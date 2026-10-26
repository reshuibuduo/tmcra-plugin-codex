from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import zipfile
from collections import OrderedDict
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..settings import ServiceSettings
from ..planner import (
    AuditedRecallPlanner,
    ScheduledRecallPlanner,
    interactive_recall_plan,
    recall_planner_from_env,
)
from ..gpu_scheduler import GpuWorkloadScheduler
from ..control_db import ControlDB
from ..costing import journal_deepseek_calls
from ..jobs import JobStore
from ..shared_core import SharedCoreVerificationError, verify_shared_core
from ..writer_pool import ResidentWriterPool, WriterPoolStatus
from ..writer_provider import (
    LOCAL_QWEN_MODEL,
    LOCAL_QWEN_PROVIDER,
    LOCAL_QWEN_PROMPT_ADAPTER,
)
from ..writer_context import (
    UNRESOLVED_CONTEXT_POLICY_VERSION,
    compact_json,
    select_unresolved_interactions,
    writer_unresolved_limits_from_env,
)
from ..usage_attribution import UNATTRIBUTED, UsageAttribution
from ..user_provider_client import normalize_user_provider_execution
from ..actor_provenance import (
    ActorProvenanceError,
    actor_metadata_json,
    actor_metadata_sha256,
    normalize_message_actor_metadata,
)


class LocalEvidenceCompilationUnavailable(RuntimeError):
    """A local model failed to finish a validated evidence plan."""


class V4AdapterError(RuntimeError):
    pass


class ContentDeletionTargetNotFound(V4AdapterError):
    """A requested memory or session selector does not exist in the scope."""


_SQLITE_SNAPSHOT_CONTRACT_DELETE_IMMUTABLE_V1 = "delete-immutable-v1"
_INCOMPLETE_RETRY_MAX_ITEMS = 32
_INCOMPLETE_RETRY_MAX_CHARS = 8_000
_INCOMPLETE_RETRY_MIN_ITEMS = 8
_INCOMPLETE_RETRY_MIN_CHARS = 2_000
_INGEST_RECOVERY_CONTRACT_VERSION = "tmcra.ingest-recovery.2"
_LOCAL_REPAIR_FINGERPRINT_CONTRACT_VERSION = (
    "tmcra.local-repair.graph-commit-lock.1"
)
_SLOW_LOCAL_REVALIDATION_FINGERPRINT_CONTRACT_VERSION = (
    "tmcra.local-repair.slow-null-counterevidence.1"
)
_SLOW_MODEL_VALIDATION_RETRY_FINGERPRINT_CONTRACT_VERSION = (
    "tmcra.model-retry.slow-validation.1"
)
_SLOW_UNATTEMPTED_QUEUE_CONTINUATION_CONTRACT_VERSION = (
    "tmcra.slow.unattempted-queue-continuation.1"
)
_LOCAL_INFERENCE_CANCELLATION_PROOF_SCHEMA = (
    "tmcra.service.local-inference-cancellation-proof.1"
)
_LOCAL_INFERENCE_CANCELLATION_PROOF_FILE = (
    "local_inference_cancellation_proof.json"
)


def _active_local_writer_model() -> str:
    """Return the configured local model alias without pinning a model family."""

    configured = str(
        os.getenv("TMCRA_WRITER_MODEL")
        or os.getenv("TMCRA_LOCAL_WRITER_MODEL")
        or ""
    ).strip()
    if configured:
        return configured
    if str(os.getenv("TMCRA_WRITER_PROVIDER") or "deepseek").strip() == "local-qwen":
        return LOCAL_QWEN_MODEL
    return "deepseek-v4-flash"


def _raw_token_estimate(content: str) -> int:
    non_empty = [char for char in content if not char.isspace()]
    cjk = sum(
        1
        for char in non_empty
        if any(
            start <= ord(char) <= end
            for start, end in (
                (0x3400, 0x4DBF),
                (0x4E00, 0x9FFF),
                (0xF900, 0xFAFF),
            )
        )
    )
    return cjk + (len(non_empty) - cjk + 3) // 4


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            # Windows does not support opening directories for fsync.
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDWR)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_read_only(path: Path) -> None:
    try:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        # The backup remains private to its generation directory even where
        # the host filesystem does not expose POSIX read-only permissions.
        pass


def _remove_tree(path: Path) -> None:
    """Remove a private tree, clearing read-only export/index artifacts."""

    root = Path(os.path.abspath(path))

    def is_within_root(value: Path) -> bool:
        try:
            return os.path.commonpath((str(root), str(Path(os.path.abspath(value))))) == str(
                root
            )
        except ValueError:
            return False

    def make_writable_and_retry(function: Any, value: str, _error: Any) -> None:
        target = Path(value)
        if not is_within_root(target):
            raise V4AdapterError("refusing to change permissions outside removal root")
        parent = target.parent
        if parent != target and is_within_root(parent):
            os.chmod(parent, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        try:
            is_directory = target.is_dir() and not target.is_symlink()
        except OSError:
            is_directory = False
        os.chmod(
            target,
            (
                stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
                if is_directory
                else stat.S_IRUSR | stat.S_IWUSR
            ),
        )
        function(value)

    shutil.rmtree(root, onerror=make_writable_and_retry)


def _sqlite_backup(source: Path, destination: Path) -> None:
    """Create a consistent, standalone SQLite backup without copying bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    source_connection: sqlite3.Connection | None = None
    destination_connection: sqlite3.Connection | None = None
    try:
        source_connection = sqlite3.connect(
            source.resolve().as_uri() + "?mode=ro", uri=True
        )
        destination_connection = sqlite3.connect(str(temporary))
        with destination_connection:
            source_connection.backup(destination_connection)
        destination_connection.close()
        destination_connection = None
        source_connection.close()
        source_connection = None
        _fsync_file(temporary)
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        _make_read_only(destination)
    finally:
        if destination_connection is not None:
            destination_connection.close()
        if source_connection is not None:
            source_connection.close()
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _identity(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class ScopePaths:
    tenant_id: str
    scope_name: str
    question_id: str
    scope_id: str
    root: Path
    database: Path
    indexes: Path
    operations: Path
    active_index: Path
    active_delta: Path


@dataclass(frozen=True)
class _GenerationValidationCacheEntry:
    """One fully validated immutable generation in this adapter process."""

    manifest_identity: tuple[str, ...]
    database_fingerprint: tuple[int, ...]
    index_fingerprint: tuple[int, ...]
    generation_directory_fingerprint: tuple[int, ...]
    sqlite_sidecar_fingerprint: tuple[int, ...]


class V4StorageAdapter:
    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        # Keep the virtualenv launcher path. Resolving it follows the symlink to
        # the system interpreter and silently drops the virtualenv packages.
        self.python = Path(sys.executable).absolute()
        self._writer_pool: ResidentWriterPool | None = None
        self._generation_validation_cache: OrderedDict[
            str, _GenerationValidationCacheEntry
        ] = OrderedDict()
        self._generation_validation_cache_max_entries = 1024
        self._generation_validation_cache_lock = threading.Lock()
        # Serialize validation per manifest without making unrelated scopes
        # wait behind one expensive database/index hash pass.
        self._generation_validation_locks = tuple(threading.RLock() for _ in range(64))

    def start(self) -> None:
        if self.settings.writer_execution_mode != "resident":
            return
        if self._writer_pool is None:
            self._writer_pool = ResidentWriterPool(
                size=self.settings.writer_pool_size,
                python=self.python,
                v4_root=self.settings.v4_root,
                repo=self.settings.integrated_repo,
                state_dir=self.settings.state_dir,
                startup_timeout=self.settings.writer_pool_startup_timeout_seconds,
                request_timeout=self.settings.writer_pool_request_timeout_seconds,
                control_db=self.settings.control_db,
                provider_key_concurrency=self.settings.provider_key_concurrency,
                provider_lease_seconds=self.settings.provider_lease_seconds,
            )
        self._writer_pool.start()

    def stop(self) -> None:
        if self._writer_pool is not None:
            self._writer_pool.stop()

    def writer_status(self) -> dict[str, Any]:
        if self.settings.writer_execution_mode == "subprocess":
            return {
                "mode": "subprocess",
                "configured": 0,
                "ready": 0,
                "alive": True,
                "pids": [],
                "protocol": "cli",
            }
        if self._writer_pool is None:
            status = WriterPoolStatus(
                configured=self.settings.writer_pool_size,
                ready=0,
                alive=False,
                pids=(),
                protocol="tmcra.writer-daemon.1",
            )
        else:
            status = self._writer_pool.status()
        return {
            "mode": "resident",
            "configured": status.configured,
            "ready": status.ready,
            "alive": status.alive,
            "pids": list(status.pids),
            "protocol": status.protocol,
            "available": status.available,
            "leased": status.leased,
        }

    def scope_paths(self, tenant_id: str, scope_name: str = "default") -> ScopePaths:
        tenant_key = _identity(tenant_id)
        scope_key = _identity(f"{tenant_id}\0{scope_name}")
        question_id = f"svc_{scope_key}"
        root = self.settings.state_dir / "tenants" / tenant_key / "scopes" / scope_key
        return ScopePaths(
            tenant_id=tenant_id,
            scope_name=scope_name,
            question_id=question_id,
            scope_id=f"tmcra_v4:{question_id}",
            root=root,
            database=root / "memory" / "native_memory.sqlite3",
            indexes=root / "indexes",
            operations=root / "operations",
            active_index=root / "active_index.json",
            active_delta=root / "active_delta_index.json",
        )

    @staticmethod
    def _active_generation_directory(
        manifest_path: Path,
        generation_root: Path,
    ) -> Path | None:
        if not manifest_path.is_file():
            return None
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise V4AdapterError("active generation manifest must be an object")
        database = Path(str(value.get("database") or "")).resolve()
        index = Path(str(value.get("index") or "")).resolve()
        directory = database.parent
        root = generation_root.resolve()
        if directory != index.parent or directory.parent != root:
            raise V4AdapterError("active generation escaped its generation root")
        if directory.name != str(value.get("generation_id") or ""):
            raise V4AdapterError("active generation directory does not match its manifest")
        if not directory.is_dir():
            raise V4AdapterError("active generation directory is missing")
        return directory

    @staticmethod
    def _generation_commit(
        paths: ScopePaths,
        directory: Path,
        *,
        kind: str,
    ) -> dict[str, Any] | None:
        name = directory.name
        if kind == "delta":
            base_name = name.split(".retry-", 1)[0]
            if not base_name.endswith("_delta"):
                return None
            job_id = base_name[: -len("_delta")]
            if not job_id:
                return None
            commit_name = "delta_commit.json"
            manifest_key = "active_delta"
        else:
            job_id = name.split(".retry-", 1)[0]
            if not job_id:
                return None
            commit_name = "index_commit.json"
            manifest_key = "active_index"
        commit_path = paths.operations / job_id / commit_name
        if not commit_path.is_file():
            return None
        try:
            commit = json.loads(commit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        manifest = commit.get(manifest_key) if isinstance(commit, dict) else None
        if not isinstance(manifest, dict) or manifest.get("generation_id") != name:
            return None
        database = Path(str(manifest.get("database") or "")).resolve()
        index = Path(str(manifest.get("index") or "")).resolve()
        if database.parent != directory.resolve() or index.parent != directory.resolve():
            return None
        if not database.is_file() or not index.is_file():
            return None
        hashes = {
            "database_sha256": str(manifest.get("database_sha256") or ""),
            "index_sha256": str(manifest.get("index_sha256") or ""),
        }
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
            for value in hashes.values()
        ):
            return None
        return {
            "commit_path": str(commit_path.resolve()),
            "activated_at": manifest.get("activated_at"),
            "source_event_seq": manifest.get(
                "source_event_seq", manifest.get("covers_through_event_seq")
            ),
            **hashes,
        }

    @staticmethod
    def _generation_size(directory: Path) -> int:
        total = 0
        for root, directories, files in os.walk(directory, followlinks=False):
            root_path = Path(root)
            for name in directories:
                if (root_path / name).is_symlink():
                    raise V4AdapterError("generation directory contains a symbolic link")
            for name in files:
                path = root_path / name
                if path.is_symlink():
                    raise V4AdapterError("generation directory contains a symbolic link")
                total += path.stat().st_size
        return total

    def prune_index_generations(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        retention: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Prune sealed, inactive index generations without weakening activation."""

        keep_count = int(
            self.settings.index_generation_retention
            if retention is None
            else retention
        )
        if keep_count <= 0:
            raise ValueError("index generation retention must be positive")
        paths = self.scope_paths(tenant_id, scope_name)
        report: dict[str, Any] = {
            "schema_version": "tmcra.service.index-generation-prune.1",
            "tenant_id": tenant_id,
            "scope_name": scope_name,
            "retention": keep_count,
            "dry_run": bool(dry_run),
            "started_at": time.time(),
            "status": "complete",
            "removed": [],
            "planned": [],
            "retained": [],
            "unsealed": [],
            "failures": [],
        }
        roots = (
            ("base", paths.indexes / "generations", paths.active_index),
            ("delta", paths.indexes / "delta-generations", paths.active_delta),
        )
        for kind, generation_root, active_manifest in roots:
            if not generation_root.is_dir():
                continue
            try:
                active = self._active_generation_directory(
                    active_manifest, generation_root
                )
            except Exception as exc:
                report["status"] = "blocked"
                report["failures"].append(
                    {"kind": kind, "generation_id": None, "error": str(exc)}
                )
                continue
            protected = {active.resolve()} if active is not None else set()
            sealed: list[tuple[Path, dict[str, Any]]] = []
            for directory in generation_root.iterdir():
                if directory.is_symlink() or not directory.is_dir():
                    report["unsealed"].append(
                        {"kind": kind, "generation_id": directory.name}
                    )
                    continue
                commit = self._generation_commit(
                    paths, directory.resolve(), kind=kind
                )
                if commit is None:
                    report["unsealed"].append(
                        {"kind": kind, "generation_id": directory.name}
                    )
                    continue
                sealed.append((directory.resolve(), commit))
            sealed.sort(
                key=lambda item: (item[0].stat().st_mtime_ns, item[0].name),
                reverse=True,
            )
            retained = set(protected)
            for directory, _commit in sealed:
                if len(retained) >= keep_count:
                    break
                retained.add(directory)
            for directory, commit in sealed:
                entry = {
                    "kind": kind,
                    "generation_id": directory.name,
                    **commit,
                }
                if directory in retained:
                    report["retained"].append(entry)
                    continue
                try:
                    entry["bytes"] = self._generation_size(directory)
                    if dry_run:
                        report["planned"].append(entry)
                    else:
                        _remove_tree(directory)
                        report["removed"].append(entry)
                except Exception as exc:
                    report["status"] = "partial"
                    report["failures"].append(
                        {**entry, "error": str(exc)}
                    )
        report["completed_at"] = time.time()
        report["removed_bytes"] = sum(
            int(item.get("bytes") or 0) for item in report["removed"]
        )
        report["planned_bytes"] = sum(
            int(item.get("bytes") or 0) for item in report["planned"]
        )
        _atomic_json(paths.indexes / "generation_prune_report.json", report)
        return report

    def _prune_index_generations_after_activation(
        self,
        *,
        tenant_id: str,
        scope_name: str,
    ) -> dict[str, Any]:
        try:
            return self.prune_index_generations(
                tenant_id=tenant_id,
                scope_name=scope_name,
            )
        except Exception as exc:
            return {
                "schema_version": "tmcra.service.index-generation-prune.1",
                "tenant_id": tenant_id,
                "scope_name": scope_name,
                "status": "failed",
                "error": str(exc),
            }

    @staticmethod
    def _validate_writer_report(
        report: Mapping[str, Any],
        *,
        paths: ScopePaths,
        job_id: str,
        stage_id: str | None = None,
        stage_attempt: int | None = None,
    ) -> None:
        """Fail closed when a Writer result is bound to another operation.

        Provider API keys are deliberately reusable credentials.  Isolation is
        therefore enforced on the operation identity carried by the work item
        and durable report, never by assigning a key to a tenant.
        """

        expected = {
            "schema_version": "tmcra.service.incremental-writer.1",
            "tenant_id": paths.tenant_id,
            "scope_name": paths.scope_name,
            "job_id": job_id,
            "stage_id": stage_id or f"{job_id}:writer",
            "operation_id": job_id,
        }
        mismatched = [
            name for name, value in expected.items() if report.get(name) != value
        ]
        if report.get("completed") is not True:
            mismatched.append("completed")
        reported_attempt = report.get("stage_attempt")
        if (
            isinstance(reported_attempt, bool)
            or not isinstance(reported_attempt, int)
            or reported_attempt <= 0
        ):
            mismatched.append("stage_attempt")
        if stage_attempt is not None and reported_attempt != stage_attempt:
            mismatched.append("stage_attempt")
        try:
            report_database = Path(str(report.get("db_path") or "")).resolve()
        except (OSError, RuntimeError, ValueError):
            report_database = Path()
        if report_database != paths.database.resolve():
            mismatched.append("db_path")
        if mismatched:
            fields = ",".join(sorted(set(mismatched)))
            raise V4AdapterError(
                f"writer report identity validation failed ({fields})"
            )

    @staticmethod
    def _writer_report_is_complete(report: Mapping[str, Any]) -> bool:
        return bool(
            report.get("schema_version") == "tmcra.service.incremental-writer.1"
            and report.get("completed") is True
            and str(report.get("status") or "").strip().lower() == "complete"
            and report.get("degraded") is False
            and report.get("input_complete") is True
            and report.get("provider_outcome_unknown") is False
        )

    @staticmethod
    def _writer_report_is_explicitly_degraded(report: Mapping[str, Any]) -> bool:
        return bool(
            report.get("schema_version") == "tmcra.service.incremental-writer.1"
            and report.get("completed") is True
            and str(report.get("status") or "").strip().lower() == "degraded"
            and report.get("degraded") is True
            and isinstance(report.get("input_complete"), bool)
            and isinstance(report.get("provider_outcome_unknown"), bool)
        )

    @staticmethod
    def _canonical_input_sha256(payload: Sequence[Mapping[str, Any]]) -> str:
        encoded = json.dumps(
            list(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        # This field is the content digest emitted by the production Writer.
        # Recovery-contract versioning belongs in the separate recovery
        # fingerprint and must not change the Writer report's hash semantics.
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _validate_complete_writer_artifacts(
        cls,
        report: Mapping[str, Any],
        *,
        paths: ScopePaths,
        operation: Path,
        expected_payload: Sequence[Mapping[str, Any]],
    ) -> None:
        """Verify the immutable Source boundary before creating a commit marker."""

        input_path = operation / "input.json"
        try:
            persisted_payload = json.loads(input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise V4AdapterError("writer input artifact is unreadable") from exc
        if persisted_payload != list(expected_payload):
            raise V4AdapterError("writer input artifact differs from the current payload")
        expected_input_sha256 = cls._canonical_input_sha256(expected_payload)
        if report.get("input_sha256") != expected_input_sha256:
            raise V4AdapterError("writer report input hash validation failed")

        if len(expected_payload) != 1 or not isinstance(expected_payload[0], Mapping):
            raise V4AdapterError("writer input artifact must contain one operation")
        raw_messages = expected_payload[0].get("messages")
        if (
            not isinstance(raw_messages, list)
            or not raw_messages
            or any(not isinstance(message, Mapping) for message in raw_messages)
        ):
            raise V4AdapterError("complete writer report has no input messages")
        messages = list(raw_messages)
        message_count = len(messages)
        for name in ("input_message_count", "verified_source_count"):
            value = report.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value != message_count:
                raise V4AdapterError(f"writer report {name} validation failed")
        new_count = report.get("new_message_count")
        replayed_count = report.get("replayed_message_count")
        if (
            isinstance(new_count, bool)
            or not isinstance(new_count, int)
            or new_count < 0
            or isinstance(replayed_count, bool)
            or not isinstance(replayed_count, int)
            or replayed_count < 0
            or new_count + replayed_count != message_count
        ):
            raise V4AdapterError("writer report message accounting validation failed")

        durable_sources = report.get("durable_sources")
        durable_count = report.get("durable_source_count")
        if (
            not isinstance(durable_sources, list)
            or any(not isinstance(item, Mapping) for item in durable_sources)
            or isinstance(durable_count, bool)
            or not isinstance(durable_count, int)
            or durable_count != len(durable_sources)
            or durable_count > message_count
        ):
            raise V4AdapterError("writer durable Source report validation failed")

        try:
            with closing(sqlite3.connect(paths.database, timeout=30.0)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA busy_timeout=30000")
                connection.execute("PRAGMA query_only=ON")
                # The online commit gate validates the immutable rows created by
                # this operation. Whole-database quick_check remains in startup,
                # explicit recovery, and offline integrity audits.
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                required_tables = {
                    "tmcra_service_messages",
                    "tmcra_service_message_actor_provenance",
                    "v4_source_journal",
                    "records",
                }
                if not required_tables.issubset(tables):
                    raise V4AdapterError("writer database lacks immutable Source tables")
                source_ids: set[str] = set()
                expected_durable: dict[str, tuple[str, int, int]] = {}
                for message in messages:
                    external_message_id = str(message.get("message_id") or "")
                    content = str(message.get("content") or "")
                    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    service_row = connection.execute(
                        "SELECT internal_message_id,session_id,role,timestamp,"
                        "content_sha256,first_operation_id,message_index "
                        "FROM tmcra_service_messages WHERE scope_id=? AND message_id=?",
                        (paths.scope_id, external_message_id),
                    ).fetchone()
                    if service_row is None:
                        raise V4AdapterError("writer service message identity is missing")
                    internal_message_id = str(service_row[0] or "").strip()
                    service_expected = (
                        str(expected_payload[0].get("session_id") or ""),
                        str(message.get("role") or "").strip().lower(),
                        str(message.get("timestamp") or "").strip(),
                        content_sha256,
                    )
                    if not internal_message_id or tuple(service_row[1:5]) != service_expected:
                        raise V4AdapterError("writer service message identity validation failed")
                    try:
                        actor = normalize_message_actor_metadata(
                            message.get("role"), message.get("metadata")
                        )
                    except ActorProvenanceError as exc:
                        raise V4AdapterError("writer actor provenance validation failed") from exc
                    actor_row = connection.execute(
                        "SELECT actor_metadata_json,actor_metadata_sha256 "
                        "FROM tmcra_service_message_actor_provenance "
                        "WHERE scope_id=? AND message_id=?",
                        (paths.scope_id, external_message_id),
                    ).fetchone()
                    if actor_row is None or tuple(actor_row) != (
                        actor_metadata_json(actor),
                        actor_metadata_sha256(actor),
                    ):
                        raise V4AdapterError("writer actor provenance identity validation failed")

                    source_row = connection.execute(
                        "SELECT scope_id,session_id,message_id,message_role,timestamp,"
                        "content,content_sha256,status,source_record_id,source_persisted_at "
                        "FROM v4_source_journal WHERE scope_id=? AND message_id=?",
                        (paths.scope_id, internal_message_id),
                    ).fetchone()
                    expected = (
                        paths.scope_id,
                        str(expected_payload[0].get("session_id") or ""),
                        internal_message_id,
                        str(message.get("role") or "").strip().lower(),
                        str(message.get("timestamp") or "").strip(),
                        content,
                        content_sha256,
                    )
                    if source_row is None or tuple(source_row[:7]) != expected:
                        raise V4AdapterError("writer immutable Source identity validation failed")
                    source_record_id = str(source_row[8] or "").strip()
                    if (
                        str(source_row[7] or "") != "enriched"
                        or not source_record_id
                        or not str(source_row[9] or "").strip()
                        or source_record_id in source_ids
                    ):
                        raise V4AdapterError("writer immutable Source durability validation failed")
                    graph_row = connection.execute(
                        "SELECT 1 FROM records WHERE scope_id=? AND memory_id=? LIMIT 1",
                        (paths.scope_id, source_record_id),
                    ).fetchone()
                    if graph_row is None:
                        raise V4AdapterError("writer immutable Source graph record is missing")
                    source_ids.add(source_record_id)
                    first_operation_id = str(service_row[5] or "").strip()
                    if first_operation_id:
                        expected_durable[source_record_id] = (
                            first_operation_id,
                            _raw_token_estimate(content),
                            int(str(message.get("role") or "").strip().lower() == "user"),
                        )
        except sqlite3.DatabaseError as exc:
            raise V4AdapterError("writer immutable Source database validation failed") from exc

        reported_durable: dict[str, tuple[str, int, int]] = {}
        for item in durable_sources:
            source_record_id = str(item.get("source_record_id") or "").strip()
            origin_operation_id = str(item.get("origin_operation_id") or "").strip()
            raw_token_estimate = item.get("raw_token_estimate")
            user_turns = item.get("user_turns")
            if (
                not source_record_id
                or source_record_id in reported_durable
                or isinstance(raw_token_estimate, bool)
                or not isinstance(raw_token_estimate, int)
                or raw_token_estimate < 0
                or isinstance(user_turns, bool)
                or not isinstance(user_turns, int)
                or user_turns not in {0, 1}
            ):
                raise V4AdapterError("writer durable Source identity validation failed")
            reported_durable[source_record_id] = (
                origin_operation_id,
                raw_token_estimate,
                user_turns,
            )
        if reported_durable != expected_durable:
            raise V4AdapterError("writer durable Source accounting validation failed")

    @staticmethod
    def _database_quick_check(database: Path) -> bool:
        if not database.is_file():
            return False
        try:
            with closing(sqlite3.connect(database, timeout=30.0)) as connection:
                row = connection.execute("PRAGMA quick_check").fetchone()
        except (OSError, sqlite3.DatabaseError):
            return False
        return bool(row and row[0] == "ok")

    @staticmethod
    def _legacy_input_operation_bindings(
        connection: sqlite3.Connection,
        *,
        paths: ScopePaths,
        existing_bindings: Mapping[str, str],
    ) -> tuple[dict[str, str], set[str]]:
        """Recover pre-provenance message bindings from immutable ingest inputs.

        Older service databases registered stable message identities before the
        ``first_operation_id`` column existed.  The original operation input is
        still content-bound on disk.  Use it only when one registered message
        maps to exactly one operation and every identity field still matches.
        """

        violations: set[str] = set()
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(tmcra_service_messages)"
            )
        }
        required = {
            "message_id",
            "internal_message_id",
            "session_id",
            "role",
            "timestamp",
            "content_sha256",
            "first_operation_id",
        }
        if not required.issubset(columns):
            return {}, set()

        service_rows = connection.execute(
            "SELECT message_id,internal_message_id,session_id,role,timestamp,"
            "content_sha256,first_operation_id FROM tmcra_service_messages "
            "WHERE scope_id=?",
            (paths.scope_id,),
        ).fetchall()
        services: dict[str, tuple[str, str, str, str, str, str]] = {}
        legacy_external_ids: set[str] = set()
        for row in service_rows:
            external_id = str(row[0] or "").strip()
            internal_id = str(row[1] or "").strip()
            if not external_id or not internal_id or external_id in services:
                violations.add("source_operation_binding_invalid")
                continue
            services[external_id] = (
                internal_id,
                str(row[2] or ""),
                str(row[3] or ""),
                str(row[4] or ""),
                str(row[5] or ""),
                str(row[6] or "").strip(),
            )
            if not str(row[6] or "").strip():
                legacy_external_ids.add(external_id)
        if not legacy_external_ids:
            return {}, violations

        candidates: dict[str, set[str]] = {}
        try:
            input_paths = sorted(paths.operations.glob("*/input.json"))
        except OSError:
            return {}, violations | {"source_operation_binding_artifacts_unreadable"}
        for input_path in input_paths:
            try:
                payload = json.loads(input_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, list):
                continue
            operation_id = input_path.parent.name
            for input_row in payload:
                if not isinstance(input_row, Mapping):
                    continue
                messages = input_row.get("messages")
                if not isinstance(messages, list):
                    continue
                external_ids = {
                    str(message.get("message_id") or "").strip()
                    for message in messages
                    if isinstance(message, Mapping)
                }
                if not (external_ids & legacy_external_ids):
                    continue
                if (
                    str(input_row.get("scope_id") or "") != paths.scope_id
                    or str(input_row.get("question_id") or "") != paths.question_id
                    or str(input_row.get("operation_id") or "") != operation_id
                    or not str(input_row.get("session_id") or "")
                    or not messages
                ):
                    violations.add("source_operation_binding_artifact_invalid")
                    continue

                session_id = str(input_row["session_id"])
                row_internal_ids: set[str] = set()
                row_legacy_ids: list[str] = []
                valid = True
                for message in messages:
                    if not isinstance(message, Mapping):
                        valid = False
                        break
                    external_id = str(message.get("message_id") or "").strip()
                    service = services.get(external_id)
                    if service is None:
                        valid = False
                        break
                    internal_id, stored_session, role, timestamp, content_sha256, _ = (
                        service
                    )
                    expected = (
                        session_id,
                        str(message.get("role") or "").strip().lower(),
                        str(message.get("timestamp") or "").strip(),
                        hashlib.sha256(
                            str(message.get("content") or "").encode("utf-8")
                        ).hexdigest(),
                    )
                    if (
                        internal_id in row_internal_ids
                        or (stored_session, role, timestamp, content_sha256)
                        != expected
                    ):
                        valid = False
                        break
                    row_internal_ids.add(internal_id)
                    if external_id in legacy_external_ids:
                        row_legacy_ids.append(internal_id)
                if not valid:
                    violations.add("source_operation_binding_artifact_invalid")
                    continue
                for internal_id in row_legacy_ids:
                    candidates.setdefault(internal_id, set()).add(operation_id)

        bindings: dict[str, str] = {}
        for internal_id, operation_ids in candidates.items():
            if len(operation_ids) != 1:
                violations.add("source_operation_binding_ambiguous")
                continue
            bindings[internal_id] = next(iter(operation_ids))
        return bindings, violations

    @staticmethod
    def _source_operation_bindings(
        connection: sqlite3.Connection,
        *,
        paths: ScopePaths,
    ) -> tuple[dict[str, str], dict[str, tuple[str, bool, bool]], set[str]]:
        """Bind immutable Source IDs to their first ingest operation.

        Current databases persist ``first_operation_id`` on the service message.
        Older production databases predate that column, but their immutable batch
        registry and hash-bound Writer request journal contain the same binding.
        Recovery may reconstruct it from those journals only when every request is
        intact and each Source appears in exactly one operation.
        """

        bindings: dict[str, str] = {}
        batch_states: dict[str, tuple[str, bool, bool]] = {}
        violations: set[str] = set()
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "tmcra_service_messages" not in tables:
            return bindings, batch_states, {
                "source_operation_binding_tables_missing"
            }

        message_columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(tmcra_service_messages)"
            )
        }
        authoritative_ids: set[str] = set()
        if "first_operation_id" in message_columns:
            for row in connection.execute(
                "SELECT internal_message_id,first_operation_id "
                "FROM tmcra_service_messages WHERE scope_id=?",
                (paths.scope_id,),
            ):
                message_id = str(row[0] or "").strip()
                operation_id = str(row[1] or "").strip()
                if not message_id:
                    violations.add("source_operation_binding_invalid")
                elif operation_id:
                    bindings[message_id] = operation_id
                    authoritative_ids.add(message_id)

        if not {"tmcra_service_batches", "v4_batch_journal"}.issubset(tables):
            if not bindings:
                violations.add("source_operation_binding_tables_missing")
            return bindings, batch_states, violations

        seen_in_batches: set[str] = set()
        rows = connection.execute(
            "SELECT batches.operation_id,journal.batch_id,journal.request_json,"
            "journal.request_sha256,journal.status,journal.response_json,"
            "journal.api_started_at "
            "FROM tmcra_service_batches AS batches "
            "JOIN v4_batch_journal AS journal "
            "ON journal.scope_id=batches.scope_id "
            "AND journal.session_id=batches.session_id "
            "AND journal.batch_index=batches.batch_index "
            "WHERE journal.scope_id=? "
            "ORDER BY journal.session_id,journal.batch_index,journal.batch_id",
            (paths.scope_id,),
        ).fetchall()
        for row in rows:
            operation_id = str(row[0] or "").strip()
            batch_id = str(row[1] or "").strip()
            request_json = str(row[2] or "")
            if (
                not operation_id
                or not batch_id
                or not request_json
                or hashlib.sha256(request_json.encode("utf-8")).hexdigest()
                != str(row[3] or "")
            ):
                violations.add("source_operation_request_hash_mismatch")
                continue
            try:
                request = json.loads(request_json)
            except json.JSONDecodeError:
                violations.add("source_operation_request_invalid")
                continue
            if (
                not isinstance(request, Mapping)
                or str(request.get("batch_id") or "") != batch_id
                or not isinstance(request.get("messages"), list)
                or not request["messages"]
            ):
                violations.add("source_operation_request_invalid")
                continue
            for message in request["messages"]:
                if not isinstance(message, Mapping):
                    violations.add("source_operation_request_invalid")
                    continue
                message_id = str(message.get("message_id") or "").strip()
                if not message_id:
                    violations.add("source_operation_binding_invalid")
                    continue
                if message_id in authoritative_ids:
                    if bindings.get(message_id) == operation_id:
                        batch_states.setdefault(
                            message_id,
                            (
                                str(row[4] or ""),
                                bool(str(row[5] or "")),
                                bool(str(row[6] or "")),
                            ),
                        )
                    continue
                if message_id in seen_in_batches:
                    violations.add("source_operation_binding_duplicate")
                seen_in_batches.add(message_id)
                prior = bindings.get(message_id)
                if prior and prior != operation_id:
                    violations.add("source_operation_binding_conflict")
                else:
                    bindings[message_id] = operation_id
                    batch_states[message_id] = (
                        str(row[4] or ""),
                        bool(str(row[5] or "")),
                        bool(str(row[6] or "")),
                    )
        legacy_bindings, legacy_violations = (
            V4StorageAdapter._legacy_input_operation_bindings(
                connection,
                paths=paths,
                existing_bindings=bindings,
            )
        )
        violations.update(legacy_violations)
        for message_id, operation_id in legacy_bindings.items():
            prior = bindings.get(message_id)
            if prior and prior != operation_id:
                violations.add("source_operation_binding_conflict")
                continue
            bindings[message_id] = operation_id
        if "first_operation_id" in message_columns:
            legacy_message_ids = {
                str(row[0] or "").strip()
                for row in connection.execute(
                    "SELECT internal_message_id FROM tmcra_service_messages "
                    "WHERE scope_id=? AND first_operation_id=''",
                    (paths.scope_id,),
                )
                if str(row[0] or "").strip()
            }
            if legacy_message_ids - set(bindings):
                violations.add("source_operation_binding_missing")
        return bindings, batch_states, violations

    def _pre_source_registered_operations(
        self,
        connection: sqlite3.Connection,
        *,
        paths: ScopePaths,
        message_ids: set[str],
        operation_bindings: Mapping[str, str],
        batch_states: Mapping[str, tuple[str, bool, bool]],
        source_message_ids: set[str],
    ) -> tuple[set[str], set[str]]:
        """Validate a durable Writer prefix followed by unprepared messages.

        ``IdentityRegistry.register_messages`` commits stable message identities
        and batch identities before immutable Source persistence.  A process may
        stop before the first batch or after a durable prefix.  Gap rows are
        resumable only when modern identity bindings, the input artifact, and a
        contiguous prepared prefix all agree that no gap message reached a
        Writer request or semantic commit.
        """

        if not message_ids:
            return set(), set()
        violations: set[str] = set()
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required_tables = {
            "tmcra_service_messages",
            "tmcra_service_batches",
            "v4_source_journal",
            "v4_message_commit_journal",
        }
        if not required_tables.issubset(tables):
            return set(), {"source_operation_binding_set_mismatch"}
        message_columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(tmcra_service_messages)"
            )
        }
        if not {"internal_message_id", "first_operation_id"}.issubset(
            message_columns
        ):
            return set(), {"source_operation_binding_set_mismatch"}

        placeholders = ",".join("?" for _ in message_ids)
        rows = connection.execute(
            "SELECT internal_message_id,first_operation_id "
            "FROM tmcra_service_messages "
            f"WHERE scope_id=? AND internal_message_id IN ({placeholders})",
            (paths.scope_id, *sorted(message_ids)),
        ).fetchall()
        registered = {
            str(row[0] or "").strip(): str(row[1] or "").strip() for row in rows
        }
        operation_ids: set[str] = set()
        for message_id in message_ids:
            operation_id = str(operation_bindings.get(message_id) or "").strip()
            registered_operation_id = registered.get(message_id)
            if (
                not operation_id
                or registered_operation_id is None
                or registered_operation_id not in {"", operation_id}
            ):
                violations.add("source_operation_binding_set_mismatch")
                continue
            operation_ids.add(operation_id)
        if violations or not operation_ids:
            return set(), violations or {"source_operation_binding_set_mismatch"}

        source_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM v4_source_journal "
                f"WHERE scope_id=? AND message_id IN ({placeholders})",
                (paths.scope_id, *sorted(message_ids)),
            ).fetchone()[0]
            or 0
        )
        commit_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM v4_message_commit_journal "
                f"WHERE scope_id=? AND message_id IN ({placeholders})",
                (paths.scope_id, *sorted(message_ids)),
            ).fetchone()[0]
            or 0
        )
        if source_count or commit_count:
            violations.add("source_operation_binding_set_mismatch")
            return set(), violations

        for operation_id in sorted(operation_ids):
            operation_message_ids = {
                message_id
                for message_id, bound_operation_id in operation_bindings.items()
                if bound_operation_id == operation_id
            }
            operation_gap_ids = operation_message_ids & message_ids
            if (
                not operation_gap_ids
                or operation_gap_ids & set(batch_states)
                or operation_gap_ids & source_message_ids
            ):
                violations.add("source_operation_binding_set_mismatch")
                continue

            input_path = paths.operations / operation_id / "input.json"
            try:
                payload = json.loads(input_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                violations.add("source_operation_binding_set_mismatch")
                continue
            if not isinstance(payload, list) or not payload:
                violations.add("source_operation_binding_set_mismatch")
                continue

            input_internal_ids: set[str] = set()
            input_valid = True
            for input_row in payload:
                if (
                    not isinstance(input_row, Mapping)
                    or str(input_row.get("scope_id") or "") != paths.scope_id
                    or str(input_row.get("question_id") or "")
                    != paths.question_id
                    or str(input_row.get("operation_id") or "") != operation_id
                    or not str(input_row.get("session_id") or "")
                    or not isinstance(input_row.get("messages"), list)
                    or not input_row["messages"]
                ):
                    input_valid = False
                    break
                session_id = str(input_row["session_id"])
                for message in input_row["messages"]:
                    if not isinstance(message, Mapping):
                        input_valid = False
                        break
                    external_id = str(message.get("message_id") or "").strip()
                    role = str(message.get("role") or "").strip().lower()
                    timestamp = str(message.get("timestamp") or "").strip()
                    content = str(message.get("content") or "")
                    service = connection.execute(
                        "SELECT internal_message_id,session_id,role,timestamp,"
                        "content_sha256 FROM tmcra_service_messages "
                        "WHERE scope_id=? AND message_id=?",
                        (paths.scope_id, external_id),
                    ).fetchone()
                    if service is None:
                        input_valid = False
                        break
                    internal_id = str(service[0] or "").strip()
                    if (
                        not external_id
                        or not internal_id
                        or internal_id in input_internal_ids
                        or str(service[1] or "") != session_id
                        or str(service[2] or "") != role
                        or str(service[3] or "") != timestamp
                        or str(service[4] or "")
                        != hashlib.sha256(content.encode("utf-8")).hexdigest()
                    ):
                        input_valid = False
                        break
                    input_internal_ids.add(internal_id)
                if not input_valid:
                    break
            if (
                not input_valid
                or not operation_message_ids.issubset(input_internal_ids)
                or not operation_gap_ids.issubset(input_internal_ids)
            ):
                violations.add("source_operation_binding_set_mismatch")
                continue

            batch_rows = connection.execute(
                "SELECT batches.local_batch_index,batches.batch_index,"
                "journal.request_json,journal.request_sha256 "
                "FROM tmcra_service_batches AS batches "
                "LEFT JOIN v4_batch_journal AS journal "
                "ON journal.scope_id=batches.scope_id "
                "AND journal.session_id=batches.session_id "
                "AND journal.batch_index=batches.batch_index "
                "WHERE batches.scope_id=? AND batches.operation_id=? "
                "ORDER BY batches.local_batch_index",
                (paths.scope_id, operation_id),
            ).fetchall()
            indexes = [int(row[0]) for row in batch_rows]
            if indexes and indexes != list(range(len(indexes))):
                violations.add("source_operation_binding_set_mismatch")
                continue
            seen_missing = False
            journal_message_ids: set[str] = set()
            valid_prefix = True
            for batch_row in batch_rows:
                request_json = str(batch_row[2] or "")
                request_sha256 = str(batch_row[3] or "")
                if not request_json:
                    seen_missing = True
                    continue
                if seen_missing or hashlib.sha256(
                    request_json.encode("utf-8")
                ).hexdigest() != request_sha256:
                    valid_prefix = False
                    break
                try:
                    request = json.loads(request_json)
                except json.JSONDecodeError:
                    valid_prefix = False
                    break
                messages = request.get("messages") if isinstance(request, Mapping) else None
                if not isinstance(messages, list) or not messages:
                    valid_prefix = False
                    break
                for message in messages:
                    message_id = (
                        str(message.get("message_id") or "").strip()
                        if isinstance(message, Mapping)
                        else ""
                    )
                    if not message_id or message_id in journal_message_ids:
                        valid_prefix = False
                        break
                    journal_message_ids.add(message_id)
                if not valid_prefix:
                    break
            if (
                not valid_prefix
                or bool(batch_rows) and not seen_missing
                or journal_message_ids & operation_gap_ids
                or not journal_message_ids.issubset(source_message_ids)
                or (operation_message_ids & source_message_ids)
                != journal_message_ids
            ):
                violations.add("source_operation_binding_set_mismatch")
                continue

        return (set() if violations else operation_ids), violations

    @staticmethod
    def _valid_ingest_commit(
        commit: Mapping[str, Any],
        *,
        paths: ScopePaths,
        job_id: str,
    ) -> bool:
        try:
            database = Path(str(commit.get("database") or "")).resolve()
        except (OSError, RuntimeError, ValueError):
            return False
        return bool(
            commit.get("schema_version") == "tmcra.service.ingest-commit.1"
            and commit.get("job_id") == job_id
            and commit.get("tenant_id") == paths.tenant_id
            and commit.get("scope_id") == paths.scope_id
            and database == paths.database.resolve()
        )

    @staticmethod
    def _archive_writer_report(
        report_path: Path,
        *,
        prior_attempt: int,
    ) -> None:
        archive = report_path.with_name(
            f"product_writer_report.attempt-{prior_attempt}.json"
        )
        content = report_path.read_bytes()
        if archive.exists():
            if archive.read_bytes() != content:
                raise V4AdapterError("writer retry archive changed immutable content")
            return
        with archive.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

    def _scope_export_root(self, tenant_id: str, scope_name: str) -> Path:
        tenant_key = _identity(tenant_id)
        scope_key = _identity(f"{tenant_id}\0{scope_name}")
        return self.settings.state_dir / "exports" / tenant_key / scope_key

    def export_scope(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        export_id: str,
        job_id: str,
        expires_at: float,
    ) -> dict[str, Any]:
        paths = self.scope_paths(tenant_id, scope_name)
        if not paths.database.is_file():
            raise V4AdapterError("scope has no native memory database to export")
        export_root = self._scope_export_root(tenant_id, scope_name)
        export_root.mkdir(parents=True, exist_ok=True)
        artifact = export_root / f"{export_id}.zip"
        if artifact.is_file():
            return {
                "export_id": export_id,
                "artifact_path": str(artifact),
                "artifact_sha256": _sha256_file(artifact),
                "size_bytes": artifact.stat().st_size,
                "expires_at": expires_at,
            }
        staging = export_root / f".{export_id}.staging.{os.getpid()}.{time.time_ns()}"
        archive_temporary = export_root / f".{export_id}.zip.tmp.{os.getpid()}.{time.time_ns()}"
        try:
            staging.mkdir(parents=False, exist_ok=False)
            database_backup = staging / "native_memory.sqlite3"
            _sqlite_backup(paths.database, database_backup)
            manifest = {
                "schema_version": "tmcra.scope-export.v1",
                "export_id": export_id,
                "job_id": job_id,
                "tenant_id": tenant_id,
                "scope_name": scope_name,
                "created_at": time.time(),
                "expires_at": expires_at,
                "files": {
                    "native_memory.sqlite3": {
                        "sha256": _sha256_file(database_backup),
                        "size_bytes": database_backup.stat().st_size,
                    }
                },
            }
            _atomic_json(staging / "manifest.json", manifest)
            with zipfile.ZipFile(
                archive_temporary,
                mode="x",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                archive.write(staging / "manifest.json", "manifest.json")
                archive.write(database_backup, "native_memory.sqlite3")
            _fsync_file(archive_temporary)
            os.replace(archive_temporary, artifact)
            return {
                "export_id": export_id,
                "artifact_path": str(artifact),
                "artifact_sha256": _sha256_file(artifact),
                "size_bytes": artifact.stat().st_size,
                "expires_at": expires_at,
            }
        finally:
            try:
                archive_temporary.unlink()
            except FileNotFoundError:
                pass
            if staging.exists():
                _remove_tree(staging)

    def delete_scope(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        job_id: str,
    ) -> dict[str, Any]:
        paths = self.scope_paths(tenant_id, scope_name)
        state_root = self.settings.state_dir.resolve()
        deleted_root = (self.settings.state_dir / "deleted").resolve()
        deleted_root.mkdir(parents=True, exist_ok=True)

        def remove_tree(source: Path, label: str) -> bool:
            source = source.resolve()
            if not source.is_relative_to(state_root):
                raise V4AdapterError(f"refusing to delete {label} outside the service state directory")
            if not source.exists():
                return False
            tombstone = (deleted_root / f"{label}.{job_id}.{time.time_ns()}").resolve()
            if not tombstone.is_relative_to(deleted_root):
                raise V4AdapterError("invalid deletion tombstone path")
            os.replace(source, tombstone)
            _remove_tree(tombstone)
            return True

        scope_removed = remove_tree(paths.root, f"scope-{_identity(tenant_id + chr(0) + scope_name)}")
        exports_removed = remove_tree(
            self._scope_export_root(tenant_id, scope_name),
            f"exports-{_identity(tenant_id + chr(0) + scope_name)}",
        )
        return {
            "scope_name": scope_name,
            "scope_id": paths.scope_id,
            "scope_removed": scope_removed,
            "exports_removed": exports_removed,
        }

    @staticmethod
    def _json_contains_identifier(raw: Any, identifiers: set[str]) -> bool:
        if not identifiers or raw in {None, ""}:
            return False
        try:
            value = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return False

        def contains(item: Any) -> bool:
            if isinstance(item, str):
                return item in identifiers
            if isinstance(item, Mapping):
                return any(contains(key) or contains(child) for key, child in item.items())
            if isinstance(item, list):
                return any(contains(child) for child in item)
            return False

        return contains(value)

    def resolve_source_memory_ids_for_messages(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        message_ids: Sequence[str],
    ) -> list[str]:
        """Resolve stable external message IDs to immutable Source memory IDs."""

        requested = {
            str(value).strip() for value in message_ids if str(value).strip()
        }
        if not requested:
            raise ValueError("message IDs are required")
        paths = self.scope_paths(tenant_id, scope_name)
        if not paths.database.is_file():
            raise ContentDeletionTargetNotFound("scope has no native memory database")
        try:
            with closing(sqlite3.connect(str(paths.database), timeout=10.0)) as connection:
                connection.row_factory = sqlite3.Row
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                if "records" not in tables or "tmcra_service_messages" not in tables:
                    raise V4AdapterError("scope message provenance tables are missing")
                service_columns = {
                    str(row[1])
                    for row in connection.execute(
                        "PRAGMA table_info(tmcra_service_messages)"
                    ).fetchall()
                }
                if not {"scope_id", "message_id"}.issubset(service_columns):
                    raise V4AdapterError("scope message provenance schema is incompatible")
                placeholders = ",".join("?" for _ in requested)
                if "internal_message_id" in service_columns:
                    service_rows = connection.execute(
                        "SELECT message_id,internal_message_id FROM "
                        "tmcra_service_messages "
                        f"WHERE scope_id=? AND message_id IN ({placeholders})",
                        (paths.scope_id, *sorted(requested)),
                    ).fetchall()
                    internal_by_external = {
                        str(row["message_id"]): str(
                            row["internal_message_id"] or ""
                        ).strip()
                        for row in service_rows
                    }
                else:
                    service_rows = connection.execute(
                        "SELECT message_id FROM tmcra_service_messages "
                        f"WHERE scope_id=? AND message_id IN ({placeholders})",
                        (paths.scope_id, *sorted(requested)),
                    ).fetchall()
                    internal_by_external = {
                        str(row["message_id"]): str(row["message_id"])
                        for row in service_rows
                    }
                missing = sorted(requested - set(internal_by_external))
                if missing:
                    raise ContentDeletionTargetNotFound(
                        "message IDs were not found: " + ",".join(missing[:10])
                    )
                if any(not value for value in internal_by_external.values()):
                    raise V4AdapterError("message provenance is incomplete")
                record_rows = connection.execute(
                    "SELECT memory_id,metadata_json FROM records WHERE scope_id=?",
                    (paths.scope_id,),
                ).fetchall()
        except (ContentDeletionTargetNotFound, V4AdapterError):
            raise
        except sqlite3.DatabaseError as exc:
            raise V4AdapterError(
                "scope message provenance could not be inspected"
            ) from exc

        requested_internal = set(internal_by_external.values())
        source_ids_by_message: dict[str, set[str]] = {
            value: set() for value in requested_internal
        }
        for row in record_rows:
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            if str(metadata.get("content_variant") or "") != "source_message":
                continue
            internal_id = str(metadata.get("message_id") or "").strip()
            if internal_id in source_ids_by_message:
                source_ids_by_message[internal_id].add(str(row["memory_id"]))
        unresolved = sorted(
            external
            for external, internal in internal_by_external.items()
            if not source_ids_by_message.get(internal)
        )
        if unresolved:
            raise ContentDeletionTargetNotFound(
                "message Source records are not ready: " + ",".join(unresolved[:10])
            )
        return sorted(
            memory_id
            for values in source_ids_by_message.values()
            for memory_id in values
        )

    def validate_content_deletion_targets(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        memory_ids: Sequence[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a deletion selector before placing the scope on hold.

        This check keeps malformed or stale client selections out of the
        durable deletion queue. The worker repeats the validation inside its
        write transaction so this read-only preflight is not a safety boundary.
        """

        explicit_ids = {
            str(value).strip() for value in (memory_ids or ()) if str(value).strip()
        }
        clean_session = str(session_id or "").strip()
        if bool(explicit_ids) == bool(clean_session):
            raise ValueError("provide exactly one of memory_ids or session_id")
        paths = self.scope_paths(tenant_id, scope_name)
        if not paths.database.is_file():
            raise ContentDeletionTargetNotFound("scope has no native memory database")

        try:
            with closing(sqlite3.connect(str(paths.database), timeout=10.0)) as connection:
                connection.row_factory = sqlite3.Row
                table = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='records'"
                ).fetchone()
                if table is None:
                    raise V4AdapterError("scope memory records table is missing")
                columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(records)").fetchall()
                }
                required = {"scope_id", "memory_id", "metadata_json"}
                if not required.issubset(columns):
                    raise V4AdapterError("scope memory records schema is incompatible")
                rows = connection.execute(
                    "SELECT memory_id,metadata_json FROM records WHERE scope_id=?",
                    (paths.scope_id,),
                ).fetchall()
        except ContentDeletionTargetNotFound:
            raise
        except sqlite3.DatabaseError as exc:
            raise V4AdapterError("scope memory database could not be inspected") from exc

        available_ids = {str(row["memory_id"]) for row in rows}
        if explicit_ids:
            missing = sorted(explicit_ids - available_ids)
            if missing:
                raise ContentDeletionTargetNotFound(
                    "memory IDs were not found: " + ",".join(missing[:10])
                )
            return {
                "mode": "memory_ids",
                "requested_memory_count": len(explicit_ids),
                "matched_memory_count": len(explicit_ids),
            }

        matched_ids: list[str] = []
        source_count = 0
        registered_message_count = 0
        for row in rows:
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            if not isinstance(metadata, dict):
                continue
            if str(metadata.get("session_id") or "") != clean_session:
                continue
            matched_ids.append(str(row["memory_id"]))
            if str(metadata.get("content_variant") or "") == "source_message":
                source_count += 1
        if not matched_ids:
            raise ContentDeletionTargetNotFound(
                "session was not found in scope memory records"
            )
        return {
            "mode": "session",
            "session_id": clean_session,
            "matched_memory_count": len(matched_ids),
            "matched_source_memory_count": source_count,
        }

    def delete_memories(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        memory_ids: Sequence[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Purge memory content from the authoritative scope database.

        Source and Fast records are removed by exact provenance. Slow capsules
        are invalidated as a set because one capsule can summarize several
        source records. The caller must activate a fresh immutable base index
        before making the scope readable again.
        """

        explicit_ids = {str(value).strip() for value in (memory_ids or ()) if str(value).strip()}
        clean_session = str(session_id or "").strip()
        if bool(explicit_ids) == bool(clean_session):
            raise ValueError("provide exactly one of memory_ids or session_id")
        paths = self.scope_paths(tenant_id, scope_name)
        if not paths.database.is_file():
            raise V4AdapterError("scope has no native memory database")

        connection = sqlite3.connect(str(paths.database), timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            connection.execute("BEGIN IMMEDIATE")
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "records" not in tables:
                raise V4AdapterError("scope memory records table is missing")
            rows = connection.execute(
                "SELECT memory_id,evidence_anchors_json,supersedes_json,metadata_json "
                "FROM records WHERE scope_id=?",
                (paths.scope_id,),
            ).fetchall()
            metadata_by_id: dict[str, dict[str, Any]] = {}
            source_ids_for_session: set[str] = set()
            message_ids: set[str] = set()
            for row in rows:
                try:
                    metadata = json.loads(str(row["metadata_json"] or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    metadata = {}
                metadata = metadata if isinstance(metadata, dict) else {}
                memory_id = str(row["memory_id"])
                metadata_by_id[memory_id] = metadata
                if clean_session and str(metadata.get("session_id") or "") == clean_session:
                    source_ids_for_session.add(memory_id)
                    message_id = str(metadata.get("message_id") or "").strip()
                    if message_id:
                        message_ids.add(message_id)

            target_ids = set(explicit_ids or source_ids_for_session)
            if explicit_ids:
                missing = sorted(explicit_ids - set(metadata_by_id))
                if missing:
                    raise V4AdapterError(
                        "memory IDs were not found: " + ",".join(missing[:10])
                    )
                for memory_id in explicit_ids:
                    metadata = metadata_by_id.get(memory_id, {})
                    source_record_id = str(metadata.get("source_record_id") or "").strip()
                    if source_record_id:
                        target_ids.add(source_record_id)
                    message_id = str(metadata.get("message_id") or "").strip()
                    if message_id:
                        message_ids.add(message_id)
            elif not source_ids_for_session:
                raise V4AdapterError("session was not found in scope memory records")

            # Expand through exact graph provenance until stable. This removes
            # Fast records grounded in a deleted Source record and revisions
            # that explicitly supersede deleted records.
            changed = True
            while changed:
                changed = False
                for row in rows:
                    memory_id = str(row["memory_id"])
                    if memory_id in target_ids:
                        continue
                    metadata = metadata_by_id.get(memory_id, {})
                    source_record_id = str(metadata.get("source_record_id") or "").strip()
                    references_target = source_record_id in target_ids or any(
                        self._json_contains_identifier(row[column], target_ids)
                        for column in (
                            "evidence_anchors_json",
                            "supersedes_json",
                            "metadata_json",
                        )
                    )
                    if references_target:
                        target_ids.add(memory_id)
                        changed = True

            slow_ids = {
                memory_id
                for memory_id, metadata in metadata_by_id.items()
                if str(metadata.get("memory_layer") or "").lower() == "slow"
                or str(metadata.get("content_variant") or "")
                == "slow_memory_capsule"
            }
            target_ids.update(slow_ids)
            if not target_ids:
                raise V4AdapterError("deletion matched no memory records")

            deleted_source_ids: set[str] = set()
            deleted_session_message_counts: dict[str, int] = {}
            for memory_id in target_ids:
                metadata = metadata_by_id.get(memory_id, {})
                if str(metadata.get("content_variant") or "") != "source_message":
                    continue
                deleted_source_ids.add(memory_id)
                message_id = str(metadata.get("message_id") or "").strip()
                if message_id:
                    message_ids.add(message_id)
                record_session_id = str(metadata.get("session_id") or "").strip()
                if record_session_id:
                    deleted_session_message_counts[record_session_id] = (
                        deleted_session_message_counts.get(record_session_id, 0) + 1
                    )

            placeholders = ",".join("?" for _ in target_ids)
            parameters = (paths.scope_id, *sorted(target_ids))
            # Clear projections before the authoritative records. Current
            # production schemas do not require every projection to declare a
            # foreign key, but deletion must remain valid when they do.
            for table in ("slot_heads", "slot_history", "subject_depth_heads"):
                if table in tables:
                    connection.execute(
                        f"DELETE FROM {table} WHERE scope_id=? AND memory_id IN ({placeholders})",
                        parameters,
                    )
            if "memory_edges" in tables:
                connection.execute(
                    f"DELETE FROM memory_edges WHERE scope_id=? "
                    f"AND (source_memory_id IN ({placeholders}) "
                    f"OR target_memory_id IN ({placeholders}))",
                    (
                        paths.scope_id,
                        *sorted(target_ids),
                        *sorted(target_ids),
                    ),
                )
            deleted_record_count = connection.execute(
                f"DELETE FROM records WHERE scope_id=? AND memory_id IN ({placeholders})",
                parameters,
            ).rowcount
            slow_patch_ids: list[str] = []
            if "slow_graph_patches" in tables:
                slow_patch_ids = [
                    str(row[0])
                    for row in connection.execute(
                        "SELECT patch_id FROM slow_graph_patches WHERE scope_id=?",
                        (paths.scope_id,),
                    ).fetchall()
                ]
            if "slow_graph_patch_operations" in tables and slow_patch_ids:
                patch_placeholders = ",".join("?" for _ in slow_patch_ids)
                connection.execute(
                    f"DELETE FROM slow_graph_patch_operations "
                    f"WHERE patch_id IN ({patch_placeholders})",
                    tuple(slow_patch_ids),
                )
            for table in (
                "slow_graph_jobs",
                "slow_graph_attempts",
                "slow_graph_batches",
                "slow_graph_patches",
                "slow_graph_provenance",
            ):
                if table in tables:
                    connection.execute(f"DELETE FROM {table} WHERE scope_id=?", (paths.scope_id,))
            for table in ("audit_turn_log", "audit_retrieval_log", "audit_answer_support"):
                if table in tables:
                    connection.execute(f"DELETE FROM {table} WHERE scope_id=?", (paths.scope_id,))

            if message_ids:
                message_placeholders = ",".join("?" for _ in message_ids)
                message_parameters = (paths.scope_id, *sorted(message_ids))
                # Record metadata uses the writer's stable internal message ID
                # (for example ``s001_m000``), while the service catalog keeps
                # the caller-facing message ID as its primary key. Resolve the
                # latter before deleting the session parent; otherwise SQLite
                # correctly rejects the parent deletion with a foreign-key
                # violation and leaves the asynchronous deletion failed.
                service_message_ids: set[str] = set()
                if "tmcra_service_messages" in tables:
                    service_columns = {
                        str(row[1])
                        for row in connection.execute(
                            "PRAGMA table_info(tmcra_service_messages)"
                        ).fetchall()
                    }
                    predicates = [f"message_id IN ({message_placeholders})"]
                    service_lookup_parameters: tuple[Any, ...] = message_parameters
                    if "internal_message_id" in service_columns:
                        predicates.append(
                            f"internal_message_id IN ({message_placeholders})"
                        )
                        service_lookup_parameters = (
                            paths.scope_id,
                            *sorted(message_ids),
                            *sorted(message_ids),
                        )
                    service_message_ids.update(
                        str(row[0])
                        for row in connection.execute(
                            "SELECT message_id FROM tmcra_service_messages "
                            "WHERE scope_id=? AND (" + " OR ".join(predicates) + ")",
                            service_lookup_parameters,
                        ).fetchall()
                    )
                if service_message_ids:
                    service_placeholders = ",".join("?" for _ in service_message_ids)
                    service_parameters = (
                        paths.scope_id,
                        *sorted(service_message_ids),
                    )
                    for table in (
                        "tmcra_service_message_actor_provenance",
                        "tmcra_service_messages",
                    ):
                        if table in tables:
                            connection.execute(
                                f"DELETE FROM {table} WHERE scope_id=? "
                                f"AND message_id IN ({service_placeholders})",
                                service_parameters,
                            )
                for table in (
                    "v4_source_journal",
                    "v4_interactions",
                    "v4_message_commit_journal",
                ):
                    if table in tables:
                        connection.execute(
                            f"DELETE FROM {table} WHERE scope_id=? "
                            f"AND message_id IN ({message_placeholders})",
                            message_parameters,
                        )
                if "v4_reconciliation_jobs" in tables:
                    connection.execute(
                        f"DELETE FROM v4_reconciliation_jobs WHERE scope_id=? "
                        f"AND message_id IN ({message_placeholders})",
                        message_parameters,
                    )
            if clean_session:
                for table in ("tmcra_service_batches", "v4_batch_journal"):
                    if table in tables:
                        connection.execute(
                            f"DELETE FROM {table} WHERE scope_id=? AND session_id=?",
                            (paths.scope_id, clean_session),
                        )
                if "tmcra_service_sessions" in tables:
                    connection.execute(
                        "DELETE FROM tmcra_service_sessions "
                        "WHERE scope_id=? AND session_id=?",
                        (paths.scope_id, clean_session),
                    )

            if "meta" in tables:
                current = connection.execute(
                    "SELECT value_json FROM meta WHERE scope_id=? AND key='storage_revision'",
                    (paths.scope_id,),
                ).fetchone()
                revision = 0
                if current is not None:
                    try:
                        revision = int(json.loads(str(current["value_json"])))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        revision = 0
                connection.execute(
                    "INSERT INTO meta(scope_id,key,value_json) VALUES(?,?,?) "
                    "ON CONFLICT(scope_id,key) DO UPDATE SET value_json=excluded.value_json",
                    (paths.scope_id, "storage_revision", json.dumps(revision + 1)),
                )
                deletion_journal = {
                    "job_id": job_id,
                    "mode": "session" if clean_session else "memory_ids",
                    "request_sha256": hashlib.sha256(
                        json.dumps(
                            {
                                "memory_ids": sorted(explicit_ids),
                                "session_id": clean_session or None,
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                    "result": {
                        "deleted_memory_count": int(deleted_record_count),
                        "deleted_message_count": len(message_ids),
                        "invalidated_slow_memory_count": len(slow_ids),
                        "deleted_source_record_ids": sorted(deleted_source_ids),
                        "deleted_session_message_counts": deleted_session_message_counts,
                    },
                    "completed_at": time.time(),
                }
                connection.execute(
                    "INSERT INTO meta(scope_id,key,value_json) VALUES(?,?,?) "
                    "ON CONFLICT(scope_id,key) DO UPDATE SET value_json=excluded.value_json",
                    (
                        paths.scope_id,
                        f"content_deletion:{job_id}",
                        json.dumps(
                            deletion_journal,
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                )
            connection.commit()
            quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            if quick_check.lower() != "ok":
                raise V4AdapterError("scope database failed SQLite quick_check")
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

        self._invalidate_generation_validation(paths.active_index)
        self._invalidate_generation_validation(paths.active_delta)
        try:
            paths.active_delta.unlink()
        except FileNotFoundError:
            pass
        return {
            "scope_id": paths.scope_id,
            "mode": "session" if clean_session else "memory_ids",
            "requested_memory_count": len(explicit_ids),
            "matched_source_memory_count": len(source_ids_for_session),
            "deleted_memory_count": int(deleted_record_count),
            "deleted_message_count": len(message_ids),
            "invalidated_slow_memory_count": len(slow_ids),
            "slow_rebuild_required": bool(slow_ids),
            "job_id": job_id,
            "_deleted_source_record_ids": sorted(deleted_source_ids),
            "_deleted_message_ids": sorted(message_ids),
            "_deleted_session_message_counts": deleted_session_message_counts,
        }

    def content_deletion_commit(
        self, *, tenant_id: str, scope_name: str, job_id: str
    ) -> dict[str, Any] | None:
        paths = self.scope_paths(tenant_id, scope_name)
        if not paths.database.is_file():
            return None
        try:
            with closing(sqlite3.connect(paths.database, timeout=30.0)) as connection:
                connection.row_factory = sqlite3.Row
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                if "meta" not in tables:
                    return None
                row = connection.execute(
                    "SELECT value_json FROM meta WHERE scope_id=? AND key=?",
                    (paths.scope_id, f"content_deletion:{job_id}"),
                ).fetchone()
        except (OSError, sqlite3.DatabaseError):
            return None
        if row is None:
            return None
        try:
            value = json.loads(str(row["value_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def compatibility(self) -> dict[str, bool]:
        benchmark_writer = self.settings.v4_root / "tmcra_v4_batch_writer.py"
        production_writer = self.settings.v4_root / "tmcra_service" / "writer.py"
        try:
            verify_shared_core(self.settings.v4_root)
            shared_core_matches = True
        except SharedCoreVerificationError:
            shared_core_matches = False
        return {
            "benchmark_writer_exists": benchmark_writer.is_file(),
            "production_writer_exists": production_writer.is_file(),
            "shared_core_manifest_matches": shared_core_matches,
        }

    def _require_compatible_writer(self) -> None:
        status = self.compatibility()
        missing = [name for name, ready in status.items() if not ready]
        if missing:
            raise V4AdapterError(
                "V4 writer lacks production incremental contracts: " + ",".join(missing)
            )

    def _journal_provider_metadata(
        self,
        values: Sequence[Mapping[str, Any]],
        *,
        tenant_id: str,
        scope_name: str,
        job_id: str | None,
        stage_id: str | None,
        operation: str,
        default_model: str,
        usage_attribution: UsageAttribution = UNATTRIBUTED,
    ) -> int:
        if stage_id is None:
            return 0
        store = JobStore(ControlDB(self.settings.control_db))
        return sum(
            journal_deepseek_calls(
                store,
                value,
                tenant_id=tenant_id,
                scope_name=scope_name,
                job_id=job_id,
                stage_id=stage_id,
                operation=operation,
                default_model=default_model,
                usage_attribution=usage_attribution,
            )
            for value in values
        )

    @staticmethod
    def _slow_call_metadata(database: Path, scope_id: str) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        with closing(sqlite3.connect(database)) as connection:
            connection.row_factory = sqlite3.Row
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='slow_graph_attempts'"
            ).fetchone()
            if exists is None:
                return []
            rows = connection.execute(
                "SELECT call_metadata_json FROM slow_graph_attempts WHERE scope_id=?",
                (scope_id,),
            ).fetchall()
        for row in rows:
            try:
                value = json.loads(str(row["call_metadata_json"] or "{}"))
            except json.JSONDecodeError as exc:
                raise V4AdapterError("slow graph call metadata is invalid JSON") from exc
            if isinstance(value, dict):
                values.append(value)
        return values

    def _run_with_writer_env(
        self,
        command: Sequence[str],
        *,
        log_path: Path,
        timeout: float | None = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        shell = 'set -a; source "$1"; shift; exec "$@"'
        environment = dict(os.environ)
        environment["TMCRA_SERVICE_CONTROL_DB"] = str(self.settings.control_db)
        environment["TMCRA_PROVIDER_KEY_CONCURRENCY"] = str(
            self.settings.provider_key_concurrency
        )
        environment["TMCRA_PROVIDER_LEASE_SECONDS"] = str(
            self.settings.provider_lease_seconds
        )
        python_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(self.settings.v4_root) + (
            os.pathsep + python_path if python_path else ""
        )
        for key, value in (extra_env or {}).items():
            key = str(key)
            if not key or "=" in key:
                raise V4AdapterError("writer extra environment keys must be valid names")
            environment[key] = str(value)
        from tmcra_local_only import enabled, validate_environment
        if enabled(environment):
            validate_environment(environment)
            invocation = list(command)
        else:
            invocation = ["bash", "-c", shell, "tmcra-service", str(self.settings.writer_env), *command]
        with log_path.open("x", encoding="utf-8") as log:
            log.write(json.dumps({"command": [str(item) for item in command]}) + "\n")
            log.flush()
            subprocess.run(
                invocation,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env=environment,
            )

    def ingest(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
        job_id: str,
        stage_id: str | None = None,
        stage_attempt: int = 1,
        usage_attribution: UsageAttribution = UNATTRIBUTED,
        provider_execution: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_compatible_writer()
        try:
            provider_execution = normalize_user_provider_execution(
                provider_execution,
                stage="writer",
            )
        except ValueError as exc:
            raise V4AdapterError(str(exc)) from exc
        stage_id = stage_id or f"{job_id}:writer"
        if (
            stage_id != f"{job_id}:writer"
            or isinstance(stage_attempt, bool)
            or not isinstance(stage_attempt, int)
            or stage_attempt <= 0
        ):
            raise V4AdapterError("writer stage identity or attempt is invalid")
        paths = self.scope_paths(tenant_id, scope_name)
        operation = paths.operations / job_id
        payload = [
            {
                "scope_id": paths.scope_id,
                "question_id": paths.question_id,
                "session_id": session_id,
                "operation_id": job_id,
                "messages": [dict(item) for item in messages],
            }
        ]
        recovery_mode = "none"
        if operation.exists():
            report = operation / "product_writer_report.json"
            if report.is_file():
                cached_report = json.loads(report.read_text(encoding="utf-8"))
                if not isinstance(cached_report, Mapping):
                    raise V4AdapterError("writer report must be an object")
                self._validate_writer_report(
                    cached_report,
                    paths=paths,
                    job_id=job_id,
                    stage_id=stage_id,
                )
                reported_attempt = cached_report.get("stage_attempt")
                if (
                    isinstance(reported_attempt, int)
                    and not isinstance(reported_attempt, bool)
                    and reported_attempt > stage_attempt
                ):
                    raise V4AdapterError("writer report belongs to a future stage attempt")
                if self._writer_report_is_complete(cached_report):
                    self._validate_complete_writer_artifacts(
                        cached_report,
                        paths=paths,
                        operation=operation,
                        expected_payload=payload,
                    )
                    commit_path = operation / "commit.json"
                    if commit_path.is_file():
                        try:
                            commit = json.loads(commit_path.read_text(encoding="utf-8"))
                        except json.JSONDecodeError as exc:
                            raise V4AdapterError("writer commit is not valid JSON") from exc
                        if not isinstance(commit, Mapping) or not self._valid_ingest_commit(
                            commit, paths=paths, job_id=job_id
                        ):
                            raise V4AdapterError("writer commit identity validation failed")
                    else:
                        _atomic_json(
                            commit_path,
                            {
                                "schema_version": "tmcra.service.ingest-commit.1",
                                "job_id": job_id,
                                "tenant_id": tenant_id,
                                "scope_id": paths.scope_id,
                                "database": str(paths.database),
                                "completed_at": time.time(),
                            },
                        )
                    return dict(cached_report)
                if (operation / "commit.json").exists():
                    raise V4AdapterError("incomplete writer report has a stale commit")
                if stage_attempt <= 1:
                    raise V4AdapterError("incomplete writer report requires a new stage attempt")
                if not self._writer_report_is_explicitly_degraded(cached_report):
                    raise V4AdapterError(
                        "writer report is neither strictly complete nor explicitly degraded"
                    )
                classified_recovery = self._incomplete_ingest_recovery_mode(
                    paths=paths,
                    operation=operation,
                    job_id=job_id,
                    expected_payload=payload,
                )
                if classified_recovery is None:
                    raise V4AdapterError(
                        f"incomplete ingest operation requires artifact audit: {job_id}"
                    )
                recovery_mode = classified_recovery
                if recovery_mode == "audited_writer_state":
                    self._prepare_audited_writer_retry(
                        paths=paths,
                        operation=operation,
                        job_id=job_id,
                    )
                elif recovery_mode == "definitive_provider_failure":
                    self._prepare_definitive_reviewer_retry(
                        paths=paths,
                        operation=operation,
                        job_id=job_id,
                    )
                elif recovery_mode == "definitive_invalid_response":
                    self._prepare_definitive_invalid_response_retry(
                        paths=paths,
                        operation=operation,
                        job_id=job_id,
                    )
                elif recovery_mode == "schema_constrained_invalid_response":
                    self._prepare_schema_constrained_invalid_response_retry(
                        paths=paths,
                        operation=operation,
                        job_id=job_id,
                    )
                elif recovery_mode == "schema_constrained_invalid_response_prepared":
                    self._validate_prepared_schema_constrained_retry(
                        paths=paths,
                        operation=operation,
                        job_id=job_id,
                    )
                elif recovery_mode == "audited_local_inference_cancelled":
                    self._prepare_cancelled_local_inference_retry(
                        paths=paths,
                        operation=operation,
                        job_id=job_id,
                    )
                self._archive_writer_report(
                    report,
                    prior_attempt=int(
                        cached_report.get("stage_attempt") or stage_attempt - 1
                    ),
                )
            else:
                classified_recovery = self._incomplete_ingest_recovery_mode(
                    paths=paths,
                    operation=operation,
                    job_id=job_id,
                    expected_payload=payload,
                )
                if classified_recovery is None:
                    raise V4AdapterError(
                        f"incomplete ingest operation requires artifact audit: {job_id}"
                    )
                recovery_mode = classified_recovery
                if recovery_mode == "audited_writer_state":
                    self._prepare_audited_writer_retry(
                        paths=paths,
                        operation=operation,
                        job_id=job_id,
                    )
                elif recovery_mode == "definitive_provider_failure":
                    self._prepare_definitive_reviewer_retry(
                        paths=paths,
                        operation=operation,
                        job_id=job_id,
                    )
                elif recovery_mode == "definitive_invalid_response":
                    self._prepare_definitive_invalid_response_retry(
                        paths=paths,
                        operation=operation,
                        job_id=job_id,
                    )
                elif recovery_mode == "schema_constrained_invalid_response":
                    self._prepare_schema_constrained_invalid_response_retry(
                        paths=paths,
                        operation=operation,
                        job_id=job_id,
                    )
                elif recovery_mode == "schema_constrained_invalid_response_prepared":
                    self._validate_prepared_schema_constrained_retry(
                        paths=paths,
                        operation=operation,
                        job_id=job_id,
                    )
                elif recovery_mode == "audited_local_inference_cancelled":
                    self._prepare_cancelled_local_inference_retry(
                        paths=paths,
                        operation=operation,
                        job_id=job_id,
                    )
        else:
            operation.mkdir(parents=True, exist_ok=False)
            paths.database.parent.mkdir(parents=True, exist_ok=True)
            input_path = operation / "input.json"
            input_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        input_path = operation / "input.json"
        if self.settings.writer_execution_mode == "resident":
            if self._writer_pool is None or not self._writer_pool.status().alive:
                raise V4AdapterError("resident Writer pool is not ready")
            log_path = self._next_operation_log(operation, "writer")
            started = time.monotonic()
            with log_path.open("x", encoding="utf-8") as log:
                log.write(
                    json.dumps(
                        {
                            "mode": "resident",
                            "operation_id": job_id,
                            "writer_pool_size": self.settings.writer_pool_size,
                            "stage_id": stage_id,
                            "stage_attempt": stage_attempt,
                            "recovery_mode": recovery_mode,
                            "usage_attribution": usage_attribution.as_dict(),
                            "provider_execution": provider_execution,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                log.flush()
                try:
                    response = self._writer_pool.execute(
                        {
                            "input_path": str(input_path),
                            "out_dir": str(operation),
                            "database": str(paths.database),
                            "operation_id": job_id,
                            "tenant_id": tenant_id,
                            "scope_name": scope_name,
                            "job_id": job_id,
                            "stage_id": stage_id,
                            "stage_attempt": stage_attempt,
                            "timeout_seconds": 180.0,
                            "max_tokens": 16384,
                            "recovery_mode": recovery_mode,
                            "usage_attribution": usage_attribution.as_dict(),
                            "provider_execution": provider_execution,
                        },
                        operation_timeout=max(
                            self.settings.writer_pool_request_timeout_seconds,
                            min(7200.0, 300.0 + 4.0 * len(messages)),
                        ),
                    )
                except Exception as exc:
                    log.write(
                        json.dumps(
                            {
                                "status": "failed",
                                "error_type": type(exc).__name__,
                                "elapsed_seconds": round(time.monotonic() - started, 6),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    log.flush()
                    raise
                log.write(
                    json.dumps(
                        {
                            "status": "complete",
                            "worker_pid": response.get("pid"),
                            "writer_elapsed_seconds": response.get("elapsed_seconds"),
                            "elapsed_seconds": round(time.monotonic() - started, 6),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        else:
            self._run_with_writer_env(
                [
                    str(self.python),
                    "-m",
                    "tmcra_service.writer",
                    "--input",
                    str(input_path),
                    "--out-dir",
                    str(operation),
                    "--database",
                    str(paths.database),
                    "--operation-id",
                    job_id,
                    "--repo",
                    str(self.settings.integrated_repo),
                    "--max-tokens",
                    "16384",
                    "--recovery-mode",
                    recovery_mode,
                    "--stage-attempt",
                    str(stage_attempt),
                    *(
                        [
                            "--provider-execution-json",
                            json.dumps(
                                provider_execution,
                                ensure_ascii=True,
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                        ]
                        if provider_execution is not None
                        else []
                    ),
                ],
                log_path=self._next_operation_log(operation, "writer"),
                extra_env={
                    "TMCRA_SERVICE_TENANT_ID": tenant_id,
                    "TMCRA_SERVICE_SCOPE_NAME": scope_name,
                    "TMCRA_SERVICE_JOB_ID": job_id,
                    "TMCRA_SERVICE_STAGE_ID": stage_id,
                    "TMCRA_SERVICE_STAGE_ATTEMPT": str(stage_attempt),
                    "TMCRA_USAGE_ATTRIBUTION_JSON": json.dumps(
                        usage_attribution.as_dict(),
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                },
            )
        report_path = operation / "product_writer_report.json"
        if not report_path.is_file():
            raise V4AdapterError(f"writer completed without a report: {job_id}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping):
            raise V4AdapterError("writer report must be an object")
        self._validate_writer_report(
            report,
            paths=paths,
            job_id=job_id,
            stage_id=stage_id,
            stage_attempt=stage_attempt,
        )
        if not self._writer_report_is_complete(report):
            return dict(report)
        self._validate_complete_writer_artifacts(
            report,
            paths=paths,
            operation=operation,
            expected_payload=payload,
        )
        _atomic_json(
            operation / "commit.json",
            {
                "schema_version": "tmcra.service.ingest-commit.1",
                "job_id": job_id,
                "tenant_id": tenant_id,
                "scope_id": paths.scope_id,
                "database": str(paths.database),
                "completed_at": time.time(),
            },
        )
        return dict(report)

    def ingest_recovery_plan(
        self, *, tenant_id: str, scope_name: str, job_id: str
    ) -> dict[str, Any]:
        """Classify a failed ingest before the recovery controller schedules it.

        ``parallel_safe`` is deliberately narrower than ``resumable``.  It is
        true only when the recovery can finish from already durable local
        artifacts without issuing another provider call.  The controller uses
        this distinction to parallelize deterministic repairs while keeping
        provider retries ordered within a scope.
        """

        paths = self.scope_paths(tenant_id, scope_name)
        operation = paths.operations / job_id
        report_path = operation / "product_writer_report.json"
        commit_path = operation / "commit.json"
        expected_payload: list[Mapping[str, Any]] | None = None
        input_path = operation / "input.json"
        if input_path.is_file():
            try:
                value = json.loads(input_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = None
            if isinstance(value, list) and all(
                isinstance(item, Mapping) for item in value
            ):
                expected_payload = list(value)
        preclassified_mode: str | None = None
        if self._operation_has_provider_outcome_unknown(
            paths=paths, job_id=job_id
        ):
            preclassified_mode = self._incomplete_ingest_recovery_mode(
                paths=paths,
                operation=operation,
                job_id=job_id,
                expected_payload=expected_payload,
            )
            if preclassified_mode != "audited_local_inference_cancelled":
                return self._blocked_ingest_recovery_plan(
                    "provider_outcome_unknown"
                )
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return self._blocked_ingest_recovery_plan("writer_report_invalid")
            if not isinstance(report, dict):
                return self._blocked_ingest_recovery_plan("writer_report_invalid")
            try:
                self._validate_writer_report(report, paths=paths, job_id=job_id)
            except V4AdapterError:
                return self._blocked_ingest_recovery_plan(
                    "writer_report_identity_mismatch"
                )
            complete = self._writer_report_is_complete(report)
            if commit_path.is_file():
                if not complete:
                    return self._blocked_ingest_recovery_plan(
                        "stale_ingest_commit"
                    )
                try:
                    if expected_payload is None:
                        return self._blocked_ingest_recovery_plan(
                            "ingest_input_invalid"
                        )
                    self._validate_complete_writer_artifacts(
                        report,
                        paths=paths,
                        operation=operation,
                        expected_payload=expected_payload,
                    )
                except V4AdapterError:
                    return self._blocked_ingest_recovery_plan(
                        "writer_artifact_validation_failed"
                    )
                try:
                    commit = json.loads(commit_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    commit = None
                if not (
                    isinstance(commit, Mapping)
                    and self._valid_ingest_commit(
                        commit, paths=paths, job_id=job_id
                    )
                ):
                    return self._blocked_ingest_recovery_plan(
                        "ingest_commit_invalid"
                    )
                return {
                    "resumable": True,
                    "mode": "committed_writer_artifacts",
                    "parallel_safe": True,
                    "external_api_calls_expected": False,
                    "deterministic_local_repair": True,
                    "recovery_fingerprint": self._ingest_recovery_fingerprint(
                        paths=paths,
                        operation=operation,
                        job_id=job_id,
                        mode="committed_writer_artifacts",
                    ),
                }
            if complete:
                try:
                    if expected_payload is None:
                        return self._blocked_ingest_recovery_plan(
                            "ingest_input_invalid"
                        )
                    self._validate_complete_writer_artifacts(
                        report,
                        paths=paths,
                        operation=operation,
                        expected_payload=expected_payload,
                    )
                except V4AdapterError:
                    return self._blocked_ingest_recovery_plan(
                        "writer_artifact_validation_failed"
                    )
                return {
                    "resumable": True,
                    "mode": "complete_writer_artifacts",
                    "parallel_safe": True,
                    "external_api_calls_expected": False,
                    "deterministic_local_repair": True,
                    "recovery_fingerprint": self._ingest_recovery_fingerprint(
                        paths=paths,
                        operation=operation,
                        job_id=job_id,
                        mode="complete_writer_artifacts",
                    ),
                }
        elif commit_path.exists():
            return self._blocked_ingest_recovery_plan("orphan_ingest_commit")

        mode = preclassified_mode or self._incomplete_ingest_recovery_mode(
            paths=paths,
            operation=operation,
            job_id=job_id,
            expected_payload=expected_payload,
        )
        if mode is None:
            return self._blocked_ingest_recovery_plan(
                "ingest_outcome_requires_manual_audit"
            )
        deterministic_local_repair = self._ingest_recovery_is_local_only(
            paths=paths,
            operation=operation,
            job_id=job_id,
            mode=mode,
        )
        parallel_safe = self._ingest_recovery_is_parallel_safe(
            paths=paths,
            operation=operation,
            job_id=job_id,
            mode=mode,
        )
        return {
            "resumable": True,
            "mode": mode,
            "parallel_safe": parallel_safe,
            "external_api_calls_expected": not deterministic_local_repair,
            "deterministic_local_repair": deterministic_local_repair,
            "recovery_fingerprint": (
                self._ingest_recovery_fingerprint(
                    paths=paths,
                    operation=operation,
                    job_id=job_id,
                    mode=mode,
                )
                if deterministic_local_repair
                else ""
            ),
        }

    def prepare_writer_source_accounting(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        session_id: str,
        job_id: str,
        messages: Sequence[Mapping[str, Any]],
        writer: Mapping[str, Any],
        default_accounting_operation_id: str,
    ) -> dict[str, Any]:
        """Add Source accounting from exact immutable operation bindings.

        This is a deterministic local reconciliation.  It does not alter the
        historical Writer report or issue a model call.  A distinct accounting
        operation is used because a Writer process can stop after committing a
        durable Source prefix but before writing its final report.  Older
        attempts may also have committed an empty source set before provenance
        repair was available.
        """

        paths = self.scope_paths(tenant_id, scope_name)
        operation = paths.operations / job_id
        expected_payload = [
            {
                "scope_id": paths.scope_id,
                "question_id": paths.question_id,
                "session_id": session_id,
                "operation_id": job_id,
                "messages": [dict(item) for item in messages],
            }
        ]
        try:
            persisted_payload = json.loads(
                (operation / "input.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise V4AdapterError(
                "legacy Source accounting input artifact is unreadable"
            ) from exc
        if persisted_payload != expected_payload:
            raise V4AdapterError(
                "legacy Source accounting input differs from the current payload"
            )

        durable = writer.get("durable_sources", [])
        if not isinstance(durable, list) or any(
            not isinstance(item, Mapping) for item in durable
        ):
            raise V4AdapterError("writer durable Source accounting is invalid")
        merged: dict[str, dict[str, Any]] = {}
        for item in durable:
            source_record_id = str(item.get("source_record_id") or "").strip()
            normalized = {
                "source_record_id": source_record_id,
                "origin_operation_id": str(
                    item.get("origin_operation_id") or ""
                ).strip(),
                "raw_token_estimate": int(item.get("raw_token_estimate", 0) or 0),
                "user_turns": int(item.get("user_turns", 0) or 0),
            }
            if (
                not source_record_id
                or not normalized["origin_operation_id"]
                or normalized["raw_token_estimate"] < 0
                or normalized["user_turns"] not in {0, 1}
                or source_record_id in merged
            ):
                raise V4AdapterError("writer durable Source accounting is invalid")
            merged[source_record_id] = normalized

        reported_source_ids = set(merged)
        validated_reported_source_ids: set[str] = set()
        validated_source_ids: set[str] = set()
        discovered_count = 0
        legacy_candidates: dict[str, dict[str, Any]] = {}
        if not paths.database.is_file():
            if reported_source_ids:
                raise V4AdapterError("writer Source database is missing")
            return {
                "writer": dict(writer),
                "accounting_operation_id": default_accounting_operation_id,
                "legacy_source_count": 0,
                "recovered_source_count": 0,
            }

        # A complete current Writer report already names the exact durable
        # Source set. Revalidate those immutable rows and reserve the historical
        # scope scan for incomplete, legacy, and recovery inputs.
        if self._writer_report_is_complete(writer) and len(merged) == len(messages):
            self._validate_complete_writer_artifacts(
                writer,
                paths=paths,
                operation=operation,
                expected_payload=expected_payload,
            )
            return {
                "writer": dict(writer),
                "accounting_operation_id": default_accounting_operation_id,
                "legacy_source_count": 0,
                "recovered_source_count": 0,
                "source_accounting_mode": "current_operation_proof_v1",
            }
        try:
            with closing(sqlite3.connect(paths.database, timeout=30.0)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA busy_timeout=30000")
                quick = connection.execute("PRAGMA quick_check").fetchone()
                if quick is None or str(quick[0]) != "ok":
                    raise V4AdapterError("writer database failed quick_check")
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                required = {
                    "records",
                    "tmcra_service_messages",
                    "v4_source_journal",
                }
                if not required.issubset(tables):
                    raise V4AdapterError(
                        "Source accounting immutable tables are missing"
                    )
                bindings, _batch_states, violations = (
                    self._source_operation_bindings(connection, paths=paths)
                )
                if violations:
                    raise V4AdapterError(
                        "Source accounting operation binding is ambiguous"
                    )
                binding_ids = {
                    message_id
                    for message_id, operation_id in bindings.items()
                    if operation_id == job_id
                }
                graph_sources_by_message: dict[str, list[sqlite3.Row]] = {}
                for graph_row in connection.execute(
                    "SELECT memory_id,turn_index,metadata_json FROM records "
                    "WHERE scope_id=? AND category='source'",
                    (paths.scope_id,),
                ).fetchall():
                    try:
                        graph_metadata = json.loads(
                            str(graph_row["metadata_json"] or "{}")
                        )
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(graph_metadata, Mapping):
                        continue
                    graph_message_id = str(
                        graph_metadata.get("message_id") or ""
                    ).strip()
                    if graph_message_id:
                        graph_sources_by_message.setdefault(
                            graph_message_id, []
                        ).append(graph_row)
                journal_bound_source_ids = {
                    str(row[0] or "").strip()
                    for row in connection.execute(
                        "SELECT source_record_id FROM v4_source_journal "
                        "WHERE scope_id=? AND source_record_id!=''",
                        (paths.scope_id,),
                    ).fetchall()
                }
                duplicated_journal_source = connection.execute(
                    "SELECT source_record_id FROM v4_source_journal "
                    "WHERE scope_id=? AND source_record_id!='' "
                    "GROUP BY source_record_id HAVING COUNT(*)>1 LIMIT 1",
                    (paths.scope_id,),
                ).fetchone()
                if duplicated_journal_source is not None:
                    raise V4AdapterError(
                        "Source accounting journal graph binding is duplicated"
                    )
                seen_external_ids: set[str] = set()
                seen_internal_ids: set[str] = set()
                for message in messages:
                    external_id = str(message.get("message_id") or "").strip()
                    content = str(message.get("content") or "")
                    if not external_id or external_id in seen_external_ids:
                        raise V4AdapterError(
                            "Source accounting input message identity is invalid"
                        )
                    seen_external_ids.add(external_id)
                    service = connection.execute(
                        "SELECT internal_message_id,session_id,role,timestamp,"
                        "content_sha256,first_operation_id,message_index "
                        "FROM tmcra_service_messages WHERE scope_id=? AND message_id=?",
                        (paths.scope_id, external_id),
                    ).fetchone()
                    # The Writer may stop before registering the tail of its
                    # input. Only rows that crossed the Source boundary are
                    # candidates for local accounting.
                    if service is None:
                        continue
                    internal_id = str(service[0] or "").strip()
                    expected_identity = (
                        session_id,
                        str(message.get("role") or "").strip().lower(),
                        str(message.get("timestamp") or "").strip(),
                        hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    )
                    if (
                        not internal_id
                        or internal_id in seen_internal_ids
                        or tuple(service[1:5]) != expected_identity
                    ):
                        raise V4AdapterError(
                            "Source accounting service identity changed"
                        )
                    seen_internal_ids.add(internal_id)
                    first_operation_id = str(service[5] or "").strip()
                    cross_job_replay = bool(
                        first_operation_id and first_operation_id != job_id
                    )
                    if (
                        not cross_job_replay
                        and str(bindings.get(internal_id) or "") != job_id
                    ):
                        raise V4AdapterError(
                            "Source accounting lacks a unique operation binding"
                        )

                    if "tmcra_service_message_actor_provenance" in tables:
                        try:
                            actor = normalize_message_actor_metadata(
                                message.get("role"), message.get("metadata")
                            )
                        except ActorProvenanceError as exc:
                            raise V4AdapterError(
                                "Source accounting actor provenance is invalid"
                            ) from exc
                        actor_row = connection.execute(
                            "SELECT actor_metadata_json,actor_metadata_sha256 "
                            "FROM tmcra_service_message_actor_provenance "
                            "WHERE scope_id=? AND message_id=?",
                            (paths.scope_id, external_id),
                        ).fetchone()
                        if actor_row is None:
                            if first_operation_id:
                                raise V4AdapterError(
                                    "Source accounting actor provenance is missing"
                                )
                        elif tuple(actor_row) != (
                            actor_metadata_json(actor),
                            actor_metadata_sha256(actor),
                        ):
                            raise V4AdapterError(
                                "Source accounting actor provenance changed"
                            )
                    elif first_operation_id:
                        raise V4AdapterError(
                            "Source accounting actor provenance table is missing"
                        )

                    source = connection.execute(
                        "SELECT session_id,session_index,message_index,message_role,"
                        "timestamp,content,content_sha256,status,source_record_id,"
                        "source_turn_index,source_persisted_at "
                        "FROM v4_source_journal WHERE scope_id=? AND message_id=?",
                        (paths.scope_id, internal_id),
                    ).fetchone()
                    if source is None:
                        continue
                    source_identity = (
                        session_id,
                        str(message.get("role") or "").strip().lower(),
                        str(message.get("timestamp") or "").strip(),
                        content,
                        hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    )
                    if (
                        str(source[0] or ""),
                        str(source[3] or ""),
                        str(source[4] or ""),
                        str(source[5] or ""),
                        str(source[6] or ""),
                    ) != source_identity or int(
                        service[6] if service[6] is not None else -1
                    ) != int(source[2]):
                        raise V4AdapterError(
                            "Source accounting journal identity changed"
                        )
                    status = str(source[7] or "")
                    source_record_id = str(source[8] or "").strip()
                    if status not in {"pending", "enriched", "failed"}:
                        raise V4AdapterError(
                            "Source accounting journal status is invalid"
                        )
                    if status in {"pending", "failed"} and not source_record_id:
                        # The graph transaction precedes the journal binding.
                        # A process can stop in between, leaving a real Source
                        # with an empty journal ID. Accept only one exact graph
                        # candidate for this immutable message.
                        graph_candidates = graph_sources_by_message.get(
                            internal_id, []
                        )
                        if not graph_candidates:
                            continue
                        if len(graph_candidates) != 1:
                            raise V4AdapterError(
                                "Source accounting graph binding is ambiguous"
                            )
                        graph_candidate = graph_candidates[0]
                        source_record_id = str(
                            graph_candidate["memory_id"] or ""
                        ).strip()
                        if (
                            not source_record_id
                            or source_record_id in journal_bound_source_ids
                        ):
                            raise V4AdapterError(
                                "Source accounting graph binding is ambiguous"
                            )
                        try:
                            graph_metadata = json.loads(
                                str(graph_candidate["metadata_json"] or "{}")
                            )
                        except json.JSONDecodeError as exc:
                            raise V4AdapterError(
                                "Source accounting graph metadata is invalid"
                            ) from exc
                        graph_sidecar = (
                            graph_metadata.get("sidecar_hint_metadata")
                            if isinstance(graph_metadata, Mapping)
                            else None
                        )
                        graph_sidecar = (
                            graph_sidecar
                            if isinstance(graph_sidecar, Mapping)
                            else {}
                        )
                        graph_actor_role = (
                            str(
                                graph_metadata.get("actor_role")
                                or graph_metadata.get("speaker")
                                or graph_sidecar.get("role")
                                or ""
                            )
                            if isinstance(graph_metadata, Mapping)
                            else ""
                        )
                        try:
                            graph_turn_index = int(graph_candidate["turn_index"])
                            graph_session_index = int(
                                graph_metadata.get("session_index", -1)
                            )
                            graph_message_index = int(
                                graph_metadata.get("message_index", -1)
                            )
                        except (TypeError, ValueError) as exc:
                            raise V4AdapterError(
                                "Source accounting graph location is invalid"
                            ) from exc
                        expected_slot = (
                            f"source.s{int(source[1]):03d}.m{int(source[2]):03d}"
                        )
                        if (
                            not isinstance(graph_metadata, Mapping)
                            or graph_metadata.get("raw_content") != content
                            or graph_metadata.get("source_span") != content
                            or graph_metadata.get("source_turn_text") != content
                            or str(graph_metadata.get("content_variant") or "")
                            != "source_message"
                            or str(graph_metadata.get("source_record_id") or "")
                            != source_record_id
                            or source_record_id
                            != f"{expected_slot}:{graph_turn_index}"
                            or str(graph_metadata.get("canonical_slot_key") or "")
                            != expected_slot
                            or str(graph_metadata.get("message_id") or "")
                            != internal_id
                            or str(graph_metadata.get("session_id") or "")
                            != session_id
                            or graph_session_index != int(source[1])
                            or graph_message_index != int(source[2])
                            or str(graph_metadata.get("timestamp") or "")
                            != str(message.get("timestamp") or "").strip()
                            or graph_actor_role
                            != str(message.get("role") or "").strip().lower()
                        ):
                            raise V4AdapterError(
                                "Source accounting graph metadata changed"
                            )
                        if source_record_id in validated_source_ids:
                            raise V4AdapterError(
                                "Source accounting graph identity is duplicated"
                            )
                        candidate = {
                            "source_record_id": source_record_id,
                            "origin_operation_id": job_id,
                            "raw_token_estimate": _raw_token_estimate(content),
                            "user_turns": int(
                                str(message.get("role") or "").strip().lower()
                                == "user"
                            ),
                        }
                        merged[source_record_id] = candidate
                        discovered_count += 1
                        validated_source_ids.add(source_record_id)
                        if not first_operation_id:
                            legacy_candidates[source_record_id] = candidate
                        continue
                    if not source_record_id or not str(source[10] or "").strip():
                        raise V4AdapterError(
                            "Source accounting lacks a durable graph binding"
                        )
                    if source_record_id in validated_source_ids:
                        raise V4AdapterError(
                            "Source accounting graph identity is duplicated"
                        )
                    record = connection.execute(
                        "SELECT category,turn_index,metadata_json FROM records "
                        "WHERE scope_id=? AND memory_id=?",
                        (paths.scope_id, source_record_id),
                    ).fetchone()
                    if record is None or str(record[0] or "") != "source":
                        raise V4AdapterError(
                            "Source accounting graph record is missing"
                        )
                    try:
                        metadata = json.loads(str(record[2] or "{}"))
                    except json.JSONDecodeError as exc:
                        raise V4AdapterError(
                            "Source accounting graph metadata is invalid"
                        ) from exc
                    sidecar = (
                        metadata.get("sidecar_hint_metadata")
                        if isinstance(metadata, Mapping)
                        else None
                    )
                    sidecar = sidecar if isinstance(sidecar, Mapping) else {}
                    metadata_message_id = (
                        str(metadata.get("message_id") or "").strip()
                        if isinstance(metadata, Mapping)
                        else ""
                    )
                    metadata_session_id = (
                        str(metadata.get("session_id") or "").strip()
                        if isinstance(metadata, Mapping)
                        else ""
                    )
                    actor_role = (
                        str(
                            metadata.get("actor_role")
                            or metadata.get("speaker")
                            or sidecar.get("role")
                            or ""
                        )
                        if isinstance(metadata, Mapping)
                        else ""
                    )
                    if (
                        not isinstance(metadata, Mapping)
                        or metadata.get("raw_content") != content
                        or str(metadata.get("source_record_id") or "")
                        != source_record_id
                        or (metadata_message_id and metadata_message_id != internal_id)
                        or (metadata_session_id and metadata_session_id != session_id)
                        or int(metadata.get("session_index", -1)) != int(source[1])
                        or int(metadata.get("message_index", -1)) != int(source[2])
                        or int(record[1] if record[1] is not None else -1)
                        != int(source[9])
                        or actor_role
                        != str(message.get("role") or "").strip().lower()
                    ):
                        raise V4AdapterError(
                            "Source accounting graph metadata changed"
                        )
                    candidate = {
                        "source_record_id": source_record_id,
                        "origin_operation_id": (
                            first_operation_id if cross_job_replay else job_id
                        ),
                        "raw_token_estimate": _raw_token_estimate(content),
                        "user_turns": int(
                            str(message.get("role") or "").strip().lower()
                            == "user"
                        ),
                    }
                    prior = merged.get(source_record_id)
                    if prior is not None and prior != candidate:
                        raise V4AdapterError(
                            "Source accounting conflicts with Writer provenance"
                        )
                    if prior is None:
                        merged[source_record_id] = candidate
                        discovered_count += 1
                    else:
                        validated_reported_source_ids.add(source_record_id)
                    validated_source_ids.add(source_record_id)
                    if not first_operation_id:
                        legacy_candidates[source_record_id] = candidate
                unexpected_binding_ids = binding_ids - seen_internal_ids
                if unexpected_binding_ids:
                    raise V4AdapterError(
                        "Source accounting input omits operation-bound messages"
                    )
        except (sqlite3.DatabaseError, TypeError, ValueError) as exc:
            raise V4AdapterError(
                "Source accounting database validation failed"
            ) from exc

        if reported_source_ids - validated_reported_source_ids:
            raise V4AdapterError(
                "writer durable Source accounting lacks immutable proof"
            )
        if not discovered_count:
            return {
                "writer": dict(writer),
                "accounting_operation_id": default_accounting_operation_id,
                "legacy_source_count": 0,
                "recovered_source_count": 0,
            }

        legacy_accounted_count = 0
        try:
            with closing(
                sqlite3.connect(self.settings.control_db, timeout=30.0)
            ) as control:
                control.row_factory = sqlite3.Row
                control.execute("PRAGMA busy_timeout=30000")
                quick = control.execute("PRAGMA quick_check").fetchone()
                if quick is None or str(quick[0]) != "ok":
                    raise V4AdapterError("control database failed quick_check")
                rows = control.execute(
                    "SELECT commits.operation_id,commits.new_message_count,"
                    "sets.source_count,commits.raw_token_estimate,commits.user_turns "
                    "FROM scope_ingest_watermark_commits AS commits "
                    "LEFT JOIN scope_ingest_source_sets AS sets "
                    "ON sets.tenant_id=commits.tenant_id "
                    "AND sets.scope_name=commits.scope_name "
                    "AND sets.operation_id=commits.operation_id "
                    "WHERE commits.tenant_id=? AND commits.scope_name=? "
                    "AND (commits.operation_id=? OR commits.operation_id LIKE ?)",
                    (
                        tenant_id,
                        scope_name,
                        job_id,
                        f"{job_id}:writer:attempt:%",
                    ),
                ).fetchall()
                legacy_rows = [
                    row
                    for row in rows
                    if row[2] is None and int(row[1] or 0) > 0
                ]
                if legacy_rows:
                    if len(legacy_rows) != 1 or not legacy_candidates:
                        raise V4AdapterError(
                            "legacy Source accounting watermark is ambiguous"
                        )
                    legacy_metrics = (
                        len(legacy_candidates),
                        sum(
                            int(item["raw_token_estimate"])
                            for item in legacy_candidates.values()
                        ),
                        sum(
                            int(item["user_turns"])
                            for item in legacy_candidates.values()
                        ),
                    )
                    committed_metrics = (
                        int(legacy_rows[0][1] or 0),
                        int(legacy_rows[0][3] or 0),
                        int(legacy_rows[0][4] or 0),
                    )
                    if committed_metrics != legacy_metrics:
                        raise V4AdapterError(
                            "legacy Source accounting watermark metrics changed"
                        )
                    for source_record_id in legacy_candidates:
                        merged.pop(source_record_id, None)
                    legacy_accounted_count = len(legacy_candidates)

                source_ids = sorted(merged)
                for offset in range(0, len(source_ids), 500):
                    chunk = source_ids[offset : offset + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    committed_rows = control.execute(
                        "SELECT source_record_id,origin_operation_id,"
                        "raw_token_estimate,user_turns "
                        "FROM scope_source_event_commits WHERE tenant_id=? "
                        "AND scope_name=? AND source_record_id IN "
                        f"({placeholders})",
                        (tenant_id, scope_name, *chunk),
                    ).fetchall()
                    for row in committed_rows:
                        source_record_id = str(row[0] or "")
                        candidate = merged.get(source_record_id)
                        if candidate is None:
                            continue
                        committed_identity = (
                            str(row[1] or ""),
                            int(row[2] or 0),
                            int(row[3] or 0),
                        )
                        candidate_identity = (
                            str(candidate["origin_operation_id"]),
                            int(candidate["raw_token_estimate"]),
                            int(candidate["user_turns"]),
                        )
                        if committed_identity != candidate_identity:
                            raise V4AdapterError(
                                "committed Source accounting metadata changed"
                            )
                        merged.pop(source_record_id, None)
        except sqlite3.DatabaseError as exc:
            raise V4AdapterError(
                "Source control accounting validation failed"
            ) from exc

        durable_sources = [merged[key] for key in sorted(merged)]
        if not durable_sources:
            return {
                "writer": {
                    **dict(writer),
                    "durable_sources": [],
                    "durable_source_count": 0,
                },
                "accounting_operation_id": default_accounting_operation_id,
                "legacy_source_count": legacy_accounted_count,
                "recovered_source_count": 0,
            }
        encoded = json.dumps(
            durable_sources,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        report = dict(writer)
        report["durable_sources"] = durable_sources
        report["durable_source_count"] = len(durable_sources)
        return {
            "writer": report,
            "accounting_operation_id": (
                f"{default_accounting_operation_id}:source-boundary-reconcile:"
                f"{hashlib.sha256(encoded).hexdigest()[:24]}"
            ),
            "legacy_source_count": legacy_accounted_count,
            "recovered_source_count": len(durable_sources),
        }

    def recover_writer_source_accounting(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        session_id: str,
        job_id: str,
        messages: Sequence[Mapping[str, Any]],
        default_accounting_operation_id: str,
    ) -> dict[str, Any]:
        """Recover only a proven durable Source prefix without a Writer report."""

        paths = self.scope_paths(tenant_id, scope_name)
        if self._operation_has_provider_outcome_unknown(paths=paths, job_id=job_id):
            raise V4AdapterError(
                "Source accounting recovery blocked by unknown provider outcome"
            )
        input_path = paths.operations / job_id / "input.json"
        if not input_path.is_file():
            return {
                "writer": {"durable_sources": [], "durable_source_count": 0},
                "accounting_operation_id": default_accounting_operation_id,
                "legacy_source_count": 0,
                "recovered_source_count": 0,
            }
        return self.prepare_writer_source_accounting(
            tenant_id=tenant_id,
            scope_name=scope_name,
            session_id=session_id,
            job_id=job_id,
            messages=messages,
            writer={"durable_sources": [], "durable_source_count": 0},
            default_accounting_operation_id=default_accounting_operation_id,
        )

    def source_accounting_recovery_plans(
        self,
        *,
        tenant_id: str,
        scope_name: str,
    ) -> list[dict[str, Any]]:
        """Plan zero-call Source ledger repairs in immutable scope order.

        Planning is read-only.  It never resumes a Writer, changes a job or
        provider-call state, or advances Slow/index watermarks.  The runtime
        applies each returned Source set through ControlDB's idempotent
        ``record_committed_source_records`` transaction.
        """

        paths = self.scope_paths(tenant_id, scope_name)
        if not paths.database.is_file():
            return []

        # A count match alone is not a set proof. Read both identities so a
        # stale or deleted control Source cannot be hidden by another graph row.
        try:
            with closing(sqlite3.connect(paths.database, timeout=30.0)) as native:
                native.execute("PRAGMA busy_timeout=30000")
                graph_source_rows = native.execute(
                    "SELECT memory_id FROM records "
                    "WHERE scope_id=? AND category='source'",
                    (paths.scope_id,),
                ).fetchall()
                graph_source_ids = {
                    str(row[0] or "").strip() for row in graph_source_rows
                }
                if "" in graph_source_ids or len(graph_source_ids) != len(
                    graph_source_rows
                ):
                    raise V4AdapterError(
                        "Source accounting graph identity set is invalid"
                    )
                graph_source_count = len(graph_source_ids)
        except sqlite3.DatabaseError as exc:
            raise V4AdapterError(
                "Source accounting recovery graph inventory failed"
            ) from exc

        plans: list[dict[str, Any]] = []
        try:
            with closing(
                sqlite3.connect(self.settings.control_db, timeout=30.0)
            ) as control:
                control.row_factory = sqlite3.Row
                control.execute("PRAGMA busy_timeout=30000")
                quick = control.execute("PRAGMA quick_check").fetchone()
                if quick is None or str(quick[0]) != "ok":
                    raise V4AdapterError("control database failed quick_check")
                state = control.execute(
                    "SELECT source_event_seq FROM scope_evolution_state "
                    "WHERE tenant_id=? AND scope_name=?",
                    (tenant_id, scope_name),
                ).fetchone()
                control_source_count = (
                    0 if state is None else int(state["source_event_seq"] or 0)
                )
                control_source_ids = {
                    str(row[0] or "").strip()
                    for row in control.execute(
                        "SELECT source_record_id FROM scope_source_event_commits "
                        "WHERE tenant_id=? AND scope_name=?",
                        (tenant_id, scope_name),
                    ).fetchall()
                }
                if (
                    "" in control_source_ids
                    or not control_source_ids.issubset(graph_source_ids)
                    or len(control_source_ids) > control_source_count
                ):
                    raise V4AdapterError(
                        "Source accounting control identity set differs from the graph"
                    )
                if control_source_count == graph_source_count:
                    return []
                if control_source_count > graph_source_count:
                    raise V4AdapterError(
                        "Source accounting control watermark is ahead of the graph"
                    )
                live_scope_work = control.execute(
                    "SELECT 1 FROM jobs WHERE tenant_id=? AND scope_name=? "
                    "AND state='running' UNION ALL "
                    "SELECT 1 FROM operation_stages WHERE tenant_id=? "
                    "AND scope_name=? AND state='running' LIMIT 1",
                    (tenant_id, scope_name, tenant_id, scope_name),
                ).fetchone()
                if live_scope_work is not None:
                    return []
                lifecycle = control.execute(
                    "SELECT state FROM scope_lifecycle "
                    "WHERE tenant_id=? AND scope_name=?",
                    (tenant_id, scope_name),
                ).fetchone()
                if lifecycle is not None and str(lifecycle["state"] or "") != "active":
                    return []
                content_deletion = control.execute(
                    "SELECT 1 FROM content_deletions "
                    "WHERE tenant_id=? AND scope_name=? "
                    "AND state IN ('requested','purging','reindexing','failed') "
                    "LIMIT 1",
                    (tenant_id, scope_name),
                ).fetchone()
                if content_deletion is not None:
                    return []
                # A local Source reconciliation must never authorize an
                # unrelated provider retry in the same Scope.  Keep this
                # check at the plan boundary because native and control
                # ledgers are separate databases.
                unknown_operations = self._provider_outcome_unknown_operations(
                    paths=paths
                )
                rows = control.execute(
                    "SELECT jobs.job_id,jobs.state,jobs.scope_seq,jobs.payload_json,"
                    "stages.stage_id,stages.state AS stage_state,stages.attempt "
                    "FROM jobs LEFT JOIN operation_stages AS stages "
                    "ON stages.job_id=jobs.job_id AND stages.stage_name='writer' "
                    "WHERE jobs.tenant_id=? AND jobs.scope_name=? "
                    "ORDER BY jobs.scope_seq,jobs.created_at,jobs.job_id",
                    (tenant_id, scope_name),
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise V4AdapterError(
                "Source accounting recovery inventory failed"
            ) from exc

        for row in rows:
            # Live and queued Writers own their Source boundary. Reconciliation
            # is only for a terminal failed Writer whose report can no longer
            # arrive; touching any other state would race normal completion.
            if (
                str(row["state"] or "") != "failed"
                or str(row["stage_state"] or "") != "failed"
            ):
                continue
            job_id = str(row["job_id"] or "").strip()
            if job_id in unknown_operations:
                # A local Source plan can coexist with an unknown operation,
                # but it must never include or authorize that operation.
                continue
            try:
                payload = json.loads(str(row["payload_json"] or "{}"))
            except json.JSONDecodeError as exc:
                raise V4AdapterError(
                    "Source accounting recovery job payload is invalid"
                ) from exc
            if not isinstance(payload, Mapping) or str(
                payload.get("job_type") or ""
            ) != "ingest":
                continue
            session_id = str(payload.get("session_id") or "").strip()
            messages = payload.get("messages")
            if not session_id or not isinstance(messages, list) or any(
                not isinstance(item, Mapping) for item in messages
            ):
                raise V4AdapterError(
                    "Source accounting recovery ingest payload is invalid"
                )
            stage_id = str(row["stage_id"] or f"{job_id}:writer").strip()
            attempt = int(row["attempt"] or 1)
            if not job_id or stage_id != f"{job_id}:writer" or attempt <= 0:
                raise V4AdapterError(
                    "Source accounting recovery stage identity is invalid"
                )
            prepared = self.recover_writer_source_accounting(
                tenant_id=tenant_id,
                scope_name=scope_name,
                session_id=session_id,
                job_id=job_id,
                messages=[dict(item) for item in messages],
                default_accounting_operation_id=f"{stage_id}:attempt:{attempt}",
            )
            writer = prepared.get("writer")
            source_count = int(prepared.get("recovered_source_count", 0) or 0)
            if source_count <= 0:
                continue
            if not isinstance(writer, Mapping):
                raise V4AdapterError(
                    "Source accounting recovery writer plan is invalid"
                )
            durable_sources = writer.get("durable_sources")
            accounting_operation_id = str(
                prepared.get("accounting_operation_id") or ""
            ).strip()
            if (
                not accounting_operation_id
                or not isinstance(durable_sources, list)
                or len(durable_sources) != source_count
            ):
                raise V4AdapterError("Source accounting recovery plan is invalid")
            plans.append(
                {
                    "job_id": job_id,
                    "job_state": str(row["state"] or ""),
                    "scope_seq": int(row["scope_seq"] or 0),
                    "writer_stage_id": stage_id,
                    "writer_stage_attempt": attempt,
                    "accounting_operation_id": accounting_operation_id,
                    "writer": dict(writer),
                    "source_count": source_count,
                }
            )
        planned_source_count = sum(int(plan["source_count"]) for plan in plans)
        expected_source_count = graph_source_count - control_source_count
        if planned_source_count > expected_source_count:
            raise V4AdapterError(
                "Source accounting recovery plans exceed the exact gap"
            )
        if not unknown_operations and planned_source_count != expected_source_count:
            raise V4AdapterError(
                "Source accounting recovery plans do not cover the exact gap"
            )
        return plans

    @staticmethod
    def _operation_has_provider_outcome_unknown(
        *, paths: ScopePaths, job_id: str
    ) -> bool:
        if not paths.database.is_file():
            return False
        try:
            database_uri = f"{paths.database.resolve().as_uri()}?mode=ro"
            with closing(
                sqlite3.connect(database_uri, timeout=30.0, uri=True)
            ) as connection:
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if not {"tmcra_service_batches", "v4_batch_journal"}.issubset(
                    tables
                ):
                    return False
                row = connection.execute(
                    "SELECT 1 FROM tmcra_service_batches AS batches "
                    "JOIN v4_batch_journal AS journal "
                    "ON journal.scope_id=batches.scope_id "
                    "AND journal.session_id=batches.session_id "
                    "AND journal.batch_index=batches.batch_index "
                    "WHERE batches.operation_id=? "
                    "AND journal.status IN ('api_started','outcome_unknown') "
                    "LIMIT 1",
                    (job_id,),
                ).fetchone()
                return row is not None
        except (OSError, sqlite3.DatabaseError):
            return True

    @staticmethod
    def _provider_outcome_unknown_operations(*, paths: ScopePaths) -> set[str]:
        """Return operation ids whose provider outcome is not terminal."""

        try:
            with closing(sqlite3.connect(paths.database, timeout=30.0)) as connection:
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if not {"tmcra_service_batches", "v4_batch_journal"}.issubset(
                    tables
                ):
                    return set()
                return {
                    str(row[0] or "").strip()
                    for row in connection.execute(
                        "SELECT DISTINCT batches.operation_id "
                        "FROM tmcra_service_batches AS batches "
                        "JOIN v4_batch_journal AS journal "
                        "ON journal.scope_id=batches.scope_id "
                        "AND journal.session_id=batches.session_id "
                        "AND journal.batch_index=batches.batch_index "
                        "WHERE journal.status IN ('api_started','outcome_unknown') "
                        "AND batches.operation_id!=''"
                    ).fetchall()
                    if str(row[0] or "").strip()
                }
        except (OSError, sqlite3.DatabaseError):
            return {"<unreadable>"}

    def validate_source_accounting_recovery_plan(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        plan: Mapping[str, Any],
    ) -> None:
        """Re-read native proof immediately before a guarded control commit."""

        job_id = str(plan.get("job_id") or "").strip()
        stage_id = str(plan.get("writer_stage_id") or "").strip()
        stage_attempt = int(plan.get("writer_stage_attempt", 0) or 0)
        planned_writer = plan.get("writer")
        planned_operation = str(
            plan.get("accounting_operation_id") or ""
        ).strip()
        if (
            not job_id
            or stage_id != f"{job_id}:writer"
            or stage_attempt <= 0
            or not isinstance(planned_writer, Mapping)
            or not planned_operation
        ):
            raise V4AdapterError("Source accounting recovery plan is invalid")
        try:
            with closing(
                sqlite3.connect(self.settings.control_db, timeout=30.0)
            ) as control:
                control.row_factory = sqlite3.Row
                control.execute("PRAGMA busy_timeout=30000")
                row = control.execute(
                    "SELECT payload_json FROM jobs WHERE job_id=? "
                    "AND tenant_id=? AND scope_name=?",
                    (job_id, tenant_id, scope_name),
                ).fetchone()
            if row is None:
                raise V4AdapterError("Source accounting recovery job is missing")
            payload = json.loads(str(row["payload_json"] or "{}"))
        except (sqlite3.DatabaseError, json.JSONDecodeError) as exc:
            raise V4AdapterError(
                "Source accounting recovery job proof is unreadable"
            ) from exc
        session_id = str(payload.get("session_id") or "").strip()
        messages = payload.get("messages")
        if not session_id or not isinstance(messages, list) or any(
            not isinstance(item, Mapping) for item in messages
        ):
            raise V4AdapterError("Source accounting recovery payload is invalid")
        current = self.recover_writer_source_accounting(
            tenant_id=tenant_id,
            scope_name=scope_name,
            session_id=session_id,
            job_id=job_id,
            messages=[dict(item) for item in messages],
            default_accounting_operation_id=f"{stage_id}:attempt:{stage_attempt}",
        )
        if (
            str(current.get("accounting_operation_id") or "").strip()
            != planned_operation
            or current.get("writer") != planned_writer
            or int(current.get("recovered_source_count", 0) or 0)
            != int(plan.get("source_count", 0) or 0)
        ):
            raise V4AdapterError(
                "Source accounting native proof changed after planning"
            )

    @contextmanager
    def source_accounting_recovery_guard(
        self,
        *,
        tenant_id: str,
        scope_name: str,
    ) -> Any:
        """Reserve the native writer slot through proof and control commit.

        Native Sources and the control ledger live in separate SQLite files, so
        one cross-database transaction is impossible.  A native ``IMMEDIATE``
        transaction prevents Writer/deletion mutations after the second proof
        read while the runtime commits the corresponding control ledger row.
        The transaction itself is read-only and is always rolled back.
        """

        paths = self.scope_paths(tenant_id, scope_name)
        if not paths.database.is_file():
            raise V4AdapterError("Source accounting recovery database is missing")
        with closing(sqlite3.connect(paths.database, timeout=30.0)) as native:
            native.execute("PRAGMA busy_timeout=30000")
            try:
                native.execute("BEGIN IMMEDIATE")
                yield
            finally:
                if native.in_transaction:
                    native.rollback()

    @staticmethod
    def _blocked_ingest_recovery_plan(reason: str) -> dict[str, Any]:
        return {
            "resumable": False,
            "mode": "manual_review",
            "parallel_safe": False,
            "external_api_calls_expected": (
                0 if reason == "provider_outcome_unknown" else None
            ),
            "deterministic_local_repair": False,
            "automatic_recovery_allowed": False,
            "reason": str(reason),
        }

    @staticmethod
    def _ingest_recovery_fingerprint(
        *, paths: ScopePaths, operation: Path, job_id: str, mode: str
    ) -> str:
        """Hash only durable recovery inputs, excluding timestamps and content."""

        state: dict[str, Any] = {
            "recovery_contract_version": _INGEST_RECOVERY_CONTRACT_VERSION,
            "mode": str(mode),
            "artifacts": {},
            "batches": [],
        }
        artifacts = state["artifacts"]
        assert isinstance(artifacts, dict)
        artifact_names = ["commit.json"]
        if mode in {"complete_writer_artifacts", "committed_writer_artifacts"}:
            artifact_names.insert(0, "product_writer_report.json")
        for name in artifact_names:
            path = operation / name
            if path.is_file():
                artifacts[name] = _sha256_file(path)

        if paths.database.is_file():
            try:
                with closing(sqlite3.connect(paths.database, timeout=30.0)) as connection:
                    connection.row_factory = sqlite3.Row
                    tables = {
                        str(row[0])
                        for row in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )
                    }
                    if {"tmcra_service_batches", "v4_batch_journal"}.issubset(
                        tables
                    ):
                        rows = connection.execute(
                            "SELECT journal.batch_id,journal.status,"
                            "journal.request_sha256,journal.response_sha256,"
                            "journal.response_json "
                            "FROM tmcra_service_batches AS batches "
                            "JOIN v4_batch_journal AS journal "
                            "ON journal.scope_id=batches.scope_id "
                            "AND journal.session_id=batches.session_id "
                            "AND journal.batch_index=batches.batch_index "
                            "WHERE batches.operation_id=? "
                            "ORDER BY batches.batch_index,journal.batch_id",
                            (job_id,),
                        ).fetchall()
                        batch_ids = [str(row["batch_id"] or "") for row in rows]
                        state["batches"] = [
                            {
                                "batch_id": str(row["batch_id"] or ""),
                                "status": str(row["status"] or ""),
                                "request_sha256": str(row["request_sha256"] or ""),
                                "response_sha256": str(row["response_sha256"] or ""),
                                "response_json_sha256": hashlib.sha256(
                                    str(row["response_json"] or "").encode("utf-8")
                                ).hexdigest(),
                            }
                            for row in rows
                        ]
                        if batch_ids:
                            placeholders = ",".join("?" for _ in batch_ids)
                            if "v4_reconciliation_jobs" in tables:
                                reconciliation = connection.execute(
                                    "SELECT batch_id,job_id,status,decision,response_json "
                                    "FROM v4_reconciliation_jobs "
                                    f"WHERE batch_id IN ({placeholders}) "
                                    "ORDER BY batch_id,job_id",
                                    tuple(batch_ids),
                                ).fetchall()
                                state["reconciliation"] = [
                                    {
                                        "batch_id": str(row["batch_id"] or ""),
                                        "job_id": str(row["job_id"] or ""),
                                        "status": str(row["status"] or ""),
                                        "decision": str(row["decision"] or ""),
                                        "response_json_sha256": hashlib.sha256(
                                            str(row["response_json"] or "").encode(
                                                "utf-8"
                                            )
                                        ).hexdigest(),
                                    }
                                    for row in reconciliation
                                ]
                            if "v4_message_commit_journal" in tables:
                                commits = connection.execute(
                                    "SELECT batch_id,commit_id,status,semantic_committed,"
                                    "response_sha256,plan_sha256 "
                                    "FROM v4_message_commit_journal "
                                    f"WHERE batch_id IN ({placeholders}) "
                                    "ORDER BY batch_id,commit_id",
                                    tuple(batch_ids),
                                ).fetchall()
                                state["message_commits"] = [
                                    {
                                        "batch_id": str(row["batch_id"] or ""),
                                        "commit_id": str(row["commit_id"] or ""),
                                        "status": str(row["status"] or ""),
                                        "semantic_committed": int(
                                            row["semantic_committed"] or 0
                                        ),
                                        "response_sha256": str(
                                            row["response_sha256"] or ""
                                        ),
                                        "plan_sha256": str(row["plan_sha256"] or ""),
                                    }
                                    for row in commits
                                ]
            except (OSError, sqlite3.DatabaseError, TypeError, ValueError):
                state["database_state"] = "unreadable"

        encoded = json.dumps(
            state, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        return f"{_LOCAL_REPAIR_FINGERPRINT_CONTRACT_VERSION}:{digest}"

    @staticmethod
    def _ingest_recovery_is_local_only(
        *, paths: ScopePaths, operation: Path, job_id: str, mode: str
    ) -> bool:
        """Prove that resuming this operation cannot reach a provider call."""

        if mode not in {"none", "validation"} or not paths.database.is_file():
            return False
        try:
            with closing(sqlite3.connect(paths.database, timeout=30.0)) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    "SELECT journal.batch_id,journal.status,"
                    "journal.response_json "
                    "FROM tmcra_service_batches AS batches "
                    "JOIN v4_batch_journal AS journal "
                    "ON journal.scope_id=batches.scope_id "
                    "AND journal.session_id=batches.session_id "
                    "AND journal.batch_index=batches.batch_index "
                    "WHERE batches.operation_id=? ORDER BY batches.batch_index",
                    (job_id,),
                ).fetchall()
                if not rows:
                    return False
                statuses = {str(row["status"] or "") for row in rows}
                if statuses == {"committed"}:
                    return True
                if mode != "validation":
                    return False
                replay_batch_ids = {
                    str(row["batch_id"] or "")
                    for row in rows
                    if str(row["status"] or "") != "committed"
                }
                if not replay_batch_ids:
                    return True
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if "v4_reconciliation_jobs" not in tables:
                    return False
                placeholders = ",".join("?" for _ in replay_batch_ids)
                reconciliation = connection.execute(
                    "SELECT batch_id,status FROM v4_reconciliation_jobs "
                    f"WHERE batch_id IN ({placeholders}) ORDER BY batch_id,job_id",
                    tuple(sorted(replay_batch_ids)),
                ).fetchall()
                jobs_by_batch: dict[str, list[str]] = {}
                for row in reconciliation:
                    jobs_by_batch.setdefault(str(row["batch_id"] or ""), []).append(
                        str(row["status"] or "")
                    )
                if all(
                    jobs_by_batch.get(batch_id)
                    and set(jobs_by_batch[batch_id]) == {"completed"}
                    for batch_id in replay_batch_ids
                ):
                    return True
                if "v4_message_commit_journal" not in tables:
                    return False
                rows_by_batch = {
                    str(row["batch_id"] or ""): row for row in rows
                }
                for batch_id in replay_batch_ids:
                    if jobs_by_batch.get(batch_id):
                        return False
                    try:
                        response = json.loads(
                            str(rows_by_batch[batch_id]["response_json"] or "")
                        )
                    except (KeyError, json.JSONDecodeError):
                        return False
                    messages = (
                        response.get("messages")
                        if isinstance(response, Mapping)
                        else None
                    )
                    if not isinstance(messages, list):
                        return False
                    expected = {
                        str(message.get("message_id") or ""): hashlib.sha256(
                            json.dumps(
                                dict(message),
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()
                        for message in messages
                        if isinstance(message, Mapping)
                        and str(message.get("message_id") or "")
                    }
                    commits = connection.execute(
                        "SELECT message_id,status,response_sha256,plan_json,plan_sha256 "
                        "FROM v4_message_commit_journal WHERE batch_id=?",
                        (batch_id,),
                    ).fetchall()
                    actual = {
                        str(commit["message_id"] or ""): commit
                        for commit in commits
                    }
                    if (
                        len(expected) != len(messages)
                        or set(actual) != set(expected)
                        or any(
                            str(actual[message_id]["status"] or "")
                            not in {"prepared", "committed"}
                            or str(actual[message_id]["response_sha256"] or "")
                            != response_sha256
                            or not str(actual[message_id]["plan_json"] or "")
                            or str(actual[message_id]["plan_sha256"] or "")
                            != hashlib.sha256(
                                str(actual[message_id]["plan_json"]).encode("utf-8")
                            ).hexdigest()
                            for message_id, response_sha256 in expected.items()
                        )
                    ):
                        return False
                return True
        except (OSError, sqlite3.DatabaseError, TypeError, ValueError):
            return False

    @staticmethod
    def _ingest_recovery_is_parallel_safe(
        *, paths: ScopePaths, operation: Path, job_id: str, mode: str
    ) -> bool:
        """Allow distinct-session recovery concurrency only on self-hosted inference."""

        if V4StorageAdapter._ingest_recovery_is_local_only(
            paths=paths,
            operation=operation,
            job_id=job_id,
            mode=mode,
        ):
            return True
        if mode not in {
            "none",
            "validation",
            "definitive_provider_failure",
            "audited_writer_state",
            "schema_constrained_invalid_response",
            "schema_constrained_invalid_response_prepared",
            "audited_local_inference_cancelled",
        }:
            return False
        return bool(
            str(os.getenv("TMCRA_WRITER_PROVIDER") or "").strip()
            == LOCAL_QWEN_PROVIDER
            and bool(_active_local_writer_model())
            and str(os.getenv("TMCRA_WRITER_PROMPT_ADAPTER") or "").strip()
            == LOCAL_QWEN_PROMPT_ADAPTER
            and str(os.getenv("TMCRA_LOCAL_WRITER_RECOVERY_CONCURRENCY") or "1")
            .strip()
            .isdigit()
            and 1
            < int(
                str(os.getenv("TMCRA_LOCAL_WRITER_RECOVERY_CONCURRENCY") or "1")
                .strip()
            )
            <= 4
        )

    def can_resume_ingest(
        self, *, tenant_id: str, scope_name: str, job_id: str
    ) -> bool:
        """Return true after a durable commit or a fail-closed journal audit."""

        return bool(
            self.ingest_recovery_plan(
                tenant_id=tenant_id,
                scope_name=scope_name,
                job_id=job_id,
            ).get("resumable")
        )

    def _unaccounted_complete_writer_operations(
        self,
        *,
        paths: ScopePaths,
        source_count: int,
        source_ids_by_operation: Mapping[str, set[str]],
        source_metrics_by_operation: Mapping[str, tuple[int, int]],
        known_failed_operation_ids: set[str],
        source_metadata_by_operation: Mapping[
            str, Mapping[str, tuple[str, int, int]]
        ] | None = None,
        legacy_source_ids_by_operation: Mapping[str, set[str]] | None = None,
    ) -> tuple[set[str], set[str], int]:
        """Bind a native/control gap to exact, locally recoverable Sources."""

        try:
            with closing(
                sqlite3.connect(self.settings.control_db, timeout=30.0)
            ) as control:
                control.row_factory = sqlite3.Row
                control.execute("PRAGMA busy_timeout=30000")
                quick = control.execute("PRAGMA quick_check").fetchone()
                if quick is None or str(quick[0]) != "ok":
                    return set(), {"control_sqlite_quick_check_failed"}, 0
                state = control.execute(
                    "SELECT source_event_seq FROM scope_evolution_state "
                    "WHERE tenant_id=? AND scope_name=?",
                    (paths.tenant_id, paths.scope_name),
                ).fetchone()
                control_source_event_seq = (
                    0 if state is None else int(state["source_event_seq"] or 0)
                )
                if control_source_event_seq > source_count:
                    return (
                        set(),
                        {"source_control_watermark_ahead"},
                        control_source_event_seq,
                    )

                jobs: dict[str, tuple[str, Mapping[str, Any]]] = {}
                for row in control.execute(
                    "SELECT job_id,state,payload_json FROM jobs WHERE tenant_id=? "
                    "AND scope_name=?",
                    (paths.tenant_id, paths.scope_name),
                ).fetchall():
                    try:
                        payload = json.loads(str(row["payload_json"] or "{}"))
                    except json.JSONDecodeError:
                        continue
                    if (
                        isinstance(payload, Mapping)
                        and str(payload.get("job_type") or "") == "ingest"
                    ):
                        jobs[str(row["job_id"])] = (
                            str(row["state"] or ""),
                            payload,
                        )

                violations: set[str] = set()
                candidates: set[str] = set()
                committed_by_source = {
                    str(row["source_record_id"]): (
                        str(row["origin_operation_id"]),
                        str(row["accounting_operation_id"]),
                        int(row["raw_token_estimate"] or 0),
                        int(row["user_turns"] or 0),
                    )
                    for row in control.execute(
                        "SELECT source_record_id,origin_operation_id,"
                        "accounting_operation_id,raw_token_estimate,user_turns "
                        "FROM scope_source_event_commits WHERE tenant_id=? "
                        "AND scope_name=?",
                        (paths.tenant_id, paths.scope_name),
                    ).fetchall()
                }
                graph_source_ids = set().union(
                    *(set(values) for values in source_ids_by_operation.values())
                ) if source_ids_by_operation else set()
                if set(committed_by_source) - graph_source_ids:
                    violations.add("source_control_accounting_set_mismatch")
                artifact_proofs: dict[str, Mapping[str, Any] | None] = {}

                def complete_artifact_proof(job_id: str) -> Mapping[str, Any] | None:
                    if job_id in artifact_proofs:
                        return artifact_proofs[job_id]
                    operation = paths.operations / job_id
                    try:
                        expected_payload = json.loads(
                            (operation / "input.json").read_text(encoding="utf-8")
                        )
                        report = json.loads(
                            (operation / "product_writer_report.json").read_text(
                                encoding="utf-8"
                            )
                        )
                    except (OSError, json.JSONDecodeError):
                        artifact_proofs[job_id] = None
                        return None
                    if not (
                        isinstance(expected_payload, list)
                        and all(isinstance(item, Mapping) for item in expected_payload)
                        and isinstance(report, Mapping)
                        and self._writer_report_is_complete(report)
                    ):
                        artifact_proofs[job_id] = None
                        return None
                    try:
                        self._validate_writer_report(report, paths=paths, job_id=job_id)
                        self._validate_complete_writer_artifacts(
                            report,
                            paths=paths,
                            operation=operation,
                            expected_payload=expected_payload,
                        )
                    except V4AdapterError:
                        artifact_proofs[job_id] = None
                        return None
                    durable_sources = report.get("durable_sources", [])
                    if not isinstance(durable_sources, list):
                        artifact_proofs[job_id] = None
                        return None
                    reported_sources = {
                        str(item.get("source_record_id") or "").strip(): (
                            str(item.get("origin_operation_id") or "").strip(),
                            int(item.get("raw_token_estimate", 0) or 0),
                            int(item.get("user_turns", 0) or 0),
                        )
                        for item in durable_sources
                        if isinstance(item, Mapping)
                    }
                    expected_sources = dict(
                        (source_metadata_by_operation or {}).get(job_id, {})
                    )
                    legacy_source_ids = set(
                        (legacy_source_ids_by_operation or {}).get(job_id, set())
                    )
                    if (
                        legacy_source_ids
                        and legacy_source_ids.issubset(expected_sources)
                    ):
                        for source_record_id in legacy_source_ids:
                            reported_sources.setdefault(
                                source_record_id,
                                expected_sources[source_record_id],
                            )
                    if expected_sources and reported_sources != expected_sources:
                        artifact_proofs[job_id] = None
                        return None
                    stage_id = str(report.get("stage_id") or "").strip()
                    stage_attempt = report.get("stage_attempt")
                    if (
                        not stage_id
                        or isinstance(stage_attempt, bool)
                        or not isinstance(stage_attempt, int)
                        or stage_attempt <= 0
                    ):
                        artifact_proofs[job_id] = None
                        return None
                    proof = {
                        "reported_sources": reported_sources,
                        "accounting_operation_id": f"{stage_id}:attempt:{stage_attempt}",
                    }
                    artifact_proofs[job_id] = proof
                    return proof

                missing_by_operation: dict[str, set[str]] = {}
                for operation_id, source_ids in source_ids_by_operation.items():
                    missing_source_ids: set[str] = set()
                    for source_record_id in source_ids:
                        committed = committed_by_source.get(source_record_id)
                        if committed is None:
                            missing_source_ids.add(source_record_id)
                        elif committed[0] != operation_id:
                            violations.add("source_control_accounting_origin_mismatch")
                    if not missing_source_ids:
                        continue

                    job = jobs.get(operation_id)
                    if (
                        job is not None
                        and operation_id not in known_failed_operation_ids
                        and missing_source_ids == set(source_ids)
                    ):
                        writer_stage = control.execute(
                            "SELECT stage_id,state,attempt FROM operation_stages "
                            "WHERE job_id=? AND stage_name='writer'",
                            (operation_id,),
                        ).fetchone()
                        if (
                            writer_stage is not None
                            and str(writer_stage["state"] or "") == "succeeded"
                            and int(writer_stage["attempt"] or 0) > 0
                        ):
                            operation_id_candidates = [
                                operation_id,
                                f"{str(writer_stage['stage_id'])}:attempt:"
                                f"{int(writer_stage['attempt'])}",
                            ]
                            placeholders = ",".join(
                                "?" for _ in operation_id_candidates
                            )
                            legacy_rows = control.execute(
                                "SELECT operation_id,source_event_seq,new_message_count,"
                                "raw_token_estimate,user_turns "
                                "FROM scope_ingest_watermark_commits "
                                "WHERE tenant_id=? AND scope_name=? "
                                f"AND operation_id IN ({placeholders})",
                                (
                                    paths.tenant_id,
                                    paths.scope_name,
                                    *operation_id_candidates,
                                ),
                            ).fetchall()
                            source_set_rows = control.execute(
                                "SELECT operation_id FROM scope_ingest_source_sets "
                                "WHERE tenant_id=? AND scope_name=? "
                                f"AND operation_id IN ({placeholders})",
                                (
                                    paths.tenant_id,
                                    paths.scope_name,
                                    *operation_id_candidates,
                                ),
                            ).fetchall()
                            token_estimate, user_turns = (
                                source_metrics_by_operation.get(operation_id, (0, 0))
                            )
                            legacy_metrics = (
                                len(source_ids),
                                token_estimate,
                                user_turns,
                            )
                            legacy_proof_valid = bool(
                                len(legacy_rows) == 1
                                and not source_set_rows
                                and (
                                    int(legacy_rows[0]["new_message_count"] or 0),
                                    int(legacy_rows[0]["raw_token_estimate"] or 0),
                                    int(legacy_rows[0]["user_turns"] or 0),
                                )
                                == legacy_metrics
                                and int(legacy_rows[0]["source_event_seq"] or 0)
                                <= control_source_event_seq
                            )
                            if legacy_proof_valid:
                                continue
                            if legacy_rows or source_set_rows:
                                violations.add(
                                    "source_control_legacy_accounting_invalid"
                                )
                    missing_by_operation[operation_id] = missing_source_ids

                expected_gap = source_count - control_source_event_seq
                observed_gap = sum(len(value) for value in missing_by_operation.values())
                if observed_gap != expected_gap:
                    violations.add("source_control_watermark_divergence")

                candidate_source_count = 0
                for job_id, missing_source_ids in sorted(
                    missing_by_operation.items()
                ):
                    if not missing_source_ids:
                        continue
                    job = jobs.get(job_id)
                    if job is None:
                        violations.add("source_control_accounting_job_state_invalid")
                        continue
                    job_state, _payload = job
                    if job_state not in {"failed", "pending"}:
                        violations.add("source_control_accounting_job_state_invalid")
                        continue

                    source_ids = set(source_ids_by_operation[job_id])
                    proof = complete_artifact_proof(job_id)
                    complete_artifacts = proof is not None
                    accounting_operation_id = ""
                    if proof is not None:
                        reported_sources = dict(proof["reported_sources"])
                        if set(reported_sources) != source_ids:
                            violations.add("source_control_accounting_set_mismatch")
                            continue
                        metadata_mismatch = False
                        for source_record_id in source_ids - missing_source_ids:
                            committed = committed_by_source.get(source_record_id)
                            reported = reported_sources.get(source_record_id)
                            if committed is None or reported is None or (
                                committed[0], committed[2], committed[3]
                            ) != reported:
                                metadata_mismatch = True
                                break
                        if metadata_mismatch:
                            violations.add(
                                "source_control_accounting_metadata_mismatch"
                            )
                            continue
                        accounting_operation_id = str(
                            proof["accounting_operation_id"]
                        )
                    if not complete_artifacts:
                        violations.add("source_control_accounting_artifacts_invalid")
                        continue
                    operation_id_candidates = [job_id]
                    if accounting_operation_id:
                        operation_id_candidates.append(accounting_operation_id)
                    placeholders_for_operations = ",".join(
                        "?" for _ in operation_id_candidates
                    )
                    operation_rows = control.execute(
                        "SELECT operation_id,new_message_count,raw_token_estimate,"
                        "user_turns FROM scope_ingest_watermark_commits "
                        "WHERE tenant_id=? AND scope_name=? "
                        f"AND operation_id IN ({placeholders_for_operations})",
                        (
                            paths.tenant_id,
                            paths.scope_name,
                            *operation_id_candidates,
                        ),
                    ).fetchall()
                    source_set_rows = control.execute(
                        "SELECT operation_id,source_count FROM scope_ingest_source_sets "
                        "WHERE tenant_id=? AND scope_name=? "
                        f"AND operation_id IN ({placeholders_for_operations})",
                        (
                            paths.tenant_id,
                            paths.scope_name,
                            *operation_id_candidates,
                        ),
                    ).fetchall()
                    operation_by_id = {
                        str(row["operation_id"]): row for row in operation_rows
                    }
                    source_set_by_id = {
                        str(row["operation_id"]): row for row in source_set_rows
                    }
                    existing_operation_ids = set(operation_by_id) | set(
                        source_set_by_id
                    )
                    empty_prior_commits_valid = all(
                        operation_id in operation_by_id
                        and operation_id in source_set_by_id
                        and int(
                            operation_by_id[operation_id]["new_message_count"] or 0
                        )
                        == 0
                        and int(
                            operation_by_id[operation_id]["raw_token_estimate"] or 0
                        )
                        == 0
                        and int(operation_by_id[operation_id]["user_turns"] or 0)
                        == 0
                        and int(source_set_by_id[operation_id]["source_count"] or 0)
                        == 0
                        for operation_id in existing_operation_ids
                    )
                    if existing_operation_ids and not empty_prior_commits_valid:
                        violations.add("source_control_accounting_partial")
                        continue
                    candidates.add(job_id)
                    candidate_source_count += len(missing_source_ids)

                if candidate_source_count != expected_gap:
                    violations.add("source_control_watermark_divergence")
                return candidates, violations, control_source_event_seq
        except (OSError, sqlite3.DatabaseError, TypeError, ValueError):
            return set(), {"source_control_accounting_audit_failed"}, 0

    def _control_source_event_seq(
        self, *, paths: ScopePaths, source_count: int
    ) -> tuple[int, set[str]]:
        """Read the control watermark without requiring terminal Writer artifacts."""

        try:
            with closing(
                sqlite3.connect(self.settings.control_db, timeout=30.0)
            ) as control:
                control.row_factory = sqlite3.Row
                control.execute("PRAGMA busy_timeout=30000")
                quick = control.execute("PRAGMA quick_check").fetchone()
                if quick is None or str(quick[0]) != "ok":
                    return 0, {"control_sqlite_quick_check_failed"}
                state = control.execute(
                    "SELECT source_event_seq FROM scope_evolution_state "
                    "WHERE tenant_id=? AND scope_name=?",
                    (paths.tenant_id, paths.scope_name),
                ).fetchone()
                watermark = 0 if state is None else int(state["source_event_seq"] or 0)
                if watermark > source_count:
                    return watermark, {"source_control_watermark_ahead"}
                return watermark, set()
        except (OSError, sqlite3.DatabaseError, TypeError, ValueError):
            return 0, {"source_control_accounting_audit_failed"}

    def audit_scope_recovery(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Perform the full immutable-Source audit used by quarantine recovery."""

        paths = self.scope_paths(tenant_id, scope_name)
        if not paths.database.is_file():
            return {
                "integrity_ok": False,
                "ready_to_release": False,
                "error_code": "scope_database_missing",
                "source_count": 0,
                "failed_source_count": 0,
                "pending_source_count": 0,
                "failed_operation_ids": [],
            }
        violations: set[str] = set()
        failed_operation_ids: set[str] = set()
        status_counts: dict[str, int] = {}
        message_commit_counts: dict[str, int] = {}
        source_count = 0
        record_source_count = 0
        pre_source_operation_count = 0
        source_ids_by_operation: dict[str, set[str]] = {}
        source_metrics_by_operation: dict[str, tuple[int, int]] = {}
        source_metadata_by_operation: dict[
            str, dict[str, tuple[str, int, int]]
        ] = {}
        legacy_source_ids_by_operation: dict[str, set[str]] = {}
        control_source_event_seq = 0
        unaccounted_operation_ids: set[str] = set()
        try:
            with closing(sqlite3.connect(paths.database, timeout=30.0)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA busy_timeout=30000")
                quick = connection.execute("PRAGMA quick_check").fetchone()
                if quick is None or str(quick[0]) != "ok":
                    violations.add("sqlite_quick_check_failed")
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                required = {
                    "records",
                    "tmcra_service_messages",
                    "v4_source_journal",
                    "v4_batch_journal",
                    "v4_message_commit_journal",
                }
                missing = sorted(required - tables)
                if missing:
                    return {
                        "integrity_ok": False,
                        "ready_to_release": False,
                        "error_code": "scope_recovery_tables_missing",
                        "missing_table_count": len(missing),
                        "source_count": 0,
                        "failed_source_count": 0,
                        "pending_source_count": 0,
                        "failed_operation_ids": [],
                    }
                operation_bindings, batch_states, binding_violations = (
                    self._source_operation_bindings(connection, paths=paths)
                )
                violations.update(binding_violations)
                message_columns = {
                    str(row[1])
                    for row in connection.execute(
                        "PRAGMA table_info(tmcra_service_messages)"
                    )
                }
                service_first_operation_expression = (
                    "messages.first_operation_id"
                    if "first_operation_id" in message_columns
                    else "''"
                )
                service_message_identity_expression = (
                    "messages.internal_message_id"
                    if "internal_message_id" in message_columns
                    else "messages.message_id"
                )
                rows = connection.execute(
                    f"""
                    SELECT source.session_id,source.message_id,
                           source.session_index,source.message_index,
                           source.message_role,source.timestamp,source.content,
                           source.content_sha256,source.status,
                           source.source_record_id,source.source_turn_index,
                           source.source_persisted_at,
                           messages.session_id AS service_session_id,
                           messages.message_index AS service_message_index,
                           messages.role AS service_role,
                           messages.timestamp AS service_timestamp,
                           messages.content_sha256 AS service_content_sha256,
                           {service_first_operation_expression}
                               AS service_first_operation_id,
                           records.category AS record_category,
                           records.turn_index AS record_turn_index,
                           records.metadata_json AS record_metadata_json
                    FROM v4_source_journal AS source
                    LEFT JOIN tmcra_service_messages AS messages
                      ON messages.scope_id=source.scope_id
                     AND {service_message_identity_expression}=source.message_id
                    LEFT JOIN records
                      ON records.scope_id=source.scope_id
                     AND records.memory_id=source.source_record_id
                    WHERE source.scope_id=?
                    ORDER BY source.session_index,source.message_index,source.message_id
                    """,
                    (paths.scope_id,),
                ).fetchall()
                source_count = len(rows)
                registered_message_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM tmcra_service_messages WHERE scope_id=?",
                        (paths.scope_id,),
                    ).fetchone()[0]
                    or 0
                )
                source_record_ids: set[str] = set()
                source_message_ids: set[str] = set()
                unbound_records_by_message: dict[str, list[sqlite3.Row]] = {}
                all_source_record_rows = connection.execute(
                    "SELECT memory_id,turn_index,metadata_json FROM records "
                    "WHERE scope_id=? AND category='source'",
                    (paths.scope_id,),
                ).fetchall()
                for record_row in all_source_record_rows:
                    try:
                        record_metadata = json.loads(
                            str(record_row["metadata_json"] or "{}")
                        )
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record_metadata, Mapping):
                        continue
                    record_message_id = str(
                        record_metadata.get("message_id") or ""
                    ).strip()
                    if record_message_id:
                        unbound_records_by_message.setdefault(
                            record_message_id, []
                        ).append(record_row)
                for row in rows:
                    status = str(row["status"] or "")
                    status_counts[status] = status_counts.get(status, 0) + 1
                    message_id = str(row["message_id"] or "")
                    if not message_id or message_id in source_message_ids:
                        violations.add("source_message_identity_invalid")
                    source_message_ids.add(message_id)
                    operation_id = operation_bindings.get(message_id, "")
                    if not operation_id:
                        violations.add("source_operation_binding_missing")
                    if status in {"failed", "pending"} and operation_id:
                        failed_operation_ids.add(operation_id)
                    if status not in {"enriched", "failed", "pending"}:
                        violations.add("source_journal_nonterminal")
                    content = str(row["content"] or "")
                    content_sha256 = hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest()
                    source_record_id = str(row["source_record_id"] or "")
                    batch_status, has_response, api_started = batch_states.get(
                        message_id, ("", False, False)
                    )
                    pending_before_call = bool(
                        status == "pending"
                        and batch_status == "prepared"
                        and not has_response
                        and not api_started
                    )
                    if not source_record_id and pending_before_call:
                        # Source graph persistence happens immediately before
                        # the provider call. The process may stop before the
                        # graph record exists or just after graph persistence
                        # but before the journal binding. Both states are
                        # safe only when a graph-side record is absent or maps
                        # uniquely back to this exact immutable Source.
                        candidates = unbound_records_by_message.get(message_id, [])
                        if len(candidates) > 1:
                            violations.add("source_record_binding_invalid")
                            continue
                        if candidates:
                            candidate = candidates[0]
                            candidate_id = str(candidate["memory_id"] or "")
                            try:
                                candidate_metadata = json.loads(
                                    str(candidate["metadata_json"] or "{}")
                                )
                            except json.JSONDecodeError:
                                candidate_metadata = None
                            sidecar = (
                                candidate_metadata.get("sidecar_hint_metadata")
                                if isinstance(candidate_metadata, Mapping)
                                else None
                            )
                            sidecar = sidecar if isinstance(sidecar, Mapping) else {}
                            actor_role = str(
                                candidate_metadata.get("actor_role")
                                or candidate_metadata.get("speaker")
                                or sidecar.get("role")
                                or ""
                            ) if isinstance(candidate_metadata, Mapping) else ""
                            if not (
                                candidate_id
                                and isinstance(candidate_metadata, Mapping)
                                and candidate_metadata.get("raw_content") == content
                                and str(candidate_metadata.get("source_record_id") or "")
                                == candidate_id
                                and str(candidate_metadata.get("session_id") or "")
                                == str(row["session_id"] or "")
                                and int(candidate_metadata.get("session_index", -1))
                                == int(row["session_index"])
                                and int(candidate_metadata.get("message_index", -1))
                                == int(row["message_index"])
                                and actor_role == str(row["message_role"] or "")
                            ):
                                violations.add("source_record_metadata_mismatch")
                                continue
                            source_record_ids.add(candidate_id)
                        continue
                    if not source_record_id or source_record_id in source_record_ids:
                        violations.add("source_record_binding_invalid")
                        continue
                    source_record_ids.add(source_record_id)
                    if (
                        not str(row["source_persisted_at"] or "")
                        or content_sha256 != str(row["content_sha256"] or "")
                        or str(row["service_session_id"] or "")
                        != str(row["session_id"] or "")
                        or int(row["service_message_index"] if row["service_message_index"] is not None else -1)
                        != int(row["message_index"])
                        or str(row["service_role"] or "")
                        != str(row["message_role"] or "")
                        or str(row["service_timestamp"] or "")
                        != str(row["timestamp"] or "")
                        or str(row["service_content_sha256"] or "")
                        != content_sha256
                        or str(row["record_category"] or "") != "source"
                        or int(row["record_turn_index"] if row["record_turn_index"] is not None else -1)
                        != int(row["source_turn_index"])
                    ):
                        violations.add("source_binding_mismatch")
                        continue
                    try:
                        metadata = json.loads(str(row["record_metadata_json"] or "{}"))
                    except json.JSONDecodeError:
                        violations.add("source_record_metadata_invalid")
                        continue
                    if not isinstance(metadata, Mapping):
                        violations.add("source_record_metadata_invalid")
                        continue
                    sidecar = metadata.get("sidecar_hint_metadata")
                    sidecar = sidecar if isinstance(sidecar, Mapping) else {}
                    raw_content = metadata.get("raw_content")
                    actor_role = str(
                        metadata.get("actor_role")
                        or metadata.get("speaker")
                        or sidecar.get("role")
                        or ""
                    )
                    if (
                        not isinstance(raw_content, str)
                        or raw_content != content
                        or str(metadata.get("source_record_id") or "")
                        != source_record_id
                        or int(metadata.get("session_index", -1))
                        != int(row["session_index"])
                        or int(metadata.get("message_index", -1))
                        != int(row["message_index"])
                        or actor_role != str(row["message_role"] or "")
                    ):
                        violations.add("source_record_metadata_mismatch")
                        continue
                    source_ids_by_operation.setdefault(operation_id, set()).add(
                        source_record_id
                    )
                    source_metadata_by_operation.setdefault(operation_id, {})[
                        source_record_id
                    ] = (
                        operation_id,
                        _raw_token_estimate(content),
                        int(str(row["message_role"] or "").lower() == "user"),
                    )
                    if not str(row["service_first_operation_id"] or "").strip():
                        legacy_source_ids_by_operation.setdefault(
                            operation_id, set()
                        ).add(source_record_id)
                    prior_tokens, prior_turns = source_metrics_by_operation.get(
                        operation_id, (0, 0)
                    )
                    source_metrics_by_operation[operation_id] = (
                        prior_tokens + _raw_token_estimate(content),
                        prior_turns
                        + int(str(row["message_role"] or "").lower() == "user"),
                    )
                record_ids = {
                    str(row["memory_id"]) for row in all_source_record_rows
                }
                record_source_count = len(record_ids)
                if record_ids != source_record_ids:
                    violations.add("source_record_set_mismatch")
                binding_message_ids = set(operation_bindings)
                if source_message_ids - binding_message_ids:
                    violations.add("source_operation_binding_set_mismatch")
                extra_binding_ids = binding_message_ids - source_message_ids
                pre_source_operations, pre_source_violations = (
                    self._pre_source_registered_operations(
                        connection,
                        paths=paths,
                        message_ids=extra_binding_ids,
                        operation_bindings=operation_bindings,
                        batch_states=batch_states,
                        source_message_ids=source_message_ids,
                    )
                )
                violations.update(pre_source_violations)
                failed_operation_ids.update(pre_source_operations)
                pre_source_operation_count = len(pre_source_operations)
                for row in connection.execute(
                    "SELECT status,COUNT(*) AS total "
                    "FROM v4_message_commit_journal WHERE scope_id=? GROUP BY status",
                    (paths.scope_id,),
                ).fetchall():
                    message_commit_counts[str(row["status"] or "")] = int(
                        row["total"] or 0
                    )
        except (OSError, sqlite3.DatabaseError, TypeError, ValueError):
            return {
                "integrity_ok": False,
                "ready_to_release": False,
                "error_code": "scope_recovery_audit_failed",
                "source_count": source_count,
                "record_source_count": record_source_count,
                "failed_source_count": status_counts.get("failed", 0),
                "pending_source_count": status_counts.get("pending", 0),
                "failed_operation_ids": sorted(failed_operation_ids),
            }
        failed_source_count = status_counts.get("failed", 0)
        pending_source_count = status_counts.get("pending", 0)
        prepared_commit_count = message_commit_counts.get("prepared", 0)
        source_recovery_incomplete = bool(
            failed_source_count
            or pending_source_count
            or prepared_commit_count
            or failed_operation_ids
        )
        if not violations and not source_recovery_incomplete:
            (
                unaccounted_operation_ids,
                accounting_violations,
                control_source_event_seq,
            ) = self._unaccounted_complete_writer_operations(
                paths=paths,
                source_count=source_count,
                source_ids_by_operation=source_ids_by_operation,
                source_metrics_by_operation=source_metrics_by_operation,
                source_metadata_by_operation=source_metadata_by_operation,
                legacy_source_ids_by_operation=legacy_source_ids_by_operation,
                known_failed_operation_ids=set(failed_operation_ids),
            )
            violations.update(accounting_violations)
            failed_operation_ids.update(unaccounted_operation_ids)
        elif not violations:
            control_source_event_seq, control_violations = (
                self._control_source_event_seq(
                    paths=paths,
                    source_count=source_count,
                )
            )
            violations.update(control_violations)
        integrity_ok = not violations
        ready_to_release = bool(
            integrity_ok
            and source_count == record_source_count
            and failed_source_count == 0
            and pending_source_count == 0
            and prepared_commit_count == 0
            and not failed_operation_ids
        )
        return {
            "integrity_ok": integrity_ok,
            "ready_to_release": ready_to_release,
            "error_code": "" if integrity_ok else sorted(violations)[0],
            "violation_codes": sorted(violations),
            "source_count": source_count,
            "registered_message_count": registered_message_count,
            "record_source_count": record_source_count,
            "enriched_source_count": status_counts.get("enriched", 0),
            "failed_source_count": failed_source_count,
            "pending_source_count": pending_source_count,
            "prepared_message_commit_count": prepared_commit_count,
            "pre_source_registered_operation_count": pre_source_operation_count,
            "control_source_event_seq": control_source_event_seq,
            "unaccounted_source_count": max(
                0, source_count - control_source_event_seq
            ),
            "unaccounted_operation_ids": sorted(unaccounted_operation_ids),
            "failed_operation_ids": sorted(failed_operation_ids),
        }

    @staticmethod
    def _has_durable_failed_batch_raw_response(
        *,
        operation: Path,
        job_id: str,
        row: sqlite3.Row,
        metadata: Mapping[str, Any],
    ) -> bool:
        """Prove a failed validation has one intact, reusable API response."""

        keys = set(row.keys())
        required = {"batch_id", "scope_id", "session_id"}
        if not required.issubset(keys):
            return False
        batch_id = str(row["batch_id"] or "")
        scope_id = str(row["scope_id"] or "")
        session_id = str(row["session_id"] or "")
        call_key = f"flash:{batch_id}"
        expected_hash = str(metadata.get("response_sha256") or "")
        artifact = operation / "product_writer_raw_responses.jsonl"
        if not all((batch_id, scope_id, session_id, expected_hash)) or not artifact.is_file():
            return False

        matches: list[Mapping[str, Any]] = []
        try:
            for line in artifact.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    return False
                if (
                    str(value.get("call_key") or "") == call_key
                    and str(value.get("raw_response_sha256") or "")
                    == expected_hash
                    and str(value.get("metadata_response_sha256") or "")
                    == expected_hash
                ):
                    matches.append(value)
        except (OSError, json.JSONDecodeError):
            return False
        if len(matches) != 1:
            return False

        record = matches[0]
        raw_response = record.get("raw_response")
        if not isinstance(raw_response, str) or not raw_response:
            return False
        try:
            parsed_response = json.loads(raw_response)
        except json.JSONDecodeError:
            return False
        raw_hash = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
        record_job_id = str(record.get("job_id") or "")
        return bool(
            isinstance(parsed_response, Mapping)
            and str(record.get("batch_id") or "") == batch_id
            and str(record.get("scope_id") or "") == scope_id
            and str(record.get("session_id") or "") == session_id
            and record_job_id in {"", job_id}
            and str(record.get("stage") or "") == "batch_flash"
            and str(record.get("model") or "") == _active_local_writer_model()
            and str(record.get("raw_response_sha256") or "") == raw_hash
            and str(record.get("metadata_response_sha256") or "") == raw_hash
            and expected_hash == raw_hash
            and (
                not str(metadata.get("physical_call_id") or "")
                or not str(record.get("physical_call_id") or "")
                or str(record.get("physical_call_id") or "")
                == str(metadata.get("physical_call_id") or "")
            )
            and (
                not str(metadata.get("request_sha256") or "")
                or not str(record.get("request_sha256") or "")
                or str(record.get("request_sha256") or "")
                == str(metadata.get("request_sha256") or "")
            )
        )

    @staticmethod
    def _cancelled_local_inference_batch_id(
        *,
        paths: ScopePaths,
        operation: Path,
        job_id: str,
        rows: Sequence[sqlite3.Row],
    ) -> str | None:
        proof_path = operation / _LOCAL_INFERENCE_CANCELLATION_PROOF_FILE
        try:
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(proof, Mapping):
            return None
        target_rows = [
            row
            for row in rows
            if str(row["status"] or "") in {"api_started", "outcome_unknown"}
        ]
        if len(target_rows) != 1:
            return None
        target = target_rows[0]
        try:
            recovery_history = json.loads(
                str(
                    target["recovery_history_json"]
                    if "recovery_history_json" in target.keys()
                    else "[]"
                )
                or "[]"
            )
        except json.JSONDecodeError:
            return None
        if not isinstance(recovery_history, list) or any(
            isinstance(item, Mapping)
            and str(item.get("reason") or "")
            == "audited_local_inference_cancelled"
            for item in recovery_history
        ):
            return None
        batch_id = str(target["batch_id"] or "")
        request_json = str(target["request_json"] or "")
        request_sha256 = str(target["request_sha256"] or "")
        evidence_sha256 = str(proof.get("evidence_sha256") or "")
        evidence_file = str(proof.get("evidence_file") or "")
        evidence_path = operation / evidence_file
        immutable = dict(proof)
        immutable.pop("proof_sha256", None)
        encoded = json.dumps(
            immutable,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        proof_valid = bool(
            proof.get("schema_version")
            == _LOCAL_INFERENCE_CANCELLATION_PROOF_SCHEMA
            and str(proof.get("job_id") or "") == job_id
            and str(proof.get("scope_id") or "") == paths.scope_id
            and str(proof.get("batch_id") or "") == batch_id
            and str(proof.get("request_sha256") or "") == request_sha256
            and request_json
            and hashlib.sha256(request_json.encode("utf-8")).hexdigest()
            == request_sha256
            and str(proof.get("provider") or "") == LOCAL_QWEN_PROVIDER
            and str(proof.get("model") or "") == _active_local_writer_model()
            and proof.get("inference_cancelled") is True
            and proof.get("completed_response_observed") is False
            and proof.get("target_request_in_provider_ledger") is False
            and int(proof.get("replacement_calls_authorized") or 0) == 1
            and int(proof.get("physical_api_calls_performed_by_audit") or 0) == 0
            and len(evidence_sha256) == 64
            and all(character in "0123456789abcdef" for character in evidence_sha256)
            and Path(evidence_file).name == evidence_file
            and evidence_path.is_file()
            and _sha256_file(evidence_path) == evidence_sha256
            and str(proof.get("proof_sha256") or "")
            == hashlib.sha256(encoded).hexdigest()
            and not str(target["response_json"] or "")
            and not str(
                target["response_metadata_json"]
                if "response_metadata_json" in target.keys()
                else ""
            ).strip().strip("{}")
        )
        if not proof_valid:
            return None
        if any(
            str(row["status"] or "")
            not in {"prepared", "validated", "committed", "api_started", "outcome_unknown"}
            for row in rows
        ):
            return None
        return batch_id

    @staticmethod
    def _incomplete_ingest_recovery_mode(
        *,
        paths: ScopePaths,
        operation: Path,
        job_id: str,
        expected_payload: Sequence[Mapping[str, Any]] | None = None,
    ) -> str | None:
        """Classify an incomplete attempt without guessing external outcomes.

        ``None`` means an operator must audit it. Validation recovery never
        issues a replacement call, while definitive-provider recovery is only
        granted to a persisted HTTP 402 rejection with no response body.
        """
        input_path = operation / "input.json"
        if (
            not operation.is_dir()
            or not input_path.is_file()
            or (operation / "commit.json").exists()
        ):
            return None
        audited_writer_state: list[str] | None = None
        definitive_reviewer_failures: list[str] | None = []
        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list) or not payload:
                return None
            if expected_payload is not None and payload != list(expected_payload):
                return None
            for row in payload:
                if (
                    not isinstance(row, dict)
                    or row.get("scope_id") != paths.scope_id
                    or row.get("question_id") != paths.question_id
                    or row.get("operation_id") != job_id
                    or not isinstance(row.get("messages"), list)
                    or not row["messages"]
                ):
                    return None
            if not paths.database.is_file():
                # The resident adapter checks pool readiness before opening a
                # Writer log. With only the immutable input artifact present,
                # no Writer process or provider call could have started.
                artifacts = {
                    path.name for path in operation.iterdir() if path.name != "input.json"
                }
                return "none" if not artifacts else None
            connection = sqlite3.connect(paths.database, timeout=30.0)
            connection.row_factory = sqlite3.Row
            try:
                quick = connection.execute("PRAGMA quick_check").fetchone()
                if not quick or quick[0] != "ok":
                    return None
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                required = {"tmcra_service_batches", "v4_batch_journal"}
                if not required.issubset(tables):
                    return "none"
                rows = connection.execute(
                    "SELECT journal.* "
                    "FROM tmcra_service_batches AS batches "
                    "JOIN v4_batch_journal AS journal "
                    "ON journal.scope_id=batches.scope_id "
                    "AND journal.session_id=batches.session_id "
                    "AND journal.batch_index=batches.batch_index "
                    "WHERE batches.operation_id=?",
                    (job_id,),
                ).fetchall()
                audited_writer_state = V4StorageAdapter._audited_writer_state_batch_ids(
                    connection,
                    paths=paths,
                    job_id=job_id,
                    rows=rows,
                )
                definitive_reviewer_failures = (
                    V4StorageAdapter._definitive_reviewer_failure_job_ids(
                        connection,
                        rows=rows,
                    )
                )
            finally:
                connection.close()
        except (OSError, sqlite3.DatabaseError, json.JSONDecodeError, TypeError, ValueError):
            return None
        cancelled_batch_id = V4StorageAdapter._cancelled_local_inference_batch_id(
            paths=paths,
            operation=operation,
            job_id=job_id,
            rows=rows,
        )
        if cancelled_batch_id:
            return "audited_local_inference_cancelled"
        if audited_writer_state:
            return "audited_writer_state"
        if definitive_reviewer_failures is None:
            return None
        if definitive_reviewer_failures:
            return "definitive_provider_failure"
        recovery_mode = "none"
        for row in rows:
            status = str(row["status"] or "")
            response = str(
                row["response_json"] if "response_json" in row.keys() else ""
            )
            if status == "prepared":
                try:
                    prepared_history = json.loads(
                        str(
                            row["recovery_history_json"]
                            if "recovery_history_json" in row.keys()
                            else "[]"
                        )
                        or "[]"
                    )
                except json.JSONDecodeError:
                    return None
                prepared_reasons = [
                    str(item.get("reason") or "")
                    for item in prepared_history
                    if isinstance(item, Mapping)
                    and str(item.get("reason") or "")
                    in {
                        "known_invalid_primary_response_replacement",
                        "schema_constrained_invalid_response_replacement",
                    }
                ]
                if prepared_reasons == [
                    "known_invalid_primary_response_replacement",
                    "schema_constrained_invalid_response_replacement",
                ]:
                    recovery_mode = "schema_constrained_invalid_response_prepared"
                continue
            if status == "committed":
                continue
            if status == "validated":
                try:
                    validated_metadata = json.loads(
                        str(
                            row["response_metadata_json"]
                            if "response_metadata_json" in row.keys()
                            else "{}"
                        )
                        or "{}"
                    )
                except json.JSONDecodeError:
                    return None
                if (
                    not isinstance(validated_metadata, Mapping)
                    or not response
                    or not V4StorageAdapter._has_durable_failed_batch_raw_response(
                    operation=operation,
                    job_id=job_id,
                    row=row,
                    metadata=validated_metadata,
                    )
                ):
                    return None
                recovery_mode = "validation"
                continue
            if status in {"api_started", "outcome_unknown"}:
                return None
            if status != "failed":
                return None
            try:
                metadata = json.loads(
                    str(
                        row["response_metadata_json"]
                        if "response_metadata_json" in row.keys()
                        else "{}"
                    )
                    or "{}"
                )
            except json.JSONDecodeError:
                return None
            if not isinstance(metadata, Mapping):
                return None
            if response:
                if (
                    str(metadata.get("status") or "") != "completed"
                    or metadata.get("physical_api_call") is not True
                    or int(metadata.get("http_status") or 0) != 200
                ):
                    return None
                if not V4StorageAdapter._has_durable_failed_batch_raw_response(
                    operation=operation,
                    job_id=job_id,
                    row=row,
                    metadata=metadata,
                ):
                    return None
                recovery_mode = "validation"
                continue
            if (
                str(metadata.get("status") or "") == "completed"
                and metadata.get("physical_api_call") is True
                and int(metadata.get("http_status") or 0) == 200
            ):
                if V4StorageAdapter._has_durable_failed_batch_raw_response(
                    operation=operation,
                    job_id=job_id,
                    row=row,
                    metadata=metadata,
                ):
                    recovery_mode = "validation"
                    continue
                error = str(row["error"] if "error" in row.keys() else "")
                try:
                    recovery_history = json.loads(
                        str(
                            row["recovery_history_json"]
                            if "recovery_history_json" in row.keys()
                            else "[]"
                        )
                        or "[]"
                    )
                except json.JSONDecodeError:
                    return None
                if (
                    not isinstance(recovery_history, list)
                    or not V4StorageAdapter._known_invalid_primary_response(
                        metadata, error=error
                    )
                ):
                    return None
                replacement_count = sum(
                    1
                    for item in recovery_history
                    if isinstance(item, Mapping)
                    and str(item.get("reason") or "")
                    in {
                        "known_invalid_primary_response_replacement",
                        "schema_constrained_invalid_response_replacement",
                    }
                )
                if replacement_count == 0:
                    recovery_mode = "definitive_invalid_response"
                    continue
                if (
                    replacement_count == 1
                    and str(metadata.get("model") or "")
                    == _active_local_writer_model()
                    and not str(metadata.get("response_schema_sha256") or "")
                ):
                    recovery_mode = "schema_constrained_invalid_response"
                    continue
                return None
            error = str(row["error"] if "error" in row.keys() else "")
            if (
                str(metadata.get("status") or "") != "http_error"
                or int(metadata.get("http_status") or 0) != 402
                or metadata.get("physical_api_call") is not True
                or not error.startswith("BatchAPIError:")
                or "HTTP 402" not in error
            ):
                return None
            recovery_mode = "definitive_provider_failure"
        return recovery_mode

    @staticmethod
    def _prepare_cancelled_local_inference_retry(
        *, paths: ScopePaths, operation: Path, job_id: str
    ) -> int:
        proof_path = operation / _LOCAL_INFERENCE_CANCELLATION_PROOF_FILE
        with closing(sqlite3.connect(paths.database, timeout=30.0)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT journal.* FROM tmcra_service_batches AS batches "
                "JOIN v4_batch_journal AS journal "
                "ON journal.scope_id=batches.scope_id "
                "AND journal.session_id=batches.session_id "
                "AND journal.batch_index=batches.batch_index "
                "WHERE batches.operation_id=? ORDER BY batches.batch_index",
                (job_id,),
            ).fetchall()
            batch_id = V4StorageAdapter._cancelled_local_inference_batch_id(
                paths=paths,
                operation=operation,
                job_id=job_id,
                rows=rows,
            )
            if not batch_id:
                raise V4AdapterError(
                    "local inference cancellation proof changed before recovery"
                )
            proof_sha256 = _sha256_file(proof_path)
            recovered_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            row = next(row for row in rows if str(row["batch_id"] or "") == batch_id)
            try:
                history = json.loads(str(row["recovery_history_json"] or "[]"))
            except (KeyError, json.JSONDecodeError) as exc:
                raise V4AdapterError(
                    "local inference cancellation recovery history is invalid"
                ) from exc
            if not isinstance(history, list) or any(
                isinstance(item, Mapping)
                and str(item.get("reason") or "")
                == "audited_local_inference_cancelled"
                for item in history
            ):
                raise V4AdapterError(
                    "local inference cancellation replacement budget is exhausted"
                )
            history.append(
                {
                    "schema_version": "tmcra.service.cancelled-local-inference-recovery.1",
                    "reason": "audited_local_inference_cancelled",
                    "model": _active_local_writer_model(),
                    "prior_request_sha256": str(row["request_sha256"] or ""),
                    "cancellation_proof_file_sha256": proof_sha256,
                    "recovered_at": recovered_at,
                    "physical_api_calls": 0,
                    "replacement_budget": 1,
                }
            )
            updated = connection.execute(
                "UPDATE v4_batch_journal SET status='prepared',api_started_at='',"
                "response_metadata_json='{}',error='',recovery_history_json=?,updated_at=? "
                "WHERE batch_id=? AND status IN ('api_started','outcome_unknown') "
                "AND response_json='' AND request_sha256=?",
                (
                    json.dumps(history, ensure_ascii=True, separators=(",", ":")),
                    recovered_at,
                    batch_id,
                    str(row["request_sha256"] or ""),
                ),
            ).rowcount
            if updated != 1:
                raise V4AdapterError(
                    "cancelled local inference batch could not recover atomically"
                )
            connection.commit()
        _atomic_json(
            operation / f"cancelled_local_inference_recovery.{time.time_ns()}.json",
            {
                "schema_version": "tmcra.service.cancelled-local-inference-recovery.1",
                "job_id": job_id,
                "batch_id": batch_id,
                "cancellation_proof_file_sha256": proof_sha256,
                "replacement_calls_authorized": 1,
                "physical_api_calls": 0,
                "recovered_at": recovered_at,
            },
        )
        return 1

    @staticmethod
    def _known_invalid_primary_response(
        metadata: Mapping[str, Any], *, error: str
    ) -> bool:
        """Prove a provider response completed but failed before validation."""

        return bool(
            str(metadata.get("status") or "") == "completed"
            and metadata.get("physical_api_call") is True
            and int(metadata.get("physical_api_calls") or 0) == 1
            and int(metadata.get("http_status") or 0) == 200
            and str(metadata.get("stage") or "") == "batch_flash"
            and str(metadata.get("physical_call_id") or "").startswith("dsc_")
            and bool(str(metadata.get("request_sha256") or ""))
            and bool(str(metadata.get("response_sha256") or ""))
            and error.startswith(("ProductWriterError:", "JSONDecodeError:"))
        )

    @staticmethod
    def _prepare_definitive_invalid_response_retry(
        *, paths: ScopePaths, operation: Path, job_id: str
    ) -> int:
        """Atomically reopen known unusable 200 responses for replacement."""

        with closing(sqlite3.connect(paths.database, timeout=30.0)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT journal.* FROM tmcra_service_batches AS batches "
                "JOIN v4_batch_journal AS journal "
                "ON journal.scope_id=batches.scope_id "
                "AND journal.session_id=batches.session_id "
                "AND journal.batch_index=batches.batch_index "
                "WHERE batches.operation_id=? ORDER BY batches.batch_index",
                (job_id,),
            ).fetchall()
            targets: list[sqlite3.Row] = []
            for row in rows:
                status = str(row["status"] or "")
                if status in {"prepared", "validated", "committed"}:
                    continue
                if status != "failed" or str(row["response_json"] or ""):
                    raise V4AdapterError(
                        "invalid-response recovery state changed before retry"
                    )
                try:
                    metadata = json.loads(
                        str(row["response_metadata_json"] or "{}")
                    )
                except json.JSONDecodeError as exc:
                    raise V4AdapterError(
                        "invalid-response recovery metadata is malformed"
                    ) from exc
                if not isinstance(metadata, Mapping) or not V4StorageAdapter._known_invalid_primary_response(
                    metadata, error=str(row["error"] or "")
                ):
                    raise V4AdapterError(
                        "invalid-response recovery proof changed before retry"
                    )
                targets.append(row)
            if not targets:
                raise V4AdapterError(
                    "invalid-response recovery has no eligible failed batch"
                )
            recovered_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            recovery_rows: list[dict[str, Any]] = []
            for row in targets:
                try:
                    history = json.loads(str(row["recovery_history_json"] or "[]"))
                except json.JSONDecodeError as exc:
                    raise V4AdapterError(
                        "invalid-response recovery history is malformed"
                    ) from exc
                if not isinstance(history, list):
                    raise V4AdapterError(
                        "invalid-response recovery history is malformed"
                    )
                if any(
                    isinstance(item, Mapping)
                    and str(item.get("reason") or "")
                    == "known_invalid_primary_response_replacement"
                    for item in history
                ):
                    raise V4AdapterError(
                        "invalid-response replacement was already consumed"
                    )
                recovery = {
                    "reason": "known_invalid_primary_response_replacement",
                    "prior_error_sha256": hashlib.sha256(
                        str(row["error"] or "").encode("utf-8")
                    ).hexdigest(),
                    "prior_response_metadata_sha256": hashlib.sha256(
                        str(row["response_metadata_json"] or "{}").encode("utf-8")
                    ).hexdigest(),
                    "recovered_at": recovered_at,
                    "physical_api_calls": 0,
                    "replacement_call_authorized": True,
                }
                history.append(recovery)
                updated = connection.execute(
                    "UPDATE v4_batch_journal SET status='prepared',api_started_at='',"
                    "response_metadata_json='{}',error='',recovery_history_json=?,updated_at=? "
                    "WHERE batch_id=? AND status='failed' AND response_json='' "
                    "AND error=? AND response_metadata_json=?",
                    (
                        json.dumps(history, ensure_ascii=True, sort_keys=True),
                        recovered_at,
                        str(row["batch_id"]),
                        str(row["error"]),
                        str(row["response_metadata_json"]),
                    ),
                ).rowcount
                if updated != 1:
                    raise V4AdapterError(
                        "invalid-response batch changed during atomic recovery"
                    )
                recovery_rows.append(
                    {
                        "batch_id": str(row["batch_id"]),
                        **recovery,
                    }
                )
            connection.commit()
        _atomic_json(
            operation / f"definitive_invalid_response_recovery.{time.time_ns()}.json",
            {
                "schema_version": "tmcra.service.invalid-response-recovery.1",
                "job_id": job_id,
                "batch_count": len(recovery_rows),
                "batches": recovery_rows,
                "physical_api_calls": 0,
                "replacement_calls_authorized": len(recovery_rows),
                "recovered_at": recovered_at,
            },
        )
        return len(recovery_rows)

    @staticmethod
    def _prepare_schema_constrained_invalid_response_retry(
        *, paths: ScopePaths, operation: Path, job_id: str
    ) -> int:
        """Authorize one final local-model replacement under decode-time schema."""

        with closing(sqlite3.connect(paths.database, timeout=30.0)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT journal.* FROM tmcra_service_batches AS batches "
                "JOIN v4_batch_journal AS journal ON journal.scope_id=batches.scope_id "
                "AND journal.session_id=batches.session_id "
                "AND journal.batch_index=batches.batch_index "
                "WHERE batches.operation_id=? ORDER BY batches.batch_index",
                (job_id,),
            ).fetchall()
            targets: list[sqlite3.Row] = []
            for row in rows:
                status = str(row["status"] or "")
                if status in {"prepared", "validated", "committed"}:
                    continue
                if status != "failed" or str(row["response_json"] or ""):
                    raise V4AdapterError(
                        "schema-constrained recovery state changed before retry"
                    )
                metadata = json.loads(str(row["response_metadata_json"] or "{}"))
                history = json.loads(str(row["recovery_history_json"] or "[]"))
                replacement_reasons = [
                    str(item.get("reason") or "")
                    for item in history
                    if isinstance(item, Mapping)
                    and str(item.get("reason") or "")
                    in {
                        "known_invalid_primary_response_replacement",
                        "schema_constrained_invalid_response_replacement",
                    }
                ]
                if (
                    not isinstance(metadata, Mapping)
                    or str(metadata.get("model") or "")
                    != _active_local_writer_model()
                    or str(metadata.get("response_schema_sha256") or "")
                    or not V4StorageAdapter._known_invalid_primary_response(
                        metadata, error=str(row["error"] or "")
                    )
                    or replacement_reasons
                    != ["known_invalid_primary_response_replacement"]
                ):
                    raise V4AdapterError(
                        "schema-constrained recovery proof changed before retry"
                    )
                targets.append(row)
            if not targets:
                raise V4AdapterError(
                    "schema-constrained recovery has no eligible failed batch"
                )
            recovered_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            recovery_rows: list[dict[str, Any]] = []
            for row in targets:
                history = json.loads(str(row["recovery_history_json"] or "[]"))
                recovery = {
                    "reason": "schema_constrained_invalid_response_replacement",
                    "prior_error_sha256": hashlib.sha256(
                        str(row["error"] or "").encode("utf-8")
                    ).hexdigest(),
                    "prior_response_metadata_sha256": hashlib.sha256(
                        str(row["response_metadata_json"] or "{}").encode("utf-8")
                    ).hexdigest(),
                    "recovered_at": recovered_at,
                    "physical_api_calls": 0,
                    "replacement_call_authorized": True,
                    "decode_constraint": "request_bound_json_schema",
                }
                history.append(recovery)
                updated = connection.execute(
                    "UPDATE v4_batch_journal SET status='prepared',api_started_at='',"
                    "response_metadata_json='{}',error='',recovery_history_json=?,updated_at=? "
                    "WHERE batch_id=? AND status='failed' AND response_json='' "
                    "AND error=? AND response_metadata_json=?",
                    (
                        json.dumps(history, ensure_ascii=True, sort_keys=True),
                        recovered_at,
                        str(row["batch_id"]),
                        str(row["error"]),
                        str(row["response_metadata_json"]),
                    ),
                ).rowcount
                if updated != 1:
                    raise V4AdapterError(
                        "schema-constrained batch changed during atomic recovery"
                    )
                recovery_rows.append({"batch_id": str(row["batch_id"]), **recovery})
            connection.commit()
        _atomic_json(
            operation / f"schema_constrained_invalid_response_recovery.{time.time_ns()}.json",
            {
                "schema_version": "tmcra.service.schema-constrained-response-recovery.1",
                "job_id": job_id,
                "batch_count": len(recovery_rows),
                "batches": recovery_rows,
                "physical_api_calls": 0,
                "replacement_calls_authorized": len(recovery_rows),
                "recovered_at": recovered_at,
            },
        )
        return len(recovery_rows)

    @staticmethod
    def _validate_prepared_schema_constrained_retry(
        *, paths: ScopePaths, operation: Path, job_id: str
    ) -> int:
        """Prove a constrained replacement was authorized but never started."""

        with closing(sqlite3.connect(paths.database, timeout=30.0)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT journal.* FROM tmcra_service_batches AS batches "
                "JOIN v4_batch_journal AS journal ON journal.scope_id=batches.scope_id "
                "AND journal.session_id=batches.session_id "
                "AND journal.batch_index=batches.batch_index "
                "WHERE batches.operation_id=? ORDER BY batches.batch_index",
                (job_id,),
            ).fetchall()
            prepared = 0
            for row in rows:
                status = str(row["status"] or "")
                if status in {"validated", "committed"}:
                    continue
                if (
                    status != "prepared"
                    or str(row["response_json"] or "")
                    or str(row["api_started_at"] or "")
                    or str(row["response_metadata_json"] or "{}") != "{}"
                    or str(row["error"] or "")
                ):
                    raise V4AdapterError(
                        "prepared schema-constrained recovery state changed"
                    )
                try:
                    history = json.loads(str(row["recovery_history_json"] or "[]"))
                except json.JSONDecodeError as exc:
                    raise V4AdapterError(
                        "prepared schema-constrained recovery history is invalid"
                    ) from exc
                replacement_reasons = [
                    str(item.get("reason") or "")
                    for item in history
                    if isinstance(item, Mapping)
                    and str(item.get("reason") or "")
                    in {
                        "known_invalid_primary_response_replacement",
                        "schema_constrained_invalid_response_replacement",
                    }
                ]
                if replacement_reasons != [
                    "known_invalid_primary_response_replacement",
                    "schema_constrained_invalid_response_replacement",
                ]:
                    raise V4AdapterError(
                        "prepared schema-constrained recovery proof is incomplete"
                    )
                prepared += 1
            if prepared <= 0:
                raise V4AdapterError(
                    "prepared schema-constrained recovery has no eligible batch"
                )
            return prepared

    @staticmethod
    def _definitive_reviewer_failure_job_ids(
        connection: sqlite3.Connection,
        *,
        rows: Sequence[sqlite3.Row],
    ) -> list[str] | None:
        """Prove nested reviewer calls were rejected before any response existed."""

        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "v4_reconciliation_jobs" not in tables:
            return []
        failed_response_batches = {
            str(row["batch_id"] or "")
            for row in rows
            if str(row["status"] or "") == "failed"
            and bool(str(row["response_json"] or ""))
            and "batch_id" in row.keys()
        }
        if not failed_response_batches:
            return []
        placeholders = ",".join("?" for _ in failed_response_batches)
        jobs = connection.execute(
            "SELECT * FROM v4_reconciliation_jobs "
            f"WHERE batch_id IN ({placeholders}) ORDER BY job_id",
            tuple(sorted(failed_response_batches)),
        ).fetchall()
        failed_jobs = [job for job in jobs if str(job["status"] or "") == "failed"]
        if not failed_jobs:
            return []
        allowed_states = {"completed", "failed", "pro_pending"}
        if any(str(job["status"] or "") not in allowed_states for job in jobs):
            return None
        eligible: list[str] = []
        for job in failed_jobs:
            try:
                metadata = json.loads(str(job["response_metadata_json"] or "{}"))
                request = json.loads(str(job["request_json"] or ""))
            except json.JSONDecodeError:
                return None
            error = str(job["error"] or "")
            if (
                not isinstance(metadata, Mapping)
                or not isinstance(request, Mapping)
                or str(job["response_json"] or "")
                or str(metadata.get("status") or "") != "http_error"
                or int(metadata.get("http_status") or 0) != 402
                or metadata.get("physical_api_call") is not True
                or int(metadata.get("physical_api_calls") or 0) != 1
                or not str(metadata.get("physical_call_id") or "").startswith("dsc_")
                or not str(metadata.get("request_sha256") or "")
                or not error.startswith("BatchAPIError:")
                or "HTTP 402" not in error
                or request.get("schema_version") != "tmcra.memory-reconcile.v4"
                or request.get("message_id") != job["message_id"]
                or request.get("canonical_slot_key") != job["canonical_slot_key"]
                or not isinstance(request.get("candidate_cited_leaves"), list)
                or type(request.get("exact_slot_match")) is not bool
                or not isinstance(request.get("new_cited_assertion"), Mapping)
            ):
                return None
            eligible.append(str(job["job_id"] or ""))
        return sorted(job_id for job_id in eligible if job_id) or None

    @staticmethod
    def _audited_writer_state_batch_ids(
        connection: sqlite3.Connection,
        *,
        paths: ScopePaths,
        job_id: str,
        rows: Sequence[sqlite3.Row],
    ) -> list[str] | None:
        """Prove a terminal local failure has no result and every Source is intact."""

        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {
            "tmcra_service_messages",
            "v4_source_journal",
            "records",
        }
        if not rows or not required.issubset(tables):
            return None
        operation_bindings, _batch_states, binding_violations = (
            V4StorageAdapter._source_operation_bindings(connection, paths=paths)
        )
        if binding_violations:
            return None
        failed_batch_ids: list[str] = []
        requested_message_ids: set[str] = set()
        for row in rows:
            keys = set(row.keys())
            required_row_keys = {
                "batch_id",
                "scope_id",
                "session_id",
                "request_json",
                "request_sha256",
                "status",
                "response_json",
                "response_metadata_json",
            }
            if not required_row_keys.issubset(keys):
                return None
            request_json = str(row["request_json"] or "")
            if (
                not request_json
                or hashlib.sha256(request_json.encode("utf-8")).hexdigest()
                != str(row["request_sha256"] or "")
            ):
                return None
            try:
                request = json.loads(request_json)
                metadata = json.loads(str(row["response_metadata_json"] or "{}"))
            except json.JSONDecodeError:
                return None
            if not isinstance(request, Mapping) or not isinstance(metadata, Mapping):
                return None
            batch_id = str(row["batch_id"] or "")
            status = str(row["status"] or "")
            if (
                str(row["scope_id"] or "") != paths.scope_id
                or str(request.get("batch_id") or "") != batch_id
            ):
                return None
            if status == "failed":
                try:
                    recovery_history = json.loads(
                        str(
                            row["recovery_history_json"]
                            if "recovery_history_json" in row.keys()
                            else "[]"
                        )
                        or "[]"
                    )
                except json.JSONDecodeError:
                    return None
                if not isinstance(recovery_history, list):
                    return None
                if V4StorageAdapter._audited_writer_failure_reason(
                    metadata,
                    response_json=str(row["response_json"] or ""),
                    error=str(row["error"] or ""),
                    recovery_history=recovery_history,
                ) is None:
                    return None
                failed_batch_ids.append(batch_id)
            elif status in {"prepared", "validated", "committed"}:
                if status in {"validated", "committed"} and not str(
                    row["response_json"] or ""
                ):
                    return None
            else:
                return None
            messages = request.get("messages")
            if not isinstance(messages, list) or not messages:
                return None
            for message in messages:
                if not isinstance(message, Mapping):
                    return None
                message_id = str(message.get("message_id") or "")
                if not message_id or message_id in requested_message_ids:
                    return None
                requested_message_ids.add(message_id)
                source = connection.execute(
                    "SELECT session_id,message_id,session_index,message_index,"
                    "message_role,timestamp,content,content_sha256,status,"
                    "source_record_id,source_turn_index,source_persisted_at "
                    "FROM v4_source_journal WHERE scope_id=? AND message_id=?",
                    (paths.scope_id, message_id),
                ).fetchone()
                service = connection.execute(
                    "SELECT session_id,message_index,role,timestamp,content_sha256 "
                    "FROM tmcra_service_messages "
                    "WHERE scope_id=? AND internal_message_id=?",
                    (paths.scope_id, message_id),
                ).fetchone()
                if source is None or service is None:
                    return None
                content = str(source[6] or "")
                content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
                source_record_id = str(source[9] or "")
                if (
                    str(source[0] or "") != str(row["session_id"] or "")
                    or str(source[1] or "") != message_id
                    or str(source[4] or "") != str(message.get("message_role") or "")
                    or str(source[5] or "") != str(message.get("timestamp") or "")
                    or str(source[7] or "") != content_sha256
                    or str(service[0] or "") != str(row["session_id"] or "")
                    or int(source[3]) != int(service[1])
                    or str(service[2] or "") != str(source[4] or "")
                    or str(service[3] or "") != str(source[5] or "")
                    or str(service[4] or "") != content_sha256
                    or operation_bindings.get(message_id) != job_id
                    or not source_record_id
                    or not str(source[11] or "")
                    or (status == "failed" and str(source[8] or "") != "failed")
                    or (status == "committed" and str(source[8] or "") != "enriched")
                ):
                    return None
                record = connection.execute(
                    "SELECT turn_index,metadata_json FROM records "
                    "WHERE scope_id=? AND memory_id=?",
                    (paths.scope_id, source_record_id),
                ).fetchone()
                if record is None:
                    return None
                try:
                    record_metadata = json.loads(str(record[1] or "{}"))
                except json.JSONDecodeError:
                    return None
                if not isinstance(record_metadata, Mapping):
                    return None
                sidecar = record_metadata.get("sidecar_hint_metadata")
                sidecar = sidecar if isinstance(sidecar, Mapping) else {}
                raw_content = record_metadata.get("raw_content")
                if (
                    not isinstance(raw_content, str)
                    or hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
                    != content_sha256
                    or str(record_metadata.get("source_record_id") or "")
                    != source_record_id
                    or int(record[0]) != int(source[10])
                    or int(record_metadata.get("session_index", -1)) != int(source[2])
                    or int(record_metadata.get("message_index", -1)) != int(source[3])
                    or str(
                        record_metadata.get("actor_role")
                        or record_metadata.get("speaker")
                        or sidecar.get("role")
                        or ""
                    )
                    != str(source[4] or "")
                ):
                    return None
        operation_message_ids = {
            message_id
            for message_id, operation_id in operation_bindings.items()
            if operation_id == job_id
        }
        operation_source_rows = []
        if operation_message_ids:
            placeholders = ",".join("?" for _ in operation_message_ids)
            operation_source_rows = connection.execute(
                "SELECT message_id,status FROM v4_source_journal "
                f"WHERE scope_id=? AND message_id IN ({placeholders})",
                (paths.scope_id, *sorted(operation_message_ids)),
            ).fetchall()
        if (
            any(str(row[1] or "") == "pending" for row in operation_source_rows)
            or {str(row[0] or "") for row in operation_source_rows}
            != requested_message_ids
        ):
            return None
        return sorted(failed_batch_ids) or None

    @staticmethod
    def _audited_writer_failure_reason(
        metadata: Mapping[str, Any],
        *,
        response_json: str,
        error: str = "",
        recovery_history: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[str, int] | None:
        """Classify bounded local inference failures safe for replacement."""

        if response_json:
            return None
        error_type = str(error or "").split(":", 1)[0]
        status = str(metadata.get("status") or "")
        if (
            error_type == "ProviderPoolExhausted"
            and (
                not metadata
                or (
                    status == "provider_pool_unavailable"
                    and metadata.get("physical_api_call") is False
                    and int(metadata.get("physical_api_calls") or 0) == 0
                )
            )
        ):
            return ("provider_lease_unavailable_before_call", 0)
        if (
            metadata.get("physical_api_call") is not True
            or str(metadata.get("model") or "") != _active_local_writer_model()
            or str(metadata.get("stage") or "") != "batch_flash"
        ):
            return None
        http_status = int(metadata.get("http_status") or 0)
        if status == "http_error" and http_status == 400:
            return ("local_provider_request_rejected", http_status)
        if status == "incomplete_response" and http_status == 200:
            return ("local_provider_incomplete_response", http_status)
        prior_reasons = {
            str(item.get("reason") or "")
            for item in recovery_history
            if isinstance(item, Mapping)
        }
        if (
            status in {"request_error", "transport_error", "timeout"}
            and int(metadata.get("physical_api_calls") or 0) == 1
            and str(metadata.get("physical_call_id") or "").startswith("dsc_")
            and bool(str(metadata.get("request_sha256") or ""))
            and "local_provider_unknown_outcome_replacement" not in prior_reasons
        ):
            return ("local_provider_unknown_outcome_replacement", http_status)
        return None

    @staticmethod
    def _prepare_audited_writer_retry(
        *, paths: ScopePaths, operation: Path, job_id: str
    ) -> int:
        with closing(sqlite3.connect(paths.database, timeout=30.0)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT journal.* FROM tmcra_service_batches AS batches "
                "JOIN v4_batch_journal AS journal "
                "ON journal.scope_id=batches.scope_id "
                "AND journal.session_id=batches.session_id "
                "AND journal.batch_index=batches.batch_index "
                "WHERE batches.operation_id=?",
                (job_id,),
            ).fetchall()
            batch_ids = V4StorageAdapter._audited_writer_state_batch_ids(
                connection,
                paths=paths,
                job_id=job_id,
                rows=rows,
            )
            if not batch_ids:
                raise V4AdapterError("audited Writer state changed before recovery")
            recovered_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            for batch_id in batch_ids:
                row = connection.execute(
                    "SELECT error,response_metadata_json,recovery_history_json,updated_at,"
                    "request_json,request_sha256 "
                    "FROM v4_batch_journal WHERE batch_id=? AND status='failed'",
                    (batch_id,),
                ).fetchone()
                if row is None:
                    raise V4AdapterError("audited Writer batch changed before recovery")
                try:
                    metadata = json.loads(str(row[1] or "{}"))
                    history = json.loads(str(row[2] or "[]"))
                except json.JSONDecodeError as exc:
                    raise V4AdapterError("audited Writer recovery metadata is invalid") from exc
                if not isinstance(metadata, Mapping) or not isinstance(history, list):
                    raise V4AdapterError("audited Writer recovery metadata is malformed")
                failure = V4StorageAdapter._audited_writer_failure_reason(
                    metadata,
                    response_json="",
                    error=str(row[0] or ""),
                    recovery_history=history,
                )
                if failure is None:
                    raise V4AdapterError("audited Writer failure classification changed")
                reason, http_status = failure
                try:
                    request = json.loads(str(row[4] or ""))
                except json.JSONDecodeError as exc:
                    raise V4AdapterError("audited Writer request is invalid") from exc
                if not isinstance(request, Mapping):
                    raise V4AdapterError("audited Writer request is malformed")
                original_messages = request.get("messages")
                original_unresolved = request.get("unresolved_interactions") or []
                if not isinstance(original_messages, list) or not isinstance(
                    original_unresolved, list
                ):
                    raise V4AdapterError("audited Writer request context is malformed")
                max_items, max_chars = writer_unresolved_limits_from_env()
                recovery_adaptation = "standard_unresolved_context_policy"
                recovery_attempt = 1
                if (
                    reason == "local_provider_incomplete_response"
                    and str(metadata.get("finish_reason") or "") == "length"
                ):
                    prior_incomplete_recoveries = sum(
                        1
                        for item in history
                        if isinstance(item, Mapping)
                        and item.get("reason") == reason
                    )
                    divisor = 2 ** min(prior_incomplete_recoveries, 2)
                    max_items = min(
                        max_items,
                        max(
                            _INCOMPLETE_RETRY_MIN_ITEMS,
                            _INCOMPLETE_RETRY_MAX_ITEMS // divisor,
                        ),
                    )
                    max_chars = min(
                        max_chars,
                        max(
                            _INCOMPLETE_RETRY_MIN_CHARS,
                            _INCOMPLETE_RETRY_MAX_CHARS // divisor,
                        ),
                    )
                    recovery_adaptation = "length_truncation_context_backoff"
                    recovery_attempt = prior_incomplete_recoveries + 1
                elif reason == "provider_lease_unavailable_before_call":
                    recovery_adaptation = "retry_after_provider_capacity_recovers"
                    recovery_attempt = 1 + sum(
                        1
                        for item in history
                        if isinstance(item, Mapping)
                        and item.get("reason") == reason
                    )
                elif reason == "local_provider_unknown_outcome_replacement":
                    recovery_adaptation = "single_bounded_pure_inference_replacement"
                    recovery_attempt = 1
                recovered_request = dict(request)
                recovered_request["unresolved_interactions"] = (
                    select_unresolved_interactions(
                        original_unresolved,
                        original_messages,
                        max_items=max_items,
                        max_chars=max_chars,
                    )
                )
                if recovered_request.get("messages") != original_messages:
                    raise V4AdapterError("audited Writer recovery changed Source messages")
                recovered_request_json = compact_json(recovered_request)
                recovered_request_sha256 = hashlib.sha256(
                    recovered_request_json.encode("utf-8")
                ).hexdigest()
                history.append(
                    {
                        "schema_version": "tmcra.service.audited-writer-recovery.1",
                        "reason": reason,
                        "http_status": http_status,
                        "model": _active_local_writer_model(),
                        "prior_error_sha256": hashlib.sha256(
                            str(row[0] or "").encode("utf-8")
                        ).hexdigest(),
                        "prior_response_metadata_sha256": hashlib.sha256(
                            str(row[1] or "{}").encode("utf-8")
                        ).hexdigest(),
                        "prior_updated_at": str(row[3] or ""),
                        "prior_request_sha256": str(row[5] or ""),
                        "recovered_request_sha256": recovered_request_sha256,
                        "unresolved_context_policy": UNRESOLVED_CONTEXT_POLICY_VERSION,
                        "recovery_adaptation": recovery_adaptation,
                        "recovery_attempt": recovery_attempt,
                        "recovered_unresolved_max_items": max_items,
                        "recovered_unresolved_max_chars": max_chars,
                        "prior_unresolved_count": len(original_unresolved),
                        "recovered_unresolved_count": len(
                            recovered_request["unresolved_interactions"]
                        ),
                        "recovered_at": recovered_at,
                        "physical_api_calls": 0,
                        "prior_physical_api_calls": int(
                            metadata.get("physical_api_calls") or 0
                        ),
                        "prior_outcome_unknown": reason
                        == "local_provider_unknown_outcome_replacement",
                        "replacement_budget": 1
                        if reason == "local_provider_unknown_outcome_replacement"
                        else None,
                    }
                )
                updated = connection.execute(
                    "UPDATE v4_batch_journal SET status='prepared',api_started_at='',"
                    "request_json=?,request_sha256=?,response_metadata_json='{}',"
                    "error='',recovery_history_json=?,updated_at=? "
                    "WHERE batch_id=? AND status='failed' AND response_json='' "
                    "AND error=? AND response_metadata_json=? AND request_json=? "
                    "AND request_sha256=?",
                    (
                        recovered_request_json,
                        recovered_request_sha256,
                        json.dumps(history, ensure_ascii=True, separators=(",", ":")),
                        recovered_at,
                        batch_id,
                        str(row[0] or ""),
                        str(row[1] or "{}"),
                        str(row[4] or ""),
                        str(row[5] or ""),
                    ),
                ).rowcount
                if updated != 1:
                    raise V4AdapterError("audited Writer batch could not recover atomically")
            connection.commit()
            return len(batch_ids)

    @staticmethod
    def _prepare_definitive_reviewer_retry(
        *, paths: ScopePaths, operation: Path, job_id: str
    ) -> int:
        recovery_rows: list[dict[str, Any]] = []
        with closing(sqlite3.connect(paths.database, timeout=30.0)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT journal.* FROM tmcra_service_batches AS batches "
                "JOIN v4_batch_journal AS journal "
                "ON journal.scope_id=batches.scope_id "
                "AND journal.session_id=batches.session_id "
                "AND journal.batch_index=batches.batch_index "
                "WHERE batches.operation_id=?",
                (job_id,),
            ).fetchall()
            reviewer_job_ids = (
                V4StorageAdapter._definitive_reviewer_failure_job_ids(
                    connection,
                    rows=rows,
                )
            )
            if reviewer_job_ids is None:
                raise V4AdapterError(
                    "definitive reviewer failure state changed before recovery"
                )
            if not reviewer_job_ids:
                connection.rollback()
                return 0
            recovered_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            for reviewer_job_id in reviewer_job_ids:
                row = connection.execute(
                    "SELECT error,response_json,response_metadata_json,updated_at "
                    "FROM v4_reconciliation_jobs "
                    "WHERE job_id=? AND status='failed'",
                    (reviewer_job_id,),
                ).fetchone()
                if row is None:
                    raise V4AdapterError(
                        "definitive reviewer failure changed before recovery"
                    )
                prior_error = str(row[0] or "")
                prior_response = str(row[1] or "")
                prior_metadata = str(row[2] or "{}")
                if prior_response:
                    raise V4AdapterError(
                        "definitive reviewer failure unexpectedly has a response"
                    )
                updated = connection.execute(
                    "UPDATE v4_reconciliation_jobs SET status='pro_pending',"
                    "error='',updated_at=? WHERE job_id=? AND status='failed' "
                    "AND response_json='' AND error=? AND response_metadata_json=?",
                    (
                        recovered_at,
                        reviewer_job_id,
                        prior_error,
                        prior_metadata,
                    ),
                ).rowcount
                if updated != 1:
                    raise V4AdapterError(
                        "definitive reviewer failure could not recover atomically"
                    )
                recovery_rows.append(
                    {
                        "reviewer_job_id": reviewer_job_id,
                        "prior_error_sha256": hashlib.sha256(
                            prior_error.encode("utf-8")
                        ).hexdigest(),
                        "prior_response_metadata_sha256": hashlib.sha256(
                            prior_metadata.encode("utf-8")
                        ).hexdigest(),
                        "prior_updated_at": str(row[3] or ""),
                    }
                )
            connection.commit()
        _atomic_json(
            operation / f"definitive_reviewer_recovery.{time.time_ns()}.json",
            {
                "schema_version": "tmcra.service.definitive-reviewer-recovery.1",
                "job_id": job_id,
                "reason": "reviewer_billing_rejected_before_response",
                "http_status": 402,
                "recoveries": recovery_rows,
                "recovered_at": recovered_at,
                "physical_api_calls": 0,
            },
        )
        return len(recovery_rows)

    @staticmethod
    def _next_operation_log(operation: Path, stem: str) -> Path:
        candidate = operation / f"{stem}.log"
        attempt = 1
        while candidate.exists():
            candidate = operation / f"{stem}.retry-{attempt}.log"
            attempt += 1
        return candidate

    @staticmethod
    def _validate_index_artifacts(
        paths: ScopePaths,
        index_path: Path,
        report_path: Path,
        database_path: Path | None = None,
    ) -> dict[str, Any]:
        database_path = (database_path or paths.database).resolve()
        if not index_path.is_file() or index_path.stat().st_size <= 0:
            raise V4AdapterError("online index artifact is missing or empty")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise V4AdapterError("online index report is unreadable") from exc
        rows = report.get("rows") if isinstance(report, dict) else None
        if (
            not isinstance(report, dict)
            or report.get("status") != "complete"
            or report.get("row_count") != 1
            or not isinstance(rows, list)
            or len(rows) != 1
        ):
            raise V4AdapterError("online index report has an invalid completion contract")
        row = rows[0]
        if not isinstance(row, Mapping):
            raise V4AdapterError("online index report row is not an object")
        if (
            row.get("scope_id") != paths.scope_id
            or Path(str(row.get("db_path", ""))).resolve() != database_path
            or Path(str(row.get("index_path", ""))).resolve() != index_path.resolve()
        ):
            raise V4AdapterError("online index report belongs to a different scope")
        return report

    @staticmethod
    def _validate_database_snapshot(
        database_path: Path, *, require_delete_journal: bool = False
    ) -> None:
        if not database_path.is_file() or database_path.stat().st_size <= 0:
            raise V4AdapterError("SQLite generation snapshot is missing or empty")
        wal_path = Path(f"{database_path}-wal")
        shm_path = Path(f"{database_path}-shm")
        try:
            if wal_path.exists():
                wal_stat = wal_path.stat()
                if not stat.S_ISREG(wal_stat.st_mode):
                    raise V4AdapterError("immutable SQLite WAL is not a regular file")
                if require_delete_journal or wal_stat.st_size != 0:
                    raise V4AdapterError(
                        "immutable SQLite generation has a forbidden WAL sidecar"
                    )
            if shm_path.exists():
                shm_stat = shm_path.stat()
                if not stat.S_ISREG(shm_stat.st_mode):
                    raise V4AdapterError("immutable SQLite SHM is not a regular file")
                if require_delete_journal:
                    raise V4AdapterError(
                        "immutable SQLite generation has a forbidden SHM sidecar"
                    )
        except V4AdapterError:
            raise
        except OSError as exc:
            raise V4AdapterError("immutable SQLite sidecars are unreadable") from exc
        try:
            with database_path.open("rb") as handle:
                header = handle.read(20)
            if (
                len(header) < 20
                or header[:16] != b"SQLite format 3\x00"
            ):
                raise V4AdapterError("SQLite generation snapshot has an invalid header")
            journal_header = header[18:20]
            if require_delete_journal and journal_header != b"\x01\x01":
                raise V4AdapterError(
                    "SQLite generation snapshot is not DELETE-journal normalized"
                )
            if not require_delete_journal and journal_header not in {
                b"\x01\x01",
                b"\x02\x02",
            }:
                raise V4AdapterError(
                    "legacy SQLite generation has an invalid journal header"
                )
            connection = sqlite3.connect(
                database_path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True
            )
            try:
                connection.execute("PRAGMA query_only=ON")
                result = connection.execute("PRAGMA quick_check").fetchone()
            finally:
                connection.close()
        except V4AdapterError:
            raise
        except (OSError, sqlite3.DatabaseError) as exc:
            raise V4AdapterError("SQLite generation snapshot is unreadable") from exc
        if not result or result[0] != "ok":
            raise V4AdapterError("SQLite generation snapshot failed integrity check")
        if require_delete_journal and (wal_path.exists() or shm_path.exists()):
            raise V4AdapterError("immutable SQLite validation created sidecar files")
        if wal_path.exists() and wal_path.stat().st_size != 0:
            raise V4AdapterError("immutable SQLite generation acquired a non-empty WAL")

    @staticmethod
    def _validate_legacy_live_database(database_path: Path) -> None:
        """Check a mutable legacy database while honoring its live WAL."""

        if not database_path.is_file() or database_path.stat().st_size <= 0:
            raise V4AdapterError("legacy SQLite database is missing or empty")
        try:
            connection = sqlite3.connect(
                database_path.resolve().as_uri() + "?mode=ro", uri=True
            )
            try:
                connection.execute("PRAGMA query_only=ON")
                result = connection.execute("PRAGMA quick_check").fetchone()
            finally:
                connection.close()
        except (OSError, sqlite3.DatabaseError) as exc:
            raise V4AdapterError("legacy SQLite database is unreadable") from exc
        if not result or result[0] != "ok":
            raise V4AdapterError("legacy SQLite database failed integrity check")

    @staticmethod
    def _normalize_uncommitted_generation_database(database_path: Path) -> None:
        """Make one private, uncommitted snapshot independent of WAL sidecars."""

        database_path = database_path.resolve()
        generation_dir = database_path.parent
        wal_path = Path(f"{database_path}-wal")
        shm_path = Path(f"{database_path}-shm")
        if not database_path.is_file() or database_path.stat().st_size <= 0:
            raise V4AdapterError("uncommitted SQLite generation is missing or empty")

        try:
            if wal_path.exists():
                wal_stat = wal_path.stat()
                if not stat.S_ISREG(wal_stat.st_mode):
                    raise V4AdapterError("uncommitted SQLite WAL is not a regular file")
                if wal_stat.st_size != 0:
                    raise V4AdapterError(
                        "uncommitted SQLite generation has a non-empty WAL"
                    )
            if shm_path.exists() and not shm_path.is_file():
                raise V4AdapterError("uncommitted SQLite SHM is not a regular file")
            generation_dir.chmod(0o700)
            database_path.chmod(0o600)
            # A previous read-only WAL probe may have left 0444 sidecars. They
            # are private to this not-yet-committed generation, so make only
            # those verified regular files writable for the checkpoint.
            for sidecar in (wal_path, shm_path):
                if sidecar.exists():
                    sidecar.chmod(0o600)
        except V4AdapterError:
            raise
        except OSError as exc:
            raise V4AdapterError(
                "uncommitted SQLite generation permissions are unusable"
            ) from exc

        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                database_path.as_uri() + "?mode=rw",
                uri=True,
                timeout=0.0,
                isolation_level=None,
            )
            connection.execute("PRAGMA busy_timeout=0")
            checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if (
                checkpoint is None
                or len(checkpoint) != 3
                or any(isinstance(value, bool) for value in checkpoint)
            ):
                raise V4AdapterError("SQLite WAL checkpoint returned an invalid result")
            busy, log_frames, checkpointed_frames = (int(value) for value in checkpoint)
            if busy != 0:
                raise V4AdapterError("uncommitted SQLite WAL checkpoint is busy")
            if (log_frames, checkpointed_frames) not in {(0, 0), (-1, -1)}:
                raise V4AdapterError(
                    "uncommitted SQLite generation has uncheckpointed WAL frames"
                )
            journal_mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
            if not journal_mode or str(journal_mode[0]).strip().lower() != "delete":
                raise V4AdapterError(
                    "uncommitted SQLite generation did not enter DELETE journal mode"
                )
        except V4AdapterError:
            raise
        except (OSError, sqlite3.DatabaseError, TypeError, ValueError) as exc:
            raise V4AdapterError(
                "uncommitted SQLite generation could not be normalized"
            ) from exc
        finally:
            if connection is not None:
                connection.close()

        try:
            if wal_path.exists() and wal_path.stat().st_size != 0:
                raise V4AdapterError(
                    "uncommitted SQLite generation retained a non-empty WAL"
                )
            for sidecar in (wal_path, shm_path):
                try:
                    sidecar.unlink()
                except FileNotFoundError:
                    pass
            with database_path.open("rb") as handle:
                header = handle.read(20)
            if (
                len(header) < 20
                or header[:16] != b"SQLite format 3\x00"
                or header[18:20] != b"\x01\x01"
            ):
                raise V4AdapterError(
                    "uncommitted SQLite generation retained WAL header state"
                )
            _fsync_file(database_path)
            database_path.chmod(0o444)
            try:
                directory_fd = os.open(str(generation_dir), os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except V4AdapterError:
            raise
        except OSError as exc:
            raise V4AdapterError(
                "normalized SQLite generation could not be made durable"
            ) from exc
        if wal_path.exists() or shm_path.exists():
            raise V4AdapterError("normalized SQLite generation retained sidecar files")

    @staticmethod
    def _validate_generation_hashes(
        active: Mapping[str, Any], database_path: Path, index_path: Path
    ) -> None:
        expected_database = str(active.get("database_sha256", ""))
        expected_index = str(active.get("index_sha256", ""))
        if expected_database and _sha256_file(database_path) != expected_database:
            raise V4AdapterError("active SQLite generation snapshot checksum mismatch")
        if expected_index and _sha256_file(index_path) != expected_index:
            raise V4AdapterError("active index artifact checksum mismatch")

    @staticmethod
    def _generation_manifest_identity(
        active: Mapping[str, Any], database_path: Path, index_path: Path
    ) -> tuple[str, ...]:
        """Bind a cache entry to the generation and the complete manifest."""

        try:
            canonical_manifest = json.dumps(
                dict(active),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise V4AdapterError("active index manifest is not canonicalizable") from exc
        return (
            str(active.get("schema_version") or ""),
            str(active.get("scope_id") or ""),
            str(active.get("generation_id") or ""),
            str(active.get("database_sha256") or ""),
            str(active.get("index_sha256") or ""),
            str(database_path),
            str(index_path),
            hashlib.sha256(canonical_manifest).hexdigest(),
        )

    @staticmethod
    def _generation_has_cacheable_hashes(active: Mapping[str, Any]) -> bool:
        """Only immutable manifests with both complete hashes may be cached."""

        hexadecimal = frozenset("0123456789abcdef")
        hashes = (
            str(active.get("database_sha256") or ""),
            str(active.get("index_sha256") or ""),
        )
        return all(len(value) == 64 and not (set(value) - hexadecimal) for value in hashes)

    @staticmethod
    def _stat_result_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
        def field(name: str, default: int = -1) -> int:
            return int(getattr(metadata, name, default))

        return (
            field("st_dev"),
            field("st_ino"),
            field("st_mode"),
            field("st_nlink"),
            field("st_uid"),
            field("st_gid"),
            field("st_size"),
            field("st_mtime_ns", int(metadata.st_mtime * 1_000_000_000)),
            field("st_ctime_ns", int(metadata.st_ctime * 1_000_000_000)),
            field("st_birthtime_ns"),
            field("st_file_attributes"),
            field("st_reparse_tag"),
        )

    @staticmethod
    def _artifact_stat_fingerprint(path: Path) -> tuple[int, ...]:
        """Return metadata that detects replacement as well as in-place writes."""

        try:
            metadata = path.stat()
        except OSError as exc:
            raise V4AdapterError("generation artifact metadata is unreadable") from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise V4AdapterError("generation artifact is missing or empty")
        return V4StorageAdapter._stat_result_fingerprint(metadata)

    @staticmethod
    def _generation_directory_stat_fingerprint(path: Path) -> tuple[int, ...]:
        try:
            metadata = path.stat()
        except OSError as exc:
            raise V4AdapterError("generation directory metadata is unreadable") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise V4AdapterError("generation directory is missing")
        return V4StorageAdapter._stat_result_fingerprint(metadata)

    @staticmethod
    def _sqlite_sidecar_stat_fingerprint(database_path: Path) -> tuple[int, ...]:
        values: list[int] = []
        for sidecar in (Path(f"{database_path}-wal"), Path(f"{database_path}-shm")):
            try:
                metadata = sidecar.stat()
            except FileNotFoundError:
                values.append(0)
                continue
            except OSError as exc:
                raise V4AdapterError("SQLite sidecar metadata is unreadable") from exc
            if not stat.S_ISREG(metadata.st_mode):
                raise V4AdapterError("SQLite sidecar is not a regular file")
            values.append(1)
            values.extend(V4StorageAdapter._stat_result_fingerprint(metadata))
        return tuple(values)

    @staticmethod
    def _generation_requires_delete_journal(active: Mapping[str, Any]) -> bool:
        contract = active.get("sqlite_snapshot_contract")
        if contract is None:
            return False
        if contract != _SQLITE_SNAPSHOT_CONTRACT_DELETE_IMMUTABLE_V1:
            raise V4AdapterError("active index has an unsupported SQLite snapshot contract")
        return True

    def _invalidate_generation_validation(self, manifest_path: Path) -> None:
        cache_key = str(manifest_path.resolve())
        with self._generation_validation_cache_lock:
            self._generation_validation_cache.pop(cache_key, None)

    def _generation_validation_lock(self, manifest_path: Path) -> threading.RLock:
        cache_key = str(manifest_path.resolve()).encode("utf-8")
        slot = int.from_bytes(hashlib.sha256(cache_key).digest()[:8], "big")
        return self._generation_validation_locks[
            slot % len(self._generation_validation_locks)
        ]

    def _cached_generation_is_valid(
        self,
        active: Mapping[str, Any],
        database_path: Path,
        index_path: Path,
        *,
        manifest_path: Path,
    ) -> bool:
        cache_key = str(manifest_path.resolve())
        if not self._generation_has_cacheable_hashes(active):
            self._invalidate_generation_validation(manifest_path)
            return False
        try:
            identity = self._generation_manifest_identity(
                active, database_path, index_path
            )
            database_fingerprint = self._artifact_stat_fingerprint(database_path)
            index_fingerprint = self._artifact_stat_fingerprint(index_path)
            directory_fingerprint = self._generation_directory_stat_fingerprint(
                database_path.parent
            )
            sidecar_fingerprint = self._sqlite_sidecar_stat_fingerprint(database_path)
        except V4AdapterError:
            self._invalidate_generation_validation(manifest_path)
            raise
        with self._generation_validation_cache_lock:
            cached = self._generation_validation_cache.get(cache_key)
            if (
                cached is not None
                and cached.manifest_identity == identity
                and cached.database_fingerprint == database_fingerprint
                and cached.index_fingerprint == index_fingerprint
                and cached.generation_directory_fingerprint == directory_fingerprint
                and cached.sqlite_sidecar_fingerprint == sidecar_fingerprint
            ):
                self._generation_validation_cache.move_to_end(cache_key)
                return True
            # A changed manifest or artifact must not leave a reusable stale
            # entry behind if the following full validation fails.
            self._generation_validation_cache.pop(cache_key, None)
        return False

    def _verify_generation_integrity(
        self,
        active: Mapping[str, Any],
        database_path: Path,
        index_path: Path,
    ) -> _GenerationValidationCacheEntry:
        identity = self._generation_manifest_identity(active, database_path, index_path)
        before_database = self._artifact_stat_fingerprint(database_path)
        before_index = self._artifact_stat_fingerprint(index_path)
        before_directory = self._generation_directory_stat_fingerprint(
            database_path.parent
        )
        before_sidecars = self._sqlite_sidecar_stat_fingerprint(database_path)
        self._validate_database_snapshot(
            database_path,
            require_delete_journal=self._generation_requires_delete_journal(active),
        )
        self._validate_generation_hashes(active, database_path, index_path)
        after_database = self._artifact_stat_fingerprint(database_path)
        after_index = self._artifact_stat_fingerprint(index_path)
        after_directory = self._generation_directory_stat_fingerprint(
            database_path.parent
        )
        after_sidecars = self._sqlite_sidecar_stat_fingerprint(database_path)
        if (
            before_database != after_database
            or before_index != after_index
            or before_directory != after_directory
            or before_sidecars != after_sidecars
        ):
            raise V4AdapterError("generation artifacts changed during integrity validation")
        return _GenerationValidationCacheEntry(
            manifest_identity=identity,
            database_fingerprint=after_database,
            index_fingerprint=after_index,
            generation_directory_fingerprint=after_directory,
            sqlite_sidecar_fingerprint=after_sidecars,
        )

    @staticmethod
    def _seal_generation_artifacts(
        database_path: Path, index_path: Path
    ) -> bool:
        """Durably seal a validated generation before it can be cached."""

        generation_dir = database_path.parent
        if index_path.parent != generation_dir:
            raise V4AdapterError("generation artifacts are not in one directory")
        try:
            database_mode = stat.S_IMODE(database_path.stat().st_mode)
            index_mode = stat.S_IMODE(index_path.stat().st_mode)
            directory_mode = stat.S_IMODE(generation_dir.stat().st_mode)
        except OSError as exc:
            raise V4AdapterError("generation permissions are unreadable") from exc
        changed = (
            database_mode != 0o444
            or index_mode != 0o444
            or directory_mode != 0o555
        )
        if not changed:
            return False
        try:
            # Writable artifacts have not yet been sealed, so flush their data
            # before removing write permission. Previously sealed retry paths
            # deliberately avoid reopening a 0444 file with O_RDWR.
            if database_mode & 0o222:
                _fsync_file(database_path)
            if index_mode & 0o222:
                _fsync_file(index_path)
            database_path.chmod(0o444)
            index_path.chmod(0o444)
            directory_fd: int | None
            try:
                directory_fd = os.open(str(generation_dir), os.O_RDONLY)
            except OSError:
                directory_fd = None
            try:
                generation_dir.chmod(0o555)
                if directory_fd is not None:
                    os.fsync(directory_fd)
            finally:
                if directory_fd is not None:
                    os.close(directory_fd)
        except OSError as exc:
            raise V4AdapterError("generation artifacts could not be sealed") from exc
        if (
            stat.S_IMODE(database_path.stat().st_mode) != 0o444
            or stat.S_IMODE(index_path.stat().st_mode) != 0o444
            or stat.S_IMODE(generation_dir.stat().st_mode) != 0o555
        ):
            raise V4AdapterError("generation artifact permissions are not immutable")
        return True

    def _verify_and_seal_generation(
        self,
        active: Mapping[str, Any],
        database_path: Path,
        index_path: Path,
        *,
        manifest_path: Path,
    ) -> _GenerationValidationCacheEntry:
        try:
            validated = self._verify_generation_integrity(
                active, database_path, index_path
            )
            if (
                self._generation_requires_delete_journal(active)
                and self._seal_generation_artifacts(database_path, index_path)
            ):
                # chmod changes mode/ctime, so create the cache entry from the
                # final sealed state and repeat the integrity check once.
                validated = self._verify_generation_integrity(
                    active, database_path, index_path
                )
            return validated
        except Exception:
            self._invalidate_generation_validation(manifest_path)
            raise

    def _remember_generation_validation(
        self,
        active: Mapping[str, Any],
        database_path: Path,
        index_path: Path,
        *,
        manifest_path: Path,
        validated: _GenerationValidationCacheEntry,
    ) -> None:
        cache_key = str(manifest_path.resolve())
        try:
            if not self._generation_has_cacheable_hashes(active):
                self._invalidate_generation_validation(manifest_path)
                return
            if validated.manifest_identity != self._generation_manifest_identity(
                active, database_path, index_path
            ):
                raise V4AdapterError(
                    "generation manifest changed after integrity validation"
                )
            current_database = self._artifact_stat_fingerprint(database_path)
            current_index = self._artifact_stat_fingerprint(index_path)
            current_directory = self._generation_directory_stat_fingerprint(
                database_path.parent
            )
            current_sidecars = self._sqlite_sidecar_stat_fingerprint(database_path)
            if (
                current_database != validated.database_fingerprint
                or current_index != validated.index_fingerprint
                or current_directory != validated.generation_directory_fingerprint
                or current_sidecars != validated.sqlite_sidecar_fingerprint
            ):
                raise V4AdapterError(
                    "generation artifacts changed after integrity validation"
                )
        except Exception:
            self._invalidate_generation_validation(manifest_path)
            raise
        with self._generation_validation_cache_lock:
            self._generation_validation_cache[cache_key] = validated
            self._generation_validation_cache.move_to_end(cache_key)
            while (
                len(self._generation_validation_cache)
                > self._generation_validation_cache_max_entries
            ):
                self._generation_validation_cache.popitem(last=False)

    def _validate_and_cache_generation_locked(
        self,
        active: Mapping[str, Any],
        database_path: Path,
        index_path: Path,
        *,
        manifest_path: Path,
    ) -> None:
        if self._cached_generation_is_valid(
            active,
            database_path,
            index_path,
            manifest_path=manifest_path,
        ):
            return
        validated = self._verify_and_seal_generation(
            active,
            database_path,
            index_path,
            manifest_path=manifest_path,
        )
        self._remember_generation_validation(
            active,
            database_path,
            index_path,
            manifest_path=manifest_path,
            validated=validated,
        )

    def _validate_and_cache_generation(
        self,
        active: Mapping[str, Any],
        database_path: Path,
        index_path: Path,
        *,
        manifest_path: Path,
    ) -> None:
        with self._generation_validation_lock(manifest_path):
            try:
                self._validate_and_cache_generation_locked(
                    active,
                    database_path,
                    index_path,
                    manifest_path=manifest_path,
                )
            except Exception:
                self._invalidate_generation_validation(manifest_path)
                raise

    def consolidate_slow(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        ledger_job_id: str | None = None,
        ledger_stage_id: str | None = None,
        usage_attribution: UsageAttribution = UNATTRIBUTED,
        provider_execution: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            provider_execution = normalize_user_provider_execution(
                provider_execution,
                stage="organizer",
            )
        except ValueError as exc:
            raise V4AdapterError(str(exc)) from exc
        provider_environment: dict[str, str] = {}
        if provider_execution is not None:
            if not ledger_job_id or not ledger_stage_id:
                raise V4AdapterError(
                    "user-provider consolidation requires job and stage identity"
                )
            provider_environment = {
                "TMCRA_USER_PROVIDER_EXECUTION_JSON": json.dumps(
                    provider_execution,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "TMCRA_SERVICE_TENANT_ID": tenant_id,
                "TMCRA_SERVICE_SCOPE_NAME": scope_name,
                "TMCRA_SERVICE_JOB_ID": ledger_job_id,
                "TMCRA_SERVICE_STAGE_ID": ledger_stage_id,
                "TMCRA_USAGE_ATTRIBUTION_JSON": json.dumps(
                    usage_attribution.as_dict(),
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
        provider_run_kwargs = (
            {"extra_env": provider_environment}
            if provider_environment
            else {}
        )
        paths = self.scope_paths(tenant_id, scope_name)
        if not paths.database.is_file():
            raise V4AdapterError("cannot consolidate a scope without a memory database")
        operation = paths.operations / job_id
        operation.mkdir(parents=True, exist_ok=True)
        commit_path = operation / "slow_commit.json"
        if commit_path.is_file():
            committed = json.loads(commit_path.read_text(encoding="utf-8"))
            if (
                committed.get("schema_version") == "tmcra.service.slow-commit.2"
                and committed.get("job_id") == job_id
                and committed.get("scope_id") == paths.scope_id
                and bool(dict(committed.get("subject_attribution") or {}).get("gate_passed"))
            ):
                return committed
            raise V4AdapterError("slow commit has an invalid production contract")

        attribution_path = operation / "subject_attribution_report.json"
        attribution: dict[str, Any] | None = None
        if attribution_path.is_file():
            try:
                value = json.loads(attribution_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = None
            if isinstance(value, dict) and value.get("status") == "complete" and value.get(
                "gate_passed"
            ):
                attribution = value
        if attribution is None:
            attribution_error: Exception | None = None
            try:
                self._run_with_writer_env(
                    [
                        str(self.python),
                        "-m",
                        "tmcra_service.subject_attribution",
                        "--database",
                        str(paths.database),
                        "--scope-id",
                        paths.scope_id,
                        "--output",
                        str(attribution_path),
                        "--apply",
                    ],
                    log_path=self._next_operation_log(operation, "subject_attribution"),
                    **provider_run_kwargs,
                )
            except Exception as exc:
                attribution_error = exc
            try:
                attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                if attribution_error is not None:
                    raise attribution_error
                raise V4AdapterError("subject attribution report is unreadable") from exc
        subject_metadata = [
            dict(result.get("call_metadata") or {})
            for result in list(attribution.get("results") or [])
            if isinstance(result, Mapping)
            and int(result.get("physical_api_calls", 0) or 0) > 0
            and isinstance(result.get("call_metadata"), Mapping)
        ]
        self._journal_provider_metadata(
            subject_metadata,
            tenant_id=tenant_id,
            scope_name=scope_name,
            job_id=ledger_job_id,
            stage_id=ledger_stage_id,
            operation="subject_attribution_pro",
            default_model=str(
                os.getenv("TMCRA_SUBJECT_ATTRIBUTION_MODEL")
                or os.getenv("TMCRA_WRITER_REVIEWER_MODEL")
                or _active_local_writer_model()
            ).strip(),
            usage_attribution=usage_attribution,
        )
        if (
            attribution.get("status") != "complete"
            or not attribution.get("gate_passed")
            or int(attribution.get("unresolved_routed_message_count", 0) or 0) != 0
        ):
            raise V4AdapterError("subject attribution gate did not resolve every routed message")

        prefix = [
            str(self.python),
            *(
                ["-m", "tmcra_service.user_provider_slow_graph"]
                if provider_execution is not None
                else [str(self.settings.v4_root / "tmcra_v4_slow_graph.py")]
            ),
            str(paths.database),
            "--repo",
            str(self.settings.integrated_repo),
        ]
        try:
            self._run_with_writer_env(
                [*prefix, "enqueue", paths.scope_id],
                log_path=self._next_operation_log(operation, "slow_enqueue"),
                **provider_run_kwargs,
            )
            self._run_with_writer_env(
                [
                    *prefix,
                    "drain",
                    "--workers",
                    str(
                        1
                        if provider_execution is not None
                        else self.settings.slow_graph_drain_concurrency
                    ),
                ],
                log_path=self._next_operation_log(operation, "slow_drain"),
                **provider_run_kwargs,
            )
            self._run_with_writer_env(
                [*prefix, "audit", paths.scope_id, "--require-promotion-coverage"],
                log_path=self._next_operation_log(operation, "slow_audit"),
                **provider_run_kwargs,
            )
        finally:
            self._journal_provider_metadata(
                self._slow_call_metadata(paths.database, paths.scope_id),
                tenant_id=tenant_id,
                scope_name=scope_name,
                job_id=ledger_job_id,
                stage_id=ledger_stage_id,
                operation="slow_graph_manager",
                default_model=str(
                    os.getenv("TMCRA_SLOW_GRAPH_MODEL")
                    or os.getenv("TMCRA_WRITER_REVIEWER_MODEL")
                    or _active_local_writer_model()
                ).strip(),
                usage_attribution=usage_attribution,
            )
        result = {
            "schema_version": "tmcra.service.slow-commit.2",
            "job_id": job_id,
            "scope_id": paths.scope_id,
            "subject_attribution": {
                "gate_passed": True,
                "report": str(attribution_path),
                "routed_message_count": int(
                    attribution.get("routed_message_count", 0) or 0
                ),
                "quarantined_count": int(attribution.get("quarantined_count", 0) or 0),
                "physical_api_calls": int(
                    attribution.get("physical_api_calls", 0) or 0
                ),
                "estimated_cost_cny": float(
                    attribution.get("estimated_cost_cny", 0.0) or 0.0
                ),
            },
            "completed_at": time.time(),
        }
        _atomic_json(commit_path, result)
        return result

    def slow_graph_recovery_plan(
        self,
        *,
        tenant_id: str,
        scope_name: str,
    ) -> dict[str, Any]:
        """Audit one recoverable failed Slow child without changing its state."""
        import tmcra_v4_slow_graph as slow_graph

        paths = self.scope_paths(tenant_id, scope_name)
        if not paths.database.is_file():
            return {"resumable": False, "reason": "memory_database_missing"}
        store = slow_graph.V4SlowGraphStore(
            paths.database,
            schema=slow_graph.load_graph_schema(self.settings.integrated_repo),
        )
        with store.connection() as connection:
            statuses = {
                str(row["status"]): int(row["count"] or 0)
                for row in connection.execute(
                    "SELECT status,COUNT(*) AS count FROM slow_graph_jobs "
                    "WHERE scope_id=? GROUP BY status",
                    (paths.scope_id,),
                )
            }
            failed = connection.execute(
                "SELECT job_id FROM slow_graph_jobs WHERE scope_id=? "
                "AND status='failed' ORDER BY created_at,job_id",
                (paths.scope_id,),
            ).fetchall()
            prepared_local = connection.execute(
                "SELECT jobs.job_id FROM slow_graph_local_revalidations AS recovery "
                "JOIN slow_graph_jobs AS jobs ON jobs.job_id=recovery.job_id "
                "WHERE jobs.scope_id=? AND jobs.status='pending' "
                "AND jobs.claim_token IS NULL ORDER BY recovery.created_at,recovery.job_id",
                (paths.scope_id,),
            ).fetchall()
            prepared_model = connection.execute(
                "SELECT jobs.job_id FROM slow_graph_model_validation_recoveries AS recovery "
                "JOIN slow_graph_jobs AS jobs ON jobs.job_id=recovery.job_id "
                "WHERE jobs.scope_id=? AND jobs.status='pending' "
                "AND jobs.claim_token IS NULL ORDER BY recovery.created_at,recovery.job_id",
                (paths.scope_id,),
            ).fetchall()
            pending_rows = connection.execute(
                "SELECT jobs.job_id,jobs.attempts,jobs.claim_token,"
                "jobs.claim_owner,jobs.lease_expires_at,"
                "COUNT(attempts.attempt_id) AS attempt_count "
                "FROM slow_graph_jobs AS jobs "
                "LEFT JOIN slow_graph_attempts AS attempts "
                "ON attempts.job_id=jobs.job_id "
                "WHERE jobs.scope_id=? AND jobs.status='pending' "
                "GROUP BY jobs.job_id,jobs.attempts,jobs.claim_token,"
                "jobs.claim_owner,jobs.lease_expires_at "
                "ORDER BY jobs.created_at,jobs.job_id",
                (paths.scope_id,),
            ).fetchall()
            recovery_audits = connection.execute(
                "SELECT recovery.recovery_id,recovery.job_id,'local' AS kind "
                "FROM slow_graph_local_revalidations AS recovery "
                "JOIN slow_graph_jobs AS jobs ON jobs.job_id=recovery.job_id "
                "WHERE jobs.scope_id=? AND recovery.state='completed' "
                "AND jobs.status='completed' AND EXISTS("
                "SELECT 1 FROM slow_graph_patches AS patches "
                "WHERE patches.job_id=jobs.job_id) "
                "UNION ALL "
                "SELECT recovery.recovery_id,recovery.job_id,'model' AS kind "
                "FROM slow_graph_model_validation_recoveries AS recovery "
                "JOIN slow_graph_jobs AS jobs ON jobs.job_id=recovery.job_id "
                "WHERE jobs.scope_id=? AND jobs.status='completed' AND EXISTS("
                "SELECT 1 FROM slow_graph_patches AS patches "
                "WHERE patches.job_id=jobs.job_id) "
                "ORDER BY 1,2,3",
                (paths.scope_id, paths.scope_id),
            ).fetchall()
        candidates = [str(row["job_id"]) for row in failed]
        prepared_local_ids = [str(row["job_id"]) for row in prepared_local]
        prepared_model_ids = [str(row["job_id"]) for row in prepared_model]
        recovery_ids = set(
            candidates + prepared_local_ids + prepared_model_ids
        )
        untouched_pending = bool(pending_rows) and all(
            int(row["attempts"] or 0) == 0
            and int(row["attempt_count"] or 0) == 0
            and row["claim_token"] is None
            and row["claim_owner"] is None
            and row["lease_expires_at"] is None
            for row in pending_rows
        )
        if (
            not recovery_ids
            and not candidates
            and int(statuses.get("failed", 0)) == 0
            and int(statuses.get("retryable", 0)) == 0
            and untouched_pending
            and recovery_audits
        ):
            pending_ids = [str(row["job_id"]) for row in pending_rows]
            audit_bindings = [
                {
                    "recovery_id": str(row["recovery_id"]),
                    "job_id": str(row["job_id"]),
                    "kind": str(row["kind"]),
                }
                for row in recovery_audits
            ]
            pending_sha256 = hashlib.sha256(
                compact_json(pending_ids).encode("utf-8")
            ).hexdigest()
            recovery_audit_sha256 = hashlib.sha256(
                compact_json(audit_bindings).encode("utf-8")
            ).hexdigest()
            recovery_id = "sgq_" + hashlib.sha256(
                compact_json(
                    {
                        "contract": (
                            _SLOW_UNATTEMPTED_QUEUE_CONTINUATION_CONTRACT_VERSION
                        ),
                        "scope_id": paths.scope_id,
                        "pending_job_ids_sha256": pending_sha256,
                        "recovery_audit_sha256": recovery_audit_sha256,
                    }
                ).encode("utf-8")
            ).hexdigest()[:32]
            completed_jobs = int(statuses.get("completed", 0))
            total_jobs = sum(statuses.values())
            evidence = {
                "recovery_id": recovery_id,
                "scope_id": paths.scope_id,
                "pending_job_count": len(pending_ids),
                "completed_job_count": completed_jobs,
                "pending_job_ids_sha256": pending_sha256,
                "recovery_audit_sha256": recovery_audit_sha256,
                "already_prepared": True,
                "queue_continuation": True,
            }
            return {
                "resumable": True,
                "mode": "audited_unattempted_queue_continuation",
                "external_api_calls_expected": len(pending_ids),
                "deterministic_local_repair": False,
                "recovery_fingerprint": (
                    f"{_SLOW_UNATTEMPTED_QUEUE_CONTINUATION_CONTRACT_VERSION}:"
                    f"{recovery_id}"
                ),
                "completed_job_count": completed_jobs,
                "pending_job_count": len(pending_ids),
                "failed_job_count": 0,
                "total_job_count": total_jobs,
                "progress_percent": (
                    round(100.0 * completed_jobs / total_jobs, 2)
                    if total_jobs
                    else 100.0
                ),
                "evidence": evidence,
            }
        if len(recovery_ids) != 1:
            return {
                "resumable": False,
                "reason": "slow_graph_failure_cardinality_not_one",
                "status_counts": statuses,
            }
        job_id = next(iter(recovery_ids))
        evidence: dict[str, Any]
        mode: str
        fingerprint: str
        deterministic_local_repair: bool
        if job_id in prepared_local_ids:
            planners = ("local",)
        elif job_id in prepared_model_ids:
            planners = ("model",)
        else:
            planners = ("local", "model")
        failures: list[str] = []
        for planner in planners:
            try:
                if planner == "local":
                    evidence = slow_graph.failed_raw_response_revalidation_plan(
                        store,
                        job_id,
                        allowed_normalization_codes=frozenset(
                            {"null_counterevidence_normalized_as_empty_list"}
                        ),
                    )
                    mode = "audited_local_saved_response_revalidation"
                    fingerprint = (
                        f"{_SLOW_LOCAL_REVALIDATION_FINGERPRINT_CONTRACT_VERSION}:"
                        f"{evidence['recovery_id']}:{evidence['state']}"
                    )
                    deterministic_local_repair = True
                else:
                    evidence = slow_graph.failed_model_validation_recovery_plan(
                        store, job_id
                    )
                    mode = "audited_model_validation_retry"
                    fingerprint = (
                        f"{_SLOW_MODEL_VALIDATION_RETRY_FINGERPRINT_CONTRACT_VERSION}:"
                        f"{evidence['recovery_id']}:"
                        f"{int(bool(evidence.get('already_prepared')))}"
                    )
                    deterministic_local_repair = False
                break
            except Exception as exc:
                failures.append(f"{planner}:{type(exc).__name__}")
        else:
            return {
                "resumable": False,
                "reason": "slow_graph_failure_not_audited_recovery_safe",
                "rejected_plans": failures,
                "status_counts": statuses,
            }
        total_jobs = sum(statuses.values())
        completed_jobs = int(statuses.get("completed", 0))
        return {
            "resumable": True,
            "mode": mode,
            "external_api_calls_expected": 0,
            "deterministic_local_repair": deterministic_local_repair,
            "recovery_fingerprint": fingerprint,
            "completed_job_count": completed_jobs,
            "pending_job_count": int(statuses.get("pending", 0)),
            "failed_job_count": int(statuses.get("failed", 0)),
            "total_job_count": total_jobs,
            "progress_percent": (
                round(100.0 * completed_jobs / total_jobs, 2)
                if total_jobs
                else 100.0
            ),
            "evidence": evidence,
        }

    def prepare_slow_graph_recovery(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        expected_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically reopen only the Slow child proven by the supplied audit."""
        import tmcra_v4_slow_graph as slow_graph

        plan = self.slow_graph_recovery_plan(
            tenant_id=tenant_id,
            scope_name=scope_name,
        )
        if not bool(plan.get("resumable")):
            raise V4AdapterError("slow graph recovery is no longer resumable")
        evidence = dict(plan.get("evidence") or {})
        expected = dict(expected_evidence)
        mode = str(plan.get("mode") or "")
        if mode == "audited_unattempted_queue_continuation":
            bound_fields = (
                "recovery_id",
                "scope_id",
                "pending_job_count",
                "completed_job_count",
                "pending_job_ids_sha256",
                "recovery_audit_sha256",
            )
        else:
            common_bound_fields = (
                "recovery_id",
                "job_id",
                "attempt_id",
                "error_sha256",
                "call_metadata_sha256",
            )
            bound_fields = common_bound_fields + (
                ("normalized_patch_sha256", "normalization_codes")
                if mode == "audited_local_saved_response_revalidation"
                else ("prior_physical_api_calls", "prompt_version")
            )
        if any(evidence.get(field) != expected.get(field) for field in bound_fields):
            raise V4AdapterError("slow graph recovery evidence changed before prepare")
        paths = self.scope_paths(tenant_id, scope_name)
        store = slow_graph.V4SlowGraphStore(
            paths.database,
            schema=slow_graph.load_graph_schema(self.settings.integrated_repo),
        )
        patch_id: str | None = None
        if mode == "audited_local_saved_response_revalidation":
            patch_id = slow_graph.revalidate_failed_raw_response(
                store,
                str(evidence["job_id"]),
                expected_recovery_id=str(evidence["recovery_id"]),
                allowed_normalization_codes=frozenset(
                    {"null_counterevidence_normalized_as_empty_list"}
                ),
            )
            result_mode = "audited_local_saved_response_revalidation_completed"
            evidence_after = {
                **evidence,
                "state": "completed",
                "already_completed": True,
            }
        elif mode == "audited_model_validation_retry":
            prepared = slow_graph.prepare_failed_model_validation_retry(
                store, str(evidence["job_id"])
            )
            if str(prepared.get("recovery_id") or "") != str(
                evidence["recovery_id"]
            ):
                raise V4AdapterError("slow graph retry identity changed during prepare")
            result_mode = "audited_model_validation_retry_prepared"
            evidence_after = {**prepared, "already_prepared": True}
        elif mode == "audited_unattempted_queue_continuation":
            result_mode = "audited_unattempted_queue_continuation_verified"
            evidence_after = {**evidence, "already_prepared": True}
        else:
            raise V4AdapterError("unsupported slow graph recovery mode")
        after = self.slow_graph_recovery_status(
            tenant_id=tenant_id,
            scope_name=scope_name,
        )
        return {
            "schema_version": "tmcra.service.slow-recovery.1",
            "mode": result_mode,
            "external_api_calls_performed": 0,
            "patch_id": patch_id,
            **after,
            "evidence": evidence_after,
        }

    def slow_graph_recovery_status(
        self,
        *,
        tenant_id: str,
        scope_name: str,
    ) -> dict[str, Any]:
        """Return sanitized child counts for recovery progress reporting."""
        import tmcra_v4_slow_graph as slow_graph

        paths = self.scope_paths(tenant_id, scope_name)
        if not paths.database.is_file():
            return {
                "completed_job_count": 0,
                "pending_job_count": 0,
                "failed_job_count": 0,
                "active_job_count": 0,
                "total_job_count": 0,
                "progress_percent": 0.0,
            }
        store = slow_graph.V4SlowGraphStore(
            paths.database,
            schema=slow_graph.load_graph_schema(self.settings.integrated_repo),
        )
        with store.connection() as connection:
            statuses = {
                str(row["status"]): int(row["count"] or 0)
                for row in connection.execute(
                    "SELECT status,COUNT(*) AS count FROM slow_graph_jobs "
                    "WHERE scope_id=? GROUP BY status",
                    (paths.scope_id,),
                )
            }
            active = int(
                connection.execute(
                    "SELECT COUNT(*) FROM slow_graph_jobs WHERE scope_id=? "
                    "AND status='pending' AND claim_token IS NOT NULL",
                    (paths.scope_id,),
                ).fetchone()[0]
                or 0
            )
        total = sum(statuses.values())
        completed = int(statuses.get("completed", 0))
        return {
            "completed_job_count": completed,
            "pending_job_count": int(statuses.get("pending", 0)),
            "failed_job_count": int(statuses.get("failed", 0)),
            "retryable_job_count": int(statuses.get("retryable", 0)),
            "active_job_count": active,
            "total_job_count": total,
            "progress_percent": (
                round(100.0 * completed / total, 2) if total else 100.0
            ),
        }

    def build_index(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        source_event_seq: int = 0,
        builder: Callable[..., Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if source_event_seq < 0:
            raise ValueError("source_event_seq must be non-negative")
        paths = self.scope_paths(tenant_id, scope_name)
        if not paths.database.is_file():
            raise V4AdapterError("cannot index a scope without a memory database")
        operation = paths.operations / job_id
        operation.mkdir(parents=True, exist_ok=True)
        paths.indexes.mkdir(parents=True, exist_ok=True)
        manifest_path = operation / "scope_manifest.jsonl"
        commit_path = operation / "index_commit.json"
        empty_report_path = operation / "empty_index_report.json"

        def activate_empty_scope(report: Mapping[str, Any]) -> dict[str, Any]:
            if (
                report.get("schema_version") != "tmcra.service.empty-index.1"
                or report.get("scope_id") != paths.scope_id
                or report.get("record_count") != 0
                or report.get("covers_through_event_seq") != 0
            ):
                raise V4AdapterError("empty index report is invalid")
            # Removing the base pointer first makes every concurrent recall
            # fail closed instead of observing a stale pre-deletion index.
            for active_path in (paths.active_index, paths.active_delta):
                with self._generation_validation_lock(active_path):
                    self._invalidate_generation_validation(active_path)
                    try:
                        active_path.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        raise V4AdapterError(
                            "empty scope index pointer could not be cleared"
                        ) from exc
            generation_prune = self._prune_index_generations_after_activation(
                tenant_id=tenant_id,
                scope_name=scope_name,
            )
            return {
                "active_index": None,
                "report": dict(report),
                "generation_prune": generation_prune,
            }

        if commit_path.is_file():
            committed = json.loads(commit_path.read_text(encoding="utf-8"))
            if committed.get("empty_scope") is True:
                report_path = Path(str(committed.get("report_path") or "")).resolve()
                if report_path != empty_report_path.resolve() or not report_path.is_file():
                    raise V4AdapterError("empty index commit references a missing report")
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise V4AdapterError("empty index report is unreadable") from exc
                if not isinstance(report, dict):
                    raise V4AdapterError("empty index report must be an object")
                return activate_empty_scope(report)
            active = dict(committed["active_index"])
            database_path = Path(str(active["database"])).resolve()
            index_path = Path(str(active["index"])).resolve()
            report_path = Path(str(committed["report_path"]))
            if index_path.is_file() and report_path.is_file():
                report = self._validate_index_artifacts(
                    paths, index_path, report_path, database_path
                )
                with self._generation_validation_lock(paths.active_index):
                    try:
                        validated = self._verify_and_seal_generation(
                            active,
                            database_path,
                            index_path,
                            manifest_path=paths.active_index,
                        )
                        current = None
                        if paths.active_index.is_file():
                            try:
                                current = json.loads(
                                    paths.active_index.read_text(encoding="utf-8")
                                )
                            except (OSError, json.JSONDecodeError):
                                current = None
                        if current != active:
                            _atomic_json(paths.active_index, active)
                        self._remember_generation_validation(
                            active,
                            database_path,
                            index_path,
                            manifest_path=paths.active_index,
                            validated=validated,
                        )
                    except Exception:
                        self._invalidate_generation_validation(paths.active_index)
                        raise
                generation_prune = self._prune_index_generations_after_activation(
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                )
                return {
                    "active_index": active,
                    "report": report,
                    "generation_prune": generation_prune,
                }
            raise V4AdapterError("index commit references missing durable artifacts")

        with closing(sqlite3.connect(paths.database)) as connection:
            record_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM records WHERE scope_id=?",
                    (paths.scope_id,),
                ).fetchone()[0]
            )
        if record_count == 0:
            if source_event_seq != 0:
                raise V4AdapterError("empty scope has a non-zero Source watermark")
            report = {
                "schema_version": "tmcra.service.empty-index.1",
                "scope_id": paths.scope_id,
                "record_count": 0,
                "covers_through_event_seq": 0,
                "activated_at": time.time(),
            }
            _atomic_json(empty_report_path, report)
            _atomic_json(
                commit_path,
                {
                    "schema_version": "tmcra.service.index-commit.1",
                    "empty_scope": True,
                    "active_index": None,
                    "report_path": str(empty_report_path.resolve()),
                    "completed_at": time.time(),
                },
            )
            return activate_empty_scope(report)

        attempt = 1
        while True:
            suffix = "" if attempt == 1 else f".retry-{attempt - 1}"
            generation_id = f"{job_id}{suffix}"
            generation_dir = paths.indexes / "generations" / generation_id
            database_path = generation_dir / "memory.sqlite3"
            index_path = generation_dir / "index.pt"
            report_path = operation / f"index_report{suffix}.json"
            log_path = operation / f"index{suffix}.log"
            if index_path.is_file() and report_path.is_file():
                try:
                    self._normalize_uncommitted_generation_database(database_path)
                    self._validate_database_snapshot(
                        database_path, require_delete_journal=True
                    )
                    self._validate_index_artifacts(
                        paths, index_path, report_path, database_path
                    )
                except V4AdapterError:
                    attempt += 1
                    continue
                break
            if not any(
                path.exists() for path in (database_path, index_path, report_path, log_path)
            ):
                break
            if database_path.is_file() and not index_path.exists() and not report_path.exists():
                try:
                    self._normalize_uncommitted_generation_database(database_path)
                    self._validate_database_snapshot(
                        database_path, require_delete_journal=True
                    )
                except V4AdapterError:
                    attempt += 1
                    continue
                break
            attempt += 1

        if database_path.is_file():
            self._normalize_uncommitted_generation_database(database_path)
            self._validate_database_snapshot(
                database_path, require_delete_journal=True
            )
        else:
            _sqlite_backup(paths.database, database_path)
            self._normalize_uncommitted_generation_database(database_path)
            self._validate_database_snapshot(
                database_path, require_delete_journal=True
            )
        manifest_path.write_text(
            json.dumps(
                {
                    "question_id": paths.question_id,
                    "scope_id": paths.scope_id,
                    "db_path": str(database_path),
                    "index_path": str(index_path),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if not (index_path.is_file() and report_path.is_file()) and builder is not None:
            builder(
                database_path=database_path,
                scope_id=paths.scope_id,
                index_path=index_path,
                report_path=report_path,
            )
        if not (index_path.is_file() and report_path.is_file()):
            command = [
                str(self.python),
                str(self.settings.v4_root / "tmcra_v4_online_runtime.py"),
                "build-index",
                "--scope-manifest",
                str(manifest_path),
                "--out-report",
                str(report_path),
                "--embedding-model",
                str(self.settings.embedding_model),
                "--device",
                self.settings.device,
                "--batch-size",
                "16",
            ]
            self._run_with_writer_env(command, log_path=log_path)
        if not index_path.is_file() or not report_path.is_file():
            raise V4AdapterError("index build completed without durable artifacts")
        report = self._validate_index_artifacts(
            paths, index_path, report_path, database_path
        )
        # The index subprocess is allowed to inspect this private generation,
        # but no WAL state may cross the immutable activation boundary.
        self._normalize_uncommitted_generation_database(database_path)
        self._validate_database_snapshot(
            database_path, require_delete_journal=True
        )
        database_sha256 = _sha256_file(database_path)
        index_sha256 = _sha256_file(index_path)
        generated_at = time.time()
        active = {
            "schema_version": "tmcra.service.active-index.1",
            "scope_id": paths.scope_id,
            "database": str(database_path),
            "index": str(index_path),
            "job_id": job_id,
            "activated_at": generated_at,
            "generation_id": generation_id,
            "covers_through_event_seq": int(source_event_seq),
            "sqlite_snapshot_contract": (
                _SQLITE_SNAPSHOT_CONTRACT_DELETE_IMMUTABLE_V1
            ),
            "generation_metadata": {
                "created_at": generated_at,
                "source_database": str(paths.database),
                "database_snapshot": str(database_path),
                "index_artifact": str(index_path),
                "covers_through_event_seq": int(source_event_seq),
            },
            "database_sha256": database_sha256,
            "index_sha256": index_sha256,
        }
        with self._generation_validation_lock(paths.active_index):
            try:
                validated = self._verify_and_seal_generation(
                    active,
                    database_path,
                    index_path,
                    manifest_path=paths.active_index,
                )
                _atomic_json(
                    commit_path,
                    {
                        "schema_version": "tmcra.service.index-commit.1",
                        "active_index": active,
                        "report_path": str(report_path),
                        "attempt": attempt,
                        "completed_at": time.time(),
                    },
                )
                # The active pointer is the last mutation. Any failure before this
                # point leaves the previous generation as the only active one.
                _atomic_json(paths.active_index, active)
                self._remember_generation_validation(
                    active,
                    database_path,
                    index_path,
                    manifest_path=paths.active_index,
                    validated=validated,
                )
            except Exception:
                self._invalidate_generation_validation(paths.active_index)
                raise
        generation_prune = self._prune_index_generations_after_activation(
            tenant_id=tenant_id,
            scope_name=scope_name,
        )
        return {
            "active_index": active,
            "report": report,
            "generation_prune": generation_prune,
        }

    def _validate_delta_manifest_locked(
        self,
        paths: ScopePaths,
        base: Mapping[str, Any],
        value: Mapping[str, Any],
        *,
        manifest_path: Path,
    ) -> dict[str, Any] | None:
        value = dict(value)
        if value.get("schema_version") != "tmcra.service.active-delta-index.1":
            raise V4AdapterError("active delta index schema is unsupported")
        if value.get("scope_id") != paths.scope_id:
            raise V4AdapterError("active delta index belongs to a different scope")
        # A base switch deliberately invalidates the old cumulative delta. It is
        # ignored rather than deleted so the activation sequence remains
        # recoverable and auditable.
        if (
            value.get("base_generation_id") != base.get("generation_id")
            or value.get("base_index_sha256") != base.get("index_sha256")
        ):
            return None
        database = Path(str(value.get("database") or "")).resolve()
        index = Path(str(value.get("index") or "")).resolve()
        try:
            database.relative_to((paths.indexes / "delta-generations").resolve())
            index.relative_to((paths.indexes / "delta-generations").resolve())
        except ValueError as exc:
            raise V4AdapterError("active delta index escaped its scope directory") from exc
        if database.parent != index.parent:
            raise V4AdapterError("active delta index and database are not one generation")
        self._validate_and_cache_generation_locked(
            value,
            database,
            index,
            manifest_path=manifest_path,
        )
        source_event_seq = value.get("source_event_seq")
        if (
            isinstance(source_event_seq, bool)
            or not isinstance(source_event_seq, int)
            or source_event_seq < 0
        ):
            raise V4AdapterError("active delta index watermark is invalid")
        return value

    def _validated_active_delta(
        self,
        paths: ScopePaths,
        base: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        path = paths.active_delta
        if not path.is_file():
            return None
        with self._generation_validation_lock(path):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise V4AdapterError("active delta index manifest is unreadable") from exc
            if not isinstance(value, dict):
                raise V4AdapterError("active delta index manifest must be an object")
            return self._validate_delta_manifest_locked(
                paths,
                base,
                value,
                manifest_path=path,
            )

    def build_delta_index(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        source_event_seq: int,
        builder: Callable[..., Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build and atomically activate a cumulative online delta generation."""

        if not callable(builder):
            raise TypeError("online delta builder must be callable")
        if source_event_seq < 0:
            raise ValueError("source_event_seq must be non-negative")
        paths = self.scope_paths(tenant_id, scope_name)
        base = self.active_snapshot(tenant_id, scope_name, include_delta=False)
        generation_id = f"{job_id}_delta"
        operation = paths.operations / job_id
        operation.mkdir(parents=True, exist_ok=True)
        commit_path = operation / "delta_commit.json"
        if commit_path.is_file():
            committed = json.loads(commit_path.read_text(encoding="utf-8"))
            active_delta = dict(committed.get("active_delta") or {})
            if not active_delta:
                raise V4AdapterError("delta commit has no active manifest")
            # Validate a replayed commit before changing the active pointer. A
            # damaged artifact must not replace a still-healthy active delta.
            with self._generation_validation_lock(commit_path):
                validated = self._validate_delta_manifest_locked(
                    paths,
                    base,
                    active_delta,
                    manifest_path=commit_path,
                )
            if validated != active_delta:
                raise V4AdapterError("delta commit cannot be reactivated against this base")
            with self._generation_validation_lock(paths.active_delta):
                _atomic_json(paths.active_delta, active_delta)
                self._invalidate_generation_validation(paths.active_delta)
                activated = self._validate_delta_manifest_locked(
                    paths,
                    base,
                    active_delta,
                    manifest_path=paths.active_delta,
                )
            if activated != active_delta:
                raise V4AdapterError("delta commit cannot be reactivated against this base")
            report = dict(committed.get("report") or {})
            snapshot = dict(base)
            snapshot["delta"] = active_delta
            generation_prune = self._prune_index_generations_after_activation(
                tenant_id=tenant_id,
                scope_name=scope_name,
            )
            return {
                "active_index": snapshot,
                "delta_index": active_delta,
                "report": report,
                "generation_prune": generation_prune,
            }

        generation_dir = paths.indexes / "delta-generations" / generation_id
        attempt = 1
        while generation_dir.exists():
            generation_id = f"{job_id}_delta.retry-{attempt}"
            generation_dir = paths.indexes / "delta-generations" / generation_id
            attempt += 1
        generation_dir.mkdir(parents=True, exist_ok=False)
        database_path = generation_dir / "memory.sqlite3"
        index_path = generation_dir / "delta.pt"
        report_path = operation / f"delta_report.retry-{attempt - 1}.json"
        previous = self._validated_active_delta(paths, base)
        _sqlite_backup(paths.database, database_path)
        self._normalize_uncommitted_generation_database(database_path)
        self._validate_database_snapshot(database_path, require_delete_journal=True)
        report = dict(
            builder(
                base_snapshot=base,
                live_database=database_path,
                source_event_seq=int(source_event_seq),
                index_path=index_path,
                report_path=report_path,
                previous_delta_path=(
                    None if previous is None else Path(str(previous["index"]))
                ),
            )
        )
        if not index_path.is_file() or index_path.stat().st_size <= 0:
            raise V4AdapterError("resident delta builder produced no index artifact")
        if not report_path.is_file():
            raise V4AdapterError("resident delta builder produced no report")
        self._validate_database_snapshot(database_path, require_delete_journal=True)
        database_sha256 = _sha256_file(database_path)
        index_sha256 = _sha256_file(index_path)
        active_delta = {
            "schema_version": "tmcra.service.active-delta-index.1",
            "scope_id": paths.scope_id,
            "base_generation_id": str(base.get("generation_id") or ""),
            "base_index_sha256": str(base.get("index_sha256") or ""),
            "database": str(database_path.resolve()),
            "index": str(index_path.resolve()),
            "database_sha256": database_sha256,
            "index_sha256": index_sha256,
            "source_event_seq": int(source_event_seq),
            "generation_id": generation_id,
            "activated_at": time.time(),
            "sqlite_snapshot_contract": _SQLITE_SNAPSHOT_CONTRACT_DELETE_IMMUTABLE_V1,
            "generation_metadata": {
                "source_database": str(paths.database.resolve()),
                "database_snapshot": str(database_path.resolve()),
                "index_artifact": str(index_path.resolve()),
                "covers_through_event_seq": int(source_event_seq),
            },
        }
        if not active_delta["base_generation_id"] or not active_delta["base_index_sha256"]:
            raise V4AdapterError("online delta requires a sealed base generation")
        with self._generation_validation_lock(paths.active_delta):
            try:
                validated_generation = self._verify_and_seal_generation(
                    active_delta,
                    database_path,
                    index_path,
                    manifest_path=paths.active_delta,
                )
                _atomic_json(
                    commit_path,
                    {
                        "schema_version": "tmcra.service.delta-index-commit.1",
                        "active_delta": active_delta,
                        "report": report,
                        "completed_at": time.time(),
                    },
                )
                _atomic_json(paths.active_delta, active_delta)
                self._remember_generation_validation(
                    active_delta,
                    database_path,
                    index_path,
                    manifest_path=paths.active_delta,
                    validated=validated_generation,
                )
            except Exception:
                self._invalidate_generation_validation(paths.active_delta)
                raise
        validated = self._validated_active_delta(paths, base)
        if validated != active_delta:
            raise V4AdapterError("activated delta failed post-commit validation")
        snapshot = dict(base)
        snapshot["delta"] = active_delta
        generation_prune = self._prune_index_generations_after_activation(
            tenant_id=tenant_id,
            scope_name=scope_name,
        )
        return {
            "active_index": snapshot,
            "delta_index": active_delta,
            "report": report,
            "generation_prune": generation_prune,
        }

    def active_snapshot(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        include_delta: bool = True,
    ) -> dict[str, Any]:
        paths = self.scope_paths(tenant_id, scope_name)
        path = paths.active_index
        with self._generation_validation_lock(path):
            try:
                if not path.is_file():
                    raise V4AdapterError("scope has no committed online index")
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise V4AdapterError("active index manifest is unreadable") from exc
                if not isinstance(value, dict):
                    raise V4AdapterError("active index manifest must be an object")
                database = Path(str(value.get("database", ""))).resolve()
                index = Path(str(value.get("index", ""))).resolve()
                if value.get("scope_id") != paths.scope_id:
                    raise V4AdapterError(
                        "active index manifest belongs to a different scope"
                    )
                legacy_manifest = not value.get("generation_id")
                if legacy_manifest:
                    if index.parent != paths.indexes.resolve() or database != paths.database:
                        raise V4AdapterError(
                            "legacy active index manifest is out of scope"
                        )
                else:
                    try:
                        index.relative_to(paths.indexes.resolve())
                    except ValueError as exc:
                        raise V4AdapterError(
                            "active index escaped the scope index directory"
                        ) from exc
                    if index.parent != database.parent:
                        raise V4AdapterError(
                            "active index and database are not one generation"
                        )
                if not database.is_file():
                    raise V4AdapterError("active scope database is missing")
                if not index.is_file() or index.stat().st_size <= 0:
                    raise V4AdapterError("active scope index is missing")
                if legacy_manifest:
                    self._invalidate_generation_validation(path)
                else:
                    self._validate_and_cache_generation_locked(
                        value,
                        database,
                        index,
                        manifest_path=path,
                    )
                if include_delta:
                    delta = self._validated_active_delta(paths, value)
                    if delta is not None:
                        value = dict(value)
                        value["delta"] = delta
                return value
            except Exception:
                self._invalidate_generation_validation(path)
                raise

    def scope_record_count(self, tenant_id: str, scope_name: str) -> int:
        """Return the durable record count without creating an empty scope."""

        paths = self.scope_paths(tenant_id, scope_name)
        if not paths.database.is_file():
            return 0
        try:
            with closing(sqlite3.connect(paths.database)) as connection:
                row = connection.execute(
                    "SELECT COUNT(*) FROM records WHERE scope_id=?",
                    (paths.scope_id,),
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise V4AdapterError("scope database is unreadable") from exc
        if row is None:
            raise V4AdapterError("scope database record count is unavailable")
        return int(row[0])

    @staticmethod
    def searchable_event_seq(snapshot: Mapping[str, Any]) -> int:
        """Return the event watermark covered by the active base plus delta."""

        base = snapshot.get("covers_through_event_seq", 0)
        if isinstance(base, bool) or not isinstance(base, int) or base < 0:
            raise V4AdapterError("active base index watermark is invalid")
        effective = int(base)
        delta = snapshot.get("delta")
        if isinstance(delta, Mapping):
            value = delta.get("source_event_seq")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise V4AdapterError("active delta index watermark is invalid")
            effective = max(effective, int(value))
        return effective

    def audit_searchable_watermarks(
        self,
        states: Sequence[Mapping[str, Any]],
        *,
        require_fresh: bool,
    ) -> dict[str, Any]:
        """Cross-check the control ledger against immutable active artifacts.

        Artifact-ahead states are recoverable after a crash between pointer
        activation and ledger advancement. Ledger-ahead states are unsafe and
        fail immediately because recall cannot satisfy the advertised watermark.
        """

        scope_count = 0
        fresh_count = 0
        stale_count = 0
        missing_count = 0
        reconciliation_count = 0
        max_lag = 0
        for raw in states:
            state = dict(raw)
            tenant_id = str(state.get("tenant_id") or "")
            scope_name = str(state.get("scope_name") or "")
            source = int(state.get("source_event_seq", 0) or 0)
            promoted = int(state.get("promoted_event_seq", 0) or 0)
            indexed = int(state.get("indexed_event_seq", 0) or 0)
            delta_indexed = int(state.get("delta_indexed_event_seq", indexed) or 0)
            if not tenant_id or not scope_name:
                raise V4AdapterError("scope watermark ledger has an invalid identity")
            if not (0 <= promoted <= indexed <= delta_indexed <= source):
                raise V4AdapterError(
                    "scope watermark ledger violates "
                    "promoted<=indexed<=delta<=source"
                )
            scope_count += 1
            if source == 0:
                fresh_count += 1
                continue
            try:
                snapshot = self.active_snapshot(tenant_id, scope_name)
            except V4AdapterError:
                if indexed > 0 or delta_indexed > 0:
                    raise V4AdapterError(
                        "scope ledger advertises searchable events without an active index"
                    )
                stale_count += 1
                missing_count += 1
                max_lag = max(max_lag, source)
                continue
            base_raw = snapshot.get("covers_through_event_seq")
            base_manifest = indexed if base_raw is None else int(base_raw)
            effective = self.searchable_event_seq(
                {**snapshot, "covers_through_event_seq": base_manifest}
            )
            if not (0 <= base_manifest <= effective <= source):
                raise V4AdapterError("active index watermark exceeds committed Source")
            if indexed > base_manifest or delta_indexed > effective:
                raise V4AdapterError("scope ledger watermark is ahead of active artifacts")
            if indexed < base_manifest or delta_indexed < effective:
                reconciliation_count += 1
            lag = source - effective
            max_lag = max(max_lag, lag)
            if lag:
                stale_count += 1
            else:
                fresh_count += 1
        ready = stale_count == 0
        return {
            "ready": ready if require_fresh else True,
            "fresh": ready,
            "scope_count": scope_count,
            "fresh_scope_count": fresh_count,
            "stale_scope_count": stale_count,
            "missing_index_scope_count": missing_count,
            "ledger_reconciliation_scope_count": reconciliation_count,
            "max_searchable_lag_events": max_lag,
        }

    def audit_active_indexes(self) -> list[dict[str, Any]]:
        """Validate every active manifest without needing unhashed tenant names."""
        snapshots: list[dict[str, Any]] = []
        pattern = self.settings.state_dir / "tenants"
        manifests = sorted(pattern.glob("*/scopes/*/active_index.json"))
        for manifest_path in manifests:
            with self._generation_validation_lock(manifest_path):
                try:
                    try:
                        value = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise V4AdapterError(
                            f"active index manifest is unreadable: {manifest_path}"
                        ) from exc
                    if not isinstance(value, dict):
                        raise V4AdapterError("active index manifest must be an object")
                    scope_root = manifest_path.parent.resolve()
                    indexes_root = (scope_root / "indexes").resolve()
                    database = Path(str(value.get("database", ""))).resolve()
                    index = Path(str(value.get("index", ""))).resolve()
                    scope_id = str(value.get("scope_id") or "")
                    if not scope_id.startswith("tmcra_v4:svc_"):
                        raise V4AdapterError(
                            "active index manifest has an invalid scope identity"
                        )
                    legacy_manifest = not value.get("generation_id")
                    if legacy_manifest:
                        expected_database = (
                            scope_root / "memory" / "native_memory.sqlite3"
                        ).resolve()
                        if database != expected_database or index.parent != indexes_root:
                            raise V4AdapterError(
                                "legacy active index manifest is out of scope"
                            )
                    else:
                        try:
                            index.relative_to(indexes_root)
                            database.relative_to(indexes_root)
                        except ValueError as exc:
                            raise V4AdapterError(
                                "active generation escaped the scope index directory"
                            ) from exc
                        if index.parent != database.parent:
                            raise V4AdapterError(
                                "active index and database are not one generation"
                            )
                    if not index.is_file() or index.stat().st_size <= 0:
                        raise V4AdapterError("active scope index is missing or empty")
                    if legacy_manifest:
                        self._invalidate_generation_validation(manifest_path)
                        self._validate_legacy_live_database(database)
                    else:
                        validated = self._verify_and_seal_generation(
                            value,
                            database,
                            index,
                            manifest_path=manifest_path,
                        )
                        self._remember_generation_validation(
                            value,
                            database,
                            index,
                            manifest_path=manifest_path,
                            validated=validated,
                        )
                    audit_paths = ScopePaths(
                        tenant_id="",
                        scope_name="",
                        question_id=scope_id.split(":", 1)[-1],
                        scope_id=scope_id,
                        root=scope_root,
                        database=(
                            scope_root / "memory" / "native_memory.sqlite3"
                        ).resolve(),
                        indexes=indexes_root,
                        operations=(scope_root / "operations").resolve(),
                        active_index=manifest_path.resolve(),
                        active_delta=(
                            scope_root / "active_delta_index.json"
                        ).resolve(),
                    )
                    delta = self._validated_active_delta(audit_paths, value)
                    snapshot = dict(value)
                    if delta is not None:
                        snapshot["delta"] = delta
                    snapshots.append(snapshot)
                except Exception:
                    self._invalidate_generation_validation(manifest_path)
                    raise
        return snapshots

    def compile_evidence(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        evidence: Mapping[str, Any],
        operation_id: str,
        ledger_stage_id: str | None = None,
        usage_attribution: UsageAttribution = UNATTRIBUTED,
    ) -> dict[str, Any]:
        paths = self.scope_paths(tenant_id, scope_name)
        operation = paths.operations / f"recall_{operation_id}"
        operation.mkdir(parents=True, exist_ok=False)
        evidence_path = operation / "evidence.jsonl"
        evidence_path.write_text(
            json.dumps(dict(evidence), ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output = operation / "compiled"
        compile_error: Exception | None = None
        try:
            self._run_with_writer_env(
                [
                    str(self.python),
                    str(self.settings.v4_root / "run_tmcra_v4_compile_evidence.py"),
                    "--evidence",
                    str(evidence_path),
                    "--out-dir",
                    str(output),
                    "--writer-env",
                    str(self.settings.writer_env),
                    "--workers",
                    "1",
                    *([
                        "--planner-provider", "openai-compatible",
                        "--planner-model", os.environ["TMCRA_EVIDENCE_COMPILER_MODEL"],
                        "--planner-base-url", os.environ["TMCRA_EVIDENCE_COMPILER_BASE_URL"],
                        "--planner-key-file", os.environ["TMCRA_LOCAL_WRITER_API_KEY_FILE"],
                        "--timeout", "600",
                    ] if os.environ.get("TMCRA_DEPLOYMENT_MODE") == "local" else []),
                ],
                log_path=operation / "compiler.log",
            )
        except Exception as exc:
            compile_error = exc
        metadata_values: list[dict[str, Any]] = []
        for artifact in sorted((output / "rows").glob("*.json")):
            if artifact.name.endswith(".failure.json"):
                continue
            try:
                value = json.loads(artifact.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise V4AdapterError("evidence compiler journal is unreadable") from exc
            if isinstance(value, Mapping) and isinstance(value.get("planner"), Mapping):
                metadata_values.append(dict(value["planner"]))
        failure_history = output / "planner_failure_history.jsonl"
        if failure_history.is_file():
            for line in failure_history.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise V4AdapterError("compiler failure history is invalid JSON") from exc
                if isinstance(value, Mapping) and isinstance(value.get("planner"), Mapping):
                    metadata_values.append(dict(value["planner"]))
        self._journal_provider_metadata(
            metadata_values,
            tenant_id=tenant_id,
            scope_name=scope_name,
            job_id=None,
            stage_id=ledger_stage_id,
            operation="evidence_compiler",
            default_model=str(
                os.getenv("TMCRA_EVIDENCE_PLANNER_MODEL")
                or os.getenv("TMCRA_WRITER_REVIEWER_MODEL")
                or _active_local_writer_model()
            ).strip(),
            usage_attribution=usage_attribution,
        )
        if compile_error is not None:
            if os.getenv("TMCRA_DEPLOYMENT_MODE") == "local":
                raise LocalEvidenceCompilationUnavailable(
                    "Local evidence compilation did not complete. Inspect the private compiler log; "
                    "the original memory and raw evidence remain available."
                ) from compile_error
            raise compile_error
        compiled_path = output / "evidence_windows.jsonl"
        if not compiled_path.is_file():
            raise V4AdapterError("evidence compiler completed without output")
        rows = [
            json.loads(line)
            for line in compiled_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != 1:
            raise V4AdapterError("evidence compiler returned an unexpected row count")
        return rows[0]


class _ResidentGraphAdapterCache(OrderedDict[tuple[str, ...], Any]):
    """Bounded scope adapter LRU with one scorer resident per serialized lane.

    Graph adapters contain scope-specific SQLite graph state, while the learned
    node/path scorer is scope-independent and expensive to load onto CUDA.  A
    V4OnlineEngine serializes every operation, so sharing one scorer across its
    bounded adapter LRU is safe and prevents scope switches from unloading and
    reloading model weights.
    """

    retain_across_scopes = True

    def __init__(
        self,
        *,
        max_entries: int,
        scorer_factory: Callable[[], Any],
    ) -> None:
        if max_entries <= 0:
            raise ValueError("graph adapter cache size must be positive")
        if not callable(scorer_factory):
            raise TypeError("graph scorer factory must be callable")
        super().__init__()
        self.max_entries = int(max_entries)
        self._scorer_factory = scorer_factory
        self._resident_scorer: Any | None = None

    @property
    def scorer_loaded(self) -> bool:
        return self._resident_scorer is not None

    def ensure_scorer(self) -> Any:
        scorer = self._resident_scorer
        if scorer is None:
            scorer = self._scorer_factory()
            if scorer is None:
                raise V4AdapterError("graph scorer factory returned no scorer")
            self._resident_scorer = scorer
        return scorer

    def get(self, key: tuple[str, ...], default: Any = None) -> Any:
        try:
            value = super().__getitem__(key)
        except KeyError:
            return default
        self.move_to_end(key)
        return value

    def __setitem__(self, key: tuple[str, ...], adapter: Any) -> None:
        if str(getattr(adapter, "retrieval_mode", "")).strip() != "hybrid_node_scored":
            raise V4AdapterError("graph adapter is not configured for learned retrieval")
        scorer = self.ensure_scorer()
        loader = getattr(adapter, "_node_scorer", None)
        if loader is None or not callable(loader):
            raise V4AdapterError("graph adapter has no learned scorer boundary")
        setattr(adapter, "_loaded_node_scorer", scorer)
        setattr(adapter, "_node_scorer_error", "")
        if loader() is not scorer:
            raise V4AdapterError("graph adapter rejected the resident scorer")
        if key in self:
            super().__delitem__(key)
        super().__setitem__(key, adapter)
        self.move_to_end(key)
        while len(self) > self.max_entries:
            self.popitem(last=False)

    def release(self) -> None:
        super().clear()
        self._resident_scorer = None


class V4OnlineEngine:
    """One serialized model replica used as a lane in the recall pool."""

    def __init__(
        self,
        settings: ServiceSettings,
        gpu_scheduler: GpuWorkloadScheduler | None = None,
    ) -> None:
        if str(settings.v4_root) not in sys.path:
            sys.path.insert(0, str(settings.v4_root))
        import tmcra_v4_online_runtime as runtime

        self.settings = settings
        self.runtime = runtime
        self.v3 = runtime._v3()
        self.args = argparse.Namespace(
            checkpoint=str(settings.checkpoint),
            cross_model=str(settings.cross_model),
            cross_max_length=1280,
            cross_batch_size=24,
            repo=str(settings.integrated_repo),
            harness=str(settings.native_harness),
            node_model=str(settings.node_model),
            path_model=str(settings.path_model),
            graph_device=settings.graph_device,
            learned_graph_enabled=settings.learned_graph_enabled,
            candidate_event_k=24,
            support_path_k=3,
            path_tunnel_rescue_k=2,
            graph_top_k=12,
            dense_k=32,
            slow_dense_k=24,
            graph_k=24,
            execution_lane="production",
            composition_mode="layered",
            packing_budget_mode="fixed",
            top_k=8,
            adaptive_simple_k=8,
            adaptive_standard_k=12,
            adaptive_complex_k=16,
            embedding_model=str(settings.embedding_model),
            text_dim=1024,
            embedding_max_length=8192,
            device=settings.device,
            subchunk_chars=1800,
            subchunk_overlap=200,
            batch_size=16,
        )
        from tmcra_local_models import apply_local_profile
        apply_local_profile(self.args)
        self.v3.graph_runtime_env(self.args)
        self.harness = self.v3.load_native_harness(
            settings.native_harness, settings.integrated_repo
        )
        self.harness.disable_topic_bucket_runtime()
        self.models = self.v3.OnlineModels(self.args)
        raw_planner = recall_planner_from_env()
        planner: Any = AuditedRecallPlanner(
            raw_planner,
            settings.state_dir / "recall_planner_repairs.jsonl",
        )
        self.planner = (
            ScheduledRecallPlanner(planner, gpu_scheduler)
            if gpu_scheduler is not None
            and getattr(raw_planner, "provider", None) == LOCAL_QWEN_PROVIDER
            else planner
        )
        self.graph_adapter_cache = (
            _ResidentGraphAdapterCache(
                max_entries=int(getattr(settings, "recall_scope_cache_size", 4)),
                scorer_factory=self._build_graph_scorer,
            )
            if settings.learned_graph_enabled
            else None
        )
        self._index_cache: OrderedDict[tuple[str, ...], Any] = OrderedDict()
        self._index_cache_max_scopes = max(
            1, int(getattr(settings, "recall_scope_cache_size", 4))
        )
        self._lock = threading.Lock()
        self._closed = False
        torch = self.v3.torch
        self._cuda_stream = (
            torch.cuda.Stream(device=self.models.device)
            if torch.cuda.is_available() and self.models.device.type == "cuda"
            else None
        )

    def _stream_context(self) -> Any:
        if self._cuda_stream is None:
            return nullcontext()
        return self.v3.torch.cuda.stream(self._cuda_stream)

    def _synchronize_stream(self) -> None:
        if self._cuda_stream is not None:
            self._cuda_stream.synchronize()

    @staticmethod
    def _is_cuda_out_of_memory(error: BaseException) -> bool:
        pending: list[BaseException] = [error]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            current_type = type(current)
            if (
                current_type.__name__ == "OutOfMemoryError"
                and current_type.__module__.startswith("torch")
            ):
                return True
            message = str(current).casefold()
            if "cuda out of memory" in message or "cuda error: out of memory" in message:
                return True
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None and current.__context__ is not current.__cause__:
                pending.append(current.__context__)
        return False

    def _release_cuda_recall_cache(self) -> None:
        """Drop temporary recall allocations before the pool retires this lane."""

        try:
            self._synchronize_stream()
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if self.v3.torch.cuda.is_available():
                self.v3.torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("recall engine replica is closed")

    def _build_graph_scorer(self) -> Any:
        from experiments.replacement.node_memory import LoadedNodeMemoryScorer

        return LoadedNodeMemoryScorer(
            node_model_path=self.settings.node_model,
            path_model_path=self.settings.path_model,
            device=self.settings.graph_device,
        )

    @staticmethod
    def _index_artifact_identity(path: Path) -> tuple[str, ...]:
        resolved = Path(path).resolve()
        metadata = resolved.stat()
        return (
            str(resolved),
            str(int(metadata.st_dev)),
            str(int(metadata.st_ino)),
            str(int(metadata.st_size)),
            str(int(metadata.st_mtime_ns)),
            str(int(metadata.st_ctime_ns)),
        )

    def _remember_index_bundle(
        self,
        key: tuple[str, ...],
        bundle: Any,
    ) -> Any:
        kind, scope_id = key[:2]
        for cached_key in list(self._index_cache):
            if cached_key[:2] == (kind, scope_id) and cached_key != key:
                self._index_cache.pop(cached_key, None)
        self._index_cache[key] = bundle
        self._index_cache.move_to_end(key)
        max_entries = self._index_cache_max_scopes * 2
        while len(self._index_cache) > max_entries:
            self._index_cache.popitem(last=False)
        return bundle

    def _load_base_index_cached(
        self,
        index_path: Path,
        database_path: Path,
        scope_id: str,
    ) -> Any:
        key = (
            "base",
            str(scope_id),
            *self._index_artifact_identity(index_path),
            *self._index_artifact_identity(database_path),
        )
        cached = self._index_cache.get(key)
        if cached is not None:
            self._index_cache.move_to_end(key)
            return cached
        bundle = self.runtime.load_online_index(index_path, database_path, scope_id)
        from tmcra_local_models import verify_index_identity
        verify_index_identity(bundle[-1], self.args)
        return self._remember_index_bundle(key, bundle)

    def _load_delta_index_cached(
        self,
        path: Path,
        *,
        expected_live_db: Path | None,
        expected_scope: str,
        expected_base_generation_id: str,
        expected_base_index_sha256: str,
    ) -> Any:
        if expected_live_db is None:
            raise RuntimeError("active delta recall requires an immutable database binding")
        key = (
            "delta",
            str(expected_scope),
            str(expected_base_generation_id),
            str(expected_base_index_sha256),
            *self._index_artifact_identity(path),
            *self._index_artifact_identity(expected_live_db),
        )
        cached = self._index_cache.get(key)
        if cached is not None:
            self._index_cache.move_to_end(key)
            return cached
        return self._remember_index_bundle(
            key,
            self.runtime.load_online_delta_index(
                path,
                expected_live_db=expected_live_db,
                expected_scope=expected_scope,
                expected_base_generation_id=expected_base_generation_id,
                expected_base_index_sha256=expected_base_index_sha256,
            ),
        )

    def build_delta_index(
        self,
        *,
        base_snapshot: Mapping[str, Any],
        live_database: Path,
        source_event_seq: int,
        index_path: Path,
        report_path: Path,
        previous_delta_path: Path | None,
    ) -> dict[str, Any]:
        """Encode one cumulative delta with the already resident BGE model."""

        torch = self.v3.torch
        with self._lock, torch.inference_mode(), self._stream_context():
            self._ensure_open()
            report = self.runtime.build_online_delta_index(
                base_db_path=Path(str(base_snapshot["database"])).resolve(),
                base_index_path=Path(str(base_snapshot["index"])).resolve(),
                base_generation_id=str(base_snapshot.get("generation_id") or ""),
                base_index_sha256=str(base_snapshot.get("index_sha256") or ""),
                live_db_path=Path(live_database).resolve(),
                scope_id=str(base_snapshot["scope_id"]),
                source_event_seq=int(source_event_seq),
                index_path=Path(index_path),
                report_path=Path(report_path),
                args=self.args,
                vectorizer=self.models.dense,
                previous_delta_path=(
                    None
                    if previous_delta_path is None
                    else Path(previous_delta_path).resolve()
                ),
            )
            self._synchronize_stream()
            return dict(report)

    def build_base_index(
        self,
        *,
        database_path: Path,
        scope_id: str,
        index_path: Path,
        report_path: Path,
    ) -> dict[str, Any]:
        """Build one full generation with the already resident BGE model."""

        torch = self.v3.torch
        with self._lock, torch.inference_mode(), self._stream_context():
            self._ensure_open()
            report = self.runtime.build_online_base_index(
                db_path=Path(database_path).resolve(),
                scope_id=str(scope_id),
                index_path=Path(index_path).resolve(),
                report_path=Path(report_path).resolve(),
                args=self.args,
                vectorizer=self.models.dense,
            )
            self._synchronize_stream()
            return dict(report)

    def warmup(self, snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Exercise all resident AI components without making a provider call."""
        torch = self.v3.torch
        with self._lock, torch.inference_mode(), self._stream_context():
            self._ensure_open()
            dense = self.models.dense.encode_one("TMCRA startup preflight")
            representations, logits = self.models.encode_cross(
                "TMCRA startup preflight", ["resident model readiness probe"]
            )
            if tuple(dense.shape) != (self.args.text_dim,) or not bool(torch.isfinite(dense).all()):
                raise RuntimeError("embedding startup probe returned an invalid vector")
            if representations.shape[0] != 1 or logits.shape[0] != 1:
                raise RuntimeError("cross encoder startup probe returned invalid shapes")
            if not bool(torch.isfinite(representations).all()) or not bool(
                torch.isfinite(logits).all()
            ):
                raise RuntimeError("cross encoder startup probe returned non-finite values")
            checkpoint_types: dict[str, str] = {}
            graph_scopes: list[str] = []
            if self.settings.learned_graph_enabled:
                assert self.graph_adapter_cache is not None
                graph_scorer = self.graph_adapter_cache.ensure_scorer()
                scorer_model = getattr(graph_scorer, "model", None)
                if scorer_model is None or bool(getattr(scorer_model, "training", True)):
                    raise RuntimeError("graph scorer startup probe is not in eval mode")
                for name, path in (
                    ("node_model", self.settings.node_model),
                    ("path_model", self.settings.path_model),
                ):
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    if not isinstance(checkpoint, Mapping) or not checkpoint:
                        raise RuntimeError(f"{name} checkpoint is empty or invalid")
                    checkpoint_types[name] = str(
                        checkpoint.get("schema_version") or "mapping"
                    )
            if snapshots and self.settings.learned_graph_enabled:
                assert self.graph_adapter_cache is not None
                recent_snapshots = sorted(
                    snapshots,
                    key=lambda item: float(item.get("activated_at") or 0.0),
                    reverse=True,
                )[: self.graph_adapter_cache.max_entries]
                self.graph_adapter_cache.clear()
                for snapshot in reversed(recent_snapshots):
                    scope_id = str(snapshot.get("scope_id") or "")
                    database = Path(str(snapshot.get("database") or "")).resolve()
                    adapter = self.harness.build_adapter(scope_id, database)
                    graph_fingerprint = self.v3.scope_fingerprint(database, scope_id)
                    self.graph_adapter_cache[
                        (scope_id, str(database), graph_fingerprint)
                    ] = adapter
                    graph_scopes.append(scope_id)
            self._synchronize_stream()
            return {
                "dense_shape": list(dense.shape),
                "cross_representation_shape": list(representations.shape),
                "cross_logit_shape": list(logits.shape),
                "checkpoint_types": checkpoint_types,
                "learned_graph_enabled": self.settings.learned_graph_enabled,
                "retrieval_mode": (
                    "hybrid_node_scored"
                    if self.settings.learned_graph_enabled
                    else "dense_fast"
                ),
                "graph_scorer_preloaded": bool(
                    self.graph_adapter_cache is not None
                    and self.graph_adapter_cache.scorer_loaded
                ),
                "graph_adapter_preloaded": bool(graph_scopes),
                "graph_scope_count": len(graph_scopes),
                "graph_scopes": graph_scopes,
            }

    def recall(
        self,
        *,
        provider_tenant_id: str | None = None,
        snapshot: Mapping[str, Any],
        query_id: str,
        query: str,
        query_time: str,
        max_windows: int,
        recall_profile: str = "quality",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        args = copy.copy(self.args)
        args.top_k = max_windows
        row = {
            "question_id": query_id,
            "question": query,
            "question_date": query_time,
            "scope_id": snapshot["scope_id"],
            "db_path": snapshot["database"],
            "index_path": snapshot["index"],
        }
        delta = snapshot.get("delta")
        if isinstance(delta, Mapping):
            row.update(
                {
                    "delta_index_path": delta["index"],
                    "live_db_path": delta["database"],
                    "base_generation_id": snapshot.get("generation_id"),
                    "base_index_sha256": snapshot.get("index_sha256"),
                }
            )
        torch = self.v3.torch
        try:
            with self._lock, torch.inference_mode(), self._stream_context():
                self._ensure_open()
                provider_user_id = ""
                if provider_tenant_id:
                    provider_user_id = "tmcra_" + hashlib.sha256(
                        (
                            str(provider_tenant_id)
                            + "\0"
                            + str(snapshot.get("scope_id") or "")
                        ).encode("utf-8")
                    ).hexdigest()[:32]
                self.planner.set_provider_user_id(provider_user_id)
                route_override = None
                route_override_metadata = None
                if recall_profile == "interactive":
                    original_query_length = len(str(query or "").strip())
                    route_override = interactive_recall_plan(query)
                    route_override_metadata = {
                        "physical_api_call": False,
                        "physical_api_calls": 0,
                        "stage": "recall_planner",
                        "status": "interactive_neutral_plan",
                        "planner_version": "tmcra-interactive-neutral-v1",
                        "prompt_version": "none",
                        "query_bounded": original_query_length > len(
                            route_override["resolved_query"]
                        ),
                        "original_query_length": original_query_length,
                        "resolved_query_length": len(route_override["resolved_query"]),
                    }
                elif recall_profile != "quality":
                    raise ValueError(f"unsupported recall profile: {recall_profile}")
                result = self.runtime.retrieve_one(
                    row,
                    args=args,
                    harness=self.harness,
                    models=self.models,
                    planner=self.planner,
                    route_override=route_override,
                    route_override_metadata=route_override_metadata,
                    graph_adapter_cache=self.graph_adapter_cache,
                    base_index_loader=self._load_base_index_cached,
                    delta_index_loader=self._load_delta_index_cached,
                )
                self._synchronize_stream()
                return result
        except BaseException as exc:
            if self._is_cuda_out_of_memory(exc):
                self._release_cuda_recall_cache()
            raise

    def close(self) -> None:
        """Release one idle replica and return its cached CUDA memory."""
        torch = self.v3.torch
        with self._lock:
            if self._closed:
                return
            self._synchronize_stream()
            if self.graph_adapter_cache is not None:
                self.graph_adapter_cache.release()
            self._index_cache.clear()
            self.models = None  # type: ignore[assignment]
            self.planner = None  # type: ignore[assignment]
            self.harness = None  # type: ignore[assignment]
            self._cuda_stream = None
            self._closed = True
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
