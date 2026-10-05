from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import math
import os
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from . import __version__
from .adapters.v4 import V4StorageAdapter
from .control_db import ControlDB
from .health import readiness
from .provider_pool import ProviderCircuitBreaker
from .planner_provider import recall_planner_route
from .runtime import LazyOnlineEngine, WorkerStatus
from .settings import (
    RELEASE_CHANNEL_RE,
    RELEASE_IDENTIFIER_RE,
    SHA256_RE,
    ServiceSettings,
)
from .writer_provider import (
    DESKTOP_LOCAL_QWEN_MODEL,
    DESKTOP_LOCAL_QWEN_MIN_CONTEXT_TOKENS,
    LOCAL_QWEN_MIN_CONTEXT_TOKENS,
    LOCAL_QWEN_PROVIDER,
    primary_writer_route,
)
from .writer_context import writer_unresolved_limits_from_env


class StartupPreflightError(RuntimeError):
    pass


class WriteAdmissionRejected(RuntimeError):
    def __init__(self, reason: str, retry_after_seconds: float) -> None:
        super().__init__(reason.replace("_", " "))
        self.reason = reason
        self.retry_after_seconds = max(0.001, float(retry_after_seconds))


@dataclass(frozen=True)
class WriteAdmissionSnapshot:
    accepting_writes: bool
    reason: str | None
    retry_after_seconds: float
    service_worker_ready: bool
    writer_mode: str
    writer_configured: int
    writer_ready: int
    provider: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepting_writes": self.accepting_writes,
            "reason": self.reason,
            "retry_after_seconds": round(self.retry_after_seconds, 3),
            "service_worker_ready": self.service_worker_ready,
            "writer": {
                "mode": self.writer_mode,
                "configured": self.writer_configured,
                "ready": self.writer_ready,
            },
            "provider": dict(self.provider),
        }


class MemoryWriteAdmission:
    """Live fail-closed gate for jobs whose first stage invokes paid Writer work."""

    def __init__(
        self,
        *,
        settings: ServiceSettings,
        storage: Any,
        worker: Any,
        provider: ProviderCircuitBreaker,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.worker = worker
        self.provider = provider

    def snapshot(
        self,
        *,
        connection: sqlite3.Connection | None = None,
        provider_required: bool = True,
    ) -> WriteAdmissionSnapshot:
        retry_default = self.settings.write_admission_retry_seconds
        if self.settings.startup_preflight_mode == "off":
            return WriteAdmissionSnapshot(
                accepting_writes=True,
                reason=None,
                retry_after_seconds=0.0,
                service_worker_ready=True,
                writer_mode="test",
                writer_configured=1,
                writer_ready=1,
                provider={
                    "pool": "test",
                    "accepting_paid_work": True,
                    "reason": None,
                    "retry_after_seconds": 0.0,
                    "enabled_keys": 1,
                    "healthy_keys": 1,
                },
            )
        try:
            worker_status = self.worker.status()
            worker_ready = bool(getattr(worker_status, "alive", False))
        except Exception:
            worker_ready = False
        try:
            writer = dict(self.storage.writer_status() or {})
        except Exception:
            writer = {}
        writer_mode = str(writer.get("mode") or "unknown")
        writer_configured = int(writer.get("configured", 0) or 0)
        writer_ready = int(writer.get("ready", 0) or 0)
        writer_alive = bool(writer.get("alive"))
        if writer_mode == "resident":
            writer_accepting = (
                writer_alive
                and writer_configured > 0
                and writer_ready == writer_configured
            )
        else:
            writer_accepting = writer_alive

        try:
            provider = self.provider.status(connection=connection)
            provider_view = provider.as_dict()
        except Exception:
            provider = None
            provider_view = {
                "pool": self.provider.pool,
                "accepting_paid_work": False,
                "reason": "provider_admission_unavailable",
                "retry_after_seconds": round(retry_default, 3),
                "circuit_kind": None,
                "circuit_open_until": None,
                "enabled_keys": 0,
                "healthy_keys": 0,
            }

        reason: str | None = None
        retry_after = 0.0
        if not worker_ready:
            reason = "service_worker_unavailable"
            retry_after = retry_default
        elif not writer_accepting:
            reason = (
                "writer_pool_starting"
                if writer_mode == "resident" and writer_ready > 0
                else "writer_pool_unavailable"
            )
            retry_after = retry_default
        elif provider_required and (provider is None or not provider.accepting_paid_work):
            reason = str(provider_view.get("reason") or "provider_unavailable")
            retry_after = float(
                provider_view.get("retry_after_seconds") or retry_default
            )
        return WriteAdmissionSnapshot(
            accepting_writes=reason is None,
            reason=reason,
            retry_after_seconds=retry_after,
            service_worker_ready=worker_ready,
            writer_mode=writer_mode,
            writer_configured=writer_configured,
            writer_ready=writer_ready,
            provider=provider_view,
        )

    def require(
        self,
        *,
        connection: sqlite3.Connection | None = None,
        provider_required: bool = True,
    ) -> None:
        snapshot = self.snapshot(
            connection=connection,
            provider_required=provider_required,
        )
        if not snapshot.accepting_writes:
            raise WriteAdmissionRejected(
                snapshot.reason or "write_admission_closed",
                snapshot.retry_after_seconds,
            )


CRITICAL_CONTROL_TABLES = frozenset(
    {
        "api_keys",
        "scope_catalog",
        "scope_sessions",
        "scope_ingest_events",
        "jobs",
        "operation_stages",
        "scope_heads",
        "scope_evolution_state",
        "scope_ingest_watermark_commits",
        "scope_source_event_commits",
        "scope_ingest_source_sets",
        "provider_calls",
        "user_provider_tasks",
        "provider_call_reconciliations",
        "provider_circuits",
        "provider_prices",
        "graph_runtime_audits",
    }
)

BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
SENSITIVE_SETTING_SUFFIXES = (
    "_credential",
    "_key",
    "_password",
    "_secret",
    "_token",
)
STAFF_VISIBLE_CHECKS = frozenset(
    {
        "settings",
        "network",
        "paths",
        "state_io",
        "control_db",
        "disk",
        "provider_pool",
        "adapter_compatibility",
        "writer_pool",
        "active_indexes",
        "gpu",
        "ai_runtime",
        "service_worker",
        "report_persistence",
    }
)
MAX_PERSISTED_PREFLIGHT_BYTES = 1024 * 1024


def _environment_value(*names: str) -> str | None:
    for name in names:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    return None


def _safe_url(value: str) -> dict[str, Any]:
    parsed = urlparse(value)
    try:
        port = parsed.port
    except ValueError:
        port = None
    return {
        "scheme": parsed.scheme.lower(),
        "host": (parsed.hostname or "").lower(),
        "port": port,
        "path": parsed.path or "/",
    }


def _configuration_fingerprint(settings: ServiceSettings) -> str:
    configuration: dict[str, Any] = {}
    for field in fields(settings):
        name = field.name
        value = getattr(settings, name)
        if name.endswith(SENSITIVE_SETTING_SUFFIXES):
            configuration[f"{name}_configured"] = bool(value)
        elif name == "public_base_url":
            configuration[name] = _safe_url(str(value))
        elif isinstance(value, Path):
            configuration[name] = str(value)
        else:
            configuration[name] = value

    raw_keys = str(os.getenv("TMCRA_WRITER_API_KEY_POOL") or "")
    key_parts = raw_keys.split(",") if raw_keys else []
    keys = [value.strip() for value in key_parts]
    configuration["startup_environment"] = {
        "tls_proxy_mode": str(
            os.getenv("TMCRA_SERVICE_TLS_PROXY_MODE") or ""
        ).strip().lower(),
        "writer_provider": {
            "base_url": _safe_url(str(os.getenv("TMCRA_WRITER_BASE_URL") or "")),
            "key_count": len(keys),
            "unique_key_count": len(set(keys)),
            "has_empty_key": any(not value for value in keys),
            "max_tokens": str(os.getenv("TMCRA_WRITER_MAX_TOKENS") or ""),
            "model": str(os.getenv("TMCRA_WRITER_MODEL") or "").strip(),
        },
    }
    payload = json.dumps(
        configuration,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _boot_id() -> str | None:
    try:
        value = BOOT_ID_PATH.read_text(encoding="ascii").strip()
    except OSError:
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return None


def service_release_metadata(settings: ServiceSettings) -> dict[str, Any]:
    release_sha256 = settings.service_release_sha256 or _environment_value(
        "TMCRA_SERVICE_RELEASE_SHA256",
        "TMCRA_RELEASE_SHA256",
        "TMCRA_ARCHIVE_SHA256",
    )
    if release_sha256 is not None and not SHA256_RE.fullmatch(release_sha256):
        release_sha256 = None
    release_id = settings.service_release_id or _environment_value(
        "TMCRA_SERVICE_RELEASE_ID", "TMCRA_RELEASE"
    )
    if release_id is not None and not RELEASE_IDENTIFIER_RE.fullmatch(release_id):
        release_id = None
    release_channel = settings.service_release_channel or _environment_value(
        "TMCRA_SERVICE_RELEASE_CHANNEL"
    )
    if (
        release_channel is not None
        and not RELEASE_CHANNEL_RE.fullmatch(release_channel)
    ):
        release_channel = None
    rollback_release_id = (
        settings.service_rollback_release_id
        or _environment_value("TMCRA_SERVICE_ROLLBACK_RELEASE_ID")
    )
    if (
        rollback_release_id is not None
        and not RELEASE_IDENTIFIER_RE.fullmatch(rollback_release_id)
    ):
        rollback_release_id = None
    raw_canary = _environment_value("TMCRA_SERVICE_CANARY_PERCENT")
    canary_percent = settings.service_canary_percent
    if canary_percent is None and raw_canary is not None:
        try:
            parsed_canary = float(raw_canary)
        except ValueError:
            parsed_canary = None
        if parsed_canary is not None and 0.0 <= parsed_canary <= 100.0:
            canary_percent = parsed_canary
    return {
        "service_version": __version__,
        "release_id": release_id,
        "release_sha256": release_sha256.lower() if release_sha256 else None,
        "release_channel": release_channel,
        "canary_percent": canary_percent,
        "rollback_release_id": rollback_release_id,
    }


def _report_metadata(settings: ServiceSettings) -> dict[str, Any]:
    return {
        **service_release_metadata(settings),
        "process_id": os.getpid(),
        "boot_id": _boot_id(),
        "configuration_fingerprint": _configuration_fingerprint(settings),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(value), handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class StartupPreflight:
    """One-time hard startup gate plus cheap cached readiness state."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.path = settings.state_dir / "startup_preflight.json"
        self._lock = threading.Lock()
        self._report: dict[str, Any] = {
            "schema_version": "tmcra.service.startup-preflight.1",
            **_report_metadata(settings),
            "mode": settings.startup_preflight_mode,
            "status": "not_run",
            "checks": {},
        }

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._report.get("status") == "passed"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._report)

    def staff_snapshot(self) -> dict[str, Any]:
        """Read and redact the persisted startup report for staff telemetry."""

        try:
            with self.path.open("rb") as handle:
                raw = handle.read(MAX_PERSISTED_PREFLIGHT_BYTES + 1)
            if len(raw) > MAX_PERSISTED_PREFLIGHT_BYTES:
                raise ValueError("report is too large")
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, Mapping):
                raise ValueError("report is not an object")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return {
                "availability": "unavailable",
                "reason": "persisted_startup_preflight_unavailable",
                "source": "startup_preflight.persisted_report",
            }

        status = str(value.get("status") or "")
        mode = str(value.get("mode") or "")
        schema_version = str(value.get("schema_version") or "")
        if (
            schema_version != "tmcra.service.startup-preflight.1"
            or status not in {"passed", "failed"}
            or mode not in {"basic", "full"}
        ):
            return {
                "availability": "unavailable",
                "reason": "persisted_startup_preflight_incomplete",
                "source": "startup_preflight.persisted_report",
            }

        checks: dict[str, dict[str, Any]] = {}
        raw_checks = value.get("checks")
        if isinstance(raw_checks, Mapping):
            for name in sorted(STAFF_VISIBLE_CHECKS):
                raw_check = raw_checks.get(name)
                if not isinstance(raw_check, Mapping):
                    continue
                check: dict[str, Any] = {"ok": bool(raw_check.get("ok"))}
                duration = raw_check.get("duration_seconds")
                if isinstance(duration, (int, float)) and not isinstance(
                    duration, bool
                ) and 0 <= float(duration) < 86_400:
                    check["duration_seconds"] = round(float(duration), 6)
                if not check["ok"]:
                    check["failure_category"] = "check_failed"
                checks[name] = check

        failed_checks = [name for name, check in checks.items() if not check["ok"]]
        result: dict[str, Any] = {
            "availability": "available",
            "source": "startup_preflight.persisted_report",
            "schema_version": "tmcra.service.startup-preflight.1",
            "mode": mode,
            "status": status,
            "hard_gate": bool(value.get("hard_gate")),
            "checks": checks,
            "failed_checks": failed_checks,
        }
        for name in ("started_at", "completed_at", "duration_seconds"):
            metric = value.get(name)
            if (
                isinstance(metric, (int, float))
                and not isinstance(metric, bool)
                and math.isfinite(float(metric))
                and float(metric) >= 0
            ):
                result[name] = float(metric)
        return result

    def _set_report(self, report: Mapping[str, Any]) -> None:
        with self._lock:
            self._report = copy.deepcopy(dict(report))

    @staticmethod
    def _check(
        checks: dict[str, Any], name: str, function: Callable[[], Mapping[str, Any] | None]
    ) -> Any:
        started = time.monotonic()
        try:
            details = dict(function() or {})
            checks[name] = {
                "ok": True,
                "duration_seconds": round(time.monotonic() - started, 6),
                **details,
            }
            return details.get("value")
        except Exception as exc:
            checks[name] = {
                "ok": False,
                "duration_seconds": round(time.monotonic() - started, 6),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            return None

    def _path_check(self) -> Mapping[str, Any]:
        file_names = {
            "audio_asr_api_key_file",
            "writer_env",
            "native_harness",
            "node_model",
            "path_model",
            "checkpoint",
        }
        checked: dict[str, str] = {}
        for name, path in self.settings.required_paths().items():
            valid = path.is_file() if name in file_names else path.is_dir()
            if not valid:
                expected = "file" if name in file_names else "directory"
                raise RuntimeError(f"{name} is not a readable {expected}: {path}")
            if not os.access(path, os.R_OK):
                raise RuntimeError(f"{name} is not readable: {path}")
            if path.is_file() and path.stat().st_size <= 0:
                raise RuntimeError(f"{name} is empty: {path}")
            checked[name] = "file" if path.is_file() else "directory"
        return {"path_types": checked}

    def _state_io_check(self) -> Mapping[str, Any]:
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        source = self.settings.state_dir / f".startup-write-{token}.tmp"
        target = self.settings.state_dir / f".startup-write-{token}.commit"
        try:
            with source.open("x", encoding="ascii") as handle:
                handle.write(token)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(source, target)
            if target.read_text(encoding="ascii") != token:
                raise RuntimeError("state directory atomic write probe changed content")
        finally:
            source.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
        return {"atomic_replace": True, "fsync": True}

    def _database_check(self) -> Mapping[str, Any]:
        with closing(sqlite3.connect(self.settings.control_db, timeout=10.0)) as connection:
            quick = connection.execute("PRAGMA quick_check").fetchone()
            if not quick or quick[0] != "ok":
                raise RuntimeError(f"control DB quick_check returned {quick!r}")
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            missing = sorted(CRITICAL_CONTROL_TABLES - tables)
            if missing:
                raise RuntimeError(
                    "control DB lacks critical tables: " + ",".join(missing)
                )
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("SELECT 1").fetchone()
            connection.rollback()
        return {
            "quick_check": "ok",
            "critical_table_count": len(CRITICAL_CONTROL_TABLES),
            "write_lock_probe": True,
        }

    def _provider_check(self) -> Mapping[str, Any]:
        try:
            route = primary_writer_route(os.environ)
            planner_route = recall_planner_route(os.environ)
        except ValueError as exc:
            raise RuntimeError(f"provider route is invalid: {exc}") from exc
        try:
            max_tokens = int(os.getenv("TMCRA_WRITER_MAX_TOKENS", "16384"))
        except ValueError as exc:
            raise RuntimeError("Writer max token setting is not an integer") from exc
        if max_tokens != 16384:
            raise RuntimeError("Writer max token setting must be 16384")
        try:
            unresolved_max_items, unresolved_max_chars = (
                writer_unresolved_limits_from_env()
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        details: dict[str, Any] = {
            "provider": route.provider,
            "pool_name": route.pool_name,
            "paid": route.paid,
            "key_count": len(route.api_keys),
            "writer_model": route.model,
            "prompt_adapter": route.prompt_adapter,
            "writer_max_tokens": max_tokens,
            "writer_unresolved_max_items": unresolved_max_items,
            "writer_unresolved_max_chars": unresolved_max_chars,
            "base_url_host": urlparse(route.base_url).netloc,
            "paid_probe": False,
            "recall_planner": {
                "provider": planner_route.provider,
                "model": planner_route.model,
                "paid": planner_route.paid,
                "prompt_adapter": planner_route.prompt_adapter,
                "base_url_host": urlparse(planner_route.base_url).netloc,
            },
        }
        if route.provider == LOCAL_QWEN_PROVIDER:
            details["local_model_context"] = self._local_model_context_check(route)
        return details

    @staticmethod
    def _local_model_context_check(route: Any) -> Mapping[str, Any]:
        endpoint = route.base_url.removesuffix("/v1") + "/props"
        request = Request(
            endpoint,
            headers={"Authorization": f"Bearer {route.api_keys[0]}"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=5.0) as response:
                payload = json.load(response)
        except Exception as exc:
            raise RuntimeError("local Qwen context probe failed") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("local Qwen context probe returned a non-object")
        generation = payload.get("default_generation_settings")
        nested = generation if isinstance(generation, Mapping) else {}
        raw_context = payload.get("n_ctx", nested.get("n_ctx"))
        if isinstance(raw_context, bool):
            raise RuntimeError("local Qwen context probe returned an invalid n_ctx")
        try:
            context_tokens = int(raw_context)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("local Qwen context probe omitted n_ctx") from exc
        required_context = (DESKTOP_LOCAL_QWEN_MIN_CONTEXT_TOKENS
                            if os.getenv("TMCRA_DEPLOYMENT_MODE") == "local"
                            and route.model == DESKTOP_LOCAL_QWEN_MODEL
                            else LOCAL_QWEN_MIN_CONTEXT_TOKENS)
        if context_tokens < required_context:
            raise RuntimeError(
                "local Qwen context is below the Writer contract: "
                f"{context_tokens} < {required_context}"
            )
        return {
            "n_ctx": context_tokens,
            "required_n_ctx": required_context,
            "paid_probe": False,
        }

    def _network_check(self) -> Mapping[str, Any]:
        address = ipaddress.ip_address(self.settings.bind_host)
        proxy_mode = os.getenv("TMCRA_SERVICE_TLS_PROXY_MODE", "").strip().lower()
        if not address.is_loopback and not (
            address.is_unspecified and proxy_mode in {"trusted_proxy", "gpuhome"}
        ):
            raise RuntimeError("public bind is not protected by the configured TLS proxy")
        public = urlparse(self.settings.public_base_url)
        from tmcra_local_only import enabled, loopback_url
        if enabled():
            if not address.is_loopback:
                raise RuntimeError("full-local service must bind loopback")
            loopback_url(self.settings.public_base_url, port=self.settings.bind_port, path="")
        elif public.scheme != "https" or not public.netloc:
            raise RuntimeError("public base URL must be HTTPS")
        return {
            "bind_mode": "loopback" if address.is_loopback else "tls_proxy",
            "public_scheme": public.scheme,
        }

    def _gpu_check(self) -> Mapping[str, Any]:
        import torch

        devices: dict[str, Any] = {}
        for configured in dict.fromkeys(
            [self.settings.device, self.settings.graph_device]
        ):
            device = torch.device(configured)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(f"CUDA is unavailable for configured device {configured}")
            value = torch.ones(16, device=device)
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"device tensor probe failed on {configured}")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                index = device.index if device.index is not None else torch.cuda.current_device()
                devices[configured] = {
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            else:
                devices[configured] = {"name": "cpu"}
        return {"devices": devices}

    def run(
        self, storage: V4StorageAdapter, online: LazyOnlineEngine
    ) -> dict[str, Any]:
        started_at = time.time()
        if self.settings.startup_preflight_mode == "off":
            completed_at = time.time()
            report = {
                "schema_version": "tmcra.service.startup-preflight.1",
                **_report_metadata(self.settings),
                "mode": "off",
                "status": "passed",
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": round(completed_at - started_at, 6),
                "checks": {"startup_preflight": {"ok": True, "skipped": True}},
                "hard_gate": False,
            }
            self.settings.state_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(self.path, report)
            self._set_report(report)
            return report
        if self.settings.startup_preflight_mode == "basic":
            ready, shallow = readiness(self.settings)
            checks = dict(shallow.get("checks") or {})
            try:
                snapshots = storage.audit_active_indexes()
                database = ControlDB(self.settings.control_db)
                states = database.list_scope_evolution_states()
                watermark_audit = storage.audit_searchable_watermarks(
                    states,
                    require_fresh=False,
                )
                checks["active_indexes"] = {
                    "ok": True,
                    "active_index_count": len(snapshots),
                    "quarantined_scope_count": database.count_quarantined_scopes(),
                    **dict(watermark_audit),
                }
            except Exception as exc:
                ready = False
                checks["active_indexes"] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            completed_at = time.time()
            report = {
                "schema_version": "tmcra.service.startup-preflight.1",
                **_report_metadata(self.settings),
                "mode": "basic",
                "status": "passed" if ready else "failed",
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": round(completed_at - started_at, 6),
                "checks": checks,
                "hard_gate": True,
            }
            self.settings.state_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(self.path, report)
            self._set_report(report)
            if not ready:
                failed = [
                    name
                    for name, value in report["checks"].items()
                    if isinstance(value, Mapping) and not value.get("ok")
                ]
                raise StartupPreflightError(
                    "basic startup preflight failed: " + ",".join(failed)
                )
            return report

        checks: dict[str, Any] = {}
        snapshots: list[dict[str, Any]] = []
        self._check(checks, "settings", lambda: (self.settings.validate() or {}))
        self._check(checks, "network", self._network_check)
        self._check(checks, "paths", self._path_check)
        self._check(checks, "state_io", self._state_io_check)
        self._check(checks, "control_db", self._database_check)
        self._check(
            checks,
            "disk",
            lambda: self._disk_details(),
        )
        self._check(checks, "provider_pool", self._provider_check)
        self._check(
            checks,
            "adapter_compatibility",
            lambda: self._adapter_details(storage),
        )
        self._check(checks, "writer_pool", lambda: self._writer_details(storage))

        def indexes() -> Mapping[str, Any]:
            nonlocal snapshots
            snapshots = storage.audit_active_indexes()
            database = ControlDB(self.settings.control_db)
            states = database.list_scope_evolution_states()
            watermark_audit = storage.audit_searchable_watermarks(
                states,
                require_fresh=False,
            )
            return {
                "active_index_count": len(snapshots),
                "quarantined_scope_count": database.count_quarantined_scopes(),
                **dict(watermark_audit),
            }

        self._check(checks, "active_indexes", indexes)
        self._check(checks, "gpu", self._gpu_check)

        def ai_runtime() -> Mapping[str, Any]:
            dispatcher = online.get()
            warmup = dispatcher.warmup(snapshots)
            status_method = getattr(online, "status", None)
            if status_method is not None and callable(status_method):
                candidate = status_method()
                pool_status: Mapping[str, Any] = (
                    candidate
                    if isinstance(candidate, Mapping)
                    else {"loaded": bool(getattr(online, "loaded", True))}
                )
            else:
                # Compatibility for the small preflight fakes used by
                # downstream deployments. A successful warmup is the gate.
                pool_status = {
                    "loaded": bool(getattr(online, "loaded", True)),
                }
            return {
                "online_engine_loaded": bool(
                    pool_status.get("loaded", getattr(online, "loaded", True))
                ),
                "warmup": warmup,
                "recall_pool": dict(pool_status),
                "paid_probe": False,
            }

        self._check(checks, "ai_runtime", ai_runtime)
        passed = all(bool(value.get("ok")) for value in checks.values())
        report = {
            "schema_version": "tmcra.service.startup-preflight.1",
            **_report_metadata(self.settings),
            "mode": "full",
            "status": "passed" if passed else "failed",
            "started_at": started_at,
            "completed_at": time.time(),
            "duration_seconds": round(time.time() - started_at, 6),
            "checks": checks,
            "hard_gate": True,
        }
        try:
            _atomic_json(self.path, report)
        except Exception as exc:
            report["status"] = "failed"
            report["checks"]["report_persistence"] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            passed = False
        self._set_report(report)
        if not passed:
            failed = [name for name, value in checks.items() if not value.get("ok")]
            raise StartupPreflightError(
                "AI startup preflight failed: " + ",".join(failed)
            )
        return report

    def _disk_details(self) -> Mapping[str, Any]:
        usage = shutil.disk_usage(self.settings.state_dir)
        if usage.free < self.settings.disk_free_min_bytes:
            raise RuntimeError(
                f"free disk {usage.free} is below required {self.settings.disk_free_min_bytes}"
            )
        return {
            "free_bytes": usage.free,
            "required_free_bytes": self.settings.disk_free_min_bytes,
        }

    @staticmethod
    def _adapter_details(storage: V4StorageAdapter) -> Mapping[str, Any]:
        compatibility = storage.compatibility()
        failed = [name for name, value in compatibility.items() if not value]
        if failed:
            raise RuntimeError("adapter compatibility failed: " + ",".join(failed))
        return {"contracts": compatibility}

    @staticmethod
    def _writer_details(storage: V4StorageAdapter) -> Mapping[str, Any]:
        storage.start()
        status = storage.writer_status()
        if not status.get("alive"):
            raise RuntimeError("resident Writer pool did not become ready")
        return status

    def record_runtime(self, worker: WorkerStatus) -> None:
        report = self.snapshot()
        checks = dict(report.get("checks") or {})
        checks["service_worker"] = {
            "ok": worker.alive,
            "worker_id": worker.worker_id,
        }
        report["checks"] = checks
        if not worker.alive:
            report["status"] = "failed"
        report["completed_at"] = time.time()
        _atomic_json(self.path, report)
        self._set_report(report)
        if self.settings.startup_preflight_mode == "full" and not worker.alive:
            raise StartupPreflightError("service worker failed to start")
