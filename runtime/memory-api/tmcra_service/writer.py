from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping, Sequence

from .actor_provenance import (
    ActorProvenanceError,
    actor_metadata_json,
    actor_metadata_sha256,
    normalize_message_actor_metadata,
)
from .provider_pool import ProviderKeyPool, ProviderPoolExhausted
from .control_db import ControlDB
from .jobs import JobStore
from .usage_attribution import UNATTRIBUTED, UsageAttribution
from .user_provider_client import (
    USER_PROVIDER,
    UserProviderBrokerClient,
    normalize_user_provider_execution,
)
from .qwen36_writer_adapter import (
    ADAPTER_ID as QWEN36_ADAPTER_ID,
    REVIEWER_ADAPTER_ID as QWEN36_REVIEWER_ADAPTER_ID,
    create_qwen36_batch_client,
    prompt_sha256 as qwen36_prompt_sha256,
)
from .writer_provider import (
    DEEPSEEK_PROVIDER,
    LOCAL_QWEN_MODEL,
    LOCAL_QWEN_PROVIDER,
    OPENAI_COMPATIBLE_PROVIDER,
    primary_writer_route,
    reviewer_writer_route,
)
from .writer_context import (
    select_unresolved_interactions,
    writer_unresolved_limits_from_env,
)


class ProductionWriterError(RuntimeError):
    pass


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _raw_token_estimate(content: str) -> int:
    """Deterministic local estimate used only for newly persisted messages."""
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
    other = len(non_empty) - cjk
    return cjk + (other + 3) // 4


DEEPSEEK_V4_PRICE_VERSION = "deepseek-v4-official-cny-2026-07-15"
DEEPSEEK_V4_PRICES_MICRO_CNY = {
    "deepseek-v4-flash": (20_000, 1_000_000, 2_000_000),
    "deepseek-v4-pro": (25_000, 3_000_000, 6_000_000),
}
DEEPSEEK_PRICING_SOURCE = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
LOCAL_QWEN_PRICE_VERSION = "tmcra-local-qwen36-iq3s-2026-08-05"
LOCAL_QWEN_PRICING_SOURCE = "self-hosted; external provider API cost is zero"
OPERATOR_PRICE_VERSION = "operator-configured-v1"
UNPRICED_MODEL_VERSION = "operator-pricing-not-configured"
WRITER_RECOVERY_MODES = frozenset(
    {
        "none",
        "validation",
        "definitive_provider_failure",
        "definitive_invalid_response",
        "schema_constrained_invalid_response",
        "schema_constrained_invalid_response_prepared",
        "audited_writer_state",
        "audited_local_inference_cancelled",
    }
)


def local_writer_recovery_concurrency_from_env() -> int:
    """Return the bounded local-model capacity reserved for recovery work."""

    raw = str(os.getenv("TMCRA_LOCAL_WRITER_RECOVERY_CONCURRENCY", "1")).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ProductionWriterError(
            "TMCRA_LOCAL_WRITER_RECOVERY_CONCURRENCY must be an integer"
        ) from exc
    if value <= 0 or value > 4:
        raise ProductionWriterError(
            "TMCRA_LOCAL_WRITER_RECOVERY_CONCURRENCY must be between 1 and 4"
        )
    return value


class IdentityRegistry:
    """Assign stable incremental identities before the benchmark writer core runs."""

    def __init__(
        self,
        database: Path,
        operation_id: str,
        *,
        expected_scope_id: str | None = None,
    ) -> None:
        self.database = database.resolve()
        self.operation_id = operation_id
        self.expected_scope_id = (
            None if expected_scope_id is None else str(expected_scope_id).strip()
        )
        if expected_scope_id is not None and not self.expected_scope_id:
            raise ProductionWriterError("expected_scope_id cannot be empty")
        self.new_message_count = 0
        self.replayed_message_count = 0
        self.new_user_turn_count = 0
        self.new_raw_token_estimate = 0
        self.registered_messages: dict[tuple[str, str], Any] = {}
        self.source_origin_operations: dict[tuple[str, str], str] = {}
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tmcra_service_sessions (
                    scope_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    session_index INTEGER NOT NULL,
                    PRIMARY KEY(scope_id, session_id),
                    UNIQUE(scope_id, session_index)
                );
                CREATE TABLE IF NOT EXISTS tmcra_service_messages (
                    scope_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    internal_message_id TEXT NOT NULL DEFAULT '',
                    session_id TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    first_operation_id TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(scope_id, message_id),
                    UNIQUE(scope_id, session_id, message_index),
                    FOREIGN KEY(scope_id, session_id)
                        REFERENCES tmcra_service_sessions(scope_id, session_id)
                );
                CREATE TABLE IF NOT EXISTS tmcra_service_batches (
                    scope_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    local_batch_index INTEGER NOT NULL,
                    batch_index INTEGER NOT NULL,
                    PRIMARY KEY(scope_id, session_id, operation_id, local_batch_index),
                    UNIQUE(scope_id, session_id, batch_index)
                );
                CREATE TABLE IF NOT EXISTS tmcra_service_message_actor_provenance (
                    scope_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    actor_metadata_json TEXT NOT NULL,
                    actor_metadata_sha256 TEXT NOT NULL,
                    PRIMARY KEY(scope_id, message_id),
                    FOREIGN KEY(scope_id, message_id)
                        REFERENCES tmcra_service_messages(scope_id, message_id)
                        ON DELETE CASCADE
                );
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(tmcra_service_messages)"
                )
            }
            if "internal_message_id" not in columns:
                connection.execute(
                    "ALTER TABLE tmcra_service_messages "
                    "ADD COLUMN internal_message_id TEXT NOT NULL DEFAULT ''"
                )
            if "first_operation_id" not in columns:
                connection.execute(
                    "ALTER TABLE tmcra_service_messages "
                    "ADD COLUMN first_operation_id TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                """
                UPDATE tmcra_service_messages
                SET internal_message_id =
                    's' || printf('%03d', (
                        SELECT session_index FROM tmcra_service_sessions AS s
                        WHERE s.scope_id=tmcra_service_messages.scope_id
                          AND s.session_id=tmcra_service_messages.session_id
                    )) || '_m' || printf('%03d', message_index)
                WHERE internal_message_id=''
                """
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "tmcra_service_messages_internal_id "
                "ON tmcra_service_messages(scope_id, internal_message_id)"
            )

    @staticmethod
    def _internal_message_id(session_index: int, message_index: int) -> str:
        return f"s{session_index:03d}_m{message_index:03d}"

    @staticmethod
    def _require_enriched_replay(
        connection: sqlite3.Connection,
        message: Any,
    ) -> None:
        """Verify a replay is already durable before excluding it from Writer input."""

        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not {"v4_source_journal", "records"}.issubset(tables):
            raise ProductionWriterError(
                f"replayed message Source proof is missing: {message.message_id}"
            )
        source = connection.execute(
            "SELECT scope_id,session_id,message_id,session_index,message_index,"
            "message_role,timestamp,content,content_sha256,status,source_record_id,"
            "source_turn_index,source_persisted_at FROM v4_source_journal "
            "WHERE scope_id=? AND message_id=?",
            (message.scope_id, message.message_id),
        ).fetchone()
        expected = (
            message.scope_id,
            message.session_id,
            message.message_id,
            int(message.session_index),
            int(message.message_index),
            message.role,
            message.timestamp,
            message.content,
            _sha256(message.content),
        )
        if source is None or tuple(source[:9]) != expected:
            raise ProductionWriterError(
                f"replayed message Source identity changed: {message.message_id}"
            )
        source_record_id = str(source[10] or "").strip()
        source_turn_index = int(source[11] or 0)
        if (
            str(source[9] or "") != "enriched"
            or not source_record_id
            or source_turn_index <= 0
            or not str(source[12] or "").strip()
        ):
            raise ProductionWriterError(
                f"replayed message Source is not release-ready: {message.message_id}"
            )
        graph = connection.execute(
            "SELECT category,value,relation,turn_index,metadata_json FROM records "
            "WHERE scope_id=? AND memory_id=?",
            (message.scope_id, source_record_id),
        ).fetchone()
        if graph is None:
            raise ProductionWriterError(
                f"replayed message graph Source is missing: {message.message_id}"
            )
        try:
            metadata = json.loads(str(graph[4] or "{}"))
        except json.JSONDecodeError as exc:
            raise ProductionWriterError(
                f"replayed message graph Source metadata is invalid: {message.message_id}"
            ) from exc
        if (
            graph[0] != "source"
            or graph[1] != message.content
            or graph[2] != "dialogue_source"
            or int(graph[3]) != source_turn_index
            or not isinstance(metadata, Mapping)
            or str(metadata.get("source_record_id") or "") != source_record_id
            or str(metadata.get("message_id") or "") != message.message_id
            or str(metadata.get("session_id") or "") != message.session_id
            or int(metadata.get("session_index", -1)) != int(message.session_index)
            or int(metadata.get("message_index", -1)) != int(message.message_index)
            or str(metadata.get("actor_role") or metadata.get("speaker") or "")
            != message.role
            or str(metadata.get("timestamp") or "") != message.timestamp
            or str(metadata.get("raw_content") or "") != message.content
            or str(metadata.get("enrichment_status") or "") != "enriched"
        ):
            raise ProductionWriterError(
                f"replayed message graph Source identity changed: {message.message_id}"
            )

    def register_messages(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        v4: Any,
        include_enriched_replays: bool = True,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        messages: list[Any] = []
        seen: set[tuple[str, str]] = set()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for row_index, row in enumerate(rows):
                    if not isinstance(row, Mapping):
                        raise ProductionWriterError(f"input row {row_index} must be an object")
                    scope_id = str(row.get("scope_id") or "").strip()
                    session_id = str(row.get("session_id") or "").strip()
                    operation_id = str(row.get("operation_id") or "").strip()
                    if not scope_id.startswith("tmcra_v4:svc_"):
                        raise ProductionWriterError("production scope_id is invalid")
                    if (
                        self.expected_scope_id is not None
                        and scope_id != self.expected_scope_id
                    ):
                        raise ProductionWriterError(
                            "production scope_id does not match tenant and scope identity"
                        )
                    if not session_id or operation_id != self.operation_id:
                        raise ProductionWriterError("session_id or operation_id is invalid")
                    session = connection.execute(
                        "SELECT session_index FROM tmcra_service_sessions "
                        "WHERE scope_id=? AND session_id=?",
                        (scope_id, session_id),
                    ).fetchone()
                    if session is None:
                        session_index = int(
                            connection.execute(
                                "SELECT COALESCE(MAX(session_index), -1)+1 "
                                "FROM tmcra_service_sessions WHERE scope_id=?",
                                (scope_id,),
                            ).fetchone()[0]
                        )
                        connection.execute(
                            "INSERT INTO tmcra_service_sessions VALUES (?, ?, ?)",
                            (scope_id, session_id, session_index),
                        )
                    else:
                        session_index = int(session["session_index"])
                    raw_messages = list(row.get("messages") or [])
                    if not raw_messages:
                        raise ProductionWriterError("production writer input has no messages")
                    for raw in raw_messages:
                        if not isinstance(raw, Mapping):
                            raise ProductionWriterError("production message must be an object")
                        message_id = str(raw.get("message_id") or "").strip()
                        role = str(raw.get("role") or "").strip().lower()
                        timestamp = str(raw.get("timestamp") or "").strip()
                        content = str(raw.get("content") or "")
                        if not message_id or not timestamp or not content.strip():
                            raise ProductionWriterError(
                                "message_id, timestamp, and non-empty content are required"
                            )
                        if role not in {"user", "assistant", "system", "tool"}:
                            raise ProductionWriterError("production message role is invalid")
                        try:
                            actor_metadata = normalize_message_actor_metadata(
                                role, raw.get("metadata")
                            )
                        except ActorProvenanceError as exc:
                            raise ProductionWriterError(str(exc)) from exc
                        actor_json = actor_metadata_json(actor_metadata)
                        actor_sha256 = actor_metadata_sha256(actor_metadata)
                        identity = (scope_id, message_id)
                        if identity in seen:
                            raise ProductionWriterError("duplicate message_id in one operation")
                        seen.add(identity)
                        content_sha256 = _sha256(content)
                        prior = connection.execute(
                            "SELECT * FROM tmcra_service_messages "
                            "WHERE scope_id=? AND message_id=?",
                            identity,
                        ).fetchone()
                        if prior is None:
                            self.new_message_count += 1
                            if role == "user":
                                self.new_user_turn_count += 1
                            self.new_raw_token_estimate += _raw_token_estimate(content)
                            message_index = int(
                                connection.execute(
                                    "SELECT COALESCE(MAX(message_index), -1)+1 "
                                    "FROM tmcra_service_messages "
                                    "WHERE scope_id=? AND session_id=?",
                                    (scope_id, session_id),
                                ).fetchone()[0]
                            )
                            internal_message_id = self._internal_message_id(
                                session_index, message_index
                            )
                            connection.execute(
                                """
                                INSERT INTO tmcra_service_messages(
                                    scope_id, message_id, internal_message_id, session_id,
                                    message_index, role, timestamp, content_sha256,
                                    first_operation_id
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    scope_id,
                                    message_id,
                                    internal_message_id,
                                    session_id,
                                    message_index,
                                    role,
                                    timestamp,
                                    content_sha256,
                                    self.operation_id,
                                ),
                            )
                            connection.execute(
                                """
                                INSERT INTO tmcra_service_message_actor_provenance(
                                    scope_id, message_id, actor_metadata_json,
                                    actor_metadata_sha256
                                ) VALUES (?, ?, ?, ?)
                                """,
                                (scope_id, message_id, actor_json, actor_sha256),
                            )
                            first_operation_id = self.operation_id
                        else:
                            self.replayed_message_count += 1
                            expected = (
                                session_id,
                                role,
                                timestamp,
                                content_sha256,
                            )
                            actual = (
                                str(prior["session_id"]),
                                str(prior["role"]),
                                str(prior["timestamp"]),
                                str(prior["content_sha256"]),
                            )
                            if actual != expected:
                                raise ProductionWriterError(
                                    f"message_id replay changed immutable content: {message_id}"
                                )
                            actor_row = connection.execute(
                                "SELECT actor_metadata_json,actor_metadata_sha256 "
                                "FROM tmcra_service_message_actor_provenance "
                                "WHERE scope_id=? AND message_id=?",
                                identity,
                            ).fetchone()
                            if actor_row is None:
                                # Databases created before actor provenance are
                                # migrated lazily only when the replay supplies
                                # no new Agent identity.  An old message cannot
                                # acquire a producer retroactively.
                                legacy_actor = normalize_message_actor_metadata(
                                    role, {}
                                )
                                if actor_json != actor_metadata_json(legacy_actor):
                                    raise ProductionWriterError(
                                        f"message_id replay changed immutable actor metadata: {message_id}"
                                    )
                                connection.execute(
                                    "INSERT INTO tmcra_service_message_actor_provenance "
                                    "VALUES (?, ?, ?, ?)",
                                    (
                                        scope_id,
                                        message_id,
                                        actor_json,
                                        actor_sha256,
                                    ),
                                )
                            else:
                                stored_json = str(actor_row["actor_metadata_json"])
                                stored_sha256 = str(actor_row["actor_metadata_sha256"])
                                if (
                                    _sha256(stored_json) != stored_sha256
                                    or stored_json != actor_json
                                    or stored_sha256 != actor_sha256
                                ):
                                    raise ProductionWriterError(
                                        f"message_id replay changed immutable actor metadata: {message_id}"
                                    )
                            message_index = int(prior["message_index"])
                            internal_message_id = str(
                                prior["internal_message_id"]
                                or self._internal_message_id(session_index, message_index)
                            )
                            if not prior["internal_message_id"]:
                                connection.execute(
                                    "UPDATE tmcra_service_messages "
                                    "SET internal_message_id=? "
                                    "WHERE scope_id=? AND message_id=?",
                                    (internal_message_id, scope_id, message_id),
                                )
                            first_operation_id = str(
                                prior["first_operation_id"] or ""
                            )
                        message = v4.SourceMessage(
                            scope_id=scope_id,
                            session_id=session_id,
                            session_index=session_index,
                            message_index=message_index,
                            message_id=internal_message_id,
                            role=role,
                            timestamp=timestamp,
                            content=content,
                            actor_metadata=actor_metadata,
                        )
                        if prior is None or include_enriched_replays:
                            messages.append(message)
                        else:
                            self._require_enriched_replay(connection, message)
                        self.registered_messages[(scope_id, internal_message_id)] = message
                        self.source_origin_operations[
                            (scope_id, internal_message_id)
                        ] = first_operation_id
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        messages.sort(
            key=lambda item: (
                item.scope_id,
                item.session_index,
                item.message_index,
            )
        )
        return messages, []

    def remap_batches(self, batches: Sequence[Any], *, v4: Any) -> list[Any]:
        output: list[Any] = []
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for batch in batches:
                    row = connection.execute(
                        """
                        SELECT batch_index FROM tmcra_service_batches
                        WHERE scope_id=? AND session_id=? AND operation_id=?
                          AND local_batch_index=?
                        """,
                        (
                            batch.scope_id,
                            batch.session_id,
                            self.operation_id,
                            batch.batch_index,
                        ),
                    ).fetchone()
                    if row is None:
                        batch_index = int(
                            connection.execute(
                                "SELECT COALESCE(MAX(batch_index), -1)+1 "
                                "FROM tmcra_service_batches "
                                "WHERE scope_id=? AND session_id=?",
                                (batch.scope_id, batch.session_id),
                            ).fetchone()[0]
                        )
                        connection.execute(
                            "INSERT INTO tmcra_service_batches VALUES (?, ?, ?, ?, ?)",
                            (
                                batch.scope_id,
                                batch.session_id,
                                self.operation_id,
                                batch.batch_index,
                                batch_index,
                            ),
                        )
                    else:
                        batch_index = int(row["batch_index"])
                    output.append(
                        v4.SourceBatch(
                            scope_id=batch.scope_id,
                            session_id=batch.session_id,
                            session_index=batch.session_index,
                            batch_index=batch_index,
                            messages=batch.messages,
                        )
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return output


def _writer_outcome_unknown(exc: BaseException) -> bool:
    metadata = getattr(exc, "metadata", None)
    values = dict(metadata) if isinstance(metadata, Mapping) else {}
    status = str(values.get("status") or "").strip().lower()
    return bool(
        status in {"request_error", "transport_error", "timeout"}
        or isinstance(exc, (TimeoutError, OSError))
        or "timeout" in str(exc).lower()
    )


def _terminalize_operation_journals(
    database: Path,
    operation_id: str,
    messages: Sequence[Any],
    *,
    error: str,
    outcome_unknown: bool,
) -> None:
    """End pending enrichment without changing immutable Source durability."""

    if not database.is_file() or not messages:
        return
    safe_error = f"{error.split(':', 1)[0]}:{_sha256(error)}"
    with closing(sqlite3.connect(database, timeout=30.0)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "v4_source_journal" not in tables:
            return
        connection.execute("BEGIN IMMEDIATE")
        try:
            for message in messages:
                connection.execute(
                    "UPDATE v4_source_journal SET status='failed',"
                    "enrichment_error=?,updated_at=? "
                    "WHERE scope_id=? AND message_id=? AND status='pending'",
                    (
                        safe_error,
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        message.scope_id,
                        message.message_id,
                    ),
                )
            if {
                "tmcra_service_batches",
                "v4_batch_journal",
            }.issubset(tables):
                rows = connection.execute(
                    "SELECT journal.rowid,journal.status "
                    "FROM tmcra_service_batches AS batches "
                    "JOIN v4_batch_journal AS journal "
                    "ON journal.scope_id=batches.scope_id "
                    "AND journal.session_id=batches.session_id "
                    "AND journal.batch_index=batches.batch_index "
                    "WHERE batches.operation_id=?",
                    (operation_id,),
                ).fetchall()
                for row in rows:
                    status = str(row["status"] or "")
                    if status in {"committed", "validated"}:
                        # A validated response remains the durable replay
                        # boundary for local graph-commit failures.  Downgrading
                        # it to failed makes recovery misclassify a frozen
                        # commit plan as another provider attempt.
                        continue
                    terminal = (
                        "outcome_unknown"
                        if outcome_unknown and status == "api_started"
                        else "failed"
                    )
                    connection.execute(
                        "UPDATE v4_batch_journal SET status=?,error=?,updated_at=? "
                        "WHERE rowid=?",
                        (
                            terminal,
                            safe_error,
                            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            int(row["rowid"]),
                        ),
                    )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


def _verified_source_bindings(
    database: Path,
    messages: Sequence[Any],
    *,
    graph_factory: Any,
) -> list[tuple[Any, dict[str, Any], Any]]:
    if not messages or not database.is_file():
        return []
    with closing(sqlite3.connect(database, timeout=30.0)) as connection:
        connection.row_factory = sqlite3.Row
        rows: list[tuple[Any, dict[str, Any], Any]] = []
        for message in messages:
            row = connection.execute(
                "SELECT * FROM v4_source_journal WHERE scope_id=? AND message_id=?",
                (message.scope_id, message.message_id),
            ).fetchone()
            if row is None:
                continue
            value = dict(row)
            expected = (
                message.session_id,
                int(message.session_index),
                int(message.message_index),
                message.role,
                message.timestamp,
                _sha256(message.content),
            )
            actual = (
                str(value.get("session_id") or ""),
                int(value.get("session_index") or 0),
                int(value.get("message_index") or 0),
                str(value.get("message_role") or ""),
                str(value.get("timestamp") or ""),
                str(value.get("content_sha256") or ""),
            )
            source_record_id = str(value.get("source_record_id") or "").strip()
            if actual != expected:
                raise ProductionWriterError(
                    f"{message.message_id}: immutable source journal identity changed"
                )
            status = str(value.get("status") or "")
            if status not in {"enriched", "failed"}:
                raise ProductionWriterError(
                    f"{message.message_id}: source journal is not terminal"
                )
            if not source_record_id:
                if status == "failed":
                    continue
                raise ProductionWriterError(
                    f"{message.message_id}: enriched source lacks its graph record"
                )
            backend = graph_factory.for_scope(message.scope_id)
            backend.verify_source(
                message,
                source_record_id,
                int(value.get("source_turn_index") or 0),
            )
            rows.append((message, value, backend))
    return rows


def _durable_source_records(
    bindings: Sequence[tuple[Any, Mapping[str, Any], Any]],
    registry: IdentityRegistry,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for message, source, _backend in bindings:
        origin_operation_id = registry.source_origin_operations.get(
            (message.scope_id, message.message_id), ""
        )
        # Rows created before source-level accounting are already represented
        # by the legacy operation watermark and must not be counted again.
        if not origin_operation_id:
            continue
        records.append(
            {
                "source_record_id": str(source["source_record_id"]),
                "origin_operation_id": origin_operation_id,
                "raw_token_estimate": _raw_token_estimate(message.content),
                "user_turns": int(message.role == "user"),
            }
        )
    records.sort(key=lambda item: item["source_record_id"])
    return records


def _write_writer_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(report), ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class LeasedDeepSeekClient:
    def __init__(
        self,
        *,
        v4: Any,
        pool: ProviderKeyPool,
        operation_id: str,
        base_url: str,
        model: str,
        timeout: float,
        max_tokens: int,
        provider: str = DEEPSEEK_PROVIDER,
        prompt_adapter: str = "none",
        ledger_database: Path | str | None = None,
        tenant_id: str | None = None,
        scope_name: str | None = None,
        job_id: str | None = None,
        stage_id: str | None = None,
        stage_name: str | None = None,
        usage_attribution: UsageAttribution = UNATTRIBUTED,
        acquire_timeout: float | None = None,
    ) -> None:
        self.v4 = v4
        self.pool = pool
        self.operation_id = operation_id
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.provider = str(provider).strip()
        self.prompt_adapter = str(prompt_adapter).strip()
        self.stage_name = stage_name or model
        self.usage_attribution = usage_attribution
        if self.provider not in {
            DEEPSEEK_PROVIDER,
            LOCAL_QWEN_PROVIDER,
            OPENAI_COMPATIBLE_PROVIDER,
        }:
            raise ProductionWriterError(f"unsupported Writer provider: {self.provider}")
        if self.provider == LOCAL_QWEN_PROVIDER and not self.model:
            raise ProductionWriterError("local Writer model alias is required")
        if self.provider == LOCAL_QWEN_PROVIDER and self.prompt_adapter not in {
            QWEN36_ADAPTER_ID,
            QWEN36_REVIEWER_ADAPTER_ID,
        }:
            raise ProductionWriterError(
                "local Qwen Writer uses an unsupported prompt adapter"
            )
        identity = {
            "tenant_id": tenant_id,
            "scope_name": scope_name,
            "job_id": job_id,
            "stage_id": stage_id,
        }
        if any(value is not None and not str(value).strip() for value in identity.values()):
            raise ProductionWriterError("provider ledger identity values cannot be empty")
        if any(value is not None for value in identity.values()) and ledger_database is None:
            raise ProductionWriterError("provider ledger database is required with ledger identity")
        if ledger_database is not None and not all(identity.values()):
            raise ProductionWriterError(
                "provider ledger requires tenant, scope, job, and stage identity"
            )
        self.tenant_id = tenant_id
        self.scope_name = scope_name
        self.provider_user_id = (
            "tmcra_"
            + hashlib.sha256(
                f"{tenant_id}\0{scope_name}".encode("utf-8")
            ).hexdigest()[:32]
            if tenant_id is not None and scope_name is not None
            else ""
        )
        self.job_id = job_id
        self.stage_id = stage_id
        self.ledger = (
            JobStore(ControlDB(ledger_database))
            if ledger_database is not None
            else None
        )
        if self.ledger is not None:
            self._register_price()
        if not math.isfinite(float(timeout)) or timeout <= 0:
            raise ProductionWriterError("provider request timeout must be positive")
        if acquire_timeout is None:
            default_acquire_timeout = (
                "90" if self.provider == LOCAL_QWEN_PROVIDER else "30"
            )
            configured_acquire_timeout = os.getenv(
                "TMCRA_LOCAL_PROVIDER_ACQUIRE_TIMEOUT_SECONDS"
                if self.provider == LOCAL_QWEN_PROVIDER
                else "TMCRA_PROVIDER_ACQUIRE_TIMEOUT_SECONDS"
            ) or os.getenv(
                "TMCRA_PROVIDER_ACQUIRE_TIMEOUT_SECONDS", default_acquire_timeout
            )
            try:
                acquire_timeout = float(configured_acquire_timeout)
            except (TypeError, ValueError) as exc:
                raise ProductionWriterError(
                    "provider pool acquire timeout must be numeric"
                ) from exc
        if not math.isfinite(float(acquire_timeout)) or acquire_timeout < 0:
            raise ProductionWriterError(
                "provider pool acquire timeout must be finite and non-negative"
            )
        self.acquire_timeout = float(acquire_timeout)
        self.heartbeat_interval = min(30.0, pool.lease_seconds / 3.0)
        if self.heartbeat_interval <= 0 or self.heartbeat_interval >= pool.lease_seconds:
            raise ProductionWriterError(
                "provider lease heartbeat interval must be shorter than lease duration"
            )

    def _register_price(self) -> None:
        rates, price_version, source = self._price_contract()
        self.ledger.upsert_provider_price(  # type: ignore[union-attr]
            self.provider,
            self.model,
            cache_hit_input_micro_cny_per_million=rates[0],
            cache_miss_input_micro_cny_per_million=rates[1],
            output_micro_cny_per_million=rates[2],
            effective_at=0.0,
            currency="CNY",
            metadata={
                "price_version": price_version,
                "source": source,
                "unit": "micro-CNY per million tokens",
            },
        )

    def _price_contract(self) -> tuple[tuple[int, int, int], str, str]:
        if self.provider == LOCAL_QWEN_PROVIDER:
            return (0, 0, 0), LOCAL_QWEN_PRICE_VERSION, LOCAL_QWEN_PRICING_SOURCE
        rates = DEEPSEEK_V4_PRICES_MICRO_CNY.get(self.model)
        if rates is not None:
            return rates, DEEPSEEK_V4_PRICE_VERSION, DEEPSEEK_PRICING_SOURCE
        names = (
            "TMCRA_WRITER_PRICE_CACHE_HIT_MICRO_CNY_PER_MILLION",
            "TMCRA_WRITER_PRICE_CACHE_MISS_MICRO_CNY_PER_MILLION",
            "TMCRA_WRITER_PRICE_OUTPUT_MICRO_CNY_PER_MILLION",
        )
        raw_rates = [str(os.getenv(name) or "").strip() for name in names]
        if any(raw_rates):
            if not all(raw_rates):
                raise ProductionWriterError(
                    "custom model pricing requires all three TMCRA_WRITER_PRICE_* values"
                )
            try:
                configured = tuple(int(value) for value in raw_rates)
            except ValueError as exc:
                raise ProductionWriterError(
                    "custom model pricing values must be non-negative integers"
                ) from exc
            if any(value < 0 for value in configured):
                raise ProductionWriterError(
                    "custom model pricing values must be non-negative integers"
                )
            return (
                configured,
                str(os.getenv("TMCRA_WRITER_PRICE_VERSION") or OPERATOR_PRICE_VERSION),
                str(os.getenv("TMCRA_WRITER_PRICING_SOURCE") or "operator configuration"),
            )
        return (
            (0, 0, 0),
            UNPRICED_MODEL_VERSION,
            "no operator pricing configured; usage is recorded without cost",
        )

    @staticmethod
    def _metadata(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _usage(metadata: Mapping[str, Any]) -> tuple[dict[str, int], str]:
        raw = metadata.get("usage")
        if not isinstance(raw, Mapping):
            return {}, "missing"

        def count(*names: str) -> int | None:
            value = next((raw.get(name) for name in names if raw.get(name) is not None), None)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) < 0:
                raise ValueError("provider usage contains a negative or non-numeric count")
            return int(value)

        prompt = count("prompt_tokens", "input_tokens")
        completion = count("completion_tokens", "output_tokens")
        hit = count("prompt_cache_hit_tokens", "cache_read_input_tokens", "cached_tokens")
        miss = count("prompt_cache_miss_tokens", "cache_miss_input_tokens")
        if prompt is None or completion is None or hit is None:
            return {}, "invalid"
        hit = int(hit)
        if miss is None:
            miss = prompt - hit
        if hit > prompt or miss < 0 or hit + miss != prompt:
            return {}, "invalid"
        total = count("total_tokens") or prompt + completion
        return {
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": int(total),
            "cache_hit_tokens": hit,
            "cache_miss_tokens": int(miss),
        }, "complete"

    def _cost(self, usage: Mapping[str, int]) -> int | None:
        if not usage:
            return None
        rates, price_version, _source = self._price_contract()
        if price_version == UNPRICED_MODEL_VERSION:
            return None
        hit_cost = int(usage["cache_hit_tokens"]) * rates[0]
        miss_cost = int(usage["cache_miss_tokens"]) * rates[1]
        output_cost = int(usage["output_tokens"]) * rates[2]
        if self.provider == LOCAL_QWEN_PROVIDER:
            return 0
        # Round up so a non-empty paid call cannot be represented as zero cost.
        return (hit_cost + miss_cost + output_cost + 999_999) // 1_000_000

    @staticmethod
    def _safe_error(exc: BaseException, metadata: Mapping[str, Any]) -> str:
        parts = [exc.__class__.__name__]
        for key in ("status", "http_status"):
            if metadata.get(key) is not None:
                parts.append(f"{key}={metadata[key]}")
        return ":".join(parts)

    @staticmethod
    def _outcome(metadata: Mapping[str, Any], exc: BaseException) -> str:
        status = str(metadata.get("status") or "").lower()
        if status == "completed":
            return "completed"
        if status in {"request_error", "transport_error", "timeout"}:
            return "unknown"
        if isinstance(exc, (TimeoutError, OSError)) or "timeout" in str(exc).lower():
            return "unknown"
        return "failed"

    @staticmethod
    def _pool_outcome(metadata: Mapping[str, Any], exc: BaseException) -> str:
        raw_status = metadata.get("status_code") or metadata.get("http_status") or 0
        try:
            status = int(raw_status)
        except (TypeError, ValueError):
            status = 0
        if status in {401, 403}:
            return "fatal_error"
        if status == 402:
            return "billing_exhausted"
        if status == 429:
            return "rate_limited"
        if status >= 500 or status in {408, 425}:
            return "transient_error"
        if 400 <= status < 500:
            return "request_error"
        transport_status = str(metadata.get("status") or "").strip().lower()
        if transport_status in {"transport_error", "timeout"}:
            return "transient_error"
        if isinstance(exc, (TimeoutError, OSError)) or "timeout" in str(exc).lower():
            return "transient_error"
        # Response schema/usage validation and local journaling failures are
        # request/service errors; they must never cool a shared credential.
        return "request_error"

    def _journal(
        self,
        metadata: Mapping[str, Any],
        *,
        status: str,
        lease: Any,
        error: BaseException | None = None,
    ) -> None:
        if self.ledger is None:
            return
        _rates, price_version, _source = self._price_contract()
        physical_call_id = str(metadata.get("physical_call_id") or ("missing_" + uuid.uuid4().hex))
        usage, usage_state = self._usage(metadata)
        request_sha256 = metadata.get("request_sha256")
        response_sha256 = metadata.get("response_sha256")
        started_at = metadata.get("started_at")
        if not isinstance(started_at, (int, float)):
            started_at = time.time()
        self.ledger.record_provider_call(
            self.tenant_id, self.provider, self.model,
            scope_name=self.scope_name, call_id=physical_call_id,
            job_id=self.job_id, stage_id=self.stage_id,
            operation=str(metadata.get("stage") or self.stage_name), status="started",
            input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"), cache_hit_tokens=usage.get("cache_hit_tokens"),
            cache_miss_tokens=usage.get("cache_miss_tokens"), usage_state=usage_state,
            price_version=price_version, key_id=lease.key_id,
            usage_attribution=self.usage_attribution,
            request_sha256=str(request_sha256) if request_sha256 else None,
            started_at=float(started_at), created_at=float(started_at),
        )
        self.ledger.transition_provider_call(
            physical_call_id, status,
            error=None if error is None else self._safe_error(error, metadata),
            input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            cost_micro_cny=self._cost(usage) if status == "completed" and usage_state == "complete" else None,
            cache_hit_tokens=usage.get("cache_hit_tokens"),
            cache_miss_tokens=usage.get("cache_miss_tokens"), usage_state=usage_state,
            price_version=price_version,
            response_sha256=str(response_sha256) if response_sha256 else None,
        )

    def _acquire_lease(self, payload: Mapping[str, Any]) -> Any:
        started = time.monotonic()
        deadline = started + self.acquire_timeout
        attempts = 0
        delay = 0.05
        while True:
            attempts += 1
            try:
                return self.pool.acquire(owner=f"{self.operation_id}:{self.model}")
            except ProviderPoolExhausted as exc:
                waited = max(0.0, time.monotonic() - started)
                saturated = str(exc).startswith("provider pool is saturated:")
                remaining = deadline - time.monotonic()
                if saturated and remaining > 0:
                    time.sleep(min(delay, remaining))
                    delay = min(0.5, delay * 1.5)
                    continue
                # Acquisition failed before a credential lease and before any
                # HTTP request. Persist that proof for audited recovery.
                exc.metadata = {
                    "status": "provider_pool_unavailable",
                    "physical_api_call": False,
                    "physical_api_calls": 0,
                    "stage": self.stage_name,
                    "model": self.model,
                    "payload_sha256": hashlib.sha256(
                        json.dumps(
                            dict(payload),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest(),
                    "provider_pool_wait_seconds": round(waited, 6),
                    "provider_pool_acquire_attempts": attempts,
                }
                raise

    def _call(self, payload: Mapping[str, Any], method_name: str) -> Any:
        try:
            lease = self._acquire_lease(payload)
        except ProviderPoolExhausted:
            raise
        pool_outcome = "success"
        try:
            client_kwargs = {
                "base_url": self.base_url,
                "model": self.model,
                "api_keys": [lease.secret],
                "timeout": self.timeout,
                "max_tokens": self.max_tokens,
            }
            if self.provider == LOCAL_QWEN_PROVIDER:
                client = create_qwen36_batch_client(v4=self.v4, **client_kwargs)
            else:
                client = self.v4.DeepSeekBatchClient(**client_kwargs)
            # API keys are reusable credentials, not user identities.  Bind
            # provider-side KV cache and scheduling isolation to a stable,
            # privacy-safe tenant/scope hash instead of the leased key.
            client.user_id = (
                self.provider_user_id if self.provider == DEEPSEEK_PROVIDER else ""
            )
            try:
                result = self._complete_with_heartbeat(
                    client, payload, lease, method_name=method_name
                )
            except Exception as exc:
                metadata = self._metadata(getattr(exc, "metadata", None))
                ledger_outcome = self._outcome(metadata, exc)
                pool_outcome = self._pool_outcome(metadata, exc)
                self._journal(metadata, status=ledger_outcome, lease=lease, error=exc)
                raise
            metadata = {}
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
                metadata = self._metadata(result[1])
            self._journal(metadata, status="completed", lease=lease)
            return result
        except Exception as exc:
            metadata = self._metadata(getattr(exc, "metadata", None))
            outcome = self._pool_outcome(metadata, exc)
            retry_after = metadata.get("retry_after") or metadata.get("retry_after_seconds")
            pool_outcome = outcome
            raise
        finally:
            metadata = locals().get("metadata", {})
            retry_after = metadata.get("retry_after") or metadata.get("retry_after_seconds")
            self.pool.release(
                lease,
                outcome=pool_outcome,
                retry_after_seconds=(float(retry_after) if retry_after else None),
            )

    def _complete_with_heartbeat(
        self,
        client: Any,
        payload: Mapping[str, Any],
        lease: Any,
        *,
        method_name: str = "complete",
    ) -> Any:
        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.wait(self.heartbeat_interval):
                try:
                    if self.pool.heartbeat(lease) is None:
                        return
                except Exception:
                    return

        thread = threading.Thread(
            target=heartbeat,
            name=f"provider-lease-heartbeat-{self.operation_id}",
            daemon=True,
        )
        thread.start()
        try:
            return getattr(client, method_name)(payload)
        finally:
            stop.set()
            thread.join(timeout=max(1.0, self.heartbeat_interval + 1.0))

    def complete(self, payload: Mapping[str, Any]) -> Any:
        return self._call(payload, "complete")

    def reconcile(self, payload: Mapping[str, Any]) -> Any:
        return self._call(payload, "reconcile")


class UserProviderWriterClient:
    def __init__(self, *, v4: Any, broker: UserProviderBrokerClient) -> None:
        self.v4 = v4
        self.broker = broker
        self.model = broker.model
        self.provider = broker.provider

    def _sync_identity(self) -> None:
        self.model = self.broker.model
        self.provider = self.broker.provider

    def complete(self, payload: Mapping[str, Any]) -> Any:
        result = self.broker.complete_prompt(
            system_prompt=str(self.v4.BATCH_SYSTEM_PROMPT),
            payload=payload,
            operation="batch_flash",
            response_schema=self.v4.batch_response_json_schema(payload),
        )
        self._sync_identity()
        return result

    def reconcile(self, payload: Mapping[str, Any]) -> Any:
        delegate = self.v4.DeepSeekBatchClient.__new__(
            self.v4.DeepSeekBatchClient
        )
        delegate.model = self.model

        def complete_through_broker(
            *,
            model: str,
            system_prompt: str,
            payload: Mapping[str, Any],
            stage: str,
        ) -> Any:
            del model
            return self.broker.complete_prompt(
                system_prompt=system_prompt,
                payload=payload,
                operation=stage,
            )

        delegate._complete = complete_through_broker
        result = delegate.reconcile(payload)
        self._sync_identity()
        return result


_MISSING = object()


def execute_writer(
    *,
    input_path: Path,
    out_dir: Path,
    database: Path,
    operation_id: str,
    repo: Path,
    tenant_id: str,
    scope_name: str,
    job_id: str,
    stage_id: str,
    stage_attempt: int = 1,
    reviewer_model: str = "deepseek-v4-pro",
    timeout_seconds: float = 180.0,
    max_tokens: int = 16384,
    recovery_mode: str = "none",
    usage_attribution: UsageAttribution = UNATTRIBUTED,
    provider_execution: Mapping[str, Any] | None = None,
    v4_module: Any | None = None,
) -> dict[str, Any]:
    """Run one production Writer operation using the unchanged V4 core.

    A resident worker calls this sequentially. The temporary V4 module hooks are
    always restored, including on failure, so one process can safely serve many
    independent operations without leaking operation-specific identity state.
    """
    identities = (tenant_id, scope_name, job_id, stage_id)
    if not all(str(value).strip() for value in identities):
        raise ProductionWriterError(
            "production writer requires tenant, scope, job, and stage ledger identity"
        )
    if operation_id != job_id or stage_id != f"{job_id}:writer":
        raise ProductionWriterError(
            "production writer operation, job, and stage identity must be bound"
        )
    if (
        isinstance(stage_attempt, bool)
        or not isinstance(stage_attempt, int)
        or stage_attempt <= 0
    ):
        raise ProductionWriterError("production writer stage attempt must be positive")
    accounting_stage_id = f"{stage_id}:attempt:{stage_attempt}"
    recovery_mode = str(recovery_mode or "none").strip()
    if recovery_mode not in WRITER_RECOVERY_MODES:
        raise ProductionWriterError("production writer recovery mode is invalid")
    repo = repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if v4_module is None:
        import tmcra_v4_batch_writer as v4
    else:
        v4 = v4_module

    rows = json.loads(input_path.resolve().read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ProductionWriterError("production writer input must be a non-empty array")
    operation_ids = {str(row.get("operation_id") or "") for row in rows}
    if operation_ids != {operation_id}:
        raise ProductionWriterError("operation identity differs from CLI contract")
    input_sha256 = hashlib.sha256(
        json.dumps(
            rows,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    database = database.resolve()
    expected_scope_key = hashlib.sha256(
        f"{tenant_id}\0{scope_name}".encode("utf-8")
    ).hexdigest()[:32]
    expected_scope_id = f"tmcra_v4:svc_{expected_scope_key}"
    registry = IdentityRegistry(
        database,
        operation_id,
        expected_scope_id=expected_scope_id,
    )
    original_normalize = getattr(v4, "normalize_source_inventory", _MISSING)
    original_build_batches = v4.build_batches
    original_build_batch_request = v4.build_batch_request
    original_prompt_version = getattr(v4, "PROMPT_VERSION", _MISSING)

    def normalize(value: Sequence[Mapping[str, Any]]) -> tuple[list[Any], list[Any]]:
        return registry.register_messages(
            value,
            v4=v4,
            include_enriched_replays=False,
        )

    def build(messages: Sequence[Any], **kwargs: Any) -> list[Any]:
        return registry.remap_batches(
            original_build_batches(messages, **kwargs), v4=v4
        )

    def build_request(
        batch: Any,
        unresolved_interactions: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        request = original_build_batch_request(batch, unresolved_interactions)
        max_items, max_chars = writer_unresolved_limits_from_env()
        request["unresolved_interactions"] = select_unresolved_interactions(
            request.get("unresolved_interactions") or [],
            request.get("messages") or [],
            max_items=max_items,
            max_chars=max_chars,
        )
        return request

    # The benchmark module remains unchanged; only this process-local call is adapted.
    v4.normalize_source_inventory = normalize
    v4.build_batches = build
    v4.build_batch_request = build_request
    try:
        control_db = str(os.getenv("TMCRA_SERVICE_CONTROL_DB") or "").strip()
        if not control_db:
            raise ProductionWriterError(
                "production writer requires an explicit control DB"
            )
        try:
            user_execution = normalize_user_provider_execution(
                provider_execution,
                stage="writer",
            )
        except ValueError as exc:
            raise ProductionWriterError(str(exc)) from exc
        base_writer_prompt = str(getattr(v4, "BATCH_SYSTEM_PROMPT", "") or "")
        if user_execution is not None:
            if not base_writer_prompt:
                raise ProductionWriterError("V4 Writer system prompt is missing")
            writer_prompt_sha256 = _sha256(base_writer_prompt)
            auth_key_id = user_execution["auth_key_id"]
            flash = UserProviderWriterClient(
                v4=v4,
                broker=UserProviderBrokerClient(
                    control_db=Path(control_db),
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    auth_key_id=auth_key_id,
                    job_id=job_id,
                    stage_id=accounting_stage_id,
                    task_stage="writer",
                    timeout=timeout_seconds,
                    max_tokens=max_tokens,
                    usage_attribution=usage_attribution,
                    record_ledger=True,
                ),
            )
            pro = UserProviderWriterClient(
                v4=v4,
                broker=UserProviderBrokerClient(
                    control_db=Path(control_db),
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    auth_key_id=auth_key_id,
                    job_id=job_id,
                    stage_id=accounting_stage_id,
                    task_stage="writer",
                    timeout=timeout_seconds,
                    max_tokens=max_tokens,
                    usage_attribution=usage_attribution,
                    record_ledger=True,
                ),
            )
            primary_provider = USER_PROVIDER
            primary_model = flash.model
            primary_prompt_adapter = "none"
            reviewer_provider = USER_PROVIDER
            reviewer_model_value = pro.model
        else:
            try:
                primary_route = primary_writer_route(os.environ)
                reviewer_route = reviewer_writer_route(
                    os.environ, fallback_model=reviewer_model
                )
            except ValueError as exc:
                raise ProductionWriterError(
                    f"invalid Writer provider route: {exc}"
                ) from exc
            if primary_route.prompt_adapter == QWEN36_ADAPTER_ID:
                if original_prompt_version is _MISSING:
                    raise ProductionWriterError("V4 Writer prompt version is missing")
                if not base_writer_prompt:
                    raise ProductionWriterError("V4 Writer system prompt is missing")
                v4.PROMPT_VERSION = (
                    f"{original_prompt_version}+{QWEN36_ADAPTER_ID}"
                )
                writer_prompt_sha256 = qwen36_prompt_sha256(base_writer_prompt)
            else:
                writer_prompt_sha256 = _sha256(
                    base_writer_prompt or str(v4.PROMPT_VERSION)
                )
            local_recovery_operation = bool(
                recovery_mode != "none" or stage_attempt > 1
            )
            primary_is_local_recovery = bool(
                primary_route.provider == LOCAL_QWEN_PROVIDER
                and local_recovery_operation
            )
            primary_pool = ProviderKeyPool(
                Path(control_db),
                pool=(
                    f"{primary_route.pool_name}-recovery"
                    if primary_is_local_recovery
                    else primary_route.pool_name
                ),
                keys=primary_route.api_keys,
                max_concurrency_per_key=(
                    local_writer_recovery_concurrency_from_env()
                    if primary_is_local_recovery
                    else 1
                    if primary_route.provider == LOCAL_QWEN_PROVIDER
                    else int(os.getenv("TMCRA_PROVIDER_KEY_CONCURRENCY", "2"))
                ),
                lease_seconds=int(os.getenv("TMCRA_PROVIDER_LEASE_SECONDS", "300")),
            )
            reviewer_is_local_recovery = bool(
                reviewer_route.provider == LOCAL_QWEN_PROVIDER
                and local_recovery_operation
            )
            reviewer_pool = ProviderKeyPool(
                Path(control_db),
                pool=(
                    f"{reviewer_route.pool_name}-recovery"
                    if reviewer_is_local_recovery
                    else reviewer_route.pool_name
                ),
                keys=reviewer_route.api_keys,
                max_concurrency_per_key=(
                    local_writer_recovery_concurrency_from_env()
                    if reviewer_is_local_recovery
                    else 1
                    if reviewer_route.provider == LOCAL_QWEN_PROVIDER
                    else int(os.getenv("TMCRA_PROVIDER_KEY_CONCURRENCY", "2"))
                ),
                lease_seconds=int(os.getenv("TMCRA_PROVIDER_LEASE_SECONDS", "300")),
            )
            flash = LeasedDeepSeekClient(
                v4=v4,
                pool=primary_pool,
                operation_id=operation_id,
                base_url=primary_route.base_url,
                model=primary_route.model,
                timeout=timeout_seconds,
                max_tokens=max_tokens,
                provider=primary_route.provider,
                prompt_adapter=primary_route.prompt_adapter,
                ledger_database=Path(control_db),
                tenant_id=tenant_id,
                scope_name=scope_name,
                job_id=job_id,
                stage_id=accounting_stage_id,
                stage_name="batch_flash",
                usage_attribution=usage_attribution,
            )
            pro = LeasedDeepSeekClient(
                v4=v4,
                pool=reviewer_pool,
                operation_id=operation_id,
                base_url=reviewer_route.base_url,
                model=reviewer_route.model,
                timeout=timeout_seconds,
                max_tokens=max_tokens,
                provider=reviewer_route.provider,
                prompt_adapter=reviewer_route.prompt_adapter,
                ledger_database=Path(control_db),
                tenant_id=tenant_id,
                scope_name=scope_name,
                job_id=job_id,
                stage_id=accounting_stage_id,
                stage_name="reconciliation_pro",
                usage_attribution=usage_attribution,
            )
            primary_provider = primary_route.provider
            primary_model = primary_route.model
            primary_prompt_adapter = primary_route.prompt_adapter
            reviewer_provider = reviewer_route.provider
            reviewer_model_value = reviewer_route.model
        graph_factory = v4.RealGraphFactory(repo=repo, database=database)
        writer = v4.V4BatchWriter(
            store=v4.V4BatchStore(database),
            flash_client=flash,
            pro_client=pro,
            graph_factory=graph_factory,
            log_dir=out_dir,
            revalidate_failed_raw_response=recovery_mode
            in {"validation", "definitive_provider_failure"},
            recover_interrupted_api_calls=recovery_mode
            in {
                "definitive_provider_failure",
                "audited_local_inference_cancelled",
            },
        )
        bindings: list[tuple[Any, dict[str, Any], Any]] = []
        try:
            report = {
                **dict(writer.run(rows)),
                "status": "complete",
                "degraded": False,
                "provider_outcome_unknown": False,
            }
            registered = list(registry.registered_messages.values())
            bindings = _verified_source_bindings(
                database,
                registered,
                graph_factory=graph_factory,
            )
            if len(bindings) != len(registered) or not registered:
                raise ProductionWriterError(
                    "writer completed without every immutable Source binding"
                )
        except Exception as exc:
            registered = list(registry.registered_messages.values())
            outcome_unknown = _writer_outcome_unknown(exc)
            error = f"{type(exc).__name__}:{exc}"
            _terminalize_operation_journals(
                database,
                operation_id,
                registered,
                error=error,
                outcome_unknown=outcome_unknown,
            )
            try:
                bindings = _verified_source_bindings(
                    database,
                    registered,
                    graph_factory=graph_factory,
                )
            except Exception:
                raise exc
            if not bindings or not registered:
                raise
            safe_error = f"{type(exc).__name__}:{_sha256(error)}"
            failed_count = 0
            enriched_count = 0
            for _message, source, backend in bindings:
                if source["status"] == "failed":
                    backend.set_enrichment_status(
                        str(source["source_record_id"]), "failed", safe_error
                    )
                    failed_count += 1
                else:
                    enriched_count += 1
            report = {
                **dict(getattr(writer, "stats", {}) or {}),
                "status": "degraded",
                "degraded": True,
                "degraded_error_type": type(exc).__name__,
                "degraded_error_sha256": _sha256(error),
                "degraded_source_count": failed_count,
                "enriched_source_count": enriched_count,
                "source_durability_boundary_reached": True,
                "provider_outcome_unknown": outcome_unknown,
            }
        registered = list(registry.registered_messages.values())
        durable_sources = _durable_source_records(bindings, registry)
        input_complete = bool(registered) and len(bindings) == len(registered)
        report.update(
            {
                "schema_version": "tmcra.service.incremental-writer.1",
                "writer_schema_version": v4.BATCH_SCHEMA_VERSION,
                "prompt_version": v4.PROMPT_VERSION,
                "writer_provider": str(getattr(flash, "provider", primary_provider)),
                "writer_model": str(getattr(flash, "model", primary_model)),
                "writer_prompt_adapter": primary_prompt_adapter,
                "writer_prompt_sha256": writer_prompt_sha256,
                "reviewer_provider": str(
                    getattr(pro, "provider", reviewer_provider)
                ),
                "reviewer_model": str(
                    getattr(pro, "model", reviewer_model_value)
                ),
                "candidate_selector_version": v4.CANDIDATE_SELECTOR_VERSION,
                "operation_id": operation_id,
                "tenant_id": tenant_id,
                "scope_name": scope_name,
                "job_id": job_id,
                "stage_id": stage_id,
                "stage_attempt": stage_attempt,
                "input_sha256": input_sha256,
                "recovery_mode": recovery_mode,
                "new_message_count": registry.new_message_count,
                "replayed_message_count": registry.replayed_message_count,
                "new_user_turn_count": registry.new_user_turn_count,
                "new_raw_token_estimate": registry.new_raw_token_estimate,
                "durable_sources": durable_sources,
                "durable_source_count": len(durable_sources),
                "verified_source_count": len(bindings),
                "input_message_count": len(registered),
                "input_messages": len(registered),
                "input_complete": input_complete,
                "estimator_version": "cjk1_other4_nonempty_v1",
                "completed": True,
                "db_path": str(database),
            }
        )
        _write_writer_report(out_dir / "product_writer_report.json", report)
        return report
    finally:
        v4.build_batch_request = original_build_batch_request
        v4.build_batches = original_build_batches
        if original_prompt_version is _MISSING:
            try:
                delattr(v4, "PROMPT_VERSION")
            except AttributeError:
                pass
        else:
            v4.PROMPT_VERSION = original_prompt_version
        if original_normalize is _MISSING:
            try:
                delattr(v4, "normalize_source_inventory")
            except AttributeError:
                pass
        else:
            v4.normalize_source_inventory = original_normalize


def _identity(cli_value: str | None, *environment_names: str) -> str:
    if cli_value and cli_value.strip():
        return cli_value.strip()
    for name in environment_names:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="TMCRA production incremental writer")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--tenant-id")
    parser.add_argument("--scope-name", "--scope")
    parser.add_argument("--job-id")
    parser.add_argument("--stage-id")
    parser.add_argument("--stage-attempt", type=int, default=1)
    parser.add_argument("--provider-execution-json")
    parser.add_argument(
        "--reviewer-model",
        default=(
            os.getenv("TMCRA_WRITER_REVIEWER_MODEL")
            or os.getenv("TMCRA_DEEPSEEK_PRO_MODEL")
            or "deepseek-v4-pro"
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--recovery-mode",
        choices=sorted(WRITER_RECOVERY_MODES),
        default="none",
    )
    parser.add_argument(
        "--recover-interrupted-api-calls",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    tenant_id = _identity(args.tenant_id, "TMCRA_SERVICE_TENANT_ID")
    scope_name = _identity(
        args.scope_name, "TMCRA_SERVICE_SCOPE_NAME", "TMCRA_SERVICE_SCOPE"
    )
    job_id = _identity(args.job_id, "TMCRA_SERVICE_JOB_ID")
    stage_id = _identity(args.stage_id, "TMCRA_SERVICE_STAGE_ID")
    attribution_raw = str(os.getenv("TMCRA_USAGE_ATTRIBUTION_JSON") or "").strip()
    try:
        attribution_value = json.loads(attribution_raw) if attribution_raw else None
    except json.JSONDecodeError as exc:
        raise ProductionWriterError("usage attribution environment is invalid") from exc
    if attribution_value is not None and not isinstance(attribution_value, Mapping):
        raise ProductionWriterError("usage attribution environment must be an object")
    usage_attribution = UsageAttribution.from_mapping(attribution_value)
    try:
        provider_execution_value = (
            json.loads(args.provider_execution_json)
            if args.provider_execution_json
            else None
        )
    except json.JSONDecodeError as exc:
        raise ProductionWriterError(
            "provider execution argument is invalid JSON"
        ) from exc
    if provider_execution_value is not None and not isinstance(
        provider_execution_value, Mapping
    ):
        raise ProductionWriterError("provider execution argument must be an object")
    recovery_mode = args.recovery_mode
    if args.recover_interrupted_api_calls:
        if recovery_mode != "none":
            parser.error(
                "--recover-interrupted-api-calls cannot be combined with --recovery-mode"
            )
        recovery_mode = "definitive_provider_failure"
    execute_writer(
        input_path=args.input,
        out_dir=args.out_dir,
        database=args.database,
        operation_id=args.operation_id,
        repo=args.repo,
        tenant_id=tenant_id,
        scope_name=scope_name,
        job_id=job_id,
        stage_id=stage_id,
        stage_attempt=args.stage_attempt,
        reviewer_model=args.reviewer_model,
        timeout_seconds=args.timeout_seconds,
        max_tokens=args.max_tokens,
        recovery_mode=recovery_mode,
        usage_attribution=usage_attribution,
        provider_execution=provider_execution_value,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
