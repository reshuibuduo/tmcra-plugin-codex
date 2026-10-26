"""Commercial control-plane contracts for the TMCRA memory service."""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import secrets
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .control_db import ControlDB
from .jobs import (
    CANCELLED,
    FAILED,
    PENDING,
    RUNNING,
    SUCCEEDED,
    Job,
    JobStateError,
    JobStore,
)


WEBHOOK_EVENTS = frozenset(
    {
        "job.succeeded",
        "job.failed",
        "job.cancelled",
        "ingest.completed",
        "consolidation.completed",
        "index.completed",
        "export.ready",
        "scope.deleted",
    }
)

AUTO_RECOVERABLE_QUARANTINE_REASONS = frozenset(
    {
        "legacy_source_journal_integrity_unverified",
        "legacy_writer_journal_failures_unresolved",
        "source_control_watermark_divergence",
        "writer_journal_failures_unresolved",
    }
)
QUARANTINE_RECOVERY_ACTIVE_STATES = frozenset(
    {"waiting", "auditing", "repairing", "consolidating", "indexing", "verifying"}
)


class CommercialContractError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WebhookDelivery:
    delivery_id: str
    endpoint_id: str
    event_id: str
    tenant_id: str
    url: str
    event_type: str
    payload: dict[str, Any]
    attempt_count: int


def validate_webhook_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() != "https":
        raise CommercialContractError("invalid_webhook_url", "webhook URL must use HTTPS")
    if not parsed.hostname or parsed.username or parsed.password:
        raise CommercialContractError("invalid_webhook_url", "webhook URL has an invalid authority")
    if parsed.fragment:
        raise CommercialContractError("invalid_webhook_url", "webhook URL must not include a fragment")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        raise CommercialContractError("unsafe_webhook_target", "local webhook targets are not allowed")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise CommercialContractError("unsafe_webhook_target", "private webhook targets are not allowed")
    return url.strip()


def _assert_public_target(url: str) -> None:
    parsed = urlsplit(validate_webhook_url(url))
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise CommercialContractError("webhook_dns_failed", "webhook hostname could not be resolved") from exc
    if not addresses:
        raise CommercialContractError("webhook_dns_failed", "webhook hostname has no addresses")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise CommercialContractError("unsafe_webhook_target", "webhook resolved to a private address")


class CommercialControl:
    def __init__(self, database: ControlDB, *, webhook_signing_key: str | None = None) -> None:
        self.database = database
        self.webhook_signing_key = webhook_signing_key

    def scope_lifecycle(self, tenant_id: str, scope_name: str) -> dict[str, Any] | None:
        self.database._validate_scope(tenant_id, scope_name)
        with self.database.transaction(immediate=False) as connection:
            row = connection.execute(
                "SELECT * FROM scope_lifecycle WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
        return None if row is None else dict(row)

    def scope_quarantine(self, tenant_id: str, scope_name: str) -> dict[str, Any] | None:
        self.database._validate_scope(tenant_id, scope_name)
        with self.database.transaction(immediate=False) as connection:
            row = connection.execute(
                "SELECT * FROM scope_quarantines WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
        return None if row is None else dict(row)

    @staticmethod
    def _ready_recovery_status() -> dict[str, Any]:
        return {
            "state": "ready",
            "phase": "ready",
            "progress_percent": 100,
            "completed_items": 0,
            "total_items": 0,
            "pending_items": 0,
            "recovery_attempts": 0,
            "automatic": True,
            "reads_available": True,
            "writes_available": True,
            "requires_support": False,
            "started_at": None,
            "updated_at": None,
            "next_attempt_at": None,
        }

    @staticmethod
    def _nonnegative_int(value: Any, default: int = 0) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            return max(0, int(default))

    @staticmethod
    def _public_recovery_status(row: Mapping[str, Any]) -> dict[str, Any]:
        try:
            decoded = json.loads(str(row.get("report_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = {}
        report = decoded if isinstance(decoded, Mapping) else {}
        internal_state = str(row.get("recovery_state") or "waiting")
        requested_phase = str(report.get("phase") or "")
        allowed_phases = {
            "waiting",
            "auditing",
            "repairing",
            "consolidating",
            "indexing",
            "verifying",
            "manual_review",
        }
        phase = (
            internal_state
            if internal_state in {"waiting", "auditing", "manual_review"}
            else requested_phase
            if requested_phase in allowed_phases
            else internal_state
        )
        if phase not in allowed_phases:
            phase = "waiting"

        source_total = CommercialControl._nonnegative_int(report.get("source_count"))
        registered_total = CommercialControl._nonnegative_int(
            report.get("registered_message_count")
        )
        if registered_total > source_total:
            source_total = registered_total
        source_complete = min(
            source_total,
            CommercialControl._nonnegative_int(report.get("enriched_source_count")),
        )
        source_pending = max(0, source_total - source_complete)
        source_event_seq = CommercialControl._nonnegative_int(
            report.get("source_event_seq")
        )
        promoted_event_seq = min(
            source_event_seq,
            CommercialControl._nonnegative_int(report.get("promoted_event_seq")),
        )
        searchable_event_seq = min(
            source_event_seq,
            CommercialControl._nonnegative_int(report.get("searchable_event_seq")),
        )

        automatic = CommercialControl.quarantine_reason_supports_auto_recovery(
            str(row.get("reason") or "")
        )
        requires_support = internal_state == "manual_review" or not automatic
        if requires_support:
            phase = "manual_review"

        source_progress = (
            10 + int(70 * source_complete / source_total)
            if source_total > 0
            else 0
        )
        if phase == "waiting":
            progress = (
                max(2, source_progress)
                if source_total > 0
                else 2
            )
        elif phase == "auditing":
            progress = max(5, source_progress)
        else:
            progress = source_progress if source_total > 0 else 10
            if phase == "consolidating":
                slow_fraction = (
                    promoted_event_seq / source_event_seq
                    if source_event_seq > 0
                    else 0.0
                )
                progress = max(progress, 80 + int(10 * slow_fraction))
            elif phase == "indexing":
                index_fraction = (
                    searchable_event_seq / source_event_seq
                    if source_event_seq > 0
                    else 0.0
                )
                progress = max(progress, 90 + int(5 * index_fraction))
            elif phase == "verifying":
                progress = max(progress, 97)
        progress = max(1, min(99, int(progress)))
        return {
            "state": "attention_required" if requires_support else "recovering",
            "phase": phase,
            "progress_percent": progress,
            "completed_items": source_complete,
            "total_items": source_total,
            "pending_items": source_pending,
            "recovery_attempts": CommercialControl._nonnegative_int(
                row.get("resumed_job_count")
            ),
            "automatic": automatic,
            # The current quarantine contract is fail closed for both reads and
            # writes until the final consistency audit succeeds.
            "reads_available": False,
            "writes_available": False,
            "requires_support": requires_support,
            "started_at": float(row["quarantined_at"]),
            "updated_at": float(
                row.get("recovery_updated_at") or row.get("quarantine_updated_at")
            ),
            "next_attempt_at": (
                None
                if requires_support or row.get("next_attempt_at") is None
                else float(row["next_attempt_at"])
            ),
        }

    def scope_recovery_statuses(
        self, tenant_id: str, scope_names: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        names = {str(value).strip() for value in scope_names if str(value).strip()}
        for scope_name in names:
            self.database._validate_scope(tenant_id, scope_name)
        statuses = {name: self._ready_recovery_status() for name in names}
        if not names:
            return statuses
        with self.database.transaction(immediate=False) as connection:
            rows = connection.execute(
                """
                SELECT quarantine.scope_name,quarantine.reason,
                       quarantine.quarantined_at,
                       quarantine.updated_at AS quarantine_updated_at,
                       recovery.state AS recovery_state,
                       recovery.resumed_job_count,recovery.next_attempt_at,
                       recovery.report_json,
                       recovery.updated_at AS recovery_updated_at
                FROM scope_quarantines AS quarantine
                LEFT JOIN scope_quarantine_recoveries AS recovery
                  ON recovery.tenant_id=quarantine.tenant_id
                 AND recovery.scope_name=quarantine.scope_name
                WHERE quarantine.tenant_id=?
                """,
                (tenant_id,),
            ).fetchall()
        for row in rows:
            scope_name = str(row["scope_name"])
            if scope_name in names:
                statuses[scope_name] = self._public_recovery_status(dict(row))
        return statuses

    def scope_recovery_status(
        self, tenant_id: str, scope_name: str
    ) -> dict[str, Any]:
        self.database._validate_scope(tenant_id, scope_name)
        return self.scope_recovery_statuses(tenant_id, (scope_name,))[scope_name]

    def _cancel_blocked_pending_jobs(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        reason_code: str,
        allowed_job_types: frozenset[str] = frozenset(
            {"export_scope", "delete_scope"}
        ),
    ) -> None:
        """Make a fail-closed scope executable for export or deletion.

        Jobs admitted before an administrative quarantine must not remain ahead
        of the control job forever. Cancellation preserves the job audit trail;
        a repaired scope requires explicit re-admission with a new operation.
        """

        with self.database.transaction(immediate=False) as connection:
            rows = connection.execute(
                "SELECT job_id,payload_json FROM jobs "
                "WHERE tenant_id=? AND scope_name=? AND state=? "
                "ORDER BY scope_seq,job_id",
                (tenant_id, scope_name, PENDING),
            ).fetchall()
        store = JobStore(self.database)
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            if str(payload.get("job_type") or "") in allowed_job_types:
                continue
            try:
                store.cancel(
                    str(row["job_id"]),
                    reason={
                        "code": reason_code,
                        "effect_state": "no_side_effects",
                    },
                )
            except JobStateError:
                # A worker that won the race still hits require_scope_active
                # before executing any storage mutation.
                continue

    def require_scope_active(self, tenant_id: str, scope_name: str) -> None:
        if self.scope_quarantine(tenant_id, scope_name) is not None:
            raise CommercialContractError(
                "scope_quarantined", "scope is quarantined"
            )
        lifecycle = self.scope_lifecycle(tenant_id, scope_name)
        if lifecycle and lifecycle["state"] != "active":
            raise CommercialContractError(
                f"scope_{lifecycle['state']}",
                f"scope is {lifecycle['state']}",
            )
        deletion = self.active_content_deletion(tenant_id, scope_name)
        if deletion is not None:
            raise CommercialContractError(
                "scope_content_deleting",
                "scope content deletion is in progress",
            )

    def require_scope_readable(self, tenant_id: str, scope_name: str) -> dict[str, Any] | None:
        """Allow only a verified stale read during automatic recovery.

        This is deliberately separate from ``require_scope_active``. Callers
        using this gate must validate the committed snapshot before serving a
        read; no mutation path may use it.
        """

        lifecycle = self.scope_lifecycle(tenant_id, scope_name)
        if lifecycle and lifecycle["state"] != "active":
            raise CommercialContractError(
                f"scope_{lifecycle['state']}",
                f"scope is {lifecycle['state']}",
            )
        if self.active_content_deletion(tenant_id, scope_name) is not None:
            raise CommercialContractError(
                "scope_content_deleting",
                "scope content deletion is in progress",
            )
        quarantine = self.scope_quarantine(tenant_id, scope_name)
        if quarantine is None:
            return None

        with self.database.transaction(immediate=False) as connection:
            recovery_row = connection.execute(
                "SELECT state FROM scope_quarantine_recoveries "
                "WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
        if recovery_row is None:
            raise CommercialContractError(
                "scope_quarantined", "scope is quarantined"
            )

        recovery = self.scope_recovery_status(tenant_id, scope_name)
        if (
            not self.quarantine_reason_supports_auto_recovery(
                str(quarantine.get("reason") or "")
            )
            or recovery["state"] != "recovering"
            or recovery["phase"] not in QUARANTINE_RECOVERY_ACTIVE_STATES
            or recovery["requires_support"]
            or recovery["phase"] == "manual_review"
        ):
            raise CommercialContractError(
                "scope_quarantined", "scope is quarantined"
            )
        return recovery

    @staticmethod
    def _decode_content_deletion(row: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(row)
        raw_result = value.pop("result_json", None)
        try:
            decoded = json.loads(str(raw_result)) if raw_result else None
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = None
        value["result"] = decoded if isinstance(decoded, dict) else None
        return value

    def active_content_deletion(
        self, tenant_id: str, scope_name: str
    ) -> dict[str, Any] | None:
        self.database._validate_scope(tenant_id, scope_name)
        with self.database.transaction(immediate=False) as connection:
            row = connection.execute(
                "SELECT * FROM content_deletions "
                "WHERE tenant_id=? AND scope_name=? "
                "AND state IN ('requested','purging','reindexing','failed') "
                "ORDER BY created_at LIMIT 1",
                (tenant_id, scope_name),
            ).fetchone()
        return None if row is None else self._decode_content_deletion(row)

    def content_deletion(
        self, tenant_id: str, scope_name: str, deletion_id: str
    ) -> dict[str, Any] | None:
        self.database._validate_scope(tenant_id, scope_name)
        with self.database.transaction(immediate=False) as connection:
            row = connection.execute(
                "SELECT * FROM content_deletions "
                "WHERE tenant_id=? AND scope_name=? AND deletion_id=?",
                (tenant_id, scope_name, deletion_id),
            ).fetchone()
        return None if row is None else self._decode_content_deletion(row)

    def cancel_jobs_for_content_deletion(
        self, tenant_id: str, scope_name: str
    ) -> None:
        self._cancel_blocked_pending_jobs(
            tenant_id,
            scope_name,
            allowed_job_types=frozenset(
                {"delete_memories", "delete_session", "delete_scope"}
            ),
            reason_code="scope_content_deleting_before_start",
        )

    def register_content_deletion_in_transaction(
        self,
        connection: Any,
        tenant_id: str,
        scope_name: str,
        *,
        deletion_id: str,
        job_id: str,
        mode: str,
        target_sha256: str,
        target_count: int,
    ) -> None:
        self.database._validate_scope(tenant_id, scope_name)
        if (
            mode not in {"memory_ids", "session"}
            or target_count < 1
            or not deletion_id
            or not job_id
            or len(target_sha256) != 64
        ):
            raise ValueError("invalid content deletion request")
        lifecycle = connection.execute(
            "SELECT state FROM scope_lifecycle WHERE tenant_id=? AND scope_name=?",
            (tenant_id, scope_name),
        ).fetchone()
        if lifecycle is not None and str(lifecycle["state"]) != "active":
            state = str(lifecycle["state"])
            raise CommercialContractError(f"scope_{state}", f"scope is {state}")
        active = connection.execute(
            "SELECT deletion_id FROM content_deletions "
            "WHERE tenant_id=? AND scope_name=? "
            "AND state IN ('requested','purging','reindexing','failed') LIMIT 1",
            (tenant_id, scope_name),
        ).fetchone()
        if active is not None:
            raise CommercialContractError(
                "scope_content_deleting",
                "scope already has a content deletion in progress",
            )
        now = time.time()
        connection.execute(
            """
            INSERT INTO content_deletions(
                deletion_id,tenant_id,scope_name,mode,target_sha256,target_count,
                job_id,state,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,'requested',?,?)
            """,
            (
                deletion_id,
                tenant_id,
                scope_name,
                mode,
                target_sha256,
                target_count,
                job_id,
                now,
                now,
            ),
        )

    def update_content_deletion(
        self,
        tenant_id: str,
        scope_name: str,
        deletion_id: str,
        job_id: str,
        *,
        state: str,
        result: Mapping[str, Any] | None = None,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        if state not in {"purging", "reindexing", "completed", "failed"}:
            raise ValueError("invalid content deletion state")
        now = time.time()
        result_json = (
            None if result is None else self.database.encode_json(dict(result))
        )
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE content_deletions
                SET state=?,result_json=?,error_code=?,updated_at=?,
                    completed_at=CASE WHEN ?='completed' THEN ? ELSE NULL END
                WHERE deletion_id=? AND tenant_id=? AND scope_name=? AND job_id=?
                """,
                (
                    state,
                    result_json,
                    error_code,
                    now,
                    state,
                    now,
                    deletion_id,
                    tenant_id,
                    scope_name,
                    job_id,
                ),
            )
            if cursor.rowcount != 1:
                raise CommercialContractError(
                    "content_deletion_missing", "content deletion was not registered"
                )
            row = connection.execute(
                "SELECT * FROM content_deletions WHERE deletion_id=?",
                (deletion_id,),
            ).fetchone()
        return self._decode_content_deletion(row)

    def resume_content_deletion(
        self, tenant_id: str, scope_name: str, deletion_id: str, job_id: str
    ) -> dict[str, Any]:
        now = time.time()
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE content_deletions
                SET state='requested',error_code=NULL,updated_at=?,completed_at=NULL
                WHERE deletion_id=? AND tenant_id=? AND scope_name=? AND job_id=?
                  AND state='failed'
                """,
                (now, deletion_id, tenant_id, scope_name, job_id),
            )
            if cursor.rowcount != 1:
                raise CommercialContractError(
                    "content_deletion_not_retryable",
                    "content deletion is not in a retryable state",
                )
            row = connection.execute(
                "SELECT * FROM content_deletions WHERE deletion_id=?",
                (deletion_id,),
            ).fetchone()
        return self._decode_content_deletion(row)

    def apply_content_deletion_control_cleanup(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        deleted_source_record_ids: Iterable[str],
        deleted_session_message_counts: Mapping[str, int],
        deleted_session_id: str | None = None,
    ) -> None:
        """Invalidate control-plane projections that can contain deleted content."""

        source_ids = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in deleted_source_record_ids
                if str(value).strip()
            )
        )
        session_counts = {
            str(key).strip(): max(0, int(value))
            for key, value in deleted_session_message_counts.items()
            if str(key).strip() and int(value) > 0
        }
        clean_session = str(deleted_session_id or "").strip()
        now = time.time()
        with self.database.transaction() as connection:
            # Provider-task rows contain bounded copies of model prompts and
            # parsed outputs. Any content deletion invalidates those recovery
            # artifacts for the scope, so purge them with the memory content.
            connection.execute(
                "DELETE FROM user_provider_tasks WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            removed_raw_tokens = 0
            removed_user_turns = 0
            if source_ids:
                placeholders = ",".join("?" for _ in source_ids)
                removed = connection.execute(
                    f"SELECT COALESCE(SUM(raw_token_estimate),0) AS raw_tokens,"
                    f"COALESCE(SUM(user_turns),0) AS user_turns "
                    f"FROM scope_source_event_commits "
                    f"WHERE tenant_id=? AND scope_name=? "
                    f"AND source_record_id IN ({placeholders})",
                    (tenant_id, scope_name, *source_ids),
                ).fetchone()
                removed_raw_tokens = int(removed["raw_tokens"] or 0)
                removed_user_turns = int(removed["user_turns"] or 0)
                connection.execute(
                    f"DELETE FROM scope_source_event_commits "
                    f"WHERE tenant_id=? AND scope_name=? "
                    f"AND source_record_id IN ({placeholders})",
                    (tenant_id, scope_name, *source_ids),
                )
                operations = connection.execute(
                    "SELECT operation_id FROM scope_ingest_source_sets "
                    "WHERE tenant_id=? AND scope_name=?",
                    (tenant_id, scope_name),
                ).fetchall()
                for operation in operations:
                    operation_id = str(operation["operation_id"])
                    remaining_rows = connection.execute(
                        """
                        SELECT source_record_id,origin_operation_id,
                               raw_token_estimate,user_turns
                        FROM scope_source_event_commits
                        WHERE tenant_id=? AND scope_name=?
                          AND accounting_operation_id=?
                        ORDER BY source_record_id
                        """,
                        (tenant_id, scope_name, operation_id),
                    ).fetchall()
                    remaining = [
                        {
                            "source_record_id": str(row["source_record_id"]),
                            "origin_operation_id": str(row["origin_operation_id"]),
                            "raw_token_estimate": int(row["raw_token_estimate"] or 0),
                            "user_turns": int(row["user_turns"] or 0),
                        }
                        for row in remaining_rows
                    ]
                    if not remaining:
                        connection.execute(
                            "DELETE FROM scope_ingest_source_sets "
                            "WHERE tenant_id=? AND scope_name=? AND operation_id=?",
                            (tenant_id, scope_name, operation_id),
                        )
                        connection.execute(
                            "DELETE FROM scope_ingest_watermark_commits "
                            "WHERE tenant_id=? AND scope_name=? AND operation_id=?",
                            (tenant_id, scope_name, operation_id),
                        )
                        continue
                    encoded = json.dumps(
                        remaining,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                    connection.execute(
                        """
                        UPDATE scope_ingest_source_sets
                        SET source_set_sha256=?,source_count=?
                        WHERE tenant_id=? AND scope_name=? AND operation_id=?
                        """,
                        (
                            hashlib.sha256(encoded).hexdigest(),
                            len(remaining),
                            tenant_id,
                            scope_name,
                            operation_id,
                        ),
                    )
                    connection.execute(
                        """
                        UPDATE scope_ingest_watermark_commits
                        SET new_message_count=?,raw_token_estimate=?,user_turns=?
                        WHERE tenant_id=? AND scope_name=? AND operation_id=?
                        """,
                        (
                            len(remaining),
                            sum(item["raw_token_estimate"] for item in remaining),
                            sum(item["user_turns"] for item in remaining),
                            tenant_id,
                            scope_name,
                            operation_id,
                        ),
                    )
            removed_source_count = len(source_ids)
            if removed_source_count:
                connection.execute(
                    """
                    UPDATE scope_evolution_state
                    SET source_event_seq=MAX(0,source_event_seq-?),
                        promoted_event_seq=MIN(
                            promoted_event_seq,MAX(0,source_event_seq-?)
                        ),
                        indexed_event_seq=MIN(
                            indexed_event_seq,MAX(0,source_event_seq-?)
                        ),
                        delta_indexed_event_seq=MIN(
                            delta_indexed_event_seq,MAX(0,source_event_seq-?)
                        ),
                        source_raw_token_estimate=MAX(
                            0,source_raw_token_estimate-?
                        ),
                        promoted_raw_token_estimate=MIN(
                            promoted_raw_token_estimate,
                            MAX(0,source_raw_token_estimate-?)
                        ),
                        source_user_turns=MAX(0,source_user_turns-?),
                        promoted_user_turns=MIN(
                            promoted_user_turns,MAX(0,source_user_turns-?)
                        ),
                        dirty_since_at=NULL,index_dirty_since_at=NULL,
                        active_evolution_job_id=NULL,
                        active_evolution_job_version=NULL,
                        active_index_job_id=NULL,active_index_job_version=NULL,
                        updated_at=?
                    WHERE tenant_id=? AND scope_name=?
                    """,
                    (
                        removed_source_count,
                        removed_source_count,
                        removed_source_count,
                        removed_source_count,
                        removed_raw_tokens,
                        removed_raw_tokens,
                        removed_user_turns,
                        removed_user_turns,
                        now,
                        tenant_id,
                        scope_name,
                    ),
                )
            connection.execute(
                "DELETE FROM memory_graph_views WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                "DELETE FROM memory_graph_refresh_queue "
                "WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            if clean_session:
                session_row = connection.execute(
                    "SELECT message_count FROM scope_sessions "
                    "WHERE tenant_id=? AND scope_name=? AND session_id=?",
                    (tenant_id, scope_name, clean_session),
                ).fetchone()
                catalog_removed_count = (
                    int(session_row["message_count"] or 0)
                    if session_row is not None
                    else sum(session_counts.values())
                )
                connection.execute(
                    "DELETE FROM session_graph_metadata "
                    "WHERE tenant_id=? AND scope_name=? AND session_id=?",
                    (tenant_id, scope_name, clean_session),
                )
                connection.execute(
                    "DELETE FROM scope_sessions "
                    "WHERE tenant_id=? AND scope_name=? AND session_id=?",
                    (tenant_id, scope_name, clean_session),
                )
            else:
                catalog_removed_count = sum(session_counts.values())
                for session_id, count in session_counts.items():
                    connection.execute(
                        """
                        UPDATE scope_sessions
                        SET message_count=MAX(0,message_count-?),last_ingest_at=?
                        WHERE tenant_id=? AND scope_name=? AND session_id=?
                        """,
                        (count, now, tenant_id, scope_name, session_id),
                    )
            if catalog_removed_count:
                connection.execute(
                    """
                    UPDATE scope_catalog
                    SET message_count=MAX(0,message_count-?),last_seen_at=?
                    WHERE tenant_id=? AND scope_name=?
                    """,
                    (catalog_removed_count, now, tenant_id, scope_name),
                )

    def quarantine_scope(
        self, tenant_id: str, scope_name: str, *, reason: str
    ) -> dict[str, Any]:
        """Fail closed for an artifact set that cannot yet be proven consistent."""

        self.database._validate_scope(tenant_id, scope_name)
        reason = str(reason or "").strip()
        if not reason or len(reason) > 500:
            raise CommercialContractError(
                "invalid_quarantine_reason",
                "scope quarantine requires a bounded reason",
            )
        automatic_recovery = self.quarantine_reason_supports_auto_recovery(reason)
        initial_recovery_state = "waiting" if automatic_recovery else "manual_review"
        now = time.time()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM scope_lifecycle WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            if row is not None and str(row["state"]) in {"deleting", "deleted"}:
                raise CommercialContractError(
                    f"scope_{row['state']}",
                    f"scope is {row['state']}",
                )
            connection.execute(
                """
                INSERT INTO scope_quarantines(
                    tenant_id,scope_name,reason,quarantined_at,updated_at
                ) VALUES(?,?,?,?,?)
                ON CONFLICT(tenant_id,scope_name) DO UPDATE SET
                    reason=excluded.reason, updated_at=excluded.updated_at
                """,
                (tenant_id, scope_name, reason, now, now),
            )
            value = connection.execute(
                "SELECT * FROM scope_quarantines WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            prior_recovery = connection.execute(
                "SELECT quarantine_started_at FROM scope_quarantine_recoveries "
                "WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            new_recovery_generation = bool(
                prior_recovery is None
                or float(prior_recovery["quarantine_started_at"])
                != float(value["quarantined_at"])
            )
            if new_recovery_generation:
                connection.execute(
                    "DELETE FROM scope_quarantine_recovery_jobs "
                    "WHERE tenant_id=? AND scope_name=?",
                    (tenant_id, scope_name),
                )
            connection.execute(
                """
                INSERT INTO scope_quarantine_recoveries(
                    tenant_id,scope_name,quarantine_started_at,state,
                    next_attempt_at,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(tenant_id,scope_name) DO UPDATE SET
                    quarantine_started_at=CASE
                        WHEN scope_quarantine_recoveries.quarantine_started_at
                             <> excluded.quarantine_started_at
                        THEN excluded.quarantine_started_at
                        ELSE scope_quarantine_recoveries.quarantine_started_at
                    END,
                    state=CASE
                        WHEN scope_quarantine_recoveries.quarantine_started_at
                             <> excluded.quarantine_started_at
                        THEN excluded.state
                        ELSE scope_quarantine_recoveries.state
                    END,
                    cycle_count=CASE
                        WHEN scope_quarantine_recoveries.quarantine_started_at
                             <> excluded.quarantine_started_at
                        THEN 0 ELSE scope_quarantine_recoveries.cycle_count END,
                    resumed_job_count=CASE
                        WHEN scope_quarantine_recoveries.quarantine_started_at
                             <> excluded.quarantine_started_at
                        THEN 0 ELSE scope_quarantine_recoveries.resumed_job_count END,
                    active_job_id=CASE
                        WHEN scope_quarantine_recoveries.quarantine_started_at
                             <> excluded.quarantine_started_at
                        THEN NULL ELSE scope_quarantine_recoveries.active_job_id END,
                    next_attempt_at=CASE
                        WHEN scope_quarantine_recoveries.quarantine_started_at
                             <> excluded.quarantine_started_at
                        THEN excluded.next_attempt_at
                        ELSE scope_quarantine_recoveries.next_attempt_at
                    END,
                    lease_owner=CASE
                        WHEN scope_quarantine_recoveries.quarantine_started_at
                             <> excluded.quarantine_started_at
                        THEN NULL ELSE scope_quarantine_recoveries.lease_owner END,
                    lease_expires_at=CASE
                        WHEN scope_quarantine_recoveries.quarantine_started_at
                             <> excluded.quarantine_started_at
                        THEN NULL ELSE scope_quarantine_recoveries.lease_expires_at END,
                    last_error_code=CASE
                        WHEN scope_quarantine_recoveries.quarantine_started_at
                             <> excluded.quarantine_started_at
                        THEN NULL ELSE scope_quarantine_recoveries.last_error_code END,
                    report_json=CASE
                        WHEN scope_quarantine_recoveries.quarantine_started_at
                             <> excluded.quarantine_started_at
                        THEN '{}' ELSE scope_quarantine_recoveries.report_json END,
                    recovered_at=CASE
                        WHEN scope_quarantine_recoveries.quarantine_started_at
                             <> excluded.quarantine_started_at
                        THEN NULL ELSE scope_quarantine_recoveries.recovered_at END,
                    updated_at=excluded.updated_at
                """,
                (
                    tenant_id,
                    scope_name,
                    float(value["quarantined_at"]),
                    initial_recovery_state,
                    now,
                    now,
                    now,
                ),
            )
            if not automatic_recovery:
                connection.execute(
                    """
                    UPDATE scope_quarantine_recoveries
                    SET state='manual_review',active_job_id=NULL,
                        lease_owner=NULL,lease_expires_at=NULL,
                        next_attempt_at=?,
                        last_error_code='manual_integrity_review_required',
                        report_json=?,updated_at=?
                    WHERE tenant_id=? AND scope_name=?
                    """,
                    (
                        now,
                        self.database.encode_json({"phase": "manual_review"}),
                        now,
                        tenant_id,
                        scope_name,
                    ),
                )
        if not automatic_recovery:
            self._cancel_blocked_pending_jobs(
                tenant_id,
                scope_name,
                reason_code="scope_quarantined_before_start",
            )
        return {**dict(value), "state": "quarantined"}

    def clear_scope_quarantine(self, tenant_id: str, scope_name: str) -> bool:
        """Re-enable a scope only after an explicit external integrity audit."""

        self.database._validate_scope(tenant_id, scope_name)
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM scope_quarantines WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            if cursor.rowcount == 1:
                now = time.time()
                connection.execute(
                    "UPDATE scope_quarantine_recoveries SET state='recovered',"
                    "active_job_id=NULL,lease_owner=NULL,lease_expires_at=NULL,"
                    "last_error_code='cleared_after_external_audit',"
                    "updated_at=?,recovered_at=? WHERE tenant_id=? AND scope_name=?",
                    (now, now, tenant_id, scope_name),
                )
        return cursor.rowcount == 1

    @staticmethod
    def quarantine_reason_supports_auto_recovery(reason: str) -> bool:
        code = str(reason or "").strip().split(":", 1)[0]
        return code in AUTO_RECOVERABLE_QUARANTINE_REASONS

    def claim_due_quarantine_recovery(
        self,
        owner: str,
        *,
        now: float | None = None,
        lease_seconds: float = 120.0,
    ) -> dict[str, Any] | None:
        """Lease one due recovery cycle without weakening the quarantine."""

        owner = str(owner or "").strip()
        if not owner or lease_seconds <= 0:
            raise ValueError("recovery owner and positive lease_seconds are required")
        moment = time.time() if now is None else float(now)
        with self.database.transaction() as connection:
            candidates = connection.execute(
                """
                SELECT quarantine.tenant_id,quarantine.scope_name,
                       quarantine.reason,quarantine.quarantined_at
                FROM scope_quarantines AS quarantine
                LEFT JOIN scope_lifecycle AS lifecycle
                  ON lifecycle.tenant_id=quarantine.tenant_id
                 AND lifecycle.scope_name=quarantine.scope_name
                LEFT JOIN scope_quarantine_recoveries AS recovery
                  ON recovery.tenant_id=quarantine.tenant_id
                 AND recovery.scope_name=quarantine.scope_name
                WHERE (lifecycle.state IS NULL OR lifecycle.state='active')
                  AND (
                    recovery.tenant_id IS NULL
                    OR (
                      recovery.state IN ('waiting','auditing','repairing','verifying')
                      AND recovery.next_attempt_at<=?
                      AND NOT (
                        recovery.state='repairing'
                        AND EXISTS (
                          SELECT 1
                          FROM scope_quarantine_recovery_jobs AS recovery_job
                          JOIN jobs AS recovery_job_record
                            ON recovery_job_record.job_id=recovery_job.job_id
                          WHERE recovery_job.tenant_id=quarantine.tenant_id
                            AND recovery_job.scope_name=quarantine.scope_name
                            AND recovery_job.state IN ('pending','running')
                            AND recovery_job_record.state IN ('pending','running')
                        )
                        AND NOT EXISTS (
                          SELECT 1
                          FROM jobs AS unmapped_active_job
                          LEFT JOIN scope_quarantine_recovery_jobs AS unmapped_mapping
                            ON unmapped_mapping.tenant_id=quarantine.tenant_id
                           AND unmapped_mapping.scope_name=quarantine.scope_name
                           AND unmapped_mapping.job_id=unmapped_active_job.job_id
                          WHERE unmapped_active_job.tenant_id=quarantine.tenant_id
                            AND unmapped_active_job.scope_name=quarantine.scope_name
                            AND unmapped_active_job.state IN ('pending','running')
                            AND unmapped_mapping.job_id IS NULL
                        )
                      )
                      AND (
                        recovery.lease_owner IS NULL
                        OR recovery.lease_expires_at IS NULL
                        OR recovery.lease_expires_at<=?
                      )
                    )
                  )
                ORDER BY quarantine.quarantined_at,
                         quarantine.tenant_id,quarantine.scope_name
                """,
                (moment, moment),
            ).fetchall()
            for candidate in candidates:
                tenant_id = str(candidate["tenant_id"])
                scope_name = str(candidate["scope_name"])
                if not self.quarantine_reason_supports_auto_recovery(
                    str(candidate["reason"] or "")
                ):
                    continue
                quarantined_at = float(candidate["quarantined_at"])
                connection.execute(
                    """
                    INSERT INTO scope_quarantine_recoveries(
                        tenant_id,scope_name,quarantine_started_at,state,
                        next_attempt_at,created_at,updated_at
                    ) VALUES(?,?,?,'waiting',?,?,?)
                    ON CONFLICT(tenant_id,scope_name) DO NOTHING
                    """,
                    (
                        tenant_id,
                        scope_name,
                        quarantined_at,
                        moment,
                        moment,
                        moment,
                    ),
                )
                updated = connection.execute(
                    """
                    UPDATE scope_quarantine_recoveries
                    SET state='auditing',cycle_count=cycle_count+1,
                        lease_owner=?,lease_expires_at=?,updated_at=?
                    WHERE tenant_id=? AND scope_name=?
                      AND quarantine_started_at=?
                      AND state IN ('waiting','auditing','repairing','verifying')
                      AND next_attempt_at<=?
                      AND NOT (
                        state='repairing'
                        AND EXISTS (
                          SELECT 1
                          FROM scope_quarantine_recovery_jobs AS recovery_job
                          JOIN jobs AS recovery_job_record
                            ON recovery_job_record.job_id=recovery_job.job_id
                          WHERE recovery_job.tenant_id=
                                scope_quarantine_recoveries.tenant_id
                            AND recovery_job.scope_name=
                                scope_quarantine_recoveries.scope_name
                            AND recovery_job.state IN ('pending','running')
                            AND recovery_job_record.state IN ('pending','running')
                        )
                        AND NOT EXISTS (
                          SELECT 1
                          FROM jobs AS unmapped_active_job
                          LEFT JOIN scope_quarantine_recovery_jobs AS unmapped_mapping
                            ON unmapped_mapping.tenant_id=
                               scope_quarantine_recoveries.tenant_id
                           AND unmapped_mapping.scope_name=
                               scope_quarantine_recoveries.scope_name
                           AND unmapped_mapping.job_id=unmapped_active_job.job_id
                          WHERE unmapped_active_job.tenant_id=
                                scope_quarantine_recoveries.tenant_id
                            AND unmapped_active_job.scope_name=
                                scope_quarantine_recoveries.scope_name
                            AND unmapped_active_job.state IN ('pending','running')
                            AND unmapped_mapping.job_id IS NULL
                        )
                      )
                      AND (
                        lease_owner IS NULL OR lease_expires_at IS NULL
                        OR lease_expires_at<=?
                      )
                    """,
                    (
                        owner,
                        moment + float(lease_seconds),
                        moment,
                        tenant_id,
                        scope_name,
                        quarantined_at,
                        moment,
                        moment,
                    ),
                ).rowcount
                if updated != 1:
                    continue
                row = connection.execute(
                    """
                    SELECT recovery.*,quarantine.reason
                    FROM scope_quarantine_recoveries AS recovery
                    JOIN scope_quarantines AS quarantine
                      ON quarantine.tenant_id=recovery.tenant_id
                     AND quarantine.scope_name=recovery.scope_name
                    WHERE recovery.tenant_id=? AND recovery.scope_name=?
                    """,
                    (tenant_id, scope_name),
                ).fetchone()
                return dict(row) if row is not None else None
        return None

    def manual_quarantine_recovery_candidates(
        self, *, limit: int = 16
    ) -> list[dict[str, Any]]:
        """Return auto-recoverable manual-review rows for a fresh audit.

        A manual-review state remains fail closed.  The worker may use this
        read-only inventory to prove that a process interruption has become
        resumable, then call ``reopen_quarantine_recovery_after_audit``.
        """

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be positive")
        with self.database.transaction(immediate=False) as connection:
            rows = connection.execute(
                """
                SELECT recovery.*,quarantine.reason,quarantine.quarantined_at
                FROM scope_quarantine_recoveries AS recovery
                JOIN scope_quarantines AS quarantine
                  ON quarantine.tenant_id=recovery.tenant_id
                 AND quarantine.scope_name=recovery.scope_name
                LEFT JOIN scope_lifecycle AS lifecycle
                  ON lifecycle.tenant_id=recovery.tenant_id
                 AND lifecycle.scope_name=recovery.scope_name
                WHERE recovery.state='manual_review'
                  AND recovery.quarantine_started_at=quarantine.quarantined_at
                  AND (lifecycle.state IS NULL OR lifecycle.state='active')
                ORDER BY recovery.updated_at,recovery.tenant_id,recovery.scope_name
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def reopen_quarantine_recovery_after_audit(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        expected_error_code: str,
        audit_report: Mapping[str, Any],
        now: float | None = None,
    ) -> bool:
        """Requeue one manual-review cycle after a new full integrity proof."""

        if not bool(audit_report.get("integrity_ok")):
            raise CommercialContractError(
                "quarantine_reaudit_failed",
                "manual recovery cannot reopen without a clean integrity audit",
            )
        moment = time.time() if now is None else float(now)
        with self.database.transaction() as connection:
            quarantine = connection.execute(
                "SELECT reason,quarantined_at FROM scope_quarantines "
                "WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            if quarantine is None or not self.quarantine_reason_supports_auto_recovery(
                str(quarantine["reason"] or "")
            ):
                return False
            updated = connection.execute(
                """
                UPDATE scope_quarantine_recoveries
                SET state='waiting',active_job_id=NULL,next_attempt_at=?,
                    lease_owner=NULL,lease_expires_at=NULL,
                    last_error_code='automatic_reaudit_verified',report_json=?,
                    updated_at=?
                WHERE tenant_id=? AND scope_name=? AND state='manual_review'
                  AND quarantine_started_at=?
                  AND COALESCE(last_error_code,'')=?
                """,
                (
                    moment,
                    self.database.encode_json(dict(audit_report)),
                    moment,
                    tenant_id,
                    scope_name,
                    float(quarantine["quarantined_at"]),
                    str(expected_error_code or ""),
                ),
            ).rowcount
        return updated == 1

    def request_quarantine_recovery_after_audit(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        audit_report: Mapping[str, Any],
        now: float | None = None,
    ) -> bool:
        """Wake an isolated scope after a user-requested, fully audited retry.

        The API may receive a retry while the automatic controller is in
        ``manual_review`` or sleeping in ``waiting`` backoff.  Moving that
        recovery to an immediately-due ``waiting`` cycle is what lets the
        worker register and execute the already-audited failed job.  An active
        recovery is left untouched so a concurrent retry request cannot steal
        its lease or clear its active job.
        """

        if not bool(audit_report.get("integrity_ok")):
            raise CommercialContractError(
                "quarantine_reaudit_failed",
                "manual recovery cannot restart without a clean integrity audit",
            )
        moment = time.time() if now is None else float(now)
        with self.database.transaction() as connection:
            quarantine = connection.execute(
                "SELECT reason,quarantined_at FROM scope_quarantines "
                "WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            recovery = connection.execute(
                "SELECT state,quarantine_started_at FROM scope_quarantine_recoveries "
                "WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            if (
                quarantine is None
                or recovery is None
                or not self.quarantine_reason_supports_auto_recovery(
                    str(quarantine["reason"] or "")
                )
                or float(recovery["quarantine_started_at"])
                != float(quarantine["quarantined_at"])
            ):
                return False
            recovery_state = str(recovery["state"])
            if recovery_state in {"auditing", "repairing", "verifying"}:
                return True
            if recovery_state not in {"manual_review", "waiting"}:
                return False
            updated = connection.execute(
                """
                UPDATE scope_quarantine_recoveries
                SET state='waiting',active_job_id=NULL,next_attempt_at=?,
                    lease_owner=NULL,lease_expires_at=NULL,
                    last_error_code='manual_retry_audit_verified',report_json=?,
                    updated_at=?
                WHERE tenant_id=? AND scope_name=?
                  AND state IN ('manual_review','waiting')
                  AND quarantine_started_at=?
                """,
                (
                    moment,
                    self.database.encode_json(dict(audit_report)),
                    moment,
                    tenant_id,
                    scope_name,
                    float(quarantine["quarantined_at"]),
                ),
            ).rowcount
        return updated == 1

    def finish_quarantine_recovery_cycle(
        self,
        tenant_id: str,
        scope_name: str,
        owner: str,
        *,
        state: str,
        next_attempt_at: float,
        error_code: str | None = None,
        report: Mapping[str, Any] | None = None,
        active_job_id: str | None = None,
    ) -> None:
        if state not in {
            "waiting",
            "repairing",
            "verifying",
            "manual_review",
        }:
            raise ValueError("invalid quarantine recovery state")
        now = time.time()
        with self.database.transaction() as connection:
            updated = connection.execute(
                """
                UPDATE scope_quarantine_recoveries
                SET state=?,active_job_id=?,next_attempt_at=?,
                    lease_owner=NULL,lease_expires_at=NULL,last_error_code=?,
                    report_json=?,updated_at=?
                WHERE tenant_id=? AND scope_name=? AND lease_owner=?
                """,
                (
                    state,
                    active_job_id,
                    float(next_attempt_at),
                    str(error_code or "") or None,
                    self.database.encode_json(dict(report or {})),
                    now,
                    tenant_id,
                    scope_name,
                    owner,
                ),
            ).rowcount
        if updated != 1:
            raise CommercialContractError(
                "quarantine_recovery_lease_lost",
                "quarantine recovery lease is no longer owned",
            )

    def publish_quarantine_recovery_job(
        self,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        owner: str,
        *,
        next_attempt_at: float,
        report: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically expose one preclaimed derived recovery job to workers."""

        now = time.time()
        with self.database.transaction() as connection:
            recovery_updated = connection.execute(
                """
                UPDATE scope_quarantine_recoveries
                SET state='repairing',active_job_id=?,next_attempt_at=?,
                    lease_owner=NULL,lease_expires_at=NULL,last_error_code=NULL,
                    report_json=?,updated_at=?
                WHERE tenant_id=? AND scope_name=? AND lease_owner=?
                  AND state='repairing'
                """,
                (
                    job_id,
                    float(next_attempt_at),
                    self.database.encode_json(dict(report or {})),
                    now,
                    tenant_id,
                    scope_name,
                    owner,
                ),
            ).rowcount
            mapping_updated = connection.execute(
                """
                UPDATE scope_quarantine_recovery_jobs
                SET state='pending',last_error_code=NULL,updated_at=?
                WHERE tenant_id=? AND scope_name=? AND job_id=?
                  AND state='authorized'
                """,
                (now, tenant_id, scope_name, job_id),
            ).rowcount
            if recovery_updated != 1 or mapping_updated != 1:
                raise CommercialContractError(
                    "quarantine_recovery_publish_conflict",
                    "preclaimed quarantine recovery job could not be published",
                )

    def prepare_quarantine_recovery_job(
        self,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        owner: str,
        *,
        next_attempt_at: float,
    ) -> None:
        """Keep an adopted job hidden while its scope claim is revalidated."""

        now = time.time()
        with self.database.transaction() as connection:
            authorized = connection.execute(
                """
                SELECT 1
                FROM scope_quarantine_recovery_jobs AS mapping
                JOIN jobs ON jobs.job_id=mapping.job_id
                WHERE mapping.tenant_id=? AND mapping.scope_name=?
                  AND mapping.job_id=? AND mapping.state='authorized'
                  AND jobs.state='pending'
                """,
                (tenant_id, scope_name, job_id),
            ).fetchone()
            updated = connection.execute(
                """
                UPDATE scope_quarantine_recoveries
                SET state='repairing',active_job_id=?,next_attempt_at=?,updated_at=?
                WHERE tenant_id=? AND scope_name=? AND lease_owner=?
                  AND state IN ('auditing','repairing')
                """,
                (
                    job_id,
                    float(next_attempt_at),
                    now,
                    tenant_id,
                    scope_name,
                    owner,
                ),
            ).rowcount
            if authorized is None or updated != 1:
                raise CommercialContractError(
                    "quarantine_recovery_prepare_conflict",
                    "authorized quarantine recovery job could not be prepared",
                )

    def authorize_quarantine_recovery_job(
        self,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        owner: str,
        *,
        max_attempts: int,
        attempt_kind: str = "provider",
        local_repair_fingerprint: str | None = None,
        max_local_repairs: int = 8,
    ) -> int:
        """Authorize one audited repair job while the scope remains isolated."""

        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if max_local_repairs <= 0:
            raise ValueError("max_local_repairs must be positive")
        if attempt_kind not in {"provider", "local"}:
            raise ValueError("attempt_kind must be 'provider' or 'local'")
        repair_fingerprint = str(local_repair_fingerprint or "").strip()
        if attempt_kind == "local" and not repair_fingerprint:
            raise ValueError("local repair requires a state fingerprint")
        now = time.time()
        with self.database.transaction() as connection:
            recovery = connection.execute(
                "SELECT * FROM scope_quarantine_recoveries "
                "WHERE tenant_id=? AND scope_name=? AND lease_owner=?",
                (tenant_id, scope_name, owner),
            ).fetchone()
            quarantine = connection.execute(
                "SELECT quarantined_at FROM scope_quarantines "
                "WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            job = connection.execute(
                "SELECT state,payload_json FROM jobs WHERE job_id=? "
                "AND tenant_id=? AND scope_name=?",
                (job_id, tenant_id, scope_name),
            ).fetchone()
            if recovery is None or quarantine is None:
                raise CommercialContractError(
                    "quarantine_recovery_not_owned",
                    "quarantine recovery is not leased",
                )
            if job is None or str(job["state"]) not in {FAILED, PENDING}:
                raise CommercialContractError(
                    "quarantine_recovery_job_changed",
                    "recovery job is neither failed nor pending",
                )
            try:
                payload = json.loads(str(job["payload_json"] or "{}"))
            except json.JSONDecodeError as exc:
                raise CommercialContractError(
                    "quarantine_recovery_job_invalid",
                    "recovery job payload is invalid",
                ) from exc
            job_type = str(payload.get("job_type") or "") if isinstance(payload, Mapping) else ""
            if job_type not in {"ingest", "reindex", "consolidate"}:
                raise CommercialContractError(
                    "quarantine_recovery_job_invalid",
                    "only failed ingest jobs or recovery derived jobs are allowed",
                )
            prior = connection.execute(
                "SELECT attempt_count,provider_attempt_count,"
                "local_repair_attempt_count,last_local_repair_fingerprint "
                "FROM scope_quarantine_recovery_jobs "
                "WHERE tenant_id=? AND scope_name=? AND job_id=?",
                (tenant_id, scope_name, job_id),
            ).fetchone()
            prior_attempts = int(prior["attempt_count"]) if prior is not None else 0
            provider_attempts = (
                int(prior["provider_attempt_count"]) if prior is not None else 0
            )
            local_repair_attempts = (
                int(prior["local_repair_attempt_count"])
                if prior is not None
                else 0
            )
            prior_repair_fingerprint = (
                str(prior["last_local_repair_fingerprint"] or "")
                if prior is not None
                else ""
            )
            repair_contract = repair_fingerprint.partition(":")[0]
            prior_repair_contract = prior_repair_fingerprint.partition(":")[0]
            adopting_pending = (
                job_type == "ingest" and str(job["state"]) == PENDING
            )
            if adopting_pending:
                # A formal retry already moved this job to pending, but the
                # quarantined worker could not claim it before registration.
                # Adopting that pending attempt must not consume another model
                # call budget.
                attempt = max(
                    1, prior_attempts
                )
            else:
                attempt = prior_attempts + 1
                if attempt_kind == "provider":
                    provider_attempts += 1
                    if provider_attempts > max_attempts:
                        raise CommercialContractError(
                            "quarantine_recovery_budget_exhausted",
                            "quarantine recovery provider retry budget is exhausted",
                        )
                else:
                    if repair_fingerprint == prior_repair_fingerprint:
                        raise CommercialContractError(
                            "quarantine_local_repair_state_repeated",
                            "quarantine local repair state was already attempted",
                        )
                    # Only an explicit audited repair-contract upgrade resets
                    # the bounded local budget. Durable state changes within
                    # one contract do not create an unbounded retry loop.
                    if (
                        repair_contract
                        and prior_repair_contract
                        and repair_contract != prior_repair_contract
                    ):
                        local_repair_attempts = 0
                    local_repair_attempts += 1
                    if local_repair_attempts > max_local_repairs:
                        raise CommercialContractError(
                            "quarantine_local_repair_budget_exhausted",
                            "quarantine local repair budget is exhausted",
                        )
                    prior_repair_fingerprint = repair_fingerprint
            connection.execute(
                """
                INSERT INTO scope_quarantine_recovery_jobs(
                    tenant_id,scope_name,job_id,state,attempt_count,
                    provider_attempt_count,local_repair_attempt_count,
                    last_local_repair_fingerprint,created_at,updated_at
                ) VALUES(?,?,?,'authorized',?,?,?,?,?,?)
                ON CONFLICT(tenant_id,scope_name,job_id) DO UPDATE SET
                    state='authorized',attempt_count=excluded.attempt_count,
                    provider_attempt_count=excluded.provider_attempt_count,
                    local_repair_attempt_count=excluded.local_repair_attempt_count,
                    last_local_repair_fingerprint=excluded.last_local_repair_fingerprint,
                    last_error_code=NULL,updated_at=excluded.updated_at
                """,
                (
                    tenant_id,
                    scope_name,
                    job_id,
                    attempt,
                    provider_attempts,
                    local_repair_attempts,
                    prior_repair_fingerprint or None,
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE scope_quarantine_recoveries SET state='repairing',"
                "active_job_id=?,resumed_job_count=resumed_job_count+1,updated_at=? "
                "WHERE tenant_id=? AND scope_name=? AND lease_owner=?",
                (job_id, now, tenant_id, scope_name, owner),
            )
        return attempt

    def mark_quarantine_recovery_job(
        self,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        *,
        state: str,
        error_code: str | None = None,
    ) -> None:
        if state not in {
            "authorized",
            "pending",
            "running",
            "succeeded",
            "failed",
            "manual_review",
        }:
            raise ValueError("invalid quarantine recovery job state")
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE scope_quarantine_recovery_jobs SET state=?,"
                "last_error_code=?,updated_at=? WHERE tenant_id=? "
                "AND scope_name=? AND job_id=?",
                (
                    state,
                    str(error_code or "") or None,
                    time.time(),
                    tenant_id,
                    scope_name,
                    job_id,
                ),
            )

    def quarantine_recovery_jobs(
        self, tenant_id: str, scope_name: str
    ) -> list[dict[str, Any]]:
        with self.database.transaction(immediate=False) as connection:
            rows = connection.execute(
                "SELECT recovery.*,jobs.state AS job_state,jobs.error AS job_error "
                "FROM scope_quarantine_recovery_jobs AS recovery "
                "JOIN jobs ON jobs.job_id=recovery.job_id "
                "WHERE recovery.tenant_id=? AND recovery.scope_name=? "
                "ORDER BY jobs.scope_seq,jobs.job_id",
                (tenant_id, scope_name),
            ).fetchall()
        return [dict(row) for row in rows]

    def is_quarantine_recovery_job(
        self, tenant_id: str, scope_name: str, job_id: str
    ) -> bool:
        with self.database.transaction(immediate=False) as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM scope_quarantine_recovery_jobs AS job
                JOIN scope_quarantine_recoveries AS recovery
                  ON recovery.tenant_id=job.tenant_id
                 AND recovery.scope_name=job.scope_name
                JOIN scope_quarantines AS quarantine
                  ON quarantine.tenant_id=job.tenant_id
                 AND quarantine.scope_name=job.scope_name
                WHERE job.tenant_id=? AND job.scope_name=? AND job.job_id=?
                  AND job.state IN ('pending','running')
                  AND recovery.state='repairing'
                  AND recovery.quarantine_started_at=quarantine.quarantined_at
                """,
                (tenant_id, scope_name, job_id),
            ).fetchone()
        return row is not None

    def complete_quarantine_recovery(
        self,
        tenant_id: str,
        scope_name: str,
        owner: str,
        *,
        report: Mapping[str, Any],
        audited_historical_failed_ingest_job_ids: Iterable[str] = (),
    ) -> bool:
        """Atomically clear only the same audited quarantine generation."""

        now = time.time()
        historical_ids = tuple(
            sorted(
                {
                    str(value).strip()
                    for value in audited_historical_failed_ingest_job_ids
                    if str(value).strip()
                }
            )
        )
        audited_count_names = (
            "source_count",
            "record_source_count",
            "enriched_source_count",
            "failed_source_count",
            "pending_source_count",
            "prepared_message_commit_count",
            "source_event_seq",
            "promoted_event_seq",
            "searchable_event_seq",
        )
        audited_counts: dict[str, int] = {}
        for name in audited_count_names:
            value = report.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return False
            audited_counts[name] = value
        if (
            report.get("integrity_ok") is not True
            or report.get("ready_to_release") is not True
            or audited_counts["record_source_count"]
            != audited_counts["source_count"]
            or audited_counts["enriched_source_count"]
            != audited_counts["source_count"]
            or audited_counts["source_event_seq"]
            != audited_counts["source_count"]
            or audited_counts["promoted_event_seq"]
            != audited_counts["source_count"]
            or audited_counts["searchable_event_seq"]
            != audited_counts["source_count"]
            or audited_counts["failed_source_count"] != 0
            or audited_counts["pending_source_count"] != 0
            or audited_counts["prepared_message_commit_count"] != 0
        ):
            return False
        with self.database.transaction() as connection:
            recovery = connection.execute(
                "SELECT * FROM scope_quarantine_recoveries WHERE tenant_id=? "
                "AND scope_name=? AND lease_owner=?",
                (tenant_id, scope_name, owner),
            ).fetchone()
            quarantine = connection.execute(
                "SELECT quarantined_at FROM scope_quarantines WHERE tenant_id=? "
                "AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            if recovery is None or quarantine is None:
                return False
            if float(recovery["quarantine_started_at"]) != float(
                quarantine["quarantined_at"]
            ):
                return False
            unfinished_rows = connection.execute(
                """
                SELECT recovery_job.job_id AS recovery_job_id,
                       jobs.job_id AS persisted_job_id,jobs.tenant_id AS job_tenant_id,
                       jobs.scope_name AS job_scope_name,jobs.state,jobs.payload_json,
                       recovery_job.state AS recovery_mapping_state
                FROM scope_quarantine_recovery_jobs AS recovery_job
                LEFT JOIN jobs ON jobs.job_id=recovery_job.job_id
                WHERE recovery_job.tenant_id=? AND recovery_job.scope_name=?
                  AND (jobs.job_id IS NULL OR jobs.state<>?)
                ORDER BY recovery_job.job_id
                """,
                (tenant_id, scope_name, SUCCEEDED),
            ).fetchall()
            unfinished_ids = {
                str(row["recovery_job_id"]) for row in unfinished_rows
            }
            if unfinished_ids != set(historical_ids):
                return False
            for row in unfinished_rows:
                if (
                    row["persisted_job_id"] is None
                    or str(row["job_tenant_id"] or "") != tenant_id
                    or str(row["job_scope_name"] or "") != scope_name
                    or str(row["state"]) != FAILED
                ):
                    return False
                if str(row["recovery_mapping_state"] or "") != "failed":
                    return False
                try:
                    payload = json.loads(str(row["payload_json"] or "{}"))
                except json.JSONDecodeError:
                    return False
                if (
                    not isinstance(payload, Mapping)
                    or str(payload.get("job_type") or "") != "ingest"
                    or str(payload.get("scope_name") or "default") != scope_name
                ):
                    return False
            watermarks = connection.execute(
                "SELECT source_event_seq,promoted_event_seq,conflict_generation,"
                "promoted_conflict_generation,source_raw_token_estimate,"
                "promoted_raw_token_estimate,source_user_turns,"
                "promoted_user_turns,indexed_event_seq,delta_indexed_event_seq "
                "FROM scope_evolution_state WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            if audited_counts["source_count"] > 0 and watermarks is None:
                return False
            if watermarks is not None and (
                int(watermarks["source_event_seq"])
                != audited_counts["source_count"]
                or int(watermarks["promoted_event_seq"])
                != audited_counts["promoted_event_seq"]
                or int(watermarks["promoted_conflict_generation"])
                < int(watermarks["conflict_generation"])
                or int(watermarks["promoted_raw_token_estimate"])
                < int(watermarks["source_raw_token_estimate"])
                or int(watermarks["promoted_user_turns"])
                < int(watermarks["source_user_turns"])
                or max(
                    int(watermarks["indexed_event_seq"]),
                    int(watermarks["delta_indexed_event_seq"]),
                )
                != audited_counts["searchable_event_seq"]
            ):
                return False
            deleted = connection.execute(
                "DELETE FROM scope_quarantines WHERE tenant_id=? AND scope_name=? "
                "AND quarantined_at=?",
                (tenant_id, scope_name, float(quarantine["quarantined_at"])),
            ).rowcount
            if deleted != 1:
                return False
            final_report = dict(report)
            if historical_ids:
                final_report.update(
                    {
                        "audited_historical_failed_ingest_count": len(
                            historical_ids
                        ),
                        "audited_historical_failed_ingest_set_sha256": hashlib.sha256(
                            self.database.encode_json(list(historical_ids)).encode(
                                "utf-8"
                            )
                        ).hexdigest(),
                    }
                )
            connection.execute(
                """
                UPDATE scope_quarantine_recoveries
                SET state='recovered',active_job_id=NULL,next_attempt_at=?,
                    lease_owner=NULL,lease_expires_at=NULL,last_error_code=NULL,
                    report_json=?,updated_at=?,recovered_at=?
                WHERE tenant_id=? AND scope_name=?
                """,
                (
                    now,
                    self.database.encode_json(final_report),
                    now,
                    now,
                    tenant_id,
                    scope_name,
                ),
            )
        return True

    def mark_scope_deleting(
        self,
        tenant_id: str,
        scope_name: str,
        deletion_job_id: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM scope_lifecycle WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            if row is not None and row["state"] == "deleted":
                if str(row["deletion_job_id"] or "") != deletion_job_id:
                    raise CommercialContractError("scope_deleted", "scope was already deleted")
            connection.execute(
                "DELETE FROM scope_quarantines WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                """
                INSERT INTO scope_lifecycle(
                    tenant_id,scope_name,state,deletion_job_id,reason,updated_at,deleted_at
                ) VALUES(?,?, 'deleting',?,?,?,NULL)
                ON CONFLICT(tenant_id,scope_name) DO UPDATE SET
                    state='deleting', deletion_job_id=excluded.deletion_job_id,
                    reason=excluded.reason, updated_at=excluded.updated_at, deleted_at=NULL
                """,
                (tenant_id, scope_name, deletion_job_id, reason, now),
            )
            value = connection.execute(
                "SELECT * FROM scope_lifecycle WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
        self._cancel_blocked_pending_jobs(
            tenant_id,
            scope_name,
            reason_code="scope_deleting_before_start",
        )
        return dict(value)

    def reopen_scope(self, tenant_id: str, scope_name: str) -> bool:
        now = time.time()
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE scope_lifecycle
                SET state='active', deletion_job_id=NULL, reason=NULL,
                    updated_at=?, deleted_at=NULL
                WHERE tenant_id=? AND scope_name=? AND state='deleted'
                """,
                (now, tenant_id, scope_name),
            )
        return cursor.rowcount == 1

    def complete_scope_deletion(
        self,
        tenant_id: str,
        scope_name: str,
        deletion_job_id: str,
        *,
        scope_id: str,
    ) -> None:
        now = time.time()
        redacted_payload = self.database.encode_json(
            {"job_type": "redacted", "scope_name": scope_name}
        )
        redacted_hash = hashlib.sha256(redacted_payload.encode("utf-8")).hexdigest()
        with self.database.transaction() as connection:
            connection.execute(
                "DELETE FROM user_provider_tasks WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                "DELETE FROM operation_stages WHERE tenant_id=? AND scope_name=? AND job_id<>?",
                (tenant_id, scope_name, deletion_job_id),
            )
            connection.execute(
                """
                UPDATE jobs SET payload_json=?, payload_hash=?, result_json=NULL,
                    error=CASE WHEN error IS NULL THEN NULL ELSE 'redacted_after_scope_deletion' END
                WHERE tenant_id=? AND scope_name=? AND job_id<>?
                """,
                (redacted_payload, redacted_hash, tenant_id, scope_name, deletion_job_id),
            )
            connection.execute(
                """
                UPDATE provider_calls SET request_json=NULL, response_json=NULL,
                    error=CASE WHEN error IS NULL THEN NULL ELSE 'redacted_after_scope_deletion' END
                WHERE tenant_id=? AND scope_name=?
                """,
                (tenant_id, scope_name),
            )
            connection.execute(
                "DELETE FROM provider_call_reconciliations "
                "WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                "DELETE FROM scope_ingest_watermark_commits WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                "DELETE FROM scope_source_event_commits WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                "DELETE FROM scope_ingest_source_sets WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                "DELETE FROM scope_evolution_state WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                "DELETE FROM graph_runtime_audits WHERE scope_id=?",
                (scope_id,),
            )
            connection.execute(
                "DELETE FROM memory_feedback WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                "DELETE FROM memory_graph_views WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                "DELETE FROM memory_graph_refresh_queue WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                "DELETE FROM session_graph_metadata WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                "DELETE FROM scope_sessions WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                "DELETE FROM scope_ingest_events WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                "DELETE FROM scope_catalog WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            )
            connection.execute(
                """
                UPDATE scope_exports SET state='expired', artifact_path=NULL,
                    artifact_sha256=NULL, size_bytes=NULL
                WHERE tenant_id=? AND scope_name=?
                """,
                (tenant_id, scope_name),
            )
            connection.execute(
                """
                INSERT INTO scope_lifecycle(
                    tenant_id,scope_name,state,deletion_job_id,updated_at,deleted_at
                ) VALUES(?,?, 'deleted',?,?,?)
                ON CONFLICT(tenant_id,scope_name) DO UPDATE SET
                    state='deleted', deletion_job_id=excluded.deletion_job_id,
                    updated_at=excluded.updated_at, deleted_at=excluded.deleted_at
                """,
                (tenant_id, scope_name, deletion_job_id, now, now),
            )

    def ensure_export(
        self,
        export_id: str,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        expires_at: float,
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO scope_exports(
                    export_id,tenant_id,scope_name,job_id,state,created_at,expires_at
                ) VALUES(?,?,?,?, 'pending',?,?)
                """,
                (export_id, tenant_id, scope_name, job_id, time.time(), expires_at),
            )

    def complete_export(
        self,
        export_id: str,
        *,
        artifact_path: Path,
        artifact_sha256: str,
        size_bytes: int,
    ) -> None:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE scope_exports SET state='ready', artifact_path=?, artifact_sha256=?,
                    size_bytes=?, completed_at=? WHERE export_id=?
                """,
                (str(artifact_path), artifact_sha256, size_bytes, time.time(), export_id),
            )
        if cursor.rowcount != 1:
            raise CommercialContractError("export_not_registered", "export record is missing")

    def fail_export(self, export_id: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE scope_exports SET state='failed', completed_at=? WHERE export_id=?",
                (time.time(), export_id),
            )

    def get_export(self, tenant_id: str, scope_name: str, export_id: str) -> dict[str, Any] | None:
        with self.database.transaction(immediate=False) as connection:
            row = connection.execute(
                """
                SELECT * FROM scope_exports
                WHERE export_id=? AND tenant_id=? AND scope_name=?
                """,
                (export_id, tenant_id, scope_name),
            ).fetchone()
        return None if row is None else dict(row)

    def set_retention_policy(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        enabled: bool,
        inactive_days: int,
        key_id: str,
    ) -> dict[str, Any]:
        if inactive_days < 1 or inactive_days > 3650:
            raise ValueError("inactive_days must be between 1 and 3650")
        now = time.time()
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO scope_retention_policies(
                    tenant_id,scope_name,enabled,inactive_days,updated_by_key_id,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(tenant_id,scope_name) DO UPDATE SET
                    enabled=excluded.enabled, inactive_days=excluded.inactive_days,
                    updated_by_key_id=excluded.updated_by_key_id, updated_at=excluded.updated_at
                """,
                (tenant_id, scope_name, int(enabled), inactive_days, key_id, now, now),
            )
            row = connection.execute(
                "SELECT * FROM scope_retention_policies WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
        return dict(row)

    def get_retention_policy(self, tenant_id: str, scope_name: str) -> dict[str, Any] | None:
        with self.database.transaction(immediate=False) as connection:
            row = connection.execute(
                "SELECT * FROM scope_retention_policies WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
        return None if row is None else dict(row)

    def due_retention_scopes(self, *, now: float | None = None) -> list[dict[str, Any]]:
        moment = time.time() if now is None else float(now)
        with self.database.transaction(immediate=False) as connection:
            rows = connection.execute(
                """
                SELECT p.*, s.last_ingest_at
                FROM scope_retention_policies p
                JOIN scope_evolution_state s
                  ON s.tenant_id=p.tenant_id AND s.scope_name=p.scope_name
                LEFT JOIN scope_lifecycle l
                  ON l.tenant_id=p.tenant_id AND l.scope_name=p.scope_name
                WHERE p.enabled=1
                  AND s.last_ingest_at IS NOT NULL
                  AND s.last_ingest_at <= ?
                  AND COALESCE(l.state, 'active')='active'
                ORDER BY s.last_ingest_at
                """,
                (moment - 86_400.0,),
            ).fetchall()
        return [
            dict(row)
            for row in rows
            if float(row["last_ingest_at"]) <= moment - int(row["inactive_days"]) * 86_400.0
        ]

    def add_feedback(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        query_id: str | None,
        rating: str,
        memory_ids: Iterable[str],
        comment: str | None,
        metadata: Mapping[str, Any],
        credential_id: str,
        operation_key: str | None = None,
    ) -> dict[str, Any]:
        feedback_id = "fb_" + (hashlib.sha256(f"{tenant_id}\0{scope_name}\0{credential_id}\0{operation_key}".encode()).hexdigest()
                               if operation_key else uuid.uuid4().hex)
        created_at = time.time()
        memory_id_values = list(dict.fromkeys(str(item) for item in memory_ids))
        with self.database.transaction() as connection:
            previous = connection.execute("SELECT * FROM memory_feedback WHERE feedback_id=?", (feedback_id,)).fetchone()
            if previous is not None:
                same = (previous["rating"] == rating and previous["comment"] == comment
                        and json.loads(previous["memory_ids_json"]) == memory_id_values
                        and json.loads(previous["metadata_json"]) == dict(metadata))
                if not same:
                    raise CommercialContractError("feedback_idempotency_conflict", "feedback key was reused for different content")
                return {"feedback_id": feedback_id, "scope_name": scope_name, "rating": rating,
                        "created_at": float(previous["created_at"])}
            connection.execute(
                """
                INSERT INTO memory_feedback(
                    feedback_id,tenant_id,scope_name,query_id,rating,memory_ids_json,
                    comment,metadata_json,created_by_credential_id,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    feedback_id,
                    tenant_id,
                    scope_name,
                    query_id,
                    rating,
                    self.database.encode_json(memory_id_values),
                    comment,
                    self.database.encode_json(dict(metadata)),
                    credential_id,
                    created_at,
                ),
            )
        return {
            "feedback_id": feedback_id,
            "scope_name": scope_name,
            "rating": rating,
            "created_at": created_at,
        }

    def resolve_feedback_targets(self, tenant_id: str, scope_name: str, memory_ids: Iterable[str]) -> list[str]:
        targets: list[str] = []
        with self.database.transaction() as connection:
            for memory_id in memory_ids:
                row = connection.execute("SELECT memory_ids_json FROM memory_feedback WHERE tenant_id=? AND scope_name=? AND feedback_id=?",
                                         (tenant_id, scope_name, memory_id)).fetchone()
                targets.extend(json.loads(row["memory_ids_json"]) if row else [memory_id])
        return list(dict.fromkeys(targets))

    def feedback_effects(self, tenant_id: str, scope_name: str) -> dict[str, Any]:
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_feedback WHERE tenant_id=? AND scope_name=? ORDER BY created_at, rowid",
                (tenant_id, scope_name),
            ).fetchall()
        effects: dict[str, Any] = {}
        correction_targets: dict[str, list[str]] = {}
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            action = metadata.get("_tmcra_action")
            if action not in {"ignore", "correct", "restore"}:
                continue
            targets = json.loads(row["memory_ids_json"])
            if action == "correct":
                correction_targets[row["feedback_id"]] = targets
            for memory_id in targets:
                if action == "restore":
                    effects.pop(memory_id, None)
                else:
                    effects[memory_id] = {"action": action, "feedback_id": row["feedback_id"],
                                          "replacement": metadata.get("_tmcra_replacement", ""),
                                          "created_at": float(row["created_at"])}
        for feedback_id, targets in correction_targets.items():
            current = {effect["feedback_id"]: effect for target in targets
                       if (effect := effects.get(target)) is not None and effect["action"] == "correct"}
            # Keep retired corrections linked even after restore. A delayed
            # index job must not make an obsolete correction authoritative again.
            effects[feedback_id] = {"action": "correction_alias", "corrections": list(current.values())}
        return effects

    def _secret_for_endpoint(self, tenant_id: str, endpoint_id: str) -> str:
        if not self.webhook_signing_key:
            raise CommercialContractError(
                "webhook_signing_not_configured",
                "TMCRA_WEBHOOK_SIGNING_KEY is required",
            )
        digest = hmac.new(
            self.webhook_signing_key.encode("utf-8"),
            f"{tenant_id}\0{endpoint_id}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    def create_webhook(
        self,
        tenant_id: str,
        *,
        label: str,
        url: str,
        events: Iterable[str],
        key_id: str,
    ) -> dict[str, Any]:
        url = validate_webhook_url(url)
        event_values = frozenset(str(item) for item in events)
        if not event_values or not event_values <= WEBHOOK_EVENTS:
            raise CommercialContractError("invalid_webhook_events", "webhook event list is invalid")
        endpoint_id = f"wh_{uuid.uuid4().hex}"
        now = time.time()
        secret = self._secret_for_endpoint(tenant_id, endpoint_id)
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO webhook_endpoints(
                    endpoint_id,tenant_id,label,url,events_json,enabled,
                    created_by_key_id,created_at,updated_at
                ) VALUES(?,?,?,?,?,1,?,?,?)
                """,
                (
                    endpoint_id,
                    tenant_id,
                    label,
                    url,
                    self.database.encode_json(sorted(event_values)),
                    key_id,
                    now,
                    now,
                ),
            )
        return {
            "endpoint_id": endpoint_id,
            "label": label,
            "url": url,
            "events": sorted(event_values),
            "enabled": True,
            "created_at": now,
            "signing_secret": secret,
        }

    def list_webhooks(self, tenant_id: str) -> list[dict[str, Any]]:
        with self.database.transaction(immediate=False) as connection:
            rows = connection.execute(
                "SELECT * FROM webhook_endpoints WHERE tenant_id=? ORDER BY created_at,endpoint_id",
                (tenant_id,),
            ).fetchall()
        return [
            {
                "endpoint_id": str(row["endpoint_id"]),
                "label": str(row["label"]),
                "url": str(row["url"]),
                "events": json.loads(str(row["events_json"])),
                "enabled": bool(row["enabled"]),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
            }
            for row in rows
        ]

    def disable_webhook(self, tenant_id: str, endpoint_id: str) -> bool:
        now = time.time()
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE webhook_endpoints SET enabled=0,disabled_at=?,updated_at=?
                WHERE endpoint_id=? AND tenant_id=? AND enabled=1
                """,
                (now, now, endpoint_id, tenant_id),
            )
            connection.execute(
                """
                UPDATE webhook_deliveries SET state='dead',updated_at=?,
                    last_error='endpoint_disabled'
                WHERE endpoint_id=? AND state IN ('pending','delivering')
                """,
                (now, endpoint_id),
            )
        return cursor.rowcount == 1

    @staticmethod
    def _job_event_types(job: Job) -> list[str]:
        values = [f"job.{job.state}"]
        job_type = str(dict(job.payload or {}).get("job_type") or "")
        if job.state == SUCCEEDED:
            values.extend(
                {
                    "ingest": ["ingest.completed"],
                    "consolidate": ["consolidation.completed"],
                    "reindex": ["index.completed"],
                    "export_scope": ["export.ready"],
                    "delete_scope": ["scope.deleted"],
                }.get(job_type, [])
            )
        return [value for value in values if value in WEBHOOK_EVENTS]

    def enqueue_job_events(self, job: Job) -> int:
        if job.state not in {SUCCEEDED, FAILED, CANCELLED}:
            return 0
        payload = dict(job.payload or {})
        created = 0
        for event_type in self._job_event_types(job):
            event_id = f"evt_{job.job_id}_{event_type.replace('.', '_')}"
            event_payload = {
                "id": event_id,
                "type": event_type,
                "created_at": job.finished_at or job.updated_at,
                "data": {
                    "job_id": job.job_id,
                    "job_type": str(payload.get("job_type") or ""),
                    "status": job.state,
                    "scope_name": str(payload.get("scope_name") or "default"),
                    "export_id": payload.get("export_id"),
                },
            }
            now = time.time()
            with self.database.transaction() as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO webhook_events(
                        event_id,tenant_id,event_type,payload_json,created_at
                    ) VALUES(?,?,?,?,?)
                    """,
                    (
                        event_id,
                        job.tenant_id,
                        event_type,
                        self.database.encode_json(event_payload),
                        now,
                    ),
                )
                if cursor.rowcount:
                    endpoints = connection.execute(
                        "SELECT endpoint_id,events_json FROM webhook_endpoints WHERE tenant_id=? AND enabled=1",
                        (job.tenant_id,),
                    ).fetchall()
                    for endpoint in endpoints:
                        if event_type not in json.loads(str(endpoint["events_json"])):
                            continue
                        connection.execute(
                            """
                            INSERT OR IGNORE INTO webhook_deliveries(
                                delivery_id,endpoint_id,event_id,state,next_attempt_at,
                                created_at,updated_at
                            ) VALUES(?,?,?,'pending',?,?,?)
                            """,
                            (
                                f"dlv_{uuid.uuid4().hex}",
                                endpoint["endpoint_id"],
                                event_id,
                                now,
                                now,
                                now,
                            ),
                        )
                    created += 1
        return created

    def reconcile_terminal_job_events(self, *, limit: int = 200) -> int:
        with self.database.transaction(immediate=False) as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs WHERE state IN (?,?,?)
                ORDER BY finished_at DESC LIMIT ?
                """,
                (SUCCEEDED, FAILED, CANCELLED, limit),
            ).fetchall()
        from .jobs import _job_from_row

        return sum(self.enqueue_job_events(_job_from_row(row)) for row in rows)

    def claim_webhook_delivery(self, *, now: float | None = None) -> WebhookDelivery | None:
        moment = time.time() if now is None else float(now)
        with self.database.transaction() as connection:
            row = connection.execute(
                """
                SELECT d.*,e.tenant_id,e.event_type,e.payload_json,w.url
                FROM webhook_deliveries d
                JOIN webhook_events e ON e.event_id=d.event_id
                JOIN webhook_endpoints w ON w.endpoint_id=d.endpoint_id
                WHERE d.state='pending' AND d.next_attempt_at<=? AND w.enabled=1
                ORDER BY d.next_attempt_at,d.created_at LIMIT 1
                """,
                (moment,),
            ).fetchone()
            if row is None:
                return None
            cursor = connection.execute(
                """
                UPDATE webhook_deliveries SET state='delivering',attempt_count=attempt_count+1,
                    updated_at=? WHERE delivery_id=? AND state='pending'
                """,
                (moment, row["delivery_id"]),
            )
            if cursor.rowcount != 1:
                return None
        return WebhookDelivery(
            delivery_id=str(row["delivery_id"]),
            endpoint_id=str(row["endpoint_id"]),
            event_id=str(row["event_id"]),
            tenant_id=str(row["tenant_id"]),
            url=str(row["url"]),
            event_type=str(row["event_type"]),
            payload=json.loads(str(row["payload_json"])),
            attempt_count=int(row["attempt_count"]) + 1,
        )

    def finish_webhook_delivery(
        self,
        delivery: WebhookDelivery,
        *,
        status_code: int | None,
        error: str | None,
        max_attempts: int = 8,
    ) -> None:
        now = time.time()
        delivered = error is None and status_code is not None and 200 <= status_code < 300
        if delivered:
            state = "delivered"
            next_attempt = now
        elif delivery.attempt_count >= max_attempts:
            state = "dead"
            next_attempt = now
        else:
            state = "pending"
            next_attempt = now + min(3600.0, 2.0 ** delivery.attempt_count)
        with self.database.transaction() as connection:
            connection.execute(
                """
                UPDATE webhook_deliveries SET state=?,next_attempt_at=?,last_error=?,
                    last_status_code=?,updated_at=?,delivered_at=? WHERE delivery_id=?
                """,
                (
                    state,
                    next_attempt,
                    None if delivered else (error or f"http_{status_code}"),
                    status_code,
                    now,
                    now if delivered else None,
                    delivery.delivery_id,
                ),
            )

    def webhook_headers(self, delivery: WebhookDelivery, body: bytes) -> dict[str, str]:
        secret = self._secret_for_endpoint(delivery.tenant_id, delivery.endpoint_id)
        timestamp = str(int(time.time()))
        signature = hmac.new(
            secret.encode("utf-8"),
            timestamp.encode("ascii") + b"." + body,
            hashlib.sha256,
        ).hexdigest()
        return {
            "Content-Type": "application/json",
            "User-Agent": "TMCRA-Webhooks/1.0",
            "X-TMCRA-Delivery": delivery.delivery_id,
            "X-TMCRA-Event": delivery.event_type,
            "X-TMCRA-Timestamp": timestamp,
            "X-TMCRA-Signature": f"v1={signature}",
        }


def _send_webhook(
    delivery: WebhookDelivery,
    headers: Mapping[str, str],
    body: bytes,
    timeout_seconds: float,
) -> int:
    _assert_public_target(delivery.url)
    request = urllib.request.Request(
        delivery.url,
        data=body,
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


class WebhookDispatcher:
    def __init__(
        self,
        control: CommercialControl,
        *,
        sender: Callable[[WebhookDelivery, Mapping[str, str], bytes, float], int] = _send_webhook,
        timeout_seconds: float = 10.0,
        poll_seconds: float = 0.5,
    ) -> None:
        self.control = control
        self.sender = sender
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="tmcra-webhooks", daemon=False)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        last_reconcile = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_reconcile >= 30.0:
                try:
                    self.control.reconcile_terminal_job_events()
                except Exception:
                    pass
                last_reconcile = now
            try:
                delivery = self.control.claim_webhook_delivery()
            except Exception:
                self._stop.wait(self.poll_seconds)
                continue
            if delivery is None:
                self._stop.wait(self.poll_seconds)
                continue
            body = json.dumps(
                delivery.payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            status_code: int | None = None
            error: str | None = None
            try:
                headers = self.control.webhook_headers(delivery, body)
                status_code = self.sender(delivery, headers, body, self.timeout_seconds)
                if not 200 <= status_code < 300:
                    error = f"http_{status_code}"
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
            self.control.finish_webhook_delivery(
                delivery,
                status_code=status_code,
                error=error,
            )
