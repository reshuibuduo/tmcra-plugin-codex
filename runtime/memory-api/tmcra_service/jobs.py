"""Persistent, idempotent job state management."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from .control_db import ControlDB
from .usage_attribution import UNATTRIBUTED, UsageAttribution


PENDING = "pending"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATES = frozenset({SUCCEEDED, FAILED, CANCELLED})
VALID_TRANSITIONS = {
    PENDING: frozenset({RUNNING, CANCELLED}),
    RUNNING: frozenset({SUCCEEDED, FAILED, CANCELLED}),
    SUCCEEDED: frozenset(),
    FAILED: frozenset(),
    CANCELLED: frozenset(),
}

STAGE_READY = "ready"
STAGE_RUNNING = "running"
STAGE_SUCCEEDED = "succeeded"
STAGE_FAILED = "failed"
STAGE_CANCELLED = "cancelled"
STAGE_TERMINAL_STATES = frozenset({STAGE_SUCCEEDED, STAGE_FAILED, STAGE_CANCELLED})
STAGE_TRANSITIONS = {
    STAGE_READY: frozenset({STAGE_RUNNING, STAGE_CANCELLED}),
    STAGE_RUNNING: frozenset({STAGE_SUCCEEDED, STAGE_FAILED, STAGE_CANCELLED}),
    STAGE_SUCCEEDED: frozenset(),
    STAGE_FAILED: frozenset(),
    STAGE_CANCELLED: frozenset(),
}


def _execution_lane(payload: Mapping[str, Any]) -> str:
    job_type = str(payload.get("job_type") or "")
    if job_type == "recall":
        return "read"
    if job_type in {
        "ingest",
        "reindex",
        "consolidate",
        "delete_memories",
        "delete_session",
    }:
        return "mutation"
    return "exclusive"


def _lanes_conflict(left: str, right: str) -> bool:
    if left == "exclusive" or right == "exclusive":
        return True
    if left == "read" or right == "read":
        return False
    return left == right


def _payload_lane(payload_json: str | None) -> str:
    if payload_json is None:
        return "exclusive"
    payload = json.loads(payload_json)
    return _execution_lane(payload if isinstance(payload, Mapping) else {})


def _validate_payload_scope(payload: Any, scope_name: str) -> None:
    if not isinstance(payload, Mapping) or "scope_name" not in payload:
        return
    payload_scope = str(payload.get("scope_name") or "default")
    if payload_scope != scope_name:
        raise ValueError("payload scope_name does not match the durable job scope")


class JobError(Exception):
    """Base class for job-store errors."""


class JobNotFound(JobError):
    pass


class JobStateError(JobError):
    pass


class IdempotencyConflict(JobError):
    pass


class JobQueueFull(JobError):
    def __init__(self, queue_scope: str, limit: int) -> None:
        super().__init__(f"{queue_scope} active-job queue reached limit {limit}")
        self.queue_scope = queue_scope
        self.limit = limit


@dataclass(frozen=True)
class Job:
    job_id: str
    tenant_id: str
    idempotency_key: str
    scope_name: str
    scope_seq: int
    payload: Any
    state: str
    result: Any
    error: str | None
    worker_id: str | None
    created_at: float
    updated_at: float
    started_at: float | None
    finished_at: float | None
    heartbeat_at: float | None
    lease_expires_at: float | None
    version: int


@dataclass(frozen=True)
class OperationStage:
    stage_id: str
    job_id: str | None
    tenant_id: str
    scope_name: str
    scope_seq: int | None
    stage_name: str
    stage_seq: int
    state: str
    attempt: int
    payload: Any
    result: Any
    error: str | None
    worker_id: str | None
    created_at: float
    updated_at: float
    started_at: float | None
    finished_at: float | None
    heartbeat_at: float | None
    lease_expires_at: float | None
    version: int


@dataclass(frozen=True)
class ProviderCall:
    call_id: str
    tenant_id: str
    scope_name: str
    job_id: str | None
    stage_id: str | None
    provider: str
    model: str
    operation: str | None
    status: str
    request: Any
    response: Any
    error: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cost_micro_cny: int | None
    cache_hit_tokens: int | None
    cache_miss_tokens: int | None
    usage_state: str
    price_version: str | None
    key_id: str | None
    client_platform: str
    integration_id: str | None
    agent_id: str | None
    attribution_source: str
    request_sha256: str | None
    response_sha256: str | None
    started_at: float | None
    finished_at: float | None
    created_at: float


@dataclass(frozen=True)
class ProviderPrice:
    provider: str
    model: str
    currency: str
    input_micro_cny_per_million: int | None
    cache_hit_input_micro_cny_per_million: int | None
    cache_miss_input_micro_cny_per_million: int | None
    output_micro_cny_per_million: int | None
    effective_at: float
    metadata: Any
    updated_at: float


@dataclass(frozen=True)
class ResumeAuthorization:
    """Explicit authorization for requeueing a failed production job."""

    reason_code: str
    resume_mode: str | None = None
    audit_fingerprint: str | None = None
    evidence: Mapping[str, Any] | None = None

    @classmethod
    def from_evidence(
        cls,
        *,
        reason_code: str,
        resume_mode: str,
        evidence: Mapping[str, Any],
    ) -> "ResumeAuthorization":
        encoded = json.dumps(
            evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return cls(
            reason_code=reason_code,
            resume_mode=resume_mode,
            audit_fingerprint=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            evidence=dict(evidence),
        )

    def as_reason(self) -> dict[str, Any]:
        return {
            "code": self.reason_code,
            "resume_mode": self.resume_mode,
            "audit_fingerprint": self.audit_fingerprint,
            "evidence": self.evidence,
        }


def _payload_json(payload: Any) -> tuple[str, str]:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _job_from_row(row: Any) -> Job:
    return Job(
        job_id=row["job_id"],
        tenant_id=row["tenant_id"],
        idempotency_key=row["idempotency_key"],
        scope_name=str(row["scope_name"]),
        scope_seq=int(row["scope_seq"]),
        payload=json.loads(row["payload_json"]),
        state=row["state"],
        result=None if row["result_json"] is None else json.loads(row["result_json"]),
        error=row["error"],
        worker_id=row["worker_id"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        started_at=None if row["started_at"] is None else float(row["started_at"]),
        finished_at=None if row["finished_at"] is None else float(row["finished_at"]),
        heartbeat_at=(
            None if row["heartbeat_at"] is None else float(row["heartbeat_at"])
        ),
        lease_expires_at=(
            None
            if row["lease_expires_at"] is None
            else float(row["lease_expires_at"])
        ),
        version=int(row["version"]),
    )


def _json_value(value: Any) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _structured_reason(
    reason: Mapping[str, Any] | str | None,
    *,
    default_code: str,
    **context: Any,
) -> dict[str, Any]:
    if reason is None:
        value: dict[str, Any] = {"code": default_code}
    elif isinstance(reason, Mapping):
        value = dict(reason)
        value.setdefault("code", default_code)
    else:
        value = {"code": str(reason).strip() or default_code}
    value.update({key: item for key, item in context.items() if item is not None})
    code = str(value.get("code") or "").strip()
    if not code:
        raise ValueError("structured reason requires a code")
    value["code"] = code
    return value


def _stage_from_row(row: Any) -> OperationStage:
    return OperationStage(
        stage_id=str(row["stage_id"]),
        job_id=row["job_id"],
        tenant_id=str(row["tenant_id"]),
        scope_name=str(row["scope_name"]),
        scope_seq=None if row["scope_seq"] is None else int(row["scope_seq"]),
        stage_name=str(row["stage_name"]),
        stage_seq=int(row["stage_seq"]),
        state=str(row["state"]),
        attempt=int(row["attempt"]),
        payload=None if row["payload_json"] is None else json.loads(row["payload_json"]),
        result=None if row["result_json"] is None else json.loads(row["result_json"]),
        error=row["error"],
        worker_id=row["worker_id"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        started_at=None if row["started_at"] is None else float(row["started_at"]),
        finished_at=None if row["finished_at"] is None else float(row["finished_at"]),
        heartbeat_at=None if row["heartbeat_at"] is None else float(row["heartbeat_at"]),
        lease_expires_at=None if row["lease_expires_at"] is None else float(row["lease_expires_at"]),
        version=int(row["version"]),
    )


def _provider_call_from_row(row: Any) -> ProviderCall:
    return ProviderCall(
        call_id=str(row["call_id"]), tenant_id=str(row["tenant_id"]), scope_name=str(row["scope_name"]),
        job_id=row["job_id"], stage_id=row["stage_id"], provider=str(row["provider"]), model=str(row["model"]),
        operation=row["operation"], status=str(row["status"]),
        request=None if row["request_json"] is None else json.loads(row["request_json"]),
        response=None if row["response_json"] is None else json.loads(row["response_json"]),
        error=row["error"], input_tokens=row["input_tokens"], output_tokens=row["output_tokens"],
        total_tokens=row["total_tokens"], cost_micro_cny=row["cost_micros"], started_at=row["started_at"],
        cache_hit_tokens=row["cache_hit_tokens"], cache_miss_tokens=row["cache_miss_tokens"],
        usage_state=str(row["usage_state"] or "missing"), price_version=row["price_version"],
        key_id=row["key_id"],
        client_platform=str(row["client_platform"] or "unattributed"),
        integration_id=row["integration_id"], agent_id=row["agent_id"],
        attribution_source=str(row["attribution_source"] or "unattributed"),
        request_sha256=row["request_sha256"],
        response_sha256=row["response_sha256"],
        finished_at=row["finished_at"], created_at=float(row["created_at"]),
    )


def _provider_price_from_row(row: Any) -> ProviderPrice:
    return ProviderPrice(
        provider=str(row["provider"]), model=str(row["model"]), currency=str(row["currency"]),
        input_micro_cny_per_million=row["input_micros_per_million"],
        cache_hit_input_micro_cny_per_million=row["cache_hit_input_micros_per_million"],
        cache_miss_input_micro_cny_per_million=row["cache_miss_input_micros_per_million"],
        output_micro_cny_per_million=row["output_micros_per_million"], effective_at=float(row["effective_at"]),
        metadata=None if row["metadata_json"] is None else json.loads(row["metadata_json"]),
        updated_at=float(row["updated_at"]),
    )


class JobStore:
    def __init__(self, db: ControlDB, *, lease_seconds: float = 120.0) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.db = db
        self.lease_seconds = float(lease_seconds)

    @staticmethod
    def _validate_identity(tenant_id: str, idempotency_key: str) -> None:
        if not tenant_id or not idempotency_key:
            raise ValueError("tenant_id and idempotency_key are required")

    @staticmethod
    def _fail_abandoned_stages(
        connection: sqlite3.Connection,
        *,
        now: float,
        tenant_id: str | None = None,
        scope_name: str | None = None,
        exclude_stage_id: str | None = None,
    ) -> int:
        """Expire stale Stage claims whose parent Job is no longer live.

        A Stage lease alone cannot block a scope forever.  The additional
        parent-Job predicate prevents a delayed Stage heartbeat from being
        reclaimed while its owning Job still has a valid running lease.
        """

        predicates = [
            "state=?",
            "lease_expires_at IS NOT NULL",
            "lease_expires_at<=?",
            "(job_id IS NULL OR NOT EXISTS ("
            "SELECT 1 FROM jobs parent WHERE parent.job_id=operation_stages.job_id "
            "AND parent.state=? AND parent.lease_expires_at IS NOT NULL "
            "AND parent.lease_expires_at>?))",
        ]
        parameters: list[Any] = [STAGE_RUNNING, now, RUNNING, now]
        if tenant_id is not None:
            predicates.append("tenant_id=?")
            parameters.append(tenant_id)
        if scope_name is not None:
            predicates.append("scope_name=?")
            parameters.append(scope_name)
        if exclude_stage_id is not None:
            predicates.append("stage_id<>?")
            parameters.append(exclude_stage_id)
        cursor = connection.execute(
            "UPDATE operation_stages SET state=?, error=?, finished_at=?, "
            "lease_expires_at=NULL, updated_at=?, version=version+1 WHERE "
            + " AND ".join(predicates),
            (
                STAGE_FAILED,
                "stage_lease_expired_after_parent_stopped",
                now,
                now,
                *parameters,
            ),
        )
        return int(cursor.rowcount)

    def fail_abandoned_stages(
        self,
        *,
        now: float | None = None,
        tenant_id: str | None = None,
        scope_name: str | None = None,
    ) -> int:
        moment = time.time() if now is None else float(now)
        with self.db.transaction() as connection:
            return self._fail_abandoned_stages(
                connection,
                now=moment,
                tenant_id=tenant_id,
                scope_name=scope_name,
            )

    def submit(
        self,
        tenant_id: str,
        idempotency_key: str,
        payload: Any,
        *,
        scope_name: str | None = None,
        tenant_queue_limit: int | None = None,
        global_queue_limit: int | None = None,
        requested_job_id: str | None = None,
        on_new_jobs: Callable[[sqlite3.Connection, tuple[str, ...]], None]
        | None = None,
    ) -> Job:
        self._validate_identity(tenant_id, idempotency_key)
        if scope_name is None:
            scope_name = (
                str(payload.get("scope_name") or "default")
                if isinstance(payload, Mapping)
                else "default"
            )
        self.db._validate_scope(tenant_id, scope_name)
        _validate_payload_scope(payload, scope_name)
        payload_json, payload_hash = _payload_json(payload)
        now = time.time()
        job_id = str(requested_job_id or uuid.uuid4().hex)
        if not job_id or len(job_id) > 200:
            raise ValueError("requested_job_id must be 1-200 characters")
        with self.db.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE tenant_id = ? AND idempotency_key = ?
                """,
                (tenant_id, idempotency_key),
            ).fetchone()
            if row is None:
                if tenant_queue_limit is not None:
                    tenant_active = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM jobs "
                            "WHERE tenant_id=? AND state IN (?, ?)",
                            (tenant_id, PENDING, RUNNING),
                        ).fetchone()[0]
                    )
                    if tenant_active >= tenant_queue_limit:
                        raise JobQueueFull("tenant", tenant_queue_limit)
                if global_queue_limit is not None:
                    global_active = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM jobs WHERE state IN (?, ?)",
                            (PENDING, RUNNING),
                        ).fetchone()[0]
                    )
                    if global_active >= global_queue_limit:
                        raise JobQueueFull("global", global_queue_limit)
                if on_new_jobs is not None:
                    on_new_jobs(connection, (idempotency_key,))
                connection.execute(
                    """
                    INSERT INTO jobs(
                        job_id, tenant_id, idempotency_key, payload_json, payload_hash,
                        state, created_at, updated_at, scope_name, scope_seq
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id, tenant_id, idempotency_key, payload_json, payload_hash,
                        PENDING, now, now, scope_name,
                        self.db._allocate_scope_seq(connection, tenant_id, scope_name, now),
                    ),
                )
                row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            elif row["payload_hash"] != payload_hash:
                raise IdempotencyConflict("idempotency key was used with a different payload")
        return _job_from_row(row)

    def submit_batch(
        self,
        tenant_id: str,
        requests: list[tuple[str, Any]],
        *,
        scope_name: str,
        tenant_queue_limit: int | None = None,
        global_queue_limit: int | None = None,
        on_new_jobs: Callable[[sqlite3.Connection, tuple[str, ...]], None]
        | None = None,
    ) -> list[Job]:
        """Atomically admit an idempotent same-scope batch.

        Replays do not consume queue capacity. Any payload conflict or capacity
        failure rolls back the entire batch, so clients never have to infer
        which prefix of a request was admitted.
        """
        if not requests:
            raise ValueError("requests cannot be empty")
        self.db._validate_scope(tenant_id, scope_name)
        for _key, payload in requests:
            _validate_payload_scope(payload, scope_name)
        keys = [key for key, _payload in requests]
        if len(keys) != len(set(keys)):
            raise ValueError("batch idempotency keys must be unique")
        for key in keys:
            self._validate_identity(tenant_id, key)
        encoded = [(*_payload_json(payload), payload) for _key, payload in requests]
        now = time.time()
        rows_by_key: dict[str, Any] = {}
        with self.db.transaction() as connection:
            placeholders = ",".join("?" for _ in keys)
            existing_rows = connection.execute(
                f"SELECT * FROM jobs WHERE tenant_id=? AND idempotency_key IN ({placeholders})",
                (tenant_id, *keys),
            ).fetchall()
            rows_by_key = {str(row["idempotency_key"]): row for row in existing_rows}
            new_count = 0
            for (key, _payload), (payload_json, payload_hash, _value) in zip(
                requests, encoded
            ):
                existing = rows_by_key.get(key)
                if existing is not None:
                    if str(existing["payload_hash"]) != payload_hash:
                        raise IdempotencyConflict(
                            "idempotency key was used with a different payload"
                        )
                    continue
                new_count += 1
            if tenant_queue_limit is not None:
                active = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM jobs WHERE tenant_id=? AND state IN (?,?)",
                        (tenant_id, PENDING, RUNNING),
                    ).fetchone()[0]
                )
                if active + new_count > tenant_queue_limit:
                    raise JobQueueFull("tenant", tenant_queue_limit)
            if global_queue_limit is not None:
                active = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM jobs WHERE state IN (?,?)",
                        (PENDING, RUNNING),
                    ).fetchone()[0]
                )
                if active + new_count > global_queue_limit:
                    raise JobQueueFull("global", global_queue_limit)
            new_keys = tuple(key for key in keys if key not in rows_by_key)
            if new_keys and on_new_jobs is not None:
                on_new_jobs(connection, new_keys)
            for (key, _payload), (payload_json, payload_hash, _value) in zip(
                requests, encoded
            ):
                if key in rows_by_key:
                    continue
                job_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO jobs(
                        job_id,tenant_id,idempotency_key,payload_json,payload_hash,
                        state,created_at,updated_at,scope_name,scope_seq
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        job_id,
                        tenant_id,
                        key,
                        payload_json,
                        payload_hash,
                        PENDING,
                        now,
                        now,
                        scope_name,
                        self.db._allocate_scope_seq(
                            connection, tenant_id, scope_name, now
                        ),
                    ),
                )
                rows_by_key[key] = connection.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
        return [_job_from_row(rows_by_key[key]) for key in keys]

    def get(self, job_id: str, *, tenant_id: str | None = None) -> Job | None:
        with self.db.transaction(immediate=False) as connection:
            if tenant_id is None:
                row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM jobs WHERE job_id = ? AND tenant_id = ?",
                    (job_id, tenant_id),
                ).fetchone()
        return None if row is None else _job_from_row(row)

    def job_execution_evidence(self, job_id: str) -> dict[str, int]:
        """Return durable evidence that a job attempt may have had effects."""

        with self.db.transaction(immediate=False) as connection:
            if connection.execute(
                "SELECT 1 FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone() is None:
                raise JobNotFound(job_id)
            stage = connection.execute(
                "SELECT COUNT(*) AS total FROM operation_stages WHERE job_id=?",
                (job_id,),
            ).fetchone()
            provider = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN status IN ('started','unknown') THEN 1 ELSE 0 END)
                           AS uncertain
                FROM provider_calls WHERE job_id=?
                """,
                (job_id,),
            ).fetchone()
        return {
            "stage_count": int(stage["total"] or 0),
            "provider_call_count": int(provider["total"] or 0),
            "uncertain_provider_call_count": int(provider["uncertain"] or 0),
        }

    def claim(
        self,
        job_id: str,
        worker_id: str,
        *,
        allow_parallel_quarantine_recovery: bool = False,
    ) -> Job:
        if not worker_id:
            raise ValueError("worker_id is required")
        now = time.time()
        with self.db.transaction() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(job_id)
            if row["state"] != PENDING:
                raise JobStateError(f"cannot claim job in state {row['state']!r}")
            candidate_lane = _payload_lane(row["payload_json"])
            prior = connection.execute(
                """
                SELECT job_id,tenant_id,scope_name,payload_json FROM jobs
                WHERE tenant_id=? AND scope_name=? AND job_id<>?
                  AND (state=? OR (state=? AND scope_seq<?))
                ORDER BY scope_seq
                """,
                (
                    row["tenant_id"],
                    row["scope_name"],
                    row["job_id"],
                    RUNNING,
                    PENDING,
                    row["scope_seq"],
                ),
            ).fetchall()
            # Retain the keyword for compatibility with older workers, but do
            # not let it bypass the per-scope Writer ordering contract.
            _ = allow_parallel_quarantine_recovery
            has_conflict = any(
                _lanes_conflict(candidate_lane, _payload_lane(str(item["payload_json"])))
                for item in prior
            )
            if has_conflict:
                raise JobStateError("a conflicting same-scope lane is not terminal")
            connection.execute(
                """
                UPDATE jobs
                SET state=?, worker_id=?, started_at=?, heartbeat_at=?,
                    lease_expires_at=?, updated_at=?, version=version+1
                WHERE job_id=? AND state=?
                """,
                (
                    RUNNING,
                    worker_id,
                    now,
                    now,
                    now + self.lease_seconds,
                    now,
                    job_id,
                    PENDING,
                ),
            )
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _job_from_row(row)

    def claim_next(
        self,
        worker_id: str,
        *,
        tenant_id: str | None = None,
        scope_name: str | None = None,
    ) -> Job | None:
        """Claim the oldest ready job, preserving order within each scope lane."""
        return self.claim_next_ready(worker_id, tenant_id=tenant_id, scope_name=scope_name)

    def claim_next_ready(
        self,
        worker_id: str,
        *,
        tenant_id: str | None = None,
        scope_name: str | None = None,
    ) -> Job | None:
        if not worker_id:
            raise ValueError("worker_id is required")
        if scope_name is not None and not scope_name.strip():
            raise ValueError("scope_name must be non-empty")
        now = time.time()
        with self.db.transaction() as connection:
            predicates = ["candidate.state=?"]
            parameters: list[Any] = [PENDING]
            if tenant_id is not None:
                predicates.append("candidate.tenant_id=?")
                parameters.append(tenant_id)
            if scope_name is not None:
                predicates.append("candidate.scope_name=?")
                parameters.append(scope_name)
            candidates = connection.execute(
                "SELECT candidate.* FROM jobs candidate WHERE "
                + " AND ".join(predicates)
                + " ORDER BY candidate.created_at, candidate.job_id",
                parameters,
            ).fetchall()
            row = None
            for candidate in candidates:
                candidate_lane = _payload_lane(str(candidate["payload_json"]))
                conflicts = connection.execute(
                    "SELECT payload_json FROM jobs WHERE tenant_id=? AND scope_name=? "
                    "AND job_id<>? AND (state=? OR (state=? AND scope_seq<?)) "
                    "ORDER BY scope_seq",
                    (
                        candidate["tenant_id"],
                        candidate["scope_name"],
                        candidate["job_id"],
                        RUNNING,
                        PENDING,
                        candidate["scope_seq"],
                    ),
                ).fetchall()
                if any(
                    _lanes_conflict(
                        candidate_lane, _payload_lane(str(item["payload_json"]))
                    )
                    for item in conflicts
                ):
                    continue
                row = candidate
                break
            if row is None:
                return None
            connection.execute(
                """
                UPDATE jobs
                SET state=?, worker_id=?, started_at=?, heartbeat_at=?,
                    lease_expires_at=?, updated_at=?, version=version+1
                WHERE job_id=? AND state=?
                """,
                (
                    RUNNING,
                    worker_id,
                    now,
                    now,
                    now + self.lease_seconds,
                    now,
                    row["job_id"],
                    PENDING,
                ),
            )
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
        return _job_from_row(row)

    def _release_terminal_claims(
        self,
        connection: sqlite3.Connection,
        row: Any,
        *,
        terminal_state: str,
        now: float,
    ) -> None:
        for claim_kind in ("evolution", "index"):
            id_column = f"active_{claim_kind}_job_id"
            version_column = f"active_{claim_kind}_job_version"
            cursor = connection.execute(
                f"""
                UPDATE scope_evolution_state
                SET {id_column}=NULL,{version_column}=NULL,updated_at=?
                WHERE tenant_id=? AND scope_name=? AND {id_column}=?
                """,
                (
                    now,
                    row["tenant_id"],
                    row["scope_name"],
                    row["job_id"],
                ),
            )
            if cursor.rowcount == 1:
                self.db._append_job_lifecycle_audit(
                    connection,
                    job_id=str(row["job_id"]),
                    tenant_id=str(row["tenant_id"]),
                    scope_name=str(row["scope_name"]),
                    scope_seq=int(row["scope_seq"]),
                    event_type="scope_claim_released",
                    stage_name=f"{claim_kind}_claim",
                    reason={
                        "code": "terminal_job_released_scope_claim",
                        "claim_kind": claim_kind,
                        "terminal_state": terminal_state,
                    },
                    from_state=str(row["state"]),
                    to_state=terminal_state,
                    worker_id=row["worker_id"],
                    created_at=now,
                )

    def transition(
        self,
        job_id: str,
        new_state: str,
        *,
        result: Any = None,
        error: str | None = None,
        worker_id: str | None = None,
        job_version: int | None = None,
        reason: Mapping[str, Any] | str | None = None,
    ) -> Job:
        if new_state not in VALID_TRANSITIONS:
            raise JobStateError(f"unknown job state {new_state!r}")
        if new_state == CANCELLED:
            return self.cancel(
                job_id,
                worker_id=worker_id,
                job_version=job_version,
                reason=reason,
            )
        now = time.time()
        result_json = None if result is None else json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        with self.db.transaction() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(job_id)
            current = row["state"]
            if current == new_state:
                if result_json is not None and row["result_json"] not in (None, result_json):
                    raise JobStateError("repeated transition has a different result")
                return _job_from_row(row)
            if new_state not in VALID_TRANSITIONS[current]:
                raise JobStateError(f"invalid transition {current!r} -> {new_state!r}")
            if current == RUNNING and worker_id is not None and row["worker_id"] != worker_id:
                raise JobStateError("worker does not own the running job")
            if job_version is not None and int(row["version"]) != int(job_version):
                raise JobStateError("job attempt version changed")
            if (
                current == RUNNING
                and job_version is not None
                and (
                    row["lease_expires_at"] is None
                    or float(row["lease_expires_at"]) <= now
                )
            ):
                raise JobStateError("job attempt lease expired")
            finished_at = now if new_state in TERMINAL_STATES else None
            connection.execute(
                """
                UPDATE jobs
                SET state=?, result_json=?, error=?, updated_at=?, finished_at=?,
                    lease_expires_at=NULL, version=version+1
                WHERE job_id=? AND state=?
                """,
                (new_state, result_json, error, now, finished_at, job_id, current),
            )
            if new_state == FAILED:
                self.db._append_job_lifecycle_audit(
                    connection,
                    job_id=job_id,
                    tenant_id=str(row["tenant_id"]),
                    scope_name=str(row["scope_name"]),
                    scope_seq=int(row["scope_seq"]),
                    event_type="job_failed",
                    from_state=current,
                    to_state=FAILED,
                    reason=_structured_reason(
                        reason,
                        default_code="job_execution_failed",
                        error_present=error is not None,
                    ),
                    worker_id=worker_id or row["worker_id"],
                    created_at=now,
                )
            if new_state in TERMINAL_STATES:
                self._release_terminal_claims(
                    connection,
                    row,
                    terminal_state=new_state,
                    now=now,
                )
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return _job_from_row(row)

    def succeed(
        self,
        job_id: str,
        result: Any,
        *,
        worker_id: str | None = None,
        job_version: int | None = None,
    ) -> Job:
        return self.transition(
            job_id,
            SUCCEEDED,
            result=result,
            worker_id=worker_id,
            job_version=job_version,
        )

    def fail(
        self,
        job_id: str,
        error: str,
        *,
        worker_id: str | None = None,
        job_version: int | None = None,
        reason: Mapping[str, Any] | str | None = None,
    ) -> Job:
        return self.transition(
            job_id,
            FAILED,
            error=error,
            worker_id=worker_id,
            job_version=job_version,
            reason=reason,
        )

    def cancel(
        self,
        job_id: str,
        *,
        worker_id: str | None = None,
        job_version: int | None = None,
        reason: Mapping[str, Any] | str | None = None,
    ) -> Job:
        now = time.time()
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise JobNotFound(job_id)
            current = str(row["state"])
            if current == CANCELLED:
                return _job_from_row(row)
            if CANCELLED not in VALID_TRANSITIONS[current]:
                raise JobStateError(f"invalid transition {current!r} -> {CANCELLED!r}")
            if current == RUNNING and worker_id is not None and row["worker_id"] != worker_id:
                raise JobStateError("worker does not own the running job")
            if job_version is not None and int(row["version"]) != int(job_version):
                raise JobStateError("job attempt version changed")
            stages = connection.execute(
                "SELECT * FROM operation_stages WHERE job_id=? ORDER BY stage_seq,stage_id",
                (job_id,),
            ).fetchall()
            provider = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN status IN ('started','unknown') THEN 1 ELSE 0 END) AS uncertain
                FROM provider_calls WHERE job_id=?
                """,
                (job_id,),
            ).fetchone()
            uncertain = int(provider["uncertain"] or 0)
            effect_state = (
                "no_side_effects"
                if not stages and int(provider["total"] or 0) == 0 and current == PENDING
                else "uncertain"
            )
            structured = _structured_reason(
                reason,
                default_code=(
                    "cancelled_before_start" if effect_state == "no_side_effects" else "cancelled_after_start"
                ),
                previous_state=current,
                effect_state=effect_state,
                stage_count=len(stages),
                uncertain_provider_call_count=uncertain,
            )
            encoded_reason = _json_value(structured)
            connection.execute(
                """
                UPDATE jobs
                SET state=?,error=?,updated_at=?,finished_at=?,lease_expires_at=NULL,
                    version=version+1
                WHERE job_id=? AND state=?
                """,
                (CANCELLED, encoded_reason, now, now, job_id, current),
            )
            self.db._append_job_lifecycle_audit(
                connection,
                job_id=job_id,
                tenant_id=str(row["tenant_id"]),
                scope_name=str(row["scope_name"]),
                scope_seq=int(row["scope_seq"]),
                event_type="job_cancelled",
                from_state=current,
                to_state=CANCELLED,
                reason=structured,
                worker_id=worker_id or row["worker_id"],
                created_at=now,
            )
            for stage in stages:
                if str(stage["state"]) not in {STAGE_READY, STAGE_RUNNING}:
                    continue
                connection.execute(
                    """
                    UPDATE operation_stages
                    SET state=?,error=?,finished_at=?,lease_expires_at=NULL,
                        updated_at=?,version=version+1
                    WHERE stage_id=? AND state=?
                    """,
                    (
                        STAGE_CANCELLED,
                        encoded_reason,
                        now,
                        now,
                        stage["stage_id"],
                        stage["state"],
                    ),
                )
                self.db._append_job_lifecycle_audit(
                    connection,
                    job_id=job_id,
                    tenant_id=str(row["tenant_id"]),
                    scope_name=str(row["scope_name"]),
                    scope_seq=int(row["scope_seq"]),
                    stage_id=str(stage["stage_id"]),
                    stage_name=str(stage["stage_name"]),
                    event_type="stage_cancelled",
                    from_state=str(stage["state"]),
                    to_state=STAGE_CANCELLED,
                    reason=structured,
                    worker_id=worker_id or stage["worker_id"],
                    created_at=now,
                )
            self._release_terminal_claims(
                connection,
                row,
                terminal_state=CANCELLED,
                now=now,
            )
            updated = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return _job_from_row(updated)

    @staticmethod
    def _validate_resume_authorization(
        connection: Any,
        row: Any,
        authorization: ResumeAuthorization | None,
    ) -> dict[str, Any]:
        if authorization is None:
            raise JobStateError("explicit resume authorization is required")
        if not isinstance(authorization, ResumeAuthorization):
            raise JobStateError("resume authorization must be structured")
        reason_code = str(authorization.reason_code or "").strip()
        if not reason_code:
            raise JobStateError("resume authorization requires a reason code")

        provider = connection.execute(
            "SELECT COUNT(*) AS total FROM provider_calls "
            "WHERE job_id=? AND status IN ('started','unknown')",
            (str(row["job_id"]),),
        ).fetchone()
        if int(provider["total"] or 0):
            raise JobStateError("cannot resume while provider outcome is unresolved")

        try:
            payload = json.loads(str(row["payload_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise JobStateError("job payload is not valid JSON") from exc
        job_type = (
            str(payload.get("job_type") or "")
            if isinstance(payload, Mapping)
            else ""
        )
        if job_type != "ingest":
            if authorization.resume_mode is not None or authorization.evidence is not None:
                raise JobStateError(
                    "non-ingest resume accepts only an explicit reason code"
                )
            return authorization.as_reason()

        mode = str(authorization.resume_mode or "").strip()
        deterministic_modes = {
            "audited_writer_state",
            "complete_writer_artifacts",
            "committed_writer_artifacts",
            "deterministic_local_repair",
        }
        provider_modes = {
            "schema_constrained_invalid_response",
            "schema_constrained_invalid_response_prepared",
            "definitive_provider_failure",
            "none",
        }
        allowed_modes = deterministic_modes | provider_modes
        if mode not in allowed_modes:
            raise JobStateError(
                "ingest resume mode is not an audited production recovery mode"
            )
        evidence = authorization.evidence
        if not isinstance(evidence, Mapping):
            raise JobStateError("ingest resume requires audit evidence")
        audit = evidence.get("audit")
        plan = evidence.get("recovery_plan")
        if not isinstance(audit, Mapping) or not isinstance(plan, Mapping):
            raise JobStateError("ingest resume evidence is incomplete")
        if audit.get("integrity_ok") is not True:
            raise JobStateError("ingest resume requires a passing Source audit")
        encoded = json.dumps(
            evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        expected_fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        if authorization.audit_fingerprint != expected_fingerprint:
            raise JobStateError("ingest resume audit fingerprint does not match evidence")
        if str(evidence.get("job_id") or "") != str(row["job_id"]):
            raise JobStateError("ingest resume evidence is bound to another job")
        if str(evidence.get("tenant_id") or "") != str(row["tenant_id"]):
            raise JobStateError("ingest resume evidence is bound to another tenant")
        if str(evidence.get("scope_name") or "") != str(row["scope_name"]):
            raise JobStateError("ingest resume evidence is bound to another scope")
        failed_operation_ids = {
            str(value)
            for value in audit.get("failed_operation_ids", ())
            if str(value)
        }
        if str(row["job_id"]) not in failed_operation_ids:
            raise JobStateError(
                "ingest resume audit does not authorize this failed operation"
            )
        if str(plan.get("mode") or mode) != mode:
            raise JobStateError("ingest resume mode does not match the audited plan")
        if mode != "audited_writer_state" and not (
            (
                mode in deterministic_modes
                and plan.get("resumable") is True
                and plan.get("parallel_safe") is True
                and plan.get("external_api_calls_expected") is False
                and plan.get("deterministic_local_repair") is True
            )
            or (
                mode in provider_modes
                and plan.get("resumable") is True
                and plan.get("parallel_safe") is False
                and plan.get("external_api_calls_expected") is True
                and plan.get("deterministic_local_repair") is False
            )
        ):
            raise JobStateError("ingest recovery is missing its audited execution proof")
        structured = authorization.as_reason()
        runtime_authorization = evidence.get("runtime_authorization")
        if isinstance(runtime_authorization, Mapping):
            compensation = runtime_authorization.get(
                "pre_writer_quarantine_gate_compensation"
            )
            if compensation is True:
                structured["pre_writer_quarantine_gate_compensation"] = True
        return structured

    def resume_failed(
        self,
        job_id: str,
        *,
        authorization: ResumeAuthorization,
    ) -> Job:
        """Requeue only after an explicit, scope-bound production audit."""
        now = time.time()
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise JobNotFound(job_id)
            if row["state"] != FAILED:
                raise JobStateError(f"cannot resume job in state {row['state']!r}")
            structured = self._validate_resume_authorization(
                connection, row, authorization
            )
            structured.update(
                {
                    "previous_error_present": row["error"] is not None,
                    "previous_job_version": int(row["version"]),
                }
            )
            self._release_terminal_claims(
                connection,
                row,
                terminal_state=FAILED,
                now=now,
            )
            connection.execute(
                """
                UPDATE jobs
                SET state=?, result_json=NULL, error=NULL, worker_id=NULL,
                    started_at=NULL, finished_at=NULL, heartbeat_at=NULL,
                    lease_expires_at=NULL, updated_at=?, version=version+1
                WHERE job_id=? AND state=?
                """,
                (PENDING, now, job_id, FAILED),
            )
            self.db._append_job_lifecycle_audit(
                connection,
                job_id=job_id,
                tenant_id=str(row["tenant_id"]),
                scope_name=str(row["scope_name"]),
                scope_seq=int(row["scope_seq"]),
                event_type="job_recovered",
                from_state=FAILED,
                to_state=PENDING,
                reason=structured,
                worker_id=row["worker_id"],
                created_at=now,
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return _job_from_row(row)

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        *,
        job_version: int | None = None,
        now: float | None = None,
    ) -> bool:
        moment = time.time() if now is None else float(now)
        with self.db.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET heartbeat_at=?, lease_expires_at=?, updated_at=?
                WHERE job_id=? AND state=? AND worker_id=?
                  AND (? IS NULL OR version=?)
                """,
                (
                    moment,
                    moment + self.lease_seconds,
                    moment,
                    job_id,
                    RUNNING,
                    worker_id,
                    job_version,
                    job_version,
                ),
            )
        return cursor.rowcount == 1

    def assert_running_attempt(
        self, job_id: str, worker_id: str, job_version: int
    ) -> Job:
        """Fence side effects to the exact durable job attempt that claimed them."""

        now = time.time()
        with self.db.transaction(immediate=False) as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        if (
            str(row["state"]) != RUNNING
            or str(row["worker_id"] or "") != worker_id
            or int(row["version"]) != int(job_version)
            or row["lease_expires_at"] is None
            or float(row["lease_expires_at"]) <= now
        ):
            raise JobStateError("job attempt ownership changed")
        return _job_from_row(row)

    def expired_running(self, *, now: float | None = None) -> list[Job]:
        moment = time.time() if now is None else float(now)
        with self.db.transaction(immediate=False) as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE state=? AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                ORDER BY lease_expires_at, job_id
                """,
                (RUNNING, moment),
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def fail_expired(
        self,
        job_id: str,
        worker_id: str,
        error: str,
        *,
        job_version: int | None = None,
        now: float | None = None,
        reason: Mapping[str, Any] | str | None = None,
    ) -> bool:
        """Fail a lease only if it is still expired at the update boundary."""
        moment = time.time() if now is None else float(now)
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                return False
            cursor = connection.execute(
                """
                UPDATE jobs
                SET state=?, error=?, finished_at=?, updated_at=?,
                    lease_expires_at=NULL, version=version+1
                WHERE job_id=? AND state=? AND worker_id=?
                  AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                  AND (? IS NULL OR version=?)
                """,
                (
                    FAILED,
                    error,
                    moment,
                    moment,
                    job_id,
                    RUNNING,
                    worker_id,
                    moment,
                    job_version,
                    job_version,
                ),
            )
            if cursor.rowcount == 1:
                structured = _structured_reason(
                    reason,
                    default_code="worker_lease_expired",
                    expired_worker_id=worker_id,
                    previous_job_version=int(row["version"]),
                )
                self.db._append_job_lifecycle_audit(
                    connection,
                    job_id=job_id,
                    tenant_id=str(row["tenant_id"]),
                    scope_name=str(row["scope_name"]),
                    scope_seq=int(row["scope_seq"]),
                    event_type="job_lease_expired",
                    from_state=RUNNING,
                    to_state=FAILED,
                    reason=structured,
                    worker_id=worker_id,
                    created_at=moment,
                )
                self._release_terminal_claims(
                    connection,
                    row,
                    terminal_state=FAILED,
                    now=moment,
                )
        return cursor.rowcount == 1

    def create_stage(
        self,
        tenant_id: str | None = None,
        scope_name: str | None = None,
        stage_name: str = "stage",
        *,
        job_id: str | None = None,
        stage_seq: int = 0,
        payload: Any = None,
        stage_id: str | None = None,
    ) -> OperationStage:
        """Create or replay a durable, ready operation stage."""
        if stage_seq < 0 or not stage_name or not stage_name.strip():
            raise ValueError("stage_name is required and stage_seq must be non-negative")
        stage_id = stage_id or uuid.uuid4().hex
        now = time.time()
        payload_json = _json_value(payload)
        with self.db.transaction() as connection:
            job = None
            if job_id is not None:
                job = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                if job is None:
                    raise JobNotFound(job_id)
                tenant_id = str(job["tenant_id"])
                scope_name = str(job["scope_name"])
                scope_seq = int(job["scope_seq"])
            else:
                if not tenant_id or not scope_name:
                    raise ValueError("tenant_id and scope_name are required without job_id")
                self.db._validate_scope(tenant_id, scope_name)
                scope_seq = None
            row = connection.execute("SELECT * FROM operation_stages WHERE stage_id=?", (stage_id,)).fetchone()
            if row is None:
                try:
                    connection.execute(
                        """
                        INSERT INTO operation_stages(
                            stage_id, job_id, tenant_id, scope_name, scope_seq, stage_name, stage_seq,
                            state, payload_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            stage_id, job_id, tenant_id, scope_name, scope_seq, stage_name, stage_seq,
                            STAGE_READY, payload_json, now, now,
                        ),
                    )
                except Exception:
                    row = connection.execute(
                        "SELECT * FROM operation_stages WHERE job_id=? AND stage_name=?",
                        (job_id, stage_name),
                    ).fetchone()
                    if row is None:
                        raise
            if row is None:
                row = connection.execute("SELECT * FROM operation_stages WHERE stage_id=?", (stage_id,)).fetchone()
            return _stage_from_row(row)

    register_stage = create_stage

    def get_stage(self, stage_id: str) -> OperationStage | None:
        with self.db.transaction(immediate=False) as connection:
            row = connection.execute("SELECT * FROM operation_stages WHERE stage_id=?", (stage_id,)).fetchone()
        return None if row is None else _stage_from_row(row)

    def list_stages(self, *, job_id: str | None = None, tenant_id: str | None = None, scope_name: str | None = None) -> list[OperationStage]:
        predicates: list[str] = []
        parameters: list[Any] = []
        if job_id is not None:
            predicates.append("job_id=?")
            parameters.append(job_id)
        if tenant_id is not None:
            predicates.append("tenant_id=?")
            parameters.append(tenant_id)
        if scope_name is not None:
            predicates.append("scope_name=?")
            parameters.append(scope_name)
        where = " WHERE " + " AND ".join(predicates) if predicates else ""
        with self.db.transaction(immediate=False) as connection:
            rows = connection.execute(
                "SELECT * FROM operation_stages" + where + " ORDER BY scope_seq, stage_seq, created_at, stage_id",
                parameters,
            ).fetchall()
        return [_stage_from_row(row) for row in rows]

    def claim_ready_stage(
        self,
        worker_id: str,
        *,
        tenant_id: str | None = None,
        scope_name: str | None = None,
    ) -> OperationStage | None:
        if not worker_id:
            raise ValueError("worker_id is required")
        now = time.time()
        with self.db.transaction() as connection:
            self._fail_abandoned_stages(
                connection,
                now=now,
                tenant_id=tenant_id,
                scope_name=scope_name,
            )
            predicates = [
                "candidate.state=?",
                "NOT EXISTS (SELECT 1 FROM operation_stages prior WHERE "
                "((prior.job_id=candidate.job_id) OR "
                "(prior.job_id IS NULL AND candidate.job_id IS NULL AND "
                "prior.tenant_id=candidate.tenant_id AND prior.scope_name=candidate.scope_name)) "
                "AND prior.stage_seq<candidate.stage_seq AND prior.state NOT IN (?, ?, ?))",
                "(candidate.scope_seq IS NULL OR NOT EXISTS (SELECT 1 FROM jobs prior_job "
                "WHERE prior_job.tenant_id=candidate.tenant_id AND prior_job.scope_name=candidate.scope_name "
                "AND prior_job.scope_seq<candidate.scope_seq AND prior_job.state IN (?, ?)))",
                "NOT EXISTS (SELECT 1 FROM operation_stages running_stage WHERE "
                "running_stage.tenant_id=candidate.tenant_id AND "
                "running_stage.scope_name=candidate.scope_name AND "
                "running_stage.stage_id<>candidate.stage_id AND running_stage.state=?)",
            ]
            parameters: list[Any] = [
                STAGE_READY,
                *STAGE_TERMINAL_STATES,
                PENDING,
                RUNNING,
                STAGE_RUNNING,
            ]
            if tenant_id is not None:
                predicates.append("candidate.tenant_id=?")
                parameters.append(tenant_id)
            if scope_name is not None:
                predicates.append("candidate.scope_name=?")
                parameters.append(scope_name)
            row = connection.execute(
                "SELECT candidate.* FROM operation_stages candidate WHERE "
                + " AND ".join(predicates)
                + " ORDER BY candidate.scope_seq, candidate.stage_seq, candidate.created_at, candidate.stage_id LIMIT 1",
                parameters,
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE operation_stages
                SET state=?, worker_id=?, attempt=attempt+1, started_at=?, heartbeat_at=?,
                    lease_expires_at=?, updated_at=?, version=version+1
                WHERE stage_id=? AND state=?
                """,
                (STAGE_RUNNING, worker_id, now, now, now + self.lease_seconds, now, row["stage_id"], STAGE_READY),
            )
            row = connection.execute("SELECT * FROM operation_stages WHERE stage_id=?", (row["stage_id"],)).fetchone()
        return _stage_from_row(row)

    claim_next_ready_stage = claim_ready_stage

    def claim_stage(
        self,
        stage_id: str,
        worker_id: str,
        *,
        job_version: int | None = None,
    ) -> OperationStage:
        """Claim one known stage without racing an unrelated ready stage."""
        if not stage_id or not worker_id:
            raise ValueError("stage_id and worker_id are required")
        now = time.time()
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM operation_stages WHERE stage_id=?", (stage_id,)
            ).fetchone()
            if row is None:
                raise JobNotFound(stage_id)
            if row["state"] == STAGE_RUNNING and row["worker_id"] == worker_id:
                return _stage_from_row(row)
            if row["state"] != STAGE_READY:
                raise JobStateError(f"cannot claim stage in state {row['state']!r}")
            if row["job_id"] is not None and job_version is not None:
                parent = connection.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)
                ).fetchone()
                if (
                    parent is None
                    or str(parent["state"]) != RUNNING
                    or str(parent["worker_id"] or "") != worker_id
                    or int(parent["version"]) != int(job_version)
                    or parent["lease_expires_at"] is None
                    or float(parent["lease_expires_at"]) <= now
                    or str(parent["tenant_id"]) != str(row["tenant_id"])
                    or str(parent["scope_name"]) != str(row["scope_name"])
                ):
                    raise JobStateError("parent job attempt ownership changed")
            self._fail_abandoned_stages(
                connection,
                now=now,
                tenant_id=str(row["tenant_id"]),
                scope_name=str(row["scope_name"]),
                exclude_stage_id=stage_id,
            )
            blocked = connection.execute(
                """
                SELECT 1 FROM operation_stages AS prior
                WHERE (
                    (prior.job_id=? AND ? IS NOT NULL)
                    OR (
                        prior.job_id IS NULL AND ? IS NULL
                        AND prior.tenant_id=? AND prior.scope_name=?
                    )
                )
                  AND prior.stage_seq<?
                  AND prior.state NOT IN (?, ?, ?)
                LIMIT 1
                """,
                (
                    row["job_id"],
                    row["job_id"],
                    row["job_id"],
                    row["tenant_id"],
                    row["scope_name"],
                    row["stage_seq"],
                    *STAGE_TERMINAL_STATES,
                ),
            ).fetchone()
            if blocked is not None:
                raise JobStateError("an earlier operation stage is not terminal")
            other_jobs = connection.execute(
                "SELECT job_id,payload_json FROM jobs WHERE tenant_id=? AND scope_name=? "
                "AND (? IS NULL OR job_id<>?) "
                "AND (state=? OR (? IS NOT NULL AND state=? AND scope_seq<?))",
                (
                    row["tenant_id"],
                    row["scope_name"],
                    row["job_id"],
                    row["job_id"],
                    RUNNING,
                    row["scope_seq"],
                    PENDING,
                    row["scope_seq"],
                ),
            ).fetchall()
            candidate_lane = _payload_lane(row["payload_json"])
            if row["job_id"] is not None:
                parent_payload = connection.execute(
                    "SELECT payload_json FROM jobs WHERE job_id=?", (row["job_id"],)
                ).fetchone()
                if parent_payload is not None:
                    candidate_lane = _payload_lane(str(parent_payload["payload_json"]))
            if any(
                _lanes_conflict(
                    candidate_lane, _payload_lane(str(other["payload_json"]))
                )
                for other in other_jobs
            ):
                raise JobStateError("a conflicting same-scope job is not terminal")
            other_stages = connection.execute(
                "SELECT stage.payload_json,stage.job_id,job.payload_json AS job_payload_json "
                "FROM operation_stages AS stage "
                "LEFT JOIN jobs AS job ON job.job_id=stage.job_id "
                "WHERE stage.tenant_id=? AND stage.scope_name=? "
                "AND stage.stage_id<>? AND stage.state=? "
                "AND (? IS NULL OR stage.job_id IS NULL OR stage.job_id<>?)",
                (
                    row["tenant_id"],
                    row["scope_name"],
                    stage_id,
                    STAGE_RUNNING,
                    row["job_id"],
                    row["job_id"],
                ),
            ).fetchall()
            if any(
                _lanes_conflict(
                    candidate_lane,
                    _payload_lane(
                        other["job_payload_json"]
                        if other["job_id"] is not None
                        else other["payload_json"]
                    ),
                )
                for other in other_stages
            ):
                raise JobStateError("a conflicting same-scope stage is running")
            cursor = connection.execute(
                """
                UPDATE operation_stages
                SET state=?, worker_id=?, attempt=attempt+1, started_at=?, heartbeat_at=?,
                    lease_expires_at=?, updated_at=?, version=version+1
                WHERE stage_id=? AND state=?
                """,
                (
                    STAGE_RUNNING,
                    worker_id,
                    now,
                    now,
                    now + self.lease_seconds,
                    now,
                    stage_id,
                    STAGE_READY,
                ),
            )
            if cursor.rowcount != 1:
                raise JobStateError("stage claim lost a concurrent race")
            row = connection.execute(
                "SELECT * FROM operation_stages WHERE stage_id=?", (stage_id,)
            ).fetchone()
        return _stage_from_row(row)

    def transition_stage(
        self,
        stage_id: str,
        new_state: str,
        *,
        result: Any = None,
        error: str | None = None,
        worker_id: str | None = None,
        stage_version: int | None = None,
    ) -> OperationStage:
        if new_state not in STAGE_TRANSITIONS:
            raise JobStateError(f"unknown stage state {new_state!r}")
        now = time.time()
        result_json = _json_value(result)
        with self.db.transaction() as connection:
            row = connection.execute("SELECT * FROM operation_stages WHERE stage_id=?", (stage_id,)).fetchone()
            if row is None:
                raise JobNotFound(stage_id)
            current = str(row["state"])
            if current == new_state:
                if result_json is not None and row["result_json"] not in (None, result_json):
                    raise JobStateError("repeated stage transition has a different result")
                return _stage_from_row(row)
            if new_state not in STAGE_TRANSITIONS[current]:
                raise JobStateError(f"invalid stage transition {current!r} -> {new_state!r}")
            if current == STAGE_RUNNING and worker_id is not None and row["worker_id"] != worker_id:
                raise JobStateError("worker does not own the running stage")
            if stage_version is not None and int(row["version"]) != int(stage_version):
                raise JobStateError("stage attempt version changed")
            if (
                current == STAGE_RUNNING
                and stage_version is not None
                and (
                    row["lease_expires_at"] is None
                    or float(row["lease_expires_at"]) <= now
                )
            ):
                raise JobStateError("stage attempt lease expired")
            finished_at = now if new_state in STAGE_TERMINAL_STATES else None
            connection.execute(
                """
                UPDATE operation_stages
                SET state=?, result_json=?, error=?, finished_at=?, lease_expires_at=NULL,
                    updated_at=?, version=version+1
                WHERE stage_id=? AND state=?
                """,
                (new_state, result_json, error, finished_at, now, stage_id, current),
            )
            row = connection.execute("SELECT * FROM operation_stages WHERE stage_id=?", (stage_id,)).fetchone()
        return _stage_from_row(row)

    def complete_stage(
        self,
        stage_id: str,
        result: Any = None,
        *,
        worker_id: str | None = None,
        stage_version: int | None = None,
    ) -> OperationStage:
        return self.transition_stage(
            stage_id,
            STAGE_SUCCEEDED,
            result=result,
            worker_id=worker_id,
            stage_version=stage_version,
        )

    def fail_stage(
        self,
        stage_id: str,
        error: str,
        *,
        worker_id: str | None = None,
        stage_version: int | None = None,
    ) -> OperationStage:
        return self.transition_stage(
            stage_id,
            STAGE_FAILED,
            error=error,
            worker_id=worker_id,
            stage_version=stage_version,
        )

    def cancel_stage(self, stage_id: str, *, worker_id: str | None = None) -> OperationStage:
        return self.transition_stage(stage_id, STAGE_CANCELLED, worker_id=worker_id)

    def retry_stage(self, stage_id: str) -> OperationStage:
        now = time.time()
        with self.db.transaction() as connection:
            row = connection.execute("SELECT * FROM operation_stages WHERE stage_id=?", (stage_id,)).fetchone()
            if row is None:
                raise JobNotFound(stage_id)
            if row["state"] != STAGE_FAILED:
                raise JobStateError(f"cannot retry stage in state {row['state']!r}")
            connection.execute(
                """
                UPDATE operation_stages
                SET state=?, result_json=NULL, error=NULL, worker_id=NULL, started_at=NULL,
                    finished_at=NULL, heartbeat_at=NULL, lease_expires_at=NULL, updated_at=?, version=version+1
                WHERE stage_id=? AND state=?
                """,
                (STAGE_READY, now, stage_id, STAGE_FAILED),
            )
            row = connection.execute("SELECT * FROM operation_stages WHERE stage_id=?", (stage_id,)).fetchone()
        return _stage_from_row(row)

    def fail_expired_stage(
        self,
        stage_id: str,
        worker_id: str,
        error: str,
        *,
        stage_version: int,
        now: float | None = None,
    ) -> bool:
        """Fence recovery to the exact still-expired stage attempt."""

        moment = time.time() if now is None else float(now)
        with self.db.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE operation_stages
                SET state=?, error=?, finished_at=?, lease_expires_at=NULL,
                    updated_at=?, version=version+1
                WHERE stage_id=? AND state=? AND worker_id=? AND version=?
                  AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                """,
                (
                    STAGE_FAILED,
                    error,
                    moment,
                    moment,
                    stage_id,
                    STAGE_RUNNING,
                    worker_id,
                    int(stage_version),
                    moment,
                ),
            )
        return cursor.rowcount == 1

    def stage_heartbeat(
        self,
        stage_id: str,
        worker_id: str,
        *,
        stage_version: int | None = None,
        now: float | None = None,
    ) -> bool:
        moment = time.time() if now is None else float(now)
        with self.db.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE operation_stages
                SET heartbeat_at=?, lease_expires_at=?, updated_at=?
                WHERE stage_id=? AND state=? AND worker_id=?
                  AND (? IS NULL OR version=?)
                """,
                (
                    moment,
                    moment + self.lease_seconds,
                    moment,
                    stage_id,
                    STAGE_RUNNING,
                    worker_id,
                    stage_version,
                    stage_version,
                ),
            )
        return cursor.rowcount == 1

    def record_provider_call(
        self,
        tenant_id: str,
        provider: str,
        model: str,
        *,
        scope_name: str = "default",
        call_id: str | None = None,
        job_id: str | None = None,
        stage_id: str | None = None,
        operation: str | None = None,
        status: str = "completed",
        request: Any = None,
        response: Any = None,
        error: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_micro_cny: int | None = None,
        cache_hit_tokens: int | None = None,
        cache_miss_tokens: int | None = None,
        usage_state: str = "missing",
        price_version: str | None = None,
        key_id: str | None = None,
        usage_attribution: UsageAttribution = UNATTRIBUTED,
        request_sha256: str | None = None,
        response_sha256: str | None = None,
        started_at: float | None = None,
        finished_at: float | None = None,
        created_at: float | None = None,
    ) -> ProviderCall:
        if not tenant_id or not provider or not model or not status:
            raise ValueError("tenant_id, provider, model, and status are required")
        if status == "succeeded":
            status = "completed"
        if status not in {"started", "completed", "failed", "unknown"}:
            raise ValueError("provider call status must be started, completed, failed, or unknown")
        if usage_state not in {"missing", "complete", "invalid", "unknown"}:
            raise ValueError("unsupported provider usage state")
        self.db._validate_scope(tenant_id, scope_name)
        call_id = call_id or uuid.uuid4().hex
        created = time.time() if created_at is None else float(created_at)
        with self.db.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM provider_calls WHERE call_id=?", (call_id,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO provider_calls(
                        call_id, tenant_id, scope_name, job_id, stage_id, provider, model, operation, status,
                        request_json, response_json, error, input_tokens, output_tokens, total_tokens, cost_micros,
                        cache_hit_tokens, cache_miss_tokens, usage_state, price_version, key_id,
                        client_platform, integration_id, agent_id, attribution_source,
                        request_sha256, response_sha256, started_at, finished_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        call_id, tenant_id, scope_name, job_id, stage_id, provider, model, operation,
                        "started", _json_value(request), None, error, input_tokens, output_tokens,
                        total_tokens, None, cache_hit_tokens, cache_miss_tokens, usage_state,
                        price_version, key_id,
                        usage_attribution.client_platform,
                        usage_attribution.integration_id,
                        usage_attribution.agent_id,
                        usage_attribution.attribution_source,
                        request_sha256, None, started_at, None, created,
                    ),
                )
            else:
                self._validate_provider_identity(
                    existing,
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    job_id=job_id,
                    stage_id=stage_id,
                    provider=provider,
                    model=model,
                    operation=operation,
                    key_id=key_id,
                    usage_attribution=usage_attribution,
                    request_sha256=request_sha256,
                )
                current = str(existing["status"])
                if current != "started" and status not in {current, "started"}:
                    raise JobStateError(
                        f"provider call {call_id} cannot transition {current} -> {status}"
                    )
            if status != "started":
                self._transition_provider_call_in_connection(
                    connection, call_id, status,
                    response=_json_value(response), error=error,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    total_tokens=total_tokens, cost_micro_cny=cost_micro_cny,
                    cache_hit_tokens=cache_hit_tokens, cache_miss_tokens=cache_miss_tokens,
                    usage_state=usage_state, price_version=price_version,
                    response_sha256=response_sha256, finished_at=finished_at,
                )
            row = connection.execute("SELECT * FROM provider_calls WHERE call_id=?", (call_id,)).fetchone()
        return _provider_call_from_row(row)

    @staticmethod
    def _validate_provider_identity(
        row: Any,
        *,
        tenant_id: str,
        scope_name: str,
        job_id: str | None,
        stage_id: str | None,
        provider: str,
        model: str,
        operation: str | None,
        key_id: str | None,
        usage_attribution: UsageAttribution,
        request_sha256: str | None,
    ) -> None:
        expected = {
            "tenant_id": tenant_id,
            "scope_name": scope_name,
            "job_id": job_id,
            "stage_id": stage_id,
            "provider": provider,
            "model": model,
            "operation": operation,
            "key_id": key_id,
            "client_platform": usage_attribution.client_platform,
            "integration_id": usage_attribution.integration_id,
            "agent_id": usage_attribution.agent_id,
            "attribution_source": usage_attribution.attribution_source,
            "request_sha256": request_sha256,
        }
        strict_attribution_columns = {
            "client_platform",
            "integration_id",
            "agent_id",
            "attribution_source",
        }
        for column, value in expected.items():
            if column in strict_attribution_columns:
                actual = row[column]
                if actual != value:
                    raise JobStateError(
                        f"provider call identity is immutable: {column}"
                    )
                continue
            if value is not None and row[column] is not None and str(row[column]) != str(value):
                raise JobStateError(f"provider call identity is immutable: {column}")

    @staticmethod
    def _transition_provider_call_in_connection(
        connection: Any,
        call_id: str,
        status: str,
        *,
        response: str | None,
        error: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        cost_micro_cny: int | None,
        cache_hit_tokens: int | None,
        cache_miss_tokens: int | None,
        usage_state: str,
        price_version: str | None,
        response_sha256: str | None,
        finished_at: float | None,
    ) -> None:
        if status not in {"completed", "failed", "unknown"}:
            raise ValueError("provider terminal status must be completed, failed, or unknown")
        row = connection.execute(
            "SELECT status FROM provider_calls WHERE call_id=?", (call_id,)
        ).fetchone()
        if row is None:
            raise JobNotFound(call_id)
        current = str(row["status"])
        if current == status:
            return
        if current != "started":
            raise JobStateError(f"provider call {call_id} cannot transition {current} -> {status}")
        moment = time.time() if finished_at is None else float(finished_at)
        connection.execute(
            """
            UPDATE provider_calls SET
                status=?, response_json=?, error=?, input_tokens=?, output_tokens=?, total_tokens=?,
                cost_micros=?, cache_hit_tokens=?, cache_miss_tokens=?, usage_state=?, price_version=?,
                response_sha256=?, finished_at=?
            WHERE call_id=? AND status='started'
            """,
            (
                status, response, error, input_tokens, output_tokens, total_tokens,
                None if status == "unknown" else cost_micro_cny, cache_hit_tokens,
                cache_miss_tokens, usage_state, price_version, response_sha256, moment, call_id,
            ),
        )

    def transition_provider_call(
        self,
        call_id: str,
        status: str,
        *,
        response: Any = None,
        error: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_micro_cny: int | None = None,
        cache_hit_tokens: int | None = None,
        cache_miss_tokens: int | None = None,
        usage_state: str = "missing",
        price_version: str | None = None,
        response_sha256: str | None = None,
        finished_at: float | None = None,
    ) -> ProviderCall:
        with self.db.transaction() as connection:
            self._transition_provider_call_in_connection(
                connection, call_id, status, response=_json_value(response), error=error,
                input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens,
                cost_micro_cny=cost_micro_cny, cache_hit_tokens=cache_hit_tokens,
                cache_miss_tokens=cache_miss_tokens, usage_state=usage_state,
                price_version=price_version, response_sha256=response_sha256,
                finished_at=finished_at,
            )
            row = connection.execute("SELECT * FROM provider_calls WHERE call_id=?", (call_id,)).fetchone()
        return _provider_call_from_row(row)

    def get_provider_call(self, call_id: str) -> ProviderCall | None:
        with self.db.transaction(immediate=False) as connection:
            row = connection.execute("SELECT * FROM provider_calls WHERE call_id=?", (call_id,)).fetchone()
        return None if row is None else _provider_call_from_row(row)

    def reconcile_committed_ingest_uncertain_calls(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        audit: Mapping[str, Any],
        reconciled_by: str,
    ) -> tuple[str, ...]:
        """Unblock derived work without rewriting an unknown cost outcome.

        This reconciliation concerns only the call's storage side effects. The
        provider call remains ``unknown`` so cost and usage reports continue to
        expose the uncertainty. A full immutable-Source audit plus operation
        watermarks must independently prove that the succeeded ingest is closed.
        """

        self.db._validate_scope(tenant_id, scope_name)
        actor = str(reconciled_by or "").strip()
        if not actor:
            raise ValueError("reconciled_by is required")
        failed_operations = tuple(
            sorted(str(value) for value in audit.get("failed_operation_ids", ()) if str(value))
        )
        audit_proof = {
            "integrity_ok": bool(audit.get("integrity_ok")),
            "ready_to_release": bool(audit.get("ready_to_release")),
            "source_count": int(audit.get("source_count", 0) or 0),
            "record_source_count": int(audit.get("record_source_count", 0) or 0),
            "failed_source_count": int(audit.get("failed_source_count", 0) or 0),
            "pending_source_count": int(audit.get("pending_source_count", 0) or 0),
            "prepared_message_commit_count": int(
                audit.get("prepared_message_commit_count", 0) or 0
            ),
            "failed_operation_ids": failed_operations,
        }
        if not (
            audit_proof["integrity_ok"]
            and audit_proof["ready_to_release"]
            and audit_proof["source_count"] > 0
            and audit_proof["source_count"] == audit_proof["record_source_count"]
            and audit_proof["failed_source_count"] == 0
            and audit_proof["pending_source_count"] == 0
            and audit_proof["prepared_message_commit_count"] == 0
            and not failed_operations
        ):
            raise JobStateError("immutable Source audit is not release-ready")

        reconciled: list[str] = []
        with self.db.transaction() as connection:
            calls = connection.execute(
                "SELECT calls.*,jobs.payload_json,jobs.state AS job_state,"
                "jobs.scope_seq FROM provider_calls AS calls "
                "JOIN jobs ON jobs.job_id=calls.job_id "
                "LEFT JOIN provider_call_reconciliations AS reconciliation "
                "ON reconciliation.call_id=calls.call_id "
                "WHERE calls.tenant_id=? AND calls.scope_name=? "
                "AND calls.status IN ('started','unknown') AND jobs.state=? "
                "AND reconciliation.call_id IS NULL ORDER BY calls.created_at",
                (tenant_id, scope_name, SUCCEEDED),
            ).fetchall()
            for call in calls:
                try:
                    payload = json.loads(str(call["payload_json"]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, Mapping) or str(
                    payload.get("job_type") or ""
                ) != "ingest":
                    continue
                job_id = str(call["job_id"] or "")
                source_rows = connection.execute(
                    "SELECT accounting_operation_id,COUNT(*) AS total "
                    "FROM scope_source_event_commits WHERE tenant_id=? "
                    "AND scope_name=? AND origin_operation_id=? "
                    "GROUP BY accounting_operation_id",
                    (tenant_id, scope_name, job_id),
                ).fetchall()
                if not source_rows:
                    continue
                committed_source_count = 0
                operation_evidence: list[dict[str, Any]] = []
                proof_complete = True
                for source_row in source_rows:
                    operation_id = str(source_row["accounting_operation_id"] or "")
                    origin_count = int(source_row["total"] or 0)
                    proof = connection.execute(
                        "SELECT source_set.source_count,source_set.source_set_sha256,"
                        "watermark.new_message_count,watermark.source_event_seq "
                        "FROM scope_ingest_source_sets AS source_set "
                        "JOIN scope_ingest_watermark_commits AS watermark "
                        "ON watermark.tenant_id=source_set.tenant_id "
                        "AND watermark.scope_name=source_set.scope_name "
                        "AND watermark.operation_id=source_set.operation_id "
                        "WHERE source_set.tenant_id=? AND source_set.scope_name=? "
                        "AND source_set.operation_id=?",
                        (tenant_id, scope_name, operation_id),
                    ).fetchone()
                    if (
                        proof is None
                        or origin_count < 1
                        or int(proof["source_count"] or 0) < origin_count
                        or int(proof["new_message_count"] or 0) != origin_count
                        or not str(proof["source_set_sha256"] or "")
                    ):
                        proof_complete = False
                        break
                    committed_source_count += origin_count
                    operation_evidence.append(
                        {
                            "operation_id": operation_id,
                            "origin_source_count": origin_count,
                            "source_set_count": int(proof["source_count"]),
                            "source_set_sha256": str(proof["source_set_sha256"]),
                            "source_event_seq": int(proof["source_event_seq"]),
                        }
                    )
                if not proof_complete or committed_source_count < 1:
                    continue
                evidence = {
                    "schema_version": "provider-side-effect-reconciliation-v1",
                    "call_status_preserved": str(call["status"]),
                    "job_state": SUCCEEDED,
                    "job_id": job_id,
                    "request_sha256": str(call["request_sha256"] or ""),
                    "committed_source_count": committed_source_count,
                    "operations": operation_evidence,
                    "scope_audit": audit_proof,
                }
                encoded = json.dumps(
                    evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                )
                digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
                moment = time.time()
                connection.execute(
                    "INSERT INTO provider_call_reconciliations("
                    "call_id,tenant_id,scope_name,job_id,reconciliation_kind,"
                    "evidence_json,evidence_sha256,reconciled_at) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        str(call["call_id"]), tenant_id, scope_name, job_id,
                        "committed_ingest_projection", encoded, digest, moment,
                    ),
                )
                self.db._append_job_lifecycle_audit(
                    connection,
                    job_id=job_id,
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    scope_seq=int(call["scope_seq"]),
                    event_type="provider_call_side_effect_reconciled",
                    reason={
                        "code": "provider_call_side_effect_reconciled",
                        "call_id": str(call["call_id"]),
                        "call_status_preserved": str(call["status"]),
                        "committed_source_count": committed_source_count,
                        "evidence_sha256": digest,
                    },
                    worker_id=actor,
                    created_at=moment,
                )
                reconciled.append(str(call["call_id"]))
        return tuple(reconciled)

    def upsert_provider_price(
        self,
        provider: str,
        model: str,
        *,
        input_micro_cny_per_million: int | None = None,
        cache_hit_input_micro_cny_per_million: int | None = None,
        cache_miss_input_micro_cny_per_million: int | None = None,
        output_micro_cny_per_million: int | None = None,
        effective_at: float,
        currency: str = "CNY",
        metadata: Any = None,
    ) -> ProviderPrice:
        if not provider or not model or not currency:
            raise ValueError("provider, model, and currency are required")
        if cache_miss_input_micro_cny_per_million is None:
            cache_miss_input_micro_cny_per_million = input_micro_cny_per_million
        if input_micro_cny_per_million is None:
            input_micro_cny_per_million = cache_miss_input_micro_cny_per_million
        now = time.time()
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO provider_prices(
                    provider, model, currency, input_micros_per_million, output_micros_per_million,
                    cache_hit_input_micros_per_million, cache_miss_input_micros_per_million,
                    effective_at, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, model, effective_at) DO UPDATE SET
                    currency=excluded.currency, input_micros_per_million=excluded.input_micros_per_million,
                    cache_hit_input_micros_per_million=excluded.cache_hit_input_micros_per_million,
                    cache_miss_input_micros_per_million=excluded.cache_miss_input_micros_per_million,
                    output_micros_per_million=excluded.output_micros_per_million,
                    metadata_json=excluded.metadata_json, updated_at=excluded.updated_at
                """,
                (
                    provider, model, currency, input_micro_cny_per_million,
                    output_micro_cny_per_million, cache_hit_input_micro_cny_per_million,
                    cache_miss_input_micro_cny_per_million, float(effective_at), _json_value(metadata), now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM provider_prices WHERE provider=? AND model=? AND effective_at=?",
                (provider, model, float(effective_at)),
            ).fetchone()
        return _provider_price_from_row(row)

    def get_provider_price(self, provider: str, model: str, *, at: float | None = None) -> ProviderPrice | None:
        moment = time.time() if at is None else float(at)
        with self.db.transaction(immediate=False) as connection:
            row = connection.execute(
                """
                SELECT * FROM provider_prices
                WHERE provider=? AND model=? AND effective_at<=?
                ORDER BY effective_at DESC LIMIT 1
                """,
                (provider, model, moment),
            ).fetchone()
        return None if row is None else _provider_price_from_row(row)

    def usage_cost_summary(
        self,
        tenant_id: str,
        *,
        scope_name: str | None = None,
        scope_prefix: str | None = None,
        from_timestamp: float | None = None,
        to_timestamp: float | None = None,
        group_by: str | None = None,
    ) -> dict[str, Any]:
        """Return registered model usage without pretending unknown calls cost zero."""
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if scope_name is not None and scope_prefix is not None:
            raise ValueError("scope_name and scope_prefix are mutually exclusive")
        if scope_name is not None:
            self.db._validate_scope(tenant_id, scope_name)
        if scope_prefix is not None:
            self.db._validate_scope(tenant_id, scope_prefix)
        if from_timestamp is not None and to_timestamp is not None:
            if float(from_timestamp) >= float(to_timestamp):
                raise ValueError("from_timestamp must be earlier than to_timestamp")
        allowed_groups = {
            None,
            "day",
            "scope",
            "stage",
            "operation",
            "provider",
            "model",
            "platform",
            "integration",
            "agent",
            "attribution_source",
        }
        if group_by not in allowed_groups:
            raise ValueError("unsupported usage group")

        def add_scope_filter(
            predicates: list[str],
            parameters: list[Any],
            *,
            column: str,
        ) -> None:
            if scope_name is not None:
                predicates.append(f"{column}=?")
                parameters.append(scope_name)
                return
            if scope_prefix is None:
                return
            escaped_prefix = (
                scope_prefix.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            predicates.append(
                f"{column} LIKE ? ESCAPE '\\' "
                f"AND substr({column},1,length(?)) = ? COLLATE BINARY"
            )
            parameters.extend((escaped_prefix + "%", scope_prefix, scope_prefix))

        predicates = ["calls.tenant_id=?"]
        parameters: list[Any] = [tenant_id]
        add_scope_filter(predicates, parameters, column="calls.scope_name")
        if from_timestamp is not None:
            predicates.append("calls.created_at>=?")
            parameters.append(float(from_timestamp))
        if to_timestamp is not None:
            predicates.append("calls.created_at<?")
            parameters.append(float(to_timestamp))
        with self.db.transaction(immediate=False) as connection:
            rows = connection.execute(
                """
                SELECT calls.*,
                       COALESCE(stages.stage_name, calls.operation, 'unbound') AS ledger_stage
                FROM provider_calls AS calls
                LEFT JOIN operation_stages AS stages ON stages.stage_id=calls.stage_id
                WHERE """
                + " AND ".join(predicates)
                + " ORDER BY calls.created_at, calls.call_id",
                parameters,
            ).fetchall()
            source_predicates = ["tenant_id=?"]
            source_parameters: list[Any] = [tenant_id]
            add_scope_filter(
                source_predicates,
                source_parameters,
                column="scope_name",
            )
            has_source_window = (
                from_timestamp is not None or to_timestamp is not None
            )
            if has_source_window:
                if from_timestamp is not None:
                    source_predicates.append("committed_at>=?")
                    source_parameters.append(float(from_timestamp))
                if to_timestamp is not None:
                    source_predicates.append("committed_at<?")
                    source_parameters.append(float(to_timestamp))
                evolution = connection.execute(
                    """
                    SELECT COUNT(DISTINCT scope_name) AS scope_count,
                           COALESCE(SUM(raw_token_estimate), 0) AS raw_tokens,
                           COALESCE(SUM(user_turns), 0) AS user_turns,
                           COALESCE(SUM(new_message_count), 0) AS source_events
                    FROM scope_ingest_watermark_commits WHERE """
                    + " AND ".join(source_predicates),
                    source_parameters,
                ).fetchone()
                source_ledger_coverage = "operation_commits_only"
            else:
                evolution = connection.execute(
                    """
                    SELECT COUNT(*) AS scope_count,
                           COALESCE(SUM(source_raw_token_estimate), 0) AS raw_tokens,
                           COALESCE(SUM(source_user_turns), 0) AS user_turns,
                           COALESCE(SUM(source_event_seq), 0) AS source_events
                    FROM scope_evolution_state WHERE """
                    + " AND ".join(source_predicates),
                    source_parameters,
                ).fetchone()
                source_ledger_coverage = "scope_evolution_totals"

            quota_predicates = ["events.tenant_id=?"]
            quota_parameters: list[Any] = [tenant_id]
            add_scope_filter(
                quota_predicates,
                quota_parameters,
                column="events.scope_name",
            )
            if from_timestamp is not None:
                quota_predicates.append("created_at>=?")
                quota_parameters.append(float(from_timestamp))
            if to_timestamp is not None:
                quota_predicates.append("created_at<?")
                quota_parameters.append(float(to_timestamp))
            quota_rows = connection.execute(
                "SELECT events.* FROM usage_events AS events WHERE "
                + " AND ".join(quota_predicates)
                + " ORDER BY events.created_at, events.event_key",
                quota_parameters,
            ).fetchall()

        totals = {
            "registered_call_count": len(rows),
            "completed_call_count": 0,
            "failed_call_count": 0,
            "unknown_call_count": 0,
            "in_flight_call_count": 0,
            "unpriced_completed_call_count": 0,
            "input_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "output_tokens": 0,
            "known_cost_micro_cny": 0,
        }
        stages: dict[str, dict[str, Any]] = {}
        buckets: dict[str, dict[str, Any]] = {}

        def group_key(row: Any, stage_name: str) -> str | None:
            if group_by is None:
                return None
            if group_by == "day":
                return time.strftime(
                    "%Y-%m-%d", time.gmtime(float(row["created_at"]))
                )
            if group_by == "scope":
                return str(row["scope_name"])
            if group_by == "stage":
                return stage_name
            column = {
                "platform": "client_platform",
                "integration": "integration_id",
                "agent": "agent_id",
            }.get(group_by, group_by)
            return str(row[column] or "unattributed")

        def usage_group_key(row: Any) -> str | None:
            if group_by is None:
                return None
            if group_by == "day":
                return time.strftime(
                    "%Y-%m-%d", time.gmtime(float(row["created_at"]))
                )
            if group_by == "scope":
                return str(row["scope_name"] or "unattributed")
            column = {
                "platform": "client_platform",
                "integration": "integration_id",
                "agent": "agent_id",
                "attribution_source": "attribution_source",
            }.get(group_by)
            if column is None:
                return None
            return str(row[column] or "unattributed")

        def empty_bucket() -> dict[str, Any]:
            return {
                "registered_call_count": 0,
                "completed_call_count": 0,
                "failed_call_count": 0,
                "unknown_call_count": 0,
                "in_flight_call_count": 0,
                "unpriced_completed_call_count": 0,
                "input_tokens": 0,
                "cache_hit_tokens": 0,
                "cache_miss_tokens": 0,
                "output_tokens": 0,
                "known_cost_micro_cny": 0,
                "ingest_raw_tokens": 0,
                "recall_requests": 0,
            }
        for row in rows:
            status = str(row["status"])
            stage_name = str(row["ledger_stage"])
            stage = stages.setdefault(
                stage_name,
                {
                    "registered_call_count": 0,
                    "completed_call_count": 0,
                    "unknown_or_unpriced_call_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "known_cost_micro_cny": 0,
                },
            )
            stage["registered_call_count"] += 1
            key = group_key(row, stage_name)
            bucket = buckets.setdefault(key, empty_bucket()) if key is not None else None
            if bucket is not None:
                bucket["registered_call_count"] += 1
            if status == "completed":
                totals["completed_call_count"] += 1
                stage["completed_call_count"] += 1
                if bucket is not None:
                    bucket["completed_call_count"] += 1
            elif status == "failed":
                totals["failed_call_count"] += 1
                if bucket is not None:
                    bucket["failed_call_count"] += 1
            elif status == "unknown":
                totals["unknown_call_count"] += 1
                stage["unknown_or_unpriced_call_count"] += 1
                if bucket is not None:
                    bucket["unknown_call_count"] += 1
            elif status == "started":
                totals["in_flight_call_count"] += 1
                stage["unknown_or_unpriced_call_count"] += 1
                if bucket is not None:
                    bucket["in_flight_call_count"] += 1
            if status == "completed" and row["cost_micros"] is None:
                totals["unpriced_completed_call_count"] += 1
                stage["unknown_or_unpriced_call_count"] += 1
                if bucket is not None:
                    bucket["unpriced_completed_call_count"] += 1
            for column in (
                "input_tokens",
                "cache_hit_tokens",
                "cache_miss_tokens",
                "output_tokens",
            ):
                value = int(row[column] or 0)
                totals[column] += value
                if column in {"input_tokens", "output_tokens"}:
                    stage[column] += value
                if bucket is not None:
                    bucket[column] += value
            cost = int(row["cost_micros"] or 0)
            totals["known_cost_micro_cny"] += cost
            stage["known_cost_micro_cny"] += cost
            if bucket is not None:
                bucket["known_cost_micro_cny"] += cost

        quota_event_totals = {
            "ingest_raw_tokens": 0,
            "recall_requests": 0,
        }
        attribution_coverage: dict[str, dict[str, int]] = {
            source: {
                "provider_call_count": 0,
                "usage_event_count": 0,
                "ingest_raw_tokens": 0,
                "recall_requests": 0,
                "known_cost_micro_cny": 0,
            }
            for source in (
                "trusted_proxy",
                "client_reported",
                "system_derived",
                "unattributed",
            )
        }
        for row in rows:
            source = str(row["attribution_source"] or "unattributed")
            coverage = attribution_coverage.setdefault(
                source,
                {
                    "provider_call_count": 0,
                    "usage_event_count": 0,
                    "ingest_raw_tokens": 0,
                    "recall_requests": 0,
                    "known_cost_micro_cny": 0,
                },
            )
            coverage["provider_call_count"] += 1
            coverage["known_cost_micro_cny"] += int(row["cost_micros"] or 0)
        for row in quota_rows:
            metric = str(row["metric"])
            units = int(row["units"] or 0)
            if metric in quota_event_totals:
                quota_event_totals[metric] += units
            source = str(row["attribution_source"] or "unattributed")
            coverage = attribution_coverage.setdefault(
                source,
                {
                    "provider_call_count": 0,
                    "usage_event_count": 0,
                    "ingest_raw_tokens": 0,
                    "recall_requests": 0,
                    "known_cost_micro_cny": 0,
                },
            )
            coverage["usage_event_count"] += 1
            if metric in {"ingest_raw_tokens", "recall_requests"}:
                coverage[metric] += units
            key = usage_group_key(row)
            if key is not None:
                bucket = buckets.setdefault(key, empty_bucket())
                if metric in {"ingest_raw_tokens", "recall_requests"}:
                    bucket[metric] += units

        raw_tokens = int(evolution["raw_tokens"] or 0)
        known_cost_micro_cny = int(totals["known_cost_micro_cny"])
        uncertainty_count = (
            int(totals["unknown_call_count"])
            + int(totals["in_flight_call_count"])
            + int(totals["unpriced_completed_call_count"])
        )
        return {
            "tenant_id": tenant_id,
            "scope_name": scope_name,
            "scope_prefix": scope_prefix,
            "from_timestamp": from_timestamp,
            "to_timestamp": to_timestamp,
            "currency": "CNY",
            "ledger_coverage": "registered_calls_only",
            "source_ledger_coverage": source_ledger_coverage,
            "complete_for_registered_calls": uncertainty_count == 0,
            "source": {
                "scope_count": int(evolution["scope_count"] or 0),
                "ingested_raw_token_estimate": raw_tokens,
                "ingested_user_turns": int(evolution["user_turns"] or 0),
                "source_event_count": int(evolution["source_events"] or 0),
            },
            "calls": totals,
            "known_cost_cny": known_cost_micro_cny / 1_000_000,
            "known_model_api_cny_per_million_ingested_raw_tokens": (
                known_cost_micro_cny / raw_tokens if raw_tokens > 0 else None
            ),
            "uncertain_cost_call_count": uncertainty_count,
            "by_stage": stages,
            "quota_events": quota_event_totals,
            "quota_event_scope_coverage": {
                "ingest_raw_tokens": "scope_attributed",
                "recall_requests": "scope_attributed_since_usage_attribution_v1",
            },
            "attribution_coverage": attribution_coverage,
            "group_by": group_by,
            "buckets": [
                {
                    "key": key,
                    **value,
                    "known_cost_cny": int(value["known_cost_micro_cny"])
                    / 1_000_000,
                }
                for key, value in sorted(buckets.items())
            ],
        }

    def list_due_evolution_scopes(self, **kwargs: Any) -> list[dict[str, object]]:
        return self.db.list_due_scopes(**kwargs)

    def list_due_index_scopes(self, **kwargs: Any) -> list[dict[str, object]]:
        return self.db.list_due_index_scopes(**kwargs)

    def claim_scope_evolution_job(self, tenant_id: str, scope_name: str, job_id: str) -> bool:
        return self.db.claim_evolution_job(tenant_id, scope_name, job_id)

    def release_scope_evolution_job(self, tenant_id: str, scope_name: str, job_id: str) -> bool:
        return self.db.release_evolution_job(tenant_id, scope_name, job_id)

    def claim_scope_index_job(self, tenant_id: str, scope_name: str, job_id: str) -> bool:
        return self.db.claim_index_job(tenant_id, scope_name, job_id)

    def release_scope_index_job(self, tenant_id: str, scope_name: str, job_id: str) -> bool:
        return self.db.release_index_job(tenant_id, scope_name, job_id)

    def advance_evolution_watermarks(self, tenant_id: str, scope_name: str, **kwargs: Any) -> dict[str, object]:
        return self.db.advance_promoted_watermarks(tenant_id, scope_name, **kwargs)

    def advance_index_watermark(self, tenant_id: str, scope_name: str, **kwargs: Any) -> dict[str, object]:
        return self.db.advance_index_watermark(tenant_id, scope_name, **kwargs)

    def advance_delta_index_watermark(self, tenant_id: str, scope_name: str, **kwargs: Any) -> dict[str, object]:
        return self.db.advance_delta_index_watermark(tenant_id, scope_name, **kwargs)
