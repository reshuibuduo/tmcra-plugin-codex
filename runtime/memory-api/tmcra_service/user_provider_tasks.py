"""Durable broker for provider calls executed on an authenticated user device."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .control_db import ControlDB


TASK_SCHEMA_VERSION = "tmcra.user-provider-task.1"
TERMINAL_STATES = frozenset({"completed", "failed", "unknown"})
ACTIVE_STATES = frozenset({"leased", "running"})
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_BYTES = 4 * 1024 * 1024


class UserProviderTaskError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        state: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.state = state
        self.metadata = dict(metadata or {})


class UserProviderTaskNotFound(UserProviderTaskError):
    def __init__(self, task_id: str) -> None:
        super().__init__(
            f"user-provider task was not found: {task_id}",
            code="user_provider_task_not_found",
        )


class UserProviderLeaseLost(UserProviderTaskError):
    def __init__(self, message: str = "user-provider task lease is no longer valid") -> None:
        super().__init__(message, code="user_provider_lease_lost")


@dataclass(frozen=True)
class UserProviderTask:
    task_id: str
    tenant_id: str
    scope_name: str
    auth_key_id: str
    job_id: str
    stage_id: str
    task_stage: str
    operation: str
    request: dict[str, Any]
    request_sha256: str
    state: str
    lease_expires_at: float | None
    provider: str | None
    model: str | None
    output: dict[str, Any] | None
    response_sha256: str | None
    usage: dict[str, int] | None
    provider_request_id: str | None
    error_code: str | None
    provider_started_at: float | None
    provider_finished_at: float | None
    created_at: float
    updated_at: float
    completed_at: float | None
    version: int


def _json(value: Any, *, maximum: int, label: str) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain JSON values") from exc
    if len(encoded.encode("utf-8")) > maximum:
        raise ValueError(f"{label} is too large")
    return encoded


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _optional_json_object(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = json.loads(str(raw))
    if not isinstance(value, dict):
        raise UserProviderTaskError(
            "user-provider task stored a non-object JSON value",
            code="user_provider_task_corrupt",
        )
    return value


def _task_from_row(row: Any) -> UserProviderTask:
    request = _optional_json_object(row["request_json"])
    if request is None:
        raise UserProviderTaskError(
            "user-provider task request is missing",
            code="user_provider_task_corrupt",
        )
    usage = _optional_json_object(row["usage_json"])
    return UserProviderTask(
        task_id=str(row["task_id"]),
        tenant_id=str(row["tenant_id"]),
        scope_name=str(row["scope_name"]),
        auth_key_id=str(row["auth_key_id"]),
        job_id=str(row["job_id"]),
        stage_id=str(row["stage_id"]),
        task_stage=str(row["task_stage"]),
        operation=str(row["operation"]),
        request=request,
        request_sha256=str(row["request_sha256"]),
        state=str(row["state"]),
        lease_expires_at=(
            None
            if row["lease_expires_at"] is None
            else float(row["lease_expires_at"])
        ),
        provider=None if row["provider"] is None else str(row["provider"]),
        model=None if row["model"] is None else str(row["model"]),
        output=_optional_json_object(row["output_json"]),
        response_sha256=(
            None if row["response_sha256"] is None else str(row["response_sha256"])
        ),
        usage=(
            None
            if usage is None
            else {str(key): int(value) for key, value in usage.items()}
        ),
        provider_request_id=(
            None
            if row["provider_request_id"] is None
            else str(row["provider_request_id"])
        ),
        error_code=(
            None if row["error_code"] is None else str(row["error_code"])
        ),
        provider_started_at=(
            None
            if row["provider_started_at"] is None
            else float(row["provider_started_at"])
        ),
        provider_finished_at=(
            None
            if row["provider_finished_at"] is None
            else float(row["provider_finished_at"])
        ),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        completed_at=(
            None if row["completed_at"] is None else float(row["completed_at"])
        ),
        version=int(row["version"]),
    )


class UserProviderTaskStore:
    def __init__(self, db: ControlDB, *, lease_seconds: float = 240.0) -> None:
        if not math.isfinite(float(lease_seconds)) or lease_seconds <= 0:
            raise ValueError("user-provider task lease must be positive")
        self.db = db
        self.lease_seconds = float(lease_seconds)

    @staticmethod
    def _validate_identity(
        tenant_id: str,
        scope_name: str,
        auth_key_id: str,
        job_id: str,
        stage_id: str,
        task_stage: str,
        operation: str,
    ) -> None:
        values = {
            "tenant_id": tenant_id,
            "scope_name": scope_name,
            "auth_key_id": auth_key_id,
            "job_id": job_id,
            "stage_id": stage_id,
            "operation": operation,
        }
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError("user-provider task identity values are required")
        if any(len(value) > 200 for value in values.values()):
            raise ValueError("user-provider task identity value is too long")
        if task_stage not in {"writer", "organizer"}:
            raise ValueError("user-provider task stage is invalid")

    @staticmethod
    def _expire_in_connection(connection: Any, now: float) -> None:
        connection.execute(
            """
            UPDATE user_provider_tasks
            SET state='queued',lease_token_sha256=NULL,lease_expires_at=NULL,
                updated_at=?,version=version+1
            WHERE state='leased' AND lease_expires_at IS NOT NULL
              AND lease_expires_at<=?
            """,
            (now, now),
        )
        connection.execute(
            """
            UPDATE user_provider_tasks
            SET state='unknown',lease_token_sha256=NULL,lease_expires_at=NULL,
                error_code='lease_expired_after_provider_start',updated_at=?,
                provider_finished_at=COALESCE(provider_finished_at,?),
                completed_at=?,version=version+1
            WHERE state='running' AND lease_expires_at IS NOT NULL
              AND lease_expires_at<=?
            """,
            (now, now, now, now),
        )

    def create(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        auth_key_id: str,
        job_id: str,
        stage_id: str,
        task_stage: str,
        operation: str,
        request: Mapping[str, Any],
    ) -> UserProviderTask:
        self._validate_identity(
            tenant_id,
            scope_name,
            auth_key_id,
            job_id,
            stage_id,
            task_stage,
            operation,
        )
        request_json = _json(dict(request), maximum=MAX_REQUEST_BYTES, label="task request")
        request_sha256 = _sha256(request_json)
        identity_json = _json(
            {
                "tenant_id": tenant_id,
                "scope_name": scope_name,
                "auth_key_id": auth_key_id,
                "job_id": job_id,
                "stage_id": stage_id,
                "task_stage": task_stage,
                "operation": operation,
                "request_sha256": request_sha256,
            },
            maximum=16_384,
            label="task identity",
        )
        task_id = "upt_" + _sha256(identity_json)[:48]
        now = time.time()
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO user_provider_tasks(
                    task_id,tenant_id,scope_name,auth_key_id,job_id,stage_id,
                    task_stage,operation,request_json,request_sha256,state,
                    created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?, 'queued',?,?)
                """,
                (
                    task_id,
                    tenant_id,
                    scope_name,
                    auth_key_id,
                    job_id,
                    stage_id,
                    task_stage,
                    operation,
                    request_json,
                    request_sha256,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM user_provider_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise UserProviderTaskError(
                    "user-provider task could not be created",
                    code="user_provider_task_create_failed",
                )
            immutable = (
                str(row["tenant_id"]),
                str(row["scope_name"]),
                str(row["auth_key_id"]),
                str(row["job_id"]),
                str(row["stage_id"]),
                str(row["task_stage"]),
                str(row["operation"]),
                str(row["request_sha256"]),
                str(row["request_json"]),
            )
            expected = (
                tenant_id,
                scope_name,
                auth_key_id,
                job_id,
                stage_id,
                task_stage,
                operation,
                request_sha256,
                request_json,
            )
            if immutable != expected:
                raise UserProviderTaskError(
                    "user-provider task identity collision",
                    code="user_provider_task_identity_conflict",
                )
        return _task_from_row(row)

    def get(self, task_id: str) -> UserProviderTask | None:
        with self.db.transaction(immediate=False) as connection:
            row = connection.execute(
                "SELECT * FROM user_provider_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return None if row is None else _task_from_row(row)

    def claim_next(
        self,
        *,
        tenant_id: str,
        auth_key_id: str,
        task_stage: str,
        scope_allowed: Callable[[str], bool],
    ) -> tuple[UserProviderTask, str] | None:
        if task_stage not in {"writer", "organizer"}:
            raise ValueError("user-provider task stage is invalid")
        now = time.time()
        with self.db.transaction() as connection:
            self._expire_in_connection(connection, now)
            rows = connection.execute(
                """
                SELECT * FROM user_provider_tasks
                WHERE tenant_id=? AND auth_key_id=? AND task_stage=?
                  AND state='queued'
                ORDER BY created_at,task_id LIMIT 100
                """,
                (tenant_id, auth_key_id, task_stage),
            ).fetchall()
            row = next(
                (candidate for candidate in rows if scope_allowed(str(candidate["scope_name"]))),
                None,
            )
            if row is None:
                return None
            lease_token = secrets.token_urlsafe(36)
            lease_sha256 = _sha256(lease_token)
            expires_at = now + self.lease_seconds
            changed = connection.execute(
                """
                UPDATE user_provider_tasks
                SET state='leased',lease_token_sha256=?,lease_expires_at=?,
                    updated_at=?,version=version+1
                WHERE task_id=? AND state='queued' AND version=?
                """,
                (lease_sha256, expires_at, now, row["task_id"], row["version"]),
            )
            if changed.rowcount != 1:
                return None
            claimed = connection.execute(
                "SELECT * FROM user_provider_tasks WHERE task_id=?", (row["task_id"],)
            ).fetchone()
        return _task_from_row(claimed), lease_token

    @staticmethod
    def _lease_matches(row: Any, lease_token: str, now: float) -> bool:
        expected = str(row["lease_token_sha256"] or "")
        expires_at = row["lease_expires_at"]
        return bool(
            expected
            and secrets.compare_digest(expected, _sha256(lease_token))
            and expires_at is not None
            and float(expires_at) > now
        )

    def _owned_row(
        self,
        connection: Any,
        *,
        task_id: str,
        tenant_id: str,
        auth_key_id: str,
    ) -> Any:
        row = connection.execute(
            """
            SELECT * FROM user_provider_tasks
            WHERE task_id=? AND tenant_id=? AND auth_key_id=?
            """,
            (task_id, tenant_id, auth_key_id),
        ).fetchone()
        if row is None:
            raise UserProviderTaskNotFound(task_id)
        return row

    def start(
        self,
        task_id: str,
        *,
        tenant_id: str,
        auth_key_id: str,
        lease_token: str,
    ) -> tuple[UserProviderTask, bool]:
        now = time.time()
        with self.db.transaction() as connection:
            self._expire_in_connection(connection, now)
            row = self._owned_row(
                connection,
                task_id=task_id,
                tenant_id=tenant_id,
                auth_key_id=auth_key_id,
            )
            if row["state"] == "running" and self._lease_matches(row, lease_token, now):
                return _task_from_row(row), True
            if row["state"] != "leased" or not self._lease_matches(
                row, lease_token, now
            ):
                raise UserProviderLeaseLost()
            expires_at = now + self.lease_seconds
            connection.execute(
                """
                UPDATE user_provider_tasks
                SET state='running',provider_started_at=?,lease_expires_at=?,
                    updated_at=?,version=version+1
                WHERE task_id=? AND state='leased'
                """,
                (now, expires_at, now, task_id),
            )
            row = connection.execute(
                "SELECT * FROM user_provider_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return _task_from_row(row), False

    def heartbeat(
        self,
        task_id: str,
        *,
        tenant_id: str,
        auth_key_id: str,
        lease_token: str,
    ) -> UserProviderTask:
        now = time.time()
        with self.db.transaction() as connection:
            self._expire_in_connection(connection, now)
            row = self._owned_row(
                connection,
                task_id=task_id,
                tenant_id=tenant_id,
                auth_key_id=auth_key_id,
            )
            if row["state"] not in ACTIVE_STATES or not self._lease_matches(
                row, lease_token, now
            ):
                raise UserProviderLeaseLost()
            expires_at = now + self.lease_seconds
            connection.execute(
                """
                UPDATE user_provider_tasks
                SET lease_expires_at=?,updated_at=?,version=version+1
                WHERE task_id=?
                """,
                (expires_at, now, task_id),
            )
            row = connection.execute(
                "SELECT * FROM user_provider_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return _task_from_row(row)

    @staticmethod
    def _normalize_usage(usage: Mapping[str, Any] | None) -> dict[str, int] | None:
        if usage is None:
            return None
        normalized = {
            str(key): int(value)
            for key, value in usage.items()
            if value is not None
        }
        return normalized or None

    def complete(
        self,
        task_id: str,
        *,
        tenant_id: str,
        auth_key_id: str,
        lease_token: str,
        provider: str,
        model: str,
        output: Mapping[str, Any],
        usage: Mapping[str, Any] | None,
        provider_request_id: str | None,
    ) -> tuple[UserProviderTask, bool]:
        output_json = _json(dict(output), maximum=MAX_OUTPUT_BYTES, label="task output")
        response_sha256 = _sha256(output_json)
        normalized_usage = self._normalize_usage(usage)
        usage_json = (
            None
            if normalized_usage is None
            else _json(normalized_usage, maximum=16_384, label="task usage")
        )
        now = time.time()
        with self.db.transaction() as connection:
            self._expire_in_connection(connection, now)
            row = self._owned_row(
                connection,
                task_id=task_id,
                tenant_id=tenant_id,
                auth_key_id=auth_key_id,
            )
            if row["state"] == "completed":
                matches = (
                    str(row["provider"] or "") == provider
                    and str(row["model"] or "") == model
                    and str(row["output_json"] or "") == output_json
                    and (None if row["usage_json"] is None else str(row["usage_json"]))
                    == usage_json
                    and (
                        None
                        if row["provider_request_id"] is None
                        else str(row["provider_request_id"])
                    )
                    == provider_request_id
                )
                if not matches:
                    raise UserProviderTaskError(
                        "completed user-provider task has different immutable output",
                        code="user_provider_result_conflict",
                    )
                return _task_from_row(row), True
            if row["state"] != "running" or not self._lease_matches(
                row, lease_token, now
            ):
                raise UserProviderLeaseLost()
            connection.execute(
                """
                UPDATE user_provider_tasks
                SET state='completed',lease_token_sha256=NULL,lease_expires_at=NULL,
                    provider=?,model=?,output_json=?,response_sha256=?,usage_json=?,
                    provider_request_id=?,provider_finished_at=?,completed_at=?,
                    updated_at=?,version=version+1
                WHERE task_id=? AND state='running'
                """,
                (
                    provider,
                    model,
                    output_json,
                    response_sha256,
                    usage_json,
                    provider_request_id,
                    now,
                    now,
                    now,
                    task_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM user_provider_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return _task_from_row(row), False

    def fail(
        self,
        task_id: str,
        *,
        tenant_id: str,
        auth_key_id: str,
        lease_token: str,
        provider: str,
        model: str,
        outcome: str,
        error_code: str,
    ) -> tuple[UserProviderTask, bool]:
        if outcome not in {"failed", "unknown"}:
            raise ValueError("user-provider failure outcome is invalid")
        now = time.time()
        with self.db.transaction() as connection:
            self._expire_in_connection(connection, now)
            row = self._owned_row(
                connection,
                task_id=task_id,
                tenant_id=tenant_id,
                auth_key_id=auth_key_id,
            )
            if row["state"] in {"failed", "unknown"}:
                matches = (
                    str(row["state"]) == outcome
                    and str(row["provider"] or "") == provider
                    and str(row["model"] or "") == model
                    and str(row["error_code"] or "") == error_code
                )
                if not matches:
                    raise UserProviderTaskError(
                        "terminal user-provider task has different failure identity",
                        code="user_provider_result_conflict",
                    )
                return _task_from_row(row), True
            if row["state"] not in ACTIVE_STATES or not self._lease_matches(
                row, lease_token, now
            ):
                raise UserProviderLeaseLost()
            connection.execute(
                """
                UPDATE user_provider_tasks
                SET state=?,lease_token_sha256=NULL,lease_expires_at=NULL,
                    provider=?,model=?,error_code=?,provider_finished_at=?,
                    completed_at=?,updated_at=?,version=version+1
                WHERE task_id=? AND state IN ('leased','running')
                """,
                (outcome, provider, model, error_code, now, now, now, task_id),
            )
            row = connection.execute(
                "SELECT * FROM user_provider_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return _task_from_row(row), False

    def await_terminal(
        self,
        task_id: str,
        *,
        timeout: float,
        poll_interval: float = 0.1,
    ) -> UserProviderTask:
        if not math.isfinite(float(timeout)) or timeout <= 0:
            raise ValueError("user-provider task timeout must be positive")
        deadline = time.monotonic() + float(timeout)
        while True:
            now = time.time()
            with self.db.transaction() as connection:
                self._expire_in_connection(connection, now)
                row = connection.execute(
                    "SELECT * FROM user_provider_tasks WHERE task_id=?", (task_id,)
                ).fetchone()
            if row is None:
                raise UserProviderTaskNotFound(task_id)
            task = _task_from_row(row)
            if task.state in TERMINAL_STATES:
                return task
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self.db.transaction() as connection:
                    row = connection.execute(
                        "SELECT * FROM user_provider_tasks WHERE task_id=?", (task_id,)
                    ).fetchone()
                    if row is not None and row["state"] not in TERMINAL_STATES:
                        terminal = "failed" if row["state"] == "queued" else "unknown"
                        error_code = (
                            "executor_unavailable"
                            if terminal == "failed"
                            else "executor_outcome_unresolved"
                        )
                        connection.execute(
                            """
                            UPDATE user_provider_tasks
                            SET state=?,lease_token_sha256=NULL,lease_expires_at=NULL,
                                error_code=?,provider_finished_at=COALESCE(
                                    provider_finished_at,?
                                ),completed_at=?,updated_at=?,version=version+1
                            WHERE task_id=? AND state NOT IN ('completed','failed','unknown')
                            """,
                            (terminal, error_code, now, now, now, task_id),
                        )
                        row = connection.execute(
                            "SELECT * FROM user_provider_tasks WHERE task_id=?", (task_id,)
                        ).fetchone()
                if row is None:
                    raise UserProviderTaskNotFound(task_id)
                return _task_from_row(row)
            time.sleep(min(max(0.01, poll_interval), remaining))
