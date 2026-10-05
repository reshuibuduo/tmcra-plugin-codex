"""SQLite-persisted concurrency and per-minute pressure gating."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from .control_db import ControlDB


@dataclass(frozen=True)
class GateDecision:
    granted: bool
    reason: str
    lease_id: str | None = None
    retry_after: float = 0.0

    def __bool__(self) -> bool:
        return self.granted


class PressureGate:
    """Atomically enforce both active leases and fixed UTC minute buckets."""

    def __init__(
        self,
        db: ControlDB,
        *,
        max_concurrency: int,
        per_minute: int,
        lease_seconds: float = 60.0,
    ) -> None:
        if max_concurrency <= 0 or per_minute <= 0 or lease_seconds <= 0:
            raise ValueError("gate limits must be positive")
        self.db = db
        self.max_concurrency = max_concurrency
        self.per_minute = per_minute
        self.lease_seconds = float(lease_seconds)

    def acquire(self, tenant_id: str, *, now: float | None = None) -> GateDecision:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        current = time.time() if now is None else float(now)
        bucket = int(current // 60)
        lease_id = uuid.uuid4().hex
        expires = current + self.lease_seconds
        with self.db.transaction() as connection:
            connection.execute(
                "DELETE FROM rate_limit_minute WHERE tenant_id=? AND bucket_start < ?",
                (tenant_id, bucket - 1),
            )
            connection.execute(
                "DELETE FROM rate_limit_leases WHERE expires_at <= ?", (current,)
            )
            active = connection.execute(
                """
                SELECT COUNT(*) AS count, MIN(expires_at) AS next_expiry
                FROM rate_limit_leases WHERE tenant_id=?
                """,
                (tenant_id,),
            ).fetchone()
            if int(active["count"]) >= self.max_concurrency:
                retry = max(0.0, float(active["next_expiry"] or current) - current)
                return GateDecision(False, "concurrency", retry_after=retry)
            minute = connection.execute(
                """
                SELECT request_count FROM rate_limit_minute
                WHERE tenant_id=? AND bucket_start=?
                """,
                (tenant_id, bucket),
            ).fetchone()
            if minute is not None and int(minute["request_count"]) >= self.per_minute:
                return GateDecision(False, "per_minute", retry_after=max(0.0, 60.0 - current % 60.0))
            if minute is None:
                connection.execute(
                    "INSERT INTO rate_limit_minute(tenant_id, bucket_start, request_count) VALUES (?, ?, 1)",
                    (tenant_id, bucket),
                )
            else:
                connection.execute(
                    """
                    UPDATE rate_limit_minute SET request_count=request_count+1
                    WHERE tenant_id=? AND bucket_start=?
                    """,
                    (tenant_id, bucket),
                )
            connection.execute(
                "INSERT INTO rate_limit_leases(lease_id, tenant_id, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
                (lease_id, tenant_id, current, expires),
            )
        return GateDecision(True, "granted", lease_id=lease_id, retry_after=0.0)

    try_acquire = acquire

    def release(self, lease_id: str) -> bool:
        with self.db.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM rate_limit_leases WHERE lease_id = ?", (lease_id,)
            )
        return cursor.rowcount == 1

    def renew(self, lease_id: str, *, now: float | None = None) -> bool:
        current = time.time() if now is None else float(now)
        with self.db.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE rate_limit_leases SET expires_at=?
                WHERE lease_id=? AND expires_at>?
                """,
                (current + self.lease_seconds, lease_id, current),
            )
        return cursor.rowcount == 1


RateLimitGate = PressureGate
