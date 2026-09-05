from __future__ import annotations

import hashlib
import math
import os
import secrets as token_source
import sqlite3
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence


class ProviderPoolError(RuntimeError):
    pass


class ProviderPoolExhausted(ProviderPoolError):
    pass


DEFAULT_BILLING_CIRCUIT_SECONDS = 900.0
DEFAULT_AUTH_CIRCUIT_SECONDS = 900.0


def _key_id(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:24]


def _configured_duration(name: str, default: float) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ProviderPoolError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ProviderPoolError(f"{name} must be positive")
    return value


def _initialize_provider_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS provider_keys (
            pool TEXT NOT NULL,
            key_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
            max_concurrency INTEGER NOT NULL CHECK(max_concurrency > 0),
            cooldown_until REAL NOT NULL DEFAULT 0,
            failure_streak INTEGER NOT NULL DEFAULT 0,
            success_count INTEGER NOT NULL DEFAULT 0,
            failure_count INTEGER NOT NULL DEFAULT 0,
            last_used_at REAL,
            updated_at REAL NOT NULL,
            PRIMARY KEY(pool, key_id),
            UNIQUE(pool, ordinal)
        );
        CREATE TABLE IF NOT EXISTS provider_leases (
            lease_token TEXT PRIMARY KEY,
            pool TEXT NOT NULL,
            key_id TEXT NOT NULL,
            owner TEXT NOT NULL,
            acquired_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            FOREIGN KEY(pool, key_id) REFERENCES provider_keys(pool, key_id)
        );
        CREATE INDEX IF NOT EXISTS provider_leases_lookup
            ON provider_leases(pool, key_id, expires_at);
        CREATE TABLE IF NOT EXISTS provider_circuits (
            pool TEXT PRIMARY KEY,
            circuit_kind TEXT NOT NULL
                CHECK(circuit_kind IN ('billing', 'auth')),
            opened_at REAL NOT NULL,
            open_until REAL NOT NULL,
            failure_count INTEGER NOT NULL DEFAULT 1
                CHECK(failure_count > 0),
            last_failure_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        """
    )


def _open_circuit(
    connection: sqlite3.Connection,
    *,
    pool: str,
    kind: str,
    now: float,
    duration_seconds: float,
) -> None:
    if kind not in {"billing", "auth"}:
        raise ProviderPoolError(f"unsupported provider circuit kind: {kind}")
    open_until = now + duration_seconds
    current = connection.execute(
        "SELECT circuit_kind, open_until, failure_count FROM provider_circuits "
        "WHERE pool=?",
        (pool,),
    ).fetchone()
    if current is None:
        connection.execute(
            """
            INSERT INTO provider_circuits(
                pool, circuit_kind, opened_at, open_until, failure_count,
                last_failure_at, updated_at
            ) VALUES (?, ?, ?, ?, 1, ?, ?)
            """,
            (pool, kind, now, open_until, now, now),
        )
        return
    current_until = float(current["open_until"])
    current_kind = str(current["circuit_kind"])
    selected_kind = kind
    if current_until > open_until or (
        current_until == open_until and current_kind == "billing"
    ):
        selected_kind = current_kind
    connection.execute(
        """
        UPDATE provider_circuits
        SET circuit_kind=?, open_until=MAX(open_until, ?),
            failure_count=?, last_failure_at=?, updated_at=?
        WHERE pool=?
        """,
        (
            selected_kind,
            open_until,
            int(current["failure_count"]) + 1,
            now,
            now,
            pool,
        ),
    )


@dataclass(frozen=True)
class ProviderAdmissionStatus:
    pool: str
    accepting_paid_work: bool
    reason: str | None
    retry_after_seconds: float
    circuit_kind: str | None
    circuit_open_until: float | None
    enabled_keys: int
    healthy_keys: int

    def as_dict(self) -> dict[str, int | float | str | bool | None]:
        return {
            "pool": self.pool,
            "accepting_paid_work": self.accepting_paid_work,
            "reason": self.reason,
            "retry_after_seconds": round(self.retry_after_seconds, 3),
            "circuit_kind": self.circuit_kind,
            "circuit_open_until": self.circuit_open_until,
            "enabled_keys": self.enabled_keys,
            "healthy_keys": self.healthy_keys,
        }


class ProviderCircuitBreaker:
    """Read-only admission view over persistent provider circuit state."""

    def __init__(self, database: Path, *, pool: str) -> None:
        self.database = database.resolve()
        self.pool = pool.strip()
        if not self.pool:
            raise ProviderPoolError("provider pool name is required")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            _initialize_provider_tables(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def status(
        self,
        *,
        connection: sqlite3.Connection | None = None,
        now: float | None = None,
    ) -> ProviderAdmissionStatus:
        checked_at = time.time() if now is None else float(now)
        owned_connection = connection is None
        active_connection = connection or self._connect()
        try:
            circuit = active_connection.execute(
                "SELECT circuit_kind, open_until FROM provider_circuits WHERE pool=?",
                (self.pool,),
            ).fetchone()
            keys = active_connection.execute(
                """
                SELECT
                    SUM(enabled) AS enabled,
                    SUM(CASE WHEN enabled=1 AND cooldown_until<=? THEN 1 ELSE 0 END)
                        AS healthy,
                    MIN(CASE WHEN enabled=1 AND cooldown_until>? THEN cooldown_until END)
                        AS next_ready
                FROM provider_keys WHERE pool=?
                """,
                (checked_at, checked_at, self.pool),
            ).fetchone()
        finally:
            if owned_connection:
                active_connection.close()

        enabled = int(keys["enabled"] or 0)
        healthy = int(keys["healthy"] or 0)
        if circuit is not None and float(circuit["open_until"]) > checked_at:
            kind = str(circuit["circuit_kind"])
            open_until = float(circuit["open_until"])
            return ProviderAdmissionStatus(
                pool=self.pool,
                accepting_paid_work=False,
                reason=f"provider_{kind}_circuit_open",
                retry_after_seconds=max(0.001, open_until - checked_at),
                circuit_kind=kind,
                circuit_open_until=open_until,
                enabled_keys=enabled,
                healthy_keys=healthy,
            )
        next_ready = keys["next_ready"]
        if enabled > 0 and healthy == 0:
            retry_after = (
                max(0.001, float(next_ready) - checked_at)
                if next_ready is not None
                else 5.0
            )
            return ProviderAdmissionStatus(
                pool=self.pool,
                accepting_paid_work=False,
                reason="provider_pool_cooldown",
                retry_after_seconds=retry_after,
                circuit_kind=None,
                circuit_open_until=None,
                enabled_keys=enabled,
                healthy_keys=healthy,
            )
        if enabled == 0:
            return ProviderAdmissionStatus(
                pool=self.pool,
                accepting_paid_work=False,
                reason="provider_keys_unavailable",
                retry_after_seconds=5.0,
                circuit_kind=None,
                circuit_open_until=None,
                enabled_keys=0,
                healthy_keys=0,
            )
        return ProviderAdmissionStatus(
            pool=self.pool,
            accepting_paid_work=True,
            reason=None,
            retry_after_seconds=0.0,
            circuit_kind=None,
            circuit_open_until=None,
            enabled_keys=enabled,
            healthy_keys=healthy,
        )


@dataclass(frozen=True)
class ProviderLease:
    pool: str
    key_id: str
    secret: str
    lease_token: str
    expires_at: float


class ProviderKeyPool:
    """Cross-process provider-key leases without persisting provider secrets."""

    def __init__(
        self,
        database: Path,
        *,
        pool: str,
        keys: Sequence[str],
        max_concurrency_per_key: int = 2,
        lease_seconds: float = 300,
        billing_circuit_seconds: float | None = None,
        auth_circuit_seconds: float | None = None,
    ) -> None:
        cleaned = [value.strip() for value in keys if value.strip()]
        if not cleaned or len(cleaned) != len(set(cleaned)):
            raise ProviderPoolError("provider key pool must be non-empty and unique")
        if (
            max_concurrency_per_key <= 0
            or not math.isfinite(float(lease_seconds))
            or lease_seconds <= 0
        ):
            raise ProviderPoolError("provider pool limits must be positive")
        self.database = database.resolve()
        self.pool = pool.strip()
        if not self.pool:
            raise ProviderPoolError("provider pool name is required")
        self._secrets = {_key_id(value): value for value in cleaned}
        self.max_concurrency_per_key = max_concurrency_per_key
        self.lease_seconds = lease_seconds
        self.billing_circuit_seconds = float(
            billing_circuit_seconds
            if billing_circuit_seconds is not None
            else _configured_duration(
                "TMCRA_PROVIDER_BILLING_CIRCUIT_SECONDS",
                DEFAULT_BILLING_CIRCUIT_SECONDS,
            )
        )
        self.auth_circuit_seconds = float(
            auth_circuit_seconds
            if auth_circuit_seconds is not None
            else _configured_duration(
                "TMCRA_PROVIDER_AUTH_CIRCUIT_SECONDS",
                DEFAULT_AUTH_CIRCUIT_SECONDS,
            )
        )
        if (
            not math.isfinite(self.billing_circuit_seconds)
            or self.billing_circuit_seconds <= 0
            or not math.isfinite(self.auth_circuit_seconds)
            or self.auth_circuit_seconds <= 0
        ):
            raise ProviderPoolError("provider circuit durations must be positive")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self._sync_keys()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            _initialize_provider_tables(connection)

    def _sync_keys(self) -> None:
        now = time.time()
        with self._transaction() as connection:
            # Move every persisted ordinal out of the desired range first. A
            # direct swap (for example [a, b] -> [b, a]) otherwise violates
            # UNIQUE(pool, ordinal) halfway through the upsert sequence.
            existing = connection.execute(
                "SELECT key_id, ordinal FROM provider_keys "
                "WHERE pool=? ORDER BY ordinal, key_id",
                (self.pool,),
            ).fetchall()
            temporary_start = max(
                (int(row["ordinal"]) for row in existing), default=-1
            ) + 1
            for offset, row in enumerate(existing):
                connection.execute(
                    "UPDATE provider_keys SET ordinal=? "
                    "WHERE pool=? AND key_id=?",
                    (temporary_start + offset, self.pool, str(row["key_id"])),
                )
            for ordinal, key_id in enumerate(self._secrets):
                connection.execute(
                    """
                    INSERT INTO provider_keys(
                        pool, key_id, ordinal, max_concurrency, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(pool, key_id) DO UPDATE SET
                        ordinal=excluded.ordinal,
                        enabled=1,
                        max_concurrency=excluded.max_concurrency,
                        updated_at=excluded.updated_at
                    """,
                    (self.pool, key_id, ordinal, self.max_concurrency_per_key, now),
                )
            placeholders = ",".join("?" for _ in self._secrets)
            connection.execute(
                f"UPDATE provider_keys SET enabled=0, updated_at=? "
                f"WHERE pool=? AND key_id NOT IN ({placeholders})",
                (now, self.pool, *self._secrets),
            )

    def acquire(self, *, owner: str) -> ProviderLease:
        lease_token = token_source.token_urlsafe(32)
        expires_at = 0.0
        with self._transaction() as connection:
            # Compute the lease window only after the write transaction is
            # acquired. Opening SQLite/WAL or waiting for another process can
            # otherwise consume the lease before it is durably published.
            now = time.time()
            expires_at = now + self.lease_seconds
            circuit = connection.execute(
                "SELECT open_until FROM provider_circuits "
                "WHERE pool=? AND open_until>?",
                (self.pool, now),
            ).fetchone()
            if circuit is not None:
                raise ProviderPoolExhausted(
                    f"provider circuit is open: {self.pool}"
                )
            connection.execute(
                "DELETE FROM provider_leases WHERE pool=? AND expires_at<=?",
                (self.pool, now),
            )
            row = connection.execute(
                """
                SELECT k.key_id
                FROM provider_keys AS k
                LEFT JOIN provider_leases AS l
                  ON l.pool=k.pool AND l.key_id=k.key_id AND l.expires_at>?
                WHERE k.pool=? AND k.enabled=1 AND k.cooldown_until<=?
                GROUP BY k.pool, k.key_id
                HAVING COUNT(l.lease_token) < k.max_concurrency
                ORDER BY COUNT(l.lease_token), COALESCE(k.last_used_at, 0), k.ordinal
                LIMIT 1
                """,
                (now, self.pool, now),
            ).fetchone()
            if row is None:
                raise ProviderPoolExhausted(f"provider pool is saturated: {self.pool}")
            key_id = str(row["key_id"])
            connection.execute(
                """
                INSERT INTO provider_leases(
                    lease_token, pool, key_id, owner, acquired_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (lease_token, self.pool, key_id, owner, now, expires_at),
            )
            connection.execute(
                "UPDATE provider_keys SET last_used_at=?, updated_at=? "
                "WHERE pool=? AND key_id=?",
                (now, now, self.pool, key_id),
            )
        return ProviderLease(
            pool=self.pool,
            key_id=key_id,
            secret=self._secrets[key_id],
            lease_token=lease_token,
            expires_at=expires_at,
        )

    def release(
        self,
        lease: ProviderLease,
        *,
        outcome: str,
        retry_after_seconds: float | None = None,
    ) -> None:
        if lease.pool != self.pool:
            raise ProviderPoolError("lease belongs to another provider pool")
        if outcome not in {
            "success",
            "request_error",
            "rate_limited",
            "billing_exhausted",
            "transient_error",
            "fatal_error",
        }:
            raise ProviderPoolError(f"unsupported provider outcome: {outcome}")
        with self._transaction() as connection:
            now = time.time()
            row = connection.execute(
                "SELECT key_id, expires_at FROM provider_leases "
                "WHERE lease_token=? AND pool=?",
                (lease.lease_token, self.pool),
            ).fetchone()
            if row is None:
                # Expiry cleanup and a prior release are both valid terminal states.
                return
            if str(row["key_id"]) != lease.key_id:
                raise ProviderPoolError("provider lease is missing or mismatched")
            if float(row["expires_at"]) <= now:
                connection.execute(
                    "DELETE FROM provider_leases WHERE lease_token=? AND pool=?",
                    (lease.lease_token, self.pool),
                )
                return
            state = connection.execute(
                "SELECT failure_streak FROM provider_keys WHERE pool=? AND key_id=?",
                (self.pool, lease.key_id),
            ).fetchone()
            streak = int(state["failure_streak"]) if state is not None else 0
            if outcome == "success":
                connection.execute(
                    """
                    UPDATE provider_keys
                    SET success_count=success_count+1, failure_streak=0,
                        cooldown_until=0, updated_at=?
                    WHERE pool=? AND key_id=?
                    """,
                    (now, self.pool, lease.key_id),
                )
                connection.execute(
                    "DELETE FROM provider_circuits WHERE pool=? AND open_until<=?",
                    (self.pool, now),
                )
            elif outcome == "request_error":
                # A bad tenant request, response-contract failure, or local
                # ledger error says nothing about credential health. Releasing
                # it must not poison the shared key pool for other tenants.
                connection.execute(
                    "UPDATE provider_keys SET updated_at=? "
                    "WHERE pool=? AND key_id=?",
                    (now, self.pool, lease.key_id),
                )
            elif outcome == "rate_limited":
                # DeepSeek concurrency is account-scoped, not API-key-scoped.
                # Rotating to another key from the same account only amplifies
                # the 429 burst, so apply Retry-After to the whole pool.
                cooldown = max(1.0, float(retry_after_seconds or 5.0))
                connection.execute(
                    """
                    UPDATE provider_keys
                    SET cooldown_until=MAX(cooldown_until, ?), updated_at=?
                    WHERE pool=? AND enabled=1
                    """,
                    (now + cooldown, now, self.pool),
                )
                connection.execute(
                    """
                    UPDATE provider_keys
                    SET failure_count=failure_count+1
                    WHERE pool=? AND key_id=?
                    """,
                    (self.pool, lease.key_id),
                )
            elif outcome == "billing_exhausted":
                # DeepSeek balance is account-scoped. Rotating across keys from
                # the same account only burns requests and fails more jobs.
                cooldown = self.billing_circuit_seconds
                _open_circuit(
                    connection,
                    pool=self.pool,
                    kind="billing",
                    now=now,
                    duration_seconds=cooldown,
                )
                connection.execute(
                    """
                    UPDATE provider_keys
                    SET cooldown_until=MAX(cooldown_until, ?), updated_at=?
                    WHERE pool=? AND enabled=1
                    """,
                    (now + cooldown, now, self.pool),
                )
                connection.execute(
                    """
                    UPDATE provider_keys
                    SET failure_count=failure_count+1,
                        failure_streak=failure_streak+1
                    WHERE pool=? AND key_id=?
                    """,
                    (self.pool, lease.key_id),
                )
            elif outcome == "fatal_error":
                streak += 1
                cooldown = self.auth_circuit_seconds
                connection.execute(
                    """
                    UPDATE provider_keys
                    SET failure_count=failure_count+1, failure_streak=?,
                        cooldown_until=MAX(cooldown_until, ?), updated_at=?
                    WHERE pool=? AND key_id=?
                    """,
                    (streak, now + cooldown, now, self.pool, lease.key_id),
                )
                healthy = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM provider_keys "
                        "WHERE pool=? AND enabled=1 AND cooldown_until<=?",
                        (self.pool, now),
                    ).fetchone()[0]
                )
                if healthy == 0:
                    _open_circuit(
                        connection,
                        pool=self.pool,
                        kind="auth",
                        now=now,
                        duration_seconds=cooldown,
                    )
            else:
                streak += 1
                default_cooldown = min(900.0, 5.0 * (2 ** min(streak - 1, 7)))
                cooldown = max(default_cooldown, float(retry_after_seconds or 0))
                connection.execute(
                    """
                    UPDATE provider_keys
                    SET failure_count=failure_count+1, failure_streak=?,
                        cooldown_until=?, updated_at=?
                    WHERE pool=? AND key_id=?
                    """,
                    (streak, now + cooldown, now, self.pool, lease.key_id),
                )
            connection.execute(
                "DELETE FROM provider_leases WHERE lease_token=?",
                (lease.lease_token,),
            )

    def heartbeat(self, lease: ProviderLease) -> ProviderLease | None:
        """Extend an active lease, returning None when it was already lost."""
        if lease.pool != self.pool:
            raise ProviderPoolError("lease belongs to another provider pool")
        expires_at = 0.0
        with self._transaction() as connection:
            # As with acquisition, start the renewed lease after any database
            # lock wait, not before it.
            now = time.time()
            expires_at = now + self.lease_seconds
            row = connection.execute(
                "SELECT key_id, expires_at FROM provider_leases "
                "WHERE lease_token=? AND pool=?",
                (lease.lease_token, self.pool),
            ).fetchone()
            if row is None:
                return None
            if str(row["key_id"]) != lease.key_id:
                raise ProviderPoolError("provider lease is missing or mismatched")
            if float(row["expires_at"]) <= now:
                connection.execute(
                    "DELETE FROM provider_leases WHERE lease_token=? AND pool=?",
                    (lease.lease_token, self.pool),
                )
                return None
            updated = connection.execute(
                "UPDATE provider_leases SET expires_at=? "
                "WHERE lease_token=? AND pool=? AND key_id=? AND expires_at>?",
                (expires_at, lease.lease_token, self.pool, lease.key_id, now),
            )
            if updated.rowcount != 1:
                return None
        return ProviderLease(
            pool=self.pool,
            key_id=lease.key_id,
            secret=lease.secret,
            lease_token=lease.lease_token,
            expires_at=expires_at,
        )

    def stats(self) -> Mapping[str, int | float | str]:
        now = time.time()
        with closing(self._connect()) as connection:
            connection.execute(
                "DELETE FROM provider_leases WHERE pool=? AND expires_at<=?",
                (self.pool, now),
            )
            key_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(enabled) AS enabled,
                    SUM(CASE WHEN enabled=1 AND cooldown_until<=? THEN 1 ELSE 0 END)
                        AS healthy,
                    SUM(success_count) AS successes,
                    SUM(failure_count) AS failures
                FROM provider_keys WHERE pool=?
                """,
                (now, self.pool),
            ).fetchone()
            lease_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM provider_leases WHERE pool=? AND expires_at>?",
                    (self.pool, now),
                ).fetchone()[0]
            )
        admission = ProviderCircuitBreaker(
            self.database, pool=self.pool
        ).status(now=now)
        return {
            "pool": self.pool,
            "total_keys": int(key_row["total"] or 0),
            "enabled_keys": int(key_row["enabled"] or 0),
            "healthy_keys": int(key_row["healthy"] or 0),
            "active_leases": lease_count,
            "successes": int(key_row["successes"] or 0),
            "failures": int(key_row["failures"] or 0),
            "accepting_paid_work": admission.accepting_paid_work,
            "admission_reason": admission.reason or "",
            "retry_after_seconds": round(admission.retry_after_seconds, 3),
            "circuit_kind": admission.circuit_kind or "",
        }
