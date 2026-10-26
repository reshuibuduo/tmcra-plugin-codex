"""User-facing scope catalog and quota accounting.

This module is deliberately independent of the TMCRA writer, planner, and
storage adapter.  It records API admission facts only; it never changes the
memory algorithm or treats provider-cost estimates as customer quota truth.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .control_db import ControlDB
from .usage_attribution import UNATTRIBUTED, UsageAttribution


QUOTA_METRICS = ("ingest_raw_tokens", "recall_requests")
PLAN_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
BILLING_GROUP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
BILLING_SUBJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,199}$")
BILLING_INTERVALS = frozenset({"monthly", "yearly", "custom"})
BILLING_GROUP_STATUSES = frozenset({"active", "suspended", "cancelled"})
BILLING_MEMBER_ROLES = frozenset({"owner", "admin", "member"})


@dataclass
class QuotaExceeded(Exception):
    metric: str
    used: int
    limit: int
    requested: int

    def __str__(self) -> str:
        return f"{self.metric} quota exceeded"


@dataclass
class BillingAccessDenied(Exception):
    reason: str
    group_id: str | None = None

    def __str__(self) -> str:
        return self.reason


class BillingConflict(ValueError):
    """A billing identifier was replayed with different immutable facts."""


class BillingNotFound(KeyError):
    """A referenced plan or billing group does not exist."""


def estimate_raw_tokens(messages: Iterable[Mapping[str, object]]) -> int:
    """Use the service writer's deterministic input-token estimate."""

    total = 0
    for message in messages:
        content = str(message.get("content") or "")
        non_empty = [character for character in content if not character.isspace()]
        cjk = sum(
            1
            for character in non_empty
            if any(
                start <= ord(character) <= end
                for start, end in (
                    (0x3400, 0x4DBF),
                    (0x4E00, 0x9FFF),
                    (0xF900, 0xFAFF),
                )
            )
        )
        total += cjk + (len(non_empty) - cjk + 3) // 4
    return total


class MemoryControlPlane:
    def __init__(self, database: ControlDB) -> None:
        self.database = database

    @staticmethod
    def principal(tenant_id: str, subject: str | None) -> str:
        return (
            MemoryControlPlane.subject_principal(subject)
            if subject is not None
            else MemoryControlPlane.tenant_principal(tenant_id)
        )

    @staticmethod
    def tenant_principal(tenant_id: str) -> str:
        clean = str(tenant_id).strip()
        if not clean:
            raise ValueError("tenant_id is required")
        return f"tenant:{clean}"

    @staticmethod
    def subject_principal(subject: str) -> str:
        clean = str(subject).strip()
        if not clean:
            raise ValueError("subject is required")
        return f"subject:{clean}"

    @staticmethod
    def billing_principal(group_id: str, period_id: str) -> str:
        clean_group = str(group_id).strip()
        clean_period = str(period_id).strip()
        if not BILLING_GROUP_ID_RE.fullmatch(clean_group) or not clean_period:
            raise ValueError("valid billing group and period IDs are required")
        return f"billing:{clean_group}:{clean_period}"

    @staticmethod
    def _effective_period_status(
        status: object, starts_at: object, ends_at: object, *, now: float | None = None
    ) -> str:
        stored = str(status)
        if stored != "active":
            return stored
        current = time.time() if now is None else float(now)
        if current < float(starts_at):
            return "scheduled"
        if current >= float(ends_at):
            return "expired"
        return "active"

    @staticmethod
    def _record_billing_member_event(
        connection: sqlite3.Connection,
        *,
        tenant_id: str,
        group_id: str,
        subject: str,
        role: str,
        event_type: str,
        created_by_key_id: str,
        created_at: float,
    ) -> None:
        connection.execute(
            """
            INSERT INTO billing_group_member_events(
                event_id,tenant_id,group_id,subject,role,event_type,
                created_by_key_id,created_at
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                uuid.uuid4().hex,
                tenant_id,
                group_id,
                subject,
                role,
                event_type,
                created_by_key_id,
                created_at,
            ),
        )

    def quota_identity(
        self,
        tenant_id: str,
        subject: str | None,
        *,
        require_active: bool = True,
    ) -> tuple[str, str, dict[str, object] | None]:
        """Resolve the shared quota owner and the concrete consuming member."""

        consumer = self.principal(tenant_id, subject)
        if subject is None:
            return consumer, consumer, None
        now = time.time()
        with closing(self.database.connect()) as connection:
            row = connection.execute(
                """
                SELECT m.group_id,m.role,g.display_name,g.status,
                       p.period_id,p.usage_principal,p.plan_code,p.plan_version,
                       p.billing_interval,p.starts_at,p.ends_at,p.status AS period_status,
                       p.max_members,p.currency,p.price_minor_units
                FROM billing_group_members AS m
                JOIN billing_groups AS g
                  ON g.tenant_id=m.tenant_id AND g.group_id=m.group_id
                JOIN billing_group_periods AS p
                  ON p.tenant_id=g.tenant_id AND p.group_id=g.group_id
                 AND p.period_id=g.active_period_id
                WHERE m.tenant_id=? AND m.subject=?
                """,
                (tenant_id, subject),
            ).fetchone()
        if row is None:
            return consumer, consumer, None
        effective_period_status = self._effective_period_status(
            row["period_status"], row["starts_at"], row["ends_at"], now=now
        )
        active = (
            str(row["status"]) == "active"
            and effective_period_status == "active"
        )
        if require_active and not active:
            reason = (
                "billing group is not active"
                if str(row["status"]) != "active"
                else "billing period is not active"
            )
            raise BillingAccessDenied(reason, str(row["group_id"]))
        billing = {key: row[key] for key in row.keys()}
        billing["period_status"] = effective_period_status
        return str(row["usage_principal"]), consumer, billing

    def consume_quota(
        self,
        tenant_id: str,
        principal: str,
        metric: str,
        units: int,
        event_key: str,
        *,
        consumer_principal: str | None = None,
        scope_name: str | None = None,
        usage_attribution: UsageAttribution = UNATTRIBUTED,
    ) -> bool:
        return bool(
            self.consume_quota_batch(
                tenant_id,
                principal,
                metric,
                [(event_key, units)],
                consumer_principal=consumer_principal,
                scope_name=scope_name,
                usage_attribution=usage_attribution,
            )
        )

    def consume_quota_batch(
        self,
        tenant_id: str,
        principal: str,
        metric: str,
        events: Sequence[tuple[str, int]],
        *,
        consumer_principal: str | None = None,
        scope_name: str | None = None,
        usage_attribution: UsageAttribution = UNATTRIBUTED,
    ) -> set[str]:
        with self.database.transaction() as connection:
            return self.consume_quota_batch_in_transaction(
                connection,
                tenant_id,
                principal,
                metric,
                events,
                consumer_principal=consumer_principal,
                scope_name=scope_name,
                usage_attribution=usage_attribution,
            )

    def consume_quota_batch_in_transaction(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        principal: str,
        metric: str,
        events: Sequence[tuple[str, int]],
        *,
        consumer_principal: str | None = None,
        scope_name: str | None = None,
        usage_attribution: UsageAttribution = UNATTRIBUTED,
    ) -> set[str]:
        """Reserve quota using the caller's active write transaction."""

        if metric not in QUOTA_METRICS:
            raise ValueError("unknown quota metric")
        if not tenant_id or not principal or not events:
            raise ValueError("tenant_id, principal, and events are required")
        if len({key for key, _units in events}) != len(events):
            raise ValueError("quota event keys must be unique")
        if any(not key or units < 0 for key, units in events):
            raise ValueError("quota event keys and non-negative units are required")
        now = time.time()
        consumer = str(consumer_principal or principal).strip()
        if not consumer or len(consumer) > 256:
            raise ValueError("consumer_principal must be 1-256 characters")
        new_events: list[tuple[str, int]] = []
        for event_key, units in events:
            existing = connection.execute(
                """
                SELECT units,consumer_principal,scope_name,client_platform,integration_id,agent_id,
                       attribution_source FROM usage_events
                WHERE tenant_id=? AND principal=? AND metric=? AND event_key=?
                """,
                (tenant_id, principal, metric, event_key),
            ).fetchone()
            if existing is not None:
                if int(existing["units"]) != int(units):
                    raise ValueError("quota event replay changed units")
                expected_attribution = (
                    consumer,
                    scope_name,
                    usage_attribution.client_platform,
                    usage_attribution.integration_id,
                    usage_attribution.agent_id,
                    usage_attribution.attribution_source,
                )
                actual_attribution = (
                    str(existing["consumer_principal"] or principal),
                    existing["scope_name"],
                    str(existing["client_platform"] or "unattributed"),
                    existing["integration_id"],
                    existing["agent_id"],
                    str(existing["attribution_source"] or "unattributed"),
                )
                if actual_attribution != expected_attribution:
                    raise ValueError("quota event replay changed attribution")
                continue
            new_events.append((event_key, int(units)))
        if not new_events:
            return set()
        usage_row = connection.execute(
            """
            SELECT used_units FROM usage_totals
            WHERE tenant_id=? AND principal=? AND metric=?
            """,
            (tenant_id, principal, metric),
        ).fetchone()
        used = 0 if usage_row is None else int(usage_row["used_units"])
        entitlement = connection.execute(
            """
            SELECT limit_units FROM usage_entitlements
            WHERE tenant_id=? AND principal=? AND metric=?
            """,
            (tenant_id, principal, metric),
        ).fetchone()
        limit = (
            None
            if entitlement is None or entitlement["limit_units"] is None
            else int(entitlement["limit_units"])
        )
        requested = sum(units for _key, units in new_events)
        if limit is not None and used + requested > limit:
            raise QuotaExceeded(metric, used, limit, requested)
        connection.executemany(
            """
            INSERT INTO usage_events(
                tenant_id,principal,consumer_principal,metric,event_key,units,scope_name,
                client_platform,integration_id,agent_id,attribution_source,
                created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    tenant_id,
                    principal,
                    consumer,
                    metric,
                    key,
                    units,
                    scope_name,
                    usage_attribution.client_platform,
                    usage_attribution.integration_id,
                    usage_attribution.agent_id,
                    usage_attribution.attribution_source,
                    now,
                )
                for key, units in new_events
            ],
        )
        connection.execute(
            """
            INSERT INTO usage_totals(
                tenant_id,principal,metric,used_units,updated_at
            ) VALUES(?,?,?,?,?)
            ON CONFLICT(tenant_id,principal,metric) DO UPDATE SET
                used_units=usage_totals.used_units + excluded.used_units,
                updated_at=excluded.updated_at
            """,
            (tenant_id, principal, metric, requested, now),
        )
        return {key for key, _units in new_events}

    def release_quota_events(
        self,
        tenant_id: str,
        principal: str,
        metric: str,
        event_keys: Iterable[str],
    ) -> None:
        keys = tuple(dict.fromkeys(str(key) for key in event_keys if key))
        if not keys:
            return
        now = time.time()
        with self.database.transaction() as connection:
            placeholders = ",".join("?" for _ in keys)
            rows = connection.execute(
                f"""
                SELECT event_key,units FROM usage_events
                WHERE tenant_id=? AND principal=? AND metric=?
                  AND event_key IN ({placeholders})
                """,
                (tenant_id, principal, metric, *keys),
            ).fetchall()
            released = sum(int(row["units"]) for row in rows)
            if not rows:
                return
            connection.execute(
                f"""
                DELETE FROM usage_events
                WHERE tenant_id=? AND principal=? AND metric=?
                  AND event_key IN ({placeholders})
                """,
                (tenant_id, principal, metric, *keys),
            )
            connection.execute(
                """
                UPDATE usage_totals
                SET used_units=MAX(0, used_units-?), updated_at=?
                WHERE tenant_id=? AND principal=? AND metric=?
                """,
                (released, now, tenant_id, principal, metric),
            )

    def admit_ingest_batch_in_transaction(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        principal: str,
        scope_name: str,
        entries: Sequence[tuple[str, str, int, int]],
        *,
        consumer_principal: str | None = None,
        usage_attribution: UsageAttribution = UNATTRIBUTED,
    ) -> None:
        """Atomically reserve quota and catalog newly admitted ingest jobs."""

        if not entries:
            return
        self.consume_quota_batch_in_transaction(
            connection,
            tenant_id,
            principal,
            "ingest_raw_tokens",
            [(key, raw_tokens) for key, _session, _messages, raw_tokens in entries],
            consumer_principal=consumer_principal,
            scope_name=scope_name,
            usage_attribution=usage_attribution,
        )
        now = time.time()
        for key, session_id, message_count, raw_token_count in entries:
            self._record_ingest_in_transaction(
                connection,
                tenant_id,
                scope_name,
                session_id,
                key,
                message_count=message_count,
                raw_token_count=raw_token_count,
                usage_attribution=usage_attribution,
                now=now,
            )

    def record_ingest(
        self,
        tenant_id: str,
        scope_name: str,
        session_id: str,
        idempotency_key: str,
        *,
        message_count: int,
        raw_token_count: int,
        usage_attribution: UsageAttribution = UNATTRIBUTED,
    ) -> bool:
        now = time.time()
        with self.database.transaction() as connection:
            return self._record_ingest_in_transaction(
                connection,
                tenant_id,
                scope_name,
                session_id,
                idempotency_key,
                message_count=message_count,
                raw_token_count=raw_token_count,
                usage_attribution=usage_attribution,
                now=now,
            )

    def _record_ingest_in_transaction(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        scope_name: str,
        session_id: str,
        idempotency_key: str,
        *,
        message_count: int,
        raw_token_count: int,
        usage_attribution: UsageAttribution,
        now: float,
    ) -> bool:
        existing = connection.execute(
            """
            SELECT scope_name,session_id,message_count,raw_token_count,
                   client_platform,integration_id,agent_id,attribution_source
            FROM scope_ingest_events
            WHERE tenant_id=? AND idempotency_key=?
            """,
            (tenant_id, idempotency_key),
        ).fetchone()
        if existing is not None:
            expected = (
                scope_name,
                session_id,
                message_count,
                raw_token_count,
                usage_attribution.client_platform,
                usage_attribution.integration_id,
                usage_attribution.agent_id,
                usage_attribution.attribution_source,
            )
            actual = (
                str(existing["scope_name"]),
                str(existing["session_id"]),
                int(existing["message_count"]),
                int(existing["raw_token_count"]),
                str(existing["client_platform"] or "unattributed"),
                existing["integration_id"],
                existing["agent_id"],
                str(existing["attribution_source"] or "unattributed"),
            )
            if actual != expected:
                raise ValueError("ingest accounting replay changed request facts")
            return False
        connection.execute(
            """
            INSERT INTO scope_ingest_events(
                tenant_id,idempotency_key,scope_name,session_id,
                message_count,raw_token_count,client_platform,integration_id,
                agent_id,attribution_source,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                tenant_id,
                idempotency_key,
                scope_name,
                session_id,
                message_count,
                raw_token_count,
                usage_attribution.client_platform,
                usage_attribution.integration_id,
                usage_attribution.agent_id,
                usage_attribution.attribution_source,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO scope_catalog(
                tenant_id,scope_name,created_at,last_seen_at,last_ingest_at,
                ingest_request_count,message_count
            ) VALUES(?,?,?,?,?,1,?)
            ON CONFLICT(tenant_id,scope_name) DO UPDATE SET
                last_seen_at=excluded.last_seen_at,
                last_ingest_at=excluded.last_ingest_at,
                ingest_request_count=scope_catalog.ingest_request_count+1,
                message_count=scope_catalog.message_count+excluded.message_count
            """,
            (tenant_id, scope_name, now, now, now, message_count),
        )
        connection.execute(
            """
            INSERT INTO scope_sessions(
                tenant_id,scope_name,session_id,created_at,last_ingest_at,
                ingest_request_count,message_count
            ) VALUES(?,?,?,?,?,1,?)
            ON CONFLICT(tenant_id,scope_name,session_id) DO UPDATE SET
                last_ingest_at=excluded.last_ingest_at,
                ingest_request_count=scope_sessions.ingest_request_count+1,
                message_count=scope_sessions.message_count+excluded.message_count
            """,
            (tenant_id, scope_name, session_id, now, now, message_count),
        )
        return True

    def record_recall(self, tenant_id: str, scope_name: str) -> None:
        now = time.time()
        with self.database.transaction() as connection:
            self._record_recall_in_transaction(
                connection, tenant_id, scope_name, now=now
            )

    @staticmethod
    def _record_recall_in_transaction(
        connection: sqlite3.Connection,
        tenant_id: str,
        scope_name: str,
        *,
        now: float,
    ) -> None:
        connection.execute(
            """
            INSERT INTO scope_catalog(
                tenant_id,scope_name,created_at,last_seen_at,last_recall_at,
                recall_request_count
            ) VALUES(?,?,?,?,?,1)
            ON CONFLICT(tenant_id,scope_name) DO UPDATE SET
                last_seen_at=excluded.last_seen_at,
                last_recall_at=excluded.last_recall_at,
                recall_request_count=scope_catalog.recall_request_count+1
            """,
            (tenant_id, scope_name, now, now, now),
        )

    def admit_recall(
        self,
        tenant_id: str,
        principal: str,
        scope_name: str,
        event_key: str,
        *,
        consumer_principal: str | None = None,
        usage_attribution: UsageAttribution = UNATTRIBUTED,
    ) -> None:
        """Atomically admit and catalog a recall that is ready to execute."""

        now = time.time()
        with self.database.transaction() as connection:
            self.consume_quota_batch_in_transaction(
                connection,
                tenant_id,
                principal,
                "recall_requests",
                [(event_key, 1)],
                consumer_principal=consumer_principal,
                scope_name=scope_name,
                usage_attribution=usage_attribution,
            )
            self._record_recall_in_transaction(
                connection, tenant_id, scope_name, now=now
            )

    def backfill_catalog_from_jobs(self) -> None:
        """Populate the user catalog once from the existing job ledger."""

        migration_id = "scope_catalog_from_jobs_v1"
        with self.database.transaction() as connection:
            applied = connection.execute(
                "SELECT 1 FROM control_migrations WHERE migration_id=?",
                (migration_id,),
            ).fetchone()
            if applied is not None:
                return
            deleted = {
                (str(row["tenant_id"]), str(row["scope_name"]))
                for row in connection.execute(
                    "SELECT tenant_id,scope_name FROM scope_lifecycle WHERE state='deleted'"
                ).fetchall()
            }
            rows = connection.execute(
                """
                SELECT tenant_id,scope_name,idempotency_key,payload_json,
                       created_at,updated_at
                FROM jobs
                ORDER BY created_at,job_id
                """
            ).fetchall()
            for row in rows:
                tenant_id = str(row["tenant_id"])
                scope_name = str(row["scope_name"] or "default")
                if (tenant_id, scope_name) in deleted:
                    continue
                try:
                    payload = json.loads(str(row["payload_json"]))
                except (TypeError, ValueError):
                    payload = {}
                payload = payload if isinstance(payload, dict) else {}
                created_at = float(row["created_at"])
                updated_at = float(row["updated_at"] or row["created_at"])
                if payload.get("job_type") == "ingest":
                    session_id = str(payload.get("session_id") or "").strip()
                    messages_value = payload.get("messages")
                    messages = messages_value if isinstance(messages_value, list) else []
                    mapped_messages = [
                        item for item in messages if isinstance(item, Mapping)
                    ]
                    if session_id:
                        self._record_ingest_in_transaction(
                            connection,
                            tenant_id,
                            scope_name,
                            session_id,
                            str(row["idempotency_key"]),
                            message_count=len(messages),
                            raw_token_count=estimate_raw_tokens(mapped_messages),
                            usage_attribution=UsageAttribution.from_mapping(
                                payload.get("_usage_attribution")
                                if isinstance(
                                    payload.get("_usage_attribution"), Mapping
                                )
                                else None
                            ),
                            now=created_at,
                        )
                        continue
                connection.execute(
                    """
                    INSERT INTO scope_catalog(
                        tenant_id,scope_name,created_at,last_seen_at
                    ) VALUES(?,?,?,?)
                    ON CONFLICT(tenant_id,scope_name) DO UPDATE SET
                        created_at=MIN(scope_catalog.created_at,excluded.created_at),
                        last_seen_at=MAX(scope_catalog.last_seen_at,excluded.last_seen_at)
                    """,
                    (tenant_id, scope_name, created_at, updated_at),
                )
            connection.execute(
                "INSERT INTO control_migrations(migration_id,applied_at) VALUES(?,?)",
                (migration_id, time.time()),
            )

    @staticmethod
    def _catalog_row(row: Mapping[str, object]) -> dict[str, object]:
        return {
            "scope_name": str(row["scope_name"]),
            "created_at": float(row["created_at"]),
            "last_seen_at": float(row["last_seen_at"]),
            "last_ingest_at": (
                None if row["last_ingest_at"] is None else float(row["last_ingest_at"])
            ),
            "last_recall_at": (
                None if row["last_recall_at"] is None else float(row["last_recall_at"])
            ),
            "session_count": int(row["session_count"]),
            "ingest_request_count": int(row["ingest_request_count"]),
            "recall_request_count": int(row["recall_request_count"]),
            "message_count": int(row["message_count"]),
        }

    def list_scopes(
        self,
        tenant_id: str,
        *,
        prefix: str | None,
        limit: int,
        allowed_scope_names: frozenset[str] | None,
        allowed_scope_prefixes: frozenset[str] | None,
    ) -> list[dict[str, object]]:
        clauses = ["catalog.tenant_id=?"]
        parameters: list[object] = [tenant_id]
        if prefix is not None:
            clauses.append(
                "substr(catalog.scope_name,1,length(?)) = ? COLLATE BINARY"
            )
            parameters.extend((prefix, prefix))
        if allowed_scope_names is not None or allowed_scope_prefixes is not None:
            selectors: list[str] = []
            exact = sorted(allowed_scope_names or ())
            if exact:
                selectors.append(
                    "catalog.scope_name IN (" + ",".join("?" for _ in exact) + ")"
                )
                parameters.extend(exact)
            for allowed_prefix in sorted(allowed_scope_prefixes or ()):
                selectors.append(
                    "substr(catalog.scope_name,1,length(?)) = ? COLLATE BINARY"
                )
                parameters.extend((allowed_prefix, allowed_prefix))
            if not selectors:
                return []
            clauses.append("(" + " OR ".join(selectors) + ")")
        parameters.append(limit)
        with self.database.transaction(immediate=False) as connection:
            rows = connection.execute(
                f"""
                SELECT catalog.*,
                       (SELECT COUNT(*) FROM scope_sessions AS sessions
                        WHERE sessions.tenant_id=catalog.tenant_id
                          AND sessions.scope_name=catalog.scope_name) AS session_count
                FROM scope_catalog AS catalog
                WHERE {' AND '.join(clauses)}
                ORDER BY catalog.last_seen_at DESC, catalog.scope_name
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [self._catalog_row(row) for row in rows]

    def scope_summary(self, tenant_id: str, scope_name: str) -> dict[str, object] | None:
        with self.database.transaction(immediate=False) as connection:
            row = connection.execute(
                """
                SELECT catalog.*,
                       (SELECT COUNT(*) FROM scope_sessions AS sessions
                        WHERE sessions.tenant_id=catalog.tenant_id
                          AND sessions.scope_name=catalog.scope_name) AS session_count
                FROM scope_catalog AS catalog
                WHERE catalog.tenant_id=? AND catalog.scope_name=?
                """,
                (tenant_id, scope_name),
            ).fetchone()
            if row is None:
                return None
            sessions = connection.execute(
                """
                SELECT session_id,created_at,last_ingest_at,
                       ingest_request_count,message_count
                FROM scope_sessions
                WHERE tenant_id=? AND scope_name=?
                ORDER BY last_ingest_at DESC,session_id
                LIMIT 1000
                """,
                (tenant_id, scope_name),
            ).fetchall()
        return {
            "scope": self._catalog_row(row),
            "sessions": [
                {
                    "session_id": str(item["session_id"]),
                    "created_at": float(item["created_at"]),
                    "last_ingest_at": float(item["last_ingest_at"]),
                    "ingest_request_count": int(item["ingest_request_count"]),
                    "message_count": int(item["message_count"]),
                }
                for item in sessions
            ],
        }

    def quota(self, tenant_id: str, principal: str) -> dict[str, object]:
        with self.database.transaction(immediate=False) as connection:
            usage_rows = connection.execute(
                """
                SELECT metric,used_units FROM usage_totals
                WHERE tenant_id=? AND principal=?
                """,
                (tenant_id, principal),
            ).fetchall()
            period = connection.execute(
                """
                SELECT p.*,g.display_name AS group_name,g.status AS group_status
                FROM billing_group_periods AS p
                JOIN billing_groups AS g
                  ON g.tenant_id=p.tenant_id AND g.group_id=p.group_id
                WHERE p.tenant_id=? AND p.usage_principal=?
                """,
                (tenant_id, principal),
            ).fetchone()
            consumers = connection.execute(
                """
                SELECT consumer_principal,metric,COALESCE(SUM(units),0) AS used_units
                FROM usage_events
                WHERE tenant_id=? AND principal=?
                GROUP BY consumer_principal,metric
                ORDER BY consumer_principal,metric
                """,
                (tenant_id, principal),
            ).fetchall()
            entitlement_rows = connection.execute(
                """
                SELECT metric,limit_units FROM usage_entitlements
                WHERE tenant_id=? AND principal=?
                """,
                (tenant_id, principal),
            ).fetchall()
        used_by_metric = {str(row["metric"]): int(row["used_units"]) for row in usage_rows}
        limit_by_metric = {
            str(row["metric"]): (
                None if row["limit_units"] is None else int(row["limit_units"])
            )
            for row in entitlement_rows
        }
        plan = "pilot" if period is None else str(period["plan_code"])
        result: dict[str, object] = {
            "tenant_id": tenant_id,
            "principal": principal,
            "plan": plan,
            "plan_version": None if period is None else str(period["plan_version"]),
            "billing_group": (
                None
                if period is None
                else {
                    "group_id": str(period["group_id"]),
                    "display_name": str(period["group_name"]),
                    "status": str(period["group_status"]),
                    "period_id": str(period["period_id"]),
                    "period_status": self._effective_period_status(
                        period["status"], period["starts_at"], period["ends_at"]
                    ),
                    "billing_interval": str(period["billing_interval"]),
                    "starts_at": float(period["starts_at"]),
                    "ends_at": float(period["ends_at"]),
                    "max_members": int(period["max_members"]),
                    "currency": str(period["currency"]),
                    "price_minor_units": (
                        None
                        if period["price_minor_units"] is None
                        else int(period["price_minor_units"])
                    ),
                }
            ),
        }
        for metric in QUOTA_METRICS:
            used = used_by_metric.get(metric, 0)
            limit = limit_by_metric.get(metric)
            result[metric] = {
                "used": used,
                "limit": limit,
                "remaining": None if limit is None else max(0, limit - used),
            }
        member_usage: dict[str, dict[str, int]] = {}
        for row in consumers:
            member = member_usage.setdefault(
                str(row["consumer_principal"]),
                {metric: 0 for metric in QUOTA_METRICS},
            )
            member[str(row["metric"])] = int(row["used_units"])
        result["member_usage"] = member_usage
        return result

    def set_entitlements(
        self,
        tenant_id: str,
        principal: str,
        limits: Mapping[str, int | None],
        *,
        updated_by_key_id: str,
    ) -> dict[str, object]:
        if not principal or len(principal) > 256:
            raise ValueError("principal must be 1-256 characters")
        if set(limits) != set(QUOTA_METRICS):
            raise ValueError("both quota metric limits are required")
        if any(value is not None and value < 0 for value in limits.values()):
            raise ValueError("quota limits must be non-negative or null")
        now = time.time()
        with self.database.transaction() as connection:
            for metric in QUOTA_METRICS:
                connection.execute(
                    """
                    INSERT INTO usage_entitlements(
                        tenant_id,principal,metric,limit_units,updated_by_key_id,updated_at
                    ) VALUES(?,?,?,?,?,?)
                    ON CONFLICT(tenant_id,principal,metric) DO UPDATE SET
                        limit_units=excluded.limit_units,
                        updated_by_key_id=excluded.updated_by_key_id,
                        updated_at=excluded.updated_at
                    """,
                    (
                        tenant_id,
                        principal,
                        metric,
                        limits[metric],
                        updated_by_key_id,
                        now,
                    ),
                )
        return self.quota(tenant_id, principal)

    @staticmethod
    def _validate_plan_payload(
        *,
        plan_code: str,
        plan_version: str,
        display_name: str,
        billing_interval: str,
        limits: Mapping[str, int | None],
        max_members: int,
        currency: str,
        price_minor_units: int | None,
    ) -> tuple[str, str, str, str, str]:
        code = str(plan_code).strip().lower()
        version = str(plan_version).strip()
        name = str(display_name).strip()
        interval = str(billing_interval).strip().lower()
        normalized_currency = str(currency).strip().upper()
        if not PLAN_CODE_RE.fullmatch(code) or not PLAN_CODE_RE.fullmatch(version):
            raise ValueError("invalid plan code or version")
        if not name or len(name) > 120:
            raise ValueError("display_name must be 1-120 characters")
        if interval not in BILLING_INTERVALS:
            raise ValueError("invalid billing interval")
        if set(limits) != set(QUOTA_METRICS):
            raise ValueError("both quota metric limits are required")
        if any(value is not None and value < 0 for value in limits.values()):
            raise ValueError("quota limits must be non-negative or null")
        if not 1 <= int(max_members) <= 100_000:
            raise ValueError("max_members must be between 1 and 100000")
        if not re.fullmatch(r"[A-Z]{3}", normalized_currency):
            raise ValueError("currency must be a three-letter code")
        if price_minor_units is not None and price_minor_units < 0:
            raise ValueError("price_minor_units must be non-negative or null")
        return code, version, name, interval, normalized_currency

    def put_plan_version(
        self,
        *,
        plan_code: str,
        plan_version: str,
        display_name: str,
        billing_interval: str,
        ingest_raw_tokens: int | None,
        recall_requests: int | None,
        max_members: int,
        currency: str,
        price_minor_units: int | None,
        entitlements: Mapping[str, object],
        updated_by: str,
    ) -> dict[str, object]:
        limits = {
            "ingest_raw_tokens": ingest_raw_tokens,
            "recall_requests": recall_requests,
        }
        code, version, name, interval, normalized_currency = self._validate_plan_payload(
            plan_code=plan_code,
            plan_version=plan_version,
            display_name=display_name,
            billing_interval=billing_interval,
            limits=limits,
            max_members=max_members,
            currency=currency,
            price_minor_units=price_minor_units,
        )
        clean_entitlements = dict(entitlements)
        encoded = json.dumps(
            clean_entitlements, ensure_ascii=False, allow_nan=False, sort_keys=True
        )
        if len(encoded.encode("utf-8")) > 32_768:
            raise ValueError("entitlements must be at most 32768 UTF-8 bytes")
        now = time.time()
        values = (
            name,
            interval,
            ingest_raw_tokens,
            recall_requests,
            int(max_members),
            normalized_currency,
            price_minor_units,
            encoded,
        )
        with self.database.transaction() as connection:
            existing = connection.execute(
                """
                SELECT display_name,billing_interval,ingest_raw_token_limit,
                       recall_request_limit,max_members,currency,price_minor_units,
                       entitlements_json,status
                FROM billing_plan_versions
                WHERE plan_code=? AND plan_version=?
                """,
                (code, version),
            ).fetchone()
            if existing is not None:
                actual = tuple(existing[key] for key in (
                    "display_name", "billing_interval", "ingest_raw_token_limit",
                    "recall_request_limit", "max_members", "currency",
                    "price_minor_units", "entitlements_json",
                ))
                if actual != values:
                    raise BillingConflict("plan version is immutable once created")
            else:
                connection.execute(
                    """
                    INSERT INTO billing_plan_versions(
                        plan_code,plan_version,display_name,status,billing_interval,
                        ingest_raw_token_limit,recall_request_limit,max_members,
                        currency,price_minor_units,entitlements_json,created_by,
                        created_at,updated_at
                    ) VALUES(?,?,?,'active',?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        code, version, name, interval, ingest_raw_tokens,
                        recall_requests, int(max_members), normalized_currency,
                        price_minor_units, encoded, updated_by, now, now,
                    ),
                )
        return self.plan_version(code, version)

    def plan_version(self, plan_code: str, plan_version: str) -> dict[str, object]:
        with closing(self.database.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM billing_plan_versions WHERE plan_code=? AND plan_version=?",
                (str(plan_code).strip().lower(), str(plan_version).strip()),
            ).fetchone()
        if row is None:
            raise BillingNotFound("billing plan version not found")
        result = {key: row[key] for key in row.keys()}
        result["entitlements"] = json.loads(str(result.pop("entitlements_json")))
        return result

    def list_plan_versions(self, *, include_retired: bool = False) -> list[dict[str, object]]:
        with closing(self.database.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM billing_plan_versions "
                + ("" if include_retired else "WHERE status='active' ")
                + "ORDER BY plan_code,created_at,plan_version"
            ).fetchall()
        return [
            {
                **{key: row[key] for key in row.keys() if key != "entitlements_json"},
                "entitlements": json.loads(str(row["entitlements_json"])),
            }
            for row in rows
        ]

    def create_billing_group(
        self,
        tenant_id: str,
        *,
        group_id: str,
        display_name: str,
        owner_subject: str,
        plan_code: str,
        plan_version: str,
        starts_at: float,
        ends_at: float,
        created_by_key_id: str,
    ) -> dict[str, object]:
        clean_group = str(group_id).strip()
        clean_name = str(display_name).strip()
        owner = str(owner_subject).strip()
        if not BILLING_GROUP_ID_RE.fullmatch(clean_group):
            raise ValueError("invalid billing group ID")
        if not clean_name or len(clean_name) > 120:
            raise ValueError("display_name must be 1-120 characters")
        if not BILLING_SUBJECT_RE.fullmatch(owner):
            raise ValueError("invalid owner subject")
        if ends_at <= starts_at:
            raise ValueError("billing period must end after it starts")
        now = time.time()
        if not starts_at <= now < ends_at:
            raise ValueError("active billing period must contain the current time")
        plan = self.plan_version(plan_code, plan_version)
        period_id = uuid.uuid4().hex
        usage_principal = self.billing_principal(clean_group, period_id)
        snapshot = {
            "schema_version": "tmcra.billing-entitlement-snapshot.1",
            "plan_code": plan["plan_code"],
            "plan_version": plan["plan_version"],
            "limits": {
                "ingest_raw_tokens": plan["ingest_raw_token_limit"],
                "recall_requests": plan["recall_request_limit"],
            },
            "max_members": plan["max_members"],
            "entitlements": plan["entitlements"],
        }
        snapshot_json = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        with self.database.transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM billing_groups WHERE tenant_id=? AND group_id=?",
                (tenant_id, clean_group),
            ).fetchone() is not None:
                raise BillingConflict("billing group already exists")
            if connection.execute(
                "SELECT 1 FROM billing_group_members WHERE tenant_id=? AND subject=?",
                (tenant_id, owner),
            ).fetchone() is not None:
                raise BillingConflict("owner already belongs to a billing group")
            connection.execute(
                """
                INSERT INTO billing_groups(
                    tenant_id,group_id,display_name,status,active_period_id,
                    created_by_key_id,created_at,updated_at
                ) VALUES(?,?,?,'active',?,?,?,?)
                """,
                (tenant_id, clean_group, clean_name, period_id, created_by_key_id, now, now),
            )
            connection.execute(
                """
                INSERT INTO billing_group_periods(
                    tenant_id,group_id,period_id,usage_principal,plan_code,
                    plan_version,billing_interval,starts_at,ends_at,status,
                    ingest_raw_token_limit,recall_request_limit,max_members,
                    currency,price_minor_units,entitlement_snapshot_json,
                    created_by_key_id,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,'active',?,?,?,?,?,?,?,?,?)
                """,
                (
                    tenant_id, clean_group, period_id, usage_principal,
                    plan["plan_code"], plan["plan_version"], plan["billing_interval"],
                    float(starts_at), float(ends_at), plan["ingest_raw_token_limit"],
                    plan["recall_request_limit"], plan["max_members"], plan["currency"],
                    plan["price_minor_units"], snapshot_json, created_by_key_id, now, now,
                ),
            )
            connection.execute(
                """
                INSERT INTO billing_group_members(
                    tenant_id,subject,group_id,role,created_by_key_id,created_at,updated_at
                ) VALUES(?,?,?,'owner',?,?,?)
                """,
                (tenant_id, owner, clean_group, created_by_key_id, now, now),
            )
            self._record_billing_member_event(
                connection,
                tenant_id=tenant_id,
                group_id=clean_group,
                subject=owner,
                role="owner",
                event_type="added",
                created_by_key_id=created_by_key_id,
                created_at=now,
            )
            for metric, limit in (
                ("ingest_raw_tokens", plan["ingest_raw_token_limit"]),
                ("recall_requests", plan["recall_request_limit"]),
            ):
                connection.execute(
                    """
                    INSERT INTO usage_entitlements(
                        tenant_id,principal,metric,limit_units,updated_by_key_id,updated_at
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (tenant_id, usage_principal, metric, limit, created_by_key_id, now),
                )
        return self.billing_group(tenant_id, clean_group)

    def billing_group(self, tenant_id: str, group_id: str) -> dict[str, object]:
        clean_group = str(group_id).strip()
        with closing(self.database.connect()) as connection:
            group = connection.execute(
                """
                SELECT g.*,p.usage_principal,p.plan_code,p.plan_version,
                       p.billing_interval,p.starts_at,p.ends_at,p.status AS period_status,
                       p.max_members,p.currency,p.price_minor_units,
                       p.entitlement_snapshot_json
                FROM billing_groups AS g
                JOIN billing_group_periods AS p
                  ON p.tenant_id=g.tenant_id AND p.group_id=g.group_id
                 AND p.period_id=g.active_period_id
                WHERE g.tenant_id=? AND g.group_id=?
                """,
                (tenant_id, clean_group),
            ).fetchone()
            members = connection.execute(
                """
                SELECT subject,role,created_at,updated_at
                FROM billing_group_members
                WHERE tenant_id=? AND group_id=?
                ORDER BY CASE role WHEN 'owner' THEN 0 WHEN 'admin' THEN 1 ELSE 2 END,
                         created_at,subject
                """,
                (tenant_id, clean_group),
            ).fetchall()
            member_events = connection.execute(
                """
                SELECT event_id,subject,role,event_type,created_by_key_id,created_at
                FROM billing_group_member_events
                WHERE tenant_id=? AND group_id=?
                ORDER BY created_at,event_id
                """,
                (tenant_id, clean_group),
            ).fetchall()
        if group is None:
            raise BillingNotFound("billing group not found")
        result = {
            **{key: group[key] for key in group.keys() if key != "entitlement_snapshot_json"},
            "entitlement_snapshot": json.loads(str(group["entitlement_snapshot_json"])),
            "members": [{key: row[key] for key in row.keys()} for row in members],
            "member_events": [
                {key: row[key] for key in row.keys()} for row in member_events
            ],
            "quota": self.quota(tenant_id, str(group["usage_principal"])),
        }
        result["period_status"] = self._effective_period_status(
            group["period_status"], group["starts_at"], group["ends_at"]
        )
        return result

    def list_billing_groups(self, tenant_id: str) -> list[dict[str, object]]:
        with closing(self.database.connect()) as connection:
            ids = [
                str(row["group_id"])
                for row in connection.execute(
                    "SELECT group_id FROM billing_groups WHERE tenant_id=? ORDER BY created_at,group_id",
                    (tenant_id,),
                ).fetchall()
            ]
        return [self.billing_group(tenant_id, group_id) for group_id in ids]

    def add_billing_member(
        self,
        tenant_id: str,
        group_id: str,
        *,
        subject: str,
        role: str,
        created_by_key_id: str,
    ) -> dict[str, object]:
        clean_subject = str(subject).strip()
        clean_role = str(role).strip().lower()
        if not BILLING_SUBJECT_RE.fullmatch(clean_subject):
            raise ValueError("invalid billing member subject")
        if clean_role not in BILLING_MEMBER_ROLES:
            raise ValueError("invalid billing member role")
        if clean_role == "owner":
            raise BillingConflict("billing group owner is assigned when the group is created")
        now = time.time()
        idempotent_replay = False
        with self.database.transaction() as connection:
            group = connection.execute(
                """
                SELECT g.status,p.max_members
                FROM billing_groups AS g
                JOIN billing_group_periods AS p
                  ON p.tenant_id=g.tenant_id AND p.group_id=g.group_id
                 AND p.period_id=g.active_period_id
                WHERE g.tenant_id=? AND g.group_id=?
                """,
                (tenant_id, group_id),
            ).fetchone()
            if group is None:
                raise BillingNotFound("billing group not found")
            if str(group["status"]) != "active":
                raise BillingAccessDenied("billing group is not active", group_id)
            existing = connection.execute(
                "SELECT group_id,role FROM billing_group_members WHERE tenant_id=? AND subject=?",
                (tenant_id, clean_subject),
            ).fetchone()
            if existing is not None:
                if str(existing["group_id"]) == group_id and str(existing["role"]) == clean_role:
                    idempotent_replay = True
                else:
                    raise BillingConflict("subject already belongs to a billing group")
            if not idempotent_replay:
                count = int(connection.execute(
                    "SELECT COUNT(*) FROM billing_group_members WHERE tenant_id=? AND group_id=?",
                    (tenant_id, group_id),
                ).fetchone()[0])
                if count >= int(group["max_members"]):
                    raise BillingConflict("billing group member limit reached")
                connection.execute(
                    """
                    INSERT INTO billing_group_members(
                        tenant_id,subject,group_id,role,created_by_key_id,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?)
                    """,
                    (
                        tenant_id,
                        clean_subject,
                        group_id,
                        clean_role,
                        created_by_key_id,
                        now,
                        now,
                    ),
                )
                self._record_billing_member_event(
                    connection,
                    tenant_id=tenant_id,
                    group_id=group_id,
                    subject=clean_subject,
                    role=clean_role,
                    event_type="added",
                    created_by_key_id=created_by_key_id,
                    created_at=now,
                )
        return self.billing_group(tenant_id, group_id)

    def remove_billing_member(
        self,
        tenant_id: str,
        group_id: str,
        subject: str,
        *,
        removed_by_key_id: str,
    ) -> dict[str, object]:
        clean_subject = str(subject).strip()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT role FROM billing_group_members WHERE tenant_id=? AND group_id=? AND subject=?",
                (tenant_id, group_id, clean_subject),
            ).fetchone()
            if row is None:
                raise BillingNotFound("billing member not found")
            if str(row["role"]) == "owner":
                raise BillingConflict("billing group owner cannot be removed")
            connection.execute(
                "DELETE FROM billing_group_members WHERE tenant_id=? AND group_id=? AND subject=?",
                (tenant_id, group_id, clean_subject),
            )
            self._record_billing_member_event(
                connection,
                tenant_id=tenant_id,
                group_id=group_id,
                subject=clean_subject,
                role=str(row["role"]),
                event_type="removed",
                created_by_key_id=removed_by_key_id,
                created_at=time.time(),
            )
        return self.billing_group(tenant_id, group_id)

    def change_billing_period(
        self,
        tenant_id: str,
        group_id: str,
        *,
        plan_code: str,
        plan_version: str,
        starts_at: float,
        ends_at: float,
        updated_by_key_id: str,
    ) -> dict[str, object]:
        if ends_at <= starts_at:
            raise ValueError("billing period must end after it starts")
        plan = self.plan_version(plan_code, plan_version)
        now = time.time()
        if not starts_at <= now < ends_at:
            raise ValueError("new active billing period must contain the current time")
        period_id = uuid.uuid4().hex
        usage_principal = self.billing_principal(group_id, period_id)
        snapshot = json.dumps(
            {
                "schema_version": "tmcra.billing-entitlement-snapshot.1",
                "plan_code": plan["plan_code"],
                "plan_version": plan["plan_version"],
                "limits": {
                    "ingest_raw_tokens": plan["ingest_raw_token_limit"],
                    "recall_requests": plan["recall_request_limit"],
                },
                "max_members": plan["max_members"],
                "entitlements": plan["entitlements"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        with self.database.transaction() as connection:
            group = connection.execute(
                "SELECT active_period_id,status FROM billing_groups WHERE tenant_id=? AND group_id=?",
                (tenant_id, group_id),
            ).fetchone()
            if group is None:
                raise BillingNotFound("billing group not found")
            if str(group["status"]) == "cancelled":
                raise BillingConflict("cancelled billing group cannot start a new period")
            member_count = int(connection.execute(
                "SELECT COUNT(*) FROM billing_group_members WHERE tenant_id=? AND group_id=?",
                (tenant_id, group_id),
            ).fetchone()[0])
            if member_count > int(plan["max_members"]):
                raise BillingConflict("new plan member limit is below current membership")
            connection.execute(
                "UPDATE billing_group_periods SET status='expired',updated_at=? "
                "WHERE tenant_id=? AND group_id=? AND period_id=? AND status='active'",
                (now, tenant_id, group_id, str(group["active_period_id"])),
            )
            connection.execute(
                """
                INSERT INTO billing_group_periods(
                    tenant_id,group_id,period_id,usage_principal,plan_code,
                    plan_version,billing_interval,starts_at,ends_at,status,
                    ingest_raw_token_limit,recall_request_limit,max_members,
                    currency,price_minor_units,entitlement_snapshot_json,
                    created_by_key_id,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,'active',?,?,?,?,?,?,?,?,?)
                """,
                (
                    tenant_id, group_id, period_id, usage_principal,
                    plan["plan_code"], plan["plan_version"], plan["billing_interval"],
                    float(starts_at), float(ends_at), plan["ingest_raw_token_limit"],
                    plan["recall_request_limit"], plan["max_members"], plan["currency"],
                    plan["price_minor_units"], snapshot, updated_by_key_id, now, now,
                ),
            )
            connection.execute(
                "UPDATE billing_groups SET active_period_id=?,status='active',updated_at=? "
                "WHERE tenant_id=? AND group_id=?",
                (period_id, now, tenant_id, group_id),
            )
            for metric, limit in (
                ("ingest_raw_tokens", plan["ingest_raw_token_limit"]),
                ("recall_requests", plan["recall_request_limit"]),
            ):
                connection.execute(
                    """
                    INSERT INTO usage_entitlements(
                        tenant_id,principal,metric,limit_units,updated_by_key_id,updated_at
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (tenant_id, usage_principal, metric, limit, updated_by_key_id, now),
                )
        return self.billing_group(tenant_id, group_id)

    def set_billing_group_status(
        self, tenant_id: str, group_id: str, status: str
    ) -> dict[str, object]:
        clean_status = str(status).strip().lower()
        if clean_status not in BILLING_GROUP_STATUSES:
            raise ValueError("invalid billing group status")
        with self.database.transaction() as connection:
            current = connection.execute(
                "SELECT status FROM billing_groups WHERE tenant_id=? AND group_id=?",
                (tenant_id, group_id),
            ).fetchone()
            if current is None:
                raise BillingNotFound("billing group not found")
            if str(current["status"]) == "cancelled" and clean_status != "cancelled":
                raise BillingConflict("cancelled billing group cannot be reactivated")
            changed = connection.execute(
                "UPDATE billing_groups SET status=?,updated_at=? "
                "WHERE tenant_id=? AND group_id=?",
                (clean_status, time.time(), tenant_id, group_id),
            ).rowcount
            if not changed:
                raise BillingNotFound("billing group not found")
        return self.billing_group(tenant_id, group_id)
