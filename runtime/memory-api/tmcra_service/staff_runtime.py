from __future__ import annotations

import hmac
import math
import threading
import time
from collections import deque
from collections.abc import Mapping
from typing import Any, Callable

from .control_db import ControlDB
from .health_monitor import CHECK_NAMES
from .settings import ServiceSettings
from .startup import StartupPreflight, service_release_metadata


STAFF_MONITORING_HEADER = "X-TMCRA-Staff-Key"
LATENCY_EXCLUDED_PATHS = frozenset(
    {
        "/docs",
        "/healthz",
        "/openapi.json",
        "/readyz",
        "/v1/internal/runtime",
    }
)


def staff_key_matches(configured: str | None, supplied: str | None) -> bool:
    """Compare a bounded staff credential without data-dependent string checks."""

    if configured is None:
        return False
    candidate = supplied or ""
    if len(candidate) > 512:
        candidate = ""
    return hmac.compare_digest(
        candidate.encode("utf-8"), configured.encode("utf-8")
    )


def _available(value: Any, *, source: str) -> dict[str, Any]:
    return {"availability": "available", "value": value, "source": source}


def _unavailable(reason: str, *, source: str) -> dict[str, Any]:
    return {
        "availability": "unavailable",
        "value": None,
        "reason": reason,
        "source": source,
    }


class RequestLatencyWindow:
    """Bounded in-memory request latency samples with no request content or paths."""

    def __init__(
        self,
        *,
        window_seconds: float,
        max_samples: int,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("latency window must be positive")
        if max_samples <= 0:
            raise ValueError("latency sample capacity must be positive")
        self.window_seconds = float(window_seconds)
        self.max_samples = int(max_samples)
        self.clock = clock
        self.wall_clock = wall_clock
        self._samples: deque[tuple[float, float, float, int]] = deque(
            maxlen=self.max_samples
        )
        self._lock = threading.Lock()

    def observe(self, *, latency_ms: float, status_code: int) -> None:
        latency = float(latency_ms)
        if latency < 0 or not math.isfinite(latency):
            return
        observed_at = self.clock()
        wall_time = self.wall_clock()
        with self._lock:
            self._prune_locked(observed_at)
            self._samples.append(
                (observed_at, wall_time, latency, int(status_code))
            )

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    @staticmethod
    def _nearest_rank(values: list[float], quantile: float) -> float:
        index = max(0, math.ceil(quantile * len(values)) - 1)
        return round(values[index], 3)

    def snapshot(self) -> dict[str, Any]:
        now = self.clock()
        with self._lock:
            self._prune_locked(now)
            samples = list(self._samples)
        source = "request_middleware.memory_window"
        sample_count = len(samples)
        base: dict[str, Any] = {
            "source": source,
            "configured_window_seconds": self.window_seconds,
            "sample_capacity": self.max_samples,
            "sample_count": sample_count,
            "stored_dimensions": ["latency_ms", "status_code", "timestamp"],
            "stores_request_path": False,
            "stores_request_content": False,
        }
        if not samples:
            reason = "no_customer_requests_observed_in_window"
            return {
                **base,
                "availability": "unavailable",
                "reason": reason,
                "observed_window_start_at": _unavailable(reason, source=source),
                "observed_window_end_at": _unavailable(reason, source=source),
                "p50_ms": _unavailable(reason, source=source),
                "p95_ms": _unavailable(reason, source=source),
                "p99_ms": _unavailable(reason, source=source),
                "status_classes": {},
            }

        values = sorted(sample[2] for sample in samples)
        status_classes: dict[str, int] = {}
        for _, _, _, status_code in samples:
            label = f"{status_code // 100}xx" if 100 <= status_code <= 599 else "other"
            status_classes[label] = status_classes.get(label, 0) + 1
        return {
            **base,
            "availability": "available",
            "observed_window_start_at": _available(
                samples[0][1], source=source
            ),
            "observed_window_end_at": _available(samples[-1][1], source=source),
            "p50_ms": _available(
                self._nearest_rank(values, 0.50), source=source
            ),
            "p95_ms": _available(
                self._nearest_rank(values, 0.95), source=source
            ),
            "p99_ms": _available(
                self._nearest_rank(values, 0.99), source=source
            ),
            "status_classes": status_classes,
        }


def _safe_error_category(error: Any) -> str:
    if error is None:
        return "unspecified_failure"
    lowered = str(error)[:8192].casefold()
    categories = (
        ("timeout", ("timeout", "timed out", "deadline")),
        ("rate_limited", ("rate limit", "rate_limit", "429")),
        ("authentication", ("unauthorized", "forbidden", "401", "403")),
        ("invalid_payload", ("json", "decode", "schema", "validation")),
        ("storage", ("sqlite", "database", "disk", "journal")),
        ("transport", ("connection", "network", "socket", "transport")),
        ("cancelled", ("cancelled", "canceled")),
    )
    for category, markers in categories:
        if any(marker in lowered for marker in markers):
            return category
    return "internal_failure"


class StaffRuntimeStatus:
    """Aggregate existing operational facts into a redacted staff contract."""

    def __init__(
        self,
        *,
        settings: ServiceSettings,
        database: ControlDB,
        startup: StartupPreflight,
        health_monitor: Any,
        latency_window: RequestLatencyWindow,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self.database = database
        self.startup = startup
        self.health_monitor = health_monitor
        self.latency_window = latency_window
        self.wall_clock = wall_clock

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "tmcra.service.staff-runtime.1",
            "generated_at": self.wall_clock(),
            "startup_preflight": self._startup_snapshot(),
            "readiness": self._readiness_snapshot(),
            "queue": self._queue_snapshot(),
            "latency": self.latency_window.snapshot(),
            "costs": self._cost_snapshot(),
            "release": self._release_snapshot(),
        }

    def _startup_snapshot(self) -> dict[str, Any]:
        try:
            return self.startup.staff_snapshot()
        except Exception:
            return {
                "availability": "unavailable",
                "reason": "persisted_startup_preflight_unavailable",
                "source": "startup_preflight.persisted_report",
            }

    def _readiness_snapshot(self) -> dict[str, Any]:
        source = "continuous_readiness_monitor.snapshot"
        try:
            snapshot = self.health_monitor.snapshot()
            checks = snapshot.get("checks")
            if not isinstance(checks, Mapping):
                raise ValueError("readiness checks are unavailable")
            return {
                "availability": "available",
                "source": source,
                "ready": bool(snapshot.get("ready")),
                "stale": bool(snapshot.get("stale")),
                "running": bool(snapshot.get("running")),
                "generation": int(snapshot.get("generation") or 0),
                "snapshot_age_seconds": snapshot.get("snapshot_age_seconds"),
                "checks": {
                    name: bool(checks.get(name)) for name in CHECK_NAMES
                },
            }
        except Exception:
            return {
                "availability": "unavailable",
                "reason": "continuous_readiness_snapshot_unavailable",
                "source": source,
            }

    def _queue_snapshot(self) -> dict[str, Any]:
        source = "control_db.jobs_and_operation_stages"
        cutoff = self.wall_clock() - self.settings.staff_recent_error_window_seconds
        limit = self.settings.staff_recent_error_limit
        try:
            with self.database.transaction(immediate=False) as connection:
                job_rows = connection.execute(
                    "SELECT state, COUNT(*) AS count FROM jobs GROUP BY state"
                ).fetchall()
                stage_rows = connection.execute(
                    "SELECT state, COUNT(*) AS count FROM operation_stages GROUP BY state"
                ).fetchall()
                recent_total = int(
                    connection.execute(
                        """
                        SELECT
                            (SELECT COUNT(*) FROM jobs
                             WHERE state='failed' AND error IS NOT NULL AND updated_at>=?) +
                            (SELECT COUNT(*) FROM operation_stages
                             WHERE state='failed' AND error IS NOT NULL AND updated_at>=?)
                        """,
                        (cutoff, cutoff),
                    ).fetchone()[0]
                )
                recent_rows = connection.execute(
                    """
                    SELECT source, error, updated_at FROM (
                        SELECT 'job' AS source, error, updated_at
                        FROM jobs
                        WHERE state='failed' AND error IS NOT NULL AND updated_at>=?
                        UNION ALL
                        SELECT 'operation_stage' AS source, error, updated_at
                        FROM operation_stages
                        WHERE state='failed' AND error IS NOT NULL AND updated_at>=?
                    ) AS recent_failures
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (cutoff, cutoff, limit),
                ).fetchall()
        except Exception:
            return {
                "availability": "unavailable",
                "reason": "control_db_queue_query_failed",
                "source": source,
            }

        job_counts = {
            "pending": 0,
            "running": 0,
            "succeeded": 0,
            "failed": 0,
            "cancelled": 0,
        }
        stage_counts = {
            "ready": 0,
            "running": 0,
            "succeeded": 0,
            "failed": 0,
            "cancelled": 0,
        }
        for row in job_rows:
            state = str(row["state"])
            if state in job_counts:
                job_counts[state] = int(row["count"])
        for row in stage_rows:
            state = str(row["state"])
            if state in stage_counts:
                stage_counts[state] = int(row["count"])
        return {
            "availability": "available",
            "source": source,
            "jobs": job_counts,
            "operation_stages": stage_counts,
            "active_job_count": job_counts["pending"] + job_counts["running"],
            "global_active_job_limit": self.settings.global_queue_limit,
            "recent_error_window_seconds": self.settings.staff_recent_error_window_seconds,
            "recent_error_total": recent_total,
            "recent_error_limit": limit,
            "recent_error_truncated": recent_total > len(recent_rows),
            "recent_errors": [
                {
                    "source": str(row["source"]),
                    "category": _safe_error_category(row["error"]),
                    "occurred_at": float(row["updated_at"]),
                }
                for row in recent_rows
            ],
            "raw_error_text_exposed": False,
        }

    def _cost_snapshot(self) -> dict[str, Any]:
        source = "control_db.provider_calls_and_scope_evolution_state"
        try:
            with self.database.transaction(immediate=False) as connection:
                calls = connection.execute(
                    """
                    SELECT
                        COUNT(*) AS registered_call_count,
                        COALESCE(SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END), 0)
                            AS completed_call_count,
                        COALESCE(SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END), 0)
                            AS failed_call_count,
                        COALESCE(SUM(CASE WHEN status='unknown' THEN 1 ELSE 0 END), 0)
                            AS unknown_call_count,
                        COALESCE(SUM(CASE WHEN status='started' THEN 1 ELSE 0 END), 0)
                            AS in_flight_call_count,
                        COALESCE(SUM(CASE
                            WHEN status='completed' AND cost_micros IS NULL THEN 1
                            ELSE 0 END), 0) AS unpriced_completed_call_count,
                        COALESCE(SUM(input_tokens), 0) AS input_tokens,
                        COALESCE(SUM(output_tokens), 0) AS output_tokens,
                        COALESCE(SUM(cache_hit_tokens), 0) AS cache_hit_tokens,
                        COALESCE(SUM(cache_miss_tokens), 0) AS cache_miss_tokens,
                        COALESCE(SUM(cost_micros), 0) AS known_cost_micro_cny,
                        MIN(created_at) AS period_start,
                        MAX(created_at) AS period_end
                    FROM provider_calls
                    """
                ).fetchone()
                evolution = connection.execute(
                    """
                    SELECT
                        COUNT(*) AS scope_count,
                        COALESCE(SUM(reserved_cost_micro_cny), 0)
                            AS reserved_cost_micro_cny,
                        COALESCE(SUM(spent_cost_micro_cny), 0)
                            AS spent_cost_micro_cny,
                        COALESCE(SUM(source_raw_token_estimate), 0)
                            AS ingested_raw_token_estimate
                    FROM scope_evolution_state
                    """
                ).fetchone()
        except Exception:
            return {
                "availability": "unavailable",
                "reason": "control_db_cost_query_failed",
                "source": source,
            }

        registered = int(calls["registered_call_count"] or 0)
        unknown = int(calls["unknown_call_count"] or 0)
        in_flight = int(calls["in_flight_call_count"] or 0)
        unpriced = int(calls["unpriced_completed_call_count"] or 0)
        period_reason = "no_registered_provider_calls"
        return {
            "availability": "available",
            "source": source,
            "currency": "CNY",
            "ledger_coverage": "registered_provider_calls_only",
            "registered_call_count": registered,
            "completed_call_count": int(calls["completed_call_count"] or 0),
            "failed_call_count": int(calls["failed_call_count"] or 0),
            "unknown_call_count": unknown,
            "in_flight_call_count": in_flight,
            "unpriced_completed_call_count": unpriced,
            "uncertain_cost_call_count": unknown + in_flight + unpriced,
            "input_tokens": int(calls["input_tokens"] or 0),
            "output_tokens": int(calls["output_tokens"] or 0),
            "cache_hit_tokens": int(calls["cache_hit_tokens"] or 0),
            "cache_miss_tokens": int(calls["cache_miss_tokens"] or 0),
            "known_cost_micro_cny": int(calls["known_cost_micro_cny"] or 0),
            "period_start": (
                _available(float(calls["period_start"]), source=source)
                if calls["period_start"] is not None
                else _unavailable(period_reason, source=source)
            ),
            "period_end": (
                _available(float(calls["period_end"]), source=source)
                if calls["period_end"] is not None
                else _unavailable(period_reason, source=source)
            ),
            "scope_evolution": {
                "scope_count": int(evolution["scope_count"] or 0),
                "reserved_cost_micro_cny": int(
                    evolution["reserved_cost_micro_cny"] or 0
                ),
                "spent_cost_micro_cny": int(
                    evolution["spent_cost_micro_cny"] or 0
                ),
                "ingested_raw_token_estimate": int(
                    evolution["ingested_raw_token_estimate"] or 0
                ),
                "counted_separately_from_provider_call_cost": True,
            },
        }

    def _release_snapshot(self) -> dict[str, Any]:
        source = "validated_deployment_environment"
        metadata = service_release_metadata(self.settings)
        fields = {
            "service_version": _available(
                metadata["service_version"], source="tmcra_service.__version__"
            ),
            "release_id": self._release_field(metadata["release_id"], source),
            "release_sha256": self._release_field(
                metadata["release_sha256"], source
            ),
            "channel": self._release_field(metadata["release_channel"], source),
            "canary_percent": self._release_field(
                metadata["canary_percent"], source
            ),
            "rollback_release_id": self._release_field(
                metadata["rollback_release_id"], source
            ),
        }
        missing = [
            name
            for name, field in fields.items()
            if field["availability"] == "unavailable"
        ]
        return {
            "availability": "partial" if missing else "available",
            "source": source,
            **fields,
            "unavailable_fields": missing,
        }

    @staticmethod
    def _release_field(value: Any, source: str) -> dict[str, Any]:
        if value is None:
            return _unavailable("deployment_metadata_not_configured", source=source)
        return _available(value, source=source)
