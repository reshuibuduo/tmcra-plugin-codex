#!/usr/bin/env python3
"""Correlate TMCRA API, worker, Job, and Stage failures for incident review."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable, Iterator


DEFAULT_ACCESS_LOG = Path("/opt/tmcra-data/tmcra_service_state/api-access.jsonl")
DEFAULT_DIAGNOSTIC_LOG = Path("/opt/tmcra-data/tmcra_service_state/api-errors.jsonl")
DEFAULT_CONTROL_DB = Path("/opt/tmcra-data/tmcra_service_state/control.sqlite3")


def files(path: Path) -> list[Path]:
    candidates = [path, *path.parent.glob(f"{path.name}.*")]
    return sorted(
        (candidate for candidate in candidates if candidate.is_file()),
        key=lambda candidate: candidate.stat().st_mtime,
    )


def lines(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        yield from stream


def events(paths: Iterable[Path], schema: str) -> Iterator[dict[str, Any]]:
    for path in paths:
        for line in lines(path):
            try:
                value = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict) and value.get("schema") == schema:
                yield value


def parsed_job_error(value: object, *, details: bool) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    raw = str(value)
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        decoded = None
    if isinstance(decoded, dict):
        result = {
            "type": str(decoded.get("type") or "unknown"),
            "message": str(decoded.get("message") or "")[:2_000],
        }
        if details and decoded.get("traceback"):
            result["traceback"] = str(decoded["traceback"])[:40_000]
        return result
    head = raw.splitlines()[0] if raw else ""
    result = {
        "type": head.split(":", 1)[0] or "unknown",
        "message": head[:2_000],
    }
    if details and len(raw.splitlines()) > 1:
        result["traceback"] = raw[:40_000]
    return result


def job_view(row: sqlite3.Row, *, details: bool) -> dict[str, Any]:
    try:
        payload = json.loads(row["payload_json"] or "{}")
    except (TypeError, ValueError):
        payload = {}
    return {
        "job_id": row["job_id"],
        "job_type": payload.get("job_type") if isinstance(payload, dict) else None,
        "tenant_id": row["tenant_id"],
        "scope_name": row["scope_name"],
        "state": row["state"],
        "worker_id": row["worker_id"],
        "version": row["version"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "error": parsed_job_error(row["error"], details=details),
    }


def stage_view(row: sqlite3.Row, *, details: bool) -> dict[str, Any]:
    error = str(row["error"] or "")
    return {
        "stage_id": row["stage_id"],
        "job_id": row["job_id"],
        "stage_name": row["stage_name"],
        "state": row["state"],
        "attempt": row["attempt"],
        "worker_id": row["worker_id"],
        "heartbeat_at": row["heartbeat_at"],
        "lease_expires_at": row["lease_expires_at"],
        "finished_at": row["finished_at"],
        "error": error[:40_000] if details else error[:2_000],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--access-log", type=Path, default=DEFAULT_ACCESS_LOG)
    parser.add_argument("--diagnostic-log", type=Path, default=DEFAULT_DIAGNOSTIC_LOG)
    parser.add_argument("--control-db", type=Path, default=DEFAULT_CONTROL_DB)
    parser.add_argument("--since-minutes", type=float, default=60.0)
    parser.add_argument("--request-id")
    parser.add_argument("--job-id")
    parser.add_argument("--scope-name")
    parser.add_argument("--component")
    parser.add_argument("--error-fingerprint")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.since_minutes <= 0 or args.limit <= 0:
        parser.error("--since-minutes and --limit must be positive")

    cutoff = time.time() - args.since_minutes * 60.0
    access = [
        event
        for event in events(files(args.access_log), "tmcra.api-access.1")
        if float(event.get("recorded_at") or 0) >= cutoff
        and (not args.request_id or event.get("request_id") == args.request_id)
        and (not args.job_id or args.job_id in (event.get("job_ids") or []))
        and (not args.scope_name or event.get("scope_name") == args.scope_name)
    ]
    diagnostic = [
        event
        for event in events(files(args.diagnostic_log), "tmcra.diagnostic.1")
        if float(event.get("recorded_at") or 0) >= cutoff
        and (not args.request_id or event.get("request_id") == args.request_id)
        and (not args.job_id or event.get("job_id") == args.job_id)
        and (not args.scope_name or event.get("scope_name") == args.scope_name)
        and (not args.component or event.get("component") == args.component)
        and (
            not args.error_fingerprint
            or event.get("error_fingerprint") == args.error_fingerprint
        )
    ]

    job_ids = {args.job_id} if args.job_id else set()
    for event in access:
        job_ids.update(str(value) for value in event.get("job_ids") or [] if value)
    for event in diagnostic:
        if event.get("job_id"):
            job_ids.add(str(event["job_id"]))

    jobs: list[sqlite3.Row] = []
    stages: list[sqlite3.Row] = []
    with sqlite3.connect(args.control_db) as connection:
        connection.row_factory = sqlite3.Row
        if job_ids:
            placeholders = ",".join("?" for _ in job_ids)
            jobs = connection.execute(
                f"SELECT * FROM jobs WHERE job_id IN ({placeholders}) "
                "ORDER BY updated_at DESC",
                sorted(job_ids),
            ).fetchall()
            stages = connection.execute(
                f"SELECT * FROM operation_stages WHERE job_id IN ({placeholders}) "
                "ORDER BY stage_seq, updated_at",
                sorted(job_ids),
            ).fetchall()
        else:
            where = "COALESCE(finished_at, updated_at)>=? AND state='failed'"
            parameters: list[Any] = [cutoff]
            if args.scope_name:
                where += " AND scope_name=?"
                parameters.append(args.scope_name)
            jobs = connection.execute(
                "SELECT * FROM jobs WHERE " + where + " ORDER BY updated_at DESC LIMIT ?",
                [*parameters, args.limit],
            ).fetchall()
            stages = connection.execute(
                "SELECT * FROM operation_stages WHERE "
                "COALESCE(finished_at, updated_at)>=? AND state IN ('failed','running') "
                + ("AND scope_name=? " if args.scope_name else "")
                + "ORDER BY updated_at DESC LIMIT ?",
                [cutoff, *([args.scope_name] if args.scope_name else []), args.limit],
            ).fetchall()

    job_views = [job_view(row, details=args.details) for row in jobs]
    stage_views = [stage_view(row, details=args.details) for row in stages]
    job_error_counts = collections.Counter(
        str((value.get("error") or {}).get("type") or "unknown")
        for value in job_views
        if value.get("error")
    )
    fingerprint_counts = collections.Counter(
        str(event.get("error_fingerprint") or "unknown") for event in diagnostic
    )
    if not args.details:
        diagnostic = [
            {
                key: event.get(key)
                for key in (
                    "recorded_at",
                    "event_id",
                    "severity",
                    "component",
                    "operation",
                    "request_id",
                    "job_id",
                    "job_type",
                    "stage_id",
                    "stage_name",
                    "stage_attempt",
                    "tenant_id",
                    "scope_name",
                    "worker_id",
                    "status_code",
                    "error_code",
                    "exception_type",
                    "exception_message",
                    "error_fingerprint",
                    "context",
                )
            }
            for event in diagnostic
        ]

    result = {
        "schema": "tmcra.diagnostic-report.1",
        "generated_at": time.time(),
        "window_minutes": args.since_minutes,
        "filters": {
            "request_id": args.request_id,
            "job_id": args.job_id,
            "scope_name": args.scope_name,
            "component": args.component,
            "error_fingerprint": args.error_fingerprint,
            "details": args.details,
        },
        "counts": {
            "access_events": len(access),
            "diagnostic_events": len(diagnostic),
            "jobs": len(job_views),
            "stages": len(stage_views),
        },
        "error_fingerprints": dict(fingerprint_counts.most_common(args.limit)),
        "job_error_types": dict(job_error_counts.most_common(args.limit)),
        "access_events": access[-args.limit :],
        "diagnostic_events": diagnostic[-args.limit :],
        "jobs": job_views[: args.limit],
        "stages": stage_views[: args.limit],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
