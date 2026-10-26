#!/usr/bin/env python3
"""Summarize the privacy-bounded TMCRA API JSONL access journal."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Iterator


DEFAULT_LOG = Path(
    "/opt/tmcra-data/tmcra_service_state/api-access.jsonl"
)


def _files(path: Path) -> list[Path]:
    candidates = [path, *path.parent.glob(f"{path.name}.*")]
    return sorted(
        (candidate for candidate in candidates if candidate.is_file()),
        key=lambda candidate: candidate.stat().st_mtime,
    )


def _lines(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        yield from stream


def _events(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        for line in _lines(path):
            try:
                value = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict) and value.get("schema") == "tmcra.api-access.1":
                yield value


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[max(0, index)], 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--since-minutes", type=float, default=60.0)
    parser.add_argument("--request-id")
    parser.add_argument("--tenant-id")
    parser.add_argument("--scope-name")
    parser.add_argument("--status-at-least", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.since_minutes <= 0 or args.limit <= 0:
        parser.error("--since-minutes and --limit must be positive")

    cutoff = time.time() - args.since_minutes * 60.0
    selected: list[dict[str, Any]] = []
    for event in _events(_files(args.log)):
        if float(event.get("recorded_at") or 0) < cutoff:
            continue
        if args.request_id and event.get("request_id") != args.request_id:
            continue
        if args.tenant_id and event.get("tenant_id") != args.tenant_id:
            continue
        if args.scope_name and event.get("scope_name") != args.scope_name:
            continue
        if int(event.get("status_code") or 0) < args.status_at_least:
            continue
        selected.append(event)

    status_counts = collections.Counter(
        str(event.get("status_code") or "unknown") for event in selected
    )
    route_counts = collections.Counter(
        str(event.get("route") or "unknown") for event in selected
    )
    error_counts = collections.Counter(
        str(event.get("error_code"))
        for event in selected
        if event.get("error_code")
    )
    latencies = [
        float(event["latency_ms"])
        for event in selected
        if isinstance(event.get("latency_ms"), (int, float))
    ]
    errors = [
        {
            key: event.get(key)
            for key in (
                "recorded_at",
                "request_id",
                "tenant_id",
                "scope_name",
                "route",
                "status_code",
                "latency_ms",
                "error_code",
                "exception_type",
                "job_ids",
            )
        }
        for event in selected
        if int(event.get("status_code") or 0) >= 400
    ][-args.limit :]
    result = {
        "schema": "tmcra.api-access-report.1",
        "generated_at": time.time(),
        "window_minutes": args.since_minutes,
        "matched_requests": len(selected),
        "status_counts": dict(status_counts.most_common()),
        "top_routes": dict(route_counts.most_common(args.limit)),
        "error_codes": dict(error_counts.most_common(args.limit)),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "recent_errors": errors,
    }
    if args.request_id:
        result["request_trace"] = selected[-args.limit :]
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
