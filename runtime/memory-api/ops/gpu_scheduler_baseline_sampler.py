#!/usr/bin/env python3
"""Collect bounded, payload-free TMCRA GPU scheduling telemetry.

The sampler is intentionally read-only.  It records aggregate GPU, Qwen slot,
recall-pool, and job-queue counters without emitting API keys, tenant IDs,
scope names, prompts, or job payloads.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _json_get(url: str, *, bearer: str = "", timeout: float = 2.0) -> Any:
    headers = {"Accept": "application/json"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _bounded_error(exc: BaseException) -> str:
    text = str(exc).replace("\n", " ").strip()
    return f"{type(exc).__name__}: {text}"[:240]


def _gpu_snapshot() -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,"
            "memory.used,memory.free,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=3.0,
    ).strip()
    values = [item.strip() for item in output.split(",")]
    if len(values) != 6:
        raise RuntimeError("unexpected nvidia-smi field count")
    return {
        "driver_timestamp": values[0],
        "gpu_util_percent": float(values[1]),
        "memory_util_percent": float(values[2]),
        "memory_used_mib": float(values[3]),
        "memory_free_mib": float(values[4]),
        "power_watts": float(values[5]),
    }


def _qwen_snapshot(base_url: str, bearer: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    props = _json_get(f"{base_url}/props", bearer=bearer)
    if isinstance(props, dict):
        result["is_sleeping"] = bool(props.get("is_sleeping", False))
        result["n_ctx"] = props.get("n_ctx")
        result["n_parallel"] = props.get("n_parallel")
    slots = _json_get(f"{base_url}/slots", bearer=bearer)
    if isinstance(slots, list):
        result["slot_count"] = len(slots)
        result["slots_processing"] = sum(
            1
            for slot in slots
            if isinstance(slot, dict)
            and (
                bool(slot.get("is_processing"))
                or int(slot.get("state", 0) or 0) != 0
            )
        )
        result["slots"] = [
            {
                "id": slot.get("id"),
                "is_processing": bool(slot.get("is_processing")),
                "state": slot.get("state"),
                "task_id": slot.get("task_id"),
            }
            for slot in slots
            if isinstance(slot, dict)
        ]
    return result


def _recall_snapshot(base_url: str) -> dict[str, Any]:
    ready = _json_get(f"{base_url}/readyz")
    recall = ready.get("recall_pool", {}) if isinstance(ready, dict) else {}
    pool = recall.get("pool", {}) if isinstance(recall, dict) else {}
    metrics = recall.get("metrics", {}) if isinstance(recall, dict) else {}
    return {
        "current_size": pool.get("current_size"),
        "desired_size": pool.get("desired_size"),
        "active": pool.get("active"),
        "idle": pool.get("idle"),
        "pending": pool.get("pending"),
        "warming": pool.get("warming"),
        "scaling": pool.get("scaling"),
        "service_time_ewma_seconds": metrics.get("service_time_ewma_seconds"),
        "arrival_rate_ewma": metrics.get("arrival_rate_ewma"),
        "offered_load": metrics.get("offered_load"),
    }


def _job_snapshot(database: Path) -> dict[str, Any]:
    by_state: dict[str, int] = {}
    by_type_state: dict[str, dict[str, int]] = {}
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=2.0)
    try:
        rows = connection.execute(
            "SELECT state, payload_json FROM jobs "
            "WHERE state IN ('queued','running','retrying')"
        ).fetchall()
    finally:
        connection.close()
    for state_value, payload_json in rows:
        state = str(state_value or "unknown")[:32]
        by_state[state] = by_state.get(state, 0) + 1
        job_type = "unknown"
        try:
            payload = json.loads(payload_json or "{}")
            if isinstance(payload, dict):
                job_type = str(payload.get("job_type") or "unknown")[:64]
        except (TypeError, ValueError, json.JSONDecodeError):
            job_type = "invalid"
        state_counts = by_type_state.setdefault(job_type, {})
        state_counts[state] = state_counts.get(state, 0) + 1
    return {"by_state": by_state, "by_type_state": by_type_state}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--detail-interval-seconds", type=float, default=5.0)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:2009")
    parser.add_argument("--qwen-base-url", default="http://127.0.0.1:11435")
    parser.add_argument(
        "--qwen-key-file",
        type=Path,
        default=Path(
            "/opt/tmcra-data/local-llm/secrets/qwen36-server-lanes.key"
        ),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("/opt/tmcra-data/tmcra_service_state/control.sqlite3"),
    )
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.interval_seconds <= 0:
        parser.error("duration and interval must be positive")

    bearer = args.qwen_key_file.read_text(encoding="utf-8").strip()
    if not bearer:
        raise RuntimeError("Qwen key file is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    next_tick = started
    next_detail = started
    sequence = 0
    with args.output.open("a", encoding="utf-8", buffering=1) as stream:
        while True:
            now = time.monotonic()
            if now - started >= args.duration_seconds:
                break
            sample: dict[str, Any] = {
                "schema": "tmcra.gpu-scheduler-baseline.v1",
                "sequence": sequence,
                "captured_at": time.time(),
                "elapsed_seconds": round(now - started, 3),
            }
            try:
                sample["gpu"] = _gpu_snapshot()
            except Exception as exc:
                sample["gpu_error"] = _bounded_error(exc)
            if now >= next_detail:
                for name, operation in (
                    (
                        "qwen",
                        lambda: _qwen_snapshot(args.qwen_base_url, bearer),
                    ),
                    ("recall", lambda: _recall_snapshot(args.api_base_url)),
                    ("jobs", lambda: _job_snapshot(args.database)),
                ):
                    try:
                        sample[name] = operation()
                    except Exception as exc:
                        sample[f"{name}_error"] = _bounded_error(exc)
                next_detail = now + args.detail_interval_seconds
            stream.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
            sequence += 1
            next_tick += args.interval_seconds
            time.sleep(max(0.0, next_tick - time.monotonic()))
        stream.write(
            json.dumps(
                {
                    "schema": "tmcra.gpu-scheduler-baseline.v1",
                    "complete": True,
                    "samples": sequence,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "captured_at": time.time(),
                },
                sort_keys=True,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
