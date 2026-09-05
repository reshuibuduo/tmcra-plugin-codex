#!/usr/bin/env python3
"""Repair V4 Fast-to-Slow coverage without rerunning the Writer."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze_tmcra_v4_slow_coverage import _analyze_db
from run_tmcra_v4_build import (
    DEFAULT_REPO,
    DEFAULT_WRITER_ENV,
    BuildError,
    _key_pool,
    _load_resume_manifest,
    _load_shell_environment,
    _slow_worker_resume,
    _verify_resume_writer,
    _worker_environment,
)
from tmcra_v4_slow_graph import (
    SLOW_PROMPT_VERSION,
    SlowGraphError,
    TieredGraphPatchManager,
    load_graph_schema,
)


SCHEMA_VERSION = "tmcra.v4.slow-coverage-repair.1"
LOCK_NAME = "SLOW_REPAIR_LOCK"
STATE_NAME = "SLOW_REPAIR_IN_PROGRESS.json"
ARCHIVED_COMPLETE_NAME = "BUILD_COMPLETE.before_slow_graph_2026-07-13.3"
FRESH_SLOW_COPY_NAME = "FRESH_SLOW_COPY_COMPLETE.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _requested_workers(raw_values: Sequence[str]) -> list[str]:
    names: list[str] = []
    for raw in raw_values:
        names.extend(item.strip() for item in raw.split(",") if item.strip())
    if not names:
        raise BuildError("at least one worker must be selected")
    if len(names) != len(set(names)):
        raise BuildError("selected workers must be unique")
    for name in names:
        if not name.startswith("worker_") or not name.removeprefix("worker_").isdigit():
            raise BuildError(f"invalid worker name: {name}")
    return names


def _select_workers(
    manifest: Mapping[str, Any], requested: Sequence[str]
) -> list[dict[str, Any]]:
    by_name = {
        Path(str(worker["worker_dir"])).name: dict(worker)
        for worker in manifest.get("workers", [])
    }
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise BuildError("selected workers are absent from manifest: " + ",".join(missing))
    return [by_name[name] for name in requested]


def _unfinished_jobs(database: Path) -> list[dict[str, Any]]:
    with closing(sqlite3.connect(database)) as con:
        con.row_factory = sqlite3.Row
        table = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='slow_graph_jobs'"
        ).fetchone()
        if table is None:
            return []
        return [
            dict(row)
            for row in con.execute(
                "SELECT job_id,status,last_error FROM slow_graph_jobs "
                "WHERE status!='completed' ORDER BY created_at,job_id"
            )
        ]


def _validate_resumable_jobs(database: Path) -> list[dict[str, Any]]:
    """Return unfinished jobs only when their transaction boundary is clean."""
    with closing(sqlite3.connect(database)) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT job_id,status,last_error,claim_token,claim_owner,lease_expires_at "
            "FROM slow_graph_jobs WHERE status!='completed' ORDER BY created_at,job_id"
        ).fetchall()
        active_attempts = con.execute(
            "SELECT attempt_id,job_id,status FROM slow_graph_attempts "
            "WHERE status='started' ORDER BY created_at,attempt_id"
        ).fetchall()
    invalid = [
        dict(row)
        for row in rows
        if row["status"] != "pending"
        or row["claim_token"] is not None
        or row["claim_owner"] is not None
        or row["lease_expires_at"] is not None
    ]
    if invalid or active_attempts:
        raise BuildError(
            "existing Slow jobs are not at a clean resumable boundary: "
            + json.dumps(
                {
                    "invalid_jobs": invalid,
                    "started_attempts": [dict(row) for row in active_attempts],
                },
                sort_keys=True,
            )
        )
    return [dict(row) for row in rows]


def _attempts(database: Path) -> dict[str, dict[str, Any]]:
    with closing(sqlite3.connect(database)) as con:
        con.row_factory = sqlite3.Row
        table = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='slow_graph_attempts'"
        ).fetchone()
        if table is None:
            return {}
        rows = con.execute(
            "SELECT attempt_id,job_id,status,call_metadata_json,error "
            "FROM slow_graph_attempts ORDER BY created_at,attempt_id"
        ).fetchall()
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            metadata = json.loads(str(row["call_metadata_json"] or "{}"))
        except json.JSONDecodeError:
            metadata = {"metadata_parse_error": True}
        output[str(row["attempt_id"])] = {
            "attempt_id": str(row["attempt_id"]),
            "job_id": str(row["job_id"]),
            "status": str(row["status"]),
            "error": str(row["error"] or ""),
            "metadata": metadata,
        }
    return output


def _attempt_summary(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    route_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    usage = Counter()
    physical_calls = 0
    estimated_cost = 0.0
    invalid_metadata = 0
    for attempt in attempts:
        status_counts[str(attempt.get("status") or "unknown")] += 1
        metadata = attempt.get("metadata")
        if not isinstance(metadata, Mapping):
            invalid_metadata += 1
            continue
        route_counts[str(metadata.get("route") or "unknown")] += 1
        physical_calls += int(metadata.get("physical_api_calls", 0) or 0)
        raw_usage = metadata.get("usage")
        if isinstance(raw_usage, Mapping):
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "cache_read_input_tokens",
                "cache_hit_tokens",
                "cache_miss_tokens",
                "total_tokens",
            ):
                usage[key] += int(raw_usage.get(key, 0) or 0)
        cost = metadata.get("cost_audit")
        if isinstance(cost, Mapping):
            estimated_cost += float(cost.get("estimated_cost", 0.0) or 0.0)
    return {
        "attempt_count": len(attempts),
        "physical_api_calls": physical_calls,
        "route_counts": dict(sorted(route_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "usage": dict(usage),
        "estimated_cost_cny": round(estimated_cost, 8),
        "invalid_metadata_count": invalid_metadata,
    }


def _coverage(database: Path) -> dict[str, Any]:
    result = _analyze_db(database)
    eligible = int(result["eligible"])
    cited = int(result["cited"])
    return {
        "eligible": eligible,
        "cited": cited,
        "uncited": int(result["uncited"]),
        "coverage_ratio": round(cited / eligible, 6) if eligible else 1.0,
        "affected_regions": int(result["affected_region_count"]),
    }


def _validated_repo(repo: Path) -> Path:
    resolved = repo.resolve()
    try:
        load_graph_schema(resolved)
    except SlowGraphError as exc:
        raise BuildError(f"invalid --repo for Slow graph schema: {resolved}: {exc}") from exc
    return resolved


def _repair_worker(
    worker: Mapping[str, Any],
    *,
    repo: Path,
    environment: Mapping[str, str],
    resume_existing: bool,
) -> dict[str, Any]:
    worker_name = Path(str(worker["worker_dir"])).name
    database = Path(str(worker["worker_dir"])) / "native_memory.sqlite3"
    before_attempts = _attempts(database)
    before_coverage = _coverage(database)
    started = time.monotonic()
    error_type = ""
    error = ""
    try:
        _slow_worker_resume(
            worker,
            repo=repo,
            environment=environment,
            enqueue=not resume_existing,
        )
        status = "passed"
    except Exception as exc:  # preserve the exact failed job for explicit review
        status = "failed"
        error_type = exc.__class__.__name__
        error = str(exc)
    after_attempts = _attempts(database)
    new_attempts = [
        attempt
        for attempt_id, attempt in after_attempts.items()
        if attempt_id not in before_attempts
    ]
    after_coverage = _coverage(database)
    return {
        "worker": worker_name,
        "worker_index": int(worker["worker_index"]),
        "question_id": str(worker["question_id"]),
        "scope_id": str(worker["scope_id"]),
        "status": status,
        "error_type": error_type,
        "error": error,
        "duration_seconds": round(time.monotonic() - started, 3),
        "coverage_before": before_coverage,
        "coverage_after": after_coverage,
        "new_attempts": _attempt_summary(new_attempts),
        "unfinished_jobs_after": _unfinished_jobs(database),
    }


def _acquire_lock(run_dir: Path, selected: Sequence[str]) -> Path:
    lock = run_dir / LOCK_NAME
    payload = {
        "schema_version": SCHEMA_VERSION,
        "pid": os.getpid(),
        "started_at": _now(),
        "prompt_version": SLOW_PROMPT_VERSION,
        "selected_workers": list(selected),
    }
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return lock


def _mark_build_incomplete(run_dir: Path, selected: Sequence[str]) -> None:
    complete = run_dir / "BUILD_COMPLETE"
    archived = run_dir / ARCHIVED_COMPLETE_NAME
    fresh_copy = run_dir / FRESH_SLOW_COPY_NAME
    state = run_dir / STATE_NAME
    if complete.exists():
        if archived.exists():
            raise BuildError("both live and archived BUILD_COMPLETE markers exist")
        os.replace(complete, archived)
    elif fresh_copy.exists():
        try:
            fresh_payload = json.loads(fresh_copy.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BuildError("fresh Slow copy marker is unreadable") from exc
        if (
            not isinstance(fresh_payload, Mapping)
            or fresh_payload.get("schema_version") != "tmcra.v4.fresh-slow-run.1"
            or fresh_payload.get("status") != "complete"
        ):
            raise BuildError("fresh Slow copy marker is incomplete or stale")
    elif not archived.exists() and not state.exists():
        raise BuildError("run has no BUILD_COMPLETE marker or prior slow-repair state")
    _write_json_atomic(
        state,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "in_progress",
            "updated_at": _now(),
            "prompt_version": SLOW_PROMPT_VERSION,
            "selected_workers": list(selected),
            "archived_build_complete": archived.name,
            "fresh_slow_copy_marker": fresh_copy.name if fresh_copy.exists() else "",
        },
    )


def repair(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise BuildError(f"run directory does not exist: {run_dir}")
    output = args.output.resolve()
    if output.exists():
        raise BuildError(f"repair report already exists: {output}")
    requested = _requested_workers(args.workers)
    manifest = _load_resume_manifest(run_dir)
    workers = _select_workers(manifest, requested)
    if args.concurrency <= 0:
        raise BuildError("concurrency must be positive")
    if args.concurrency > len(workers):
        raise BuildError("concurrency cannot exceed selected worker count")
    repo = _validated_repo(args.repo)

    shell_environment = _load_shell_environment(args.writer_env.resolve())
    base_environment = {**os.environ, **shell_environment}
    keys = _key_pool(base_environment)
    environments: dict[str, dict[str, str]] = {}
    for worker in workers:
        _verify_resume_writer(worker)
        database = Path(str(worker["worker_dir"])) / "native_memory.sqlite3"
        unfinished = _unfinished_jobs(database)
        if unfinished and not args.resume_existing:
            raise BuildError(
                f"{Path(str(worker['worker_dir'])).name} has unfinished slow jobs: "
                + json.dumps(unfinished, sort_keys=True)
            )
        if unfinished:
            _validate_resumable_jobs(database)
        environment = _worker_environment(
            base_environment, keys, int(worker["worker_index"])
        )
        if int(environment.get("TMCRA_WRITER_MAX_TOKENS", "0")) != 16384:
            raise BuildError("Writer/Slow max token preflight is not 16384")
        environments[Path(str(worker["worker_dir"])).name] = environment

    # Instantiate both clients before changing the run completion marker. This is
    # configuration validation only and performs no network request.
    first_environment = environments[requested[0]]
    previous_environment = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(first_environment)
        manager = TieredGraphPatchManager.from_env()
        if manager.flash is None or manager.pro is None:
            raise BuildError("Slow Flash/Pro clients are not fully configured")
    finally:
        os.environ.clear()
        os.environ.update(previous_environment)

    lock = _acquire_lock(run_dir, requested)
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        _mark_build_incomplete(run_dir, requested)
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    _repair_worker,
                    worker,
                    repo=repo,
                    environment=environments[Path(str(worker["worker_dir"])).name],
                    resume_existing=args.resume_existing,
                ): Path(str(worker["worker_dir"])).name
                for worker in workers
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: int(item["worker_index"]))
        physical_calls = sum(
            int(item["new_attempts"]["physical_api_calls"]) for item in results
        )
        estimated_cost = sum(
            float(item["new_attempts"]["estimated_cost_cny"]) for item in results
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "passed" if all(item["status"] == "passed" for item in results) else "failed"
            ),
            "run_dir": str(run_dir),
            "prompt_version": SLOW_PROMPT_VERSION,
            "selected_workers": requested,
            "concurrency": args.concurrency,
            "duration_seconds": round(time.monotonic() - started, 3),
            "physical_api_calls": physical_calls,
            "estimated_cost_cny": round(estimated_cost, 8),
            "workers": results,
        }
        _write_json_atomic(output, report)
        state = json.loads((run_dir / STATE_NAME).read_text(encoding="utf-8"))
        state.update(
            {
                "updated_at": _now(),
                "last_report": str(output),
                "last_status": report["status"],
                "last_physical_api_calls": physical_calls,
            }
        )
        _write_json_atomic(run_dir / STATE_NAME, state)
        return report
    finally:
        lock.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair selected V4 Slow regions without rerunning Writer"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "continue existing pending Slow jobs only after interrupted attempts "
            "have been explicitly recovered and reopened"
        ),
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--writer-env", type=Path, default=DEFAULT_WRITER_ENV)
    args = parser.parse_args()
    report = repair(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
