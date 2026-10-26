#!/usr/bin/env python3
"""Finalize a fresh Slow build only after every explicit quality gate passes."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ops.repair_tmcra_v4_slow_coverage import LOCK_NAME, STATE_NAME, _write_json_atomic
from run_tmcra_v4_build import BuildError, _finalize_build, _load_resume_manifest, _stage
from tmcra_v4_cost_report import build_report, collect_calls


SCHEMA_VERSION = "tmcra.v4.slow-quality-gate-finalization.1"
MARKER_NAME = "V4_SLOW_QUALITY_GATE_COMPLETE.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"quality-gate artifact is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise BuildError(f"quality-gate artifact is not an object: {path}")
    return dict(payload)


def _require_status(payload: Mapping[str, Any], expected: str, label: str) -> None:
    if payload.get("status") != expected:
        raise BuildError(f"{label} status is not {expected}")


def _validate_worker_database(database: Path) -> dict[str, int]:
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as con:
        statuses = {
            str(status): int(count)
            for status, count in con.execute(
                "SELECT status,COUNT(*) FROM slow_graph_jobs GROUP BY status"
            )
        }
        started = int(
            con.execute(
                "SELECT COUNT(*) FROM slow_graph_attempts WHERE status='started'"
            ).fetchone()[0]
        )
    if set(statuses) != {"completed"} or started:
        raise BuildError(
            f"Slow database is unfinished: {database}: statuses={statuses}, started={started}"
        )
    return {"completed_jobs": statuses["completed"], "started_attempts": started}


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise BuildError(f"run directory does not exist: {run_dir}")
    for forbidden in ("BUILD_COMPLETE", "FAILED", LOCK_NAME, MARKER_NAME):
        if (run_dir / forbidden).exists():
            raise BuildError(f"run contains a forbidden finalization marker: {forbidden}")

    manifest = _load_resume_manifest(run_dir)
    workers = list(manifest["workers"])
    worker_names = [Path(str(worker["worker_dir"])).name for worker in workers]
    repair = _json_object(args.repair_report.resolve())
    recovery = _json_object(args.recovery_report.resolve())
    audit = _json_object(args.build_audit.resolve())
    diff = _json_object(args.partition_diff.resolve())
    manual = _json_object(args.manual_review.resolve())
    index = _json_object(run_dir / "index_report.json")

    _require_status(repair, "passed", "Slow repair")
    _require_status(recovery, "passed", "interruption recovery")
    _require_status(audit, "passed", "build audit")
    _require_status(diff, "passed", "partition diff")
    if not str(manual.get("status") or "").startswith("passed"):
        raise BuildError("manual partition review did not pass")
    if int(manual.get("blocking_issue_count", -1)) != 0:
        raise BuildError("manual partition review contains blocking issues")
    if int(diff.get("blocking_issue_count", -1)) != 0:
        raise BuildError("partition diff contains blocking issues")
    if list(repair.get("selected_workers") or []) != worker_names:
        raise BuildError("repair workers differ from the frozen manifest")
    if list(recovery.get("selected_workers") or []) != worker_names:
        raise BuildError("recovery workers differ from the frozen manifest")
    coverage = audit.get("slow_promotion_coverage")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("complete") is not True
        or float(coverage.get("coverage_ratio", 0.0)) != 1.0
        or int(coverage.get("semantic_integrity_issue_count", -1)) != 0
    ):
        raise BuildError("build audit Slow promotion coverage is incomplete")

    index_rows = index.get("rows")
    if (
        int(index.get("row_count", -1)) != len(workers)
        or not isinstance(index_rows, list)
        or len(index_rows) != len(workers)
    ):
        raise BuildError("index report does not cover every frozen worker")
    indexed_qids = {str(row.get("question_id")) for row in index_rows if isinstance(row, Mapping)}
    expected_qids = {str(worker["question_id"]) for worker in workers}
    if indexed_qids != expected_qids:
        raise BuildError("index report question IDs differ from the frozen manifest")
    for row in index_rows:
        index_path = Path(str(row.get("index_path") or ""))
        if not index_path.is_file():
            raise BuildError(f"index file is missing: {index_path}")

    databases = [
        Path(str(worker["worker_dir"])) / "native_memory.sqlite3"
        for worker in workers
    ]
    database_audit = {
        worker_names[index]: _validate_worker_database(database)
        for index, database in enumerate(databases)
    }
    cost = build_report(collect_calls([], databases))
    unknown = int(recovery.get("unknown_external_call_outcomes", -1))
    if unknown < 0 or int(cost.get("unknown_outcome_call_count", -2)) != unknown:
        raise BuildError("cost report does not expose every unknown Slow call outcome")
    if unknown and cost.get("exact_cost_cny") is not None:
        raise BuildError("cost report incorrectly claims an exact total")

    _stage(run_dir, "index_complete", resumed=True)
    build = _finalize_build(
        out_dir=run_dir,
        workers=workers,
        writer_concurrency=0,
        slow_concurrency=len(workers),
        recovered=True,
    )
    if int(build.get("interrupted_calls_without_usage", -1)) != unknown:
        raise BuildError("final build report lost unknown Slow call outcomes")

    state_path = run_dir / STATE_NAME
    state = _json_object(state_path)
    state.update(
        {
            "status": "complete",
            "updated_at": _now(),
            "quality_gate_marker": str(run_dir / MARKER_NAME),
        }
    )
    _write_json_atomic(state_path, state)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": _now(),
        "run_dir": str(run_dir),
        "workers": worker_names,
        "database_audit": database_audit,
        "promotion_coverage": dict(coverage),
        "partition_diff": {
            "blocking_issue_count": diff["blocking_issue_count"],
            "changed_support_count": diff["partition_changed_support_count"],
            "changed_region_count": diff["partition_changed_region_count"],
        },
        "manual_review": {
            "status": manual["status"],
            "blocking_issue_count": manual["blocking_issue_count"],
            "nonblocking_observation_count": manual.get(
                "nonblocking_observation_count", 0
            ),
        },
        "index": {
            "row_count": index["row_count"],
            "parent_count": index.get("parent_count"),
            "candidate_count": index.get("candidate_count"),
        },
        "cost": {
            "physical_call_count": cost["physical_call_count"],
            "definite_physical_call_count": cost["definite_physical_call_count"],
            "unknown_outcome_call_count": cost["unknown_outcome_call_count"],
            "exact_cost_cny": cost["exact_cost_cny"],
            "known_priced_exact_component_cny": cost[
                "known_priced_exact_component_cny"
            ],
        },
        "build_report": build,
    }
    _write_json_atomic(run_dir / MARKER_NAME, marker)
    return marker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument("--recovery-report", type=Path, required=True)
    parser.add_argument("--build-audit", type=Path, required=True)
    parser.add_argument("--partition-diff", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    args = parser.parse_args()
    report = finalize(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
