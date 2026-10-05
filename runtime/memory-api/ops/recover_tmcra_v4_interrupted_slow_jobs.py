#!/usr/bin/env python3
"""Audit and explicitly reopen Slow jobs interrupted at an API boundary."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ops.repair_tmcra_v4_slow_coverage import (
    LOCK_NAME,
    STATE_NAME,
    _requested_workers,
    _select_workers,
    _write_json_atomic,
)
from run_tmcra_v4_build import DEFAULT_REPO, BuildError, _load_resume_manifest
from tmcra_v4_slow_graph import (
    PROCESS_LOSS_INTERRUPTION_ERROR,
    SLOW_PROCESS_LOSS_PHYSICAL_CALLS_MAX,
    SLOW_PROMPT_VERSION,
    SlowGraphStore,
    load_graph_schema,
)


SCHEMA_VERSION = "tmcra.v4.interrupted-slow-recovery.1"
INTERRUPTION_ERROR = PROCESS_LOSS_INTERRUPTION_ERROR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _claim_owner_pid(owner: str) -> int:
    parts = owner.split(":", 2)
    if len(parts) != 3 or parts[0] != "pid" or not parts[1].isdigit():
        raise BuildError(f"invalid Slow claim owner: {owner}")
    return int(parts[1])


def _load_stale_lock(run_dir: Path, requested: Sequence[str]) -> dict[str, Any]:
    path = run_dir / LOCK_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BuildError("Slow repair lock is absent") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError("Slow repair lock is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise BuildError("Slow repair lock is not an object")
    pid = payload.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        raise BuildError("Slow repair lock PID is invalid")
    if _pid_is_alive(pid):
        raise BuildError(f"Slow repair controller is still alive: {pid}")
    if list(payload.get("selected_workers") or []) != list(requested):
        raise BuildError("requested workers do not exactly match the stale repair lock")
    if payload.get("prompt_version") != SLOW_PROMPT_VERSION:
        raise BuildError("stale repair lock prompt version does not match current code")
    return dict(payload)


def _snapshot_database(database: Path) -> dict[str, Any]:
    now = int(time.time())
    with closing(sqlite3.connect(database)) as con:
        con.row_factory = sqlite3.Row
        jobs = con.execute(
            "SELECT job_id,scope_id,region_key,status,attempts,last_error,"
            "claim_token,claim_owner,lease_expires_at FROM slow_graph_jobs "
            "WHERE status!='completed' ORDER BY created_at,job_id"
        ).fetchall()
        started = con.execute(
            "SELECT attempt_id,job_id,status,call_metadata_json,error,created_at,"
            "completed_at,claim_token,claim_owner FROM slow_graph_attempts "
            "WHERE status='started' ORDER BY created_at,attempt_id"
        ).fetchall()
        completed = int(
            con.execute(
                "SELECT COUNT(*) FROM slow_graph_jobs WHERE status='completed'"
            ).fetchone()[0]
        )
    by_claim = {
        (str(row["job_id"]), str(row["claim_token"]), str(row["claim_owner"])): row
        for row in started
    }
    candidates: list[dict[str, Any]] = []
    for job in jobs:
        if job["claim_token"] is None:
            continue
        if job["status"] != "pending":
            raise BuildError("claimed interrupted job is not pending")
        lease = job["lease_expires_at"]
        if lease is None or int(lease) >= now:
            raise BuildError(f"Slow job claim has not expired: {job['job_id']}")
        owner = str(job["claim_owner"] or "")
        owner_pid = _claim_owner_pid(owner)
        if _pid_is_alive(owner_pid):
            raise BuildError(f"Slow claim owner is still alive: {owner_pid}")
        key = (str(job["job_id"]), str(job["claim_token"]), owner)
        attempt = by_claim.get(key)
        if attempt is None:
            raise BuildError(f"claimed job has no matching started attempt: {job['job_id']}")
        try:
            metadata = json.loads(str(attempt["call_metadata_json"]))
        except json.JSONDecodeError as exc:
            raise BuildError("started attempt metadata is invalid JSON") from exc
        if metadata != {} or str(attempt["error"] or ""):
            raise BuildError(
                f"started attempt already contains a durable outcome: {attempt['attempt_id']}"
            )
        candidates.append(
            {
                "job_id": str(job["job_id"]),
                "scope_id": str(job["scope_id"]),
                "region_key": str(job["region_key"]),
                "job_attempts_before": int(job["attempts"]),
                "attempt_id": str(attempt["attempt_id"]),
                "claim_token": str(job["claim_token"]),
                "claim_owner": owner,
                "claim_owner_pid": owner_pid,
                "lease_expires_at": int(lease),
                "attempt_created_at": int(attempt["created_at"]),
                "call_metadata": metadata,
            }
        )
    candidate_attempt_ids = {item["attempt_id"] for item in candidates}
    unexpected_started = [
        str(row["attempt_id"])
        for row in started
        if str(row["attempt_id"]) not in candidate_attempt_ids
    ]
    if unexpected_started:
        raise BuildError(
            "started attempts exist outside expired claimed jobs: "
            + ",".join(unexpected_started)
        )
    return {
        "completed_jobs": completed,
        "unfinished_jobs": len(jobs),
        "interrupted_attempts": candidates,
    }


def _verify_reopened(database: Path, candidates: Sequence[Mapping[str, Any]]) -> None:
    with closing(sqlite3.connect(database)) as con:
        con.row_factory = sqlite3.Row
        remaining_started = int(
            con.execute(
                "SELECT count(*) FROM slow_graph_attempts WHERE status='started'"
            ).fetchone()[0]
        )
        if remaining_started:
            raise BuildError(
                f"recovered database still contains {remaining_started} started attempts"
            )
        for candidate in candidates:
            job = con.execute(
                "SELECT status,attempts,last_error,claim_token,claim_owner,lease_expires_at "
                "FROM slow_graph_jobs WHERE job_id=?",
                (candidate["job_id"],),
            ).fetchone()
            attempt = con.execute(
                "SELECT status,error,completed_at FROM slow_graph_attempts "
                "WHERE attempt_id=?",
                (candidate["attempt_id"],),
            ).fetchone()
            if (
                job is None
                or job["status"] != "pending"
                or int(job["attempts"]) != int(candidate["job_attempts_before"]) + 1
                or str(job["last_error"] or "")
                or job["claim_token"] is not None
                or job["claim_owner"] is not None
                or job["lease_expires_at"] is not None
            ):
                raise BuildError(f"reopened Slow job is inconsistent: {candidate['job_id']}")
            if (
                attempt is None
                or attempt["status"] != "expired"
                or attempt["error"] != INTERRUPTION_ERROR
                or attempt["completed_at"] is None
            ):
                raise BuildError(
                    f"expired Slow attempt is inconsistent: {candidate['attempt_id']}"
                )


def recover(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    if not run_dir.is_dir():
        raise BuildError(f"run directory does not exist: {run_dir}")
    if output.parent != run_dir:
        raise BuildError("recovery report must be written inside the run directory")
    if output.exists():
        raise BuildError(f"recovery report already exists: {output}")
    requested = _requested_workers(args.workers)
    stale_lock = _load_stale_lock(run_dir, requested)
    repo = args.repo.resolve()
    schema = load_graph_schema(repo)
    workers = _select_workers(_load_resume_manifest(run_dir), requested)

    snapshots: list[dict[str, Any]] = []
    for worker in workers:
        worker_name = Path(str(worker["worker_dir"])).name
        database = Path(str(worker["worker_dir"])) / "native_memory.sqlite3"
        snapshots.append(
            {
                "worker": worker_name,
                "database": str(database),
                **_snapshot_database(database),
            }
        )
    interrupted_count = sum(
        len(item["interrupted_attempts"]) for item in snapshots
    )
    if interrupted_count == 0:
        raise BuildError("no expired interrupted Slow attempts were found")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "created_at": _now(),
        "run_dir": str(run_dir),
        "prompt_version": SLOW_PROMPT_VERSION,
        "selected_workers": requested,
        "stale_lock": stale_lock,
        "unknown_external_call_outcomes": interrupted_count,
        "potential_duplicate_physical_calls_min": 0,
        "potential_duplicate_physical_calls_max": (
            interrupted_count * SLOW_PROCESS_LOSS_PHYSICAL_CALLS_MAX
        ),
        "physical_api_calls_during_recovery": 0,
        "workers": snapshots,
    }
    _write_json_atomic(output, report)

    try:
        for item in snapshots:
            candidates = item["interrupted_attempts"]
            store = SlowGraphStore(Path(item["database"]), schema=schema)
            for candidate in candidates:
                store.recover_interrupted_process_loss(
                    str(candidate["job_id"]),
                    expected_attempt_id=str(candidate["attempt_id"]),
                )
            _verify_reopened(Path(item["database"]), candidates)
            item["recovered_and_reopened"] = len(candidates)

        state_path = run_dir / STATE_NAME
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update(
            {
                "updated_at": _now(),
                "interruption_recovery_report": str(output),
                "unknown_external_call_outcomes": interrupted_count,
            }
        )
        _write_json_atomic(state_path, state)
        report.update(
            {
                "status": "recovered_with_stale_lock",
                "completed_at": _now(),
            }
        )
        _write_json_atomic(output, report)
        (run_dir / LOCK_NAME).unlink()
        report.update({"status": "passed", "stale_lock_removed": True})
        _write_json_atomic(output, report)
        return report
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "failed_at": _now(),
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
        )
        _write_json_atomic(output, report)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly recover Slow jobs interrupted at an API boundary"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    args = parser.parse_args()
    report = recover(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
