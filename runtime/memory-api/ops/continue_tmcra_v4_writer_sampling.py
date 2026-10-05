#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from run_tmcra_v4_build import (  # noqa: E402
    DEFAULT_REPO,
    DEFAULT_WRITER_ENV,
    _key_pool,
    _load_shell_environment,
    _worker_environment,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _indices(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _last_nonempty_line(path: Path) -> str:
    if not path.is_file():
        return ""
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.strip():
            return line.strip()[-1000:]
    return ""


def _writer_complete(worker_dir: Path) -> bool:
    report_path = worker_dir / "product_writer_report.json"
    if not report_path.is_file():
        return False
    try:
        report = _load_json(report_path)
    except (OSError, json.JSONDecodeError, RuntimeError):
        return False
    return report.get("completed") is True


def _run(command: list[str], log_path: Path, environment: Mapping[str, str]) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            env=dict(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Finish one first-pass Writer attempt per worker while preserving known "
            "failures for later diagnosis. This command never starts slow graph or indexing."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--known-failed-indices", required=True)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--writer-env", type=Path, default=DEFAULT_WRITER_ENV)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = _load_json(run_dir / "input_manifest.json")
    workers = list(manifest.get("workers") or [])
    if not workers:
        raise RuntimeError("input manifest has no workers")
    known_failed = _indices(args.known_failed_indices)
    valid_indices = {int(worker["worker_index"]) for worker in workers}
    unknown = sorted(known_failed - valid_indices)
    if unknown:
        raise RuntimeError(f"known failure indices are outside the manifest: {unknown}")

    result_path = run_dir / "writer_first_pass_continuation_results.jsonl"
    report_path = run_dir / "writer_first_pass_continuation_report.json"
    plan_path = run_dir / "writer_first_pass_continuation_plan.json"
    prior_failed: set[int] = set()
    if result_path.is_file():
        for raw_line in result_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if row.get("status") == "failed":
                prior_failed.add(int(row["index"]))

    base_environment = {
        **os.environ,
        **_load_shell_environment(args.writer_env.resolve()),
    }
    keys = _key_pool(base_environment)

    selected: list[tuple[Mapping[str, Any], str]] = []
    counts = {
        "complete_existing": 0,
        "known_failure_skipped": 0,
        "prior_continuation_failure_skipped": 0,
        "resume_interrupted": 0,
        "fresh": 0,
    }
    for worker in workers:
        index = int(worker["worker_index"])
        worker_dir = Path(str(worker["worker_dir"]))
        if _writer_complete(worker_dir):
            counts["complete_existing"] += 1
            continue
        if index in known_failed:
            counts["known_failure_skipped"] += 1
            continue
        if index in prior_failed:
            counts["prior_continuation_failure_skipped"] += 1
            continue
        if (worker_dir / "native_memory.sqlite3").is_file() or (
            worker_dir / "writer.log"
        ).is_file():
            action = "resume_interrupted"
        else:
            action = "fresh"
        counts[action] += 1
        selected.append((worker, action))

    plan = {
        "schema_version": "tmcra.v4.writer-first-pass-continuation.1",
        "created_at": _now(),
        "run_dir": str(run_dir),
        "worker_count": len(workers),
        "known_failed_indices": sorted(known_failed),
        "prior_continuation_failed_indices": sorted(prior_failed),
        "counts": counts,
        "selected_indices": [int(worker["worker_index"]) for worker, _ in selected],
    }
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "plan", **plan}, sort_keys=True), flush=True)
    if args.plan_only:
        return 0

    def execute(worker: Mapping[str, Any], action: str) -> dict[str, Any]:
        index = int(worker["worker_index"])
        worker_dir = Path(str(worker["worker_dir"]))
        environment = _worker_environment(base_environment, keys, index)
        writer_log = worker_dir / "writer.first_pass_continuation.log"
        audit_log = worker_dir / "writer_audit.first_pass_continuation.log"
        started_at = _now()
        try:
            command = [
                sys.executable,
                str(BASE / "tmcra_v4_batch_writer.py"),
                "--input",
                str(worker["input"]),
                "--out-dir",
                str(worker_dir),
                "--repo",
                str(args.repo.resolve()),
            ]
            if action == "resume_interrupted":
                command.extend(
                    [
                        "--revalidate-failed-raw-response",
                        "--recover-interrupted-api-calls",
                    ]
                )
            _run(command, writer_log, environment)
            if not _writer_complete(worker_dir):
                raise RuntimeError("Writer exited successfully without a complete report")
            audit_command = [
                sys.executable,
                str(BASE / "audit_tmcra_v4_chain.py"),
                "--run-dir",
                str(worker_dir),
                "--output",
                str(worker_dir / "writer_chain_audit.json"),
                "--worker-db",
                f"worker={worker_dir / 'native_memory.sqlite3'}",
            ]
            with audit_log.open("w", encoding="utf-8") as log:
                audit_result = subprocess.run(
                    audit_command,
                    env=dict(environment),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            return {
                "at": _now(),
                "started_at": started_at,
                "index": index,
                "question_id": str(worker["question_id"]),
                "action": action,
                "status": "completed",
                "error": "",
                "audit_passed": audit_result.returncode == 0,
                "audit_exit_code": audit_result.returncode,
                "audit_last_log_line": _last_nonempty_line(audit_log),
            }
        except BaseException as exc:
            return {
                "at": _now(),
                "started_at": started_at,
                "index": index,
                "question_id": str(worker["question_id"]),
                "action": action,
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
                "last_log_line": _last_nonempty_line(writer_log),
                "traceback": traceback.format_exc(),
            }

    results: list[dict[str, Any]] = []
    if selected:
        with ThreadPoolExecutor(
            max_workers=min(max(1, args.concurrency), len(selected))
        ) as executor:
            futures = {
                executor.submit(execute, worker, action): int(worker["worker_index"])
                for worker, action in selected
            }
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                _append_jsonl(result_path, row)
                print(json.dumps({"event": "worker_terminal", **row}, sort_keys=True), flush=True)

    completed_now = sum(row["status"] == "completed" for row in results)
    failed_now = sum(row["status"] == "failed" for row in results)
    report = {
        "schema_version": "tmcra.v4.writer-first-pass-continuation.1",
        "status": "complete",
        "completed_at": _now(),
        "run_dir": str(run_dir),
        "worker_count": len(workers),
        "plan_counts": counts,
        "selected": len(selected),
        "completed_now": completed_now,
        "failed_now": failed_now,
        "known_failure_count": len(known_failed),
        "prior_continuation_failure_count": len(prior_failed),
        "all_workers_first_attempted": (
            counts["complete_existing"]
            + counts["known_failure_skipped"]
            + counts["prior_continuation_failure_skipped"]
            + completed_now
            + failed_now
            == len(workers)
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "complete", **report}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
