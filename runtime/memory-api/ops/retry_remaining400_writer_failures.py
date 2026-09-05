#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _writer_complete(worker_dir: Path) -> bool:
    report_path = worker_dir / "product_writer_report.json"
    if not report_path.is_file():
        return False
    try:
        return _load_object(report_path).get("completed") is True
    except (OSError, json.JSONDecodeError, RuntimeError):
        return False


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        return sum(1 for line in handle if line.strip())


def _last_nonempty_line(path: Path) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        if line.strip():
            return line.strip()[-2000:]
    return ""


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _indices(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retry only the frozen remaining400 Writer first-pass failures."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--failure-analysis",
        type=Path,
        help="Defaults to RUN_DIR/writer_first_pass_failure_analysis.json.",
    )
    parser.add_argument("--expected-failed-count", type=int, default=53)
    parser.add_argument("--length-indices", default="102,297")
    parser.add_argument("--normal-max-tokens", type=int, default=16384)
    parser.add_argument("--length-max-tokens", type=int, default=32768)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--writer-env", type=Path, default=DEFAULT_WRITER_ENV)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    if args.concurrency <= 0:
        raise RuntimeError("concurrency must be positive")
    if args.normal_max_tokens <= 0 or args.length_max_tokens <= 0:
        raise RuntimeError("token limits must be positive")
    if args.length_max_tokens <= args.normal_max_tokens:
        raise RuntimeError("length recovery limit must exceed the normal limit")

    run_dir = args.run_dir.resolve()
    analysis_path = (
        args.failure_analysis.resolve()
        if args.failure_analysis
        else run_dir / "writer_first_pass_failure_analysis.json"
    )
    analysis = _load_object(analysis_path)
    if Path(str(analysis.get("run_dir") or "")).resolve() != run_dir:
        raise RuntimeError("failure analysis belongs to a different run directory")
    failed_indices = {int(value) for value in analysis.get("failed_indices") or []}
    if len(failed_indices) != args.expected_failed_count:
        raise RuntimeError(
            f"frozen failure set changed: {len(failed_indices)} != "
            f"{args.expected_failed_count}"
        )

    manifest_path = run_dir / "input_manifest.json"
    manifest = _load_object(manifest_path)
    workers = list(manifest.get("workers") or [])
    by_index = {int(worker["worker_index"]): worker for worker in workers}
    if len(workers) != 400 or len(by_index) != 400:
        raise RuntimeError("remaining400 manifest must contain 400 unique workers")
    if not failed_indices <= set(by_index):
        raise RuntimeError("failure analysis contains indices outside the manifest")

    length_indices = _indices(args.length_indices)
    if not length_indices <= failed_indices:
        raise RuntimeError("length-recovery indices are not all in the frozen failure set")

    completed_outside_failure_set = {
        index
        for index, worker in by_index.items()
        if index not in failed_indices and _writer_complete(Path(str(worker["worker_dir"])))
    }
    expected_completed = set(by_index) - failed_indices
    if completed_outside_failure_set != expected_completed:
        missing = sorted(expected_completed - completed_outside_failure_set)
        raise RuntimeError(
            "a previously successful worker lost its complete report: "
            + json.dumps(missing)
        )

    selected = [
        by_index[index]
        for index in sorted(failed_indices)
        if not _writer_complete(Path(str(by_index[index]["worker_dir"])))
    ]
    already_recovered = sorted(
        index
        for index in failed_indices
        if _writer_complete(Path(str(by_index[index]["worker_dir"])))
    )

    plan_path = run_dir / "writer_targeted_repair_plan.json"
    result_path = run_dir / "writer_targeted_repair_results.jsonl"
    report_path = run_dir / "writer_targeted_repair_report.json"
    plan = {
        "schema_version": "tmcra.v4.remaining400-writer-targeted-repair.1",
        "created_at": _now(),
        "run_dir": str(run_dir),
        "manifest_sha256": _sha256(manifest_path),
        "failure_analysis_sha256": _sha256(analysis_path),
        "writer_sha256": _sha256(BASE / "tmcra_v4_batch_writer.py"),
        "frozen_failed_indices": sorted(failed_indices),
        "already_recovered_indices": already_recovered,
        "selected_indices": [int(worker["worker_index"]) for worker in selected],
        "length_recovery_indices": sorted(length_indices),
        "normal_max_tokens": args.normal_max_tokens,
        "length_max_tokens": args.length_max_tokens,
        "concurrency": min(args.concurrency, max(1, len(selected))),
        "successful_workers_protected": len(completed_outside_failure_set),
    }
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "plan", **plan}, ensure_ascii=False, sort_keys=True), flush=True)
    if args.plan_only:
        return 0

    base_environment = {
        **os.environ,
        **_load_shell_environment(args.writer_env.resolve()),
    }
    keys = _key_pool(base_environment)

    def execute(worker: Mapping[str, Any]) -> dict[str, Any]:
        index = int(worker["worker_index"])
        worker_dir = Path(str(worker["worker_dir"])).resolve()
        log_path = worker_dir / "writer.targeted_repair.log"
        environment = _worker_environment(base_environment, keys, index)
        calls_path = worker_dir / "product_writer_calls.jsonl"
        raw_path = worker_dir / "product_writer_raw_responses.jsonl"
        calls_before = _line_count(calls_path)
        raw_before = _line_count(raw_path)
        max_tokens = (
            args.length_max_tokens if index in length_indices else args.normal_max_tokens
        )
        started_at = _now()
        command = [
            sys.executable,
            str(BASE / "tmcra_v4_batch_writer.py"),
            "--input",
            str(worker["input"]),
            "--out-dir",
            str(worker_dir),
            "--repo",
            str(args.repo.resolve()),
            "--max-tokens",
            str(max_tokens),
            "--revalidate-failed-raw-response",
            "--recover-interrupted-api-calls",
            "--recover-incomplete-api-calls",
        ]
        try:
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if not _writer_complete(worker_dir):
                raise RuntimeError("Writer exited successfully without a complete report")
            status = "completed"
            error = ""
            failure_traceback = ""
        except BaseException as exc:
            status = "failed"
            error = f"{exc.__class__.__name__}: {exc}"
            failure_traceback = traceback.format_exc()
        calls_after = _line_count(calls_path)
        raw_after = _line_count(raw_path)
        return {
            "at": _now(),
            "started_at": started_at,
            "index": index,
            "question_id": str(worker["question_id"]),
            "status": status,
            "error": error,
            "traceback": failure_traceback,
            "max_tokens": max_tokens,
            "api_call_rows_before": calls_before,
            "api_call_rows_after": calls_after,
            "api_call_rows_added": calls_after - calls_before,
            "raw_response_rows_before": raw_before,
            "raw_response_rows_after": raw_after,
            "raw_response_rows_added": raw_after - raw_before,
            "last_log_line": _last_nonempty_line(log_path),
        }

    results: list[dict[str, Any]] = []
    if selected:
        with ThreadPoolExecutor(
            max_workers=min(args.concurrency, len(selected))
        ) as executor:
            futures = {executor.submit(execute, worker): worker for worker in selected}
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                _append_jsonl(result_path, row)
                print(
                    json.dumps(
                        {"event": "worker_terminal", **row},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )

    results.sort(key=lambda row: int(row["index"]))
    completed_now = [int(row["index"]) for row in results if row["status"] == "completed"]
    failed_now = [int(row["index"]) for row in results if row["status"] == "failed"]
    all_recovered = all(
        _writer_complete(Path(str(by_index[index]["worker_dir"])))
        for index in failed_indices
    )
    report = {
        "schema_version": "tmcra.v4.remaining400-writer-targeted-repair.1",
        "status": "complete" if not failed_now else "completed_with_failures",
        "completed_at": _now(),
        "run_dir": str(run_dir),
        "frozen_failure_count": len(failed_indices),
        "already_recovered_before_run": already_recovered,
        "selected_count": len(selected),
        "completed_now": completed_now,
        "failed_now": failed_now,
        "all_frozen_failures_recovered": all_recovered,
        "api_call_rows_added": sum(int(row["api_call_rows_added"]) for row in results),
        "raw_response_rows_added": sum(
            int(row["raw_response_rows_added"]) for row in results
        ),
        "results": results,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", **report}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if all_recovered and not failed_now else 1


if __name__ == "__main__":
    raise SystemExit(main())
