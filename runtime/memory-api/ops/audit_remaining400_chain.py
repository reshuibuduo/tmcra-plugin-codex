#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


BASE = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the strict V4 chain audit against only the 400 workers frozen "
            "in input_manifest.json. This controller makes no API calls."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    progress = args.progress.resolve()
    manifest = _load_object(run_dir / "input_manifest.json")
    workers = list(manifest.get("workers") or [])
    indices = [int(worker["worker_index"]) for worker in workers]
    if len(workers) != 400 or len(set(indices)) != 400:
        raise RuntimeError("remaining400 manifest must contain 400 unique workers")

    targets: list[tuple[int, str, Path, Path]] = []
    seen_databases: set[Path] = set()
    for worker in workers:
        index = int(worker["worker_index"])
        question_id = str(worker.get("question_id") or "")
        worker_dir = Path(str(worker["worker_dir"])).resolve()
        expected_dir = (run_dir / "writer" / f"worker_{index:03d}").resolve()
        if worker_dir != expected_dir:
            raise RuntimeError(f"worker {index} directory is outside frozen layout")
        database = (worker_dir / "native_memory.sqlite3").resolve()
        if not database.is_file():
            raise RuntimeError(f"worker {index} database is missing")
        if database in seen_databases:
            raise RuntimeError(f"duplicate database target: {database}")
        seen_databases.add(database)
        targets.append((index, question_id, worker_dir, database))

    output.parent.mkdir(parents=True, exist_ok=True)
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text("", encoding="utf-8")
    complete_marker = run_dir / "WRITER_CHAIN_AUDIT_COMPLETE"
    failed_marker = run_dir / "WRITER_CHAIN_AUDIT_FAILED"
    complete_marker.unlink(missing_ok=True)
    failed_marker.unlink(missing_ok=True)

    def execute(target: tuple[int, str, Path, Path]) -> dict[str, Any]:
        index, question_id, worker_dir, database = target
        report_path = worker_dir / "writer_chain_audit.post_provenance.json"
        log_path = worker_dir / "writer_audit.post_provenance.log"
        started_at = _now()
        started = time.monotonic()
        try:
            command = [
                sys.executable,
                str(BASE / "audit_tmcra_v4_chain.py"),
                "--run-dir",
                str(worker_dir),
                "--output",
                str(report_path),
                "--worker-db",
                f"worker={database}",
            ]
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            report = dict(_load_object(report_path))
            passed = result.returncode == 0 and report.get("passed") is True
            issues = list(report.get("issues") or [])
            if result.returncode == 0 and report.get("passed") is not True:
                issues.append("audit exited successfully without passed=true")
            if result.returncode != 0 and not issues:
                issues.append(f"audit exited with code {result.returncode}")
            return {
                "at": _now(),
                "started_at": started_at,
                "duration_seconds": round(time.monotonic() - started, 3),
                "index": index,
                "question_id": question_id,
                "worker_dir": str(worker_dir),
                "database": str(database),
                "status": "passed" if passed else "failed",
                "exit_code": result.returncode,
                "issues": issues,
                "counts": dict(report.get("counts") or {}),
                "slow_promotion_coverage": dict(
                    report.get("slow_promotion_coverage") or {}
                ),
                "report": str(report_path),
                "log": str(log_path),
            }
        except BaseException as exc:
            return {
                "at": _now(),
                "started_at": started_at,
                "duration_seconds": round(time.monotonic() - started, 3),
                "index": index,
                "question_id": question_id,
                "worker_dir": str(worker_dir),
                "database": str(database),
                "status": "failed",
                "exit_code": None,
                "issues": [f"{exc.__class__.__name__}: {exc}"],
                "traceback": traceback.format_exc(),
            }

    results: list[dict[str, Any]] = []
    started_at = _now()
    started = time.monotonic()
    with ThreadPoolExecutor(
        max_workers=max(1, min(args.concurrency, 16))
    ) as executor:
        futures = {executor.submit(execute, target): target for target in targets}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            _append_jsonl(progress, row)
            print(
                json.dumps(
                    {
                        "event": "worker_terminal",
                        "completed": len(results),
                        "passed": sum(item["status"] == "passed" for item in results),
                        "failed": sum(item["status"] == "failed" for item in results),
                        "index": row["index"],
                        "status": row["status"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    results.sort(key=lambda row: int(row["index"]))
    failures = [row for row in results if row["status"] != "passed"]
    count_fields = (
        "input_messages",
        "nonempty_input_messages",
        "excluded_empty_input_messages",
        "source_records",
        "fast_leaves",
        "slow_records",
        "interactions",
        "edges",
    )
    report = {
        "schema_version": "tmcra.v4.remaining400-chain-audit.1",
        "status": "passed" if not failures else "failed",
        "started_at": started_at,
        "completed_at": _now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "run_dir": str(run_dir),
        "worker_count": len(results),
        "passed_workers": len(results) - len(failures),
        "failed_workers": len(failures),
        "failure_indices": [int(row["index"]) for row in failures],
        "physical_api_calls": 0,
        "totals": {
            field: sum(int((row.get("counts") or {}).get(field) or 0) for row in results)
            for field in count_fields
        },
        "slow_promotion": {
            "enforced_workers": sum(
                bool((row.get("slow_promotion_coverage") or {}).get("enforced"))
                for row in results
            ),
            "complete_workers": sum(
                bool((row.get("slow_promotion_coverage") or {}).get("complete"))
                for row in results
            ),
            "eligible_current_durable_count": sum(
                int(
                    (row.get("slow_promotion_coverage") or {}).get(
                        "eligible_current_durable_count"
                    )
                    or 0
                )
                for row in results
            ),
        },
        "failures": failures,
        "workers": results,
    }
    _write_json(output, report)
    marker = complete_marker if not failures else failed_marker
    marker.write_text(
        json.dumps(
            {
                "at": report["completed_at"],
                "report": str(output),
                "status": report["status"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key != "workers"}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
