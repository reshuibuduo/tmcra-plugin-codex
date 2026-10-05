#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_tmcra_v4_build import (
    DEFAULT_REPO,
    DEFAULT_WRITER_ENV,
    _key_pool,
    _load_shell_environment,
    _worker_environment,
)


BASE = Path("/opt/tmcra")


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume selected Writer workers and audit each result")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--recover-incomplete-api-calls", action="store_true")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--writer-env", type=Path, default=DEFAULT_WRITER_ENV)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    base_environment = {
        **os.environ,
        **_load_shell_environment(args.writer_env.resolve()),
    }
    keys = _key_pool(base_environment)
    result_path = run_dir / "writer_repair_results.jsonl"

    def execute(position: int, index: int) -> dict[str, object]:
        worker_dir = run_dir / "writer" / f"worker_{index:03d}"
        environment = _worker_environment(base_environment, keys, position)
        try:
            with (worker_dir / "writer.repair.log").open("w", encoding="utf-8") as log:
                writer_command = [
                        sys.executable,
                        str(BASE / "tmcra_v4_batch_writer.py"),
                        "--input",
                        str(worker_dir / "input.json"),
                        "--out-dir",
                        str(worker_dir),
                        "--repo",
                        str(args.repo.resolve()),
                        "--revalidate-failed-raw-response",
                    ]
                if args.recover_incomplete_api_calls:
                    writer_command.append("--recover-incomplete-api-calls")
                subprocess.run(
                    writer_command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            with (worker_dir / "writer_audit.repair.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(BASE / "audit_tmcra_v4_chain.py"),
                        "--run-dir",
                        str(worker_dir),
                        "--output",
                        str(worker_dir / "writer_chain_audit.json"),
                        "--worker-db",
                        f"worker={worker_dir / 'native_memory.sqlite3'}",
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            return {"index": index, "status": "completed", "error": ""}
        except BaseException as exc:
            return {
                "index": index,
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(execute, position, index): index
            for position, index in enumerate(indices)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")

    results.sort(key=lambda item: int(item["index"]))
    report = {
        "status": "complete",
        "requested": len(indices),
        "completed": sum(item["status"] == "completed" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "results": results,
    }
    (run_dir / "writer_repair_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
