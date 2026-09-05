#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from migrate_tmcra_v4_fast_exact_evidence import (  # noqa: E402
    MIGRATION_VERSION,
    migrate_database,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _load_object(run_dir / "input_manifest.json")
    workers = list(manifest.get("workers") or [])
    indices = [int(worker["worker_index"]) for worker in workers]
    if len(workers) != 400 or len(set(indices)) != 400:
        raise RuntimeError("remaining400 manifest must contain 400 unique workers")

    targets: list[tuple[int, str, Path]] = []
    seen: set[Path] = set()
    for worker in workers:
        index = int(worker["worker_index"])
        worker_dir = Path(str(worker["worker_dir"])).resolve()
        expected = (run_dir / "writer" / f"worker_{index:03d}").resolve()
        if worker_dir != expected:
            raise RuntimeError(f"worker {index} directory is outside frozen layout")
        database = (worker_dir / "native_memory.sqlite3").resolve()
        if not database.is_file() or database in seen:
            raise RuntimeError(f"worker {index} database is missing or duplicated")
        seen.add(database)
        targets.append((index, str(worker.get("question_id") or ""), database))

    def execute(target: tuple[int, str, Path]) -> dict[str, Any]:
        index, question_id, database = target
        try:
            return {
                "index": index,
                "question_id": question_id,
                "status": "passed",
                "error": "",
                **migrate_database(database, apply=args.apply),
            }
        except BaseException as exc:
            return {
                "index": index,
                "question_id": question_id,
                "database": str(database),
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
            }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=max(1, min(args.concurrency, 16))
    ) as executor:
        futures = {executor.submit(execute, target): target for target in targets}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: int(row["index"]))
    failures = [row for row in results if row["status"] != "passed"]
    report = {
        "schema_version": "tmcra.v4.remaining400-fast-exact-evidence-migration.1",
        "migration_version": MIGRATION_VERSION,
        "status": "passed" if not failures else "failed",
        "mode": "apply" if args.apply else "dry_run",
        "completed_at": _now(),
        "run_dir": str(run_dir),
        "worker_count": len(results),
        "passed_workers": len(results) - len(failures),
        "failed_workers": len(failures),
        "failure_indices": [int(row["index"]) for row in failures],
        "changed_worker_count": sum(
            int(int(row.get("changed_record_count") or 0) > 0) for row in results
        ),
        "changed_record_count": sum(
            int(row.get("changed_record_count") or 0) for row in results
        ),
        "physical_api_calls": 0,
        "failures": failures,
        "workers": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key != "workers"}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
