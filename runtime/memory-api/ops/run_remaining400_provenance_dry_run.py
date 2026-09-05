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

from migrate_tmcra_v4_provenance_offsets import (  # noqa: E402
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
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run provenance offset migration against only the 400 databases "
            "frozen in input_manifest.json."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _load_object(run_dir / "input_manifest.json")
    workers = list(manifest.get("workers") or [])
    indices = [int(worker["worker_index"]) for worker in workers]
    if len(workers) != 400 or len(set(indices)) != 400:
        raise RuntimeError("remaining400 manifest must contain 400 unique workers")

    targets: list[tuple[int, str, Path]] = []
    seen_databases: set[Path] = set()
    for worker in workers:
        index = int(worker["worker_index"])
        worker_dir = Path(str(worker["worker_dir"])).resolve()
        database = (worker_dir / "native_memory.sqlite3").resolve()
        expected_dir = (run_dir / "writer" / f"worker_{index:03d}").resolve()
        if worker_dir != expected_dir:
            raise RuntimeError(f"worker {index} directory is outside frozen layout")
        if not database.is_file():
            raise RuntimeError(f"worker {index} database is missing")
        if database in seen_databases:
            raise RuntimeError(f"duplicate database target: {database}")
        seen_databases.add(database)
        targets.append((index, str(worker.get("question_id") or ""), database))

    def execute(target: tuple[int, str, Path]) -> dict[str, Any]:
        index, question_id, database = target
        try:
            result = migrate_database(database, apply=False)
            return {
                "index": index,
                "question_id": question_id,
                "status": "passed",
                "error": "",
                **result,
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
    with ThreadPoolExecutor(max_workers=max(1, min(args.concurrency, 16))) as executor:
        futures = {executor.submit(execute, target): target for target in targets}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: int(row["index"]))
    failures = [row for row in results if row["status"] != "passed"]
    report = {
        "schema_version": "tmcra.v4.remaining400-provenance-dry-run.1",
        "migration_version": MIGRATION_VERSION,
        "status": "passed" if not failures else "failed",
        "mode": "dry_run",
        "completed_at": _now(),
        "run_dir": str(run_dir),
        "manifest_database_count": len(targets),
        "passed_databases": len(results) - len(failures),
        "failed_databases": len(failures),
        "failure_indices": [int(row["index"]) for row in failures],
        "changed_database_count": sum(
            int(int(row.get("changed_record_count") or 0) > 0) for row in results
        ),
        "changed_record_count": sum(
            int(row.get("changed_record_count") or 0) for row in results
        ),
        "added_offset_count": sum(
            int(row.get("added_offset_count") or 0) for row in results
        ),
        "already_complete_count": sum(
            int(row.get("already_complete_count") or 0) for row in results
        ),
        "journal_disambiguated_count": sum(
            int(row.get("journal_disambiguated_count") or 0) for row in results
        ),
        "provenance_count": sum(
            int(row.get("provenance_count") or 0) for row in results
        ),
        "physical_api_calls": 0,
        "failures": failures,
        "databases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key != "databases"}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
