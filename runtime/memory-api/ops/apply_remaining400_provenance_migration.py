#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import traceback
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


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _verify_applied_database(
    database: Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            raise RuntimeError(f"SQLite quick_check failed: {quick_check}")
        row = connection.execute(
            "SELECT changed_record_count,changed_provenance_count,before_digest,"
            "after_digest,report_json FROM v4_graph_repair_journal WHERE repair_id=?",
            (MIGRATION_VERSION,),
        ).fetchone()
        if row is None:
            raise RuntimeError("migration repair journal is missing")
        if int(row["changed_record_count"]) != int(expected["changed_record_count"]):
            raise RuntimeError("repair journal changed-record count differs from dry-run")
        if int(row["changed_provenance_count"]) != int(expected["added_offset_count"]):
            raise RuntimeError("repair journal offset count differs from dry-run")
        if str(row["before_digest"]) != str(expected["before_digest"]):
            raise RuntimeError("repair journal before digest differs from dry-run")
        if str(row["after_digest"]) != str(expected["after_digest"]):
            raise RuntimeError("repair journal after digest differs from dry-run")
        journal = json.loads(row["report_json"])
        rollback_records = journal.get("rollback_records")
        if not isinstance(rollback_records, list):
            raise RuntimeError("repair journal lacks rollback records")
        if len(rollback_records) != int(expected["changed_record_count"]):
            raise RuntimeError("rollback record count differs from dry-run")
        for rollback in rollback_records:
            if not isinstance(rollback, Mapping):
                raise RuntimeError("rollback record is not an object")
            before_metadata = str(rollback.get("before_metadata_json") or "")
            try:
                before_canonical = _canonical_json(json.loads(before_metadata))
            except (TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError("rollback metadata is invalid JSON") from exc
            if _sha(before_canonical) != rollback.get("before_canonical_sha256"):
                raise RuntimeError("rollback before hash is invalid")
            current = connection.execute(
                "SELECT metadata_json FROM records WHERE scope_id=? AND memory_id=?",
                (rollback.get("scope_id"), rollback.get("memory_id")),
            ).fetchone()
            if current is None:
                raise RuntimeError("migrated record is missing")
            if _sha(str(current[0])) != rollback.get("after_metadata_sha256"):
                raise RuntimeError("migrated record differs from journaled after hash")
    return {
        "quick_check": "ok",
        "rollback_record_count": int(expected["changed_record_count"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a passed remaining400 provenance dry-run sequentially, with "
            "transactional logical rollback records in each SQLite repair journal."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        raise RuntimeError("explicit --apply is required")

    run_dir = args.run_dir.resolve()
    dry_run_path = args.dry_run_report.resolve()
    dry_run = _load_object(dry_run_path)
    if dry_run.get("status") != "passed" or dry_run.get("mode") != "dry_run":
        raise RuntimeError("provenance dry-run report did not pass")
    if dry_run.get("migration_version") != MIGRATION_VERSION:
        raise RuntimeError("provenance dry-run used a different migration version")
    if int(dry_run.get("manifest_database_count") or 0) != 400:
        raise RuntimeError("provenance dry-run did not cover exactly 400 databases")

    manifest = _load_object(run_dir / "input_manifest.json")
    workers = list(manifest.get("workers") or [])
    indices = [int(worker["worker_index"]) for worker in workers]
    if len(workers) != 400 or len(set(indices)) != 400:
        raise RuntimeError("remaining400 manifest must contain 400 unique workers")
    expected_by_index = {
        int(row["index"]): row for row in dry_run.get("databases") or []
    }
    if set(expected_by_index) != set(indices):
        raise RuntimeError("dry-run database set differs from frozen manifest")

    completed_indices: set[int] = set()
    if args.progress.is_file():
        for line in args.progress.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "completed":
                completed_indices.add(int(row["index"]))

    applied_now: list[int] = []
    resumed_indices: list[int] = []
    failure: dict[str, Any] | None = None
    total_rollback_records = 0
    for worker in sorted(workers, key=lambda item: int(item["worker_index"])):
        index = int(worker["worker_index"])
        database = (Path(str(worker["worker_dir"])).resolve() / "native_memory.sqlite3")
        expected = expected_by_index[index]
        started_at = _now()
        try:
            if index in completed_indices:
                verification = _verify_applied_database(database, expected=expected)
                resumed_indices.append(index)
                action = "verify_prior_progress"
            else:
                with sqlite3.connect(database) as connection:
                    table_exists = connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='v4_graph_repair_journal'"
                    ).fetchone()
                    journal_exists = bool(
                        table_exists
                        and connection.execute(
                            "SELECT 1 FROM v4_graph_repair_journal WHERE repair_id=?",
                            (MIGRATION_VERSION,),
                        ).fetchone()
                    )
                if journal_exists:
                    action = "verify_existing_journal"
                    resumed_indices.append(index)
                else:
                    action = "apply"
                    result = migrate_database(database, apply=True)
                    for field in (
                        "changed_record_count",
                        "added_offset_count",
                        "before_digest",
                        "after_digest",
                    ):
                        if result.get(field) != expected.get(field):
                            raise RuntimeError(
                                f"apply result {field} differs from frozen dry-run"
                            )
                    if result.get("applied") is not True:
                        raise RuntimeError("migration did not report an applied transaction")
                    applied_now.append(index)
                verification = _verify_applied_database(database, expected=expected)
            total_rollback_records += int(verification["rollback_record_count"])
            progress_row = {
                "at": _now(),
                "started_at": started_at,
                "index": index,
                "question_id": str(worker.get("question_id") or ""),
                "database": str(database),
                "action": action,
                "status": "completed",
                **verification,
            }
            _append_jsonl(args.progress, progress_row)
        except BaseException as exc:
            failure = {
                "at": _now(),
                "started_at": started_at,
                "index": index,
                "question_id": str(worker.get("question_id") or ""),
                "database": str(database),
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            _append_jsonl(args.progress, failure)
            break

    terminal_completed = completed_indices | set(applied_now) | set(resumed_indices)
    report = {
        "schema_version": "tmcra.v4.remaining400-provenance-apply.1",
        "migration_version": MIGRATION_VERSION,
        "status": "complete" if failure is None and len(terminal_completed) == 400 else "failed",
        "completed_at": _now(),
        "run_dir": str(run_dir),
        "dry_run_report": str(dry_run_path),
        "manifest_database_count": 400,
        "completed_database_count": len(terminal_completed),
        "applied_now_count": len(applied_now),
        "resumed_database_count": len(resumed_indices),
        "rollback_record_count": total_rollback_records,
        "added_offset_count": int(dry_run.get("added_offset_count") or 0),
        "physical_api_calls": 0,
        "failure": failure,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
