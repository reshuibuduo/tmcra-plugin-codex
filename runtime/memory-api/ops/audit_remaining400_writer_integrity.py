#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REQUIRED_TABLES = {
    "records",
    "v4_batch_journal",
    "v4_message_commit_journal",
    "v4_source_journal",
    "v4_reconciliation_jobs",
    "v4_interactions",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        return sum(1 for line in handle if line.strip())


def _status_counts(connection: sqlite3.Connection, table: str) -> dict[str, int]:
    return {
        str(status): int(count)
        for status, count in connection.execute(
            f"SELECT status,COUNT(*) FROM {table} GROUP BY status"
        )
    }


def _input_message_count(rows: Any) -> int:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("input is not a non-empty benchmark row array")
    count = 0
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"input row {row_index} is not an object")
        sessions = row.get("haystack_sessions")
        session_ids = row.get("haystack_session_ids")
        dates = row.get("haystack_dates")
        if not isinstance(sessions, list) or not sessions:
            raise RuntimeError(f"input row {row_index} lacks haystack_sessions")
        if not isinstance(session_ids, list) or len(session_ids) != len(sessions):
            raise RuntimeError(f"input row {row_index} session IDs do not align")
        if not isinstance(dates, list) or len(dates) != len(sessions):
            raise RuntimeError(f"input row {row_index} dates do not align")
        for session_index, session in enumerate(sessions):
            if not isinstance(session, list):
                raise RuntimeError(
                    f"input row {row_index} session {session_index} is not an array"
                )
            count += len(session)
    return count


def _audit_worker(worker: Mapping[str, Any]) -> dict[str, Any]:
    index = int(worker["worker_index"])
    worker_dir = Path(str(worker["worker_dir"])).resolve()
    input_path = Path(str(worker["input"])).resolve()
    database = worker_dir / "native_memory.sqlite3"
    report_path = worker_dir / "product_writer_report.json"
    errors: list[str] = []

    try:
        input_rows = json.loads(input_path.read_text(encoding="utf-8"))
        input_count = _input_message_count(input_rows)
    except Exception as exc:
        errors.append(f"input read failed: {exc}")
        input_count = 0

    try:
        report = dict(_load_object(report_path))
    except Exception as exc:
        errors.append(f"Writer report read failed: {exc}")
        report = {}

    if report.get("completed") is not True:
        errors.append("Writer report is not complete")
    reported_db = str(report.get("db_path") or "")
    if reported_db and Path(reported_db).resolve() != database.resolve():
        errors.append("Writer report database path differs from manifest worker")

    statuses: dict[str, dict[str, int]] = {}
    semantic_commit_mismatches = 0
    quick_check = "missing"
    record_count = 0
    try:
        with sqlite3.connect(database) as connection:
            quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            missing_tables = sorted(REQUIRED_TABLES - tables)
            if missing_tables:
                errors.append("missing tables: " + ",".join(missing_tables))
            else:
                for table in (
                    "v4_batch_journal",
                    "v4_message_commit_journal",
                    "v4_source_journal",
                    "v4_reconciliation_jobs",
                    "v4_interactions",
                ):
                    statuses[table] = _status_counts(connection, table)
                for message_id, plan_json, semantic_committed in connection.execute(
                    "SELECT message_id,plan_json,semantic_committed "
                    "FROM v4_message_commit_journal"
                ):
                    try:
                        plan = json.loads(plan_json)
                        decisions = plan.get("decisions") or {}
                        if not isinstance(decisions, Mapping):
                            raise TypeError("decisions is not an object")
                        expected_committed = sum(
                            str(decision) != "quarantine"
                            for decision in decisions.values()
                        )
                        if int(semantic_committed) != expected_committed:
                            semantic_commit_mismatches += 1
                    except Exception as exc:
                        errors.append(
                            f"message {message_id} semantic plan is invalid: {exc}"
                        )
                record_count = int(
                    connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
                )
    except Exception as exc:
        errors.append(f"SQLite audit failed: {exc}")

    if quick_check != "ok":
        errors.append(f"SQLite quick_check={quick_check}")
    expected_states = {
        "v4_batch_journal": {"committed"},
        "v4_message_commit_journal": {"committed"},
        "v4_source_journal": {"enriched"},
        "v4_reconciliation_jobs": {"completed"},
    }
    for table, expected in expected_states.items():
        actual = set(statuses.get(table) or {})
        if actual - expected:
            errors.append(f"{table} has nonterminal states: {sorted(actual - expected)}")
    if semantic_commit_mismatches:
        errors.append(
            "message semantic commit count mismatches plan decisions: "
            f"{semantic_commit_mismatches}"
        )

    message_count = sum((statuses.get("v4_message_commit_journal") or {}).values())
    source_count = sum((statuses.get("v4_source_journal") or {}).values())
    batch_count = sum((statuses.get("v4_batch_journal") or {}).values())
    for label, value in (("input_messages", input_count), ("source_messages", source_count)):
        if report and int(report.get(label) or -1) != value:
            errors.append(f"report {label} does not equal durable count")
    excluded_empty = int(report.get("excluded_empty_source_messages") or 0)
    if message_count != source_count or source_count + excluded_empty != input_count:
        errors.append(
            "input/message/source/excluded counts disagree: "
            f"{input_count}/{message_count}/{source_count}/{excluded_empty}"
        )
    if report and int(report.get("batches") or -1) != batch_count:
        errors.append("report batch count does not equal journal count")

    return {
        "index": index,
        "question_id": str(worker.get("question_id") or ""),
        "worker_dir": str(worker_dir),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "quick_check": quick_check,
        "input_messages": input_count,
        "batch_count": batch_count,
        "message_count": message_count,
        "source_count": source_count,
        "record_count": record_count,
        "statuses": statuses,
        "prompt_version": str(report.get("prompt_version") or ""),
        "writer_schema_version": str(report.get("writer_schema_version") or ""),
        "candidate_selector_version": str(
            report.get("candidate_selector_version") or ""
        ),
        "validation_warnings": int(report.get("validation_warnings") or 0),
        "reported_reconciliation_quarantines": int(
            report.get("reconciliation_response_quarantines") or 0
        ),
        "durable_reconciliation_quarantines": _jsonl_rows(
            worker_dir / "product_writer_reconciliation_quarantines.jsonl"
        ),
        "excluded_empty_source_messages": int(
            report.get("excluded_empty_source_messages") or 0
        ),
        "incomplete_call_recoveries": int(
            report.get("incomplete_call_recoveries") or 0
        ),
        "interrupted_call_recoveries": int(
            report.get("interrupted_call_recoveries") or 0
        ),
        "database_bytes": database.stat().st_size if database.is_file() else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit exact Writer integrity for a frozen remaining400 manifest."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = _load_object(run_dir / "input_manifest.json")
    workers = list(manifest.get("workers") or [])
    indices = [int(worker["worker_index"]) for worker in workers]
    if len(workers) != 400 or len(set(indices)) != 400:
        raise RuntimeError("remaining400 manifest must contain 400 unique workers")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.concurrency, 32))) as executor:
        futures = {executor.submit(_audit_worker, worker): worker for worker in workers}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: int(row["index"]))

    versions: dict[str, dict[str, int]] = {}
    for field in (
        "prompt_version",
        "writer_schema_version",
        "candidate_selector_version",
    ):
        versions[field] = dict(Counter(str(row[field]) for row in results))
    failures = [row for row in results if row["status"] != "passed"]
    report = {
        "schema_version": "tmcra.v4.remaining400-writer-integrity-audit.1",
        "status": "passed" if not failures else "failed",
        "completed_at": _now(),
        "run_dir": str(run_dir),
        "worker_count": len(results),
        "passed_workers": len(results) - len(failures),
        "failed_workers": len(failures),
        "failure_indices": [int(row["index"]) for row in failures],
        "total_input_messages": sum(int(row["input_messages"]) for row in results),
        "total_batches": sum(int(row["batch_count"]) for row in results),
        "total_records": sum(int(row["record_count"]) for row in results),
        "total_validation_warnings": sum(
            int(row["validation_warnings"]) for row in results
        ),
        "total_durable_reconciliation_quarantines": sum(
            int(row["durable_reconciliation_quarantines"]) for row in results
        ),
        "total_excluded_empty_source_messages": sum(
            int(row["excluded_empty_source_messages"]) for row in results
        ),
        "total_incomplete_call_recoveries": sum(
            int(row["incomplete_call_recoveries"]) for row in results
        ),
        "total_interrupted_call_recoveries": sum(
            int(row["interrupted_call_recoveries"]) for row in results
        ),
        "total_database_bytes": sum(int(row["database_bytes"]) for row in results),
        "versions": versions,
        "failures": failures,
        "workers": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key not in {"workers"}}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
