from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(database: Path, sql: str) -> dict[str, dict[str, Any]]:
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            raise RuntimeError(f"{database}: SQLite quick_check failed: {quick_check}")
        return {str(row[0]): dict(row) for row in connection.execute(sql)}


def _verify_progress(source: Path, target: Path) -> dict[str, Any]:
    source_db = source / "native_memory.sqlite3"
    target_db = target / "native_memory.sqlite3"
    if _sha256(source / "input.json") != _sha256(target / "input.json"):
        raise RuntimeError("source and target Writer inputs differ")

    source_batches = _rows(
        source_db,
        "SELECT batch_id,status,request_sha256,response_sha256 FROM v4_batch_journal",
    )
    target_batches = _rows(
        target_db,
        "SELECT batch_id,status,request_sha256,response_sha256 FROM v4_batch_journal",
    )
    source_messages = _rows(
        source_db,
        "SELECT commit_id,status,plan_sha256,semantic_committed FROM v4_message_commit_journal",
    )
    target_messages = _rows(
        target_db,
        "SELECT commit_id,status,plan_sha256,semantic_committed FROM v4_message_commit_journal",
    )

    for batch_id, target_row in target_batches.items():
        source_row = source_batches.get(batch_id)
        if source_row is None:
            raise RuntimeError(f"source DB lost target batch {batch_id}")
        if target_row["request_sha256"] != source_row["request_sha256"]:
            raise RuntimeError(f"source DB changed frozen request {batch_id}")
        if target_row["status"] == "committed":
            if source_row["status"] != "committed":
                raise RuntimeError(f"source DB regressed committed batch {batch_id}")
            if target_row["response_sha256"] != source_row["response_sha256"]:
                raise RuntimeError(f"source DB changed committed response {batch_id}")

    for commit_id, target_row in target_messages.items():
        source_row = source_messages.get(commit_id)
        if source_row is None:
            raise RuntimeError(f"source DB lost target message commit {commit_id}")
        if target_row["status"] == "committed":
            if source_row["status"] != "committed":
                raise RuntimeError(f"source DB regressed committed message {commit_id}")
            if (
                target_row["plan_sha256"] != source_row["plan_sha256"]
                or target_row["semantic_committed"]
                != source_row["semantic_committed"]
            ):
                raise RuntimeError(f"source DB changed committed message {commit_id}")

    target_committed = sum(
        row["status"] == "committed" for row in target_batches.values()
    )
    source_committed = sum(
        row["status"] == "committed" for row in source_batches.values()
    )
    if source_committed <= target_committed:
        raise RuntimeError(
            f"smoke DB has no forward progress: {source_committed} <= {target_committed}"
        )
    return {
        "target_committed_batches": target_committed,
        "source_committed_batches": source_committed,
        "advanced_batches": source_committed - target_committed,
        "source_batch_statuses": {
            status: sum(row["status"] == status for row in source_batches.values())
            for status in sorted({row["status"] for row in source_batches.values()})
        },
    }


def _identity(filename: str, row: dict[str, Any]) -> str:
    keys = {
        "product_writer_calls.jsonl": ("call_key",),
        "product_writer_raw_responses.jsonl": ("call_key",),
        "product_write_messages.jsonl": ("message_key",),
        "product_writer_interrupted_calls.jsonl": ("call_key",),
        "product_writer_reconciliation_revalidations.jsonl": ("job_id",),
        "product_writer_revalidations.jsonl": ("batch_id",),
        "product_writer_validated_batch_recoveries.jsonl": ("batch_id",),
    }.get(filename, ("action_id", "job_id", "batch_id", "message_key", "call_key"))
    for key in keys:
        value = str(row.get(key) or "")
        if value:
            return f"{key}:{value}"
    return "row:" + hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _merge_jsonl(
    source: Path,
    target: Path,
    filename: str,
    *,
    write: bool,
) -> dict[str, int]:
    rows: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    counts = {"target": 0, "source": 0, "merged": 0}
    for label, path in (("target", target / filename), ("source", source / filename)):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identity = _identity(filename, row)
            prior = rows.get(identity)
            if prior is not None and prior != row:
                raise RuntimeError(f"{filename}: conflicting duplicate identity {identity}")
            if prior is None:
                rows[identity] = row
                order.append(identity)
            counts[label] += 1
    counts["merged"] = len(order)
    if not order or not write:
        return counts
    temporary = target / f".{filename}.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as handle:
        for identity in order:
            handle.write(json.dumps(rows[identity], ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target / filename)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    target = args.target_dir.resolve()
    progress = _verify_progress(source, target)

    jsonl_files = sorted(
        {path.name for path in source.glob("*.jsonl")}
        | {path.name for path in target.glob("*.jsonl")}
    )
    merge_counts: dict[str, dict[str, int]] = {}
    for filename in jsonl_files:
        merge_counts[filename] = _merge_jsonl(
            source,
            target,
            filename,
            write=args.apply,
        )
    if args.apply:
        database_tmp = target / f".native_memory.sqlite3.tmp.{os.getpid()}"
        shutil.copy2(source / "native_memory.sqlite3", database_tmp)
        with sqlite3.connect(database_tmp) as connection:
            if str(connection.execute("PRAGMA quick_check").fetchone()[0]) != "ok":
                raise RuntimeError("copied smoke database failed quick_check")
        database_tmp.replace(target / "native_memory.sqlite3")

    report = {
        "schema_version": "tmcra.v4.writer-smoke-promotion.1",
        "mode": "apply" if args.apply else "dry_run",
        "status": "complete",
        "source_dir": str(source),
        "target_dir": str(target),
        "input_sha256": _sha256(source / "input.json"),
        "progress": progress,
        "jsonl_merge_counts": merge_counts,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
