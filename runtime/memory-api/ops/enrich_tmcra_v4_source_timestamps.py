#!/usr/bin/env python3
"""Bind persisted source timestamps to frozen V4 retrieval evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class TimestampEnrichmentError(RuntimeError):
    pass


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_metadata(
    db_path: Path, scope_id: str, source_record_ids: set[str]
) -> dict[str, dict[str, Any]]:
    if not db_path.is_file():
        raise TimestampEnrichmentError(f"source database is missing: {db_path}")
    output: dict[str, dict[str, Any]] = {}
    connection = sqlite3.connect(db_path)
    try:
        for source_record_id in sorted(source_record_ids):
            row = connection.execute(
                "SELECT metadata_json FROM records WHERE scope_id=? AND memory_id=?",
                (scope_id, source_record_id),
            ).fetchone()
            if row is None:
                raise TimestampEnrichmentError(
                    f"{scope_id}: source record is missing: {source_record_id}"
                )
            try:
                metadata = json.loads(row[0])
            except (TypeError, json.JSONDecodeError) as exc:
                raise TimestampEnrichmentError(
                    f"{source_record_id}: source metadata is invalid JSON"
                ) from exc
            if not isinstance(metadata, dict):
                raise TimestampEnrichmentError(
                    f"{source_record_id}: source metadata is not an object"
                )
            output[source_record_id] = metadata
    finally:
        connection.close()
    return output


def _enrich_source(
    source: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    scope_id: str,
) -> dict[str, Any]:
    source_record_id = _text(source.get("source_record_id"))
    session_id = _text(source.get("session_id"))
    try:
        session_index = int(source["session_index"])
        parent_chunk_index = int(source["parent_chunk_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TimestampEnrichmentError(
            f"{source_record_id}: source coordinates are invalid"
        ) from exc
    timestamp = _text(metadata.get("timestamp"))
    historical_date = _text(
        metadata.get("historical_date")
        or dict(metadata.get("sidecar_hint_metadata") or {}).get("historical_date")
    )
    message_role = _text(
        metadata.get("speaker")
        or dict(metadata.get("sidecar_hint_metadata") or {}).get("role")
    )
    metadata_scope_id = _text(metadata.get("scope_id"))
    if (
        _text(metadata.get("content_variant")) != "source_message"
        or _text(metadata.get("source_record_id")) != source_record_id
        or (metadata_scope_id and metadata_scope_id != scope_id)
        or _text(metadata.get("session_id")) != session_id
        or int(metadata.get("session_index", -1)) != session_index
        or int(metadata.get("message_index", -1)) != parent_chunk_index
        or metadata.get("raw_content") != source.get("text")
        or not timestamp
        or not historical_date
        or message_role not in {"user", "assistant", "system", "tool"}
    ):
        raise TimestampEnrichmentError(
            f"{source_record_id}: persisted source identity or temporal metadata differs"
        )
    enriched = dict(source)
    for field, value in (
        ("historical_date", historical_date),
        ("timestamp", timestamp),
        ("message_role", message_role),
    ):
        existing = _text(enriched.get(field))
        if existing and existing != value:
            raise TimestampEnrichmentError(
                f"{source_record_id}: frozen {field} conflicts with persisted source"
            )
        enriched[field] = value
    return enriched


def enrich_row(row: Mapping[str, Any]) -> dict[str, Any]:
    qid = _text(row.get("question_id"))
    windows = row.get("evidence_windows")
    if not qid or not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)):
        raise TimestampEnrichmentError("retrieval row lacks question ID or evidence windows")
    grouped: dict[tuple[Path, str], set[str]] = {}
    for window in windows:
        if not isinstance(window, Mapping):
            raise TimestampEnrichmentError(f"{qid}: evidence window is not an object")
        raw_db_path = _text(window.get("db_path"))
        db_path = Path(raw_db_path)
        scope_id = _text(window.get("scope_id"))
        source_record_id = _text(window.get("source_record_id"))
        if not raw_db_path or not scope_id or not source_record_id:
            raise TimestampEnrichmentError(
                f"{qid}: evidence window lacks database, scope, or source identity"
            )
        key = (db_path, scope_id)
        grouped.setdefault(key, set()).add(source_record_id)
        context = window.get("source_group_context") or []
        if not isinstance(context, Sequence) or isinstance(context, (str, bytes)):
            raise TimestampEnrichmentError(f"{qid}: source group context is invalid")
        for member in context:
            if not isinstance(member, Mapping):
                raise TimestampEnrichmentError(
                    f"{qid}: source group context member is not an object"
                )
            context_source_id = _text(member.get("source_record_id"))
            if not context_source_id:
                raise TimestampEnrichmentError(
                    f"{qid}: source group context member lacks source identity"
                )
            grouped[key].add(context_source_id)

    metadata_by_key = {
        key: _source_metadata(key[0], key[1], source_ids)
        for key, source_ids in grouped.items()
    }
    enriched_windows: list[dict[str, Any]] = []
    for window in windows:
        db_path = Path(_text(window.get("db_path")))
        scope_id = _text(window.get("scope_id"))
        metadata = metadata_by_key[(db_path, scope_id)]
        source_record_id = _text(window.get("source_record_id"))
        enriched = _enrich_source(
            window, metadata[source_record_id], scope_id=scope_id
        )
        enriched["source_group_context"] = [
            _enrich_source(
                member,
                metadata[_text(member.get("source_record_id"))],
                scope_id=scope_id,
            )
            for member in list(window.get("source_group_context") or [])
        ]
        enriched_windows.append(enriched)
    output = dict(row)
    output["evidence_windows"] = enriched_windows
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--qid-list", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise TimestampEnrichmentError(f"output already exists: {args.out}")
    rows = _read_jsonl(args.evidence)
    if args.qid_list:
        wanted = [
            line.strip()
            for line in args.qid_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not wanted or len(wanted) != len(set(wanted)):
            raise TimestampEnrichmentError("qid list is empty or contains duplicates")
        by_qid = {_text(row.get("question_id")): row for row in rows}
        missing = [qid for qid in wanted if qid not in by_qid]
        if missing:
            raise TimestampEnrichmentError(f"qid list contains unknown rows: {missing}")
        rows = [by_qid[qid] for qid in wanted]
    enriched = [enrich_row(row) for row in rows]
    _write_jsonl_atomic(args.out, enriched)
    report = {
        "status": "complete",
        "row_count": len(enriched),
        "source_window_count": sum(len(row["evidence_windows"]) for row in enriched),
        "source_context_count": sum(
            len(window.get("source_group_context") or [])
            for row in enriched
            for window in row["evidence_windows"]
        ),
        "input_sha256": hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "api_call_count": 0,
    }
    report_path = args.report or args.out.with_suffix(args.out.suffix + ".report.json")
    _write_json_atomic(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
