#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping


ACTIVE_STATES = {"active", "parallel_active", "promoted"}


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return value


def _row(connection: sqlite3.Connection, query: str, values: tuple[Any, ...]) -> dict[str, Any] | None:
    row = connection.execute(query, values).fetchone()
    return dict(row) if row is not None else None


def _record_snapshot(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    metadata = _decode(row.get("metadata_json"))
    return {
        "memory_id": row.get("memory_id"),
        "slot_key": row.get("slot_key"),
        "turn_index": row.get("turn_index"),
        "state": row.get("state"),
        "value": row.get("value"),
        "supersedes": _decode(row.get("supersedes_json")),
        "metadata": metadata,
    }


def _all_occurrences(content: str, quote: str) -> list[int]:
    if not quote:
        return []
    output: list[int] = []
    cursor = 0
    while True:
        index = content.find(quote, cursor)
        if index < 0:
            return output
        output.append(index)
        cursor = index + 1


def _journal_assertions(
    connection: sqlite3.Connection,
    *,
    message_id: str,
    evidence_span_id: str,
    proposal_index: int,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for batch_id, response_json in connection.execute(
        "SELECT batch_id,response_json FROM v4_batch_journal "
        "WHERE status='committed' ORDER BY batch_index"
    ):
        response = _decode(response_json)
        messages = response.get("messages") if isinstance(response, Mapping) else None
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, Mapping) or str(message.get("message_id") or "") != message_id:
                continue
            v3 = message.get("v3")
            assertions = v3.get("assertions") if isinstance(v3, Mapping) else None
            if not isinstance(assertions, list):
                continue
            for index, assertion in enumerate(assertions):
                if not isinstance(assertion, Mapping):
                    continue
                same_span = (
                    evidence_span_id
                    and str(assertion.get("evidence_span_id") or "") == evidence_span_id
                )
                if same_span or index == proposal_index:
                    matches.append(
                        {
                            "batch_id": str(batch_id),
                            "assertion_index": index,
                            "same_evidence_span_id": bool(same_span),
                            "assertion": dict(assertion),
                        }
                    )
    return matches


def _fast_diagnostic(connection: sqlite3.Connection, memory_id: str) -> dict[str, Any]:
    record = _row(
        connection,
        "SELECT * FROM records WHERE memory_id=?",
        (memory_id,),
    )
    if record is None:
        return {"kind": "fast_grounding", "memory_id": memory_id, "error": "record missing"}
    metadata = _decode(record.get("metadata_json"))
    if not isinstance(metadata, Mapping):
        return {"kind": "fast_grounding", "memory_id": memory_id, "error": "metadata invalid"}
    source_id = str(metadata.get("source_record_id") or "")
    source = _row(
        connection,
        "SELECT * FROM records WHERE scope_id=? AND memory_id=?",
        (record.get("scope_id"), source_id),
    )
    source_metadata = _decode(source.get("metadata_json")) if source else {}
    content = (
        str(source_metadata.get("raw_content") or "")
        if isinstance(source_metadata, Mapping)
        else ""
    )
    quote = str(
        metadata.get("evidence_quote")
        or metadata.get("raw_content")
        or metadata.get("source_span")
        or ""
    ).strip()
    start = metadata.get("evidence_char_start")
    end = metadata.get("evidence_char_end")
    try:
        start_value, end_value = int(start), int(end)
        actual_slice = content[start_value:end_value]
    except (TypeError, ValueError):
        start_value, end_value, actual_slice = None, None, ""
    proposal_index = int(metadata.get("llm_write_proposal_index", -1) or -1)
    journal = _journal_assertions(
        connection,
        message_id=str(metadata.get("message_id") or ""),
        evidence_span_id=str(metadata.get("evidence_span_id") or ""),
        proposal_index=proposal_index,
    )
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    edges: list[dict[str, Any]] = []
    if "memory_edges" in tables:
        for edge in connection.execute(
            "SELECT * FROM memory_edges WHERE source_memory_id=?",
            (memory_id,),
        ):
            value = dict(edge)
            if "metadata_json" in value:
                value["metadata"] = _decode(value.pop("metadata_json"))
            edges.append(value)
    return {
        "kind": "fast_grounding",
        "memory_id": memory_id,
        "record": _record_snapshot(record),
        "source_record_id": source_id,
        "source_message_id": (
            source_metadata.get("message_id") if isinstance(source_metadata, Mapping) else ""
        ),
        "source_raw_content": content,
        "evidence_quote": quote,
        "persisted_offsets": [start, end],
        "persisted_slice": actual_slice,
        "slice_equals_quote": actual_slice == quote,
        "exact_quote_occurrences": _all_occurrences(content, quote),
        "journal_assertions": journal,
        "outgoing_edges": edges,
    }


def _slot_head_diagnostic(connection: sqlite3.Connection, memory_id: str) -> dict[str, Any]:
    head = _row(
        connection,
        "SELECT * FROM slot_heads WHERE memory_id=?",
        (memory_id,),
    )
    if head is None:
        return {"kind": "slot_head", "memory_id": memory_id, "error": "head missing"}
    scope_id = str(head.get("scope_id") or "")
    slot_key = str(head.get("slot_key") or "")
    records = [
        _record_snapshot(dict(row))
        for row in connection.execute(
            "SELECT * FROM records WHERE scope_id=? AND slot_key=? "
            "ORDER BY turn_index,memory_id",
            (scope_id, slot_key),
        )
    ]
    history = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM slot_history WHERE scope_id=? AND slot_key=? ORDER BY ordinal",
            (scope_id, slot_key),
        )
    ]
    return {
        "kind": "slot_head",
        "memory_id": memory_id,
        "head": dict(head),
        "records": records,
        "active_record_ids": [
            str(record.get("memory_id"))
            for record in records
            if isinstance(record, Mapping) and str(record.get("state")) in ACTIVE_STATES
        ],
        "slot_history": history,
    }


def _keep_parallel_diagnostic(connection: sqlite3.Connection, job_id: str) -> dict[str, Any]:
    job = _row(
        connection,
        "SELECT * FROM v4_reconciliation_jobs WHERE job_id=?",
        (job_id,),
    )
    if job is None:
        return {"kind": "keep_parallel", "job_id": job_id, "error": "job missing"}
    request = _decode(job.get("request_json"))
    response = _decode(job.get("response_json"))
    scope_id = str(job.get("scope_id") or "")
    selected_id = (
        str(response.get("selected_memory_id") or "")
        if isinstance(response, Mapping)
        else ""
    )
    selected = _row(
        connection,
        "SELECT * FROM records WHERE scope_id=? AND memory_id=?",
        (scope_id, selected_id),
    )
    selected_metadata = _decode(selected.get("metadata_json")) if selected else {}
    incoming_id = (
        str(selected_metadata.get("superseded_by") or "")
        if isinstance(selected_metadata, Mapping)
        else ""
    )
    incoming = _row(
        connection,
        "SELECT * FROM records WHERE scope_id=? AND memory_id=?",
        (scope_id, incoming_id),
    )
    return {
        "kind": "keep_parallel",
        "job_id": job_id,
        "job": {
            key: (_decode(value) if key.endswith("_json") else value)
            for key, value in job.items()
        },
        "request": request,
        "response": response,
        "selected": _record_snapshot(selected),
        "incoming": _record_snapshot(incoming),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = _load_object(args.audit_report.resolve())
    results: list[dict[str, Any]] = []
    for failure in list(report.get("failures") or []):
        database = Path(str(failure["database"])).resolve()
        worker = {
            "index": int(failure["index"]),
            "question_id": str(failure.get("question_id") or ""),
            "database": str(database),
            "issues": list(failure.get("issues") or []),
            "diagnostics": [],
        }
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            seen: set[tuple[str, str]] = set()
            for issue in worker["issues"]:
                fast = re.match(r"fast leaf (.+?): (?:evidence|source)", str(issue))
                if fast and ("fast", fast.group(1)) not in seen:
                    seen.add(("fast", fast.group(1)))
                    worker["diagnostics"].append(
                        _fast_diagnostic(connection, fast.group(1))
                    )
                if "slot_heads targets non-active record " in str(issue):
                    memory_id = str(issue).rsplit(" ", 1)[-1]
                    if ("slot", memory_id) not in seen:
                        seen.add(("slot", memory_id))
                        worker["diagnostics"].append(
                            _slot_head_diagnostic(connection, memory_id)
                        )
                if "keep_parallel decision was overwritten by graph policy: " in str(issue):
                    job_id = str(issue).rsplit(": ", 1)[-1]
                    if ("keep", job_id) not in seen:
                        seen.add(("keep", job_id))
                        worker["diagnostics"].append(
                            _keep_parallel_diagnostic(connection, job_id)
                        )
        results.append(worker)
    output = {
        "schema_version": "tmcra.v4.remaining400-chain-failure-diagnostics.1",
        "status": "complete",
        "worker_count": len(results),
        "physical_api_calls": 0,
        "workers": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "worker_count": output["worker_count"],
                "physical_api_calls": 0,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
