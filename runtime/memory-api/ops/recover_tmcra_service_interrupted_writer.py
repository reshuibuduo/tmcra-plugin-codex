#!/usr/bin/env python3
"""Audit and reopen one local Writer call interrupted by host process loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tmcra_v4_batch_writer as v4
from tmcra_service.writer import LOCAL_QWEN_MODEL


SCHEMA_VERSION = "tmcra.service.local-writer-process-loss-recovery.1"


class RecoveryAuditError(RuntimeError):
    """Raised when an interrupted call cannot be proven safe to replace."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RecoveryAuditError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RecoveryAuditError(f"{label} must be an object")
    return value


def _jsonl_call_count(path: Path, call_key: str) -> int:
    if not path.exists():
        return 0
    count = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecoveryAuditError(
                f"{path.name}:{line_number} is not valid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise RecoveryAuditError(
                f"{path.name}:{line_number} must contain an object"
            )
        count += int(str(value.get("call_key") or "") == call_key)
    return count


def _source_text(message: Mapping[str, Any]) -> str:
    spans = message.get("source_spans")
    if not isinstance(spans, list) or not spans:
        raise RecoveryAuditError("Writer request message has no source spans")
    parts: list[str] = []
    for span in spans:
        if not isinstance(span, Mapping) or not isinstance(span.get("text"), str):
            raise RecoveryAuditError("Writer request source span is malformed")
        parts.append(str(span["text"]))
    return "".join(parts)


def _verify_source_binding(
    connection: sqlite3.Connection,
    *,
    operation_id: str,
    scope_id: str,
    session_id: str,
    message: Mapping[str, Any],
    expected_status: str,
) -> str:
    message_id = str(message.get("message_id") or "")
    if not message_id:
        raise RecoveryAuditError("Writer request message ID is missing")
    source = connection.execute(
        "SELECT session_id,message_id,session_index,message_index,message_role,"
        "timestamp,content,content_sha256,status,source_record_id,"
        "source_turn_index,source_persisted_at FROM v4_source_journal "
        "WHERE scope_id=? AND message_id=?",
        (scope_id, message_id),
    ).fetchone()
    service = connection.execute(
        "SELECT session_id,message_index,role,timestamp,content_sha256,"
        "first_operation_id FROM tmcra_service_messages "
        "WHERE scope_id=? AND internal_message_id=?",
        (scope_id, message_id),
    ).fetchone()
    if source is None or service is None:
        raise RecoveryAuditError(f"{message_id}: Source or service row is missing")
    content = str(source["content"] or "")
    content_sha256 = _sha256_text(content)
    source_record_id = str(source["source_record_id"] or "")
    if (
        str(source["session_id"] or "") != session_id
        or str(source["message_role"] or "")
        != str(message.get("message_role") or "")
        or str(source["timestamp"] or "") != str(message.get("timestamp") or "")
        or str(source["content_sha256"] or "") != content_sha256
        or _source_text(message) != content
        or str(source["status"] or "") != expected_status
        or not source_record_id
        or not str(source["source_persisted_at"] or "")
        or str(service["session_id"] or "") != session_id
        or int(service["message_index"]) != int(source["message_index"])
        or str(service["role"] or "") != str(source["message_role"] or "")
        or str(service["timestamp"] or "") != str(source["timestamp"] or "")
        or str(service["content_sha256"] or "") != content_sha256
        or str(service["first_operation_id"] or "") != operation_id
    ):
        raise RecoveryAuditError(f"{message_id}: immutable Source binding differs")
    record = connection.execute(
        "SELECT turn_index,metadata_json FROM records "
        "WHERE scope_id=? AND memory_id=?",
        (scope_id, source_record_id),
    ).fetchone()
    if record is None:
        raise RecoveryAuditError(f"{message_id}: Source graph record is missing")
    metadata = _load_json_object(
        str(record["metadata_json"] or "{}"), f"{message_id} record metadata"
    )
    sidecar = metadata.get("sidecar_hint_metadata")
    sidecar = sidecar if isinstance(sidecar, Mapping) else {}
    actor_role = (
        metadata.get("actor_role")
        or metadata.get("speaker")
        or sidecar.get("role")
        or ""
    )
    raw_content = metadata.get("raw_content")
    if (
        not isinstance(raw_content, str)
        or _sha256_text(raw_content) != content_sha256
        or str(metadata.get("source_record_id") or "") != source_record_id
        or int(record["turn_index"]) != int(source["source_turn_index"])
        or int(metadata.get("session_index", -1)) != int(source["session_index"])
        or int(metadata.get("message_index", -1)) != int(source["message_index"])
        or str(actor_role) != str(source["message_role"] or "")
    ):
        raise RecoveryAuditError(f"{message_id}: Source graph metadata differs")
    return message_id


def audit_recovery(
    *,
    database: Path,
    operation_dir: Path,
    operation_id: str,
    batch_id: str,
) -> dict[str, Any]:
    if not database.is_file():
        raise RecoveryAuditError("native memory database is missing")
    if not operation_dir.is_dir():
        raise RecoveryAuditError("operation directory is missing")
    if (operation_dir / "commit.json").exists():
        raise RecoveryAuditError("operation already has a durable commit")
    input_path = operation_dir / "input.json"
    if not input_path.is_file():
        raise RecoveryAuditError("operation input artifact is missing")
    input_rows = json.loads(input_path.read_text(encoding="utf-8"))
    if (
        not isinstance(input_rows, list)
        or not input_rows
        or any(
            not isinstance(row, Mapping)
            or str(row.get("operation_id") or "") != operation_id
            for row in input_rows
        )
    ):
        raise RecoveryAuditError("operation input identity is invalid")

    with closing(sqlite3.connect(database, timeout=30.0)) as connection:
        connection.row_factory = sqlite3.Row
        quick = connection.execute("PRAGMA quick_check").fetchone()
        if not quick or quick[0] != "ok":
            raise RecoveryAuditError("native memory database quick_check failed")
        rows = connection.execute(
            "SELECT journal.*,batches.operation_id,batches.local_batch_index "
            "FROM tmcra_service_batches AS batches "
            "JOIN v4_batch_journal AS journal "
            "ON journal.scope_id=batches.scope_id "
            "AND journal.session_id=batches.session_id "
            "AND journal.batch_index=batches.batch_index "
            "WHERE batches.operation_id=? ORDER BY batches.local_batch_index",
            (operation_id,),
        ).fetchall()
        if not rows:
            raise RecoveryAuditError("operation has no Writer batch journal")
        interrupted = [row for row in rows if str(row["batch_id"]) == batch_id]
        if len(interrupted) != 1:
            raise RecoveryAuditError("requested Writer batch is not unique")
        target = interrupted[0]
        if (
            str(target["status"] or "") != "api_started"
            or str(target["response_json"] or "")
            or str(target["error"] or "")
        ):
            raise RecoveryAuditError(
                "Writer batch is not an unanswered api_started call"
            )
        if any(
            str(row["status"] or "") != "committed"
            for row in rows
            if str(row["batch_id"] or "") != batch_id
        ):
            raise RecoveryAuditError(
                "another Writer batch in the operation is not committed"
            )
        request_raw = str(target["request_json"] or "")
        if _sha256_text(request_raw) != str(target["request_sha256"] or ""):
            raise RecoveryAuditError("Writer request hash differs")
        request = _load_json_object(request_raw, "Writer request")
        if str(request.get("batch_id") or "") != batch_id:
            raise RecoveryAuditError("Writer request batch identity differs")
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RecoveryAuditError("Writer request has no messages")

        requested_ids: set[str] = set()
        operation_ids: set[str] = set()
        for row in rows:
            row_request = _load_json_object(
                str(row["request_json"] or ""), "operation Writer request"
            )
            row_messages = row_request.get("messages")
            if not isinstance(row_messages, list) or not row_messages:
                raise RecoveryAuditError("operation Writer request has no messages")
            expected_status = (
                "pending" if str(row["batch_id"] or "") == batch_id else "enriched"
            )
            for message in row_messages:
                if not isinstance(message, Mapping):
                    raise RecoveryAuditError("Writer request message is malformed")
                message_id = _verify_source_binding(
                    connection,
                    operation_id=operation_id,
                    scope_id=str(row["scope_id"] or ""),
                    session_id=str(row["session_id"] or ""),
                    message=message,
                    expected_status=expected_status,
                )
                if message_id in operation_ids:
                    raise RecoveryAuditError("operation repeats a Writer message ID")
                operation_ids.add(message_id)
                if expected_status == "pending":
                    requested_ids.add(message_id)
        source_rows = connection.execute(
            "SELECT journal.message_id FROM v4_source_journal AS journal "
            "JOIN tmcra_service_messages AS messages "
            "ON messages.scope_id=journal.scope_id "
            "AND messages.internal_message_id=journal.message_id "
            "WHERE messages.first_operation_id=?",
            (operation_id,),
        ).fetchall()
        source_ids = {str(row[0] or "") for row in source_rows}
        if source_ids != operation_ids:
            raise RecoveryAuditError("operation Source set differs from Writer requests")

    call_key = f"flash:{batch_id}"
    raw_count = _jsonl_call_count(
        operation_dir / "product_writer_raw_responses.jsonl", call_key
    )
    call_count = _jsonl_call_count(
        operation_dir / "product_writer_calls.jsonl", call_key
    )
    if raw_count or call_count:
        raise RecoveryAuditError(
            "interrupted call has a durable response or call artifact"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "operation_id": operation_id,
        "batch_id": batch_id,
        "scope_id": str(target["scope_id"] or ""),
        "session_id": str(target["session_id"] or ""),
        "batch_index": int(target["batch_index"]),
        "operation_batch_count": len(rows),
        "pending_source_count": len(requested_ids),
        "operation_source_count": len(operation_ids),
        "durable_raw_response_count": raw_count,
        "durable_call_record_count": call_count,
        "request_sha256": str(target["request_sha256"] or ""),
        "api_started_at": str(target["api_started_at"] or ""),
        "audit_passed": True,
    }


def apply_recovery(
    *,
    database: Path,
    operation_dir: Path,
    audit: Mapping[str, Any],
    model: str,
) -> dict[str, Any]:
    batch_id = str(audit["batch_id"])
    batch = v4.SourceBatch(
        scope_id=str(audit["scope_id"]),
        session_id=str(audit["session_id"]),
        session_index=0,
        batch_index=int(audit["batch_index"]),
        messages=(),
    )
    if batch.batch_id != batch_id:
        raise RecoveryAuditError("reconstructed Writer batch identity differs")
    writer = v4.V4BatchWriter(
        store=v4.V4BatchStore(database),
        flash_client=object(),
        log_dir=operation_dir,
        recover_interrupted_api_calls=True,
    )
    call_key = f"flash:{batch_id}"
    writer._assert_interrupted_call_has_no_response(call_key)
    writer._record_interrupted_call(
        call_key=call_key,
        batch=batch,
        stage="batch_flash_interrupted",
        model=model,
        job_id=str(audit["operation_id"]),
    )
    recovered = writer.store.abandon_interrupted_batch_call(batch_id)
    if str(recovered["status"] or "") != "prepared":
        raise RecoveryAuditError("Writer batch did not return to prepared")
    result = dict(audit)
    result.update(
        {
            "applied": True,
            "model": model,
            "replacement_call_authorized": True,
            "result_status": "prepared",
            "interrupted_call_artifact": str(
                operation_dir / "product_writer_interrupted_calls.jsonl"
            ),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit one local Writer call interrupted by confirmed host process loss"
        )
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--operation-dir", type=Path, required=True)
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--model", default=LOCAL_QWEN_MODEL)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--confirm-batch-id",
        help="must exactly equal --batch-id when --apply is used",
    )
    args = parser.parse_args()
    audit = audit_recovery(
        database=args.database.resolve(),
        operation_dir=args.operation_dir.resolve(),
        operation_id=args.operation_id,
        batch_id=args.batch_id,
    )
    if not args.apply:
        print(json.dumps({**audit, "applied": False}, sort_keys=True))
        return 0
    if args.confirm_batch_id != args.batch_id:
        raise RecoveryAuditError(
            "--confirm-batch-id must exactly equal --batch-id"
        )
    result = apply_recovery(
        database=args.database.resolve(),
        operation_dir=args.operation_dir.resolve(),
        audit=audit,
        model=str(args.model),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
