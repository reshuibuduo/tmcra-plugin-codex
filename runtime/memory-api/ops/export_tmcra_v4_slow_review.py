#!/usr/bin/env python3
"""Export raw Slow repair inputs and committed outputs for human review."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping


def _json(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _worker_names(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("workers must be a non-empty unique comma-separated list")
    return names


def _public_fast_record(row: sqlite3.Row) -> dict[str, Any]:
    metadata = _json(row["metadata_json"], {})
    return {
        "memory_id": str(row["memory_id"]),
        "state": str(row["state"]),
        "value": str(row["value"]),
        "canonical_slot": metadata.get("canonical_slot_key"),
        "durability": metadata.get("durability"),
        "temporal_status": metadata.get("temporal_status")
        or metadata.get("target_status"),
        "polarity": metadata.get("polarity"),
        "write_operation": metadata.get("write_operation"),
        "evidence_role": (
            "counterevidence"
            if bool(metadata.get("counterevidence"))
            or bool(metadata.get("is_counterevidence"))
            else "support"
        ),
    }


def _request_payload(call: Mapping[str, Any]) -> dict[str, Any]:
    request = call.get("request")
    if not isinstance(request, Mapping):
        return {}
    for message in request.get("messages") or []:
        if isinstance(message, Mapping) and message.get("role") == "user":
            payload = _json(message.get("content"), {})
            return payload if isinstance(payload, dict) else {}
    return {}


def _operation_errors(operation: Any) -> list[str]:
    if not isinstance(operation, Mapping):
        return ["operation must be an object"]
    action = operation.get("action")
    if not isinstance(action, str) or not action.strip():
        return ["operation.action must be a non-empty string"]
    if action not in {"create", "revise", "challenge", "resolve_challenge", "retire", "noop"}:
        return [f"unknown operation action: {action}"]
    return []


def _operation_capsules(
    con: sqlite3.Connection, patch_id: str
) -> dict[int, str]:
    try:
        rows = con.execute(
            "SELECT ordinal,capsule_id FROM slow_graph_patch_operations "
            "WHERE patch_id=? ORDER BY ordinal",
            (patch_id,),
        )
    except sqlite3.OperationalError:
        return {}
    return {
        int(row["ordinal"]): str(row["capsule_id"])
        for row in rows
        if row["capsule_id"]
    }


def _export_database(
    database: Path, *, worker: str, prompt_version: str, include_noop: bool
) -> list[dict[str, Any]]:
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as con:
        con.row_factory = sqlite3.Row
        records = list(
            con.execute("SELECT memory_id,state,value,metadata_json FROM records")
        )
        record_by_id = {str(row["memory_id"]): row for row in records}
        resulting_by_patch_capsule: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in records:
            metadata = _json(row["metadata_json"], {})
            patch_id = str(metadata.get("patch_id") or "")
            capsule_id = str(metadata.get("capsule_id") or "")
            if patch_id and capsule_id:
                resulting_by_patch_capsule.setdefault((patch_id, capsule_id), []).append(
                    {
                        "memory_id": str(row["memory_id"]),
                        "state": str(row["state"]),
                        "value": str(row["value"]),
                        "metadata": metadata,
                    }
                )

        def resulting_capsule(
            patch_id: str, capsule_id: str | None
        ) -> dict[str, Any] | None:
            if capsule_id:
                candidates = resulting_by_patch_capsule.get((patch_id, capsule_id), [])
            else:
                candidates = [
                    capsule
                    for (candidate_patch_id, _), values in resulting_by_patch_capsule.items()
                    if candidate_patch_id == patch_id
                    for capsule in values
                ]
            if not candidates:
                return None
            return max(
                candidates,
                key=lambda capsule: int(capsule["metadata"].get("revision", 0) or 0),
            )

        def context(
            *,
            job: sqlite3.Row,
            call: Mapping[str, Any],
            payload: Mapping[str, Any],
            patch: Mapping[str, Any],
            patch_id: str,
        ) -> dict[str, Any]:
            evidence_ids = [str(item) for item in _json(job["evidence_ids_json"], [])]
            return {
                "worker": worker,
                "job_id": str(job["job_id"]),
                "scope_id": str(job["scope_id"]),
                "region_key": str(job["region_key"]),
                "job_status": str(job["status"]),
                "job_attempts": int(job["attempts"]),
                "job_evidence": [
                    _public_fast_record(record_by_id[memory_id])
                    for memory_id in evidence_ids
                    if memory_id in record_by_id
                ],
                "route": call.get("route"),
                "route_reason": call.get("route_reason"),
                "prompt_version": call.get("prompt_version"),
                "physical_api_calls": int(call.get("physical_api_calls", 0) or 0),
                "usage": call.get("usage"),
                "cost_audit": call.get("cost_audit"),
                "tier_calls": call.get("tier_calls"),
                "request_region": payload.get("region"),
                "request_capsules": payload.get("capsules"),
                "patch_id": patch_id,
                "patch": patch,
            }

        output: list[dict[str, Any]] = []
        for job in con.execute(
            "SELECT * FROM slow_graph_jobs ORDER BY created_at,job_id"
        ):
            job_metadata = _json(job["metadata_json"], {})
            model_config = job_metadata.get("model_config")
            if not isinstance(model_config, Mapping) or model_config.get(
                "prompt_version"
            ) != prompt_version:
                continue
            patch_rows = list(
                con.execute(
                    "SELECT * FROM slow_graph_patches WHERE job_id=? "
                    "ORDER BY applied_at,patch_id",
                    (job["job_id"],),
                )
            )
            if len(patch_rows) != 1:
                output.append(
                    {
                        "worker": worker,
                        "job_id": str(job["job_id"]),
                        "region_key": str(job["region_key"]),
                        "job_status": str(job["status"]),
                        "operation_index": None,
                        "operation": None,
                        "review_errors": [f"expected one patch, found {len(patch_rows)}"],
                        "export_error": f"expected one patch, found {len(patch_rows)}",
                    }
                )
                continue
            patch_row = patch_rows[0]
            raw_patch = _json(patch_row["patch_json"], {})
            patch = raw_patch if isinstance(raw_patch, Mapping) else {}
            patch_id = str(patch_row["patch_id"])
            call_value = _json(patch_row["call_metadata_json"], {})
            call = call_value if isinstance(call_value, Mapping) else {}
            payload = _request_payload(call)
            base = context(
                job=job,
                call=call,
                payload=payload,
                patch=patch,
                patch_id=patch_id,
            )
            operations = patch.get("operations")
            if not isinstance(operations, list):
                output.append(
                    {
                        **base,
                        "operation_index": None,
                        "operation": None,
                        "capsule_id": None,
                        "review_errors": ["patch.operations must be a list"],
                        "resulting_capsule": None,
                    }
                )
                continue
            if not operations:
                output.append(
                    {
                        **base,
                        "operation_index": None,
                        "operation": None,
                        "capsule_id": None,
                        "review_errors": ["patch.operations is empty"],
                        "resulting_capsule": None,
                    }
                )
                continue
            operation_capsules = _operation_capsules(con, patch_id)
            patch_capsules = [
                capsule
                for (candidate_patch_id, _), values in resulting_by_patch_capsule.items()
                if candidate_patch_id == patch_id
                for capsule in values
            ]
            for operation_index, operation in enumerate(operations):
                errors = _operation_errors(operation)
                action = (
                    str(operation.get("action") or "")
                    if isinstance(operation, Mapping)
                    else ""
                )
                if action == "noop" and not include_noop and not errors:
                    continue
                capsule_id = (
                    str(operation.get("capsule_id") or "")
                    if isinstance(operation, Mapping)
                    else ""
                )
                capsule_id = capsule_id or operation_capsules.get(operation_index, "")
                if not errors and not capsule_id and len(patch_capsules) == 1:
                    capsule_id = str(patch_capsules[0]["metadata"].get("capsule_id") or "")
                output.append(
                    {
                        **base,
                        "operation_index": operation_index,
                        "operation": operation,
                        "capsule_id": capsule_id or None,
                        "review_errors": errors,
                        "resulting_capsule": (
                            resulting_capsule(patch_id, capsule_id or None)
                            if not errors
                            else None
                        ),
                    }
                )
        return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", required=True)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-noop", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    workers = _worker_names(args.workers)
    entries: list[dict[str, Any]] = []
    for worker in workers:
        database = args.run_dir / "writer" / worker / "native_memory.sqlite3"
        if not database.is_file():
            raise SystemExit(f"database is missing: {database}")
        entries.extend(
            _export_database(
                database,
                worker=worker,
                prompt_version=args.prompt_version,
                include_noop=args.include_noop,
            )
        )
    action_counts = Counter(
        str(entry["operation"].get("action") or "error")
        if isinstance(entry.get("operation"), Mapping)
        else "error"
        for entry in entries
    )
    route_counts = Counter(str(entry.get("route") or "unknown") for entry in entries)
    report = {
        "schema_version": "tmcra.v4.slow-human-review-export.2",
        "read_only": True,
        "run_dir": str(args.run_dir.resolve()),
        "prompt_version": args.prompt_version,
        "workers": workers,
        "entry_count": len(entries),
        "action_counts": dict(sorted(action_counts.items())),
        "route_counts": dict(sorted(route_counts.items())),
        "entries": entries,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: report[key] for key in ("entry_count", "action_counts", "route_counts")},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
