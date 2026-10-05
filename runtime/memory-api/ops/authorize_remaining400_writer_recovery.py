from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _artifact_count(path: Path, call_key: str) -> int:
    if not path.is_file():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        count += int(str(value.get("call_key") or "") == call_key)
    return count


def _action_id(worker_index: int, kind: str, identity: str) -> str:
    return hashlib.sha256(f"{worker_index}:{kind}:{identity}".encode()).hexdigest()[:32]


def _inspect_worker(worker: Path, index: int) -> list[dict[str, Any]]:
    database = worker / "native_memory.sqlite3"
    if not database.is_file():
        raise RuntimeError(f"worker {index}: database is missing")
    actions: list[dict[str, Any]] = []
    writer_model = str(
        os.getenv("TMCRA_WRITER_MODEL")
        or os.getenv("TMCRA_DEEPSEEK_FLASH_MODEL")
        or "deepseek-v4-flash"
    ).strip()
    reviewer_model = str(
        os.getenv("TMCRA_WRITER_REVIEWER_MODEL")
        or os.getenv("TMCRA_DEEPSEEK_PRO_MODEL")
        or "deepseek-v4-pro"
    ).strip()
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            raise RuntimeError(f"worker {index}: SQLite quick_check failed: {quick_check}")

        for row in connection.execute(
            "SELECT * FROM v4_batch_journal WHERE status!='committed' ORDER BY batch_index"
        ):
            batch = dict(row)
            if batch["status"] != "failed" or str(batch.get("response_json") or ""):
                continue
            metadata = json.loads(str(batch.get("response_metadata_json") or "{}"))
            error = str(batch.get("error") or "")
            if not error.startswith("IncompleteRead:") or metadata:
                continue
            call_key = f"flash:{batch['batch_id']}"
            call_count = _artifact_count(worker / "product_writer_calls.jsonl", call_key)
            raw_count = _artifact_count(
                worker / "product_writer_raw_responses.jsonl", call_key
            )
            if call_count or raw_count:
                raise RuntimeError(
                    f"worker {index}: {call_key} transport recovery has unexpected artifacts"
                )
            actions.append(
                {
                    "action_id": _action_id(index, "flash_transport", batch["batch_id"]),
                    "worker_index": index,
                    "worker_dir": str(worker),
                    "kind": "flash_transport_to_prepared",
                    "identity": str(batch["batch_id"]),
                    "prior_status": "failed",
                    "prior_error": error,
                    "call_artifact_count": call_count,
                    "raw_response_artifact_count": raw_count,
                    "replacement_model": writer_model,
                    "replacement_authorized": True,
                }
            )

        for row in connection.execute(
            "SELECT * FROM v4_reconciliation_jobs WHERE status!='completed' ORDER BY rowid"
        ):
            job = dict(row)
            if job["status"] != "failed" or str(job.get("response_json") or ""):
                continue
            metadata = json.loads(str(job.get("response_metadata_json") or "{}"))
            error = str(job.get("error") or "")
            http_retry = (
                str(metadata.get("status") or "") == "http_error"
                and int(metadata.get("http_status") or 0) >= 500
                and metadata.get("physical_api_call") is True
            )
            incomplete_retry = error.startswith("IncompleteRead:") and not metadata
            if not (http_retry or incomplete_retry):
                continue
            call_key = f"pro:{job['job_id']}"
            call_count = _artifact_count(worker / "product_writer_calls.jsonl", call_key)
            raw_count = _artifact_count(
                worker / "product_writer_raw_responses.jsonl", call_key
            )
            expected_calls = 1 if http_retry else 0
            if call_count != expected_calls or raw_count != 0:
                raise RuntimeError(
                    f"worker {index}: {call_key} transport recovery artifact mismatch: "
                    f"calls={call_count}, raw={raw_count}, expected_calls={expected_calls}"
                )
            actions.append(
                {
                    "action_id": _action_id(index, "pro_transport", job["job_id"]),
                    "worker_index": index,
                    "worker_dir": str(worker),
                    "kind": "pro_transport_to_pending",
                    "identity": str(job["job_id"]),
                    "batch_id": str(job["batch_id"]),
                    "prior_status": "failed",
                    "prior_error": error,
                    "prior_response_metadata": metadata,
                    "call_artifact_count": call_count,
                    "raw_response_artifact_count": raw_count,
                    "replacement_model": reviewer_model,
                    "replacement_authorized": True,
                }
            )
    return actions


def _apply_action(action: dict[str, Any]) -> None:
    database = Path(action["worker_dir"]) / "native_memory.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN IMMEDIATE")
        if action["kind"] == "flash_transport_to_prepared":
            row = connection.execute(
                "SELECT status,error,response_json,response_metadata_json FROM v4_batch_journal "
                "WHERE batch_id=?",
                (action["identity"],),
            ).fetchone()
            if (
                row is None
                or row["status"] != "failed"
                or str(row["error"] or "") != action["prior_error"]
                or str(row["response_json"] or "")
                or json.loads(str(row["response_metadata_json"] or "{}"))
            ):
                raise RuntimeError(
                    f"{action['action_id']}: Flash journal changed after review"
                )
            updated = connection.execute(
                "UPDATE v4_batch_journal SET status='prepared',api_started_at='',error='',updated_at=? "
                "WHERE batch_id=? AND status='failed' AND response_json=''",
                (_now(), action["identity"]),
            ).rowcount
        else:
            row = connection.execute(
                "SELECT status,error,response_json,response_metadata_json FROM v4_reconciliation_jobs "
                "WHERE job_id=?",
                (action["identity"],),
            ).fetchone()
            if (
                row is None
                or row["status"] != "failed"
                or str(row["error"] or "") != action["prior_error"]
                or str(row["response_json"] or "")
                or json.loads(str(row["response_metadata_json"] or "{}"))
                != action["prior_response_metadata"]
            ):
                raise RuntimeError(
                    f"{action['action_id']}: Pro journal changed after review"
                )
            updated = connection.execute(
                "UPDATE v4_reconciliation_jobs SET status='pro_pending',error='',updated_at=? "
                "WHERE job_id=? AND status='failed' AND response_json=''",
                (_now(), action["identity"]),
            ).rowcount
        if updated != 1:
            raise RuntimeError(f"{action['action_id']}: journal transition was not atomic")
        connection.commit()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    writer_root = args.run_dir / "writer"
    actions: list[dict[str, Any]] = []
    for worker in sorted(
        writer_root.glob("worker_*"),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    ):
        if (worker / "product_writer_report.json").is_file():
            continue
        index = int(worker.name.rsplit("_", 1)[-1])
        actions.extend(_inspect_worker(worker, index))

    if args.apply:
        for action in actions:
            _apply_action(action)

    completed_at = _now()
    report = {
        "schema_version": "tmcra.v4.writer-recovery-authorization.1",
        "run_dir": str(args.run_dir),
        "mode": "apply" if args.apply else "dry_run",
        "status": "complete",
        "action_count": len(actions),
        "actions": [
            {
                **action,
                "authorized_at": completed_at if args.apply else "",
            }
            for action in actions
        ],
        "completed_at": completed_at,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "action_count": len(actions),
                "worker_indices": sorted({item["worker_index"] for item in actions}),
                "kinds": sorted({item["kind"] for item in actions}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
