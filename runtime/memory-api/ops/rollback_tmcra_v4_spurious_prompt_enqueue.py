#!/usr/bin/env python3
"""Remove a proven zero-call prompt-version enqueue from frozen V4 stores."""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_tmcra_v4_build import BuildError, _load_resume_manifest
from tmcra_v4_cost_report import SLOW_INTERRUPTION_ERROR


SCHEMA_VERSION = "tmcra.v4.spurious-prompt-enqueue-rollback.1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _prompt_version(metadata_json: str) -> str:
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise BuildError("Slow job metadata is not JSON") from exc
    if not isinstance(metadata, Mapping):
        raise BuildError("Slow job metadata is not an object")
    model_config = metadata.get("model_config")
    if not isinstance(model_config, Mapping):
        return ""
    return str(model_config.get("prompt_version") or "")


def _json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BuildError(f"{label} is not JSON") from exc
    if not isinstance(value, Mapping):
        raise BuildError(f"{label} is not an object")
    return dict(value)


def _inspect_database(database: Path, prompt_version: str) -> dict[str, Any]:
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as con:
        con.row_factory = sqlite3.Row
        jobs = [
            dict(row)
            for row in con.execute("SELECT * FROM slow_graph_jobs ORDER BY job_id")
            if _prompt_version(str(row["metadata_json"])) == prompt_version
        ]
        target_jobs = {str(row["job_id"]) for row in jobs}
        empty = {
            "database": str(database),
            "prompt_version": prompt_version,
            "target_job_count": 0,
            "target_attempt_count": 0,
            "target_patch_count": 0,
            "target_operation_count": 0,
            "target_batch_count": 0,
            "target_job_ids": [],
            "target_patch_ids": [],
            "target_batch_ids": [],
            "unknown_attempt_ids": [],
            "status_counts": {},
        }
        if not target_jobs:
            return empty
        if any(
            row["claim_token"] is not None
            or row["claim_owner"] is not None
            or row["lease_expires_at"] is not None
            for row in jobs
        ):
            raise BuildError(f"target Slow jobs are still claimed: {database}")

        attempts = [
            dict(row)
            for row in con.execute("SELECT * FROM slow_graph_attempts")
            if str(row["job_id"]) in target_jobs
        ]
        unknown_attempt_ids: list[str] = []
        for attempt in attempts:
            metadata = _json_object(
                str(attempt["call_metadata_json"] or "{}"),
                "target Slow attempt metadata",
            )
            interrupted_unknown = (
                str(attempt["status"]) == "expired"
                and str(attempt["error"] or "") == SLOW_INTERRUPTION_ERROR
                and metadata == {}
            )
            if interrupted_unknown:
                unknown_attempt_ids.append(str(attempt["attempt_id"]))
            elif (
                metadata.get("physical_api_call") is not False
                or int(metadata.get("physical_api_calls", -1)) != 0
            ):
                raise BuildError(
                    f"target prompt enqueue contains a physical API call: {database}"
                )

        patches = [
            dict(row)
            for row in con.execute("SELECT * FROM slow_graph_patches")
            if str(row["job_id"]) in target_jobs
        ]
        target_patches = {str(row["patch_id"]) for row in patches}
        for patch in patches:
            payload = _json_object(str(patch["patch_json"]), "target Slow patch")
            operations = payload.get("operations")
            if (
                not isinstance(operations, list)
                or len(operations) != 1
                or not isinstance(operations[0], Mapping)
                or operations[0].get("action") != "noop"
            ):
                raise BuildError(
                    f"target prompt enqueue contains a non-noop patch: {database}"
                )

        operations = [
            dict(row)
            for row in con.execute("SELECT * FROM slow_graph_patch_operations")
            if str(row["patch_id"]) in target_patches
        ]
        if len(operations) != len(patches) or any(
            str(row["action"]) != "noop" for row in operations
        ):
            raise BuildError(f"target patch-operation audit is not noop-only: {database}")
        provenance_count = sum(
            1
            for row in con.execute("SELECT patch_id FROM slow_graph_provenance")
            if str(row["patch_id"]) in target_patches
        )
        if provenance_count:
            raise BuildError(f"target prompt enqueue created provenance: {database}")

        target_batches: list[str] = []
        for row in con.execute("SELECT batch_id,job_ids_json FROM slow_graph_batches"):
            try:
                job_ids = json.loads(str(row["job_ids_json"]))
            except json.JSONDecodeError as exc:
                raise BuildError("Slow batch job IDs are not JSON") from exc
            if not isinstance(job_ids, list):
                raise BuildError("Slow batch job IDs are not a list")
            batch_jobs = {str(item) for item in job_ids}
            overlap = target_jobs.intersection(batch_jobs)
            if overlap:
                if overlap != batch_jobs:
                    raise BuildError(
                        f"target jobs share a batch with retained jobs: {database}"
                    )
                target_batches.append(str(row["batch_id"]))
        status_counts = {
            status: sum(str(row["status"]) == status for row in jobs)
            for status in {str(row["status"]) for row in jobs}
        }
        return {
            "database": str(database),
            "prompt_version": prompt_version,
            "target_job_count": len(target_jobs),
            "target_attempt_count": len(attempts),
            "target_patch_count": len(patches),
            "target_operation_count": len(operations),
            "target_batch_count": len(target_batches),
            "status_counts": dict(sorted(status_counts.items())),
            "target_job_ids": sorted(target_jobs),
            "target_patch_ids": sorted(target_patches),
            "target_batch_ids": sorted(target_batches),
            "unknown_attempt_ids": sorted(unknown_attempt_ids),
        }


def _backup_database(database: Path, backup: Path) -> None:
    backup.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as source, closing(
        sqlite3.connect(backup)
    ) as target:
        source.backup(target)


def _rollback_database(database: Path, inspection: Mapping[str, Any]) -> None:
    if int(inspection["target_job_count"]) == 0:
        return
    with closing(sqlite3.connect(database)) as con:
        con.execute("BEGIN IMMEDIATE")
        con.execute("CREATE TEMP TABLE rollback_jobs(job_id TEXT PRIMARY KEY)")
        con.execute("CREATE TEMP TABLE rollback_patches(patch_id TEXT PRIMARY KEY)")
        con.execute("CREATE TEMP TABLE rollback_batches(batch_id TEXT PRIMARY KEY)")
        con.executemany(
            "INSERT INTO rollback_jobs VALUES(?)",
            ((item,) for item in inspection["target_job_ids"]),
        )
        con.executemany(
            "INSERT INTO rollback_patches VALUES(?)",
            ((item,) for item in inspection["target_patch_ids"]),
        )
        con.executemany(
            "INSERT INTO rollback_batches VALUES(?)",
            ((item,) for item in inspection["target_batch_ids"]),
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS slow_graph_archived_attempts("
            "attempt_id TEXT PRIMARY KEY,original_job_id TEXT NOT NULL,scope_id TEXT NOT NULL,"
            "status TEXT NOT NULL,call_metadata_json TEXT NOT NULL,error TEXT NOT NULL,"
            "created_at INTEGER,completed_at INTEGER,claim_token TEXT,claim_owner TEXT,"
            "prompt_version TEXT NOT NULL,archive_reason TEXT NOT NULL,archived_at TEXT NOT NULL)"
        )
        for attempt_id in inspection["unknown_attempt_ids"]:
            row = con.execute(
                "SELECT * FROM slow_graph_attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise BuildError(f"unknown target attempt disappeared: {attempt_id}")
            columns = [item[1] for item in con.execute("PRAGMA table_info(slow_graph_attempts)")]
            attempt = dict(zip(columns, row))
            con.execute(
                "INSERT INTO slow_graph_archived_attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt_id,
                    attempt["job_id"],
                    attempt["scope_id"],
                    attempt["status"],
                    attempt["call_metadata_json"],
                    attempt["error"],
                    attempt["created_at"],
                    attempt["completed_at"],
                    attempt["claim_token"],
                    attempt["claim_owner"],
                    inspection["prompt_version"],
                    "spurious_prompt_enqueue_rollback",
                    _now(),
                ),
            )
        con.execute(
            "DELETE FROM slow_graph_provenance WHERE patch_id IN "
            "(SELECT patch_id FROM rollback_patches)"
        )
        con.execute(
            "DELETE FROM slow_graph_patch_operations WHERE patch_id IN "
            "(SELECT patch_id FROM rollback_patches)"
        )
        con.execute(
            "DELETE FROM slow_graph_patches WHERE patch_id IN "
            "(SELECT patch_id FROM rollback_patches)"
        )
        con.execute(
            "DELETE FROM slow_graph_attempts WHERE job_id IN "
            "(SELECT job_id FROM rollback_jobs)"
        )
        con.execute(
            "DELETE FROM slow_graph_batches WHERE batch_id IN "
            "(SELECT batch_id FROM rollback_batches)"
        )
        con.execute(
            "DELETE FROM slow_graph_jobs WHERE job_id IN "
            "(SELECT job_id FROM rollback_jobs)"
        )
        con.commit()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archived_unknown_attempt_ids(database: Path) -> set[str]:
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as con:
        table = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='slow_graph_archived_attempts'"
        ).fetchone()
        if table is None:
            return set()
        return {
            str(row[0])
            for row in con.execute(
                "SELECT attempt_id FROM slow_graph_archived_attempts "
                "WHERE archive_reason='spurious_prompt_enqueue_rollback'"
            )
        }


def rollback(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    backup_dir = args.backup_dir.resolve()
    if output.exists():
        raise BuildError("rollback output already exists")
    if backup_dir.exists() and not args.resume:
        raise BuildError("rollback backup directory already exists without --resume")
    if (run_dir / "SLOW_REPAIR_LOCK").exists():
        raise BuildError("Slow repair lock is active")
    manifest = _load_resume_manifest(run_dir)
    databases = [
        (
            Path(str(worker["worker_dir"])).name,
            Path(str(worker["worker_dir"])) / "native_memory.sqlite3",
        )
        for worker in manifest["workers"]
    ]
    plans: list[dict[str, Any]] = []
    for worker, database in databases:
        backup = backup_dir / worker / "native_memory.sqlite3"
        current = _inspect_database(database, args.prompt_version)
        original = (
            _inspect_database(backup, args.prompt_version)
            if backup.is_file()
            else current
        )
        current_ids = set(current["target_job_ids"])
        original_ids = set(original["target_job_ids"])
        if current_ids and current_ids != original_ids:
            raise BuildError(f"partially rolled back target jobs in {database}")
        if not current_ids:
            missing_archived = set(original["unknown_attempt_ids"]) - (
                _archived_unknown_attempt_ids(database)
            )
            if missing_archived:
                raise BuildError(
                    f"rolled-back database lost archived unknown attempts: {database}"
                )
        plans.append(
            {
                "worker": worker,
                "database": database,
                "backup": backup,
                "current": current,
                "original": original,
            }
        )
    if not any(int(item["original"]["target_job_count"]) for item in plans):
        raise BuildError("no target prompt-version jobs were found")
    backup_dir.mkdir(parents=True, exist_ok=args.resume)
    rows: list[dict[str, Any]] = []
    for plan in plans:
        worker = plan["worker"]
        database = plan["database"]
        backup = plan["backup"]
        inspection = plan["original"]
        if not backup.exists():
            _backup_database(database, backup)
        before_sha256 = _sha256(backup)
        if int(plan["current"]["target_job_count"]):
            _rollback_database(database, plan["current"])
        after = _inspect_database(database, args.prompt_version)
        if int(after["target_job_count"]) != 0:
            raise BuildError(f"rollback left target jobs in {database}")
        rows.append(
            {
                "worker": worker,
                "database": str(database),
                "backup": str(backup),
                "backup_sha256": before_sha256,
                "removed_job_count": inspection["target_job_count"],
                "removed_attempt_count": inspection["target_attempt_count"],
                "removed_patch_count": inspection["target_patch_count"],
                "removed_operation_count": inspection["target_operation_count"],
                "removed_batch_count": inspection["target_batch_count"],
                "archived_unknown_attempt_count": len(
                    inspection["unknown_attempt_ids"]
                ),
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "created_at": _now(),
        "run_dir": str(run_dir),
        "prompt_version": args.prompt_version,
        "backup_dir": str(backup_dir),
        "resumed": bool(args.resume),
        "worker_count": len(rows),
        "removed_job_count": sum(row["removed_job_count"] for row in rows),
        "removed_attempt_count": sum(row["removed_attempt_count"] for row in rows),
        "removed_patch_count": sum(row["removed_patch_count"] for row in rows),
        "removed_operation_count": sum(row["removed_operation_count"] for row in rows),
        "removed_batch_count": sum(row["removed_batch_count"] for row in rows),
        "archived_unknown_attempt_count": sum(
            row["archived_unknown_attempt_count"] for row in rows
        ),
        "workers": rows,
    }
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = rollback(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
