#!/usr/bin/env python3
"""Atomically restore selected worker databases from verified SQLite snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any


TABLES = (
    "records",
    "memory_edges",
    "slow_graph_jobs",
    "slow_graph_attempts",
    "slow_graph_patches",
    "slow_graph_patch_operations",
    "slow_graph_provenance",
    "slot_heads",
    "slot_history",
)
SLOW_CONTROL_PLANE_TABLES = (
    "slow_graph_jobs",
    "slow_graph_attempts",
    "slow_graph_patches",
    "slow_graph_patch_operations",
    "slow_graph_provenance",
    "slot_heads",
    "slot_history",
)
BACKUP_TAG_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_sha256(connection: sqlite3.Connection, table: str) -> str:
    quoted_table = '"' + table.replace('"', '""') + '"'
    columns = [
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({quoted_table})")
    ]
    digest = hashlib.sha256()
    if not columns:
        return digest.hexdigest()
    quoted_column_list = [
        '"' + column.replace('"', '""') + '"' for column in columns
    ]
    quoted_columns = ",".join(quoted_column_list)
    order_by = ",".join(quoted_column_list)
    for row in connection.execute(
        f"SELECT {quoted_columns} FROM {quoted_table} ORDER BY {order_by}"
    ):
        digest.update(
            json.dumps(
                list(row),
                ensure_ascii=False,
                separators=(",", ":"),
                default=lambda value: {"__bytes__": value.hex()},
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def inspect_database(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        table_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        counts = {
            table: int(
                connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            )
            for table in TABLES
            if table in table_names
        }
        table_sha256 = {
            table: _table_sha256(connection, table)
            for table in TABLES
            if table in table_names
        }
        control_plane_counts = {
            table: counts[table]
            for table in SLOW_CONTROL_PLANE_TABLES
            if table in counts
        }
        control_plane_sha256 = hashlib.sha256(
            json.dumps(
                {
                    table: table_sha256[table]
                    for table in SLOW_CONTROL_PLANE_TABLES
                    if table in table_sha256
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    finally:
        connection.close()
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "quick_check": quick_check,
        "counts": counts,
        "table_sha256": table_sha256,
        "slow_control_plane_counts": control_plane_counts,
        "slow_control_plane_sha256": control_plane_sha256,
    }


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_journal(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def ensure_within(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the run directory: {path}") from exc


def validate_backup_tag(value: str) -> str:
    tag = str(value).strip()
    if not BACKUP_TAG_RE.fullmatch(tag):
        raise RuntimeError(
            "backup tag must be 1-80 characters using letters, digits, dot, underscore, or hyphen"
        )
    return tag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--workers", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backup-tag",
        default="rejected_slow_graph_2026-07-13.3",
        help="unique sibling backup suffix for the current target databases",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    snapshot_dir = args.snapshot_dir.resolve()
    output = args.output.resolve()
    backup_tag = validate_backup_tag(args.backup_tag)
    workers = [item.strip() for item in args.workers.split(",") if item.strip()]
    if not run_dir.is_dir() or not snapshot_dir.is_dir():
        raise RuntimeError("run and snapshot directories must exist")
    if not workers or len(workers) != len(set(workers)):
        raise RuntimeError("workers must be a non-empty unique list")
    ensure_within(output, run_dir, "output")

    manifest = json.loads(
        (snapshot_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != "tmcra.v4.sqlite-snapshot.1"
        or int(manifest.get("physical_api_calls", -1)) != 0
    ):
        raise RuntimeError("snapshot manifest contract is invalid")
    expected = {item["worker"]: item for item in manifest.get("workers", [])}
    if set(workers) != set(expected):
        raise RuntimeError("requested workers do not exactly match snapshot manifest")

    backup_dir = run_dir.parent / (run_dir.name + "." + backup_tag)
    backup_dir.mkdir(exist_ok=True)
    journal = output.with_suffix(output.suffix + ".journal.jsonl")
    report = {
        "schema_version": "tmcra.v4.worker-database-restore.1",
        "status": "in_progress",
        "physical_api_calls": 0,
        "run_dir": str(run_dir),
        "snapshot_dir": str(snapshot_dir),
        "backup_dir": str(backup_dir),
        "backup_tag": backup_tag,
        "workers": [],
    }
    if output.exists() or journal.exists():
        raise RuntimeError("restore output or journal already exists")
    atomic_json(output, report)

    for worker in workers:
        snapshot = snapshot_dir / worker / "native_memory.sqlite3"
        target = run_dir / "writer" / worker / "native_memory.sqlite3"
        backup = backup_dir / worker / "native_memory.sqlite3"
        ensure_within(target, run_dir, "target database")
        ensure_within(backup, run_dir.parent, "backup database")
        if not snapshot.is_file() or not target.is_file():
            raise RuntimeError(f"snapshot or target is missing for {worker}")
        snapshot_state = inspect_database(snapshot)
        expected_item = expected[worker]
        expected_counts = expected_item.get("counts")
        expected_table_sha256 = expected_item.get("table_sha256")
        expected_control_plane_counts = expected_item.get("slow_control_plane_counts")
        expected_control_plane_sha256 = expected_item.get("slow_control_plane_sha256")
        if (
            snapshot_state["quick_check"] != "ok"
            or snapshot_state["sha256"] != expected_item["snapshot_sha256"]
            or snapshot_state["bytes"] != int(expected_item["snapshot_bytes"])
            or snapshot_state["counts"] != expected_counts
            or (
                expected_table_sha256 is not None
                and snapshot_state["table_sha256"] != expected_table_sha256
            )
            or (
                expected_control_plane_counts is not None
                and snapshot_state["slow_control_plane_counts"]
                != expected_control_plane_counts
            )
            or (
                expected_control_plane_sha256 is not None
                and snapshot_state["slow_control_plane_sha256"]
                != expected_control_plane_sha256
            )
        ):
            raise RuntimeError(f"snapshot verification failed for {worker}")
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(target) + suffix)
            if suffix == "-wal" and sidecar.exists() and sidecar.stat().st_size:
                raise RuntimeError(f"non-empty WAL blocks restore for {worker}")
        before = inspect_database(target)
        if before["quick_check"] != "ok":
            raise RuntimeError(f"target quick_check failed for {worker}")
        if backup.exists():
            raise RuntimeError(f"rejected-version backup already exists for {worker}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        os.link(target, backup)

        temporary = target.with_name(target.name + ".restore." + uuid.uuid4().hex)
        try:
            shutil.copy2(snapshot, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary_state = inspect_database(temporary)
            if temporary_state != snapshot_state:
                raise RuntimeError(f"copied snapshot differs for {worker}")
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(target) + suffix)
                if sidecar.exists():
                    sidecar.unlink()
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        after = inspect_database(target)
        if after != snapshot_state:
            raise RuntimeError(f"restored target differs for {worker}")
        entry = {
            "worker": worker,
            "target": str(target),
            "backup": str(backup),
            "before": before,
            "after": after,
        }
        report["workers"].append(entry)
        append_journal(journal, entry)
        atomic_json(output, report)

    report["status"] = "complete"
    report["restored_count"] = len(report["workers"])
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
