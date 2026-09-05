#!/usr/bin/env python3
"""Create a Fast/Source-identical database copy with no prior Slow state."""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SLOW_TABLES = (
    "slow_graph_attempts",
    "slow_graph_batches",
    "slow_graph_patch_operations",
    "slow_graph_provenance",
    "slow_graph_patches",
    "slow_graph_jobs",
    "slow_graph_zero_call_recoveries",
    "slow_graph_zero_call_promotion_recoveries",
    "slow_graph_zero_call_projection_recoveries",
)
RUNTIME_AUDIT_TABLES = (
    "audit_answer_support",
    "audit_retrieval_log",
)


class FreshSlowCopyError(RuntimeError):
    pass


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: item.hex() if isinstance(item, bytes) else str(item),
    )


def _tables(con: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def _fingerprint(rows: Iterable[Sequence[Any]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        digest.update(_json(list(row)).encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def _record_inventory(
    con: sqlite3.Connection,
) -> tuple[set[tuple[str, str]], tuple[int, str]]:
    rows = con.execute(
        "SELECT scope_id,memory_id,category,slot_key,value,relation,"
        "anchor_concepts_json,evidence_anchors_json,salience,confidence,"
        "source_kind,turn_index,state,supersedes_json,metadata_json "
        "FROM records ORDER BY scope_id,memory_id"
    ).fetchall()
    slow_ids: set[tuple[str, str]] = set()
    preserved: list[Sequence[Any]] = []
    for row in rows:
        try:
            metadata = json.loads(str(row[-1]))
        except json.JSONDecodeError as exc:
            raise FreshSlowCopyError(
                f"record {row[0]}:{row[1]} has invalid metadata_json"
            ) from exc
        if not isinstance(metadata, Mapping):
            raise FreshSlowCopyError(
                f"record {row[0]}:{row[1]} metadata is not an object"
            )
        if (
            metadata.get("content_variant") == "slow_memory_capsule"
            or metadata.get("memory_layer") == "slow"
        ):
            slow_ids.add((str(row[0]), str(row[1])))
        else:
            preserved.append(row)
    return slow_ids, _fingerprint(preserved)


def _preserved_edge_inventory(
    con: sqlite3.Connection, slow_ids: set[tuple[str, str]]
) -> tuple[int, str]:
    rows = con.execute(
        "SELECT scope_id,edge_id,source_memory_id,target_memory_id,edge_type,"
        "score,model_score,evidence_turn,evidence,metadata_json "
        "FROM memory_edges ORDER BY scope_id,edge_id"
    ).fetchall()
    preserved = [
        row
        for row in rows
        if (str(row[0]), str(row[2])) not in slow_ids
        and (str(row[0]), str(row[3])) not in slow_ids
    ]
    return _fingerprint(preserved)


def _preserved_slot_inventory(
    con: sqlite3.Connection,
    table: str,
    slow_ids: set[tuple[str, str]],
) -> tuple[int, str]:
    if table not in _tables(con):
        return (0, hashlib.sha256().hexdigest())
    columns = (
        "scope_id,slot_key,memory_id"
        if table == "slot_heads"
        else "scope_id,slot_key,ordinal,memory_id"
    )
    order_by = (
        "scope_id,slot_key,memory_id"
        if table == "slot_heads"
        else "scope_id,slot_key,ordinal,memory_id"
    )
    rows = con.execute(
        f'SELECT {columns} FROM "{table}" ORDER BY {order_by}'
    ).fetchall()
    preserved = [
        row
        for row in rows
        if not str(row[1]).startswith("slow.")
        and (str(row[0]), str(row[-1])) not in slow_ids
    ]
    return _fingerprint(preserved)


def prepare_copy(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise FreshSlowCopyError("source and output databases must differ")
    if not source.is_file():
        raise FreshSlowCopyError(f"source database does not exist: {source}")
    if output.exists():
        raise FreshSlowCopyError(f"output database already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    source_uri = f"file:{source.as_posix()}?mode=ro"
    with closing(sqlite3.connect(source_uri, uri=True)) as src, closing(
        sqlite3.connect(output)
    ) as dst:
        src.row_factory = sqlite3.Row
        if src.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise FreshSlowCopyError("source database quick_check failed")
        src.backup(dst)

    with closing(sqlite3.connect(output)) as con:
        con.row_factory = sqlite3.Row
        tables = _tables(con)
        required = {"records", "memory_edges"}
        if not required <= tables:
            raise FreshSlowCopyError(
                "database lacks required tables: " + ",".join(sorted(required - tables))
            )
        unexpected = sorted(
            table
            for table in tables
            if table.startswith("slow_graph_") and table not in SLOW_TABLES
        )
        if unexpected:
            raise FreshSlowCopyError(
                "unknown slow-graph tables require an explicit migration: "
                + ",".join(unexpected)
            )

        slow_ids, preserved_records_before = _record_inventory(con)
        preserved_edges_before = _preserved_edge_inventory(con, slow_ids)
        preserved_heads_before = _preserved_slot_inventory(
            con, "slot_heads", slow_ids
        )
        preserved_history_before = _preserved_slot_inventory(
            con, "slot_history", slow_ids
        )
        cleared_tables: dict[str, int] = {}
        con.execute("BEGIN IMMEDIATE")
        try:
            removed_edges = 0
            for scope_id, memory_id in sorted(slow_ids):
                cursor = con.execute(
                    "DELETE FROM memory_edges WHERE scope_id=? AND "
                    "(source_memory_id=? OR target_memory_id=?)",
                    (scope_id, memory_id, memory_id),
                )
                removed_edges += max(0, int(cursor.rowcount))
            con.executemany(
                "DELETE FROM records WHERE scope_id=? AND memory_id=?",
                sorted(slow_ids),
            )
            removed_slow_slot_heads = 0
            removed_slow_slot_history_rows = 0
            for table, count_name in (
                ("slot_heads", "heads"),
                ("slot_history", "history"),
            ):
                if table not in tables:
                    continue
                unsafe = con.execute(
                    f'SELECT s.scope_id,s.slot_key,s.memory_id FROM "{table}" s '
                    "JOIN records r ON r.scope_id=s.scope_id AND r.memory_id=s.memory_id "
                    "WHERE s.slot_key LIKE ? AND NOT ("
                    "json_extract(r.metadata_json,'$.content_variant')='slow_memory_capsule' "
                    "OR json_extract(r.metadata_json,'$.memory_layer')='slow') LIMIT 1",
                    ("slow.%",),
                ).fetchone()
                if unsafe is not None:
                    raise FreshSlowCopyError(
                        f"{table} uses a reserved Slow slot for a non-Slow record: "
                        f"{unsafe[0]}:{unsafe[1]}:{unsafe[2]}"
                    )
                removed = 0
                for scope_id, memory_id in sorted(slow_ids):
                    cursor = con.execute(
                        f'DELETE FROM "{table}" WHERE scope_id=? AND memory_id=?',
                        (scope_id, memory_id),
                    )
                    removed += max(0, int(cursor.rowcount))
                cursor = con.execute(
                    f'DELETE FROM "{table}" WHERE slot_key LIKE ?',
                    ("slow.%",),
                )
                removed += max(0, int(cursor.rowcount))
                if count_name == "heads":
                    removed_slow_slot_heads = removed
                else:
                    removed_slow_slot_history_rows = removed
            for table in (*SLOW_TABLES, *RUNTIME_AUDIT_TABLES):
                if table not in tables:
                    continue
                before = int(con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
                con.execute(f'DELETE FROM "{table}"')
                cleared_tables[table] = before
            con.commit()
        except Exception:
            con.rollback()
            raise

        remaining_slow_ids, preserved_records_after = _record_inventory(con)
        preserved_edges_after = _preserved_edge_inventory(con, set())
        preserved_heads_after = _preserved_slot_inventory(con, "slot_heads", set())
        preserved_history_after = _preserved_slot_inventory(
            con, "slot_history", set()
        )
        quick_check = str(con.execute("PRAGMA quick_check").fetchone()[0])
        if remaining_slow_ids:
            raise FreshSlowCopyError("Slow records remain after reset")
        if preserved_records_after != preserved_records_before:
            raise FreshSlowCopyError("non-Slow record fingerprint changed")
        if preserved_edges_after != preserved_edges_before:
            raise FreshSlowCopyError("non-Slow edge fingerprint changed")
        if preserved_heads_after != preserved_heads_before:
            raise FreshSlowCopyError("non-Slow slot-head fingerprint changed")
        if preserved_history_after != preserved_history_before:
            raise FreshSlowCopyError("non-Slow slot-history fingerprint changed")
        for table in ("slot_heads", "slot_history"):
            if table in tables and con.execute(
                f'SELECT 1 FROM "{table}" WHERE slot_key LIKE ? LIMIT 1',
                ("slow.%",),
            ).fetchone() is not None:
                raise FreshSlowCopyError(f"Slow {table} rows remain after reset")
        if quick_check != "ok":
            raise FreshSlowCopyError("output database quick_check failed")

    return {
        "schema_version": "tmcra.v4.fresh-slow-copy.2",
        "status": "complete",
        "source_db": str(source),
        "output_db": str(output),
        "physical_api_calls": 0,
        "removed_slow_records": len(slow_ids),
        "removed_slow_edges": removed_edges,
        "removed_slow_slot_heads": removed_slow_slot_heads,
        "removed_slow_slot_history_rows": removed_slow_slot_history_rows,
        "cleared_table_rows": dict(sorted(cleared_tables.items())),
        "preserved_record_count": preserved_records_after[0],
        "preserved_record_sha256": preserved_records_after[1],
        "preserved_edge_count": preserved_edges_after[0],
        "preserved_edge_sha256": preserved_edges_after[1],
        "quick_check": quick_check,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = prepare_copy(args.source_db, args.output_db)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(_json(report) + "\n", encoding="utf-8")
    print(_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
