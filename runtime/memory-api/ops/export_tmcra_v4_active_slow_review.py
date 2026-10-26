#!/usr/bin/env python3
"""Export the Slow capsule heads that production indexing can actually see."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping


def _workers(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("workers must be a non-empty unique comma-separated list")
    return values


def _overrides(raw: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for item in raw:
        worker, separator, path = item.partition("=")
        if not separator or not worker or not path or worker in output:
            raise ValueError("worker DB overrides must use unique WORKER=PATH values")
        output[worker] = Path(path).resolve()
    return output


def _metadata(raw: Any, *, memory_id: str) -> dict[str, Any]:
    try:
        value = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{memory_id}: metadata_json is invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{memory_id}: metadata_json is not an object")
    return dict(value)


def _export_database(database: Path, worker: str) -> list[dict[str, Any]]:
    if not database.is_file():
        raise ValueError(f"database is missing: {database}")
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as con:
        rows = list(
            con.execute(
                "SELECT memory_id,value,state,metadata_json FROM records "
                "ORDER BY memory_id"
            )
        )
    revisions: dict[str, list[tuple[str, str, str, dict[str, Any]]]] = {}
    for memory_id, value, state, raw_metadata in rows:
        metadata = _metadata(raw_metadata, memory_id=str(memory_id))
        if (
            str(metadata.get("memory_layer") or "") != "slow"
            or str(metadata.get("content_variant") or "")
            != "slow_memory_capsule"
        ):
            continue
        capsule_id = str(metadata.get("capsule_id") or "")
        revision = metadata.get("revision")
        if not capsule_id or not isinstance(revision, int) or revision < 1:
            raise ValueError(f"{memory_id}: invalid Slow capsule identity")
        revisions.setdefault(capsule_id, []).append(
            (str(memory_id), str(value), str(state), metadata)
        )

    output: list[dict[str, Any]] = []
    for capsule_id, candidates in sorted(revisions.items()):
        latest_revision = max(int(item[3]["revision"]) for item in candidates)
        latest = [
            item for item in candidates if int(item[3]["revision"]) == latest_revision
        ]
        if len(latest) != 1:
            raise ValueError(
                f"{worker}: capsule {capsule_id} lacks a unique latest revision"
            )
        memory_id, value, state, metadata = latest[0]
        if state != "active" or str(metadata.get("status") or "") not in {
            "active",
            "challenged",
        }:
            continue
        claims = metadata.get("claims")
        if not isinstance(claims, list) or not claims:
            raise ValueError(f"{memory_id}: current Slow capsule has no claims")
        output.append(
            {
                "worker": worker,
                "region_key": str(metadata.get("region_key") or ""),
                "capsule_id": capsule_id,
                "memory_id": memory_id,
                "resulting_capsule": {
                    "memory_id": memory_id,
                    "state": state,
                    "value": value,
                    "metadata": metadata,
                },
            }
        )
    return output


def export_active(
    run_dir: Path,
    workers: list[str],
    *,
    worker_databases: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    overrides = dict(worker_databases or {})
    unknown = sorted(set(overrides) - set(workers))
    if unknown:
        raise ValueError(f"worker DB overrides are outside the selection: {unknown}")
    entries: list[dict[str, Any]] = []
    for worker in workers:
        database = overrides.get(
            worker, run_dir / "writer" / worker / "native_memory.sqlite3"
        )
        entries.extend(_export_database(database, worker))
    return {
        "schema_version": "tmcra.v4.active-slow-review-export.1",
        "read_only": True,
        "run_dir": str(run_dir.resolve()),
        "prompt_version": "current-production-slow-heads",
        "workers": workers,
        "entry_count": len(entries),
        "state_counts": dict(
            sorted(
                Counter(
                    str(entry["resulting_capsule"]["state"]) for entry in entries
                ).items()
            )
        ),
        "worker_db_overrides": {
            worker: str(path) for worker, path in sorted(overrides.items())
        },
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", required=True)
    parser.add_argument("--worker-db", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    try:
        report = export_active(
            args.run_dir.resolve(),
            _workers(args.workers),
            worker_databases=_overrides(args.worker_db),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "entries"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
