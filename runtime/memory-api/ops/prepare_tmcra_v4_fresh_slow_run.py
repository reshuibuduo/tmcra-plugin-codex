#!/usr/bin/env python3
"""Prepare an isolated Writer/Fast-identical run for a fresh Slow rebuild."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ops.prepare_tmcra_v4_fresh_slow_copy import prepare_copy


SCHEMA_VERSION = "tmcra.v4.fresh-slow-run.1"
MARKER_NAME = "FRESH_SLOW_COPY_COMPLETE.json"
REQUIRED_WORKER_ARTIFACTS = (
    "input.json",
    "product_write_messages.jsonl",
    "product_writer_calls.jsonl",
    "product_writer_raw_responses.jsonl",
    "product_writer_report.json",
    "source_exclusions.json",
    "writer_chain_audit.json",
    "writer.log",
)


class FreshSlowRunError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_json(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshSlowRunError(f"cannot read JSON artifact {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise FreshSlowRunError(f"{path}:{line_number} is not an object")
            rows.append(dict(value))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshSlowRunError(f"cannot read JSONL artifact {path}: {exc}") from exc
    return rows


def _worker_names(raw_values: Sequence[str]) -> list[str]:
    names: list[str] = []
    for raw in raw_values:
        names.extend(item.strip() for item in raw.split(",") if item.strip())
    if not names:
        raise FreshSlowRunError("at least one worker must be selected")
    if len(names) != len(set(names)):
        raise FreshSlowRunError("selected workers must be unique")
    for name in names:
        if not name.startswith("worker_") or not name.removeprefix("worker_").isdigit():
            raise FreshSlowRunError(f"invalid worker name: {name}")
    return names


def _selected_rows(
    rows: Sequence[Mapping[str, Any]], qids: Sequence[str], *, artifact: str
) -> list[dict[str, Any]]:
    by_qid: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("question_id") or "").strip()
        if not qid or qid in by_qid:
            raise FreshSlowRunError(f"{artifact} has a missing or duplicate question_id")
        by_qid[qid] = dict(row)
    missing = [qid for qid in qids if qid not in by_qid]
    if missing:
        raise FreshSlowRunError(f"{artifact} lacks selected qids: {','.join(missing)}")
    return [by_qid[qid] for qid in qids]


def _rewrite_manifest_row(
    row: Mapping[str, Any], output_run: Path, worker_name: str
) -> dict[str, Any]:
    result = dict(row)
    qid = str(result["question_id"])
    result["db_path"] = str(output_run / "writer" / worker_name / "native_memory.sqlite3")
    result["index_path"] = str(output_run / "indexes" / f"{qid}.pt")
    return result


def prepare_run(source_run: Path, output_run: Path, names: Sequence[str]) -> dict[str, Any]:
    source_run = source_run.resolve()
    output_run = output_run.resolve()
    if not source_run.is_dir():
        raise FreshSlowRunError(f"source run does not exist: {source_run}")
    if output_run.exists():
        raise FreshSlowRunError(f"output run already exists: {output_run}")

    manifest = _load_json(source_run / "input_manifest.json")
    if not isinstance(manifest, Mapping) or manifest.get("status") != "prepared":
        raise FreshSlowRunError("source input manifest is incomplete")
    workers = manifest.get("workers")
    if not isinstance(workers, list):
        raise FreshSlowRunError("source input manifest has no workers")
    by_name = {
        Path(str(worker.get("worker_dir") or "")).name: dict(worker)
        for worker in workers
        if isinstance(worker, Mapping)
    }
    missing_workers = [name for name in names if name not in by_name]
    if missing_workers:
        raise FreshSlowRunError(
            "source manifest lacks selected workers: " + ",".join(missing_workers)
        )
    selected_workers = [by_name[name] for name in names]
    qids = [str(worker["question_id"]) for worker in selected_workers]
    if len(qids) != len(set(qids)):
        raise FreshSlowRunError("selected workers have duplicate question ids")

    temporary = output_run.with_name(output_run.name + f".preparing.{os.getpid()}")
    if temporary.exists():
        raise FreshSlowRunError(f"temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    source_db_hashes_before: dict[str, str] = {}
    source_db_hashes_after: dict[str, str] = {}
    copy_reports: list[dict[str, Any]] = []
    try:
        rewritten_workers: list[dict[str, Any]] = []
        for name, worker in zip(names, selected_workers, strict=True):
            source_worker = Path(str(worker["worker_dir"])).resolve()
            source_db = source_worker / "native_memory.sqlite3"
            if not source_db.is_file():
                raise FreshSlowRunError(f"source worker database is missing: {source_db}")
            source_db_hashes_before[name] = _sha256(source_db)

            destination_worker = temporary / "writer" / name
            destination_worker.mkdir(parents=True)
            for artifact in REQUIRED_WORKER_ARTIFACTS:
                source_artifact = source_worker / artifact
                if not source_artifact.is_file():
                    raise FreshSlowRunError(
                        f"required writer artifact is missing: {source_artifact}"
                    )
                shutil.copy2(source_artifact, destination_worker / artifact)
            # Recovery, migration, quarantine, and warning sidecars are part of
            # the immutable Writer proof chain even though they are optional
            # for workers that never exercised those paths.
            for source_artifact in sorted(source_worker.glob("product_writer_*.jsonl")):
                destination = destination_worker / source_artifact.name
                if not destination.exists():
                    shutil.copy2(source_artifact, destination)

            copy_report = prepare_copy(
                source_db, destination_worker / "native_memory.sqlite3"
            )
            copy_report["worker"] = name
            copy_report["output_db"] = str(
                output_run / "writer" / name / "native_memory.sqlite3"
            )
            copy_reports.append(copy_report)
            _write_json(destination_worker / "fresh_slow_copy_report.json", copy_report)

            rewritten = dict(worker)
            final_worker = output_run / "writer" / name
            rewritten["worker_dir"] = str(final_worker)
            rewritten["input"] = str(final_worker / "input.json")
            rewritten_workers.append(rewritten)

        for name, worker in zip(names, selected_workers, strict=True):
            source_db = Path(str(worker["worker_dir"])).resolve() / "native_memory.sqlite3"
            source_db_hashes_after[name] = _sha256(source_db)
        changed_sources = [
            name
            for name in names
            if source_db_hashes_before[name] != source_db_hashes_after[name]
        ]
        if changed_sources:
            raise FreshSlowRunError(
                "source databases changed during backup: " + ",".join(changed_sources)
            )

        writer_rows_raw = _load_json(source_run / "writer_input.json")
        if not isinstance(writer_rows_raw, list):
            raise FreshSlowRunError("writer_input is not an array")
        writer_rows = _selected_rows(writer_rows_raw, qids, artifact="writer_input")
        writer_input_path = temporary / "writer_input.json"
        _write_json(writer_input_path, writer_rows)

        scope_rows = _selected_rows(
            _load_jsonl(source_run / "scope_manifest.jsonl"), qids, artifact="scope_manifest"
        )
        query_rows = _selected_rows(
            _load_jsonl(source_run / "query_manifest.jsonl"), qids, artifact="query_manifest"
        )
        name_by_qid = dict(zip(qids, names, strict=True))
        scope_rows = [
            _rewrite_manifest_row(row, output_run, name_by_qid[str(row["question_id"])])
            for row in scope_rows
        ]
        query_rows = [
            _rewrite_manifest_row(row, output_run, name_by_qid[str(row["question_id"])])
            for row in query_rows
        ]
        _write_jsonl(temporary / "scope_manifest.jsonl", scope_rows)
        _write_jsonl(temporary / "query_manifest.jsonl", query_rows)
        (temporary / "qids.txt").write_text(
            "".join(qid + "\n" for qid in qids), encoding="utf-8"
        )

        selected_manifest = dict(manifest)
        selected_manifest.update(
            {
                "combined_writer_input": str(output_run / "writer_input.json"),
                "duplicate_session_id_occurrence_count": sum(
                    int(worker.get("duplicate_session_id_occurrence_count", 0) or 0)
                    for worker in selected_workers
                ),
                "duplicate_session_id_qids": [
                    str(worker["question_id"])
                    for worker in selected_workers
                    if int(worker.get("duplicate_session_id_occurrence_count", 0) or 0)
                ],
                "empty_message_count": sum(
                    int(worker.get("empty_message_count", 0) or 0)
                    for worker in selected_workers
                ),
                "input_message_count": sum(
                    int(worker.get("message_count", 0) or 0) for worker in selected_workers
                ),
                "nonempty_message_count": sum(
                    int(worker.get("nonempty_message_count", 0) or 0)
                    for worker in selected_workers
                ),
                "qids": qids,
                "query_manifest": str(output_run / "query_manifest.jsonl"),
                "row_count": len(qids),
                "scope_manifest": str(output_run / "scope_manifest.jsonl"),
                "subset_source_run": str(source_run),
                "workers": rewritten_workers,
                "writer_input_sha256": _sha256(writer_input_path),
            }
        )
        _write_json(temporary / "input_manifest.json", selected_manifest)

        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "created_at": _now(),
            "source_run": str(source_run),
            "output_run": str(output_run),
            "workers": list(names),
            "qids": qids,
            "physical_api_calls": 0,
            "source_database_sha256_before": source_db_hashes_before,
            "source_database_sha256_after": source_db_hashes_after,
            "copies": copy_reports,
        }
        _write_json(temporary / MARKER_NAME, report)
        temporary.rename(output_run)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--workers", action="append", required=True)
    args = parser.parse_args()
    report = prepare_run(args.source_run, args.output_run, _worker_names(args.workers))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
