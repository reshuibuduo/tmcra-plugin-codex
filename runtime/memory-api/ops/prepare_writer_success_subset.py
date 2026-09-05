#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_root(value: Any, source: Path, destination: Path) -> Any:
    if isinstance(value, str):
        return value.replace(str(source), str(destination))
    if isinstance(value, list):
        return [_replace_root(item, source, destination) for item in value]
    if isinstance(value, dict):
        return {key: _replace_root(item, source, destination) for key, item in value.items()}
    return value


def _selected_indices(source: Path, manifest: dict[str, Any], policy: str) -> set[int]:
    if policy == "soak_completed":
        results = _read_jsonl(source / "writer_soak_results.jsonl")
        return {
            int(result["index"])
            for result in results
            if result.get("status") == "completed"
        }
    if policy != "audit_passed":
        raise RuntimeError(f"unsupported selection policy: {policy}")
    selected: set[int] = set()
    for worker in manifest["workers"]:
        index = int(worker["worker_index"])
        worker_dir = Path(worker["worker_dir"])
        required = (
            worker_dir / "native_memory.sqlite3",
            worker_dir / "product_writer_report.json",
            worker_dir / "writer_chain_audit.json",
        )
        if not all(path.is_file() for path in required):
            continue
        audit = json.loads(required[2].read_text(encoding="utf-8"))
        report = json.loads(required[1].read_text(encoding="utf-8"))
        if audit.get("status") == "passed" and report.get("completed") is True:
            selected.add(index)
    return selected


def prepare(
    source: Path,
    destination: Path,
    *,
    selection_policy: str = "soak_completed",
) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise RuntimeError(f"destination already exists: {destination}")

    manifest = json.loads((source / "input_manifest.json").read_text(encoding="utf-8"))
    completed_indices = _selected_indices(source, manifest, selection_policy)
    workers = [
        worker
        for worker in manifest["workers"]
        if int(worker["worker_index"]) in completed_indices
    ]
    if len(workers) != len(completed_indices):
        raise RuntimeError("completed results do not match input manifest workers")

    destination.mkdir(parents=True)
    (destination / "writer").mkdir()
    for worker in workers:
        source_worker = Path(worker["worker_dir"])
        relative = source_worker.relative_to(source)
        shutil.copytree(source_worker, destination / relative, copy_function=shutil.copy2)

    qids = [str(worker["question_id"]) for worker in workers]
    qid_set = set(qids)
    worker_input = json.loads((source / "writer_input.json").read_text(encoding="utf-8"))
    filtered_input = [row for row in worker_input if str(row["question_id"]) in qid_set]
    (destination / "writer_input.json").write_text(
        json.dumps(filtered_input, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for name in ("query_manifest.jsonl", "scope_manifest.jsonl"):
        rows = [row for row in _read_jsonl(source / name) if str(row["question_id"]) in qid_set]
        _write_jsonl(destination / name, _replace_root(rows, source, destination))

    (destination / "evaluation_only").mkdir()
    references = [
        row
        for row in _read_jsonl(source / "evaluation_only" / "references.jsonl")
        if str(row["question_id"]) in qid_set
    ]
    _write_jsonl(destination / "evaluation_only" / "references.jsonl", references)
    (destination / "qids.txt").write_text("".join(f"{qid}\n" for qid in qids), encoding="utf-8")

    source_worker_count = len(manifest["workers"])
    manifest = _replace_root(manifest, source, destination)
    manifest["qids"] = qids
    manifest["row_count"] = len(workers)
    manifest["workers"] = [_replace_root(worker, source, destination) for worker in workers]
    manifest["input_message_count"] = sum(int(worker["message_count"]) for worker in workers)
    manifest["nonempty_message_count"] = sum(int(worker["nonempty_message_count"]) for worker in workers)
    manifest["empty_message_count"] = sum(int(worker["empty_message_count"]) for worker in workers)
    manifest["subset_source_run"] = str(source)
    manifest["subset_policy"] = selection_policy
    manifest["writer_input_sha256"] = _sha256(destination / "writer_input.json")
    (destination / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = {
        "status": "prepared",
        "source_run": str(source),
        "destination_run": str(destination),
        "completed_workers": len(workers),
        "excluded_workers": source_worker_count - len(workers),
        "selection_policy": selection_policy,
        "qids": qids,
    }
    (destination / "writer_subset_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "WRITER_SUBSET_PREPARED").write_text("complete\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze successful Writer soak workers into a build run")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--selection-policy",
        choices=("soak_completed", "audit_passed"),
        default="soak_completed",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source,
                args.destination,
                selection_policy=args.selection_policy,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
