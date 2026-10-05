from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path


GRAPH_ZERO_RE = re.compile(
    r"(?P<message_id>s\d+_m\d+): proposal \d+ resolved to 0 persisted records"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _classify(error: str) -> str:
    value = error.lower()
    if "resolved to 0 persisted records" in value:
        return "graph_commit_zero"
    if "exact message coverage" in value or "exactly one entry for every" in value:
        return "message_coverage"
    if "clean json" in value or "json object" in value or "jsondecode" in value:
        return "json_format"
    if any(
        token in value
        for token in (
            "incompleteread",
            "remotedisconnected",
            "connection reset",
            "timed out",
            "transport",
            "urlerror",
            "http 502",
        )
    ):
        return "transport"
    return "unknown"


def _signature_collision(plan: dict) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for assertion in list(dict(plan.get("extraction") or {}).get("assertions") or []):
        key = (
            assertion.get("canonical_key"),
            assertion.get("evidence_quote"),
            int(assertion.get("evidence_char_start", 0) or 0),
            int(assertion.get("evidence_char_end", 0) or 0),
            assertion.get("polarity"),
        )
        groups.setdefault(key, []).append(assertion)
    collisions = []
    for key, assertions in groups.items():
        claims = sorted({str(item.get("claim_text") or "") for item in assertions})
        if len(claims) < 2:
            continue
        collisions.append(
            {
                "canonical_key": key[0],
                "evidence_char_start": key[2],
                "evidence_char_end": key[3],
                "polarity": key[4],
                "claim_count": len(claims),
                "claims": claims,
            }
        )
    return {
        "collision_count": len(collisions),
        "collisions": collisions,
    }


def _worker_failure(worker: Path, index: int) -> dict:
    input_payload = _read_json(worker / "input.json")
    question_id = str(input_payload[0].get("question_id") or "")
    database = worker / "native_memory.sqlite3"
    result = {
        "index": index,
        "question_id": question_id,
        "worker_dir": str(worker),
        "database": str(database),
    }
    if not database.is_file():
        result.update(category="missing_database", error="native database is missing")
        return result

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        batches = [
            dict(row)
            for row in connection.execute(
                "SELECT batch_id,status,error,length(response_json) AS response_length "
                "FROM v4_batch_journal ORDER BY batch_index"
            )
        ]
        messages = [
            dict(row)
            for row in connection.execute(
                "SELECT commit_id,batch_id,message_id,status,error,plan_json "
                "FROM v4_message_commit_journal ORDER BY rowid"
            )
        ]
        sources = dict(
            connection.execute(
                "SELECT status,COUNT(*) FROM v4_source_journal GROUP BY status"
            ).fetchall()
        )

    errors = [
        str(row.get("error") or "")
        for row in [*batches, *messages]
        if str(row.get("error") or "")
    ]
    error = errors[-1] if errors else ""
    category = _classify(error)
    result.update(
        category=category,
        error=error,
        quick_check=quick_check,
        batch_statuses=dict(Counter(str(row["status"]) for row in batches)),
        message_statuses=dict(Counter(str(row["status"]) for row in messages)),
        source_statuses=sources,
        response_lengths=[
            int(row.get("response_length") or 0)
            for row in batches
            if str(row.get("status")) != "committed"
        ],
    )

    match = GRAPH_ZERO_RE.search(error)
    if category == "graph_commit_zero" and match:
        message_id = match.group("message_id")
        row = next(
            (item for item in messages if str(item["message_id"]) == message_id),
            None,
        )
        if row and str(row.get("plan_json") or ""):
            result["graph_identity_analysis"] = _signature_collision(
                json.loads(str(row["plan_json"]))
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    writer_root = args.run_dir / "writer"
    workers = sorted(
        writer_root.glob("worker_*"),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    failures = []
    completed = []
    for worker in workers:
        index = int(worker.name.rsplit("_", 1)[-1])
        if (worker / "product_writer_report.json").is_file():
            completed.append(index)
            continue
        failures.append(_worker_failure(worker, index))

    category_counts = Counter(item["category"] for item in failures)
    graph_zero = [item for item in failures if item["category"] == "graph_commit_zero"]
    report = {
        "schema_version": "tmcra.v4.remaining400-writer-failure-analysis.1",
        "run_dir": str(args.run_dir),
        "worker_count": len(workers),
        "completed_count": len(completed),
        "failed_count": len(failures),
        "category_counts": dict(sorted(category_counts.items())),
        "graph_zero_with_distinct_claim_collision": sum(
            bool(item.get("graph_identity_analysis", {}).get("collision_count"))
            for item in graph_zero
        ),
        "completed_indices": completed,
        "failed_indices": [item["index"] for item in failures],
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "worker_count",
        "completed_count",
        "failed_count",
        "category_counts",
        "graph_zero_with_distinct_claim_collision",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
