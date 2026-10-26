#!/usr/bin/env python3
"""Apply an explicit human disposition to a raw Slow partition diff."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return dict(value)


def _strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a string list")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicates")
    return sorted(value)


def approve(raw: Mapping[str, Any], manual: Mapping[str, Any]) -> dict[str, Any]:
    if not str(manual.get("status") or "").startswith("passed"):
        raise ValueError("manual review status did not pass")
    if int(manual.get("blocking_issue_count", -1)) != 0:
        raise ValueError("manual review contains blocking issues")
    fields = {
        "missing_support_ids": "approved_missing_support_ids",
        "added_support_ids": "approved_added_support_ids",
        "duplicate_support_ids": "approved_duplicate_support_ids",
    }
    approved_count = 0
    for raw_field, manual_field in fields.items():
        actual = _strings(raw.get(raw_field), raw_field)
        approved = _strings(manual.get(manual_field), manual_field)
        if actual != approved:
            raise ValueError(f"manual review does not exactly disposition {raw_field}")
        approved_count += len(actual)
    raw_slots = raw.get("slot_changes")
    if not isinstance(raw_slots, list) or any(
        not isinstance(item, Mapping) or not isinstance(item.get("support_id"), str)
        for item in raw_slots
    ):
        raise ValueError("slot_changes is invalid")
    slot_ids = sorted(str(item["support_id"]) for item in raw_slots)
    approved_slots = _strings(
        manual.get("approved_slot_change_support_ids"),
        "approved_slot_change_support_ids",
    )
    if slot_ids != approved_slots:
        raise ValueError("manual review does not exactly disposition slot_changes")
    approved_count += len(slot_ids)
    output = dict(raw)
    output.update(
        {
            "schema_version": "tmcra.v4.reviewed-slow-partition-diff.1",
            "raw_status": raw.get("status"),
            "raw_blocking_issue_count": int(raw.get("blocking_issue_count", -1)),
            "status": "passed",
            "blocking_issue_count": 0,
            "approved_issue_count": approved_count,
            "manual_review_status": manual.get("status"),
            "manual_review_decision": manual.get("decision"),
        }
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-diff", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    try:
        report = approve(_object(args.raw_diff), _object(args.manual_review))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
