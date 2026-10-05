#!/usr/bin/env python3
"""Compare two read-only Slow review exports by evidence and capsule partition."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "tmcra.v4.slow-review-comparison.1"


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").split())


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Slow review export is unreadable: {path}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("entries"), list):
        raise ValueError(f"Slow review export is invalid: {path}")
    return dict(payload)


def _index(payload: Mapping[str, Any]) -> dict[str, Any]:
    support_to_claim: dict[str, dict[str, Any]] = {}
    support_to_capsule: dict[str, dict[str, Any]] = {}
    duplicate_support_ids: list[str] = []
    capsules: list[dict[str, Any]] = []
    for entry in payload["entries"]:
        if not isinstance(entry, Mapping):
            raise ValueError("Slow review entry must be an object")
        resulting = entry.get("resulting_capsule")
        metadata = resulting.get("metadata") if isinstance(resulting, Mapping) else None
        claims = metadata.get("claims") if isinstance(metadata, Mapping) else None
        if not isinstance(claims, list):
            raise ValueError("Slow review entry has no resulting capsule claims")
        capsule = {
            "worker": str(entry.get("worker") or ""),
            "region_key": str(entry.get("region_key") or ""),
            "capsule_key": str(metadata.get("capsule_key") or ""),
            "summary": str(resulting.get("value") or ""),
            "support_ids": [],
        }
        for claim in claims:
            if not isinstance(claim, Mapping):
                raise ValueError("Slow claim must be an object")
            support = claim.get("support")
            if not isinstance(support, list) or not support:
                raise ValueError("Slow claim support must be a non-empty list")
            for raw_support_id in support:
                support_id = str(raw_support_id)
                if support_id in support_to_claim:
                    duplicate_support_ids.append(support_id)
                support_to_claim[support_id] = {
                    "worker": capsule["worker"],
                    "region_key": capsule["region_key"],
                    "canonical_slot": str(claim.get("canonical_slot") or ""),
                    "text": str(claim.get("text") or ""),
                }
                capsule["support_ids"].append(support_id)
        capsule["support_ids"] = sorted(capsule["support_ids"])
        capsules.append(capsule)
        for support_id in capsule["support_ids"]:
            support_to_capsule[support_id] = capsule
    return {
        "capsules": capsules,
        "support_to_claim": support_to_claim,
        "support_to_capsule": support_to_capsule,
        "duplicate_support_ids": sorted(set(duplicate_support_ids)),
    }


def compare(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    old = _index(baseline)
    new = _index(candidate)
    old_ids = set(old["support_to_claim"])
    new_ids = set(new["support_to_claim"])
    shared = sorted(old_ids & new_ids)
    claim_changes: list[dict[str, Any]] = []
    slot_changes: list[dict[str, Any]] = []
    changed_support_ids: list[str] = []
    changed_regions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for support_id in shared:
        old_claim = old["support_to_claim"][support_id]
        new_claim = new["support_to_claim"][support_id]
        if old_claim["canonical_slot"] != new_claim["canonical_slot"]:
            slot_changes.append(
                {
                    "support_id": support_id,
                    "baseline_slot": old_claim["canonical_slot"],
                    "candidate_slot": new_claim["canonical_slot"],
                }
            )
        if _normalize(old_claim["text"]) != _normalize(new_claim["text"]):
            claim_changes.append(
                {
                    "support_id": support_id,
                    "worker": new_claim["worker"],
                    "region_key": new_claim["region_key"],
                    "baseline_text": old_claim["text"],
                    "candidate_text": new_claim["text"],
                }
            )
        old_group = old["support_to_capsule"][support_id]["support_ids"]
        new_group = new["support_to_capsule"][support_id]["support_ids"]
        if old_group != new_group:
            changed_support_ids.append(support_id)
            changed_regions[(new_claim["worker"], new_claim["region_key"])].add(
                support_id
            )

    region_changes: list[dict[str, Any]] = []
    for (worker, region_key), support_ids in sorted(changed_regions.items()):
        old_capsules = {
            tuple(old["support_to_capsule"][support_id]["support_ids"]): old[
                "support_to_capsule"
            ][support_id]
            for support_id in support_ids
        }
        new_capsules = {
            tuple(new["support_to_capsule"][support_id]["support_ids"]): new[
                "support_to_capsule"
            ][support_id]
            for support_id in support_ids
        }
        region_changes.append(
            {
                "worker": worker,
                "region_key": region_key,
                "changed_support_count": len(support_ids),
                "baseline_capsules": list(old_capsules.values()),
                "candidate_capsules": list(new_capsules.values()),
            }
        )

    missing = sorted(old_ids - new_ids)
    added = sorted(new_ids - old_ids)
    duplicates = sorted(
        set(old["duplicate_support_ids"]) | set(new["duplicate_support_ids"])
    )
    blocking = bool(missing or added or duplicates or slot_changes)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed" if blocking else "passed",
        "blocking_issue_count": (
            len(missing) + len(added) + len(duplicates) + len(slot_changes)
        ),
        "baseline": {
            "prompt_version": baseline.get("prompt_version"),
            "capsule_count": len(old["capsules"]),
            "support_count": len(old_ids),
        },
        "candidate": {
            "prompt_version": candidate.get("prompt_version"),
            "capsule_count": len(new["capsules"]),
            "support_count": len(new_ids),
        },
        "missing_support_ids": missing,
        "added_support_ids": added,
        "duplicate_support_ids": duplicates,
        "slot_changes": slot_changes,
        "claim_text_changes": claim_changes,
        "partition_changed_support_count": len(changed_support_ids),
        "partition_changed_region_count": len(region_changes),
        "region_changes": region_changes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    report = compare(_load(args.baseline), _load(args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
