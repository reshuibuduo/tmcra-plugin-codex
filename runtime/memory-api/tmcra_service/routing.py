from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


EvidenceMode = Literal["raw", "auto", "compiled"]


@dataclass(frozen=True)
class EvidenceRoute:
    requested: EvidenceMode
    selected: Literal["raw", "compiled"]
    reasons: tuple[str, ...]


def select_evidence_route(
    requested: EvidenceMode, evidence: Mapping[str, Any]
) -> EvidenceRoute:
    if requested == "raw":
        return EvidenceRoute(requested, "raw", ("caller_selected_raw",))
    if requested == "compiled":
        return EvidenceRoute(requested, "compiled", ("caller_selected_compiled",))
    if requested != "auto":
        raise ValueError(f"unknown evidence mode: {requested}")

    plan = dict(evidence.get("recall_plan") or {})
    windows = [dict(item) for item in list(evidence.get("evidence_windows") or [])]
    reasons: list[str] = []
    if str(plan.get("query_kind") or "") == "comparison":
        reasons.append("planner_comparison")
    if str(plan.get("temporal_focus") or "") == "mixed":
        reasons.append("planner_mixed_temporal_focus")
    if any(
        bool(dict(item.get("retrieval_metadata") or {}).get("newer_fast_override"))
        or bool(dict(item.get("retrieval_metadata") or {}).get("conflict_detected"))
        for item in windows
    ):
        reasons.append("retrieval_conflict")
    selected = "compiled" if reasons else "raw"
    if not reasons:
        reasons.append("no_high_risk_evidence_condition")
    return EvidenceRoute(requested, selected, tuple(reasons))

