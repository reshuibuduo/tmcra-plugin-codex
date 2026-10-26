from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

from experiments.replacement.adapters.base import AdapterResponse
from experiments.replacement.adapters.memory_adapters import GraphSessionMemoryAdapter
from experiments.replacement.memory_graph import guess_slot_key
from experiments.replacement_overlay.pipeline import ReasoningRequest, TMCRAReasoningPipeline


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _dedupe(items: Iterable[object]) -> List[str]:
    values: List[str] = []
    seen = set()
    for item in items:
        text = _clean_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(text)
    return values


@dataclass(slots=True)
class RuntimeFeatureFlags:
    TMCRA_REASONING_V2_ENABLED: bool = False
    TMCRA_REASONING_V2_SHADOW: bool = True
    TMCRA_TEMPORAL_REASONING_ENABLED: bool = True
    TMCRA_SLOT_RESOLUTION_ENABLED: bool = True

    def to_dict(self) -> Dict[str, bool]:
        return {
            "TMCRA_REASONING_V2_ENABLED": bool(self.TMCRA_REASONING_V2_ENABLED),
            "TMCRA_REASONING_V2_SHADOW": bool(self.TMCRA_REASONING_V2_SHADOW),
            "TMCRA_TEMPORAL_REASONING_ENABLED": bool(self.TMCRA_TEMPORAL_REASONING_ENABLED),
            "TMCRA_SLOT_RESOLUTION_ENABLED": bool(self.TMCRA_SLOT_RESOLUTION_ENABLED),
        }


@dataclass(slots=True)
class SessionMemoryNormalizationAdapter:
    default_source_kind: str = "session_memory"

    def normalize_records(self, records: Sequence[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for index, raw in enumerate(records):
            if isinstance(raw, dict):
                category = _clean_text(raw.get("category", "memory")) or "memory"
                value = _clean_text(raw.get("value", ""))
                anchors = [_clean_text(anchor) for anchor in raw.get("anchor_concepts", raw.get("anchors", [])) or [] if _clean_text(anchor)]
                turn_index = int(raw.get("turn_index", 0) or 0)
                source_kind = _clean_text(raw.get("source_kind", "")) or self.default_source_kind
                slot_key = _clean_text(raw.get("slot_key", raw.get("slot", ""))) or guess_slot_key(category=category, value=value, anchors=anchors)
                if value:
                    normalized.append(
                        {
                            "category": category,
                            "slot": slot_key,
                            "value": value,
                            "anchors": anchors[:8],
                            "relation": _clean_text(raw.get("relation", "")) or f"{category}_memory",
                            "source_kind": source_kind,
                            "turn_index": turn_index,
                            "metadata": dict(raw.get("metadata", {}) or {}),
                        }
                    )
                continue
            category = _clean_text(getattr(raw, "category", "memory")) or "memory"
            value = _clean_text(getattr(raw, "value", ""))
            anchors = [_clean_text(anchor) for anchor in getattr(raw, "anchor_concepts", []) or [] if _clean_text(anchor)]
            if not value:
                continue
            normalized.append(
                {
                    "category": category,
                    "slot": guess_slot_key(category=category, value=value, anchors=anchors),
                    "value": value,
                    "anchors": anchors[:8],
                    "relation": _clean_text(getattr(raw, "relation", "")) or f"{category}_memory",
                    "source_kind": _clean_text(getattr(raw, "source_kind", "")) or self.default_source_kind,
                    "turn_index": int(getattr(raw, "turn_index", 0) or 0),
                    "metadata": dict(getattr(raw, "metadata", {}) or {}),
                }
            )
        return normalized

    def build_memory_adapter(
        self,
        *,
        records: Sequence[Any] | None = None,
        session_turns: Sequence[Dict[str, Any]] | None = None,
    ) -> GraphSessionMemoryAdapter:
        adapter = GraphSessionMemoryAdapter(auto_extract=False)
        if session_turns:
            for turn in session_turns:
                adapter.ingest_turn(
                    _clean_text(turn.get("user_text", "")),
                    _clean_text(turn.get("assistant_text", "")),
                    answer_payload=dict(turn.get("answer_payload", {}) or {}),
                    extraction_result=dict(turn.get("extraction_result", {}) or {}),
                )
        normalized = self.normalize_records(records or [])
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for item in normalized:
            grouped.setdefault(int(item.get("turn_index", 0) or 0), []).append(item)
        for turn_index in sorted(grouped.keys()):
            adapter.ingest_turn(
                f"normalized_session_memory_turn_{turn_index}",
                "",
                answer_payload={"replacement_memory_records": grouped[turn_index], "metadata": {"source": "main_chain_shadow"}},
            )
        return adapter


class MainChainReasoningAdapter:
    def __init__(
        self,
        *,
        flags: RuntimeFeatureFlags | None = None,
        normalizer: SessionMemoryNormalizationAdapter | None = None,
        pipeline: TMCRAReasoningPipeline | None = None,
    ) -> None:
        self.flags = flags or RuntimeFeatureFlags()
        self.normalizer = normalizer or SessionMemoryNormalizationAdapter()
        self.pipeline = pipeline or TMCRAReasoningPipeline()

    def answer(
        self,
        query: str,
        *,
        answer_mode: str = "transparent",
        legacy_response: AdapterResponse | None = None,
        session_records: Sequence[Any] | None = None,
        session_turns: Sequence[Dict[str, Any]] | None = None,
        top_k: int = 6,
        memory_name: str = "session_memory",
    ) -> AdapterResponse:
        adapter = self.normalizer.build_memory_adapter(records=session_records, session_turns=session_turns)
        shadow_bundle = self.pipeline.run(
            ReasoningRequest(query=query, answer_mode=answer_mode, top_k=top_k, metadata={"source": "main_chain_shadow"}),
            memory_adapter=adapter,
            base_response=legacy_response,
            reasoner_name="tmcra_reasoning_v2",
            memory_name=memory_name,
        )
        if self.flags.TMCRA_REASONING_V2_ENABLED:
            response = shadow_bundle.response
        else:
            response = legacy_response or AdapterResponse(
                answer="",
                answer_mode=answer_mode,
                reasoner_name="main_chain_legacy",
                memory_name=memory_name,
            )
        response = AdapterResponse(
            answer=response.answer,
            answer_mode=response.answer_mode,
            reasoner_name=response.reasoner_name,
            memory_name=response.memory_name,
            confidence=response.confidence,
            paths=list(response.paths),
            facts=list(response.facts),
            candidate_scores=list(response.candidate_scores),
            memory_hits=list(response.memory_hits),
            evidence_consistent=bool(response.evidence_consistent),
            unsupported_claims=list(response.unsupported_claims),
            pillar_scores=dict(response.pillar_scores or {}),
            latency_seconds=response.latency_seconds,
            trace={
                **dict(response.trace or {}),
                "tmcra_reasoning_v2_shadow": {
                    "enabled": bool(self.flags.TMCRA_REASONING_V2_ENABLED),
                    "shadow": bool(self.flags.TMCRA_REASONING_V2_SHADOW),
                    "flags": self.flags.to_dict(),
                    "shadow_response": shadow_bundle.response.to_dict(),
                },
            },
            metadata={
                **dict(response.metadata or {}),
                "tmcra_reasoning_v2": {
                    "enabled": bool(self.flags.TMCRA_REASONING_V2_ENABLED),
                    "shadow": bool(self.flags.TMCRA_REASONING_V2_SHADOW),
                    "flags": self.flags.to_dict(),
                    "shadow_bundle": shadow_bundle.to_dict(),
                    "normalized_slots": _dedupe(view["slot_key"] for view in shadow_bundle.context.slot_resolution.to_dict().get("views", [])),
                },
            },
        )
        return response


def create_reasoning_v2_shadow_adapter(*, flags: RuntimeFeatureFlags | None = None) -> MainChainReasoningAdapter:
    return MainChainReasoningAdapter(flags=flags or RuntimeFeatureFlags())
