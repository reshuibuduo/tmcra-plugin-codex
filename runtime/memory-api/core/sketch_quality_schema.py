from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


SKETCH_QUALITY_SCHEMA_VERSION = "tmcra_sketch_quality_v1"
DISTANCE_TO_TARGET_VALUES = ("close", "medium", "far")
SCENE_COVERAGE_GROUPS = ("nature", "urban", "street", "indoor", "surreal", "general")
FOCUS_MODES = (
    "class_separation",
    "silhouette_stability",
    "line_cleanup",
    "layout_to_scene_structure",
    "depth_and_perspective",
    "support_object_coverage",
    "editability",
    "target_style_alignment",
)

DIMENSION_TO_FOCUS_MODES: dict[str, tuple[str, ...]] = {
    "subject_readability": ("class_separation", "silhouette_stability"),
    "line_cleanliness": ("line_cleanup", "target_style_alignment"),
    "editable_control": ("editability", "layout_to_scene_structure"),
    "scene_layering": ("layout_to_scene_structure", "support_object_coverage"),
    "perspective_logic": ("depth_and_perspective",),
    "scene_richness": ("support_object_coverage",),
}


@dataclass(frozen=True)
class SketchQualityDimension:
    key: str
    label: str
    weight: float
    description: str
    scoring_focus: tuple[str, ...]
    downgrade_signals: tuple[str, ...]


CANONICAL_SKETCH_QUALITY_DIMENSIONS: tuple[SketchQualityDimension, ...] = (
    SketchQualityDimension(
        key="subject_readability",
        label="Subject Readability",
        weight=0.24,
        description="Main subject class, silhouette, and semantic intent are obvious at a glance.",
        scoring_focus=("class separation", "silhouette stability", "clear primary subject"),
        downgrade_signals=("ambiguous subject", "merged classes", "missing key parts"),
    ),
    SketchQualityDimension(
        key="line_cleanliness",
        label="Line Cleanliness",
        weight=0.2,
        description="Contours are clean, stable, and usable as editable sketch structure.",
        scoring_focus=("clean contour control", "consistent stroke quality", "low clutter"),
        downgrade_signals=("scratchy lines", "double edges", "noisy decorative detail"),
    ),
    SketchQualityDimension(
        key="editable_control",
        label="Editable Control",
        weight=0.18,
        description="Geometry is controllable, decomposable, and easy to edit downstream.",
        scoring_focus=("stable structure", "part boundaries", "layout controllability"),
        downgrade_signals=("collapsed forms", "uneditable tangles", "weak part separation"),
    ),
    SketchQualityDimension(
        key="scene_layering",
        label="Scene Layering",
        weight=0.15,
        description="Foreground, midground, background, and support objects form a readable scene stack.",
        scoring_focus=("depth grouping", "support object placement", "clear near-far layout"),
        downgrade_signals=("flat layout", "missing support context", "foreground/background confusion"),
    ),
    SketchQualityDimension(
        key="perspective_logic",
        label="Perspective Logic",
        weight=0.13,
        description="Scale, overlap, road/bridge direction, skyline placement, and depth cues are coherent.",
        scoring_focus=("correct scale falloff", "consistent vanishing logic", "physical plausibility"),
        downgrade_signals=("broken depth", "conflicting scale", "perspective drift"),
    ),
    SketchQualityDimension(
        key="scene_richness",
        label="Scene Richness",
        weight=0.1,
        description="The scene includes enough secondary elements to support the subject without clutter.",
        scoring_focus=("useful support objects", "scene completeness", "non-empty composition"),
        downgrade_signals=("too empty", "support objects missing", "overcrowded clutter"),
    ),
)


LEGACY_TMCRA_DIMENSION_CROSSWALK: dict[str, tuple[str, ...]] = {
    "concept_clarity": ("subject_readability",),
    "structural_integrity": ("editable_control", "scene_layering"),
    "line_quality": ("line_cleanliness",),
    "style_consistency": ("line_cleanliness", "editable_control"),
    "creative_expression": ("scene_richness",),
    "semantic_accuracy": ("subject_readability", "perspective_logic"),
}


TMCRA_SKETCH_PHILOSOPHY = (
    "TMCRA sketch training should favor semantic clarity, controllable structure, and useful scene support "
    "over decorative detail. The goal is not photorealism. The goal is a readable, editable structural sketch "
    "that can carry intent through the product chain: query -> scene plan -> scene line -> final sketch quality."
)


def dimension_keys() -> list[str]:
    return [item.key for item in CANONICAL_SKETCH_QUALITY_DIMENSIONS]


def dimension_specs() -> list[SketchQualityDimension]:
    return list(CANONICAL_SKETCH_QUALITY_DIMENSIONS)


def build_output_template() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SKETCH_QUALITY_SCHEMA_VERSION,
        "overall_score": 0.0,
    }
    for item in CANONICAL_SKETCH_QUALITY_DIMENSIONS:
        payload[item.key] = 0.0
    payload.update(
        {
            "distance_to_target": "medium",
            "major_gaps": ["gap"],
            "retrain_focus": ["class_separation"],
            "verdict": "needs_targeted_retrain",
            "adaptive_training_policy": {
                "generalization_strength": 0.0,
                "specialization_strength": 0.0,
                "synthetic_data_priority": 0.0,
                "base_data_priority": 0.0,
                "scene_coverage_groups": ["general"],
                "focus_modes": ["layout_to_scene_structure"],
                "rationale": "short reason",
            },
        }
    )
    return payload


def output_template_json() -> str:
    return json.dumps(build_output_template(), ensure_ascii=False, indent=2)


def build_vlm_judge_prompt() -> str:
    lines = [
        "You are reviewing a clean-line sketch preview generated during TMCRA training.",
        TMCRA_SKETCH_PHILOSOPHY,
        "",
        "Score every primary dimension from 0.0 to 10.0.",
        "Use harsher penalties for unreadable structure, weak class separation, bad depth logic, and messy outlines than for lack of decorative detail.",
        "",
        "Primary dimensions:",
    ]
    for item in CANONICAL_SKETCH_QUALITY_DIMENSIONS:
        focus = ", ".join(item.scoring_focus)
        penalties = ", ".join(item.downgrade_signals)
        lines.append(
            f"- {item.key} ({item.label}, weight {item.weight:.2f}): {item.description} "
            f"Focus on {focus}. Downgrade for {penalties}."
        )
    lines.extend(
        [
            "",
            "Older TMCRA internal rubric ideas should be translated into the production-facing dimensions above using this crosswalk:",
        ]
    )
    for legacy_key, canonical_keys in LEGACY_TMCRA_DIMENSION_CROSSWALK.items():
        lines.append(f"- {legacy_key} -> {', '.join(canonical_keys)}")
    lines.extend(
        [
            "",
            "Important scoring rule: favor readability, controllable structure, class separation, clean editable outlines, correct near-far logic, coherent perspective, and rich but uncluttered scene support over decorative detail.",
            "If foreground/background size, overlap, road direction, skyline placement, bridge perspective, or object depth logic is weak, score scene_layering and perspective_logic aggressively lower.",
            "If the scene is too empty, lacks support objects, or fails to provide useful secondary elements around the subject, score scene_richness lower even if the main subject is readable.",
            "Use adaptive_training_policy to describe what the next training round should emphasize.",
            "",
            "Return strict JSON:",
            output_template_json(),
        ]
    )
    return "\n".join(lines)


def _clamp_float(value: Any, *, lower: float, upper: float, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return max(lower, min(upper, parsed))


def _clean_string_list(values: Any) -> list[str]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, dict)):
        return []
    cleaned: list[str] = []
    for value in values:
        token = str(value or "").strip()
        if token:
            cleaned.append(token)
    return cleaned


def _mapped_legacy_scores(parsed: dict[str, Any]) -> dict[str, list[float]]:
    mapped: dict[str, list[float]] = {item.key: [] for item in CANONICAL_SKETCH_QUALITY_DIMENSIONS}
    for legacy_key, canonical_keys in LEGACY_TMCRA_DIMENSION_CROSSWALK.items():
        if legacy_key not in parsed:
            continue
        value = _clamp_float(parsed.get(legacy_key), lower=0.0, upper=10.0, default=0.0)
        for canonical_key in canonical_keys:
            mapped.setdefault(canonical_key, []).append(value)
    return mapped


def normalize_sketch_quality_result(parsed: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}
    normalized: dict[str, Any] = {}
    mapped_legacy = _mapped_legacy_scores(parsed)
    weighted_total = 0.0
    weight_sum = 0.0
    for item in CANONICAL_SKETCH_QUALITY_DIMENSIONS:
        raw_value = parsed.get(item.key)
        if raw_value is None and mapped_legacy.get(item.key):
            raw_value = sum(mapped_legacy[item.key]) / len(mapped_legacy[item.key])
        value = _clamp_float(raw_value, lower=0.0, upper=10.0, default=0.0)
        normalized[item.key] = value
        weighted_total += value * item.weight
        weight_sum += item.weight
    overall_default = weighted_total / weight_sum if weight_sum > 0 else 0.0
    normalized["schema_version"] = str(parsed.get("schema_version") or SKETCH_QUALITY_SCHEMA_VERSION)
    normalized["overall_score"] = _clamp_float(
        parsed.get("overall_score"),
        lower=0.0,
        upper=10.0,
        default=overall_default,
    )
    distance = str(parsed.get("distance_to_target") or "medium").strip().lower()
    distance_aliases = {
        "moderate": "medium",
        "mid": "medium",
        "good": "close",
        "poor": "far",
    }
    distance = distance_aliases.get(distance, distance)
    normalized["distance_to_target"] = distance if distance in DISTANCE_TO_TARGET_VALUES else "medium"
    fallback_gap_dimensions = [
        item.label
        for item in CANONICAL_SKETCH_QUALITY_DIMENSIONS
        if normalized[item.key] < 6.0
    ]
    major_gaps = _clean_string_list(parsed.get("major_gaps"))
    if not major_gaps:
        major_gaps = [f"{label} is weak" for label in fallback_gap_dimensions[:3]]
    normalized["major_gaps"] = major_gaps
    retrain_focus = [token for token in _clean_string_list(parsed.get("retrain_focus")) if token in FOCUS_MODES]
    if not retrain_focus:
        weakest_dimensions = sorted(
            CANONICAL_SKETCH_QUALITY_DIMENSIONS,
            key=lambda item: normalized[item.key],
        )[:2]
        derived_focus: list[str] = []
        for item in weakest_dimensions:
            for token in DIMENSION_TO_FOCUS_MODES.get(item.key, ()):
                if token not in derived_focus:
                    derived_focus.append(token)
        retrain_focus = derived_focus
    normalized["retrain_focus"] = retrain_focus
    verdict = str(parsed.get("verdict") or "").strip().lower()
    if not verdict:
        minimum_dimension = min(normalized[item.key] for item in CANONICAL_SKETCH_QUALITY_DIMENSIONS)
        if normalized["overall_score"] >= 8.5 and minimum_dimension >= 8.0:
            verdict = "target_close_enough"
        elif normalized["overall_score"] >= 7.0:
            verdict = "continue_with_targeted_cleanup"
        else:
            verdict = "needs_targeted_retrain"
    normalized["verdict"] = verdict
    adaptive_policy = parsed.get("adaptive_training_policy")
    policy_payload = adaptive_policy if isinstance(adaptive_policy, dict) else {}
    normalized["adaptive_training_policy"] = {
        "generalization_strength": _clamp_float(
            policy_payload.get("generalization_strength"), lower=0.0, upper=1.0, default=0.0
        ),
        "specialization_strength": _clamp_float(
            policy_payload.get("specialization_strength"), lower=0.0, upper=1.0, default=0.0
        ),
        "synthetic_data_priority": _clamp_float(
            policy_payload.get("synthetic_data_priority"), lower=0.0, upper=1.0, default=0.0
        ),
        "base_data_priority": _clamp_float(
            policy_payload.get("base_data_priority"), lower=0.0, upper=1.0, default=0.0
        ),
        "scene_coverage_groups": [
            token
            for token in (item.strip().lower() for item in _clean_string_list(policy_payload.get("scene_coverage_groups")))
            if token in SCENE_COVERAGE_GROUPS
        ],
        "focus_modes": [
            token
            for token in _clean_string_list(policy_payload.get("focus_modes"))
            if token in FOCUS_MODES
        ],
        "rationale": str(policy_payload.get("rationale") or "").strip(),
    }
    return normalized


__all__ = [
    "CANONICAL_SKETCH_QUALITY_DIMENSIONS",
    "DISTANCE_TO_TARGET_VALUES",
    "DIMENSION_TO_FOCUS_MODES",
    "FOCUS_MODES",
    "LEGACY_TMCRA_DIMENSION_CROSSWALK",
    "SCENE_COVERAGE_GROUPS",
    "SKETCH_QUALITY_SCHEMA_VERSION",
    "SketchQualityDimension",
    "TMCRA_SKETCH_PHILOSOPHY",
    "build_output_template",
    "build_vlm_judge_prompt",
    "dimension_keys",
    "dimension_specs",
    "normalize_sketch_quality_result",
    "output_template_json",
]
