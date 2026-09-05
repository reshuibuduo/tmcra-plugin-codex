from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import torch
from torch.utils.data import Dataset

from .object_sketch_backend import (
    STYLE_VARIANTS,
    list_object_shape_variants,
    shape_recipe_to_strokes,
    strokes_to_shape_recipe,
)
from .sketch_style_spec import (
    build_stroke_style_profile,
    default_part_graph,
    default_readability_rank,
    default_region_masks,
    infer_sketch_family,
)


TRAINING_STYLE_IDS = tuple(item["id"] for item in STYLE_VARIANTS)
DEFAULT_MAX_STROKES = 12
DEFAULT_POINTS_PER_STROKE = 48
REAL_PROVENANCE_TYPES = {"real_handdrawn", "cleaned_real", "retraced"}
DEFAULT_PROVENANCE_WEIGHTS = {
    "real_handdrawn": 1.35,
    "cleaned_real": 1.15,
    "retraced": 0.95,
    "bootstrap": 0.35,
    "unknown": 0.8,
}


@dataclass(slots=True)
class SketchTrainingClassSpec:
    class_id: str
    scene_type: str
    sketch_family: str
    priority: int = 1
    min_target_samples: int = 192
    bootstrap_from_runtime: bool = True


TARGET_TRAINING_CLASS_SPECS: tuple[SketchTrainingClassSpec, ...] = (
    SketchTrainingClassSpec("building", "scene", "scene_subject", priority=1, min_target_samples=320),
    SketchTrainingClassSpec("house", "scene", "scene_subject", priority=1, min_target_samples=280),
    SketchTrainingClassSpec("window", "scene", "scene_detail", priority=1, min_target_samples=240),
    SketchTrainingClassSpec("door", "scene", "scene_detail", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("tree", "scene", "scene_support", priority=1, min_target_samples=320),
    SketchTrainingClassSpec("cloud", "scene", "scene_environment", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("sun", "scene", "scene_environment", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("person", "scene", "scene_subject", priority=1, min_target_samples=320),
    SketchTrainingClassSpec("car", "scene", "scene_subject", priority=1, min_target_samples=320),
    SketchTrainingClassSpec("street_lamp", "scene", "scene_support", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("table", "scene", "scene_support", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("chair", "scene", "scene_support", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("dog", "scene", "scene_subject", priority=1, min_target_samples=240),
    SketchTrainingClassSpec("desk_lamp", "scene", "scene_support", priority=2, min_target_samples=180),
    SketchTrainingClassSpec("road", "scene", "scene_environment", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("bicycle", "scene", "scene_subject", priority=2, min_target_samples=180),
    SketchTrainingClassSpec("bus", "scene", "scene_subject", priority=2, min_target_samples=180),
    SketchTrainingClassSpec("bridge", "scene", "scene_environment", priority=2, min_target_samples=160),
    SketchTrainingClassSpec("river", "scene", "scene_environment", priority=2, min_target_samples=160),
    SketchTrainingClassSpec("mountain", "scene", "scene_environment", priority=2, min_target_samples=160),
    SketchTrainingClassSpec("bench", "scene", "scene_support", priority=2, min_target_samples=160),
    SketchTrainingClassSpec("bird", "scene", "scene_support", priority=2, min_target_samples=160),
    SketchTrainingClassSpec("flower", "scene", "scene_detail", priority=2, min_target_samples=160),
    SketchTrainingClassSpec("boat", "scene", "scene_subject", priority=2, min_target_samples=160),
    SketchTrainingClassSpec("grass", "scene", "scene_environment", priority=2, min_target_samples=160),
    SketchTrainingClassSpec("cycle", "process", "process_motif", priority=1, min_target_samples=240),
    SketchTrainingClassSpec("flow_node", "process", "process_motif", priority=1, min_target_samples=240),
    SketchTrainingClassSpec("energy_wave", "process", "process_motif", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("vapor", "process", "process_motif", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("leaf", "process", "process_motif", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("raindrop", "process", "process_motif", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("airplane", "process", "process_motif", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("cell", "process", "process_motif", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("photosynthesis", "process", "process_motif", priority=2, min_target_samples=180),
    SketchTrainingClassSpec("heat_flow", "process", "process_motif", priority=2, min_target_samples=180),
    SketchTrainingClassSpec("water_cycle", "process", "process_motif", priority=2, min_target_samples=180),
    SketchTrainingClassSpec("airflow", "process", "process_motif", priority=2, min_target_samples=180),
    SketchTrainingClassSpec("battery", "schematic", "schematic_symbol", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("led", "schematic", "schematic_symbol", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("resistor", "schematic", "schematic_symbol", priority=1, min_target_samples=220),
    SketchTrainingClassSpec("capacitor", "schematic", "schematic_symbol", priority=1, min_target_samples=200),
    SketchTrainingClassSpec("diode", "schematic", "schematic_symbol", priority=1, min_target_samples=200),
    SketchTrainingClassSpec("board", "schematic", "schematic_symbol", priority=1, min_target_samples=200),
    SketchTrainingClassSpec("module", "schematic", "schematic_symbol", priority=1, min_target_samples=200),
    SketchTrainingClassSpec("branch", "schematic", "schematic_symbol", priority=2, min_target_samples=180),
    SketchTrainingClassSpec("switch", "schematic", "schematic_symbol", priority=2, min_target_samples=180),
    SketchTrainingClassSpec("wire", "schematic", "schematic_symbol", priority=2, min_target_samples=180),
    SketchTrainingClassSpec("motor", "schematic", "schematic_symbol", priority=2, min_target_samples=180),
    SketchTrainingClassSpec("sensor", "schematic", "schematic_symbol", priority=2, min_target_samples=180),
    SketchTrainingClassSpec("ground", "schematic", "schematic_symbol", priority=2, min_target_samples=180),
    SketchTrainingClassSpec("chip", "schematic", "schematic_symbol", priority=2, min_target_samples=180),
    SketchTrainingClassSpec("transistor", "schematic", "schematic_symbol", priority=2, min_target_samples=180),
)


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _normalize_provenance(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value in REAL_PROVENANCE_TYPES:
        return value
    if value in {"runtime_variant", "demo_dataset", "trained_export_v3", "bootstrap"}:
        return "bootstrap"
    if not value:
        return "unknown"
    return value


def _infer_provenance(row: Dict[str, Any]) -> str:
    explicit = _normalize_provenance(row.get("provenance"))
    if explicit != "unknown":
        return explicit
    source = str(row.get("source") or "").lower()
    origin_path = str(row.get("origin_path") or "").lower()
    merged = " ".join(part for part in (source, origin_path) if part)
    if any(token in merged for token in ("bootstrap", "runtime_variant", "demo", "trained_export", "synthetic")):
        return "bootstrap"
    if any(token in merged for token in ("retraced", "trace", "vectorized")):
        return "retraced"
    if any(token in merged for token in ("handdrawn", "hand_drawn", "quickdraw", "ink", "tu_berlin", "sketchyscene")):
        return "real_handdrawn"
    if any(token in merged for token in ("clean", "cleaned", "curated", "manual")):
        return "cleaned_real"
    return "unknown"


def _default_sample_weight(*, provenance: str, priority: int) -> float:
    base = DEFAULT_PROVENANCE_WEIGHTS.get(provenance, DEFAULT_PROVENANCE_WEIGHTS["unknown"])
    if priority <= 1:
        base *= 1.12
    elif priority >= 3:
        base *= 0.94
    return round(max(0.1, min(2.0, base)), 4)


def _default_style_cluster_id(*, style_id: str, sketch_family: str) -> str:
    family = str(sketch_family or "generic").strip() or "generic"
    style = str(style_id or "scribble_line").strip() or "scribble_line"
    return f"{family}:{style}"


def _raw_stroke_lengths(
    strokes: Sequence[Sequence[Sequence[float]]],
    *,
    max_strokes: int = DEFAULT_MAX_STROKES,
    points_per_stroke: int = DEFAULT_POINTS_PER_STROKE,
) -> tuple[List[float], int]:
    lengths: List[float] = []
    sequence_length = 0
    for stroke in list(strokes)[:max_strokes]:
        raw_len = 0
        for point in stroke:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                raw_len += 1
        raw_len = max(0, raw_len)
        sequence_length += min(points_per_stroke, raw_len)
        lengths.append(min(1.0, raw_len / max(1, points_per_stroke)))
    return lengths, sequence_length


def iter_target_class_specs(*, priority_at_most: int | None = None) -> List[SketchTrainingClassSpec]:
    specs = list(TARGET_TRAINING_CLASS_SPECS)
    if priority_at_most is not None:
        specs = [item for item in specs if item.priority <= priority_at_most]
    return specs


def target_class_manifest(*, priority_at_most: int | None = None) -> Dict[str, Any]:
    specs = iter_target_class_specs(priority_at_most=priority_at_most)
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for spec in specs:
        by_scene.setdefault(spec.scene_type, []).append(asdict(spec))
    return {
        "style_ids": list(TRAINING_STYLE_IDS),
        "class_count": len(specs),
        "classes": [asdict(item) for item in specs],
        "by_scene_type": by_scene,
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resample_points(points: Sequence[Sequence[float]], target_points: int) -> List[List[float]]:
    cleaned = [[float(px), float(py)] for px, py in points]
    if not cleaned:
        return []
    if len(cleaned) == 1:
        return cleaned * max(2, target_points)
    distances = [0.0]
    for index in range(1, len(cleaned)):
        distances.append(distances[-1] + math.dist(cleaned[index - 1], cleaned[index]))
    total = distances[-1]
    if total <= 1e-6:
        return [cleaned[0] for _ in range(max(2, target_points))]
    sampled: List[List[float]] = []
    for step in range(max(2, target_points)):
        target = total * step / max(1, target_points - 1)
        for index in range(1, len(cleaned)):
            if distances[index] >= target:
                start = cleaned[index - 1]
                end = cleaned[index]
                span = max(1e-6, distances[index] - distances[index - 1])
                ratio = (target - distances[index - 1]) / span
                sampled.append([start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio])
                break
        else:
            sampled.append(cleaned[-1])
    return sampled


def canonicalize_strokes(
    strokes: Sequence[Sequence[Sequence[float]]],
    *,
    max_strokes: int = DEFAULT_MAX_STROKES,
    points_per_stroke: int = DEFAULT_POINTS_PER_STROKE,
) -> List[List[List[float]]]:
    normalized: List[List[List[float]]] = []
    for stroke in strokes[:max_strokes]:
        cleaned = []
        for point in stroke:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            px = max(0.0, min(1.0, float(point[0])))
            py = max(0.0, min(1.0, float(point[1])))
            cleaned.append([px, py])
        if len(cleaned) >= 2:
            normalized.append(_resample_points(cleaned, points_per_stroke))
    return normalized


def augment_strokes(
    strokes: Sequence[Sequence[Sequence[float]]],
    rng: random.Random,
    *,
    max_strokes: int = DEFAULT_MAX_STROKES,
    points_per_stroke: int = DEFAULT_POINTS_PER_STROKE,
    aggressive: bool = False,
) -> List[List[List[float]]]:
    scale = rng.uniform(0.88, 1.12 if aggressive else 1.06)
    shift_x = rng.uniform(-0.06 if aggressive else -0.03, 0.06 if aggressive else 0.03)
    shift_y = rng.uniform(-0.06 if aggressive else -0.03, 0.06 if aggressive else 0.03)
    angle = math.radians(rng.uniform(-10.0 if aggressive else -4.0, 10.0 if aggressive else 4.0))
    jitter = rng.uniform(0.004, 0.02 if aggressive else 0.012)
    mirror = rng.random() < (0.12 if aggressive else 0.05)
    keep_ratio = rng.uniform(0.6, 1.0) if aggressive else rng.uniform(0.78, 1.0)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    augmented: List[List[List[float]]] = []
    for stroke in strokes:
        if len(augmented) >= max_strokes:
            break
        points: List[List[float]] = []
        for px, py in stroke:
            cx = float(px) - 0.5
            cy = float(py) - 0.5
            rx = cx * cos_a - cy * sin_a
            ry = cx * sin_a + cy * cos_a
            nx = rx * scale + 0.5 + shift_x + rng.uniform(-jitter, jitter)
            ny = ry * scale + 0.5 + shift_y + rng.uniform(-jitter, jitter)
            if mirror:
                nx = 1.0 - nx
            points.append([max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))])
        if rng.random() < 0.22 and len(points) > 8:
            target_len = max(6, int(len(points) * keep_ratio))
            points = points[:target_len]
        if len(points) >= 2:
            augmented.append(_resample_points(points, points_per_stroke))
    if aggressive and augmented and rng.random() < 0.08:
        augmented = augmented[1:] or augmented
    return augmented[:max_strokes]


def flatten_stroke_sequence(
    strokes: Sequence[Sequence[Sequence[float]]],
    *,
    max_strokes: int = DEFAULT_MAX_STROKES,
    points_per_stroke: int = DEFAULT_POINTS_PER_STROKE,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence: List[List[float]] = []
    mask: List[float] = []
    for stroke_index in range(max_strokes):
        stroke = strokes[stroke_index] if stroke_index < len(strokes) else []
        for point_index in range(points_per_stroke):
            if point_index < len(stroke):
                px, py = stroke[point_index]
                pen_break = 1.0 if point_index == len(stroke) - 1 else 0.0
                sequence.append([float(px), float(py), pen_break])
                mask.append(1.0)
            else:
                sequence.append([0.0, 0.0, 1.0])
                mask.append(0.0)
    return torch.tensor(sequence, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def sequence_to_strokes(
    sequence: torch.Tensor,
    *,
    max_strokes: int = DEFAULT_MAX_STROKES,
    points_per_stroke: int = DEFAULT_POINTS_PER_STROKE,
) -> List[List[List[float]]]:
    values = sequence.detach().cpu().tolist()
    strokes: List[List[List[float]]] = []
    current: List[List[float]] = []
    for index, (px, py, pen_break) in enumerate(values[: max_strokes * points_per_stroke]):
        current.append([max(0.0, min(1.0, float(px))), max(0.0, min(1.0, float(py)))])
        if pen_break >= 0.5 or (index + 1) % points_per_stroke == 0:
            if len(current) >= 2:
                strokes.append(current)
            current = []
    if current and len(current) >= 2:
        strokes.append(current)
    return strokes[:max_strokes]


def compute_bbox(strokes: Sequence[Sequence[Sequence[float]]]) -> List[float]:
    points = [(float(px), float(py)) for stroke in strokes for px, py in stroke]
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
    min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
    return [min_x, min_y, max(0.0, max_x - min_x), max(0.0, max_y - min_y)]


def _stable_hash(text: str) -> float:
    digest = hashlib.md5(text.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def normalize_training_row(
    row: Dict[str, Any],
    *,
    fallback_class_id: str | None = None,
    fallback_scene_type: str | None = None,
    fallback_style_id: str = "scribble_line",
    max_strokes: int = DEFAULT_MAX_STROKES,
    points_per_stroke: int = DEFAULT_POINTS_PER_STROKE,
) -> Dict[str, Any] | None:
    class_id = str(row.get("class_id") or fallback_class_id or "").strip()
    if not class_id:
        return None
    scene_type = str(row.get("scene_type") or fallback_scene_type or "scene").strip() or "scene"
    style_id = str(row.get("style_id") or fallback_style_id).strip() or fallback_style_id
    sketch_family = str(row.get("sketch_family") or infer_sketch_family(class_id, scene_type=scene_type)).strip()
    shape_recipe = _copy(row.get("shape_recipe") or {})
    stroke_payload = _copy(row.get("stroke_payload") or [])
    raw_strokes = row.get("strokes") or stroke_payload or shape_recipe_to_strokes(shape_recipe)
    raw_lengths, sequence_length = _raw_stroke_lengths(
        raw_strokes,
        max_strokes=max_strokes,
        points_per_stroke=points_per_stroke,
    )
    strokes = canonicalize_strokes(
        raw_strokes,
        max_strokes=max_strokes,
        points_per_stroke=points_per_stroke,
    )
    if not strokes:
        return None
    region_masks = _copy(row.get("region_masks") or default_region_masks(class_id, scene_type=scene_type))
    part_graph = _copy(row.get("part_graph") or default_part_graph(class_id, scene_type=scene_type))
    readability_rank = int(row.get("readability_rank", default_readability_rank(class_id, scene_type=scene_type)) or default_readability_rank(class_id, scene_type=scene_type))
    priority = int(row.get("priority", 2) or 2)
    provenance = _infer_provenance(row)
    style_cluster_id = str(row.get("style_cluster_id") or _default_style_cluster_id(style_id=style_id, sketch_family=sketch_family))
    stroke_style_profile = _copy(
        row.get("stroke_style_profile")
        or build_stroke_style_profile(
            class_id,
            scene_type=scene_type,
            style_variant=style_id,
            sketch_family=sketch_family,
        )
    )
    variant_id = str(row.get("variant_id") or row.get("id") or f"{class_id}:{style_id}:{abs(hash(json.dumps(strokes, ensure_ascii=False))) % 10_000_000}")
    payload = {
        "class_id": class_id,
        "scene_type": scene_type,
        "style_id": style_id,
        "variant_id": variant_id,
        "strokes": strokes,
        "stroke_payload": _copy(strokes),
        "shape_recipe": strokes_to_shape_recipe(strokes),
        "source": str(row.get("source") or "unknown"),
        "region_masks": region_masks,
        "part_graph": part_graph,
        "stroke_style_profile": stroke_style_profile,
        "stroke_render_profile": _copy(row.get("stroke_render_profile") or stroke_style_profile),
        "sketch_family": sketch_family,
        "readability_rank": readability_rank,
        "stroke_count": len(strokes),
        "bbox": compute_bbox(strokes),
        "render_representation": str(row.get("render_representation") or "stroke_native"),
        "stroke_variant_id": str(row.get("stroke_variant_id") or variant_id),
        "stroke_payload_source": str(row.get("stroke_payload_source") or row.get("source") or "unknown"),
        "naturalness_score": float(row.get("naturalness_score", row.get("quality_score", 0.6)) or 0.6),
        "quality_score": float(row.get("quality_score", row.get("naturalness_score", 0.6)) or 0.6),
        "provenance": provenance,
        "sample_weight": float(row.get("sample_weight", _default_sample_weight(provenance=provenance, priority=priority)) or _default_sample_weight(provenance=provenance, priority=priority)),
        "style_cluster_id": style_cluster_id,
        "stroke_lengths": raw_lengths[:max_strokes],
        "sequence_length": int(sequence_length or len(strokes) * points_per_stroke),
        "priority": priority,
    }
    extra_keys = ("origin_path", "layout_condition", "object_boxes", "depth_order", "relation_graph", "condition_maps", "target_sketch_path", "layout_quality_score")
    for key in extra_keys:
        if key in row:
            payload[key] = _copy(row[key])
    return payload


def build_bootstrap_rows(
    *,
    class_specs: Sequence[SketchTrainingClassSpec] | None = None,
    max_strokes: int = DEFAULT_MAX_STROKES,
    points_per_stroke: int = DEFAULT_POINTS_PER_STROKE,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    specs = list(class_specs or TARGET_TRAINING_CLASS_SPECS)
    for spec in specs:
        if not spec.bootstrap_from_runtime:
            continue
        variants = list_object_shape_variants(spec.class_id, spec.class_id, spec.scene_type)
        if not variants:
            continue
        for variant in variants:
            for style_id in TRAINING_STYLE_IDS:
                normalized = normalize_training_row(
                    {
                        "class_id": spec.class_id,
                        "scene_type": spec.scene_type,
                        "style_id": style_id,
                        "variant_id": variant.get("id") or f"{spec.class_id}:{style_id}",
                        "render_representation": variant.get("render_representation", "stroke_native"),
                        "stroke_variant_id": variant.get("stroke_variant_id") or variant.get("id") or f"{spec.class_id}:{style_id}",
                        "stroke_payload": _copy(variant.get("stroke_payload") or []),
                        "stroke_payload_source": variant.get("stroke_payload_source") or variant.get("source") or "runtime_variant",
                        "stroke_render_profile": _copy(variant.get("stroke_render_profile") or variant.get("stroke_style_profile") or {}),
                        "shape_recipe": _copy(variant.get("shape_recipe") or {}),
                        "region_masks": _copy(variant.get("region_masks") or []),
                        "part_graph": _copy(variant.get("part_graph") or []),
                        "stroke_style_profile": _copy(variant.get("stroke_style_profile") or {}),
                        "sketch_family": spec.sketch_family,
                        "readability_rank": variant.get("readability_rank", default_readability_rank(spec.class_id, scene_type=spec.scene_type)),
                        "source": str(variant.get("source") or "runtime_variant"),
                        "priority": spec.priority,
                        "provenance": "bootstrap",
                    },
                    fallback_class_id=spec.class_id,
                    fallback_scene_type=spec.scene_type,
                    max_strokes=max_strokes,
                    points_per_stroke=points_per_stroke,
                )
                if normalized:
                    rows.append(normalized)
    return rows


def expand_rows_to_targets(
    rows: Sequence[Dict[str, Any]],
    *,
    class_specs: Sequence[SketchTrainingClassSpec] | None = None,
    seed: int = 42,
    max_multiplier: float = 1.0,
    max_strokes: int = DEFAULT_MAX_STROKES,
    points_per_stroke: int = DEFAULT_POINTS_PER_STROKE,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    specs = list(class_specs or TARGET_TRAINING_CLASS_SPECS)
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        normalized = normalize_training_row(
            row,
            max_strokes=max_strokes,
            points_per_stroke=points_per_stroke,
        )
        if normalized:
            normalized_rows.append(normalized)
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in normalized_rows:
        grouped.setdefault((row["class_id"], row["style_id"]), []).append(row)
    spec_lookup = {item.class_id: item for item in specs}
    result = list(normalized_rows)
    for spec in specs:
        per_style_target = max(24, math.ceil(spec.min_target_samples * max_multiplier / max(1, len(TRAINING_STYLE_IDS))))
        for style_id in TRAINING_STYLE_IDS:
            key = (spec.class_id, style_id)
            candidates = list(grouped.get(key, []))
            if not candidates:
                fallback_rows = [
                    row
                    for row in normalized_rows
                    if row["class_id"] == spec.class_id
                ]
                candidates = []
                for item in fallback_rows:
                    cloned = _copy(item)
                    cloned["style_id"] = style_id
                    cloned["stroke_style_profile"] = build_stroke_style_profile(
                        spec.class_id,
                        scene_type=spec.scene_type,
                        style_variant=style_id,
                        sketch_family=spec.sketch_family,
                    )
                    normalized = normalize_training_row(
                        cloned,
                        fallback_class_id=spec.class_id,
                        fallback_scene_type=spec.scene_type,
                        max_strokes=max_strokes,
                        points_per_stroke=points_per_stroke,
                    )
                    if normalized:
                        candidates.append(normalized)
                if not candidates:
                    continue
                grouped[key] = list(candidates)
                result.extend(candidates)
            while len(grouped.get(key, [])) < per_style_target:
                base = _copy(rng.choice(candidates))
                base["variant_id"] = f'{base["variant_id"]}::auto_aug_{len(grouped.get(key, [])) + 1}'
                base["source"] = f'{base.get("source", "unknown")}:scale_augmented'
                base["style_id"] = style_id
                base["strokes"] = augment_strokes(
                    base.get("strokes", []),
                    rng,
                    max_strokes=max_strokes,
                    points_per_stroke=points_per_stroke,
                    aggressive=len(grouped.get(key, [])) > per_style_target * 0.6,
                )
                base["stroke_payload"] = _copy(base["strokes"])
                base["render_representation"] = "stroke_native"
                base["stroke_payload_source"] = str(base.get("source") or "scale_augmented")
                base["stroke_variant_id"] = str(base.get("stroke_variant_id") or base.get("variant_id") or "")
                base["stroke_style_profile"] = build_stroke_style_profile(
                    spec.class_id,
                    scene_type=spec.scene_type,
                    style_variant=style_id,
                    sketch_family=spec.sketch_family,
                )
                normalized = normalize_training_row(
                    base,
                    fallback_class_id=spec.class_id,
                    fallback_scene_type=spec.scene_type,
                    max_strokes=max_strokes,
                    points_per_stroke=points_per_stroke,
                )
                if not normalized:
                    break
                grouped.setdefault(key, []).append(normalized)
                result.append(normalized)
    return result


def split_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    train_ratio: float = 0.9,
    val_ratio: float = 0.07,
) -> Dict[str, List[Dict[str, Any]]]:
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    boundary_train = max(0.0, min(1.0, train_ratio))
    boundary_val = max(boundary_train, min(1.0, train_ratio + val_ratio))
    for row in rows:
        key = f'{row.get("class_id", "")}|{row.get("style_id", "")}|{row.get("variant_id", "")}'
        score = _stable_hash(key)
        if score < boundary_train:
            train.append(_copy(row))
        elif score < boundary_val:
            val.append(_copy(row))
        else:
            test.append(_copy(row))
    return {"train": train, "val": val, "test": test}


def build_index_maps(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    class_ids = sorted({str(row.get("class_id") or "") for row in rows if row.get("class_id")})
    style_ids = sorted({str(row.get("style_id") or "") for row in rows if row.get("style_id")})
    scene_ids = sorted({str(row.get("scene_type") or "scene") for row in rows})
    family_ids = sorted({str(row.get("sketch_family") or "") for row in rows if row.get("sketch_family")})
    provenance_ids = sorted({_normalize_provenance(row.get("provenance")) for row in rows} | {"unknown"})
    style_cluster_ids = sorted({str(row.get("style_cluster_id") or _default_style_cluster_id(style_id=row.get("style_id"), sketch_family=row.get("sketch_family"))) for row in rows})
    return {
        "class_to_idx": {item: index for index, item in enumerate(class_ids)},
        "style_to_idx": {item: index for index, item in enumerate(style_ids)},
        "scene_to_idx": {item: index for index, item in enumerate(scene_ids)},
        "family_to_idx": {item: index for index, item in enumerate(family_ids)},
        "provenance_to_idx": {item: index for index, item in enumerate(provenance_ids)},
        "style_cluster_to_idx": {item: index for index, item in enumerate(style_cluster_ids)},
    }


def build_dataset_manifest(
    split_rows_map: Dict[str, Sequence[Dict[str, Any]]],
    *,
    class_specs: Sequence[SketchTrainingClassSpec] | None = None,
) -> Dict[str, Any]:
    specs = list(class_specs or TARGET_TRAINING_CLASS_SPECS)
    spec_lookup = {item.class_id: item for item in specs}
    all_rows = [row for rows in split_rows_map.values() for row in rows]
    mappings = build_index_maps(all_rows)
    split_summary: Dict[str, Any] = {}
    for split_name, rows in split_rows_map.items():
        class_counts: Dict[str, int] = {}
        style_counts: Dict[str, int] = {}
        scene_counts: Dict[str, int] = {}
        source_counts: Dict[str, int] = {}
        provenance_counts: Dict[str, int] = {}
        readability = []
        stroke_counts = []
        naturalness = []
        for row in rows:
            class_counts[row["class_id"]] = class_counts.get(row["class_id"], 0) + 1
            style_counts[row["style_id"]] = style_counts.get(row["style_id"], 0) + 1
            scene_counts[row["scene_type"]] = scene_counts.get(row["scene_type"], 0) + 1
            source = str(row.get("source") or "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            provenance = _normalize_provenance(row.get("provenance"))
            provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1
            readability.append(int(row.get("readability_rank", 0) or 0))
            stroke_counts.append(int(row.get("stroke_count", 0) or 0))
            naturalness.append(float(row.get("naturalness_score", row.get("quality_score", 0.0)) or 0.0))
        real_rows = sum(count for key, count in provenance_counts.items() if key in REAL_PROVENANCE_TYPES)
        split_summary[split_name] = {
            "row_count": len(rows),
            "class_counts": class_counts,
            "style_counts": style_counts,
            "scene_counts": scene_counts,
            "source_counts": source_counts,
            "provenance_counts": provenance_counts,
            "avg_readability_rank": round(sum(readability) / max(1, len(readability)), 3),
            "avg_stroke_count": round(sum(stroke_counts) / max(1, len(stroke_counts)), 3),
            "avg_naturalness_score": round(sum(naturalness) / max(1, len(naturalness)), 4),
            "real_data_ratio": round(real_rows / max(1, len(rows)), 4),
            "bootstrap_ratio": round(provenance_counts.get("bootstrap", 0) / max(1, len(rows)), 4),
        }
    coverage = []
    for spec in specs:
        class_rows = [row for row in all_rows if row.get("class_id") == spec.class_id]
        total = len(class_rows)
        real_rows = sum(1 for row in class_rows if _normalize_provenance(row.get("provenance")) in REAL_PROVENANCE_TYPES)
        bootstrap_rows = sum(1 for row in class_rows if _normalize_provenance(row.get("provenance")) == "bootstrap")
        real_ratio = real_rows / max(1, total)
        target_real_ratio = 0.7 if spec.priority == 1 else 0.55 if spec.priority == 2 else 0.45
        coverage.append(
            {
                "class_id": spec.class_id,
                "scene_type": spec.scene_type,
                "sketch_family": spec.sketch_family,
                "priority": spec.priority,
                "min_target_samples": spec.min_target_samples,
                "actual_samples": total,
                "coverage_ratio": round(total / max(1, spec.min_target_samples), 3),
                "real_samples": real_rows,
                "real_ratio": round(real_ratio, 4),
                "bootstrap_ratio": round(bootstrap_rows / max(1, total), 4),
                "target_real_ratio": target_real_ratio,
            }
        )
    return {
        "target_taxonomy": target_class_manifest(),
        "mappings": mappings,
        "splits": split_summary,
        "coverage": coverage,
        "row_count": len(all_rows),
        "class_count": len({row["class_id"] for row in all_rows}),
        "style_count": len({row["style_id"] for row in all_rows}),
        "scene_type_count": len({row["scene_type"] for row in all_rows}),
        "priority_gaps": [
            item
            for item in coverage
            if item["priority"] == 1
            and (
                item["actual_samples"] < spec_lookup[item["class_id"]].min_target_samples
                or item["real_ratio"] < item["target_real_ratio"]
            )
        ],
    }


class PreparedObjectSketchDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        mappings: Dict[str, Dict[str, int]],
        *,
        max_strokes: int = DEFAULT_MAX_STROKES,
        points_per_stroke: int = DEFAULT_POINTS_PER_STROKE,
    ):
        self.rows = list(rows)
        self.mappings = mappings
        self.max_strokes = max_strokes
        self.points_per_stroke = points_per_stroke

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        sequence, mask = flatten_stroke_sequence(
            row.get("strokes", []),
            max_strokes=self.max_strokes,
            points_per_stroke=self.points_per_stroke,
        )
        class_to_idx = self.mappings["class_to_idx"]
        style_to_idx = self.mappings["style_to_idx"]
        scene_to_idx = self.mappings["scene_to_idx"]
        family_to_idx = self.mappings["family_to_idx"]
        provenance_to_idx = self.mappings.get("provenance_to_idx", {"unknown": 0})
        style_cluster_to_idx = self.mappings.get("style_cluster_to_idx", {})
        stroke_lengths = [float(value) for value in (row.get("stroke_lengths") or [])[: self.max_strokes]]
        if len(stroke_lengths) < self.max_strokes:
            stroke_lengths.extend([0.0] * (self.max_strokes - len(stroke_lengths)))
        provenance = _normalize_provenance(row.get("provenance"))
        style_cluster_id = str(row.get("style_cluster_id") or _default_style_cluster_id(style_id=row.get("style_id"), sketch_family=row.get("sketch_family")))
        return {
            "sequence": sequence,
            "mask": mask,
            "class_id": torch.tensor(class_to_idx[row["class_id"]], dtype=torch.long),
            "style_id": torch.tensor(style_to_idx[row["style_id"]], dtype=torch.long),
            "scene_id": torch.tensor(scene_to_idx[row["scene_type"]], dtype=torch.long),
            "family_id": torch.tensor(family_to_idx[row["sketch_family"]], dtype=torch.long),
            "stroke_count": torch.tensor(min(self.max_strokes, int(row.get("stroke_count", len(row.get("strokes", []))) or 0)), dtype=torch.long),
            "bbox": torch.tensor(row.get("bbox") or compute_bbox(row.get("strokes", [])), dtype=torch.float32),
            "readability_rank": torch.tensor(float(row.get("readability_rank", 0) or 0.0), dtype=torch.float32),
            "naturalness_score": torch.tensor(float(row.get("naturalness_score", row.get("quality_score", 0.0)) or 0.0), dtype=torch.float32),
            "sample_weight": torch.tensor(float(row.get("sample_weight", 1.0) or 1.0), dtype=torch.float32),
            "provenance_id": torch.tensor(provenance_to_idx.get(provenance, provenance_to_idx.get("unknown", 0)), dtype=torch.long),
            "style_cluster_id": torch.tensor(style_cluster_to_idx.get(style_cluster_id, 0), dtype=torch.long),
            "stroke_lengths": torch.tensor(stroke_lengths, dtype=torch.float32),
            "sequence_length": torch.tensor(int(row.get("sequence_length", int(mask.sum().item())) or 0), dtype=torch.long),
        }


def collate_object_sketch_batch(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sequence": torch.stack([item["sequence"] for item in items], dim=0),
        "mask": torch.stack([item["mask"] for item in items], dim=0),
        "class_id": torch.stack([item["class_id"] for item in items], dim=0),
        "style_id": torch.stack([item["style_id"] for item in items], dim=0),
        "scene_id": torch.stack([item["scene_id"] for item in items], dim=0),
        "family_id": torch.stack([item["family_id"] for item in items], dim=0),
        "stroke_count": torch.stack([item["stroke_count"] for item in items], dim=0),
        "bbox": torch.stack([item["bbox"] for item in items], dim=0),
        "readability_rank": torch.stack([item["readability_rank"] for item in items], dim=0),
        "naturalness_score": torch.stack([item["naturalness_score"] for item in items], dim=0),
        "sample_weight": torch.stack([item["sample_weight"] for item in items], dim=0),
        "provenance_id": torch.stack([item["provenance_id"] for item in items], dim=0),
        "style_cluster_id": torch.stack([item["style_cluster_id"] for item in items], dim=0),
        "stroke_lengths": torch.stack([item["stroke_lengths"] for item in items], dim=0),
        "sequence_length": torch.stack([item["sequence_length"] for item in items], dim=0),
    }
