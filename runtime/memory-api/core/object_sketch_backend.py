from __future__ import annotations

import json
import math
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .sketch_style_spec import (
    build_stroke_style_profile,
    default_part_graph,
    default_readability_rank,
    default_region_masks,
    infer_sketch_family,
)
from .visual_prototypes import FAMILY_FALLBACKS, resolve_visual_prototype
from .default_model_paths import (
    LEGACY_OBJECT_STROKE_VARIANTS_PATH,
    LEGACY_OBJECT_VARIANTS_PATH,
    resolve_default_object_stroke_variants_path,
    resolve_default_object_variants_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "object_sketch"
DEFAULT_EXPORTED_VARIANTS_PATH = LEGACY_OBJECT_VARIANTS_PATH
DEFAULT_EXPORTED_STROKE_VARIANTS_PATH = LEGACY_OBJECT_STROKE_VARIANTS_PATH
EXPORTED_VARIANTS_PATH = DEFAULT_EXPORTED_VARIANTS_PATH
PUBLIC_SEED_PATH = DATA_DIR / "public_seed_dataset.jsonl"
MANUAL_SEED_PATH = DATA_DIR / "manual_seed_dataset.jsonl"

DEMO_BACKEND_MODE = "hybrid_demo"
SCENE_OBJECT_CLASSES = ("building", "house", "window", "door", "tree", "cloud", "sun", "person", "car", "street_lamp", "table", "chair", "dog", "desk_lamp", "road")
PROCESS_MOTIF_CLASSES = ("cycle", "flow_node", "energy_wave", "vapor", "leaf", "raindrop", "airplane", "cell", "branch", "module")
SCHEMATIC_SYMBOL_CLASSES = ("battery", "led", "resistor", "capacitor", "diode", "board", "module", "branch", "flow_node")
TRAINED_DEMO_CLASSES = tuple(dict.fromkeys([*SCENE_OBJECT_CLASSES, *PROCESS_MOTIF_CLASSES, *SCHEMATIC_SYMBOL_CLASSES]).keys())
CLASS_SCENE_MAP = {item: "scene" for item in SCENE_OBJECT_CLASSES} | {item: "process" for item in PROCESS_MOTIF_CLASSES} | {item: "schematic" for item in SCHEMATIC_SYMBOL_CLASSES}
STYLE_VARIANTS = ({"id": "scribble_line", "label": "Scribble line"}, {"id": "clean_line", "label": "Clean line"})


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def resolve_exported_variants_path() -> Path:
    override = os.getenv("TMCRA_OBJECT_VARIANTS_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return resolve_default_object_variants_path()


def resolve_exported_stroke_variants_path() -> Path:
    override = os.getenv("TMCRA_OBJECT_STROKE_VARIANTS_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return resolve_default_object_stroke_variants_path()


def _normalize_stroke_payload(payload: Any) -> List[List[List[float]]]:
    strokes: List[List[List[float]]] = []
    for stroke in payload if isinstance(payload, list) else []:
        if not isinstance(stroke, list):
            continue
        cleaned: List[List[float]] = []
        for point in stroke:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            cleaned.append([max(0.0, min(1.0, float(point[0]))), max(0.0, min(1.0, float(point[1])))])
        if len(cleaned) >= 2:
            strokes.append(cleaned)
    return strokes


def _resolve_sketch_engine() -> str:
    raw = str(os.getenv("TMCRA_SKETCH_ENGINE", "auto") or "auto").strip().lower()
    if raw in {"mechanical", "mechanical_v1"}:
        return "mechanical_v1"
    if raw in {"natural", "natural_v1"}:
        return "natural_v1"
    return "auto"


def _shape(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"version": 1, "viewbox": [0, 0, 1, 1], "parts": parts}


def _rect(x: float, y: float, w: float, h: float, *, rx: float = 0.0, fill_role: str = "fill", stroke_role: str = "line", stroke_width: float = 0.02) -> Dict[str, Any]:
    return {"kind": "rect", "x": x, "y": y, "w": w, "h": h, "rx": rx, "fill_role": fill_role, "stroke_role": stroke_role, "stroke_width": stroke_width, "opacity": 1.0}


def _ellipse(x: float, y: float, w: float, h: float, *, fill_role: str = "fill", stroke_role: str = "line", stroke_width: float = 0.02) -> Dict[str, Any]:
    return {"kind": "ellipse", "x": x, "y": y, "w": w, "h": h, "fill_role": fill_role, "stroke_role": stroke_role, "stroke_width": stroke_width, "opacity": 1.0}


def _line(x1: float, y1: float, x2: float, y2: float, *, stroke_role: str = "line", stroke_width: float = 0.02, dash: List[float] | None = None) -> Dict[str, Any]:
    part = {"kind": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "stroke_role": stroke_role, "stroke_width": stroke_width, "opacity": 1.0}
    if dash:
        part["dash"] = list(dash)
    return part


def _polygon(points: Sequence[Sequence[float]], *, fill_role: str = "fill", stroke_role: str = "line", stroke_width: float = 0.02) -> Dict[str, Any]:
    return {"kind": "polygon", "points": [[float(px), float(py)] for px, py in points], "fill_role": fill_role, "stroke_role": stroke_role, "stroke_width": stroke_width, "opacity": 1.0}


def _polyline(points: Sequence[Sequence[float]], *, stroke_role: str = "line", stroke_width: float = 0.02) -> Dict[str, Any]:
    return {"kind": "polyline", "points": [[float(px), float(py)] for px, py in points], "stroke_role": stroke_role, "stroke_width": stroke_width, "opacity": 1.0}


def _variant(asset_key: str, suffix: str, label: str, parts: List[Dict[str, Any]], *, scene_type: str = "", source: str = "demo_library", confidence: float = 0.82) -> Dict[str, Any]:
    resolved_scene = scene_type or CLASS_SCENE_MAP.get(asset_key, "scene")
    family = infer_sketch_family(asset_key, scene_type=resolved_scene)
    return {
        "id": f"{asset_key}:{suffix}",
        "label": label,
        "asset_key": asset_key,
        "shape_recipe": _shape(parts),
        "source": source,
        "confidence": float(confidence),
        "style_variants": [item["id"] for item in STYLE_VARIANTS],
        "default_style": "scribble_line",
        "part_graph": default_part_graph(asset_key, scene_type=resolved_scene),
        "region_masks": default_region_masks(asset_key, scene_type=resolved_scene),
        "stroke_style_profile": build_stroke_style_profile(asset_key, scene_type=resolved_scene, style_variant="scribble_line", sketch_family=family),
        "readability_rank": default_readability_rank(asset_key, scene_type=resolved_scene),
        "sketch_family": family,
    }


def _fallback_variant(asset_key: str, concept: str = "", scene_type: str = "scene") -> Dict[str, Any]:
    prototype = resolve_visual_prototype(asset_key, concept, scene_type)
    family = infer_sketch_family(asset_key, scene_type=scene_type)
    return {
        "id": f"{asset_key}:rule_base",
        "label": f"{asset_key} base",
        "asset_key": asset_key,
        "shape_recipe": _copy(prototype.get("shape_recipe", {})),
        "source": "rule_prototype",
        "confidence": 0.64 if asset_key not in TRAINED_DEMO_CLASSES else 0.78,
        "style_variants": [item["id"] for item in STYLE_VARIANTS],
        "default_style": "scribble_line",
        "part_graph": default_part_graph(asset_key, scene_type=scene_type),
        "region_masks": default_region_masks(asset_key, scene_type=scene_type),
        "stroke_style_profile": build_stroke_style_profile(asset_key, scene_type=scene_type, style_variant="scribble_line", sketch_family=family),
        "readability_rank": default_readability_rank(asset_key, scene_type=scene_type),
        "sketch_family": family,
    }


@lru_cache(maxsize=1)
def _builtin_variant_library() -> Dict[str, List[Dict[str, Any]]]:
    library = {
        "building": [_variant("building", "city_block", "City block", [_rect(0.12, 0.1, 0.76, 0.88, rx=0.03), _rect(0.22, 0.2, 0.1, 0.1, fill_role="none", stroke_role="accent", stroke_width=0.014), _rect(0.42, 0.2, 0.1, 0.1, fill_role="none", stroke_role="accent", stroke_width=0.014), _rect(0.62, 0.2, 0.1, 0.1, fill_role="none", stroke_role="accent", stroke_width=0.014), _rect(0.44, 0.72, 0.12, 0.24, fill_role="none", stroke_role="accent", stroke_width=0.014)], confidence=0.84)],
        "house": [_variant("house", "gable", "Gable house", [_polygon([[0.5, 0.02], [0.08, 0.34], [0.92, 0.34]]), _rect(0.16, 0.34, 0.68, 0.62, rx=0.03), _rect(0.42, 0.58, 0.16, 0.38, fill_role="none", stroke_role="accent", stroke_width=0.014), _rect(0.24, 0.46, 0.12, 0.12, fill_role="none", stroke_role="accent", stroke_width=0.014), _rect(0.64, 0.46, 0.12, 0.12, fill_role="none", stroke_role="accent", stroke_width=0.014)], confidence=0.84)],
        "window": [_variant("window", "cross", "Window", [_rect(0.08, 0.08, 0.84, 0.84, rx=0.04), _line(0.5, 0.1, 0.5, 0.9, stroke_role="accent", stroke_width=0.018), _line(0.1, 0.5, 0.9, 0.5, stroke_role="accent", stroke_width=0.018)], confidence=0.84)],
        "door": [_variant("door", "rounded", "Rounded door", [_rect(0.14, 0.06, 0.72, 0.9, rx=0.14), _ellipse(0.7, 0.5, 0.06, 0.06, fill_role="accent_fill", stroke_role="accent", stroke_width=0.01)], confidence=0.82)],
        "tree": [_variant("tree", "round_canopy", "Round canopy", [_rect(0.44, 0.58, 0.12, 0.38, fill_role="region_alt", stroke_role="line", stroke_width=0.018), _ellipse(0.22, 0.16, 0.56, 0.42), _ellipse(0.08, 0.34, 0.28, 0.24), _ellipse(0.62, 0.32, 0.22, 0.22)], confidence=0.86)],
        "cloud": [_variant("cloud", "puffy", "Puffy cloud", [_ellipse(0.06, 0.42, 0.3, 0.26), _ellipse(0.28, 0.2, 0.34, 0.34), _ellipse(0.54, 0.36, 0.28, 0.24), _line(0.16, 0.72, 0.78, 0.72, stroke_width=0.016)], confidence=0.87)],
        "sun": [_variant("sun", "classic", "Classic sun", [_ellipse(0.24, 0.24, 0.52, 0.52), _line(0.5, 0.0, 0.5, 0.18, stroke_role="accent", stroke_width=0.018), _line(0.5, 0.82, 0.5, 1.0, stroke_role="accent", stroke_width=0.018), _line(0.0, 0.5, 0.18, 0.5, stroke_role="accent", stroke_width=0.018), _line(0.82, 0.5, 1.0, 0.5, stroke_role="accent", stroke_width=0.018)], confidence=0.87)],
        "person": [_variant("person", "standing", "Standing person", [_ellipse(0.36, 0.04, 0.28, 0.22), _line(0.5, 0.28, 0.5, 0.68, stroke_width=0.024), _line(0.5, 0.38, 0.24, 0.52, stroke_width=0.02), _line(0.5, 0.38, 0.76, 0.52, stroke_width=0.02), _line(0.5, 0.68, 0.26, 0.98, stroke_width=0.02), _line(0.5, 0.68, 0.74, 0.98, stroke_width=0.02)], confidence=0.86)],
        "car": [_variant("car", "sedan", "Sedan", [_rect(0.08, 0.42, 0.84, 0.3, rx=0.12), _polygon([[0.24, 0.42], [0.36, 0.18], [0.7, 0.18], [0.82, 0.42]]), _ellipse(0.2, 0.72, 0.18, 0.18, fill_role="region_alt", stroke_width=0.018), _ellipse(0.62, 0.72, 0.18, 0.18, fill_role="region_alt", stroke_width=0.018)], confidence=0.87)],
        "street_lamp": [_variant("street_lamp", "classic", "Classic lamp", [_line(0.5, 0.98, 0.5, 0.14, stroke_width=0.034), _line(0.5, 0.16, 0.86, 0.16, stroke_width=0.026), _ellipse(0.76, 0.22, 0.14, 0.14, fill_role="accent_fill", stroke_role="accent", stroke_width=0.01)], confidence=0.84)],
        "table": [_variant("table", "desk", "Desk", [_rect(0.1, 0.18, 0.8, 0.16, rx=0.04), _line(0.2, 0.34, 0.2, 0.96, stroke_width=0.028), _line(0.8, 0.34, 0.8, 0.96, stroke_width=0.028)], confidence=0.83)],
        "chair": [_variant("chair", "straight", "Straight chair", [_rect(0.28, 0.12, 0.34, 0.2, rx=0.06), _line(0.3, 0.32, 0.3, 0.96, stroke_width=0.024), _line(0.62, 0.32, 0.62, 0.96, stroke_width=0.024), _line(0.62, 0.14, 0.8, 0.02, stroke_width=0.022), _line(0.8, 0.02, 0.8, 0.68, stroke_width=0.022)], confidence=0.83)],
        "dog": [_variant("dog", "standing", "Standing dog", [_ellipse(0.2, 0.34, 0.54, 0.3), _ellipse(0.66, 0.24, 0.22, 0.18), _line(0.26, 0.64, 0.22, 0.98, stroke_width=0.022), _line(0.44, 0.64, 0.42, 0.98, stroke_width=0.022), _line(0.62, 0.64, 0.62, 0.98, stroke_width=0.022), _line(0.16, 0.42, 0.04, 0.24, stroke_width=0.022)], confidence=0.82)],
        "desk_lamp": [_variant("desk_lamp", "task", "Task lamp", [_line(0.38, 0.98, 0.48, 0.62, stroke_width=0.028), _line(0.48, 0.62, 0.64, 0.34, stroke_width=0.024), _polygon([[0.58, 0.28], [0.8, 0.18], [0.72, 0.42]], fill_role="none", stroke_role="line", stroke_width=0.022), _ellipse(0.16, 0.9, 0.32, 0.08, fill_role="region_alt")], confidence=0.82)],
        "road": [_variant("road", "perspective", "Perspective road", [_polygon([[0.32, 0.18], [0.68, 0.18], [0.94, 0.96], [0.06, 0.96]], fill_role="region_alt"), _line(0.5, 0.26, 0.5, 0.94, stroke_role="accent", stroke_width=0.016, dash=[0.06])], confidence=0.82)],
        "cycle": [_variant("cycle", "loop", "Cycle loop", [_ellipse(0.16, 0.18, 0.68, 0.68, fill_role="none", stroke_role="line", stroke_width=0.028), _polygon([[0.62, 0.18], [0.88, 0.28], [0.7, 0.42]], fill_role="accent_fill", stroke_role="accent", stroke_width=0.01), _polygon([[0.22, 0.82], [0.1, 0.58], [0.34, 0.64]], fill_role="accent_fill", stroke_role="accent", stroke_width=0.01)], scene_type="process", confidence=0.84)],
        "flow_node": [_variant("flow_node", "stage_card", "Stage node", [_rect(0.12, 0.2, 0.76, 0.52, rx=0.16), _line(0.24, 0.34, 0.76, 0.34, stroke_role="accent", stroke_width=0.018), _line(0.24, 0.52, 0.62, 0.52, stroke_width=0.016), _line(0.78, 0.46, 0.9, 0.46, stroke_role="accent", stroke_width=0.016)], scene_type="process", confidence=0.82)],
        "energy_wave": [_variant("energy_wave", "arc", "Energy arc", [_polyline([[0.08, 0.68], [0.24, 0.48], [0.4, 0.62], [0.56, 0.34], [0.72, 0.48], [0.9, 0.2]], stroke_role="accent", stroke_width=0.032), _polyline([[0.12, 0.84], [0.3, 0.68], [0.46, 0.82], [0.62, 0.56], [0.78, 0.68]], stroke_width=0.018)], scene_type="process", confidence=0.8)],
        "vapor": [_variant("vapor", "steam", "Steam plume", [_polyline([[0.28, 0.92], [0.24, 0.72], [0.32, 0.54], [0.26, 0.34], [0.36, 0.16]], stroke_width=0.026), _polyline([[0.48, 0.92], [0.44, 0.68], [0.54, 0.5], [0.46, 0.28], [0.58, 0.08]], stroke_width=0.026), _polyline([[0.68, 0.92], [0.64, 0.72], [0.72, 0.56], [0.66, 0.36], [0.74, 0.18]], stroke_width=0.026)], scene_type="process", confidence=0.82)],
        "leaf": [_variant("leaf", "simple", "Leaf", [_polygon([[0.5, 0.04], [0.82, 0.48], [0.5, 0.96], [0.18, 0.48]]), _line(0.5, 0.1, 0.5, 0.9, stroke_role="accent", stroke_width=0.016), _line(0.5, 0.44, 0.72, 0.28, stroke_role="accent", stroke_width=0.012)], scene_type="process", confidence=0.82)],
        "raindrop": [_variant("raindrop", "drop", "Raindrop", [_polygon([[0.5, 0.02], [0.78, 0.42], [0.66, 0.82], [0.34, 0.82], [0.22, 0.42]])], scene_type="process", confidence=0.82)],
        "airplane": [_variant("airplane", "jet", "Jet", [_polyline([[0.08, 0.5], [0.52, 0.5], [0.72, 0.3], [0.88, 0.36], [0.7, 0.5], [0.88, 0.64], [0.72, 0.7], [0.52, 0.5]], stroke_width=0.028), _line(0.42, 0.5, 0.28, 0.24, stroke_width=0.02), _line(0.42, 0.5, 0.28, 0.76, stroke_width=0.02)], scene_type="process", confidence=0.82)],
        "cell": [_variant("cell", "nucleus", "Cell", [_ellipse(0.08, 0.12, 0.84, 0.76), _ellipse(0.36, 0.32, 0.28, 0.24, fill_role="region_alt"), _ellipse(0.26, 0.26, 0.12, 0.1, fill_role="none", stroke_role="accent", stroke_width=0.014)], scene_type="process", confidence=0.82)],
        "battery": [_variant("battery", "cell", "Battery cell", [_line(0.08, 0.5, 0.34, 0.5, stroke_width=0.022), _line(0.42, 0.18, 0.42, 0.82, stroke_width=0.026), _line(0.58, 0.28, 0.58, 0.72, stroke_width=0.02), _line(0.66, 0.5, 0.92, 0.5, stroke_width=0.022)], scene_type="schematic", confidence=0.84)],
        "led": [_variant("led", "symbol", "LED symbol", [_line(0.06, 0.5, 0.22, 0.5, stroke_width=0.022), _polygon([[0.24, 0.2], [0.24, 0.8], [0.62, 0.5]]), _line(0.68, 0.18, 0.68, 0.82, stroke_width=0.024), _line(0.68, 0.5, 0.94, 0.5, stroke_width=0.022), _line(0.66, 0.24, 0.86, 0.1, stroke_role="accent", stroke_width=0.016), _line(0.62, 0.46, 0.86, 0.26, stroke_role="accent", stroke_width=0.016)], scene_type="schematic", confidence=0.84)],
        "resistor": [_variant("resistor", "zigzag", "Resistor", [_line(0.04, 0.5, 0.18, 0.5, stroke_width=0.022), _polyline([[0.18, 0.5], [0.28, 0.24], [0.4, 0.76], [0.52, 0.24], [0.64, 0.76], [0.76, 0.24], [0.86, 0.5]], stroke_width=0.024), _line(0.86, 0.5, 0.98, 0.5, stroke_width=0.022)], scene_type="schematic", confidence=0.83)],
        "capacitor": [_variant("capacitor", "parallel", "Capacitor", [_line(0.08, 0.5, 0.36, 0.5, stroke_width=0.022), _line(0.42, 0.18, 0.42, 0.82, stroke_width=0.026), _line(0.58, 0.18, 0.58, 0.82, stroke_width=0.026), _line(0.64, 0.5, 0.92, 0.5, stroke_width=0.022)], scene_type="schematic", confidence=0.82)],
        "diode": [_variant("diode", "standard", "Diode", [_line(0.06, 0.5, 0.22, 0.5, stroke_width=0.022), _polygon([[0.24, 0.22], [0.24, 0.78], [0.6, 0.5]]), _line(0.64, 0.18, 0.64, 0.82, stroke_width=0.024), _line(0.66, 0.5, 0.94, 0.5, stroke_width=0.022)], scene_type="schematic", confidence=0.82)],
        "board": [_variant("board", "chip", "Board module", [_rect(0.12, 0.14, 0.76, 0.72, rx=0.06), _rect(0.24, 0.26, 0.28, 0.22, fill_role="none", stroke_role="accent", stroke_width=0.014), _rect(0.58, 0.28, 0.14, 0.18, fill_role="none", stroke_role="accent", stroke_width=0.014), _line(0.08, 0.22, 0.12, 0.22, stroke_width=0.016), _line(0.88, 0.28, 0.92, 0.28, stroke_width=0.016)], scene_type="schematic", confidence=0.8)],
        "module": [_variant("module", "io", "I/O module", [_rect(0.1, 0.18, 0.8, 0.62, rx=0.08), _line(0.22, 0.32, 0.78, 0.32, stroke_role="accent", stroke_width=0.016), _ellipse(0.14, 0.48, 0.08, 0.08, fill_role="accent_fill", stroke_role="accent", stroke_width=0.01), _ellipse(0.78, 0.48, 0.08, 0.08, fill_role="accent_fill", stroke_role="accent", stroke_width=0.01)], scene_type="schematic", confidence=0.8)],
        "branch": [_variant("branch", "split", "Split branch", [_line(0.12, 0.5, 0.46, 0.5, stroke_width=0.026), _line(0.46, 0.5, 0.78, 0.24, stroke_width=0.022), _line(0.46, 0.5, 0.78, 0.76, stroke_width=0.022), _ellipse(0.78, 0.18, 0.1, 0.1, fill_role="accent_fill", stroke_role="accent", stroke_width=0.01), _ellipse(0.78, 0.72, 0.1, 0.1, fill_role="accent_fill", stroke_role="accent", stroke_width=0.01)], scene_type="schematic", confidence=0.76)],
    }
    return library


def _coerce_variant(asset_key: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    item = _copy(raw)
    scene_type = CLASS_SCENE_MAP.get(asset_key, "scene")
    family = infer_sketch_family(asset_key, scene_type=scene_type)
    variant_id = str(item.get("id") or f"{asset_key}:external")
    stroke_payload = _normalize_stroke_payload(item.get("stroke_payload"))
    shape_recipe = _copy(item.get("shape_recipe") or {})
    if not stroke_payload and shape_recipe:
        stroke_payload = shape_recipe_to_strokes(shape_recipe)
    if not shape_recipe and stroke_payload:
        shape_recipe = strokes_to_shape_recipe(stroke_payload)
    render_representation = str(item.get("render_representation") or ("stroke_native" if stroke_payload else "shape_recipe"))
    stroke_variant_id = str(item.get("stroke_variant_id") or variant_id)
    return {
        "id": variant_id,
        "label": str(item.get("label") or variant_id),
        "asset_key": asset_key,
        "shape_recipe": shape_recipe,
        "source": str(item.get("source") or "trained_export"),
        "confidence": float(item.get("confidence", 0.8) or 0.8),
        "style_variants": list(item.get("style_variants") or [style["id"] for style in STYLE_VARIANTS]),
        "default_style": str(item.get("default_style") or "scribble_line"),
        "part_graph": _copy(item.get("part_graph") or default_part_graph(asset_key, scene_type=scene_type)),
        "region_masks": _copy(item.get("region_masks") or default_region_masks(asset_key, scene_type=scene_type)),
        "stroke_style_profile": _copy(item.get("stroke_style_profile") or build_stroke_style_profile(asset_key, scene_type=scene_type, style_variant=str(item.get("default_style") or "scribble_line"), sketch_family=family)),
        "stroke_payload": stroke_payload,
        "stroke_variant_id": stroke_variant_id,
        "stroke_payload_source": str(item.get("stroke_payload_source") or item.get("source") or "trained_export"),
        "stroke_render_profile": _copy(item.get("stroke_render_profile") or item.get("stroke_style_profile") or build_stroke_style_profile(asset_key, scene_type=scene_type, style_variant=str(item.get("default_style") or "scribble_line"), sketch_family=family)),
        "render_representation": render_representation,
        "readability_rank": int(item.get("readability_rank", default_readability_rank(asset_key, scene_type=scene_type)) or default_readability_rank(asset_key, scene_type=scene_type)),
        "sketch_family": str(item.get("sketch_family") or family),
    }


@lru_cache(maxsize=8)
def _load_exported_variant_library_cached(path_key: str) -> Dict[str, List[Dict[str, Any]]]:
    export_path = Path(path_key)
    if not export_path.exists():
        return {}
    try:
        payload = json.loads(export_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw_variants = payload.get("variants") if isinstance(payload, dict) else {}
    if not isinstance(raw_variants, dict):
        return {}
    library: Dict[str, List[Dict[str, Any]]] = {}
    for asset_key, variants in raw_variants.items():
        if not isinstance(variants, list):
            continue
        cleaned = [_coerce_variant(str(asset_key), item) for item in variants if isinstance(item, dict)]
        if cleaned:
            library[str(asset_key)] = cleaned
    return library


def load_exported_variant_library() -> Dict[str, List[Dict[str, Any]]]:
    return _load_exported_variant_library_cached(str(resolve_exported_variants_path().resolve()))


@lru_cache(maxsize=8)
def _load_exported_stroke_variant_library_cached(path_key: str) -> Dict[str, List[Dict[str, Any]]]:
    export_path = Path(path_key)
    if not export_path.exists():
        return {}
    try:
        payload = json.loads(export_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw_variants = payload.get("variants") if isinstance(payload, dict) else {}
    if not isinstance(raw_variants, dict):
        return {}
    library: Dict[str, List[Dict[str, Any]]] = {}
    for asset_key, variants in raw_variants.items():
        if not isinstance(variants, list):
            continue
        cleaned = [_coerce_variant(str(asset_key), item) for item in variants if isinstance(item, dict)]
        if cleaned:
            library[str(asset_key)] = cleaned
    return library


def load_exported_stroke_variant_library() -> Dict[str, List[Dict[str, Any]]]:
    return _load_exported_stroke_variant_library_cached(str(resolve_exported_stroke_variants_path().resolve()))


def _merge_variants(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    for item in items:
        variant_id = str(item.get("id") or "")
        if variant_id:
            unique[variant_id] = _copy(item)
    return list(unique.values())


def list_object_shape_variants(asset_key: str, concept: str = "", scene_type: str = "scene") -> List[Dict[str, Any]]:
    key = str(asset_key or "").strip() or FAMILY_FALLBACKS.get(scene_type, "blob")
    return _merge_variants([
        *load_exported_variant_library().get(key, []),
        *_builtin_variant_library().get(key, []),
        _fallback_variant(key, concept, scene_type),
        *load_exported_stroke_variant_library().get(key, []),
    ])


def _stable_index(key: str, total: int) -> int:
    if total <= 0:
        return 0
    return sum((index + 1) * ord(char) for index, char in enumerate(key)) % total


def _normalize_style_variant_id(style_variant: str | None) -> str:
    raw = str(style_variant or "").strip().lower()
    if raw in {"minimal", "clean", "clean_line"}:
        return "clean_line"
    if raw in {"line_art", "scribble", "scribble_line"}:
        return "scribble_line"
    return raw or ""


def _resolve_default_style_variant(asset_key: str, scene_type: str, style_variant: str | None) -> str:
    explicit = _normalize_style_variant_id(style_variant)
    if explicit:
        return explicit
    env_default = _normalize_style_variant_id(os.getenv("TMCRA_GENERATION_STYLE_DEFAULT", ""))
    if env_default:
        return env_default
    if asset_key in SCENE_OBJECT_CLASSES or scene_type == "scene":
        return "clean_line"
    return "scribble_line"


def _resolve_variant_selection_policy() -> str:
    raw = str(os.getenv("TMCRA_OBJECT_VARIANT_POLICY", "current_generation_v1") or "").strip().lower()
    if raw in {"stable", "legacy_stable", "legacy"}:
        return "legacy_stable"
    return "current_generation_v1"


def _source_selection_bonus(
    asset_key: str,
    *,
    scene_type: str,
    source: str,
    render_representation: str,
) -> float:
    if scene_type == "scene" and asset_key in SCENE_OBJECT_CLASSES:
        if source == "demo_library":
            return 1.0
        if source == "rule_prototype":
            return 0.82
        if source.startswith("trained_export") and render_representation == "stroke_native":
            return 0.35
    if source.startswith("trained_export"):
        return 0.78
    if source == "demo_library":
        return 0.58
    if source == "rule_prototype":
        return 0.42
    return 0.0


def _variant_priority_tuple(
    asset_key: str,
    variant: Dict[str, Any],
    *,
    concept: str,
    scene_type: str,
    preferred_style: str,
    stroke_seed: int | str | None,
) -> tuple[float, ...]:
    styles = {_normalize_style_variant_id(item) for item in (variant.get("style_variants") or [])}
    default_style = _normalize_style_variant_id(variant.get("default_style"))
    source = str(variant.get("source") or "")
    confidence = float(variant.get("confidence", 0.0) or 0.0)
    readability = float(variant.get("readability_rank", default_readability_rank(asset_key, scene_type=scene_type)) or 0.0)
    stroke_payload = _normalize_stroke_payload(variant.get("stroke_payload"))
    render_representation = str(variant.get("render_representation") or "")
    sketch_engine = _resolve_sketch_engine()
    style_hit = 1.0 if preferred_style and preferred_style in styles else 0.0
    style_default_hit = 1.0 if preferred_style and default_style == preferred_style else 0.0
    clean_bonus = 1.0 if "clean_line" in styles else 0.0
    source_bonus = _source_selection_bonus(
        asset_key,
        scene_type=scene_type,
        source=source,
        render_representation=render_representation,
    )
    native_stroke_bonus = 1.0 if stroke_payload and render_representation == "stroke_native" else 0.0
    natural_engine_bonus = 1.0 if stroke_payload and sketch_engine != "mechanical_v1" else 0.0
    scene_clean_bonus = 1.0 if preferred_style == "clean_line" and (asset_key in SCENE_OBJECT_CLASSES or scene_type == "scene") else 0.0
    tie_break = -float(_stable_index(f"{asset_key}|{concept}|{variant.get('id') or ''}|{stroke_seed or ''}", 1_000_000))
    return (
        style_hit,
        confidence,
        source_bonus,
        readability,
        style_default_hit,
        scene_clean_bonus * clean_bonus,
        clean_bonus,
        natural_engine_bonus,
        native_stroke_bonus,
        tie_break,
    )


def resolve_object_shape(asset_key: str, *, concept: str = "", scene_type: str = "scene", preferred_variant_id: str | None = None, style_variant: str | None = None, stroke_seed: int | str | None = None) -> Dict[str, Any]:
    variants = list_object_shape_variants(asset_key, concept, scene_type)
    variant_map = {item["id"]: item for item in variants}
    preferred_style = _resolve_default_style_variant(asset_key, scene_type, style_variant)
    explicit_variant = variant_map.get(str(preferred_variant_id or "").strip())
    if explicit_variant:
        chosen = explicit_variant
        selection_policy = "preferred_variant_id"
    elif _resolve_variant_selection_policy() == "legacy_stable":
        chosen = variants[_stable_index(f"{asset_key}|{concept}|{stroke_seed or ''}", len(variants))]
        selection_policy = "legacy_stable"
    else:
        chosen = max(
            variants,
            key=lambda item: _variant_priority_tuple(
                asset_key,
                item,
                concept=concept,
                scene_type=scene_type,
                preferred_style=preferred_style,
                stroke_seed=stroke_seed,
            ),
        )
        selection_policy = "current_generation_v1"
    source = str(chosen.get("source") or "rule_prototype")
    sketch_engine = _resolve_sketch_engine()
    stroke_payload = _normalize_stroke_payload(chosen.get("stroke_payload"))
    shape_recipe = _copy(chosen.get("shape_recipe") or {})
    if not stroke_payload and shape_recipe:
        stroke_payload = shape_recipe_to_strokes(shape_recipe)
    render_representation = "shape_recipe" if sketch_engine == "mechanical_v1" else str(chosen.get("render_representation") or ("stroke_native" if stroke_payload else "shape_recipe"))
    if render_representation == "stroke_native" and not stroke_payload:
        render_representation = "shape_recipe"
    sketch_backend = "trained" if source.startswith("trained_export") else "hybrid" if asset_key in TRAINED_DEMO_CLASSES else "rule"
    return {
        "sketch_backend": sketch_backend,
        "shape_variant_id": chosen["id"],
        "shape_recipe": shape_recipe,
        "shape_recipe_source": source,
        "shape_confidence": float(chosen.get("confidence", 0.78) or 0.78),
        "stroke_seed": str(stroke_seed or ""),
        "style_variant": preferred_style or str(chosen.get("default_style") or "scribble_line"),
        "part_graph": _copy(chosen.get("part_graph") or []),
        "region_masks": _copy(chosen.get("region_masks") or []),
        "stroke_style_profile": _copy(chosen.get("stroke_style_profile") or {}),
        "render_representation": render_representation,
        "stroke_variant_id": str(chosen.get("stroke_variant_id") or chosen["id"]),
        "stroke_payload": stroke_payload,
        "stroke_payload_source": str(chosen.get("stroke_payload_source") or source),
        "stroke_render_profile": _copy(chosen.get("stroke_render_profile") or chosen.get("stroke_style_profile") or {}),
        "readability_rank": int(chosen.get("readability_rank", default_readability_rank(asset_key, scene_type=scene_type)) or default_readability_rank(asset_key, scene_type=scene_type)),
        "sketch_family": str(chosen.get("sketch_family") or infer_sketch_family(asset_key, scene_type=scene_type)),
        "variant_selection_policy": selection_policy,
        "available_shape_variants": [{"id": item["id"], "label": item.get("label", item["id"]), "asset_key": item.get("asset_key", asset_key), "source": item.get("source", "rule_prototype"), "confidence": float(item.get("confidence", 0.75) or 0.75), "style_variants": list(item.get("style_variants") or [style["id"] for style in STYLE_VARIANTS]), "shape_recipe": _copy(item.get("shape_recipe") or {}), "part_graph": _copy(item.get("part_graph") or []), "region_masks": _copy(item.get("region_masks") or []), "stroke_style_profile": _copy(item.get("stroke_style_profile") or {}), "readability_rank": item.get("readability_rank"), "sketch_family": item.get("sketch_family"), "render_representation": item.get("render_representation"), "stroke_variant_id": item.get("stroke_variant_id"), "stroke_payload_source": item.get("stroke_payload_source"), "stroke_render_profile": _copy(item.get("stroke_render_profile") or {}), "stroke_payload": _normalize_stroke_payload(item.get("stroke_payload"))} for item in variants],
    }


def summarize_scene_backend(scene_spec: Dict[str, Any] | None) -> Dict[str, Any]:
    objects = list((scene_spec or {}).get("object_instances") or [])
    return {
        "mode": DEMO_BACKEND_MODE,
        "object_count": len(objects),
        "backend_counts": dict(Counter(str(item.get("sketch_backend") or "rule") for item in objects)),
        "source_counts": dict(Counter(str(item.get("shape_recipe_source") or "rule_prototype") for item in objects)),
        "render_representation_counts": dict(Counter(str(item.get("render_representation") or "shape_recipe") for item in objects)),
        "family_counts": dict(Counter(str(item.get("sketch_family") or "") for item in objects if item.get("sketch_family"))),
        "trained_class_count": len(TRAINED_DEMO_CLASSES),
        "export_library_loaded": bool(load_exported_variant_library()),
        "stroke_export_library_loaded": bool(load_exported_stroke_variant_library()),
    }


def scene_shape_variant_payload(scene_spec: Dict[str, Any] | None) -> Dict[str, Any]:
    objects = list((scene_spec or {}).get("object_instances") or [])
    return {
        "available_shape_variants": {str(item.get("id")): [{"id": variant.get("id"), "label": variant.get("label"), "source": variant.get("source"), "confidence": variant.get("confidence"), "style_variants": variant.get("style_variants"), "readability_rank": variant.get("readability_rank"), "sketch_family": variant.get("sketch_family")} for variant in list(item.get("available_shape_variants") or [])] for item in objects if item.get("id")},
        "shape_variant_id": {str(item.get("id")): str(item.get("shape_variant_id") or "") for item in objects if item.get("id")},
        "shape_recipe_source": {str(item.get("id")): str(item.get("shape_recipe_source") or "") for item in objects if item.get("id")},
        "render_representation": {str(item.get("id")): str(item.get("render_representation") or "shape_recipe") for item in objects if item.get("id")},
        "stroke_variant_id": {str(item.get("id")): str(item.get("stroke_variant_id") or "") for item in objects if item.get("id")},
        "stroke_payload_source": {str(item.get("id")): str(item.get("stroke_payload_source") or "") for item in objects if item.get("id")},
        "region_masks": {str(item.get("id")): _copy(item.get("region_masks") or []) for item in objects if item.get("id")},
    }


def _resample_points(points: Sequence[Sequence[float]], target_points: int) -> List[List[float]]:
    cleaned = [[float(px), float(py)] for px, py in points]
    if not cleaned:
        return []
    if len(cleaned) == 1:
        return cleaned * max(1, target_points)
    distances = [0.0]
    for index in range(1, len(cleaned)):
        distances.append(distances[-1] + math.dist(cleaned[index - 1], cleaned[index]))
    total = distances[-1]
    if total <= 1e-6:
        return [cleaned[0] for _ in range(max(1, target_points))]
    sampled: List[List[float]] = []
    for step in range(max(1, target_points)):
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


def _shape_part_to_points(part: Dict[str, Any], sample_points: int = 24) -> List[List[float]]:
    kind = str(part.get("kind") or "").lower()
    if kind == "line":
        return [[float(part.get("x1", 0.0)), float(part.get("y1", 0.0))], [float(part.get("x2", 0.0)), float(part.get("y2", 0.0))]]
    if kind in {"polyline", "polygon"}:
        points = [[float(px), float(py)] for px, py in part.get("points") or []]
        return points + ([points[0]] if kind == "polygon" and points else [])
    if kind == "rect":
        x, y, w, h = float(part.get("x", 0.0)), float(part.get("y", 0.0)), float(part.get("w", 0.0)), float(part.get("h", 0.0))
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]
    if kind == "ellipse":
        x, y, w, h = float(part.get("x", 0.0)), float(part.get("y", 0.0)), float(part.get("w", 0.0)), float(part.get("h", 0.0))
        cx, cy, rx, ry = x + w / 2.0, y + h / 2.0, w / 2.0, h / 2.0
        return [[cx + math.cos(2 * math.pi * step / sample_points) * rx, cy + math.sin(2 * math.pi * step / sample_points) * ry] for step in range(sample_points + 1)]
    return []


def shape_recipe_to_strokes(shape_recipe: Dict[str, Any], *, max_strokes: int = 8, points_per_stroke: int = 32) -> List[List[List[float]]]:
    strokes: List[List[List[float]]] = []
    for part in list((shape_recipe or {}).get("parts") or []):
        points = _shape_part_to_points(part, sample_points=max(12, points_per_stroke))
        if points:
            strokes.append(_resample_points(points, points_per_stroke))
        if len(strokes) >= max_strokes:
            break
    return strokes


def strokes_to_shape_recipe(strokes: Sequence[Sequence[Sequence[float]]], *, stroke_width: float = 0.02, stroke_role: str = "line") -> Dict[str, Any]:
    return _shape([{"kind": "polyline", "points": [[max(0.0, min(1.0, float(px))), max(0.0, min(1.0, float(py)))] for px, py in stroke], "stroke_role": stroke_role, "stroke_width": stroke_width, "opacity": 1.0} for stroke in strokes if len(stroke) >= 2])


def _load_seed_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def bootstrap_training_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in [*_load_seed_rows(PUBLIC_SEED_PATH), *_load_seed_rows(MANUAL_SEED_PATH)]:
        key = (str(row.get("class_id") or ""), str(row.get("style_id") or ""), str(row.get("variant_id") or "seed"))
        if all(key) and key not in seen:
            rows.append(_copy(row))
            seen.add(key)
    for asset_key in TRAINED_DEMO_CLASSES:
        scene_type = CLASS_SCENE_MAP.get(asset_key, "scene")
        for variant in list_object_shape_variants(asset_key, asset_key, scene_type):
            if str(variant.get("source") or "") == "rule_prototype":
                continue
            for style in STYLE_VARIANTS:
                key = (asset_key, style["id"], str(variant.get("id") or ""))
                if key in seen:
                    continue
                rows.append({"class_id": asset_key, "scene_type": scene_type, "style_id": style["id"], "variant_id": variant["id"], "strokes": shape_recipe_to_strokes(variant.get("shape_recipe") or {}), "shape_recipe": _copy(variant.get("shape_recipe") or {}), "region_masks": _copy(variant.get("region_masks") or []), "part_graph": _copy(variant.get("part_graph") or []), "stroke_style_profile": _copy(variant.get("stroke_style_profile") or {}), "sketch_family": str(variant.get("sketch_family") or infer_sketch_family(asset_key, scene_type=scene_type)), "readability_rank": int(variant.get("readability_rank", default_readability_rank(asset_key, scene_type=scene_type)) or default_readability_rank(asset_key, scene_type=scene_type)), "source": str(variant.get("source") or "demo_library")})
                seen.add(key)
    return rows
