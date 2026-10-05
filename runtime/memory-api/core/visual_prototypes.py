from __future__ import annotations

import json
from typing import Any, Dict, List


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    rx: float = 0.0,
    fill_role: str = "fill",
    stroke_role: str = "line",
    stroke_width: float = 0.02,
    opacity: float = 1.0,
) -> Dict[str, Any]:
    return {
        "kind": "rect",
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "rx": rx,
        "fill_role": fill_role,
        "stroke_role": stroke_role,
        "stroke_width": stroke_width,
        "opacity": opacity,
    }


def _ellipse(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill_role: str = "fill",
    stroke_role: str = "line",
    stroke_width: float = 0.02,
    opacity: float = 1.0,
) -> Dict[str, Any]:
    return {
        "kind": "ellipse",
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "fill_role": fill_role,
        "stroke_role": stroke_role,
        "stroke_width": stroke_width,
        "opacity": opacity,
    }


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke_role: str = "line",
    stroke_width: float = 0.02,
    opacity: float = 1.0,
    dash: List[float] | None = None,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "kind": "line",
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "stroke_role": stroke_role,
        "stroke_width": stroke_width,
        "opacity": opacity,
    }
    if dash:
        item["dash"] = dash
    return item


def _polygon(
    points: List[List[float]],
    *,
    fill_role: str = "fill",
    stroke_role: str = "line",
    stroke_width: float = 0.02,
    opacity: float = 1.0,
) -> Dict[str, Any]:
    return {
        "kind": "polygon",
        "points": points,
        "fill_role": fill_role,
        "stroke_role": stroke_role,
        "stroke_width": stroke_width,
        "opacity": opacity,
    }


def _polyline(
    points: List[List[float]],
    *,
    stroke_role: str = "line",
    stroke_width: float = 0.02,
    opacity: float = 1.0,
) -> Dict[str, Any]:
    return {
        "kind": "polyline",
        "points": points,
        "stroke_role": stroke_role,
        "stroke_width": stroke_width,
        "opacity": opacity,
    }


def _prototype(
    prototype_id: str,
    visual_family: str,
    parts: List[Dict[str, Any]],
    *,
    style_variant: str = "scribble_line",
    part_slots: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    return {
        "prototype_id": prototype_id,
        "visual_family": visual_family,
        "style_variant": style_variant,
        "part_slots": part_slots or [],
        "shape_recipe": {
            "version": 1,
            "viewbox": [0, 0, 1, 1],
            "parts": parts,
        },
    }


PROTOTYPE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "generic_object": _prototype(
        "generic_object",
        "scene_object",
        [
            _ellipse(0.08, 0.14, 0.84, 0.68, fill_role="fill", stroke_role="line"),
            _line(0.2, 0.78, 0.8, 0.24, stroke_role="accent", stroke_width=0.018),
        ],
        part_slots=[{"name": "center", "x": 0.5, "y": 0.5}],
    ),
    "generic_panel": _prototype(
        "generic_panel",
        "process_motif",
        [
            _rect(0.08, 0.14, 0.84, 0.7, rx=0.12, fill_role="fill", stroke_role="line"),
            _line(0.18, 0.28, 0.82, 0.28, stroke_role="accent", stroke_width=0.018),
            _line(0.22, 0.48, 0.74, 0.48, stroke_role="line", stroke_width=0.016),
            _line(0.22, 0.62, 0.62, 0.62, stroke_role="line", stroke_width=0.016),
        ],
        part_slots=[{"name": "center", "x": 0.5, "y": 0.5}],
    ),
    "generic_circle": _prototype(
        "generic_circle",
        "scene_object",
        [
            _ellipse(0.1, 0.1, 0.8, 0.8, fill_role="fill", stroke_role="line"),
            _line(0.3, 0.7, 0.7, 0.3, stroke_role="accent", stroke_width=0.02),
        ],
    ),
    "blob": _prototype(
        "blob",
        "scene_object",
        [
            _ellipse(0.1, 0.2, 0.8, 0.58, fill_role="fill", stroke_role="line"),
            _ellipse(0.22, 0.1, 0.22, 0.18, fill_role="fill", stroke_role="line"),
            _ellipse(0.58, 0.08, 0.2, 0.2, fill_role="fill", stroke_role="line"),
        ],
    ),
    "tower": _prototype(
        "tower",
        "scene_object",
        [
            _rect(0.22, 0.06, 0.56, 0.88, rx=0.04, fill_role="fill", stroke_role="line"),
            _rect(0.34, 0.22, 0.1, 0.1, fill_role="none", stroke_role="accent", stroke_width=0.014),
            _rect(0.56, 0.22, 0.1, 0.1, fill_role="none", stroke_role="accent", stroke_width=0.014),
            _rect(0.34, 0.42, 0.1, 0.1, fill_role="none", stroke_role="accent", stroke_width=0.014),
            _rect(0.56, 0.42, 0.1, 0.1, fill_role="none", stroke_role="accent", stroke_width=0.014),
            _rect(0.45, 0.72, 0.1, 0.22, fill_role="none", stroke_role="accent", stroke_width=0.014),
        ],
        part_slots=[
            {"name": "facade", "x": 0.5, "y": 0.45},
            {"name": "roof", "x": 0.5, "y": 0.08},
            {"name": "ground", "x": 0.5, "y": 0.94},
        ],
    ),
    "capsule": _prototype(
        "capsule",
        "process_motif",
        [
            _ellipse(0.12, 0.18, 0.76, 0.56, fill_role="fill", stroke_role="line"),
            _line(0.24, 0.46, 0.76, 0.46, stroke_role="accent", stroke_width=0.018),
        ],
    ),
    "branch": _prototype(
        "branch",
        "process_motif",
        [
            _line(0.18, 0.78, 0.48, 0.18, stroke_role="line", stroke_width=0.03),
            _line(0.46, 0.28, 0.82, 0.16, stroke_role="line", stroke_width=0.025),
            _ellipse(0.5, 0.06, 0.2, 0.18, fill_role="fill", stroke_role="accent", stroke_width=0.014),
            _ellipse(0.12, 0.62, 0.22, 0.18, fill_role="fill", stroke_role="accent", stroke_width=0.014),
            _ellipse(0.68, 0.1, 0.18, 0.16, fill_role="fill", stroke_role="accent", stroke_width=0.014),
        ],
    ),
    "module": _prototype(
        "module",
        "schematic_symbol",
        [
            _rect(0.08, 0.16, 0.84, 0.68, rx=0.08, fill_role="fill", stroke_role="line"),
            _line(0.2, 0.3, 0.8, 0.3, stroke_role="accent", stroke_width=0.016),
            _ellipse(0.16, 0.46, 0.08, 0.08, fill_role="accent_fill", stroke_role="accent", stroke_width=0.01),
            _ellipse(0.76, 0.46, 0.08, 0.08, fill_role="accent_fill", stroke_role="accent", stroke_width=0.01),
        ],
    ),
    "switch": _prototype(
        "switch",
        "schematic_symbol",
        [
            _ellipse(0.14, 0.42, 0.1, 0.1, fill_role="accent_fill", stroke_role="accent", stroke_width=0.01),
            _ellipse(0.76, 0.42, 0.1, 0.1, fill_role="accent_fill", stroke_role="accent", stroke_width=0.01),
            _line(0.24, 0.5, 0.72, 0.26, stroke_role="line", stroke_width=0.03),
            _line(0.08, 0.5, 0.14, 0.5, stroke_role="line", stroke_width=0.022),
            _line(0.86, 0.5, 0.94, 0.5, stroke_role="line", stroke_width=0.022),
        ],
    ),
    "building": _prototype(
        "building",
        "scene_object",
        [
            _rect(0.1, 0.08, 0.8, 0.9, rx=0.03),
            _rect(0.2, 0.18, 0.1, 0.1, fill_role="none", stroke_role="accent", stroke_width=0.014),
            _rect(0.4, 0.18, 0.1, 0.1, fill_role="none", stroke_role="accent", stroke_width=0.014),
            _rect(0.6, 0.18, 0.1, 0.1, fill_role="none", stroke_role="accent", stroke_width=0.014),
            _rect(0.2, 0.38, 0.1, 0.1, fill_role="none", stroke_role="accent", stroke_width=0.014),
            _rect(0.4, 0.38, 0.1, 0.1, fill_role="none", stroke_role="accent", stroke_width=0.014),
            _rect(0.6, 0.38, 0.1, 0.1, fill_role="none", stroke_role="accent", stroke_width=0.014),
            _rect(0.44, 0.72, 0.12, 0.26, fill_role="none", stroke_role="accent", stroke_width=0.014),
        ],
        part_slots=[
            {"name": "facade", "x": 0.5, "y": 0.45},
            {"name": "roof", "x": 0.5, "y": 0.08},
            {"name": "ground", "x": 0.5, "y": 0.98},
        ],
    ),
    "house": _prototype(
        "house",
        "scene_object",
        [
            _polygon([[0.5, 0.02], [0.08, 0.34], [0.92, 0.34]]),
            _rect(0.16, 0.34, 0.68, 0.62, rx=0.03),
            _rect(0.42, 0.58, 0.16, 0.38, fill_role="none", stroke_role="accent", stroke_width=0.014),
            _rect(0.24, 0.46, 0.12, 0.12, fill_role="none", stroke_role="accent", stroke_width=0.014),
            _rect(0.64, 0.46, 0.12, 0.12, fill_role="none", stroke_role="accent", stroke_width=0.014),
        ],
        part_slots=[
            {"name": "facade", "x": 0.5, "y": 0.54},
            {"name": "roof", "x": 0.5, "y": 0.14},
            {"name": "ground", "x": 0.5, "y": 0.96},
        ],
    ),
    "window": _prototype(
        "window",
        "scene_object",
        [
            _rect(0.06, 0.06, 0.88, 0.88, rx=0.04),
            _line(0.5, 0.08, 0.5, 0.92, stroke_role="accent", stroke_width=0.02),
            _line(0.08, 0.5, 0.92, 0.5, stroke_role="accent", stroke_width=0.02),
        ],
        part_slots=[{"name": "center", "x": 0.5, "y": 0.5}],
    ),
    "door": _prototype(
        "door",
        "scene_object",
        [
            _rect(0.12, 0.04, 0.76, 0.92, rx=0.12),
            _ellipse(0.72, 0.5, 0.06, 0.06, fill_role="accent_fill", stroke_role="accent", stroke_width=0.01),
        ],
        part_slots=[{"name": "ground", "x": 0.5, "y": 0.96}],
    ),
    "tree": _prototype(
        "tree",
        "scene_object",
        [
            _rect(0.42, 0.56, 0.16, 0.4, fill_role="region_alt", stroke_role="line", stroke_width=0.018),
            _ellipse(0.24, 0.14, 0.52, 0.42),
            _ellipse(0.06, 0.3, 0.34, 0.28),
            _ellipse(0.58, 0.28, 0.28, 0.26),
        ],
        part_slots=[{"name": "ground", "x": 0.5, "y": 0.96}],
    ),
    "cloud": _prototype(
        "cloud",
        "scene_object",
        [
            _ellipse(0.04, 0.42, 0.34, 0.3),
            _ellipse(0.28, 0.2, 0.34, 0.36),
            _ellipse(0.52, 0.38, 0.3, 0.28),
            _line(0.16, 0.72, 0.76, 0.72, stroke_role="line", stroke_width=0.016),
        ],
        part_slots=[{"name": "sky", "x": 0.5, "y": 0.52}],
    ),
    "sun": _prototype(
        "sun",
        "scene_object",
        [
            _ellipse(0.24, 0.24, 0.52, 0.52),
            _line(0.5, 0.0, 0.5, 0.18, stroke_role="accent", stroke_width=0.018),
            _line(0.5, 0.82, 0.5, 1.0, stroke_role="accent", stroke_width=0.018),
            _line(0.0, 0.5, 0.18, 0.5, stroke_role="accent", stroke_width=0.018),
            _line(0.82, 0.5, 1.0, 0.5, stroke_role="accent", stroke_width=0.018),
            _line(0.16, 0.16, 0.28, 0.28, stroke_role="accent", stroke_width=0.018),
            _line(0.72, 0.72, 0.84, 0.84, stroke_role="accent", stroke_width=0.018),
            _line(0.16, 0.84, 0.28, 0.72, stroke_role="accent", stroke_width=0.018),
            _line(0.72, 0.28, 0.84, 0.16, stroke_role="accent", stroke_width=0.018),
        ],
        part_slots=[{"name": "sky", "x": 0.5, "y": 0.5}],
    ),
    "dog": _prototype(
        "dog",
        "scene_object",
        [
            _ellipse(0.18, 0.34, 0.52, 0.3),
            _ellipse(0.62, 0.22, 0.22, 0.2),
            _polygon([[0.68, 0.18], [0.62, 0.04], [0.74, 0.12]]),
            _polygon([[0.78, 0.2], [0.74, 0.06], [0.86, 0.16]]),
            _line(0.26, 0.62, 0.22, 0.96, stroke_role="line", stroke_width=0.02),
            _line(0.4, 0.62, 0.38, 0.96, stroke_role="line", stroke_width=0.02),
            _line(0.58, 0.62, 0.56, 0.96, stroke_role="line", stroke_width=0.02),
            _line(0.72, 0.56, 0.72, 0.94, stroke_role="line", stroke_width=0.02),
            _polyline([[0.18, 0.42], [0.08, 0.28], [0.02, 0.2]], stroke_role="accent", stroke_width=0.018),
        ],
        part_slots=[{"name": "ground", "x": 0.48, "y": 0.96}],
    ),
    "car": _prototype(
        "car",
        "scene_object",
        [
            _rect(0.08, 0.38, 0.84, 0.34, rx=0.12),
            _polygon([[0.24, 0.38], [0.36, 0.16], [0.7, 0.16], [0.82, 0.38]]),
            _ellipse(0.2, 0.72, 0.18, 0.18, fill_role="region_alt", stroke_role="line", stroke_width=0.018),
            _ellipse(0.62, 0.72, 0.18, 0.18, fill_role="region_alt", stroke_role="line", stroke_width=0.018),
            _line(0.38, 0.24, 0.64, 0.24, stroke_role="accent", stroke_width=0.016),
        ],
        part_slots=[{"name": "road", "x": 0.5, "y": 0.82}],
    ),
    "road": _prototype(
        "road",
        "scene_object",
        [
            _polygon([[0.16, 0.08], [0.84, 0.08], [1.0, 0.94], [0.0, 0.94]], fill_role="region_alt", stroke_role="line", stroke_width=0.016),
            _line(0.5, 0.18, 0.5, 0.92, stroke_role="accent", stroke_width=0.02, dash=[0.08, 0.06]),
        ],
        part_slots=[{"name": "center", "x": 0.5, "y": 0.55}],
    ),
    "person": _prototype(
        "person",
        "scene_object",
        [
            _ellipse(0.36, 0.04, 0.28, 0.24),
            _line(0.5, 0.28, 0.5, 0.68, stroke_role="line", stroke_width=0.024),
            _line(0.5, 0.38, 0.24, 0.52, stroke_role="line", stroke_width=0.02),
            _line(0.5, 0.38, 0.76, 0.52, stroke_role="line", stroke_width=0.02),
            _line(0.5, 0.68, 0.26, 0.98, stroke_role="line", stroke_width=0.02),
            _line(0.5, 0.68, 0.74, 0.98, stroke_role="line", stroke_width=0.02),
        ],
        part_slots=[{"name": "ground", "x": 0.5, "y": 0.96}],
    ),
    "street_lamp": _prototype(
        "street_lamp",
        "scene_object",
        [
            _line(0.5, 0.98, 0.5, 0.14, stroke_role="line", stroke_width=0.034),
            _line(0.5, 0.16, 0.86, 0.16, stroke_role="line", stroke_width=0.026),
            _ellipse(0.76, 0.22, 0.14, 0.14, fill_role="accent_fill", stroke_role="accent", stroke_width=0.01),
        ],
        part_slots=[{"name": "ground", "x": 0.5, "y": 0.98}],
    ),
    "desk_lamp": _prototype(
        "desk_lamp",
        "scene_object",
        [
            _ellipse(0.26, 0.82, 0.32, 0.12, fill_role="region_alt", stroke_role="line", stroke_width=0.018),
            _line(0.42, 0.82, 0.44, 0.54, stroke_role="line", stroke_width=0.024),
            _line(0.44, 0.54, 0.58, 0.36, stroke_role="line", stroke_width=0.022),
            _polygon([[0.58, 0.34], [0.84, 0.26], [0.72, 0.52], [0.52, 0.46]], fill_role="fill", stroke_role="line", stroke_width=0.018),
            _line(0.58, 0.36, 0.74, 0.58, stroke_role="accent", stroke_width=0.014),
        ],
        part_slots=[{"name": "base", "x": 0.42, "y": 0.88}],
    ),
    "table": _prototype(
        "table",
        "scene_object",
        [
            _rect(0.1, 0.18, 0.8, 0.16, rx=0.04),
            _line(0.2, 0.34, 0.2, 0.96, stroke_role="line", stroke_width=0.028),
            _line(0.8, 0.34, 0.8, 0.96, stroke_role="line", stroke_width=0.028),
        ],
        part_slots=[{"name": "center", "x": 0.5, "y": 0.24}],
    ),
    "chair": _prototype(
        "chair",
        "scene_object",
        [
            _rect(0.28, 0.12, 0.34, 0.2, rx=0.06),
            _line(0.3, 0.32, 0.3, 0.96, stroke_role="line", stroke_width=0.024),
            _line(0.62, 0.32, 0.62, 0.96, stroke_role="line", stroke_width=0.024),
            _line(0.62, 0.14, 0.8, 0.02, stroke_role="line", stroke_width=0.022),
            _line(0.8, 0.02, 0.8, 0.68, stroke_role="line", stroke_width=0.022),
        ],
    ),
    "battery": _prototype(
        "battery",
        "schematic_symbol",
        [
            _rect(0.14, 0.18, 0.62, 0.62, rx=0.08),
            _rect(0.76, 0.36, 0.1, 0.26, rx=0.02),
            _line(0.3, 0.5, 0.48, 0.5, stroke_role="accent", stroke_width=0.024),
            _line(0.39, 0.41, 0.39, 0.59, stroke_role="accent", stroke_width=0.024),
            _line(0.54, 0.5, 0.66, 0.5, stroke_role="accent", stroke_width=0.02),
        ],
    ),
    "led": _prototype(
        "led",
        "schematic_symbol",
        [
            _line(0.06, 0.5, 0.22, 0.5, stroke_role="line", stroke_width=0.022),
            _polygon([[0.24, 0.2], [0.24, 0.8], [0.62, 0.5]]),
            _line(0.68, 0.18, 0.68, 0.82, stroke_role="line", stroke_width=0.024),
            _line(0.68, 0.5, 0.94, 0.5, stroke_role="line", stroke_width=0.022),
            _line(0.66, 0.24, 0.86, 0.1, stroke_role="accent", stroke_width=0.016),
            _line(0.62, 0.46, 0.86, 0.26, stroke_role="accent", stroke_width=0.016),
        ],
    ),
    "resistor": _prototype(
        "resistor",
        "schematic_symbol",
        [
            _line(0.04, 0.5, 0.18, 0.5, stroke_role="line", stroke_width=0.022),
            _polyline([[0.18, 0.5], [0.28, 0.24], [0.4, 0.76], [0.52, 0.24], [0.64, 0.76], [0.76, 0.24], [0.86, 0.5]], stroke_role="line", stroke_width=0.024),
            _line(0.86, 0.5, 0.98, 0.5, stroke_role="line", stroke_width=0.022),
        ],
    ),
    "capacitor": _prototype(
        "capacitor",
        "schematic_symbol",
        [
            _line(0.08, 0.5, 0.36, 0.5, stroke_role="line", stroke_width=0.022),
            _line(0.42, 0.18, 0.42, 0.82, stroke_role="line", stroke_width=0.026),
            _line(0.58, 0.18, 0.58, 0.82, stroke_role="line", stroke_width=0.026),
            _line(0.64, 0.5, 0.92, 0.5, stroke_role="line", stroke_width=0.022),
        ],
    ),
    "diode": _prototype(
        "diode",
        "schematic_symbol",
        [
            _line(0.06, 0.5, 0.24, 0.5, stroke_role="line", stroke_width=0.022),
            _polygon([[0.24, 0.2], [0.24, 0.8], [0.62, 0.5]]),
            _line(0.68, 0.18, 0.68, 0.82, stroke_role="line", stroke_width=0.024),
            _line(0.68, 0.5, 0.94, 0.5, stroke_role="line", stroke_width=0.022),
        ],
    ),
    "board": _prototype(
        "board",
        "schematic_symbol",
        [
            _rect(0.06, 0.08, 0.88, 0.84, rx=0.08),
            _rect(0.16, 0.22, 0.2, 0.16, fill_role="none", stroke_role="accent", stroke_width=0.014, rx=0.03),
            _rect(0.56, 0.2, 0.2, 0.3, fill_role="none", stroke_role="accent", stroke_width=0.014, rx=0.03),
            _ellipse(0.18, 0.68, 0.05, 0.05, fill_role="accent_fill", stroke_role="accent", stroke_width=0.008),
            _ellipse(0.3, 0.68, 0.05, 0.05, fill_role="accent_fill", stroke_role="accent", stroke_width=0.008),
            _ellipse(0.42, 0.68, 0.05, 0.05, fill_role="accent_fill", stroke_role="accent", stroke_width=0.008),
        ],
    ),
    "airplane": _prototype(
        "airplane",
        "scene_object",
        [
            _line(0.12, 0.5, 0.92, 0.5, stroke_role="line", stroke_width=0.03),
            _polygon([[0.3, 0.5], [0.54, 0.18], [0.6, 0.18], [0.5, 0.5]]),
            _polygon([[0.42, 0.5], [0.58, 0.82], [0.64, 0.82], [0.56, 0.5]]),
            _polygon([[0.76, 0.5], [0.9, 0.3], [0.9, 0.7]]),
        ],
    ),
    "leaf": _prototype(
        "leaf",
        "process_motif",
        [
            _ellipse(0.14, 0.18, 0.72, 0.64),
            _line(0.22, 0.72, 0.8, 0.3, stroke_role="accent", stroke_width=0.018),
        ],
    ),
    "raindrop": _prototype(
        "raindrop",
        "process_motif",
        [
            _polygon([[0.5, 0.06], [0.8, 0.44], [0.74, 0.92], [0.26, 0.92], [0.2, 0.44]]),
        ],
    ),
    "cell": _prototype(
        "cell",
        "process_motif",
        [
            _ellipse(0.08, 0.16, 0.84, 0.68),
            _ellipse(0.34, 0.34, 0.32, 0.24, fill_role="region_alt", stroke_role="accent", stroke_width=0.014),
            _ellipse(0.46, 0.42, 0.08, 0.08, fill_role="accent_fill", stroke_role="accent", stroke_width=0.008),
        ],
    ),
    "cycle": _prototype(
        "cycle",
        "process_motif",
        [
            _ellipse(0.16, 0.18, 0.68, 0.68, fill_role="none", stroke_role="line", stroke_width=0.028),
            _polygon([[0.62, 0.18], [0.88, 0.28], [0.7, 0.42]], fill_role="accent_fill", stroke_role="accent", stroke_width=0.01),
            _polygon([[0.22, 0.82], [0.1, 0.58], [0.34, 0.64]], fill_role="accent_fill", stroke_role="accent", stroke_width=0.01),
        ],
    ),
    "flow_node": _prototype(
        "flow_node",
        "process_motif",
        [
            _rect(0.12, 0.2, 0.76, 0.52, rx=0.16, fill_role="fill", stroke_role="line"),
            _line(0.24, 0.34, 0.76, 0.34, stroke_role="accent", stroke_width=0.018),
            _line(0.24, 0.52, 0.62, 0.52, stroke_role="line", stroke_width=0.016),
            _line(0.78, 0.46, 0.9, 0.46, stroke_role="accent", stroke_width=0.016),
        ],
    ),
    "energy_wave": _prototype(
        "energy_wave",
        "process_motif",
        [
            _polyline([[0.08, 0.68], [0.24, 0.48], [0.4, 0.62], [0.56, 0.34], [0.72, 0.48], [0.9, 0.2]], stroke_role="accent", stroke_width=0.032),
            _polyline([[0.12, 0.84], [0.3, 0.68], [0.46, 0.82], [0.62, 0.56], [0.78, 0.68]], stroke_role="line", stroke_width=0.018),
        ],
    ),
    "vapor": _prototype(
        "vapor",
        "process_motif",
        [
            _polyline([[0.28, 0.92], [0.24, 0.72], [0.32, 0.54], [0.26, 0.34], [0.36, 0.16]], stroke_role="line", stroke_width=0.026),
            _polyline([[0.48, 0.92], [0.44, 0.68], [0.54, 0.5], [0.46, 0.28], [0.58, 0.08]], stroke_role="line", stroke_width=0.026),
            _polyline([[0.68, 0.92], [0.64, 0.72], [0.72, 0.56], [0.66, 0.36], [0.74, 0.18]], stroke_role="line", stroke_width=0.026),
        ],
    ),
}


FAMILY_FALLBACKS = {
    "scene": "blob",
    "process": "flow_node",
    "schematic": "module",
}


TOKEN_FALLBACKS = {
    "tower": ("塔", "楼", "高楼", "建筑", "烟囱", "柱"),
    "cycle": ("循环", "周期", "回路", "轮回", "往复"),
    "vapor": ("蒸发", "蒸汽", "水汽", "雾气", "气化"),
    "energy_wave": ("能量", "热量", "热", "光照", "光", "辐射", "波", "传播"),
    "flow_node": ("过程", "机制", "作用", "阶段", "步骤", "输入", "输出", "结果", "原因", "条件", "变化", "转化"),
    "branch": ("分支", "发散", "传播", "分化", "树枝", "网络"),
    "module": ("模块", "单元", "系统", "控制", "信号", "输入", "输出", "电源"),
    "switch": ("开关", "按钮", "按键", "拨动", "切换"),
    "capsule": ("囊泡", "胶囊", "包裹体"),
    "blob": ("物体", "东西", "目标", "主体"),
}


def fallback_prototype_id(concept: str, scene_type: str = "scene") -> str:
    text = str(concept or "").strip().lower()
    for prototype_id, tokens in TOKEN_FALLBACKS.items():
        if any(token.lower() in text for token in tokens):
            if scene_type == "schematic" and prototype_id in {"tower", "blob"}:
                continue
            if scene_type == "process" and prototype_id == "tower":
                continue
            return prototype_id
    return FAMILY_FALLBACKS.get(scene_type, "blob")


def resolve_visual_prototype(asset_key: str, concept: str = "", scene_type: str = "scene") -> Dict[str, Any]:
    prototype_id = asset_key if asset_key in PROTOTYPE_LIBRARY else fallback_prototype_id(concept, scene_type)
    resolved = _copy(PROTOTYPE_LIBRARY.get(prototype_id, PROTOTYPE_LIBRARY[FAMILY_FALLBACKS.get(scene_type, "blob")]))
    resolved["prototype_id"] = prototype_id
    return resolved
