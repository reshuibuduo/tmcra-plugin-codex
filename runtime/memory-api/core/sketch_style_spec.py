from __future__ import annotations

import json
from typing import Any, Dict, List


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


SKETCH_VIEW_MODE_ALIASES: Dict[str, str] = {
    "rough": "rough",
    "base": "rough",
    "structure": "structure",
    "structural": "structure",
    "annotated": "annotated",
    "annotation": "annotated",
    "region": "region",
    "region_overlay": "region",
}


DEFAULT_LAYOUT_FLAGS: Dict[str, Any] = {
    "sketch_view_mode": "structure",
    "annotation_level": "light",
    "region_edit_enabled": True,
}


REGION_ACTION_LABELS: Dict[str, str] = {
    "hide": "隐藏",
    "weaken": "弱化",
    "emphasize": "强调",
    "replace": "替换",
}


STROKE_STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "scribble_line": {
        "line_width": 1.0,
        "secondary_line_width": 0.76,
        "guide_opacity": 0.28,
        "roughness": 0.72,
        "fill_opacity": 0.1,
        "annotation_color": "#f97316",
        "region_color": "#38bdf8",
        "region_fill": "rgba(56, 189, 248, 0.14)",
    },
    "clean_line": {
        "line_width": 0.94,
        "secondary_line_width": 0.72,
        "guide_opacity": 0.2,
        "roughness": 0.18,
        "fill_opacity": 0.08,
        "annotation_color": "#eab308",
        "region_color": "#38bdf8",
        "region_fill": "rgba(56, 189, 248, 0.12)",
    },
    "blueprint": {
        "line_width": 0.92,
        "secondary_line_width": 0.72,
        "guide_opacity": 0.24,
        "roughness": 0.12,
        "fill_opacity": 0.06,
        "annotation_color": "#fb7185",
        "region_color": "#7dd3fc",
        "region_fill": "rgba(125, 211, 252, 0.12)",
    },
    "wireframe": {
        "line_width": 0.88,
        "secondary_line_width": 0.68,
        "guide_opacity": 0.18,
        "roughness": 0.06,
        "fill_opacity": 0.03,
        "annotation_color": "#0ea5e9",
        "region_color": "#94a3b8",
        "region_fill": "rgba(148, 163, 184, 0.08)",
    },
}


def normalize_view_mode(value: str | None) -> str:
    token = str(value or "").strip().lower()
    return SKETCH_VIEW_MODE_ALIASES.get(token, DEFAULT_LAYOUT_FLAGS["sketch_view_mode"])


def normalize_annotation_level(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in {"off", "light", "edit"}:
        return token
    return DEFAULT_LAYOUT_FLAGS["annotation_level"]


def normalize_style_variant(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in STROKE_STYLE_PRESETS:
        return token
    if token == "line_art":
        return "scribble_line"
    if token == "minimal":
        return "clean_line"
    return "scribble_line"


def normalize_region_action(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in REGION_ACTION_LABELS:
        return token
    return ""


def action_label(value: str | None) -> str:
    return REGION_ACTION_LABELS.get(normalize_region_action(value), "")


def build_stroke_style_profile(
    asset_key: str,
    *,
    scene_type: str = "scene",
    style_variant: str = "scribble_line",
    sketch_family: str = "",
) -> Dict[str, Any]:
    normalized_style = normalize_style_variant(style_variant)
    preset = _copy(STROKE_STYLE_PRESETS.get(normalized_style, STROKE_STYLE_PRESETS["scribble_line"]))
    family = sketch_family or infer_sketch_family(asset_key, scene_type=scene_type)
    if family == "scene_subject":
        preset["line_width"] = round(float(preset["line_width"]) * 1.12, 3)
        preset["fill_opacity"] = round(float(preset["fill_opacity"]) + 0.03, 3)
    elif family == "process_motif":
        preset["secondary_line_width"] = round(float(preset["secondary_line_width"]) * 0.92, 3)
    elif family == "schematic_symbol":
        preset["roughness"] = round(float(preset["roughness"]) * 0.4, 3)
        preset["fill_opacity"] = round(float(preset["fill_opacity"]) * 0.5, 3)
    preset["style_variant"] = normalized_style
    preset["scene_type"] = scene_type
    preset["sketch_family"] = family
    return preset


def infer_sketch_family(asset_key: str, *, scene_type: str = "scene") -> str:
    key = str(asset_key or "").strip().lower()
    if key in {"building", "house", "tree", "person", "car", "dog", "street_lamp", "table", "chair", "desk_lamp"}:
        return "scene_subject" if key in {"building", "house", "person", "car", "dog"} else "scene_support"
    if key in {"window", "door"}:
        return "scene_detail"
    if key in {"cloud", "sun", "road"}:
        return "scene_environment"
    if key in {"cycle", "flow_node", "energy_wave", "vapor", "raindrop", "leaf", "airplane", "cell"}:
        return "process_motif"
    if key in {"battery", "led", "resistor", "capacitor", "diode", "board", "module", "branch"}:
        return "schematic_symbol"
    if scene_type == "process":
        return "process_motif"
    if scene_type == "schematic":
        return "schematic_symbol"
    return "scene_support"


def default_readability_rank(asset_key: str, *, scene_type: str = "scene") -> int:
    key = str(asset_key or "").strip().lower()
    if key in {"building", "house", "tree", "person", "car", "cloud", "sun", "street_lamp", "battery", "led", "resistor"}:
        return 5
    if key in {"window", "door", "road", "leaf", "airplane", "cycle", "flow_node", "vapor", "energy_wave"}:
        return 4
    if key in {"module", "branch", "cell", "board", "capacitor", "diode"}:
        return 3
    if scene_type == "scene":
        return 2
    return 3


def _rect_region(region_id: str, label: str, x: float, y: float, width: float, height: float, **extra: Any) -> Dict[str, Any]:
    payload = {
        "id": region_id,
        "label": label,
        "shape": "rect",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "editable": True,
    }
    payload.update(extra)
    return payload


def _ellipse_region(region_id: str, label: str, x: float, y: float, width: float, height: float, **extra: Any) -> Dict[str, Any]:
    payload = {
        "id": region_id,
        "label": label,
        "shape": "ellipse",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "editable": True,
    }
    payload.update(extra)
    return payload


def default_region_masks(asset_key: str, *, scene_type: str = "scene") -> List[Dict[str, Any]]:
    key = str(asset_key or "").strip().lower()
    if key in {"building", "house"}:
        return [
            _rect_region("roofline", "屋顶区域", 0.08, 0.02, 0.84, 0.22, actions=["emphasize", "weaken"]),
            _rect_region("window_row", "窗户区域", 0.16, 0.16, 0.68, 0.42, actions=["replace", "weaken", "hide"]),
            _rect_region("door_zone", "门区", 0.38, 0.54, 0.24, 0.42, actions=["replace", "hide"]),
            _rect_region("facade", "立面", 0.1, 0.12, 0.8, 0.84, actions=["emphasize", "weaken"]),
        ]
    if key == "person":
        return [
            _ellipse_region("head", "头部", 0.32, 0.02, 0.36, 0.26, actions=["emphasize", "weaken"]),
            _ellipse_region("face", "脸部", 0.36, 0.08, 0.28, 0.18, actions=["replace", "weaken"]),
            _rect_region("beard", "胡子区域", 0.38, 0.16, 0.24, 0.12, actions=["hide", "replace", "weaken"]),
            _rect_region("torso", "上身", 0.28, 0.26, 0.44, 0.34, actions=["emphasize", "weaken"]),
            _rect_region("legs", "下身", 0.24, 0.58, 0.52, 0.4, actions=["emphasize", "weaken"]),
        ]
    if key == "tree":
        return [
            _ellipse_region("canopy", "树冠", 0.08, 0.04, 0.84, 0.58, actions=["hide", "replace", "weaken", "emphasize"]),
            _rect_region("trunk", "树干", 0.4, 0.54, 0.2, 0.42, actions=["weaken", "emphasize"]),
        ]
    if key == "car":
        return [
            _rect_region("cabin", "驾驶舱", 0.22, 0.18, 0.58, 0.28, actions=["replace", "weaken"]),
            _rect_region("body", "车身", 0.08, 0.38, 0.84, 0.34, actions=["emphasize", "weaken"]),
            _ellipse_region("front_wheel", "前轮", 0.14, 0.68, 0.22, 0.22, actions=["hide", "replace"]),
            _ellipse_region("rear_wheel", "后轮", 0.6, 0.68, 0.22, 0.22, actions=["hide", "replace"]),
        ]
    if key == "street_lamp":
        return [
            _rect_region("pole", "灯杆", 0.42, 0.18, 0.16, 0.78, actions=["weaken", "emphasize"]),
            _ellipse_region("lamp_head", "灯头", 0.68, 0.18, 0.18, 0.18, actions=["replace", "hide"]),
            _ellipse_region("light_cone", "光照范围", 0.5, 0.26, 0.38, 0.38, actions=["weaken", "emphasize"]),
        ]
    if key == "cloud":
        return [_ellipse_region("cloud_mass", "云团", 0.04, 0.18, 0.88, 0.58, actions=["hide", "replace", "weaken"])]
    if key == "sun":
        return [
            _ellipse_region("sun_core", "太阳主体", 0.24, 0.24, 0.52, 0.52, actions=["hide", "replace", "weaken"]),
            _rect_region("rays", "光线", 0.04, 0.04, 0.92, 0.92, actions=["weaken", "emphasize"]),
        ]
    if key in {"cycle", "flow_node", "energy_wave", "vapor"}:
        return [
            _rect_region("core_flow", "主流程", 0.12, 0.16, 0.76, 0.56, actions=["replace", "weaken", "emphasize"]),
            _rect_region("markers", "辅助标记", 0.08, 0.04, 0.84, 0.2, actions=["hide", "replace"]),
        ]
    if key in {"battery", "led", "resistor", "capacitor", "diode", "board", "module"}:
        return [
            _rect_region("body", "主体区域", 0.12, 0.18, 0.76, 0.56, actions=["replace", "weaken"]),
            _rect_region("terminals", "连接端", 0.02, 0.36, 0.96, 0.28, actions=["hide", "emphasize"]),
        ]
    return [_rect_region("core", "主体区域", 0.14, 0.14, 0.72, 0.72, actions=["replace", "weaken", "hide", "emphasize"])]


def default_part_graph(asset_key: str, *, scene_type: str = "scene") -> List[Dict[str, Any]]:
    key = str(asset_key or "").strip().lower()
    graphs: Dict[str, List[Dict[str, Any]]] = {
        "building": [
            {"id": "roofline", "label": "屋顶", "kind": "part", "region_id": "roofline"},
            {"id": "window_row", "label": "窗户组", "kind": "part", "region_id": "window_row"},
            {"id": "door_zone", "label": "门区", "kind": "part", "region_id": "door_zone"},
        ],
        "house": [
            {"id": "roofline", "label": "屋顶", "kind": "part", "region_id": "roofline"},
            {"id": "window_row", "label": "窗户组", "kind": "part", "region_id": "window_row"},
            {"id": "door_zone", "label": "门区", "kind": "part", "region_id": "door_zone"},
        ],
        "person": [
            {"id": "head", "label": "头部", "kind": "part", "region_id": "head"},
            {"id": "face", "label": "脸部", "kind": "part", "region_id": "face"},
            {"id": "beard", "label": "胡子", "kind": "part", "region_id": "beard"},
            {"id": "torso", "label": "上身", "kind": "part", "region_id": "torso"},
            {"id": "legs", "label": "下身", "kind": "part", "region_id": "legs"},
        ],
        "tree": [
            {"id": "canopy", "label": "树冠", "kind": "part", "region_id": "canopy"},
            {"id": "trunk", "label": "树干", "kind": "part", "region_id": "trunk"},
        ],
        "car": [
            {"id": "cabin", "label": "驾驶舱", "kind": "part", "region_id": "cabin"},
            {"id": "body", "label": "车身", "kind": "part", "region_id": "body"},
            {"id": "front_wheel", "label": "前轮", "kind": "part", "region_id": "front_wheel"},
            {"id": "rear_wheel", "label": "后轮", "kind": "part", "region_id": "rear_wheel"},
        ],
        "street_lamp": [
            {"id": "pole", "label": "灯杆", "kind": "part", "region_id": "pole"},
            {"id": "lamp_head", "label": "灯头", "kind": "part", "region_id": "lamp_head"},
            {"id": "light_cone", "label": "光照范围", "kind": "part", "region_id": "light_cone"},
        ],
    }
    if key in graphs:
        return _copy(graphs[key])
    return [{"id": "core", "label": "主体", "kind": "part", "region_id": "core"}]


def normalize_layout_options(layout: Dict[str, Any] | None, sketch_options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    merged = _copy(layout or {})
    options = sketch_options or {}
    merged["sketch_view_mode"] = normalize_view_mode(options.get("sketch_view_mode") or merged.get("sketch_view_mode"))
    merged["annotation_level"] = normalize_annotation_level(options.get("annotation_level") or merged.get("annotation_level"))
    merged["region_edit_enabled"] = bool(options.get("region_edit_enabled", merged.get("region_edit_enabled", True)))
    merged["scene_generation_backend"] = str(
        options.get("scene_generation_backend")
        or merged.get("scene_generation_backend")
        or "unified_scene_v3"
    )
    return merged


def summarize_region_overrides(scene_spec: Dict[str, Any] | None) -> str:
    if not isinstance(scene_spec, dict):
        return ""
    edits: List[str] = []
    for obj in scene_spec.get("object_instances", []) or []:
        concept = str(obj.get("concept") or obj.get("asset_key") or "对象").strip()
        region_lookup = {
            str(item.get("id") or ""): str(item.get("label") or item.get("id") or "").strip()
            for item in obj.get("region_masks", []) or []
            if isinstance(item, dict)
        }
        overrides = obj.get("region_overrides") if isinstance(obj, dict) else {}
        if not isinstance(overrides, dict):
            continue
        for region_id, payload in overrides.items():
            if not isinstance(payload, dict):
                continue
            action = action_label(payload.get("action"))
            label = str(payload.get("label") or region_lookup.get(str(region_id), region_id)).strip()
            if action and label:
                edits.append(f"{concept}的{label}{action}")
    return "；".join(edits[:12])
