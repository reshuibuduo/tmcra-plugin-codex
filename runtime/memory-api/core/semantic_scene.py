from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Tuple

from .visual_prototypes import fallback_prototype_id, resolve_visual_prototype


CANVAS_DEFAULT = (1024, 768)


ASSET_LIBRARY: Dict[str, Dict[str, Any]] = {
    "generic_object": {
        "label": "通用对象",
        "category": "通用",
        "silhouette_key": "generic_object",
        "default_size": (152, 132),
        "keywords": [],
        "anchors": ["center", "top", "bottom", "left", "right"],
        "editor_visible": True,
    },
    "generic_panel": {
        "label": "通用面板",
        "category": "通用",
        "silhouette_key": "generic_panel",
        "default_size": (180, 120),
        "keywords": [],
        "anchors": ["center", "top", "bottom"],
        "editor_visible": True,
    },
    "generic_circle": {
        "label": "通用圆形",
        "category": "通用",
        "silhouette_key": "generic_circle",
        "default_size": (120, 120),
        "keywords": [],
        "anchors": ["center"],
        "editor_visible": True,
    },
    "blob": {
        "label": "语义团块",
        "category": "兜底",
        "silhouette_key": "blob",
        "default_size": (168, 128),
        "keywords": [],
        "anchors": ["center"],
        "editor_visible": True,
    },
    "tower": {
        "label": "高塔体",
        "category": "兜底",
        "silhouette_key": "tower",
        "default_size": (180, 260),
        "keywords": [],
        "anchors": ["facade", "roof", "ground", "center"],
        "editor_visible": True,
    },
    "capsule": {
        "label": "过程胶囊",
        "category": "兜底",
        "silhouette_key": "capsule",
        "default_size": (172, 116),
        "keywords": [],
        "anchors": ["center", "left", "right"],
        "editor_visible": True,
    },
    "branch": {
        "label": "分支母题",
        "category": "兜底",
        "silhouette_key": "branch",
        "default_size": (180, 140),
        "keywords": [],
        "anchors": ["center"],
        "editor_visible": True,
    },
    "module": {
        "label": "功能模块",
        "category": "兜底",
        "silhouette_key": "module",
        "default_size": (180, 120),
        "keywords": [],
        "anchors": ["left", "right", "center"],
        "editor_visible": True,
    },
    "building": {
        "label": "楼体",
        "category": "建筑",
        "silhouette_key": "building",
        "default_size": (240, 280),
        "keywords": ["楼", "楼房", "建筑", "大楼", "高楼", "工厂", "教学楼", "公寓"],
        "anchors": ["facade", "roof", "ground", "center"],
        "editor_visible": True,
    },
    "house": {
        "label": "房子",
        "category": "建筑",
        "silhouette_key": "house",
        "default_size": (220, 220),
        "keywords": ["房子", "房屋", "小屋", "住宅", "家"],
        "anchors": ["facade", "roof", "ground", "center"],
        "editor_visible": True,
    },
    "window": {
        "label": "窗户",
        "category": "建筑",
        "silhouette_key": "window",
        "default_size": (72, 72),
        "keywords": ["窗", "窗户", "玻璃窗"],
        "anchors": ["facade", "center"],
        "editor_visible": True,
    },
    "door": {
        "label": "门",
        "category": "建筑",
        "silhouette_key": "door",
        "default_size": (72, 124),
        "keywords": ["门", "大门", "房门"],
        "anchors": ["facade", "ground", "center"],
        "editor_visible": True,
    },
    "tree": {
        "label": "树",
        "category": "自然",
        "silhouette_key": "tree",
        "default_size": (180, 220),
        "keywords": ["树", "树林", "树木"],
        "anchors": ["ground", "center"],
        "editor_visible": True,
    },
    "cloud": {
        "label": "云",
        "category": "自然",
        "silhouette_key": "cloud",
        "default_size": (150, 90),
        "keywords": ["云", "云朵", "乌云"],
        "anchors": ["sky", "center"],
        "editor_visible": True,
    },
    "sun": {
        "label": "太阳",
        "category": "自然",
        "silhouette_key": "sun",
        "default_size": (110, 110),
        "keywords": ["太阳", "阳光", "日光"],
        "anchors": ["sky", "center"],
        "editor_visible": True,
    },
    "car": {
        "label": "车",
        "category": "交通",
        "silhouette_key": "car",
        "default_size": (170, 92),
        "keywords": ["车", "汽车", "轿车", "卡车"],
        "anchors": ["road", "ground", "center"],
        "editor_visible": True,
    },
    "road": {
        "label": "道路",
        "category": "交通",
        "silhouette_key": "road",
        "default_size": (320, 96),
        "keywords": ["路", "道路", "公路", "街道"],
        "anchors": ["ground", "center"],
        "editor_visible": True,
    },
    "person": {
        "label": "人物",
        "category": "角色",
        "silhouette_key": "person",
        "default_size": (96, 172),
        "keywords": ["人", "人物", "学生", "工人", "孩子", "成人"],
        "anchors": ["ground", "center"],
        "editor_visible": True,
    },
    "street_lamp": {
        "label": "路灯",
        "category": "城市",
        "silhouette_key": "street_lamp",
        "default_size": (72, 220),
        "keywords": ["路灯", "灯杆", "街灯"],
        "anchors": ["ground", "center"],
        "editor_visible": True,
    },
    "table": {
        "label": "桌子",
        "category": "室内",
        "silhouette_key": "table",
        "default_size": (180, 116),
        "keywords": ["桌", "桌子", "课桌", "餐桌"],
        "anchors": ["ground", "center"],
        "editor_visible": True,
    },
    "chair": {
        "label": "椅子",
        "category": "室内",
        "silhouette_key": "chair",
        "default_size": (96, 138),
        "keywords": ["椅", "椅子", "凳子"],
        "anchors": ["ground", "center"],
        "editor_visible": True,
    },
    "battery": {
        "label": "电池",
        "category": "自动科技",
        "silhouette_key": "battery",
        "default_size": (120, 80),
        "keywords": ["电池", "电源", "供电"],
        "anchors": ["left", "right", "center"],
        "editor_visible": False,
    },
    "led": {
        "label": "LED",
        "category": "自动科技",
        "silhouette_key": "led",
        "default_size": (104, 92),
        "keywords": ["led", "发光二极管", "灯泡"],
        "anchors": ["left", "right", "center"],
        "editor_visible": False,
    },
    "resistor": {
        "label": "电阻",
        "category": "自动科技",
        "silhouette_key": "resistor",
        "default_size": (120, 62),
        "keywords": ["电阻", "电阻器"],
        "anchors": ["left", "right", "center"],
        "editor_visible": False,
    },
    "capacitor": {
        "label": "电容",
        "category": "自动科技",
        "silhouette_key": "capacitor",
        "default_size": (92, 88),
        "keywords": ["电容", "电容器"],
        "anchors": ["left", "right", "center"],
        "editor_visible": False,
    },
    "diode": {
        "label": "二极管",
        "category": "自动科技",
        "silhouette_key": "diode",
        "default_size": (112, 72),
        "keywords": ["二极管", "整流管"],
        "anchors": ["left", "right", "center"],
        "editor_visible": False,
    },
    "board": {
        "label": "电路板",
        "category": "自动科技",
        "silhouette_key": "board",
        "default_size": (200, 140),
        "keywords": ["arduino", "电路板", "开发板", "主板"],
        "anchors": ["center"],
        "editor_visible": False,
    },
    "airplane": {
        "label": "飞机",
        "category": "自动科技",
        "silhouette_key": "airplane",
        "default_size": (220, 140),
        "keywords": ["飞机", "机翼", "飞行器"],
        "anchors": ["sky", "center"],
        "editor_visible": False,
    },
    "leaf": {
        "label": "叶片",
        "category": "自动科技",
        "silhouette_key": "leaf",
        "default_size": (132, 92),
        "keywords": ["叶片", "叶子", "树叶"],
        "anchors": ["center"],
        "editor_visible": False,
    },
    "raindrop": {
        "label": "雨滴",
        "category": "自动科技",
        "silhouette_key": "raindrop",
        "default_size": (72, 96),
        "keywords": ["雨滴", "水滴"],
        "anchors": ["sky", "center"],
        "editor_visible": False,
    },
    "cell": {
        "label": "细胞",
        "category": "自动科技",
        "silhouette_key": "cell",
        "default_size": (140, 140),
        "keywords": ["细胞"],
        "anchors": ["center"],
        "editor_visible": False,
    },
}


EDITOR_LIBRARY_ORDER = [
    "building",
    "house",
    "window",
    "door",
    "tree",
    "cloud",
    "sun",
    "car",
    "road",
    "person",
    "street_lamp",
    "table",
    "chair",
    "battery",
    "led",
    "resistor",
    "capacitor",
    "diode",
    "board",
    "airplane",
    "leaf",
    "raindrop",
    "cell",
    "tower",
    "capsule",
    "branch",
    "module",
    "blob",
    "generic_object",
    "generic_panel",
    "generic_circle",
]


SCHEMATIC_TOKENS = {
    "led",
    "电路",
    "电阻",
    "电容",
    "二极管",
    "三极管",
    "arduino",
    "电池",
    "导线",
    "串联",
    "并联",
    "电流",
    "电压",
}


SCENE_TOKENS = {
    "场景",
    "图片",
    "画面",
    "草图",
    "房子",
    "楼",
    "树",
    "云",
    "太阳",
    "路",
    "车",
    "人物",
    "街道",
}


PROCESS_TOKENS = {"为什么", "解释", "原理", "过程", "机制", "如何", "怎么", "因果"}


def _deep_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _slugify(text: str) -> str:
    stripped = re.sub(r"\s+", "_", _clean_text(text).lower())
    stripped = re.sub(r"[^0-9a-z_\u4e00-\u9fff]+", "", stripped)
    return stripped or "item"


def get_asset_definition(asset_key: str) -> Dict[str, Any]:
    return ASSET_LIBRARY.get(asset_key, ASSET_LIBRARY["generic_object"])


ATTACHMENT_COMPATIBILITY: Dict[str, List[str]] = {
    "window": ["building", "house", "tower"],
    "door": ["building", "house", "tower"],
    "cloud": ["bg_sky"],
    "sun": ["bg_sky"],
    "car": ["road"],
    "chair": ["table"],
    "street_lamp": ["road", "building", "house"],
}


def build_editor_asset_library() -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for asset_key in EDITOR_LIBRARY_ORDER:
        asset = get_asset_definition(asset_key)
        category = asset.get("category", "其他")
        grouped.setdefault(category, []).append(
            {
                "asset_key": asset_key,
                "label": asset.get("label", asset_key),
                "silhouette_key": asset.get("silhouette_key", asset_key),
                "default_width": asset.get("default_size", (120, 120))[0],
                "default_height": asset.get("default_size", (120, 120))[1],
                "anchors": asset.get("anchors", []),
            }
        )
    return [{"category": category, "items": items} for category, items in grouped.items()]


def pick_asset_key(concept: str, scene_type: str = "scene") -> str:
    concept_text = _clean_text(concept)
    lowered = concept_text.lower()
    for asset_key, asset in ASSET_LIBRARY.items():
        for keyword in asset.get("keywords", []):
            if keyword and keyword.lower() in lowered:
                return asset_key
    if scene_type == "schematic":
        if any(token in lowered for token in ("电流", "电压", "导电", "功率", "效率")):
            return "generic_panel"
        return "generic_object"
    if scene_type == "process" and any(token in concept_text for token in ("作用", "过程", "机制", "功能", "能力")):
        return "generic_panel"
    return "generic_object"


def infer_scene_type(
    query: str,
    understanding_result: Dict[str, Any] | None,
    extraction_result: Dict[str, Any] | None,
    answer_bundle: Dict[str, Any] | None,
) -> str:
    query_text = _clean_text(query)
    joined_parts = [query_text]
    for source in (understanding_result or {}, extraction_result or {}, answer_bundle or {}):
        joined_parts.append(json.dumps(source, ensure_ascii=False))
    haystack = " ".join(joined_parts).lower()
    if any(token.lower() in haystack for token in SCHEMATIC_TOKENS):
        return "schematic"
    if any(token.lower() in haystack for token in SCENE_TOKENS) and not any(token in query_text for token in PROCESS_TOKENS):
        return "scene"
    if any(token in query_text for token in PROCESS_TOKENS):
        return "process"
    return "scene"


def _collect_concepts(
    understanding_result: Dict[str, Any] | None,
    extraction_result: Dict[str, Any] | None,
    answer_bundle: Dict[str, Any] | None,
    best_path_concepts: List[str] | None,
) -> tuple[List[str], Dict[str, str], Dict[str, float]]:
    ordered: List[str] = []
    concept_types: Dict[str, str] = {}
    importance: Dict[str, float] = {}

    def push(concept: str, concept_type: str = "general", delta: float = 1.0) -> None:
        name = _clean_text(concept)
        if not name:
            return
        if name not in ordered:
            ordered.append(name)
        if concept_type and name not in concept_types:
            concept_types[name] = concept_type
        importance[name] = importance.get(name, 0.0) + delta

    focus = _clean_text((answer_bundle or {}).get("focus_concept") or (understanding_result or {}).get("focus_concept"))
    if focus:
        push(focus, "focus", 4.0)

    for concept in best_path_concepts or []:
        push(concept, "path", 2.8)

    for item in (extraction_result or {}).get("concepts", []) or []:
        push(item.get("concept", ""), item.get("type", "general"), 1.3)

    for item in (understanding_result or {}).get("concepts", []) or []:
        push(item.get("concept", ""), item.get("type", "general"), 1.0)

    for relation in (answer_bundle or {}).get("primary_chain", []) or []:
        push(relation.get("from", ""), "relation", 2.2)
        push(relation.get("to", ""), "relation", 2.2)

    for relation in (answer_bundle or {}).get("supporting_relations", []) or []:
        push(relation.get("from", ""), "support", 1.0)
        push(relation.get("to", ""), "support", 1.0)

    return ordered, concept_types, importance


def _collect_relations(
    answer_bundle: Dict[str, Any] | None,
    extraction_result: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for group in (
        (answer_bundle or {}).get("primary_chain", []) or [],
        (answer_bundle or {}).get("supporting_relations", []) or [],
        (extraction_result or {}).get("relations", []) or [],
    ):
        for relation in group:
            src = _clean_text(relation.get("from"))
            dst = _clean_text(relation.get("to"))
            label = _clean_text(relation.get("relation"))
            if not src or not dst or not label:
                continue
            key = (src, dst, label)
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "from": src,
                    "to": dst,
                    "relation": label,
                    "weight": float(relation.get("weight", 0.5) or 0.5),
                    "source": relation.get("source", "evidence"),
                }
            )
    return merged


def _make_id(prefix: str, label: str, index: int) -> str:
    return f"{prefix}_{index}_{_slugify(label)}"


def _build_background_layer(
    layer_id: str,
    layer_type: str,
    label: str,
    x: int,
    y: int,
    width: int,
    height: int,
    z_index: int,
    source: str = "auto",
) -> Dict[str, Any]:
    return {
        "id": layer_id,
        "type": layer_type,
        "label": label,
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
        "z_index": int(z_index),
        "source": source,
    }


def _build_object_instance(
    object_id: str,
    concept: str,
    asset_key: str,
    role: str,
    depth_band: str,
    x: float,
    y: float,
    width: float,
    height: float,
    z_index: int,
    source: str = "auto",
    rotation: float = 0.0,
    scale: float = 1.0,
    editable: bool = True,
    visible: bool = True,
    depth_z: float | None = None,
) -> Dict[str, Any]:
    asset = get_asset_definition(asset_key)
    return {
        "id": object_id,
        "concept": _clean_text(concept),
        "asset_key": asset_key,
        "source": source,
        "role": role,
        "depth_band": depth_band,
        "x": int(round(x)),
        "y": int(round(y)),
        "width": int(round(width)),
        "height": int(round(height)),
        "rotation": float(rotation),
        "scale": float(scale),
        "silhouette_key": asset.get("silhouette_key", asset_key),
        "z_index": int(z_index),
        "editable": bool(editable),
        "visible": bool(visible),
        "depth_z": float(depth_z) if depth_z is not None else {
            "background": 0.18,
            "midground": 0.52,
            "foreground": 0.84,
        }.get(depth_band, 0.52),
    }


def _connector_type(label: str, scene_type: str) -> str:
    text = _clean_text(label)
    if any(keyword in text for keyword in ("光", "照射", "发射", "辐射")):
        return "beam"
    if scene_type == "schematic" or any(keyword in text for keyword in ("连接", "串联", "并联", "导通", "供电")):
        return "wire"
    if any(keyword in text for keyword in ("导致", "推动", "产生", "影响", "驱动", "引发", "减少", "增加", "支持")):
        return "arrow"
    return "relation"


def _build_connector(
    connector_id: str,
    from_id: str,
    to_id: str,
    label: str,
    scene_type: str,
    visible: bool = True,
) -> Dict[str, Any]:
    return {
        "id": connector_id,
        "type": _connector_type(label, scene_type),
        "from_id": from_id,
        "to_id": to_id,
        "label": _clean_text(label),
        "visible": bool(visible),
    }


def _build_attachment(attachment_id: str, host_id: str, child_id: str, anchor_name: str, mode: str = "attach") -> Dict[str, Any]:
    return {
        "id": attachment_id,
        "host_id": host_id,
        "child_id": child_id,
        "anchor_name": anchor_name,
        "mode": mode,
    }


def _build_scene_shell(canvas_size: Tuple[int, int], sketch_options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    sketch_options = sketch_options or {}
    width = max(512, int(canvas_size[0]))
    height = max(384, int(canvas_size[1]))
    return {
        "version": 2,
        "canvas_size": {"width": width, "height": height},
        "layout_options": {
            "mode": "semantic_composition",
            "sketch_style": sketch_options.get("sketch_style", "line_art"),
            "show_grid": bool(sketch_options.get("show_grid", True)),
            "show_labels": bool(sketch_options.get("show_labels", True)),
            "show_guides": bool(sketch_options.get("show_guides", True)),
            "node_scale": float(sketch_options.get("node_scale", 1.0)),
            "spacing_scale": float(sketch_options.get("spacing_scale", 1.0)),
        },
        "background_layers": [],
        "object_instances": [],
        "attachments": [],
        "connectors": [],
        "render_hints": {},
        "concept_order": [],
    }


def _render_hints_for_scene(scene: Dict[str, Any], scene_type: str) -> Dict[str, str]:
    objects = scene.get("object_instances", []) or []
    backgrounds = scene.get("background_layers", []) or []
    subjects = [item["concept"] for item in objects if item.get("role") in {"subject", "focus", "core_subject"}][:3]
    user_items = [item["concept"] for item in objects if item.get("source") == "user"][:6]
    bg_labels = [item["label"] for item in backgrounds][:4]
    connectors = [item["label"] for item in scene.get("connectors", []) if item.get("visible")][:4]
    style_name = scene.get("layout_options", {}).get("sketch_style", "line_art")
    return {
        "scene_summary": "、".join(subjects) + (" 场景草图" if subjects else "语义构图草图"),
        "subject_summary": "主体: " + ("、".join(subjects) if subjects else "未显式主体"),
        "style_summary": f"风格: {style_name} | 类型: {scene_type}",
        "preview_summary": "背景: " + ("、".join(bg_labels) if bg_labels else "无") + " | 关系: " + ("、".join(connectors) if connectors else "弱化显示"),
        "user_added_summary": "用户新增: " + ("、".join(user_items) if user_items else "无"),
    }


def _outdoor_backgrounds(width: int, height: int, include_road: bool = False, include_water: bool = False) -> List[Dict[str, Any]]:
    layers = [
        _build_background_layer("bg_sky", "sky", "天空", 0, 0, width, int(height * 0.58), -30),
    ]
    ground_label = "水面" if include_water else "地面"
    ground_type = "water" if include_water else "ground"
    layers.append(_build_background_layer("bg_ground", ground_type, ground_label, 0, int(height * 0.58), width, int(height * 0.42), -20))
    if include_road:
        road_height = max(72, int(height * 0.16))
        layers.append(_build_background_layer("bg_road", "road", "道路", 0, height - road_height, width, road_height, -10))
    return layers


def _compose_scene_layout(
    query: str,
    concept_names: List[str],
    concept_types: Dict[str, str],
    importance: Dict[str, float],
    relations: List[Dict[str, Any]],
    focus_concept: str,
    canvas_size: Tuple[int, int],
    sketch_options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    scene = _build_scene_shell(canvas_size, sketch_options)
    width = scene["canvas_size"]["width"]
    height = scene["canvas_size"]["height"]
    indoor_mode = any(token in query for token in ("室内", "房间")) or any(
        pick_asset_key(name, "scene") in {"table", "chair", "desk_lamp"} for name in concept_names
    )
    include_road = any(pick_asset_key(name) == "car" or "路" in name for name in concept_names)
    include_water = any("水" in name for name in concept_names)
    if indoor_mode:
        scene["background_layers"] = [
            _build_background_layer("bg_wall", "wall", "墙面", 0, 0, width, int(height * 0.64), -30),
            _build_background_layer("bg_floor", "ground", "地面", 0, int(height * 0.64), width, int(height * 0.36), -20),
        ]
    else:
        scene["background_layers"] = _outdoor_backgrounds(width, height, include_road=include_road, include_water=include_water)

    ranked = sorted(concept_names, key=lambda name: (-importance.get(name, 0.0), concept_names.index(name)))
    subject = focus_concept if focus_concept in concept_names else (ranked[0] if ranked else "")
    if indoor_mode:
        table_subject = next((name for name in ranked if pick_asset_key(name, "scene") == "table"), "")
        if table_subject:
            subject = table_subject
    if pick_asset_key(subject) in {"window", "door", "cloud", "sun"} and len(ranked) > 1:
        subject = ranked[1]

    subject_id = ""
    if subject:
        asset_key = pick_asset_key(subject, "scene")
        default_w, default_h = get_asset_definition(asset_key).get("default_size", (180, 180))
        subject_x = width * 0.37
        subject_y = height * 0.34
        subject_w = default_w * 1.15
        subject_h = default_h * 1.15
        if indoor_mode and asset_key == "table":
            subject_x = width * 0.28
            subject_y = height * 0.5
            subject_w = min(width * 0.42, default_w * 1.35)
            subject_h = min(height * 0.26, default_h * 1.15)
        scene["object_instances"].append(
            _build_object_instance(
                "obj_subject",
                subject,
                asset_key,
                "subject",
                "midground",
                subject_x,
                subject_y,
                subject_w,
                subject_h,
                30,
            )
        )
        subject_id = "obj_subject"

    side_toggle = 0
    detail_index = 0
    for concept in ranked:
        if concept == subject:
            continue
        asset_key = pick_asset_key(concept, "scene")
        default_w, default_h = get_asset_definition(asset_key).get("default_size", (120, 120))
        if indoor_mode and asset_key in {"chair", "desk_lamp"} and subject_id and pick_asset_key(subject, "scene") == "table":
            host = next((item for item in scene["object_instances"] if item["id"] == subject_id), None)
            if host:
                child_id = _make_id("obj", concept, detail_index + 1)
                if asset_key == "chair":
                    child_w = min(default_w, host["width"] * 0.34)
                    child_h = min(default_h, host["height"] * 1.26)
                    child_x = host["x"] - child_w * (0.58 if detail_index % 2 == 0 else -1.08)
                    child_y = host["y"] - host["height"] * 0.1
                    anchor = "center"
                    role_name = "support"
                else:
                    child_w = min(default_w, host["width"] * 0.24)
                    child_h = min(default_h, max(84, host["height"] * 0.96))
                    child_x = host["x"] + host["width"] * 0.58
                    child_y = host["y"] - child_h * 0.42
                    anchor = "center"
                    role_name = "detail"
                scene["object_instances"].append(
                    _build_object_instance(child_id, concept, asset_key, role_name, "midground", child_x, child_y, child_w, child_h, 36)
                )
                scene["attachments"].append(_build_attachment(_make_id("att", concept, detail_index + 1), host["id"], child_id, anchor))
                detail_index += 1
                continue
        if asset_key in {"window", "door"} and subject_id and pick_asset_key(subject, "scene") in {"building", "house"}:
            host = next((item for item in scene["object_instances"] if item["id"] == subject_id), None)
            if not host:
                continue
            child_id = _make_id("obj", concept, detail_index + 1)
            child_w = min(default_w, host["width"] * 0.22)
            child_h = min(default_h, host["height"] * (0.26 if asset_key == "door" else 0.18))
            child_x = host["x"] + host["width"] * (0.2 + 0.22 * (detail_index % 3))
            child_y = host["y"] + host["height"] * (0.2 + 0.2 * (detail_index // 3))
            anchor = "facade"
            if asset_key == "door":
                child_y = host["y"] + host["height"] - child_h - 6
                anchor = "ground"
            scene["object_instances"].append(
                _build_object_instance(child_id, concept, asset_key, "detail", "midground", child_x, child_y, child_w, child_h, 36)
            )
            scene["attachments"].append(_build_attachment(_make_id("att", concept, detail_index + 1), host["id"], child_id, anchor))
            detail_index += 1
            continue

        depth_band = "background"
        role = "environment"
        z_index = 12
        x = width * (0.08 if side_toggle % 2 == 0 else 0.68)
        y = height * (0.4 if side_toggle % 2 == 0 else 0.46)
        scale = 0.84
        if asset_key in {"cloud", "sun"}:
            depth_band = "background"
            role = "effect"
            z_index = 4
            y = height * 0.1
            x = width * (0.12 + 0.2 * (side_toggle % 4))
            scale = 0.8
        elif asset_key in {"tree", "person", "street_lamp", "car"}:
            depth_band = "foreground"
            role = "support"
            z_index = 42
            y = height * 0.56
            scale = 0.95
        elif asset_key == "dog":
            depth_band = "foreground"
            role = "support"
            z_index = 40
            y = height * 0.62
            scale = 0.96
        elif asset_key in {"building", "house"}:
            depth_band = "background"
            role = "support"
            z_index = 18
            y = height * 0.24
            scale = 0.9
        elif asset_key == "road":
            depth_band = "foreground"
            role = "support"
            z_index = 8
            y = height * 0.8
            scale = 1.0
        elif indoor_mode and asset_key == "table":
            depth_band = "midground"
            role = "support"
            z_index = 28
            x = width * 0.58
            y = height * 0.52
            scale = 1.05

        object_id = _make_id("obj", concept, side_toggle + 1)
        object_width = default_w * scale
        object_height = default_h * scale
        scene["object_instances"].append(
            _build_object_instance(object_id, concept, asset_key, role, depth_band, x, y, object_width, object_height, z_index)
        )
        side_toggle += 1

    object_by_concept = {item["concept"]: item for item in scene["object_instances"]}
    for index, relation in enumerate(relations[:8], start=1):
        from_obj = object_by_concept.get(relation["from"])
        to_obj = object_by_concept.get(relation["to"])
        if not from_obj or not to_obj:
            continue
        connector = _build_connector(
            _make_id("conn", relation["relation"], index),
            from_obj["id"],
            to_obj["id"],
            relation["relation"],
            "scene",
            visible=_connector_type(relation["relation"], "scene") in {"beam", "arrow"},
        )
        if connector["visible"]:
            scene["connectors"].append(connector)

    scene["concept_order"] = [item["concept"] for item in scene["object_instances"]]
    scene["render_hints"] = _render_hints_for_scene(scene, "scene")
    scene["layout_options"]["scene_type"] = "scene"
    return scene


def _compose_process_layout(
    concept_names: List[str],
    importance: Dict[str, float],
    relations: List[Dict[str, Any]],
    focus_concept: str,
    canvas_size: Tuple[int, int],
    sketch_options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    scene = _build_scene_shell(canvas_size, sketch_options)
    width = scene["canvas_size"]["width"]
    height = scene["canvas_size"]["height"]
    panel_width = width // 3
    scene["background_layers"] = [
        _build_background_layer("panel_input", "panel", "输入", 0, 92, panel_width, height - 132, -20),
        _build_background_layer("panel_process", "panel", "过程", panel_width, 92, panel_width, height - 132, -19),
        _build_background_layer("panel_output", "panel", "结果", panel_width * 2, 92, width - panel_width * 2, height - 132, -18),
    ]

    ordered = concept_names[:6]
    if focus_concept and focus_concept in ordered:
        ordered = [focus_concept] + [item for item in ordered if item != focus_concept]
    if not ordered and relations:
        ordered = [relations[0]["from"], relations[0]["to"]]

    x_positions = [width * 0.12, width * 0.4, width * 0.72]
    y_base = [height * 0.32, height * 0.5]

    object_by_concept: Dict[str, Dict[str, Any]] = {}
    for index, concept in enumerate(ordered):
        asset_key = pick_asset_key(concept, "process")
        if asset_key == "generic_object":
            asset_key = "generic_panel"
        default_w, default_h = get_asset_definition(asset_key).get("default_size", (180, 120))
        importance_scale = 0.9 + min(0.45, importance.get(concept, 1.0) * 0.05)
        column = min(2, index // 2)
        row = index % 2
        role = "subject" if concept == focus_concept else ("stage" if column == 1 else "support")
        object_id = _make_id("obj", concept, index + 1)
        obj = _build_object_instance(
            object_id,
            concept,
            asset_key,
            role,
            "midground",
            x_positions[column] - (default_w * importance_scale) / 2,
            y_base[row] - (default_h * importance_scale) / 2,
            default_w * importance_scale,
            default_h * importance_scale,
            25 + column,
        )
        scene["object_instances"].append(obj)
        object_by_concept[concept] = obj

    for index, relation in enumerate(relations[:10], start=1):
        from_obj = object_by_concept.get(relation["from"])
        to_obj = object_by_concept.get(relation["to"])
        if not from_obj or not to_obj:
            continue
        scene["connectors"].append(
            _build_connector(_make_id("conn", relation["relation"], index), from_obj["id"], to_obj["id"], relation["relation"], "process", visible=True)
        )

    scene["concept_order"] = [item["concept"] for item in scene["object_instances"]]
    scene["render_hints"] = _render_hints_for_scene(scene, "process")
    scene["layout_options"]["scene_type"] = "process"
    return scene


def _compose_schematic_layout(
    concept_names: List[str],
    importance: Dict[str, float],
    relations: List[Dict[str, Any]],
    focus_concept: str,
    canvas_size: Tuple[int, int],
    sketch_options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    scene = _build_scene_shell(canvas_size, sketch_options)
    width = scene["canvas_size"]["width"]
    height = scene["canvas_size"]["height"]
    scene["background_layers"] = [
        _build_background_layer("board_layer", "board", "电路画板", 18, 72, width - 36, height - 110, -18),
    ]

    ordered = concept_names[:8]
    if focus_concept and focus_concept in ordered:
        ordered = [focus_concept] + [item for item in ordered if item != focus_concept]
    if not ordered and relations:
        ordered = [relations[0]["from"], relations[0]["to"]]
    if not ordered:
        ordered = ["电路", "信号", "输出"]

    cell_concept = next((name for name in ordered if pick_asset_key(name, "schematic") == "cell"), "")
    if cell_concept:
        object_by_concept: Dict[str, Dict[str, Any]] = {}
        cell_w = width * 0.42
        cell_h = height * 0.46
        cell_x = width * 0.29
        cell_y = height * 0.24
        cell_obj = _build_object_instance("obj_cell", cell_concept, "cell", "subject", "midground", cell_x, cell_y, cell_w, cell_h, 30)
        scene["object_instances"].append(cell_obj)
        object_by_concept[cell_concept] = cell_obj

        ring_added = False
        for index, concept in enumerate(ordered, start=1):
            if concept == cell_concept:
                continue
            asset_key = pick_asset_key(concept, "schematic")
            object_id = _make_id("obj", concept, index)
            if "膜" in concept:
                membrane = _build_object_instance(
                    object_id,
                    concept,
                    "generic_circle",
                    "detail",
                    "midground",
                    cell_x + cell_w * 0.05,
                    cell_y + cell_h * 0.06,
                    cell_w * 0.9,
                    cell_h * 0.86,
                    31,
                )
                scene["object_instances"].append(membrane)
                scene["attachments"].append(_build_attachment(_make_id("att", concept, index), cell_obj["id"], object_id, "center"))
                object_by_concept[concept] = membrane
                ring_added = True
                continue
            if "核" in concept:
                nucleus = _build_object_instance(
                    object_id,
                    concept,
                    "generic_circle",
                    "detail",
                    "midground",
                    cell_x + cell_w * 0.4,
                    cell_y + cell_h * 0.34,
                    cell_w * 0.18,
                    cell_h * 0.18,
                    34,
                )
                scene["object_instances"].append(nucleus)
                scene["attachments"].append(_build_attachment(_make_id("att", concept, index), cell_obj["id"], object_id, "center"))
                object_by_concept[concept] = nucleus
                continue
            if "质" in concept:
                cytoplasm = _build_object_instance(
                    object_id,
                    concept,
                    "generic_circle",
                    "support",
                    "midground",
                    cell_x + cell_w * 0.22,
                    cell_y + cell_h * 0.25,
                    cell_w * 0.56,
                    cell_h * 0.4,
                    32,
                )
                scene["object_instances"].append(cytoplasm)
                scene["attachments"].append(_build_attachment(_make_id("att", concept, index), cell_obj["id"], object_id, "center"))
                object_by_concept[concept] = cytoplasm
                continue

            default_w, default_h = get_asset_definition(asset_key).get("default_size", (140, 90))
            side = -1 if index % 2 == 0 else 1
            x = width * (0.08 if side < 0 else 0.7)
            y = height * (0.22 + 0.18 * ((index - 1) % 3))
            obj = _build_object_instance(
                object_id,
                concept,
                asset_key,
                "component",
                "midground",
                x,
                y,
                default_w * 0.95,
                default_h * 0.95,
                25,
            )
            scene["object_instances"].append(obj)
            object_by_concept[concept] = obj
            scene["connectors"].append(
                _build_connector(_make_id("conn", concept, index), obj["id"], cell_obj["id"], "关联", "schematic", visible=True)
            )

        if not ring_added:
            membrane = _build_object_instance(
                "obj_cell_membrane",
                "细胞膜",
                "generic_circle",
                "detail",
                "midground",
                cell_x + cell_w * 0.05,
                cell_y + cell_h * 0.06,
                cell_w * 0.9,
                cell_h * 0.86,
                31,
            )
            scene["object_instances"].append(membrane)
            scene["attachments"].append(_build_attachment("att_cell_membrane", cell_obj["id"], membrane["id"], "center"))
            object_by_concept["细胞膜"] = membrane

        for index, relation in enumerate(relations[:12], start=1):
            from_obj = object_by_concept.get(relation["from"])
            to_obj = object_by_concept.get(relation["to"])
            if not from_obj or not to_obj:
                continue
            scene["connectors"].append(
                _build_connector(_make_id("conn", relation["relation"], index), from_obj["id"], to_obj["id"], relation["relation"], "schematic", visible=True)
            )

        scene["concept_order"] = [item["concept"] for item in scene["object_instances"]]
        scene["render_hints"] = _render_hints_for_scene(scene, "schematic")
        scene["layout_options"]["scene_type"] = "schematic"
        return scene

    gap = max(110, int((width - 160) / max(1, len(ordered) - 1)))
    object_by_concept: Dict[str, Dict[str, Any]] = {}
    y = height * 0.5
    for index, concept in enumerate(ordered):
        asset_key = pick_asset_key(concept, "schematic")
        default_w, default_h = get_asset_definition(asset_key).get("default_size", (140, 90))
        importance_scale = 0.85 + min(0.4, importance.get(concept, 1.0) * 0.04)
        x = 80 + gap * index
        role = "subject" if concept == focus_concept else "component"
        object_id = _make_id("obj", concept, index + 1)
        obj = _build_object_instance(
            object_id,
            concept,
            asset_key,
            role,
            "midground",
            x - (default_w * importance_scale) / 2,
            y - (default_h * importance_scale) / 2,
            default_w * importance_scale,
            default_h * importance_scale,
            28 + index,
        )
        scene["object_instances"].append(obj)
        object_by_concept[concept] = obj

    for index, relation in enumerate(relations[:12], start=1):
        from_obj = object_by_concept.get(relation["from"])
        to_obj = object_by_concept.get(relation["to"])
        if not from_obj or not to_obj:
            continue
        scene["connectors"].append(
            _build_connector(_make_id("conn", relation["relation"], index), from_obj["id"], to_obj["id"], relation["relation"], "schematic", visible=True)
        )

    scene["concept_order"] = [item["concept"] for item in scene["object_instances"]]
    scene["render_hints"] = _render_hints_for_scene(scene, "schematic")
    scene["layout_options"]["scene_type"] = "schematic"
    return scene


def compose_semantic_scene_spec(
    query: str,
    understanding_result: Dict[str, Any] | None,
    extraction_result: Dict[str, Any] | None,
    answer_bundle: Dict[str, Any] | None,
    best_path_concepts: List[str] | None = None,
    canvas_size: Tuple[int, int] = CANVAS_DEFAULT,
    sketch_options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    scene_type = infer_scene_type(query, understanding_result, extraction_result, answer_bundle)
    concept_names, concept_types, importance = _collect_concepts(
        understanding_result,
        extraction_result,
        answer_bundle,
        best_path_concepts,
    )
    relations = _collect_relations(answer_bundle, extraction_result)
    focus_concept = _clean_text((answer_bundle or {}).get("focus_concept") or (understanding_result or {}).get("focus_concept"))

    if scene_type == "schematic":
        scene = _compose_schematic_layout(concept_names, importance, relations, focus_concept, canvas_size, sketch_options)
    elif scene_type == "process":
        scene = _compose_process_layout(concept_names, importance, relations, focus_concept, canvas_size, sketch_options)
    else:
        scene = _compose_scene_layout(query, concept_names, concept_types, importance, relations, focus_concept, canvas_size, sketch_options)

    scene["render_hints"]["concept_type_summary"] = "、".join(
        f"{concept}:{concept_types.get(concept, 'general')}" for concept in concept_names[:6]
    )
    scene["render_hints"]["query_summary"] = _clean_text(query)
    scene["debug_legacy"] = {
        "concept_order": concept_names,
        "relations": relations[:12],
    }
    return scene


def _normalize_background_layers(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    layers = []
    for index, layer in enumerate(scene.get("background_layers", []) or [], start=1):
        width = max(1, int(layer.get("width", 1)))
        height = max(1, int(layer.get("height", 1)))
        layers.append(
            _build_background_layer(
                layer.get("id") or f"bg_{index}",
                layer.get("type") or "panel",
                layer.get("label") or f"区域{index}",
                int(layer.get("x", 0)),
                int(layer.get("y", 0)),
                width,
                height,
                int(layer.get("z_index", -10)),
                _clean_text(layer.get("source") or "auto"),
            )
        )
    return layers


def _normalize_objects(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    objects = []
    for index, obj in enumerate(scene.get("object_instances", []) or [], start=1):
        concept = _clean_text(obj.get("concept") or f"对象{index}")
        asset_key = obj.get("asset_key") or pick_asset_key(concept, scene.get("layout_options", {}).get("scene_type", "scene"))
        default_w, default_h = get_asset_definition(asset_key).get("default_size", (140, 100))
        width = max(36, int(obj.get("width", default_w)))
        height = max(36, int(obj.get("height", default_h)))
        objects.append(
            _build_object_instance(
                obj.get("id") or f"obj_{index}",
                concept,
                asset_key,
                _clean_text(obj.get("role") or "support"),
                _clean_text(obj.get("depth_band") or "midground"),
                float(obj.get("x", 80 + index * 40)),
                float(obj.get("y", 120 + index * 24)),
                width,
                height,
                int(obj.get("z_index", 20 + index)),
                _clean_text(obj.get("source") or "auto"),
                float(obj.get("rotation", 0.0) or 0.0),
                float(obj.get("scale", 1.0) or 1.0),
                bool(obj.get("editable", True)),
                bool(obj.get("visible", True)),
                obj.get("depth_z"),
            )
        )
    return objects


def _normalize_connectors(scene: Dict[str, Any], object_ids: Iterable[str]) -> List[Dict[str, Any]]:
    valid_ids = set(object_ids)
    connectors = []
    for index, connector in enumerate(scene.get("connectors", []) or [], start=1):
        from_id = _clean_text(connector.get("from_id"))
        to_id = _clean_text(connector.get("to_id"))
        if from_id not in valid_ids or to_id not in valid_ids:
            continue
        connectors.append(
            {
                "id": connector.get("id") or f"conn_{index}",
                "type": _clean_text(connector.get("type") or "arrow"),
                "from_id": from_id,
                "to_id": to_id,
                "label": _clean_text(connector.get("label") or "连接"),
                "visible": bool(connector.get("visible", True)),
            }
        )
    return connectors


def _normalize_attachments(scene: Dict[str, Any], object_ids: Iterable[str]) -> List[Dict[str, Any]]:
    valid_ids = set(object_ids)
    attachments = []
    for index, attachment in enumerate(scene.get("attachments", []) or [], start=1):
        host_id = _clean_text(attachment.get("host_id"))
        child_id = _clean_text(attachment.get("child_id"))
        if host_id not in valid_ids or child_id not in valid_ids or host_id == child_id:
            continue
        attachments.append(
            _build_attachment(
                attachment.get("id") or f"att_{index}",
                host_id,
                child_id,
                _clean_text(attachment.get("anchor_name") or "center"),
                _clean_text(attachment.get("mode") or "attach"),
            )
        )
    return attachments


def legacy_scene_spec_to_v2(scene_spec: Dict[str, Any], sketch_options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    sketch_options = sketch_options or {}
    canvas = scene_spec.get("canvas_size") or {}
    canvas_size = (
        max(512, int(canvas.get("width", sketch_options.get("canvas_width", CANVAS_DEFAULT[0])))),
        max(384, int(canvas.get("height", sketch_options.get("canvas_height", CANVAS_DEFAULT[1])))),
    )
    scene = _build_scene_shell(canvas_size, sketch_options)
    legacy_nodes = scene_spec.get("nodes", []) or []
    legacy_relations = scene_spec.get("relations", []) or []
    inferred_type = "schematic" if legacy_relations else "scene"
    scene["layout_options"]["scene_type"] = inferred_type

    object_instances = []
    concept_order = []
    for index, node in enumerate(legacy_nodes, start=1):
        concept = _clean_text(node.get("concept") or f"节点{index}")
        concept_order.append(concept)
        asset_key = pick_asset_key(concept, inferred_type)
        object_instances.append(
            _build_object_instance(
                node.get("id") or f"obj_{index}",
                concept,
                asset_key,
                _clean_text(node.get("role") or "support"),
                "midground",
                float(node.get("x", 80 + index * 120)),
                float(node.get("y", 240)),
                float(node.get("width", get_asset_definition(asset_key).get("default_size", (140, 100))[0])),
                float(node.get("height", get_asset_definition(asset_key).get("default_size", (140, 100))[1])),
                int(node.get("z_index", 20 + index)),
                _clean_text(node.get("source") or "auto"),
                float(node.get("rotation", 0.0) or 0.0),
                float(node.get("scale", 1.0) or 1.0),
                bool(node.get("editable", True)),
            )
        )

    scene["object_instances"] = object_instances
    scene["connectors"] = [
        _build_connector(
            relation.get("id") or f"conn_{index}",
            _clean_text(relation.get("from_id")),
            _clean_text(relation.get("to_id")),
            relation.get("relation") or relation.get("label") or "连接",
            inferred_type,
            visible=True,
        )
        for index, relation in enumerate(legacy_relations, start=1)
        if _clean_text(relation.get("from_id")) and _clean_text(relation.get("to_id"))
    ]
    scene["concept_order"] = concept_order
    scene["render_hints"] = _render_hints_for_scene(scene, inferred_type)
    scene["debug_legacy"] = {
        "concept_order": scene_spec.get("concept_order", concept_order),
        "nodes": legacy_nodes,
        "relations": legacy_relations,
    }
    return scene


def normalize_scene_spec_v2(scene_spec: Dict[str, Any] | None, sketch_options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    sketch_options = sketch_options or {}
    raw = _deep_copy(scene_spec or {})
    if not raw:
        return _build_scene_shell(CANVAS_DEFAULT, sketch_options)
    if int(raw.get("version", 0) or 0) != 2 and ("object_instances" not in raw and "background_layers" not in raw):
        raw = legacy_scene_spec_to_v2(raw, sketch_options)

    canvas = raw.get("canvas_size") or {}
    width = max(512, int(canvas.get("width", sketch_options.get("canvas_width", CANVAS_DEFAULT[0]))))
    height = max(384, int(canvas.get("height", sketch_options.get("canvas_height", CANVAS_DEFAULT[1]))))
    normalized = _build_scene_shell((width, height), sketch_options)
    normalized["version"] = 2
    normalized["layout_options"].update(raw.get("layout_options") or {})
    normalized["layout_options"]["mode"] = "semantic_composition"
    normalized["layout_options"]["sketch_style"] = sketch_options.get(
        "sketch_style",
        normalized["layout_options"].get("sketch_style", "line_art"),
    )
    normalized["layout_options"]["show_grid"] = bool(
        sketch_options.get("show_grid", normalized["layout_options"].get("show_grid", True))
    )
    normalized["layout_options"]["show_labels"] = bool(
        sketch_options.get("show_labels", normalized["layout_options"].get("show_labels", True))
    )
    normalized["layout_options"]["show_guides"] = bool(
        sketch_options.get("show_guides", normalized["layout_options"].get("show_guides", True))
    )
    normalized["layout_options"]["node_scale"] = float(
        sketch_options.get("node_scale", normalized["layout_options"].get("node_scale", 1.0))
    )
    normalized["layout_options"]["spacing_scale"] = float(
        sketch_options.get("spacing_scale", normalized["layout_options"].get("spacing_scale", 1.0))
    )

    normalized["background_layers"] = _normalize_background_layers(raw)
    normalized["object_instances"] = _normalize_objects(raw)
    object_ids = [item["id"] for item in normalized["object_instances"]]
    normalized["attachments"] = _normalize_attachments(raw, object_ids)
    normalized["connectors"] = _normalize_connectors(raw, object_ids)
    normalized["concept_order"] = [
        _clean_text(item.get("concept"))
        for item in normalized["object_instances"]
        if _clean_text(item.get("concept"))
    ]
    normalized["render_hints"] = {
        **_render_hints_for_scene(
            {
                "background_layers": normalized["background_layers"],
                "object_instances": normalized["object_instances"],
                "connectors": normalized["connectors"],
                "layout_options": normalized["layout_options"],
            },
            normalized["layout_options"].get("scene_type", "scene"),
        ),
        **(raw.get("render_hints") or {}),
    }
    if raw.get("debug_legacy"):
        normalized["debug_legacy"] = raw["debug_legacy"]
    return normalized


def summarize_scene_spec(scene_spec: Dict[str, Any] | None) -> str:
    if not scene_spec:
        return ""
    scene = normalize_scene_spec_v2(scene_spec)
    backgrounds = scene.get("background_layers", []) or []
    objects = scene.get("object_instances", []) or []
    attachments = scene.get("attachments", []) or []
    connectors = [item for item in scene.get("connectors", []) or [] if item.get("visible")]

    bg_summary = ", ".join(
        f"{item['label']}@({item['x']},{item['y']},{item['width']}x{item['height']})"
        for item in backgrounds[:5]
    )
    object_summary = ", ".join(
        f"{item['concept']}[{item['asset_key']}]@({item['x']},{item['y']},{item['width']}x{item['height']})/{item['depth_band']}"
        for item in objects[:8]
    )
    attachment_summary = ", ".join(
        f"{item['child_id']}->{item['host_id']}:{item['anchor_name']}"
        for item in attachments[:6]
    )
    connector_summary = ", ".join(
        f"{item['type']}:{item['label']} {item['from_id']}->{item['to_id']}"
        for item in connectors[:6]
    )
    hints = scene.get("render_hints", {}) or {}
    parts = [
        f"scene_type={scene.get('layout_options', {}).get('scene_type', 'scene')}",
        f"style={scene.get('layout_options', {}).get('sketch_style', 'line_art')}",
        f"backgrounds={bg_summary}" if bg_summary else "",
        f"objects={object_summary}" if object_summary else "",
        f"attachments={attachment_summary}" if attachment_summary else "",
        f"connectors={connector_summary}" if connector_summary else "",
        hints.get("scene_summary", ""),
        hints.get("subject_summary", ""),
        hints.get("user_added_summary", ""),
    ]
    return " | ".join(part for part in parts if part)


def scene_spec_counts(scene_spec: Dict[str, Any] | None) -> Tuple[int, int]:
    if not scene_spec:
        return 0, 0
    scene = normalize_scene_spec_v2(scene_spec)
    return len(scene.get("object_instances", []) or []), len(scene.get("connectors", []) or [])
