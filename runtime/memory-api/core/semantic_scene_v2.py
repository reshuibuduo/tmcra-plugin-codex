from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from . import semantic_scene as legacy
from .natural_layout import apply_natural_layout
from .object_sketch_backend import resolve_object_shape
from .sketch_style_spec import (
    build_stroke_style_profile,
    default_part_graph,
    default_readability_rank,
    default_region_masks,
    infer_sketch_family,
    normalize_layout_options,
    normalize_style_variant,
    summarize_region_overrides,
)
from .visual_prototypes import fallback_prototype_id, resolve_visual_prototype


CANVAS_DEFAULT = legacy.CANVAS_DEFAULT


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


FALLBACK_ASSETS: Dict[str, Dict[str, Any]] = {
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
        "label": "高塔母题",
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
}


EXTRA_ASSETS: Dict[str, Dict[str, Any]] = {
    "dog": {
        "label": "狗",
        "category": "角色",
        "silhouette_key": "dog",
        "default_size": (180, 128),
        "keywords": ["狗", "小狗", "犬", "dog"],
        "anchors": ["center", "ground"],
        "editor_visible": True,
    },
    "desk_lamp": {
        "label": "台灯",
        "category": "室内",
        "silhouette_key": "desk_lamp",
        "default_size": (120, 180),
        "keywords": ["台灯", "桌灯", "desk lamp"],
        "anchors": ["base", "center"],
        "editor_visible": True,
    },
    "cycle": {
        "label": "循环母题",
        "category": "过程",
        "silhouette_key": "cycle",
        "default_size": (180, 180),
        "keywords": ["循环", "周期", "回路", "cycle"],
        "anchors": ["center"],
        "editor_visible": True,
    },
    "flow_node": {
        "label": "流程节点",
        "category": "过程",
        "silhouette_key": "flow_node",
        "default_size": (180, 120),
        "keywords": ["阶段", "步骤", "过程", "机制", "flow"],
        "anchors": ["center", "left", "right"],
        "editor_visible": True,
    },
    "energy_wave": {
        "label": "能量波",
        "category": "过程",
        "silhouette_key": "energy_wave",
        "default_size": (180, 100),
        "keywords": ["能量", "热", "光", "传播", "波"],
        "anchors": ["center", "left", "right"],
        "editor_visible": True,
    },
    "vapor": {
        "label": "蒸汽母题",
        "category": "过程",
        "silhouette_key": "vapor",
        "default_size": (160, 180),
        "keywords": ["蒸发", "蒸汽", "水汽", "vapor"],
        "anchors": ["center", "sky"],
        "editor_visible": True,
    },
    "switch": {
        "label": "开关",
        "category": "电路",
        "silhouette_key": "switch",
        "default_size": (170, 110),
        "keywords": ["开关", "按钮", "按键", "拨动", "toggle", "switch"],
        "anchors": ["left", "right", "center"],
        "editor_visible": True,
    },
}


ASSET_LIBRARY: Dict[str, Dict[str, Any]] = _copy(legacy.ASSET_LIBRARY)
ASSET_LIBRARY.update(FALLBACK_ASSETS)
ASSET_LIBRARY.update(EXTRA_ASSETS)


EDITOR_LIBRARY_ORDER = [
    *[item for item in legacy.EDITOR_LIBRARY_ORDER if item not in {"tower", "capsule", "branch", "module", "blob"}],
    "dog",
    "desk_lamp",
    "cycle",
    "flow_node",
    "energy_wave",
    "vapor",
    "switch",
    "tower",
    "branch",
    "module",
    "blob",
]


ATTACHMENT_COMPATIBILITY: Dict[str, List[str]] = {
    "window": ["building", "house", "tower"],
    "door": ["building", "house", "tower"],
    "cloud": ["bg_sky"],
    "sun": ["bg_sky"],
    "car": ["road"],
    "chair": ["table"],
    "desk_lamp": ["table"],
    "street_lamp": ["road", "building", "house"],
}


NOISE_EXACT = {
    "一个",
    "一种",
    "一些",
    "东西",
    "内容",
    "图片",
    "图像",
    "画面",
    "场景",
    "草图",
    "示意图",
    "问题",
    "答案",
    "原因",
    "结果",
    "条件",
    "关系",
    "结构",
    "对象",
    "主体",
    "元素",
}


NOISE_CONTAINS = {
    "这张图",
    "该图片",
    "这个场景",
    "这个东西",
    "一张图",
    "某个东西",
    "visualized intent",
}


def _sync_legacy() -> None:
    legacy.ASSET_LIBRARY = ASSET_LIBRARY
    legacy.EDITOR_LIBRARY_ORDER = EDITOR_LIBRARY_ORDER
    legacy.pick_asset_key = pick_asset_key


def _normalized_text(text: Any) -> str:
    value = _clean_text(text).lower()
    value = re.sub(r"[\s\-_·•,.;:!?！？、，。；：“”\"'（）()【】\[\]<>《》/\\]+", "", value)
    return value


def _contains_any(text: str, tokens: tuple[str, ...] | list[str]) -> bool:
    return any(token and token in text for token in tokens)


_QUERY_NEGATION_PATTERNS = (
    re.compile(r"(不要|别|避免|勿|去掉|移除|不是|并非|非)\s*$"),
    re.compile(r"(?:^|[\s,;:()\[\]{}'\"/_-])(no|not|avoid|avoiding|without|exclude|excluding|remove|minus)\s*$"),
)


def _iter_query_token_hits(text: str, lowered: str, marker: str):
    cleaned = _clean_text(marker)
    if not cleaned:
        return
    ascii_phrase = all(ord(ch) < 128 for ch in cleaned)
    if ascii_phrase:
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(cleaned.lower())}(?![a-z0-9])")
        for match in pattern.finditer(lowered):
            yield lowered, match.start(), match.end()
        return
    start = 0
    while True:
        index = text.find(cleaned, start)
        if index < 0:
            break
        yield text, index, index + len(cleaned)
        start = index + len(cleaned)


def _is_negated_query_hit(source: str, start: int) -> bool:
    prefix = source[max(0, start - 28):start].strip()
    if not prefix:
        return False
    return any(pattern.search(prefix) for pattern in _QUERY_NEGATION_PATTERNS)


def _contains_query_token(text: str, lowered: str, tokens: tuple[str, ...] | list[str]) -> bool:
    for token in tokens:
        marker = _clean_text(token)
        if not marker:
            continue
        for source, start, _ in _iter_query_token_hits(text, lowered, marker):
            if not _is_negated_query_hit(source, start):
                return True
    return False


def _is_switch_like_concept(text: str) -> bool:
    raw = _clean_text(text)
    lowered = raw.lower()
    return _contains_any(raw, ("开关", "按钮", "按键", "拨动", "切换")) or any(
        token in lowered for token in ("switch", "toggle", "button")
    )


def _is_explicit_board_concept(text: str) -> bool:
    raw = _clean_text(text)
    lowered = raw.lower()
    return _contains_any(raw, ("电路板", "开发板", "主板", "面包板", "板子", "控制板")) or any(
        token in lowered for token in ("board", "breadboard", "arduino", "pcb", "controller board", "control board")
    )


def _normalize_schematic_connector_label(label: Any) -> str:
    raw = _clean_text(label)
    if not raw:
        return "连接"
    return "连接"


def _is_generic_process_stage(text: str) -> bool:
    raw = _clean_text(text)
    return _contains_any(raw, ("过程", "机制", "阶段", "步骤", "输入", "输出", "结果", "原因", "条件", "变化", "转化"))


def _is_noise_concept(text: Any) -> bool:
    raw = _clean_text(text)
    if not raw:
        return True
    normalized = _normalized_text(raw)
    if normalized in NOISE_EXACT:
        return True
    if any(token in raw for token in NOISE_CONTAINS):
        return True
    if len(normalized) <= 2 and normalized in {"它", "他", "她", "其", "该", "此", "某"}:
        return True
    return False


def get_asset_definition(asset_key: str) -> Dict[str, Any]:
    return ASSET_LIBRARY.get(asset_key, ASSET_LIBRARY["generic_object"])


def _asset_scene_type(asset_key: str, fallback_scene_type: str) -> str:
    key = _clean_text(asset_key)
    if key in {"cycle", "flow_node", "energy_wave", "vapor", "leaf", "raindrop", "airplane", "cell"}:
        return "process"
    if key in {"battery", "led", "resistor", "capacitor", "diode", "board", "module", "branch", "switch"}:
        return "schematic"
    return fallback_scene_type


def _preferred_scene_style(scene_type: str) -> str:
    return "clean_line" if _clean_text(scene_type) == "scene" else "scribble_line"


def _preferred_variant_id_for_style(variants: List[Dict[str, Any]], style_variant: str) -> str:
    normalized_style = normalize_style_variant(style_variant or "")
    if not normalized_style:
        return ""
    for variant in variants:
        styles = {normalize_style_variant(item) for item in (variant.get("style_variants") or [])}
        if normalized_style in styles and normalize_style_variant(variant.get("default_style")) == normalized_style:
            return str(variant.get("id") or "")
    for variant in variants:
        styles = {normalize_style_variant(item) for item in (variant.get("style_variants") or [])}
        if normalized_style in styles:
            return str(variant.get("id") or "")
    return ""


def build_editor_asset_library() -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for asset_key in EDITOR_LIBRARY_ORDER:
        asset = get_asset_definition(asset_key)
        scene_type = _asset_scene_type(asset_key, "scene")
        prototype = resolve_visual_prototype(asset_key, asset.get("label", asset_key), scene_type)
        requested_style = normalize_style_variant(_preferred_scene_style(scene_type) if scene_type == "scene" else (prototype.get("style_variant") or _preferred_scene_style(scene_type)))
        sketch_variant = resolve_object_shape(
            asset_key,
            concept=asset.get("label", asset_key),
            scene_type=scene_type,
            style_variant=requested_style,
            stroke_seed=asset_key,
        )
        category = asset.get("category", "其他")
        grouped.setdefault(category, []).append(
            {
                "asset_key": asset_key,
                "label": asset.get("label", asset_key),
                "silhouette_key": asset.get("silhouette_key", asset_key),
                "default_width": asset.get("default_size", (120, 120))[0],
                "default_height": asset.get("default_size", (120, 120))[1],
                "anchors": asset.get("anchors", []),
                "prototype_id": prototype.get("prototype_id", asset_key),
                "visual_family": prototype.get("visual_family", "scene_object"),
                "style_variant": sketch_variant.get("style_variant", requested_style),
                "part_slots": prototype.get("part_slots", []),
                "shape_recipe": sketch_variant.get("shape_recipe") or prototype.get("shape_recipe", {}),
                "shape_variant_id": sketch_variant.get("shape_variant_id", f"{asset_key}:rule_base"),
                "shape_recipe_source": sketch_variant.get("shape_recipe_source", "rule_prototype"),
                "sketch_backend": sketch_variant.get("sketch_backend", "rule"),
                "shape_confidence": float(sketch_variant.get("shape_confidence", 0.7) or 0.7),
                "available_shape_variants": sketch_variant.get("available_shape_variants", []),
                "render_representation": sketch_variant.get("render_representation", "shape_recipe"),
                "stroke_variant_id": sketch_variant.get("stroke_variant_id", sketch_variant.get("shape_variant_id", f"{asset_key}:rule_base")),
                "stroke_payload": sketch_variant.get("stroke_payload", []),
                "stroke_payload_source": sketch_variant.get("stroke_payload_source", sketch_variant.get("shape_recipe_source", "rule_prototype")),
                "stroke_render_profile": sketch_variant.get("stroke_render_profile", sketch_variant.get("stroke_style_profile", {})),
                "part_graph": sketch_variant.get("part_graph") or default_part_graph(asset_key, scene_type=scene_type),
                "region_masks": sketch_variant.get("region_masks") or default_region_masks(asset_key, scene_type=scene_type),
                "stroke_style_profile": sketch_variant.get("stroke_style_profile")
                or build_stroke_style_profile(
                    asset_key,
                    scene_type=scene_type,
                    style_variant=sketch_variant.get("style_variant", requested_style),
                    sketch_family=sketch_variant.get("sketch_family", infer_sketch_family(asset_key, scene_type=scene_type)),
                ),
                "readability_rank": int(sketch_variant.get("readability_rank", default_readability_rank(asset_key, scene_type=scene_type)) or default_readability_rank(asset_key, scene_type=scene_type)),
                "sketch_family": sketch_variant.get("sketch_family", infer_sketch_family(asset_key, scene_type=scene_type)),
                "attach_to": ATTACHMENT_COMPATIBILITY.get(asset_key, []),
            }
        )
    return [{"category": category, "items": items} for category, items in grouped.items()]


def _process_asset_from_concept(text: str) -> str:
    if _contains_any(text, ("蒸发", "蒸汽", "水汽", "雾气", "气化")):
        return "vapor"
    if _contains_any(text, ("雨", "降水", "降雨", "水滴")):
        return "raindrop"
    if _contains_any(text, ("云", "凝结")):
        return "cloud"
    if _contains_any(text, ("太阳", "日照", "光合作用", "叶", "叶片", "植物")):
        return "leaf" if _contains_any(text, ("光合作用", "叶", "叶片", "植物")) else "sun"
    if _contains_any(text, ("能量", "热量", "热", "光照", "光", "辐射")):
        return "sun"
    if _contains_any(text, ("飞机", "机翼", "升力", "飞行")):
        return "airplane"
    if _contains_any(text, ("细胞", "线粒体", "叶绿体")):
        return "cell"
    return ""


def _schematic_asset_from_concept(text: str) -> str:
    if _contains_any(text, ("电池", "电源", "供电", "battery", "power")):
        return "battery"
    if _contains_any(text, ("led", "发光二极管", "灯泡")):
        return "led"
    if _contains_any(text, ("电阻", "resistor")):
        return "resistor"
    if _contains_any(text, ("电容", "capacitor")):
        return "capacitor"
    if _contains_any(text, ("二极管", "diode")):
        return "diode"
    if _is_switch_like_concept(text):
        return "switch"
    if _contains_any(text, ("电路板", "开发板", "主板", "board", "arduino")):
        return "board"
    return ""


def pick_asset_key(concept: str, scene_type: str = "scene") -> str:
    concept_text = _clean_text(concept)
    lowered = concept_text.lower()
    normalized = _normalized_text(concept_text)

    english_map = (
        "dog",
        "desk_lamp",
        "window",
        "door",
        "building",
        "house",
        "tree",
        "cloud",
        "sun",
        "car",
        "road",
        "person",
        "chair",
        "table",
        "cycle",
        "flow_node",
        "energy_wave",
        "vapor",
        "battery",
        "led",
        "switch",
        "resistor",
        "capacitor",
        "diode",
        "board",
        "airplane",
        "leaf",
        "raindrop",
        "cell",
    )
    for asset_key in english_map:
        if asset_key in lowered:
            return asset_key

    chinese_token_map = [
        ("dog", ("狗", "小狗", "柯基", "犬")),
        ("desk_lamp", ("台灯", "桌灯")),
        ("street_lamp", ("路灯", "街灯", "灯杆")),
        ("building", ("楼房", "大楼", "建筑", "高楼")),
        ("house", ("房子", "房屋", "住宅")),
        ("window", ("窗户", "窗", "玻璃窗")),
        ("door", ("门", "大门", "房门")),
        ("tree", ("树", "树木", "树林")),
        ("cloud", ("云", "云朵")),
        ("sun", ("太阳", "阳光", "日光")),
        ("car", ("汽车", "轿车", "车辆")),
        ("road", ("道路", "路面", "街道", "公路")),
        ("person", ("人物", "人", "行人", "学生", "孩子")),
        ("table", ("桌子", "桌")),
        ("chair", ("椅子", "椅")),
        ("cycle", ("循环", "周期", "回路")),
        ("flow_node", ("步骤", "阶段", "流程节点")),
        ("energy_wave", ("能量", "热量", "波", "辐射", "传播")),
        ("vapor", ("蒸发", "蒸汽", "水汽", "雾气")),
        ("battery", ("电池", "电源")),
        ("led", ("LED", "发光二极管", "灯泡")),
        ("switch", ("开关", "按钮", "按键", "拨动")),
        ("resistor", ("电阻",)),
        ("capacitor", ("电容",)),
        ("diode", ("二极管",)),
        ("board", ("电路板", "开发板", "主板", "面包板", "板子", "控制板")),
        ("airplane", ("飞机", "机翼", "飞行器")),
        ("leaf", ("叶子", "树叶", "叶片")),
        ("raindrop", ("雨滴", "水滴")),
        ("cell", ("细胞",)),
    ]
    for asset_key, tokens in chinese_token_map:
        if any(token in concept_text for token in tokens):
            return asset_key

    if any(token in concept_text for token in ("细胞核", "细胞质", "细胞膜")):
        return "generic_circle"

    for asset_key, asset in ASSET_LIBRARY.items():
        for keyword in asset.get("keywords", []):
            if keyword and keyword.lower() in lowered:
                return asset_key

    if scene_type == "process":
        process_asset = _process_asset_from_concept(concept_text)
        if process_asset:
            return process_asset

    if scene_type == "schematic":
        schematic_asset = _schematic_asset_from_concept(concept_text)
        if schematic_asset:
            return schematic_asset

    if scene_type == "scene" and _contains_any(normalized, ("塔", "楼", "大厦", "烟囱", "柱")):
        return "tower"

    prototype_id = fallback_prototype_id(concept_text, scene_type)
    if prototype_id in ASSET_LIBRARY:
        return prototype_id
    if scene_type == "process":
        return "flow_node"
    if scene_type == "schematic":
        return "module"
    return "blob"


def _sanitize_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    objects = []
    removed_ids: set[str] = set()
    for obj in scene.get("object_instances", []) or []:
        item = _copy(obj)
        concept = _clean_text(item.get("concept"))
        source = _clean_text(item.get("source"))
        if not concept and source != "user":
            removed_ids.add(str(item.get("id") or ""))
            continue
        if source != "user" and _is_noise_concept(concept):
            removed_ids.add(str(item.get("id") or ""))
            continue
        objects.append(item)
    scene["object_instances"] = objects
    if removed_ids:
        scene["attachments"] = [
            item
            for item in scene.get("attachments", []) or []
            if str(item.get("host_id") or "") not in removed_ids and str(item.get("child_id") or "") not in removed_ids
        ]
        scene["connectors"] = [
            item
            for item in scene.get("connectors", []) or []
            if str(item.get("from_id") or "") not in removed_ids and str(item.get("to_id") or "") not in removed_ids
        ]
    concept_order = [_clean_text(item) for item in scene.get("concept_order", []) or []]
    scene["concept_order"] = [item for item in concept_order if item and not _is_noise_concept(item)]
    if not scene["concept_order"]:
        scene["concept_order"] = [item.get("concept", "") for item in objects if _clean_text(item.get("concept"))]
    return scene


def _repair_asset_key(asset_key: str, concept: str, scene_type: str) -> str:
    current = _clean_text(asset_key)
    if scene_type == "process" and current in {"capsule", "generic_panel", "generic_object", "blob"}:
        current = ""
    if scene_type == "schematic" and current in {"generic_panel", "generic_object", "blob"}:
        current = ""
    if scene_type == "scene" and current in {"generic_panel"}:
        current = ""
    return current or pick_asset_key(concept, scene_type)


def _rebalance_generic_objects(scene: Dict[str, Any]) -> Dict[str, Any]:
    scene_type = _clean_text(scene.get("layout_options", {}).get("scene_type") or "scene")
    for obj in scene.get("object_instances", []) or []:
        concept = _clean_text(obj.get("concept"))
        repaired = _repair_asset_key(_clean_text(obj.get("asset_key")), concept, scene_type)
        obj["asset_key"] = repaired
        obj["silhouette_key"] = get_asset_definition(repaired).get("silhouette_key", repaired)
    return scene


def _apply_visual_metadata(scene: Dict[str, Any]) -> Dict[str, Any]:
    scene_type = _clean_text(scene.get("layout_options", {}).get("scene_type") or "scene")
    objects: List[Dict[str, Any]] = []
    for obj in scene.get("object_instances", []) or []:
        item = _copy(obj)
        concept = _clean_text(item.get("concept"))
        asset_key = _repair_asset_key(_clean_text(item.get("asset_key")), concept, scene_type)
        prototype = resolve_visual_prototype(
            _clean_text(item.get("prototype_id") or asset_key),
            concept,
            scene_type,
        )
        raw_item_style = normalize_style_variant(_clean_text(item.get("style_variant")))
        if scene_type == "scene" and raw_item_style == "scribble_line" and _clean_text(item.get("source")) != "user":
            raw_item_style = ""
        style_variant = normalize_style_variant(raw_item_style or (_preferred_scene_style(scene_type) if scene_type == "scene" else (prototype.get("style_variant") or _preferred_scene_style(scene_type))))
        preferred_variant_id = _clean_text(item.get("shape_variant_id"))
        if style_variant == "clean_line" and "clean_line" not in preferred_variant_id:
            preferred_variant_id = ""
        sketch_variant = resolve_object_shape(
            asset_key,
            concept=concept,
            scene_type=scene_type,
            preferred_variant_id=preferred_variant_id,
            style_variant=style_variant,
            stroke_seed=item.get("stroke_seed") or item.get("id") or concept or asset_key,
        )
        sketch_family = item.get("sketch_family") or sketch_variant.get("sketch_family") or infer_sketch_family(asset_key, scene_type=scene_type)
        resolved_style = normalize_style_variant(sketch_variant.get("style_variant") or style_variant or _preferred_scene_style(scene_type))
        preferred_style_variant_id = _preferred_variant_id_for_style(
            sketch_variant.get("available_shape_variants") or [],
            resolved_style,
        )
        item["asset_key"] = asset_key
        item["silhouette_key"] = item.get("silhouette_key") or get_asset_definition(asset_key).get("silhouette_key", asset_key)
        item["prototype_id"] = prototype.get("prototype_id", asset_key)
        item["visual_family"] = item.get("visual_family") or prototype.get("visual_family", "scene_object")
        item["shape_recipe"] = item.get("shape_recipe") or sketch_variant.get("shape_recipe") or prototype.get("shape_recipe", {})
        item["part_slots"] = item.get("part_slots") or prototype.get("part_slots", [])
        item["style_variant"] = resolved_style
        item["shape_variant_id"] = preferred_style_variant_id or sketch_variant.get("shape_variant_id", f"{asset_key}:rule_base")
        item["shape_recipe_source"] = item.get("shape_recipe_source") or sketch_variant.get("shape_recipe_source", "rule_prototype")
        item["sketch_backend"] = item.get("sketch_backend") or sketch_variant.get("sketch_backend", "rule")
        item["stroke_seed"] = str(item.get("stroke_seed") or sketch_variant.get("stroke_seed") or item.get("id") or concept or asset_key)
        item["shape_confidence"] = float(item.get("shape_confidence") or sketch_variant.get("shape_confidence") or 0.7)
        item["available_shape_variants"] = item.get("available_shape_variants") or sketch_variant.get("available_shape_variants", [])
        item["render_representation"] = item.get("render_representation") or sketch_variant.get("render_representation", "shape_recipe")
        item["stroke_variant_id"] = item.get("stroke_variant_id") or sketch_variant.get("stroke_variant_id") or item["shape_variant_id"]
        item["stroke_payload"] = item.get("stroke_payload") or sketch_variant.get("stroke_payload") or []
        item["stroke_payload_source"] = item.get("stroke_payload_source") or sketch_variant.get("stroke_payload_source") or item["shape_recipe_source"]
        item["stroke_render_profile"] = item.get("stroke_render_profile") or sketch_variant.get("stroke_render_profile") or sketch_variant.get("stroke_style_profile") or {}
        item["part_graph"] = item.get("part_graph") or sketch_variant.get("part_graph") or default_part_graph(asset_key, scene_type=scene_type)
        item["region_masks"] = item.get("region_masks") or sketch_variant.get("region_masks") or default_region_masks(asset_key, scene_type=scene_type)
        item["stroke_style_profile"] = item.get("stroke_style_profile") or sketch_variant.get("stroke_style_profile") or build_stroke_style_profile(
            asset_key,
            scene_type=scene_type,
            style_variant=resolved_style,
            sketch_family=sketch_family,
        )
        item["readability_rank"] = int(item.get("readability_rank") or sketch_variant.get("readability_rank") or default_readability_rank(asset_key, scene_type=scene_type))
        item["sketch_family"] = sketch_family
        item["region_overrides"] = item.get("region_overrides") if isinstance(item.get("region_overrides"), dict) else {}
        objects.append(item)
    scene["object_instances"] = objects
    return scene


def _soften_backgrounds(scene: Dict[str, Any]) -> Dict[str, Any]:
    scene_type = _clean_text(scene.get("layout_options", {}).get("scene_type") or "scene")
    updated: List[Dict[str, Any]] = []
    for index, layer in enumerate(scene.get("background_layers", []) or [], start=1):
        item = _copy(layer)
        if scene_type == "process" and item.get("type") == "panel":
            item["type"] = "process_band"
            item["source"] = item.get("source", "auto")
            item["z_index"] = -22 + index
        elif scene_type == "schematic" and item.get("type") == "board":
            item["type"] = "board"
            item["z_index"] = -18
        updated.append(item)
    scene["background_layers"] = updated
    return scene


def _upgrade_layout_defaults(scene: Dict[str, Any], sketch_options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    sketch_options = sketch_options or {}
    layout = scene.setdefault("layout_options", {})
    layout["mode"] = "semantic_composition"
    layout["sketch_style"] = normalize_style_variant(_clean_text(sketch_options.get("sketch_style") or layout.get("sketch_style") or _preferred_scene_style(_clean_text(layout.get("scene_type") or "scene"))))
    layout["show_labels"] = bool(sketch_options.get("show_labels", layout.get("show_labels", False)))
    layout["show_grid"] = bool(sketch_options.get("show_grid", layout.get("show_grid", True)))
    layout["show_guides"] = bool(sketch_options.get("show_guides", layout.get("show_guides", False)))
    layout["node_scale"] = float(sketch_options.get("node_scale", layout.get("node_scale", 1.0)))
    layout["spacing_scale"] = float(sketch_options.get("spacing_scale", layout.get("spacing_scale", 1.0)))
    layout["composition_mode"] = layout.get("scene_type", "scene")
    layout["layout_engine"] = _clean_text(sketch_options.get("layout_engine") or layout.get("layout_engine") or "auto") or "auto"
    layout["layout_candidate_count"] = int(sketch_options.get("layout_candidate_count", layout.get("layout_candidate_count", 4)) or 4)
    layout["layout_manual_override"] = bool(layout.get("layout_manual_override", False))
    layout.update(normalize_layout_options(layout, sketch_options))
    return scene


def _upgrade_render_hints(scene: Dict[str, Any]) -> Dict[str, Any]:
    scene_type = _clean_text(scene.get("layout_options", {}).get("scene_type") or "scene")
    hints = dict(scene.get("render_hints") or {})
    objects = scene.get("object_instances", []) or []
    subjects = [item.get("concept", "") for item in objects if item.get("role") in {"subject", "focus", "core_subject"}][:3]
    user_items = [item.get("concept", "") for item in objects if item.get("source") == "user"][:6]
    families = [item.get("prototype_id", item.get("asset_key", "")) for item in objects[:8]]
    edit_summary = summarize_region_overrides(scene)
    hints["scene_summary"] = hints.get("scene_summary") or ("、".join(filter(None, subjects)) if subjects else "语义构图")
    hints["subject_summary"] = "主体: " + ("、".join(filter(None, subjects)) if subjects else "未显式主体")
    hints["style_summary"] = f"风格: {scene.get('layout_options', {}).get('sketch_style', 'scribble_line')} | 类型: {scene_type}"
    hints["prototype_summary"] = "原型: " + ("、".join(filter(None, families)) if families else "无")
    hints["user_added_summary"] = "用户新增: " + ("、".join(filter(None, user_items)) if user_items else "无")
    hints["edit_summary"] = "局部编辑: " + (edit_summary if edit_summary else "无")
    hints["sketch_view_mode"] = scene.get("layout_options", {}).get("sketch_view_mode", "structure")
    hints["annotation_level"] = scene.get("layout_options", {}).get("annotation_level", "light")
    scene["render_hints"] = hints
    return scene


def _infer_scene_type_from_query(query: str) -> str:
    text = _clean_text(query)
    lowered = text.lower()

    def score(*groups: tuple[tuple[str, ...], int]) -> int:
        total = 0
        for tokens, weight in groups:
            if _contains_query_token(text, lowered, tokens):
                total += weight
        return total

    scene_score = score(
        (("场景", "整图", "街景", "室内", "户外", "房间", "街道", "城市", "自然场景", "scene", "whole-scene", "street-view", "indoor", "outdoor", "room", "city", "landscape"), 2),
        (("房子", "建筑", "桌子", "椅子", "台灯", "树", "人", "汽车", "house", "building", "table", "chair", "desk lamp", "tree", "person", "car"), 1),
    )
    schematic_score = score(
        (("电路", "原理图", "线路图", "串联", "并联", "示意图", "circuit", "schematic", "wiring"), 4),
        (("LED", "电阻", "电容", "二极管", "发光二极管", "led", "resistor", "capacitor", "diode"), 3),
        (("电池", "电源", "battery"), 2),
        (("开发板", "电路板", "主板", "arduino", "board"), 2),
        (("开关", "按钮", "按键", "拨动", "switch", "toggle"), 1),
    )
    process_score = score(
        (("为什么", "形成", "原理", "机制", "过程", "循环", "如何产生", "怎么产生", "作用", "机理"), 3),
        (("process", "workflow", "cycle", "mechanism", "explanation", "how it works"), 3),
        (("evaporation", "condensation", "rain", "rainfall", "photosynthesis", "water cycle", "plant-energy"), 4),
        (("升力", "气流", "空气流动", "空气动力", "lift", "airflow", "aerodynamic"), 4),
    )
    if _contains_query_token(text, lowered, ("飞机", "飞行", "airplane", "aircraft", "flight")) and _contains_query_token(
        text,
        lowered,
        ("原理", "机制", "解释", "升力", "气流", "空气流动", "mechanism", "explanation", "lift", "airflow"),
    ):
        process_score += 3
    if _contains_query_token(text, lowered, ("叶", "叶子", "leaf")) and _contains_query_token(
        text,
        lowered,
        ("太阳", "阳光", "sun", "sunlight"),
    ) and _contains_query_token(
        text,
        lowered,
        ("能量", "光合作用", "光照", "energy", "plant-energy", "photosynthesis"),
    ):
        process_score += 3

    if schematic_score >= max(process_score + 2, scene_score + 2, 4):
        return "schematic"
    if process_score >= max(schematic_score + 1, scene_score + 1, 3):
        return "process"
    return "scene"


def _collect_query_assets(query: str, scene_type: str) -> List[str]:
    text = _clean_text(query)
    lowered = text.lower()
    keys: List[str] = []
    def add(key: str) -> None:
        if key and key not in keys:
            keys.append(key)

    reliable_scene_tokens = [
        ("building", ("\u697c", "\u5927\u697c", "\u697c\u623f", "\u5efa\u7b51", "building", "architecture")),
        ("house", ("\u623f\u5b50", "\u5c0f\u623f\u5b50", "\u623f\u5c4b", "\u4f4f\u5b85", "house", "home", "cottage")),
        ("road", ("\u8def", "\u8857\u9053", "\u9053\u8def", "\u516c\u8def", "\u9a6c\u8def", "\u8857\u666f", "road", "street", "avenue")),
        ("car", ("\u6c7d\u8f66", "\u8f66", "\u8f66\u8f86", "car", "vehicle")),
        ("street_lamp", ("\u8def\u706f", "\u8857\u706f", "\u706f\u6746", "street lamp", "streetlight", "lamp post")),
        ("person", ("\u4eba", "\u4eba\u7269", "\u884c\u4eba", "\u5b66\u751f", "\u5b69\u5b50", "\u4e00\u4e2a\u4eba", "person", "people", "human", "pedestrian")),
        ("tree", ("\u6811", "\u6811\u6728", "\u6811\u6797", "\u4e00\u68f5\u6811", "tree", "trees")),
        ("cloud", ("\u4e91", "\u4e91\u6735", "cloud")),
        ("sun", ("\u592a\u9633", "\u9633\u5149", "sun")),
        ("dog", ("\u72d7", "\u5c0f\u72d7", "\u72ac", "dog")),
        ("table", ("\u684c", "\u684c\u5b50", "table")),
        ("chair", ("\u6905", "\u6905\u5b50", "chair")),
        ("desk_lamp", ("\u53f0\u706f", "\u684c\u706f", "desk lamp")),
    ]
    reliable_process_tokens = [
        ("sun", ("\u592a\u9633", "\u9633\u5149", "sun")),
        ("vapor", ("\u84b8\u53d1", "\u84b8\u6c7d", "\u6c34\u84b8\u6c14", "vapor")),
        ("cloud", ("\u4e91", "\u4e91\u6735", "\u51dd\u7ed3", "cloud")),
        ("raindrop", ("\u96e8", "\u96e8\u6c34", "\u964d\u96e8", "\u96e8\u6ef4", "\u6c34\u6ef4", "rain", "raindrop")),
        ("leaf", ("\u53f6", "\u53f6\u5b50", "\u690d\u7269", "leaf")),
        ("energy_wave", ("\u80fd\u91cf", "\u4f20\u64ad", "\u8f90\u5c04", "energy")),
        ("airplane", ("\u98de\u673a", "\u98de\u884c", "airplane")),
        ("cell", ("\u7ec6\u80de", "cell")),
    ]
    reliable_schematic_tokens = [
        ("battery", ("\u7535\u6c60", "\u7535\u6e90", "\u4f9b\u7535", "battery", "power")),
        ("resistor", ("\u7535\u963b", "resistor")),
        ("led", ("led", "\u53d1\u5149\u4e8c\u6781\u7ba1", "\u706f\u6ce1")),
        ("switch", ("\u5f00\u5173", "\u6309\u94ae", "\u6309\u952e", "\u62e8\u52a8", "switch", "toggle", "button")),
        ("capacitor", ("\u7535\u5bb9", "capacitor")),
        ("diode", ("\u4e8c\u6781\u7ba1", "diode")),
        ("board", ("\u5f00\u53d1\u677f", "\u7535\u8def\u677f", "\u4e3b\u677f", "\u9762\u5305\u677f", "\u677f\u5b50", "\u63a7\u5236\u677f", "arduino", "breadboard", "board")),
    ]
    if scene_type == "scene":
        for key, tokens in reliable_scene_tokens:
            if _contains_query_token(text, lowered, tokens):
                add(key)
    elif scene_type == "process":
        for key, tokens in reliable_process_tokens:
            if _contains_query_token(text, lowered, tokens):
                add(key)
    elif scene_type == "schematic":
        for key, tokens in reliable_schematic_tokens:
            if _contains_query_token(text, lowered, tokens):
                add(key)

    if scene_type == "schematic":
        for key, tokens in [
            ("battery", ("电池", "电源", "供电", "battery")),
            ("resistor", ("电阻", "resistor")),
            ("led", ("LED", "发光二极管", "灯泡", "led")),
            ("switch", ("开关", "按钮", "按键", "拨动", "switch", "toggle", "button")),
            ("capacitor", ("电容", "capacitor")),
            ("diode", ("二极管", "diode")),
            ("board", ("开发板", "电路板", "主板", "面包板", "板子", "控制板", "arduino", "breadboard", "board")),
        ]:
            if _contains_any(text, tokens) or _contains_any(lowered, tokens):
                add(key)
        if not keys and (_contains_any(text, ("电路", "结构", "信号")) or _contains_any(lowered, ("circuit", "schematic", "wiring"))):
            keys = ["battery", "switch", "resistor", "led"]
        return keys[:6]

    if scene_type == "process":
        if _contains_any(text, ("下雨", "降雨", "降水", "雨水")) or _contains_any(lowered, ("rain", "rainfall", "water cycle")):
            return ["sun", "vapor", "cloud", "raindrop"]
        if _contains_any(text, ("光合作用",)) or _contains_any(lowered, ("photosynthesis",)):
            return ["sun", "leaf", "cloud"]
        if _contains_any(text, ("飞机", "升力", "飞行")) or _contains_any(lowered, ("airplane", "flight", "lift")):
            return ["airplane", "cloud"]
        for key, tokens in [
            ("vapor", ("蒸发", "蒸汽", "水汽", "vapor")),
            ("cloud", ("云", "凝结", "cloud")),
            ("raindrop", ("雨", "降雨", "水滴", "raindrop")),
            ("sun", ("太阳", "光", "热", "sun")),
            ("leaf", ("叶", "植物", "leaf")),
            ("energy_wave", ("能量", "传播", "辐射", "energy")),
            ("airplane", ("飞机", "机翼", "airplane")),
            ("cell", ("细胞", "cell")),
        ]:
            if _contains_any(text, tokens) or _contains_any(lowered, tokens):
                add(key)
        if not keys:
            keys = ["sun", "cloud"]
        return keys[:6]

    for key, tokens in [
        ("building", ("楼", "大楼", "楼房", "建筑", "building", "architecture")),
        ("house", ("房子", "小房子", "房屋", "住宅", "house", "home", "cottage")),
        ("road", ("路", "街道", "道路", "公路", "马路", "街景", "road", "street", "avenue")),
        ("car", ("汽车", "轿车", "车辆", "车", "car", "vehicle")),
        ("street_lamp", ("路灯", "街灯", "灯杆", "street lamp", "streetlight", "lamp post")),
        ("person", ("人", "人物", "行人", "学生", "孩子", "一个人", "两个人", "person", "people", "human", "pedestrian")),
        ("tree", ("树", "树林", "树木", "一棵树", "tree", "trees")),
        ("cloud", ("云", "云朵", "cloud")),
        ("sun", ("太阳", "阳光", "sun")),
        ("dog", ("狗", "小狗", "犬", "dog")),
        ("table", ("桌", "桌子", "table")),
        ("chair", ("椅", "椅子", "chair")),
        ("desk_lamp", ("台灯", "桌灯", "desk lamp")),
    ]:
        if _contains_any(text, tokens) or _contains_any(lowered, tokens):
            add(key)
    return keys[:8]


def _simple_background_layers(scene_type: str, width: int, height: int, assets: List[str]) -> List[Dict[str, Any]]:
    if scene_type == "process":
        return [
            {"id": "bg_process_sky", "type": "sky", "label": "天空", "x": 0, "y": 0, "width": width, "height": int(height * 0.58), "z_index": -20, "source": "auto"},
            {"id": "bg_process_ground", "type": "ground", "label": "地面", "x": 0, "y": int(height * 0.58), "width": width, "height": int(height * 0.42), "z_index": -19, "source": "auto"},
        ]
    if scene_type == "schematic":
        return []
    layers = [
        {"id": "bg_sky", "type": "sky", "label": "天空", "x": 0, "y": 0, "width": width, "height": int(height * 0.54), "z_index": -20, "source": "auto"},
        {"id": "bg_ground", "type": "ground", "label": "地面", "x": 0, "y": int(height * 0.56), "width": width, "height": int(height * 0.44), "z_index": -19, "source": "auto"},
    ]
    if "road" in assets:
        layers.append({"id": "bg_road", "type": "road", "label": "道路", "x": int(width * 0.1), "y": int(height * 0.62), "width": int(width * 0.8), "height": int(height * 0.2), "z_index": -18, "source": "auto"})
    return layers


def _query_scene_object_seed(
    asset_key: str,
    *,
    scene_assets: List[str],
    index: int,
) -> Tuple[float, float, str, str]:
    has_person = "person" in scene_assets
    has_house = any(item in {"house", "building"} for item in scene_assets)
    house_on_right = True
    if has_house:
        house_index = next((pos for pos, item in enumerate(scene_assets) if item in {"house", "building"}), 0)
        house_on_right = (house_index % 2) == 0
    if asset_key == "road":
        return 0.5, 0.78, "environment", "foreground"
    if asset_key == "sun":
        return 0.16 + 0.18 * (index % 3), 0.12, "detail", "background"
    if asset_key == "cloud":
        return 0.28 + 0.18 * (index % 3), 0.18, "detail", "background"
    if asset_key in {"person", "dog"}:
        return 0.48, 0.66, "subject", "foreground"
    if asset_key == "car":
        return 0.58, 0.72, "support", "foreground"
    if asset_key in {"house", "building"}:
        return (0.72 if house_on_right else 0.28), 0.5, ("support" if has_person else "subject"), "midground"
    if asset_key == "tree":
        if has_house:
            return (0.22 if house_on_right else 0.78), 0.58, "support", "midground"
        return (0.26 if (index % 2 == 0) else 0.74), 0.58, "support", "midground"
    if asset_key == "street_lamp":
        return (0.12 if house_on_right else 0.88), 0.62, "detail", "foreground"
    if asset_key in {"table", "chair", "desk_lamp"}:
        return 0.5 + (0.08 if index % 2 else -0.08), 0.6, ("subject" if not has_person else "detail"), "midground"
    fallback_slots = [
        (0.32, 0.56, "subject", "midground"),
        (0.68, 0.52, "support", "midground"),
        (0.22, 0.62, "detail", "foreground"),
        (0.78, 0.26, "detail", "background"),
        (0.18, 0.24, "detail", "background"),
    ]
    return fallback_slots[min(index, len(fallback_slots) - 1)]


def _query_process_object_seed(asset_key: str, index: int) -> Tuple[float, float, str, str]:
    slots = {
        "sun": (0.16, 0.16, "support", "background"),
        "cloud": (0.58, 0.22, "subject", "background"),
        "vapor": (0.34, 0.48, "detail", "midground"),
        "raindrop": (0.64, 0.48, "detail", "midground"),
        "leaf": (0.54, 0.74, "subject", "foreground"),
        "airplane": (0.48, 0.26, "subject", "background"),
        "cell": (0.50, 0.58, "subject", "midground"),
        "energy_wave": (0.28, 0.34, "detail", "background"),
    }
    fallback_slots = [
        (0.22, 0.58, "subject", "midground"),
        (0.50, 0.34, "support", "background"),
        (0.72, 0.58, "detail", "midground"),
        (0.38, 0.74, "detail", "foreground"),
    ]
    return slots.get(asset_key, fallback_slots[min(index, len(fallback_slots) - 1)])


def _query_schematic_object_seed(asset_key: str, index: int) -> Tuple[float, float, str, str]:
    slots = {
        "battery": (0.18, 0.54, "subject", "midground"),
        "switch": (0.42, 0.34, "subject", "foreground"),
        "resistor": (0.50, 0.54, "subject", "midground"),
        "led": (0.78, 0.54, "subject", "midground"),
        "capacitor": (0.48, 0.76, "detail", "midground"),
        "diode": (0.66, 0.34, "detail", "midground"),
        "board": (0.50, 0.54, "support", "background"),
    }
    fallback_slots = [
        (0.24, 0.54, "subject", "midground"),
        (0.50, 0.34, "subject", "foreground"),
        (0.76, 0.54, "subject", "midground"),
        (0.50, 0.76, "detail", "midground"),
    ]
    return slots.get(asset_key, fallback_slots[min(index, len(fallback_slots) - 1)])


def _compose_query_only_scene(query: str, canvas_size: Tuple[int, int], sketch_options: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    scene_type = _infer_scene_type_from_query(query)
    assets = _collect_query_assets(query, scene_type)
    if not assets:
        return None
    width, height = canvas_size
    objects: List[Dict[str, Any]] = []
    connectors: List[Dict[str, Any]] = []
    backgrounds = _simple_background_layers(scene_type, width, height, assets)

    object_assets = [item for item in assets if not (scene_type == "scene" and item == "road")]
    for index, asset_key in enumerate(object_assets):
        asset = get_asset_definition(asset_key)
        default_width, default_height = asset.get("default_size", (160, 120))
        if scene_type == "scene":
            px, py, role, depth = _query_scene_object_seed(
                asset_key,
                scene_assets=object_assets,
                index=index,
            )
            if asset_key == "road":
                default_width = int(width * 0.72)
                default_height = int(height * 0.18)
        elif scene_type == "process":
            px, py, role, depth = _query_process_object_seed(asset_key, index)
        else:
            px, py, role, depth = _query_schematic_object_seed(asset_key, index)
            if asset_key == "board":
                default_width = int(width * 0.58)
                default_height = int(height * 0.44)

        obj_width = int(default_width if scene_type != "schematic" else default_width * 0.92)
        obj_height = int(default_height if scene_type != "schematic" else default_height * 0.92)
        objects.append(
            {
                "id": f"obj_{index + 1}",
                "concept": asset.get("label", asset_key),
                "asset_key": asset_key,
                "source": "auto",
                "role": role,
                "depth_band": depth,
                "x": int(width * px - obj_width / 2),
                "y": int(height * py - obj_height / 2),
                "width": obj_width,
                "height": obj_height,
                "rotation": 0,
                "scale": 1,
                "z_index": 20 + index,
                "editable": True,
            }
        )

    if scene_type == "process":
        connectors = []
    elif scene_type == "schematic":
        for index in range(len(objects) - 1):
            connectors.append(
                {
                    "id": f"conn_{index + 1}",
                    "type": "wire",
                    "from_id": objects[index]["id"],
                    "to_id": objects[index + 1]["id"],
                    "label": "连接",
                    "visible": True,
                }
            )

    scene = {
        "version": 2,
        "canvas_size": {"width": width, "height": height},
        "layout_options": {
            "scene_type": scene_type,
            "composition_mode": scene_type,
            "sketch_style": _clean_text((sketch_options or {}).get("sketch_style") or _preferred_scene_style(scene_type)),
            "show_grid": bool((sketch_options or {}).get("show_grid", True)),
            "show_labels": bool((sketch_options or {}).get("show_labels", False)),
            "show_guides": bool((sketch_options or {}).get("show_guides", False)),
            "layout_engine": _clean_text((sketch_options or {}).get("layout_engine") or "auto") or "auto",
            "layout_candidate_count": int((sketch_options or {}).get("layout_candidate_count", 4) or 4),
        },
        "background_layers": backgrounds,
        "object_instances": objects,
        "attachments": [],
        "connectors": connectors,
        "render_hints": {
            "scene_summary": query[:48],
        },
        "concept_order": [item["concept"] for item in objects],
    }
    return apply_natural_layout(scene, sketch_options)


def _prefer_sd_semantic_upstream(sketch_options: Dict[str, Any] | None = None) -> bool:
    backend = _clean_text((sketch_options or {}).get("sketch_backend")).lower()
    return backend in {"sd", "sketch_v2"}


def _scene_query_asset_gap(scene: Dict[str, Any], query: str) -> Tuple[str, List[str], List[str], List[str], int]:
    scene_type = _clean_text(scene.get("layout_options", {}).get("scene_type") or _infer_scene_type_from_query(query))
    query_assets = _collect_query_assets(query, scene_type)
    current_assets = [str(item.get("asset_key") or "") for item in scene.get("object_instances", []) or []]
    missing_query_assets = [item for item in query_assets if item not in current_assets]
    generic_assets = {"module", "branch", "road", "blob", "generic_object", "generic_panel", "flow_node", "capsule"}
    generic_count = sum(1 for item in current_assets if item in generic_assets)
    return scene_type, query_assets, current_assets, missing_query_assets, generic_count


def _scene_needs_query_boost(
    scene: Dict[str, Any],
    query: str,
    sketch_options: Dict[str, Any] | None = None,
) -> bool:
    scene_type, query_assets, current_assets, missing_query_assets, generic_count = _scene_query_asset_gap(scene, query)
    if not query_assets:
        return False
    if not current_assets:
        return True
    if _prefer_sd_semantic_upstream(sketch_options):
        strong_scene_assets = {
            "building",
            "house",
            "road",
            "car",
            "street_lamp",
            "person",
            "tree",
            "dog",
            "table",
            "chair",
            "desk_lamp",
        }
        if len(current_assets) <= 1 and len(query_assets) >= 2 and bool(missing_query_assets):
            return True
        if scene_type == "scene" and generic_count > 0 and len(query_assets) >= 2:
            return True
        if scene_type == "process":
            explicit_process_assets = {"sun", "vapor", "cloud", "raindrop", "leaf", "airplane", "cell"}
            if any(item in {"cycle", "flow_node", "branch"} for item in current_assets):
                return True
            if explicit_process_assets.intersection(query_assets) and len(explicit_process_assets.intersection(current_assets)) < len(explicit_process_assets.intersection(query_assets)):
                return True
        if scene_type == "schematic":
            explicit_schematic_assets = {"battery", "resistor", "led", "capacitor", "diode", "board", "switch"}
            if any(item in {"module", "branch", "road"} for item in current_assets):
                return True
            if explicit_schematic_assets.intersection(query_assets) and len(explicit_schematic_assets.intersection(current_assets)) < len(explicit_schematic_assets.intersection(query_assets)):
                return True
        if generic_count >= max(1, len(current_assets) - 1) and bool(missing_query_assets):
            return True
        if strong_scene_assets.intersection(missing_query_assets) and len(missing_query_assets) >= max(1, len(query_assets) // 2):
            return True
    if scene_type == "schematic":
        specific_assets = {"battery", "resistor", "led", "capacitor", "diode", "board", "switch"}
        return bool(specific_assets.intersection(query_assets)) and bool(missing_query_assets)
    if scene_type == "process":
        return generic_count >= max(1, len(current_assets) // 2) and bool(missing_query_assets)
    return generic_count == len(current_assets) and bool(missing_query_assets)


def _strip_sd_symbolic_objects(scene: Dict[str, Any], sketch_options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not _prefer_sd_semantic_upstream(sketch_options):
        return scene

    scene_type = _clean_text(scene.get("layout_options", {}).get("scene_type") or "scene")
    width = int(scene.get("canvas_size", {}).get("width", CANVAS_DEFAULT[0]) or CANVAS_DEFAULT[0])
    height = int(scene.get("canvas_size", {}).get("height", CANVAS_DEFAULT[1]) or CANVAS_DEFAULT[1])
    kept_objects: List[Dict[str, Any]] = []
    kept_ids: set[str] = set()
    removed_ids: set[str] = set()

    for obj in scene.get("object_instances", []) or []:
        item = _copy(obj)
        object_id = _clean_text(item.get("id"))
        concept = _clean_text(item.get("concept") or item.get("label"))
        asset_key = _clean_text(item.get("asset_key") or item.get("silhouette_key")).lower()

        if scene_type == "process":
            if asset_key in {"cycle", "flow_node", "branch", "capsule", "generic_panel", "generic_object", "blob"}:
                if object_id:
                    removed_ids.add(object_id)
                continue
            if _is_generic_process_stage(concept) and asset_key in {"energy_wave", "cycle", "flow_node", "branch"}:
                if object_id:
                    removed_ids.add(object_id)
                continue
        elif scene_type == "schematic":
            if _is_switch_like_concept(concept):
                item["asset_key"] = "switch"
                item["silhouette_key"] = "switch"
            elif asset_key in {"branch", "road", "generic_panel", "generic_object", "blob"}:
                if object_id:
                    removed_ids.add(object_id)
                continue
            elif asset_key == "module":
                if object_id:
                    removed_ids.add(object_id)
                continue
            elif asset_key == "board" and not _is_explicit_board_concept(concept):
                if object_id:
                    removed_ids.add(object_id)
                continue
            if concept and _contains_any(concept, ("电路结构", "可编辑", "结构说明", "直接编辑")):
                if object_id:
                    removed_ids.add(object_id)
                continue

        kept_objects.append(item)
        if object_id:
            kept_ids.add(object_id)

    scene["object_instances"] = kept_objects
    scene["concept_order"] = [_clean_text(item.get("concept")) for item in kept_objects if _clean_text(item.get("concept"))]
    scene["attachments"] = [
        item
        for item in scene.get("attachments", []) or []
        if str(item.get("host_id") or "") in kept_ids and str(item.get("child_id") or "") in kept_ids
    ]
    if scene_type == "process":
        scene["connectors"] = []
        scene["background_layers"] = _simple_background_layers("process", width, height, [])
    elif scene_type == "schematic":
        normalized_connectors: List[Dict[str, Any]] = []
        seen_pairs: set[Tuple[str, str]] = set()
        for item in scene.get("connectors", []) or []:
            from_id = str(item.get("from_id") or "")
            to_id = str(item.get("to_id") or "")
            if str(item.get("type") or "").strip().lower() != "wire" or from_id not in kept_ids or to_id not in kept_ids:
                continue
            pair_key = (from_id, to_id)
            if pair_key in seen_pairs:
                continue
            cleaned = _copy(item)
            cleaned["type"] = "wire"
            cleaned["label"] = _normalize_schematic_connector_label(cleaned.get("label"))
            cleaned["visible"] = True
            normalized_connectors.append(cleaned)
            seen_pairs.add(pair_key)
        scene["connectors"] = normalized_connectors
        scene["background_layers"] = [
            item for item in scene.get("background_layers", []) or [] if str(item.get("type") or "").strip().lower() not in {"board", "process_band"}
        ]
    else:
        scene["connectors"] = [
            item
            for item in scene.get("connectors", []) or []
            if str(item.get("from_id") or "") in kept_ids and str(item.get("to_id") or "") in kept_ids
        ]
        scene["background_layers"] = [
            item for item in scene.get("background_layers", []) or [] if str(item.get("type") or "").strip().lower() != "process_band"
        ]

    hints = dict(scene.get("render_hints") or {})
    hints["sd_upstream_cleanup"] = {
        "applied": True,
        "scene_type": scene_type,
        "removed_ids": sorted(item for item in removed_ids if item),
    }
    scene["render_hints"] = hints
    return scene


def _merge_query_boost_scene(
    base_scene: Dict[str, Any],
    fallback_scene: Dict[str, Any],
    query: str,
    sketch_options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    merged = _copy(fallback_scene)
    base_scene = base_scene if isinstance(base_scene, dict) else {}
    base_hints = _copy(base_scene.get("render_hints") or {})
    merged_hints = _copy(merged.get("render_hints") or {})
    merged["canvas_size"] = _copy(base_scene.get("canvas_size") or merged.get("canvas_size") or {})
    merged["render_hints"] = {
        **base_hints,
        **merged_hints,
        "scene_summary": _clean_text(merged_hints.get("scene_summary") or base_hints.get("scene_summary") or query[:48]),
        "query_boost_applied": True,
        "query_boost_reason": "sd_upstream_scene_rebuild" if _prefer_sd_semantic_upstream(sketch_options) else "query_scene_rebuild",
        "query_boost_query": _clean_text(query),
        "query_boost_assets": _collect_query_assets(query, _clean_text((merged.get("layout_options") or {}).get("scene_type") or "scene")),
    }
    merged["layout_options"] = {
        **_copy(base_scene.get("layout_options") or {}),
        **_copy(merged.get("layout_options") or {}),
    }
    if _prefer_sd_semantic_upstream(sketch_options):
        merged["layout_options"]["semantic_source"] = "query_boost_sd_upstream"
    merged["concept_order"] = [item.get("concept") for item in merged.get("object_instances", []) or [] if item.get("concept")]
    return merged


def normalize_scene_spec_v2(scene_spec: Dict[str, Any] | None, sketch_options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    _sync_legacy()
    scene = legacy.normalize_scene_spec_v2(scene_spec, sketch_options)
    scene = _sanitize_scene(scene)
    scene = _rebalance_generic_objects(scene)
    scene = _strip_sd_symbolic_objects(scene, sketch_options)
    scene = _apply_visual_metadata(scene)
    scene = _soften_backgrounds(scene)
    scene = _upgrade_layout_defaults(scene, sketch_options)
    scene = apply_natural_layout(scene, sketch_options)
    scene = _upgrade_render_hints(scene)
    return scene


def legacy_scene_spec_to_v2(scene_spec: Dict[str, Any], sketch_options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    _sync_legacy()
    legacy_scene = legacy.legacy_scene_spec_to_v2(scene_spec, sketch_options)
    return normalize_scene_spec_v2(legacy_scene, sketch_options)


def compose_semantic_scene_spec(
    query: str,
    understanding_result: Dict[str, Any] | None,
    extraction_result: Dict[str, Any] | None,
    answer_bundle: Dict[str, Any] | None,
    best_path_concepts: List[str] | None = None,
    canvas_size: Tuple[int, int] = CANVAS_DEFAULT,
    sketch_options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    _sync_legacy()
    base_scene = legacy.compose_semantic_scene_spec(
        query=query,
        understanding_result=understanding_result,
        extraction_result=extraction_result,
        answer_bundle=answer_bundle,
        best_path_concepts=best_path_concepts,
        canvas_size=canvas_size,
        sketch_options=sketch_options,
    )
    scene = normalize_scene_spec_v2(base_scene, sketch_options)
    boost_needed = _scene_needs_query_boost(scene, query, sketch_options)
    if scene.get("object_instances") and not boost_needed:
        return scene
    fallback_scene = _compose_query_only_scene(query, canvas_size, sketch_options)
    if fallback_scene:
        boosted_scene = _merge_query_boost_scene(scene, fallback_scene, query, sketch_options)
        return normalize_scene_spec_v2(boosted_scene, sketch_options)
    return scene


def summarize_scene_spec(scene_spec: Dict[str, Any] | None) -> str:
    if not scene_spec:
        return ""
    scene = normalize_scene_spec_v2(scene_spec)
    backgrounds = scene.get("background_layers", []) or []
    objects = scene.get("object_instances", []) or []
    attachments = scene.get("attachments", []) or []
    connectors = [item for item in scene.get("connectors", []) or [] if item.get("visible")]
    hints = scene.get("render_hints", {}) or {}
    bg_summary = ", ".join(
        f"{item.get('label', item.get('type', 'layer'))}@({item.get('x', 0)},{item.get('y', 0)},{item.get('width', 0)}x{item.get('height', 0)})"
        for item in backgrounds[:4]
    )
    object_summary = ", ".join(
        f"{item.get('concept', '')}[{item.get('prototype_id', item.get('asset_key', ''))}]@({item.get('x', 0)},{item.get('y', 0)},{item.get('width', 0)}x{item.get('height', 0)})/{item.get('depth_band', '')}"
        for item in objects[:8]
    )
    attachment_summary = ", ".join(
        f"{item.get('child_id', '')}->{item.get('host_id', '')}:{item.get('anchor_name', '')}"
        for item in attachments[:6]
    )
    connector_summary = ", ".join(
        f"{item.get('type', '')}:{item.get('label', '')} {item.get('from_id', '')}->{item.get('to_id', '')}"
        for item in connectors[:6]
    )
    parts = [
        f"scene_type={scene.get('layout_options', {}).get('scene_type', 'scene')}",
        f"style={scene.get('layout_options', {}).get('sketch_style', 'scribble_line')}",
        f"view={scene.get('layout_options', {}).get('sketch_view_mode', 'structure')}",
        f"backgrounds={bg_summary}" if bg_summary else "",
        f"objects={object_summary}" if object_summary else "",
        f"attachments={attachment_summary}" if attachment_summary else "",
        f"connectors={connector_summary}" if connector_summary else "",
        hints.get("scene_summary", ""),
        hints.get("subject_summary", ""),
        hints.get("prototype_summary", ""),
        hints.get("user_added_summary", ""),
        hints.get("edit_summary", ""),
    ]
    return " | ".join(part for part in parts if part)


def scene_spec_counts(scene_spec: Dict[str, Any] | None) -> Tuple[int, int]:
    if not scene_spec:
        return 0, 0
    scene = normalize_scene_spec_v2(scene_spec)
    return len(scene.get("object_instances", []) or []), len(scene.get("connectors", []) or [])
