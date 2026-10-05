from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .semantic_scene_v2 import pick_asset_key


VISUAL_QUERY_HINTS = [
    "生成",
    "画",
    "图",
    "画面",
    "场景",
    "草图",
    "渲染",
    "构图",
    "近景",
    "远景",
    "海报",
    "插画",
    "示意图",
    "简笔",
    "预演",
]

META_CONCEPT_HINTS = {
    "用户输入",
    "含义",
    "解释",
    "内容",
    "问题",
    "概念",
    "机制",
    "原理",
    "场景",
    "画面",
    "草图",
    "控制草图",
    "示意图",
    "构图",
    "unknown",
    "ngram",
}

SCENE_TOKENS = {
    "街道",
    "街景",
    "路边",
    "草地",
    "草坪",
    "树",
    "云",
    "天空",
    "太阳",
    "房子",
    "楼房",
    "建筑",
    "汽车",
    "路灯",
    "室内",
    "房间",
    "桌子",
    "椅子",
    "台灯",
    "窗户",
    "门",
}

PROCESS_TOKENS = {
    "过程",
    "机制",
    "形成",
    "循环",
    "蒸发",
    "凝结",
    "降雨",
    "光合作用",
    "吸收",
    "产生能量",
    "升力",
}

SCHEMATIC_TOKENS = {
    "电路",
    "电池",
    "电阻",
    "电容",
    "二极管",
    "发光二极管",
    "led",
    "串联",
    "并联",
    "电流",
    "电压",
    "细胞",
    "细胞核",
    "细胞质",
    "细胞膜",
}

NON_OBJECT_TERMS = {
    "控制草图",
    "草图",
    "画面",
    "场景",
    "图片",
    "图像",
    "示意图",
    "简笔画",
    "构图",
    "问题",
    "内容",
    "原因",
    "机制",
    "原理",
    "室内",
    "房间",
    "一个",
    "过程图",
    "电路图",
    "基础构图",
    "后续",
    "空间",
}

DETAIL_ASSETS = {"window", "door", "cloud", "sun", "generic_circle"}
UNKNOWN_ASSETS = {"blob", "capsule", "module", "generic_object", "generic_panel"}

OBJECT_SPECS: List[Dict[str, Any]] = [
    {"concept": "狗", "aliases": ["柯基", "小狗", "狗", "犬"], "type": "animal", "role": "subject", "importance": 3.8},
    {"concept": "人", "aliases": ["人物", "行人", "人", "孩子", "学生"], "type": "role", "role": "subject", "importance": 3.4},
    {"concept": "楼房", "aliases": ["楼房", "高楼", "大楼", "楼体", "楼", "建筑"], "type": "building", "role": "subject", "importance": 3.6},
    {"concept": "房子", "aliases": ["房子", "房屋", "住宅", "小屋"], "type": "building", "role": "subject", "importance": 3.4},
    {"concept": "窗户", "aliases": ["窗户", "窗"], "type": "detail", "role": "detail", "importance": 2.3, "countable": True},
    {"concept": "门", "aliases": ["房门", "大门", "门"], "type": "detail", "role": "detail", "importance": 2.2},
    {"concept": "树", "aliases": ["树木", "树", "树叶"], "type": "environment", "role": "support", "importance": 2.8},
    {"concept": "云", "aliases": ["云朵", "云"], "type": "environment", "role": "effect", "importance": 2.7},
    {"concept": "太阳", "aliases": ["太阳", "阳光", "日光"], "type": "environment", "role": "effect", "importance": 2.8},
    {"concept": "汽车", "aliases": ["轿车", "汽车", "车"], "type": "vehicle", "role": "support", "importance": 3.0},
    {"concept": "路灯", "aliases": ["路灯", "街灯", "灯杆"], "type": "detail", "role": "detail", "importance": 2.4},
    {"concept": "桌子", "aliases": ["桌子", "餐桌", "桌"], "type": "indoor", "role": "subject", "importance": 3.2},
    {"concept": "椅子", "aliases": ["椅子", "座椅", "椅"], "type": "indoor", "role": "support", "importance": 2.8},
    {"concept": "台灯", "aliases": ["台灯", "桌灯"], "type": "indoor", "role": "detail", "importance": 2.6},
    {"concept": "室内", "aliases": ["室内", "房间"], "type": "environment", "role": "environment", "importance": 1.7},
    {"concept": "街道", "aliases": ["街道", "街景", "道路", "公路", "路面"], "type": "environment", "role": "environment", "importance": 1.9},
    {"concept": "地面", "aliases": ["地面", "地上"], "type": "environment", "role": "environment", "importance": 1.4},
    {"concept": "大气", "aliases": ["大气", "空气"], "type": "environment", "role": "environment", "importance": 1.5},
    {"concept": "草地", "aliases": ["草地", "草坪"], "type": "environment", "role": "environment", "importance": 2.1},
    {"concept": "植物", "aliases": ["植物"], "type": "life", "role": "subject", "importance": 3.0},
    {"concept": "叶片", "aliases": ["叶片", "叶子", "树叶", "叶"], "type": "life", "role": "support", "importance": 2.6},
    {"concept": "蒸发", "aliases": ["蒸发"], "type": "process", "role": "stage", "importance": 3.0},
    {"concept": "凝结", "aliases": ["凝结"], "type": "process", "role": "stage", "importance": 3.0},
    {"concept": "降雨", "aliases": ["降雨", "下雨", "降水"], "type": "process", "role": "stage", "importance": 3.1},
    {"concept": "雨滴", "aliases": ["雨滴", "水滴"], "type": "process", "role": "stage", "importance": 2.8},
    {"concept": "光合作用", "aliases": ["光合作用"], "type": "process", "role": "stage", "importance": 3.1},
    {"concept": "电池", "aliases": ["电池", "电源"], "type": "schematic", "role": "component", "importance": 3.0},
    {"concept": "电阻", "aliases": ["限流电阻", "电阻"], "type": "schematic", "role": "component", "importance": 3.0},
    {"concept": "LED", "aliases": ["发光二极管", "LED", "led"], "type": "schematic", "role": "component", "importance": 3.0},
    {"concept": "电容", "aliases": ["电容"], "type": "schematic", "role": "component", "importance": 2.6},
    {"concept": "二极管", "aliases": ["二极管"], "type": "schematic", "role": "component", "importance": 2.6},
    {"concept": "电路板", "aliases": ["电路板", "开发板", "主板"], "type": "schematic", "role": "environment", "importance": 2.0},
    {"concept": "细胞", "aliases": ["细胞"], "type": "structure", "role": "subject", "importance": 3.4},
    {"concept": "细胞核", "aliases": ["细胞核"], "type": "structure", "role": "detail", "importance": 2.8},
    {"concept": "细胞质", "aliases": ["细胞质"], "type": "structure", "role": "support", "importance": 2.8},
    {"concept": "细胞膜", "aliases": ["细胞膜"], "type": "structure", "role": "detail", "importance": 2.7},
    {"concept": "氧气", "aliases": ["氧气"], "type": "process", "role": "support", "importance": 2.6},
    {"concept": "能量", "aliases": ["能量"], "type": "process", "role": "support", "importance": 2.6},
    {"concept": "消化", "aliases": ["消化", "消化吸收"], "type": "process", "role": "stage", "importance": 2.7},
    {"concept": "血液", "aliases": ["血液", "血"], "type": "process", "role": "support", "importance": 2.7},
    {"concept": "水", "aliases": ["水"], "type": "process", "role": "support", "importance": 2.6},
]

RELATION_HINTS = [
    (("串联", "series"), "串联"),
    (("并联", "parallel"), "并联"),
    (("光照", "照射", "照到", "照向", "射向"), "照射"),
    (("附着", "装在", "贴在", "挂在"), "附着"),
    (("放在", "摆在", "置于"), "放在"),
    (("邻近", "旁边", "靠近", "周边"), "邻近"),
    (("进入", "吸收", "输送"), "进入"),
    (("连接", "接到", "导通"), "连接"),
    (("导致", "引发", "形成", "产生", "转化", "变成"), "导致"),
    (("位于", "在里面", "内部", "包含"), "位于"),
]

COUNT_PATTERNS = [
    (re.compile(r"(两|二|2)(?:个|扇|辆|盏|朵|栋)?"), 2),
    (re.compile(r"(三|3)(?:个|扇|辆|盏|朵|栋)?"), 3),
]


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value or {}, ensure_ascii=False)
    except Exception:
        return ""


def _normalize_text(value: Any) -> str:
    text = _clean_text(value).lower()
    return re.sub(r"\s+", "", text)


def is_meta_concept(text: Any) -> bool:
    cleaned = _clean_text(text)
    lowered = _normalize_text(cleaned)
    if not cleaned:
        return True
    if cleaned in META_CONCEPT_HINTS or lowered in META_CONCEPT_HINTS:
        return True
    return any(token in cleaned for token in ("用户输入", "含义", "问题", "机制", "原理"))


def is_visual_query(query: str) -> bool:
    text = _clean_text(query)
    if not text:
        return False
    lowered = text.lower()
    if any(token.lower() in lowered for token in VISUAL_QUERY_HINTS):
        return True
    has_domain_tokens = any(token in text for token in SCENE_TOKENS | PROCESS_TOKENS | SCHEMATIC_TOKENS)
    has_visual_intent = any(token in text for token in ("图", "画", "草图", "示意", "构图", "渲染", "预演"))
    return has_domain_tokens and has_visual_intent


def normalize_relation_label(label: Any) -> str:
    text = _normalize_text(label)
    if not text:
        return ""
    for tokens, canonical in RELATION_HINTS:
        if any(token in text for token in tokens):
            return canonical
    return _clean_text(label)


def _canonicalize_concept(name: str) -> str:
    text = _clean_text(name)
    lowered = text.lower()
    for spec in OBJECT_SPECS:
        for alias in spec["aliases"]:
            alias_lower = alias.lower()
            if alias_lower and alias_lower in lowered:
                return spec["concept"]
    return text


def _spec_for_concept(name: str) -> Dict[str, Any] | None:
    concept = _canonicalize_concept(name)
    for spec in OBJECT_SPECS:
        if spec["concept"] == concept:
            return spec
    return None


def _infer_scene_type(
    query: str,
    understanding_result: Dict[str, Any] | None,
    extraction_result: Dict[str, Any] | None,
    answer_bundle: Dict[str, Any] | None,
) -> str:
    query_text = _clean_text(query).lower()
    if any(token.lower() in query_text for token in SCHEMATIC_TOKENS):
        return "schematic"
    if any(token.lower() in query_text for token in PROCESS_TOKENS):
        return "process"
    if any(token in _clean_text(query) for token in SCENE_TOKENS):
        return "scene"

    context_terms: List[str] = []
    for item in (understanding_result or {}).get("concepts", []) or []:
        if isinstance(item, dict):
            context_terms.append(_clean_text(item.get("concept")))
    for item in (extraction_result or {}).get("concepts", []) or []:
        if isinstance(item, dict):
            context_terms.append(_clean_text(item.get("concept")))
    for item in (answer_bundle or {}).get("core_concepts", []) or []:
        context_terms.append(_clean_text(item))
    haystack = " ".join(term for term in context_terms if term).lower()
    if any(token.lower() in haystack for token in SCHEMATIC_TOKENS):
        return "schematic"
    if any(token.lower() in haystack for token in PROCESS_TOKENS):
        return "process"
    return "scene"


def _count_for_alias(query: str, alias: str, default: int = 1) -> int:
    if not alias or alias not in query:
        return 0
    prefix = query[: query.index(alias)]
    tail = prefix[-6:]
    for pattern, value in COUNT_PATTERNS:
        if pattern.search(tail):
            return value
    return default


def _object_role(spec: Dict[str, Any] | None, scene_type: str) -> str:
    if not spec:
        return "support"
    role = str(spec.get("role", "support"))
    if scene_type == "process" and role not in {"stage", "support"}:
        return "stage" if spec.get("type") == "process" else "support"
    if scene_type == "schematic":
        return "component" if role != "environment" else "environment"
    return role


def _append_object(
    objects: List[Dict[str, Any]],
    object_index: Dict[str, Dict[str, Any]],
    *,
    concept: str,
    scene_type: str,
    source: str,
    importance: float = 1.0,
    count: int = 1,
    position: int = 9999,
) -> None:
    base_concept = _canonicalize_concept(concept)
    if not base_concept or base_concept in NON_OBJECT_TERMS or is_meta_concept(base_concept):
        return
    spec = _spec_for_concept(base_concept)
    role = _object_role(spec, scene_type)
    total = max(1, int(count))
    for index in range(total):
        concept_name = base_concept if total == 1 else f"{base_concept}{index + 1}"
        item = object_index.get(concept_name)
        if item:
            item["importance"] = max(float(item.get("importance", 1.0) or 1.0), float(importance))
            item["position"] = min(int(item.get("position", 9999) or 9999), int(position))
            if source not in item["sources"]:
                item["sources"].append(source)
            continue
        asset_key = pick_asset_key(base_concept, scene_type)
        record = {
            "concept": concept_name,
            "base_concept": base_concept,
            "type": spec.get("type", "general") if spec else "general",
            "role": role,
            "asset_key": asset_key,
            "importance": float(importance),
            "position": int(position),
            "sources": [source],
        }
        object_index[concept_name] = record
        objects.append(record)


def _extract_query_objects(query: str, scene_type: str) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    object_index: Dict[str, Dict[str, Any]] = {}
    query_text = _clean_text(query)
    lowered = query_text.lower()
    for spec in OBJECT_SPECS:
        positions = []
        count = 0
        for alias in spec["aliases"]:
            alias_text = _clean_text(alias)
            idx = lowered.find(alias_text.lower())
            if idx >= 0:
                positions.append(idx)
                count = max(count, _count_for_alias(query_text, alias_text, 1))
        if not positions:
            continue
        _append_object(
            objects,
            object_index,
            concept=spec["concept"],
            scene_type=scene_type,
            source="query",
            importance=float(spec.get("importance", 1.0) or 1.0),
            count=count or 1,
            position=min(positions),
        )
    objects.sort(key=lambda item: (int(item.get("position", 9999)), -float(item.get("importance", 0.0))))
    return objects


def _iter_context_concepts(
    understanding_result: Dict[str, Any] | None,
    extraction_result: Dict[str, Any] | None,
    answer_bundle: Dict[str, Any] | None,
) -> List[str]:
    names: List[str] = []
    for item in (understanding_result or {}).get("concepts", []) or []:
        names.append(_clean_text(item.get("concept")))
    for item in (extraction_result or {}).get("concepts", []) or []:
        names.append(_clean_text(item.get("concept")))
    for item in (answer_bundle or {}).get("core_concepts", []) or []:
        names.append(_clean_text(item))
    for rel in (answer_bundle or {}).get("primary_chain", []) or []:
        names.extend([_clean_text(rel.get("from")), _clean_text(rel.get("to"))])
    for rel in (answer_bundle or {}).get("supporting_relations", []) or []:
        names.extend([_clean_text(rel.get("from")), _clean_text(rel.get("to"))])
    return [name for name in names if name]


def _augment_context_objects(
    objects: List[Dict[str, Any]],
    scene_type: str,
    understanding_result: Dict[str, Any] | None,
    extraction_result: Dict[str, Any] | None,
    answer_bundle: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    object_index = {item["concept"]: item for item in objects}
    for position, name in enumerate(_iter_context_concepts(understanding_result, extraction_result, answer_bundle), start=500):
        if is_meta_concept(name):
            continue
        canonical = _canonicalize_concept(name)
        if canonical in NON_OBJECT_TERMS:
            continue
        if any(token in canonical for token in ("画一个", "保留", "包括", "控制草图", "基础构图")):
            continue
        importance = 1.1
        spec = _spec_for_concept(canonical)
        if spec is None:
            continue
        if spec:
            importance = max(importance, float(spec.get("importance", 1.0) or 1.0) * 0.8)
        _append_object(
            objects,
            object_index,
            concept=canonical,
            scene_type=scene_type,
            source="context",
            importance=importance,
            position=position,
        )
    objects.sort(key=lambda item: (int(item.get("position", 9999)), -float(item.get("importance", 0.0))))
    return objects


def _find_host(objects: List[Dict[str, Any]], *base_concepts: str) -> str | None:
    base_set = set(base_concepts)
    for item in objects:
        if item.get("base_concept") in base_set:
            return item["concept"]
    return None


def _map_object_name(name: str, objects: List[Dict[str, Any]]) -> str:
    target = _canonicalize_concept(name)
    for item in objects:
        if item.get("base_concept") == target or item.get("concept") == target:
            return str(item["concept"])
    return target


def _add_relation(relations: List[Dict[str, Any]], seen: set[tuple[str, str, str]], src: str, dst: str, label: str, source: str, weight: float = 0.72) -> None:
    relation = normalize_relation_label(label)
    if not src or not dst or not relation or src == dst:
        return
    key = (src, dst, relation)
    if key in seen:
        return
    seen.add(key)
    relations.append({"from": src, "to": dst, "relation": relation, "weight": float(weight), "source": source})


def _collect_context_relations(
    objects: List[Dict[str, Any]],
    understanding_result: Dict[str, Any] | None,
    extraction_result: Dict[str, Any] | None,
    answer_bundle: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    relations: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    object_names = {item["concept"] for item in objects}
    for group_name, group in (
        ("understanding", (understanding_result or {}).get("relations", []) or []),
        ("extraction", (extraction_result or {}).get("relations", []) or []),
        ("answer_primary", (answer_bundle or {}).get("primary_chain", []) or []),
        ("answer_support", (answer_bundle or {}).get("supporting_relations", []) or []),
    ):
        for relation in group:
            src = _map_object_name(_clean_text(relation.get("from")), objects)
            dst = _map_object_name(_clean_text(relation.get("to")), objects)
            label = normalize_relation_label(relation.get("relation"))
            if src not in object_names or dst not in object_names:
                continue
            _add_relation(relations, seen, src, dst, label, group_name, float(relation.get("weight", 0.65) or 0.65))
    return relations


def _apply_rule_relations(query: str, scene_type: str, objects: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text = _clean_text(query)
    seen = {(item["from"], item["to"], item["relation"]) for item in relations}
    building = _find_host(objects, "楼房", "房子")
    cell = _find_host(objects, "细胞")
    table = _find_host(objects, "桌子")
    road = _find_host(objects, "街道")
    plant = _find_host(objects, "植物", "叶片")

    for item in objects:
        base = item.get("base_concept")
        if base == "窗户" and building:
            _add_relation(relations, seen, item["concept"], building, "附着", "rule_attachment", 0.82)
        elif base == "门" and building:
            _add_relation(relations, seen, item["concept"], building, "附着", "rule_attachment", 0.82)
        elif base == "椅子" and table:
            _add_relation(relations, seen, item["concept"], table, "邻近", "rule_attachment", 0.78)
        elif base == "台灯" and table:
            _add_relation(relations, seen, item["concept"], table, "放在", "rule_attachment", 0.8)
        elif base == "汽车" and road:
            _add_relation(relations, seen, item["concept"], road, "位于", "rule_layout", 0.76)
        elif base == "路灯" and road:
            _add_relation(relations, seen, item["concept"], road, "位于", "rule_layout", 0.74)
        elif base in {"细胞核", "细胞质"} and cell:
            _add_relation(relations, seen, item["concept"], cell, "位于", "rule_structure", 0.8)
        elif base == "细胞膜" and cell:
            _add_relation(relations, seen, item["concept"], cell, "附着", "rule_structure", 0.8)

    if scene_type == "process":
        evaporation = _find_host(objects, "蒸发")
        condensation = _find_host(objects, "凝结")
        rainfall = _find_host(objects, "降雨", "雨滴")
        if evaporation and condensation:
            _add_relation(relations, seen, evaporation, condensation, "导致", "rule_process", 0.84)
        if condensation and rainfall:
            _add_relation(relations, seen, condensation, rainfall, "导致", "rule_process", 0.84)
        sun = _find_host(objects, "太阳")
        if sun and plant:
            _add_relation(relations, seen, sun, plant, "照射", "rule_process", 0.84)
        photosynthesis = _find_host(objects, "光合作用")
        if plant and photosynthesis:
            _add_relation(relations, seen, plant, photosynthesis, "导致", "rule_process", 0.78)

    if scene_type == "schematic":
        battery = _find_host(objects, "电池")
        resistor = _find_host(objects, "电阻")
        led = _find_host(objects, "LED")
        if battery and resistor:
            _add_relation(relations, seen, battery, resistor, "串联", "rule_schematic", 0.88)
        if resistor and led:
            _add_relation(relations, seen, resistor, led, "串联", "rule_schematic", 0.88)

    if "上" in text and table and _find_host(objects, "台灯"):
        _add_relation(relations, seen, _find_host(objects, "台灯"), table, "放在", "rule_text", 0.8)
    return relations


def _constraint(label: str, target: str = "", source: str = "query") -> Dict[str, Any]:
    return {"label": label, "target": target, "source": source}


def _collect_constraints(query: str, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text = _clean_text(query)
    focus = objects[0]["concept"] if objects else ""
    constraints: List[Dict[str, Any]] = []
    if "近景" in text or "前景" in text:
        constraints.append(_constraint("foreground", focus))
    if "远景" in text or "背景" in text:
        constraints.append(_constraint("background", focus))
    if "左" in text:
        constraints.append(_constraint("left", focus))
    if "右" in text:
        constraints.append(_constraint("right", focus))
    if "上方" in text or "顶部" in text:
        constraints.append(_constraint("top", focus))
    if "下方" in text or "底部" in text:
        constraints.append(_constraint("bottom", focus))
    if "居中" in text or "中心" in text:
        constraints.append(_constraint("center", focus))
    return constraints


def _infer_focus(
    objects: List[Dict[str, Any]],
    scene_type: str,
    understanding_result: Dict[str, Any] | None,
    answer_bundle: Dict[str, Any] | None,
) -> str:
    preferred = [_clean_text((answer_bundle or {}).get("focus_concept")), _clean_text((understanding_result or {}).get("focus_concept"))]
    object_names = {item["concept"] for item in objects}
    for item in preferred:
        mapped = _map_object_name(item, objects)
        if mapped in object_names:
            selected = next((obj for obj in objects if obj.get("concept") == mapped), None)
            role = str((selected or {}).get("role", ""))
            if scene_type == "process" and role != "stage":
                continue
            if scene_type == "schematic" and role != "component":
                continue
            if scene_type == "scene" and role == "detail":
                continue
            return mapped
    ranked = sorted(
        objects,
        key=lambda item: (
            1 if item.get("role") in {"subject", "stage", "component"} else 0,
            float(item.get("importance", 0.0)),
            -int(item.get("position", 9999)),
        ),
        reverse=True,
    )
    for item in ranked:
        if item.get("asset_key") not in DETAIL_ASSETS:
            return item["concept"]
    return ranked[0]["concept"] if ranked else ""


def parse_visual_query(
    query: str,
    understanding_result: Dict[str, Any] | None = None,
    extraction_result: Dict[str, Any] | None = None,
    answer_bundle: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    query_text = _clean_text(query)
    if not is_visual_query(query_text):
        return None
    scene_type = _infer_scene_type(query_text, understanding_result, extraction_result, answer_bundle)
    objects = _extract_query_objects(query_text, scene_type)
    objects = _augment_context_objects(objects, scene_type, understanding_result, extraction_result, answer_bundle)
    if not objects:
        return None
    relations = _collect_context_relations(objects, understanding_result, extraction_result, answer_bundle)
    relations = _apply_rule_relations(query_text, scene_type, objects, relations)
    constraints = _collect_constraints(query_text, objects)
    focus_concept = _infer_focus(objects, scene_type, understanding_result, answer_bundle)
    unknown_assets = set(UNKNOWN_ASSETS)
    if scene_type == "process":
        unknown_assets.discard("capsule")
    unknown_object_count = sum(1 for item in objects if str(item.get("asset_key", "")) in unknown_assets)
    focus_item = next((item for item in objects if item["concept"] == focus_concept), None)
    unknown_primary_object = bool(focus_item and str(focus_item.get("asset_key", "")) in unknown_assets)
    fallback_reason = ""
    if unknown_primary_object:
        fallback_reason = "focus_object_fell_back_to_generic"
    elif unknown_object_count:
        fallback_reason = "some_objects_require_generic_fallback"
    elif not relations:
        fallback_reason = "objects_found_without_explicit_relations"
    return {
        "scene_type": scene_type,
        "focus_concept": focus_concept,
        "objects": objects,
        "relations": relations,
        "constraints": constraints,
        "best_path_concepts": [item["concept"] for item in objects],
        "object_base_concepts": [item.get("base_concept", item["concept"]) for item in objects],
        "parse_source": "visual_query_parser",
        "fallback_reason": fallback_reason,
        "unknown_object_count": int(unknown_object_count),
        "unknown_primary_object": unknown_primary_object,
    }


def build_visual_extraction(parsed: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not parsed:
        return None
    concepts = [
        {
            "concept": item.get("concept"),
            "type": item.get("type", "general"),
            "importance": float(item.get("importance", 1.0) or 1.0),
            "source": "visual_query_parser",
        }
        for item in parsed.get("objects", []) or []
    ]
    return {
        "concepts": concepts,
        "relations": list(parsed.get("relations", []) or []),
        "contexts": {
            "scene_type": parsed.get("scene_type"),
            "constraints": parsed.get("constraints", []),
            "parse_source": parsed.get("parse_source"),
            "fallback_reason": parsed.get("fallback_reason"),
        },
    }


def build_visual_understanding(parsed: Dict[str, Any] | None, base: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    if not parsed:
        return base
    merged = dict(base or {})
    visual_extraction = build_visual_extraction(parsed) or {"concepts": [], "relations": []}
    merged.setdefault("intent", "visual_composition")
    merged["normalized_query"] = merged.get("normalized_query") or ""
    merged["focus_concept"] = parsed.get("focus_concept") or merged.get("focus_concept")
    merged["confidence"] = max(float(merged.get("confidence", 0.0) or 0.0), 0.66)
    merged["concepts"] = (merged.get("concepts") or []) or visual_extraction["concepts"]
    merged["relations"] = (merged.get("relations") or []) or list(parsed.get("relations", []) or [])
    merged["constraints"] = list(merged.get("constraints") or []) + [str(item.get("label")) for item in parsed.get("constraints", []) or [] if str(item.get("label", "")).strip()]
    return merged


def build_visual_answer_bundle(parsed: Dict[str, Any] | None, base: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    if not parsed:
        return base
    merged = dict(base or {})
    relations = list(parsed.get("relations", []) or [])
    merged["focus_concept"] = parsed.get("focus_concept") or merged.get("focus_concept")
    merged["answer_source"] = merged.get("answer_source") or "seeded_relations"
    merged["core_concepts"] = list(dict.fromkeys(list(merged.get("core_concepts") or []) + list(parsed.get("best_path_concepts") or [])))[:12]
    if not merged.get("primary_chain"):
        merged["primary_chain"] = relations[:4]
    merged["supporting_relations"] = list(merged.get("supporting_relations") or [])
    seen = {
        (_clean_text(item.get("from")), _clean_text(item.get("to")), normalize_relation_label(item.get("relation")))
        for item in merged["supporting_relations"]
    }
    for relation in relations:
        key = (_clean_text(relation.get("from")), _clean_text(relation.get("to")), normalize_relation_label(relation.get("relation")))
        if key in seen:
            continue
        seen.add(key)
        merged["supporting_relations"].append(relation)
    merged["constraints"] = list(dict.fromkeys(list(merged.get("constraints") or []) + [item.get("label") for item in parsed.get("constraints", []) or [] if item.get("label")]))
    merged["confidence"] = max(float(merged.get("confidence", 0.0) or 0.0), 0.58)
    merged["has_forward_path"] = bool(merged.get("has_forward_path", False))
    merged["has_boundary_path"] = bool(merged.get("has_boundary_path", False))
    merged["has_reverse_path"] = bool(merged.get("has_reverse_path", False))
    return merged


def build_visual_scene_context(
    query: str,
    understanding_result: Dict[str, Any] | None = None,
    extraction_result: Dict[str, Any] | None = None,
    answer_bundle: Dict[str, Any] | None = None,
    best_path_concepts: List[str] | None = None,
) -> Dict[str, Any] | None:
    parsed = parse_visual_query(query, understanding_result, extraction_result, answer_bundle)
    if not parsed:
        return None
    visual_extraction = build_visual_extraction(parsed) or {"concepts": [], "relations": [], "contexts": {}}
    merged_understanding = build_visual_understanding(parsed, understanding_result)
    merged_extraction = dict(extraction_result or {})

    existing_concepts = [
        _clean_text(item.get("concept"))
        for item in (merged_extraction.get("concepts") or [])
        if isinstance(item, dict) and _clean_text(item.get("concept"))
    ]
    existing_relations = merged_extraction.get("relations") or []
    if not existing_concepts or all(is_meta_concept(name) for name in existing_concepts):
        merged_extraction["concepts"] = visual_extraction["concepts"]
    else:
        seen_concepts = set(existing_concepts)
        for item in visual_extraction["concepts"]:
            name = _clean_text(item.get("concept"))
            if name and name not in seen_concepts:
                seen_concepts.add(name)
                merged_extraction.setdefault("concepts", []).append(item)
    if not existing_relations:
        merged_extraction["relations"] = visual_extraction["relations"]
    else:
        seen_relations = {
            (_clean_text(item.get("from")), _clean_text(item.get("to")), normalize_relation_label(item.get("relation")))
            for item in existing_relations
            if isinstance(item, dict)
        }
        for relation in visual_extraction["relations"]:
            key = (_clean_text(relation.get("from")), _clean_text(relation.get("to")), normalize_relation_label(relation.get("relation")))
            if key in seen_relations:
                continue
            seen_relations.add(key)
            merged_extraction.setdefault("relations", []).append(relation)
    merged_extraction["contexts"] = {**(merged_extraction.get("contexts") or {}), **(visual_extraction.get("contexts") or {})}

    merged_answer_bundle = build_visual_answer_bundle(parsed, answer_bundle)
    path_concepts = list(best_path_concepts or [])
    if not path_concepts or all(is_meta_concept(item) for item in path_concepts):
        path_concepts = list(parsed.get("best_path_concepts") or [])

    return {
        "query": _clean_text(query),
        "understanding_result": merged_understanding,
        "extraction_result": merged_extraction,
        "answer_bundle": merged_answer_bundle,
        "best_path_concepts": path_concepts,
        "start_concept": merged_answer_bundle.get("focus_concept") if isinstance(merged_answer_bundle, dict) else parsed.get("focus_concept"),
        "visual_parse": parsed,
    }
