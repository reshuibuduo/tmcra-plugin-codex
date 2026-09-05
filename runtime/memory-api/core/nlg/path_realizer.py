from __future__ import annotations

import random
from typing import List

from .pattern_bank import PatternBank


RELATION_CATEGORY_MAP = {
    "导致": "causal",
    "引起": "causal",
    "造成": "causal",
    "使得": "causal",
    "触发": "causal",
    "产生": "causal",
    "因为": "causal",
    "由于": "causal",
    "causes": "causal",
    "cause": "causal",
    "leads to": "causal",
    "lead to": "causal",
    "results in": "causal",
    "result in": "causal",
    "drives": "causal",
    "trigger": "causal",
    "组成": "structural",
    "构成": "structural",
    "包含": "structural",
    "包括": "structural",
    "属于": "structural",
    "是": "structural",
    "具有": "property",
    "具备": "property",
    "属性": "property",
    "特性": "property",
    "used for": "functional",
    "used to": "functional",
    "用于": "functional",
    "用来": "functional",
    "适用于": "functional",
    "能够": "functional",
    "限制": "limiting",
    "阻止": "limiting",
    "抑制": "limiting",
    "阻碍": "limiting",
    "相关": "generic",
    "related": "generic",
    "related to": "generic",
}

FALLBACK_PATTERNS = {
    "causal": "{X}会导致{Y}",
    "structural": "{X}由{Y}组成",
    "property": "{X}具有{Y}",
    "functional": "{X}用于{Y}",
    "limiting": "{X}限制{Y}",
    "generic": "{X}与{Y}相关",
}


def _category_for_relation(relation: str) -> str:
    lowered = relation.lower()
    for key, category in RELATION_CATEGORY_MAP.items():
        if key.lower() in lowered:
            return category
    return "generic"


def _realize_edge(pattern_bank: PatternBank, source: str, relation: str, target: str) -> str:
    category = _category_for_relation(relation)
    pattern = pattern_bank.pick(category) or FALLBACK_PATTERNS.get(category)
    if not pattern:
        pattern = "{X}与{Y}相关"
    return pattern.replace("{X}", source).replace("{Y}", target)


def _decorate_with_connector(pattern_bank: PatternBank, sentence: str) -> str:
    if not sentence:
        return sentence
    if random.random() < 0.6:
        connector = pattern_bank.pick_connector()
        if connector:
            if sentence.startswith(connector):
                return sentence
            return f"{connector}，{sentence}"
    return sentence


def realize_answer(
    query: str,
    concepts: List[str],
    relations: List[str],
    pattern_bank: PatternBank,
    *,
    intent: str = "general",
) -> str:
    sentences: List[str] = []
    if concepts and relations and len(concepts) == len(relations) + 1:
        for idx, relation in enumerate(relations):
            sentence = _realize_edge(pattern_bank, concepts[idx], relation, concepts[idx + 1])
            sentences.append(sentence)
    else:
        for idx in range(len(concepts) - 1):
            sentences.append(_realize_edge(pattern_bank, concepts[idx], "相关", concepts[idx + 1]))

    if not sentences:
        return "暂未形成足够的概念关系，建议补充语料或知识库。"

    decorated: List[str] = []
    for idx, sentence in enumerate(sentences):
        if idx == 0:
            decorated.append(sentence)
        else:
            decorated.append(_decorate_with_connector(pattern_bank, sentence))

    intro = pattern_bank.pick_intro(intent)
    if not intro:
        if intent == "necessity":
            intro = "需要这样做的原因是："
        elif intent == "explanation":
            intro = "其机理可以概括为："
        elif intent == "how_to":
            intro = "关键步骤与原因如下："
        else:
            intro = ""

    body = "。".join(decorated)
    if intro:
        return f"{intro}{body}。"
    return f"{body}。"
