from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional

from loguru import logger

from .kb.conceptnet_store import ConceptNetStore, _has_cjk
from .kb.embedding_store import EmbeddingStore


CAUSAL_MARKERS_ZH = (
    "导致", "引起", "造成", "使得", "因此", "从而", "所以", "促使", "引发", "产生",
)
CAUSAL_MARKERS_EN = (
    "causes", "cause", "leads to", "lead to", "results in", "result in",
    "drives", "drive", "triggers", "trigger",
)

FUNCTION_MARKERS_ZH = ("用于", "用来", "适用于")
STRUCTURAL_MARKERS_ZH = ("由", "组成", "包括", "包含")

PROCESS_HINTS_ZH = ("过程", "变化", "流动", "转换", "生成", "传递", "驱动", "扩散", "反应")
PROPERTY_HINTS_ZH = ("温度", "电压", "速度", "压力", "浓度", "功率", "能量", "电流", "阻力")
PROCESS_HINTS_EN = ("process", "flow", "transfer", "conversion", "generation", "drive", "trigger")
PROPERTY_HINTS_EN = ("temperature", "voltage", "speed", "pressure", "density", "power", "energy", "current", "resistance")

STOPWORDS_EN = {
    "what", "is", "are", "was", "were", "do", "does", "did",
    "a", "an", "the", "to", "of", "for", "in", "on", "with",
    "and", "or", "as", "by", "from", "into", "that", "this",
    "these", "those", "used", "use", "using", "about", "why",
    "how", "when", "where", "which", "who", "whom", "whose",
}

STOPWORDS_ZH = {
    "什么", "为何", "为什么", "怎么", "怎样", "如何", "是否", "是不是",
    "的", "了", "在", "与", "和", "以及", "或", "及", "对", "为",
    "有", "没有", "能", "不能", "会", "不会", "应该", "可能",
    "用于", "导致", "能够", "由", "组成", "包括", "限制", "阻止",
    "如果", "把", "将", "使", "让", "会产生", "会导致", "作用", "功能",
}

RELATION_MAP = {
    "Causes": "导致",
    "CausesDesire": "触发",
    "HasA": "具有",
    "PartOf": "组成",
    "IsA": "属于",
    "UsedFor": "用途",
    "CapableOf": "能够",
    "HasProperty": "具有",
    "LocatedNear": "靠近",
    "AtLocation": "位于",
    "RelatedTo": "相关",
    "ReceivesAction": "受到",
    "CreatedBy": "产生",
    "MotivatedByGoal": "动机",
    "ObstructedBy": "阻碍",
}


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"[。！？；;!?]\s*", text)
    return [part.strip() for part in parts if part.strip()]


def _tokenize_en(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9\-]*", text.lower())


def _extract_cjk_sequences(text: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fff]{2,}", text)


def _order_concepts_by_text(text: str, concepts: Iterable[str]) -> List[str]:
    hits = []
    for concept in concepts:
        idx = text.find(concept)
        if idx >= 0:
            hits.append((idx, concept))
    hits.sort(key=lambda item: item[0])
    return [concept for _, concept in hits]


def _generate_ngrams(tokens: List[str], min_n: int, max_n: int) -> List[str]:
    results: List[str] = []
    for n in range(min_n, max_n + 1):
        for i in range(0, max(len(tokens) - n + 1, 0)):
            results.append(" ".join(tokens[i:i + n]))
    return results


def _generate_cjk_ngrams(text: str, min_n: int, max_n: int) -> List[str]:
    results: List[str] = []
    length = len(text)
    for n in range(min_n, max_n + 1):
        for i in range(0, max(length - n + 1, 0)):
            results.append(text[i:i + n])
    return results


def _infer_type(concept: str) -> str:
    lowered = concept.lower()
    if _has_cjk(concept):
        if any(hint in concept for hint in PROCESS_HINTS_ZH):
            return "process"
        if any(hint in concept for hint in PROPERTY_HINTS_ZH):
            return "property"
        return "entity"
    if any(hint in lowered for hint in PROCESS_HINTS_EN):
        return "process"
    if any(hint in lowered for hint in PROPERTY_HINTS_EN):
        return "property"
    return "entity"


def _dedupe_relations(relations: Iterable[Dict]) -> List[Dict]:
    seen = set()
    unique: List[Dict] = []
    for rel in relations:
        key = (rel.get("from"), rel.get("to"), rel.get("relation"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(rel)
    return unique


class LocalConceptExtractor:
    """Rule-based concept extractor backed by ConceptNet."""

    def __init__(self, store: Optional[ConceptNetStore] = None, embedding_store: Optional[EmbeddingStore] = None) -> None:
        self.store = store
        self.embedding_store = embedding_store
        self.MAX_CONCEPTS = None
        self.MAX_RELATIONS = 25
        self.DEDUPE_SIM_THRESHOLD = 0.88
        self.last_warnings: List[str] = []

    def _apply_limit(self, items, limit):
        if limit and limit > 0:
            return items[:limit]
        return items

    def _filter_candidates(self, candidates: List[str]) -> List[str]:
        filtered = []
        for term in candidates:
            if not term:
                continue
            if len(term) > 40:
                continue
            if _has_cjk(term):
                if len(term) < 2:
                    continue
                if term in STOPWORDS_ZH:
                    continue
            else:
                if len(term) < 2:
                    continue
                if term.lower() in STOPWORDS_EN:
                    continue
            filtered.append(term)
        return filtered

    def _dedupe_by_embedding(self, concepts: List[str]) -> List[str]:
        if not concepts:
            return []
        if not self.embedding_store or not self.embedding_store.available:
            return self._apply_limit(concepts, self.MAX_CONCEPTS)
        kept: List[str] = []
        for concept in concepts:
            duplicate = False
            for existing in kept:
                score = self.embedding_store.cosine_similarity(concept, existing)
                if score is not None and score >= self.DEDUPE_SIM_THRESHOLD:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(concept)
            if self.MAX_CONCEPTS and self.MAX_CONCEPTS > 0 and len(kept) >= self.MAX_CONCEPTS:
                break
        return kept

    def _extract_concepts(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        candidates: List[str] = []
        cjk_segments = _extract_cjk_sequences(text)
        for seg in cjk_segments:
            candidates.extend(_generate_cjk_ngrams(seg, 2, 4))
            candidates.append(seg)
        en_tokens = _tokenize_en(text)
        candidates.extend(_generate_ngrams(en_tokens, 1, 3))
        candidates = self._filter_candidates(candidates)
        if len(candidates) > 600:
            unique = list(dict.fromkeys(candidates))
            unique.sort(key=lambda item: (-len(item), item))
            candidates = unique[:600]

        if self.store and self.store.available:
            zh_candidates = [c for c in candidates if _has_cjk(c)]
            en_candidates = [c for c in candidates if not _has_cjk(c)]
            known: List[str] = []
            if zh_candidates:
                known.extend(self.store.find_concepts(zh_candidates, lang="zh"))
            if en_candidates:
                known.extend(self.store.find_concepts(en_candidates, lang="en"))
            if known:
                counts = Counter(candidates)
                unique_known = list(dict.fromkeys(known))
                unique_known.sort(key=lambda item: (-counts.get(item, 0), -len(item), item))
                return self._dedupe_by_embedding(unique_known)

        counts = Counter(candidates)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
        ranked_terms = [term for term, _ in ranked]
        ranked_terms = self._apply_limit(ranked_terms, self.MAX_CONCEPTS)
        return self._dedupe_by_embedding(ranked_terms)

    def _extract_marker_relations(self, sentences: List[str], concepts: List[str]) -> List[Dict]:
        relations: List[Dict] = []
        if not concepts:
            return relations
        concept_set = set(concepts)
        for sentence in sentences:
            lowered = sentence.lower()
            for marker in CAUSAL_MARKERS_ZH:
                if marker in sentence:
                    left, right = sentence.split(marker, 1)
                    left_concepts = _order_concepts_by_text(left, concept_set)
                    right_concepts = _order_concepts_by_text(right, concept_set)
                    if left_concepts and right_concepts:
                        relations.append({
                            "from": left_concepts[-1],
                            "to": right_concepts[0],
                            "relation": "导致",
                            "weight": 0.85,
                            "source": "text",
                        })
            for marker in CAUSAL_MARKERS_EN:
                if marker in lowered:
                    parts = lowered.split(marker, 1)
                    if len(parts) != 2:
                        continue
                    left, right = parts
                    left_concepts = _order_concepts_by_text(left, [c for c in concept_set if c.lower() in left])
                    right_concepts = _order_concepts_by_text(right, [c for c in concept_set if c.lower() in right])
                    if left_concepts and right_concepts:
                        relations.append({
                            "from": left_concepts[-1],
                            "to": right_concepts[0],
                            "relation": "causes",
                            "weight": 0.85,
                            "source": "text",
                        })
            for marker in FUNCTION_MARKERS_ZH:
                if marker in sentence:
                    left, right = sentence.split(marker, 1)
                    left_concepts = _order_concepts_by_text(left, concept_set)
                    right_concepts = _order_concepts_by_text(right, concept_set)
                    if left_concepts and right_concepts:
                        relations.append({
                            "from": left_concepts[-1],
                            "to": right_concepts[0],
                            "relation": "用于",
                            "weight": 0.82,
                            "source": "text",
                        })
            for marker in STRUCTURAL_MARKERS_ZH:
                if marker in sentence:
                    left, right = sentence.split(marker, 1)
                    left_concepts = _order_concepts_by_text(left, concept_set)
                    right_concepts = _order_concepts_by_text(right, concept_set)
                    if left_concepts and right_concepts:
                        relations.append({
                            "from": left_concepts[-1],
                            "to": right_concepts[0],
                            "relation": "组成",
                            "weight": 0.8,
                            "source": "text",
                        })
        return relations

    def _extract_kb_relations(self, concepts: List[str]) -> List[Dict]:
        if not self.store or not self.store.available:
            return []
        kb_edges = self.store.get_edges_between(concepts, limit_per_concept=20)
        relations: List[Dict] = []
        for edge in kb_edges:
            weight = min(0.7, max(0.35, float(edge.get("weight", 0.5)) / 2.0))
            relations.append({
                "from": edge.get("source"),
                "to": edge.get("target"),
                "relation": RELATION_MAP.get(edge.get("relation", ""), edge.get("relation", "related_to")),
                "weight": weight,
                "source": "kb",
            })
        return relations

    def extract(self, text: str) -> Optional[Dict]:
        self.last_warnings = []
        concepts = self._extract_concepts(text)
        if not concepts:
            self.last_warnings.append("未命中ConceptNet词表，已退化为文本关键词抽取。")
            fallback_candidates = _tokenize_en(text)
            for seg in _extract_cjk_sequences(text):
                fallback_candidates.append(seg)
            concepts = self._apply_limit(self._filter_candidates(fallback_candidates), self.MAX_CONCEPTS)

        concept_records = [{"concept": concept, "type": _infer_type(concept)} for concept in concepts]

        sentences = _split_sentences(text)
        relations = []
        relations.extend(self._extract_marker_relations(sentences, concepts))
        relations.extend(self._extract_kb_relations(concepts))
        relations = _dedupe_relations(relations)
        relations = sorted(relations, key=lambda item: (-item.get("weight", 0.5), item.get("relation", "")))
        relations = self._apply_limit(relations, self.MAX_RELATIONS)

        if not relations:
            self.last_warnings.append("未抽取到显式关系，建议提高文本中的因果连接词密度。")

        logger.info("Local extractor: {} concepts, {} relations", len(concept_records), len(relations))
        return {"concepts": concept_records, "relations": relations}
