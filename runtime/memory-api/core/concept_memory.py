"""
Concept long-term memory for TMCRA.
Stores concepts, successful paths, and explicit facts with relation types.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List

from loguru import logger


class ConceptMemory:
    """Long-term concept memory."""

    def __init__(self, memory_file: str = "data/concept_memory.json"):
        self.memory_file = memory_file
        self.max_concepts = None
        self.max_paths = None
        self.max_facts = 2000
        self.importance_threshold = 0.2
        self.reinforcement_factor = 0.05

        self.concepts: Dict[str, Dict] = {}
        self.paths: List[Dict] = []
        self.facts: List[Dict] = []
        self.edge_counts: Dict[tuple, int] = defaultdict(int)

        self._load_memory()
        logger.info(
            "✅ Concept memory initialized: {} concepts, {} paths, {} facts",
            len(self.concepts),
            len(self.paths),
            len(self.facts),
        )

    def _normalize_path_record(self, record: Dict) -> Dict:
        path_concepts = [str(item).strip() for item in (record.get("path") or []) if str(item).strip()]
        mode = str(record.get("mode") or "forward").strip().lower() or "forward"
        if mode not in {"forward", "reverse", "boundary"}:
            mode = "forward"
        source = str(record.get("source") or "engine_runtime").strip() or "engine_runtime"
        try:
            score = float(record.get("score", 0.8))
        except Exception:
            score = 0.8
        try:
            uses = int(record.get("uses", 1) or 1)
        except Exception:
            uses = 1

        normalized = {
            "path": path_concepts,
            "score": max(0.0, min(1.0, score)),
            "uses": max(1, uses),
            "mode": mode,
            "source": source,
        }
        for key, value in record.items():
            if key not in normalized:
                normalized[key] = value
        return normalized

    def _load_memory(self) -> None:
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        if not os.path.exists(self.memory_file):
            self._init_empty_memory()
            return
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.concepts = data.get("concepts", {})
            self.paths = [self._normalize_path_record(item) for item in data.get("paths", []) if isinstance(item, dict)]
            self.facts = data.get("facts", [])

            for path in self.paths:
                concept_list = path.get("path", [])
                for i in range(len(concept_list) - 1):
                    edge = (concept_list[i], concept_list[i + 1])
                    self.edge_counts[edge] += path.get("uses", 1)

            logger.info(
                "📥 Loaded memory: {} concepts, {} paths, {} facts",
                len(self.concepts),
                len(self.paths),
                len(self.facts),
            )
        except Exception as exc:
            logger.error("❌ Failed to load memory: {}", exc)
            self._init_empty_memory()

    def _init_empty_memory(self) -> None:
        self.concepts = {}
        self.paths = []
        self.facts = []
        self.edge_counts = defaultdict(int)
        self._save_memory()

    def _save_memory(self) -> None:
        try:
            data = {
                "concepts": self.concepts,
                "paths": [self._normalize_path_record(item) for item in self.paths],
                "facts": self.facts,
            }
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error("❌ Failed to save memory: {}", exc)

    def _cleanup_memory(self) -> None:
        if self.max_concepts and self.max_concepts > 0 and len(self.concepts) > self.max_concepts:
            sorted_concepts = sorted(
                self.concepts.items(),
                key=lambda x: x[1].get("importance_score", 0),
                reverse=True,
            )
            keep_count = int(self.max_concepts * 0.9)
            self.concepts = dict(sorted_concepts[:keep_count])
            logger.info("🧹 Trim concepts: kept {}", keep_count)

        if self.max_paths and self.max_paths > 0 and len(self.paths) > self.max_paths:
            sorted_paths = sorted(
                self.paths,
                key=lambda x: (x.get("uses", 0), x.get("score", 0)),
                reverse=True,
            )
            keep_count = int(self.max_paths * 0.9)
            self.paths = sorted_paths[:keep_count]
            logger.info("🧹 Trim paths: kept {}", keep_count)

        if len(self.facts) > self.max_facts:
            sorted_facts = sorted(
                self.facts,
                key=lambda x: (x.get("uses", 0), x.get("weight", 0)),
                reverse=True,
            )
            keep_count = int(self.max_facts * 0.9)
            self.facts = sorted_facts[:keep_count]
            logger.info("🧹 Trim facts: kept {}", keep_count)

        self._save_memory()

    def update_concept_importance(self, concept: str, score_delta: float = 0.1):
        if concept in self.concepts:
            self.concepts[concept]["importance_score"] = min(
                1.0,
                self.concepts[concept].get("importance_score", 0) + score_delta,
            )
        else:
            self.concepts[concept] = {
                "concept": concept,
                "type": "unknown",
                "importance_score": max(0.3, score_delta),
                "created_from": "inferred",
            }
        self._save_memory()

    def add_concept(
        self,
        concept: str,
        concept_type: str = "general",
        created_from: str = "original",
        importance_score: float = 0.5,
    ):
        if concept in self.concepts:
            self.concepts[concept]["importance_score"] = max(
                self.concepts[concept]["importance_score"], importance_score
            )
            if concept_type != "general" and self.concepts[concept].get("type") == "unknown":
                self.concepts[concept]["type"] = concept_type
        else:
            self.concepts[concept] = {
                "concept": concept,
                "type": concept_type,
                "importance_score": importance_score,
                "created_from": created_from,
            }
        self._cleanup_memory()

    def save_successful_path(
        self,
        path_concepts: List[str],
        score: float = 0.8,
        *,
        mode: str = "forward",
        source: str = "engine_runtime",
    ):
        if not path_concepts or len(path_concepts) < 2:
            return
        normalized_mode = str(mode or "forward").strip().lower() or "forward"
        if normalized_mode not in {"forward", "reverse", "boundary"}:
            normalized_mode = "forward"
        normalized_source = str(source or "engine_runtime").strip() or "engine_runtime"
        existing_idx = None
        for i, p in enumerate(self.paths):
            normalized = self._normalize_path_record(p)
            if normalized.get("path") == path_concepts and normalized.get("mode") == normalized_mode:
                existing_idx = i
                break
        if existing_idx is not None:
            current = self._normalize_path_record(self.paths[existing_idx])
            current["uses"] = current.get("uses", 1) + 1
            current["score"] = max(float(current.get("score", 0.8)), score)
            current["mode"] = normalized_mode
            current["source"] = normalized_source
            self.paths[existing_idx] = current
        else:
            self.paths.append(
                self._normalize_path_record(
                    {
                        "path": path_concepts,
                        "score": score,
                        "uses": 1,
                        "mode": normalized_mode,
                        "source": normalized_source,
                    }
                )
            )

        for i in range(len(path_concepts) - 1):
            edge = (path_concepts[i], path_concepts[i + 1])
            self.edge_counts[edge] += 1
        for concept in path_concepts:
            self.update_concept_importance(concept, 0.05)

        self._cleanup_memory()

    def add_fact(self, src: str, relation: str, dst: str, weight: float = 0.7):
        if not src or not dst or not relation:
            return
        for fact in self.facts:
            if fact.get("from") == src and fact.get("to") == dst and fact.get("relation") == relation:
                fact["uses"] = fact.get("uses", 1) + 1
                fact["weight"] = max(fact.get("weight", weight), weight)
                self._cleanup_memory()
                return
        self.facts.append(
            {"from": src, "to": dst, "relation": relation, "weight": weight, "uses": 1}
        )
        self._cleanup_memory()

    def retrieve_related_concepts(self, concept: str, max_results: int = 10) -> List[Dict]:
        def _is_injectable(item: Dict) -> bool:
            concept_type = str(item.get("type", "")).strip().lower()
            return concept_type not in {"ngram", "unknown"}

        related: List[Dict] = []
        concept_lower = concept.lower()
        if concept in self.concepts:
            related.append(self.concepts[concept].copy())
        for path in self.paths:
            path_concepts = path.get("path", [])
            if concept in path_concepts:
                idx = path_concepts.index(concept)
                if idx > 0:
                    prev_concept = path_concepts[idx - 1]
                    if prev_concept in self.concepts and not any(c["concept"] == prev_concept for c in related):
                        candidate = self.concepts[prev_concept].copy()
                        if _is_injectable(candidate):
                            related.append(candidate)
                if idx < len(path_concepts) - 1:
                    next_concept = path_concepts[idx + 1]
                    if next_concept in self.concepts and not any(c["concept"] == next_concept for c in related):
                        candidate = self.concepts[next_concept].copy()
                        if _is_injectable(candidate):
                            related.append(candidate)
        if len(concept_lower.strip()) >= 2:
            for mem_concept, data in self.concepts.items():
                if concept_lower in mem_concept.lower() and not any(c["concept"] == mem_concept for c in related):
                    candidate = data.copy()
                    if _is_injectable(candidate):
                        related.append(candidate)
                    if len(related) >= max_results:
                        break
        related.sort(key=lambda x: x.get("importance_score", 0), reverse=True)
        return related[:max_results]

    def get_related_facts(self, concept: str, max_results: int = 10) -> List[Dict]:
        related = [f for f in self.facts if f.get("from") == concept or f.get("to") == concept]
        related.sort(key=lambda x: (x.get("uses", 0), x.get("weight", 0)), reverse=True)
        return related[:max_results]

    def get_edge_reinforcement(self, from_concept: str, to_concept: str) -> float:
        edge = (from_concept, to_concept)
        count = self.edge_counts.get(edge, 0)
        return min(1.0, count * self.reinforcement_factor)

    def get_all_concepts(self) -> Dict[str, Dict]:
        return self.concepts.copy()

    def get_all_paths(self) -> List[Dict]:
        return [self._normalize_path_record(item) for item in self.paths]

    def get_all_facts(self) -> List[Dict]:
        return self.facts.copy()

    def clear_memory(self):
        self._init_empty_memory()
        logger.warning("🧹 All memory cleared.")


if __name__ == "__main__":
    memory = ConceptMemory()
    memory.add_concept("Arduino", "device", "original", 0.8)
    memory.add_concept("LED", "component", "expansion", 0.7)
    memory.save_successful_path(["Arduino", "LED", "电阻", "GND"], 0.9)
    memory.add_fact("电阻", "用于", "限制电流", 0.85)
    print("Facts:", memory.get_related_facts("电阻"))
