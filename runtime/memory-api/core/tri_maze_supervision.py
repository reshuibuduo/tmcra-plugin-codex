from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence

from .concept_memory import ConceptMemory


CANONICAL_FIELDS = [
    "source_dataset",
    "sample_id",
    "query",
    "query_type",
    "focus_concept",
    "concepts",
    "forward_paths",
    "facts",
    "score",
    "source",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "more",
    "need",
    "necessary",
    "of",
    "on",
    "or",
    "should",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "will",
    "with",
    "would",
}

ATOMIC_RELATIONS = {
    "xNeed",
    "xEffect",
    "xIntent",
    "xWant",
    "xAttr",
    "oEffect",
    "oWant",
    "oReact",
    "xReact",
    "isBefore",
    "isAfter",
    "HasSubEvent",
    "HinderedBy",
}


@dataclass(slots=True)
class DatasetAdapter:
    dataset_id: str
    supervision_role: str
    convert: Callable[[Sequence[Dict[str, Any]]], List[Dict[str, Any]]]


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _collapse_ws(value.replace("\u00a0", " ").strip(" \t\r\n\"'"))
    return _collapse_ws(str(value))


def _truncate(text: str, *, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _iter_strings(value: Any) -> Iterator[str]:
    if value is None:
        return
    if isinstance(value, str):
        cleaned = _normalize_text(value)
        if cleaned:
            yield cleaned
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_strings(nested)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_strings(item)


def _first_text(container: Any, *keys: str) -> str:
    if not isinstance(container, dict):
        return ""
    for key in keys:
        value = container.get(key)
        if isinstance(value, str):
            cleaned = _normalize_text(value)
            if cleaned:
                return cleaned
    return ""


def _dedupe_preserve(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        cleaned = _normalize_text(value)
        if not cleaned:
            continue
        marker = cleaned.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(cleaned)
    return ordered


def _clamp_score(value: Any, *, default: float = 0.8) -> float:
    try:
        return max(0.1, min(1.0, float(value)))
    except Exception:
        return default


def _extract_keywords(*texts: str, limit: int = 12) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for text in texts:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text or ""):
            marker = token.casefold()
            if marker in STOPWORDS or marker in seen:
                continue
            seen.add(marker)
            ordered.append(token.lower())
            if len(ordered) >= limit:
                return ordered
    return ordered


def _infer_query_type(query: str) -> str:
    normalized = (query or "").strip().lower()
    if normalized.startswith("why ") or normalized.startswith("why?") or " why " in f" {normalized} ":
        return "explanation"
    if normalized.startswith("how ") or normalized.startswith("how do") or normalized.startswith("how can"):
        return "how_to"
    if normalized.startswith("what if") or "what would happen" in normalized:
        return "counterfactual"
    if normalized.startswith("is ") or normalized.startswith("are ") or normalized.startswith("does "):
        return "verification"
    if "need" in normalized or "necessary" in normalized or "why must" in normalized:
        return "necessity"
    return "query"


def _as_path_nodes(values: Iterable[str], *, min_length: int = 2) -> List[str]:
    cleaned = _dedupe_preserve(_truncate(_normalize_text(value), limit=220) for value in values)
    if len(cleaned) < min_length:
        return []
    return cleaned


def _path_to_facts(nodes: Sequence[str], *, relation: str, weight: float) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for index in range(len(nodes) - 1):
        facts.append(
            {
                "from": nodes[index],
                "relation": relation,
                "to": nodes[index + 1],
                "weight": max(0.1, min(1.0, float(weight))),
            }
        )
    return facts


def _extract_choice_text(row: Dict[str, Any], answer_key: str) -> str:
    if not answer_key:
        return ""
    choices_value = row.get("choices")
    if not choices_value and isinstance(row.get("question"), dict):
        choices_value = row["question"].get("choices")
    if not isinstance(choices_value, list):
        return ""
    normalized_key = answer_key.strip().upper()
    for choice in choices_value:
        if not isinstance(choice, dict):
            continue
        label = str(choice.get("label") or choice.get("key") or "").strip().upper()
        if label == normalized_key:
            return _normalize_text(choice.get("text") or choice.get("label_text") or "")
    return ""


def _statement_from_query_answer(query: str, answer: str) -> str:
    query_clean = _normalize_text(query)
    answer_clean = _normalize_text(answer)
    if not query_clean:
        return answer_clean
    if not answer_clean:
        return query_clean
    if query_clean.endswith("?"):
        return _collapse_ws(f"{query_clean[:-1]}: {answer_clean}")
    return _collapse_ws(f"{query_clean} => {answer_clean}")


def _collect_clauses(text: str, *, limit: int = 6) -> List[str]:
    raw = _normalize_text(text)
    if not raw:
        return []
    parts = re.split(r"(?:\s*;\s*|\s+\.\s+|\s*,\s+|\s+because\s+|\s+therefore\s+|\s+so\s+)", raw)
    return _dedupe_preserve(_truncate(part, limit=180) for part in parts if _normalize_text(part))[:limit]


def _normalize_fact(raw: Any) -> Dict[str, Any] | None:
    if isinstance(raw, dict):
        src = _normalize_text(raw.get("from") or raw.get("src") or raw.get("head"))
        relation = _normalize_text(raw.get("relation") or raw.get("predicate") or raw.get("type"))
        dst = _normalize_text(raw.get("to") or raw.get("dst") or raw.get("tail"))
        if not src or not relation or not dst:
            return None
        return {
            "from": _truncate(src),
            "relation": relation.lower(),
            "to": _truncate(dst),
            "weight": _clamp_score(raw.get("weight", 0.8), default=0.8),
        }
    if isinstance(raw, str):
        text = _normalize_text(raw)
        if "->" in text:
            left, right = text.split("->", 1)
            src = _normalize_text(left)
            dst = _normalize_text(right)
            if src and dst:
                return {"from": _truncate(src), "relation": "related_to", "to": _truncate(dst), "weight": 0.7}
    return None


def normalize_supervision_row(row: Dict[str, Any]) -> Dict[str, Any]:
    source_dataset = _normalize_text(row.get("source_dataset") or row.get("source") or "unknown").lower()
    sample_id = _normalize_text(row.get("sample_id") or row.get("id") or "")
    query = _truncate(_normalize_text(row.get("query") or row.get("question") or sample_id), limit=260)
    query_type = _normalize_text(row.get("query_type") or _infer_query_type(query)).lower() or "query"
    focus_concept = _truncate(_normalize_text(row.get("focus_concept") or ""), limit=220)
    concepts = _dedupe_preserve(_truncate(item, limit=120) for item in _iter_strings(row.get("concepts")))

    forward_paths: List[List[str]] = []
    raw_paths = row.get("forward_paths") or []
    if isinstance(raw_paths, list):
        for raw_path in raw_paths:
            if isinstance(raw_path, str):
                path = _as_path_nodes(re.split(r"\s*->\s*", raw_path))
            else:
                path = _as_path_nodes(_iter_strings(raw_path))
            if path:
                forward_paths.append(path)

    facts = []
    for raw_fact in row.get("facts") or []:
        normalized_fact = _normalize_fact(raw_fact)
        if normalized_fact:
            facts.append(normalized_fact)

    if not concepts:
        concept_seed = [query, focus_concept]
        for path in forward_paths:
            concept_seed.extend(path)
        for fact in facts:
            concept_seed.extend([fact["from"], fact["to"]])
        concepts = _extract_keywords(*concept_seed, limit=16)

    normalized = {
        "source_dataset": source_dataset or "unknown",
        "sample_id": sample_id or f"{source_dataset}-{hashlib.md5(query.encode('utf-8', errors='ignore')).hexdigest()[:12]}",
        "query": query,
        "query_type": query_type,
        "focus_concept": focus_concept,
        "concepts": concepts,
        "forward_paths": forward_paths,
        "facts": facts,
        "score": _clamp_score(row.get("score", 0.8), default=0.8),
        "source": _normalize_text(row.get("source") or source_dataset or "unknown").lower() or "unknown",
    }
    return normalized


def validate_supervision_row(row: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    normalized = normalize_supervision_row(row)
    if not normalized["source_dataset"]:
        issues.append("missing source_dataset")
    if not normalized["sample_id"]:
        issues.append("missing sample_id")
    if not normalized["query"]:
        issues.append("missing query")
    for path_index, path in enumerate(normalized["forward_paths"]):
        if len(path) < 2:
            issues.append(f"path[{path_index}] shorter than 2 nodes")
        for idx in range(len(path) - 1):
            if path[idx].casefold() == path[idx + 1].casefold():
                issues.append(f"path[{path_index}] contains self loop at step {idx}")
    for fact_index, fact in enumerate(normalized["facts"]):
        if not fact["from"] or not fact["relation"] or not fact["to"]:
            issues.append(f"fact[{fact_index}] is incomplete")
    return issues


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(normalize_supervision_row(row), ensure_ascii=False) + "\n")


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    target = Path(path)
    if target.is_dir():
        rows: List[Dict[str, Any]] = []
        for child in sorted(target.rglob("*")):
            if child.suffix.lower() not in {".json", ".jsonl"}:
                continue
            rows.extend(load_records(child))
        return rows

    if target.suffix.lower() == ".jsonl":
        return read_jsonl(target)

    payload = json.loads(target.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "data", "examples", "items", "train", "dev", "validation", "test"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def load_supervision_rows(path: str | Path) -> List[Dict[str, Any]]:
    return [normalize_supervision_row(item) for item in load_records(path)]


def summarize_supervision_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    memory: ConceptMemory | None = None,
) -> Dict[str, Any]:
    normalized_rows = [normalize_supervision_row(row) for row in rows]
    query_type_counts: Dict[str, int] = {}
    dataset_counts: Dict[str, int] = {}
    forward_path_count = 0
    forward_step_count = 0
    rows_with_paths = 0
    rows_with_facts = 0
    fact_count = 0
    invalid_rows = 0
    duplicate_paths = 0
    path_markers = set()
    concept_nodes = set()

    for row in normalized_rows:
        dataset_counts[row["source_dataset"]] = dataset_counts.get(row["source_dataset"], 0) + 1
        query_type_counts[row["query_type"]] = query_type_counts.get(row["query_type"], 0) + 1
        issues = validate_supervision_row(row)
        if issues:
            invalid_rows += 1
        if row["forward_paths"]:
            rows_with_paths += 1
        if row["facts"]:
            rows_with_facts += 1
        for concept in row["concepts"]:
            concept_nodes.add(concept)
        for path in row["forward_paths"]:
            forward_path_count += 1
            forward_step_count += max(0, len(path) - 1)
            concept_nodes.update(path)
            marker = " -> ".join(path).casefold()
            if marker in path_markers:
                duplicate_paths += 1
            else:
                path_markers.add(marker)
        for fact in row["facts"]:
            fact_count += 1
            concept_nodes.add(fact["from"])
            concept_nodes.add(fact["to"])

    overlap = {"base_concepts": 0, "matched_concepts": 0, "match_rate": 0.0}
    if memory is not None:
        base_concepts = {name.casefold() for name in memory.get_all_concepts().keys()}
        matched = {item for item in concept_nodes if item.casefold() in base_concepts}
        overlap = {
            "base_concepts": len(base_concepts),
            "matched_concepts": len(matched),
            "match_rate": round(len(matched) / max(1, len(concept_nodes)), 4),
        }

    avg_path_length = 0.0
    if forward_path_count > 0:
        avg_path_length = round((forward_step_count + forward_path_count) / forward_path_count, 4)

    return {
        "row_count": len(normalized_rows),
        "dataset_counts": dataset_counts,
        "query_type_counts": query_type_counts,
        "rows_with_paths": rows_with_paths,
        "rows_with_facts": rows_with_facts,
        "forward_path_count": forward_path_count,
        "forward_step_count": forward_step_count,
        "avg_path_length": avg_path_length,
        "fact_count": fact_count,
        "unique_concept_count": len(concept_nodes),
        "duplicate_path_count": duplicate_paths,
        "invalid_row_count": invalid_rows,
        "concept_overlap": overlap,
    }


def import_supervision_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    memory: ConceptMemory,
    min_path_len: int = 2,
    concept_type: str = "proposition",
) -> Dict[str, Any]:
    normalized_rows = [normalize_supervision_row(row) for row in rows]
    concept_attempts = 0
    fact_attempts = 0
    path_attempts = 0
    incoming_fact_total = sum(len(row.get("facts") or []) for row in normalized_rows)
    if getattr(memory, "max_facts", 0):
        memory.max_facts = max(int(memory.max_facts), len(memory.facts) + incoming_fact_total)

    path_lookup = {}
    for index, path_record in enumerate(memory.paths):
        normalized_path = memory._normalize_path_record(path_record)
        marker = (tuple(normalized_path.get("path") or []), normalized_path.get("mode") or "forward")
        path_lookup[marker] = index

    fact_lookup = {}
    for index, fact in enumerate(memory.facts):
        marker = (
            _normalize_text(fact.get("from")),
            _normalize_text(fact.get("relation")).lower(),
            _normalize_text(fact.get("to")),
        )
        fact_lookup[marker] = index

    for row in normalized_rows:
        dataset_id = row["source_dataset"]
        source_tag = f"offline_import:{dataset_id}"
        concept_seed: List[str] = list(row["concepts"])
        for path in row["forward_paths"]:
            concept_seed.extend(path)
        for fact in row["facts"]:
            concept_seed.extend([fact["from"], fact["to"]])

        for concept in _dedupe_preserve(concept_seed):
            existing = memory.concepts.get(concept)
            if existing is not None:
                existing["importance_score"] = max(
                    float(existing.get("importance_score", 0.0) or 0.0),
                    max(0.3, min(1.0, float(row["score"]))),
                )
                if concept_type != "general" and existing.get("type") == "unknown":
                    existing["type"] = concept_type
                if not existing.get("created_from"):
                    existing["created_from"] = dataset_id
            else:
                memory.concepts[concept] = {
                    "concept": concept,
                    "type": concept_type,
                    "importance_score": max(0.3, min(1.0, float(row["score"]))),
                    "created_from": dataset_id,
                }
            concept_attempts += 1

        for fact in row["facts"]:
            marker = (
                _normalize_text(fact["from"]),
                _normalize_text(fact["relation"]).lower(),
                _normalize_text(fact["to"]),
            )
            existing_fact_index = fact_lookup.get(marker)
            if existing_fact_index is not None:
                existing_fact = memory.facts[existing_fact_index]
                existing_fact["uses"] = int(existing_fact.get("uses", 1) or 1) + 1
                existing_fact["weight"] = max(float(existing_fact.get("weight", 0.7) or 0.7), float(fact["weight"]))
            else:
                memory.facts.append(
                    {
                        "from": fact["from"],
                        "relation": fact["relation"],
                        "to": fact["to"],
                        "weight": float(fact["weight"]),
                        "uses": 1,
                    }
                )
                fact_lookup[marker] = len(memory.facts) - 1
            fact_attempts += 1

        for path in row["forward_paths"]:
            if len(path) < min_path_len:
                continue
            marker = (tuple(path), "forward")
            existing_path_index = path_lookup.get(marker)
            if existing_path_index is not None:
                current = memory._normalize_path_record(memory.paths[existing_path_index])
                current["uses"] = current.get("uses", 1) + 1
                current["score"] = max(float(current.get("score", 0.8)), float(row["score"]))
                current["mode"] = "forward"
                current["source"] = source_tag
                memory.paths[existing_path_index] = current
            else:
                memory.paths.append(
                    memory._normalize_path_record(
                        {
                            "path": list(path),
                            "score": row["score"],
                            "uses": 1,
                            "mode": "forward",
                            "source": source_tag,
                        }
                    )
                )
                path_lookup[marker] = len(memory.paths) - 1
            for index in range(len(path) - 1):
                edge = (path[index], path[index + 1])
                memory.edge_counts[edge] += 1
            path_attempts += 1

    memory._cleanup_memory()

    return {
        "row_count": len(normalized_rows),
        "concept_attempts": concept_attempts,
        "fact_attempts": fact_attempts,
        "path_attempts": path_attempts,
        "memory_file": memory.memory_file,
    }


def merge_supervision_rows(inputs: Sequence[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for rows in inputs:
        for row in rows:
            normalized = normalize_supervision_row(row)
            marker = (normalized["source_dataset"], normalized["sample_id"])
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(normalized)
    return merged


def _clean_statement_text(text: Any) -> str:
    cleaned = _normalize_text(text)
    cleaned = re.sub(r"^(?:sent|statement|fact|int|hypothesis)\d*\s*[:=]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;")
    return _truncate(cleaned, limit=220)


def _is_placeholder_token(text: str) -> bool:
    return bool(re.fullmatch(r"(?i)(?:sent|statement|fact|int|hypothesis)\d+", _normalize_text(text)))


def _keep_statement(text: str, *, min_words: int = 3) -> bool:
    cleaned = _clean_statement_text(text)
    if not cleaned:
        return False
    if _is_placeholder_token(cleaned):
        return False
    words = re.findall(r"[A-Za-z0-9_-]+", cleaned)
    return len(words) >= min_words


def _proof_lookup(record: Dict[str, Any], key: str) -> Dict[str, str]:
    value = record.get(key)
    if not value and isinstance(record.get("meta"), dict):
        value = record["meta"].get(key)
    mapping: Dict[str, str] = {}
    if isinstance(value, dict):
        for raw_key, raw_value in value.items():
            cleaned = ""
            if isinstance(raw_value, dict):
                cleaned = _first_text(raw_value, "text", "sentence", "statement", "original_text")
                if not cleaned:
                    cleaned = next(iter(_iter_strings(raw_value)), "")
            elif isinstance(raw_value, list):
                cleaned = next(iter(_iter_strings(raw_value)), "")
            else:
                cleaned = _normalize_text(raw_value)
            cleaned = _clean_statement_text(cleaned)
            if _keep_statement(cleaned, min_words=2):
                normalized_key = _normalize_text(raw_key)
                mapping[normalized_key] = cleaned
                mapping[normalized_key.lower()] = cleaned
    return mapping


def _resolve_proof_token(token: str, lookup: Dict[str, str]) -> str:
    cleaned = _normalize_text(token)
    if not cleaned:
        return ""
    if cleaned in lookup:
        return lookup[cleaned]
    if cleaned.lower() in lookup:
        return lookup[cleaned.lower()]
    if _is_placeholder_token(cleaned):
        return ""
    resolved = _clean_statement_text(cleaned)
    if not _keep_statement(resolved, min_words=2):
        return ""
    return resolved


def _extract_entailment_question(record: Dict[str, Any]) -> str:
    question = _first_text(record, "question", "query")
    if question:
        return question
    meta = record.get("meta")
    if isinstance(meta, dict):
        question = _first_text(meta, "question", "question_text", "query")
        if question:
            return question
    if isinstance(record.get("answer"), dict):
        question = _first_text(record["answer"], "text")
        if question:
            return question
    return ""


def _extract_entailment_hypothesis(record: Dict[str, Any]) -> str:
    hypothesis = _first_text(record, "hypothesis", "answer")
    if hypothesis:
        return _clean_statement_text(hypothesis)
    meta = record.get("meta")
    if isinstance(meta, dict):
        hypothesis = _first_text(meta, "hypothesis", "answer")
        if hypothesis:
            return _clean_statement_text(hypothesis)
    return ""


def _extract_entailment_proof_steps(record: Dict[str, Any]) -> List[tuple[List[str], str]]:
    proof_text = _first_text(record, "full_text_proof", "proof")
    if not proof_text and isinstance(record.get("meta"), dict):
        proof_text = _first_text(record["meta"], "full_text_proof", "proof")
    if not proof_text:
        return []

    lookup: Dict[str, str] = {}
    lookup.update(_proof_lookup(record, "triples"))
    lookup.update(_proof_lookup(record, "intermediate_conclusions"))
    lookup.update(_proof_lookup(record, "sentences"))
    lookup.update(_proof_lookup(record, "worldtree_provenance"))

    steps: List[tuple[List[str], str]] = []
    for line in re.split(r"[;\n]+", proof_text):
        cleaned_line = _normalize_text(line)
        if "->" not in cleaned_line:
            continue
        left, right = cleaned_line.split("->", 1)
        output = _resolve_proof_token(right, lookup)
        if not output:
            continue
        inputs = []
        for token in re.split(r"\s*(?:and|&|,)\s*", left, flags=re.IGNORECASE):
            resolved = _resolve_proof_token(token, lookup)
            if resolved:
                inputs.append(resolved)
        inputs = _dedupe_preserve(inputs)
        if inputs:
            steps.append((inputs, output))
    return steps


def _extract_entailment_paths_and_facts(record: Dict[str, Any], *, max_paths: int = 4) -> tuple[List[List[str]], List[Dict[str, Any]]]:
    steps = _extract_entailment_proof_steps(record)
    hypothesis = _extract_entailment_hypothesis(record)
    if not steps:
        fallback_path = _as_path_nodes([hypothesis] if hypothesis else [])
        return ([fallback_path] if fallback_path else []), _path_to_facts(fallback_path, relation="entails", weight=0.95)

    produced_by: Dict[str, List[str]] = {}
    used_as_input = set()
    facts: List[Dict[str, Any]] = []
    for inputs, output in steps:
        produced_by.setdefault(output, [])
        for item in inputs:
            if item not in produced_by[output]:
                produced_by[output].append(item)
            used_as_input.add(item)
            facts.append({"from": item, "relation": "entails", "to": output, "weight": 0.95})

    terminal_targets: List[str] = []
    if hypothesis and hypothesis in produced_by:
        terminal_targets.append(hypothesis)
    elif hypothesis:
        candidate = _clean_statement_text(hypothesis)
        if candidate in produced_by:
            terminal_targets.append(candidate)
    if not terminal_targets:
        terminal_targets = [node for node in produced_by.keys() if node not in used_as_input]
    terminal_targets = _dedupe_preserve(terminal_targets)[:max_paths]

    def walk(node: str, trail: tuple[str, ...]) -> List[List[str]]:
        if node in trail:
            return []
        parents = produced_by.get(node) or []
        if not parents:
            return [[node]]
        paths: List[List[str]] = []
        for parent in parents:
            parent_paths = walk(parent, trail + (node,))
            if not parent_paths:
                paths.append([parent, node])
                continue
            for parent_path in parent_paths:
                candidate = parent_path + [node]
                if 2 <= len(candidate) <= 6:
                    paths.append(candidate)
        return paths

    raw_paths: List[List[str]] = []
    for target in terminal_targets:
        raw_paths.extend(walk(target, tuple()))

    cleaned_paths: List[List[str]] = []
    seen = set()
    for path in raw_paths:
        normalized_path = _as_path_nodes(path)
        if len(normalized_path) < 2:
            continue
        marker = tuple(node.casefold() for node in normalized_path)
        if marker in seen:
            continue
        seen.add(marker)
        cleaned_paths.append(normalized_path)
        if len(cleaned_paths) >= max_paths:
            break

    if not cleaned_paths and hypothesis:
        fallback_nodes = _as_path_nodes(list(produced_by.keys())[:1] + [hypothesis])
        if fallback_nodes:
            cleaned_paths.append(fallback_nodes)

    deduped_facts: List[Dict[str, Any]] = []
    fact_markers = set()
    for fact in facts:
        marker = (fact["from"].casefold(), fact["relation"], fact["to"].casefold())
        if marker in fact_markers:
            continue
        fact_markers.add(marker)
        deduped_facts.append(fact)
    return cleaned_paths, deduped_facts


def convert_entailmentbank(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        query = _extract_entailment_question(record)
        hypothesis = _extract_entailment_hypothesis(record)
        forward_paths, facts = _extract_entailment_paths_and_facts(record)
        concept_strings = [query, hypothesis]
        for path in forward_paths:
            concept_strings.extend(path)
        rows.append(
            normalize_supervision_row(
                {
                    "source_dataset": "entailmentbank",
                    "sample_id": _normalize_text(record.get("id") or record.get("uid") or f"entailmentbank-{index}"),
                    "query": query or hypothesis or f"entailmentbank sample {index}",
                    "query_type": _infer_query_type(query or hypothesis),
                    "focus_concept": hypothesis,
                    "concepts": _extract_keywords(*concept_strings, limit=16),
                    "forward_paths": forward_paths,
                    "facts": facts,
                    "score": 0.95,
                    "source": "entailmentbank",
                }
            )
        )
    return rows


def _extract_qasc_question(record: Dict[str, Any]) -> str:
    question_value = record.get("question")
    if isinstance(question_value, dict):
        return _first_text(question_value, "stem", "question")
    return _normalize_text(question_value or record.get("formatted_question") or record.get("question_text"))


def _extract_qasc_choice_context(record: Dict[str, Any], answer_key: str) -> List[str]:
    if not answer_key:
        return []
    question_value = record.get("question")
    if isinstance(question_value, dict):
        choices = question_value.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                label = _normalize_text(choice.get("label") or choice.get("key")).upper()
                if label != answer_key.upper():
                    continue
                support: List[str] = []
                support.extend(_iter_strings(choice.get("facts")))
                para = _normalize_text(choice.get("para") or choice.get("support"))
                if para:
                    support.extend(_collect_clauses(para, limit=2))
                return _dedupe_preserve(_clean_statement_text(item) for item in support if _keep_statement(item, min_words=3))
    return []


def _qasc_answer_statement(query: str, answer_text: str, combined: str) -> str:
    candidate = _clean_statement_text(combined)
    if _keep_statement(candidate, min_words=4):
        return candidate
    return _clean_statement_text(_statement_from_query_answer(query, answer_text))


def convert_qasc(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        query = _extract_qasc_question(record)
        answer_key = _normalize_text(record.get("answerKey") or record.get("answer_key"))
        answer_text = _extract_choice_text(record, answer_key) or _normalize_text(record.get("answer"))
        fact1 = _clean_statement_text(_first_text(record, "fact1"))
        fact2 = _clean_statement_text(_first_text(record, "fact2"))
        combined = _first_text(record, "combinedfact", "combined_fact", "explanation")
        answer_statement = _qasc_answer_statement(query, answer_text, combined)
        path_nodes = _as_path_nodes(
            [
                fact1 if _keep_statement(fact1, min_words=3) else "",
                fact2 if _keep_statement(fact2, min_words=3) else "",
                answer_statement if _keep_statement(answer_statement, min_words=3) else "",
            ]
        )
        choice_context = _extract_qasc_choice_context(record, answer_key)
        weak_support = _as_path_nodes(choice_context[:2] + ([answer_statement] if answer_statement else []))
        facts = _path_to_facts(path_nodes, relation="supports", weight=0.9)
        if not facts and weak_support:
            facts = _path_to_facts(weak_support, relation="supports", weight=0.55)
        forward_paths = [path_nodes] if path_nodes else []
        rows.append(
            normalize_supervision_row(
                {
                    "source_dataset": "qasc",
                    "sample_id": _normalize_text(record.get("id") or f"qasc-{index}"),
                    "query": query or f"qasc sample {index}",
                    "query_type": _infer_query_type(query),
                    "focus_concept": answer_text,
                    "concepts": _extract_keywords(query, fact1, fact2, answer_text, answer_statement, *choice_context, limit=16),
                    "forward_paths": forward_paths,
                    "facts": facts,
                    "score": 0.92 if forward_paths else 0.45,
                    "source": "qasc",
                }
            )
        )
    return rows


def _wiqa_answer_text(record: Dict[str, Any]) -> str:
    answer_key = _normalize_text(record.get("answer_label") or record.get("answerKey") or record.get("label"))
    if answer_key in {"A", "MORE"}:
        return "more likely / more"
    if answer_key in {"B", "LESS"}:
        return "less likely / less"
    if answer_key in {"C", "NO_EFFECT", "NO EFFECT"}:
        return "no effect"
    answer_text = _extract_choice_text(record, answer_key)
    if answer_text:
        return answer_text
    return _normalize_text(record.get("answer") or record.get("answer_text"))


def convert_wiqa(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        query = _first_text(record, "question_stem", "question", "stem")
        question_value = record.get("question")
        if not query and isinstance(question_value, dict):
            query = _first_text(question_value, "stem", "question")
        step_texts = _dedupe_preserve(
            [
                _first_text(record, "question_para_step", "question_para_steps"),
                *_iter_strings(record.get("steps")),
                *_iter_strings(record.get("metadata")),
            ]
        )
        answer_text = _wiqa_answer_text(record)
        outcome = _statement_from_query_answer(query or "wiqa effect", answer_text or "effect")
        path_nodes = _as_path_nodes(step_texts[:2] + [outcome])
        if not path_nodes and query and outcome:
            path_nodes = _as_path_nodes([query, outcome])
        facts = _path_to_facts(path_nodes, relation="causes", weight=0.8)
        rows.append(
            normalize_supervision_row(
                {
                    "source_dataset": "wiqa",
                    "sample_id": _normalize_text(record.get("id") or record.get("qid") or f"wiqa-{index}"),
                    "query": query or f"wiqa sample {index}",
                    "query_type": "counterfactual" if "what if" in (query or "").lower() else _infer_query_type(query),
                    "focus_concept": answer_text,
                    "concepts": _extract_keywords(query, answer_text, *path_nodes, limit=16),
                    "forward_paths": [path_nodes] if path_nodes else [],
                    "facts": facts,
                    "score": 0.8,
                    "source": "wiqa",
                }
            )
        )
    return rows


def convert_quartz(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        question_value = record.get("question")
        query = _normalize_text(question_value if isinstance(question_value, str) else "")
        if not query and isinstance(question_value, dict):
            query = _first_text(question_value, "stem", "question")
        para = _first_text(record, "para", "paragraph")
        annotations = _dedupe_preserve(_iter_strings(record.get("para_anno")))
        answer_key = _normalize_text(record.get("answerKey") or record.get("answer_key"))
        answer_text = _extract_choice_text(record, answer_key) or _normalize_text(record.get("answer"))
        path_nodes = _as_path_nodes(annotations[:2] or _collect_clauses(para, limit=2))
        if answer_text:
            path_nodes = _as_path_nodes(list(path_nodes) + [answer_text])
        facts = _path_to_facts(path_nodes, relation="qualitative_relation", weight=0.85)
        rows.append(
            normalize_supervision_row(
                {
                    "source_dataset": "quartz",
                    "sample_id": _normalize_text(record.get("id") or f"quartz-{index}"),
                    "query": query or para or f"quartz sample {index}",
                    "query_type": _infer_query_type(query or para),
                    "focus_concept": answer_text,
                    "concepts": _extract_keywords(query, para, answer_text, *path_nodes, limit=16),
                    "forward_paths": [path_nodes] if path_nodes else [],
                    "facts": facts,
                    "score": 0.85,
                    "source": "quartz",
                }
            )
        )
    return rows


def _convert_query_pool(records: Sequence[Dict[str, Any]], *, dataset_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        question_value = record.get("question")
        if isinstance(question_value, dict):
            query = _first_text(question_value, "stem", "question")
        else:
            query = _normalize_text(question_value or record.get("question_stem") or record.get("stem"))
        answer_key = _normalize_text(record.get("answerKey") or record.get("answer_key") or record.get("label"))
        answer_text = _extract_choice_text(record, answer_key) or _normalize_text(record.get("answer") or "")
        rows.append(
            normalize_supervision_row(
                {
                    "source_dataset": dataset_id,
                    "sample_id": _normalize_text(record.get("id") or f"{dataset_id}-{index}"),
                    "query": query or f"{dataset_id} sample {index}",
                    "query_type": _infer_query_type(query),
                    "focus_concept": answer_text,
                    "concepts": _extract_keywords(query, answer_text, limit=12),
                    "forward_paths": [],
                    "facts": [],
                    "score": 0.3,
                    "source": dataset_id,
                }
            )
        )
    return rows


def convert_arc(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _convert_query_pool(records, dataset_id="arc")


def convert_openbookqa(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _convert_query_pool(records, dataset_id="openbookqa")


def convert_commonsenseqa(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _convert_query_pool(records, dataset_id="commonsenseqa")


def convert_csqa2(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _convert_query_pool(records, dataset_id="csqa2")


def convert_atomic(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        event = _first_text(record, "event", "input")
        facts: List[Dict[str, Any]] = []
        concepts = [event]
        for relation in ATOMIC_RELATIONS:
            raw_targets = record.get(relation)
            targets = _dedupe_preserve(_iter_strings(raw_targets))
            for target in targets[:8]:
                facts.append(
                    {
                        "from": _truncate(event, limit=220),
                        "relation": relation.lower(),
                        "to": _truncate(target, limit=220),
                        "weight": 0.7,
                    }
                )
                concepts.append(target)
        rows.append(
            normalize_supervision_row(
                {
                    "source_dataset": "atomic",
                    "sample_id": _normalize_text(record.get("id") or f"atomic-{index}"),
                    "query": event or f"atomic sample {index}",
                    "query_type": "commonsense",
                    "focus_concept": event,
                    "concepts": _extract_keywords(*concepts, limit=16),
                    "forward_paths": [],
                    "facts": facts,
                    "score": 0.6,
                    "source": "atomic",
                }
            )
        )
    return rows


def convert_ai2d(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        query = _first_text(record, "question", "query", "caption")
        annotations = _dedupe_preserve(_iter_strings(record.get("annotations") or record.get("objects")))
        rows.append(
            normalize_supervision_row(
                {
                    "source_dataset": "ai2d",
                    "sample_id": _normalize_text(record.get("id") or record.get("image_id") or f"ai2d-{index}"),
                    "query": query or f"ai2d sample {index}",
                    "query_type": _infer_query_type(query),
                    "focus_concept": "",
                    "concepts": _extract_keywords(query, *annotations[:6], limit=16),
                    "forward_paths": [],
                    "facts": [],
                    "score": 0.25,
                    "source": "ai2d",
                }
            )
        )
    return rows


DATASET_ADAPTERS: Dict[str, DatasetAdapter] = {
    "entailmentbank": DatasetAdapter("entailmentbank", "strong_path_supervision", convert_entailmentbank),
    "qasc": DatasetAdapter("qasc", "strong_path_supervision", convert_qasc),
    "wiqa": DatasetAdapter("wiqa", "causal_path_supervision", convert_wiqa),
    "quartz": DatasetAdapter("quartz", "qualitative_relation_supervision", convert_quartz),
    "arc": DatasetAdapter("arc", "query_pool_eval", convert_arc),
    "openbookqa": DatasetAdapter("openbookqa", "query_pool_eval", convert_openbookqa),
    "commonsenseqa": DatasetAdapter("commonsenseqa", "hard_negative_query_pool", convert_commonsenseqa),
    "csqa2": DatasetAdapter("csqa2", "hard_negative_query_pool", convert_csqa2),
    "atomic": DatasetAdapter("atomic", "graph_fact_expansion", convert_atomic),
    "ai2d": DatasetAdapter("ai2d", "diagram_query_pool", convert_ai2d),
}


def available_dataset_ids() -> List[str]:
    return sorted(DATASET_ADAPTERS)


def convert_dataset_records(dataset_id: str, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    adapter = DATASET_ADAPTERS.get(dataset_id.strip().lower())
    if adapter is None:
        known = ", ".join(available_dataset_ids())
        raise ValueError(f"unknown dataset adapter '{dataset_id}'. available: {known}")
    return adapter.convert(records)
