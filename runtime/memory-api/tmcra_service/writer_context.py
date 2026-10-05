from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping, Sequence


UNRESOLVED_CONTEXT_POLICY_VERSION = "tmcra.writer-unresolved-context.v1"
DEFAULT_UNRESOLVED_MAX_ITEMS = 64
DEFAULT_UNRESOLVED_MAX_CHARS = 16_000
_TERM_RE = re.compile(r"[a-z0-9_]{2,}|[\u3400-\u9fff]", re.IGNORECASE)


def writer_unresolved_limits_from_env() -> tuple[int, int]:
    try:
        max_items = int(
            os.getenv(
                "TMCRA_WRITER_UNRESOLVED_MAX_ITEMS",
                str(DEFAULT_UNRESOLVED_MAX_ITEMS),
            )
        )
        max_chars = int(
            os.getenv(
                "TMCRA_WRITER_UNRESOLVED_MAX_CHARS",
                str(DEFAULT_UNRESOLVED_MAX_CHARS),
            )
        )
    except ValueError as exc:
        raise ValueError("Writer unresolved-context limits must be integers") from exc
    if max_items <= 0 or max_chars < 1_000:
        raise ValueError("Writer unresolved-context limits are invalid")
    return max_items, max_chars


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _query_terms(messages: Sequence[Mapping[str, Any]]) -> set[str]:
    text: list[str] = []
    for message in messages:
        spans = message.get("source_spans")
        if not isinstance(spans, list):
            continue
        for span in spans:
            if isinstance(span, Mapping):
                text.append(str(span.get("text") or ""))
    return set(_TERM_RE.findall("\n".join(text).casefold()))


def select_unresolved_interactions(
    interactions: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, Any]],
    *,
    max_items: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    """Bound open-interaction context with deterministic recency and relevance."""

    if max_items <= 0 or max_chars < 1_000:
        raise ValueError("unresolved-context limits are invalid")
    items = [dict(item) for item in interactions]
    if len(items) <= max_items and len(compact_json(items)) <= max_chars:
        return items

    encoded = [compact_json(item) for item in items]
    query_terms = _query_terms(messages)
    recent_item_limit = max(1, max_items // 2)
    recent_char_limit = max(500, max_chars // 2)
    selected: set[int] = set()

    def selected_size(indices: set[int]) -> int:
        return len(compact_json([items[index] for index in sorted(indices)]))

    for index in range(len(items) - 1, -1, -1):
        if len(selected) >= recent_item_limit:
            break
        candidate = {*selected, index}
        if selected_size(candidate) <= recent_char_limit:
            selected = candidate

    scored: list[tuple[int, int]] = []
    if query_terms:
        for index, value in enumerate(encoded):
            if index in selected:
                continue
            overlap = len(query_terms.intersection(_TERM_RE.findall(value.casefold())))
            if overlap:
                scored.append((overlap, index))
        scored.sort(key=lambda row: (-row[0], -row[1]))

    scored_indices = {index for _score, index in scored}
    priorities = [index for _score, index in scored]
    priorities.extend(
        index
        for index in range(len(items) - 1, -1, -1)
        if index not in selected and index not in scored_indices
    )
    for index in priorities:
        if len(selected) >= max_items:
            break
        candidate = {*selected, index}
        if selected_size(candidate) <= max_chars:
            selected = candidate

    return [items[index] for index in sorted(selected)]
