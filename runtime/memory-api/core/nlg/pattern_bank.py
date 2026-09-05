from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional


class PatternBank:
    """Lightweight pattern bank for relation-to-sentence patterns."""

    def __init__(self, patterns: Dict[str, List[Dict]], meta: Optional[Dict] = None) -> None:
        self.patterns = patterns
        self.meta = meta or {}

    @classmethod
    def load(cls, path: str) -> "PatternBank":
        file_path = Path(path)
        if not file_path.exists():
            return cls({})
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "_meta" in data:
            meta = data.get("_meta") or {}
            patterns = {k: v for k, v in data.items() if k != "_meta"}
            return cls(patterns or {}, meta=meta)
        return cls(data or {})

    def pick(self, category: str, *, top_k: int = 6) -> Optional[str]:
        items = self.patterns.get(category, [])
        if not items:
            return None
        items = sorted(items, key=lambda item: -item.get("count", 0))
        pool = items[: max(top_k, 1)]
        weights = [max(item.get("count", 1), 1) for item in pool]
        return random.choices([item.get("pattern") for item in pool], weights=weights, k=1)[0]

    def pick_connector(self) -> Optional[str]:
        connectors = self.meta.get("connectors", [])
        if not connectors:
            return None
        connectors = sorted(connectors, key=lambda item: -item.get("count", 0))
        pool = connectors[:6]
        weights = [max(item.get("count", 1), 1) for item in pool]
        return random.choices([item.get("text") for item in pool], weights=weights, k=1)[0]

    def pick_intro(self, intent: str) -> Optional[str]:
        intros = self.meta.get("intros", {}).get(intent)
        if not intros:
            return None
        intros = sorted(intros, key=lambda item: -item.get("count", 0))
        pool = intros[:4]
        weights = [max(item.get("count", 1), 1) for item in pool]
        return random.choices([item.get("text") for item in pool], weights=weights, k=1)[0]
