from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _normalize_en(text: str) -> str:
    text = text.strip().lower().replace("_", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def _normalize_zh(text: str) -> str:
    return text.strip()


def _guess_lang(text: str) -> str:
    return "zh" if _has_cjk(text) else "en"


class ConceptNetStore:
    """Read-only ConceptNet SQLite store."""

    def __init__(self, db_path: str, vocab_path: Optional[str] = None) -> None:
        self.db_path = Path(db_path)
        self.vocab_path = Path(vocab_path) if vocab_path else None
        self._conn: Optional[sqlite3.Connection] = None
        self._vocab: Optional[set[str]] = None
        self._concept_cache: Dict[Tuple[str, str], Optional[str]] = {}
        self._open()
        self._load_vocab()

    @classmethod
    def from_env(cls) -> Optional["ConceptNetStore"]:
        root = Path(__file__).resolve().parents[2]
        default_db = root / "data" / "kb" / "conceptnet_full.db"
        default_vocab = root / "data" / "kb" / "conceptnet_vocab.txt"
        db_path = Path(os.getenv("TMCRA_KB_PATH", str(default_db)))
        vocab_path = Path(os.getenv("TMCRA_KB_VOCAB", str(default_vocab)))
        if not db_path.exists():
            logger.warning("ConceptNet DB not found at {}", db_path)
            return None
        return cls(str(db_path), str(vocab_path) if vocab_path.exists() else None)

    def _open(self) -> None:
        if not self.db_path.exists():
            self._conn = None
            return
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _load_vocab(self) -> None:
        if not self.vocab_path or not self.vocab_path.exists():
            self._vocab = None
            return
        try:
            with self.vocab_path.open("r", encoding="utf-8") as handle:
                self._vocab = {line.strip() for line in handle if line.strip()}
            logger.info("Loaded ConceptNet vocab: {} entries", len(self._vocab))
        except Exception as exc:
            logger.warning("Failed to load vocab: {}", exc)
            self._vocab = None

    @property
    def available(self) -> bool:
        return self._conn is not None

    def _normalize(self, text: str, lang: Optional[str] = None) -> Tuple[str, str]:
        language = lang or _guess_lang(text)
        if language == "zh":
            return language, _normalize_zh(text)
        return language, _normalize_en(text)

    def concept_exists(self, text: str, lang: Optional[str] = None) -> bool:
        normalized_lang, normalized = self._normalize(text, lang)
        cache_key = (normalized_lang, normalized)
        if cache_key in self._concept_cache:
            return self._concept_cache[cache_key] is not None
        if self._vocab is not None:
            exists = normalized in self._vocab
            self._concept_cache[cache_key] = normalized if exists else None
            return exists
        if not self._conn:
            self._concept_cache[cache_key] = None
            return False
        row = self._conn.execute(
            "SELECT concept FROM concepts WHERE lang=? AND normalized=? LIMIT 1",
            (normalized_lang, normalized),
        ).fetchone()
        self._concept_cache[cache_key] = row["concept"] if row else None
        return row is not None

    def resolve_concept(self, text: str, lang: Optional[str] = None) -> Optional[str]:
        normalized_lang, normalized = self._normalize(text, lang)
        cache_key = (normalized_lang, normalized)
        if cache_key in self._concept_cache:
            return self._concept_cache[cache_key]
        if self._vocab is not None:
            if normalized in self._vocab:
                self._concept_cache[cache_key] = normalized
                return normalized
            self._concept_cache[cache_key] = None
            return None
        if not self._conn:
            self._concept_cache[cache_key] = None
            return None
        row = self._conn.execute(
            "SELECT concept FROM concepts WHERE lang=? AND normalized=? LIMIT 1",
            (normalized_lang, normalized),
        ).fetchone()
        concept = row["concept"] if row else None
        self._concept_cache[cache_key] = concept
        return concept

    def find_concepts(self, candidates: Iterable[str], lang: Optional[str] = None) -> List[str]:
        if not candidates:
            return []
        normalized_lang = lang or "en"
        normalized_map: Dict[str, str] = {}
        for item in candidates:
            language, normalized = self._normalize(item, lang)
            normalized_lang = language
            if normalized:
                normalized_map[normalized] = normalized
        if not normalized_map:
            return []
        normalized_list = list(normalized_map.keys())
        if self._vocab is not None:
            return [n for n in normalized_list if n in self._vocab]
        if not self._conn:
            return []
        results: List[str] = []
        chunk_size = 900
        for i in range(0, len(normalized_list), chunk_size):
            chunk = normalized_list[i:i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT concept, normalized FROM concepts WHERE lang=? AND normalized IN ({placeholders})"
            rows = self._conn.execute(query, (normalized_lang, *chunk)).fetchall()
            results.extend([row["concept"] for row in rows])
        return list(dict.fromkeys(results))

    def get_neighbors(self, concept: str, limit: int = 12) -> List[Dict]:
        if not self._conn:
            return []
        canonical = self.resolve_concept(concept) or concept
        rows = self._conn.execute(
            "SELECT source, target, relation, weight FROM edges WHERE source=? ORDER BY weight DESC LIMIT ?",
            (canonical, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_edges_between(self, concepts: Sequence[str], limit_per_concept: int = 25) -> List[Dict]:
        if not self._conn or not concepts:
            return []
        canonical_concepts = [self.resolve_concept(concept) or concept for concept in concepts]
        concept_set = set(canonical_concepts)
        results: List[Dict] = []
        chunk_size = 700
        for source in canonical_concepts:
            canonical = self.resolve_concept(source) or source
            targets = list(concept_set)
            for i in range(0, len(targets), chunk_size):
                chunk = targets[i:i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                query = (
                    f"SELECT source, target, relation, weight FROM edges "
                    f"WHERE source=? AND target IN ({placeholders}) "
                    f"ORDER BY weight DESC LIMIT ?"
                )
                rows = self._conn.execute(query, (canonical, *chunk, limit_per_concept)).fetchall()
                results.extend([dict(row) for row in rows])
        return results
