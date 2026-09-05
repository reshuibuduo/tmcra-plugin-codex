from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


class EmbeddingStore:
    """Lazy-loading embedding store for Numberbatch vectors."""

    def __init__(self, emb_path: str, vocab_path: str) -> None:
        self.emb_path = Path(emb_path)
        self.vocab_path = Path(vocab_path)
        self._vectors = None
        self._vocab: Dict[str, int] = {}
        self._load_vocab()

    @classmethod
    def from_env(cls) -> Optional["EmbeddingStore"]:
        root = Path(__file__).resolve().parents[2]
        default_emb = root / "data" / "kb" / "numberbatch.npy"
        default_vocab = root / "data" / "kb" / "numberbatch_vocab.json"
        emb_path = Path(os.getenv("TMCRA_EMB_PATH", str(default_emb)))
        vocab_path = Path(os.getenv("TMCRA_EMB_VOCAB", str(default_vocab)))
        if not emb_path.exists() or not vocab_path.exists():
            logger.warning("Embedding files not found: {} / {}", emb_path, vocab_path)
            return None
        return cls(str(emb_path), str(vocab_path))

    def _load_vocab(self) -> None:
        if not self.vocab_path.exists():
            return
        try:
            with self.vocab_path.open("r", encoding="utf-8") as handle:
                self._vocab = json.load(handle)
        except Exception as exc:
            logger.warning("Failed to load embedding vocab: {}", exc)
            self._vocab = {}

    def _load_vectors(self) -> None:
        if self._vectors is None and self.emb_path.exists():
            self._vectors = np.load(self.emb_path, mmap_mode="r")

    @property
    def available(self) -> bool:
        return bool(self._vocab) and self.emb_path.exists()

    def _normalize_key(self, concept: str) -> str:
        if _has_cjk(concept):
            return concept.strip()
        text = concept.strip().lower()
        return text.replace(" ", "_")

    def _resolve_index(self, concept: str) -> Optional[int]:
        if not self._vocab:
            return None
        key = self._normalize_key(concept)
        if key in self._vocab:
            return int(self._vocab[key])
        alt = key.replace("_", " ")
        if alt in self._vocab:
            return int(self._vocab[alt])
        if concept in self._vocab:
            return int(self._vocab[concept])
        if _has_cjk(concept):
            prefixed = f"/c/zh/{key}"
        else:
            prefixed = f"/c/en/{key}"
        if prefixed in self._vocab:
            return int(self._vocab[prefixed])
        return None

    def get_vector(self, concept: str) -> Optional[np.ndarray]:
        idx = self._resolve_index(concept)
        if idx is None:
            return None
        self._load_vectors()
        if self._vectors is None:
            return None
        return self._vectors[idx]

    def cosine_similarity(self, left: str, right: str) -> Optional[float]:
        vec_left = self.get_vector(left)
        vec_right = self.get_vector(right)
        if vec_left is None or vec_right is None:
            return None
        denom = float(np.linalg.norm(vec_left) * np.linalg.norm(vec_right))
        if denom <= 0:
            return None
        return float(np.dot(vec_left, vec_right) / denom)
