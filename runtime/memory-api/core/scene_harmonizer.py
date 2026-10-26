from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageFilter

from .scene_line_generator import SceneLineGeneratorRuntime


@dataclass
class SceneSketchHarmonizerRuntime:
    checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        harmonizer_path = self.checkpoint_path or os.getenv("TMCRA_SCENE_HARMONIZER_PATH", "").strip() or None
        self._runtime = SceneLineGeneratorRuntime(checkpoint_path=harmonizer_path) if harmonizer_path else SceneLineGeneratorRuntime()
        self._state: Dict[str, Any] = {
            "enabled": bool(self._runtime.status().get("enabled")),
            "model_id": "scene_sketch_harmonizer_v1",
            "checkpoint_path": harmonizer_path or self._runtime.status().get("checkpoint_path", ""),
            "fallback_model_id": self._runtime.status().get("model_id", "scene_line_generator_v1"),
            "loaded": bool(self._runtime.status().get("loaded")),
        }
        if self._runtime.status().get("error"):
            self._state["error"] = self._runtime.status()["error"]

    def status(self) -> Dict[str, Any]:
        payload = dict(self._state)
        payload["fallback_status"] = self._runtime.status()
        return payload

    def harmonize(self, *, base_image: Image.Image, condition_maps: np.ndarray) -> Image.Image | None:
        refined = self._runtime.refine(base_image=base_image, condition_maps=condition_maps)
        if refined is None:
            return None
        softened = refined.filter(ImageFilter.GaussianBlur(radius=0.35))
        lines = refined.convert("L").point(lambda value: 255 - value)
        lines = lines.filter(ImageFilter.GaussianBlur(radius=0.6))
        overlay = Image.merge("RGB", (lines, lines, lines))
        blended = Image.blend(softened.convert("RGB"), overlay, alpha=0.08)
        return blended
