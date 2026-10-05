from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency path
    torch = None
    nn = None


if nn is not None:
    class _ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
            )

        def forward(self, x):  # type: ignore[override]
            return self.net(x)


    class SceneLineGenerator(nn.Module):
        """Trainable second-stage whole-scene line sketch refiner."""

        def __init__(self, in_channels: int = 6, base_channels: int = 32):
            super().__init__()
            self.enc1 = _ConvBlock(in_channels, base_channels)
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = _ConvBlock(base_channels, base_channels * 2)
            self.pool2 = nn.MaxPool2d(2)
            self.enc3 = _ConvBlock(base_channels * 2, base_channels * 4)
            self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
            self.dec2 = _ConvBlock(base_channels * 4, base_channels * 2)
            self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
            self.dec1 = _ConvBlock(base_channels * 2, base_channels)
            self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

        def forward(self, x):  # type: ignore[override]
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            d2 = self.up2(e3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return torch.sigmoid(self.head(d1))


@dataclass
class SceneLineGeneratorRuntime:
    checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        self.checkpoint_path = self.checkpoint_path or os.getenv("TMCRA_SCENE_LINE_GENERATOR_PATH", "").strip() or None
        self._model = None
        self._state: Dict[str, Any] = {
            "enabled": False,
            "model_id": "scene_line_generator_v1",
            "checkpoint_path": self.checkpoint_path or "",
            "loaded": False,
        }
        if torch is None or nn is None or not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return
        try:
            model = SceneLineGenerator()
            payload = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = payload.get("state_dict") if isinstance(payload, dict) and "state_dict" in payload else payload
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            self._model = model
            self._state.update({"enabled": True, "loaded": True})
        except Exception as exc:  # pragma: no cover - runtime guard
            self._state["error"] = str(exc)

    def status(self) -> Dict[str, Any]:
        return dict(self._state)

    def refine(self, *, base_image: Image.Image, condition_maps: np.ndarray) -> Image.Image | None:
        if self._model is None or torch is None:
            return None
        height, width = condition_maps.shape[:2]
        tensor = torch.from_numpy(condition_maps.transpose(2, 0, 1)).unsqueeze(0).float()
        with torch.no_grad():
            prediction = self._model(tensor)[0, 0].cpu().numpy()
        prediction = np.clip(prediction, 0.0, 1.0)
        refined = base_image.convert("RGB").resize((width, height))
        arr = np.asarray(refined, dtype=np.uint8).copy()
        line_mask = prediction > 0.42
        arr[line_mask] = np.minimum(arr[line_mask], np.array([44, 52, 64], dtype=np.uint8))
        return Image.fromarray(arr, mode="RGB")
