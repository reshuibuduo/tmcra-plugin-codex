from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


ROLE_PRIORITIES = {
    "subject": 1.0,
    "focus": 1.0,
    "core_subject": 1.0,
    "support": 0.72,
    "environment": 0.62,
    "detail": 0.48,
}
DEPTH_VALUES = {"background": 0.2, "midground": 0.55, "foreground": 0.9}
SCENE_VALUES = {"scene": 0.2, "process": 0.6, "schematic": 0.9}


@dataclass(slots=True)
class NaturalLayoutTrainerConfig:
    max_objects: int = 12
    feature_dim: int = 10
    hidden_dim: int = 128


if nn is not None:
    class NaturalLayoutProposalNet(nn.Module):
        def __init__(self, config: NaturalLayoutTrainerConfig):
            super().__init__()
            self.config = config
            input_dim = config.max_objects * config.feature_dim
            self.net = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.max_objects * 5),
            )

        def forward(self, features):  # type: ignore[override]
            batch = features.shape[0]
            output = self.net(features.reshape(batch, -1))
            return output.reshape(batch, self.config.max_objects, 5)


    class NaturalLayoutRanker(nn.Module):
        def __init__(self, config: NaturalLayoutTrainerConfig):
            super().__init__()
            input_dim = config.max_objects * (config.feature_dim + 5)
            self.net = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )

        def forward(self, features, boxes):  # type: ignore[override]
            batch = features.shape[0]
            merged = torch.cat([features, boxes], dim=-1)
            return self.net(merged.reshape(batch, -1)).squeeze(-1)
else:  # pragma: no cover
    class NaturalLayoutProposalNet:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch not available; install torch to use NaturalLayoutProposalNet")


    class NaturalLayoutRanker:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch not available; install torch to use NaturalLayoutRanker")


def _role_value(value: Any) -> float:
    return ROLE_PRIORITIES.get(str(value or "").strip(), 0.4)


def _depth_value(value: Any) -> float:
    return DEPTH_VALUES.get(str(value or "").strip(), 0.55)


def _scene_value(value: Any) -> float:
    return SCENE_VALUES.get(str(value or "").strip(), 0.2)


def encode_layout_row(row: Dict[str, Any], *, max_objects: int = 12) -> Dict[str, Any]:
    layout_condition = row.get("layout_condition") or {}
    objects = list(layout_condition.get("objects") or row.get("objects") or [])
    boxes = list(row.get("object_boxes") or [])
    relation_graph = list(row.get("relation_graph") or [])
    canvas = layout_condition.get("canvas_size") or {"width": 1024, "height": 768}
    width = max(1.0, float(canvas.get("width", 1024) or 1024))
    height = max(1.0, float(canvas.get("height", 768) or 768))
    degree_map: Dict[str, int] = {}
    for edge in relation_graph:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source:
            degree_map[source] = degree_map.get(source, 0) + 1
        if target:
            degree_map[target] = degree_map.get(target, 0) + 1
    feature_rows: List[List[float]] = []
    target_boxes: List[List[float]] = []
    mask: List[float] = []
    scene_value = _scene_value(layout_condition.get("scene_type") or row.get("scene_type"))
    for index in range(max_objects):
        if index < len(boxes):
            box = boxes[index]
            meta = objects[index] if index < len(objects) else {}
            object_id = str(meta.get("id") or box.get("id") or f"obj_{index}")
            x = float(box.get("x", 0.0)) / width
            y = float(box.get("y", 0.0)) / height
            w = float(box.get("width", 0.0)) / width
            h = float(box.get("height", 0.0)) / height
            rotation = float(box.get("rotation", 0.0)) / 45.0
            role = _role_value(meta.get("role"))
            is_subject = 1.0 if role >= 0.95 else 0.0
            depth = _depth_value(meta.get("depth_band") or box.get("depth_band"))
            degree = min(1.0, degree_map.get(object_id, 0) / 6.0)
            importance = float(meta.get("importance", 1.0) or 1.0)
            importance = max(0.0, min(1.0, importance / 3.0 if importance > 1.0 else importance))
            feature_rows.append([x, y, w, h, role, is_subject, depth, degree, scene_value, importance])
            target_boxes.append([x, y, w, h, rotation])
            mask.append(1.0)
        else:
            feature_rows.append([0.0] * 10)
            target_boxes.append([0.0] * 5)
            mask.append(0.0)
    return {
        "features": feature_rows,
        "target_boxes": target_boxes,
        "mask": mask,
        "quality": float(row.get("layout_quality_score", row.get("naturalness_score", 0.8)) or 0.8),
        "scene_type": str(layout_condition.get("scene_type") or row.get("scene_type") or "scene"),
    }
