from __future__ import annotations

import hashlib
import json
import math
import os
import random
from typing import Any, Dict, List, Tuple

try:
    import torch
except Exception:  # pragma: no cover - optional dependency path
    torch = None

try:
    from .natural_layout_trainer import (
        NaturalLayoutProposalNet,
        NaturalLayoutRanker,
        NaturalLayoutTrainerConfig,
        encode_layout_row,
    )
except Exception:  # pragma: no cover - optional dependency path
    NaturalLayoutProposalNet = None  # type: ignore[assignment]
    NaturalLayoutRanker = None  # type: ignore[assignment]
    NaturalLayoutTrainerConfig = None  # type: ignore[assignment]
    encode_layout_row = None  # type: ignore[assignment]


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _engine_value(value: Any) -> str:
    raw = _clean_text(value).lower()
    if raw in {"mechanical", "mechanical_v1"}:
        return "mechanical_layout_v1"
    if raw in {"natural", "natural_v1", "natural_layout_v1"}:
        return "natural_layout_v1"
    return "auto"


def resolve_layout_engine(scene: Dict[str, Any] | None, sketch_options: Dict[str, Any] | None = None) -> str:
    env_value = _engine_value(os.getenv("TMCRA_LAYOUT_ENGINE", ""))
    option_value = _engine_value((sketch_options or {}).get("layout_engine"))
    scene_value = _engine_value(((scene or {}).get("layout_options") or {}).get("layout_engine"))
    for candidate in (option_value, scene_value, env_value):
        if candidate in {"natural_layout_v1", "mechanical_layout_v1"}:
            return candidate
    return "natural_layout_v1"


def resolve_layout_candidate_count(scene: Dict[str, Any] | None, sketch_options: Dict[str, Any] | None = None) -> int:
    for raw in (
        (sketch_options or {}).get("layout_candidate_count"),
        ((scene or {}).get("layout_options") or {}).get("layout_candidate_count"),
        os.getenv("TMCRA_LAYOUT_CANDIDATES", ""),
    ):
        try:
            value = int(raw)
        except Exception:
            continue
        if value > 0:
            return max(3, min(8, value))
    return 4


def _stable_seed(scene: Dict[str, Any], scene_type: str, salt: str = "") -> int:
    payload = {
        "scene_type": scene_type,
        "objects": [
            {
                "id": item.get("id"),
                "asset_key": item.get("asset_key"),
                "role": item.get("role"),
                "depth_band": item.get("depth_band"),
                "concept": item.get("concept"),
            }
            for item in scene.get("object_instances", []) or []
        ],
        "concept_order": scene.get("concept_order", []) or [],
        "summary": ((scene.get("render_hints") or {}).get("scene_summary") or ""),
        "salt": salt,
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _object_center(item: Dict[str, Any]) -> Tuple[float, float]:
    return float(item.get("x", 0)) + float(item.get("width", 0)) / 2.0, float(item.get("y", 0)) + float(item.get("height", 0)) / 2.0


def _role_priority(item: Dict[str, Any]) -> float:
    role = str(item.get("role") or "")
    if role in {"subject", "focus", "core_subject"}:
        return 3.0
    if role in {"support", "environment"}:
        return 2.0
    return 1.0


def _depth_rank(depth_band: str) -> int:
    return {"background": 0, "midground": 1, "foreground": 2}.get(str(depth_band or ""), 1)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _repo_checkpoint_path(filename: str) -> str | None:
    root = os.path.dirname(os.path.dirname(__file__))
    candidate = os.path.join(root, filename)
    return candidate if os.path.exists(candidate) else None


class NaturalLayoutModelRuntime:
    def __init__(self) -> None:
        self.proposal_path = (
            os.getenv("TMCRA_LAYOUT_PROPOSAL_PATH", "").strip()
            or _repo_checkpoint_path("tmp_layout_proposal.pt")
            or ""
        )
        self.ranker_path = (
            os.getenv("TMCRA_LAYOUT_RANKER_PATH", "").strip()
            or _repo_checkpoint_path("tmp_layout_ranker.pt")
            or ""
        )
        self._proposal = None
        self._ranker = None
        self._config = None
        self._state: Dict[str, Any] = {
            "enabled": False,
            "loaded": False,
            "proposal_path": self.proposal_path,
            "ranker_path": self.ranker_path,
            "model_id": "natural_layout_runtime_v1",
        }
        self._load()

    def _load(self) -> None:
        if (
            torch is None
            or NaturalLayoutProposalNet is None
            or NaturalLayoutRanker is None
            or NaturalLayoutTrainerConfig is None
            or encode_layout_row is None
            or not self.proposal_path
            or not self.ranker_path
            or not os.path.exists(self.proposal_path)
            or not os.path.exists(self.ranker_path)
        ):
            return
        try:
            proposal_payload = torch.load(self.proposal_path, map_location="cpu", weights_only=False)
            ranker_payload = torch.load(self.ranker_path, map_location="cpu", weights_only=False)
            config_payload = proposal_payload.get("config") or ranker_payload.get("config") or {}
            config = NaturalLayoutTrainerConfig(**config_payload)
            proposal = NaturalLayoutProposalNet(config)
            ranker = NaturalLayoutRanker(config)
            proposal.load_state_dict(proposal_payload.get("state_dict") or proposal_payload, strict=False)
            ranker.load_state_dict(ranker_payload.get("state_dict") or ranker_payload, strict=False)
            proposal.eval()
            ranker.eval()
            self._proposal = proposal
            self._ranker = ranker
            self._config = config
            self._state.update(
                {
                    "enabled": True,
                    "loaded": True,
                    "proposal_model_id": proposal_payload.get("model_id", "natural_layout_proposal_v1"),
                    "ranker_model_id": ranker_payload.get("model_id", "natural_layout_ranker_v1"),
                }
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            self._state["error"] = str(exc)

    def status(self) -> Dict[str, Any]:
        return dict(self._state)

    def build_candidate(self, scene: Dict[str, Any], scene_type: str, locked_ids: set[str]) -> Dict[str, Any] | None:
        if (
            self._proposal is None
            or self._ranker is None
            or self._config is None
            or encode_layout_row is None
            or torch is None
        ):
            return None
        objects = list(scene.get("object_instances", []) or [])
        if not objects:
            return None
        width = float(scene.get("canvas_size", {}).get("width", 1024) or 1024)
        height = float(scene.get("canvas_size", {}).get("height", 768) or 768)
        encoded = encode_layout_row(
            {
                "scene_type": scene_type,
                "layout_quality_score": float(scene.get("layout_score", 0.8) or 0.8),
                "layout_condition": {
                    "scene_type": scene_type,
                    "canvas_size": {"width": width, "height": height},
                    "objects": [
                        {
                            "id": str(item.get("id") or f"obj_{index}"),
                            "role": str(item.get("role") or ""),
                            "depth_band": str(item.get("depth_band") or "midground"),
                            "importance": float(item.get("importance", 1.0) or 1.0),
                        }
                        for index, item in enumerate(objects)
                    ],
                },
                "object_boxes": [
                    {
                        "id": str(item.get("id") or f"obj_{index}"),
                        "x": float(item.get("x", 0.0) or 0.0),
                        "y": float(item.get("y", 0.0) or 0.0),
                        "width": float(item.get("width", 0.0) or 0.0),
                        "height": float(item.get("height", 0.0) or 0.0),
                        "rotation": float(item.get("rotation", 0.0) or 0.0),
                        "depth_band": str(item.get("depth_band") or "midground"),
                    }
                    for index, item in enumerate(objects)
                ],
                "relation_graph": list(scene.get("connectors", []) or []),
            },
            max_objects=int(self._config.max_objects),
        )
        feature_tensor = torch.tensor(encoded["features"], dtype=torch.float32).unsqueeze(0)
        box_tensor = torch.tensor(encoded["target_boxes"], dtype=torch.float32).unsqueeze(0)
        proposal_input = feature_tensor.clone()
        proposal_input[..., :5] = box_tensor
        with torch.no_grad():
            predicted = self._proposal(proposal_input)
            rank_logit = self._ranker(feature_tensor, predicted)
            rank_score = float(torch.sigmoid(rank_logit)[0].item())
        updates: Dict[str, Dict[str, Any]] = {}
        active_count = min(len(objects), int(self._config.max_objects))
        subject_centers: List[Tuple[float, float]] = []
        for index, item in enumerate(objects[:active_count]):
            object_id = _clean_text(item.get("id"))
            if not object_id:
                continue
            if object_id in locked_ids:
                updates[object_id] = {
                    "x": int(item.get("x", 0)),
                    "y": int(item.get("y", 0)),
                    "width": int(item.get("width", 120)),
                    "height": int(item.get("height", 100)),
                    "rotation": float(item.get("rotation", 0.0) or 0.0),
                    "depth_band": str(item.get("depth_band") or "midground"),
                    "z_index": int(item.get("z_index", 20 + index)),
                }
                continue
            px, py, pw, ph, prot = [float(value) for value in predicted[0, index].tolist()]
            px = _clamp(px, 0.04, 0.92)
            py = _clamp(py, 0.06, 0.90)
            pw = _clamp(pw, 0.05, 0.8)
            ph = _clamp(ph, 0.05, 0.8)
            x = int(round(px * width))
            y = int(round(py * height))
            w = max(36, int(round(pw * width)))
            h = max(36, int(round(ph * height)))
            x = max(12, min(int(width) - w - 12, x))
            y = max(40, min(int(height) - h - 12, y))
            center_y = y + h / 2.0
            depth_band = "foreground" if center_y > height * 0.62 else "background" if center_y < height * 0.30 else "midground"
            z_index = 20 + _depth_rank(depth_band) * 20 + index
            if str(item.get("role") or "") in {"subject", "focus", "core_subject"}:
                z_index += 12
                subject_centers.append((x + w / 2.0, y + h / 2.0))
            updates[object_id] = {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "rotation": round(_clamp(prot * 45.0, -18.0, 18.0), 2),
                "depth_band": depth_band,
                "z_index": z_index,
            }
        focus_x = 0.5
        focus_y = 0.46
        if subject_centers:
            focus_x = round(sum(item[0] for item in subject_centers) / len(subject_centers) / width, 3)
            focus_y = round(sum(item[1] for item in subject_centers) / len(subject_centers) / height, 3)
        return {
            "id": "layout_model_1",
            "object_updates": updates,
            "camera_bias": {
                "focus_x": focus_x,
                "focus_y": focus_y,
                "perspective_strength": 0.58,
            },
            "model_score": round(rank_score, 4),
            "score_source": "proposal_ranker",
        }


_LAYOUT_MODEL_RUNTIME: NaturalLayoutModelRuntime | None = None


def _layout_model_runtime() -> NaturalLayoutModelRuntime:
    global _LAYOUT_MODEL_RUNTIME
    if _LAYOUT_MODEL_RUNTIME is None:
        _LAYOUT_MODEL_RUNTIME = NaturalLayoutModelRuntime()
    return _LAYOUT_MODEL_RUNTIME


def _negative_space_mask(scene: Dict[str, Any], objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    width = int(scene.get("canvas_size", {}).get("width", 1024))
    height = int(scene.get("canvas_size", {}).get("height", 768))
    occupancy = [0.0, 0.0, 0.0, 0.0]
    quadrants = [
        {"x": 0, "y": 0, "width": width * 0.5, "height": height * 0.5},
        {"x": width * 0.5, "y": 0, "width": width * 0.5, "height": height * 0.5},
        {"x": 0, "y": height * 0.5, "width": width * 0.5, "height": height * 0.5},
        {"x": width * 0.5, "y": height * 0.5, "width": width * 0.5, "height": height * 0.5},
    ]
    for item in objects:
        x0 = float(item.get("x", 0))
        y0 = float(item.get("y", 0))
        x1 = x0 + float(item.get("width", 0))
        y1 = y0 + float(item.get("height", 0))
        for index, quad in enumerate(quadrants):
            qx0 = quad["x"]
            qy0 = quad["y"]
            qx1 = qx0 + quad["width"]
            qy1 = qy0 + quad["height"]
            overlap_w = max(0.0, min(x1, qx1) - max(x0, qx0))
            overlap_h = max(0.0, min(y1, qy1) - max(y0, qy0))
            occupancy[index] += overlap_w * overlap_h
    ranked = sorted(range(len(quadrants)), key=lambda index: occupancy[index])
    return [
        {
            "id": f"negative_space_{slot + 1}",
            "x": int(quadrants[index]["x"]),
            "y": int(quadrants[index]["y"]),
            "width": int(quadrants[index]["width"]),
            "height": int(quadrants[index]["height"]),
            "weight": round(1.0 - (occupancy[index] / max(1.0, quadrants[index]["width"] * quadrants[index]["height"])), 3),
        }
        for slot, index in enumerate(ranked[:2], start=1)
    ]


def _candidate_object_updates(
    scene: Dict[str, Any],
    scene_type: str,
    locked_ids: set[str],
    *,
    candidate_index: int,
) -> Dict[str, Dict[str, Any]]:
    width = int(scene.get("canvas_size", {}).get("width", 1024))
    height = int(scene.get("canvas_size", {}).get("height", 768))
    objects = sorted(
        [_copy(item) for item in scene.get("object_instances", []) or []],
        key=lambda item: (-_role_priority(item), _clean_text(item.get("id"))),
    )
    rng = random.Random(_stable_seed(scene, scene_type, salt=f"candidate:{candidate_index}"))
    updates: Dict[str, Dict[str, Any]] = {}
    total = max(1, len(objects))
    scene_focus_slots = [(0.34, 0.43), (0.62, 0.41), (0.42, 0.54), (0.58, 0.52)]
    horizon_y = height * 0.56

    for index, item in enumerate(objects):
        object_id = _clean_text(item.get("id"))
        if not object_id:
            continue
        if object_id in locked_ids:
            updates[object_id] = {
                "x": int(item.get("x", 0)),
                "y": int(item.get("y", 0)),
                "width": int(item.get("width", 120)),
                "height": int(item.get("height", 100)),
                "rotation": float(item.get("rotation", 0.0) or 0.0),
                "depth_band": str(item.get("depth_band") or "midground"),
                "z_index": int(item.get("z_index", 20 + index)),
            }
            continue

        base_width = max(36, int(item.get("width", 120)))
        base_height = max(36, int(item.get("height", 100)))
        role = str(item.get("role") or "")
        asset_key = str(item.get("asset_key") or "")
        scale_jitter = 1.0 + rng.uniform(-0.14, 0.18)
        forced_depth_band = ""

        if role in {"subject", "focus", "core_subject"}:
            scale_jitter += 0.08
        elif role in {"detail"}:
            scale_jitter -= 0.05

        object_width = max(36, int(base_width * scale_jitter))
        object_height = max(36, int(base_height * scale_jitter))

        if scene_type == "process":
            process_slots = {
                "sun": (0.16, 0.16, "background"),
                "cloud": (0.58, 0.22, "background"),
                "vapor": (0.34, 0.48, "midground"),
                "raindrop": (0.64, 0.50, "midground"),
                "leaf": (0.54, 0.74, "foreground"),
                "cell": (0.50, 0.58, "midground"),
                "airplane": (0.48, 0.26, "background"),
                "energy_wave": (0.28, 0.34, "background"),
            }
            default_slot = (0.22 + 0.5 * (index / max(1, total - 1)), 0.54, "midground")
            base_x, base_y, forced_depth_band = process_slots.get(asset_key, default_slot)
            x_ratio = base_x + rng.uniform(-0.04, 0.04)
            y_ratio = base_y + rng.uniform(-0.04, 0.04)
            if forced_depth_band == "background":
                object_width = max(36, int(object_width * (0.88 + rng.uniform(-0.04, 0.04))))
                object_height = max(36, int(object_height * (0.88 + rng.uniform(-0.04, 0.04))))
            elif forced_depth_band == "foreground":
                object_width = max(36, int(object_width * (1.12 + rng.uniform(-0.04, 0.06))))
                object_height = max(36, int(object_height * (1.12 + rng.uniform(-0.04, 0.06))))
            rotation = rng.uniform(-8, 8)
        elif scene_type == "schematic":
            schematic_slots = {
                "battery": (0.18, 0.54, "midground"),
                "switch": (0.42, 0.34, "foreground"),
                "resistor": (0.50, 0.54, "midground"),
                "led": (0.78, 0.54, "midground"),
                "capacitor": (0.48, 0.76, "midground"),
                "diode": (0.66, 0.34, "midground"),
                "board": (0.50, 0.54, "background"),
            }
            default_slot = (0.22 + 0.5 * (index / max(1, total - 1)), 0.54, "midground")
            base_x, base_y, forced_depth_band = schematic_slots.get(asset_key, default_slot)
            x_ratio = base_x + rng.uniform(-0.035, 0.04)
            y_ratio = base_y + rng.uniform(-0.04, 0.04)
            if asset_key == "board":
                object_width = max(object_width, int(width * 0.44))
                object_height = max(object_height, int(height * 0.30))
            rotation = rng.uniform(-3, 3)
        else:
            focus_x, focus_y = scene_focus_slots[candidate_index % len(scene_focus_slots)]
            lane_bias = -1 if (index + candidate_index) % 2 == 0 else 1
            side_lane = 0.18 if lane_bias < 0 else 0.82
            depth_hint = str(item.get("depth_band") or "")
            if asset_key in {"sun"}:
                depth_hint = "background"
            elif asset_key in {"cloud"}:
                depth_hint = "background"
            elif asset_key in {"road", "car", "street_lamp", "person"}:
                depth_hint = "midground" if asset_key != "road" else "foreground"
            elif role in {"subject", "focus", "core_subject"}:
                depth_hint = depth_hint or "midground"
            elif not depth_hint:
                depth_hint = "background" if asset_key in {"building", "tree", "house"} else "midground"

            if depth_hint == "foreground":
                y_ratio = 0.74 + rng.uniform(-0.03, 0.04)
                depth_scale = 1.18 + rng.uniform(-0.04, 0.08)
            elif depth_hint == "background":
                y_ratio = 0.34 + rng.uniform(-0.05, 0.04)
                depth_scale = 0.86 + rng.uniform(-0.08, 0.04)
            else:
                y_ratio = 0.54 + rng.uniform(-0.05, 0.05)
                depth_scale = 1.0 + rng.uniform(-0.08, 0.06)

            object_width = max(36, int(object_width * depth_scale))
            object_height = max(36, int(object_height * depth_scale))

            x_ratio = focus_x + rng.uniform(-0.18, 0.18)
            if asset_key in {"building", "house"}:
                x_ratio = side_lane + rng.uniform(-0.08, 0.08)
                y_ratio = min(y_ratio, 0.5 + rng.uniform(-0.03, 0.03))
            elif asset_key == "tree":
                x_ratio = (0.22 if lane_bias < 0 else 0.76) + rng.uniform(-0.07, 0.07)
                y_ratio += 0.02
            elif asset_key == "street_lamp":
                x_ratio = (0.14 if lane_bias < 0 else 0.86) + rng.uniform(-0.03, 0.03)
                y_ratio = max(y_ratio, 0.58 + rng.uniform(-0.04, 0.04))
            elif asset_key == "car":
                x_ratio = 0.5 + rng.uniform(-0.12, 0.12)
                y_ratio = max(y_ratio, 0.7 + rng.uniform(-0.02, 0.03))
            elif role in {"subject", "focus", "core_subject"}:
                x_ratio = focus_x + rng.uniform(-0.08, 0.08)
                y_ratio = 0.5 + rng.uniform(-0.04, 0.05)
            if asset_key == "sun":
                x_ratio = 0.18 + 0.18 * (candidate_index % 3)
                y_ratio = 0.12 + rng.uniform(-0.02, 0.02)
            elif asset_key == "cloud":
                x_ratio = 0.24 + 0.18 * ((index + candidate_index) % 3)
                y_ratio = 0.18 + rng.uniform(-0.04, 0.03)
            elif asset_key == "road":
                x_ratio = 0.5
                y_ratio = 0.76
                object_width = max(object_width, int(width * 0.62))
                object_height = max(object_height, int(height * 0.16))
            if role not in {"subject", "focus", "core_subject"}:
                x_ratio = x_ratio + (0.06 if x_ratio < focus_x else -0.06) * rng.uniform(0.4, 1.0)
            y_ratio = max(y_ratio, (horizon_y / height) + 0.02 if depth_hint != "background" else 0.12)
            rotation = rng.uniform(-10, 10)

        x = int(width * _clamp(x_ratio, 0.12, 0.88) - object_width / 2)
        y = int(height * _clamp(y_ratio, 0.14, 0.84) - object_height / 2)
        x = max(12, min(width - object_width - 12, x))
        y = max(40, min(height - object_height - 12, y))

        if scene_type == "scene":
            depth_band = "foreground" if y > height * 0.58 else "background" if y < height * 0.32 else "midground"
        else:
            depth_band = forced_depth_band or ("midground" if role not in {"subject", "focus", "core_subject"} else "foreground")
        z_index = 20 + _depth_rank(depth_band) * 20 + index
        if role in {"subject", "focus", "core_subject"}:
            z_index += 12

        updates[object_id] = {
            "x": x,
            "y": y,
            "width": object_width,
            "height": object_height,
            "rotation": round(rotation, 2),
            "depth_band": depth_band,
            "z_index": z_index,
        }
    if scene_type == "scene":
        updates = _refine_scene_object_updates(scene, updates, candidate_index=candidate_index)
    return updates


def _refine_scene_object_updates(
    scene: Dict[str, Any],
    updates: Dict[str, Dict[str, Any]],
    *,
    candidate_index: int,
) -> Dict[str, Dict[str, Any]]:
    width = int(scene.get("canvas_size", {}).get("width", 1024))
    height = int(scene.get("canvas_size", {}).get("height", 768))
    objects_by_id = {str(item.get("id") or ""): item for item in scene.get("object_instances", []) or []}
    refined = _copy(updates)
    subject_boxes: List[Tuple[int, int, int, int]] = []

    for object_id, payload in refined.items():
        item = objects_by_id.get(object_id) or {}
        role = str(item.get("role") or "")
        asset_key = str(item.get("asset_key") or "")
        if role in {"subject", "focus", "core_subject"}:
            subject_boxes.append(
                (
                    int(payload.get("x", 0)),
                    int(payload.get("y", 0)),
                    int(payload.get("x", 0)) + int(payload.get("width", 0)),
                    int(payload.get("y", 0)) + int(payload.get("height", 0)),
                )
            )
        if asset_key in {"building", "house"}:
            payload["rotation"] = 0.0
        elif asset_key in {"road"}:
            payload["rotation"] = 0.0
            payload["x"] = max(0, min(width - int(payload["width"]), int(width * 0.5 - int(payload["width"]) / 2)))
        elif asset_key in {"street_lamp"}:
            payload["rotation"] = round(-2 + candidate_index * 0.8, 2)

    for object_id, payload in refined.items():
        item = objects_by_id.get(object_id) or {}
        role = str(item.get("role") or "")
        asset_key = str(item.get("asset_key") or "")
        if role in {"subject", "focus", "core_subject"} or asset_key in {"road", "sun", "cloud"}:
            continue
        x0 = int(payload.get("x", 0))
        y0 = int(payload.get("y", 0))
        x1 = x0 + int(payload.get("width", 0))
        y1 = y0 + int(payload.get("height", 0))
        for sx0, sy0, sx1, sy1 in subject_boxes:
            overlap_w = max(0, min(x1, sx1) - max(x0, sx0))
            overlap_h = max(0, min(y1, sy1) - max(y0, sy0))
            if overlap_w * overlap_h <= 0:
                continue
            shift = max(18, overlap_w + 12)
            if x0 < sx0:
                x0 = max(12, x0 - shift)
            else:
                x0 = min(width - int(payload["width"]) - 12, x0 + shift)
            y0 = min(height - int(payload["height"]) - 12, max(40, y0 + max(0, overlap_h // 2)))
            x1 = x0 + int(payload["width"])
            y1 = y0 + int(payload["height"])
        payload["x"] = x0
        payload["y"] = y0
    return refined


def _score_layout(scene: Dict[str, Any], objects: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    width = float(scene.get("canvas_size", {}).get("width", 1024) or 1024)
    height = float(scene.get("canvas_size", {}).get("height", 768) or 768)
    if not objects:
        return 0.0, {"focus_score": 0.0, "layering_score": 0.0, "negative_space_score": 0.0, "symmetry_penalty": 0.0, "connector_penalty": 0.0, "uniformity_penalty": 0.0}

    subject_objects = [item for item in objects if str(item.get("role") or "") in {"subject", "focus", "core_subject"}]
    focus_targets = [(width * 0.36, height * 0.42), (width * 0.62, height * 0.42)]
    focus_score = 0.0
    for subject in subject_objects or objects[:1]:
        center = _object_center(subject)
        nearest = min(math.dist(center, target) for target in focus_targets)
        focus_score += 1.0 - _clamp(nearest / max(width, height), 0.0, 1.0)
    focus_score /= max(1, len(subject_objects or objects[:1]))

    centers_x = sorted(_object_center(item)[0] for item in objects)
    x_gaps = [centers_x[index + 1] - centers_x[index] for index in range(len(centers_x) - 1)]
    if x_gaps:
        avg_gap = sum(x_gaps) / len(x_gaps)
        gap_variance = sum((gap - avg_gap) ** 2 for gap in x_gaps) / len(x_gaps)
        uniformity_penalty = 1.0 - _clamp(math.sqrt(gap_variance) / max(24.0, avg_gap), 0.0, 1.0)
    else:
        uniformity_penalty = 0.0

    left_weight = 0.0
    right_weight = 0.0
    for item in objects:
        area = float(item.get("width", 0)) * float(item.get("height", 0))
        if _object_center(item)[0] <= width / 2:
            left_weight += area
        else:
            right_weight += area
    symmetry_penalty = 1.0 - _clamp(abs(left_weight - right_weight) / max(1.0, left_weight + right_weight), 0.0, 1.0)

    y_centers = [_object_center(item)[1] for item in objects]
    y_span = max(y_centers) - min(y_centers) if len(y_centers) > 1 else 0.0
    depth_bands = {str(item.get("depth_band") or "midground") for item in objects}
    layering_score = 0.5 * _clamp(y_span / max(1.0, height * 0.42), 0.0, 1.0) + 0.5 * (len(depth_bands) / 3.0)

    occupancy = sum(float(item.get("width", 0)) * float(item.get("height", 0)) for item in objects) / max(1.0, width * height)
    negative_space_score = _clamp(1.0 - occupancy, 0.0, 1.0)

    objects_by_id = {str(item.get("id") or ""): item for item in objects}
    connector_vectors: List[float] = []
    for connector in scene.get("connectors", []) or []:
        if not connector.get("visible", True):
            continue
        from_obj = objects_by_id.get(str(connector.get("from_id") or ""))
        to_obj = objects_by_id.get(str(connector.get("to_id") or ""))
        if not from_obj or not to_obj:
            continue
        start = _object_center(from_obj)
        end = _object_center(to_obj)
        connector_vectors.append(abs(start[1] - end[1]) / max(24.0, abs(start[0] - end[0]) + abs(start[1] - end[1])))
    connector_penalty = sum(connector_vectors) / len(connector_vectors) if connector_vectors else 0.35

    semantic_score = 0.7
    scene_type = _clean_text(((scene.get("layout_options") or {}).get("scene_type")) or "scene")
    if scene_type == "scene":
        semantic_checks: List[float] = []
        for item in objects:
            asset_key = str(item.get("asset_key") or "").strip().lower()
            center_x, center_y = _object_center(item)
            x_ratio = center_x / max(1.0, width)
            y_ratio = center_y / max(1.0, height)
            visible_area = (float(item.get("width", 0) or 0) * float(item.get("height", 0) or 0)) / max(1.0, width * height)
            if asset_key in {"person", "dog"}:
                y_target = 1.0 - _clamp(abs(y_ratio - 0.68) / 0.24, 0.0, 1.0)
                x_target = 1.0 - _clamp(abs(x_ratio - 0.5) / 0.34, 0.0, 1.0)
                semantic_checks.append(0.65 * y_target + 0.35 * x_target)
            elif asset_key in {"house", "building"}:
                y_target = 1.0 - _clamp(abs(y_ratio - 0.52) / 0.22, 0.0, 1.0)
                side_target = _clamp(abs(x_ratio - 0.5) / 0.24, 0.0, 1.0)
                semantic_checks.append(0.68 * y_target + 0.32 * side_target)
            elif asset_key in {"tree", "street_lamp"}:
                y_target = 1.0 - _clamp(abs(y_ratio - 0.6) / 0.28, 0.0, 1.0)
                side_target = _clamp(abs(x_ratio - 0.5) / 0.28, 0.0, 1.0)
                semantic_checks.append(0.56 * y_target + 0.44 * side_target)
            elif asset_key in {"car", "road"}:
                y_target = 1.0 - _clamp(abs(y_ratio - 0.74) / 0.18, 0.0, 1.0)
                semantic_checks.append(y_target)
            elif asset_key == "sun":
                semantic_checks.append(1.0 - _clamp(abs(y_ratio - 0.14) / 0.12, 0.0, 1.0))
            elif asset_key == "cloud":
                semantic_checks.append(1.0 - _clamp(abs(y_ratio - 0.18) / 0.14, 0.0, 1.0))
            elif asset_key:
                semantic_checks.append(0.84 if visible_area <= 0.24 else 0.62)

        if semantic_checks:
            semantic_score = sum(semantic_checks) / len(semantic_checks)
        depth_kinds = {str(item.get("depth_band") or "") for item in objects if str(item.get("depth_band") or "")}
        if len(objects) >= 3 and len(depth_kinds) <= 1:
            semantic_score *= 0.72
        if subject_objects:
            subject_box = subject_objects[0]
            sx0 = float(subject_box.get("x", 0) or 0)
            sy0 = float(subject_box.get("y", 0) or 0)
            sx1 = sx0 + float(subject_box.get("width", 0) or 0)
            sy1 = sy0 + float(subject_box.get("height", 0) or 0)
            support_overlap_penalty = 0.0
            support_count = 0
            for item in objects:
                if item is subject_box:
                    continue
                ix0 = float(item.get("x", 0) or 0)
                iy0 = float(item.get("y", 0) or 0)
                ix1 = ix0 + float(item.get("width", 0) or 0)
                iy1 = iy0 + float(item.get("height", 0) or 0)
                overlap_w = max(0.0, min(ix1, sx1) - max(ix0, sx0))
                overlap_h = max(0.0, min(iy1, sy1) - max(iy0, sy0))
                if overlap_w * overlap_h <= 0:
                    continue
                support_count += 1
                overlap_ratio = (overlap_w * overlap_h) / max(1.0, (sx1 - sx0) * (sy1 - sy0))
                support_overlap_penalty += overlap_ratio
            if support_count:
                semantic_score *= max(0.55, 1.0 - (support_overlap_penalty / support_count) * 0.9)

    naturalness = (
        focus_score * 0.26
        + layering_score * 0.18
        + negative_space_score * 0.14
        + (1.0 - symmetry_penalty) * 0.08
        + connector_penalty * 0.06
        + (1.0 - uniformity_penalty) * 0.06
        + semantic_score * 0.22
    )
    features = {
        "focus_score": round(focus_score, 3),
        "layering_score": round(layering_score, 3),
        "negative_space_score": round(negative_space_score, 3),
        "symmetry_penalty": round(symmetry_penalty, 3),
        "connector_penalty": round(connector_penalty, 3),
        "uniformity_penalty": round(uniformity_penalty, 3),
        "semantic_score": round(semantic_score, 3),
    }
    return round(naturalness, 4), features


def _apply_candidate_to_objects(scene: Dict[str, Any], candidate: Dict[str, Any]) -> None:
    updates = candidate.get("object_updates") if isinstance(candidate.get("object_updates"), dict) else {}
    for item in scene.get("object_instances", []) or []:
        object_id = _clean_text(item.get("id"))
        payload = updates.get(object_id) if object_id else None
        if not isinstance(payload, dict):
            continue
        item["x"] = int(payload.get("x", item.get("x", 0)))
        item["y"] = int(payload.get("y", item.get("y", 0)))
        item["width"] = int(payload.get("width", item.get("width", 120)))
        item["height"] = int(payload.get("height", item.get("height", 100)))
        item["rotation"] = float(payload.get("rotation", item.get("rotation", 0.0) or 0.0))
        item["depth_band"] = str(payload.get("depth_band", item.get("depth_band", "midground")))
        item["z_index"] = int(payload.get("z_index", item.get("z_index", 20)))


def _manual_layout_payload(scene: Dict[str, Any]) -> Dict[str, Any]:
    objects = [_copy(item) for item in scene.get("object_instances", []) or []]
    score, features = _score_layout(scene, objects)
    return {
        "id": str(scene.get("layout_candidate_id") or "manual"),
        "score": score,
        "features": features,
        "camera_bias": _copy(scene.get("camera_bias") or {"focus_x": 0.5, "focus_y": 0.5, "perspective_strength": 0.4}),
        "negative_space_mask": _copy(scene.get("negative_space_mask") or _negative_space_mask(scene, objects)),
        "object_updates": {
            str(item.get("id")): {
                "x": int(item.get("x", 0)),
                "y": int(item.get("y", 0)),
                "width": int(item.get("width", 120)),
                "height": int(item.get("height", 100)),
                "rotation": float(item.get("rotation", 0.0) or 0.0),
                "depth_band": str(item.get("depth_band", "midground")),
                "z_index": int(item.get("z_index", 20)),
            }
            for item in objects
            if item.get("id")
        },
    }


def apply_natural_layout(scene_spec: Dict[str, Any] | None, sketch_options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    scene = _copy(scene_spec or {})
    scene.setdefault("layout_options", {})
    layout_options = scene["layout_options"]
    scene_type = _clean_text(layout_options.get("scene_type") or "scene")
    resolved_engine = resolve_layout_engine(scene, sketch_options)
    model_runtime = _layout_model_runtime()
    runtime_status = _copy(model_runtime.status())
    scene.setdefault("render_hints", {})
    layout_options["layout_engine"] = resolved_engine
    layout_options["layout_model_status"] = runtime_status
    if resolved_engine != "natural_layout_v1" or not (scene.get("object_instances") or []):
        scene["layout_engine"] = "mechanical_layout_v1" if resolved_engine == "mechanical_layout_v1" else resolved_engine
        scene.setdefault("layout_candidate_id", "mechanical")
        scene.setdefault("layout_score", 0.0)
        scene.setdefault("layout_features", {})
        scene.setdefault("camera_bias", {"focus_x": 0.5, "focus_y": 0.5, "perspective_strength": 0.3})
        scene.setdefault("negative_space_mask", [])
        scene["render_hints"]["layout_runtime"] = {
            **runtime_status,
            "fallback_used": True,
            "selected_source": "mechanical" if resolved_engine == "mechanical_layout_v1" else "bypass",
        }
        return scene

    if bool(layout_options.get("layout_manual_override")):
        current = _manual_layout_payload(scene)
        scene["layout_engine"] = "natural_layout_v1"
        scene["layout_candidate_id"] = current["id"]
        scene["layout_score"] = current["score"]
        scene["layout_features"] = current["features"]
        scene["camera_bias"] = current["camera_bias"]
        scene["negative_space_mask"] = current["negative_space_mask"]
        scene["layout_candidates"] = [current]
        scene["render_hints"]["layout_runtime"] = {
            **runtime_status,
            "fallback_used": False,
            "selected_source": "manual_override",
            "selected_candidate_id": current["id"],
        }
        return scene

    candidate_count = resolve_layout_candidate_count(scene, sketch_options)
    locked_ids = {
        str(item.get("id"))
        for item in scene.get("object_instances", []) or []
        if item.get("layout_locked") and item.get("id")
    }
    candidates: List[Dict[str, Any]] = []
    for candidate_index in range(candidate_count):
        candidate = {
            "id": f"layout_{candidate_index + 1}",
            "object_updates": _candidate_object_updates(scene, scene_type, locked_ids, candidate_index=candidate_index),
            "camera_bias": {
                "focus_x": round(0.36 + 0.08 * (candidate_index % 3), 3),
                "focus_y": round(0.42 + 0.05 * ((candidate_index + 1) % 2), 3),
                "perspective_strength": round(0.34 + 0.08 * (candidate_index % 4), 3),
            },
        }
        preview_scene = _copy(scene)
        _apply_candidate_to_objects(preview_scene, candidate)
        score, features = _score_layout(preview_scene, list(preview_scene.get("object_instances", []) or []))
        candidate["score"] = score
        candidate["features"] = features
        candidate["negative_space_mask"] = _negative_space_mask(preview_scene, list(preview_scene.get("object_instances", []) or []))
        candidates.append(candidate)

    model_candidate = model_runtime.build_candidate(scene, scene_type, locked_ids)
    if isinstance(model_candidate, dict):
        preview_scene = _copy(scene)
        _apply_candidate_to_objects(preview_scene, model_candidate)
        heuristic_score, heuristic_features = _score_layout(preview_scene, list(preview_scene.get("object_instances", []) or []))
        model_score = float(model_candidate.get("model_score", 0.0) or 0.0)
        combined_score = round(heuristic_score * 0.58 + model_score * 0.42, 4)
        model_candidate["score"] = combined_score
        model_candidate["features"] = {
            **heuristic_features,
            "model_score": round(model_score, 3),
            "heuristic_score": round(heuristic_score, 3),
            "score_source": "proposal_ranker+heuristic",
        }
        model_candidate["negative_space_mask"] = _negative_space_mask(preview_scene, list(preview_scene.get("object_instances", []) or []))
        candidates.append(model_candidate)

    requested_id = _clean_text(scene.get("layout_candidate_id") or layout_options.get("layout_candidate_id"))
    candidate_map = {candidate["id"]: candidate for candidate in candidates}
    chosen = candidate_map.get(requested_id)
    fallback_used = False
    if chosen is None:
        chosen = max(candidates, key=lambda item: float(item.get("score", 0.0) or 0.0))
        fallback_used = isinstance(model_candidate, dict) and str(chosen.get("id", "")) != str(model_candidate.get("id", ""))
    _apply_candidate_to_objects(scene, chosen)
    scene["layout_engine"] = "natural_layout_v1"
    scene["layout_candidate_id"] = str(chosen["id"])
    scene["layout_score"] = float(chosen.get("score", 0.0) or 0.0)
    scene["layout_features"] = _copy(chosen.get("features") or {})
    scene["camera_bias"] = _copy(chosen.get("camera_bias") or {})
    scene["negative_space_mask"] = _copy(chosen.get("negative_space_mask") or [])
    scene["render_hints"]["layout_runtime"] = {
        **runtime_status,
        "fallback_used": fallback_used,
        "selected_candidate_id": str(chosen["id"]),
        "selected_source": "model_candidate" if "model" in str(chosen.get("id", "")).lower() else "heuristic_candidate",
    }
    scene["layout_candidates"] = [
        {
            "id": str(candidate["id"]),
            "score": float(candidate.get("score", 0.0) or 0.0),
            "features": _copy(candidate.get("features") or {}),
            "camera_bias": _copy(candidate.get("camera_bias") or {}),
            "negative_space_mask": _copy(candidate.get("negative_space_mask") or []),
            "object_updates": _copy(candidate.get("object_updates") or {}),
        }
        for candidate in sorted(candidates, key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    ]
    return scene
