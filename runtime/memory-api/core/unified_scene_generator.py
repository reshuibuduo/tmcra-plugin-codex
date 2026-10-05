from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .scene_harmonizer import SceneSketchHarmonizerRuntime


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


NOISE_TERMS = {
    "透视",
    "近大远小",
    "远近",
    "线条草图",
    "草图",
    "风格",
    "构图",
    "布局",
    "可读性",
    "场景",
    "street scene",
    "clean line sketch",
    "clear perspective",
    "near large far small",
}


SPATIAL_PHRASES = ("前景", "中景", "背景", "foreground", "midground", "background")


class UnifiedSceneGenerator:
    """Whole-scene preview generator that avoids asset-by-asset symbol composition."""

    def __init__(self, renderer: Any):
        self.renderer = renderer
        self.harmonizer_runtime = SceneSketchHarmonizerRuntime()

    def render_scene(
        self,
        scene_spec: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
        title: str,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        scene = _copy(scene_spec)
        canvas = scene.get("canvas_size", {}) if isinstance(scene.get("canvas_size"), dict) else {}
        width = int(canvas.get("width", 1024))
        height = int(canvas.get("height", 768))
        image = Image.new("RGBA", (width, height), (*palette["background"], 255))
        draw = ImageDraw.Draw(image)

        self._draw_background(draw, scene, palette)
        selected_objects, skipped = self._select_objects(scene)

        for obj in selected_objects:
            self._draw_object(image, obj, palette)

        refined = self.harmonizer_runtime.harmonize(
            base_image=image.convert("RGB"),
            condition_maps=self._build_condition_maps(scene, selected_objects, width, height),
        )
        if refined is not None:
            image = refined.convert("RGBA")

        metadata = {
            "id": "unified_scene_v3",
            "title": str(title or ""),
            "selected_object_ids": [str(item.get("id", "")) for item in selected_objects],
            "selected_object_count": len(selected_objects),
            "skipped_objects": skipped,
            "second_stage_generator": self.harmonizer_runtime.status(),
        }
        return image.convert("RGB"), metadata

    def _draw_background(
        self,
        draw: ImageDraw.ImageDraw,
        scene: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
    ) -> None:
        width = int(scene.get("canvas_size", {}).get("width", 1024))
        height = int(scene.get("canvas_size", {}).get("height", 768))
        horizon_y = int(height * 0.58)
        draw.rectangle([0, 0, width, horizon_y], fill=self.renderer._with_alpha(palette["background"], 1.0))
        draw.rectangle(
            [0, horizon_y, width, height],
            fill=self.renderer._with_alpha(palette["region_fill"], 0.18),
        )
        for layer in sorted(scene.get("background_layers", []) or [], key=lambda item: item.get("z_index", 0)):
            layer_type = str(layer.get("type", "") or "")
            if layer_type == "road":
                self._draw_road(draw, layer, palette)
                continue
            self.renderer._draw_scene_background_layer(draw, layer, palette, filled=True)

    def _draw_road(
        self,
        draw: ImageDraw.ImageDraw,
        layer: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
    ) -> None:
        x0, y0, x1, y1 = self.renderer._scene_bbox(layer)
        top_y = int(y0 + (y1 - y0) * 0.12)
        road_poly = [(x0, y1), (x0 + 46, top_y), (x1 - 46, top_y), (x1, y1)]
        draw.polygon(road_poly, fill=self.renderer._with_alpha(palette["region_alt"], 0.26))
        draw.line([road_poly[0], road_poly[1]], fill=self.renderer._with_alpha(palette["line"], 0.44), width=2)
        draw.line([road_poly[2], road_poly[3]], fill=self.renderer._with_alpha(palette["line"], 0.44), width=2)

    def _select_objects(self, scene: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        selected: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        road_kept = False
        ranked = sorted(
            [_copy(item) for item in scene.get("object_instances", []) or []],
            key=self._object_priority,
            reverse=True,
        )
        for obj in ranked:
            reason = self._skip_reason(obj)
            if reason:
                skipped.append({"object_id": str(obj.get("id", "")), "reason": reason})
                continue
            asset_key = str(obj.get("asset_key") or obj.get("silhouette_key") or "").strip().lower()
            if asset_key == "road":
                if road_kept:
                    skipped.append({"object_id": str(obj.get("id", "")), "reason": "duplicate_road"})
                    continue
                road_kept = True
            duplicate_index = self._duplicate_index(selected, obj)
            if duplicate_index >= 0:
                skipped.append({"object_id": str(obj.get("id", "")), "reason": "overlap_duplicate"})
                continue
            selected.append(obj)
        selected.sort(key=lambda item: item.get("z_index", 0))
        return selected, skipped

    def _object_priority(self, obj: Dict[str, Any]) -> Tuple[float, float, float]:
        role = str(obj.get("role") or "")
        backend = str(obj.get("sketch_backend") or "")
        readability = float(obj.get("readability_rank", 0) or 0)
        role_score = 3.0 if role in {"subject", "focus", "core_subject"} else 2.0 if role == "support" else 1.0
        backend_score = 3.0 if backend == "trained" else 2.0 if backend == "hybrid" else 1.0
        area = float(obj.get("width", 0) or 0) * float(obj.get("height", 0) or 0)
        return role_score, backend_score + readability * 0.1, area

    def _skip_reason(self, obj: Dict[str, Any]) -> str | None:
        concept = str(obj.get("concept") or "").strip().lower()
        asset_key = str(obj.get("asset_key") or obj.get("silhouette_key") or "").strip().lower()
        variant = str(obj.get("shape_variant_id") or "")
        backend = str(obj.get("sketch_backend") or "")
        if variant.startswith("blob:") or asset_key == "blob":
            return "rule_blob"
        if any(term in concept for term in NOISE_TERMS):
            return "semantic_noise"
        if any(term in concept for term in SPATIAL_PHRASES) and backend == "rule":
            return "spatial_phrase_noise"
        if asset_key in {"generic_object", "generic_panel", "module"} and backend != "trained":
            return "generic_symbol"
        return None

    def _duplicate_index(self, selected: List[Dict[str, Any]], candidate: Dict[str, Any]) -> int:
        asset_key = str(candidate.get("asset_key") or candidate.get("silhouette_key") or "").strip().lower()
        cx0, cy0, cx1, cy1 = self.renderer._scene_bbox(candidate)
        c_area = max(1, (cx1 - cx0) * (cy1 - cy0))
        for index, existing in enumerate(selected):
            e_asset = str(existing.get("asset_key") or existing.get("silhouette_key") or "").strip().lower()
            if e_asset != asset_key:
                continue
            ex0, ey0, ex1, ey1 = self.renderer._scene_bbox(existing)
            inter_w = max(0, min(cx1, ex1) - max(cx0, ex0))
            inter_h = max(0, min(cy1, ey1) - max(cy0, ey0))
            inter_area = inter_w * inter_h
            e_area = max(1, (ex1 - ex0) * (ey1 - ey0))
            overlap = inter_area / float(min(c_area, e_area))
            if overlap >= 0.38:
                return index
        return -1

    def _draw_object(
        self,
        image: Image.Image,
        obj: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
    ) -> None:
        sprite = self.renderer._build_object_sprite(obj, palette, filled=False)
        x0, y0, x1, y1 = self.renderer._scene_bbox(obj)
        left = int(x0 + (x1 - x0) / 2 - sprite.width / 2)
        top = int(y0 + (y1 - y0) / 2 - sprite.height / 2)

        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        layer.alpha_composite(sprite.convert("RGBA"), (left, top))
        depth = str(obj.get("depth_band") or "")
        opacity = 0.96 if depth == "foreground" else 0.88 if depth == "midground" else 0.72
        if opacity < 1.0:
            alpha = layer.getchannel("A").point(lambda value: int(value * opacity))
            layer.putalpha(alpha)
        image.alpha_composite(layer)

    def _build_condition_maps(
        self,
        scene: Dict[str, Any],
        objects: List[Dict[str, Any]],
        width: int,
        height: int,
    ) -> np.ndarray:
        channels: List[np.ndarray] = []
        channels.append(self._mask_from_background(scene, width, height))
        channels.append(self._mask_from_objects(objects, width, height, roles={"subject", "focus", "core_subject"}))
        channels.append(self._mask_from_objects(objects, width, height, roles={"support", "environment", "detail"}))
        channels.append(self._mask_from_connectors(scene, width, height))
        channels.append(self._depth_map(objects, width, height))
        channels.append(self._line_seed(objects, width, height))
        return np.stack(channels, axis=2).astype(np.float32)

    def _mask_from_background(self, scene: Dict[str, Any], width: int, height: int) -> np.ndarray:
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        for layer in scene.get("background_layers", []) or []:
            x0, y0, x1, y1 = self.renderer._scene_bbox(layer)
            draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=255)
        return np.asarray(image, dtype=np.float32) / 255.0

    def _mask_from_objects(self, objects: List[Dict[str, Any]], width: int, height: int, *, roles: set[str]) -> np.ndarray:
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        for obj in objects:
            if str(obj.get("role") or "") not in roles:
                continue
            box = self.renderer._scene_bbox(obj)
            if not self.renderer._draw_object_mask(draw, box, obj, value=255):
                self.renderer._draw_asset_symbol_mask(
                    draw,
                    box,
                    str(obj.get("asset_key") or obj.get("silhouette_key") or "generic_object"),
                    value=255,
                )
        return np.asarray(image, dtype=np.float32) / 255.0

    def _mask_from_connectors(self, scene: Dict[str, Any], width: int, height: int) -> np.ndarray:
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        objects_by_id = {str(item.get("id") or ""): item for item in scene.get("object_instances", []) or []}
        for connector in scene.get("connectors", []) or []:
            if not connector.get("visible", True):
                continue
            from_obj = objects_by_id.get(str(connector.get("from_id") or ""))
            to_obj = objects_by_id.get(str(connector.get("to_id") or ""))
            if not from_obj or not to_obj:
                continue
            start, end = self.renderer._connector_points_for_scene(from_obj, to_obj)
            draw.line([start, end], fill=255, width=3)
        return np.asarray(image, dtype=np.float32) / 255.0

    def _depth_map(self, objects: List[Dict[str, Any]], width: int, height: int) -> np.ndarray:
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        for obj in objects:
            depth = str(obj.get("depth_band") or "")
            value = 224 if depth == "foreground" else 160 if depth == "midground" else 96
            draw.rounded_rectangle(self.renderer._scene_bbox(obj), radius=18, fill=value)
        return np.asarray(image, dtype=np.float32) / 255.0

    def _line_seed(self, objects: List[Dict[str, Any]], width: int, height: int) -> np.ndarray:
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        for obj in objects:
            box = self.renderer._scene_bbox(obj)
            if not self.renderer._draw_object_mask(draw, box, obj, value=255):
                continue
        return np.asarray(image, dtype=np.float32) / 255.0
