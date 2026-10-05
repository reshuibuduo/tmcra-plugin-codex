from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .scene_harmonizer import SceneSketchHarmonizerRuntime


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


DISPLAY_LABELS = {
    "building": "Building",
    "house": "House",
    "window": "Window",
    "door": "Door",
    "tree": "Tree",
    "cloud": "Cloud",
    "sun": "Sun",
    "person": "Person",
    "car": "Car",
    "street_lamp": "Lamp",
    "table": "Table",
    "chair": "Chair",
    "dog": "Dog",
    "desk_lamp": "Desk lamp",
    "road": "Road",
}

REGION_LABELS = {
    "roofline": "Roof",
    "window_row": "Windows",
    "door_zone": "Door",
    "facade": "Facade",
    "head": "Head",
    "face": "Face",
    "beard": "Beard",
    "torso": "Torso",
    "legs": "Legs",
    "canopy": "Canopy",
    "trunk": "Trunk",
    "body": "Body",
    "windshield": "Windshield",
    "wheel_front": "Front wheel",
    "wheel_rear": "Rear wheel",
    "front_wheel": "Front wheel",
    "rear_wheel": "Rear wheel",
    "cabin": "Cabin",
    "body": "Body",
    "lamp_head": "Lamp head",
    "pole": "Pole",
    "light_cone": "Light cone",
}


class WholeSceneSketchGenerator:
    """Render a full-scene clean-line sketch and emit synchronized annotations."""

    def __init__(self, renderer: Any):
        self.renderer = renderer
        self.harmonizer_runtime = SceneSketchHarmonizerRuntime()

    def render_scene(
        self,
        scene_spec: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
        title: str,
        *,
        annotated: bool = False,
        show_regions: bool = False,
        include_title: bool = False,
        view_mode: str = "structure",
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        scene = _copy(scene_spec)
        layout_options = scene.get("layout_options", {}) if isinstance(scene.get("layout_options"), dict) else {}
        canvas_size = (
            int(scene.get("canvas_size", {}).get("width", 1024)),
            int(scene.get("canvas_size", {}).get("height", 768)),
        )
        generator_style = str(layout_options.get("generator_style") or "clean_line")
        image = Image.new("RGBA", canvas_size, (*palette["background"], 255))
        draw = ImageDraw.Draw(image)

        if view_mode != "rich_preview":
            self._draw_scene_skeleton(draw, scene, palette)
        self._draw_background_layers(draw, scene, palette, view_mode=view_mode)
        if view_mode != "rich_preview":
            self._draw_connectors(draw, scene, palette)
        self._draw_objects(draw, scene, palette, generator_style=generator_style, view_mode=view_mode)

        refined = self.harmonizer_runtime.harmonize(
            base_image=image.convert("RGB"),
            condition_maps=self._build_condition_maps(scene),
        )
        if refined is not None:
            image = refined.convert("RGBA")
            draw = ImageDraw.Draw(image)

        annotation_bundle = self.build_annotation_bundle(
            scene,
            generator_style=generator_style,
            title=title,
            second_stage_status=self.harmonizer_runtime.status(),
        )

        if include_title and title:
            draw.text((24, 20), self._safe_text(title), fill=palette["text"], font=self.renderer.font)
            draw.text(
                (24, 50),
                f"{layout_options.get('scene_type', 'scene')} | {generator_style} | whole_scene_structural_v2 | {view_mode}",
                fill=self.renderer._with_alpha(palette["text"], 0.8),
                font=self.renderer.font,
            )
        if annotated:
            self._draw_annotation_overlay(draw, annotation_bundle, palette, show_regions=show_regions)
        return image.convert("RGB"), annotation_bundle

    def build_annotation_bundle(
        self,
        scene_spec: Dict[str, Any],
        *,
        generator_style: str,
        title: str,
        second_stage_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        scene = _copy(scene_spec)
        layout_options = scene.get("layout_options", {}) if isinstance(scene.get("layout_options"), dict) else {}
        canvas = scene.get("canvas_size", {}) if isinstance(scene.get("canvas_size"), dict) else {}
        scene_type = str(layout_options.get("scene_type", "scene"))
        objects = sorted(scene.get("object_instances", []) or [], key=lambda item: (item.get("z_index", 0), item.get("id", "")))
        background_layers = sorted(scene.get("background_layers", []) or [], key=lambda item: (item.get("z_index", 0), item.get("id", "")))
        connectors = [item for item in scene.get("connectors", []) or [] if item.get("visible", True)]

        object_annotations: List[Dict[str, Any]] = []
        region_annotations: List[Dict[str, Any]] = []
        for obj in objects:
            object_annotations.append(
                {
                    "object_id": str(obj.get("id", "")),
                    "concept": str(obj.get("concept", "")),
                    "display_label": self._display_label(obj),
                    "asset_key": str(obj.get("asset_key", "")),
                    "role": str(obj.get("role", "")),
                    "depth_band": str(obj.get("depth_band", "")),
                    "bbox": self._bbox_payload(self.renderer._scene_bbox(obj)),
                    "label_anchor": self._label_anchor_for_object(obj),
                    "editable": bool(obj.get("editable", True)),
                    "controls": self._edit_controls_for_object(obj),
                    "style_variant": generator_style,
                    "render_representation": str(obj.get("render_representation", "shape_recipe")),
                    "shape_variant_id": str(obj.get("shape_variant_id", "")),
                    "stroke_variant_id": str(obj.get("stroke_variant_id", "")),
                    "shape_recipe_source": str(obj.get("shape_recipe_source", "")),
                    "stroke_payload_source": str(obj.get("stroke_payload_source", "")),
                    "sketch_backend": str(obj.get("sketch_backend", "")),
                    "readability_rank": int(obj.get("readability_rank", 0) or 0),
                }
            )
            for region in obj.get("region_masks", []) or []:
                if not isinstance(region, dict):
                    continue
                region_box = self.renderer._region_box_for_object(obj, region)
                region_annotations.append(
                    {
                        "object_id": str(obj.get("id", "")),
                        "region_id": str(region.get("id", "")),
                        "label": str(region.get("label", "") or region.get("id", "")),
                        "display_label": self._region_display_label(region),
                        "bbox": self._bbox_payload(region_box),
                        "shape": str(region.get("shape", "rect")),
                        "actions": list(region.get("actions") or []),
                        "label_anchor": {"x": int(region_box[2] + 12), "y": int(region_box[1] + 18)},
                    }
                )

        connector_annotations: List[Dict[str, Any]] = []
        objects_by_id = {str(item.get("id", "")): item for item in objects}
        for connector in connectors:
            from_obj = objects_by_id.get(str(connector.get("from_id", "")))
            to_obj = objects_by_id.get(str(connector.get("to_id", "")))
            if not from_obj or not to_obj:
                continue
            start, end = self.renderer._connector_points_for_scene(from_obj, to_obj)
            connector_annotations.append(
                {
                    "connector_id": str(connector.get("id", "")),
                    "type": str(connector.get("type", "relation")),
                    "label": str(connector.get("label", "")),
                    "display_label": self._connector_display_label(connector),
                    "from_id": str(connector.get("from_id", "")),
                    "to_id": str(connector.get("to_id", "")),
                    "start": {"x": int(start[0]), "y": int(start[1])},
                    "end": {"x": int(end[0]), "y": int(end[1])},
                    "label_anchor": {"x": int((start[0] + end[0]) / 2), "y": int((start[1] + end[1]) / 2 - 18)},
                }
            )

        background_annotations = [
            {
                "layer_id": str(layer.get("id", "")),
                "type": str(layer.get("type", "panel")),
                "label": str(layer.get("label", "")),
                "display_label": self._layer_display_label(layer),
                "bbox": self._bbox_payload(self.renderer._scene_bbox(layer)),
            }
            for layer in background_layers
        ]

        return {
            "version": 2,
            "generator_id": "whole_scene_structural_v2",
            "title": self._safe_text(title),
            "scene_type": scene_type,
            "sketch_style": generator_style,
            "canvas_size": {"width": int(canvas.get("width", 1024)), "height": int(canvas.get("height", 768))},
            "quality_targets": {
                "subject_coverage": "required",
                "scene_layering": "required",
                "editable_control": "required",
                "line_cleanliness": "required",
                "detail_readability": "required",
            },
            "second_stage_generator": second_stage_status,
            "layout_runtime": _copy(((scene.get("render_hints") or {}).get("layout_runtime") or {})),
            "scene_markers": self._scene_markers(scene),
            "background_annotations": background_annotations,
            "object_annotations": object_annotations,
            "region_annotations": region_annotations,
            "connector_annotations": connector_annotations,
            "edit_layers": {
                "object_edit_count": sum(1 for item in object_annotations if item.get("editable")),
                "region_edit_count": len(region_annotations),
                "connector_edit_count": len(connector_annotations),
            },
            "metrics": {
                "background_count": len(background_annotations),
                "object_count": len(object_annotations),
                "region_count": len(region_annotations),
                "connector_count": len(connector_annotations),
            },
        }

    def _draw_scene_skeleton(
        self,
        draw: ImageDraw.ImageDraw,
        scene: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
    ) -> None:
        scene_type = str((scene.get("layout_options", {}) or {}).get("scene_type", "scene"))
        if scene_type != "scene":
            return
        width = int(scene.get("canvas_size", {}).get("width", 1024))
        height = int(scene.get("canvas_size", {}).get("height", 768))
        horizon_y = int(height * 0.56)
        vanishing_x = int(width * 0.54)
        guide = self.renderer._with_alpha(palette["guide"], 0.18)
        draw.line([(0, horizon_y), (width, horizon_y)], fill=guide, width=1)
        for offset in (-0.28, -0.1, 0.12, 0.28):
            x = int(width * (0.5 + offset))
            draw.line([(x, height), (vanishing_x, horizon_y)], fill=guide, width=1)

    def _draw_background_layers(
        self,
        draw: ImageDraw.ImageDraw,
        scene: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
        *,
        view_mode: str,
    ) -> None:
        for layer in sorted(scene.get("background_layers", []) or [], key=lambda item: item.get("z_index", 0)):
            layer_type = str(layer.get("type", "panel"))
            x0, y0, x1, y1 = self.renderer._scene_bbox(layer)
            if layer_type == "sky":
                if view_mode == "rich_preview":
                    draw.line([(0, y1), (x1, y1)], fill=self.renderer._with_alpha(palette["guide"], 0.08), width=1)
                    continue
                draw.line([(0, y1), (x1, y1)], fill=self.renderer._with_alpha(palette["guide"], 0.18), width=1)
                continue
            if layer_type == "road":
                top_y = int(y0 + (y1 - y0) * 0.12)
                bottom_y = int(y1)
                mid_x = int((x0 + x1) / 2)
                road_poly = [(x0, bottom_y), (x0 + 50, top_y), (x1 - 50, top_y), (x1, bottom_y)]
                if view_mode == "rich_preview":
                    draw.polygon(road_poly, fill=self.renderer._with_alpha(palette["region_alt"], 0.16))
                    draw.line([road_poly[0], road_poly[1]], fill=self.renderer._with_alpha(palette["line"], 0.36), width=2)
                    draw.line([road_poly[2], road_poly[3]], fill=self.renderer._with_alpha(palette["line"], 0.36), width=2)
                else:
                    draw.line(road_poly + [road_poly[0]], fill=self.renderer._with_alpha(palette["line"], 0.5), width=2)
                    self.renderer._draw_dashed_line(
                        draw,
                        (mid_x, top_y + 10),
                        (mid_x, bottom_y - 14),
                        palette["accent"],
                        width=2,
                        dash_length=16,
                    )
                continue
            self.renderer._draw_scene_background_layer(draw, layer, palette, filled=view_mode == "rich_preview")

    def _draw_connectors(
        self,
        draw: ImageDraw.ImageDraw,
        scene: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
    ) -> None:
        objects_by_id = {item["id"]: item for item in scene.get("object_instances", []) or [] if item.get("id")}
        for connector in scene.get("connectors", []) or []:
            if not connector.get("visible", True):
                continue
            from_obj = objects_by_id.get(str(connector.get("from_id", "")))
            to_obj = objects_by_id.get(str(connector.get("to_id", "")))
            if not from_obj or not to_obj:
                continue
            start, end = self.renderer._connector_points_for_scene(from_obj, to_obj)
            if str(connector.get("type", "relation")) == "beam":
                self._draw_light_beam(draw, start, end, palette)
                continue
            self.renderer._draw_scene_connector(draw, connector, objects_by_id, palette, show_labels=False)

    def _draw_objects(
        self,
        draw: ImageDraw.ImageDraw,
        scene: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
        *,
        generator_style: str,
        view_mode: str,
    ) -> None:
        for obj in sorted(scene.get("object_instances", []) or [], key=lambda item: item.get("z_index", 0)):
            self._render_object(draw, obj, palette, generator_style=generator_style, view_mode=view_mode)

    def _render_object(
        self,
        draw: ImageDraw.ImageDraw,
        obj: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
        *,
        generator_style: str,
        view_mode: str,
    ) -> None:
        bbox = self.renderer._scene_bbox(obj)
        asset_key = str(obj.get("asset_key", "") or obj.get("silhouette_key", "")).strip().lower()
        if view_mode == "rich_preview":
            rendered = False
            if str(obj.get("render_representation") or "") == "stroke_native" or obj.get("stroke_payload"):
                rendered = self.renderer._draw_stroke_payload(
                    draw,
                    bbox,
                    obj.get("stroke_payload") if isinstance(obj.get("stroke_payload"), list) else [],
                    palette,
                    style_variant=generator_style,
                    stroke_render_profile=obj.get("stroke_render_profile") if isinstance(obj.get("stroke_render_profile"), dict) else obj.get("stroke_style_profile"),
                )
            if not rendered:
                shape_recipe = obj.get("shape_recipe") if isinstance(obj.get("shape_recipe"), dict) else {}
                rendered = self.renderer._draw_shape_recipe(
                    draw,
                    bbox,
                    shape_recipe,
                    palette,
                    filled=False,
                    style_variant=generator_style,
                )
            if rendered:
                return
        if asset_key == "building":
            self._draw_building(draw, bbox, palette)
            return
        if asset_key == "house":
            self._draw_house(draw, bbox, palette)
            return
        if asset_key == "tree":
            self._draw_tree(draw, bbox, palette)
            return
        if asset_key == "person":
            self._draw_person(draw, bbox, palette)
            return
        if asset_key == "car":
            self._draw_car(draw, bbox, palette)
            return
        if asset_key == "street_lamp":
            self._draw_street_lamp(draw, bbox, palette)
            return
        if asset_key == "road":
            self._draw_road(draw, bbox, palette)
            return
        if asset_key == "window":
            self._draw_window(draw, bbox, palette)
            return
        if asset_key == "door":
            self._draw_door(draw, bbox, palette)
            return
        rendered = False
        if str(obj.get("render_representation") or "") == "stroke_native" or obj.get("stroke_payload"):
            rendered = self.renderer._draw_stroke_payload(
                draw,
                bbox,
                obj.get("stroke_payload") if isinstance(obj.get("stroke_payload"), list) else [],
                palette,
                style_variant=generator_style,
                stroke_render_profile=obj.get("stroke_render_profile") if isinstance(obj.get("stroke_render_profile"), dict) else obj.get("stroke_style_profile"),
            )
        if not rendered:
            shape_recipe = obj.get("shape_recipe") if isinstance(obj.get("shape_recipe"), dict) else {}
            rendered = self.renderer._draw_shape_recipe(
                draw,
                bbox,
                shape_recipe,
                palette,
                filled=False,
                style_variant=generator_style,
            )
        if not rendered:
            self.renderer._draw_asset_symbol(draw, bbox, asset_key or "generic_object", palette, filled=False)

    def _draw_annotation_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        annotation_bundle: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
        *,
        show_regions: bool,
    ) -> None:
        accent = palette["accent"]
        text_color = palette["text"]
        muted = self.renderer._with_alpha(palette["guide"], 0.86)
        for marker in annotation_bundle.get("scene_markers", []) or []:
            if marker.get("type") == "light_source":
                draw.text(
                    (int(marker.get("x", 0)) + 10, int(marker.get("y", 0)) - 12),
                    str(marker.get("display_label", "Light")),
                    fill=accent,
                    font=self.renderer.font,
                )

        for item in annotation_bundle.get("object_annotations", []) or []:
            bbox = item.get("bbox", {})
            anchor = item.get("label_anchor", {})
            center_x = int(bbox.get("x", 0) + bbox.get("width", 0) / 2)
            center_y = int(bbox.get("y", 0) + bbox.get("height", 0) / 2)
            label_x = int(anchor.get("x", center_x))
            label_y = int(anchor.get("y", center_y))
            draw.line([(center_x, center_y), (label_x, label_y)], fill=accent, width=2)
            label = str(item.get("display_label", "Object"))
            role = str(item.get("role", ""))
            if role:
                label = f"{label} [{role}]"
            draw.text((label_x + 4, label_y - 10), label, fill=text_color, font=self.renderer.font)

        if show_regions:
            label_budget: Dict[str, int] = {}
            for region in annotation_bundle.get("region_annotations", []) or []:
                bbox = region.get("bbox", {})
                box = (
                    int(bbox.get("x", 0)),
                    int(bbox.get("y", 0)),
                    int(bbox.get("x", 0) + bbox.get("width", 0)),
                    int(bbox.get("y", 0) + bbox.get("height", 0)),
                )
                self.renderer._draw_region_shape(
                    draw,
                    box,
                    region,
                    outline=self.renderer._with_alpha(accent, 0.68),
                    width=2,
                    dash_length=6,
                )
                object_id = str(region.get("object_id", ""))
                label_budget[object_id] = label_budget.get(object_id, 0) + 1
                if label_budget[object_id] > 2:
                    continue
                anchor = region.get("label_anchor", {})
                label_x = int(anchor.get("x", box[2] + 12))
                label_y = int(anchor.get("y", box[1] + 18))
                draw.line([(box[2], int((box[1] + box[3]) / 2)), (label_x, label_y)], fill=accent, width=1)
                actions = list(region.get("actions") or [])
                action_text = f" [{' / '.join(actions[:2])}]" if actions else ""
                draw.text(
                    (label_x + 4, label_y - 10),
                    f"{region.get('display_label', 'Region')}{action_text}",
                    fill=accent,
                    font=self.renderer.font,
                )

        for layer in annotation_bundle.get("background_annotations", []) or []:
            bbox = layer.get("bbox", {})
            draw.text(
                (int(bbox.get("x", 0)) + 8, int(bbox.get("y", 0)) + 8),
                str(layer.get("display_label", "")),
                fill=muted,
                font=self.renderer.font,
            )

    def _draw_building(self, draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], palette: Dict[str, Tuple[int, int, int]]) -> None:
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        h = y1 - y0
        line = palette["line"]
        accent = palette["accent"]
        draw.rectangle([x0 + w * 0.12, y0 + h * 0.04, x1 - w * 0.08, y1], outline=line, width=3)
        side = [(x1 - w * 0.08, y0 + h * 0.04), (x1, y0 + h * 0.1), (x1, y1 - h * 0.02), (x1 - w * 0.08, y1)]
        draw.line(side + [side[0]], fill=self.renderer._with_alpha(line, 0.72), width=2)
        rows, cols = 4, 3
        for row in range(rows):
            for col in range(cols):
                win_w = w * 0.12
                win_h = h * 0.1
                wx = x0 + w * (0.2 + col * 0.18)
                wy = y0 + h * (0.14 + row * 0.16)
                draw.rectangle([wx, wy, wx + win_w, wy + win_h], outline=accent, width=2)
        self._draw_door(draw, (int(x0 + w * 0.44), int(y0 + h * 0.7), int(x0 + w * 0.62), y1), palette)
        for offset in (0.18, 0.34, 0.5, 0.66):
            y = int(y0 + h * offset)
            draw.line([(x0 + w * 0.14, y), (x1 - w * 0.1, y)], fill=self.renderer._with_alpha(line, 0.22), width=1)

    def _draw_house(self, draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], palette: Dict[str, Tuple[int, int, int]]) -> None:
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        h = y1 - y0
        line = palette["line"]
        roof = [(x0 + w * 0.5, y0), (x0 + w * 0.12, y0 + h * 0.28), (x1 - w * 0.12, y0 + h * 0.28)]
        draw.line(roof + [roof[0]], fill=line, width=3)
        draw.rectangle([x0 + w * 0.16, y0 + h * 0.28, x1 - w * 0.16, y1], outline=line, width=3)
        self._draw_window(draw, (int(x0 + w * 0.26), int(y0 + h * 0.42), int(x0 + w * 0.4), int(y0 + h * 0.56)), palette)
        self._draw_window(draw, (int(x0 + w * 0.6), int(y0 + h * 0.42), int(x0 + w * 0.74), int(y0 + h * 0.56)), palette)
        self._draw_door(draw, (int(x0 + w * 0.42), int(y0 + h * 0.56), int(x0 + w * 0.58), y1), palette)

    def _draw_tree(self, draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], palette: Dict[str, Tuple[int, int, int]]) -> None:
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        h = y1 - y0
        line = palette["line"]
        draw.line([(x0 + w * 0.5, y0 + h * 0.44), (x0 + w * 0.5, y1)], fill=line, width=4)
        draw.line([(x0 + w * 0.5, y0 + h * 0.56), (x0 + w * 0.36, y0 + h * 0.78)], fill=line, width=2)
        draw.line([(x0 + w * 0.5, y0 + h * 0.5), (x0 + w * 0.64, y0 + h * 0.74)], fill=line, width=2)
        crowns = [
            [x0 + w * 0.2, y0 + h * 0.12, x0 + w * 0.54, y0 + h * 0.5],
            [x0 + w * 0.42, y0 + h * 0.02, x0 + w * 0.82, y0 + h * 0.42],
            [x0 + w * 0.58, y0 + h * 0.14, x0 + w * 0.92, y0 + h * 0.48],
        ]
        for crown in crowns:
            draw.ellipse(crown, outline=line, width=3)
        draw.line([(x0 + w * 0.28, y0 + h * 0.32), (x0 + w * 0.74, y0 + h * 0.26)], fill=self.renderer._with_alpha(line, 0.22), width=1)

    def _draw_person(self, draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], palette: Dict[str, Tuple[int, int, int]]) -> None:
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        h = y1 - y0
        line = palette["line"]
        head_box = [x0 + w * 0.34, y0, x0 + w * 0.64, y0 + h * 0.24]
        draw.ellipse(head_box, outline=line, width=3)
        neck = (x0 + w * 0.49, y0 + h * 0.24)
        chest = (x0 + w * 0.49, y0 + h * 0.48)
        hip = (x0 + w * 0.5, y0 + h * 0.62)
        draw.line([neck, chest, hip], fill=line, width=4)
        draw.line([(x0 + w * 0.5, y0 + h * 0.32), (x0 + w * 0.3, y0 + h * 0.48)], fill=line, width=3)
        draw.line([(x0 + w * 0.5, y0 + h * 0.32), (x0 + w * 0.68, y0 + h * 0.46)], fill=line, width=3)
        draw.line([hip, (x0 + w * 0.34, y1)], fill=line, width=4)
        draw.line([hip, (x0 + w * 0.68, y1)], fill=line, width=4)
        draw.line([(x0 + w * 0.5, y0 + h * 0.4), (x0 + w * 0.44, y0 + h * 0.58)], fill=self.renderer._with_alpha(line, 0.3), width=1)
        draw.line([(x0 + w * 0.5, y0 + h * 0.4), (x0 + w * 0.58, y0 + h * 0.58)], fill=self.renderer._with_alpha(line, 0.3), width=1)

    def _draw_car(self, draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], palette: Dict[str, Tuple[int, int, int]]) -> None:
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        h = y1 - y0
        line = palette["line"]
        accent = palette["accent"]
        body = [x0 + w * 0.08, y0 + h * 0.38, x1 - w * 0.08, y0 + h * 0.76]
        draw.rounded_rectangle(body, radius=max(10, int(w * 0.08)), outline=line, width=3)
        roof = [(x0 + w * 0.24, y0 + h * 0.38), (x0 + w * 0.38, y0 + h * 0.16), (x0 + w * 0.7, y0 + h * 0.16), (x0 + w * 0.84, y0 + h * 0.38)]
        draw.line(roof + [roof[0]], fill=line, width=3)
        draw.line([(x0 + w * 0.42, y0 + h * 0.2), (x0 + w * 0.34, y0 + h * 0.38)], fill=accent, width=2)
        draw.line([(x0 + w * 0.56, y0 + h * 0.2), (x0 + w * 0.66, y0 + h * 0.38)], fill=accent, width=2)
        wheel_r = max(6, int(min(w, h) * 0.12))
        self.renderer._draw_circle(draw, (x0 + w * 0.3, y0 + h * 0.78), wheel_r, outline=line, fill=None, width=3)
        self.renderer._draw_circle(draw, (x0 + w * 0.72, y0 + h * 0.78), wheel_r, outline=line, fill=None, width=3)

    def _draw_street_lamp(self, draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], palette: Dict[str, Tuple[int, int, int]]) -> None:
        x0, y0, x1, y1 = bbox
        w = x1 - x0
        h = y1 - y0
        line = palette["line"]
        accent = palette["accent"]
        pole_x = x0 + w * 0.48
        top_y = y0 + h * 0.18
        draw.line([(pole_x, y1), (pole_x, top_y)], fill=line, width=4)
        draw.line([(pole_x, top_y), (x0 + w * 0.86, top_y)], fill=line, width=3)
        lamp_box = [x0 + w * 0.72, y0 + h * 0.16, x0 + w * 0.92, y0 + h * 0.26]
        draw.arc(lamp_box, start=180, end=360, fill=line, width=3)
        draw.line([(x0 + w * 0.82, y0 + h * 0.26), (x0 + w * 0.74, y0 + h * 0.34)], fill=line, width=2)
        for ratio in (0.0, -0.08, 0.08):
            draw.line(
                [(x0 + w * 0.82, y0 + h * 0.28), (x0 + w * (0.62 + ratio), y0 + h * 0.62)],
                fill=self.renderer._with_alpha(accent, 0.5),
                width=1,
            )

    def _draw_window(self, draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], palette: Dict[str, Tuple[int, int, int]]) -> None:
        x0, y0, x1, y1 = bbox
        accent = palette["accent"]
        draw.rectangle([x0, y0, x1, y1], outline=accent, width=2)
        draw.line([((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1)], fill=accent, width=1)

    def _draw_door(self, draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], palette: Dict[str, Tuple[int, int, int]]) -> None:
        x0, y0, x1, y1 = bbox
        accent = palette["accent"]
        draw.rectangle([x0, y0, x1, y1], outline=accent, width=2)
        self.renderer._draw_circle(draw, (x1 - 5, (y0 + y1) / 2), 2, outline=accent, fill=accent, width=1)

    def _draw_road(self, draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], palette: Dict[str, Tuple[int, int, int]]) -> None:
        x0, y0, x1, y1 = bbox
        mid_x = int((x0 + x1) / 2)
        draw.line([(x0, y1), (x0 + 30, y0 + 20)], fill=self.renderer._with_alpha(palette["line"], 0.5), width=2)
        draw.line([(x1, y1), (x1 - 30, y0 + 20)], fill=self.renderer._with_alpha(palette["line"], 0.5), width=2)
        self.renderer._draw_dashed_line(draw, (mid_x, y0 + 20), (mid_x, y1 - 12), palette["accent"], width=2, dash_length=16)

    def _draw_light_beam(
        self,
        draw: ImageDraw.ImageDraw,
        start: Tuple[float, float],
        end: Tuple[float, float],
        palette: Dict[str, Tuple[int, int, int]],
    ) -> None:
        accent = self.renderer._with_alpha(palette["accent"], 0.42)
        self.renderer._draw_dashed_line(draw, start, end, accent, width=1, dash_length=12)

    def _scene_markers(self, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
        width = int(scene.get("canvas_size", {}).get("width", 1024))
        height = int(scene.get("canvas_size", {}).get("height", 768))
        markers: List[Dict[str, Any]] = []
        if str((scene.get("layout_options", {}) or {}).get("scene_type", "scene")) == "scene":
            markers.append({"type": "horizon_line", "x": 0, "y": int(height * 0.56), "width": width, "display_label": "Horizon"})
        for obj in scene.get("object_instances", []) or []:
            asset_key = str(obj.get("asset_key", ""))
            if asset_key in {"sun", "street_lamp", "desk_lamp"}:
                bbox = self.renderer._scene_bbox(obj)
                markers.append(
                    {
                        "type": "light_source",
                        "object_id": str(obj.get("id", "")),
                        "label": str(obj.get("concept", "") or asset_key),
                        "display_label": "Light",
                        "x": int((bbox[0] + bbox[2]) / 2),
                        "y": bbox[1],
                    }
                )
        return markers

    def _label_anchor_for_object(self, obj: Dict[str, Any]) -> Dict[str, int]:
        x0, y0, x1, y1 = self.renderer._scene_bbox(obj)
        role = str(obj.get("role", ""))
        depth = str(obj.get("depth_band", ""))
        if role in {"subject", "focus", "core_subject"}:
            return {"x": x1 + 18, "y": y0 + 26}
        if depth == "background":
            return {"x": x0 + 10, "y": max(24, y0 - 18)}
        return {"x": x1 + 10, "y": y0 + 16}

    def _edit_controls_for_object(self, obj: Dict[str, Any]) -> List[str]:
        controls = ["move", "scale", "delete"]
        if obj.get("editable", True):
            controls.extend(["swap_variant", "restyle"])
        if obj.get("region_masks"):
            controls.append("edit_regions")
        return controls

    def _bbox_payload(self, bbox: Tuple[int, int, int, int]) -> Dict[str, int]:
        x0, y0, x1, y1 = bbox
        return {"x": int(x0), "y": int(y0), "width": int(x1 - x0), "height": int(y1 - y0)}

    def _safe_text(self, value: Any) -> str:
        text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
        return " ".join(text.split())

    def _display_label(self, obj: Dict[str, Any]) -> str:
        asset_key = str(obj.get("asset_key", "")).strip().lower()
        concept = self._safe_text(obj.get("concept", ""))
        if concept and concept.isascii():
            return concept
        return DISPLAY_LABELS.get(asset_key, asset_key or "Object")

    def _region_display_label(self, region: Dict[str, Any]) -> str:
        region_id = str(region.get("id", "")).strip().lower()
        if region_id in REGION_LABELS:
            return REGION_LABELS[region_id]
        label = self._safe_text(region.get("label", ""))
        if label and label.isascii():
            return label
        if region_id and region_id.isascii():
            return region_id.replace("_", " ").title()
        return "Region"

    def _layer_display_label(self, layer: Dict[str, Any]) -> str:
        layer_type = str(layer.get("type", "panel")).strip().lower()
        return {"sky": "Sky", "road": "Road", "ground": "Ground", "water": "Water"}.get(layer_type, layer_type.title())

    def _connector_display_label(self, connector: Dict[str, Any]) -> str:
        connector_type = str(connector.get("type", "relation")).strip().lower()
        return {"beam": "Light direction", "arrow": "Relation", "wire": "Connection"}.get(connector_type, "Relation")

    def _build_condition_maps(self, scene: Dict[str, Any]) -> np.ndarray:
        width = int(scene.get("canvas_size", {}).get("width", 1024))
        height = int(scene.get("canvas_size", {}).get("height", 768))
        channels: List[np.ndarray] = []
        channels.append(self._mask_from_background(scene, width, height))
        channels.append(self._mask_from_objects(scene, width, height, roles={"subject", "focus", "core_subject"}))
        channels.append(self._mask_from_objects(scene, width, height, roles={"support", "detail", ""}))
        channels.append(self._mask_from_connectors(scene, width, height))
        channels.append(self._mask_from_regions(scene, width, height))
        channels.append(self._depth_map(scene, width, height))
        return np.stack(channels, axis=2).astype(np.float32)

    def _mask_from_background(self, scene: Dict[str, Any], width: int, height: int) -> np.ndarray:
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        for layer in scene.get("background_layers", []) or []:
            x0, y0, x1, y1 = self.renderer._scene_bbox(layer)
            draw.rectangle([x0, y0, x1, y1], fill=255)
        return np.asarray(image, dtype=np.float32) / 255.0

    def _mask_from_objects(self, scene: Dict[str, Any], width: int, height: int, roles: set[str]) -> np.ndarray:
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        for obj in scene.get("object_instances", []) or []:
            if roles and str(obj.get("role", "")) not in roles:
                continue
            draw.rectangle(self.renderer._scene_bbox(obj), fill=255)
        return np.asarray(image, dtype=np.float32) / 255.0

    def _mask_from_connectors(self, scene: Dict[str, Any], width: int, height: int) -> np.ndarray:
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        objects_by_id = {item["id"]: item for item in scene.get("object_instances", []) or [] if item.get("id")}
        for connector in scene.get("connectors", []) or []:
            if not connector.get("visible", True):
                continue
            from_obj = objects_by_id.get(str(connector.get("from_id", "")))
            to_obj = objects_by_id.get(str(connector.get("to_id", "")))
            if not from_obj or not to_obj:
                continue
            start, end = self.renderer._connector_points_for_scene(from_obj, to_obj)
            draw.line([start, end], fill=255, width=3)
        return np.asarray(image, dtype=np.float32) / 255.0

    def _mask_from_regions(self, scene: Dict[str, Any], width: int, height: int) -> np.ndarray:
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        for obj in scene.get("object_instances", []) or []:
            for region in obj.get("region_masks", []) or []:
                if not isinstance(region, dict):
                    continue
                box = self.renderer._region_box_for_object(obj, region)
                if str(region.get("shape", "rect")) == "ellipse":
                    draw.ellipse(box, fill=255)
                else:
                    draw.rectangle(box, fill=255)
        return np.asarray(image, dtype=np.float32) / 255.0

    def _depth_map(self, scene: Dict[str, Any], width: int, height: int) -> np.ndarray:
        image = np.zeros((height, width), dtype=np.float32)
        mapping = {"background": 0.25, "midground": 0.58, "foreground": 0.9}
        for obj in scene.get("object_instances", []) or []:
            x0, y0, x1, y1 = self.renderer._scene_bbox(obj)
            image[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = mapping.get(str(obj.get("depth_band", "")), 0.46)
        return image
