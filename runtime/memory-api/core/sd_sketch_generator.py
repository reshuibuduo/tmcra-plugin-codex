from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from loguru import logger
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from .comfyui_client import ComfyUIClient
from .semantic_scene_v2 import summarize_scene_spec


class SDSketchGenerator:
    def __init__(self, output_dir: str = "outputs", sd_api_url: str = "") -> None:
        self.output_dir = output_dir
        self.sd_api_url = str(sd_api_url or "").strip().rstrip("/")
        self.comfy_client = ComfyUIClient(api_url=self.sd_api_url, output_dir=output_dir)
        self.last_conditioning_report: Dict[str, Any] = {}
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return self.sd_api_url.startswith("http")

    def set_sd_api_url(self, api_url: str) -> None:
        self.sd_api_url = str(api_url or "").strip().rstrip("/")
        self.comfy_client.set_api_url(self.sd_api_url)

    def _encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")

    def _control_image_path(self, preview: Dict[str, Any]) -> str:
        sketch_bundle = preview.get("sketch_bundle", {}) if isinstance(preview.get("sketch_bundle"), dict) else {}
        for candidate in (
            sketch_bundle.get("sd_upstream_control"),
            sketch_bundle.get("rich_preview"),
            preview.get("sd_upstream_control_path"),
            preview.get("rich_preview_path"),
            preview.get("render_control_path"),
            preview.get("low_preview_path"),
            preview.get("image_path"),
        ):
            value = str(candidate or "").strip()
            if value and os.path.exists(value):
                return value
        return ""

    def _style_prompt(self, sketch_style: str) -> str:
        style = str(sketch_style or "").strip().lower()
        mapping = {
            "scribble_line": "expressive monochrome concept sketch, loose but readable contour lines",
            "line_art": "clean monochrome line art sketch, readable contour lines, minimal shading",
            "clean_line": "clean monochrome line art sketch, readable contour lines, minimal shading",
            "wireframe": "technical wireframe sketch, structural contour drawing, monochrome",
            "blueprint": "blueprint-style technical sketch, clean construction lines, minimal fill",
            "minimal": "minimal line drawing, sparse contour lines, clean white background",
        }
        return mapping.get(style, "clean monochrome line art sketch, readable contour lines")

    def _build_prompt(
        self,
        scene_spec: Dict[str, Any] | None,
        sketch_options: Dict[str, Any] | None = None,
        title: str | None = None,
    ) -> str:
        sketch_options = sketch_options or {}
        style_prompt = self._style_prompt(str(sketch_options.get("sketch_style", "line_art")))
        scene_summary = summarize_scene_spec(scene_spec or {}) if isinstance(scene_spec, dict) else ""
        style_hint = str(sketch_options.get("style_hint", "") or "").strip()
        prompt_suffix = str(sketch_options.get("prompt_suffix", "") or "").strip()
        parts = [
            style_prompt,
            "preserve composition, object placement, scale, and layering from the input control image",
            "white or light paper background",
            "high readability",
            "no photorealistic shading",
        ]
        if title:
            parts.append(str(title).strip())
        if scene_summary:
            parts.append(scene_summary)
        if style_hint:
            parts.append(style_hint)
        if prompt_suffix:
            parts.append(prompt_suffix)
        return ", ".join(part for part in parts if part)

    def _build_negative_prompt(self, sketch_style: str) -> str:
        negative = [
            "photorealistic",
            "full color rendering",
            "oil painting",
            "watercolor",
            "3d render",
            "ui screenshot",
            "text",
            "labels",
            "arrows",
            "boxes",
            "watermark",
            "blurry",
            "messy composition",
            "heavy shadows",
            "thick paint texture",
        ]
        if str(sketch_style or "").strip().lower() == "blueprint":
            negative.extend(["black paper", "dark background"])
        return ", ".join(negative)

    def _fallback_preview(self, preview: Dict[str, Any], reason: str) -> Dict[str, Any]:
        payload = dict(preview or {})
        sketch_bundle = dict(payload.get("sketch_bundle") or {})
        if payload.get("image_path") and not sketch_bundle.get("native_structural_sketch"):
            sketch_bundle["native_structural_sketch"] = payload.get("image_path")
        sketch_bundle["active_sketch_backend"] = "native"
        payload["sketch_bundle"] = sketch_bundle
        payload["backend"] = payload.get("backend", "native_scene_spec_preview")
        payload["sketch_backend"] = "native"
        payload["note"] = reason
        return payload

    def _conditioning_action_rank(self, action: str) -> int:
        mapping = {
            "hide": 0,
            "replace": 1,
            "inpaint": 2,
            "emphasize": 3,
            "weaken": 4,
            "show": 5,
            "transform": 8,
            "idle": 9,
        }
        return mapping.get(str(action or "idle").strip().lower(), 9)

    def _region_conditioning_prompt(self, base_prompt: str, region: Dict[str, Any]) -> str:
        label = str(region.get("label") or region.get("region_id") or "region").strip()
        action = str(region.get("action") or region.get("edit_state") or "idle").strip().lower()
        render_intent = region.get("render_intent") if isinstance(region.get("render_intent"), dict) else {}
        prompt_hint = str(render_intent.get("prompt") or region.get("prompt") or "").strip()
        if action in {"idle", "transform"} and not prompt_hint:
            return ""
        instruction = ""
        if action == "hide":
            instruction = f"remove the masked {label} and blend the surrounding structure naturally"
        elif action == "replace":
            instruction = f"replace the masked {label} with {prompt_hint or f'a clearer {label} matching the scene'}"
        elif action == "inpaint":
            instruction = f"redraw only the masked {label} region as {prompt_hint or f'a clearer {label}'}"
        elif action == "emphasize":
            instruction = f"make the masked {label} more prominent, clearer, and easier to read"
        elif action == "weaken":
            instruction = f"make the masked {label} subtler, lighter, and less dominant"
        elif action == "show":
            instruction = f"restore a readable {label} in the masked region consistent with the scene"
        elif prompt_hint:
            instruction = f"refine the masked {label} region as {prompt_hint}"
        if not instruction:
            return ""
        return f"{base_prompt}. Apply only inside the masked region: {instruction}. Keep everything outside the mask stable."

    def _patch_conditioning_prompt(self, base_prompt: str, patch: Dict[str, Any]) -> str:
        kind = str(patch.get("kind") or "").strip().lower()
        prompt_hint = str(patch.get("prompt") or "").strip()
        if kind == "erase_region":
            instruction = prompt_hint or "remove the masked content and blend it naturally"
        elif kind == "brush_mask":
            instruction = prompt_hint or "clean and simplify the masked area while keeping the composition stable"
        elif kind == "inpaint_region":
            instruction = prompt_hint or "repaint the masked area so it is clean and readable"
        else:
            instruction = prompt_hint
        if not instruction:
            return ""
        return f"{base_prompt}. Apply only inside the masked region: {instruction}. Keep everything outside the mask stable."

    def _region_mask_canvas(self, region: Dict[str, Any], canvas_size: tuple[int, int]) -> Image.Image:
        canvas = Image.new("L", canvas_size, 0)
        rect = region.get("current_rect") if isinstance(region.get("current_rect"), dict) else {}
        if not rect:
            return canvas
        x = int(round(float(rect.get("x", 0) or 0)))
        y = int(round(float(rect.get("y", 0) or 0)))
        width = max(1, int(round(float(rect.get("width", 1) or 1))))
        height = max(1, int(round(float(rect.get("height", 1) or 1))))
        mask_path = str(region.get("mask_image_path") or "").strip()
        if mask_path and os.path.exists(mask_path):
            region_mask = Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.LANCZOS)
        else:
            region_mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(region_mask)
            shape = str(region.get("shape") or "rect").strip().lower()
            if shape == "ellipse":
                draw.ellipse([0, 0, max(0, width - 1), max(0, height - 1)], fill=255)
            else:
                draw.rounded_rectangle([0, 0, max(0, width - 1), max(0, height - 1)], radius=max(4, int(min(width, height) * 0.08)), fill=255)
        layer = Image.new("L", canvas_size, 0)
        layer.paste(region_mask, (x, y))
        rotation = float(region.get("rotation", 0.0) or 0.0)
        if abs(rotation) > 0.01:
            center = (x + width / 2.0, y + height / 2.0)
            layer = layer.rotate(rotation, resample=Image.Resampling.BICUBIC, center=center)
        return layer.filter(ImageFilter.GaussianBlur(radius=1.6))

    def _patch_mask_canvas(self, patch: Dict[str, Any], canvas_size: tuple[int, int]) -> Image.Image:
        canvas = Image.new("L", canvas_size, 0)
        rect = patch.get("rect") if isinstance(patch.get("rect"), dict) else {}
        if not rect:
            return canvas
        x = int(round(float(rect.get("x", 0) or 0)))
        y = int(round(float(rect.get("y", 0) or 0)))
        width = max(1, int(round(float(rect.get("width", 1) or 1))))
        height = max(1, int(round(float(rect.get("height", 1) or 1))))
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(
            [x, y, x + width, y + height],
            radius=max(6, int(min(width, height) * 0.12)),
            fill=255,
        )
        blur_radius = 8 if str(patch.get("kind") or "").strip().lower() == "brush_mask" else 4
        return canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    def _save_mask_canvas(self, mask: Image.Image, filename_prefix: str) -> str:
        rgba = Image.new("RGBA", mask.size, (255, 255, 255, 0))
        rgba.putalpha(mask)
        output_path = os.path.join(self.output_dir, f"{filename_prefix}_{abs(hash(mask.tobytes()))}.png")
        rgba.save(output_path)
        return output_path

    def _save_luma_canvas(self, image: Image.Image, filename_prefix: str, suffix: str) -> str:
        output_path = os.path.join(self.output_dir, f"{filename_prefix}_{suffix}_{abs(hash(image.tobytes()))}.png")
        image.convert("L").save(output_path)
        return output_path

    def _scene_canvas_size(self, scene_spec: Dict[str, Any] | None, fallback_size: tuple[int, int]) -> tuple[int, int]:
        scene_spec = scene_spec if isinstance(scene_spec, dict) else {}
        canvas = scene_spec.get("canvas_size") if isinstance(scene_spec.get("canvas_size"), dict) else {}
        width = max(1, int(canvas.get("width", fallback_size[0]) or fallback_size[0]))
        height = max(1, int(canvas.get("height", fallback_size[1]) or fallback_size[1]))
        return width, height

    def _scene_bbox(self, item: Dict[str, Any], canvas_size: tuple[int, int]) -> tuple[int, int, int, int]:
        width_limit, height_limit = canvas_size
        x0 = max(0, min(width_limit - 1, int(round(float(item.get("x", 0) or 0)))))
        y0 = max(0, min(height_limit - 1, int(round(float(item.get("y", 0) or 0)))))
        width = max(2, int(round(float(item.get("width", 1) or 1))))
        height = max(2, int(round(float(item.get("height", 1) or 1))))
        x1 = max(x0 + 2, min(width_limit, x0 + width))
        y1 = max(y0 + 2, min(height_limit, y0 + height))
        return x0, y0, x1, y1

    def _scaled_points(self, points: List[List[float]], box: tuple[int, int, int, int]) -> List[Tuple[float, float]]:
        x0, y0, x1, y1 = box
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        return [(x0 + width * float(px), y0 + height * float(py)) for px, py in points if len([px, py]) == 2]

    def _draw_dashed_line(self, draw: ImageDraw.ImageDraw, start: Tuple[float, float], end: Tuple[float, float], *, fill: int, width: int = 1, dash_length: int = 10) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        steps = max(1, int(length / max(2, dash_length)))
        for index in range(steps):
            start_ratio = index / steps
            end_ratio = min(1.0, (index + 0.55) / steps)
            sx = start[0] + dx * start_ratio
            sy = start[1] + dy * start_ratio
            ex = start[0] + dx * end_ratio
            ey = start[1] + dy * end_ratio
            draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)

    def _draw_shape_recipe_control(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        shape_recipe: Dict[str, Any],
        *,
        fill_value: int,
        outline_value: int,
        accent_value: int,
    ) -> bool:
        parts = list((shape_recipe or {}).get("parts") or [])
        if not parts:
            return False
        x0, y0, x1, y1 = box
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        drew = False

        for part in parts:
            kind = str(part.get("kind") or "").strip().lower()
            fill_role = str(part.get("fill_role", "fill") or "fill").strip().lower()
            stroke_role = str(part.get("stroke_role", "line") or "line").strip().lower()
            stroke_value = accent_value if stroke_role == "accent" else outline_value
            stroke_width = max(1, int(round(max(width, height) * float(part.get("stroke_width", 0.02) or 0.02))))
            fill = None if fill_role in {"none", "transparent"} else fill_value

            if kind == "rect":
                rect = [
                    x0 + width * float(part.get("x", 0.0)),
                    y0 + height * float(part.get("y", 0.0)),
                    x0 + width * (float(part.get("x", 0.0)) + float(part.get("w", 0.0))),
                    y0 + height * (float(part.get("y", 0.0)) + float(part.get("h", 0.0))),
                ]
                rx = max(0, int(min(width, height) * float(part.get("rx", 0.0) or 0.0)))
                draw.rounded_rectangle(rect, radius=rx, fill=fill, outline=stroke_value, width=stroke_width)
                drew = True
                continue
            if kind == "ellipse":
                rect = [
                    x0 + width * float(part.get("x", 0.0)),
                    y0 + height * float(part.get("y", 0.0)),
                    x0 + width * (float(part.get("x", 0.0)) + float(part.get("w", 0.0))),
                    y0 + height * (float(part.get("y", 0.0)) + float(part.get("h", 0.0))),
                ]
                draw.ellipse(rect, fill=fill, outline=stroke_value, width=stroke_width)
                drew = True
                continue
            if kind == "line":
                start = (x0 + width * float(part.get("x1", 0.0)), y0 + height * float(part.get("y1", 0.0)))
                end = (x0 + width * float(part.get("x2", 0.0)), y0 + height * float(part.get("y2", 0.0)))
                dash = list(part.get("dash") or [])
                if dash:
                    dash_len = max(6, int(max(width, height) * float(dash[0] or 0.08)))
                    self._draw_dashed_line(draw, start, end, fill=stroke_value, width=stroke_width, dash_length=dash_len)
                else:
                    draw.line([start, end], fill=stroke_value, width=stroke_width)
                drew = True
                continue
            if kind == "polygon":
                points = self._scaled_points(list(part.get("points") or []), box)
                if points:
                    draw.polygon(points, fill=fill, outline=stroke_value)
                    if len(points) >= 2 and stroke_width > 1:
                        draw.line(points + [points[0]], fill=stroke_value, width=stroke_width)
                    drew = True
                continue
            if kind in {"polyline", "path"}:
                points = self._scaled_points(list(part.get("points") or []), box)
                if len(points) >= 2:
                    draw.line(points, fill=stroke_value, width=stroke_width)
                    drew = True
        return drew

    def _draw_structure_asset_prior(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        asset_key: str,
        *,
        fill_value: int,
        outline_value: int,
        accent_value: int,
    ) -> None:
        x0, y0, x1, y1 = box
        width = max(2, x1 - x0)
        height = max(2, y1 - y0)
        key = str(asset_key or "").strip().lower()
        line_w = max(2, int(round(min(width, height) * 0.03)))
        accent_w = max(1, line_w - 1)

        if key in {"house", "home"}:
            body = [x0 + width * 0.16, y0 + height * 0.30, x1 - width * 0.16, y1 - height * 0.02]
            roof = [(x0 + width * 0.5, y0 + height * 0.03), (x0 + width * 0.10, y0 + height * 0.34), (x1 - width * 0.10, y0 + height * 0.34)]
            draw.polygon(roof, fill=fill_value, outline=outline_value)
            draw.line(roof + [roof[0]], fill=outline_value, width=line_w)
            draw.rounded_rectangle(body, radius=max(6, int(min(width, height) * 0.06)), fill=fill_value, outline=outline_value, width=line_w)
            door = [x0 + width * 0.43, y0 + height * 0.58, x0 + width * 0.57, y1 - height * 0.02]
            left_window = [x0 + width * 0.24, y0 + height * 0.46, x0 + width * 0.36, y0 + height * 0.58]
            right_window = [x0 + width * 0.64, y0 + height * 0.46, x0 + width * 0.76, y0 + height * 0.58]
            for rect in (door, left_window, right_window):
                draw.rectangle(rect, outline=accent_value, width=accent_w)
            return

        if key in {"tree", "leaf", "plant", "bush"}:
            trunk = [x0 + width * 0.44, y0 + height * 0.56, x0 + width * 0.56, y1]
            draw.rounded_rectangle(trunk, radius=max(3, int(min(width, height) * 0.04)), fill=max(0, fill_value - 14), outline=outline_value, width=line_w)
            canopy_boxes = [
                [x0 + width * 0.08, y0 + height * 0.18, x0 + width * 0.56, y0 + height * 0.74],
                [x0 + width * 0.26, y0 + height * 0.02, x0 + width * 0.78, y0 + height * 0.66],
                [x0 + width * 0.48, y0 + height * 0.16, x1, y0 + height * 0.80],
            ]
            for canopy in canopy_boxes:
                draw.ellipse(canopy, fill=fill_value, outline=outline_value, width=line_w)
            return

        if key in {"person", "human", "figure", "character"}:
            head = [x0 + width * 0.37, y0 + height * 0.02, x0 + width * 0.63, y0 + height * 0.18]
            torso = [x0 + width * 0.37, y0 + height * 0.20, x0 + width * 0.63, y0 + height * 0.60]
            draw.ellipse(head, fill=fill_value, outline=outline_value, width=line_w)
            draw.rounded_rectangle(torso, radius=max(4, int(min(width, height) * 0.06)), fill=fill_value, outline=outline_value, width=line_w)
            draw.line([(x0 + width * 0.37, y0 + height * 0.30), (x0 + width * 0.22, y0 + height * 0.52)], fill=outline_value, width=line_w)
            draw.line([(x0 + width * 0.63, y0 + height * 0.30), (x0 + width * 0.78, y0 + height * 0.52)], fill=outline_value, width=line_w)
            draw.line([(x0 + width * 0.46, y0 + height * 0.60), (x0 + width * 0.38, y1)], fill=outline_value, width=line_w)
            draw.line([(x0 + width * 0.54, y0 + height * 0.60), (x0 + width * 0.62, y1)], fill=outline_value, width=line_w)
            return

        if key in {"building", "tower"}:
            draw.rounded_rectangle([x0 + width * 0.12, y0 + height * 0.04, x1 - width * 0.12, y1], radius=max(6, int(min(width, height) * 0.04)), fill=fill_value, outline=outline_value, width=line_w)
            for row in range(3):
                for col in range(3):
                    wx = x0 + width * (0.24 + col * 0.18)
                    wy = y0 + height * (0.18 + row * 0.18)
                    ww = width * 0.08
                    wh = height * 0.08
                    draw.rectangle([wx, wy, wx + ww, wy + wh], outline=accent_value, width=accent_w)
            return

        draw.rounded_rectangle(
            [x0 + width * 0.08, y0 + height * 0.06, x1 - width * 0.08, y1 - height * 0.04],
            radius=max(8, int(min(width, height) * 0.12)),
            fill=fill_value,
            outline=outline_value,
            width=line_w,
        )

    def _structure_values(self, obj: Dict[str, Any], *, purpose: str) -> tuple[int, int, int]:
        depth_band = str(obj.get("depth_band") or "").strip().lower()
        role = str(obj.get("role") or "").strip().lower()
        fill = {"background": 226, "midground": 208, "foreground": 190}.get(depth_band, 206)
        outline = {"background": 108, "midground": 88, "foreground": 72}.get(depth_band, 84)
        accent = {"background": 138, "midground": 116, "foreground": 96}.get(depth_band, 112)
        if role in {"subject", "focus", "core_subject", "primary"}:
            fill = max(150, fill - 10)
            outline = max(46, outline - 10)
            accent = max(62, accent - 8)
        if purpose == "final_render":
            fill = max(142, fill - 8)
            outline = max(42, outline - 6)
            accent = max(58, accent - 6)
        return fill, outline, accent

    def _draw_structure_background_layer(self, draw: ImageDraw.ImageDraw, layer: Dict[str, Any], *, canvas_size: tuple[int, int]) -> None:
        x0, y0, x1, y1 = self._scene_bbox(layer, canvas_size)
        layer_type = str(layer.get("type") or "").strip().lower()
        if layer_type == "process_band":
            return
        if layer_type == "sky":
            draw.rectangle([x0, y0, x1, y1], fill=246)
            draw.line([(x0, y1), (x1, y1)], fill=198, width=2)
            return
        if layer_type == "ground":
            draw.rectangle([x0, y0, x1, y1], fill=238)
            draw.line([(x0, y0), (x1, y0)], fill=184, width=2)
            return
        if layer_type == "road":
            draw.rounded_rectangle([x0, y0, x1, y1], radius=max(8, int(min(x1 - x0, y1 - y0) * 0.08)), fill=232, outline=182, width=2)
            center_y = (y0 + y1) / 2.0
            self._draw_dashed_line(draw, (x0 + 22, center_y), (x1 - 22, center_y), fill=170, width=2, dash_length=22)
            return
        if layer_type == "water":
            draw.rounded_rectangle([x0, y0, x1, y1], radius=max(8, int(min(x1 - x0, y1 - y0) * 0.08)), fill=240, outline=188, width=2)
            wave_y = y0 + (y1 - y0) * 0.35
            self._draw_dashed_line(draw, (x0 + 18, wave_y), (x1 - 18, wave_y), fill=180, width=2, dash_length=18)
            return
        draw.rounded_rectangle([x0, y0, x1, y1], radius=max(8, int(min(x1 - x0, y1 - y0) * 0.08)), fill=242, outline=196, width=2)

    def _draw_structure_connectors(self, draw: ImageDraw.ImageDraw, scene_spec: Dict[str, Any], *, canvas_size: tuple[int, int]) -> None:
        scene_type = str(((scene_spec.get("layout_options") or {}).get("scene_type")) or "scene").strip().lower()
        if scene_type == "scene":
            return
        objects_by_id = {
            str(item.get("id")): item
            for item in (scene_spec.get("object_instances") or [])
            if isinstance(item, dict) and item.get("id") and item.get("visible", True) is not False
        }
        for connector in scene_spec.get("connectors", []) or []:
            if not isinstance(connector, dict):
                continue
            from_id = str(connector.get("from_id") or connector.get("source_id") or "")
            to_id = str(connector.get("to_id") or connector.get("target_id") or "")
            if not from_id or not to_id or from_id not in objects_by_id or to_id not in objects_by_id:
                continue
            fx0, fy0, fx1, fy1 = self._scene_bbox(objects_by_id[from_id], canvas_size)
            tx0, ty0, tx1, ty1 = self._scene_bbox(objects_by_id[to_id], canvas_size)
            start = ((fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0)
            end = ((tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0)
            self._draw_dashed_line(draw, start, end, fill=156, width=2, dash_length=14)

    def _draw_structure_region_constraints(
        self,
        draw: ImageDraw.ImageDraw,
        scene_spec: Dict[str, Any] | None,
        *,
        purpose: str,
    ) -> None:
        if purpose != "final_render":
            return
        render_hints = (scene_spec or {}).get("render_hints") if isinstance((scene_spec or {}).get("render_hints"), dict) else {}
        constraints = list(render_hints.get("region_edit_constraints") or [])
        for item in constraints:
            if not isinstance(item, dict):
                continue
            rect = item.get("current_rect") if isinstance(item.get("current_rect"), dict) else item.get("source_rect") if isinstance(item.get("source_rect"), dict) else {}
            if not rect:
                continue
            x = int(round(float(rect.get("x", 0) or 0)))
            y = int(round(float(rect.get("y", 0) or 0)))
            width = max(2, int(round(float(rect.get("width", 1) or 1))))
            height = max(2, int(round(float(rect.get("height", 1) or 1))))
            box = [x, y, x + width, y + height]
            action = str(item.get("action") or "").strip().lower()
            visible = bool(item.get("visible", True))
            if action == "hide" or not visible:
                draw.rounded_rectangle(box, radius=max(4, int(min(width, height) * 0.12)), fill=248, outline=214, width=2)
                continue
            emphasis = 84 if action in {"replace", "show", "inpaint"} else 98 if action == "emphasize" else 116
            draw.rounded_rectangle(box, radius=max(4, int(min(width, height) * 0.12)), fill=None, outline=emphasis, width=3)

    def _explicit_structure_canvas(
        self,
        scene_spec: Dict[str, Any] | None,
        *,
        control_image_path: str,
        filename_prefix: str,
        purpose: str,
    ) -> str:
        scene_spec = scene_spec if isinstance(scene_spec, dict) else {}
        with Image.open(control_image_path) as control_image:
            canvas_size = self._scene_canvas_size(scene_spec, control_image.size)
        image = Image.new("L", canvas_size, 248)
        draw = ImageDraw.Draw(image)
        scene_type = str(((scene_spec.get("layout_options") or {}).get("scene_type")) or "scene").strip().lower()

        for layer in sorted(scene_spec.get("background_layers", []) or [], key=lambda item: item.get("z_index", 0) if isinstance(item, dict) else 0):
            if isinstance(layer, dict):
                self._draw_structure_background_layer(draw, layer, canvas_size=canvas_size)

        objects = [obj for obj in (scene_spec.get("object_instances") or []) if isinstance(obj, dict) and obj.get("visible", True) is not False]
        if scene_type in {"process", "schematic"}:
            objects = [obj for obj in objects if not self._should_skip_direct_object(scene_type, obj)]
        objects.sort(key=lambda item: (float(item.get("depth_z", 0.5) or 0.5), int(item.get("z_index", 0) or 0)))
        prior_overlay_keys = {
            "house",
            "home",
            "tree",
            "leaf",
            "plant",
            "bush",
            "person",
            "human",
            "figure",
            "character",
            "building",
            "tower",
            "sun",
            "cloud",
            "vapor",
            "raindrop",
            "battery",
            "resistor",
            "led",
            "switch",
            "diode",
            "board",
        }
        for obj in objects:
            box = self._scene_bbox(obj, canvas_size)
            fill_value, outline_value, accent_value = self._structure_values(obj, purpose=purpose)
            asset_key = str(obj.get("asset_key") or obj.get("silhouette_key") or obj.get("concept") or "")
            shape_recipe = obj.get("shape_recipe") if isinstance(obj.get("shape_recipe"), dict) else {}
            drew = self._draw_shape_recipe_control(
                draw,
                box,
                shape_recipe,
                fill_value=fill_value,
                outline_value=outline_value,
                accent_value=accent_value,
            )
            should_overlay_prior = asset_key.strip().lower() in prior_overlay_keys
            if not drew or should_overlay_prior:
                if should_overlay_prior:
                    self._draw_direct_scene_object_mass(draw, obj, scene_type=scene_type, canvas_size=canvas_size)
                else:
                    self._draw_structure_asset_prior(
                        draw,
                        box,
                        asset_key,
                        fill_value=fill_value,
                        outline_value=outline_value,
                        accent_value=accent_value,
                    )

        self._draw_structure_connectors(draw, scene_spec, canvas_size=canvas_size)
        self._draw_structure_region_constraints(draw, scene_spec, purpose=purpose)

        image = ImageOps.autocontrast(image)
        if purpose == "sketch_upstream":
            image = image.filter(ImageFilter.GaussianBlur(radius=0.2))
        return self._save_luma_canvas(image, filename_prefix, "structure_control")

    def _edge_control_canvas(self, control_image_path: str, filename_prefix: str) -> str:
        with Image.open(control_image_path) as image:
            gray = ImageOps.autocontrast(image.convert("L"))
            smoothed = gray.filter(ImageFilter.GaussianBlur(radius=0.8))
            edges = smoothed.filter(ImageFilter.FIND_EDGES).point(lambda px: 255 if px > 18 else 0)
            line_seed = gray.point(lambda px: 255 if px < 214 else 0).filter(ImageFilter.MaxFilter(size=3))
            merged = ImageChops.lighter(edges, line_seed).filter(ImageFilter.MaxFilter(size=3))
            merged = merged.filter(ImageFilter.GaussianBlur(radius=0.6)).point(lambda px: 255 if px > 18 else 0)
        return self._save_luma_canvas(merged, filename_prefix, "edge_control")

    def _direct_scene_type(self, scene_spec: Dict[str, Any] | None) -> str:
        scene_spec = scene_spec if isinstance(scene_spec, dict) else {}
        layout_options = scene_spec.get("layout_options") if isinstance(scene_spec.get("layout_options"), dict) else {}
        return str(layout_options.get("scene_type") or layout_options.get("composition_mode") or "scene").strip().lower() or "scene"

    def _direct_scene_asset_allowlist(self, scene_type: str) -> set[str]:
        if scene_type == "process":
            return {"sun", "vapor", "cloud", "raindrop", "leaf", "energy_wave", "airplane", "cell"}
        if scene_type == "schematic":
            return {"battery", "resistor", "led", "switch", "capacitor", "diode", "board"}
        return {"person", "house", "home", "building", "tree", "leaf", "plant", "bush", "road", "car", "street_lamp", "cloud", "sun", "dog", "table", "chair", "desk_lamp"}

    def _should_skip_direct_object(self, scene_type: str, obj: Dict[str, Any]) -> bool:
        asset_key = str(obj.get("asset_key") or obj.get("silhouette_key") or "").strip().lower()
        concept = str(obj.get("concept") or obj.get("label") or "").strip().lower()
        allowlist = self._direct_scene_asset_allowlist(scene_type)
        if asset_key and allowlist and asset_key not in allowlist:
            return True
        noise_terms = ("tri-maze", "scene sketch", "semantic sketch", "readable", "editable", "direct edit", "后续", "可编辑", "直接编辑", "包含", "contains", "结构", "说明", "描述")
        if concept and any(token in concept for token in noise_terms):
            return True
        if scene_type == "schematic" and asset_key == "road":
            return True
        return False

    def _depth_value(self, obj: Dict[str, Any]) -> float:
        depth_z = obj.get("depth_z")
        if depth_z is not None:
            try:
                return max(0.0, min(1.0, float(depth_z)))
            except Exception:
                pass
        return {"background": 0.28, "midground": 0.56, "foreground": 0.86}.get(str(obj.get("depth_band") or "").strip().lower(), 0.48)

    def _depth_control_canvas(
        self,
        scene_spec: Dict[str, Any] | None,
        *,
        canvas_size: tuple[int, int],
        filename_prefix: str,
    ) -> str:
        scene_spec = scene_spec if isinstance(scene_spec, dict) else {}
        scene_type = self._direct_scene_type(scene_spec)
        image = Image.new("L", canvas_size, 40)
        draw = ImageDraw.Draw(image)
        objects = [
            obj
            for obj in (scene_spec.get("object_instances") or [])
            if isinstance(obj, dict) and obj.get("visible", True) is not False and not self._should_skip_direct_object(scene_type, obj)
        ]
        for obj in sorted(objects, key=lambda item: float(item.get("depth_z", 0.5) or 0.5)):
            x0 = int(round(float(obj.get("x", 0) or 0)))
            y0 = int(round(float(obj.get("y", 0) or 0)))
            width = max(1, int(round(float(obj.get("width", 1) or 1))))
            height = max(1, int(round(float(obj.get("height", 1) or 1))))
            pad = max(8, int(min(width, height) * 0.08))
            rect = [
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(canvas_size[0], x0 + width + pad),
                min(canvas_size[1], y0 + height + pad),
            ]
            depth_value = int(round(self._depth_value(obj) * 255))
            draw.rounded_rectangle(rect, radius=max(10, int(min(width, height) * 0.14)), fill=depth_value)
        image = image.filter(ImageFilter.GaussianBlur(radius=10))
        return self._save_luma_canvas(image, filename_prefix, "depth_control")

    def _draw_direct_scene_background(
        self,
        draw: ImageDraw.ImageDraw,
        scene_spec: Dict[str, Any] | None,
        *,
        canvas_size: tuple[int, int],
    ) -> None:
        scene_spec = scene_spec if isinstance(scene_spec, dict) else {}
        scene_type = self._direct_scene_type(scene_spec)
        width, height = canvas_size
        if scene_type == "schematic":
            return
        for layer in sorted(scene_spec.get("background_layers", []) or [], key=lambda item: item.get("z_index", 0) if isinstance(item, dict) else 0):
            if not isinstance(layer, dict):
                continue
            x0, y0, x1, y1 = self._scene_bbox(layer, canvas_size)
            layer_type = str(layer.get("type") or "").strip().lower()
            if scene_type == "process" and layer_type not in {"sky", "ground", "water"}:
                continue
            if layer_type == "sky":
                draw.rectangle([x0, y0, x1, y1], fill=246)
                continue
            if layer_type == "ground":
                draw.rectangle([x0, y0, x1, y1], fill=232)
                draw.line([(0, y0), (width, y0)], fill=220, width=max(2, height // 256))
                continue
            if layer_type == "road":
                if scene_type != "scene":
                    continue
                draw.rounded_rectangle([x0, y0, x1, y1], radius=max(14, int(min(x1 - x0, y1 - y0) * 0.08)), fill=226)
                continue
            if layer_type == "water":
                draw.rounded_rectangle([x0, y0, x1, y1], radius=max(14, int(min(x1 - x0, y1 - y0) * 0.08)), fill=236)
                continue
            if scene_type == "process":
                continue
            draw.rounded_rectangle([x0, y0, x1, y1], radius=max(12, int(min(x1 - x0, y1 - y0) * 0.08)), fill=240)

    def _scene_region_box(
        self,
        obj: Dict[str, Any],
        region: Dict[str, Any],
        canvas_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self._scene_bbox(obj, canvas_size)
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        rx0 = x0 + width * float(region.get("x", 0.0) or 0.0)
        ry0 = y0 + height * float(region.get("y", 0.0) or 0.0)
        rx1 = rx0 + width * float(region.get("width", 0.0) or 0.0)
        ry1 = ry0 + height * float(region.get("height", 0.0) or 0.0)
        return (
            int(round(rx0)),
            int(round(ry0)),
            max(int(round(rx0)) + 2, int(round(rx1))),
            max(int(round(ry0)) + 2, int(round(ry1))),
        )

    def _mass_fill_value(self, obj: Dict[str, Any], *, delta: int = 0) -> int:
        depth_band = str(obj.get("depth_band") or "").strip().lower()
        base = {"background": 206, "midground": 160, "foreground": 116}.get(depth_band, 150)
        role = str(obj.get("role") or "").strip().lower()
        if role in {"subject", "focus", "core_subject", "primary"}:
            base = max(88, base - 12)
        return max(56, min(228, base + delta))

    def _draw_direct_scene_object_mass(
        self,
        draw: ImageDraw.ImageDraw,
        obj: Dict[str, Any],
        *,
        scene_type: str,
        canvas_size: tuple[int, int],
    ) -> None:
        asset_key = str(obj.get("asset_key") or obj.get("concept") or "").strip().lower()
        concept = str(obj.get("concept") or obj.get("label") or "").strip().lower()
        x0, y0, x1, y1 = self._scene_bbox(obj, canvas_size)
        width = max(2, x1 - x0)
        height = max(2, y1 - y0)
        primary_fill = self._mass_fill_value(obj)
        secondary_fill = self._mass_fill_value(obj, delta=18)
        accent_fill = self._mass_fill_value(obj, delta=-12)
        regions = [item for item in (obj.get("region_masks") or []) if isinstance(item, dict)]
        if self._should_skip_direct_object(scene_type, obj):
            return

        if regions:
            for index, region in enumerate(regions):
                rx0, ry0, rx1, ry1 = self._scene_region_box(obj, region, canvas_size)
                fill_value = max(48, min(236, primary_fill + index * 8))
                shape = str(region.get("shape") or "rect").strip().lower()
                if shape == "ellipse":
                    draw.ellipse([rx0, ry0, rx1, ry1], fill=fill_value)
                else:
                    draw.rounded_rectangle(
                        [rx0, ry0, rx1, ry1],
                        radius=max(8, int(min(rx1 - rx0, ry1 - ry0) * 0.16)),
                        fill=fill_value,
                    )
            return

        if asset_key in {"person", "human", "figure", "character"}:
            head = [x0 + width * 0.34, y0 + height * 0.04, x0 + width * 0.66, y0 + height * 0.24]
            torso = [x0 + width * 0.30, y0 + height * 0.22, x0 + width * 0.70, y0 + height * 0.62]
            leg_left = [x0 + width * 0.34, y0 + height * 0.58, x0 + width * 0.48, y1]
            leg_right = [x0 + width * 0.52, y0 + height * 0.58, x0 + width * 0.66, y1]
            draw.ellipse(head, fill=primary_fill)
            draw.rounded_rectangle(torso, radius=max(10, int(min(width, height) * 0.12)), fill=accent_fill)
            draw.rounded_rectangle(leg_left, radius=max(6, int(min(width, height) * 0.08)), fill=secondary_fill)
            draw.rounded_rectangle(leg_right, radius=max(6, int(min(width, height) * 0.08)), fill=secondary_fill)
            return

        if asset_key in {"house", "home"}:
            roof = [(x0 + width * 0.50, y0 + height * 0.02), (x0 + width * 0.08, y0 + height * 0.34), (x1 - width * 0.08, y0 + height * 0.34)]
            body = [x0 + width * 0.14, y0 + height * 0.28, x1 - width * 0.14, y1]
            draw.polygon(roof, fill=accent_fill)
            draw.rounded_rectangle(body, radius=max(10, int(min(width, height) * 0.08)), fill=primary_fill)
            return

        if asset_key in {"tree", "leaf", "plant", "bush"}:
            canopy_a = [x0 + width * 0.06, y0 + height * 0.10, x0 + width * 0.60, y0 + height * 0.72]
            canopy_b = [x0 + width * 0.30, y0 + height * 0.02, x1 - width * 0.02, y0 + height * 0.68]
            trunk = [x0 + width * 0.42, y0 + height * 0.56, x0 + width * 0.58, y1]
            draw.ellipse(canopy_a, fill=primary_fill)
            draw.ellipse(canopy_b, fill=secondary_fill)
            draw.rounded_rectangle(trunk, radius=max(6, int(min(width, height) * 0.06)), fill=accent_fill)
            return

        if asset_key in {"building", "tower"}:
            draw.rounded_rectangle([x0 + width * 0.10, y0 + height * 0.04, x1 - width * 0.10, y1], radius=max(10, int(min(width, height) * 0.06)), fill=primary_fill)
            return

        if asset_key == "sun":
            halo = [x0 + width * 0.08, y0 + height * 0.08, x1 - width * 0.08, y1 - height * 0.08]
            core = [x0 + width * 0.22, y0 + height * 0.22, x1 - width * 0.22, y1 - height * 0.22]
            draw.ellipse(halo, fill=secondary_fill)
            draw.ellipse(core, fill=accent_fill)
            return

        if asset_key == "cloud":
            draw.ellipse([x0 + width * 0.06, y0 + height * 0.28, x0 + width * 0.46, y1 - height * 0.08], fill=secondary_fill)
            draw.ellipse([x0 + width * 0.28, y0 + height * 0.08, x0 + width * 0.72, y1 - height * 0.16], fill=primary_fill)
            draw.ellipse([x0 + width * 0.52, y0 + height * 0.22, x1 - width * 0.04, y1 - height * 0.06], fill=secondary_fill)
            return

        if asset_key == "vapor":
            for index in range(3):
                left = x0 + width * (0.18 + index * 0.18)
                top = y0 + height * (0.16 + (index % 2) * 0.08)
                right = left + width * 0.16
                bottom = y1 - height * 0.08
                draw.ellipse([left, top, right, bottom], fill=primary_fill if index == 1 else secondary_fill)
            return

        if asset_key == "raindrop":
            drop = [
                (x0 + width * 0.50, y0 + height * 0.06),
                (x0 + width * 0.24, y0 + height * 0.42),
                (x0 + width * 0.30, y1 - height * 0.10),
                (x0 + width * 0.70, y1 - height * 0.10),
                (x0 + width * 0.76, y0 + height * 0.42),
            ]
            draw.polygon(drop, fill=primary_fill)
            return

        if asset_key in {"cycle", "flow_node"}:
            outer = [x0 + width * 0.12, y0 + height * 0.16, x1 - width * 0.12, y1 - height * 0.16]
            inner = [x0 + width * 0.30, y0 + height * 0.34, x1 - width * 0.30, y1 - height * 0.34]
            draw.ellipse(outer, fill=secondary_fill)
            draw.ellipse(inner, fill=248)
            return

        if asset_key == "battery":
            body = [x0 + width * 0.14, y0 + height * 0.28, x1 - width * 0.12, y1 - height * 0.18]
            nub = [x1 - width * 0.16, y0 + height * 0.38, x1 - width * 0.04, y0 + height * 0.58]
            draw.rounded_rectangle(body, radius=max(8, int(min(width, height) * 0.12)), fill=primary_fill)
            draw.rounded_rectangle(nub, radius=max(4, int(min(width, height) * 0.06)), fill=accent_fill)
            return

        if asset_key == "resistor":
            left_lead = [x0 + width * 0.04, y0 + height * 0.46, x0 + width * 0.20, y0 + height * 0.54]
            core = [x0 + width * 0.18, y0 + height * 0.30, x1 - width * 0.18, y1 - height * 0.30]
            right_lead = [x1 - width * 0.20, y0 + height * 0.46, x1 - width * 0.04, y0 + height * 0.54]
            draw.rounded_rectangle(left_lead, radius=max(4, int(min(width, height) * 0.05)), fill=secondary_fill)
            draw.rounded_rectangle(core, radius=max(8, int(min(width, height) * 0.10)), fill=primary_fill)
            draw.rounded_rectangle(right_lead, radius=max(4, int(min(width, height) * 0.05)), fill=secondary_fill)
            return

        if asset_key in {"led", "diode"}:
            left_lead = [x0 + width * 0.06, y0 + height * 0.46, x0 + width * 0.24, y0 + height * 0.54]
            bulb = [x0 + width * 0.22, y0 + height * 0.22, x0 + width * 0.72, y1 - height * 0.18]
            base = [x0 + width * 0.56, y0 + height * 0.36, x1 - width * 0.08, y0 + height * 0.62]
            draw.rounded_rectangle(left_lead, radius=max(4, int(min(width, height) * 0.05)), fill=secondary_fill)
            draw.ellipse(bulb, fill=primary_fill)
            draw.rounded_rectangle(base, radius=max(6, int(min(width, height) * 0.08)), fill=accent_fill)
            return

        if asset_key == "switch" or (asset_key in {"module", "board"} and any(token in concept for token in ("开关", "switch", "toggle", "button"))):
            left_contact = [x0 + width * 0.14, y0 + height * 0.42, x0 + width * 0.28, y0 + height * 0.58]
            right_contact = [x1 - width * 0.28, y0 + height * 0.42, x1 - width * 0.14, y0 + height * 0.58]
            draw.ellipse(left_contact, fill=secondary_fill)
            draw.ellipse(right_contact, fill=secondary_fill)
            draw.line(
                [(x0 + width * 0.28, y0 + height * 0.50), (x1 - width * 0.18, y0 + height * 0.26)],
                fill=accent_fill,
                width=max(4, int(min(width, height) * 0.08)),
            )
            draw.rounded_rectangle(
                [x0 + width * 0.08, y0 + height * 0.26, x1 - width * 0.08, y1 - height * 0.20],
                radius=max(8, int(min(width, height) * 0.10)),
                outline=primary_fill,
                width=max(3, int(min(width, height) * 0.05)),
            )
            return

        if asset_key in {"module", "board"}:
            body = [x0 + width * 0.10, y0 + height * 0.16, x1 - width * 0.10, y1 - height * 0.16]
            draw.rounded_rectangle(body, radius=max(8, int(min(width, height) * 0.10)), fill=primary_fill)
            notch_size = max(6, int(min(width, height) * 0.12))
            draw.rounded_rectangle([x0 + width * 0.18, y0 + height * 0.28, x0 + width * 0.34, y0 + height * 0.44], radius=notch_size // 3, fill=secondary_fill)
            draw.rounded_rectangle([x0 + width * 0.58, y0 + height * 0.56, x0 + width * 0.76, y0 + height * 0.72], radius=notch_size // 3, fill=secondary_fill)
            return

        if asset_key == "branch":
            node = [x0 + width * 0.34, y0 + height * 0.30, x0 + width * 0.66, y0 + height * 0.62]
            draw.ellipse(node, fill=primary_fill)
            draw.rounded_rectangle([x0 + width * 0.08, y0 + height * 0.44, x0 + width * 0.34, y0 + height * 0.52], radius=max(4, int(min(width, height) * 0.05)), fill=secondary_fill)
            draw.rounded_rectangle([x0 + width * 0.66, y0 + height * 0.44, x1 - width * 0.08, y0 + height * 0.52], radius=max(4, int(min(width, height) * 0.05)), fill=secondary_fill)
            return

        if asset_key == "road" and scene_type == "schematic":
            center_y = y0 + height * 0.50
            draw.rounded_rectangle([x0 + width * 0.04, center_y - height * 0.08, x1 - width * 0.04, center_y + height * 0.08], radius=max(6, int(min(width, height) * 0.06)), fill=secondary_fill)
            return

        if asset_key == "road" and scene_type == "scene":
            draw.rounded_rectangle([x0 + width * 0.04, y0 + height * 0.28, x1 - width * 0.04, y1 - height * 0.10], radius=max(10, int(min(width, height) * 0.08)), fill=secondary_fill)
            return

        if asset_key == "car":
            body = [x0 + width * 0.12, y0 + height * 0.34, x1 - width * 0.10, y1 - height * 0.10]
            roof = [x0 + width * 0.28, y0 + height * 0.12, x0 + width * 0.72, y0 + height * 0.40]
            wheel_a = [x0 + width * 0.18, y1 - height * 0.24, x0 + width * 0.38, y1 - height * 0.02]
            wheel_b = [x0 + width * 0.62, y1 - height * 0.24, x0 + width * 0.82, y1 - height * 0.02]
            draw.rounded_rectangle(body, radius=max(10, int(min(width, height) * 0.10)), fill=primary_fill)
            draw.rounded_rectangle(roof, radius=max(8, int(min(width, height) * 0.08)), fill=secondary_fill)
            draw.ellipse(wheel_a, fill=accent_fill)
            draw.ellipse(wheel_b, fill=accent_fill)
            return

        draw.rounded_rectangle(
            [x0 + width * 0.06, y0 + height * 0.06, x1 - width * 0.06, y1 - height * 0.04],
            radius=max(12, int(min(width, height) * 0.14)),
            fill=primary_fill,
        )

    def _draw_direct_scene_connectors(
        self,
        draw: ImageDraw.ImageDraw,
        scene_spec: Dict[str, Any] | None,
        *,
        canvas_size: tuple[int, int],
        scene_type: str,
    ) -> None:
        if scene_type == "scene":
            return
        scene_spec = scene_spec if isinstance(scene_spec, dict) else {}
        objects_by_id = {
            str(item.get("id")): item
            for item in (scene_spec.get("object_instances") or [])
            if isinstance(item, dict) and item.get("id") and item.get("visible", True) is not False and not self._should_skip_direct_object(scene_type, item)
        }
        for connector in scene_spec.get("connectors", []) or []:
            if not isinstance(connector, dict):
                continue
            from_id = str(connector.get("from_id") or connector.get("source_id") or "")
            to_id = str(connector.get("to_id") or connector.get("target_id") or "")
            if not from_id or not to_id or from_id not in objects_by_id or to_id not in objects_by_id:
                continue
            fx0, fy0, fx1, fy1 = self._scene_bbox(objects_by_id[from_id], canvas_size)
            tx0, ty0, tx1, ty1 = self._scene_bbox(objects_by_id[to_id], canvas_size)
            start = ((fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0)
            end = ((tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0)
            if scene_type == "schematic":
                mid_x = (start[0] + end[0]) / 2.0
                draw.line([start, (mid_x, start[1]), (mid_x, end[1]), end], fill=170, width=3)
            else:
                self._draw_dashed_line(draw, start, end, fill=176, width=2, dash_length=16)

    def _direct_scene_background_plate(
        self,
        scene_spec: Dict[str, Any] | None,
        *,
        canvas_size: tuple[int, int],
        filename_prefix: str,
    ) -> str:
        image = Image.new("RGB", canvas_size, (248, 246, 241))
        gray = image.convert("L")
        draw = ImageDraw.Draw(gray)
        self._draw_direct_scene_background(draw, scene_spec, canvas_size=canvas_size)
        output_path = os.path.join(self.output_dir, f"{filename_prefix}_direct_base.png")
        gray.convert("RGB").save(output_path)
        return output_path

    def _direct_scene_layout_canvas(
        self,
        scene_spec: Dict[str, Any] | None,
        *,
        canvas_size: tuple[int, int],
        filename_prefix: str,
    ) -> str:
        scene_spec = scene_spec if isinstance(scene_spec, dict) else {}
        scene_type = self._direct_scene_type(scene_spec)
        image = Image.new("L", canvas_size, 248)
        draw = ImageDraw.Draw(image)
        self._draw_direct_scene_background(draw, scene_spec, canvas_size=canvas_size)
        objects = [
            obj
            for obj in (scene_spec.get("object_instances") or [])
            if isinstance(obj, dict) and obj.get("visible", True) is not False and not self._should_skip_direct_object(scene_type, obj)
        ]
        objects.sort(key=lambda item: (float(item.get("depth_z", 0.5) or 0.5), int(item.get("z_index", 0) or 0)))
        for obj in objects:
            self._draw_direct_scene_object_mass(draw, obj, scene_type=scene_type, canvas_size=canvas_size)
        self._draw_direct_scene_connectors(draw, scene_spec, canvas_size=canvas_size, scene_type=scene_type)
        blur_radius = 8.0 if scene_type == "scene" else 5.2 if scene_type == "process" else 4.4
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        image = ImageOps.autocontrast(image, cutoff=1)
        return self._save_luma_canvas(image, filename_prefix, "direct_scene_layout")

    def build_direct_scene_prior(
        self,
        *,
        scene_spec: Dict[str, Any] | None = None,
        filename_prefix: str = "direct_scene",
        fallback_size: tuple[int, int] = (1024, 768),
    ) -> Dict[str, Any]:
        canvas_size = self._scene_canvas_size(scene_spec, fallback_size)
        base_plate_path = self._direct_scene_background_plate(
            scene_spec,
            canvas_size=canvas_size,
            filename_prefix=filename_prefix,
        )
        layout_control_path = self._direct_scene_layout_canvas(
            scene_spec,
            canvas_size=canvas_size,
            filename_prefix=filename_prefix,
        )
        depth_control_path = self._depth_control_canvas(
            scene_spec,
            canvas_size=canvas_size,
            filename_prefix=filename_prefix,
        )
        return {
            "canvas_size": {"width": canvas_size[0], "height": canvas_size[1]},
            "base_plate_path": base_plate_path,
            "layout_control_path": layout_control_path,
            "depth_control_path": depth_control_path,
        }

    def build_direct_scene_controlnet_bundle(
        self,
        *,
        scene_spec: Dict[str, Any] | None = None,
        prior_bundle: Dict[str, Any] | None = None,
        filename_prefix: str = "direct_scene",
    ) -> Dict[str, Any]:
        if not self.comfy_client.is_comfyui_server():
            return {}
        prior_bundle = prior_bundle if isinstance(prior_bundle, dict) else {}
        available = self.comfy_client.get_available_controlnets()
        if not available:
            return {}
        available_markers = [name.casefold() for name in available]
        layout_path = str(prior_bundle.get("layout_control_path") or "").strip()
        depth_path = str(prior_bundle.get("depth_control_path") or "").strip()
        base_plate_path = str(prior_bundle.get("base_plate_path") or layout_path or "").strip()
        inputs: list[Dict[str, Any]] = []

        scene_type = self._direct_scene_type(scene_spec)
        structure_path = ""
        if scene_spec and base_plate_path:
            try:
                structure_path = self._explicit_structure_canvas(
                    scene_spec,
                    control_image_path=base_plate_path,
                    filename_prefix=f"{filename_prefix}_direct",
                    purpose="sketch_candidate",
                )
            except Exception as exc:
                logger.warning(f"failed to build direct scene structure control: {exc}")
                structure_path = ""
        if structure_path:
            inputs.append(
                {
                    "image_path": structure_path,
                    "control_net_name": self.comfy_client.pick_controlnet(["lineart", "sketch", "scribble", "canny"]),
                    "strength": 0.54 if scene_type == "schematic" else 0.48 if scene_type == "process" else 0.40,
                    "start_percent": 0.0,
                    "end_percent": 0.90 if scene_type == "schematic" else 0.86 if scene_type == "process" else 0.78,
                    "kind": "scene_structure",
                }
            )
        if layout_path:
            control_name = self.comfy_client.pick_controlnet(["scribble", "sketch", "lineart", "canny"])
            control_path = layout_path
            if "canny" in str(control_name).casefold():
                control_path = self._edge_control_canvas(layout_path, filename_prefix)
            inputs.append(
                {
                    "image_path": control_path,
                    "control_net_name": control_name,
                    "strength": 0.66 if scene_type in {"process", "schematic"} else 0.58,
                    "start_percent": 0.0,
                    "end_percent": 0.90 if scene_type in {"process", "schematic"} else 0.84,
                    "kind": "scene_layout",
                }
            )
        if depth_path and any("depth" in marker for marker in available_markers):
            inputs.append(
                {
                    "image_path": depth_path,
                    "control_net_name": self.comfy_client.pick_controlnet(["depth"]),
                    "strength": 0.24 if scene_type in {"process", "schematic"} else 0.42,
                    "start_percent": 0.0,
                    "end_percent": 0.82 if scene_type in {"process", "schematic"} else 0.88,
                    "kind": "scene_depth",
                }
            )
        if not inputs or not base_plate_path:
            return {}
        init_image_path = base_plate_path or layout_path
        return {
            "purpose": "direct_scene",
            "comfy_mode": "img2img_controlnet" if scene_type in {"process", "schematic"} else "txt2img_controlnet",
            "init_image_path": init_image_path,
            "inputs": inputs,
            "layout_control_path": layout_path,
            "depth_control_path": depth_path,
            "structure_control_path": structure_path,
        }

    def build_controlnet_bundle(
        self,
        *,
        control_image_path: str,
        scene_spec: Dict[str, Any] | None = None,
        filename_prefix: str = "controlnet",
        purpose: str = "sketch_candidate",
    ) -> Dict[str, Any]:
        if not self.comfy_client.is_comfyui_server():
            return {}
        available = self.comfy_client.get_available_controlnets()
        if not available:
            return {}

        available_markers = [name.casefold() for name in available]
        with Image.open(control_image_path) as image:
            canvas_size = self._scene_canvas_size(scene_spec, image.size)

        structure_path = ""
        if scene_spec:
            try:
                structure_path = self._explicit_structure_canvas(
                    scene_spec,
                    control_image_path=control_image_path,
                    filename_prefix=filename_prefix,
                    purpose=purpose,
                )
            except Exception as exc:
                logger.warning(f"failed to build explicit structure control: {exc}")
                structure_path = ""

        inputs: list[Dict[str, Any]] = []
        init_image_path = control_image_path

        if purpose in {"sketch_upstream", "sketch_candidate"} and structure_path:
            sketch_model = self.comfy_client.pick_controlnet(["sketch", "scribble", "lineart", "canny"])
            inputs.append(
                {
                    "image_path": structure_path,
                    "control_net_name": sketch_model,
                    "strength": 0.86 if purpose == "sketch_upstream" else 0.78,
                    "start_percent": 0.0,
                    "end_percent": 1.0,
                    "kind": "structure_sketch",
                }
            )
            canny_model = self.comfy_client.pick_controlnet(["canny", "lineart", "sketch"])
            if "canny" in str(canny_model).casefold():
                structure_edge_path = self._edge_control_canvas(structure_path, filename_prefix)
                inputs.append(
                    {
                        "image_path": structure_edge_path,
                        "control_net_name": canny_model,
                        "strength": 0.48 if purpose == "sketch_upstream" else 0.42,
                        "start_percent": 0.0,
                        "end_percent": 0.88,
                        "kind": "structure_canny",
                    }
                )
        else:
            if any(marker for marker in available_markers if any(tag in marker for tag in ("canny", "lineart", "sketch", "scribble"))):
                edge_path = self._edge_control_canvas(control_image_path, filename_prefix)
                if purpose == "final_render":
                    strength = 0.9
                elif purpose == "sketch_upstream":
                    strength = 1.0
                else:
                    strength = 0.96
                inputs.append(
                    {
                        "image_path": edge_path,
                        "control_net_name": self.comfy_client.pick_controlnet(
                            ["canny", "lineart", "sketch", "scribble"] if purpose != "final_render" else ["sketch", "lineart", "scribble", "canny"]
                        ),
                        "strength": strength,
                        "start_percent": 0.0,
                        "end_percent": 1.0,
                        "kind": "edge",
                    }
                )

        if purpose == "final_render":
            sketch_model = ""
            try:
                sketch_model = self.comfy_client.pick_controlnet(["sketch", "scribble", "lineart"])
            except Exception:
                sketch_model = ""
            if sketch_model:
                inputs.insert(
                    0,
                    {
                        "image_path": control_image_path,
                        "control_net_name": sketch_model,
                        "strength": 0.88,
                        "start_percent": 0.0,
                        "end_percent": 1.0,
                        "kind": "edited_sketch",
                    },
                )
            if structure_path:
                inputs.append(
                    {
                        "image_path": structure_path,
                        "control_net_name": self.comfy_client.pick_controlnet(["sketch", "scribble", "lineart", "canny"]),
                        "strength": 0.42,
                        "start_percent": 0.0,
                        "end_percent": 0.82,
                        "kind": "scene_structure",
                    }
                )

        if scene_spec and any("depth" in marker for marker in available_markers):
            depth_path = self._depth_control_canvas(scene_spec, canvas_size=canvas_size, filename_prefix=filename_prefix)
            inputs.append(
                {
                    "image_path": depth_path,
                    "control_net_name": self.comfy_client.pick_controlnet(["depth"]),
                    "strength": 0.58 if purpose != "final_render" else 0.46,
                    "start_percent": 0.0,
                    "end_percent": 0.86,
                    "kind": "depth",
                }
            )

        if not inputs:
            return {}
        return {
            "purpose": purpose,
            "comfy_mode": "img2img_controlnet",
            "init_image_path": init_image_path,
            "structure_image_path": structure_path,
            "inputs": inputs,
        }

    def _iter_conditioning_ops(self, conditioning_bundle: Dict[str, Any] | None = None) -> list[Dict[str, Any]]:
        bundle = conditioning_bundle if isinstance(conditioning_bundle, dict) else {}
        ops: list[Dict[str, Any]] = []
        for region in bundle.get("region_layers", []) or []:
            if not isinstance(region, dict):
                continue
            action = str(region.get("action") or region.get("edit_state") or "idle").strip().lower()
            if action in {"idle", "transform"} and not str(((region.get("render_intent") or {}).get("prompt") or "")).strip():
                continue
            if action == "show" and not str(((region.get("render_intent") or {}).get("prompt") or "")).strip():
                continue
            ops.append({"kind": "region", "action": action, "payload": region})
        for patch in bundle.get("patch_layers", []) or []:
            if not isinstance(patch, dict):
                continue
            kind = str(patch.get("kind") or "").strip().lower()
            if kind not in {"erase_region", "brush_mask", "inpaint_region"}:
                continue
            ops.append({"kind": "patch", "action": kind, "payload": patch})
        ops.sort(key=lambda item: (self._conditioning_action_rank(item.get("action", "")), item.get("kind", "")))
        return ops[:6]

    def render_inpaint(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        init_image_path: str,
        mask_image_path: str,
        denoising_strength: float,
        steps: int,
        cfg_scale: float,
        sampler_name: str,
        filename_prefix: str,
    ) -> str:
        with Image.open(init_image_path) as image:
            width, height = image.size

        if self.comfy_client.is_comfyui_server():
            return self.comfy_client.render_inpaint(
                prompt=prompt,
                negative_prompt=negative_prompt,
                init_image_path=init_image_path,
                mask_image_path=mask_image_path,
                steps=steps,
                cfg_scale=cfg_scale,
                denoising_strength=denoising_strength,
                sampler_name=sampler_name,
                filename_prefix=filename_prefix,
            )

        payload = {
            "init_images": [self._encode_image_to_base64(init_image_path)],
            "mask": self._encode_image_to_base64(mask_image_path),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "denoising_strength": float(denoising_strength),
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "width": width,
            "height": height,
            "sampler_name": str(sampler_name or "DPM++ 2M Karras"),
            "resize_mode": 0,
            "mask_blur": 4,
            "inpainting_fill": 1,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 32,
        }
        response = requests.post(
            url=f"{self.sd_api_url}/sdapi/v1/img2img",
            json=payload,
            timeout=240,
        )
        response.raise_for_status()
        result = response.json()
        images = result.get("images") or []
        if not images:
            raise RuntimeError("SD inpaint backend returned no images")
        output_path = os.path.join(self.output_dir, f"{filename_prefix}_{abs(hash(prompt + init_image_path + mask_image_path))}.png")
        with open(output_path, "wb") as handle:
            handle.write(base64.b64decode(images[0]))
        return output_path

    def _apply_conditioning_passes(
        self,
        *,
        base_image_path: str,
        prompt: str,
        negative_prompt: str,
        conditioning_bundle: Dict[str, Any] | None = None,
        steps: int,
        cfg_scale: float,
        denoising_strength: float,
        sampler_name: str,
        filename_prefix: str,
    ) -> str:
        bundle = conditioning_bundle if isinstance(conditioning_bundle, dict) else {}
        ops = self._iter_conditioning_ops(bundle)
        self.last_conditioning_report = {
            "applied": False,
            "base_image_path": base_image_path,
            "final_image_path": base_image_path,
            "operations": [],
        }
        if not ops or not os.path.exists(base_image_path):
            return base_image_path
        canvas = bundle.get("canvas_size") if isinstance(bundle.get("canvas_size"), dict) else {}
        with Image.open(base_image_path) as base_image:
            canvas_size = (
                max(1, int(canvas.get("width", base_image.size[0]) or base_image.size[0])),
                max(1, int(canvas.get("height", base_image.size[1]) or base_image.size[1])),
            )
        current_image_path = base_image_path
        for index, op in enumerate(ops, start=1):
            payload = op.get("payload") if isinstance(op.get("payload"), dict) else {}
            if op.get("kind") == "region":
                mask = self._region_mask_canvas(payload, canvas_size)
                local_prompt = self._region_conditioning_prompt(prompt, payload)
            else:
                mask = self._patch_mask_canvas(payload, canvas_size)
                local_prompt = self._patch_conditioning_prompt(prompt, payload)
            if not local_prompt or mask.getbbox() is None:
                continue
            mask_path = self._save_mask_canvas(mask, f"{filename_prefix}_mask_{index}")
            try:
                next_image_path = self.render_inpaint(
                    prompt=local_prompt,
                    negative_prompt=negative_prompt,
                    init_image_path=current_image_path,
                    mask_image_path=mask_path,
                    denoising_strength=max(0.18, min(0.62, float(denoising_strength) * (0.95 if op.get("kind") == "region" else 0.88))),
                    steps=max(10, min(int(steps), 20)),
                    cfg_scale=float(cfg_scale),
                    sampler_name=sampler_name,
                    filename_prefix=f"{filename_prefix}_cond_{index}",
                )
                current_image_path = next_image_path
                self.last_conditioning_report["operations"].append(
                    {
                        "kind": op.get("kind"),
                        "action": op.get("action"),
                        "status": "applied",
                        "mask_image_path": mask_path,
                        "output_image_path": next_image_path,
                    }
                )
            except Exception as exc:
                logger.warning(f"conditioning inpaint failed: {exc}")
                self.last_conditioning_report["operations"].append(
                    {
                        "kind": op.get("kind"),
                        "action": op.get("action"),
                        "status": "failed",
                        "mask_image_path": mask_path,
                        "error": str(exc),
                    }
                )
        self.last_conditioning_report["applied"] = any(item.get("status") == "applied" for item in self.last_conditioning_report["operations"])
        self.last_conditioning_report["final_image_path"] = current_image_path
        return current_image_path

    def render_img2img(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        control_image_path: str,
        denoising_strength: float,
        steps: int,
        cfg_scale: float,
        sampler_name: str,
        filename_prefix: str,
        conditioning_bundle: Dict[str, Any] | None = None,
        controlnet_bundle: Dict[str, Any] | None = None,
        checkpoint_name: str = "",
        lora_name: str = "",
        lora_strength_model: float = 1.0,
        lora_strength_clip: float = 1.0,
    ) -> str:
        init_image_path = str((controlnet_bundle or {}).get("init_image_path") or control_image_path).strip() or control_image_path
        with Image.open(init_image_path) as image:
            width, height = image.size

        if self.comfy_client.is_comfyui_server():
            controlnet_inputs = list((controlnet_bundle or {}).get("inputs") or [])
            comfy_mode = str((controlnet_bundle or {}).get("comfy_mode") or "img2img_controlnet").strip().lower()
            if controlnet_inputs:
                if comfy_mode == "txt2img_controlnet":
                    base_output_path = self.comfy_client.render_controlnet_txt2img(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        control_inputs=controlnet_inputs,
                        width=width,
                        height=height,
                        steps=max(steps, 28),
                        cfg_scale=cfg_scale,
                        sampler_name=sampler_name,
                        filename_prefix=filename_prefix,
                        checkpoint_name=checkpoint_name,
                        lora_name=lora_name,
                        lora_strength_model=lora_strength_model,
                        lora_strength_clip=lora_strength_clip,
                    )
                else:
                    base_output_path = self.comfy_client.render_controlnet_img2img(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        init_image_path=init_image_path,
                        control_inputs=controlnet_inputs,
                        width=width,
                        height=height,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        denoising_strength=denoising_strength,
                        sampler_name=sampler_name,
                        filename_prefix=filename_prefix,
                        checkpoint_name=checkpoint_name,
                        lora_name=lora_name,
                        lora_strength_model=lora_strength_model,
                        lora_strength_clip=lora_strength_clip,
                    )
            else:
                base_output_path = self.comfy_client.render_img2img(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    control_image_path=init_image_path,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    denoising_strength=denoising_strength,
                    sampler_name=sampler_name,
                    filename_prefix=filename_prefix,
                    checkpoint_name=checkpoint_name,
                    lora_name=lora_name,
                    lora_strength_model=lora_strength_model,
                    lora_strength_clip=lora_strength_clip,
                )
        else:
            payload = {
                "init_images": [self._encode_image_to_base64(init_image_path)],
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "denoising_strength": float(denoising_strength),
                "steps": int(steps),
                "cfg_scale": float(cfg_scale),
                "width": width,
                "height": height,
                "sampler_name": str(sampler_name or "DPM++ 2M Karras"),
                "resize_mode": 0,
            }
            response = requests.post(
                url=f"{self.sd_api_url}/sdapi/v1/img2img",
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            result = response.json()
            images = result.get("images") or []
            if not images:
                raise RuntimeError("SD sketch backend returned no images")
            base_output_path = os.path.join(self.output_dir, f"{filename_prefix}_{abs(hash(prompt + init_image_path))}.png")
            with open(base_output_path, "wb") as handle:
                handle.write(base64.b64decode(images[0]))

        final_output_path = self._apply_conditioning_passes(
            base_image_path=base_output_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            conditioning_bundle=conditioning_bundle,
            steps=steps,
            cfg_scale=cfg_scale,
            denoising_strength=denoising_strength,
            sampler_name=sampler_name,
            filename_prefix=filename_prefix,
        )
        if controlnet_bundle:
            self.last_conditioning_report["controlnet_bundle"] = controlnet_bundle
            self.last_conditioning_report["base_image_path"] = base_output_path
            self.last_conditioning_report["final_image_path"] = final_output_path
        return final_output_path

    def render_txt2img(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        sampler_name: str,
        filename_prefix: str,
        checkpoint_name: str = "",
        lora_name: str = "",
        lora_strength_model: float = 1.0,
        lora_strength_clip: float = 1.0,
    ) -> str:
        if self.comfy_client.is_comfyui_server():
            return self.comfy_client.render_txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler_name=sampler_name,
                filename_prefix=filename_prefix,
                checkpoint_name=checkpoint_name,
                lora_name=lora_name,
                lora_strength_model=lora_strength_model,
                lora_strength_clip=lora_strength_clip,
            )

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "width": int(width),
            "height": int(height),
            "sampler_name": str(sampler_name or "DPM++ 2M Karras"),
        }
        response = requests.post(
            url=f"{self.sd_api_url}/sdapi/v1/txt2img",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
        images = result.get("images") or []
        if not images:
            raise RuntimeError("SD image backend returned no images")
        output_path = os.path.join(self.output_dir, f"{filename_prefix}_{abs(hash(prompt + negative_prompt))}.png")
        with open(output_path, "wb") as handle:
            handle.write(base64.b64decode(images[0]))
        return output_path

    def render_scene_direct(
        self,
        *,
        scene_spec: Dict[str, Any] | None,
        prompt: str,
        negative_prompt: str,
        sketch_options: Dict[str, Any] | None = None,
        filename_prefix: str = "sd_direct_scene",
        prior_bundle: Dict[str, Any] | None = None,
        controlnet_bundle: Dict[str, Any] | None = None,
        checkpoint_name: str = "",
        lora_name: str = "",
        lora_strength_model: float = 1.0,
        lora_strength_clip: float = 1.0,
    ) -> Dict[str, Any]:
        sketch_options = sketch_options or {}
        prior_bundle = prior_bundle if isinstance(prior_bundle, dict) else self.build_direct_scene_prior(
            scene_spec=scene_spec,
            filename_prefix=filename_prefix,
        )
        controlnet_bundle = controlnet_bundle if isinstance(controlnet_bundle, dict) else self.build_direct_scene_controlnet_bundle(
            scene_spec=scene_spec,
            prior_bundle=prior_bundle,
            filename_prefix=filename_prefix,
        )
        canvas = prior_bundle.get("canvas_size") if isinstance(prior_bundle.get("canvas_size"), dict) else {}
        width = max(512, int(canvas.get("width", 1024) or 1024))
        height = max(384, int(canvas.get("height", 768) or 768))
        scene_type = self._direct_scene_type(scene_spec)
        default_denoising = 0.82 if scene_type == "scene" else 0.78 if scene_type == "schematic" else 0.80
        default_steps = 30 if scene_type == "scene" else 36
        default_cfg = 6.2 if scene_type == "scene" else 6.5 if scene_type == "schematic" else 6.4

        if controlnet_bundle.get("inputs"):
            image_path = self.render_img2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image_path=str(controlnet_bundle.get("init_image_path") or prior_bundle.get("base_plate_path") or prior_bundle.get("layout_control_path") or ""),
                denoising_strength=float(sketch_options.get("sd_direct_denoising", default_denoising)),
                steps=int(sketch_options.get("sd_direct_steps", sketch_options.get("sd_sketch_steps", default_steps))),
                cfg_scale=float(sketch_options.get("sd_direct_cfg_scale", sketch_options.get("sd_sketch_cfg_scale", default_cfg))),
                sampler_name=str(sketch_options.get("sd_direct_sampler_name", sketch_options.get("sd_sketch_sampler_name", "DPM++ 2M Karras"))),
                filename_prefix=filename_prefix,
                controlnet_bundle=controlnet_bundle,
                checkpoint_name=checkpoint_name,
                lora_name=lora_name,
                lora_strength_model=lora_strength_model,
                lora_strength_clip=lora_strength_clip,
            )
        else:
            image_path = self.render_txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=int(sketch_options.get("sd_direct_steps", sketch_options.get("sd_sketch_steps", 30))),
                cfg_scale=float(sketch_options.get("sd_direct_cfg_scale", sketch_options.get("sd_sketch_cfg_scale", 6.6))),
                sampler_name=str(sketch_options.get("sd_direct_sampler_name", sketch_options.get("sd_sketch_sampler_name", "DPM++ 2M Karras"))),
                filename_prefix=filename_prefix,
                checkpoint_name=checkpoint_name,
                lora_name=lora_name,
                lora_strength_model=lora_strength_model,
                lora_strength_clip=lora_strength_clip,
            )

        return {
            "image_path": image_path,
            "prior_bundle": prior_bundle,
            "controlnet_bundle": controlnet_bundle,
        }

    def render_from_preview(
        self,
        preview: Dict[str, Any],
        sketch_options: Dict[str, Any] | None = None,
        title: str | None = None,
    ) -> Dict[str, Any]:
        preview = dict(preview or {})
        sketch_options = sketch_options or {}

        if not self.available:
            return self._fallback_preview(preview, "SD sketch backend is not configured; falling back to native sketch.")

        control_image_path = self._control_image_path(preview)
        if not control_image_path:
            return self._fallback_preview(preview, "SD sketch backend has no usable control image; falling back to native sketch.")

        prompt = self._build_prompt(preview.get("scene_spec"), sketch_options, title=title)
        negative_prompt = self._build_negative_prompt(str(sketch_options.get("sketch_style", "line_art")))
        controlnet_bundle = self.build_controlnet_bundle(
            control_image_path=control_image_path,
            scene_spec=preview.get("scene_spec"),
            filename_prefix="sd_sketch_preview",
            purpose="sketch_candidate",
        )

        try:
            output_path = self.render_img2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image_path=control_image_path,
                denoising_strength=float(sketch_options.get("sd_sketch_denoising", 0.42)),
                steps=int(sketch_options.get("sd_sketch_steps", 20)),
                cfg_scale=float(sketch_options.get("sd_sketch_cfg_scale", 6.5)),
                sampler_name=str(sketch_options.get("sd_sketch_sampler_name", "DPM++ 2M Karras")),
                filename_prefix="sd_sketch",
                controlnet_bundle=controlnet_bundle,
            )
        except Exception as exc:
            logger.error(f"SD sketch generation failed: {exc}")
            return self._fallback_preview(preview, f"SD sketch generation failed, falling back to native sketch: {exc}")

        payload = dict(preview)
        sketch_bundle = dict(payload.get("sketch_bundle") or {})
        native_structural = sketch_bundle.get("structural_sketch") or payload.get("image_path")
        if native_structural:
            sketch_bundle["native_structural_sketch"] = native_structural
        sketch_bundle["structural_sketch"] = output_path
        sketch_bundle["sd_sketch"] = output_path
        sketch_bundle["active_sketch_backend"] = "sd"

        payload["image_path"] = output_path
        payload["save_path"] = output_path
        payload["sd_sketch_path"] = output_path
        payload["backend"] = "comfyui_sketch_preview" if self.comfy_client.is_comfyui_server() else "sd_sketch_preview"
        payload["sketch_backend"] = "sd"
        payload["generated_prompt"] = prompt
        payload["negative_prompt"] = negative_prompt
        payload["sketch_bundle"] = sketch_bundle
        logger.info(f"SD sketch generation succeeded: {output_path}")
        return payload
