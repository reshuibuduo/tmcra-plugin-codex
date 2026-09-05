"""
Tri-Maze 原生多模态生成器
完全基于三迷宫架构的推理链路原生生成多模态数据，不依赖任何外部生成 API
实现全新的生成思维：推理路径 → 生成单元 → 组合渲染 → 最终产物
"""
import numpy as np
try:
    import cv2
except Exception:  # pragma: no cover - optional dependency for video export only
    cv2 = None
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
import os
import json
import math
from typing import Dict, List, Any, Tuple, TYPE_CHECKING
from loguru import logger
from .object_sketch_backend import scene_shape_variant_payload, summarize_scene_backend
from .semantic_scene_v2 import (
    build_editor_asset_library,
    compose_semantic_scene_spec,
    get_asset_definition,
    normalize_scene_spec_v2,
)
from .sketch_style_spec import normalize_annotation_level, normalize_view_mode
from .unified_scene_generator import UnifiedSceneGenerator
from .whole_scene_sketch_generator import WholeSceneSketchGenerator

if TYPE_CHECKING:
    from .maze_engine import MazePath, MazeNode


class NativeGenerator:
    """
    Tri-Maze原生生成器
    完全基于推理路径的节点、关系、阻力直接生成多模态数据
    不需要任何外部生成API，原生实现生成逻辑
    """
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.whole_scene_sketch_generator = WholeSceneSketchGenerator(self)
        self.unified_scene_generator = UnifiedSceneGenerator(self)
        
        # 概念原生渲染库
        self.concept_renderers = {
            # 电子元件
            "电阻": self._render_resistor,
            "LED": self._render_led,
            "Arduino": self._render_arduino,
            "电源": self._render_battery,
            "GND": self._render_ground,
            "电容": self._render_capacitor,
            "二极管": self._render_diode,
            "三极管": self._render_transistor,
            
            # 基础形状
            "矩形": self._render_rectangle,
            "圆形": self._render_circle,
            "三角形": self._render_triangle,
            "箭头": self._render_arrow,
            
            # 生物
            "猫": self._render_cat,
            "树": self._render_tree,
            "房子": self._render_house,
        }
        
        # 关系布局规则
        self.relation_layout = {
            "串联": self._layout_horizontal_series,
            "并联": self._layout_horizontal_parallel,
            "连接": self._layout_connect,
            "控制": self._layout_top_to_bottom,
            "包含": self._layout_inside,
            "产生": self._layout_left_to_right,
            "导致": self._layout_left_to_right,
        }
        
        # 颜色映射
        self.color_map = {
            "红色": (255, 0, 0),
            "绿色": (0, 255, 0),
            "蓝色": (0, 0, 255),
            "黄色": (255, 255, 0),
            "黑色": (0, 0, 0),
            "白色": (255, 255, 255),
            "灰色": (128, 128, 128),
            "棕色": (165, 42, 42),
            "橙色": (255, 165, 0),
            "紫色": (128, 0, 128),
        }
        
        # 字体
        font_candidates = [
            "C:\\Windows\\Fonts\\msyh.ttc",
            "C:\\Windows\\Fonts\\simhei.ttf",
            "C:\\Windows\\Fonts\\simsun.ttc",
            "SimHei.ttf",
            "NotoSansCJK-Regular.ttc",
        ]
        self.font = None
        for font_path in font_candidates:
            try:
                self.font = ImageFont.truetype(font_path, 20)
                break
            except Exception:
                continue
        if self.font is None:
            self.font = ImageFont.load_default()
        
        logger.info("✅ Tri-Maze 原生生成器初始化完成，完全不依赖外部 API")
    
    def _draw_circle(self, draw: ImageDraw, center: Tuple[float, float], radius: float, fill=None, outline=None, width: int = 1):
        """兼容 Pillow 版本的圆形绘制"""
        cx, cy = center
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=fill, outline=outline, width=width)
    
    def _draw_arrow_line(self, draw: ImageDraw, start: Tuple[float, float], end: Tuple[float, float], fill, width: int = 2, arrow_size: int = 12):
        """兼容 Pillow 的箭头绘制"""
        draw.line([start, end], fill=fill, width=width)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux = dx / length
        uy = dy / length
        base_x = end[0] - ux * arrow_size
        base_y = end[1] - uy * arrow_size
        perp_x = -uy
        perp_y = ux
        left = (base_x + perp_x * arrow_size * 0.5, base_y + perp_y * arrow_size * 0.5)
        right = (base_x - perp_x * arrow_size * 0.5, base_y - perp_y * arrow_size * 0.5)
        draw.polygon([end, left, right], fill=fill)
    
    def _draw_dashed_line(self, draw: ImageDraw, start: Tuple[float, float], end: Tuple[float, float], fill, width: int = 1, dash_length: int = 6):
        """兼容 Pillow 的虚线绘制"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
        dash_count = max(1, int(length / dash_length))
        for index in range(dash_count):
            start_ratio = index / dash_count
            end_ratio = min((index + 0.5) / dash_count, 1.0)
            sx = start[0] + dx * start_ratio
            sy = start[1] + dy * start_ratio
            ex = start[0] + dx * end_ratio
            ey = start[1] + dy * end_ratio
            draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
    
    def _draw_dashed_rectangle(self, draw: ImageDraw, box: Tuple[float, float, float, float], outline, width: int = 1, dash_length: int = 6):
        """兼容 Pillow 的虚线矩形绘制"""
        x0, y0, x1, y1 = box
        edges = [
            ((x0, y0), (x1, y0)),
            ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)),
            ((x0, y1), (x0, y0)),
        ]
        for start, end in edges:
            self._draw_dashed_line(draw, start, end, outline, width=width, dash_length=dash_length)
    
    def _build_preview_palette(self, sketch_style: str) -> Dict[str, Tuple[int, int, int]]:
        sketch_style = {
            "line_art": "scribble_line",
            "minimal": "clean_line",
            "wireframe": "wireframe",
            "blueprint": "blueprint",
        }.get(sketch_style, sketch_style)
        palettes = {
            "scribble_line": {
                "background": (250, 247, 240),
                "grid": (226, 220, 208),
                "line": (46, 50, 56),
                "accent": (73, 110, 168),
                "text": (34, 37, 41),
                "node_fill": (255, 255, 255),
                "guide": (170, 170, 170),
                "region_fill": (236, 232, 222),
                "region_alt": (224, 232, 240),
            },
            "clean_line": {
                "background": (255, 255, 255),
                "grid": (236, 238, 242),
                "line": (48, 54, 61),
                "accent": (53, 104, 214),
                "text": (17, 24, 39),
                "node_fill": (251, 252, 254),
                "guide": (188, 195, 207),
                "region_fill": (244, 246, 249),
                "region_alt": (233, 239, 248),
            },
            "blueprint": {
                "background": (8, 24, 48),
                "grid": (32, 72, 110),
                "line": (132, 220, 255),
                "accent": (121, 255, 198),
                "text": (224, 245, 255),
                "node_fill": (17, 58, 90),
                "guide": (74, 126, 160),
                "region_fill": (14, 48, 76),
                "region_alt": (20, 66, 92),
            },
            "wireframe": {
                "background": (248, 250, 252),
                "grid": (226, 232, 240),
                "line": (51, 65, 85),
                "accent": (14, 165, 233),
                "text": (15, 23, 42),
                "node_fill": (241, 245, 249),
                "guide": (148, 163, 184),
                "region_fill": (241, 245, 249),
                "region_alt": (226, 232, 240),
            },
        }
        return palettes.get(sketch_style, palettes["scribble_line"])

    def _build_low_preview_palette(self, sketch_style: str) -> Dict[str, Tuple[int, int, int]]:
        sketch_style = {
            "line_art": "scribble_line",
            "minimal": "clean_line",
            "wireframe": "wireframe",
            "blueprint": "blueprint",
        }.get(sketch_style, sketch_style)
        palettes = {
            "scribble_line": {
                "background": (245, 241, 232),
                "grid": (229, 222, 210),
                "line": (84, 88, 94),
                "accent": (218, 154, 84),
                "text": (43, 46, 51),
                "node_fill": (255, 252, 246),
                "guide": (188, 180, 166),
                "region_fill": (214, 227, 244),
                "region_alt": (208, 228, 203),
            },
            "clean_line": {
                "background": (252, 253, 255),
                "grid": (235, 239, 245),
                "line": (82, 92, 105),
                "accent": (238, 132, 60),
                "text": (31, 41, 55),
                "node_fill": (255, 255, 255),
                "guide": (190, 199, 211),
                "region_fill": (228, 236, 247),
                "region_alt": (226, 242, 232),
            },
            "blueprint": {
                "background": (14, 32, 60),
                "grid": (32, 72, 110),
                "line": (167, 228, 255),
                "accent": (255, 196, 105),
                "text": (232, 246, 255),
                "node_fill": (47, 88, 122),
                "guide": (87, 126, 154),
                "region_fill": (24, 62, 92),
                "region_alt": (40, 86, 112),
            },
            "wireframe": {
                "background": (243, 246, 251),
                "grid": (227, 234, 241),
                "line": (75, 85, 99),
                "accent": (14, 165, 233),
                "text": (15, 23, 42),
                "node_fill": (255, 255, 255),
                "guide": (148, 163, 184),
                "region_fill": (225, 236, 249),
                "region_alt": (230, 243, 233),
            },
        }
        return palettes.get(sketch_style, palettes["scribble_line"])

    def _build_rich_preview_palette(self, sketch_style: str) -> Dict[str, Tuple[int, int, int]]:
        base = dict(self._build_preview_palette(sketch_style))
        if sketch_style == "clean_line":
            base.update(
                {
                    "background": (252, 252, 249),
                    "accent": (120, 124, 130),
                    "guide": (198, 202, 206),
                    "region_fill": (246, 244, 238),
                    "region_alt": (236, 234, 229),
                }
            )
        else:
            base.update(
                {
                    "accent": (128, 118, 102),
                    "guide": (186, 178, 168),
                }
            )
        return base
    
    def _draw_preview_grid(self, draw: ImageDraw, canvas_size: Tuple[int, int], palette: Dict[str, Tuple[int, int, int]], step: int = 40):
        width, height = canvas_size
        for x in range(0, width + 1, step):
            draw.line([(x, 0), (x, height)], fill=palette["grid"], width=1)
        for y in range(0, height + 1, step):
            draw.line([(0, y), (width, y)], fill=palette["grid"], width=1)

    def _with_alpha(self, color: Tuple[int, int, int], opacity: float = 1.0) -> Tuple[int, int, int, int]:
        alpha = max(0, min(255, int(round(255 * opacity))))
        return color[0], color[1], color[2], alpha

    def _recipe_fill(self, palette: Dict[str, Tuple[int, int, int]], fill_role: str, filled: bool = False) -> Tuple[int, int, int, int] | None:
        role = str(fill_role or "fill")
        if role in {"none", "transparent"}:
            return None
        base = palette["node_fill"] if role == "fill" else palette["region_fill"]
        if role == "region_alt":
            base = palette["region_alt"]
        elif role == "accent_fill":
            base = palette["accent"]
        elif role == "background":
            base = palette["background"]
        opacity = 0.96 if filled else 0.18
        if role in {"region_fill", "region_alt"}:
            opacity = 0.36 if filled else 0.16
        if role == "accent_fill":
            opacity = 0.3 if not filled else 0.8
        return self._with_alpha(base, opacity)

    def _recipe_stroke(self, palette: Dict[str, Tuple[int, int, int]], stroke_role: str, opacity: float = 1.0) -> Tuple[int, int, int, int] | None:
        role = str(stroke_role or "line")
        if role in {"none", "transparent"}:
            return None
        color = palette["line"] if role == "line" else palette["accent"] if role == "accent" else palette["guide"]
        return self._with_alpha(color, opacity)

    def _scaled_points(self, points: List[List[float]], box: Tuple[int, int, int, int]) -> List[Tuple[float, float]]:
        x0, y0, x1, y1 = box
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        return [(x0 + width * float(px), y0 + height * float(py)) for px, py in points]

    def _draw_shape_recipe(
        self,
        draw: ImageDraw,
        box: Tuple[int, int, int, int],
        shape_recipe: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
        *,
        filled: bool = False,
        style_variant: str = "",
    ) -> bool:
        parts = list((shape_recipe or {}).get("parts") or [])
        if not parts:
            return False
        x0, y0, x1, y1 = box
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        rough = "scribble" in str(style_variant or "")

        for part in parts:
            kind = str(part.get("kind") or "").lower()
            opacity = float(part.get("opacity", 1.0) or 1.0)
            fill = self._recipe_fill(palette, str(part.get("fill_role", "fill")), filled=filled)
            outline = self._recipe_stroke(palette, str(part.get("stroke_role", "line")), opacity=opacity)
            stroke_width = max(1, int(round(max(width, height) * float(part.get("stroke_width", 0.02) or 0.02))))
            dash = part.get("dash") or []

            if kind == "rect":
                rx = float(part.get("rx", 0.0) or 0.0)
                rect = [
                    x0 + width * float(part.get("x", 0.0)),
                    y0 + height * float(part.get("y", 0.0)),
                    x0 + width * (float(part.get("x", 0.0)) + float(part.get("w", 0.0))),
                    y0 + height * (float(part.get("y", 0.0)) + float(part.get("h", 0.0))),
                ]
                draw.rounded_rectangle(rect, radius=max(0, int(min(width, height) * rx)), fill=fill, outline=outline, width=stroke_width)
                if rough and outline:
                    offset = max(1, stroke_width // 3)
                    shifted = [rect[0] + offset, rect[1] + offset, rect[2] + offset, rect[3] + offset]
                    draw.rounded_rectangle(shifted, radius=max(0, int(min(width, height) * rx)), fill=None, outline=outline, width=max(1, stroke_width - 1))
                continue

            if kind == "ellipse":
                rect = [
                    x0 + width * float(part.get("x", 0.0)),
                    y0 + height * float(part.get("y", 0.0)),
                    x0 + width * (float(part.get("x", 0.0)) + float(part.get("w", 0.0))),
                    y0 + height * (float(part.get("y", 0.0)) + float(part.get("h", 0.0))),
                ]
                draw.ellipse(rect, fill=fill, outline=outline, width=stroke_width)
                if rough and outline:
                    offset = max(1, stroke_width // 3)
                    shifted = [rect[0] + offset, rect[1] + offset, rect[2] + offset, rect[3] + offset]
                    draw.ellipse(shifted, fill=None, outline=outline, width=max(1, stroke_width - 1))
                continue

            if kind == "line":
                start = (
                    x0 + width * float(part.get("x1", 0.0)),
                    y0 + height * float(part.get("y1", 0.0)),
                )
                end = (
                    x0 + width * float(part.get("x2", 0.0)),
                    y0 + height * float(part.get("y2", 0.0)),
                )
                if dash:
                    dash_len = max(6, int(max(width, height) * float(dash[0])))
                    self._draw_dashed_line(draw, start, end, outline or palette["line"], width=stroke_width, dash_length=dash_len)
                else:
                    draw.line([start, end], fill=outline or palette["line"], width=stroke_width)
                    if rough and outline:
                        draw.line([(start[0] + 1, start[1] + 1), (end[0] + 1, end[1] + 1)], fill=outline, width=max(1, stroke_width - 1))
                continue

            if kind == "polygon":
                points = self._scaled_points(part.get("points") or [], box)
                if points:
                    draw.polygon(points, fill=fill, outline=outline)
                    if outline and stroke_width > 1:
                        draw.line(points + [points[0]], fill=outline, width=stroke_width)
                        if rough:
                            shifted = [(px + 1, py + 1) for px, py in points]
                            draw.line(shifted + [shifted[0]], fill=outline, width=max(1, stroke_width - 1))
                continue

            if kind == "polyline":
                points = self._scaled_points(part.get("points") or [], box)
                if points:
                    draw.line(points, fill=outline or palette["line"], width=stroke_width)
                    if rough and outline:
                        shifted = [(px + 1, py + 1) for px, py in points]
                        draw.line(shifted, fill=outline, width=max(1, stroke_width - 1))
                continue

        return True

    def _draw_stroke_payload(
        self,
        draw: ImageDraw,
        box: Tuple[int, int, int, int],
        stroke_payload: List[List[List[float]]] | None,
        palette: Dict[str, Tuple[int, int, int]],
        *,
        style_variant: str = "",
        stroke_render_profile: Dict[str, Any] | None = None,
    ) -> bool:
        strokes = [stroke for stroke in list(stroke_payload or []) if len(stroke) >= 2]
        if not strokes:
            return False
        x0, y0, x1, y1 = box
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        rough = "scribble" in str(style_variant or "")
        profile = stroke_render_profile if isinstance(stroke_render_profile, dict) else {}
        min_width = float(profile.get("line_width_min", profile.get("min_width", 0.014)) or 0.014)
        max_width = float(profile.get("line_width_max", profile.get("max_width", 0.034)) or 0.034)
        opacity = float(profile.get("opacity", 0.96) or 0.96)
        accent_ratio = float(profile.get("accent_ratio", 0.2) or 0.2)
        for index, stroke in enumerate(strokes):
            points = [(x0 + width * float(px), y0 + height * float(py)) for px, py in stroke]
            if len(points) < 2:
                continue
            ratio = index / max(1, len(strokes) - 1) if len(strokes) > 1 else 0.0
            line_width = max(1, int(round(max(width, height) * (min_width + (max_width - min_width) * ratio))))
            line_role = "accent" if ratio <= accent_ratio else "line"
            color = self._recipe_stroke(palette, line_role, opacity=opacity) or self._with_alpha(palette["line"], opacity)
            draw.line(points, fill=color, width=line_width)
            if rough:
                shifted = [(px + 1.0, py + 1.0) for px, py in points]
                draw.line(shifted, fill=self._recipe_stroke(palette, line_role, opacity=max(0.18, opacity * 0.55)) or color, width=max(1, line_width - 1))
        return True

    def _scene_bbox(self, item: Dict[str, Any]) -> Tuple[int, int, int, int]:
        x0 = int(item.get("x", 0))
        y0 = int(item.get("y", 0))
        x1 = x0 + int(item.get("width", 0))
        y1 = y0 + int(item.get("height", 0))
        return x0, y0, x1, y1

    def _scene_center(self, item: Dict[str, Any]) -> Tuple[float, float]:
        x0, y0, x1, y1 = self._scene_bbox(item)
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def _connector_points_for_scene(self, from_item: Dict[str, Any], to_item: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        from_center = self._scene_center(from_item)
        to_center = self._scene_center(to_item)
        if from_center[0] <= to_center[0]:
            start = (from_item["x"] + from_item["width"], from_center[1])
            end = (to_item["x"], to_center[1])
        else:
            start = (from_item["x"], from_center[1])
            end = (to_item["x"] + to_item["width"], to_center[1])
        return start, end

    def _draw_scene_background_layer(self, draw: ImageDraw, layer: Dict[str, Any], palette: Dict[str, Tuple[int, int, int]], filled: bool = False):
        x0, y0, x1, y1 = self._scene_bbox(layer)
        layer_type = str(layer.get("type", "panel"))
        fill = palette["region_alt"] if layer_type in {"ground", "road", "board", "water"} else palette["region_fill"]
        outline = palette["guide"] if filled else palette["line"]
        fill_alpha = self._with_alpha(fill, 0.45 if filled else 0.16)
        outline_alpha = self._with_alpha(outline, 0.5 if filled else 0.3)
        if layer_type == "road":
            draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=fill_alpha, outline=outline_alpha, width=2)
            lane_y = int((y0 + y1) / 2)
            self._draw_dashed_line(draw, (x0 + 24, lane_y), (x1 - 24, lane_y), palette["accent"], width=2, dash_length=18)
            return
        if layer_type == "sky":
            draw.rectangle([x0, y0, x1, y1], fill=self._with_alpha(fill, 0.18 if not filled else 0.42))
            draw.line([(x0, y1), (x1, y1)], fill=outline_alpha, width=2)
            return
        if layer_type == "process_band":
            draw.rounded_rectangle(
                [x0 + 8, y0 + 10, x1 - 8, y1 - 10],
                radius=28,
                fill=self._with_alpha(fill, 0.06 if not filled else 0.14),
                outline=self._with_alpha(outline, 0.14 if not filled else 0.2),
                width=1,
            )
            return
        draw.rounded_rectangle([x0, y0, x1, y1], radius=20, fill=fill_alpha, outline=outline_alpha, width=2)

    def _draw_asset_symbol(
        self,
        draw: ImageDraw,
        box: Tuple[int, int, int, int],
        silhouette_key: str,
        palette: Dict[str, Tuple[int, int, int]],
        filled: bool = False,
    ):
        x0, y0, x1, y1 = box
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        line = palette["line"]
        accent = palette["accent"]
        fill = palette["node_fill"] if filled else palette["background"]
        if silhouette_key in {"generic_panel", "generic_object"}:
            draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=fill, outline=line, width=3)
            if silhouette_key == "generic_panel":
                draw.line([(x0 + 18, y0 + 24), (x1 - 18, y0 + 24)], fill=accent, width=2)
            return
        if silhouette_key == "generic_circle":
            draw.ellipse([x0, y0, x1, y1], fill=fill, outline=line, width=3)
            return
        if silhouette_key == "building":
            draw.rectangle([x0 + width * 0.1, y0 + height * 0.08, x1 - width * 0.1, y1], fill=fill, outline=line, width=3)
            for row in range(3):
                for col in range(3):
                    wx = x0 + width * (0.18 + col * 0.22)
                    wy = y0 + height * (0.16 + row * 0.22)
                    ww = width * 0.12
                    wh = height * 0.12
                    draw.rectangle([wx, wy, wx + ww, wy + wh], outline=accent, width=2)
            return
        if silhouette_key == "house":
            draw.polygon([(x0 + width * 0.5, y0), (x0, y0 + height * 0.34), (x1, y0 + height * 0.34)], fill=fill, outline=line, width=3)
            draw.rectangle([x0 + width * 0.14, y0 + height * 0.34, x1 - width * 0.14, y1], fill=fill, outline=line, width=3)
            draw.rectangle([x0 + width * 0.42, y0 + height * 0.56, x0 + width * 0.58, y1], outline=accent, width=2)
            return
        if silhouette_key == "window":
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=line, width=3)
            draw.line([(x0 + width / 2, y0), (x0 + width / 2, y1)], fill=accent, width=2)
            draw.line([(x0, y0 + height / 2), (x1, y0 + height / 2)], fill=accent, width=2)
            return
        if silhouette_key == "door":
            draw.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=fill, outline=line, width=3)
            knob = (x1 - width * 0.18, y0 + height * 0.56)
            self._draw_circle(draw, knob, max(3, width * 0.04), fill=accent, outline=accent, width=1)
            return
        if silhouette_key == "tree":
            trunk_w = width * 0.18
            draw.rectangle([x0 + width * 0.41, y0 + height * 0.56, x0 + width * 0.41 + trunk_w, y1], fill=accent if filled else fill, outline=line, width=3)
            draw.ellipse([x0 + width * 0.12, y0, x0 + width * 0.88, y0 + height * 0.66], fill=fill, outline=line, width=3)
            draw.ellipse([x0, y0 + height * 0.12, x0 + width * 0.56, y0 + height * 0.66], fill=fill, outline=line, width=2)
            draw.ellipse([x0 + width * 0.44, y0 + height * 0.12, x1, y0 + height * 0.66], fill=fill, outline=line, width=2)
            return
        if silhouette_key == "cloud":
            draw.ellipse([x0 + width * 0.08, y0 + height * 0.28, x0 + width * 0.46, y1], fill=fill, outline=line, width=3)
            draw.ellipse([x0 + width * 0.26, y0, x0 + width * 0.7, y0 + height * 0.9], fill=fill, outline=line, width=3)
            draw.ellipse([x0 + width * 0.54, y0 + height * 0.22, x1, y1], fill=fill, outline=line, width=3)
            return
        if silhouette_key == "sun":
            cx = x0 + width / 2
            cy = y0 + height / 2
            radius = min(width, height) * 0.28
            self._draw_circle(draw, (cx, cy), radius, fill=fill, outline=line, width=3)
            for ray in range(8):
                angle = math.radians(ray * 45)
                sx = cx + math.cos(angle) * radius * 1.3
                sy = cy + math.sin(angle) * radius * 1.3
                ex = cx + math.cos(angle) * radius * 1.85
                ey = cy + math.sin(angle) * radius * 1.85
                draw.line([(sx, sy), (ex, ey)], fill=accent, width=2)
            return
        if silhouette_key == "car":
            draw.rounded_rectangle([x0 + width * 0.08, y0 + height * 0.34, x1 - width * 0.08, y1 - height * 0.12], radius=14, fill=fill, outline=line, width=3)
            draw.polygon(
                [(x0 + width * 0.24, y0 + height * 0.34), (x0 + width * 0.38, y0 + height * 0.1), (x0 + width * 0.72, y0 + height * 0.1), (x0 + width * 0.84, y0 + height * 0.34)],
                fill=fill,
                outline=line,
                width=3,
            )
            self._draw_circle(draw, (x0 + width * 0.3, y1 - height * 0.1), min(width, height) * 0.11, fill=palette["background"], outline=line, width=3)
            self._draw_circle(draw, (x0 + width * 0.72, y1 - height * 0.1), min(width, height) * 0.11, fill=palette["background"], outline=line, width=3)
            return
        if silhouette_key == "road":
            draw.rounded_rectangle([x0, y0 + height * 0.2, x1, y1], radius=18, fill=fill, outline=line, width=3)
            self._draw_dashed_line(draw, (x0 + width * 0.08, y0 + height * 0.62), (x1 - width * 0.08, y0 + height * 0.62), accent, width=2, dash_length=18)
            return
        if silhouette_key == "person":
            cx = x0 + width / 2
            head_r = min(width, height) * 0.16
            self._draw_circle(draw, (cx, y0 + head_r * 1.25), head_r, fill=fill, outline=line, width=3)
            draw.line([(cx, y0 + head_r * 2.5), (cx, y0 + height * 0.72)], fill=line, width=3)
            draw.line([(cx, y0 + height * 0.36), (x0 + width * 0.22, y0 + height * 0.5)], fill=line, width=3)
            draw.line([(cx, y0 + height * 0.36), (x1 - width * 0.22, y0 + height * 0.5)], fill=line, width=3)
            draw.line([(cx, y0 + height * 0.72), (x0 + width * 0.24, y1)], fill=line, width=3)
            draw.line([(cx, y0 + height * 0.72), (x1 - width * 0.24, y1)], fill=line, width=3)
            return
        if silhouette_key == "street_lamp":
            draw.line([(x0 + width * 0.5, y1), (x0 + width * 0.5, y0 + height * 0.18)], fill=line, width=4)
            draw.line([(x0 + width * 0.5, y0 + height * 0.2), (x1, y0 + height * 0.2)], fill=line, width=3)
            draw.arc([x1 - width * 0.36, y0 + height * 0.12, x1, y0 + height * 0.44], 200, 360, fill=line, width=3)
            self._draw_circle(draw, (x1 - width * 0.06, y0 + height * 0.4), max(4, width * 0.05), fill=accent, outline=accent, width=2)
            return
        if silhouette_key == "table":
            draw.rectangle([x0 + width * 0.08, y0 + height * 0.24, x1 - width * 0.08, y0 + height * 0.38], fill=fill, outline=line, width=3)
            for ratio in (0.18, 0.82):
                draw.line([(x0 + width * ratio, y0 + height * 0.38), (x0 + width * ratio, y1)], fill=line, width=3)
            return
        if silhouette_key == "chair":
            draw.line([(x0 + width * 0.24, y1), (x0 + width * 0.24, y0 + height * 0.44)], fill=line, width=3)
            draw.line([(x0 + width * 0.76, y1), (x0 + width * 0.76, y0 + height * 0.44)], fill=line, width=3)
            draw.line([(x0 + width * 0.24, y0 + height * 0.62), (x0 + width * 0.76, y0 + height * 0.62)], fill=line, width=3)
            draw.line([(x0 + width * 0.24, y0 + height * 0.44), (x0 + width * 0.24, y0 + height * 0.1)], fill=line, width=3)
            draw.line([(x0 + width * 0.24, y0 + height * 0.1), (x0 + width * 0.76, y0 + height * 0.1)], fill=line, width=3)
            return
        if silhouette_key == "battery":
            draw.rounded_rectangle([x0, y0 + height * 0.12, x1, y1], radius=10, fill=fill, outline=line, width=3)
            draw.rectangle([x0 + width * 0.4, y0, x0 + width * 0.6, y0 + height * 0.14], fill=fill, outline=line, width=2)
            draw.line([(x0 + width * 0.25, y0 + height * 0.5), (x0 + width * 0.42, y0 + height * 0.5)], fill=accent, width=2)
            draw.line([(x0 + width * 0.66, y0 + height * 0.5), (x0 + width * 0.82, y0 + height * 0.5)], fill=accent, width=2)
            draw.line([(x0 + width * 0.74, y0 + height * 0.42), (x0 + width * 0.74, y0 + height * 0.58)], fill=accent, width=2)
            return
        if silhouette_key == "led":
            draw.ellipse([x0 + width * 0.24, y0, x0 + width * 0.76, y0 + height * 0.58], fill=fill, outline=line, width=3)
            draw.line([(x0 + width * 0.4, y0 + height * 0.56), (x0 + width * 0.36, y1)], fill=line, width=3)
            draw.line([(x0 + width * 0.6, y0 + height * 0.56), (x0 + width * 0.66, y1)], fill=line, width=3)
            self._draw_arrow_line(draw, (x0 + width * 0.76, y0 + height * 0.18), (x1, y0), accent, width=2, arrow_size=10)
            self._draw_arrow_line(draw, (x0 + width * 0.78, y0 + height * 0.34), (x1, y0 + height * 0.16), accent, width=2, arrow_size=10)
            return
        if silhouette_key == "resistor":
            points = [
                (x0, y0 + height * 0.5),
                (x0 + width * 0.14, y0 + height * 0.5),
                (x0 + width * 0.24, y0 + height * 0.24),
                (x0 + width * 0.38, y0 + height * 0.76),
                (x0 + width * 0.52, y0 + height * 0.24),
                (x0 + width * 0.66, y0 + height * 0.76),
                (x0 + width * 0.78, y0 + height * 0.5),
                (x1, y0 + height * 0.5),
            ]
            draw.line(points, fill=line, width=3)
            return
        if silhouette_key == "capacitor":
            draw.line([(x0 + width * 0.22, y0 + height * 0.16), (x0 + width * 0.22, y1)], fill=line, width=3)
            draw.line([(x0 + width * 0.46, y0 + height * 0.16), (x0 + width * 0.46, y1)], fill=line, width=3)
            draw.line([(x0, y0 + height * 0.58), (x0 + width * 0.22, y0 + height * 0.58)], fill=accent, width=2)
            draw.line([(x0 + width * 0.46, y0 + height * 0.58), (x1, y0 + height * 0.58)], fill=accent, width=2)
            return
        if silhouette_key == "diode":
            draw.line([(x0, y0 + height * 0.5), (x0 + width * 0.18, y0 + height * 0.5)], fill=line, width=3)
            draw.polygon([(x0 + width * 0.18, y0 + height * 0.16), (x0 + width * 0.18, y1 - height * 0.16), (x0 + width * 0.64, y0 + height * 0.5)], fill=fill, outline=line, width=3)
            draw.line([(x0 + width * 0.74, y0 + height * 0.16), (x0 + width * 0.74, y1 - height * 0.16)], fill=line, width=3)
            draw.line([(x0 + width * 0.74, y0 + height * 0.5), (x1, y0 + height * 0.5)], fill=line, width=3)
            return
        if silhouette_key == "board":
            draw.rounded_rectangle([x0, y0, x1, y1], radius=16, fill=fill, outline=line, width=3)
            for row in range(2):
                for col in range(4):
                    px = x0 + width * (0.12 + col * 0.18)
                    py = y0 + height * (0.18 + row * 0.28)
                    draw.rectangle([px, py, px + width * 0.1, py + height * 0.12], outline=accent, width=2)
            return
        if silhouette_key == "airplane":
            draw.line([(x0 + width * 0.08, y0 + height * 0.52), (x1, y0 + height * 0.52)], fill=line, width=4)
            draw.polygon([(x0 + width * 0.3, y0 + height * 0.52), (x0 + width * 0.56, y0 + height * 0.14), (x0 + width * 0.52, y0 + height * 0.52)], fill=fill, outline=line, width=3)
            draw.polygon([(x0 + width * 0.38, y0 + height * 0.52), (x0 + width * 0.62, y1 - height * 0.12), (x0 + width * 0.54, y0 + height * 0.52)], fill=fill, outline=line, width=3)
            draw.polygon([(x0 + width * 0.12, y0 + height * 0.52), (x0 + width * 0.22, y0 + height * 0.24), (x0 + width * 0.24, y0 + height * 0.52)], fill=fill, outline=line, width=3)
            return
        if silhouette_key == "leaf":
            draw.ellipse([x0, y0 + height * 0.16, x1, y1], fill=fill, outline=line, width=3)
            draw.line([(x0 + width * 0.12, y0 + height * 0.84), (x1 - width * 0.12, y0 + height * 0.24)], fill=accent, width=2)
            return
        if silhouette_key == "raindrop":
            draw.polygon([(x0 + width * 0.5, y0), (x1, y0 + height * 0.54), (x0 + width * 0.78, y1), (x0 + width * 0.22, y1), (x0, y0 + height * 0.54)], fill=fill, outline=line, width=3)
            return
        if silhouette_key == "cell":
            draw.ellipse([x0, y0, x1, y1], fill=fill, outline=line, width=3)
            draw.ellipse([x0 + width * 0.28, y0 + height * 0.28, x0 + width * 0.72, y0 + height * 0.72], fill=palette["region_alt"], outline=accent, width=2)
            return
        draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=fill, outline=line, width=3)

    def _build_object_sprite(self, obj: Dict[str, Any], palette: Dict[str, Tuple[int, int, int]], filled: bool = False) -> Image.Image:
        width = max(40, int(obj.get("width", 80)))
        height = max(40, int(obj.get("height", 80)))
        padding = 18
        sprite = Image.new("RGBA", (width + padding * 2, height + padding * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(sprite)
        style_variant = str(obj.get("style_variant") or obj.get("layout_style") or "")
        stroke_render_profile = obj.get("stroke_render_profile") if isinstance(obj.get("stroke_render_profile"), dict) else obj.get("stroke_style_profile")
        rendered = False
        if str(obj.get("render_representation") or "") == "stroke_native" or obj.get("stroke_payload"):
            rendered = self._draw_stroke_payload(
                draw,
                (padding, padding, padding + width, padding + height),
                obj.get("stroke_payload") if isinstance(obj.get("stroke_payload"), list) else [],
                palette,
                style_variant=style_variant,
                stroke_render_profile=stroke_render_profile if isinstance(stroke_render_profile, dict) else {},
            )
        if not rendered:
            shape_recipe = obj.get("shape_recipe") if isinstance(obj.get("shape_recipe"), dict) else {}
            rendered = self._draw_shape_recipe(
                draw,
                (padding, padding, padding + width, padding + height),
                shape_recipe,
                palette,
                filled=filled,
                style_variant=style_variant,
            )
        if not rendered:
            self._draw_asset_symbol(
                draw,
                (padding, padding, padding + width, padding + height),
                str(obj.get("silhouette_key") or obj.get("asset_key") or "generic_object"),
                palette,
                filled=filled,
            )
        rotation = float(obj.get("rotation", 0.0) or 0.0)
        if abs(rotation) > 0.1:
            sprite = sprite.rotate(-rotation, expand=True, resample=Image.BICUBIC)
        return sprite

    def _draw_scene_connector(
        self,
        draw: ImageDraw,
        connector: Dict[str, Any],
        objects: Dict[str, Dict[str, Any]],
        palette: Dict[str, Tuple[int, int, int]],
        show_labels: bool = True,
    ):
        from_obj = objects.get(connector.get("from_id"))
        to_obj = objects.get(connector.get("to_id"))
        if not from_obj or not to_obj or not connector.get("visible", True):
            return
        start, end = self._connector_points_for_scene(from_obj, to_obj)
        connector_type = str(connector.get("type", "relation"))
        label = str(connector.get("label", "连接"))
        line_color = palette["accent"] if connector_type in {"beam", "arrow"} else palette["line"]
        if connector_type == "beam":
            self._draw_dashed_line(draw, start, end, line_color, width=3, dash_length=12)
        elif connector_type == "arrow":
            self._draw_arrow_line(draw, start, end, line_color, width=3, arrow_size=12)
        elif connector_type == "wire":
            draw.line([start, end], fill=line_color, width=3)
        else:
            self._draw_dashed_line(draw, start, end, line_color, width=2, dash_length=10)
        if show_labels and label:
            mid_x = int((start[0] + end[0]) / 2)
            mid_y = int((start[1] + end[1]) / 2 - 16)
            draw.text((mid_x, mid_y), label, fill=palette["text"], font=self.font, anchor="mm")

    def _resolve_scene_view_flags(
        self,
        layout_options: Dict[str, Any],
        *,
        view_mode: str | None = None,
        force_show_labels: bool | None = None,
        force_show_grid: bool | None = None,
        force_show_guides: bool | None = None,
    ) -> Dict[str, Any]:
        resolved_view = normalize_view_mode(view_mode or layout_options.get("sketch_view_mode"))
        annotation_level = normalize_annotation_level(layout_options.get("annotation_level"))
        show_grid = bool(layout_options.get("show_grid", True))
        show_labels = bool(layout_options.get("show_labels", False))
        show_guides = bool(layout_options.get("show_guides", False))

        if resolved_view == "rough":
            show_labels = False
            show_guides = False
        elif resolved_view == "structure":
            show_labels = show_labels and annotation_level != "off"
            show_guides = False
        elif resolved_view == "annotated":
            show_labels = annotation_level != "off"
            show_guides = True
        elif resolved_view == "region":
            show_labels = annotation_level != "off"
            show_guides = True

        if force_show_labels is not None:
            show_labels = bool(force_show_labels)
        if force_show_grid is not None:
            show_grid = bool(force_show_grid)
        if force_show_guides is not None:
            show_guides = bool(force_show_guides)

        return {
            "view_mode": resolved_view,
            "annotation_level": annotation_level,
            "show_grid": show_grid,
            "show_labels": show_labels,
            "show_guides": show_guides,
            "show_regions": resolved_view == "region",
            "show_override_regions": resolved_view in {"annotated", "region"},
        }

    def _region_box_for_object(self, obj: Dict[str, Any], region: Dict[str, Any]) -> Tuple[int, int, int, int]:
        x0 = int(obj.get("x", 0) + obj.get("width", 0) * float(region.get("x", 0.0)))
        y0 = int(obj.get("y", 0) + obj.get("height", 0) * float(region.get("y", 0.0)))
        x1 = int(x0 + obj.get("width", 0) * float(region.get("width", 0.0)))
        y1 = int(y0 + obj.get("height", 0) * float(region.get("height", 0.0)))
        return x0, y0, x1, y1

    def _draw_region_shape(
        self,
        draw: ImageDraw,
        box: Tuple[int, int, int, int],
        region: Dict[str, Any],
        *,
        outline: Tuple[int, int, int] | Tuple[int, int, int, int],
        width: int = 2,
        fill: Tuple[int, int, int, int] | None = None,
        dash_length: int = 8,
    ) -> None:
        shape = str(region.get("shape") or "rect")
        if shape == "ellipse":
            draw.ellipse(box, outline=outline, fill=fill, width=width)
            return
        if dash_length > 0 and fill is None:
            self._draw_dashed_rectangle(draw, box, outline, width=width, dash_length=dash_length)
            return
        draw.rounded_rectangle(box, radius=max(4, int(min(box[2] - box[0], box[3] - box[1]) * 0.12)), outline=outline, fill=fill, width=width)

    def _region_style(self, action: str, palette: Dict[str, Tuple[int, int, int]]) -> Dict[str, Any]:
        normalized = str(action or "").strip().lower()
        if normalized == "hide":
            return {
                "outline": (244, 63, 94),
                "fill": self._with_alpha((244, 63, 94), 0.1),
                "text": (251, 113, 133),
                "dash": 7,
            }
        if normalized == "weaken":
            return {
                "outline": (56, 189, 248),
                "fill": self._with_alpha((56, 189, 248), 0.12),
                "text": (125, 211, 252),
                "dash": 8,
            }
        if normalized == "emphasize":
            return {
                "outline": (245, 158, 11),
                "fill": self._with_alpha((245, 158, 11), 0.1),
                "text": (253, 224, 71),
                "dash": 0,
            }
        if normalized == "replace":
            return {
                "outline": (168, 85, 247),
                "fill": self._with_alpha((168, 85, 247), 0.12),
                "text": (196, 181, 253),
                "dash": 6,
            }
        return {
            "outline": palette["accent"],
            "fill": self._with_alpha(palette["region_fill"], 0.12),
            "text": palette["accent"],
            "dash": 8,
        }

    def _draw_region_overlays(
        self,
        draw: ImageDraw,
        scene: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
        *,
        labels: bool = True,
        only_overrides: bool = False,
    ) -> None:
        for obj in scene.get("object_instances", []) or []:
            regions = [item for item in obj.get("region_masks", []) or [] if isinstance(item, dict)]
            overrides = obj.get("region_overrides") if isinstance(obj.get("region_overrides"), dict) else {}
            if only_overrides:
                regions = [item for item in regions if overrides.get(str(item.get("id") or ""))]
            if not regions:
                continue
            for region in regions:
                region_id = str(region.get("id") or "")
                override = overrides.get(region_id) if isinstance(overrides, dict) else None
                action = str((override or {}).get("action") or "").strip().lower()
                style = self._region_style(action, palette)
                box = self._region_box_for_object(obj, region)
                self._draw_region_shape(
                    draw,
                    box,
                    region,
                    outline=style["outline"],
                    width=3 if action else 2,
                    fill=style["fill"] if action else None,
                    dash_length=style["dash"],
                )
                if labels:
                    label = str(region.get("label") or region_id or "region")
                    action_label = {"hide": "隐藏", "weaken": "弱化", "emphasize": "强调", "replace": "替换"}.get(action, "")
                    if action:
                        label = f"{label} · {action_label or action}"
                    draw.text((box[0] + 6, max(10, box[1] - 12)), label, fill=style["text"], font=self.font)

    def _render_semantic_scene_image(
        self,
        scene_spec: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
        title: str,
        filled: bool = False,
        *,
        include_title: bool = True,
        view_mode: str | None = None,
        force_show_labels: bool | None = None,
        force_show_grid: bool | None = None,
        force_show_guides: bool | None = None,
    ) -> Image.Image:
        scene = normalize_scene_spec_v2(scene_spec)
        canvas_size = (scene["canvas_size"]["width"], scene["canvas_size"]["height"])
        base = Image.new("RGBA", canvas_size, (*palette["background"], 255))
        draw = ImageDraw.Draw(base)
        layout_options = scene.get("layout_options", {})
        view_flags = self._resolve_scene_view_flags(
            layout_options,
            view_mode=view_mode,
            force_show_labels=force_show_labels,
            force_show_grid=force_show_grid,
            force_show_guides=force_show_guides,
        )
        if view_flags["show_grid"]:
            self._draw_preview_grid(draw, canvas_size, palette)
        for layer in sorted(scene.get("background_layers", []), key=lambda item: item.get("z_index", 0)):
            self._draw_scene_background_layer(draw, layer, palette, filled=filled)
            if view_flags["show_labels"] and view_flags["view_mode"] in {"annotated", "region"}:
                draw.text((layer["x"] + 16, layer["y"] + 20), str(layer.get("label", "")), fill=palette["text"], font=self.font)
        if include_title and title:
            draw.text((24, 20), title, fill=palette["text"], font=self.font)
            style_desc = f"{layout_options.get('scene_type', 'scene')} | {layout_options.get('sketch_style', 'line_art')} | {view_flags['view_mode']}"
            draw.text((24, 52), style_desc, fill=palette["text"], font=self.font)
        objects_by_id = {item["id"]: item for item in scene.get("object_instances", [])}
        for connector in scene.get("connectors", []):
            self._draw_scene_connector(draw, connector, objects_by_id, palette, show_labels=view_flags["show_labels"])
        for obj in sorted(scene.get("object_instances", []), key=lambda item: item.get("z_index", 0)):
            if obj.get("visible", True) is False:
                continue
            if view_flags["show_guides"]:
                x0, y0, x1, y1 = self._scene_bbox(obj)
                self._draw_dashed_rectangle(draw, (x0, y0, x1, y1), palette["guide"], width=1, dash_length=8)
            sprite = self._build_object_sprite(obj, palette, filled=filled)
            left = int(obj["x"] + obj["width"] / 2 - sprite.width / 2)
            top = int(obj["y"] + obj["height"] / 2 - sprite.height / 2)
            base.alpha_composite(sprite, (left, top))
            if view_flags["show_labels"]:
                draw.text((obj["x"] + obj["width"] / 2, obj["y"] - 18), str(obj.get("concept", "")), fill=palette["text"], font=self.font, anchor="mm")
        if view_flags["show_regions"]:
            self._draw_region_overlays(draw, scene, palette, labels=True, only_overrides=False)
        elif view_flags["show_override_regions"]:
            self._draw_region_overlays(draw, scene, palette, labels=view_flags["show_labels"], only_overrides=True)
        return base.convert("RGB")

    def _render_control_palette(self) -> Dict[str, Tuple[int, int, int]]:
        return {
            "background": (242, 241, 237),
            "sky": (224, 229, 236),
            "ground": (223, 226, 216),
            "road": (214, 216, 219),
            "water": (214, 223, 228),
            "wall": (229, 226, 221),
            "board": (226, 222, 214),
            "process_band": (223, 219, 231),
            "scene_object": (150, 158, 169),
            "process_motif": (160, 151, 176),
            "schematic_symbol": (144, 159, 148),
            "subject": (128, 141, 158),
            "user": (163, 146, 126),
            "connector": (132, 137, 144),
        }

    def _compose_rich_preview_image(
        self,
        structural_sketch_image: Image.Image,
        low_preview_image: Image.Image,
        *,
        sketch_style: str,
    ) -> Image.Image:
        structural = structural_sketch_image.convert("RGBA")
        low_preview = low_preview_image.convert("RGBA")
        base = Image.blend(low_preview, structural, 0.42 if sketch_style == "clean_line" else 0.5)
        overlay = structural.copy()
        overlay.putalpha(216 if sketch_style == "clean_line" else 192)
        merged = Image.alpha_composite(base, overlay)
        return merged.convert("RGB")

    def _line_alpha_mask(
        self,
        image: Image.Image,
        *,
        gain: float = 1.0,
        bias: int = 0,
    ) -> Image.Image:
        inverted = ImageOps.invert(image.convert("L"))
        if gain != 1.0 or bias != 0:
            inverted = inverted.point(lambda px: max(0, min(255, int(px * gain) + bias)))
        return inverted

    def _soft_mass_outline_mask(self, mask: Image.Image) -> Image.Image:
        inner = mask.filter(ImageFilter.GaussianBlur(radius=1.6))
        outer = mask.filter(ImageFilter.GaussianBlur(radius=7.5))
        outline = ImageChops.difference(inner, outer)
        return outline.point(lambda px: 0 if px < 8 else min(255, int(px * 1.18)))

    def _balanced_upstream_box(
        self,
        box: Tuple[int, int, int, int],
        *,
        canvas_size: Tuple[int, int],
        asset_key: str,
        multi_anchor_scene: bool,
    ) -> Tuple[int, int, int, int]:
        if not multi_anchor_scene:
            return box
        x0, y0, x1, y1 = box
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        short_side = min(width, height)
        long_side = max(width, height)
        key = str(asset_key or "").strip().lower()
        min_short = 140
        max_long = 280
        if key in {"house", "home", "building"}:
            min_short = 168
        elif key in {"tree", "leaf", "plant", "bush"}:
            min_short = 156
            max_long = 260
        elif key in {"person", "human", "figure", "character"}:
            min_short = 148
            max_long = 292
        scale = 1.0
        if short_side < min_short:
            scale = max(scale, min(1.55, float(min_short) / max(1.0, float(short_side))))
        if long_side > max_long:
            scale = min(scale, max(0.78, float(max_long) / max(1.0, float(long_side))))
        if abs(scale - 1.0) < 0.02:
            return box
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        new_width = max(24, int(round(width * scale)))
        new_height = max(24, int(round(height * scale)))
        if key in {"person", "human", "figure", "character"} and new_height > new_width:
            aspect = float(new_height) / max(1.0, float(new_width))
            if aspect > 1.46:
                target_width = int(round(new_height / 1.46))
                new_width = max(new_width, min(canvas_size[0], target_width))
                new_height = max(24, int(round(new_height * 0.94)))
        new_x0 = max(0, int(round(center_x - new_width / 2.0)))
        new_y0 = max(0, int(round(center_y - new_height / 2.0)))
        new_x1 = min(canvas_size[0], new_x0 + new_width)
        new_y1 = min(canvas_size[1], new_y0 + new_height)
        if new_x1 - new_x0 < 24:
            new_x0 = max(0, new_x1 - 24)
        if new_y1 - new_y0 < 24:
            new_y0 = max(0, new_y1 - 24)
        return (new_x0, new_y0, new_x1, new_y1)

    def _sd_upstream_background_color(self, layer_type: str) -> Tuple[Tuple[int, int, int], int, int]:
        key = str(layer_type or "").strip().lower()
        if key in {"ground", "road", "board"}:
            return (212, 217, 223), 36, 10
        if key in {"water"}:
            return (206, 216, 224), 34, 10
        if key in {"sky", "cloud_band"}:
            return (238, 236, 232), 18, 16
        if key in {"process_band"}:
            return (220, 221, 226), 26, 8
        return (234, 232, 228), 16, 12

    def _sd_upstream_object_style(self, obj: Dict[str, Any], scene_type: str = "scene") -> Tuple[Tuple[int, int, int], int, int]:
        depth_band = str(obj.get("depth_band") or "").strip().lower()
        role = str(obj.get("role") or "").strip().lower()
        tone = (162, 164, 170)
        alpha = 96
        blur = 14
        if depth_band == "foreground":
            tone = (136, 138, 146)
            alpha = 126
            blur = 9
        elif depth_band == "background":
            tone = (184, 186, 192)
            alpha = 72
            blur = 18
        if role in {"subject", "focus", "core_subject"}:
            tone = (118, 121, 130)
            alpha = max(alpha, 134)
            blur = max(7, blur - 2)
        if scene_type == "process":
            alpha = min(168, alpha + 8)
            blur = max(6, blur - 4)
        elif scene_type == "schematic":
            tone = tuple(max(96, min(188, channel - 8)) for channel in tone)
            alpha = min(164, alpha + 10)
            blur = max(5, blur - 5)
        return tone, alpha, blur

    def _render_sd_upstream_control_image(
        self,
        scene_spec: Dict[str, Any],
        *,
        structural_sketch_image: Image.Image,
    ) -> Image.Image:
        scene = normalize_scene_spec_v2(scene_spec)
        canvas_size = (scene["canvas_size"]["width"], scene["canvas_size"]["height"])
        base = Image.new("RGBA", canvas_size, (245, 242, 236, 255))
        layout_options = scene.get("layout_options", {}) or {}
        scene_type = str(layout_options.get("scene_type") or "scene")

        background_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        background_draw = ImageDraw.Draw(background_layer, "RGBA")
        for layer in sorted(scene.get("background_layers", []), key=lambda item: item.get("z_index", 0)):
            color, alpha, radius = self._sd_upstream_background_color(str(layer.get("type") or ""))
            background_draw.rounded_rectangle(
                [layer["x"], layer["y"], layer["x"] + layer["width"], layer["y"] + layer["height"]],
                radius=radius,
                fill=(*color, alpha),
                outline=None,
            )
        background_layer = background_layer.filter(ImageFilter.GaussianBlur(radius=8 if scene_type == "scene" else 5))
        base.alpha_composite(background_layer)

        visible_objects = [obj for obj in scene.get("object_instances", []) if obj.get("visible", True) is not False]
        multi_anchor_scene = len(visible_objects) > 1
        object_support_mask = Image.new("L", canvas_size, 0)
        for obj in sorted(visible_objects, key=lambda item: item.get("z_index", 0)):
            if obj.get("visible", True) is False:
                continue
            x0, y0, x1, y1 = self._scene_bbox(obj)
            width = max(24, x1 - x0)
            height = max(24, y1 - y0)
            key = str(obj.get("asset_key") or obj.get("silhouette_key") or obj.get("concept") or "").strip().lower()
            padding_ratio = 0.1 if scene_type == "scene" else 0.06
            if key in {"person", "human", "figure", "character"}:
                padding_ratio = 0.045 if scene_type == "scene" else 0.035
            padding = max(6 if key in {"person", "human", "figure", "character"} else 8, int(min(width, height) * padding_ratio))
            raw_box = (
                max(0, x0 - padding),
                max(0, y0 - padding),
                min(canvas_size[0], x1 + padding),
                min(canvas_size[1], y1 + padding),
            )
            box = self._balanced_upstream_box(
                raw_box,
                canvas_size=canvas_size,
                asset_key=str(obj.get("asset_key") or obj.get("silhouette_key") or obj.get("concept") or ""),
                multi_anchor_scene=multi_anchor_scene,
            )
            mask = Image.new("L", canvas_size, 0)
            mask_draw = ImageDraw.Draw(mask)
            self._draw_upstream_object_mass_mask(mask_draw, box, obj)
            mask = self._apply_region_overrides_to_mask(mask, obj)
            object_support_mask = ImageChops.lighter(
                object_support_mask,
                mask.filter(ImageFilter.MaxFilter(size=9)),
            )
            tone, alpha, blur_radius = self._sd_upstream_object_style(obj, scene_type=scene_type)
            if multi_anchor_scene:
                alpha = max(alpha, 116)
                blur_radius = max(6, blur_radius - 2)
                if key in {"house", "home", "building"}:
                    tone = (128, 130, 138)
                    alpha = max(alpha, 144)
                    blur_radius = max(5, blur_radius - 2)
                elif key in {"person", "human", "figure", "character"}:
                    tone = (126, 128, 136)
                    alpha = max(alpha, 130)
                    blur_radius = max(5, blur_radius - 3)
                elif key in {"tree", "leaf", "plant", "bush"}:
                    alpha = max(alpha, 132)
            self._apply_soft_mask(base, mask, tone, alpha, blur_radius)

        object_support_mask = object_support_mask.filter(ImageFilter.GaussianBlur(radius=4))
        outline_alpha = self._soft_mass_outline_mask(object_support_mask)
        outline_layer = Image.new("RGBA", canvas_size, (104, 101, 96, 0))
        outline_layer.putalpha(outline_alpha.point(lambda px: min(255, int(px * 0.55))))
        base.alpha_composite(outline_layer)

        preserve_structural_lines = bool(layout_options.get("sd_upstream_preserve_structural_lines", False))
        if preserve_structural_lines:
            guide_alpha = ImageChops.multiply(
                self._line_alpha_mask(structural_sketch_image, gain=0.58, bias=-96),
                object_support_mask,
            ).filter(ImageFilter.GaussianBlur(radius=2.2))
            warm_guide = Image.new("RGBA", canvas_size, (116, 112, 108, 0))
            warm_guide.putalpha(guide_alpha)
            base.alpha_composite(warm_guide)
        return base.convert("RGB")

    def _draw_shape_recipe_mask(
        self,
        draw: ImageDraw,
        box: Tuple[int, int, int, int],
        shape_recipe: Dict[str, Any],
        value: int = 255,
    ) -> bool:
        parts = list((shape_recipe or {}).get("parts") or [])
        if not parts:
            return False
        x0, y0, x1, y1 = box
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        drawn = False

        for part in parts:
            kind = str(part.get("kind") or "").lower()
            fill_role = str(part.get("fill_role", "fill") or "fill").lower()
            stroke_width = max(2, int(round(max(width, height) * float(part.get("stroke_width", 0.03) or 0.03))))

            if kind == "rect" and fill_role not in {"none", "transparent"}:
                rect = [
                    x0 + width * float(part.get("x", 0.0)),
                    y0 + height * float(part.get("y", 0.0)),
                    x0 + width * (float(part.get("x", 0.0)) + float(part.get("w", 0.0))),
                    y0 + height * (float(part.get("y", 0.0)) + float(part.get("h", 0.0))),
                ]
                rx = max(0, int(min(width, height) * float(part.get("rx", 0.0) or 0.0)))
                draw.rounded_rectangle(rect, radius=rx, fill=value)
                drawn = True
                continue

            if kind == "ellipse" and fill_role not in {"none", "transparent"}:
                rect = [
                    x0 + width * float(part.get("x", 0.0)),
                    y0 + height * float(part.get("y", 0.0)),
                    x0 + width * (float(part.get("x", 0.0)) + float(part.get("w", 0.0))),
                    y0 + height * (float(part.get("y", 0.0)) + float(part.get("h", 0.0))),
                ]
                draw.ellipse(rect, fill=value)
                drawn = True
                continue

            if kind == "polygon" and fill_role not in {"none", "transparent"}:
                points = self._scaled_points(list(part.get("points") or []), box)
                if points:
                    draw.polygon(points, fill=value)
                    drawn = True
                continue

            if kind == "line":
                start = (
                    x0 + width * float(part.get("x1", 0.0)),
                    y0 + height * float(part.get("y1", 0.0)),
                )
                end = (
                    x0 + width * float(part.get("x2", 0.0)),
                    y0 + height * float(part.get("y2", 0.0)),
                )
                draw.line([start, end], fill=value, width=stroke_width)
                drawn = True
                continue

            if kind in {"polyline", "path"}:
                points = self._scaled_points(list(part.get("points") or []), box)
                if len(points) >= 2:
                    draw.line(points, fill=value, width=stroke_width)
                    drawn = True

        return drawn

    def _draw_stroke_payload_mask(
        self,
        draw: ImageDraw,
        box: Tuple[int, int, int, int],
        stroke_payload: List[List[List[float]]] | None,
        value: int = 255,
        stroke_render_profile: Dict[str, Any] | None = None,
    ) -> bool:
        strokes = [stroke for stroke in list(stroke_payload or []) if len(stroke) >= 2]
        if not strokes:
            return False
        x0, y0, x1, y1 = box
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        profile = stroke_render_profile if isinstance(stroke_render_profile, dict) else {}
        min_width = float(profile.get("line_width_min", profile.get("min_width", 0.018)) or 0.018)
        max_width = float(profile.get("line_width_max", profile.get("max_width", 0.04)) or 0.04)
        for index, stroke in enumerate(strokes):
            ratio = index / max(1, len(strokes) - 1) if len(strokes) > 1 else 0.0
            line_width = max(2, int(round(max(width, height) * (min_width + (max_width - min_width) * ratio))))
            points = [(x0 + width * float(px), y0 + height * float(py)) for px, py in stroke]
            if len(points) >= 2:
                draw.line(points, fill=value, width=line_width)
        return True

    def _draw_object_mask(self, draw: ImageDraw, box: Tuple[int, int, int, int], obj: Dict[str, Any], value: int = 255) -> bool:
        stroke_render_profile = obj.get("stroke_render_profile") if isinstance(obj.get("stroke_render_profile"), dict) else obj.get("stroke_style_profile")
        if str(obj.get("render_representation") or "") == "stroke_native" or obj.get("stroke_payload"):
            drew = self._draw_stroke_payload_mask(
                draw,
                box,
                obj.get("stroke_payload") if isinstance(obj.get("stroke_payload"), list) else [],
                value=value,
                stroke_render_profile=stroke_render_profile if isinstance(stroke_render_profile, dict) else {},
            )
            if drew:
                return True
        return self._draw_shape_recipe_mask(draw, box, obj.get("shape_recipe") or {}, value=value)

    def _draw_upstream_object_mass_mask(
        self,
        draw: ImageDraw,
        box: Tuple[int, int, int, int],
        obj: Dict[str, Any],
        value: int = 255,
    ) -> None:
        x0, y0, x1, y1 = box
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        key = str(obj.get("asset_key") or obj.get("silhouette_key") or obj.get("concept") or "generic_object").strip().lower()

        if key in {"person", "human", "figure", "character"}:
            shoulder_y = y0 + height * 0.22
            hip_y = y0 + height * 0.58
            center_x = x0 + width * 0.5
            draw.ellipse([x0 + width * 0.38, y0 + height * 0.02, x0 + width * 0.62, y0 + height * 0.18], fill=value)
            draw.rounded_rectangle(
                [x0 + width * 0.43, y0 + height * 0.16, x0 + width * 0.57, y0 + height * 0.24],
                radius=max(3, int(min(width, height) * 0.03)),
                fill=value,
            )
            draw.ellipse([x0 + width * 0.25, shoulder_y, x0 + width * 0.75, y0 + height * 0.48], fill=value)
            draw.rounded_rectangle(
                [x0 + width * 0.34, y0 + height * 0.24, x0 + width * 0.66, hip_y],
                radius=max(6, int(min(width, height) * 0.08)),
                fill=value,
            )
            draw.polygon(
                [
                    (x0 + width * 0.28, y0 + height * 0.26),
                    (x0 + width * 0.18, y0 + height * 0.52),
                    (x0 + width * 0.24, y0 + height * 0.58),
                    (x0 + width * 0.38, y0 + height * 0.36),
                ],
                fill=value,
            )
            draw.polygon(
                [
                    (x0 + width * 0.72, y0 + height * 0.26),
                    (x0 + width * 0.82, y0 + height * 0.52),
                    (x0 + width * 0.76, y0 + height * 0.58),
                    (x0 + width * 0.62, y0 + height * 0.36),
                ],
                fill=value,
            )
            draw.polygon(
                [
                    (x0 + width * 0.38, hip_y),
                    (center_x - width * 0.04, y1),
                    (center_x, y1),
                    (center_x - width * 0.01, y0 + height * 0.72),
                ],
                fill=value,
            )
            draw.polygon(
                [
                    (x0 + width * 0.62, hip_y),
                    (center_x + width * 0.04, y1),
                    (center_x, y1),
                    (center_x + width * 0.01, y0 + height * 0.72),
                ],
                fill=value,
            )
            return

        if key in {"house", "home"}:
            draw.polygon(
                [
                    (x0 + width * 0.5, y0 + height * 0.02),
                    (x0 + width * 0.16, y0 + height * 0.34),
                    (x1 - width * 0.16, y0 + height * 0.34),
                ],
                fill=value,
            )
            draw.rounded_rectangle(
                [x0 + width * 0.18, y0 + height * 0.28, x1 - width * 0.18, y1],
                radius=max(8, int(min(width, height) * 0.1)),
                fill=value,
            )
            return

        if key in {"building", "tower"}:
            draw.rounded_rectangle(
                [x0 + width * 0.08, y0 + height * 0.04, x1 - width * 0.08, y1],
                radius=max(8, int(min(width, height) * 0.08)),
                fill=value,
            )
            return

        if key in {"tree", "leaf", "plant", "bush"}:
            draw.rounded_rectangle(
                [x0 + width * 0.42, y0 + height * 0.54, x0 + width * 0.58, y1],
                radius=max(4, int(min(width, height) * 0.05)),
                fill=value,
            )
            draw.ellipse([x0 + width * 0.06, y0 + height * 0.18, x0 + width * 0.54, y0 + height * 0.76], fill=value)
            draw.ellipse([x0 + width * 0.28, y0, x1 - width * 0.12, y0 + height * 0.68], fill=value)
            draw.ellipse([x0 + width * 0.48, y0 + height * 0.2, x1, y0 + height * 0.82], fill=value)
            return

        if width >= height * 1.25:
            draw.rounded_rectangle(
                [x0, y0 + height * 0.12, x1, y1 - height * 0.12],
                radius=max(8, int(min(width, height) * 0.12)),
                fill=value,
            )
            return

        draw.rounded_rectangle(
            [x0 + width * 0.08, y0 + height * 0.04, x1 - width * 0.08, y1 - height * 0.04],
            radius=max(10, int(min(width, height) * 0.18)),
            fill=value,
        )

    def _draw_asset_symbol_mask(
        self,
        draw: ImageDraw,
        box: Tuple[int, int, int, int],
        asset_key: str,
        value: int = 255,
    ) -> None:
        x0, y0, x1, y1 = box
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        key = str(asset_key or "generic_object")

        if key == "sun":
            draw.ellipse([x0, y0, x1, y1], fill=value)
            return
        if key == "cloud":
            draw.ellipse([x0, y0 + height * 0.2, x0 + width * 0.38, y1], fill=value)
            draw.ellipse([x0 + width * 0.18, y0, x0 + width * 0.72, y0 + height * 0.78], fill=value)
            draw.ellipse([x0 + width * 0.5, y0 + height * 0.16, x1, y0 + height * 0.9], fill=value)
            return
        if key in {"building", "tower"}:
            draw.rounded_rectangle([x0, y0, x1, y1], radius=max(8, int(min(width, height) * 0.06)), fill=value)
            return
        if key == "house":
            draw.polygon([(x0 + width * 0.5, y0), (x0, y0 + height * 0.34), (x1, y0 + height * 0.34)], fill=value)
            draw.rounded_rectangle([x0 + width * 0.12, y0 + height * 0.3, x1 - width * 0.12, y1], radius=max(8, int(min(width, height) * 0.06)), fill=value)
            return
        if key in {"tree", "leaf"}:
            draw.rectangle([x0 + width * 0.42, y0 + height * 0.56, x0 + width * 0.58, y1], fill=value)
            draw.ellipse([x0 + width * 0.08, y0, x1 - width * 0.08, y0 + height * 0.72], fill=value)
            return
        if key == "person":
            draw.ellipse([x0 + width * 0.32, y0, x0 + width * 0.68, y0 + height * 0.28], fill=value)
            draw.rounded_rectangle([x0 + width * 0.34, y0 + height * 0.22, x0 + width * 0.66, y1], radius=max(6, int(min(width, height) * 0.08)), fill=value)
            return
        if key == "car":
            draw.rounded_rectangle([x0 + width * 0.08, y0 + height * 0.36, x1 - width * 0.08, y0 + height * 0.78], radius=max(8, int(min(width, height) * 0.08)), fill=value)
            draw.polygon([(x0 + width * 0.24, y0 + height * 0.36), (x0 + width * 0.42, y0 + height * 0.12), (x0 + width * 0.72, y0 + height * 0.12), (x0 + width * 0.84, y0 + height * 0.36)], fill=value)
            return
        if key in {"window", "door", "generic_panel", "module"}:
            draw.rounded_rectangle([x0, y0, x1, y1], radius=max(6, int(min(width, height) * 0.08)), fill=value)
            return
        draw.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=max(10, int(min(width, height) * 0.16)),
            fill=value,
        )

    def _draw_object_region_mask(
        self,
        draw: ImageDraw,
        obj: Dict[str, Any],
        region: Dict[str, Any],
        value: int = 255,
    ) -> None:
        box = self._region_box_for_object(obj, region)
        shape = str(region.get("shape") or "rect")
        if shape == "ellipse":
            draw.ellipse(box, fill=value)
            return
        draw.rounded_rectangle(
            box,
            radius=max(4, int(min(box[2] - box[0], box[3] - box[1]) * 0.12)),
            fill=value,
        )

    def _apply_region_overrides_to_mask(self, mask: Image.Image, obj: Dict[str, Any]) -> Image.Image:
        overrides = obj.get("region_overrides") if isinstance(obj.get("region_overrides"), dict) else {}
        if not overrides:
            return mask
        adjusted = mask.copy()
        for region in obj.get("region_masks", []) or []:
            if not isinstance(region, dict):
                continue
            region_id = str(region.get("id") or "")
            action = str((overrides.get(region_id) or {}).get("action") or "").strip().lower()
            if not action:
                continue
            region_mask = Image.new("L", adjusted.size, 0)
            region_draw = ImageDraw.Draw(region_mask)
            self._draw_object_region_mask(region_draw, obj, region, value=255)
            if action == "hide":
                adjusted.paste(0, mask=region_mask)
                continue
            if action == "replace":
                subtract = Image.new("L", adjusted.size, 0)
                subtract.paste(120, mask=region_mask)
                adjusted = ImageChops.subtract(adjusted, subtract)
                continue
            if action == "weaken":
                subtract = Image.new("L", adjusted.size, 0)
                subtract.paste(96, mask=region_mask)
                adjusted = ImageChops.subtract(adjusted, subtract)
                continue
            if action == "emphasize":
                boost = Image.new("L", adjusted.size, 0)
                boost.paste(84, mask=region_mask)
                adjusted = ImageChops.add(adjusted, boost)
        return adjusted

    def _color_from_index(self, index: int) -> Tuple[int, int, int]:
        return (
            32 + (index * 73) % 192,
            36 + (index * 91) % 184,
            44 + (index * 57) % 176,
        )

    def _render_hit_map_image(self, scene_spec: Dict[str, Any]) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        scene = normalize_scene_spec_v2(scene_spec)
        canvas_size = (scene["canvas_size"]["width"], scene["canvas_size"]["height"])
        image = Image.new("RGB", canvas_size, (0, 0, 0))
        draw = ImageDraw.Draw(image)
        legend: List[Dict[str, Any]] = []
        color_index = 1

        for obj in sorted(scene.get("object_instances", []), key=lambda item: item.get("z_index", 0)):
            if obj.get("visible", True) is False:
                continue
            box = self._scene_bbox(obj)
            color = self._color_from_index(color_index)
            color_index += 1
            if not self._draw_object_mask(draw, box, obj, value=color):
                self._draw_asset_symbol_mask(draw, box, str(obj.get("asset_key") or obj.get("silhouette_key") or "generic_object"), value=color)
            legend.append(
                {
                    "object_id": obj.get("id"),
                    "region_id": "",
                    "concept": obj.get("concept", ""),
                    "rgb": color,
                }
            )
            for region in obj.get("region_masks", []) or []:
                if not isinstance(region, dict):
                    continue
                region_color = self._color_from_index(color_index)
                color_index += 1
                self._draw_object_region_mask(draw, obj, region, value=region_color)
                legend.append(
                    {
                        "object_id": obj.get("id"),
                        "region_id": region.get("id", ""),
                        "concept": obj.get("concept", ""),
                        "label": region.get("label", ""),
                        "rgb": region_color,
                    }
                )
        return image, legend

    def _control_layer_color(self, layer: Dict[str, Any], control_palette: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
        layer_type = str(layer.get("type") or "layer")
        return control_palette.get(layer_type, control_palette.get("wall", (226, 224, 220)))

    def _control_object_style(
        self,
        obj: Dict[str, Any],
        control_palette: Dict[str, Tuple[int, int, int]],
        scene_type: str = "scene",
    ) -> Tuple[Tuple[int, int, int], int, int]:
        visual_family = str(obj.get("visual_family") or "scene_object")
        role = str(obj.get("role") or "")
        depth_band = str(obj.get("depth_band") or "")
        source = str(obj.get("source") or "")
        color = control_palette.get(visual_family, control_palette["scene_object"])
        if role in {"subject", "focus", "core_subject"}:
            color = control_palette["subject"]
        elif source == "user":
            color = control_palette["user"]

        alpha = 82
        if depth_band == "foreground":
            alpha = 118
        elif depth_band == "midground":
            alpha = 98
        elif depth_band == "background":
            alpha = 74
        if role in {"subject", "focus", "core_subject"}:
            alpha += 24
        if source == "user":
            alpha += 10
        blur = 18
        if depth_band == "foreground":
            blur = 14
        elif depth_band == "background":
            blur = 22
        if scene_type == "process":
            blur = max(6, blur - 8)
            alpha = min(196, alpha + 8)
        elif scene_type == "schematic":
            blur = max(4, blur - 10)
            alpha = min(184, alpha + 4)
        return color, max(40, min(182, alpha)), blur

    def _apply_soft_mask(
        self,
        base: Image.Image,
        mask: Image.Image,
        color: Tuple[int, int, int],
        alpha: int,
        blur_radius: int,
    ) -> None:
        if blur_radius > 0:
            soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        else:
            soft_mask = mask
        soft_alpha = soft_mask.point(lambda px: min(255, int(px * alpha / 255)))
        soft_layer = Image.new("RGBA", base.size, (*color, 0))
        soft_layer.putalpha(soft_alpha)
        base.alpha_composite(soft_layer)

        inner_alpha = max(0, int(alpha * 0.42))
        if inner_alpha > 0:
            inner_mask = mask.point(lambda px: min(255, int(px * inner_alpha / 255)))
            inner_layer = Image.new("RGBA", base.size, (*color, 0))
            inner_layer.putalpha(inner_mask)
            base.alpha_composite(inner_layer)

    def _draw_control_connectors(
        self,
        base: Image.Image,
        scene: Dict[str, Any],
        control_palette: Dict[str, Tuple[int, int, int]],
    ) -> None:
        layout_options = scene.get("layout_options", {}) or {}
        scene_type = str(layout_options.get("scene_type") or "scene")
        if scene_type == "scene":
            return
        connector_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(connector_layer, "RGBA")
        objects_by_id = {item["id"]: item for item in scene.get("object_instances", []) if item.get("id")}
        alpha = 56 if scene_type == "process" else 72
        line_width = 8
        blur_radius = 5
        if scene_type == "process":
            alpha = 64
            line_width = 6
            blur_radius = 3
        elif scene_type == "schematic":
            alpha = 78
            line_width = 6
            blur_radius = 2
        line_color = (*control_palette["connector"], alpha)
        for connector in scene.get("connectors", []):
            if not connector.get("visible", True):
                continue
            from_obj = objects_by_id.get(connector.get("from_id"))
            to_obj = objects_by_id.get(connector.get("to_id"))
            if not from_obj or not to_obj:
                continue
            start, end = self._connector_points_for_scene(from_obj, to_obj)
            draw.line([start, end], fill=line_color, width=line_width)
        connector_layer = connector_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        base.alpha_composite(connector_layer)

    def _render_external_control_image(
        self,
        scene_spec: Dict[str, Any],
        palette: Dict[str, Tuple[int, int, int]],
    ) -> Image.Image:
        scene = normalize_scene_spec_v2(scene_spec)
        canvas_size = (scene["canvas_size"]["width"], scene["canvas_size"]["height"])
        control_palette = self._render_control_palette()
        base = Image.new("RGBA", canvas_size, (*control_palette["background"], 255))
        layout_options = scene.get("layout_options", {}) or {}
        scene_type = str(layout_options.get("scene_type") or "scene")

        background_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        background_draw = ImageDraw.Draw(background_layer, "RGBA")
        for layer in sorted(scene.get("background_layers", []), key=lambda item: item.get("z_index", 0)):
            fill = self._control_layer_color(layer, control_palette)
            layer_type = str(layer.get("type") or "")
            alpha = 98 if layer_type in {"ground", "road", "board", "water"} else 82
            radius = 28
            if scene_type == "process":
                if layer_type == "process_band":
                    alpha = 26
                else:
                    alpha = min(alpha, 34)
                radius = 18
            elif scene_type == "schematic":
                alpha = 18 if layer_type == "board" else 14
                radius = 12
            background_draw.rounded_rectangle(
                [layer["x"], layer["y"], layer["x"] + layer["width"], layer["y"] + layer["height"]],
                radius=radius,
                fill=(*fill, alpha),
                outline=None,
            )
        background_blur = 18
        if scene_type == "process":
            background_blur = 8
        elif scene_type == "schematic":
            background_blur = 4
        background_layer = background_layer.filter(ImageFilter.GaussianBlur(radius=background_blur))
        base.alpha_composite(background_layer)

        for obj in sorted(scene.get("object_instances", []), key=lambda item: item.get("z_index", 0)):
            if obj.get("visible", True) is False:
                continue
            x0, y0, x1, y1 = self._scene_bbox(obj)
            width = max(24, x1 - x0)
            height = max(24, y1 - y0)
            padding_scale = 0.16
            if scene_type == "process":
                padding_scale = 0.12
            elif scene_type == "schematic":
                padding_scale = 0.08
            padding = max(12, int(min(width, height) * padding_scale))
            box = (
                max(0, x0 - padding),
                max(0, y0 - padding),
                min(canvas_size[0], x1 + padding),
                min(canvas_size[1], y1 + padding),
            )
            mask = Image.new("L", canvas_size, 0)
            mask_draw = ImageDraw.Draw(mask)
            drew = self._draw_object_mask(mask_draw, box, obj)
            if not drew:
                self._draw_asset_symbol_mask(
                    mask_draw,
                    box,
                    str(obj.get("asset_key") or obj.get("silhouette_key") or "generic_object"),
                )
            mask = self._apply_region_overrides_to_mask(mask, obj)
            color, alpha, blur_radius = self._control_object_style(obj, control_palette, scene_type=scene_type)
            self._apply_soft_mask(base, mask, color, alpha, blur_radius)

            role = str(obj.get("role") or "")
            if role in {"subject", "focus", "core_subject"}:
                halo = Image.new("L", canvas_size, 0)
                halo_draw = ImageDraw.Draw(halo)
                halo_pad_x = max(18, int(width * 0.22))
                halo_pad_y = max(18, int(height * 0.22))
                halo_draw.ellipse(
                    [
                        max(0, x0 - halo_pad_x),
                        max(0, y0 - halo_pad_y),
                        min(canvas_size[0], x1 + halo_pad_x),
                        min(canvas_size[1], y1 + halo_pad_y),
                    ],
                    fill=255,
                )
                self._apply_soft_mask(base, halo, control_palette["subject"], min(74, alpha // 2), 26)

        self._draw_control_connectors(base, scene, control_palette)
        return base.convert("RGB")
    
    def _get_connection_points(self, from_pos: Dict, to_pos: Dict) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        start = (from_pos["x"] + from_pos["width"], from_pos["y"] + from_pos["height"] // 2)
        end = (to_pos["x"], to_pos["y"] + to_pos["height"] // 2)
        return start, end
    def _calculate_layout(self, path: 'MazePath', canvas_size: Tuple[int, int] = (1024, 768), layout_options: Dict[str, Any] | None = None) -> Dict:
        """
        基于推理路径的节点和关系自动计算布局
        :param path: Tri-Maze 推理路径
        :param canvas_size: 画布大小
        :return: 布局信息，包含每个节点的位置、大小、权重
        """
        layout_options = layout_options or {}
        width, height = canvas_size
        node_count = len(path.nodes)
        if node_count == 0:
            return {
                "canvas_size": canvas_size,
                "positions": [],
                "connections": [],
                "path": path,
                "layout_options": layout_options,
            }
            
        node_scale = max(0.4, float(layout_options.get("node_scale", 1.0)))
        spacing_scale = max(0.6, float(layout_options.get("spacing_scale", 1.0)))
        vertical_offset = int(layout_options.get("vertical_offset", 0))
        
        weights = []
        for i, node in enumerate(path.nodes):
            if i < len(path.edges):
                resistance = path.edges[i].resistance
            else:
                resistance = 0.0
            weight = 1.0 - resistance
            weights.append(max(0.3, weight))
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        positions = []
        margin_x = 100
        available_width = max(200, width - 2 * margin_x)
        step_count = max(1, node_count - 1)
        base_spacing = available_width / step_count if node_count > 1 else 0
        spacing = base_spacing * spacing_scale
        total_span = spacing * (node_count - 1)
        start_x = (width - total_span) / 2 if node_count > 1 else width / 2
        
        for i in range(node_count):
            node_width = max(72, min(240, 100 * normalized_weights[i] * 2 * node_scale))
            x = start_x + (spacing * i) - node_width / 2 if node_count > 1 else (width - node_width) / 2
            y = height // 2 - (node_width * 0.8) / 2 + vertical_offset
            positions.append({
                "x": int(x),
                "y": int(y),
                "width": int(node_width),
                "height": int(node_width * 0.8),
                "weight": normalized_weights[i],
                "node": path.nodes[i]
            })
        
        connections = []
        for i in range(len(path.edges)):
            from_pos = positions[i]
            to_pos = positions[i + 1]
            start_point, end_point = self._get_connection_points(from_pos, to_pos)
            connections.append({
                "from": from_pos,
                "to": to_pos,
                "edge": path.edges[i],
                "start_point": start_point,
                "end_point": end_point
            })
        
        return {
            "canvas_size": canvas_size,
            "positions": positions,
            "connections": connections,
            "path": path,
            "layout_options": layout_options,
        }
    
    def build_scene_spec(
        self,
        path: 'MazePath | None',
        canvas_size: Tuple[int, int] = (1024, 768),
        sketch_options: Dict[str, Any] | None = None,
        scene_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """将推理路径转换为可编辑的 SceneSpec。"""
        sketch_options = sketch_options or {}
        scene_context = scene_context or {}
        if scene_context:
            best_path_concepts = scene_context.get("best_path_concepts") or [
                node.concept for node in getattr(path, "nodes", []) if getattr(node, "concept", "")
            ]
            return compose_semantic_scene_spec(
                query=str(scene_context.get("query", "") or ""),
                understanding_result=scene_context.get("understanding_result"),
                extraction_result=scene_context.get("extraction_result"),
                answer_bundle=scene_context.get("answer_bundle"),
                best_path_concepts=best_path_concepts,
                canvas_size=canvas_size,
                sketch_options=sketch_options,
            )

        if path is None:
            return normalize_scene_spec_v2(
                {
                    "version": 2,
                    "canvas_size": {"width": canvas_size[0], "height": canvas_size[1]},
                    "layout_options": {
                        "scene_type": "scene",
                        "composition_mode": "scene",
                        "sketch_style": sketch_options.get("sketch_style", "scribble_line"),
                        "show_grid": bool(sketch_options.get("show_grid", True)),
                        "show_labels": bool(sketch_options.get("show_labels", False)),
                        "show_guides": bool(sketch_options.get("show_guides", False)),
                    },
                    "background_layers": [],
                    "object_instances": [],
                    "attachments": [],
                    "connectors": [],
                    "render_hints": {},
                    "concept_order": [],
                },
                sketch_options,
            )

        layout = self._calculate_layout(path, canvas_size, sketch_options)
        positions = layout["positions"]
        nodes = []
        relations = []
        for index, pos in enumerate(positions):
            node_id = f"node_{index + 1}"
            nodes.append({
                "id": node_id,
                "concept": pos["node"].concept,
                "x": pos["x"],
                "y": pos["y"],
                "width": pos["width"],
                "height": pos["height"],
                "weight": round(pos["weight"], 4),
                "role": "core" if pos["weight"] >= 0.2 else "secondary",
            })
        for index, edge in enumerate(path.edges):
            relations.append({
                "id": f"edge_{index + 1}",
                "from_id": nodes[index]["id"],
                "to_id": nodes[index + 1]["id"],
                "from": nodes[index]["concept"],
                "to": nodes[index + 1]["concept"],
                "relation": edge.relation,
                "resistance": edge.resistance,
            })
        legacy_scene = {
            "canvas_size": {"width": canvas_size[0], "height": canvas_size[1]},
            "layout_options": {
                "node_scale": float(sketch_options.get("node_scale", 1.0)),
                "spacing_scale": float(sketch_options.get("spacing_scale", 1.0)),
                "sketch_style": sketch_options.get("sketch_style", "line_art"),
                "show_grid": bool(sketch_options.get("show_grid", True)),
                "show_labels": bool(sketch_options.get("show_labels", True)),
                "show_guides": bool(sketch_options.get("show_guides", True)),
            },
            "concept_order": [node["concept"] for node in nodes],
            "nodes": nodes,
            "relations": relations,
        }
        return normalize_scene_spec_v2(legacy_scene, sketch_options)
    
    def _normalize_scene_spec(self, scene_spec: Dict[str, Any], sketch_options: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return normalize_scene_spec_v2(scene_spec, sketch_options)
    
    def _scene_node_shape(self, concept: str) -> str:
        text = str(concept or "")
        if any(keyword in text for keyword in {"圆形", "LED", "细胞"}):
            return "ellipse"
        if any(keyword in text for keyword in {"三角形", "箭头", "二极管"}):
            return "triangle"
        return "rect"
    
    def _render_scene_spec_relation(self, draw: ImageDraw, scene_spec: Dict[str, Any], relation: Dict[str, Any], palette: Dict[str, Tuple[int, int, int]], show_labels: bool = True):
        nodes = {node["id"]: node for node in scene_spec.get("nodes", [])}
        from_node = nodes.get(relation.get("from_id"))
        to_node = nodes.get(relation.get("to_id"))
        if not from_node or not to_node:
            return
        start, end = self._get_connection_points(from_node, to_node)
        relation_text = str(relation.get("relation", "连接"))
        if relation_text == "并联":
            draw.line([start, (start[0], start[1] - 28), (end[0], end[1] - 28), end], fill=palette["line"], width=2)
            draw.line([start, (start[0], start[1] + 28), (end[0], end[1] + 28), end], fill=palette["line"], width=2)
        elif relation_text == "包含":
            self._draw_dashed_rectangle(
                draw,
                (
                    from_node["x"] - 6,
                    from_node["y"] - 6,
                    from_node["x"] + from_node["width"] + 6,
                    from_node["y"] + from_node["height"] + 6,
                ),
                palette["line"],
                width=1,
            )
        elif relation_text in {"控制", "产生", "导致", "指向", "驱动"}:
            self._draw_arrow_line(draw, start, end, fill=palette["accent"], width=2)
        else:
            draw.line([start, end], fill=palette["line"], width=2)
        if show_labels:
            mid_x = int((start[0] + end[0]) / 2)
            mid_y = int((start[1] + end[1]) / 2 - 18)
            draw.text((mid_x, mid_y), relation_text, fill=palette["text"], font=self.font)
    
    def render_scene_spec_preview(self, scene_spec: Dict[str, Any], sketch_options: Dict[str, Any] | None = None, title: str | None = None) -> Dict[str, Any]:
        """根据 SceneSpec 渲染语义草图与低清预演。"""
        scene = self._normalize_scene_spec(scene_spec, sketch_options)
        layout_options = scene.get("layout_options", {})
        variant_payload = scene_shape_variant_payload(scene)
        backend_status = summarize_scene_backend(scene)
        sketch_style = layout_options.get("sketch_style", "line_art")
        generator_style = layout_options.get("generator_style") or sketch_style
        scene_generation_backend = str(layout_options.get("scene_generation_backend") or "unified_scene_v3")
        concept_order = scene.get("concept_order", []) or [
            obj.get("concept", "") for obj in scene.get("object_instances", []) if obj.get("concept")
        ]
        title = title or (
            f"Tri-Maze 语义草图 · {' · '.join(concept_order[:4])}" if concept_order else "Tri-Maze 语义草图"
        )
        palette = self._build_preview_palette(sketch_style)
        low_palette = self._build_low_preview_palette(sketch_style)
        rich_palette = self._build_rich_preview_palette(str(generator_style))
        base_sketch_image = self._render_semantic_scene_image(
            scene,
            palette,
            title,
            filled=False,
            include_title=False,
            view_mode="rough",
        )
        structural_sketch_image, annotation_bundle = self.whole_scene_sketch_generator.render_scene(
            scene,
            palette,
            title,
            annotated=False,
            show_regions=False,
            include_title=False,
        )
        annotated_sketch_image, _ = self.whole_scene_sketch_generator.render_scene(
            scene,
            palette,
            title,
            annotated=True,
            show_regions=False,
            include_title=False,
        )
        region_overlay_image, _ = self.whole_scene_sketch_generator.render_scene(
            scene,
            palette,
            title,
            annotated=True,
            show_regions=True,
            include_title=False,
        )
        low_preview_image = self._render_semantic_scene_image(
            scene,
            low_palette,
            "",
            filled=True,
            include_title=False,
            force_show_labels=False,
            force_show_grid=False,
            force_show_guides=False,
            view_mode="structure",
        )
        render_control_image = self._render_external_control_image(scene, low_palette)
        sd_upstream_control_image = self._render_sd_upstream_control_image(
            scene,
            structural_sketch_image=structural_sketch_image,
        )
        hit_map_image, hit_map_legend = self._render_hit_map_image(scene)
        scene.setdefault("render_hints", {})
        scene["render_hints"]["layout_runtime"] = json.loads(json.dumps(
            scene["render_hints"].get("layout_runtime")
            or layout_options.get("layout_model_status")
            or {},
            ensure_ascii=False,
        ))
        scene["render_hints"]["annotation_bundle"] = annotation_bundle
        scene["render_hints"]["structural_generator"] = {
            "id": str((annotation_bundle or {}).get("generator_id") or "whole_scene_structural_v2"),
            "style_variant": str(generator_style),
            "synchronized_annotations": True,
            "second_stage": (annotation_bundle or {}).get("second_stage_generator", {}),
        }
        scene["layout_options"]["generator_style"] = str(generator_style)
        scene["layout_options"]["scene_generation_backend"] = scene_generation_backend

        sketch_hash = hash(json.dumps({"scene": scene, "title": title}, ensure_ascii=False, sort_keys=True))
        image_path = os.path.join(self.output_dir, f"structural_sketch_{sketch_hash}.png")
        base_sketch_path = os.path.join(self.output_dir, f"base_sketch_{sketch_hash}.png")
        annotated_sketch_path = os.path.join(self.output_dir, f"annotated_sketch_{sketch_hash}.png")
        region_overlay_path = os.path.join(self.output_dir, f"region_overlay_{sketch_hash}.png")
        low_preview_path = os.path.join(self.output_dir, f"low_preview_{sketch_hash}.png")
        rich_preview_path = os.path.join(self.output_dir, f"rich_preview_{sketch_hash}.png")
        render_control_path = os.path.join(self.output_dir, f"render_control_{sketch_hash}.png")
        sd_upstream_control_path = os.path.join(self.output_dir, f"sd_upstream_control_{sketch_hash}.png")
        hit_map_path = os.path.join(self.output_dir, f"hit_map_{sketch_hash}.png")
        annotation_bundle_path = os.path.join(self.output_dir, f"scene_annotation_{sketch_hash}.json")
        if scene_generation_backend == "legacy_object_library_v2":
            rich_preview_image, rich_preview_meta = self.whole_scene_sketch_generator.render_scene(
                scene,
                rich_palette,
                title,
                annotated=False,
                show_regions=False,
                include_title=False,
                view_mode="rich_preview",
            )
        else:
            rich_preview_image, rich_preview_meta = self.unified_scene_generator.render_scene(
                scene,
                rich_palette,
                title,
            )
        scene["render_hints"]["scene_preview_generation"] = rich_preview_meta
        base_sketch_image.save(base_sketch_path)
        structural_sketch_image.save(image_path)
        annotated_sketch_image.save(annotated_sketch_path)
        region_overlay_image.save(region_overlay_path)
        low_preview_image.save(low_preview_path)
        rich_preview_image.save(rich_preview_path)
        render_control_image.save(render_control_path)
        sd_upstream_control_image.save(sd_upstream_control_path)
        hit_map_image.save(hit_map_path)
        with open(annotation_bundle_path, "w", encoding="utf-8") as handle:
            json.dump(annotation_bundle, handle, ensure_ascii=False, indent=2)
        sketch_bundle = {
            "base_sketch": base_sketch_path,
            "structural_sketch": image_path,
            "annotated_sketch": annotated_sketch_path,
            "region_overlay": region_overlay_path,
            "low_preview": low_preview_path,
            "rich_preview": rich_preview_path,
            "sd_upstream_control": sd_upstream_control_path,
            "hit_map": hit_map_path,
            "annotation_bundle": annotation_bundle_path,
            "scene_preview_generation": rich_preview_meta,
        }
        scene["render_hints"]["sd_upstream_control_path"] = sd_upstream_control_path
        return {
            "success": True,
            "type": "control_preview",
            "image_path": image_path,
            "base_sketch_path": base_sketch_path,
            "annotated_sketch_path": annotated_sketch_path,
            "region_overlay_path": region_overlay_path,
            "low_preview_path": low_preview_path,
            "rich_preview_path": rich_preview_path,
            "render_control_path": render_control_path,
            "sd_upstream_control_path": sd_upstream_control_path,
            "hit_map_path": hit_map_path,
            "annotation_bundle_path": annotation_bundle_path,
            "save_path": image_path,
            "scene_spec": scene,
            "scene_spec_version": scene.get("version", 2),
            "composition_mode": layout_options.get("composition_mode", layout_options.get("scene_type", "scene")),
            "editor_palette": build_editor_asset_library() if "build_editor_asset_library" in globals() else [],
            "sketch_backend_status": backend_status,
            "available_shape_variants": variant_payload.get("available_shape_variants", {}),
            "shape_variant_id": variant_payload.get("shape_variant_id", {}),
            "shape_recipe_source": variant_payload.get("shape_recipe_source", {}),
            "render_representation": variant_payload.get("render_representation", {}),
            "stroke_variant_id": variant_payload.get("stroke_variant_id", {}),
            "stroke_payload_source": variant_payload.get("stroke_payload_source", {}),
            "annotation_bundle": annotation_bundle,
            "scene_preview_generation": rich_preview_meta,
            "overlay_defaults": {
                "show_labels": bool(layout_options.get("show_labels", False)),
                "show_grid": bool(layout_options.get("show_grid", True)),
                "show_guides": bool(layout_options.get("show_guides", False)),
                "view_mode": layout_options.get("sketch_view_mode", "structure"),
                "annotation_level": layout_options.get("annotation_level", "light"),
            },
            "sketch_bundle": sketch_bundle,
            "hit_map_legend": hit_map_legend,
            "description": "Tri-Maze 整图结构草图（同步标注）、人类可读整图预览与低清预演，可作为后续图片渲染的统一结构约束",
            "sketch_options": scene["layout_options"],
        }
    
    async def generate_control_preview(
        self,
        path: 'MazePath | None',
        canvas_size: Tuple[int, int] = (1024, 768),
        sketch_options: Dict[str, Any] | None = None,
        scene_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """生成控制草图，不依赖外部图片 API。"""
        scene_spec = self.build_scene_spec(
            path,
            canvas_size=canvas_size,
            sketch_options=sketch_options,
            scene_context=scene_context,
        )
        hints = scene_spec.get("render_hints", {}) if isinstance(scene_spec, dict) else {}
        title_core = hints.get("scene_summary") or " · ".join(scene_spec.get("concept_order", [])[:4])
        title = f"Tri-Maze 语义草图 · {title_core}" if title_core else "Tri-Maze 语义草图"
        return self.render_scene_spec_preview(scene_spec, sketch_options=sketch_options, title=title)

    def _render_resistor(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染电阻"""
        color = color or self.color_map["棕色"]
        # 主体
        draw.rectangle([x, y, x + width, y + height], fill=color, outline=self.color_map["黑色"], width=2)
        # 引脚
        draw.line([(x - 20, y + height//2), (x, y + height//2)], fill=self.color_map["灰色"], width=3)
        draw.line([(x + width, y + height//2), (x + width + 20, y + height//2)], fill=self.color_map["灰色"], width=3)
        # 色环
        ring_width = width // 5
        for i in range(4):
            ring_color = [self.color_map["棕色"], self.color_map["黑色"], self.color_map["红色"], self.color_map["金色"]][i]
            draw.rectangle([x + i*ring_width, y, x + (i+1)*ring_width, y + height], fill=ring_color)
    
    def _render_led(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染LED"""
        color = color or self.color_map["红色"]
        # 主体
        draw.ellipse([x, y, x + width, y + height], fill=color, outline=self.color_map["黑色"], width=2)
        # 正极
        draw.line([(x + width//2, y + height), (x + width//2, y + height + 20)], fill=self.color_map["灰色"], width=3)
        # 负极
        draw.line([(x + width//4, y + height), (x + width//4, y + height + 15)], fill=self.color_map["灰色"], width=3)
        # 发光效果
        glow_color = (min(255, color[0] + 100), min(255, color[1] + 100), min(255, color[2] + 100))
        draw.ellipse([x-10, y-10, x + width + 10, y + height + 10], outline=glow_color, width=3)
    
    def _render_arduino(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染Arduino开发板"""
        color = color or self.color_map["蓝色"]
        # 主板
        draw.rectangle([x, y, x + width, y + height], fill=color, outline=self.color_map["黑色"], width=2)
        # USB口
        draw.rectangle([x + 10, y + height//2 - 10, x + 30, y + height//2 + 10], fill=self.color_map["灰色"], outline=self.color_map["黑色"])
        # 引脚
        for i in range(10):
            draw.rectangle([x + width - 5, y + 10 + i*15, x + width + 5, y + 20 + i*15], fill=self.color_map["金色"])
        # 文字
        draw.text((x + width//2 - 30, y + height//2 - 10), "Arduino", fill=self.color_map["白色"], font=self.font)
    
    def _render_battery(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染电源/电池"""
        color = color or self.color_map["黑色"]
        # 主体
        draw.rectangle([x, y, x + width, y + height], fill=self.color_map["灰色"], outline=color, width=2)
        # 正极
        draw.rectangle([x + width, y + height//3, x + width + 15, y + height*2//3], fill=self.color_map["红色"])
        # 负极
        draw.rectangle([x -15, y + height//3, x, y + height*2//3], fill=self.color_map["黑色"])
        # 正负极符号
        draw.text((x + width + 20, y + height//2 - 10), "+", fill=self.color_map["红色"], font=self.font)
        draw.text((x - 35, y + height//2 - 10), "-", fill=self.color_map["黑色"], font=self.font)
    
    def _render_ground(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染接地符号"""
        color = color or self.color_map["黑色"]
        draw.line([(x + width//2, y), (x + width//2, y + height//2)], fill=color, width=3)
        draw.line([(x, y + height//2), (x + width, y + height//2)], fill=color, width=3)
        draw.line([(x + width*0.25, y + height*0.7), (x + width*0.75, y + height*0.7)], fill=color, width=3)
        draw.line([(x + width*0.4, y + height*0.9), (x + width*0.6, y + height*0.9)], fill=color, width=3)
    
    def _render_capacitor(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染电容"""
        color = color or self.color_map["灰色"]
        # 两个极板
        draw.rectangle([x, y, x + width//3, y + height], fill=color, outline=self.color_map["黑色"], width=2)
        draw.rectangle([x + width*2//3, y, x + width, y + height], fill=color, outline=self.color_map["黑色"], width=2)
        # 引脚
        draw.line([(x + width//6, y - 20), (x + width//6, y)], fill=self.color_map["灰色"], width=3)
        draw.line([(x + width*5//6, y - 20), (x + width*5//6, y)], fill=self.color_map["灰色"], width=3)
        draw.line([(x + width//6, y + height), (x + width//6, y + height + 20)], fill=self.color_map["灰色"], width=3)
        draw.line([(x + width*5//6, y + height), (x + width*5//6, y + height + 20)], fill=self.color_map["灰色"], width=3)
    
    def _render_diode(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染二极管"""
        color = color or self.color_map["黑色"]
        # 三角形
        points = [
            (x, y + height//2),
            (x + width*0.7, y),
            (x + width*0.7, y + height)
        ]
        draw.polygon(points, fill=self.color_map["灰色"], outline=color, width=2)
        # 竖线
        draw.line([(x + width*0.7, y), (x + width*0.7, y + height)], fill=color, width=3)
        # 引脚
        draw.line([(x - 20, y + height//2), (x, y + height//2)], fill=self.color_map["灰色"], width=3)
        draw.line([(x + width, y + height//2), (x + width + 20, y + height//2)], fill=self.color_map["灰色"], width=3)
    
    def _render_transistor(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染三极管"""
        color = color or self.color_map["黑色"]
        # 主体
        self._draw_circle(draw, (x + width//2, y + height//2), width//2, fill=self.color_map["灰色"], outline=color, width=2)
        # 三个引脚
        draw.line([(x + width//2, y - 20), (x + width//2, y)], fill=self.color_map["灰色"], width=3)  # 基极
        draw.line([(x - 20, y + height*0.8), (x, y + height*0.8)], fill=self.color_map["灰色"], width=3)  # 发射极
        draw.line([(x + width, y + height*0.2), (x + width + 20, y + height*0.2)], fill=self.color_map["灰色"], width=3)  # 集电极
    
    def _render_rectangle(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染矩形"""
        color = color or self.color_map["蓝色"]
        draw.rectangle([x, y, x + width, y + height], fill=color, outline=self.color_map["黑色"], width=2)
    
    def _render_circle(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染圆形"""
        color = color or self.color_map["红色"]
        draw.ellipse([x, y, x + width, y + height], fill=color, outline=self.color_map["黑色"], width=2)
    
    def _render_triangle(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染三角形"""
        color = color or self.color_map["黄色"]
        points = [(x + width//2, y), (x, y + height), (x + width, y + height)]
        draw.polygon(points, fill=color, outline=self.color_map["黑色"], width=2)
    
    def _render_arrow(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染箭头"""
        color = color or self.color_map["黑色"]
        # 箭身
        draw.line([(x, y + height//2), (x + width*0.7, y + height//2)], fill=color, width=3)
        # 箭头
        points = [
            (x + width*0.7, y),
            (x + width, y + height//2),
            (x + width*0.7, y + height)
        ]
        draw.polygon(points, fill=color)
    
    def _render_cat(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染猫"""
        color = color or self.color_map["橙色"]
        # 身体
        draw.ellipse([x + width*0.2, y + height*0.3, x + width*0.8, y + height*0.9], fill=color, outline=self.color_map["黑色"], width=2)
        # 头
        draw.ellipse([x + width*0.3, y, x + width*0.7, y + height*0.4], fill=color, outline=self.color_map["黑色"], width=2)
        # 耳朵
        draw.polygon([(x + width*0.3, y), (x + width*0.4, y + height*0.1), (x + width*0.2, y + height*0.2)], fill=color, outline=self.color_map["黑色"])
        draw.polygon([(x + width*0.7, y), (x + width*0.6, y + height*0.1), (x + width*0.8, y + height*0.2)], fill=color, outline=self.color_map["黑色"])
        # 眼睛
        self._draw_circle(draw, (x + width*0.4, y + height*0.2), height*0.05, fill=self.color_map["黄色"])
        self._draw_circle(draw, (x + width*0.6, y + height*0.2), height*0.05, fill=self.color_map["黄色"])
        self._draw_circle(draw, (x + width*0.4, y + height*0.2), height*0.02, fill=self.color_map["黑色"])
        self._draw_circle(draw, (x + width*0.6, y + height*0.2), height*0.02, fill=self.color_map["黑色"])
        # 胡子
        draw.line([(x + width*0.2, y + height*0.25), (x + width*0.3, y + height*0.27)], fill=self.color_map["黑色"], width=1)
        draw.line([(x + width*0.2, y + height*0.3), (x + width*0.3, y + height*0.3)], fill=self.color_map["黑色"], width=1)
        draw.line([(x + width*0.7, y + height*0.27), (x + width*0.8, y + height*0.25)], fill=self.color_map["黑色"], width=1)
        draw.line([(x + width*0.7, y + height*0.3), (x + width*0.8, y + height*0.3)], fill=self.color_map["黑色"], width=1)
    
    def _render_tree(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染树"""
        # 树干
        draw.rectangle([x + width*0.4, y + height*0.6, x + width*0.6, y + height], fill=self.color_map["棕色"], outline=self.color_map["黑色"])
        # 树叶
        self._draw_circle(draw, (x + width//2, y + height*0.2), width*0.3, fill=self.color_map["绿色"])
        self._draw_circle(draw, (x + width*0.3, y + height*0.4), width*0.25, fill=self.color_map["绿色"])
        self._draw_circle(draw, (x + width*0.7, y + height*0.4), width*0.25, fill=self.color_map["绿色"])
    
    def _render_house(self, draw: ImageDraw, x: int, y: int, width: int, height: int, color: Tuple = None):
        """原生渲染房子"""
        # 墙
        draw.rectangle([x + width*0.2, y + height*0.4, x + width*0.8, y + height], fill=self.color_map["黄色"], outline=self.color_map["黑色"], width=2)
        # 屋顶
        draw.polygon([(x, y + height*0.4), (x + width//2, y), (x + width, y + height*0.4)], fill=self.color_map["红色"], outline=self.color_map["黑色"], width=2)
        # 门
        draw.rectangle([x + width*0.45, y + height*0.7, x + width*0.55, y + height], fill=self.color_map["棕色"], outline=self.color_map["黑色"])
        # 窗户
        draw.rectangle([x + width*0.3, y + height*0.5, x + width*0.4, y + height*0.6], fill=self.color_map["蓝色"], outline=self.color_map["黑色"])
        draw.rectangle([x + width*0.6, y + height*0.5, x + width*0.7, y + height*0.6], fill=self.color_map["蓝色"], outline=self.color_map["黑色"])
    
    def _layout_horizontal_series(self, draw: ImageDraw, from_pos: Dict, to_pos: Dict, edge):
        """水平串联布局"""
        start, end = self._get_connection_points(from_pos, to_pos)
        draw.line([start, end], fill=self.color_map["黑色"], width=2)
        mid_x = (start[0] + end[0]) // 2
        mid_y = (start[1] + end[1]) // 2 - 30
        draw.text((mid_x, mid_y), edge.relation, fill=self.color_map["黑色"], font=self.font)
    
    def _layout_horizontal_parallel(self, draw: ImageDraw, from_pos: Dict, to_pos: Dict, edge):
        """并联布局"""
        start, end = self._get_connection_points(from_pos, to_pos)
        draw.line([start, (start[0], start[1] - 30), (end[0], end[1] - 30), end], fill=self.color_map["黑色"], width=2)
        draw.line([start, (start[0], start[1] + 30), (end[0], end[1] + 30), end], fill=self.color_map["黑色"], width=2)
        mid_x = (start[0] + end[0]) // 2
        mid_y = min(start[1], end[1]) - 50
        draw.text((mid_x, mid_y), edge.relation, fill=self.color_map["黑色"], font=self.font)
    
    def _layout_connect(self, draw: ImageDraw, from_pos: Dict, to_pos: Dict, edge):
        """普通连接"""
        start, end = self._get_connection_points(from_pos, to_pos)
        draw.line([start, end], fill=self.color_map["黑色"], width=2)
    
    def _layout_top_to_bottom(self, draw: ImageDraw, from_pos: Dict, to_pos: Dict, edge):
        """从上到下布局（控制关系）"""
        start = (from_pos["x"] + from_pos["width"]//2, from_pos["y"] + from_pos["height"])
        end = (to_pos["x"] + to_pos["width"]//2, to_pos["y"])
        self._draw_arrow_line(draw, start, end, fill=self.color_map["黑色"], width=2)
        mid_x = (start[0] + end[0]) // 2
        mid_y = (start[1] + end[1]) // 2
        draw.text((mid_x, mid_y), edge.relation, fill=self.color_map["黑色"], font=self.font)
    
    def _layout_inside(self, draw: ImageDraw, from_pos: Dict, to_pos: Dict, edge):
        """包含关系：to在from内部"""
        to_pos["x"] = from_pos["x"] + from_pos["width"] * 0.2
        to_pos["y"] = from_pos["y"] + from_pos["height"] * 0.2
        to_pos["width"] = from_pos["width"] * 0.6
        to_pos["height"] = from_pos["height"] * 0.6
        self._draw_dashed_rectangle(
            draw,
            (
                from_pos["x"] - 5,
                from_pos["y"] - 5,
                from_pos["x"] + from_pos["width"] + 5,
                from_pos["y"] + from_pos["height"] + 5,
            ),
            outline=self.color_map["灰色"],
            width=1,
            dash_length=5,
        )
        draw.text((from_pos["x"], from_pos["y"] - 20), f"包含{to_pos['node'].concept}", fill=self.color_map["黑色"], font=self.font)
    
    def _layout_left_to_right(self, draw: ImageDraw, from_pos: Dict, to_pos: Dict, edge):
        """从左到右布局（产生/导致关系）"""
        start, end = self._get_connection_points(from_pos, to_pos)
        self._draw_arrow_line(draw, start, end, fill=self.color_map["黑色"], width=2)
        mid_x = (start[0] + end[0]) // 2
        mid_y = (start[1] + end[1]) // 2 - 20
        draw.text((mid_x, mid_y), edge.relation, fill=self.color_map["黑色"], font=self.font)
    async def generate_image(self, path: 'MazePath', canvas_size: Tuple[int, int] = (1024, 768)) -> Dict[str, Any]:
        """
        原生生成图片：完全基于推理路径，不需要任何外部API
        :param path: Tri-Maze推理路径
        :param canvas_size: 画布大小
        :return: 生成结果
        """
        logger.info(f"🎨 原生生成图片，推理路径: {path.get_concept_list()}")
        
        try:
            # 1. 计算布局
            layout = self._calculate_layout(path, canvas_size)
            positions = layout["positions"]
            connections = layout["connections"]
            
            # 2. 创建画布
            image = Image.new("RGB", canvas_size, self.color_map["白色"])
            draw = ImageDraw.Draw(image)
            
            # 3. 渲染连接关系
            for conn in connections:
                edge = conn["edge"]
                layout_func = self.relation_layout.get(edge.relation, self._layout_connect)
                layout_func(draw, conn["from"], conn["to"], edge)
            
            # 4. 渲染每个节点
            for pos in positions:
                node = pos["node"]
                renderer = self.concept_renderers.get(node.concept, self._render_rectangle)
                renderer(
                    draw, 
                    pos["x"], 
                    pos["y"], 
                    pos["width"], 
                    pos["height"]
                )
                # 标注节点名称
                draw.text(
                    (pos["x"], pos["y"] - 25), 
                    node.concept, 
                    fill=self.color_map["黑色"], 
                    font=self.font
                )
            
            # 5. 保存图片
            image_path = f"{self.output_dir}/native_image_{hash(str(path.get_concept_list()))}.png"
            image.save(image_path)
            
            logger.info(f"✅ 原生图片生成成功，保存到: {image_path}")
            
            return {
                "success": True,
                "type": "native_image",
                "image_path": image_path,
                "save_path": image_path,
                "layout": layout,
                "description": "完全基于Tri-Maze推理路径原生生成的图片，无外部API依赖",
                "concepts": path.get_concept_list()
            }
            
        except Exception as e:
            logger.error(f"❌ 原生图片生成失败: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def generate_video(self, path: 'MazePath', duration: int = 5, fps: int = 24) -> Dict[str, Any]:
        """
        原生生成视频：基于推理路径生成动画视频
        :param path: Tri-Maze推理路径
        :param duration: 视频时长（秒）
        :param fps: 帧率
        :return: 生成结果
        """
        logger.info(f"🎬 原生生成视频，推理路径: {path.get_concept_list()}")
        
        try:
            if cv2 is None:
                raise RuntimeError("OpenCV is not available in the current environment")
            # 计算布局
            layout = self._calculate_layout(path, (1024, 768))
            total_frames = duration * fps
            
            # 创建视频写入器
            video_path = f"{self.output_dir}/native_video_{hash(str(path.get_concept_list()))}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, fps, (1024, 768))
            
            # 生成每一帧
            for frame_idx in range(total_frames):
                # 进度：0.0到1.0
                progress = frame_idx / total_frames
                
                # 创建帧
                image = Image.new("RGB", (1024, 768), self.color_map["白色"])
                draw = ImageDraw.Draw(image)
                
                # 渲染已探索的节点和连接
                explored_count = int(len(layout["connections"]) * progress) + 1
                
                # 渲染连接
                for i, conn in enumerate(layout["connections"][:explored_count]):
                    edge = conn["edge"]
                    # 绘制逐步显示的动画
                    if i < explored_count - 1:
                        alpha = 1.0
                    else:
                        alpha = progress * len(layout["connections"]) - (explored_count - 1)
                    
                    start = conn["start_point"]
                    end = conn["end_point"]
                    current_end = (
                        int(start[0] + (end[0] - start[0]) * alpha),
                        int(start[1] + (end[1] - start[1]) * alpha)
                    )
                    draw.line([start, current_end], fill=self.color_map["黑色"], width=2)
                
                # 渲染节点
                for i, pos in enumerate(layout["positions"][:explored_count]):
                    node = pos["node"]
                    renderer = self.concept_renderers.get(node.concept, self._render_rectangle)
                    renderer(draw, pos["x"], pos["y"], pos["width"], pos["height"])
                    draw.text((pos["x"], pos["y"] - 25), node.concept, fill=self.color_map["黑色"], font=self.font)
                
                # 转换为OpenCV格式
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                video.write(frame)
            
            video.release()
            cv2.destroyAllWindows()
            
            logger.info(f"✅ 原生视频生成成功，保存到: {video_path}")
            
            return {
                "success": True,
                "type": "native_video",
                "video_path": video_path,
                "save_path": video_path,
                "duration": duration,
                "fps": fps,
                "description": "完全基于Tri-Maze推理路径原生生成的动画视频，无外部API依赖",
                "concepts": path.get_concept_list()
            }
            
        except Exception as e:
            logger.error(f"❌ 原生视频生成失败: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def generate(self, path: 'MazePath', generation_type: str = "image") -> Dict[str, Any]:
        """
        原生生成多模态产物
        :param path: Tri-Maze推理路径
        :param generation_type: 生成类型：image/video
        :return: 生成结果
        """
        if generation_type == "image":
            return await self.generate_image(path)
        elif generation_type == "video":
            return await self.generate_video(path)
        else:
            return {"success": False, "error": f"不支持的原生生成类型: {generation_type}"}
    
    def get_supported_types(self) -> List[str]:
        """获取支持的原生生成类型"""
        return ["image", "video"]








