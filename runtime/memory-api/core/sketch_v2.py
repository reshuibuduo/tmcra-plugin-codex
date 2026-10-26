from __future__ import annotations

import base64
import json
import mimetypes
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import requests
from loguru import logger
from PIL import Image

from .default_model_paths import resolve_default_sketch_lora_alias
from .sd_sketch_generator import SDSketchGenerator
from .semantic_scene_v2 import normalize_scene_spec_v2, summarize_scene_spec


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


class SketchV2Generator:
    def __init__(
        self,
        *,
        output_dir: str = "outputs",
        sd_api_url: str = "",
        image_api_url: str = "",
        image_api_key: str = "",
        image_api_model: str = "",
        image_api_size: str = "",
    ) -> None:
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.sd_generator = SDSketchGenerator(output_dir=output_dir, sd_api_url=sd_api_url)
        self.image_api_url = str(image_api_url or "").strip().rstrip("/")
        self.image_api_key = str(image_api_key or "").strip()
        self.image_api_model = str(image_api_model or "").strip()
        self.image_api_size = str(image_api_size or "").strip()

    @property
    def sd_available(self) -> bool:
        return self.sd_generator.available

    @property
    def image_api_available(self) -> bool:
        return self.image_api_url.startswith("http") and bool(self.image_api_key)

    def set_sd_api_url(self, api_url: str) -> None:
        self.sd_generator.set_sd_api_url(api_url)

    def set_image_api_url(self, api_url: str) -> None:
        self.image_api_url = str(api_url or "").strip().rstrip("/")

    def set_image_api_key(self, api_key: str) -> None:
        self.image_api_key = str(api_key or "").strip()

    def set_image_api_model(self, model_name: str) -> None:
        self.image_api_model = str(model_name or "").strip()

    def set_image_api_size(self, image_size: str) -> None:
        self.image_api_size = str(image_size or "").strip()

    def _clean_text(self, value: Any) -> str:
        text = str(value or "").replace("\n", " ").replace("\r", " ").replace("|", " ").strip()
        return " ".join(text.split()).strip(" ,;，；。")

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

    def _upstream_control_image_path(self, preview: Dict[str, Any]) -> str:
        sketch_bundle = preview.get("sketch_bundle", {}) if isinstance(preview.get("sketch_bundle"), dict) else {}
        render_bundle = preview.get("render_bundle", {}) if isinstance(preview.get("render_bundle"), dict) else {}
        visible_outputs = render_bundle.get("visible_outputs", {}) if isinstance(render_bundle.get("visible_outputs"), dict) else {}
        for candidate in (
            sketch_bundle.get("sd_upstream_control"),
            visible_outputs.get("sd_upstream_control_path"),
            preview.get("sd_upstream_control_path"),
            sketch_bundle.get("rich_preview"),
            visible_outputs.get("rich_preview_path"),
            preview.get("rich_preview_path"),
            sketch_bundle.get("structural_sketch"),
            visible_outputs.get("structural_sketch_path"),
            preview.get("low_preview_path"),
            preview.get("render_control_path"),
            preview.get("image_path"),
        ):
            value = str(candidate or "").strip()
            if value and os.path.exists(value):
                return value
        return ""

    def _scene_type(self, scene_spec: Dict[str, Any] | None) -> str:
        if not isinstance(scene_spec, dict):
            return "scene"
        layout_options = scene_spec.get("layout_options", {}) if isinstance(scene_spec.get("layout_options"), dict) else {}
        return str(layout_options.get("scene_type") or layout_options.get("composition_mode") or "scene").strip().lower() or "scene"

    def _prompt_anchor_name(self, item: Dict[str, Any]) -> str:
        asset_key = self._clean_text(item.get("asset_key") or "")
        concept = self._clean_text(item.get("concept") or item.get("label") or "")
        if asset_key and concept and concept.lower() != asset_key.lower():
            return f"{asset_key} ({concept})"
        return asset_key or concept

    def _scene_asset_allowlist(self, scene_type: str) -> set[str]:
        if scene_type == "process":
            return {"sun", "vapor", "cloud", "raindrop", "leaf", "energy_wave", "airplane", "cell"}
        if scene_type == "schematic":
            return {"battery", "resistor", "led", "switch", "capacitor", "diode", "board"}
        return {"person", "house", "home", "building", "tree", "leaf", "plant", "bush", "road", "car", "street_lamp", "cloud", "sun", "dog", "table", "chair", "desk_lamp"}

    def _noise_prompt_terms(self) -> tuple[str, ...]:
        return (
            "tri-maze",
            "scene sketch",
            "semantic sketch",
            "readable",
            "editable",
            "direct edit",
            "directly edit",
            "include",
            "contains",
            "containing",
            "后续",
            "可编辑",
            "直接编辑",
            "语义草图",
            "场景草图",
            "包含",
            "结构",
            "说明",
            "描述",
            "阶段一",
            "阶段二",
            "阶段三",
            "电路结构",
        )

    def _should_skip_object_hint(self, scene_type: str, item: Dict[str, Any]) -> bool:
        asset_key = self._clean_text(item.get("asset_key") or "").lower()
        concept = self._clean_text(item.get("concept") or item.get("label") or "")
        lowered_concept = concept.lower()
        if not asset_key and not concept:
            return True
        allowlist = self._scene_asset_allowlist(scene_type)
        if asset_key and allowlist and asset_key not in allowlist:
            return True
        if concept and any(token in concept or token in lowered_concept for token in self._noise_prompt_terms()):
            return True
        if scene_type == "schematic" and asset_key == "switch":
            return False
        if scene_type == "scene" and asset_key in {"road", "car", "street_lamp"} and any(token in concept for token in ("电路", "结构", "模块", "元件")):
            return True
        return False

    def _simple_anchor_name(self, item: Dict[str, Any]) -> str:
        asset_key = self._clean_text(item.get("asset_key") or "").lower()
        concept = self._clean_text(item.get("concept") or item.get("label") or "")
        if asset_key == "street_lamp":
            return "street lamp"
        if asset_key == "desk_lamp":
            return "desk lamp"
        if asset_key == "energy_wave":
            return "sunlight ray"
        if asset_key == "raindrop":
            return "rain"
        if asset_key == "vapor":
            return "evaporation vapor"
        if asset_key == "switch":
            return "switch"
        if asset_key == "board":
            return "circuit board"
        if asset_key == "module":
            if "开关" in concept:
                return "switch"
            if "传感" in concept:
                return "sensor module"
            return "module"
        return asset_key or concept.lower()

    def _sd_anchor_phrases(self, scene_plan: Dict[str, Any]) -> List[str]:
        scene_type = str(scene_plan.get("scene_type", "scene") or "scene")
        phrases: List[str] = []
        seen_names: set[str] = set()
        for item in scene_plan.get("object_hints") or []:
            if not isinstance(item, dict):
                continue
            name = self._simple_anchor_name(item)
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            position = self._clean_text(str(item.get("position") or "").replace("-", " "))
            depth_band = self._clean_text(item.get("depth_band") or "")
            phrase = f"one {name}"
            if scene_type == "scene" and name == "person":
                phrase = "one full-body person"
            if position and position not in {"middle center", "center"}:
                phrase += f" at {position}"
            if depth_band in {"foreground", "background"}:
                phrase += f" in {depth_band}"
            phrases.append(phrase)
        return phrases[:6]

    def _build_sd_prompt(self, scene_plan: Dict[str, Any], variation_index: int) -> str:
        scene_type = str(scene_plan.get("scene_type", "scene") or "scene")
        asset_keys = {
            str(item).strip().lower()
            for item in (scene_plan.get("asset_keys") or [])
            if str(item).strip()
        }
        anchors = self._sd_anchor_phrases(scene_plan)
        parts: List[str] = [
            "monochrome pencil line sketch",
            "plain white paper background",
            "wide landscape composition",
            "all requested subjects fully visible inside the frame",
            "clean readable contours",
            "light sketch shading only",
            "no text in image",
        ]
        if scene_type == "process":
            parts.extend(
                [
                    "single integrated process scene",
                    "natural spatial flow instead of boxed infographic panels",
                    "no title area, no panel borders, no ornamental frame",
                ]
            )
        elif scene_type == "schematic":
            parts.extend(
                [
                    "technical hand-drawn circuit sketch",
                    "clear component spacing on blank paper",
                    "simple connection structure",
                    "no blueprint title block, no product design sheet",
                ]
            )
        else:
            parts.extend(
                [
                    "outdoor whole-scene street view",
                    "natural scene composition",
                    "not a close-up crop",
                ]
            )
        if anchors:
            parts.append(", ".join(anchors))
        if scene_type == "scene":
            if {"house", "building"} & asset_keys:
                parts.append("show roofline, windows, facade, and door clearly")
            if "tree" in asset_keys:
                parts.append("show tree canopy and trunk clearly")
            if "person" in asset_keys:
                parts.append("show head, torso, arms, and legs clearly")
            if {"house", "tree", "person"} <= asset_keys:
                parts.append("show one full-body person standing near the house and tree, not omitted and not replaced by another object")
                parts.append("place the person in the open space between the tree and the house, clearly separated from both and not hidden behind the tree")
        elif scene_type == "process":
            if {"sun", "vapor", "cloud", "raindrop"} & asset_keys:
                parts.append("show evaporation rising, cloud forming, and rain falling in one readable scene")
                parts.append("educational water cycle illustration, sun heating water, vapor rising upward, cloud above, rain falling down, runoff flowing back across the ground")
            parts.append("avoid diagram boxes and avoid decorative poster layout")
        elif scene_type == "schematic":
            if {"battery", "resistor", "led"} & asset_keys:
                parts.append("show battery, resistor, led, and switch as simple readable components")
                parts.append("single-loop hand-drawn circuit on paper, visible wires connecting battery to switch, resistor, and led in order")
            parts.append("connected engineering sketch, not an industrial product concept page")
        variations = [
            "balanced composition and stronger subject readability",
            "clearer silhouettes and cleaner scene hierarchy",
            "less clutter and more stable object identity",
            "more open negative space and clearer separation between anchors",
        ]
        parts.append(variations[variation_index % len(variations)])
        return ", ".join(self._clean_text(part) for part in parts if self._clean_text(part))

    def _default_background_summary(self, scene_type: str, objects: List[Dict[str, str]]) -> str:
        anchors = ", ".join(
            item.get("prompt_name") or item.get("concept") or ""
            for item in objects[:4]
            if item.get("prompt_name") or item.get("concept")
        )
        if scene_type == "process":
            return "plain open background with natural process flow and no boxed panels"
        if scene_type == "schematic":
            return "blank paper with open negative space for readable components and wire connections"
        if anchors:
            return f"simple street-scene environment around {anchors}"
        return "simple readable scene environment with open sky and ground plane"

    def _editable_detail_summary(self, objects: List[Dict[str, str]]) -> str:
        asset_keys = {str(item.get("asset_key") or "").strip().lower() for item in objects}
        details: List[str] = []
        if {"house", "building"} & asset_keys:
            details.append("roofline, windows, and door stay clear and editable")
        if "person" in asset_keys:
            details.append("human silhouette, head, torso, and legs stay readable")
        if "tree" in asset_keys:
            details.append("tree canopy and trunk stay readable")
        return ", ".join(details[:3])

    def _prompt_title_text(self, title: str | None = None) -> str:
        cleaned = self._clean_text(title or "")
        if not cleaned:
            return ""
        return cleaned if len(cleaned) <= 48 else ""

    def _anchor_layout_summary(self, objects: List[Dict[str, str]]) -> str:
        anchors: List[str] = []
        for item in objects[:6]:
            anchor = self._clean_text(item.get("asset_key") or item.get("concept") or item.get("prompt_name") or "")
            position = self._clean_text(str(item.get("position") or "").replace("-", " "))
            depth = self._clean_text(item.get("depth_band") or "")
            if not anchor:
                continue
            phrase = f"one {anchor}"
            if position:
                phrase += f" at {position}"
            if depth:
                phrase += f" in {depth}"
            anchors.append(phrase)
        return "; ".join(anchors)

    def _object_hint_priority(self, item: Dict[str, Any]) -> Tuple[int, int, float, float]:
        role = str(item.get("role") or "").strip().lower()
        depth_band = str(item.get("depth_band") or "").strip().lower()
        asset_key = str(item.get("asset_key") or "").strip().lower()
        role_priority = {
            "subject": 0,
            "focus": 0,
            "core_subject": 0,
            "primary": 1,
            "support": 2,
            "detail": 3,
        }.get(role, 4)
        depth_priority = {
            "midground": 0,
            "foreground": 1,
            "background": 2,
        }.get(depth_band, 3)
        human_bonus = 0 if asset_key in {"person", "human", "figure", "character"} else 1
        position_bonus = abs(float(item.get("_x_center", 0.5) or 0.5) - 0.5) + abs(float(item.get("_y_center", 0.5) or 0.5) - 0.5) * 0.6
        return (role_priority, human_bonus + depth_priority * 2, position_bonus, -float(item.get("_area", 0.0) or 0.0))

    def _anchor_presence_summary(self, scene_plan: Dict[str, Any]) -> str:
        objects = list(scene_plan.get("object_hints") or [])
        if not objects:
            return ""
        must_keep: List[str] = []
        asset_keys = {
            str(item.get("asset_key") or "").strip().lower()
            for item in objects
            if str(item.get("asset_key") or "").strip()
        }
        if "person" in asset_keys:
            must_keep.append("Keep one clearly visible standing person; head, torso, arms, and legs must remain readable and must not disappear.")
        if "house" in asset_keys:
            must_keep.append("Keep one readable house with roofline, facade, windows, and door still visible.")
        if "tree" in asset_keys:
            must_keep.append("Keep one readable tree with both canopy and trunk visible.")
        return " ".join(must_keep[:3])

    def _object_hints(self, scene_spec: Dict[str, Any] | None) -> List[Dict[str, str]]:
        hints: List[Dict[str, str]] = []
        if not isinstance(scene_spec, dict):
            return hints
        scene_type = self._scene_type(scene_spec)
        canvas = scene_spec.get("canvas_size", {}) if isinstance(scene_spec.get("canvas_size"), dict) else {}
        width = max(1.0, float(canvas.get("width", 1024) or 1024))
        height = max(1.0, float(canvas.get("height", 768) or 768))
        for item in scene_spec.get("object_instances", []) or []:
            if not isinstance(item, dict):
                continue
            if self._should_skip_object_hint(scene_type, item):
                continue
            asset_key = self._clean_text(item.get("asset_key") or "")
            concept = self._clean_text(item.get("concept") or item.get("label") or asset_key or "")
            prompt_name = self._prompt_anchor_name(
                {
                    "asset_key": asset_key,
                    "concept": concept,
                    "label": self._clean_text(item.get("label") or ""),
                }
            )
            if not prompt_name:
                continue
            x_center = (float(item.get("x", 0) or 0) + float(item.get("width", 0) or 0) / 2.0) / width
            y_center = (float(item.get("y", 0) or 0) + float(item.get("height", 0) or 0) / 2.0) / height
            if x_center <= 0.28:
                horizontal = "left"
            elif x_center >= 0.72:
                horizontal = "right"
            else:
                horizontal = "center"
            if y_center <= 0.34:
                vertical = "upper"
            elif y_center >= 0.66:
                vertical = "lower"
            else:
                vertical = "middle"
            hints.append(
                {
                    "concept": concept,
                    "asset_key": asset_key,
                    "prompt_name": prompt_name,
                    "role": str(item.get("role", "") or ""),
                    "depth_band": str(item.get("depth_band", "") or ""),
                    "position": f"{vertical}-{horizontal}",
                    "_x_center": x_center,
                    "_y_center": y_center,
                    "_area": float(item.get("width", 0) or 0) * float(item.get("height", 0) or 0),
                    "size": "large"
                    if float(item.get("width", 0) or 0) * float(item.get("height", 0) or 0) >= width * height * 0.1
                    else "medium"
                    if float(item.get("width", 0) or 0) * float(item.get("height", 0) or 0) >= width * height * 0.035
                    else "small",
                }
            )
        hints.sort(key=self._object_hint_priority)
        return hints[:8]

    def build_scene_plan(
        self,
        scene_spec: Dict[str, Any] | None,
        sketch_options: Dict[str, Any] | None = None,
        *,
        title: str | None = None,
    ) -> Dict[str, Any]:
        sketch_options = sketch_options or {}
        scene_spec = scene_spec if isinstance(scene_spec, dict) else {}
        scene_type = self._scene_type(scene_spec)
        render_hints = scene_spec.get("render_hints", {}) if isinstance(scene_spec.get("render_hints"), dict) else {}
        objects = self._object_hints(scene_spec)
        subject_hints = [item.get("prompt_name") or item["concept"] for item in objects if item.get("role") in {"subject", "focus", "core_subject"}]
        if not subject_hints:
            subject_hints = [item.get("prompt_name") or item["concept"] for item in objects[:3]]
        backgrounds = []
        for item in scene_spec.get("background_layers", []) or []:
            if not isinstance(item, dict):
                continue
            label = self._clean_text(item.get("label") or item.get("type") or "")
            if label:
                backgrounds.append(label)
        composition_bits = [f'{item.get("prompt_name") or item["concept"]} {item["position"]}' for item in objects[:5]]
        depth_counts: Dict[str, int] = {}
        for item in objects:
            depth = str(item.get("depth_band") or "")
            if not depth:
                continue
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        depth_summary = ", ".join(f"{key}:{value}" for key, value in depth_counts.items()) or "foreground, midground, background separation"
        style_summary = ", ".join(
            part
            for part in [
                str(sketch_options.get("sketch_style", "scribble_line") or "scribble_line"),
                self._clean_text(sketch_options.get("style_hint") or ""),
            ]
            if part
        )
        if not style_summary:
            style_summary = "readable whole-scene sketch"
        negative_constraints = [
            "symbol collage",
            "node boxes",
            "arrow labels",
            "flat icon layout",
            "unreadable overlapping objects",
        ]
        if scene_type == "process":
            negative_constraints.extend(["ppt slide", "flowchart boxes", "mind map"])
        elif scene_type == "schematic":
            negative_constraints.extend(["pcb photo", "chip macro photo", "motherboard photo"])
        else:
            negative_constraints.extend(["sticker collage", "poster layout"])
        subject_summary_hint = self._clean_text(render_hints.get("subject_summary") or "")
        if len(objects) > 1:
            subject_summary_hint = ""
        scene_summary_hint = self._clean_text(render_hints.get("scene_summary") or "")
        cleaned_title = self._clean_text(title or "")
        if (
            scene_summary_hint == cleaned_title
            or (cleaned_title and scene_summary_hint and (scene_summary_hint in cleaned_title or cleaned_title in scene_summary_hint))
            or len(scene_summary_hint) > 96
        ):
            scene_summary_hint = ""
        if scene_summary_hint and any(token in scene_summary_hint for token in ("请", "生成", "不要", "方便", "草图", "后续", "可读")):
            scene_summary_hint = ""
        must_include_summary = ", ".join(
            item.get("prompt_name") or item.get("concept") or ""
            for item in objects[:6]
            if item.get("prompt_name") or item.get("concept")
        )
        asset_counts: Counter[str] = Counter(
            str(item.get("asset_key") or "").strip().lower()
            for item in objects
            if str(item.get("asset_key") or "").strip()
        )
        exact_elements_summary = ", ".join(
            f'{count} {asset_key}' if count > 1 else f'one {asset_key}'
            for asset_key, count in asset_counts.items()
        )
        anchor_layout_summary = self._anchor_layout_summary(objects)
        subject_summary = self._clean_text(
            subject_summary_hint
            or (must_include_summary if len(asset_counts) > 1 else ", ".join(subject_hints[:4]))
            or cleaned_title
            or "main scene subject"
        )
        background_summary = self._default_background_summary(scene_type, objects)
        if scene_type == "scene" and backgrounds:
            background_summary = ", ".join(backgrounds[:4])
        return {
            "scene_type": scene_type,
            "subject_summary": subject_summary,
            "background_summary": self._clean_text(scene_summary_hint or background_summary),
            "composition_summary": self._clean_text(", ".join(composition_bits) or "clear hierarchy, readable layout, coherent scene composition"),
            "depth_summary": depth_summary,
            "style_summary": style_summary,
            "negative_constraints": negative_constraints,
            "object_hints": objects,
            "must_include_summary": must_include_summary,
            "anchor_layout_summary": anchor_layout_summary,
            "editable_detail_summary": self._editable_detail_summary(objects),
            "asset_keys": list(asset_counts.keys()),
            "asset_counts": dict(asset_counts),
            "exact_elements_summary": exact_elements_summary,
        }

    def _prompt_prefix(self, scene_type: str) -> str:
        if scene_type == "process":
            return "Create a highly readable whole-scene process sketch with coherent stages and spatial flow."
        if scene_type == "schematic":
            return "Create a highly readable technical sketch with clear structure and clean spatial organization."
        return "Create a highly readable whole-scene sketch with natural composition and coherent silhouettes."

    def _variation_suffix(self, provider: str, index: int, scene_type: str) -> str:
        variations = {
            "sd": [
                "Favor clean silhouettes, stronger subject readability, and confident contour continuity.",
                "Favor expressive but readable line rhythm, better depth layering, and less symbol-like geometry.",
            ],
            "image_api": [
                "Favor clearer whole-scene readability, soft hand-drawn variation, and natural scene balance.",
                "Favor stronger visual storytelling, cleaner hierarchy, and less mechanical object repetition.",
            ],
        }
        bucket = variations.get(provider) or variations["image_api"]
        value = bucket[index % len(bucket)]
        if scene_type == "process":
            value += " Keep transitions stage-like without turning into diagram boxes."
        elif scene_type == "schematic":
            value += " Keep the technical structure readable without turning into PCB photography."
        return value

    def build_prompt(
        self,
        scene_plan: Dict[str, Any],
        *,
        provider: str,
        variation_index: int,
        title: str | None = None,
    ) -> str:
        if provider == "sd":
            return self._build_sd_prompt(scene_plan, variation_index)
        scene_type = str(scene_plan.get("scene_type", "scene") or "scene")
        prompt_title = self._prompt_title_text(title)
        parts = [
            self._prompt_prefix(scene_type),
            "Use the provided control image only as a soft layout prior for subject placement, layering, and scale.",
            "Do not copy geometric helper lines, annotation labels, arrows, boxes, colored masks, or icon-like symbols into the final sketch.",
            prompt_title,
            f'Subject: {self._clean_text(scene_plan.get("subject_summary") or "")}',
            f'Background: {self._clean_text(scene_plan.get("background_summary") or "")}',
            f'Composition: {self._clean_text(scene_plan.get("composition_summary") or "")}',
            f'Depth: {self._clean_text(scene_plan.get("depth_summary") or "")}',
            f'Style: {self._clean_text(scene_plan.get("style_summary") or "")}',
            self._variation_suffix(provider, variation_index, scene_type),
        ]
        must_include_summary = self._clean_text(scene_plan.get("must_include_summary") or "")
        if must_include_summary:
            parts.append(f"Must clearly include all anchors: {must_include_summary}.")
        exact_elements_summary = self._clean_text(scene_plan.get("exact_elements_summary") or "")
        if exact_elements_summary:
            parts.append(f"Exact visible elements only: {exact_elements_summary}. Do not add extra major objects or substitute object types.")
        anchor_layout_summary = self._clean_text(scene_plan.get("anchor_layout_summary") or "")
        if anchor_layout_summary:
            parts.append(f"Exact anchor layout: {anchor_layout_summary}.")
        editable_detail_summary = self._clean_text(scene_plan.get("editable_detail_summary") or "")
        if editable_detail_summary:
            parts.append(f"Keep editable details clear: {editable_detail_summary}.")
        anchor_presence_summary = self._clean_text(self._anchor_presence_summary(scene_plan))
        if anchor_presence_summary:
            parts.append(anchor_presence_summary)
        object_hint_text = ", ".join(
            f'{item.get("prompt_name") or item.get("concept", "")} {item.get("position", "")} {item.get("depth_band", "")}'.strip()
            for item in (scene_plan.get("object_hints") or [])[:6]
            if item.get("prompt_name") or item.get("concept")
        )
        if object_hint_text:
            parts.append(f"Object hints: {object_hint_text}.")
        return ", ".join(part for part in parts if part)

    def build_negative_prompt(self, scene_plan: Dict[str, Any], sketch_style: str) -> str:
        scene_type = str(scene_plan.get("scene_type", "scene") or "scene")
        asset_keys = {
            str(item).strip().lower()
            for item in (scene_plan.get("asset_keys") or [])
            if str(item).strip()
        }
        asset_counts = {
            str(key).strip().lower(): int(value)
            for key, value in dict(scene_plan.get("asset_counts") or {}).items()
            if str(key).strip()
        }
        negatives = [
            "photorealistic",
            "full color rendering",
            "text",
            "letters",
            "chinese characters",
            "caption",
            "title block",
            "labels",
            "arrows",
            "boxes",
            "symbol collage",
            "flat icon composition",
            "decorative border",
            "ornate frame",
            "calligraphy",
            "mechanical repeated geometry",
            "unreadable overlapping objects",
            "watermark",
            "messy composition",
            "blurry",
        ]
        negatives.extend(str(item) for item in (scene_plan.get("negative_constraints") or []))
        style_token = str(sketch_style or "").strip().lower()
        if style_token == "blueprint":
            negatives.extend(["dark paper", "blueprint UI overlay"])
        if scene_type == "process":
            negatives.extend(["ppt slide", "flowchart", "poster frame", "certificate border", "storybook frame", "mountain painting", "split panels", "comic panel", "forest landscape only", "empty field", "trees only", "plain countryside"])
        elif scene_type == "schematic":
            negatives.extend(["pcb photo", "motherboard", "chip macro", "car", "vehicle", "wheel", "industrial design sheet", "product concept sheet", "annotated blueprint", "pen", "marker", "stationery", "writing tool"])
        else:
            negatives.extend(["sphere", "orb", "balloon", "wire", "cable", "abstract sculpture", "close-up portrait", "interior room", "cropped subject", "single facade close-up"])
            if "car" not in asset_keys:
                negatives.extend(["car", "vehicle", "garage focus"])
        if "person" in asset_keys and not (asset_keys & {"dog", "bird"}):
            negatives.extend(["animal", "deer", "antlers", "horns", "multiple people", "crowd", "group portrait", "missing person", "person omitted", "person replaced by house", "person replaced by tree", "person hidden behind tree", "tiny distant person"])
        if "house" in asset_keys and "bird" not in asset_keys:
            negatives.extend(["bird", "wings", "winged object", "flying ornament", "facade omitted", "house replaced by person"])
            if asset_counts.get("house", 0) <= 1 and asset_counts.get("building", 0) <= 1:
                negatives.extend(["multiple houses", "pagoda", "temple", "pavilion", "gazebo"])
        if "tree" in asset_keys:
            negatives.extend(["tree replaced by person", "tree replaced by rock"])
            if asset_counts.get("tree", 0) <= 1:
                negatives.extend(["multiple trees", "forest", "grove", "woodland"])
        return ", ".join(self._clean_text(item) for item in negatives if self._clean_text(item))

    def _encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")

    def _encode_image_to_data_url(self, image_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/png"
        return f"data:{mime_type};base64,{self._encode_image_to_base64(image_path)}"

    def _fingerprint(self, provider: str, prompt: str, negative_prompt: str, control_image_path: str, variation_index: int) -> str:
        return json.dumps(
            {
                "provider": provider,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "control_image_path": control_image_path,
                "variation_index": variation_index,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _render_sd_candidate(
        self,
        *,
        provider: str,
        variation_index: int,
        prompt: str,
        negative_prompt: str,
        control_image_path: str,
        sketch_options: Dict[str, Any],
        control_kind: str = "default",
        scene_spec: Dict[str, Any] | None = None,
    ) -> str:
        if not self.sd_available:
            raise RuntimeError("SD sketch backend unavailable")
        if control_kind == "raw_anchor_control":
            denoising_values = [0.56, 0.62, 0.68, 0.72]
            cfg_values = [6.0, 6.4, 6.8, 7.2]
            step_values = [26, 28, 30, 34]
        elif control_kind == "fused_anchor_scene":
            denoising_values = [0.42, 0.48, 0.54, 0.58]
            cfg_values = [5.8, 6.2, 6.6, 7.0]
            step_values = [24, 26, 28, 30]
        else:
            denoising_values = [0.3, 0.36, 0.42, 0.48]
            cfg_values = [5.8, 6.4, 7.0, 7.6]
            step_values = [20, 22, 24, 28]
        output_name = f"sketch_v2_{provider}_{abs(hash(self._fingerprint(provider, prompt, negative_prompt, control_image_path, variation_index)))}"
        controlnet_bundle = self.sd_generator.build_controlnet_bundle(
            control_image_path=control_image_path,
            scene_spec=scene_spec,
            filename_prefix=output_name,
            purpose="sketch_candidate",
        )
        return self.sd_generator.render_img2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image_path=control_image_path,
            denoising_strength=float(sketch_options.get("sd_sketch_denoising", denoising_values[variation_index % len(denoising_values)])),
            steps=int(sketch_options.get("sd_sketch_steps", step_values[variation_index % len(step_values)])),
            cfg_scale=float(sketch_options.get("sd_sketch_cfg_scale", cfg_values[variation_index % len(cfg_values)])),
            sampler_name=str(sketch_options.get("sd_sketch_sampler_name", "DPM++ 2M Karras")),
            filename_prefix=output_name,
            controlnet_bundle=controlnet_bundle,
            **self._sd_lora_kwargs(sketch_options),
        )

    def _get_image_api_endpoint(self) -> str:
        if not self.image_api_url:
            return ""
        if self.image_api_url.endswith("/images/generations"):
            return self.image_api_url
        return f"{self.image_api_url}/images/generations"

    def _is_volc_ark_image_api(self) -> bool:
        url = self.image_api_url.lower()
        return any(keyword in url for keyword in ["ark.", "volces.com", "volcengine"])

    def _build_generic_image_payload(self, prompt: str, control_image_path: str, variation_index: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "response_format": "url",
        }
        if not self._is_volc_ark_image_api():
            payload["n"] = 1
        if self.image_api_model:
            payload["model"] = self.image_api_model
        if self.image_api_size:
            payload["size"] = self.image_api_size
        elif self._is_volc_ark_image_api():
            payload["size"] = "2K"
        if control_image_path:
            payload["image"] = self._encode_image_to_data_url(control_image_path)
        if self._is_volc_ark_image_api():
            payload["stream"] = False
            payload["watermark"] = False
            payload["sequential_image_generation"] = "disabled"
            payload["metadata"] = {"candidate_index": variation_index}
        return payload

    def _save_generated_image(self, response_payload: Dict[str, Any], provider: str, fingerprint: str) -> str:
        data = response_payload.get("data") or []
        if not data:
            raise RuntimeError("Image API returned no data")
        first_item = data[0] or {}
        output_path = os.path.join(self.output_dir, f"sketch_v2_{provider}_{abs(hash(fingerprint))}.png")
        b64_json = first_item.get("b64_json")
        if b64_json:
            with open(output_path, "wb") as handle:
                handle.write(base64.b64decode(b64_json))
            return output_path
        image_url = first_item.get("url")
        if not image_url:
            raise RuntimeError("Image API returned neither b64_json nor url")
        image_response = requests.get(image_url, timeout=180)
        image_response.raise_for_status()
        with open(output_path, "wb") as handle:
            handle.write(image_response.content)
        return output_path

    def _render_image_api_candidate(
        self,
        *,
        provider: str,
        variation_index: int,
        prompt: str,
        negative_prompt: str,
        control_image_path: str,
    ) -> str:
        if not self.image_api_available:
            raise RuntimeError("Image API sketch backend unavailable")
        payload = self._build_generic_image_payload(prompt, control_image_path, variation_index)
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        response = requests.post(
            self._get_image_api_endpoint(),
            json=payload,
            headers={
                "Authorization": f"Bearer {self.image_api_key}",
                "Content-Type": "application/json",
            },
            timeout=180,
        )
        response.raise_for_status()
        return self._save_generated_image(
            response.json(),
            provider,
            self._fingerprint(provider, prompt, negative_prompt, control_image_path, variation_index),
        )

    def _provider_plan(self) -> List[str]:
        return self._provider_plan_for_options()

    def _provider_plan_for_options(self, sketch_options: Dict[str, Any] | None = None) -> List[str]:
        sketch_options = sketch_options or {}
        if self._use_direct_scene_mode(sketch_options) and self.sd_available:
            return ["sd", "sd", "sd", "sd"]
        if self.sd_available and self.image_api_available:
            return ["sd", "sd", "image_api", "image_api"]
        if self.sd_available:
            return ["sd", "sd", "sd", "sd"]
        if self.image_api_available:
            return ["image_api", "image_api", "image_api", "image_api"]
        return []

    def _use_direct_scene_mode(self, sketch_options: Dict[str, Any] | None = None) -> bool:
        sketch_options = sketch_options or {}
        mode = str(sketch_options.get("sketch_v2_mode", "direct_sd") or "direct_sd").strip().lower()
        if mode in {"legacy", "legacy_control", "native_control"}:
            return False
        if bool(sketch_options.get("sketch_v2_use_legacy_control", False)):
            return False
        return True

    def _sd_lora_kwargs(self, sketch_options: Dict[str, Any] | None = None) -> Dict[str, Any]:
        sketch_options = sketch_options or {}
        use_lora = bool(sketch_options.get("sd_sketch_use_lora", False))
        lora_name = str(
            sketch_options.get("sd_sketch_lora_name")
            or sketch_options.get("sd_lora_name")
            or (resolve_default_sketch_lora_alias() if use_lora else "")
        ).strip()
        if not lora_name:
            return {}
        return {
            "lora_name": lora_name,
            "lora_strength_model": float(sketch_options.get("sd_sketch_lora_strength_model", 0.92)),
            "lora_strength_clip": float(sketch_options.get("sd_sketch_lora_strength_clip", 0.86)),
        }

    def _prefer_sd_upstream(self, sketch_options: Dict[str, Any] | None = None) -> bool:
        sketch_options = sketch_options or {}
        backend = str(sketch_options.get("sketch_backend", "native") or "native").strip().lower()
        return (
            backend == "sketch_v2"
            and self.sd_available
            and not self._use_direct_scene_mode(sketch_options)
            and not bool(sketch_options.get("disable_sd_upstream_pass", False))
        )

    def _direct_scene_control_path(self, prior_bundle: Dict[str, Any] | None = None) -> str:
        prior_bundle = prior_bundle if isinstance(prior_bundle, dict) else {}
        return str(
            prior_bundle.get("layout_control_path")
            or prior_bundle.get("base_plate_path")
            or prior_bundle.get("depth_control_path")
            or ""
        ).strip()

    def _clamp(self, value: Any, low: float, high: float) -> float:
        return max(low, min(high, float(value or 0.0)))

    def _direct_sd_variant_recipe(
        self,
        scene_type: str,
        variation_index: int,
        sketch_options: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_index = max(0, int(variation_index) % 4)
        recipe_order = {
            "scene": [3, 2, 1, 0],
            "process": [2, 3, 1, 0],
            "schematic": [3, 2, 1, 0],
        }.get(scene_type, [0, 1, 2, 3])
        index = recipe_order[base_index % len(recipe_order)]
        if scene_type in {"process", "schematic"}:
            base_denoising = float(sketch_options.get("sd_direct_denoising", 0.80 if scene_type == "process" else 0.78))
            base_cfg = float(sketch_options.get("sd_direct_cfg_scale", 6.4 if scene_type == "process" else 6.5))
            base_steps = int(sketch_options.get("sd_direct_steps", sketch_options.get("sd_sketch_steps", 36)))
            denoising_offsets = [0.0, 0.06, 0.12, 0.18]
            cfg_offsets = [0.0, 0.05, 0.15, 0.30]
            step_offsets = [0, 0, 2, 4]
            layout_scales = [1.00, 0.90, 0.78, 0.66]
            depth_scales = [1.00, 0.92, 0.80, 0.68]
            structure_scales = [1.00, 0.96, 0.90, 0.84]
            layout_end_scales = [1.00, 0.98, 0.94, 0.90]
            depth_end_scales = [1.00, 0.96, 0.92, 0.88]
            structure_end_scales = [1.00, 0.98, 0.94, 0.90]
        else:
            base_denoising = float(sketch_options.get("sd_direct_denoising", 0.82))
            base_cfg = float(sketch_options.get("sd_direct_cfg_scale", 6.2))
            base_steps = int(sketch_options.get("sd_direct_steps", sketch_options.get("sd_sketch_steps", 30)))
            denoising_offsets = [0.0, 0.03, 0.08, 0.12]
            cfg_offsets = [0.0, 0.05, 0.15, 0.25]
            step_offsets = [0, 2, 2, 4]
            layout_scales = [1.00, 0.96, 0.90, 0.84]
            depth_scales = [1.00, 0.96, 0.90, 0.84]
            structure_scales = [1.00, 0.98, 0.94, 0.90]
            layout_end_scales = [1.00, 0.98, 0.94, 0.90]
            depth_end_scales = [1.00, 0.98, 0.94, 0.90]
            structure_end_scales = [1.00, 0.98, 0.94, 0.90]
        denoising = self._clamp(base_denoising + denoising_offsets[index], 0.42, 1.0)
        cfg_scale = self._clamp(base_cfg + cfg_offsets[index], 3.5, 12.0)
        steps = max(16, base_steps + step_offsets[index])
        return {
            "scene_type": scene_type,
            "variation_index": index,
            "sd_direct_denoising": round(denoising, 4),
            "sd_direct_cfg_scale": round(cfg_scale, 4),
            "sd_direct_steps": steps,
            "layout_strength_scale": layout_scales[index],
            "depth_strength_scale": depth_scales[index],
            "structure_strength_scale": structure_scales[index],
            "layout_end_scale": layout_end_scales[index],
            "depth_end_scale": depth_end_scales[index],
            "structure_end_scale": structure_end_scales[index],
            "note": (
                f"direct_scene_prior recipe={scene_type}:{index}"
                f" denoise={denoising:.2f} cfg={cfg_scale:.2f} steps={steps}"
                f" layout_scale={layout_scales[index]:.2f} depth_scale={depth_scales[index]:.2f}"
                f" structure_scale={structure_scales[index]:.2f}"
            ),
        }

    def _direct_sd_variant_controlnet(
        self,
        controlnet_bundle: Dict[str, Any] | None,
        recipe: Dict[str, Any],
    ) -> Dict[str, Any]:
        bundle = _copy(controlnet_bundle or {})
        inputs = bundle.get("inputs") if isinstance(bundle.get("inputs"), list) else []
        adjusted_inputs: List[Dict[str, Any]] = []
        for item in inputs:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            kind = str(entry.get("kind") or "").strip().lower()
            if kind == "scene_layout":
                entry["strength"] = round(
                    self._clamp(float(entry.get("strength", 0.0)) * float(recipe.get("layout_strength_scale", 1.0)), 0.05, 1.35),
                    4,
                )
                entry["end_percent"] = round(
                    self._clamp(float(entry.get("end_percent", 1.0)) * float(recipe.get("layout_end_scale", 1.0)), 0.12, 1.0),
                    4,
                )
            elif kind == "scene_depth":
                entry["strength"] = round(
                    self._clamp(float(entry.get("strength", 0.0)) * float(recipe.get("depth_strength_scale", 1.0)), 0.05, 1.35),
                    4,
                )
                entry["end_percent"] = round(
                    self._clamp(float(entry.get("end_percent", 1.0)) * float(recipe.get("depth_end_scale", 1.0)), 0.12, 1.0),
                    4,
                )
            elif kind == "scene_structure":
                entry["strength"] = round(
                    self._clamp(float(entry.get("strength", 0.0)) * float(recipe.get("structure_strength_scale", 1.0)), 0.05, 1.35),
                    4,
                )
                entry["end_percent"] = round(
                    self._clamp(float(entry.get("end_percent", 1.0)) * float(recipe.get("structure_end_scale", 1.0)), 0.12, 1.0),
                    4,
                )
            adjusted_inputs.append(entry)
        if adjusted_inputs:
            bundle["inputs"] = adjusted_inputs
        return bundle

    def _render_sd_direct_candidate(
        self,
        *,
        provider: str,
        variation_index: int,
        prompt: str,
        negative_prompt: str,
        scene_spec: Dict[str, Any] | None,
        sketch_options: Dict[str, Any],
        prior_bundle: Dict[str, Any] | None = None,
        controlnet_bundle: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if not self.sd_available:
            raise RuntimeError("SD sketch backend unavailable")
        scene_type = self._scene_type(scene_spec)
        recipe = self._direct_sd_variant_recipe(scene_type, variation_index, sketch_options)
        variant_sketch_options = dict(sketch_options)
        variant_sketch_options["sd_direct_denoising"] = recipe["sd_direct_denoising"]
        variant_sketch_options["sd_direct_cfg_scale"] = recipe["sd_direct_cfg_scale"]
        variant_sketch_options["sd_direct_steps"] = recipe["sd_direct_steps"]
        variant_controlnet_bundle = self._direct_sd_variant_controlnet(controlnet_bundle, recipe)
        output_name = f"sketch_v2_direct_{provider}_{abs(hash(self._fingerprint(provider, prompt, negative_prompt, self._direct_scene_control_path(prior_bundle), variation_index)))}"
        render_result = self.sd_generator.render_scene_direct(
            scene_spec=scene_spec,
            prompt=prompt,
            negative_prompt=negative_prompt,
            sketch_options=variant_sketch_options,
            filename_prefix=output_name,
            prior_bundle=prior_bundle,
            controlnet_bundle=variant_controlnet_bundle,
            **self._sd_lora_kwargs(sketch_options),
        )
        return {
            "image_path": render_result.get("image_path"),
            "control_image_path": self._direct_scene_control_path(render_result.get("prior_bundle") or prior_bundle),
            "prior_bundle": _copy(render_result.get("prior_bundle") or prior_bundle or {}),
            "controlnet_bundle": _copy(render_result.get("controlnet_bundle") or variant_controlnet_bundle or {}),
            "sd_recipe": recipe,
            "note": str(recipe.get("note") or "").strip(),
        }

    def _direct_preview_seed(
        self,
        scene_spec: Dict[str, Any],
        sketch_options: Dict[str, Any],
        *,
        title: str | None = None,
        preview_seed: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        normalized_scene = normalize_scene_spec_v2(scene_spec or {}, sketch_options)
        sketch_style = str(sketch_options.get("sketch_style", "scribble_line") or "scribble_line")
        scene_plan = self.build_scene_plan(normalized_scene, sketch_options=sketch_options, title=title)
        prior_prefix = f"sketch_v2_direct_prior_{abs(hash(json.dumps({'scene': normalized_scene, 'title': str(title or '')}, ensure_ascii=False, sort_keys=True)))}"
        prior_bundle = self.sd_generator.build_direct_scene_prior(
            scene_spec=normalized_scene,
            filename_prefix=prior_prefix,
        )
        controlnet_bundle = self.sd_generator.build_direct_scene_controlnet_bundle(
            scene_spec=normalized_scene,
            prior_bundle=prior_bundle,
            filename_prefix=prior_prefix,
        )
        control_path = self._direct_scene_control_path(prior_bundle)
        payload = dict(preview_seed or {})
        payload["success"] = True
        payload["type"] = "control_preview"
        payload["scene_spec"] = normalized_scene
        payload["image_path"] = control_path
        payload["save_path"] = control_path
        payload["low_preview_path"] = str(prior_bundle.get("base_plate_path") or control_path or "").strip()
        payload["render_control_path"] = control_path
        payload["sd_upstream_control_path"] = ""
        payload["backend"] = "sketch_v2_direct_seed"
        payload["sketch_backend"] = "sketch_v2"
        payload["scene_plan"] = scene_plan
        payload["direct_scene_prior"] = _copy(prior_bundle)
        payload["overlay_defaults"] = {
            "show_labels": bool((normalized_scene.get("layout_options") or {}).get("show_labels", False)),
            "show_grid": bool((normalized_scene.get("layout_options") or {}).get("show_grid", True)),
            "show_guides": bool((normalized_scene.get("layout_options") or {}).get("show_guides", False)),
            "view_mode": (normalized_scene.get("layout_options") or {}).get("sketch_view_mode", "structure"),
            "annotation_level": (normalized_scene.get("layout_options") or {}).get("annotation_level", "light"),
        }
        sketch_bundle = dict(payload.get("sketch_bundle") or {})
        sketch_bundle["active_sketch_backend"] = "sketch_v2"
        sketch_bundle["direct_scene_prior"] = _copy(prior_bundle)
        sketch_bundle["direct_scene_controlnet"] = _copy(controlnet_bundle)
        sketch_bundle["direct_scene_prompt_mode"] = "scene_spec_to_sd"
        payload["sketch_bundle"] = sketch_bundle
        payload["description"] = "SceneSpec direct-SD sketch seed without native structural sketch conversion."
        payload["generated_prompt"] = self.build_prompt(scene_plan, provider="sd", variation_index=0, title=title)
        payload["negative_prompt"] = self.build_negative_prompt(scene_plan, sketch_style)
        return payload

    def _render_direct_scene(
        self,
        scene_spec: Dict[str, Any],
        *,
        sketch_options: Dict[str, Any] | None = None,
        title: str | None = None,
        preview_seed: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        sketch_options = sketch_options or {}
        preview = self._direct_preview_seed(scene_spec, sketch_options, title=title, preview_seed=preview_seed)
        scene_plan = _copy(preview.get("scene_plan") or {})
        prior_bundle = _copy(preview.get("direct_scene_prior") or {})
        controlnet_bundle = _copy(((preview.get("sketch_bundle") or {}).get("direct_scene_controlnet")) or {})
        control_path = self._direct_scene_control_path(prior_bundle)
        candidates: List[Dict[str, Any]] = []
        for index, provider in enumerate(self._provider_plan_for_options(sketch_options)):
            prompt = self.build_prompt(scene_plan, provider=provider, variation_index=index, title=title)
            negative_prompt = self.build_negative_prompt(scene_plan, str(sketch_options.get("sketch_style", "scribble_line")))
            try:
                if provider == "sd":
                    render_result = self._render_sd_direct_candidate(
                        provider=provider,
                        variation_index=index,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        scene_spec=preview.get("scene_spec"),
                        sketch_options=sketch_options,
                        prior_bundle=prior_bundle,
                        controlnet_bundle=controlnet_bundle,
                    )
                    image_path = str(render_result.get("image_path") or "").strip()
                    candidate_control_path = str(render_result.get("control_image_path") or control_path).strip()
                else:
                    image_path = self._render_image_api_candidate(
                        provider=provider,
                        variation_index=index,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        control_image_path=control_path,
                    )
                    candidate_control_path = control_path
            except Exception as exc:
                logger.warning(f"sketch_v2 direct candidate failed | provider={provider} index={index} error={exc}")
                continue
            if not image_path:
                continue
            candidates.append(
                {
                    "candidate_id": f"sketch_v2_{provider}_{len(candidates) + 1}",
                    "provider": provider,
                    "image_path": image_path,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "control_image_path": candidate_control_path,
                    "note": str(render_result.get("note") or "direct_scene_prior").strip() if provider == "sd" else "direct_scene_prior",
                    "sd_recipe": _copy(render_result.get("sd_recipe") or {}) if provider == "sd" else {},
                }
            )
        if not candidates:
            candidates = [
                self._fallback_candidate(
                    preview,
                    scene_plan,
                    "No direct SD sketch candidate succeeded; falling back to current preview seed.",
                )
            ]
        active = candidates[0]
        sketch_bundle = dict(preview.get("sketch_bundle") or {})
        sketch_bundle["active_sketch_backend"] = "sketch_v2"
        sketch_bundle["active_sketch_path"] = active["image_path"]
        sketch_bundle["sketch_v2_candidates"] = [item["image_path"] for item in candidates]
        sketch_bundle["candidate_control_sources"] = [{"kind": "scene_prior_layout", "path": control_path}] if control_path else []
        payload = dict(preview)
        payload["image_path"] = active["image_path"]
        payload["save_path"] = active["image_path"]
        payload["backend"] = "sketch_v2_direct_preview"
        payload["sketch_bundle"] = sketch_bundle
        payload["sketch_candidates"] = [_copy(item) for item in candidates]
        payload["active_sketch_candidate_id"] = active["candidate_id"]
        payload["active_sketch_backend"] = "sketch_v2"
        payload["active_sketch_provider"] = active["provider"]
        payload["active_sketch_path"] = active["image_path"]
        payload["generated_prompt"] = active["prompt"]
        payload["negative_prompt"] = active["negative_prompt"]
        payload["native_control_image_path"] = control_path
        payload["upstream_control_image_path"] = ""
        payload["sd_upstream_control_path"] = ""
        payload["upstream_control_provider"] = "direct_scene_prior" if control_path else ""
        if active.get("note"):
            payload["note"] = active["note"]
        return payload

    def render_from_scene_spec(
        self,
        scene_spec: Dict[str, Any] | None,
        sketch_options: Dict[str, Any] | None = None,
        title: str | None = None,
    ) -> Dict[str, Any]:
        sketch_options = sketch_options or {}
        return self._render_direct_scene(scene_spec or {}, sketch_options=sketch_options, title=title, preview_seed={})

    def _build_upstream_sd_prompt(self, scene_plan: Dict[str, Any], title: str | None = None) -> str:
        scene_type = str(scene_plan.get("scene_type", "scene") or "scene")
        prompt_title = self._prompt_title_text(title)
        parts = [
            self._prompt_prefix(scene_type),
            "Create an upstream whole-scene grayscale draft for later sketch refinement and editing.",
            "Replace geometric placeholders with natural scene silhouettes, readable object masses, and coherent scene depth.",
            "Use the control image only as a soft prior for horizon, placement, scale, and front-mid-back layering.",
            "Do not preserve symbol collage, object blobs, icon stickers, helper boxes, arrows, labels, or rigid repeated geometry.",
            "Prefer tonal scene masses, believable object shapes, and readable environment context over isolated contour symbols.",
            "People, buildings, trees, roads, and devices must read as real scene elements instead of stick figures or pictograms.",
            prompt_title,
            f'Subject: {self._clean_text(scene_plan.get("subject_summary") or "")}',
            f'Background: {self._clean_text(scene_plan.get("background_summary") or "")}',
            f'Composition: {self._clean_text(scene_plan.get("composition_summary") or "")}',
            f'Depth: {self._clean_text(scene_plan.get("depth_summary") or "")}',
            f'Style: {self._clean_text(scene_plan.get("style_summary") or "")}',
            "Favor natural whole-image readability over symbolic object assembly.",
        ]
        must_include_summary = self._clean_text(scene_plan.get("must_include_summary") or "")
        if must_include_summary:
            parts.append(f"Must clearly include all anchors: {must_include_summary}.")
        exact_elements_summary = self._clean_text(scene_plan.get("exact_elements_summary") or "")
        if exact_elements_summary:
            parts.append(f"Exact visible elements only: {exact_elements_summary}. Do not add extra major objects or substitute object types.")
        anchor_layout_summary = self._clean_text(scene_plan.get("anchor_layout_summary") or "")
        if anchor_layout_summary:
            parts.append(f"Exact anchor layout: {anchor_layout_summary}.")
        editable_detail_summary = self._clean_text(scene_plan.get("editable_detail_summary") or "")
        if editable_detail_summary:
            parts.append(f"Keep editable details clear: {editable_detail_summary}.")
        anchor_presence_summary = self._clean_text(self._anchor_presence_summary(scene_plan))
        if anchor_presence_summary:
            parts.append(anchor_presence_summary)
        object_hint_text = ", ".join(
            f'{item.get("prompt_name") or item.get("concept", "")} {item.get("position", "")} {item.get("depth_band", "")}'.strip()
            for item in (scene_plan.get("object_hints") or [])[:8]
            if item.get("prompt_name") or item.get("concept")
        )
        if object_hint_text:
            parts.append(f"Scene anchors: {object_hint_text}.")
        return ", ".join(part for part in parts if part)

    def _build_upstream_sd_negative_prompt(self, scene_plan: Dict[str, Any], sketch_style: str) -> str:
        negatives = [
            self.build_negative_prompt(scene_plan, sketch_style),
            "generic blob object",
            "placeholder silhouette",
            "stick figure",
            "floating isolated icon",
            "pictogram",
            "clipart",
            "mechanical layout diagram",
            "sticker sheet composition",
            "hard geometric glyph",
            "empty white page",
            "single object on blank background",
            "missing center subject",
        ]
        return ", ".join(self._clean_text(item) for item in negatives if self._clean_text(item))

    def _render_sd_upstream_guide(
        self,
        *,
        preview: Dict[str, Any],
        scene_plan: Dict[str, Any],
        control_image_path: str,
        sketch_options: Dict[str, Any],
        title: str | None = None,
    ) -> Dict[str, str]:
        if not self._prefer_sd_upstream(sketch_options) or not control_image_path:
            return {}
        prompt = self._build_upstream_sd_prompt(scene_plan, title=title)
        negative_prompt = self._build_upstream_sd_negative_prompt(
            scene_plan,
            str(sketch_options.get("sketch_style", "scribble_line")),
        )
        control_name = os.path.basename(control_image_path).lower()
        stronger_tonal_control = control_name.startswith("sd_upstream_control_")
        output_name = f"sketch_v2_upstream_sd_{abs(hash(self._fingerprint('sd_upstream', prompt, negative_prompt, control_image_path, 0)))}"
        controlnet_bundle = self.sd_generator.build_controlnet_bundle(
            control_image_path=control_image_path,
            scene_spec=preview.get("scene_spec"),
            filename_prefix=output_name,
            purpose="sketch_upstream",
        )
        try:
            image_path = self.sd_generator.render_img2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image_path=control_image_path,
                denoising_strength=float(sketch_options.get("sd_upstream_denoising", 0.72 if stronger_tonal_control else 0.58)),
                steps=int(sketch_options.get("sd_upstream_steps", 30 if stronger_tonal_control else 24)),
                cfg_scale=float(sketch_options.get("sd_upstream_cfg_scale", 6.4 if stronger_tonal_control else 6.2)),
                sampler_name=str(sketch_options.get("sd_upstream_sampler_name", sketch_options.get("sd_sketch_sampler_name", "DPM++ 2M Karras"))),
                filename_prefix=output_name,
                controlnet_bundle=controlnet_bundle,
                **self._sd_lora_kwargs(sketch_options),
            )
        except Exception as exc:
            logger.warning(f"sketch_v2 upstream sd guide failed: {exc}")
            return {}
        return {
            "image_path": image_path,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "provider": "sd_upstream",
        }

    def _fallback_candidate(self, preview: Dict[str, Any], scene_plan: Dict[str, Any], reason: str) -> Dict[str, Any]:
        image_path = str(preview.get("image_path") or "")
        return {
            "candidate_id": "sketch_v2_fallback_native_1",
            "provider": "fallback_native",
            "image_path": image_path,
            "prompt": self.build_prompt(scene_plan, provider="image_api", variation_index=0),
            "negative_prompt": self.build_negative_prompt(scene_plan, str((preview.get("scene_spec") or {}).get("layout_options", {}).get("sketch_style", "scribble_line"))),
            "control_image_path": self._control_image_path(preview) or image_path,
            "note": reason,
        }

    def _fuse_control_image(
        self,
        *,
        raw_control_image_path: str,
        upstream_sd_guide_path: str,
    ) -> str:
        raw_path = str(raw_control_image_path or "").strip()
        upstream_path = str(upstream_sd_guide_path or "").strip()
        if not raw_path or not upstream_path or raw_path == upstream_path:
            return ""
        try:
            upstream_image = Image.open(upstream_path).convert("RGBA")
            raw_image = Image.open(raw_path).convert("RGBA").resize(upstream_image.size)
            raw_alpha = raw_image.convert("L").point(lambda px: max(0, min(255, int((255 - px) * 0.72))))
            raw_overlay = raw_image.copy()
            raw_overlay.putalpha(raw_alpha)
            fused = Image.alpha_composite(upstream_image, raw_overlay).convert("RGB")
            output_name = f"sketch_v2_fused_control_{abs(hash(self._fingerprint('fused_control', raw_path, upstream_path, '', 0)))}.png"
            output_path = os.path.join(self.output_dir, output_name)
            fused.save(output_path)
            return output_path
        except Exception as exc:
            logger.warning(f"sketch_v2 fused control build failed: {exc}")
            return ""

    def _candidate_control_sources(
        self,
        *,
        raw_control_image_path: str,
        upstream_sd_guide_path: str,
        fused_control_image_path: str,
    ) -> List[Dict[str, str]]:
        sources: List[Dict[str, str]] = []
        for kind, path in [
            ("fused_anchor_scene", fused_control_image_path),
            ("raw_anchor_control", raw_control_image_path),
            ("sd_upstream_scene", upstream_sd_guide_path),
        ]:
            cleaned = str(path or "").strip()
            if not cleaned:
                continue
            if any(item.get("path") == cleaned for item in sources):
                continue
            sources.append({"kind": kind, "path": cleaned})
        return sources

    def render_from_preview(
        self,
        preview: Dict[str, Any],
        sketch_options: Dict[str, Any] | None = None,
        title: str | None = None,
    ) -> Dict[str, Any]:
        preview = dict(preview or {})
        sketch_options = sketch_options or {}
        if self._use_direct_scene_mode(sketch_options):
            return self._render_direct_scene(
                preview.get("scene_spec") if isinstance(preview.get("scene_spec"), dict) else {},
                sketch_options=sketch_options,
                title=title,
                preview_seed=preview,
            )
        scene_plan = self.build_scene_plan(preview.get("scene_spec"), sketch_options=sketch_options, title=title)
        control_image_path = self._upstream_control_image_path(preview) or self._control_image_path(preview)
        sketch_bundle = dict(preview.get("sketch_bundle") or {})
        native_structural = sketch_bundle.get("structural_sketch") or preview.get("image_path")
        upstream_sd_guide = self._render_sd_upstream_guide(
            preview=preview,
            scene_plan=scene_plan,
            control_image_path=control_image_path,
            sketch_options=sketch_options,
            title=title,
        )
        upstream_guide_path = str(upstream_sd_guide.get("image_path") or "").strip()
        fused_control_path = self._fuse_control_image(
            raw_control_image_path=control_image_path,
            upstream_sd_guide_path=upstream_guide_path,
        )
        control_sources = self._candidate_control_sources(
            raw_control_image_path=control_image_path,
            upstream_sd_guide_path=upstream_guide_path,
            fused_control_image_path=fused_control_path,
        )
        upstream_control_path = control_sources[0]["path"] if control_sources else str(control_image_path or upstream_guide_path or "").strip()
        candidates: List[Dict[str, Any]] = []
        for index, provider in enumerate(self._provider_plan_for_options(sketch_options)):
            prompt = self.build_prompt(scene_plan, provider=provider, variation_index=index, title=title)
            negative_prompt = self.build_negative_prompt(scene_plan, str(sketch_options.get("sketch_style", "scribble_line")))
            control_source = control_sources[index % len(control_sources)] if control_sources else {"kind": "default", "path": upstream_control_path}
            try:
                if provider == "sd":
                    image_path = self._render_sd_candidate(
                        provider=provider,
                        variation_index=index,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        control_image_path=control_source["path"],
                        sketch_options=sketch_options,
                        control_kind=str(control_source.get("kind") or "default"),
                        scene_spec=preview.get("scene_spec"),
                    )
                else:
                    image_path = self._render_image_api_candidate(
                        provider=provider,
                        variation_index=index,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        control_image_path=control_source["path"],
                    )
            except Exception as exc:
                logger.warning(f"sketch_v2 candidate failed | provider={provider} index={index} error={exc}")
                continue
            candidates.append(
                {
                    "candidate_id": f"sketch_v2_{provider}_{len(candidates) + 1}",
                    "provider": provider,
                    "image_path": image_path,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "control_image_path": control_source["path"],
                    "note": control_source["kind"],
                }
            )
        if not candidates:
            candidates = [
                self._fallback_candidate(
                    preview,
                    scene_plan,
                    "No sketch_v2 provider succeeded; falling back to native composition sketch.",
                )
            ]
        active = candidates[0]
        if native_structural:
            sketch_bundle.setdefault("native_structural_sketch", native_structural)
        if upstream_sd_guide.get("image_path"):
            sketch_bundle["sd_upstream_guide"] = upstream_sd_guide.get("image_path")
            sketch_bundle["upstream_control_provider"] = upstream_sd_guide.get("provider")
            sketch_bundle["upstream_control_prompt"] = upstream_sd_guide.get("prompt")
        if fused_control_path:
            sketch_bundle["fused_control_image"] = fused_control_path
        if control_sources:
            sketch_bundle["candidate_control_sources"] = _copy(control_sources)
        sketch_bundle["active_sketch_backend"] = "sketch_v2"
        sketch_bundle["active_sketch_path"] = active["image_path"]
        sketch_bundle["sketch_v2_candidates"] = [item["image_path"] for item in candidates]
        payload = dict(preview)
        payload["image_path"] = active["image_path"]
        payload["save_path"] = active["image_path"]
        payload["backend"] = "sketch_v2_preview"
        payload["sketch_backend"] = "sketch_v2"
        payload["generated_prompt"] = active["prompt"]
        payload["negative_prompt"] = active["negative_prompt"]
        payload["sketch_bundle"] = sketch_bundle
        payload["sketch_candidates"] = [_copy(item) for item in candidates]
        payload["active_sketch_candidate_id"] = active["candidate_id"]
        payload["active_sketch_backend"] = "sketch_v2"
        payload["active_sketch_provider"] = active["provider"]
        payload["active_sketch_path"] = active["image_path"]
        payload["native_control_image_path"] = control_image_path
        payload["upstream_control_image_path"] = upstream_control_path if upstream_control_path and upstream_control_path != control_image_path else ""
        payload["sd_upstream_control_path"] = preview.get("sd_upstream_control_path") or sketch_bundle.get("sd_upstream_control") or ""
        payload["upstream_control_provider"] = upstream_sd_guide.get("provider", "")
        payload["scene_plan"] = scene_plan
        if active.get("note"):
            payload["note"] = active["note"]
        return payload

