"""
多模态生成模块
基于 Tri-Maze 推理链路（节点/关系/阻力）直接生成结构化多模态参数
完全基于推理路径驱动，不依赖自然语言Prompt中转
"""
import json
import asyncio
import requests
import base64
import os
import mimetypes
from typing import Dict, List, Any, Optional
from loguru import logger
from openai import OpenAI
from io import BytesIO
from PIL import Image
from .native_generator import NativeGenerator
from .sd_sketch_generator import SDSketchGenerator
from .sketch_v2 import SketchV2Generator
from .sketch_edit_v1 import build_render_conditioning_bundle as build_editable_render_conditioning_bundle
from .semantic_scene_v2 import summarize_scene_spec
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .maze_engine import MazePath


def _safe_scene_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


class MultimodalGenerator:
    """多模态生成器：基于Tri-Maze推理链路直接驱动多模态生成
    完全基于推理路径的节点、关系、阻力生成参数，不需要自然语言Prompt中转
    """
    
    def __init__(self, llm_client: OpenAI = None, config: Dict = None):
        self.llm_client = llm_client
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 生成API配置
        self.config = config or {}
        self.sd_api_url = self.config.get("sd_api_url", "").strip()  # Stable Diffusion API地址
        self.image_api_url = self.config.get("image_api_url", "").strip()
        self.image_api_key = self.config.get("image_api_key", "").strip()
        self.image_api_model = self.config.get("image_api_model", "").strip()
        self.image_api_control_model = self.config.get("image_api_control_model", "").strip()
        self.image_api_size = self.config.get("image_api_size", "").strip()
        self.dalle_api_key = self.config.get("dalle_api_key", "").strip() or (self.image_api_key if not self.image_api_url else "")
        self.pika_api_key = self.config.get("pika_api_key", "")
        self.runway_api_key = self.config.get("runway_api_key", "")
        
        # 原生生成器：完全不依赖外部API
        self.native_generator = NativeGenerator(output_dir=self.output_dir)
        self.sd_sketch_generator = SDSketchGenerator(output_dir=self.output_dir, sd_api_url=self.sd_api_url)
        self.sketch_v2_generator = SketchV2Generator(
            output_dir=self.output_dir,
            sd_api_url=self.sd_api_url,
            image_api_url=self.image_api_url,
            image_api_key=self.image_api_key,
            image_api_model=self.image_api_model,
            image_api_size=self.image_api_size,
        )
        self.use_native_generation = self.config.get("use_native_generation", True)  # 默认使用原生生成
        
        # 概念到视觉特征的映射规则
        self.concept_feature_map = {
            # 电子电路
            "电阻": {"shape": "cylindrical", "color": ["beige", "brown", "red"], "tags": ["electronic component", "resistor"]},
            "LED": {"shape": "diode", "color": ["transparent", "red", "green", "blue"], "tags": ["light-emitting diode", "electronic component"]},
            "Arduino": {"shape": "rectangular board", "color": ["blue", "green"], "tags": ["microcontroller", "development board"]},
            "电源": {"shape": "battery", "color": ["black", "red"], "tags": ["power supply", "battery"]},
            "GND": {"shape": "ground symbol", "color": ["black"], "tags": ["ground", "electrical ground"]},
            "电路": {"style": "schematic diagram", "background": "white", "tags": ["circuit diagram", "schematic"]},
            
            # 物理
            "力": {"visual": "arrow", "color": ["red"], "tags": ["force vector", "physics"]},
            "速度": {"visual": "arrow", "color": ["blue"], "tags": ["velocity vector", "physics"]},
            "能量": {"visual": "glow", "color": ["yellow", "orange"], "tags": ["energy", "glowing effect"]},
            "光": {"visual": "rays", "color": ["white", "yellow"], "tags": ["light rays", "bright"]},
            
            # 生物
            "猫": {"shape": "feline", "color": ["various"], "tags": ["cat", "animal", "feline"]},
            "毛皮": {"texture": "fur", "color": ["soft"], "tags": ["fur", "animal fur"]},
            "羽毛": {"texture": "soft", "color": ["light"], "tags": ["feathers", "bird"]},
            "细胞": {"shape": "circular", "color": ["transparent"], "tags": ["cell", "biology"]},
            
            # 通用
            "抽象概念": {"style": "abstract", "color": ["gradient"], "tags": ["abstract art"]},
            "机械结构": {"style": "technical drawing", "color": ["gray", "metal"], "tags": ["mechanical design"]}
        }
        
        # 阻力到视觉权重的映射：阻力越低，视觉上越突出
        self.resistance_weight_map = {
            (0.0, 0.2): {"weight": 1.0, "opacity": 1.0, "size_multiplier": 1.5},
            (0.2, 0.4): {"weight": 0.8, "opacity": 0.9, "size_multiplier": 1.3},
            (0.4, 0.6): {"weight": 0.6, "opacity": 0.8, "size_multiplier": 1.1},
            (0.6, 0.8): {"weight": 0.4, "opacity": 0.7, "size_multiplier": 1.0},
            (0.8, 1.01): {"weight": 0.2, "opacity": 0.5, "size_multiplier": 0.8},
        }
        
        logger.info("✅ 多模态生成器初始化完成（基于推理链路直接生成）")
    
    def _get_weight_config(self, resistance: float) -> Dict[str, float]:
        for (lower, upper), config in self.resistance_weight_map.items():
            if lower <= resistance < upper:
                return config
        return self.resistance_weight_map[(0.4, 0.6)]
    
    def _get_canvas_size(self, sketch_options: Dict[str, Any] | None = None) -> tuple[int, int]:
        sketch_options = sketch_options or {}
        width = int(sketch_options.get("canvas_width", 1024))
        height = int(sketch_options.get("canvas_height", 768))
        return max(512, width), max(384, height)

    def _clean_prompt_text(self, value: Any) -> str:
        text = str(value or "").replace("\n", " ").replace("\r", " ").replace("|", " ").strip()
        return " ".join(text.split()).strip(" ,;，；。")

    def _resolve_sketch_backend(self, sketch_options: Dict[str, Any] | None = None) -> str:
        sketch_options = sketch_options or {}
        backend = str(sketch_options.get("sketch_backend", self.config.get("sketch_backend", "native")) or "native").strip().lower()
        return backend if backend in {"native", "sd", "sketch_v2"} else "native"

    def _find_sketch_candidate(
        self,
        candidates: List[Dict[str, Any]] | None = None,
        candidate_id: str | None = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(candidates, list) or not candidates:
            return None
        if candidate_id:
            marker = str(candidate_id).strip()
            for item in candidates:
                if isinstance(item, dict) and str(item.get("candidate_id") or "").strip() == marker:
                    return item
        for item in candidates:
            if isinstance(item, dict) and str(item.get("image_path") or "").strip():
                return item
        return None

    def resolve_used_control_image_path(
        self,
        *,
        sketch_options: Dict[str, Any] | None = None,
        preview: Dict[str, Any] | None = None,
        render_options: Dict[str, Any] | None = None,
        control_image_path: str | None = None,
        low_preview_path: str | None = None,
        render_control_path: str | None = None,
    ) -> str | None:
        sketch_options = sketch_options or {}
        preview = preview if isinstance(preview, dict) else {}
        render_options = render_options if isinstance(render_options, dict) else {}
        for key in ("editable_sketch_composited_path",):
            candidate = str(render_options.get(key) or preview.get(key) or "").strip()
            if candidate:
                return candidate
        backend = self._resolve_sketch_backend(sketch_options)
        if backend == "sketch_v2":
            candidates = render_options.get("sketch_candidates")
            if not isinstance(candidates, list) or not candidates:
                candidates = preview.get("sketch_candidates")
            selected_id = (
                render_options.get("selected_sketch_candidate_id")
                or render_options.get("active_sketch_candidate_id")
                or sketch_options.get("selected_sketch_candidate_id")
                or preview.get("active_sketch_candidate_id")
            )
            candidate = self._find_sketch_candidate(candidates, selected_id)
            if candidate:
                candidate_path = str(candidate.get("image_path") or "").strip()
                if candidate_path:
                    return candidate_path
            for key in ("active_sketch_path", "image_path", "native_control_image_path"):
                value = str(render_options.get(key) or preview.get(key) or "").strip()
                if value:
                    return value
        return (
            str(render_control_path or "").strip()
            or str(low_preview_path or "").strip()
            or str(control_image_path or "").strip()
            or None
        )

    def _conditioning_summary(self, conditioning_bundle: Dict[str, Any] | None = None) -> str:
        bundle = conditioning_bundle if isinstance(conditioning_bundle, dict) else {}
        counts = bundle.get("counts") if isinstance(bundle.get("counts"), dict) else {}
        edited_regions = int(counts.get("edited_regions", 0) or 0)
        patches = int(counts.get("patches", 0) or 0)
        if not edited_regions and not patches:
            return ""
        parts = []
        if edited_regions:
            parts.append(f"edited_regions={edited_regions}")
        if patches:
            parts.append(f"patches={patches}")
        return " | ".join(parts)

    def _persist_conditioning_bundle(self, conditioning_bundle: Dict[str, Any] | None = None) -> str:
        bundle = conditioning_bundle if isinstance(conditioning_bundle, dict) else {}
        if not bundle:
            return ""
        revision_id = str(bundle.get("revision_id") or "").strip()
        digest = abs(hash(json.dumps(bundle, ensure_ascii=False, sort_keys=True)))
        filename = f"render_conditioning_{revision_id or digest}.json"
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(bundle, handle, ensure_ascii=False, indent=2)
        return output_path

    def _build_render_conditioning_bundle(
        self,
        *,
        render_options: Dict[str, Any] | None = None,
        scene_spec: Dict[str, Any] | None = None,
        used_control_image_path: str | None = None,
    ) -> Dict[str, Any]:
        render_options = render_options if isinstance(render_options, dict) else {}
        scene_spec = scene_spec if isinstance(scene_spec, dict) else {}
        editable_doc = render_options.get("editable_sketch_doc") if isinstance(render_options.get("editable_sketch_doc"), dict) else None
        if editable_doc:
            bundle = build_editable_render_conditioning_bundle(editable_doc)
        else:
            render_hints = scene_spec.get("render_hints", {}) if isinstance(scene_spec.get("render_hints"), dict) else {}
            region_constraints = _safe_scene_copy(render_hints.get("region_edit_constraints") or [])
            patch_constraints = _safe_scene_copy(render_hints.get("render_patch_constraints") or [])
            if not region_constraints and not patch_constraints:
                return {}
            bundle = {
                "version": 1,
                "source": "scene_spec_render_hints",
                "revision_id": "",
                "session_id": "",
                "sketch_backend": str(render_options.get("sketch_backend") or ""),
                "source_candidate_id": str(render_options.get("active_sketch_candidate_id") or ""),
                "base_image_path": "",
                "background_plate_path": "",
                "composited_image_path": str(render_hints.get("editable_sketch_composited_path") or ""),
                "canvas_size": _safe_scene_copy(scene_spec.get("canvas_size") or {}),
                "camera_state": {},
                "depth_model": {},
                "edit_summary": str(render_hints.get("edit_summary") or ""),
                "region_edit_summary": str(render_hints.get("region_edit_summary") or ""),
                "region_edit_constraints": region_constraints,
                "render_patch_constraints": patch_constraints,
                "object_layers": [],
                "region_layers": [],
                "patch_layers": [],
                "counts": {
                    "objects": 0,
                    "regions": len(region_constraints),
                    "edited_regions": len(region_constraints),
                    "patches": len(patch_constraints),
                },
            }
        if not bundle:
            return {}
        bundle["used_control_image_path"] = str(used_control_image_path or bundle.get("composited_image_path") or "")
        bundle["conditioning_summary"] = self._conditioning_summary(bundle)
        bundle["conditioning_bundle_path"] = self._persist_conditioning_bundle(bundle)
        return bundle

    def render_scene_spec_preview(
        self,
        scene_spec: Dict[str, Any],
        sketch_options: Dict[str, Any] | None = None,
        title: str | None = None,
    ) -> Dict[str, Any]:
        sketch_options = sketch_options or {}
        backend = self._resolve_sketch_backend(sketch_options)
        if backend == "sketch_v2" and self.sketch_v2_generator._use_direct_scene_mode(sketch_options):
            return self.sketch_v2_generator.render_from_scene_spec(scene_spec, sketch_options=sketch_options, title=title)
        preview = self.native_generator.render_scene_spec_preview(scene_spec, sketch_options=sketch_options, title=title)
        sketch_bundle = dict(preview.get("sketch_bundle") or {})
        native_structural = sketch_bundle.get("structural_sketch") or preview.get("image_path")
        if native_structural:
            sketch_bundle.setdefault("native_structural_sketch", native_structural)
        sketch_bundle["active_sketch_backend"] = "native"
        preview["sketch_bundle"] = sketch_bundle
        preview.setdefault("backend", "native_scene_spec_preview")
        preview["sketch_backend"] = "native"
        if backend == "sd":
            return self.sd_sketch_generator.render_from_preview(preview, sketch_options=sketch_options, title=title)
        if backend == "sketch_v2":
            return self.sketch_v2_generator.render_from_preview(preview, sketch_options=sketch_options, title=title)
        if backend != "sd":
            return preview
        return self.sd_sketch_generator.render_from_preview(preview, sketch_options=sketch_options, title=title)

    def _dedupe_prompt_items(self, items: List[str]) -> List[str]:
        seen = set()
        output: List[str] = []
        for item in items:
            cleaned = self._clean_prompt_text(item)
            if not cleaned:
                continue
            marker = cleaned.casefold()
            if marker in seen:
                continue
            seen.add(marker)
            output.append(cleaned)
        return output

    def _scene_type_caption(self, scene_type: str) -> str:
        mapping = {
            "scene": "具象场景图像",
            "process": "过程图解画面",
            "schematic": "技术结构示意图",
        }
        return mapping.get(str(scene_type or "scene"), "完整图像")

    def _position_phrase(self, bbox: Dict[str, Any], depth_band: str = "") -> str:
        x_center = float(bbox.get("x_norm", 0.0)) + float(bbox.get("width_norm", 0.0)) / 2.0
        y_center = float(bbox.get("y_norm", 0.0)) + float(bbox.get("height_norm", 0.0)) / 2.0
        if depth_band == "foreground" or y_center >= 0.68:
            depth = "前景"
        elif depth_band == "background" or y_center <= 0.34:
            depth = "背景"
        else:
            depth = "中景"
        if x_center <= 0.28:
            horizontal = "偏左"
        elif x_center >= 0.72:
            horizontal = "偏右"
        else:
            horizontal = "居中"
        return f"{depth}{horizontal}"

    def _size_phrase(self, bbox: Dict[str, Any]) -> str:
        area = float(bbox.get("width_norm", 0.0)) * float(bbox.get("height_norm", 0.0))
        if area >= 0.2:
            return "超大"
        if area >= 0.1:
            return "较大"
        if area >= 0.04:
            return "中等"
        return "小型"

    def _layer_region_phrase(self, bbox: Dict[str, Any]) -> str:
        x_norm = float(bbox.get("x_norm", 0.0))
        y_norm = float(bbox.get("y_norm", 0.0))
        width_norm = float(bbox.get("width_norm", 0.0))
        height_norm = float(bbox.get("height_norm", 0.0))
        if width_norm >= 0.9 and height_norm >= 0.45:
            if y_norm <= 0.12:
                return "上半部"
            if y_norm >= 0.45:
                return "下半部"
            return "大部分区域"
        if y_norm <= 0.2:
            vertical = "上方"
        elif y_norm >= 0.55:
            vertical = "下方"
        else:
            vertical = "中部"
        if x_norm <= 0.25:
            horizontal = "偏左"
        elif x_norm >= 0.55:
            horizontal = "偏右"
        else:
            horizontal = ""
        return f"{vertical}{horizontal}"

    def _describe_render_object(self, obj: Dict[str, Any]) -> str:
        name = self._clean_prompt_text(obj.get("concept") or obj.get("asset_key") or "对象")
        bbox = obj.get("bbox") or {}
        role = str(obj.get("role") or "")
        source = str(obj.get("source") or "")
        prefix: List[str] = []
        if role in {"subject", "focus", "core_subject"}:
            prefix.append("主体")
        elif role in {"detail", "support"}:
            prefix.append("辅助元素")
        if source == "user":
            prefix.append("用户补充")
        qualifier = "".join(prefix)
        return f"{qualifier}{self._position_phrase(bbox, str(obj.get('depth_band') or ''))}的{self._size_phrase(bbox)}{name}"

    def _scene_query_clause(self, scene_context: Dict[str, Any] | None = None) -> str:
        if not isinstance(scene_context, dict):
            return ""
        query = self._clean_prompt_text(scene_context.get("query") or "")
        if not query:
            return ""
        if len(query) > 96:
            query = query[:96].rstrip("，,；;。.!！？? ") + "…"
        return query

    def _scene_style_booster(self, scene_type: str, query_clause: str = "") -> list[str]:
        boosters: list[str] = []
        if scene_type == "scene":
            boosters.append("严格保持用户指定的主体数量、动作姿态、相对方位与远近层次。")
            boosters.append("以自然完整的真实场景方式呈现，不要把对象拼贴成素材板。")
        elif scene_type == "process":
            boosters.append("整体表现为浅色科普过程图解，按阶段展开，但每个阶段都必须是可识别对象，不要做成 PPT 卡片。")
            boosters.append("过程箭头只作辅助，画面主体仍然是蒸发、凝结、降雨等可识别元素。")
        elif scene_type == "schematic":
            boosters.append("整体表现为浅色背景的二维工程示意图，使用符号化元件、清晰导线、正视角和规则排布。")
            boosters.append("不要生成真实 PCB 主板照片、微距芯片特写或产品摄影。")
        if query_clause:
            boosters.append(f"必须忠实满足原始需求：{query_clause}。")
        return boosters

    def _scene_spec_semantic_sections(self, scene_spec: Dict[str, Any] | None = None) -> Dict[str, str]:
        if not isinstance(scene_spec, dict):
            return {}
        constraints = self._render_constraints(scene_spec)
        layout_options = scene_spec.get("layout_options", {}) if isinstance(scene_spec, dict) else {}
        scene_type = str(layout_options.get("scene_type", "scene") or "scene")
        canvas_size = scene_spec.get("canvas_size", {}) if isinstance(scene_spec, dict) else {}
        canvas_width = int(canvas_size.get("width", 1024) or 1024)
        canvas_height = int(canvas_size.get("height", 768) or 768)
        orientation = "横向" if canvas_width >= canvas_height else "竖向"

        background_lines = []
        for layer in constraints.get("background_layers", [])[:4]:
            layer_type = str(layer.get("type") or "")
            label = self._clean_prompt_text(layer.get("label") or layer_type or "背景层")
            if scene_type == "schematic" and layer_type == "board":
                label = "浅色技术底图"
            elif scene_type == "process" and layer_type == "process_band":
                label = "阶段区域"
            background_lines.append(f"{label}铺在画面{self._layer_region_phrase(layer.get('bbox') or {})}")
        environment = "、".join(self._dedupe_prompt_items(background_lines))

        objects = list(constraints.get("object_instances", []) or [])
        role_order = {"subject": 0, "focus": 0, "core_subject": 0, "support": 1, "detail": 2, "environment": 3}

        def _sort_key(item: Dict[str, Any]) -> tuple[float, float]:
            bbox = item.get("bbox") or {}
            area = float(bbox.get("width_norm", 0.0)) * float(bbox.get("height_norm", 0.0))
            return (float(role_order.get(str(item.get("role") or ""), 4)), -area)

        sorted_objects = sorted(objects, key=_sort_key)
        subjects = [self._describe_render_object(item) for item in sorted_objects[:6]]
        subject_text = "、".join(self._dedupe_prompt_items(subjects[:4]))

        depth_counts = constraints.get("depth_band_counts", {}) or {}
        depth_chunks = []
        for depth_band in ("foreground", "midground", "background"):
            count = int(depth_counts.get(depth_band, 0) or 0)
            if count > 0:
                depth_label = {"foreground": "前景", "midground": "中景", "background": "背景"}[depth_band]
                depth_chunks.append(f"{depth_label}{count}个主要对象")
        composition_bits = [f"{orientation}构图"]
        if depth_chunks:
            composition_bits.append("层次分布为" + "、".join(depth_chunks))
        if sorted_objects:
            composition_bits.append("视觉重心放在" + self._position_phrase((sorted_objects[0].get("bbox") or {}), str(sorted_objects[0].get("depth_band") or "")))
        composition = "，".join(self._dedupe_prompt_items(composition_bits))

        lookup = {
            str(item.get("id") or ""): self._clean_prompt_text(item.get("concept") or item.get("asset_key") or "对象")
            for item in objects
            if item.get("id")
        }
        attachment_lines = []
        for item in constraints.get("attachments", [])[:6]:
            child = lookup.get(str(item.get("child_id") or ""), "附着元素")
            host = lookup.get(str(item.get("host_id") or ""), "宿主")
            anchor = self._clean_prompt_text(item.get("anchor_name") or "对应位置")
            attachment_lines.append(f"{child}附着在{host}的{anchor}")
        attachments = "、".join(self._dedupe_prompt_items(attachment_lines))

        connector_lines = []
        for item in constraints.get("connectors", [])[:4]:
            from_name = lookup.get(str(item.get("from_id") or ""), "起点")
            to_name = lookup.get(str(item.get("to_id") or ""), "终点")
            relation = self._clean_prompt_text(item.get("label") or item.get("type") or "连接")
            connector_lines.append(f"{from_name}与{to_name}通过{relation}形成联系")
        connectors = "、".join(self._dedupe_prompt_items(connector_lines))

        user_added = []
        for item in sorted_objects:
            if str(item.get("source") or "") == "user":
                user_added.append(self._describe_render_object(item))
        user_added_text = "、".join(self._dedupe_prompt_items(user_added[:5]))
        render_hints = scene_spec.get("render_hints", {}) if isinstance(scene_spec, dict) else {}
        edit_summary = self._clean_prompt_text(render_hints.get("edit_summary") or "")
        region_edit_summary = self._clean_prompt_text(render_hints.get("region_edit_summary") or "")

        return {
            "scene_type": scene_type,
            "scene_caption": self._scene_type_caption(scene_type),
            "environment": environment,
            "subjects": subject_text,
            "composition": composition,
            "attachments": attachments,
            "connectors": connectors,
            "user_added": user_added_text,
            "edit_summary": edit_summary,
            "region_edit_summary": region_edit_summary,
        }

    def _scene_spec_to_prompt(self, scene_spec: Dict[str, Any] | None = None) -> str:
        sections = self._scene_spec_semantic_sections(scene_spec)
        if not sections:
            return ""
        prompt_parts = [f"画面类型：{sections.get('scene_caption', '完整图像')}"]
        if sections.get("subjects"):
            prompt_parts.append(f"主体：{sections['subjects']}")
        if sections.get("environment"):
            prompt_parts.append(f"环境：{sections['environment']}")
        if sections.get("composition"):
            prompt_parts.append(f"构图：{sections['composition']}")
        if sections.get("attachments"):
            prompt_parts.append(f"附着关系：{sections['attachments']}")
        if sections.get("connectors") and sections.get("scene_type") in {"process", "schematic"}:
            prompt_parts.append(f"结构关系：{sections['connectors']}")
        if sections.get("user_added"):
            prompt_parts.append(f"用户新增：{sections['user_added']}")
        if sections.get("edit_summary"):
            prompt_parts.append(sections["edit_summary"])
        if sections.get("region_edit_summary"):
            prompt_parts.append(f"局部编辑：{sections['region_edit_summary']}")
        return "；".join(self._dedupe_prompt_items(prompt_parts))

    def _build_image_render_prompt(
        self,
        params: Dict[str, Any],
        scene_spec: Dict[str, Any] | None = None,
        style_hint: str = "",
        prompt_suffix: str = "",
        scene_context: Dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        return self._structured_params_to_sd_prompt(
            params,
            scene_spec=scene_spec,
            style_hint=style_hint,
            prompt_suffix=prompt_suffix,
            scene_context=scene_context,
        )

    def _scene_canvas_size(self, scene_spec: Dict[str, Any] | None = None) -> tuple[int, int]:
        canvas = (scene_spec or {}).get("canvas_size", {}) if isinstance(scene_spec, dict) else {}
        width = int(canvas.get("width", 1024) or 1024)
        height = int(canvas.get("height", 768) or 768)
        return max(1, width), max(1, height)

    def _normalized_bbox(self, item: Dict[str, Any], canvas_width: int, canvas_height: int) -> Dict[str, float]:
        x = float(item.get("x", 0) or 0)
        y = float(item.get("y", 0) or 0)
        width = float(item.get("width", 0) or 0)
        height = float(item.get("height", 0) or 0)
        return {
            "x": round(x, 2),
            "y": round(y, 2),
            "width": round(width, 2),
            "height": round(height, 2),
            "x_norm": round(x / canvas_width, 4),
            "y_norm": round(y / canvas_height, 4),
            "width_norm": round(width / canvas_width, 4),
            "height_norm": round(height / canvas_height, 4),
        }

    def _render_constraints(self, scene_spec: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if not isinstance(scene_spec, dict):
            return {
                "background_layers": [],
                "object_instances": [],
                "attachments": [],
                "connectors": [],
                "depth_band_counts": {},
            }
        canvas_width, canvas_height = self._scene_canvas_size(scene_spec)
        backgrounds = []
        for layer in scene_spec.get("background_layers", []) or []:
            if not isinstance(layer, dict):
                continue
            backgrounds.append({
                "id": str(layer.get("id", "")),
                "type": str(layer.get("type", "")),
                "label": str(layer.get("label", "")),
                "bbox": self._normalized_bbox(layer, canvas_width, canvas_height),
                "z_index": int(layer.get("z_index", 0) or 0),
                "source": str(layer.get("source", "")),
            })

        objects = []
        depth_band_counts: Dict[str, int] = {}
        for obj in scene_spec.get("object_instances", []) or []:
            if not isinstance(obj, dict):
                continue
            depth_band = str(obj.get("depth_band", "") or "")
            if depth_band:
                depth_band_counts[depth_band] = depth_band_counts.get(depth_band, 0) + 1
            objects.append({
                "id": str(obj.get("id", "")),
                "concept": str(obj.get("concept", "")),
                "asset_key": str(obj.get("asset_key", "")),
                "prototype_id": str(obj.get("prototype_id", obj.get("asset_key", ""))),
                "role": str(obj.get("role", "")),
                "depth_band": depth_band,
                "depth_z": float(obj.get("depth_z", 0.0) or 0.0),
                "source": str(obj.get("source", "")),
                "bbox": self._normalized_bbox(obj, canvas_width, canvas_height),
                "rotation": float(obj.get("rotation", 0.0) or 0.0),
                "scale": float(obj.get("scale", 1.0) or 1.0),
                "visible": bool(obj.get("visible", True)),
                "z_index": int(obj.get("z_index", 0) or 0),
                "editable": bool(obj.get("editable", True)),
            })

        attachments = []
        for item in scene_spec.get("attachments", []) or []:
            if not isinstance(item, dict):
                continue
            attachments.append({
                "id": str(item.get("id", "")),
                "host_id": str(item.get("host_id", "")),
                "child_id": str(item.get("child_id", "")),
                "anchor_name": str(item.get("anchor_name", "")),
                "mode": str(item.get("mode", "")),
            })

        connectors = []
        for item in scene_spec.get("connectors", []) or []:
            if not isinstance(item, dict):
                continue
            connectors.append({
                "id": str(item.get("id", "")),
                "type": str(item.get("type", "")),
                "from_id": str(item.get("from_id", "")),
                "to_id": str(item.get("to_id", "")),
                "label": str(item.get("label", "")),
                "visible": bool(item.get("visible", True)),
            })

        return {
            "background_layers": backgrounds,
            "object_instances": objects,
            "attachments": attachments,
            "connectors": connectors,
            "depth_band_counts": depth_band_counts,
        }

    def build_render_bundle(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        scene_spec: Dict[str, Any] | None = None,
        structured_params: Dict[str, Any] | None = None,
        control_image_path: str | None = None,
        low_preview_path: str | None = None,
        render_control_path: str | None = None,
        used_control_image_path: str | None = None,
        sketch_bundle: Dict[str, Any] | None = None,
        conditioning_bundle: Dict[str, Any] | None = None,
        backend: str = "",
        final_image_path: str | None = None,
        workflow_state: str = "previewed",
    ) -> Dict[str, Any]:
        scene_spec = scene_spec if isinstance(scene_spec, dict) else {}
        sketch_bundle = sketch_bundle if isinstance(sketch_bundle, dict) else {}
        conditioning_bundle = conditioning_bundle if isinstance(conditioning_bundle, dict) else {}
        layout_options = scene_spec.get("layout_options", {}) if isinstance(scene_spec, dict) else {}
        render_hints = scene_spec.get("render_hints", {}) if isinstance(scene_spec.get("render_hints"), dict) else {}
        render_strategy = "text_only"
        if conditioning_bundle and (
            int(((conditioning_bundle.get("counts") or {}).get("edited_regions", 0) or 0)) > 0
            or int(((conditioning_bundle.get("counts") or {}).get("patches", 0) or 0)) > 0
        ):
            render_strategy = "full_state_conditioned_img2img"
        elif render_control_path:
            render_strategy = "render_control_v2"
        elif low_preview_path and used_control_image_path and os.path.normpath(str(used_control_image_path)) == os.path.normpath(str(low_preview_path)):
            render_strategy = "low_preview_bridge"
        elif control_image_path and used_control_image_path and os.path.normpath(str(used_control_image_path)) == os.path.normpath(str(control_image_path)):
            render_strategy = "semantic_sketch_bridge"
        elif used_control_image_path:
            render_strategy = "external_control_image"
        return {
            "version": 2,
            "pipeline": "render_bridge_v2",
            "workflow_state": workflow_state,
            "text_prompt": str(prompt or ""),
            "negative_prompt": str(negative_prompt or ""),
            "scene_spec_version": int(scene_spec.get("version", 2) or 2) if isinstance(scene_spec, dict) else 2,
            "composition_mode": str(layout_options.get("composition_mode", layout_options.get("scene_type", "scene"))),
            "scene_type": str(layout_options.get("scene_type", "scene")),
            "sketch_style": str(layout_options.get("sketch_style", "scribble_line")),
            "scene_summary": summarize_scene_spec(scene_spec) if scene_spec else "",
            "edit_summary": str(render_hints.get("edit_summary", "")),
            "region_edit_summary": str(render_hints.get("region_edit_summary", "")),
            "depth_summary": str(render_hints.get("depth_summary", "")),
            "camera_summary": str(render_hints.get("camera_summary", "")),
            "region_edit_constraints": _safe_scene_copy(render_hints.get("region_edit_constraints") or []),
            "conditioning_summary": str(conditioning_bundle.get("conditioning_summary", "")),
            "dual_conditioning": {
                "text": True,
                "sketch": bool(used_control_image_path or sketch_bundle),
                "mode": "text+sketch",
            },
            "visible_outputs": {
                "semantic_sketch_path": str(control_image_path or ""),
                "base_sketch_path": str(sketch_bundle.get("base_sketch") or ""),
                "structural_sketch_path": str(sketch_bundle.get("structural_sketch") or control_image_path or ""),
                "annotated_sketch_path": str(sketch_bundle.get("annotated_sketch") or ""),
                "region_overlay_path": str(sketch_bundle.get("region_overlay") or ""),
                "low_preview_path": str(low_preview_path or ""),
                "hit_map_path": str(sketch_bundle.get("hit_map") or ""),
                "annotation_bundle_path": str(sketch_bundle.get("annotation_bundle") or ""),
                "final_image_path": str(final_image_path or ""),
                "editable_sketch_composited_path": str(render_hints.get("editable_sketch_composited_path", "")),
            },
            "model_inputs": {
                "render_control_path": str(render_control_path or ""),
                "used_control_image_path": str(used_control_image_path or ""),
                "control_strategy": render_strategy,
                "backend": str(backend or ""),
                "conditioning_bundle_path": str(conditioning_bundle.get("conditioning_bundle_path", "")),
            },
            "text_constraints": {
                "scene_summary": str((scene_spec.get("render_hints", {}) or {}).get("scene_summary", "")),
                "subject_summary": str((scene_spec.get("render_hints", {}) or {}).get("subject_summary", "")),
                "edit_summary": str(render_hints.get("edit_summary", "")),
                "region_edit_summary": str(render_hints.get("region_edit_summary", "")),
                "depth_summary": str(render_hints.get("depth_summary", "")),
                "camera_summary": str(render_hints.get("camera_summary", "")),
                "conditioning_summary": str(conditioning_bundle.get("conditioning_summary", "")),
            },
            "sketch_constraints": {
                "structural_sketch_path": str(sketch_bundle.get("structural_sketch") or control_image_path or ""),
                "annotated_sketch_path": str(sketch_bundle.get("annotated_sketch") or ""),
                "region_overlay_path": str(sketch_bundle.get("region_overlay") or ""),
                "annotation_bundle_path": str(sketch_bundle.get("annotation_bundle") or ""),
                "low_preview_path": str(low_preview_path or ""),
                "render_control_path": str(render_control_path or ""),
                "render_patch_constraints": _safe_scene_copy(render_hints.get("render_patch_constraints") or []),
                "region_edit_constraints": _safe_scene_copy(render_hints.get("region_edit_constraints") or []),
                "editable_sketch_composited_path": str(render_hints.get("editable_sketch_composited_path", "")),
                "conditioning_bundle_path": str(conditioning_bundle.get("conditioning_bundle_path", "")),
            },
            "conditioning": {
                "source": str(conditioning_bundle.get("source", "")),
                "summary": str(conditioning_bundle.get("conditioning_summary", "")),
                "bundle_path": str(conditioning_bundle.get("conditioning_bundle_path", "")),
                "counts": _safe_scene_copy(conditioning_bundle.get("counts") or {}),
            },
            "annotation_bundle": _safe_scene_copy((scene_spec.get("render_hints", {}) or {}).get("annotation_bundle") or {}),
            "structured_params": structured_params or {},
            "constraints": self._render_constraints(scene_spec),
        }
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _encode_image_to_data_url(self, image_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/png"
        return f"data:{mime_type};base64,{self._encode_image_to_base64(image_path)}"

    def _get_image_api_endpoint(self) -> str:
        base_url = (self.image_api_url or "").strip().rstrip("/")
        if not base_url:
            return ""
        if base_url.endswith("/images/generations"):
            return base_url
        return f"{base_url}/images/generations"

    def _is_volc_ark_image_api(self) -> bool:
        url = (self.image_api_url or "").lower()
        return any(keyword in url for keyword in ["ark.", "volces.com", "volcengine"])

    def _default_ark_image_model(self) -> str:
        return "doubao-seedream-5-0-260128"

    def _default_ark_control_fallback_model(self) -> str:
        return "doubao-seededit-3-0-i2i-250628"

    def _infer_image_model(self, use_control_image: bool = False) -> Optional[str]:
        if not self.image_api_url:
            return None
        if self.image_api_model:
            return self.image_api_model
        if use_control_image and self.image_api_control_model:
            return self.image_api_control_model
        if self._is_volc_ark_image_api():
            return self._default_ark_image_model()
        return None

    def _infer_image_size(self, render_options: Dict[str, Any] | None = None) -> Optional[str]:
        render_options = render_options or {}
        explicit_size = str(render_options.get("image_size", "") or "").strip()
        if explicit_size:
            return explicit_size
        if self.image_api_size:
            return self.image_api_size
        if self._is_volc_ark_image_api():
            return "2K"
        return None

    def _save_generated_image(self, response_payload: Dict[str, Any], filename_prefix: str) -> str:
        data = response_payload.get("data") or []
        if not data:
            raise ValueError("图片接口未返回 data 字段")
        first_item = data[0] or {}
        image_path = os.path.join(self.output_dir, f"{filename_prefix}_{abs(hash(json.dumps(first_item, ensure_ascii=False, sort_keys=True)))}.png")

        b64_json = first_item.get("b64_json")
        if b64_json:
            with open(image_path, "wb") as output_file:
                output_file.write(base64.b64decode(b64_json))
            return image_path

        image_url = first_item.get("url")
        if image_url:
            image_response = requests.get(image_url, timeout=180)
            image_response.raise_for_status()
            with open(image_path, "wb") as output_file:
                output_file.write(image_response.content)
            return image_path

        raise ValueError("图片接口返回中既没有 url 也没有 b64_json")

    def _call_generic_image_api(self, payload: Dict[str, Any], filename_prefix: str) -> str:
        endpoint = self._get_image_api_endpoint()
        if not endpoint or not self.image_api_key:
            raise ValueError("未配置图片 API URL 或 API Key")
        headers = {
            "Authorization": f"Bearer {self.image_api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(endpoint, json=payload, headers=headers, timeout=180)
        if not response.ok:
            error_body = response.text.strip()
            raise ValueError(f"图片接口调用失败（HTTP {response.status_code}）：{error_body[:800]}")
        return self._save_generated_image(response.json(), filename_prefix)

    def _build_generic_image_payload(
        self,
        prompt: str,
        control_image_path: str | None = None,
        render_options: Dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> Dict[str, Any]:
        render_options = render_options or {}
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "response_format": "url",
        }
        if not self._is_volc_ark_image_api():
            payload["n"] = 1
        model = (model_override or "").strip() or self._infer_image_model(use_control_image=bool(control_image_path))
        if model:
            payload["model"] = model

        image_size = self._infer_image_size(render_options)
        if image_size:
            payload["size"] = image_size

        if self._is_volc_ark_image_api():
            payload["stream"] = bool(render_options.get("stream", False))
            payload["watermark"] = bool(render_options.get("watermark", True))
            payload["sequential_image_generation"] = str(
                render_options.get("sequential_image_generation", "disabled")
            ).strip() or "disabled"

        if control_image_path:
            payload["image"] = self._encode_image_to_data_url(control_image_path)

        return payload
    
    def _path_to_structured_params(self, path: 'MazePath', generation_type: str) -> Dict[str, Any]:
        """
        直接将Tri-Maze推理路径转化为多模态生成的结构化参数
        完全基于节点、关系、阻力生成，不需要自然语言中转
        :param path: Tri-Maze推理路径
        :param generation_type: 生成类型
        :return: 结构化生成参数
        """
        params = {
            "concepts": [],
            "relations": [],
            "core_elements": [],
            "secondary_elements": [],
            "style": "",
            "weights": {}
        }
        
        # 提取路径中的概念和关系，按阻力分配权重
        for i, (node, edge) in enumerate(zip(path.nodes, path.edges + [None])):
            concept = node.concept
            resistance = edge.resistance if edge else 0.0
            
            # 获取概念的视觉特征
            features = self.concept_feature_map.get(concept, {
                "shape": "object", 
                "color": ["natural"], 
                "tags": [concept.lower().replace(" ", "_")]
            })
            
            # 获取阻力对应的权重
            weight_config = self._get_weight_config(resistance)
            
            concept_info = {
                "name": concept,
                "features": features,
                "resistance": resistance,
                "weight": weight_config["weight"],
                "opacity": weight_config["opacity"],
                "size_multiplier": weight_config["size_multiplier"],
                "position": "core" if resistance < 0.4 else "secondary"
            }
            
            params["concepts"].append(concept_info)
            
            if concept_info["position"] == "core":
                params["core_elements"].append(concept)
            else:
                params["secondary_elements"].append(concept)
            
            if edge:
                params["relations"].append({
                    "from": path.nodes[i].concept,
                    "to": path.nodes[i+1].concept,
                    "relation": edge.relation,
                    "resistance": edge.resistance
                })
        
        # 根据生成类型设置风格
        if generation_type == "image":
            if len([c for c in params["core_elements"] if c in ["电路", "电阻", "LED", "Arduino"]]) > 0:
                params["style"] = "professional electronic schematic diagram, white background, clear lines, technical illustration"
            elif len([c for c in params["core_elements"] if c in ["猫", "动物", "生物"]]) > 0:
                params["style"] = "photorealistic, natural lighting, high detail"
            elif len([c for c in params["core_elements"] if c in ["机械结构", "机器", "工程"]]) > 0:
                params["style"] = "technical drawing, blueprint style, precise lines"
            else:
                params["style"] = "photorealistic, high quality, 8k"
        
        elif generation_type == "video":
            params["style"] = "smooth motion, natural transitions, high quality video"
            params["duration"] = "5 seconds"
            params["motion"] = "slow pan over the core concepts, showing the relations between them"
        
        elif generation_type == "3d_model":
            params["style"] = "3D model, PBR materials, high polygon, realistic rendering"
        
        return params
    
    def _structured_params_to_sd_prompt(
        self,
        params: Dict,
        scene_spec: Dict[str, Any] | None = None,
        style_hint: str = "",
        prompt_suffix: str = "",
        scene_context: Dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """将结构化参数和 SceneSpec 组织成更适合最终渲染的语义提示词。"""
        sections = self._scene_spec_semantic_sections(scene_spec)
        scene_type = sections.get("scene_type", "scene")
        scene_caption = sections.get("scene_caption", self._scene_type_caption(scene_type))
        query_clause = self._scene_query_clause(scene_context)

        core_elements = self._dedupe_prompt_items(list(params.get("core_elements") or []))
        secondary_elements = self._dedupe_prompt_items(list(params.get("secondary_elements") or []))
        relations = self._dedupe_prompt_items([rel.get("relation", "") for rel in params.get("relations", [])])
        style_value = self._clean_prompt_text(params.get("style") or "")
        scene_prompt = self._scene_spec_to_prompt(scene_spec)

        prompt_sections = [f"请生成一张最终可用的高质量{scene_caption}。"]
        if scene_prompt:
            prompt_sections.append(scene_prompt + "。")
        elif core_elements:
            prompt_sections.append(f"核心内容围绕{ '、'.join(core_elements[:4]) }展开。")

        if secondary_elements:
            prompt_sections.append(f"补充元素可以包含{ '、'.join(secondary_elements[:5]) }。")
        if relations and scene_type in {"process", "schematic"}:
            prompt_sections.append(f"重点表达的关系包括{ '、'.join(relations[:4]) }。")
        if style_value:
            prompt_sections.append(f"整体风格：{style_value}。")

        extra_style = self._clean_prompt_text(style_hint)
        if extra_style:
            prompt_sections.append(f"额外风格要求：{extra_style}。")
        extra_suffix = self._clean_prompt_text(prompt_suffix)
        if extra_suffix:
            prompt_sections.append(f"补充要求：{extra_suffix}。")

        prompt_sections.extend(self._scene_style_booster(scene_type, query_clause=query_clause))

        prompt_sections.append(
            "上传的控制图只用于约束构图、位置、大小占比、前后层级与附着关系，不要把控制图里的草图线条、遮罩色块、几何图标、标签文字、箭头、边框、面板或涂鸦直接渲染进最终画面。"
        )
        if scene_type == "scene":
            prompt_sections.append("画面要自然完整，主体清晰，避免示意图感、流程图感和拼贴感。")
        elif scene_type == "process":
            prompt_sections.append("请把机制过程转成清晰的阶段式图解，用可识别对象和空间层次来表达，不要退回脑图式方框。")
        else:
            prompt_sections.append("请保持技术结构清楚，用元件与空间布局表达结构，不要生成文字节点图。")

        prompt = " ".join(self._dedupe_prompt_items(prompt_sections))

        negative_items = [
            "sketch lines",
            "doodle overlay",
            "wireframe",
            "blueprint",
            "diagram",
            "mind map",
            "labels",
            "text",
            "arrows",
            "boxes",
            "panels",
            "colored masks",
            "watermark",
            "low quality",
            "blurry",
            "distorted",
            "duplicate objects",
            "messy composition",
            "草图线稿",
            "节点框",
            "关系箭头",
            "文字标签",
            "面板框",
            "遮罩色块入镜",
        ]
        if scene_type == "scene":
            negative_items.extend(["flat collage", "sticker style", "UI screenshot"])
        elif scene_type == "process":
            negative_items.extend(["flowchart boxes", "presentation slide", "ppt slide", "ui cards", "large rounded cards"])
        elif scene_type == "schematic":
            negative_items.extend(
                [
                    "mind map layout",
                    "cartoon poster",
                    "pcb board photo",
                    "green motherboard",
                    "microchip macro",
                    "electronic product photography",
                    "realistic PCB",
                    "芯片微距",
                    "主板照片",
                    "绿色电路板实拍",
                ]
            )
        negative_prompt = ", ".join(self._dedupe_prompt_items(negative_items))
        return prompt, negative_prompt
    
    async def _call_stable_diffusion(self, params: Dict) -> Optional[str]:
        """调用Stable Diffusion API生成图片，基于结构化参数"""
        try:
            prompt, negative_prompt = self._structured_params_to_sd_prompt(params)
            logger.info(f"🎨 基于推理路径生成SD Prompt: {prompt[:150]}...")

            image_path = self.sd_sketch_generator.render_txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=1024,
                height=1024,
                steps=25,
                cfg_scale=7.0,
                sampler_name="DPM++ 2M Karras",
                filename_prefix="image",
            )
            logger.info(f"✅ 图片生成成功，保存到: {image_path}")
            return image_path, prompt
        except Exception as e:
            logger.error(f"❌ Stable Diffusion调用失败: {str(e)}")
            return None, None
    
    async def _call_stable_diffusion_img2img(self, prompt: str, negative_prompt: str, control_image_path: str, render_options: Dict[str, Any] | None = None):
        """调用 Stable Diffusion img2img，将控制草图转成最终图片"""
        render_options = render_options or {}
        scene_spec = render_options.get("scene_spec") if isinstance(render_options.get("scene_spec"), dict) else {}
        layout_options = scene_spec.get("layout_options", {}) if isinstance(scene_spec, dict) else {}
        scene_type = str(layout_options.get("scene_type") or layout_options.get("composition_mode") or "scene").strip().lower() or "scene"
        default_strength = {"scene": 0.42, "process": 0.34, "schematic": 0.3}.get(scene_type, 0.35)
        default_steps = {"scene": 30, "process": 28, "schematic": 26}.get(scene_type, 28)
        default_cfg = {"scene": 7.0, "process": 6.4, "schematic": 6.0}.get(scene_type, 7.0)
        controlnet_bundle = self.sd_sketch_generator.build_controlnet_bundle(
            control_image_path=control_image_path,
            scene_spec=scene_spec,
            filename_prefix="controlled_image",
            purpose="final_render",
        )
        image_path = self.sd_sketch_generator.render_img2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image_path=control_image_path,
            denoising_strength=float(render_options.get("control_strength", default_strength)),
            steps=int(render_options.get("steps", default_steps)),
            cfg_scale=float(render_options.get("cfg_scale", default_cfg)),
            sampler_name=str(render_options.get("sampler_name", "DPM++ 2M Karras")),
            filename_prefix="controlled_image",
            conditioning_bundle=render_options.get("conditioning_bundle") if isinstance(render_options.get("conditioning_bundle"), dict) else None,
            controlnet_bundle=controlnet_bundle,
        )
        logger.info(f"✅ 草图约束图片生成成功，保存到: {image_path}")
        return image_path
    
    async def preview_controlled_image(
        self,
        reasoning_path: 'MazePath | None',
        sketch_options: Dict[str, Any] | None = None,
        scene_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """生成前置控制草图与结构化场景说明"""
        sketch_options = sketch_options or {}
        canvas_size = self._get_canvas_size(sketch_options)
        params = self._path_to_structured_params(reasoning_path, "image") if reasoning_path else {}
        scene_spec = self.native_generator.build_scene_spec(
            reasoning_path,
            canvas_size=canvas_size,
            sketch_options=sketch_options,
            scene_context=scene_context,
        )
        hints = scene_spec.get("render_hints", {}) if isinstance(scene_spec, dict) else {}
        title_core = hints.get("scene_summary") or " · ".join(scene_spec.get("concept_order", [])[:4])
        title = f"Tri-Maze 语义草图 · {title_core}" if title_core else "Tri-Maze 语义草图"
        preview = self.render_scene_spec_preview(
            scene_spec,
            sketch_options=sketch_options,
            title=title,
        )
        prompt, negative_prompt = self._build_image_render_prompt(
            params,
            preview.get("scene_spec"),
            style_hint=sketch_options.get("style_hint", ""),
            prompt_suffix=sketch_options.get("prompt_suffix", ""),
            scene_context=scene_context,
        )
        preview["structured_params"] = params
        preview["generated_prompt"] = prompt
        preview["negative_prompt"] = negative_prompt
        preview.setdefault("backend", "tri_maze_control_preview")
        used_control_image_path = self.resolve_used_control_image_path(
            sketch_options=sketch_options,
            preview=preview,
            control_image_path=preview.get("image_path"),
            low_preview_path=preview.get("low_preview_path"),
            render_control_path=preview.get("render_control_path"),
        )
        preview["render_bundle"] = self.build_render_bundle(
            prompt=prompt,
            negative_prompt=negative_prompt,
            scene_spec=preview.get("scene_spec"),
            structured_params=params,
            control_image_path=preview.get("image_path"),
            low_preview_path=preview.get("low_preview_path"),
            render_control_path=preview.get("render_control_path"),
            used_control_image_path=used_control_image_path,
            sketch_bundle=preview.get("sketch_bundle"),
            backend="tri_maze_control_preview",
            workflow_state="composed",
        )
        return preview
    
    async def render_image_from_preview(
        self,
        reasoning_path: 'MazePath | None',
        sketch_options: Dict[str, Any] | None = None,
        render_options: Dict[str, Any] | None = None,
        scene_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """基于控制草图调用外部图片 API 生成最终图片"""
        sketch_options = sketch_options or {}
        render_options = render_options or {}
        preview_result = None
        control_image_path = render_options.get("control_image_path")
        scene_spec = render_options.get("scene_spec")
        low_preview_path = render_options.get("low_preview_path")
        render_control_path = render_options.get("render_control_path")
        sketch_bundle = render_options.get("sketch_bundle") if isinstance(render_options.get("sketch_bundle"), dict) else None
        external_control_image_path = self.resolve_used_control_image_path(
            sketch_options=sketch_options,
            render_options=render_options,
            control_image_path=control_image_path,
            low_preview_path=low_preview_path,
            render_control_path=render_control_path,
        )
        if self._resolve_sketch_backend(sketch_options) == "sketch_v2" and external_control_image_path:
            control_image_path = external_control_image_path
        if not control_image_path:
            preview_result = await self.preview_controlled_image(
                reasoning_path,
                sketch_options,
                scene_context=scene_context,
            )
            control_image_path = preview_result.get("image_path")
            scene_spec = preview_result.get("scene_spec")
            low_preview_path = preview_result.get("low_preview_path")
            render_control_path = preview_result.get("render_control_path")
            sketch_bundle = preview_result.get("sketch_bundle")
            external_control_image_path = self.resolve_used_control_image_path(
                sketch_options=sketch_options,
                preview=preview_result,
                render_options=render_options,
                control_image_path=control_image_path,
                low_preview_path=low_preview_path,
                render_control_path=render_control_path,
            )
            if self._resolve_sketch_backend(sketch_options) == "sketch_v2" and external_control_image_path:
                control_image_path = external_control_image_path
        params = self._path_to_structured_params(reasoning_path, "image") if reasoning_path else {}
        prompt, default_negative = self._build_image_render_prompt(
            params,
            scene_spec,
            style_hint=render_options.get("style_hint", sketch_options.get("style_hint", "")),
            prompt_suffix=render_options.get("prompt_suffix", sketch_options.get("prompt_suffix", "")),
            scene_context=scene_context,
        )
        negative_prompt = render_options.get("negative_prompt") or default_negative
        conditioning_bundle = self._build_render_conditioning_bundle(
            render_options=render_options,
            scene_spec=scene_spec,
            used_control_image_path=external_control_image_path or control_image_path,
        )
        if conditioning_bundle:
            render_options["conditioning_bundle"] = conditioning_bundle
        image_path = None
        backend = None
        note = ""
        
        try:
            if self.sd_api_url and "http" in self.sd_api_url and control_image_path:
                image_path = await self._call_stable_diffusion_img2img(prompt, negative_prompt, control_image_path, render_options)
                backend = "comfyui_img2img" if self.sd_sketch_generator.comfy_client.is_comfyui_server() else "stable_diffusion_img2img"
                conditioning_report = self.sd_sketch_generator.last_conditioning_report if isinstance(self.sd_sketch_generator.last_conditioning_report, dict) else {}
                applied_ops = [
                    item for item in (conditioning_report.get("operations") or [])
                    if isinstance(item, dict) and item.get("status") == "applied"
                ]
                if applied_ops:
                    note = f"已应用 {len(applied_ops)} 个局部 conditioning pass（region/patch 级）到最终渲染。"
        except Exception as e:
            logger.error(f"草图约束渲染失败: {str(e)}")
            note = f"Stable Diffusion img2img 调用失败：{str(e)}"
        
        if not image_path and self.image_api_url and self.image_api_key:
            try:
                primary_model = self._infer_image_model(use_control_image=bool(external_control_image_path))
                payload = self._build_generic_image_payload(
                    prompt,
                    external_control_image_path,
                    render_options,
                    model_override=primary_model,
                )
                image_path = self._call_generic_image_api(payload, "generic_controlled")
                backend = "volc_ark_image_api" if self._is_volc_ark_image_api() else "generic_image_api"
                if external_control_image_path and self._is_volc_ark_image_api():
                    if note:
                        note += " | "
                    note += f"已使用火山方舟图片接口进行构图约束渲染（model={payload.get('model', '')}）。"
            except Exception as e:
                logger.error(f"通用图片接口草图约束渲染失败: {str(e)}")
                if note:
                    note += " | "
                note += f"通用图片接口渲染失败：{str(e)}"

        if (
            not image_path
            and external_control_image_path
            and self.image_api_url
            and self.image_api_key
            and self._is_volc_ark_image_api()
        ):
            fallback_model = self.image_api_control_model.strip() or self._default_ark_control_fallback_model()
            primary_model = self._infer_image_model(use_control_image=True)
            if fallback_model and fallback_model != primary_model:
                try:
                    payload = self._build_generic_image_payload(
                        prompt,
                        external_control_image_path,
                        render_options,
                        model_override=fallback_model,
                    )
                    image_path = self._call_generic_image_api(payload, "ark_control_fallback")
                    backend = "volc_ark_image_api_control_fallback"
                    if note:
                        note += " | "
                    note += f"主模型不接受当前草图约束时，已回退到控制图模型（model={fallback_model}）。"
                except Exception as e:
                    logger.error(f"火山方舟控制图回退模型渲染失败: {str(e)}")
                    if note:
                        note += " | "
                    note += f"火山方舟控制图回退模型失败：{str(e)}"

        if not image_path and self.image_api_url and self.image_api_key:
            try:
                payload = self._build_generic_image_payload(prompt, None, render_options)
                image_path = self._call_generic_image_api(payload, "generic_prompt_only")
                backend = "volc_ark_text_fallback" if self._is_volc_ark_image_api() else "generic_image_api_text_fallback"
                if note:
                    note += " | "
                note += "外部图片接口未使用控制草图，仅复用结构化场景提示。"
            except Exception as e:
                logger.error(f"通用图片接口文本渲染失败: {str(e)}")
                if note:
                    note += " | "
                note += f"通用图片接口文本渲染失败：{str(e)}"

        if not image_path and self.dalle_api_key:
            client = OpenAI(api_key=self.dalle_api_key)
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            image_response = requests.get(image_url, timeout=180)
            image_path = f"{self.output_dir}/dalle_controlled_{hash(prompt)}.png"
            with open(image_path, "wb") as output_file:
                output_file.write(image_response.content)
            backend = "dall_e_text_fallback"
            if note:
                note += " | "
            note += "DALL-E 兜底不直接读取草图，仅复用结构化场景提示。"

        if not image_path and scene_spec:
            try:
                native_result = self.native_generator.render_scene_spec_preview(
                    scene_spec,
                    sketch_options=sketch_options,
                    title="Tri-Maze 本地结构化渲染",
                )
                image_path = native_result.get("low_preview_path") or native_result.get("image_path")
                control_image_path = control_image_path or native_result.get("image_path")
                low_preview_path = native_result.get("low_preview_path") or low_preview_path
                scene_spec = native_result.get("scene_spec", scene_spec)
                sketch_bundle = native_result.get("sketch_bundle") or sketch_bundle
                backend = "native_scene_spec_fallback"
                if note:
                    note += " | "
                note += "当前未配置外部生图 API，已回退为本地结构化渲染图，可继续编辑 SceneSpec 后再重渲染。"
            except Exception as e:
                logger.error(f"本地结构化兜底渲染失败: {str(e)}")
                if note:
                    note += " | "
                note += f"本地结构化兜底渲染失败：{str(e)}"
        
        if not image_path:
            return {
                "success": False,
                "error": note or "未配置可用的外部图片渲染 API",
                "control_image_path": control_image_path,
                "used_control_image_path": external_control_image_path,
                "low_preview_path": low_preview_path,
                "render_control_path": render_control_path,
                "scene_spec": scene_spec,
                "generated_prompt": prompt,
                "negative_prompt": negative_prompt,
                "render_bundle": self.build_render_bundle(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    scene_spec=scene_spec,
                    structured_params=params,
                    control_image_path=control_image_path,
                    low_preview_path=low_preview_path,
                    render_control_path=render_control_path,
                    used_control_image_path=external_control_image_path,
                    sketch_bundle=sketch_bundle or (preview_result or {}).get("sketch_bundle"),
                    conditioning_bundle=conditioning_bundle,
                    backend=backend or "",
                    workflow_state="render_failed",
                ),
            }
        
        result = {
            "success": True,
            "type": "controlled_image",
            "backend": backend,
            "image_path": image_path,
            "save_path": image_path,
            "control_image_path": control_image_path,
            "used_control_image_path": external_control_image_path,
            "low_preview_path": low_preview_path,
            "render_control_path": render_control_path,
            "scene_spec": scene_spec,
            "structured_params": params,
            "generated_prompt": prompt,
            "negative_prompt": negative_prompt,
            "sketch_bundle": sketch_bundle,
            "description": "先生成 Tri-Maze 控制草图，再交给外部图片 API 或本地结构化渲染生成最终图片",
            "render_bundle": self.build_render_bundle(
                prompt=prompt,
                negative_prompt=negative_prompt,
                scene_spec=scene_spec,
                structured_params=params,
                control_image_path=control_image_path,
                low_preview_path=low_preview_path,
                render_control_path=render_control_path,
                used_control_image_path=external_control_image_path,
                sketch_bundle=sketch_bundle or (preview_result or {}).get("sketch_bundle"),
                conditioning_bundle=conditioning_bundle,
                backend=backend or "",
                final_image_path=image_path,
                workflow_state="rendered",
            ),
        }
        if note:
            result["note"] = note
        if preview_result:
            result["control_preview"] = preview_result
        return result
    async def _generate_image(self, reasoning_path: 'MazePath') -> Dict[str, Any]:
        """生成图片：完全基于Tri-Maze推理路径的结构化参数生成，不需要自然语言Prompt"""
        try:
            # 直接将推理路径转化为结构化参数
            params = self._path_to_structured_params(reasoning_path, "image")
            
            # 尝试调用SD生成真实图片
            image_path = None
            generated_prompt = None
            if self.sd_api_url and "http" in self.sd_api_url:
                image_path, generated_prompt = await self._call_stable_diffusion(params)
            
            if not image_path and self.image_api_url and self.image_api_key:
                prompt_desc = f"Generate a high quality image of: {', '.join(params['core_elements'])}. The image should show the relations: {', '.join([r['relation'] for r in params['relations']])}. Style: {params['style']}"
                payload = self._build_generic_image_payload(prompt_desc)
                image_path = self._call_generic_image_api(payload, "generic_image")
                generated_prompt = prompt_desc
                logger.info(f"✅ 通用图片接口生成成功，保存到: {image_path}")

            # 尝试DALL-E
            if not image_path and self.dalle_api_key:
                # 将结构化参数转化为DALL-E Prompt
                prompt_desc = f"Generate a high quality image of: {', '.join(params['core_elements'])}. The image should show the relations: {', '.join([r['relation'] for r in params['relations']])}. Style: {params['style']}"
                
                client = OpenAI(api_key=self.dalle_api_key)
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt_desc,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
                
                image_url = response.data[0].url
                image_response = requests.get(image_url)
                image_path = f"{self.output_dir}/image_{hash(str(params))}.png"
                with open(image_path, "wb") as f:
                    f.write(image_response.content)
                generated_prompt = prompt_desc
                logger.info(f"✅ DALL-E图片生成成功，保存到: {image_path}")
            
            result = {
                "success": True,
                "type": "image",
                "structured_params": params,
                "generated_prompt": generated_prompt,
                "description": "基于Tri-Maze推理链路直接生成的图片"
            }
            
            if image_path:
                result["image_path"] = image_path
                result["save_path"] = image_path
            else:
                result["note"] = "未配置图片生成API，已生成结构化参数和Prompt，可直接用于生成"
            
            return result
            
        except Exception as e:
            logger.error(f"生成图片失败：{str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _generate_video(self, reasoning_path: 'MazePath') -> Dict[str, Any]:
        """生成视频：基于Tri-Maze推理路径生成结构化参数"""
        try:
            # 直接将推理路径转化为结构化参数
            params = self._path_to_structured_params(reasoning_path, "video")
            
            # 生成视频Prompt
            prompt = f"Video showing: {', '.join(params['core_elements'])}. Motion: {params['motion']}. Style: {params['style']}. Duration: {params['duration']}."
            
            # 这里可以对接Pika/Runway API
            video_path = None
            # if self.pika_api_key:
            #     video_path = await self._call_pika(params)
            
            result = {
                "success": True,
                "type": "video",
                "structured_params": params,
                "generated_prompt": prompt,
                "description": "基于Tri-Maze推理链路直接生成的视频"
            }
            
            if video_path:
                result["video_path"] = video_path
                result["save_path"] = video_path
            else:
                result["note"] = "未配置视频生成API，已生成结构化参数和Prompt，可直接用于Pika/Runway生成"
            
            return result
            
        except Exception as e:
            logger.error(f"生成视频失败：{str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _generate_code(self, reasoning_path: 'MazePath') -> Dict[str, Any]:
        """生成代码：基于Tri-Maze推理路径的逻辑生成"""
        if not self.llm_client:
            return {"success": False, "error": "缺少LLM客户端"}
        
        try:
            # 提取路径中的逻辑关系
            path_concepts = [n.concept for n in reasoning_path.nodes]
            path_relations = [e.relation for e in reasoning_path.edges]
            
            # 构建代码生成逻辑提示，基于推理路径
            logic_desc = "Implement code based on the following logical path:\n"
            for i in range(len(path_relations)):
                logic_desc += f"- {path_concepts[i]} → {path_relations[i]} → {path_concepts[i+1]}\n"
            
            prompt = f"""
{logic_desc}

Generate complete, runnable code that implements this logic. Add necessary comments. The code should directly reflect the logical relationships in the path.
"""
            
            code_resp = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            code_content = code_resp.choices[0].message.content.strip()
            
            # 保存代码到文件
            code_path = f"{self.output_dir}/code_{hash(str(reasoning_path.get_concept_list()))}.py"
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code_content)
            
            return {
                "success": True,
                "type": "code",
                "content": code_content,
                "save_path": code_path,
                "logic_path": path_concepts,
                "description": "基于Tri-Maze推理链路逻辑生成的可运行代码"
            }
            
        except Exception as e:
            logger.error(f"生成代码失败：{str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _generate_document(self, reasoning_path: 'MazePath') -> Dict[str, Any]:
        """生成文档：基于Tri-Maze推理路径的结构生成"""
        if not self.llm_client:
            return {"success": False, "error": "缺少LLM客户端"}
        
        try:
            # 提取路径中的逻辑结构
            path_concepts = [n.concept for n in reasoning_path.nodes]
            path_relations = [e.relation for e in reasoning_path.edges]
            
            # 构建文档结构，基于推理路径
            structure = f"Document structure based on reasoning path:\n"
            for i, concept in enumerate(path_concepts):
                structure += f"## {i+1}. {concept}\n"
                if i < len(path_relations):
                    structure += f"   关系：{path_relations[i]}\n"
            
            prompt = f"""
{structure}

Generate a complete, well-structured Markdown document based on this structure. The document should follow the logical flow of the reasoning path.
"""
            
            doc_resp = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=3000
            )
            doc_content = doc_resp.choices[0].message.content.strip()
            
            # 保存文档到文件
            doc_path = f"{self.output_dir}/doc_{hash(str(reasoning_path.get_concept_list()))}.md"
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(doc_content)
            
            return {
                "success": True,
                "type": "document",
                "content": doc_content,
                "save_path": doc_path,
                "structure": path_concepts,
                "description": "基于Tri-Maze推理链路结构生成的文档"
            }
            
        except Exception as e:
            logger.error(f"生成文档失败：{str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _generate_design(self, reasoning_path: 'MazePath') -> Dict[str, Any]:
        """生成设计：基于Tri-Maze推理路径生成设计说明"""
        try:
            params = self._path_to_structured_params(reasoning_path, "design")
            
            design_desc = f"""
Design Description:
Core Elements: {', '.join(params['core_elements'])}
Relations: {', '.join([r['relation'] for r in params['relations']])}
Style: {params['style']}
Element Weights: { {c['name']: c['weight'] for c in params['concepts']} }
"""
            
            design_path = f"{self.output_dir}/design_{hash(str(reasoning_path.get_concept_list()))}.md"
            with open(design_path, "w", encoding="utf-8") as f:
                f.write(design_desc)
            
            return {
                "success": True,
                "type": "design",
                "content": design_desc,
                "save_path": design_path,
                "structured_params": params,
                "description": "基于Tri-Maze推理链路生成的设计说明"
            }
            
        except Exception as e:
            logger.error(f"生成设计失败：{str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _generate_audio(self, reasoning_path: 'MazePath') -> Dict[str, Any]:
        """生成音频：基于Tri-Maze推理路径生成音频参数"""
        try:
            # 将推理路径转化为音频参数
            path_concepts = [n.concept for n in reasoning_path.nodes]
            total_resistance = reasoning_path.total_resistance
            
            # 阻力映射到音频参数
            tempo = max(60, min(180, 120 + (0.5 - total_resistance) * 60))  # 阻力越低节奏越快
            pitch = 440 + (0.5 - total_resistance) * 220  # 阻力越低音调越高
            
            prompt = f"Audio representing: {', '.join(path_concepts)}. Tempo: {tempo} BPM. Pitch: {pitch} Hz. Style: ambient sound effects."
            
            result = {
                "success": True,
                "type": "audio",
                "structured_params": {
                    "tempo": tempo,
                    "pitch": pitch,
                    "concepts": path_concepts
                },
                "generated_prompt": prompt,
                "description": "基于Tri-Maze推理链路生成的音频参数"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"生成音频失败：{str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _generate_3d_model(self, reasoning_path: 'MazePath') -> Dict[str, Any]:
        """生成3D模型：基于Tri-Maze推理路径生成3D参数"""
        try:
            params = self._path_to_structured_params(reasoning_path, "3d_model")
            
            model_desc = f"""
3D Model Description:
Core Elements: {', '.join(params['core_elements'])}
Relations: {', '.join([r['relation'] for r in params['relations']])}
Style: {params['style']}
Element Sizes: { {c['name']: c['size_multiplier'] for c in params['concepts']} }
"""
            
            model_path = f"{self.output_dir}/3d_{hash(str(reasoning_path.get_concept_list()))}.md"
            with open(model_path, "w", encoding="utf-8") as f:
                f.write(model_desc)
            
            return {
                "success": True,
                "type": "3d_model",
                "content": model_desc,
                "save_path": model_path,
                "structured_params": params,
                "description": "基于Tri-Maze推理链路生成的3D模型说明"
            }
            
        except Exception as e:
            logger.error(f"生成3D模型失败：{str(e)}")
            return {"success": False, "error": str(e)}
    
    async def generate(self, reasoning_path: 'MazePath', generation_type: str = "auto") -> Dict[str, Any]:
        """
        生成多模态产物
        :param reasoning_path: Tri-Maze推理路径对象（包含节点、关系、阻力）
        :param generation_type: 生成类型：auto/image/video/code/document/design/audio/3d_model
        :return: 生成结果
        """
        # 自动判断生成类型
        if generation_type == "auto":
            generation_type = self._detect_generation_type(str(reasoning_path))
            logger.info(f"自动识别生成类型：{generation_type}")
        
        # 优先使用原生生成（不需要外部API）
        if self.use_native_generation and generation_type in ["image", "video"]:
            logger.info(f"使用Tri-Maze原生生成器生成{generation_type}，无外部API依赖")
            return await self.native_generator.generate(reasoning_path, generation_type)
        
        # 否则使用外部API生成
        generators = {
            "image": self._generate_image,
            "video": self._generate_video,
            "code": self._generate_code,
            "document": self._generate_document,
            "design": self._generate_design,
            "audio": self._generate_audio,
            "3d_model": self._generate_3d_model
        }
        
        if generation_type not in generators:
            return {"success": False, "error": f"不支持的生成类型：{generation_type}"}
        
        generator = generators[generation_type]
        return await generator(reasoning_path)
    
    def _detect_generation_type(self, query: str) -> str:
        """自动识别用户需要的生成类型"""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ["画", "图片", "图像", "绘画", "生成图", "插图", "海报", "设计图"]):
            return "image"
        elif any(keyword in query_lower for keyword in ["视频", "动画", "短片", "mv", "video"]):
            return "video"
        elif any(keyword in query_lower for keyword in ["代码", "程序", "脚本", "python", "java", "c++", "编程"]):
            return "code"
        elif any(keyword in query_lower for keyword in ["文档", "报告", "方案", "说明书", "markdown", "文章", "论文"]):
            return "document"
        elif any(keyword in query_lower for keyword in ["设计", "ui", "界面", "原型", "平面设计", "视觉设计"]):
            return "design"
        elif any(keyword in query_lower for keyword in ["音频", "声音", "音乐", "配音", "音效", "audio"]):
            return "audio"
        elif any(keyword in query_lower for keyword in ["3d", "模型", "三维", "建模", "blender", "maya"]):
            return "3d_model"
        else:
            return "document"  # 默认生成文档
    
    def get_supported_types(self) -> List[str]:
        """获取支持的生成类型"""
        return ["image", "video", "code", "document", "design", "audio", "3d_model", "auto"]
    
    def set_sd_api(self, api_url: str):
        """设置Stable Diffusion API地址"""
        self.sd_api_url = api_url
        self.sd_sketch_generator.set_sd_api_url(api_url)
        self.sketch_v2_generator.set_sd_api_url(api_url)
        logger.info(f"✅ Stable Diffusion API已设置: {api_url}")

    def set_image_api_url(self, api_url: str):
        """设置通用图片 API 地址"""
        self.image_api_url = api_url.strip()
        self.sketch_v2_generator.set_image_api_url(self.image_api_url)
        logger.info(f"✅ 通用图片 API URL已设置: {self.image_api_url}")

    def set_image_api_key(self, api_key: str):
        """设置通用图片 API Key"""
        self.image_api_key = api_key.strip()
        self.sketch_v2_generator.set_image_api_key(self.image_api_key)
        logger.info("✅ 通用图片 API Key已设置")

    def set_image_api_model(self, model_name: str):
        """设置通用图片 API 主模型"""
        self.image_api_model = model_name.strip()
        self.sketch_v2_generator.set_image_api_model(self.image_api_model)
        logger.info(f"✅ 通用图片 API 主模型已设置: {self.image_api_model}")

    def set_image_api_control_model(self, model_name: str):
        """设置通用图片 API 控制图回退模型"""
        self.image_api_control_model = model_name.strip()
        logger.info(f"✅ 通用图片 API 控制图回退模型已设置: {self.image_api_control_model}")

    def set_image_api_size(self, image_size: str):
        """设置通用图片 API 目标尺寸"""
        self.image_api_size = image_size.strip()
        self.sketch_v2_generator.set_image_api_size(self.image_api_size)
        logger.info(f"✅ 通用图片 API 图片尺寸已设置: {self.image_api_size}")
    
    def set_dalle_api_key(self, api_key: str):
        """设置DALL-E API Key"""
        self.dalle_api_key = api_key
        logger.info(f"✅ DALL-E API Key已设置")
    
    def set_pika_api_key(self, api_key: str):
        """设置Pika API Key"""
        self.pika_api_key = api_key
        logger.info(f"✅ Pika API Key已设置")


