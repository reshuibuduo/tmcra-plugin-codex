from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter


DEPTH_BAND_Z = {
    "background": 0.18,
    "midground": 0.52,
    "foreground": 0.84,
}

DEFAULT_REGION_TRANSFORM = {
    "tx": 0.0,
    "ty": 0.0,
    "scale_x": 1.0,
    "scale_y": 1.0,
    "rotation": 0.0,
}

SUPPORTED_OPS = [
    "transform_object",
    "set_depth",
    "hide_object",
    "show_object",
    "reorder_layer",
    "transform_region",
    "hide_region",
    "show_region",
    "replace_region",
    "emphasize_region",
    "weaken_region",
    "restore_region",
    "erase_region",
    "brush_mask",
    "inpaint_region",
]

REGION_ONLY_OPS = [
    "transform_region",
    "hide_region",
    "show_region",
    "replace_region",
    "emphasize_region",
    "weaken_region",
    "restore_region",
    "inpaint_region",
]


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _slug(value: Any) -> str:
    raw = "".join(ch if str(ch).isalnum() else "_" for ch in str(value or "").strip())
    raw = raw.strip("_")
    return raw[:64] or "item"


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp_rect(rect: Dict[str, Any], width: int, height: int) -> Dict[str, int]:
    x = max(0, min(width - 1, _int(rect.get("x"), 0)))
    y = max(0, min(height - 1, _int(rect.get("y"), 0)))
    w = max(1, _int(rect.get("width"), 1))
    h = max(1, _int(rect.get("height"), 1))
    if x + w > width:
        w = max(1, width - x)
    if y + h > height:
        h = max(1, height - y)
    return {"x": x, "y": y, "width": w, "height": h}


def _band_depth(depth_band: str, depth_z: Any = None) -> float:
    if depth_z is not None:
        try:
            return max(0.0, min(1.0, float(depth_z)))
        except Exception:
            pass
    return float(DEPTH_BAND_Z.get(str(depth_band or "midground"), DEPTH_BAND_Z["midground"]))


def _new_revision_id() -> str:
    return f"esk_{uuid.uuid4().hex[:12]}"


def _default_region_transform() -> Dict[str, float]:
    return dict(DEFAULT_REGION_TRANSFORM)


def _candidate_from_preview(preview: Dict[str, Any], candidate_id: str | None = None) -> Dict[str, Any] | None:
    candidates = preview.get("sketch_candidates")
    if not isinstance(candidates, list):
        candidates = []
    marker = str(candidate_id or preview.get("active_sketch_candidate_id") or "").strip()
    if marker:
        for item in candidates:
            if isinstance(item, dict) and str(item.get("candidate_id") or "").strip() == marker:
                return _copy(item)
    active_path = str(preview.get("active_sketch_path") or preview.get("image_path") or "").strip()
    active_provider = str(preview.get("active_sketch_provider") or "").strip()
    if active_path:
        for item in candidates:
            if isinstance(item, dict) and str(item.get("image_path") or "").strip() == active_path:
                return _copy(item)
    if active_path:
        return {
            "candidate_id": marker or "active",
            "provider": active_provider or "unknown",
            "image_path": active_path,
        }
    for item in candidates:
        if isinstance(item, dict) and str(item.get("image_path") or "").strip():
            return _copy(item)
    return None


def _estimate_background_rgb(image: Image.Image, *, stride: int = 4) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    border = max(1, min(width, height) // 18)
    samples: List[tuple[int, int, int, int]] = []

    def _push_pixel(px: int, py: int) -> None:
        r, g, b = rgb.getpixel((px, py))
        luminance = int((r + g + b) / 3)
        samples.append((luminance, r, g, b))

    for x in range(0, width, max(1, stride)):
        for y in range(0, min(height, border), max(1, stride)):
            _push_pixel(x, y)
        for y in range(max(0, height - border), height, max(1, stride)):
            _push_pixel(x, y)
    for y in range(border, max(border, height - border), max(1, stride)):
        for x in range(0, min(width, border), max(1, stride)):
            _push_pixel(x, y)
        for x in range(max(0, width - border), width, max(1, stride)):
            _push_pixel(x, y)

    if not samples:
        return (248, 246, 241)
    samples.sort(key=lambda item: item[0], reverse=True)
    keep = samples[: max(1, len(samples) // 3)]
    count = max(1, len(keep))
    return (
        int(round(sum(item[1] for item in keep) / count)),
        int(round(sum(item[2] for item in keep) / count)),
        int(round(sum(item[3] for item in keep) / count)),
    )


def _white_alpha_from_crop(crop: Image.Image) -> Image.Image:
    rgb = crop.convert("RGB")
    bg_r, bg_g, bg_b = _estimate_background_rgb(rgb, stride=3)
    background_luma = int((bg_r + bg_g + bg_b) / 3)
    alpha = Image.new("L", crop.size, 0)
    src = rgb.load()
    dst = alpha.load()
    width, height = crop.size
    for x in range(width):
        for y in range(height):
            r, g, b = src[x, y]
            luminance = int((r + g + b) / 3)
            contrast = max(
                0,
                background_luma - luminance,
                abs(r - bg_r),
                abs(g - bg_g),
                abs(b - bg_b),
            )
            if contrast <= 12:
                value = 0
            elif contrast >= 60:
                value = 255
            else:
                value = int((contrast - 12) * 255 / 48)
            dst[x, y] = value
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.8))
    alpha = alpha.point(lambda px: 0 if px < 14 else min(255, int(px * 1.08)))
    return alpha


def _expand_box(box: Dict[str, int], canvas_width: int, canvas_height: int, padding: int) -> Dict[str, int]:
    return _clamp_rect(
        {
            "x": box["x"] - padding,
            "y": box["y"] - padding,
            "width": box["width"] + padding * 2,
            "height": box["height"] + padding * 2,
        },
        canvas_width,
        canvas_height,
    )


def _shape_mask(width: int, height: int, shape: str) -> Image.Image:
    mask = Image.new("L", (max(1, width), max(1, height)), 0)
    draw = ImageDraw.Draw(mask)
    shape_name = str(shape or "rect").strip().lower()
    if shape_name == "ellipse":
        draw.ellipse([0, 0, max(0, width - 1), max(0, height - 1)], fill=255)
    else:
        draw.rounded_rectangle([0, 0, max(0, width - 1), max(0, height - 1)], radius=max(2, int(min(width, height) * 0.08)), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=0.8))


def _transform_is_default(transform: Dict[str, Any] | None) -> bool:
    transform = transform if isinstance(transform, dict) else {}
    return (
        abs(_float(transform.get("tx"), 0.0)) < 0.01
        and abs(_float(transform.get("ty"), 0.0)) < 0.01
        and abs(_float(transform.get("scale_x"), 1.0) - 1.0) < 0.01
        and abs(_float(transform.get("scale_y"), 1.0) - 1.0) < 0.01
        and abs(_float(transform.get("rotation"), 0.0)) < 0.01
    )


def _normalize_region_action(value: str) -> str:
    raw = str(value or "").strip().lower()
    mapping = {
        "transform_region": "transform",
        "hide_region": "hide",
        "show_region": "show",
        "replace_region": "replace",
        "emphasize_region": "emphasize",
        "weaken_region": "weaken",
        "restore_region": "restore",
        "inpaint_region": "inpaint",
    }
    return mapping.get(raw, raw or "idle")


def _region_action_text(action: str) -> str:
    mapping = {
        "transform": "移动或变形",
        "hide": "隐藏",
        "show": "显示",
        "replace": "替换",
        "emphasize": "强调",
        "weaken": "弱化",
        "restore": "恢复",
        "inpaint": "局部重绘",
        "idle": "未编辑",
    }
    return mapping.get(str(action or "idle"), str(action or "未编辑"))


def _region_state_from_action(action: str) -> str:
    mapping = {
        "transform": "transformed",
        "hide": "hidden",
        "show": "shown",
        "replace": "replaced",
        "emphasize": "emphasized",
        "weaken": "weakened",
        "restore": "idle",
        "inpaint": "inpainted",
    }
    return mapping.get(str(action or "transform"), "transformed")


def _rotate_point(x: float, y: float, center_x: float, center_y: float, angle_deg: float) -> tuple[float, float]:
    if abs(angle_deg) < 0.01:
        return x, y
    radians = math.radians(angle_deg)
    rel_x = x - center_x
    rel_y = y - center_y
    cos_v = math.cos(radians)
    sin_v = math.sin(radians)
    return (
        center_x + rel_x * cos_v - rel_y * sin_v,
        center_y + rel_x * sin_v + rel_y * cos_v,
    )


def _resolve_object_layer(doc: Dict[str, Any], op: Dict[str, Any]) -> Dict[str, Any] | None:
    wanted = str(op.get("object_id") or op.get("scene_object_id") or op.get("layer_id") or "").strip()
    if not wanted:
        return None
    for item in doc.get("object_layers", []) or []:
        if str(item.get("id") or "") == wanted or str(item.get("scene_object_id") or "") == wanted:
            return item
    return None


def _resolve_region_layer(doc: Dict[str, Any], op: Dict[str, Any]) -> Dict[str, Any] | None:
    wanted = str(op.get("region_id") or op.get("region_layer_id") or "").strip()
    parent_object_id = str(op.get("object_id") or "").strip()
    for item in doc.get("region_layers", []) or []:
        if not isinstance(item, dict):
            continue
        region_id = str(item.get("region_id") or item.get("id") or "").strip()
        object_id = str(item.get("parent_object_id") or item.get("object_id") or "").strip()
        if wanted and region_id != wanted:
            continue
        if parent_object_id and object_id != parent_object_id:
            continue
        if wanted or parent_object_id:
            return item
    return None


def _resolve_scene_object(scene_spec: Dict[str, Any], layer: Dict[str, Any]) -> Dict[str, Any] | None:
    wanted = str(layer.get("scene_object_id") or layer.get("id") or "").strip()
    for item in scene_spec.get("object_instances", []) or []:
        if isinstance(item, dict) and str(item.get("id") or "").strip() == wanted:
            return item
    return None


def _object_rect_ratios(object_layer: Dict[str, Any], region: Dict[str, Any]) -> Dict[str, float]:
    source_bbox = object_layer.get("source_bbox") if isinstance(object_layer.get("source_bbox"), dict) else {}
    source_rect = region.get("source_rect") if isinstance(region.get("source_rect"), dict) else {}
    source_width = max(1, _float(source_bbox.get("width"), 1.0))
    source_height = max(1, _float(source_bbox.get("height"), 1.0))
    return {
        "x": (_float(source_rect.get("x"), 0.0) - _float(source_bbox.get("x"), 0.0)) / source_width,
        "y": (_float(source_rect.get("y"), 0.0) - _float(source_bbox.get("y"), 0.0)) / source_height,
        "width": _float(source_rect.get("width"), 0.0) / source_width,
        "height": _float(source_rect.get("height"), 0.0) / source_height,
    }


def _region_base_rect(object_layer: Dict[str, Any], region: Dict[str, Any]) -> Dict[str, float]:
    ratios = region.get("source_rect_ratios") if isinstance(region.get("source_rect_ratios"), dict) else _object_rect_ratios(object_layer, region)
    object_x = _float(object_layer.get("x"), 0.0)
    object_y = _float(object_layer.get("y"), 0.0)
    object_width = max(1.0, _float(object_layer.get("width"), 1.0))
    object_height = max(1.0, _float(object_layer.get("height"), 1.0))
    base_width = max(1.0, object_width * _float(ratios.get("width"), 0.0))
    base_height = max(1.0, object_height * _float(ratios.get("height"), 0.0))
    base_left = object_x + object_width * _float(ratios.get("x"), 0.0)
    base_top = object_y + object_height * _float(ratios.get("y"), 0.0)
    base_center_x = base_left + base_width / 2.0
    base_center_y = base_top + base_height / 2.0
    object_center_x = object_x + object_width / 2.0
    object_center_y = object_y + object_height / 2.0
    rotated_center_x, rotated_center_y = _rotate_point(
        base_center_x,
        base_center_y,
        object_center_x,
        object_center_y,
        _float(object_layer.get("rotation"), 0.0),
    )
    return {
        "x": rotated_center_x - base_width / 2.0,
        "y": rotated_center_y - base_height / 2.0,
        "width": base_width,
        "height": base_height,
        "center_x": rotated_center_x,
        "center_y": rotated_center_y,
    }


def _region_display_state(doc: Dict[str, Any], region: Dict[str, Any]) -> Dict[str, Any]:
    object_layer = _resolve_object_layer(doc, {"object_id": region.get("parent_object_id") or region.get("object_id")})
    if not object_layer:
        current_rect = _copy(region.get("source_rect") or {"x": 0, "y": 0, "width": 1, "height": 1})
        return {
            "base_rect": current_rect,
            "current_rect": current_rect,
            "rotation": _float((region.get("local_transform") or {}).get("rotation"), 0.0),
            "depth_z": 0.52,
            "z_index": 0,
        }
    base_rect = _region_base_rect(object_layer, region)
    local_transform = region.get("local_transform") if isinstance(region.get("local_transform"), dict) else {}
    current_width = max(1.0, base_rect["width"] * max(0.2, _float(local_transform.get("scale_x"), 1.0)))
    current_height = max(1.0, base_rect["height"] * max(0.2, _float(local_transform.get("scale_y"), 1.0)))
    center_x = base_rect["center_x"] + _float(local_transform.get("tx"), 0.0)
    center_y = base_rect["center_y"] + _float(local_transform.get("ty"), 0.0)
    current_rect = {
        "x": int(round(center_x - current_width / 2.0)),
        "y": int(round(center_y - current_height / 2.0)),
        "width": max(1, int(round(current_width))),
        "height": max(1, int(round(current_height))),
    }
    return {
        "base_rect": {
            "x": int(round(base_rect["x"])),
            "y": int(round(base_rect["y"])),
            "width": max(1, int(round(base_rect["width"]))),
            "height": max(1, int(round(base_rect["height"]))),
        },
        "current_rect": current_rect,
        "rotation": _float(object_layer.get("rotation"), 0.0) + _float(local_transform.get("rotation"), 0.0),
        "depth_z": _float(object_layer.get("depth_z"), 0.52),
        "z_index": _int(object_layer.get("z_index"), 0),
    }


def _region_is_edited(region: Dict[str, Any]) -> bool:
    return (
        bool(region.get("promoted"))
        or not bool(region.get("visible", True))
        or str(region.get("edit_state") or "idle") != "idle"
        or not _transform_is_default(region.get("local_transform"))
        or bool(str(((region.get("render_intent") or {}).get("prompt") or "")).strip())
    )


def _doc_has_visual_edits(doc: Dict[str, Any]) -> bool:
    if doc.get("patch_layers"):
        return True
    if any(isinstance(region, dict) and _region_is_edited(region) for region in doc.get("region_layers", []) or []):
        return True
    return any(
        str(entry.get("type") or "")
        in {"transform_object", "set_depth", "hide_object", "show_object", "reorder_layer"}
        for entry in doc.get("edit_history", []) or []
        if isinstance(entry, dict)
    )


def _overlay_from_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    region_layers = []
    for item in doc.get("region_layers", []) or []:
        if not isinstance(item, dict):
            continue
        display = _region_display_state(doc, item)
        region_layers.append(
            {
                "id": item.get("region_id") or item.get("id"),
                "region_id": item.get("region_id") or item.get("id"),
                "parent_object_id": item.get("parent_object_id") or item.get("object_id"),
                "label": item.get("label"),
                "shape": item.get("shape"),
                "actions": _copy(item.get("actions") or []),
                "visible": item.get("visible", True),
                "promoted": bool(item.get("promoted", False)),
                "edit_state": str(item.get("edit_state") or "idle"),
                "render_intent": _copy(item.get("render_intent") or {}),
                "local_transform": _copy(item.get("local_transform") or _default_region_transform()),
                "source_rect": _copy(item.get("source_rect") or {}),
                "base_rect": _copy(display.get("base_rect") or {}),
                "current_rect": _copy(display.get("current_rect") or {}),
                "rotation": display.get("rotation"),
                "depth_z": display.get("depth_z"),
                "z_index": display.get("z_index"),
                "crop_image_path": item.get("crop_image_path"),
                "mask_image_path": item.get("mask_image_path"),
            }
        )
    return {
        "revision_id": doc.get("revision_id"),
        "canvas_size": _copy(doc.get("canvas_size") or {}),
        "camera_state": _copy(doc.get("camera_state") or {}),
        "object_layers": [
            {
                "id": item.get("id"),
                "scene_object_id": item.get("scene_object_id"),
                "concept": item.get("concept"),
                "x": item.get("x"),
                "y": item.get("y"),
                "width": item.get("width"),
                "height": item.get("height"),
                "rotation": item.get("rotation"),
                "scale": item.get("scale"),
                "visible": item.get("visible", True),
                "depth_band": item.get("depth_band"),
                "depth_z": item.get("depth_z"),
                "z_index": item.get("z_index"),
                "crop_image_path": item.get("display_image_path") or item.get("crop_image_path"),
                "base_crop_image_path": item.get("base_crop_image_path"),
                "source_bbox": _copy(item.get("source_bbox") or {}),
            }
            for item in doc.get("object_layers", []) or []
        ],
        "region_layers": region_layers,
        "patch_layers": [
            {
                **_copy(item),
                "rect": _copy(_resolve_patch_rect(doc, item) or item.get("rect") or {}),
            }
            for item in doc.get("patch_layers", []) or []
            if isinstance(item, dict)
        ],
    }


def _camera_summary(doc: Dict[str, Any]) -> str:
    camera = doc.get("camera_state") if isinstance(doc.get("camera_state"), dict) else {}
    return (
        f"pan=({camera.get('pan_x', 0)},{camera.get('pan_y', 0)}) | "
        f"zoom={float(camera.get('zoom', 1.0) or 1.0):.2f} | "
        f"parallax={float(camera.get('parallax_strength', 0.12) or 0.12):.2f}"
    )


def _depth_summary(doc: Dict[str, Any]) -> str:
    counts = {"foreground": 0, "midground": 0, "background": 0}
    for item in doc.get("object_layers", []) or []:
        if not item.get("visible", True):
            continue
        band = str(item.get("depth_band") or "midground")
        counts[band] = counts.get(band, 0) + 1
    return f"foreground={counts.get('foreground', 0)} | midground={counts.get('midground', 0)} | background={counts.get('background', 0)}"


def _render_patch_constraints(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    constraints: List[Dict[str, Any]] = []
    for patch in doc.get("patch_layers", []) or []:
        constraints.append(
            {
                "patch_id": patch.get("id"),
                "kind": patch.get("kind"),
                "rect": _copy(patch.get("rect") or {}),
                "prompt": str(patch.get("prompt") or ""),
                "note": str(patch.get("note") or ""),
            }
        )
    return constraints


def _region_edit_constraints(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    constraints: List[Dict[str, Any]] = []
    for region in doc.get("region_layers", []) or []:
        if not isinstance(region, dict) or not _region_is_edited(region):
            continue
        display = _region_display_state(doc, region)
        render_intent = region.get("render_intent") if isinstance(region.get("render_intent"), dict) else {}
        constraints.append(
            {
                "parent_object_id": region.get("parent_object_id") or region.get("object_id"),
                "region_id": region.get("region_id") or region.get("id"),
                "label": str(region.get("label") or region.get("region_id") or "region"),
                "action": str(render_intent.get("action") or _normalize_region_action(region.get("edit_state") or "idle")),
                "visible": bool(region.get("visible", True)),
                "source_rect": _copy(region.get("source_rect") or {}),
                "current_rect": _copy(display.get("current_rect") or {}),
                "transform": _copy(region.get("local_transform") or _default_region_transform()),
                "prompt": str(render_intent.get("prompt") or ""),
            }
        )
    return constraints


def _region_edit_summary(doc: Dict[str, Any]) -> str:
    parts: List[str] = []
    for region in doc.get("region_layers", []) or []:
        if not isinstance(region, dict) or not _region_is_edited(region):
            continue
        render_intent = region.get("render_intent") if isinstance(region.get("render_intent"), dict) else {}
        action = str(render_intent.get("action") or _normalize_region_action(region.get("edit_state") or "idle"))
        text = f"{region.get('label') or region.get('region_id')}: {_region_action_text(action)}"
        prompt = str(render_intent.get("prompt") or "").strip()
        if prompt:
            text += f"（{prompt}）"
        parts.append(text)
    if not parts:
        return ""
    return "；".join(parts[:10])


def _edit_summary(doc: Dict[str, Any]) -> str:
    object_changes = len(
        [
            entry
            for entry in doc.get("edit_history", []) or []
            if str(entry.get("type") or "") in {"transform_object", "set_depth", "hide_object", "show_object", "reorder_layer"}
        ]
    )
    region_changes = len([region for region in doc.get("region_layers", []) or [] if isinstance(region, dict) and _region_is_edited(region)])
    patch_changes = len(doc.get("patch_layers", []) or [])
    return f"对象直改 {object_changes} 次；部件直改 {region_changes} 处；局部 patch {patch_changes} 层"


def _update_render_sync_state(doc: Dict[str, Any]) -> None:
    render_sync = doc.setdefault("render_sync_state", {})
    render_sync["ready"] = bool(doc.get("composited_image_path"))
    render_sync["used_control_image_path"] = doc.get("composited_image_path")
    render_sync["edit_summary"] = _edit_summary(doc)
    render_sync["region_edit_summary"] = _region_edit_summary(doc)
    render_sync["depth_summary"] = _depth_summary(doc)
    render_sync["camera_summary"] = _camera_summary(doc)
    render_sync["render_patch_constraints"] = _render_patch_constraints(doc)
    render_sync["region_edit_constraints"] = _region_edit_constraints(doc)


def _rect_from_points(points: List[List[float]] | None, width: int, height: int) -> Dict[str, int] | None:
    valid = [item for item in (points or []) if isinstance(item, (list, tuple)) and len(item) >= 2]
    if not valid:
        return None
    xs = [max(0, min(width, _float(item[0], 0.0))) for item in valid]
    ys = [max(0, min(height, _float(item[1], 0.0))) for item in valid]
    return _clamp_rect(
        {
            "x": min(xs),
            "y": min(ys),
            "width": max(1, max(xs) - min(xs)),
            "height": max(1, max(ys) - min(ys)),
        },
        width,
        height,
    )


def _resolve_region_rect(doc: Dict[str, Any], object_id: str | None = None, region_id: str | None = None) -> Dict[str, int] | None:
    for region in doc.get("region_layers", []) or []:
        if object_id and str(region.get("parent_object_id") or region.get("object_id") or "") != str(object_id):
            continue
        if region_id and str(region.get("region_id") or region.get("id") or "") != str(region_id):
            continue
        display = _region_display_state(doc, region)
        rect = display.get("current_rect")
        if isinstance(rect, dict):
            return _copy(rect)
    return None


def _resolve_patch_rect(doc: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, int] | None:
    canvas = doc.get("canvas_size") if isinstance(doc.get("canvas_size"), dict) else {}
    width = max(1, _int(canvas.get("width"), 1))
    height = max(1, _int(canvas.get("height"), 1))
    if patch.get("follow_region") and patch.get("anchor_region_id"):
        target_region = _resolve_region_layer(doc, {"region_id": patch.get("anchor_region_id"), "object_id": patch.get("anchor_object_id")})
        relative = patch.get("relative_rect") if isinstance(patch.get("relative_rect"), dict) else {}
        if target_region:
            target_rect = _region_display_state(doc, target_region).get("current_rect") or {}
            rect = {
                "x": _int(target_rect.get("x"), 0) + int(round(float(relative.get("x", 0.0) or 0.0) * _int(target_rect.get("width"), 1))),
                "y": _int(target_rect.get("y"), 0) + int(round(float(relative.get("y", 0.0) or 0.0) * _int(target_rect.get("height"), 1))),
                "width": int(round(float(relative.get("width", 0.0) or 0.0) * _int(target_rect.get("width"), 1))),
                "height": int(round(float(relative.get("height", 0.0) or 0.0) * _int(target_rect.get("height"), 1))),
            }
            return _clamp_rect(rect, width, height)
    if patch.get("follow_object") and patch.get("anchor_object_id"):
        target = _resolve_object_layer(doc, {"object_id": patch.get("anchor_object_id")})
        relative = patch.get("relative_rect") if isinstance(patch.get("relative_rect"), dict) else {}
        if target:
            rect = {
                "x": _int(target.get("x"), 0) + int(round(float(relative.get("x", 0.0) or 0.0) * _int(target.get("width"), 1))),
                "y": _int(target.get("y"), 0) + int(round(float(relative.get("y", 0.0) or 0.0) * _int(target.get("height"), 1))),
                "width": int(round(float(relative.get("width", 0.0) or 0.0) * _int(target.get("width"), 1))),
                "height": int(round(float(relative.get("height", 0.0) or 0.0) * _int(target.get("height"), 1))),
            }
            return _clamp_rect(rect, width, height)
    rect = patch.get("rect")
    if isinstance(rect, dict):
        return _clamp_rect(rect, width, height)
    return None


def _asset_root(output_dir: str | Path, session_id: str, candidate_id: str) -> Path:
    root = Path(output_dir) / "sketch_edit_v1" / _slug(session_id) / _slug(candidate_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _object_promoted_regions(doc: Dict[str, Any], object_id: str) -> List[Dict[str, Any]]:
    regions = [
        item
        for item in doc.get("region_layers", []) or []
        if isinstance(item, dict)
        and str(item.get("parent_object_id") or item.get("object_id") or "") == str(object_id)
        and bool(item.get("promoted"))
    ]
    return sorted(regions, key=lambda item: str(item.get("region_id") or item.get("id") or ""))


def _build_object_residual_image(doc: Dict[str, Any], object_layer: Dict[str, Any]) -> str:
    promoted_regions = _object_promoted_regions(doc, str(object_layer.get("scene_object_id") or ""))
    if not promoted_regions:
        object_layer["display_image_path"] = object_layer.get("base_crop_image_path") or object_layer.get("crop_image_path")
        return str(object_layer.get("display_image_path") or "")
    crop_path = str(object_layer.get("base_crop_image_path") or object_layer.get("crop_image_path") or "").strip()
    if not crop_path or not Path(crop_path).exists():
        return ""
    crop_image = Image.open(crop_path).convert("RGBA")
    alpha = crop_image.getchannel("A")
    crop_box = object_layer.get("crop_box") if isinstance(object_layer.get("crop_box"), dict) else {}
    for region in promoted_regions:
        mask_path = str(region.get("mask_image_path") or "").strip()
        source_rect = region.get("source_rect") if isinstance(region.get("source_rect"), dict) else {}
        if not mask_path or not Path(mask_path).exists() or not source_rect:
            continue
        local_x = _int(source_rect.get("x"), 0) - _int(crop_box.get("x"), 0)
        local_y = _int(source_rect.get("y"), 0) - _int(crop_box.get("y"), 0)
        mask = Image.open(mask_path).convert("L")
        alpha.paste(0, (local_x, local_y), mask=mask)
    crop_image.putalpha(alpha)
    asset_root = Path(str(doc.get("asset_root") or "")).resolve()
    residual_path = asset_root / f"residual_{doc.get('revision_id')}_{_slug(object_layer.get('scene_object_id'))}.png"
    crop_image.save(residual_path)
    object_layer["display_image_path"] = str(residual_path)
    return str(residual_path)


def _render_object_crop(base: Image.Image, object_layer: Dict[str, Any], image_path: str) -> None:
    crop_path = str(image_path or "").strip()
    if not crop_path or not Path(crop_path).exists():
        return
    source_bbox = object_layer.get("source_bbox") if isinstance(object_layer.get("source_bbox"), dict) else {}
    crop_box = object_layer.get("crop_box") if isinstance(object_layer.get("crop_box"), dict) else {}
    current_x = _int(object_layer.get("x"), 0)
    current_y = _int(object_layer.get("y"), 0)
    current_w = max(1, _int(object_layer.get("width"), source_bbox.get("width", 1)))
    current_h = max(1, _int(object_layer.get("height"), source_bbox.get("height", 1)))
    source_w = max(1, _int(source_bbox.get("width"), current_w))
    source_h = max(1, _int(source_bbox.get("height"), current_h))
    ratio_x = current_w / source_w
    ratio_y = current_h / source_h
    crop_w = max(1, int(round(_int(crop_box.get("width"), current_w) * ratio_x)))
    crop_h = max(1, int(round(_int(crop_box.get("height"), current_h) * ratio_y)))
    offset_x = int(round(_int(object_layer.get("crop_offset_x"), 0) * ratio_x))
    offset_y = int(round(_int(object_layer.get("crop_offset_y"), 0) * ratio_y))
    paste_x = current_x - offset_x
    paste_y = current_y - offset_y
    crop = Image.open(crop_path).convert("RGBA").resize((crop_w, crop_h), Image.Resampling.LANCZOS)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.alpha_composite(crop, (paste_x, paste_y))
    rotation = _float(object_layer.get("rotation"), 0.0)
    if abs(rotation) > 0.01:
        center = (current_x + current_w / 2.0, current_y + current_h / 2.0)
        layer = layer.rotate(rotation, resample=Image.Resampling.BICUBIC, center=center)
    base.alpha_composite(layer)


def _apply_region_visual_intent(crop: Image.Image, region: Dict[str, Any]) -> Image.Image:
    edit_state = str(region.get("edit_state") or "idle")
    if edit_state in {"idle", "transformed", "shown"}:
        return crop
    result = crop.convert("RGBA")
    if edit_state == "weakened":
        alpha = result.getchannel("A").point(lambda px: int(px * 0.38))
        result.putalpha(alpha)
        return result
    if edit_state == "emphasized":
        alpha = result.getchannel("A").point(lambda px: min(255, int(px * 1.25)))
        result.putalpha(alpha)
        return ImageEnhance.Contrast(result).enhance(1.25)
    if edit_state in {"replaced", "inpainted"}:
        tinted = Image.new("RGBA", result.size, (255, 255, 255, 0))
        alpha = result.getchannel("A")
        if edit_state == "replaced":
            tinted.putalpha(alpha.point(lambda px: min(255, int(px * 0.78))))
            return Image.alpha_composite(result.filter(ImageFilter.GaussianBlur(radius=1.2)), tinted)
        tinted.putalpha(alpha.point(lambda px: min(255, int(px * 0.88))))
        return Image.alpha_composite(result.filter(ImageFilter.GaussianBlur(radius=2.0)), tinted)
    return result


def _render_region_crop(base: Image.Image, doc: Dict[str, Any], region: Dict[str, Any]) -> None:
    crop_path = str(region.get("crop_image_path") or "").strip()
    if not crop_path or not Path(crop_path).exists():
        return
    display = _region_display_state(doc, region)
    rect = display.get("current_rect") if isinstance(display.get("current_rect"), dict) else {}
    if not rect:
        return
    current_w = max(1, _int(rect.get("width"), 1))
    current_h = max(1, _int(rect.get("height"), 1))
    paste_x = _int(rect.get("x"), 0)
    paste_y = _int(rect.get("y"), 0)
    crop = Image.open(crop_path).convert("RGBA").resize((current_w, current_h), Image.Resampling.LANCZOS)
    crop = _apply_region_visual_intent(crop, region)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.alpha_composite(crop, (paste_x, paste_y))
    rotation = _float(display.get("rotation"), 0.0)
    if abs(rotation) > 0.01:
        center = (paste_x + current_w / 2.0, paste_y + current_h / 2.0)
        layer = layer.rotate(rotation, resample=Image.Resampling.BICUBIC, center=center)
    base.alpha_composite(layer)


def _compose_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    canvas = doc.get("canvas_size") if isinstance(doc.get("canvas_size"), dict) else {}
    canvas_width = max(1, _int(canvas.get("width"), 1))
    canvas_height = max(1, _int(canvas.get("height"), 1))
    revision_id = str(doc.get("revision_id") or _new_revision_id())
    doc["revision_id"] = revision_id
    if not _doc_has_visual_edits(doc):
        base_image_path = str(doc.get("base_image_path") or "").strip()
        if base_image_path and Path(base_image_path).exists():
            doc["composited_image_path"] = base_image_path
            doc["overlay"] = _overlay_from_doc(doc)
            _update_render_sync_state(doc)
            return doc
    background_path = str(doc.get("background_plate_path") or "").strip()
    if background_path and Path(background_path).exists():
        base = Image.open(background_path).convert("RGBA")
    else:
        base = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 255))

    object_layers = sorted(
        [item for item in doc.get("object_layers", []) or [] if isinstance(item, dict) and item.get("visible", True)],
        key=lambda item: (_float(item.get("depth_z"), 0.0), _int(item.get("z_index"), 0)),
    )
    for item in object_layers:
        display_path = _build_object_residual_image(doc, item)
        _render_object_crop(base, item, display_path)

    region_layers = sorted(
        [
            item
            for item in doc.get("region_layers", []) or []
            if isinstance(item, dict) and item.get("promoted") and item.get("visible", True)
        ],
        key=lambda item: (
            _float(_region_display_state(doc, item).get("depth_z"), 0.0),
            _int(_region_display_state(doc, item).get("z_index"), 0),
            str(item.get("region_id") or item.get("id") or ""),
        ),
    )
    for item in region_layers:
        _render_region_crop(base, doc, item)

    for patch in doc.get("patch_layers", []) or []:
        rect = _resolve_patch_rect(doc, patch)
        if not rect:
            continue
        patch_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(patch_layer, "RGBA")
        kind = str(patch.get("kind") or "erase_region")
        fill = (255, 255, 255, 255) if kind in {"erase_region", "brush_mask", "inpaint_region"} else (255, 255, 255, 190)
        draw.rounded_rectangle(
            [rect["x"], rect["y"], rect["x"] + rect["width"], rect["y"] + rect["height"]],
            radius=max(8, int(min(rect["width"], rect["height"]) * 0.12)),
            fill=fill,
            outline=None,
        )
        if kind == "brush_mask":
            patch_layer = patch_layer.filter(ImageFilter.GaussianBlur(radius=8))
        elif kind == "inpaint_region":
            patch_layer = patch_layer.filter(ImageFilter.GaussianBlur(radius=4))
        base.alpha_composite(patch_layer)

    asset_root = Path(str(doc.get("asset_root") or "")).resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    composited_path = asset_root / f"composited_{revision_id}.png"
    base.convert("RGB").save(composited_path)
    doc["composited_image_path"] = str(composited_path)
    doc["overlay"] = _overlay_from_doc(doc)
    _update_render_sync_state(doc)
    return doc


def _make_object_layer(
    base_image: Image.Image,
    scene_object: Dict[str, Any],
    asset_root: Path,
    index: int,
    canvas_width: int,
    canvas_height: int,
) -> Dict[str, Any]:
    source_bbox = _clamp_rect(
        {
            "x": scene_object.get("x", 0),
            "y": scene_object.get("y", 0),
            "width": scene_object.get("width", 1),
            "height": scene_object.get("height", 1),
        },
        canvas_width,
        canvas_height,
    )
    padding = max(8, int(min(source_bbox["width"], source_bbox["height"]) * 0.16))
    crop_box = _expand_box(source_bbox, canvas_width, canvas_height, padding)
    crop = base_image.crop(
        (
            crop_box["x"],
            crop_box["y"],
            crop_box["x"] + crop_box["width"],
            crop_box["y"] + crop_box["height"],
        )
    ).convert("RGBA")
    alpha = _white_alpha_from_crop(crop)
    rgba = crop.copy()
    rgba.putalpha(alpha)
    crop_path = asset_root / f"layer_{index:03d}_{_slug(scene_object.get('id') or scene_object.get('concept'))}.png"
    rgba.save(crop_path)
    return {
        "id": f"layer_{_slug(scene_object.get('id') or index)}",
        "scene_object_id": str(scene_object.get("id") or f"obj_{index}"),
        "concept": str(scene_object.get("concept") or scene_object.get("asset_key") or f"对象{index}"),
        "base_crop_image_path": str(crop_path),
        "crop_image_path": str(crop_path),
        "display_image_path": str(crop_path),
        "source_bbox": source_bbox,
        "crop_box": crop_box,
        "crop_offset_x": source_bbox["x"] - crop_box["x"],
        "crop_offset_y": source_bbox["y"] - crop_box["y"],
        "x": source_bbox["x"],
        "y": source_bbox["y"],
        "width": source_bbox["width"],
        "height": source_bbox["height"],
        "rotation": _float(scene_object.get("rotation"), 0.0),
        "scale": _float(scene_object.get("scale"), 1.0),
        "depth_band": str(scene_object.get("depth_band") or "midground"),
        "depth_z": _band_depth(scene_object.get("depth_band"), scene_object.get("depth_z")),
        "z_index": _int(scene_object.get("z_index"), index),
        "visible": bool(scene_object.get("visible", True)),
    }


def _build_region_layer(
    scene_object: Dict[str, Any],
    object_layer: Dict[str, Any],
    asset_root: Path,
    region_spec: Dict[str, Any],
    index: int,
) -> Dict[str, Any] | None:
    source_rect = _clamp_rect(
        {
            "x": _float(scene_object.get("x"), 0.0) + _float(region_spec.get("x"), 0.0) * _float(scene_object.get("width"), 1.0),
            "y": _float(scene_object.get("y"), 0.0) + _float(region_spec.get("y"), 0.0) * _float(scene_object.get("height"), 1.0),
            "width": _float(region_spec.get("width"), 0.0) * _float(scene_object.get("width"), 1.0),
            "height": _float(region_spec.get("height"), 0.0) * _float(scene_object.get("height"), 1.0),
        },
        max(1, _int((scene_object.get("canvas_size") or {}).get("width"), object_layer.get("width", 1) + object_layer.get("x", 0) + 1)),
        max(1, _int((scene_object.get("canvas_size") or {}).get("height"), object_layer.get("height", 1) + object_layer.get("y", 0) + 1)),
    )
    object_crop_path = str(object_layer.get("base_crop_image_path") or object_layer.get("crop_image_path") or "").strip()
    if not object_crop_path or not Path(object_crop_path).exists():
        return None
    object_crop = Image.open(object_crop_path).convert("RGBA")
    crop_box = object_layer.get("crop_box") if isinstance(object_layer.get("crop_box"), dict) else {}
    local_x = _int(source_rect.get("x"), 0) - _int(crop_box.get("x"), 0)
    local_y = _int(source_rect.get("y"), 0) - _int(crop_box.get("y"), 0)
    local_w = max(1, _int(source_rect.get("width"), 1))
    local_h = max(1, _int(source_rect.get("height"), 1))
    region_crop = object_crop.crop((local_x, local_y, local_x + local_w, local_y + local_h)).convert("RGBA")
    shape = str(region_spec.get("shape") or "rect")
    region_mask = _shape_mask(local_w, local_h, shape)
    region_alpha = ImageChops.multiply(_white_alpha_from_crop(region_crop), region_mask)
    region_crop.putalpha(region_alpha)
    object_id = str(scene_object.get("id") or object_layer.get("scene_object_id") or "object")
    region_id = str(region_spec.get("id") or f"{object_id}_region_{index}")
    crop_path = asset_root / f"region_{_slug(object_id)}_{_slug(region_id)}.png"
    mask_path = asset_root / f"region_{_slug(object_id)}_{_slug(region_id)}_mask.png"
    region_crop.save(crop_path)
    region_mask.save(mask_path)
    ratios = _object_rect_ratios(object_layer, {"source_rect": source_rect})
    return {
        "id": region_id,
        "region_id": region_id,
        "parent_object_id": object_id,
        "label": str(region_spec.get("label") or region_id),
        "shape": shape,
        "source_rect": source_rect,
        "source_rect_ratios": ratios,
        "crop_image_path": str(crop_path),
        "mask_image_path": str(mask_path),
        "actions": [str(item) for item in (region_spec.get("actions") or []) if str(item).strip()],
        "visible": True,
        "local_transform": _default_region_transform(),
        "promoted": False,
        "edit_state": "idle",
        "render_intent": {"action": "idle", "prompt": "", "note": ""},
    }


def _build_region_layers(scene_spec: Dict[str, Any], object_layers: List[Dict[str, Any]], asset_root: Path) -> List[Dict[str, Any]]:
    object_map = {str(item.get("scene_object_id") or ""): item for item in object_layers if isinstance(item, dict)}
    regions: List[Dict[str, Any]] = []
    for scene_object in scene_spec.get("object_instances", []) or []:
        if not isinstance(scene_object, dict):
            continue
        object_id = str(scene_object.get("id") or "")
        object_layer = object_map.get(object_id)
        if not object_layer:
            continue
        scene_object = _copy(scene_object)
        scene_object["canvas_size"] = _copy(scene_spec.get("canvas_size") or {})
        region_specs = scene_object.get("region_masks") or []
        if not region_specs:
            region_specs = [
                {
                    "id": "core",
                    "label": "主体区域",
                    "shape": "rect",
                    "x": 0.14,
                    "y": 0.14,
                    "width": 0.72,
                    "height": 0.72,
                    "actions": ["replace", "weaken", "hide", "emphasize"],
                }
            ]
        for index, region_spec in enumerate(region_specs, start=1):
            if not isinstance(region_spec, dict):
                continue
            layer = _build_region_layer(scene_object, object_layer, asset_root, region_spec, index)
            if layer:
                regions.append(layer)
    return regions


def _default_depth_model(scene_spec: Dict[str, Any]) -> Dict[str, Any]:
    counts = {"foreground": 0, "midground": 0, "background": 0}
    for item in scene_spec.get("object_instances", []) or []:
        if not isinstance(item, dict):
            continue
        band = str(item.get("depth_band") or "midground")
        counts[band] = counts.get(band, 0) + 1
    return {"bands": counts, "mode": "parallax_v1"}


def _build_background_plate(base_image: Image.Image, object_layers: List[Dict[str, Any]], asset_root: Path) -> str:
    background = base_image.copy()
    fill_rgb = _estimate_background_rgb(base_image, stride=6)
    for layer in object_layers:
        crop_path = str(layer.get("base_crop_image_path") or layer.get("crop_image_path") or "").strip()
        if not crop_path or not Path(crop_path).exists():
            continue
        crop = Image.open(crop_path).convert("RGBA")
        alpha = crop.getchannel("A").filter(ImageFilter.GaussianBlur(radius=1.8))
        wipe = Image.new("RGBA", crop.size, (*fill_rgb, 255))
        background.paste(
            wipe,
            (_int(layer.get("crop_box", {}).get("x"), 0), _int(layer.get("crop_box", {}).get("y"), 0)),
            mask=alpha,
        )
    output_path = asset_root / "background_plate.png"
    background.convert("RGB").save(output_path)
    return str(output_path)


def create_editable_sketch_doc(
    *,
    preview: Dict[str, Any],
    scene_spec: Dict[str, Any],
    session_id: str,
    output_dir: str | Path = "outputs",
    sketch_backend: str = "sketch_v2",
    source_candidate_id: str | None = None,
) -> Dict[str, Any] | None:
    if str(sketch_backend or "").strip().lower() != "sketch_v2":
        return None
    active_candidate = _candidate_from_preview(preview, source_candidate_id)
    if not active_candidate:
        return None
    if str(active_candidate.get("provider") or "").strip().lower() != "sd":
        return None
    image_path = str(active_candidate.get("image_path") or "").strip()
    if not image_path or not Path(image_path).exists():
        return None

    base_image = Image.open(image_path).convert("RGBA")
    canvas_width, canvas_height = base_image.size
    asset_root = _asset_root(output_dir, session_id, str(active_candidate.get("candidate_id") or "active"))
    object_layers: List[Dict[str, Any]] = []
    for index, item in enumerate(scene_spec.get("object_instances", []) or [], start=1):
        if not isinstance(item, dict):
            continue
        object_layers.append(_make_object_layer(base_image, item, asset_root, index, canvas_width, canvas_height))

    region_layers = _build_region_layers(scene_spec, object_layers, asset_root)
    doc = {
        "revision_id": _new_revision_id(),
        "session_id": session_id,
        "sketch_backend": "sketch_v2",
        "source_candidate_id": str(active_candidate.get("candidate_id") or "active"),
        "source_provider": str(active_candidate.get("provider") or "sd"),
        "base_image_path": image_path,
        "asset_root": str(asset_root),
        "canvas_size": {"width": canvas_width, "height": canvas_height},
        "camera_state": {
            "pan_x": 0,
            "pan_y": 0,
            "zoom": 1.0,
            "parallax_strength": 0.12,
            "preview_enabled": False,
        },
        "depth_model": _default_depth_model(scene_spec),
        "object_layers": object_layers,
        "region_layers": region_layers,
        "patch_layers": [],
        "edit_history": [],
        "scene_sync_state": {
            "synced_object_ids": [item.get("scene_object_id") for item in object_layers],
            "pending_region_layers": 0,
            "pending_patch_layers": 0,
            "last_scene_delta": {"objects": [], "regions": [], "patches": []},
        },
        "render_sync_state": {
            "ready": False,
            "used_control_image_path": "",
            "edit_summary": "",
            "region_edit_summary": "",
            "depth_summary": "",
            "camera_summary": "",
            "render_patch_constraints": [],
            "region_edit_constraints": [],
        },
    }
    doc["background_plate_path"] = _build_background_plate(base_image, object_layers, asset_root)
    return _compose_doc(doc)


def _update_scene_render_hints(scene_spec: Dict[str, Any], doc: Dict[str, Any]) -> None:
    render_hints = scene_spec.setdefault("render_hints", {})
    render_sync = doc.get("render_sync_state") if isinstance(doc.get("render_sync_state"), dict) else {}
    render_hints["edit_summary"] = str(render_sync.get("edit_summary") or "")
    render_hints["region_edit_summary"] = str(render_sync.get("region_edit_summary") or "")
    render_hints["depth_summary"] = str(render_sync.get("depth_summary") or "")
    render_hints["camera_summary"] = str(render_sync.get("camera_summary") or "")
    render_hints["render_patch_constraints"] = _copy(render_sync.get("render_patch_constraints") or [])
    render_hints["region_edit_constraints"] = _copy(render_sync.get("region_edit_constraints") or [])
    render_hints["editable_sketch_composited_path"] = str(doc.get("composited_image_path") or "")


def build_render_conditioning_bundle(doc: Dict[str, Any] | None) -> Dict[str, Any]:
    doc = doc if isinstance(doc, dict) else {}
    if not doc:
        return {}
    object_layers = []
    for item in doc.get("object_layers", []) or []:
        if not isinstance(item, dict):
            continue
        object_layers.append(
            {
                "object_id": item.get("scene_object_id") or item.get("id"),
                "concept": item.get("concept"),
                "x": _int(item.get("x"), 0),
                "y": _int(item.get("y"), 0),
                "width": _int(item.get("width"), 0),
                "height": _int(item.get("height"), 0),
                "rotation": _float(item.get("rotation"), 0.0),
                "scale": _float(item.get("scale"), 1.0),
                "visible": bool(item.get("visible", True)),
                "depth_band": str(item.get("depth_band") or "midground"),
                "depth_z": _float(item.get("depth_z"), 0.52),
                "z_index": _int(item.get("z_index"), 0),
                "display_image_path": str(item.get("display_image_path") or item.get("crop_image_path") or ""),
                "base_crop_image_path": str(item.get("base_crop_image_path") or ""),
                "source_bbox": _copy(item.get("source_bbox") or {}),
            }
        )
    region_layers = []
    for item in doc.get("region_layers", []) or []:
        if not isinstance(item, dict):
            continue
        display = _region_display_state(doc, item)
        render_intent = item.get("render_intent") if isinstance(item.get("render_intent"), dict) else {}
        region_layers.append(
            {
                "region_id": item.get("region_id") or item.get("id"),
                "parent_object_id": item.get("parent_object_id") or item.get("object_id"),
                "label": str(item.get("label") or item.get("region_id") or "region"),
                "shape": str(item.get("shape") or "rect"),
                "visible": bool(item.get("visible", True)),
                "promoted": bool(item.get("promoted", False)),
                "edit_state": str(item.get("edit_state") or "idle"),
                "action": str(render_intent.get("action") or _normalize_region_action(item.get("edit_state") or "idle")),
                "render_intent": _copy(render_intent),
                "local_transform": _copy(item.get("local_transform") or _default_region_transform()),
                "source_rect": _copy(item.get("source_rect") or {}),
                "base_rect": _copy(display.get("base_rect") or {}),
                "current_rect": _copy(display.get("current_rect") or {}),
                "rotation": _float(display.get("rotation"), 0.0),
                "depth_z": _float(display.get("depth_z"), 0.52),
                "z_index": _int(display.get("z_index"), 0),
                "crop_image_path": str(item.get("crop_image_path") or ""),
                "mask_image_path": str(item.get("mask_image_path") or ""),
                "edited": bool(_region_is_edited(item)),
            }
        )
    patch_layers = []
    for item in doc.get("patch_layers", []) or []:
        if not isinstance(item, dict):
            continue
        patch_layers.append(
            {
                "patch_id": item.get("id"),
                "kind": str(item.get("kind") or ""),
                "prompt": str(item.get("prompt") or ""),
                "note": str(item.get("note") or ""),
                "rect": _copy(_resolve_patch_rect(doc, item) or item.get("rect") or {}),
                "anchor_object_id": str(item.get("anchor_object_id") or ""),
                "anchor_region_id": str(item.get("anchor_region_id") or item.get("region_id") or ""),
                "follow_region": bool(item.get("follow_region", False)),
                "follow_object": bool(item.get("follow_object", False)),
            }
        )
    return {
        "version": 1,
        "source": "editable_sketch_doc",
        "revision_id": str(doc.get("revision_id") or ""),
        "session_id": str(doc.get("session_id") or ""),
        "sketch_backend": str(doc.get("sketch_backend") or ""),
        "source_candidate_id": str(doc.get("source_candidate_id") or ""),
        "base_image_path": str(doc.get("base_image_path") or ""),
        "background_plate_path": str(doc.get("background_plate_path") or ""),
        "composited_image_path": str(doc.get("composited_image_path") or ""),
        "canvas_size": _copy(doc.get("canvas_size") or {}),
        "camera_state": _copy(doc.get("camera_state") or {}),
        "depth_model": _copy(doc.get("depth_model") or {}),
        "edit_summary": _edit_summary(doc),
        "region_edit_summary": _region_edit_summary(doc),
        "region_edit_constraints": _region_edit_constraints(doc),
        "render_patch_constraints": _render_patch_constraints(doc),
        "object_layers": object_layers,
        "region_layers": region_layers,
        "patch_layers": patch_layers,
        "counts": {
            "objects": len(object_layers),
            "regions": len(region_layers),
            "edited_regions": len([item for item in region_layers if item.get("edited")]),
            "patches": len(patch_layers),
        },
    }


def attach_doc_to_preview(preview: Dict[str, Any], scene_spec: Dict[str, Any], doc: Dict[str, Any]) -> Dict[str, Any]:
    updated_preview = _copy(preview)
    updated_scene = _copy(scene_spec)
    _update_scene_render_hints(updated_scene, doc)
    updated_preview["scene_spec"] = updated_scene
    updated_preview["editable_sketch_doc"] = _copy(doc)
    updated_preview["editable_sketch_enabled"] = True
    updated_preview["editable_sketch_revision_id"] = doc.get("revision_id")
    updated_preview["editable_sketch_base_candidate_id"] = doc.get("source_candidate_id")
    updated_preview["editable_sketch_source_candidate_id"] = doc.get("source_candidate_id")
    updated_preview["editable_sketch_base_image_path"] = doc.get("base_image_path")
    updated_preview["editable_sketch_composited_path"] = doc.get("composited_image_path")
    updated_preview["editable_sketch_overlay"] = _copy(doc.get("overlay") or _overlay_from_doc(doc))
    updated_preview["editable_sketch_scene_sync"] = _copy(doc.get("scene_sync_state") or {})
    updated_preview["editable_sketch_history"] = _copy(doc.get("edit_history") or [])
    updated_preview["editable_sketch_capabilities"] = {
        "supported_ops": list(SUPPORTED_OPS),
        "region_ops": list(REGION_ONLY_OPS),
        "modes": ["select", "erase", "inpaint", "depth"],
        "parallax": True,
        "direct_on_active_sd_sketch": True,
    }
    render_bundle = updated_preview.get("render_bundle")
    if isinstance(render_bundle, dict):
        render_bundle = _copy(render_bundle)
        model_inputs = render_bundle.get("model_inputs") if isinstance(render_bundle.get("model_inputs"), dict) else {}
        model_inputs["used_control_image_path"] = str(doc.get("composited_image_path") or "")
        render_bundle["model_inputs"] = model_inputs
        updated_preview["render_bundle"] = render_bundle
    return updated_preview


def detach_doc_from_preview(preview: Dict[str, Any], note: str = "") -> Dict[str, Any]:
    updated_preview = _copy(preview)
    for key in [
        "editable_sketch_doc",
        "editable_sketch_revision_id",
        "editable_sketch_base_candidate_id",
        "editable_sketch_source_candidate_id",
        "editable_sketch_base_image_path",
        "editable_sketch_composited_path",
        "editable_sketch_overlay",
        "editable_sketch_scene_sync",
        "editable_sketch_history",
        "editable_sketch_capabilities",
    ]:
        updated_preview.pop(key, None)
    updated_preview["editable_sketch_enabled"] = False
    if note:
        updated_preview["editable_sketch_note"] = note
    return updated_preview


def ensure_editable_preview(
    *,
    preview: Dict[str, Any] | None,
    session_id: str,
    sketch_backend: str,
    output_dir: str | Path = "outputs",
    force_rebuild: bool = False,
    source_candidate_id: str | None = None,
) -> Dict[str, Any] | None:
    if not isinstance(preview, dict):
        return preview
    updated_preview = _copy(preview)
    if str(sketch_backend or "").strip().lower() != "sketch_v2":
        return detach_doc_from_preview(updated_preview, "Direct sketch editing is available only under sketch_v2.")
    active_candidate = _candidate_from_preview(updated_preview, source_candidate_id)
    if not active_candidate:
        return detach_doc_from_preview(updated_preview, "No editable sketch candidate is available in the current session.")
    provider = str(active_candidate.get("provider") or "").strip().lower()
    if provider != "sd":
        return detach_doc_from_preview(updated_preview, "Direct editing currently requires an SD sketch candidate. Please switch to an SD candidate.")
    existing_doc = updated_preview.get("editable_sketch_doc") if isinstance(updated_preview.get("editable_sketch_doc"), dict) else None
    same_candidate = existing_doc and str(existing_doc.get("source_candidate_id") or "") == str(active_candidate.get("candidate_id") or "")
    same_base = existing_doc and str(existing_doc.get("base_image_path") or "") == str(active_candidate.get("image_path") or "")
    if existing_doc and same_candidate and same_base and not force_rebuild:
        return attach_doc_to_preview(updated_preview, updated_preview.get("scene_spec") or {}, existing_doc)
    scene_spec = updated_preview.get("scene_spec") if isinstance(updated_preview.get("scene_spec"), dict) else {}
    if not scene_spec:
        return detach_doc_from_preview(updated_preview, "No SceneSpec is available in the current session.")
    doc = create_editable_sketch_doc(
        preview=updated_preview,
        scene_spec=scene_spec,
        session_id=session_id,
        output_dir=output_dir,
        sketch_backend=sketch_backend,
        source_candidate_id=str(active_candidate.get("candidate_id") or ""),
    )
    if not doc:
        return detach_doc_from_preview(updated_preview, "Direct editing currently requires an SD sketch candidate. Please switch to an SD candidate.")
    return attach_doc_to_preview(updated_preview, scene_spec, doc)


def _append_history(doc: Dict[str, Any], op: Dict[str, Any], note: str = "") -> None:
    history = doc.setdefault("edit_history", [])
    history.append(
        {
            "id": f"hist_{uuid.uuid4().hex[:10]}",
            "type": str(op.get("type") or op.get("op") or ""),
            "note": note,
            "payload": _copy(op),
        }
    )
    if len(history) > 120:
        del history[:-120]


def _resolve_op_rect(doc: Dict[str, Any], op: Dict[str, Any]) -> Dict[str, int] | None:
    canvas = doc.get("canvas_size") if isinstance(doc.get("canvas_size"), dict) else {}
    width = max(1, _int(canvas.get("width"), 1))
    height = max(1, _int(canvas.get("height"), 1))
    rect = op.get("rect")
    if isinstance(rect, dict):
        return _clamp_rect(rect, width, height)
    points_rect = _rect_from_points(op.get("points") if isinstance(op.get("points"), list) else None, width, height)
    if points_rect:
        return points_rect
    region_layer = _resolve_region_layer(doc, op)
    if region_layer:
        region_rect = _resolve_region_rect(doc, region_layer.get("parent_object_id"), region_layer.get("region_id"))
        if region_rect:
            return region_rect
    object_layer = _resolve_object_layer(doc, op)
    if object_layer:
        return _clamp_rect(
            {
                "x": object_layer.get("x"),
                "y": object_layer.get("y"),
                "width": object_layer.get("width"),
                "height": object_layer.get("height"),
            },
            width,
            height,
        )
    return None


def _sync_object_to_scene(scene_obj: Dict[str, Any], layer: Dict[str, Any]) -> None:
    scene_obj["x"] = _int(layer.get("x"), scene_obj.get("x", 0))
    scene_obj["y"] = _int(layer.get("y"), scene_obj.get("y", 0))
    scene_obj["width"] = _int(layer.get("width"), scene_obj.get("width", 1))
    scene_obj["height"] = _int(layer.get("height"), scene_obj.get("height", 1))
    scene_obj["rotation"] = _float(layer.get("rotation"), scene_obj.get("rotation", 0.0))
    scene_obj["scale"] = _float(layer.get("scale"), scene_obj.get("scale", 1.0))
    scene_obj["depth_band"] = str(layer.get("depth_band") or scene_obj.get("depth_band") or "midground")
    scene_obj["depth_z"] = _float(layer.get("depth_z"), scene_obj.get("depth_z", _band_depth(layer.get("depth_band"))))
    scene_obj["z_index"] = _int(layer.get("z_index"), scene_obj.get("z_index", 0))
    scene_obj["visible"] = bool(layer.get("visible", True))


def _ensure_region_promoted(region: Dict[str, Any], *, action: str, prompt: str = "", note: str = "") -> None:
    region["promoted"] = True
    region["edit_state"] = _region_state_from_action(action)
    render_intent = region.setdefault("render_intent", {})
    render_intent["action"] = action
    render_intent["prompt"] = str(prompt or "")
    render_intent["note"] = str(note or "")


def _apply_transform_object(doc: Dict[str, Any], scene_spec: Dict[str, Any], op: Dict[str, Any], scene_delta: Dict[str, Any]) -> None:
    layer = _resolve_object_layer(doc, op)
    if not layer:
        return
    source_bbox = layer.get("source_bbox") if isinstance(layer.get("source_bbox"), dict) else {}
    if "dx" in op:
        layer["x"] = _int(layer.get("x"), 0) + _int(op.get("dx"), 0)
    if "dy" in op:
        layer["y"] = _int(layer.get("y"), 0) + _int(op.get("dy"), 0)
    if "x" in op:
        layer["x"] = _int(op.get("x"), layer.get("x", 0))
    if "y" in op:
        layer["y"] = _int(op.get("y"), layer.get("y", 0))
    if "width" in op:
        layer["width"] = max(24, _int(op.get("width"), layer.get("width", 24)))
    if "height" in op:
        layer["height"] = max(24, _int(op.get("height"), layer.get("height", 24)))
    if "scale" in op and "width" not in op and "height" not in op:
        scale = max(0.2, min(3.0, _float(op.get("scale"), layer.get("scale", 1.0))))
        layer["scale"] = scale
        if source_bbox:
            layer["width"] = max(24, int(round(_int(source_bbox.get("width"), layer.get("width", 24)) * scale)))
            layer["height"] = max(24, int(round(_int(source_bbox.get("height"), layer.get("height", 24)) * scale)))
    else:
        layer["scale"] = max(0.2, min(3.0, _float(op.get("scale"), layer.get("scale", 1.0))))
        if source_bbox and _int(source_bbox.get("width"), 0) > 0:
            scale_x = _int(layer.get("width"), 1) / max(1, _int(source_bbox.get("width"), 1))
            scale_y = _int(layer.get("height"), 1) / max(1, _int(source_bbox.get("height"), 1))
            layer["scale"] = round((scale_x + scale_y) / 2.0, 4)
    if "rotation" in op:
        layer["rotation"] = _float(op.get("rotation"), layer.get("rotation", 0.0))
    scene_obj = _resolve_scene_object(scene_spec, layer)
    if scene_obj:
        _sync_object_to_scene(scene_obj, layer)
    scene_delta["objects"].append(
        {
            "object_id": layer.get("scene_object_id"),
            "x": layer.get("x"),
            "y": layer.get("y"),
            "width": layer.get("width"),
            "height": layer.get("height"),
            "rotation": layer.get("rotation"),
            "scale": layer.get("scale"),
        }
    )
    _append_history(doc, op, note=f"Updated object {layer.get('concept')}")


def _apply_set_depth(doc: Dict[str, Any], scene_spec: Dict[str, Any], op: Dict[str, Any], scene_delta: Dict[str, Any]) -> None:
    layer = _resolve_object_layer(doc, op)
    if not layer:
        return
    depth_band = str(op.get("depth_band") or op.get("band") or layer.get("depth_band") or "midground")
    layer["depth_band"] = depth_band
    layer["depth_z"] = _band_depth(depth_band, op.get("depth_z"))
    if "z_index" in op:
        layer["z_index"] = _int(op.get("z_index"), layer.get("z_index", 0))
    scene_obj = _resolve_scene_object(scene_spec, layer)
    if scene_obj:
        _sync_object_to_scene(scene_obj, layer)
    scene_delta["objects"].append(
        {
            "object_id": layer.get("scene_object_id"),
            "depth_band": layer.get("depth_band"),
            "depth_z": layer.get("depth_z"),
            "z_index": layer.get("z_index"),
        }
    )
    _append_history(doc, op, note=f"Updated depth for {layer.get('concept')}")


def _apply_visibility(doc: Dict[str, Any], scene_spec: Dict[str, Any], op: Dict[str, Any], visible: bool, scene_delta: Dict[str, Any]) -> None:
    layer = _resolve_object_layer(doc, op)
    if not layer:
        return
    layer["visible"] = bool(visible)
    scene_obj = _resolve_scene_object(scene_spec, layer)
    if scene_obj:
        _sync_object_to_scene(scene_obj, layer)
    scene_delta["objects"].append({"object_id": layer.get("scene_object_id"), "visible": visible})
    _append_history(doc, op, note=f"{'Showed' if visible else 'Hid'} object {layer.get('concept')}")


def _apply_reorder(doc: Dict[str, Any], scene_spec: Dict[str, Any], op: Dict[str, Any], scene_delta: Dict[str, Any]) -> None:
    layer = _resolve_object_layer(doc, op)
    if not layer:
        return
    z_values = [_int(item.get("z_index"), 0) for item in doc.get("object_layers", []) or []]
    direction = str(op.get("direction") or "").strip().lower()
    if "z_index" in op:
        layer["z_index"] = _int(op.get("z_index"), layer.get("z_index", 0))
    elif direction == "front":
        layer["z_index"] = (max(z_values) if z_values else 0) + 1
    elif direction == "back":
        layer["z_index"] = (min(z_values) if z_values else 0) - 1
    scene_obj = _resolve_scene_object(scene_spec, layer)
    if scene_obj:
        _sync_object_to_scene(scene_obj, layer)
    scene_delta["objects"].append({"object_id": layer.get("scene_object_id"), "z_index": layer.get("z_index")})
    _append_history(doc, op, note=f"Reordered object {layer.get('concept')}")


def _apply_transform_region(doc: Dict[str, Any], op: Dict[str, Any], scene_delta: Dict[str, Any]) -> None:
    region = _resolve_region_layer(doc, op)
    if not region:
        return
    base_rect = _region_display_state(doc, region).get("base_rect") or {}
    transform = region.setdefault("local_transform", _default_region_transform())
    if "dx" in op:
        transform["tx"] = _float(transform.get("tx"), 0.0) + _float(op.get("dx"), 0.0)
    if "dy" in op:
        transform["ty"] = _float(transform.get("ty"), 0.0) + _float(op.get("dy"), 0.0)
    if "x" in op:
        transform["tx"] = _float(op.get("x"), base_rect.get("x", 0.0)) - _float(base_rect.get("x"), 0.0)
    if "y" in op:
        transform["ty"] = _float(op.get("y"), base_rect.get("y", 0.0)) - _float(base_rect.get("y"), 0.0)
    if "scale" in op:
        scale = max(0.25, min(3.0, _float(op.get("scale"), 1.0)))
        transform["scale_x"] = scale
        transform["scale_y"] = scale
    if "scale_x" in op:
        transform["scale_x"] = max(0.25, min(3.0, _float(op.get("scale_x"), 1.0)))
    if "scale_y" in op:
        transform["scale_y"] = max(0.25, min(3.0, _float(op.get("scale_y"), 1.0)))
    if "width" in op and _float(base_rect.get("width"), 0.0) > 0:
        transform["scale_x"] = max(0.25, min(3.0, _float(op.get("width"), base_rect.get("width", 1.0)) / _float(base_rect.get("width"), 1.0)))
    if "height" in op and _float(base_rect.get("height"), 0.0) > 0:
        transform["scale_y"] = max(0.25, min(3.0, _float(op.get("height"), base_rect.get("height", 1.0)) / _float(base_rect.get("height"), 1.0)))
    if "rotation" in op:
        transform["rotation"] = _float(op.get("rotation"), transform.get("rotation", 0.0))
    _ensure_region_promoted(region, action="transform", prompt=str(op.get("prompt") or ""))
    display = _region_display_state(doc, region)
    scene_delta["regions"].append(
        {
            "region_id": region.get("region_id"),
            "parent_object_id": region.get("parent_object_id"),
            "action": "transform",
            "current_rect": _copy(display.get("current_rect") or {}),
            "transform": _copy(transform),
        }
    )
    _append_history(doc, op, note=f"Updated region {region.get('label')}")


def _apply_region_visibility(doc: Dict[str, Any], op: Dict[str, Any], visible: bool, scene_delta: Dict[str, Any]) -> None:
    region = _resolve_region_layer(doc, op)
    if not region:
        return
    region["visible"] = bool(visible)
    action = "show" if visible else "hide"
    _ensure_region_promoted(region, action=action, prompt=str(op.get("prompt") or ""))
    scene_delta["regions"].append(
        {
            "region_id": region.get("region_id"),
            "parent_object_id": region.get("parent_object_id"),
            "action": action,
            "visible": bool(visible),
        }
    )
    _append_history(doc, op, note=f"{'Showed' if visible else 'Hid'} region {region.get('label')}")


def _apply_region_semantic_op(doc: Dict[str, Any], op: Dict[str, Any], action: str, scene_delta: Dict[str, Any]) -> None:
    region = _resolve_region_layer(doc, op)
    if not region:
        return
    prompt = str(op.get("prompt") or "").strip()
    _ensure_region_promoted(region, action=action, prompt=prompt, note=str(op.get("note") or ""))
    region["visible"] = True
    scene_delta["regions"].append(
        {
            "region_id": region.get("region_id"),
            "parent_object_id": region.get("parent_object_id"),
            "action": action,
            "prompt": prompt,
            "current_rect": _copy((_region_display_state(doc, region).get("current_rect") or {})),
        }
    )
    _append_history(doc, op, note=f"{_region_action_text(action)} region {region.get('label')}")


def _apply_restore_region(doc: Dict[str, Any], op: Dict[str, Any], scene_delta: Dict[str, Any]) -> bool:
    region = _resolve_region_layer(doc, op)
    if region:
        region["visible"] = True
        region["promoted"] = False
        region["edit_state"] = "idle"
        region["local_transform"] = _default_region_transform()
        region["render_intent"] = {"action": "idle", "prompt": "", "note": ""}
        scene_delta["regions"].append(
            {
                "region_id": region.get("region_id"),
                "parent_object_id": region.get("parent_object_id"),
                "action": "restore",
            }
        )
        _append_history(doc, op, note=f"Restored region {region.get('label')}")
        return True
    return False


def _apply_patch_layer(doc: Dict[str, Any], op: Dict[str, Any], kind: str, scene_delta: Dict[str, Any]) -> None:
    rect = _resolve_op_rect(doc, op)
    if not rect:
        return
    region_layer = _resolve_region_layer(doc, op)
    object_layer = _resolve_object_layer(doc, op)
    follow_region = bool(op.get("follow_region", False) and region_layer)
    follow_object = bool(op.get("follow_object", False) and object_layer and not follow_region)
    patch: Dict[str, Any] = {
        "id": f"patch_{uuid.uuid4().hex[:10]}",
        "kind": kind,
        "rect": rect,
        "prompt": str(op.get("prompt") or ""),
        "note": str(op.get("note") or ""),
        "anchor_object_id": object_layer.get("scene_object_id") if object_layer else "",
        "anchor_region_id": region_layer.get("region_id") if region_layer else "",
        "region_id": region_layer.get("region_id") if region_layer else "",
        "follow_region": follow_region,
        "follow_object": follow_object,
    }
    if follow_region and region_layer:
        region_rect = _region_display_state(doc, region_layer).get("current_rect") or {}
        region_w = max(1, _int(region_rect.get("width"), 1))
        region_h = max(1, _int(region_rect.get("height"), 1))
        patch["relative_rect"] = {
            "x": (rect["x"] - _int(region_rect.get("x"), 0)) / region_w,
            "y": (rect["y"] - _int(region_rect.get("y"), 0)) / region_h,
            "width": rect["width"] / region_w,
            "height": rect["height"] / region_h,
        }
    elif follow_object and object_layer:
        layer_w = max(1, _int(object_layer.get("width"), 1))
        layer_h = max(1, _int(object_layer.get("height"), 1))
        patch["relative_rect"] = {
            "x": (rect["x"] - _int(object_layer.get("x"), 0)) / layer_w,
            "y": (rect["y"] - _int(object_layer.get("y"), 0)) / layer_h,
            "width": rect["width"] / layer_w,
            "height": rect["height"] / layer_h,
        }
    doc.setdefault("patch_layers", []).append(patch)
    scene_delta["patches"].append({"patch_id": patch["id"], "kind": kind, "rect": rect})
    _append_history(doc, op, note=f"Added {kind} patch")


def _apply_restore_patch(doc: Dict[str, Any], op: Dict[str, Any], scene_delta: Dict[str, Any]) -> None:
    patch_layers = doc.get("patch_layers")
    if not isinstance(patch_layers, list) or not patch_layers:
        return
    target_id = str(op.get("patch_id") or "").strip()
    removed = None
    if target_id:
        for index, item in enumerate(list(patch_layers)):
            if str(item.get("id") or "") == target_id:
                removed = patch_layers.pop(index)
                break
    if removed is None:
        object_id = str(op.get("object_id") or "").strip()
        region_id = str(op.get("region_id") or "").strip()
        for index in range(len(patch_layers) - 1, -1, -1):
            item = patch_layers[index]
            if object_id and str(item.get("anchor_object_id") or "") != object_id:
                continue
            if region_id and str(item.get("region_id") or item.get("anchor_region_id") or "") != region_id:
                continue
            removed = patch_layers.pop(index)
            break
    if removed is None and patch_layers:
        removed = patch_layers.pop()
    if removed is not None:
        scene_delta["patches"].append({"restored_patch_id": removed.get("id")})
        _append_history(doc, op, note=f"Restored patch {removed.get('id')}")


def apply_patch_ops(
    *,
    preview: Dict[str, Any],
    session_id: str,
    sketch_backend: str = "sketch_v2",
    output_dir: str | Path = "outputs",
    ops: List[Dict[str, Any]] | None = None,
    source_candidate_id: str | None = None,
) -> Dict[str, Any]:
    prepared = ensure_editable_preview(
        preview=preview,
        session_id=session_id,
        sketch_backend=sketch_backend,
        output_dir=output_dir,
        force_rebuild=False,
        source_candidate_id=source_candidate_id,
    )
    if not isinstance(prepared, dict):
        return {"preview": preview, "scene_sync_delta": {"objects": [], "regions": [], "patches": []}, "render_sync_ready": False}
    if not prepared.get("editable_sketch_enabled"):
        return {"preview": prepared, "scene_sync_delta": {"objects": [], "regions": [], "patches": []}, "render_sync_ready": False}

    doc = _copy(prepared.get("editable_sketch_doc") or {})
    scene_spec = _copy(prepared.get("scene_spec") or {})
    ops = [item for item in (ops or []) if isinstance(item, dict)]
    scene_delta = {"objects": [], "regions": [], "patches": []}

    for op in ops:
        op_type = str(op.get("type") or op.get("op") or "").strip()
        if op_type == "transform_object":
            _apply_transform_object(doc, scene_spec, op, scene_delta)
        elif op_type == "set_depth":
            _apply_set_depth(doc, scene_spec, op, scene_delta)
        elif op_type == "hide_object":
            _apply_visibility(doc, scene_spec, op, False, scene_delta)
        elif op_type == "show_object":
            _apply_visibility(doc, scene_spec, op, True, scene_delta)
        elif op_type == "reorder_layer":
            _apply_reorder(doc, scene_spec, op, scene_delta)
        elif op_type == "transform_region":
            _apply_transform_region(doc, op, scene_delta)
        elif op_type == "hide_region":
            _apply_region_visibility(doc, op, False, scene_delta)
        elif op_type == "show_region":
            _apply_region_visibility(doc, op, True, scene_delta)
        elif op_type == "replace_region":
            _apply_region_semantic_op(doc, op, "replace", scene_delta)
        elif op_type == "emphasize_region":
            _apply_region_semantic_op(doc, op, "emphasize", scene_delta)
        elif op_type == "weaken_region":
            _apply_region_semantic_op(doc, op, "weaken", scene_delta)
        elif op_type == "restore_region":
            if not _apply_restore_region(doc, op, scene_delta):
                _apply_restore_patch(doc, op, scene_delta)
        elif op_type == "inpaint_region":
            if _resolve_region_layer(doc, op):
                _apply_region_semantic_op(doc, op, "inpaint", scene_delta)
            else:
                _apply_patch_layer(doc, op, op_type, scene_delta)
        elif op_type in {"erase_region", "brush_mask"}:
            _apply_patch_layer(doc, op, op_type, scene_delta)

    doc["revision_id"] = _new_revision_id()
    doc.setdefault("scene_sync_state", {})
    doc["scene_sync_state"]["synced_object_ids"] = sorted(
        {
            item.get("object_id")
            for item in scene_delta.get("objects", [])
            if isinstance(item, dict) and str(item.get("object_id") or "").strip()
        }
    )
    doc["scene_sync_state"]["pending_region_layers"] = len(
        [item for item in doc.get("region_layers", []) or [] if isinstance(item, dict) and _region_is_edited(item)]
    )
    doc["scene_sync_state"]["pending_patch_layers"] = len(doc.get("patch_layers", []) or [])
    doc["scene_sync_state"]["last_scene_delta"] = _copy(scene_delta)
    doc["depth_model"] = _default_depth_model({"object_instances": doc.get("object_layers", []) or []})
    doc = _compose_doc(doc)
    updated_preview = attach_doc_to_preview(prepared, scene_spec, doc)
    return {
        "preview": updated_preview,
        "scene_sync_delta": scene_delta,
        "render_sync_ready": bool((doc.get("render_sync_state") or {}).get("ready")),
    }
