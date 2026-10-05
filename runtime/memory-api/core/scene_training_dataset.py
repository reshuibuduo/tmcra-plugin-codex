from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_IMAGE_SIZE = 256


@dataclass(slots=True)
class SceneTrainingRecord:
    dataset_id: str
    scene_type: str
    style_id: str
    source_family: str
    recommended_use: str
    split: str
    image_path: str
    component_count: int
    mapped_component_count: int
    mapped_internal_classes: List[str]
    metadata: Dict[str, Any]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _split_from_bucket(bucket: str, fallback: str = "train") -> str:
    text = str(bucket or "").strip().lower()
    if text in {"primary_val", "style_val"}:
        return "val"
    if text in {"primary_test", "style_test", "holdout_eval"}:
        return "test"
    if text:
        return "train"
    return fallback


def _infer_image_path(row: Dict[str, Any]) -> str:
    drawing_png = str(row.get("drawing_png") or "").strip()
    if drawing_png:
        return drawing_png
    drawing_svg = str(row.get("drawing_svg") or "").strip()
    return drawing_svg


def registry_rows_to_scene_records(
    rows: Sequence[Dict[str, Any]],
    *,
    min_mapped_components: int = 1,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("record_type") or "") != "scene_sample":
            continue
        if str(row.get("readiness") or "") != "ready":
            continue
        image_path = _infer_image_path(row)
        if not image_path or not Path(image_path).exists():
            continue
        mapped_component_count = int(row.get("mapped_component_count", 0) or 0)
        allow_unmapped_training = bool(row.get("allow_unmapped_training"))
        if mapped_component_count < min_mapped_components and not allow_unmapped_training:
            continue
        recommended_bucket = str(row.get("recommended_bucket") or "")
        records.append(
            asdict(
                SceneTrainingRecord(
                    dataset_id=str(row.get("dataset_id") or "unknown"),
                    scene_type=str(row.get("scene_type") or "scene"),
                    style_id=str(row.get("style_variant") or "unknown"),
                    source_family=str(row.get("source_family") or "unknown"),
                    recommended_use=str(row.get("recommended_use") or "scene_sketch_pretrain"),
                    split=_split_from_bucket(recommended_bucket),
                    image_path=image_path,
                    component_count=int(row.get("component_count", 0) or 0),
                    mapped_component_count=mapped_component_count,
                    mapped_internal_classes=[
                        str(item)
                        for item in (row.get("mapped_internal_classes") or [])
                        if str(item).strip()
                    ],
                    metadata={
                        "dataset_id": row.get("dataset_id"),
                        "archive_name": row.get("archive_name"),
                        "scene_id": row.get("scene_id"),
                        "recommended_bucket": recommended_bucket,
                        "recommended_extract_dir": row.get("recommended_extract_dir"),
                        "component_labels": _copy(row.get("component_labels") or []),
                        "allow_unmapped_training": allow_unmapped_training,
                    },
                )
            )
        )
    return records


def build_scene_index_maps(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    scene_types = sorted({str(row.get("scene_type") or "scene") for row in rows})
    style_ids = sorted({str(row.get("style_id") or "unknown") for row in rows})
    source_families = sorted({str(row.get("source_family") or "unknown") for row in rows})
    recommended_uses = sorted({str(row.get("recommended_use") or "scene_sketch_pretrain") for row in rows})
    internal_classes = sorted(
        {
            str(class_id)
            for row in rows
            for class_id in (row.get("mapped_internal_classes") or [])
            if str(class_id).strip()
        }
    )
    return {
        "scene_to_idx": {value: idx for idx, value in enumerate(scene_types)},
        "style_to_idx": {value: idx for idx, value in enumerate(style_ids)},
        "source_family_to_idx": {value: idx for idx, value in enumerate(source_families)},
        "use_to_idx": {value: idx for idx, value in enumerate(recommended_uses)},
        "class_to_idx": {value: idx for idx, value in enumerate(internal_classes)},
    }


def split_scene_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    payload = {"train": [], "val": [], "test": []}
    for row in rows:
        split = str(row.get("split") or "train")
        payload.setdefault(split, []).append(_copy(row))
    return payload


def build_scene_manifest(split_rows_payload: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    all_rows = [row for rows in split_rows_payload.values() for row in rows]
    by_dataset = Counter(str(row.get("dataset_id") or "") for row in all_rows)
    by_scene = Counter(str(row.get("scene_type") or "scene") for row in all_rows)
    by_style = Counter(str(row.get("style_id") or "unknown") for row in all_rows)
    return {
        "row_count": len(all_rows),
        "dataset_count": len(by_dataset),
        "scene_type_count": len(by_scene),
        "style_count": len(by_style),
        "splits": {
            split_name: {"row_count": len(rows)}
            for split_name, rows in split_rows_payload.items()
        },
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_scene_type": dict(sorted(by_scene.items())),
        "by_style": dict(sorted(by_style.items())),
    }


def prepare_scene_dataset_from_registry(
    *,
    registry_path: Path,
    output_dir: Path,
    min_mapped_components: int = 1,
) -> Dict[str, Any]:
    registry_rows = read_jsonl(registry_path)
    scene_rows = registry_rows_to_scene_records(
        registry_rows,
        min_mapped_components=min_mapped_components,
    )
    split_map = split_scene_rows(scene_rows)
    mappings = build_scene_index_maps(scene_rows)
    manifest = build_scene_manifest(split_map)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in split_map.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "mappings.json", mappings)
    return {
        "output_dir": str(output_dir),
        "row_count": len(scene_rows),
        "manifest": manifest,
    }


def _load_image_grayscale(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((image_size, image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor


def _augment_image(image: torch.Tensor, rng: random.Random) -> torch.Tensor:
    output = image.clone()
    if rng.random() < 0.5:
        output = torch.flip(output, dims=[2])
    if rng.random() < 0.25:
        output = torch.flip(output, dims=[1])
    if rng.random() < 0.3:
        output = torch.clamp(output + rng.uniform(-0.08, 0.08), 0.0, 1.0)
    if rng.random() < 0.25:
        noise = torch.randn_like(output) * rng.uniform(0.01, 0.04)
        output = torch.clamp(output + noise, 0.0, 1.0)
    return output


class PreparedSceneTrainingDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        mappings: Dict[str, Dict[str, int]],
        *,
        image_size: int = DEFAULT_IMAGE_SIZE,
        augment: bool = False,
        seed: int = 42,
    ):
        self.rows = list(rows)
        self.mappings = mappings
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.class_count = len(mappings.get("class_to_idx") or {})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        path = Path(str(row.get("image_path") or ""))
        image = _load_image_grayscale(path, self.image_size)
        if self.augment:
            image = _augment_image(image, random.Random(self.seed + index))
        class_target = torch.zeros(self.class_count, dtype=torch.float32)
        for class_id in row.get("mapped_internal_classes") or []:
            class_index = self.mappings["class_to_idx"].get(str(class_id))
            if class_index is not None:
                class_target[class_index] = 1.0
        return {
            "image": image,
            "scene_id": torch.tensor(self.mappings["scene_to_idx"][str(row.get("scene_type") or "scene")], dtype=torch.long),
            "style_id": torch.tensor(self.mappings["style_to_idx"][str(row.get("style_id") or "unknown")], dtype=torch.long),
            "source_family_id": torch.tensor(self.mappings["source_family_to_idx"][str(row.get("source_family") or "unknown")], dtype=torch.long),
            "use_id": torch.tensor(self.mappings["use_to_idx"][str(row.get("recommended_use") or "scene_sketch_pretrain")], dtype=torch.long),
            "component_count": torch.tensor(math.log1p(float(row.get("component_count", 0) or 0)), dtype=torch.float32),
            "mapped_component_count": torch.tensor(math.log1p(float(row.get("mapped_component_count", 0) or 0)), dtype=torch.float32),
            "class_target": class_target,
            "image_path": str(path),
            "raw_row": row,
        }


def collate_scene_training_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "scene_id": torch.stack([item["scene_id"] for item in batch], dim=0),
        "style_id": torch.stack([item["style_id"] for item in batch], dim=0),
        "source_family_id": torch.stack([item["source_family_id"] for item in batch], dim=0),
        "use_id": torch.stack([item["use_id"] for item in batch], dim=0),
        "component_count": torch.stack([item["component_count"] for item in batch], dim=0),
        "mapped_component_count": torch.stack([item["mapped_component_count"] for item in batch], dim=0),
        "class_target": torch.stack([item["class_target"] for item in batch], dim=0),
        "image_path": [item["image_path"] for item in batch],
        "raw_row": [item["raw_row"] for item in batch],
    }
