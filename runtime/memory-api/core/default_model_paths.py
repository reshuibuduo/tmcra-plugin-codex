from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_IMPORTS_DIR = PROJECT_ROOT / "outputs" / "remote_imports"

LEGACY_OBJECT_VARIANTS_PATH = PROJECT_ROOT / "data" / "object_sketch" / "exported_shape_variants.json"
LEGACY_OBJECT_STROKE_VARIANTS_PATH = PROJECT_ROOT / "data" / "object_sketch" / "exported_stroke_variants_v1.json"
LEGACY_SCENE_DATASET_CANDIDATES = (
    PROJECT_ROOT / "data" / "scene_training_prep" / "scene_stage1_v2_fscoco_legacy-user_remote",
    PROJECT_ROOT / "data" / "scene_training_prep" / "scene_stage1_v2_fscoco_remote",
    PROJECT_ROOT / "data" / "scene_training_prep" / "scene_stage1_v1",
)
LEGACY_SKETCH_LORA_RUN_DIR = PROJECT_ROOT / "outputs" / "sketch_v2_lora_v1"
DEFAULT_SKETCH_LORA_ALIAS = "tmcra_sketch_v2_preview.safetensors"


def _env_path(name: str) -> Path | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _pick_existing(*candidates: Path | None) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def resolve_remote_import_root() -> Path | None:
    env_root = _env_path("TMCRA_IMPORTED_ARTIFACT_ROOT")
    if env_root and env_root.exists():
        return env_root
    if not REMOTE_IMPORTS_DIR.exists():
        return None
    candidates = sorted(
        [item for item in REMOTE_IMPORTS_DIR.iterdir() if item.is_dir()],
        key=lambda item: (item.name, item.stat().st_mtime),
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_default_object_variants_path() -> Path:
    remote_root = resolve_remote_import_root()
    candidate = _pick_existing(
        remote_root / "object" / "runtime_export" / "exported_shape_variants_scale.json" if remote_root else None,
        remote_root / "object" / "runtime_export" / "exported_shape_variants.json" if remote_root else None,
        LEGACY_OBJECT_VARIANTS_PATH,
    )
    return candidate or LEGACY_OBJECT_VARIANTS_PATH


def resolve_default_object_stroke_variants_path() -> Path:
    remote_root = resolve_remote_import_root()
    candidate = _pick_existing(
        remote_root / "object" / "runtime_export" / "exported_stroke_variants_v1.json" if remote_root else None,
        LEGACY_OBJECT_STROKE_VARIANTS_PATH,
    )
    return candidate or LEGACY_OBJECT_STROKE_VARIANTS_PATH


def resolve_default_scene_dataset_dir() -> Path:
    remote_root = resolve_remote_import_root()
    candidate = _pick_existing(
        remote_root / "scene" / "datasets" / "scene_stage1_v2_fscoco_legacy-user_remote" if remote_root else None,
        remote_root / "scene" / "datasets" / "scene_stage1_fscoco_only_legacy-user_remote" if remote_root else None,
        *LEGACY_SCENE_DATASET_CANDIDATES,
    )
    return candidate or LEGACY_SCENE_DATASET_CANDIDATES[0]


def resolve_default_sketch_lora_run_dir() -> Path:
    remote_root = resolve_remote_import_root()
    candidate = _pick_existing(
        remote_root / "sketch" / "latest_run" if remote_root else None,
        LEGACY_SKETCH_LORA_RUN_DIR,
    )
    return candidate or LEGACY_SKETCH_LORA_RUN_DIR


def resolve_default_sketch_lora_path() -> Path:
    run_dir = resolve_default_sketch_lora_run_dir()
    candidate = _pick_existing(
        run_dir / "pytorch_lora_weights.safetensors",
        run_dir / "checkpoint-2000" / "pytorch_lora_weights.safetensors",
    )
    return candidate or (run_dir / "pytorch_lora_weights.safetensors")


def resolve_default_sketch_lora_alias() -> str:
    return str(os.getenv("TMCRA_SKETCH_LORA_ALIAS", DEFAULT_SKETCH_LORA_ALIAS) or DEFAULT_SKETCH_LORA_ALIAS).strip()
