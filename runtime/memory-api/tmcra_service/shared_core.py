from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "tmcra.service.shared-core-manifest.1"


class SharedCoreVerificationError(RuntimeError):
    pass


def verify_shared_core(
    root: str | Path, manifest_path: str | Path | None = None
) -> dict[str, str]:
    checkout = Path(root).resolve()
    manifest_file = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else checkout / "tmcra_service" / "shared_core_manifest.json"
    )
    try:
        manifest: Any = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SharedCoreVerificationError(
            f"shared-core manifest is unreadable: {manifest_file}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise SharedCoreVerificationError("shared-core manifest schema is invalid")
    files = manifest.get("algorithm_files")
    if not isinstance(files, list) or not files:
        raise SharedCoreVerificationError("shared-core manifest has no algorithm files")

    verified: dict[str, str] = {}
    for item in files:
        if not isinstance(item, dict):
            raise SharedCoreVerificationError("shared-core manifest entry is invalid")
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise SharedCoreVerificationError("shared-core path is invalid")
        if not isinstance(expected, str) or len(expected) != 64:
            raise SharedCoreVerificationError(
                f"shared-core digest is invalid: {relative}"
            )
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in verified:
            raise SharedCoreVerificationError(
                f"shared-core path is unsafe or duplicated: {relative}"
            )
        resolved = (checkout / path).resolve()
        try:
            resolved.relative_to(checkout)
        except ValueError as exc:
            raise SharedCoreVerificationError(
                f"shared-core path escaped the checkout: {relative}"
            ) from exc
        if not resolved.is_file():
            raise SharedCoreVerificationError(
                f"shared-core file is missing: {relative}"
            )
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != expected.lower():
            raise SharedCoreVerificationError(
                f"shared-core hash mismatch: {relative}"
            )
        verified[relative] = actual
    return verified
