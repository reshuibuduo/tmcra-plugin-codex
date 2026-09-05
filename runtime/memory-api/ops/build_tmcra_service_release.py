#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Iterable


STATIC_FILES = (
    "deploy/install-tmcra.sh",
    "deploy/tmcra",
    "deploy/tmcra-local-llm-control.sh",
    "deploy/tmcra-memory-api-control.sh",
    "deploy/tmcra-memory-api.service",
    "deploy/tmcra-production-maintenance.sh",
    "deploy/model-manifests/bge-reranker-v2-m3.TMCRA_MODEL_MANIFEST.json",
    "deploy/tmcra-service.env.example",
    "deploy/writer.env.example",
    "requirements-tmcra-service.txt",
    "models/tmcra_v3_reranker.pt",
    "models/README.md",
    "ops/build_tmcra_service_release.py",
    "ops/export_tmcra_openapi.py",
    "ops/run_tmcra_service_preflight.py",
    "ops/run_commercial_api_smoke.py",
)

EXCLUDED_TREE_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "node_modules",
}

EXCLUDED_TREE_SUFFIXES = (
    ".egg-info",
    ".tgz",
)

FORBIDDEN_SUFFIXES = (
    ".db",
    ".jsonl",
    ".log",
    ".pyc",
    ".sqlite",
    ".sqlite3",
    ".sqlite3-shm",
    ".sqlite3-wal",
)

SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
)

ENV_SECRET_ASSIGNMENT = re.compile(
    rb"(?m)^(?:[A-Z][A-Z0-9_]*_)?(?:API_KEY|TOKEN|SECRET|PASSWORD)="
    rb"(?P<value>[^\r\n#]+)"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_placeholder(value: bytes) -> bool:
    normalized = value.strip().lower()
    return not normalized or any(
        marker in normalized
        for marker in (b"example", b"changeme", b"replace", b"xxxx", b"<")
    )


def _scan_payload(relative_path: Path, payload: bytes) -> None:
    for pattern in SECRET_PATTERNS:
        if pattern.search(payload):
            raise ValueError(f"suspected secret in release member: {relative_path}")
    for match in ENV_SECRET_ASSIGNMENT.finditer(payload):
        if not _is_placeholder(match.group("value")):
            raise ValueError(
                f"populated secret assignment in release member: {relative_path}"
            )


def _validate_relative_path(relative_path: Path) -> None:
    posix = relative_path.as_posix()
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"unsafe release path: {relative_path}")
    if "__pycache__" in relative_path.parts or posix.endswith(FORBIDDEN_SUFFIXES):
        raise ValueError(f"runtime artifact cannot enter release: {relative_path}")
    if relative_path.name.endswith(".key.json"):
        raise ValueError(f"private key metadata cannot enter release: {relative_path}")
    if relative_path.name.endswith(".env"):
        raise ValueError(f"live environment file cannot enter release: {relative_path}")


def _algorithm_files(root: Path) -> list[Path]:
    manifest_path = root / "tmcra_service" / "shared_core_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: list[Path] = []
    for entry in manifest.get("algorithm_files", []):
        relative_path = Path(str(entry["path"]))
        path = root / relative_path
        if _file_sha256(path) != str(entry["sha256"]):
            raise ValueError(f"shared-core hash mismatch: {relative_path}")
        files.append(relative_path)
    if not files:
        raise ValueError("shared-core manifest contains no algorithm files")
    return files


def _local_module_path(root: Path, module_name: str) -> Path | None:
    if not module_name or any(not part for part in module_name.split(".")):
        return None
    base = root.joinpath(*module_name.split("."))
    candidates = (base.with_suffix(".py"), base / "__init__.py")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.relative_to(root)
    return None


def _imported_module_names(relative_path: Path, payload: str) -> set[str]:
    try:
        tree = ast.parse(payload, filename=relative_path.as_posix())
    except SyntaxError as exc:
        raise ValueError(f"cannot parse Python release member: {relative_path}") from exc

    package_parts = list(relative_path.parent.parts)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            trim = max(0, node.level - 1)
            base_parts = package_parts[: len(package_parts) - trim]
            if node.module:
                base_parts.extend(node.module.split("."))
            base = ".".join(base_parts)
        else:
            base = node.module or ""
        if base:
            names.add(base)
        for alias in node.names:
            if alias.name != "*" and base:
                names.add(f"{base}.{alias.name}")
    return names


def _local_python_dependencies(root: Path, seed_members: Iterable[Path]) -> set[Path]:
    discovered: set[Path] = set()
    pending = [path for path in seed_members if path.suffix == ".py"]
    parsed: set[Path] = set()
    while pending:
        relative_path = pending.pop()
        if relative_path in parsed:
            continue
        parsed.add(relative_path)
        path = root / relative_path
        for module_name in _imported_module_names(
            relative_path, path.read_text(encoding="utf-8")
        ):
            dependency = _local_module_path(root, module_name)
            if dependency is None or dependency in parsed or dependency in discovered:
                continue
            discovered.add(dependency)
            pending.append(dependency)
    return discovered


def release_members(root: Path) -> list[Path]:
    members = {Path(item) for item in STATIC_FILES}
    members.update(_algorithm_files(root))
    members.update(
        path.relative_to(root)
        for path in (root / "tmcra_service").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    members.update(
        path.relative_to(root)
        for path in root.glob("test_tmcra_service_*.py")
        if path.is_file()
    )
    members.update(_local_python_dependencies(root, members))

    ordered = sorted(members, key=lambda item: item.as_posix())
    for relative_path in ordered:
        _validate_relative_path(relative_path)
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"required release member missing: {relative_path}")
        _scan_payload(relative_path, path.read_bytes())
    return ordered


def _normalized_tar_info(path: Path, arcname: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(arcname)
    info.size = path.stat().st_size
    info.mode = 0o755 if path.suffix == ".sh" or path.name == "build_tmcra_service_release.py" else 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _bytes_tar_info(arcname: str, payload: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(arcname)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def build_release(root: Path, output: Path) -> dict[str, object]:
    root = root.resolve()
    output = output.resolve()
    members = release_members(root)
    file_hashes = {
        relative_path.as_posix(): _file_sha256(root / relative_path)
        for relative_path in members
    }
    release_manifest = {
        "schema_version": "tmcra.memory-service-release.1",
        "files": file_hashes,
        "forbidden_runtime_state_included": False,
    }
    manifest_payload = (
        json.dumps(release_manifest, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, compresslevel=9, mtime=0
        ) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                archive.addfile(
                    _bytes_tar_info("RELEASE_MANIFEST.json", manifest_payload),
                    io.BytesIO(manifest_payload),
                )
                for relative_path in members:
                    path = root / relative_path
                    with path.open("rb") as member_handle:
                        archive.addfile(
                            _normalized_tar_info(path, relative_path.as_posix()),
                            member_handle,
                        )

    return {
        "output": str(output),
        "sha256": _file_sha256(output),
        "size_bytes": output.stat().st_size,
        "member_count": len(members) + 1,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, secret-scanned TMCRA service release."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_release(args.root, args.output)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
