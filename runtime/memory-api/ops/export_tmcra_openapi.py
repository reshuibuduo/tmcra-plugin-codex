#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmcra_service.app import create_app
from tmcra_service.settings import ServiceSettings


def _public_service_version() -> str:
    manifest_path = ROOT / "tmcra_service" / "shared_core_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("service_version")
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(f"missing service_version in {manifest_path}")
    return version.strip()


def _schema_settings(root: Path, *, server_url: str) -> ServiceSettings:
    required_files = {
        "writer_env": root / "writer.env",
        "native_harness": root / "native_harness.py",
        "node_model": root / "node.pt",
        "path_model": root / "path.pt",
        "checkpoint": root / "checkpoint.pt",
    }
    for path in required_files.values():
        path.write_text("openapi-export\n", encoding="utf-8")
    return ServiceSettings(
        state_dir=root / "state",
        control_db=root / "state" / "control.sqlite3",
        bind_host="127.0.0.1",
        bind_port=2009,
        public_base_url=server_url.rstrip("/"),
        v4_root=root,
        integrated_repo=root,
        writer_env=required_files["writer_env"],
        embedding_model=root,
        native_harness=required_files["native_harness"],
        node_model=required_files["node_model"],
        path_model=required_files["path_model"],
        checkpoint=required_files["checkpoint"],
        cross_model=root,
        device="cpu",
        graph_device="cpu",
        request_body_limit=2 * 1024 * 1024,
        provider_lease_seconds=300,
        provider_key_concurrency=1,
        disk_free_min_bytes=1,
        preload_online_engine=False,
    )


def normalized_openapi(app: FastAPI, *, server_url: str) -> dict[str, Any]:
    schema = dict(app.openapi())
    info = dict(schema.get("info") or {})
    info["version"] = _public_service_version()
    schema["info"] = info
    schema["servers"] = [
        {"url": server_url.rstrip("/"), "description": "TMCRA Memory API"}
    ]
    return schema


def write_openapi(app: FastAPI, output: Path, *, server_url: str) -> None:
    schema = normalized_openapi(app, server_url=server_url)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(schema, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the TMCRA public OpenAPI contract.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-url", default="https://api.tmcra.example")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="tmcra-openapi-") as directory:
        settings = _schema_settings(Path(directory), server_url=args.server_url)
        write_openapi(
            create_app(settings),
            args.output,
            server_url=args.server_url,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
