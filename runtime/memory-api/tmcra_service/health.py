from __future__ import annotations

import os
import shutil
import sqlite3
from contextlib import closing
from typing import Any

from .settings import ServiceSettings


def readiness(settings: ServiceSettings) -> tuple[bool, dict[str, Any]]:
    checks: dict[str, Any] = {}
    for name, path in settings.required_paths().items():
        path_ok = (
            path.is_file()
            if name
            in {
                "audio_asr_api_key_file",
                "writer_env",
                "native_harness",
                "node_model",
                "path_model",
                "checkpoint",
            }
            else path.exists()
        )
        checks[name] = {"ok": path_ok, "path": str(path)}
    database_ok = False
    database_error = ""
    quick_check_result: Any = None
    try:
        settings.control_db.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(settings.control_db, timeout=5.0)) as connection:
            row = connection.execute("PRAGMA quick_check").fetchone()
            quick_check_result = row[0] if row and len(row) == 1 else None
            if quick_check_result != "ok":
                raise RuntimeError(
                    f"SQLite quick_check returned {quick_check_result!r}, expected 'ok'"
                )
            connection.execute("SELECT 1").fetchone()
        database_ok = True
    except Exception as exc:
        database_error = f"{type(exc).__name__}: {exc}"
    checks["control_db"] = {
        "ok": database_ok,
        "path": str(settings.control_db),
        "quick_check": quick_check_result,
        "error": database_error,
    }
    usage = shutil.disk_usage(settings.state_dir.parent)
    disk_ok = usage.free >= settings.disk_free_min_bytes
    checks["disk"] = {
        "ok": disk_ok,
        "free_bytes": usage.free,
        "required_free_bytes": settings.disk_free_min_bytes,
    }
    raw_keys = os.getenv("TMCRA_WRITER_API_KEY_POOL") or os.getenv(
        "TMCRA_DEEPSEEK_WRITER_KEY_POOL", ""
    )
    parts = raw_keys.split(",") if raw_keys else []
    keys = [value.strip() for value in parts]
    base_url = os.getenv("TMCRA_WRITER_BASE_URL") or os.getenv(
        "TMCRA_DEEPSEEK_WRITER_BASE_URL", ""
    )
    key_error = ""
    if not raw_keys:
        key_error = "writer API key pool is missing"
    elif any(not value for value in keys):
        key_error = "writer API key pool contains an empty entry"
    elif len(keys) != len(set(keys)):
        key_error = "writer API key pool contains duplicate keys"
    if not base_url:
        key_error = (
            (key_error + "; " if key_error else "")
            + "writer base URL is missing"
        )
    checks["provider_pool"] = {
        "ok": not key_error,
        "key_count": len(keys),
        "unique_key_count": len(set(keys)),
        "error": key_error,
        "base_url": base_url,
        "writer_model": os.getenv("TMCRA_WRITER_MODEL", ""),
        "writer_max_tokens": os.getenv("TMCRA_WRITER_MAX_TOKENS", ""),
    }
    ready = all(bool(value["ok"]) for value in checks.values())
    return ready, {"status": "ready" if ready else "not_ready", "checks": checks}
