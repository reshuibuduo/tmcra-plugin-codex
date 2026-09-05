#!/usr/bin/env python3
"""Parse production API configs without printing secrets or making calls."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_tmcra_v4_build import (
    DEFAULT_WRITER_ENV,
    _key_pool,
    _load_shell_environment,
    _worker_environment,
)
from run_tmcra_v4_gpt54_answers import load_harness
from tmcra_v4_slow_graph import TieredGraphPatchManager


ANSWER_ENV = Path(
    "/opt/tmcra-data/migration/legacy/"
    "tmcra_longmemeval/env/answer-vectorengine-gpt54.env"
)
HARNESS = Path(
    "/opt/tmcra-data/migration/legacy/"
    "tmcra_longmemeval/scripts/run_lme_s10_native_tmcra.py"
)


def main() -> int:
    writer_environment = _load_shell_environment(DEFAULT_WRITER_ENV)
    keys = _key_pool(writer_environment)
    worker_environment = _worker_environment(writer_environment, keys, 0)
    with patch.dict(os.environ, worker_environment, clear=True):
        manager = TieredGraphPatchManager.from_env()
    if manager.flash is None or manager.pro is None:
        raise RuntimeError("slow-graph writer/reviewer configuration is incomplete")

    answer_environment = _load_shell_environment(ANSWER_ENV)
    with patch.dict(os.environ, answer_environment, clear=True):
        harness = load_harness(HARNESS)
        answer_base_url, answer_model, answer_key = harness.answer_llm_config()
    if not answer_model or not answer_base_url or not answer_key:
        raise RuntimeError("answer model configuration is incomplete")

    report = {
        "schema_version": "tmcra.v4.api-config-preflight.1",
        "physical_api_calls": 0,
        "deepseek_key_count": len(keys),
        "writer_max_tokens": int(
            worker_environment.get("TMCRA_WRITER_MAX_TOKENS", "16384")
        ),
        "slow_flash_model": manager.flash.config.model,
        "slow_pro_model": manager.pro.config.model,
        "slow_max_tokens": manager.flash.config.max_tokens,
        "deepseek_endpoint_host": urlparse(manager.flash.config.base_url).netloc,
        "answer_model": answer_model,
        "answer_endpoint_host": urlparse(answer_base_url).netloc,
        "answer_key_present": bool(answer_key),
        "status": "passed",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
