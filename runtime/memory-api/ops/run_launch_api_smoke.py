#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from tmcra_client import IngestRequest, MemoryMessage, RecallRequest, SyncClient
except ModuleNotFoundError:
    SDK_ROOT = (
        Path(__file__).resolve().parents[2]
        / "06-tmcra-sdk-integrations"
        / "sdk"
        / "python"
    )
    if SDK_ROOT.is_dir():
        sys.path.insert(0, str(SDK_ROOT))
    from tmcra_client import IngestRequest, MemoryMessage, RecallRequest, SyncClient


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real TMCRA ingest and recall through the Python SDK."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--scope", default="")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    api_key = os.getenv("TMCRA_SMOKE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TMCRA_SMOKE_API_KEY is required")

    run_id = uuid.uuid4().hex
    marker = f"tmcra-launch-{run_id[:12]}"
    scope = args.scope or f"launch-smoke-{run_id[:12]}"
    session_id = f"session-{run_id[:12]}"
    now = datetime.now(timezone.utc)
    request = IngestRequest(
        session_id=session_id,
        messages=[
            MemoryMessage(
                message_id=f"user-{run_id}",
                role="user",
                content=f"Remember that my launch verification code is {marker}.",
                timestamp=now,
            ),
            MemoryMessage(
                message_id=f"assistant-{run_id}",
                role="assistant",
                content=f"I will remember the verification code {marker}.",
                timestamp=now,
            ),
        ],
        consistency="read_your_writes",
        slow_policy="auto",
        metadata={"source": "launch-api-smoke", "run_id": run_id},
    )

    started = time.monotonic()
    with SyncClient(args.base_url, api_key=api_key, timeout=60.0) as client:
        health = client.healthz()
        ready = client.readyz()
        accepted = client.ingest(
            scope,
            request,
            idempotency_key=f"launch-smoke-{run_id}",
        )
        completed = client.wait_for_job(
            accepted.job_id,
            timeout=args.timeout,
            poll_interval=1.0,
            max_poll_interval=5.0,
        )
        if not completed.succeeded:
            raise RuntimeError(
                f"ingest job ended as {completed.status}: {completed.error}"
            )
        recalled = client.recall(
            scope,
            RecallRequest(
                query="What is my launch verification code?",
                evidence_mode="auto",
                max_windows=8,
                wait_for_job_id=completed.job_id,
            ),
        )

    evidence = recalled.prompt_evidence.content
    if marker not in evidence:
        raise RuntimeError("recall completed but did not contain the ingested marker")
    report = {
        "schema_version": "tmcra.launch-api-smoke.1",
        "status": "passed",
        "scope": scope,
        "job_id": completed.job_id,
        "query_id": recalled.query_id,
        "selected_evidence_mode": recalled.evidence_route.selected,
        "route_reasons": list(recalled.evidence_route.reasons),
        "prompt_evidence_character_count": len(evidence),
        "health": health.status,
        "readiness": ready.status,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
