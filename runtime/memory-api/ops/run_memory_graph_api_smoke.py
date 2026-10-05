#!/usr/bin/env python3
"""Run an isolated, real-data smoke test for the memory graph API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tmcra_service.auth import APIKeyAuth
from tmcra_service.control_db import ControlDB


class SmokeError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an isolated tenant, ingest two neutral messages, validate all "
            "memory-graph endpoints, and revoke the temporary API key."
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--control-db", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--reuse-tenant", default="")
    parser.add_argument("--reuse-scope", default="")
    return parser


def _request_json(
    method: str,
    url: str,
    *,
    api_key: str = "",
    payload: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: int = 120,
) -> tuple[int, dict[str, Any]]:
    data = None
    request_headers = {"Accept": "application/json", **dict(headers or {})}
    if api_key:
        request_headers["Authorization"] = f"Bearer {api_key}"
    if payload is not None:
        data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json; charset=utf-8"
        request_headers["Content-Length"] = str(len(data))
    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {}
        return exc.code, body
    except (TimeoutError, urllib.error.URLError) as exc:
        raise SmokeError(f"request failed: {method} {url}: {exc}") from exc


def _expect_status(
    actual: int,
    expected: int,
    operation: str,
    body: Mapping[str, Any],
) -> None:
    if actual == expected:
        return
    detail = json.dumps(dict(body), ensure_ascii=False, sort_keys=True)[:1000]
    raise SmokeError(f"{operation} returned HTTP {actual}, expected {expected}: {detail}")


def _wait_job(
    base_url: str,
    api_key: str,
    job_id: str,
    *,
    timeout: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    encoded_job_id = urllib.parse.quote(job_id, safe="")
    while True:
        status, job = _request_json(
            "GET",
            f"{base_url}/v1/jobs/{encoded_job_id}",
            api_key=api_key,
            timeout=30,
        )
        _expect_status(status, 200, "job status", job)
        state = str(job.get("status") or "")
        if state == "succeeded":
            return job
        if state in {"failed", "cancelled"}:
            error = job.get("error") or {}
            raise SmokeError(
                f"ingest job ended as {state}: "
                f"{json.dumps(error, ensure_ascii=False, sort_keys=True)[:1000]}"
            )
        if time.monotonic() >= deadline:
            raise SmokeError(f"ingest job timed out in state {state!r}")
        time.sleep(2)


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(report), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run(
    *,
    base_url: str,
    control_db: Path,
    timeout: int,
    cleanup: dict[str, Any],
    reuse_tenant: str = "",
    reuse_scope: str = "",
) -> dict[str, Any]:
    if not base_url.startswith("https://"):
        raise SmokeError("--base-url must use HTTPS")
    base_url = base_url.rstrip("/")
    if not control_db.is_file():
        raise SmokeError(f"control database does not exist: {control_db}")

    db = ControlDB(control_db)
    auth = APIKeyAuth(db)
    run_id = uuid.uuid4().hex
    if bool(reuse_tenant) != bool(reuse_scope):
        raise SmokeError("--reuse-tenant and --reuse-scope must be supplied together")
    tenant_id = reuse_tenant or f"graph-smoke-{run_id[:12]}"
    scope_name = reuse_scope or f"graph-scope-{run_id[:12]}"
    permissions = {"memory:read", "memory:write", "memory:consolidate"}
    if not reuse_tenant:
        auth.set_tenant_scopes(tenant_id, permissions)
    issued = auth.create_key(tenant_id, permissions)
    cleanup.update(auth=auth, issued=issued)

    encoded_scope = urllib.parse.quote(scope_name, safe="")
    scope_url = f"{base_url}/v1/scopes/{encoded_scope}"
    if reuse_tenant:
        ingest_status = "reused_committed_snapshot"
    else:
        now = datetime.now(timezone.utc).isoformat()
        status, accepted = _request_json(
            "POST",
            f"{scope_url}/ingest",
            api_key=issued.api_key,
            payload={
                "session_id": "memory-graph-release-smoke",
                "messages": [
                    {
                        "message_id": f"{run_id}-user",
                        "role": "user",
                        "content": (
                            "For the release check, use the canary channel and the "
                            "09:30 UTC deployment window."
                        ),
                        "timestamp": now,
                    },
                    {
                        "message_id": f"{run_id}-assistant",
                        "role": "assistant",
                        "content": (
                            "I will retain the canary channel and 09:30 UTC deployment "
                            "window for the release check."
                        ),
                        "timestamp": now,
                    },
                ],
                "consistency": "read_your_writes",
                "slow_policy": "force",
                "metadata": {"source": "memory-graph-api-smoke"},
            },
            headers={"Idempotency-Key": f"memory-graph-smoke-{run_id}"},
            timeout=60,
        )
        _expect_status(status, 202, "ingest", accepted)
        job_id = str(accepted.get("job_id") or "")
        if not job_id:
            raise SmokeError("ingest response has no job_id")
        job = _wait_job(base_url, issued.api_key, job_id, timeout=timeout)
        ingest_status = str(job.get("status") or "")

    status, overview = _request_json(
        "GET",
        f"{scope_url}/memory-graph?layers=slow&limit=20",
        api_key=issued.api_key,
        timeout=60,
    )
    _expect_status(status, 200, "memory graph overview", overview)
    nodes = list(overview.get("nodes") or [])
    if not nodes:
        raise SmokeError("memory graph overview returned no nodes")
    if any("text" in node or "text" in (node.get("attributes") or {}) for node in nodes):
        raise SmokeError("memory graph overview leaked Source text")

    root_id = urllib.parse.quote(str(nodes[0].get("id") or ""), safe="")
    status, neighbors = _request_json(
        "GET",
        (
            f"{scope_url}/memory-graph/nodes/{root_id}/neighbors"
            "?depth=2&layers=slow,fast,source&limit=30"
        ),
        api_key=issued.api_key,
        timeout=60,
    )
    _expect_status(status, 200, "memory graph neighbors", neighbors)
    expanded_nodes = list(neighbors.get("nodes") or [])
    if any(
        "text" in node or "text" in (node.get("attributes") or {})
        for node in expanded_nodes
    ):
        raise SmokeError("memory graph neighbors leaked Source text")

    candidates = nodes + expanded_nodes
    evidence_node = next(
        (node for node in candidates if int(node.get("evidence_count") or 0) > 0),
        nodes[0],
    )
    evidence_id = urllib.parse.quote(str(evidence_node.get("id") or ""), safe="")
    status, evidence = _request_json(
        "GET",
        f"{scope_url}/memory-graph/nodes/{evidence_id}/evidence?limit=5",
        api_key=issued.api_key,
        timeout=60,
    )
    _expect_status(status, 200, "memory graph evidence", evidence)
    evidence_items = list(evidence.get("items") or [])
    for item in evidence_items:
        text = str(item.get("text") or "")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if item.get("source_text_verbatim") is not True or digest != item.get(
            "text_sha256"
        ):
            raise SmokeError("memory graph evidence digest mismatch")

    status, trace = _request_json(
        "POST",
        f"{scope_url}/memory-graph/trace",
        api_key=issued.api_key,
        payload={
            "query": "What deployment channel and time are relevant?",
            "max_windows": 8,
            "debug": False,
        },
        timeout=300,
    )
    _expect_status(status, 200, "memory graph trace", trace)
    if trace.get("view") != "recall_trace":
        raise SmokeError("memory graph trace returned an unexpected view")

    unauthenticated_status, _ = _request_json(
        "GET",
        f"{scope_url}/memory-graph?limit=1",
        timeout=30,
    )
    if unauthenticated_status != 401:
        raise SmokeError(
            "unauthenticated memory graph request returned "
            f"HTTP {unauthenticated_status}, expected 401"
        )

    status, usage = _request_json(
        "GET",
        f"{base_url}/v1/usage/costs?scope_name={encoded_scope}",
        api_key=issued.api_key,
        timeout=30,
    )
    _expect_status(status, 200, "cost ledger", usage)
    return {
        "schema_version": "tmcra.memory-graph-smoke.1",
        "status": "passed",
        "service_version": "0.3.0-rc2",
        "ingest_status": ingest_status,
        "overview_nodes": len(nodes),
        "overview_resolved_layers": overview.get("resolved_layers"),
        "neighbor_nodes": len(expanded_nodes),
        "evidence_items_checked": len(evidence_items),
        "trace_selected_nodes": len(trace.get("selected_memory_ids") or []),
        "unauthenticated_status": unauthenticated_status,
        "known_model_api_cost_cny": usage.get("known_cost_cny"),
        "public_https": True,
        "retained_audit_scope": scope_name,
    }


def main() -> int:
    args = _parser().parse_args()
    started = time.monotonic()
    cleanup: dict[str, Any] = {}
    try:
        report = _run(
            base_url=args.base_url,
            control_db=args.control_db.resolve(),
            timeout=args.timeout,
            cleanup=cleanup,
            reuse_tenant=args.reuse_tenant.strip(),
            reuse_scope=args.reuse_scope.strip(),
        )
    except Exception as exc:
        report = {
            "schema_version": "tmcra.memory-graph-smoke.1",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }
    finally:
        auth = cleanup.get("auth")
        issued = cleanup.get("issued")
        if auth is not None and issued is not None:
            report["temporary_key_revoked"] = auth.revoke_key(issued.key_id)
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        _write_report(args.report.resolve(), report)

    printable = {key: value for key, value in report.items() if key != "retained_audit_scope"}
    print(json.dumps(printable, ensure_ascii=True, sort_keys=True))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
