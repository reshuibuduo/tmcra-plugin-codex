"""Opt-in live synthetic acceptance test. Never reads user/provider credentials to stdout."""
import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--recall-only", action="store_true")
parser.add_argument("--organize", action="store_true")
parser.add_argument("--compiled", action="store_true")
args = parser.parse_args()
receipt = json.loads((args.root / "installation.json").read_text(encoding="utf-8"))
state = args.root / "state" / receipt["profile"]
credentials = json.loads((state / "secrets/client.json").read_text(encoding="utf-8"))
base = f"http://127.0.0.1:{receipt['api_port']}"
headers = {"Authorization": "Bearer " + credentials["api_key"], "Content-Type": "application/json"}
scope = "synthetic-local-acceptance"
report_path = state / "synthetic-acceptance.json"
results = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}


def request(path, payload=None, key=None):
    request_headers = dict(headers)
    if key:
        request_headers["Idempotency-Key"] = key
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers=request_headers, method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"API {exc.code}: {detail[:2000]}") from None


def wait_job(job, timeout=1500):
    job_id = job["job_id"]
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        status = request("/v1/jobs/" + job_id)
        phase = status.get("status") or status.get("state")
        if phase != last:
            print(json.dumps({"job_id": job_id, "status": phase}), flush=True)
            last = phase
        if phase == "succeeded":
            return status
        if phase in {"failed", "cancelled"}:
            raise RuntimeError(json.dumps(status, ensure_ascii=False)[:4000])
        time.sleep(2)
    raise TimeoutError("synthetic memory job still incomplete; inspect the retained job")


results["health"] = request("/healthz")
results["ready"] = request("/readyz")
if not args.recall_only:
    started = time.monotonic()
    job = request(f"/v1/scopes/{scope}/ingest", {
        "session_id": "local-synthetic-20260906", "consistency": "read_your_writes", "slow_policy": "deferred",
        "messages": [{"message_id": "synthetic-01", "role": "user", "timestamp": "2026-09-06T01:00:00Z",
                      "content": "这是一条合成测试记忆。我把测试项目命名为蓝鲸，演示安排在周五下午三点。我喜欢用 Markdown 保存项目说明。"}],
    }, "local-synthetic-acceptance-v1")
    results["ingest"] = wait_job(job)
    results["ingest_seconds"] = round(time.monotonic() - started, 3)
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
started = time.monotonic()
results["recall"] = request(f"/v1/scopes/{scope}/recall", {
    "query": "蓝鲸测试项目的演示安排在什么时候？", "evidence_mode": "raw", "recall_profile": "interactive"})
results["recall_seconds"] = round(time.monotonic() - started, 3)
report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
rendered = json.dumps(results["recall"], ensure_ascii=False)
if "周五" not in rendered or "三点" not in rendered:
    raise AssertionError("recall omitted the synthetic source fact")
if args.compiled:
    started = time.monotonic()
    results["compiled"] = request(f"/v1/scopes/{scope}/recall", {
        "query": "蓝鲸测试项目的演示安排在什么时候？", "evidence_mode": "compiled", "recall_profile": "quality"})
    results["compiled_seconds"] = round(time.monotonic() - started, 3)
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"test": "compiled_evidence", "seconds": results["compiled_seconds"]}), flush=True)
if args.organize:
    started = time.monotonic()
    job = request(f"/v1/scopes/{scope}/consolidate", {}, "local-synthetic-organize-v1")
    results["consolidation"] = wait_job(job)
    results["consolidation_seconds"] = round(time.monotonic() - started, 3)
    results["knowledge"] = request(f"/v1/scopes/{scope}/knowledge-base")
    results["graph"] = request(f"/v1/scopes/{scope}/memory-graph/visual-atlas")
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"test": "consolidation", "seconds": results["consolidation_seconds"]}), flush=True)
print(json.dumps({"test": "synthetic_ingest_recall", "passed": True,
                  "ingest_seconds": results.get("ingest_seconds"), "recall_seconds": results["recall_seconds"],
                  "report": str(report_path)}, ensure_ascii=False), flush=True)
