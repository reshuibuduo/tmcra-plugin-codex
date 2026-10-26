#!/usr/bin/env python3
"""Explicitly recover reviewed Slow jobs that failed local model validation."""

from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ops.repair_tmcra_v4_slow_coverage import _attempt_summary, _attempts, _coverage
from run_tmcra_v4_build import (
    DEFAULT_REPO,
    DEFAULT_WRITER_ENV,
    BuildError,
    _key_pool,
    _load_resume_manifest,
    _load_shell_environment,
    _worker_environment,
)
from tmcra_v4_slow_graph import (
    SLOW_PROMPT_MIGRATION_SOURCE_VERSION,
    SLOW_PROMPT_MIGRATION_SOURCE_VERSIONS,
    SLOW_PROMPT_VERSION,
    load_graph_schema,
)


SCHEMA_VERSION = "tmcra.v4.failed-slow-model-validation-recovery.3"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _job_specs(values: Sequence[str]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise BuildError("--job must be WORKER=JOB_ID")
        worker, job_id = (item.strip() for item in value.split("=", 1))
        if (
            not worker.startswith("worker_")
            or not worker.removeprefix("worker_").isdigit()
            or not job_id.startswith("sgj_")
        ):
            raise BuildError(f"invalid recovery job specification: {value}")
        result.append((worker, job_id))
    if not result or len(result) != len(set(result)):
        raise BuildError("recovery jobs must be non-empty and unique")
    workers = [worker for worker, _ in result]
    if len(workers) != len(set(workers)):
        raise BuildError("only one recovery job per worker is allowed")
    return result


def _recovery_mode(database: Path, job_id: str) -> tuple[str, str]:
    with closing(
        sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    ) as connection:
        connection.row_factory = sqlite3.Row
        job = connection.execute(
            "SELECT status,attempts,claim_token,last_error FROM slow_graph_jobs "
            "WHERE job_id=?",
            (job_id,),
        ).fetchone()
        attempts = connection.execute(
            "SELECT status,error,call_metadata_json FROM slow_graph_attempts "
            "WHERE job_id=? ORDER BY created_at,attempt_id",
            (job_id,),
        ).fetchall()
        patch_count = int(
            connection.execute(
                "SELECT count(*) FROM slow_graph_patches WHERE job_id=?", (job_id,)
            ).fetchone()[0]
        )
    if (
        job is None
        or job["status"] != "failed"
        or int(job["attempts"] or 0) != len(attempts)
        or job["claim_token"] is not None
        or not attempts
        or patch_count != 0
    ):
        raise BuildError(
            f"{job_id} is not an explicit unclaimed failure without an applied patch"
        )
    parsed_attempts: list[tuple[sqlite3.Row, Mapping[str, Any], str]] = []
    for attempt in attempts:
        try:
            metadata = json.loads(str(attempt["call_metadata_json"]))
        except json.JSONDecodeError as exc:
            raise BuildError(f"{job_id} failed metadata is not JSON") from exc
        if not isinstance(metadata, Mapping):
            raise BuildError(f"{job_id} failed metadata is not an object")
        parsed_attempts.append((attempt, metadata, str(attempt["error"] or "").strip()))
    attempt, metadata, error = parsed_attempts[-1]
    if error != str(job["last_error"] or "").strip():
        raise BuildError(f"{job_id} job and attempt errors differ")
    if (
        len(parsed_attempts) == 1
        and
        metadata.get("physical_api_call") is False
        and int(metadata.get("physical_api_calls", -1)) == 0
        and str(metadata.get("route") or "") == "deterministic_noop"
        and error.startswith(
            "noop cannot consume uncited current durable Fast evidence: "
        )
    ):
        return "zero_call_promotion", "resume-zero-call-promotion-failure"
    if (
        len(parsed_attempts) == 1
        and
        metadata.get("physical_api_call") is True
        and int(metadata.get("physical_api_calls", 0) or 0) >= 1
        and str(metadata.get("route") or "") in {"pro", "flash_to_pro"}
        and str(metadata.get("status") or "")
        in {"completed", "response_received", "semantic_correction_rejected"}
        and int(metadata.get("http_status", 0) or 0) == 200
        and str(metadata.get("finish_reason") or "") == "stop"
    ):
        return "model_validation", "resume-failed-model-validation"
    if len(parsed_attempts) == 2 and all(
        row[0]["status"] == "failed"
        and row[2].startswith(
            "atomic Fast evidence may belong to only one resulting claim: "
        )
        and row[1].get("physical_api_call") is True
        and int(row[1].get("physical_api_calls", 0) or 0) >= 1
        and str(row[1].get("route") or "") in {"pro", "flash_to_pro"}
        and str(row[1].get("status") or "") == "semantic_correction_rejected"
        and int(row[1].get("http_status", 0) or 0) == 200
        and str(row[1].get("finish_reason") or "") == "stop"
        and str(row[1].get("prompt_version") or "")
        in SLOW_PROMPT_MIGRATION_SOURCE_VERSIONS
        for row in parsed_attempts
    ) and str(parsed_attempts[-1][1].get("prompt_version") or "") == (
        SLOW_PROMPT_MIGRATION_SOURCE_VERSION
    ):
        return "prompt_contract_migration", "resume-failed-prompt-migration"
    raise BuildError(f"{job_id} is not an approved reviewed recovery class")


def _run_one(
    *,
    worker: Mapping[str, Any],
    job_id: str,
    repo: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    worker_dir = Path(str(worker["worker_dir"])).resolve()
    database = worker_dir / "native_memory.sqlite3"
    recovery_mode, recovery_command = _recovery_mode(database, job_id)
    before_attempts = _attempts(database)
    before_coverage = _coverage(database)
    log = worker_dir / (
        f"slow_reviewed_failure_recovery.{recovery_mode}.{job_id}.log"
    )
    started = time.monotonic()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tmcra_v4_slow_graph.py"),
        str(database),
        "--repo",
        str(repo),
        recovery_command,
        job_id,
    ]
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=dict(environment),
            check=False,
        )
    after_attempts = _attempts(database)
    new_attempts = [
        attempt
        for attempt_id, attempt in after_attempts.items()
        if attempt_id not in before_attempts
    ]
    return {
        "worker": Path(str(worker["worker_dir"])).name,
        "worker_index": int(worker["worker_index"]),
        "question_id": str(worker["question_id"]),
        "scope_id": str(worker["scope_id"]),
        "job_id": job_id,
        "recovery_mode": recovery_mode,
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "log": str(log),
        "coverage_before": before_coverage,
        "coverage_after": _coverage(database),
        "new_attempts": _attempt_summary(new_attempts),
    }


def recover(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    if not run_dir.is_dir():
        raise BuildError(f"run directory does not exist: {run_dir}")
    if output.exists():
        raise BuildError(f"recovery report already exists: {output}")
    if (run_dir / "SLOW_REPAIR_LOCK").exists():
        raise BuildError("Slow repair lock is active")
    specs = _job_specs(args.job)
    if args.concurrency <= 0 or args.concurrency > len(specs):
        raise BuildError("recovery concurrency is out of range")
    repo = args.repo.resolve()
    load_graph_schema(repo)
    manifest = _load_resume_manifest(run_dir)
    by_name = {
        Path(str(worker["worker_dir"])).name: worker
        for worker in manifest["workers"]
    }
    missing = [worker for worker, _ in specs if worker not in by_name]
    if missing:
        raise BuildError("recovery workers are absent from manifest: " + ",".join(missing))

    shell_environment = _load_shell_environment(args.writer_env.resolve())
    base_environment = {**os.environ, **shell_environment}
    keys = _key_pool(base_environment)
    selected = [(by_name[worker], job_id) for worker, job_id in specs]
    environments = {
        worker: _worker_environment(
            base_environment, keys, int(by_name[worker]["worker_index"])
        )
        for worker, _ in specs
    }
    for environment in environments.values():
        if int(environment.get("TMCRA_WRITER_MAX_TOKENS", "0")) != 16384:
            raise BuildError("Writer/Slow max token preflight is not 16384")

    results: list[dict[str, Any]] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _run_one,
                worker=worker,
                job_id=job_id,
                repo=repo,
                environment=environments[Path(str(worker["worker_dir"])).name],
            ): Path(str(worker["worker_dir"])).name
            for worker, job_id in selected
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: int(item["worker_index"]))
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed" if all(item["status"] == "passed" for item in results) else "failed"
        ),
        "created_at": _now(),
        "run_dir": str(run_dir),
        "prompt_version": SLOW_PROMPT_VERSION,
        "concurrency": args.concurrency,
        "duration_seconds": round(time.monotonic() - started, 3),
        "physical_api_calls": sum(
            int(item["new_attempts"]["physical_api_calls"]) for item in results
        ),
        "estimated_cost_cny": round(
            sum(float(item["new_attempts"]["estimated_cost_cny"]) for item in results),
            8,
        ),
        "workers": results,
    }
    _write_json_atomic(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--writer-env", type=Path, default=DEFAULT_WRITER_ENV)
    args = parser.parse_args()
    report = recover(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
