from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from .writer import WRITER_RECOVERY_MODES, execute_writer
from .usage_attribution import UsageAttribution


PROTOCOL_VERSION = "tmcra.writer-daemon.5"
REQUEST_STATUS_SCHEMA_VERSION = "tmcra.writer-request-status.1"
RECOVERY_MODES = WRITER_RECOVERY_MODES


def _emit(value: Mapping[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(dict(value), ensure_ascii=True) + "\n")
    sys.__stdout__.flush()


def _request_path(value: Mapping[str, Any], name: str) -> Path:
    raw = value.get(name)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"writer daemon request lacks {name}")
    return Path(raw).resolve()


def _request_text(value: Mapping[str, Any], name: str) -> str:
    raw = value.get(name)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"writer daemon request lacks {name}")
    return raw.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_identity(value: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Return the deterministic identity for one immutable Writer attempt."""

    recovery_mode = str(value.get("recovery_mode") or "none").strip()
    if recovery_mode not in RECOVERY_MODES:
        raise ValueError("writer daemon recovery_mode is invalid")
    input_path = _request_path(value, "input_path")
    stage_attempt = value.get("stage_attempt")
    if (
        isinstance(stage_attempt, bool)
        or not isinstance(stage_attempt, int)
        or stage_attempt <= 0
    ):
        raise ValueError("writer daemon stage_attempt must be positive")
    provider_execution_value = value.get("provider_execution")
    if provider_execution_value is not None and not isinstance(
        provider_execution_value, Mapping
    ):
        raise ValueError("writer daemon provider_execution must be an object")
    provider_execution = (
        None
        if provider_execution_value is None
        else {
            str(key): str(item)
            for key, item in provider_execution_value.items()
        }
    )
    contract = {
        "protocol": PROTOCOL_VERSION,
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "out_dir": str(_request_path(value, "out_dir")),
        "database": str(_request_path(value, "database")),
        "operation_id": _request_text(value, "operation_id"),
        "tenant_id": _request_text(value, "tenant_id"),
        "scope_name": _request_text(value, "scope_name"),
        "job_id": _request_text(value, "job_id"),
        "stage_id": _request_text(value, "stage_id"),
        "stage_attempt": stage_attempt,
        "recovery_mode": recovery_mode,
        "timeout_seconds": float(value.get("timeout_seconds", 180.0)),
        "max_tokens": int(value.get("max_tokens", 16384)),
        "usage_attribution": UsageAttribution.from_mapping(
            value.get("usage_attribution")
            if isinstance(value.get("usage_attribution"), Mapping)
            else None
        ).as_dict(),
        "provider_execution": provider_execution,
    }
    encoded = json.dumps(
        contract, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    request_sha256 = hashlib.sha256(encoded).hexdigest()
    return f"wrq_{request_sha256}", request_sha256, contract


def request_status_path(root: Path, request_id: str) -> Path:
    if not request_id.startswith("wrq_") or len(request_id) != 68:
        raise ValueError("writer request ID is invalid")
    return root.resolve() / request_id[:6] / f"{request_id}.json"


def read_request_status(
    path: Path, *, request_id: str, request_sha256: str
) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("writer request status is unreadable") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != REQUEST_STATUS_SCHEMA_VERSION
        or value.get("protocol") != PROTOCOL_VERSION
        or value.get("request_id") != request_id
        or value.get("request_sha256") != request_sha256
    ):
        raise ValueError("writer request status identity mismatch")
    return value


def write_request_status(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(dict(value), ensure_ascii=True, indent=2, sort_keys=True)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _failure_state(exc: BaseException) -> str:
    metadata = getattr(exc, "metadata", None)
    values = dict(metadata) if isinstance(metadata, Mapping) else {}
    status = str(values.get("status") or "").strip().lower()
    if status in {"request_error", "transport_error", "timeout"}:
        return "outcome_unknown"
    if isinstance(exc, (TimeoutError, OSError)) or "timeout" in str(exc).lower():
        return "outcome_unknown"
    return "failed"


def _preload(repo: Path) -> Any:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    with contextlib.redirect_stdout(sys.stderr):
        v4 = importlib.import_module("tmcra_v4_batch_writer")
        # RealGraphBackend imports this transitively on every cold CLI launch.
        # Preloading it once keeps the exact graph implementation while avoiding
        # repeated torch startup for every ingest operation.
        importlib.import_module("experiments.replacement.adapters.memory_adapters")
    return v4


def serve(repo: Path) -> int:
    started = time.monotonic()
    try:
        status_root_raw = str(os.getenv("TMCRA_WRITER_REQUEST_STATE_DIR") or "").strip()
        if not status_root_raw:
            raise ValueError("TMCRA_WRITER_REQUEST_STATE_DIR is required")
        status_root = Path(status_root_raw).resolve()
        status_root.mkdir(parents=True, exist_ok=True)
        v4 = _preload(repo)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit(
            {
                "type": "hello",
                "protocol": PROTOCOL_VERSION,
                "ok": False,
                "pid": os.getpid(),
                "error_type": type(exc).__name__,
            }
        )
        return 2
    _emit(
        {
            "type": "hello",
            "protocol": PROTOCOL_VERSION,
            "ok": True,
            "pid": os.getpid(),
            "preload_seconds": round(time.monotonic() - started, 6),
            "request_status_schema": REQUEST_STATUS_SCHEMA_VERSION,
            "writer_schema_version": str(v4.BATCH_SCHEMA_VERSION),
            "prompt_version": str(v4.PROMPT_VERSION),
            "candidate_selector_version": str(v4.CANDIDATE_SELECTOR_VERSION),
        }
    )
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        request_id = ""
        request_sha256 = ""
        status_path: Path | None = None
        status_value: dict[str, Any] | None = None
        try:
            request = json.loads(raw_line)
            if not isinstance(request, dict):
                raise ValueError("writer daemon request must be an object")
            request_id = _request_text(request, "request_id")
            if request.get("type") == "shutdown":
                _emit({"type": "shutdown", "request_id": request_id, "ok": True})
                return 0
            if request.get("type") != "execute":
                raise ValueError("writer daemon request type is invalid")
            expected_id, request_sha256, contract = request_identity(request)
            if request_id != expected_id:
                raise ValueError("writer daemon request identity differs from its content")
            if str(request.get("request_sha256") or "") != request_sha256:
                raise ValueError("writer daemon request hash differs from its content")
            status_path = request_status_path(status_root, request_id)
            status_value = read_request_status(
                status_path,
                request_id=request_id,
                request_sha256=request_sha256,
            )
            if status_value is not None:
                prior_state = str(status_value.get("state") or "")
                prior_response = status_value.get("response")
                if prior_state in {"succeeded", "failed", "outcome_unknown"}:
                    if not isinstance(prior_response, Mapping):
                        raise ValueError("terminal writer request lacks its response")
                    _emit(dict(prior_response))
                    continue
                raise ValueError(
                    "writer request is already running and cannot be replayed"
                )
            operation_started = time.monotonic()
            accepted_at = time.time()
            write_request_status(
                status_path,
                {
                    "schema_version": REQUEST_STATUS_SCHEMA_VERSION,
                    "protocol": PROTOCOL_VERSION,
                    "request_id": request_id,
                    "request_sha256": request_sha256,
                    "state": "running",
                    "contract": contract,
                    "worker_pid": os.getpid(),
                    "accepted_at": accepted_at,
                    "updated_at": accepted_at,
                },
            )
            with contextlib.redirect_stdout(sys.stderr):
                report = execute_writer(
                    input_path=_request_path(request, "input_path"),
                    out_dir=_request_path(request, "out_dir"),
                    database=_request_path(request, "database"),
                    operation_id=_request_text(request, "operation_id"),
                    repo=repo,
                    tenant_id=_request_text(request, "tenant_id"),
                    scope_name=_request_text(request, "scope_name"),
                    job_id=_request_text(request, "job_id"),
                    stage_id=_request_text(request, "stage_id"),
                    stage_attempt=int(request.get("stage_attempt", 0) or 0),
                    reviewer_model=str(
                        os.getenv("TMCRA_WRITER_REVIEWER_MODEL")
                        or os.getenv("TMCRA_DEEPSEEK_PRO_MODEL")
                        or "deepseek-v4-pro"
                    ).strip(),
                    timeout_seconds=float(request.get("timeout_seconds", 180.0)),
                    max_tokens=int(request.get("max_tokens", 16384)),
                    recovery_mode=str(request.get("recovery_mode") or "none"),
                    usage_attribution=UsageAttribution.from_mapping(
                        request.get("usage_attribution")
                        if isinstance(request.get("usage_attribution"), Mapping)
                        else None
                    ),
                    provider_execution=(
                        request.get("provider_execution")
                        if isinstance(request.get("provider_execution"), Mapping)
                        else None
                    ),
                    v4_module=v4,
                )
            response = {
                "type": "result",
                "request_id": request_id,
                "request_sha256": request_sha256,
                "request_state": "succeeded",
                "ok": True,
                "pid": os.getpid(),
                "elapsed_seconds": round(time.monotonic() - operation_started, 6),
                "report_schema_version": str(report.get("schema_version") or ""),
                "report_status": str(report.get("status") or ""),
            }
            completed_at = time.time()
            write_request_status(
                status_path,
                {
                    "schema_version": REQUEST_STATUS_SCHEMA_VERSION,
                    "protocol": PROTOCOL_VERSION,
                    "request_id": request_id,
                    "request_sha256": request_sha256,
                    "state": "succeeded",
                    "contract": contract,
                    "worker_pid": os.getpid(),
                    "accepted_at": accepted_at,
                    "updated_at": completed_at,
                    "completed_at": completed_at,
                    "response": response,
                },
            )
            _emit(response)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            state = _failure_state(exc)
            response = {
                "type": "result",
                "request_id": request_id,
                "request_sha256": request_sha256,
                "request_state": state,
                "ok": False,
                "pid": os.getpid(),
                "error_type": type(exc).__name__,
            }
            if status_path is not None and request_sha256:
                now = time.time()
                prior = status_value or {}
                write_request_status(
                    status_path,
                    {
                        "schema_version": REQUEST_STATUS_SCHEMA_VERSION,
                        "protocol": PROTOCOL_VERSION,
                        "request_id": request_id,
                        "request_sha256": request_sha256,
                        "state": state,
                        "contract": prior.get("contract", contract),
                        "worker_pid": os.getpid(),
                        "accepted_at": prior.get("accepted_at", now),
                        "updated_at": now,
                        "completed_at": now,
                        "error_sha256": hashlib.sha256(
                            f"{type(exc).__name__}:{exc}".encode("utf-8")
                        ).hexdigest(),
                        "response": response,
                    },
                )
            _emit(response)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Resident TMCRA Writer worker")
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    return serve(args.repo.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
