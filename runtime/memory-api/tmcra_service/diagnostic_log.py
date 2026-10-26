"""Private structured diagnostics for API and asynchronous worker failures.

This journal is deliberately separate from the public-facing access journal.
It may contain sanitized exception messages and server-side stack locations,
but never captures request bodies, headers, query strings, cookies, local
variables, model prompts, or memory payloads.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Mapping


DIAGNOSTIC_LOG_SCHEMA = "tmcra.diagnostic.1"
MAX_MESSAGE_CHARS = 2_000
MAX_CHAIN_DEPTH = 8
MAX_TRACEBACK_FRAMES = 60

_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_NAMED_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|"
    r"cookie|password|passwd|secret)\b(\s*[:=]\s*)([^\s,;]+)"
)
_URL_QUERY_RE = re.compile(r"(https?://[^\s?#]+)\?[^\s#]*")
_LONG_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9_-]{48,}(?![A-Za-z0-9])")


class _StrictTimedRotatingFileHandler(TimedRotatingFileHandler):
    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        raise


def _gzip_namer(filename: str) -> str:
    return filename + ".gz"


def _gzip_rotator(source: str, destination: str) -> None:
    with open(source, "rb") as input_stream, gzip.open(
        destination, "wb"
    ) as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
    os.remove(source)


def redact_diagnostic_text(value: object, *, limit: int = MAX_MESSAGE_CHARS) -> str:
    """Bound and redact an exception string without logging its arguments."""

    text = str(value or "").replace("\x00", "�")
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _NAMED_SECRET_RE.sub(lambda match: match.group(1) + match.group(2) + "[REDACTED]", text)
    text = _URL_QUERY_RE.sub(r"\1?[REDACTED]", text)
    text = _LONG_TOKEN_RE.sub("[REDACTED_TOKEN]", text)
    if len(text) > limit:
        return text[: max(0, limit - 15)] + "...[truncated]"
    return text


def _exception_chain(exc: BaseException) -> list[tuple[str, BaseException]]:
    result: list[tuple[str, BaseException]] = []
    current: BaseException | None = exc
    relation = "raised"
    seen: set[int] = set()
    while current is not None and len(result) < MAX_CHAIN_DEPTH:
        identity = id(current)
        if identity in seen:
            break
        seen.add(identity)
        result.append((relation, current))
        if current.__cause__ is not None:
            current = current.__cause__
            relation = "caused_by"
        elif current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
            relation = "during_handling"
        else:
            current = None
    return result


def exception_details(exc: BaseException) -> dict[str, Any]:
    """Serialize an exception chain without source lines or local variables."""

    chain: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    fingerprint_parts: list[str] = []
    for relation, current in _exception_chain(exc):
        exception_type = type(current).__name__
        exception_module = type(current).__module__
        message = redact_diagnostic_text(current)
        chain.append(
            {
                "relation": relation,
                "type": exception_type,
                "module": exception_module,
                "message": message,
            }
        )
        fingerprint_parts.extend((relation, exception_module, exception_type))
        extracted = traceback.extract_tb(current.__traceback__)
        remaining = MAX_TRACEBACK_FRAMES - len(frames)
        for frame in extracted[-max(0, remaining) :]:
            value = {
                "file": str(frame.filename),
                "line": int(frame.lineno),
                "function": str(frame.name),
            }
            frames.append(value)
            fingerprint_parts.extend(
                (Path(frame.filename).name, str(frame.lineno), str(frame.name))
            )
        if len(frames) >= MAX_TRACEBACK_FRAMES:
            break
    root = chain[0] if chain else {
        "type": type(exc).__name__,
        "module": type(exc).__module__,
        "message": redact_diagnostic_text(exc),
    }
    return {
        "exception_type": root["type"],
        "exception_module": root["module"],
        "exception_message": root["message"],
        "exception_chain": chain,
        "traceback_frames": frames,
        "error_fingerprint": hashlib.sha256(
            "\x1f".join(fingerprint_parts).encode("utf-8", errors="replace")
        ).hexdigest()[:24],
    }


def diagnostic_exception_event(
    exc: BaseException,
    *,
    component: str,
    operation: str,
    severity: str = "error",
    request_id: str | None = None,
    job_id: str | None = None,
    job_type: str | None = None,
    stage_id: str | None = None,
    stage_name: str | None = None,
    stage_attempt: int | None = None,
    tenant_id: str | None = None,
    scope_name: str | None = None,
    worker_id: str | None = None,
    status_code: int | None = None,
    error_code: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fixed, payload-free diagnostic event."""

    safe_context: dict[str, Any] = {}
    for key, value in dict(context or {}).items():
        if value is None or isinstance(value, (bool, int, float)):
            safe_context[str(key)[:100]] = value
        elif isinstance(value, str):
            safe_context[str(key)[:100]] = redact_diagnostic_text(value, limit=500)
    return {
        "event_id": uuid.uuid4().hex,
        "severity": str(severity).lower(),
        "event": "exception",
        "component": str(component)[:100],
        "operation": str(operation)[:200],
        "request_id": request_id,
        "job_id": job_id,
        "job_type": job_type,
        "stage_id": stage_id,
        "stage_name": stage_name,
        "stage_attempt": stage_attempt,
        "tenant_id": tenant_id,
        "scope_name": scope_name,
        "worker_id": worker_id,
        "status_code": status_code,
        "error_code": error_code,
        "process_id": os.getpid(),
        "thread_name": threading.current_thread().name,
        "context": safe_context,
        **exception_details(exc),
    }


class DiagnosticJournal:
    """Append private JSONL diagnostic events without failing production work."""

    def __init__(self, path: Path | None, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.path = path.resolve() if path is not None else None
        self._lock = threading.Lock()
        self._written_events = 0
        self._write_failures = 0
        self._last_event_at: float | None = None
        self._last_failure_at: float | None = None
        self._handler: TimedRotatingFileHandler | None = None
        self._logger: logging.Logger | None = None
        if not self.enabled:
            return
        if self.path is None:
            raise ValueError("enabled diagnostic journal requires a path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        handler = _StrictTimedRotatingFileHandler(
            self.path,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
            delay=False,
            utc=True,
        )
        handler.namer = _gzip_namer
        handler.rotator = _gzip_rotator
        handler.setFormatter(logging.Formatter("%(message)s"))
        try:
            self.path.chmod(0o600)
        except OSError:
            pass
        logger = logging.Logger(
            "tmcra.diagnostic."
            + hashlib.sha256(str(self.path).encode()).hexdigest()[:12],
            level=logging.INFO,
        )
        logger.propagate = False
        logger.addHandler(handler)
        self._handler = handler
        self._logger = logger

    def record(self, event: Mapping[str, Any]) -> None:
        if not self.enabled or self._logger is None:
            return
        now = time.time()
        payload = {
            "schema": DIAGNOSTIC_LOG_SCHEMA,
            "recorded_at": now,
            **dict(event),
        }
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            self._logger.info(encoded)
        except Exception:
            with self._lock:
                self._write_failures += 1
                self._last_failure_at = now
            return
        with self._lock:
            self._written_events += 1
            self._last_event_at = now

    def record_exception(self, exc: BaseException, **fields: Any) -> None:
        self.record(diagnostic_exception_event(exc, **fields))

    def status(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {
                "enabled": self.enabled,
                "written_events": self._written_events,
                "write_failures": self._write_failures,
                "last_event_at": self._last_event_at,
                "last_failure_at": self._last_failure_at,
                "rotation": "utc_midnight",
                "retained_files": 30,
                "compressed_rotations": True,
                "captures_request_bodies": False,
                "captures_headers": False,
                "captures_query_strings": False,
                "captures_local_variables": False,
            }
        if self.path is not None:
            result["filename"] = self.path.name
            try:
                result["size_bytes"] = self.path.stat().st_size
            except OSError:
                result["size_bytes"] = None
        return result

    def close(self) -> None:
        handler = self._handler
        logger = self._logger
        self._handler = None
        self._logger = None
        if handler is None:
            return
        if logger is not None:
            logger.removeHandler(handler)
        try:
            handler.flush()
        finally:
            handler.close()
