"""Privacy-bounded structured HTTP access journal.

The journal intentionally records request metadata only. Request bodies,
query strings, authorization headers, cookies, and raw user identifiers never
cross this boundary.
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
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Mapping


ACCESS_LOG_SCHEMA = "tmcra.api-access.1"
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class _StrictTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Surface I/O failures to the journal without failing the request."""

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        # ``logging.StreamHandler.emit`` normally swallows write failures and
        # optionally prints them to stderr.  The journal needs an accurate
        # failure counter, so re-raise inside the handler's active exception
        # context; ``ApiAccessJournal.record`` catches it at the request-safe
        # boundary.
        raise


def _gzip_namer(filename: str) -> str:
    return filename + ".gz"


def _gzip_rotator(source: str, destination: str) -> None:
    with open(source, "rb") as input_stream, gzip.open(
        destination, "wb"
    ) as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
    os.remove(source)


def normalize_request_id(value: str | None, *, generated: str) -> str:
    """Accept a bounded caller correlation ID or use the server value."""

    candidate = str(value or "").strip()
    if candidate and REQUEST_ID_RE.fullmatch(candidate):
        return candidate
    return generated


def private_identifier_hash(value: object) -> str | None:
    """Return a stable short fingerprint without persisting the identifier."""

    text = str(value or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def bounded_content_length(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0 or parsed > 2**63 - 1:
        return None
    return parsed


class ApiAccessJournal:
    """Append JSONL access events while keeping logging failure non-fatal."""

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
            raise ValueError("enabled API access journal requires a path")
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
            f"tmcra.api_access.{hashlib.sha256(str(self.path).encode()).hexdigest()[:12]}",
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
            "schema": ACCESS_LOG_SCHEMA,
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


def request_access_event(
    *,
    request_id: str,
    method: str,
    route: str,
    status_code: int,
    latency_ms: float,
    request_bytes: int | None,
    response_bytes: int | None,
    auth_context: object | None,
    auth_kind: str | None,
    scope_name: str | None,
    job_ids: list[str],
    client_platform: str | None,
    integration_id: str | None,
    agent_id: str | None,
    error_code: str | None,
    exception_type: str | None,
    unmatched_path: str | None,
) -> dict[str, Any]:
    """Build the fixed, body-free access-event contract."""

    subject = getattr(auth_context, "subject", None)
    tenant_id = getattr(auth_context, "tenant_id", None)
    credential_id = getattr(auth_context, "credential_id", None)
    credential_type = getattr(auth_context, "credential_type", None)
    event: dict[str, Any] = {
        "request_id": request_id,
        "method": str(method).upper(),
        "route": route,
        "status_code": int(status_code),
        "latency_ms": round(max(0.0, float(latency_ms)), 3),
        "request_bytes": request_bytes,
        "response_bytes": response_bytes,
        "auth_kind": auth_kind or ("authenticated" if auth_context else "anonymous"),
        "tenant_id": str(tenant_id) if tenant_id else None,
        "credential_id": str(credential_id) if credential_id else None,
        "credential_type": str(credential_type) if credential_type else None,
        "subject_hash": private_identifier_hash(subject),
        "scope_name": scope_name,
        "job_ids": job_ids[:100],
        "job_count": len(job_ids),
        "client_platform": client_platform,
        "integration_id_hash": private_identifier_hash(integration_id),
        "agent_id_hash": private_identifier_hash(agent_id),
        "error_code": error_code,
        "exception_type": exception_type,
    }
    if unmatched_path is not None:
        event["unmatched_path_length"] = len(unmatched_path)
        event["unmatched_path_hash"] = hashlib.sha256(
            unmatched_path.encode("utf-8", errors="replace")
        ).hexdigest()
    return event
