"""User-facing TMCRA CLI client and machine-readable receipt sidecar.

This module deliberately stays separate from the service administration CLI.
It talks only to the public HTTP contract and never prints the bearer token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen


CLIENT_COMMANDS = frozenset({"recall", "ingest", "job", "turn"})
TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
RECEIPT_SCHEMA = "tmcra.cli.receipt.v1"
CONTRACT_SCHEMA = "tmcra.receipts.v1"
DEFAULT_BASE_URL = "https://api.tmcra.com"
CLI_VERSION = "0.5.0"


class ClientCLIError(RuntimeError):
    """A user-facing error that can be represented without leaking secrets."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "cli_error",
        status_code: int | None = None,
        request_id: str | None = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.request_id = request_id
        self.details = details


@dataclass(frozen=True)
class ClientConfig:
    base_url: str
    api_key: str
    timeout_seconds: float = 30.0


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def deterministic_idempotency_key(
    operation: str,
    *,
    scope: str,
    payload: Mapping[str, Any],
    supplied: str | None = None,
) -> str:
    """Return a stable write key without putting message text in the key."""

    if supplied is not None:
        key = supplied.strip()
        if not 8 <= len(key) <= 200:
            raise ClientCLIError(
                "idempotency key must contain between 8 and 200 characters",
                code="invalid_idempotency_key",
            )
        return key
    digest = hashlib.sha256(
        _canonical_json({"operation": operation, "scope": scope, "payload": payload})
    ).hexdigest()
    return f"tmcra-cli-{operation}-{digest[:48]}"


def _redact(value: Any, *, secret: str | None = None) -> Any:
    """Remove credential-like fields before a payload reaches stdout."""

    sensitive = {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "client_secret",
        "password",
        "secret",
        "token",
        "signing_secret",
    }
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]"
            if str(key).lower() in sensitive
            else _redact(item, secret=secret)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, secret=secret) for item in value]
    if isinstance(value, tuple):
        return [_redact(item, secret=secret) for item in value]
    if secret and isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _receipt(
    operation: str,
    *,
    status: str,
    scope: str | None = None,
    data: Any = None,
    idempotency_key: str | None = None,
    request_id: str | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    terminal = status in TERMINAL_JOB_STATUSES or (operation == "recall" and status == "succeeded")
    submitted_status = "completed" if operation == "recall" and status == "succeeded" else "submitted"
    final_status: str | None = (
        "completed" if operation == "recall" and status == "succeeded" else
        status if status in TERMINAL_JOB_STATUSES else None
    )
    value: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "contract_schema_version": CONTRACT_SCHEMA,
        "operation": operation,
        "status": status,
        "submitted_status": submitted_status,
        "final_status": final_status,
        "submitted": True,
        "final": terminal,
        "watermarks": _extract_watermarks(data),
        "created_at": _now(),
    }
    if scope is not None:
        value["scope_name"] = scope
    if idempotency_key is not None:
        value["idempotency_key"] = idempotency_key
    if request_id:
        value["request_id"] = request_id
    if data is not None:
        value["data"] = data
    if error is not None:
        value["error"] = dict(error)
    return value


def _extract_watermarks(value: Any) -> dict[str, Any]:
    """Project service watermarks without making them a runtime dependency."""

    found: dict[str, Any] = {}

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key in (
                "source_event_seq",
                "promoted_event_seq",
                "indexed_event_seq",
                "source_raw_token_estimate",
            ):
                if key not in found and isinstance(item.get(key), int) and not isinstance(item.get(key), bool):
                    found[key] = item[key]
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return {
        key: found.get(key)
        for key in (
            "source_event_seq",
            "promoted_event_seq",
            "indexed_event_seq",
            "source_raw_token_estimate",
        )
    } | {"available": bool(found)}


def _validate_job(payload: Any, *, require_status_url: bool = True) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ClientCLIError("job response must be a JSON object", code="invalid_job_response")
    required = ("job_id", "status", "scope_name")
    missing = [name for name in required if not str(payload.get(name) or "").strip()]
    if require_status_url and not str(payload.get("status_url") or "").strip():
        missing.append("status_url")
    status = str(payload.get("status") or "")
    if status and status not in TERMINAL_JOB_STATUSES | {"pending", "running", "queued"}:
        raise ClientCLIError(
            f"job response has unsupported status: {status}",
            code="invalid_job_response",
        )
    if missing:
        raise ClientCLIError(
            f"job response is missing: {', '.join(missing)}",
            code="invalid_job_response",
        )
    return dict(payload)


def _validate_recall(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ClientCLIError("recall response must be a JSON object", code="invalid_recall_response")
    required = ("query_id", "scope_name", "evidence_route", "prompt_evidence")
    missing = [name for name in required if payload.get(name) in (None, "")]
    route = payload.get("evidence_route")
    prompt = payload.get("prompt_evidence")
    if not isinstance(route, Mapping) or not route.get("selected"):
        missing.append("evidence_route.selected")
    if not isinstance(prompt, Mapping) or not isinstance(prompt.get("content"), str):
        missing.append("prompt_evidence.content")
    if missing:
        raise ClientCLIError(
            f"recall response is incomplete: {', '.join(missing)}",
            code="invalid_recall_response",
        )
    return dict(payload)


class HTTPClient:
    """Small stdlib HTTP client with an injectable request function for tests."""

    def __init__(
        self,
        config: ClientConfig,
        *,
        requester: Callable[..., tuple[int, Mapping[str, str], Any]] | None = None,
    ) -> None:
        self.config = config
        self._requester = requester

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[int, Mapping[str, str], Any]:
        url = urljoin(self.config.base_url.rstrip("/") + "/", path.lstrip("/"))
        encoded = _canonical_json(body) if body is not None else None
        request = Request(
            url,
            data=encoded,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "User-Agent": f"tmcra-cli/{CLI_VERSION}",
                **({"Idempotency-Key": idempotency_key} if idempotency_key else {}),
                **({"Content-Type": "application/json"} if encoded is not None else {}),
            },
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read()
                return _check_response(
                    (response.status, dict(response.headers.items()), _decode_json(raw))
                )
        except HTTPError as exc:
            raw = exc.read()
            payload = _decode_json(raw)
            raise ClientCLIError(
                _error_message(payload, exc.code),
                code=_error_code(payload),
                status_code=exc.code,
                request_id=_request_id(payload, exc.headers),
                details=_error_details(payload),
            ) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise ClientCLIError(
                f"TMCRA transport error: {exc}",
                code="transport_error",
            ) from exc

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[int, Mapping[str, str], Any]:
        if self._requester is not None:
            response = self._requester(
                method,
                path,
                body=body,
                idempotency_key=idempotency_key,
            )
            return _check_response(response)
        return self._request(
            method,
            path,
            body=body,
            idempotency_key=idempotency_key,
        )


def _check_response(
    response: tuple[int, Mapping[str, str], Any],
) -> tuple[int, Mapping[str, str], Any]:
    status, headers, payload = response
    if status >= 400:
        raise ClientCLIError(
            _error_message(payload, status),
            code=_error_code(payload),
            status_code=status,
            request_id=_request_id(payload, headers),
            details=_error_details(payload),
        )
    if status < 200:
        raise ClientCLIError(
            f"TMCRA returned unexpected HTTP status {status}",
            code="unexpected_http_status",
            status_code=status,
        )
    return response


def _decode_json(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClientCLIError("TMCRA returned invalid JSON", code="invalid_json_response") from exc


def _error_object(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    value = payload.get("error", payload.get("detail", {}))
    return value if isinstance(value, Mapping) else {"message": str(value)}


def _error_message(payload: Any, status: int) -> str:
    value = _error_object(payload).get("message")
    return str(value or f"TMCRA returned HTTP {status}")


def _error_code(payload: Any) -> str:
    return str(_error_object(payload).get("code") or "http_error")


def _error_details(payload: Any) -> Any:
    return _error_object(payload).get("details")


def _request_id(payload: Any, headers: Mapping[str, str]) -> str | None:
    error = _error_object(payload)
    header_request_id = next(
        (value for key, value in headers.items() if str(key).lower() == "x-request-id"),
        "",
    )
    return str(error.get("request_id") or header_request_id or "") or None


def load_config(args: argparse.Namespace) -> ClientConfig:
    api_key_name = str(args.api_key_env or "TMCRA_API_KEY")
    api_key = os.getenv(api_key_name, "").strip()
    if not api_key:
        raise ClientCLIError(
            f"missing API credential in environment variable {api_key_name}",
            code="missing_api_key",
        )
    timeout = float(args.request_timeout)
    if timeout <= 0:
        raise ClientCLIError("request timeout must be positive", code="invalid_timeout")
    base_url = (args.base_url or os.getenv("TMCRA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
    if not base_url.startswith("https://") and not base_url.startswith("http://localhost"):
        raise ClientCLIError(
            "TMCRA base URL must use HTTPS (or localhost for development)",
            code="insecure_base_url",
        )
    return ClientConfig(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout,
    )


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=None, help="TMCRA base URL (or TMCRA_BASE_URL)")
    parser.add_argument("--api-key-env", default="TMCRA_API_KEY", help=argparse.SUPPRESS)
    parser.add_argument("--request-timeout", type=float, default=30.0, help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", help="emit one JSON receipt")


def _scope_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scope", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TMCRA user-side memory API CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    recall = sub.add_parser("recall", help="retrieve evidence for a query")
    _common_parser(recall)
    _scope_arg(recall)
    recall.add_argument("--query", required=True)
    recall.add_argument("--evidence-mode", choices=("raw", "auto", "compiled"), default="auto")
    recall.add_argument("--recall-policy", choices=("strict", "lenient"), default="strict")
    recall.add_argument("--wait-for-job-id")

    ingest = sub.add_parser("ingest", help="submit a user/assistant memory turn")
    _common_parser(ingest)
    _scope_arg(ingest)
    ingest.add_argument("--session-id", required=True)
    ingest.add_argument("--messages-file", type=Path)
    ingest.add_argument("--messages-json")
    ingest.add_argument("--consistency", choices=("eventual", "read_your_writes"), default="read_your_writes")
    ingest.add_argument("--slow-policy", choices=("auto", "deferred", "force"), default="auto")
    ingest.add_argument("--idempotency-key")
    ingest.add_argument("--wait", action="store_true")
    ingest.add_argument("--wait-timeout", type=float, default=120.0)
    ingest.add_argument("--poll-interval", type=float, default=1.5)

    job = sub.add_parser("job", help="inspect or wait for a job")
    _common_parser(job)
    job_sub = job.add_subparsers(dest="job_command", required=True)
    get = job_sub.add_parser("get")
    get.add_argument("job_id")
    wait = job_sub.add_parser("wait")
    wait.add_argument("job_id")
    wait.add_argument("--timeout", type=float, default=120.0)
    wait.add_argument("--poll-interval", type=float, default=1.5)

    turn = sub.add_parser("turn", help="recall, then submit a complete user/assistant turn")
    _common_parser(turn)
    _scope_arg(turn)
    turn.add_argument("--session-id", required=True)
    turn.add_argument("--user-message", required=True)
    turn.add_argument("--assistant-message", required=True)
    turn.add_argument("--query")
    turn.add_argument("--recall-policy", choices=("strict", "lenient"), default="strict")
    turn.add_argument("--evidence-mode", choices=("raw", "auto", "compiled"), default="auto")
    turn.add_argument("--consistency", choices=("eventual", "read_your_writes"), default="read_your_writes")
    turn.add_argument("--slow-policy", choices=("auto", "deferred", "force"), default="auto")
    turn.add_argument("--idempotency-key")
    turn.add_argument("--wait", action="store_true")
    turn.add_argument("--wait-timeout", type=float, default=120.0)
    turn.add_argument("--poll-interval", type=float, default=1.5)
    return parser


def _normalize_global_options(argv: Sequence[str]) -> list[str]:
    """Accept connection options before or after the client subcommand."""

    values = {"--base-url", "--api-key-env", "--request-timeout"}
    command_index = next(
        (index for index, token in enumerate(argv) if token in CLIENT_COMMANDS),
        None,
    )
    if command_index in (None, 0):
        return list(argv)
    prefix: list[str] = []
    remaining: list[str] = []
    index = 0
    while index < command_index:
        token = argv[index]
        if token in values:
            if index + 1 >= command_index:
                raise ClientCLIError(f"{token} requires a value", code="invalid_cli_options")
            prefix.extend((token, argv[index + 1]))
            index += 2
            continue
        if token == "--json":
            prefix.append(token)
            index += 1
            continue
        remaining.append(token)
        index += 1
    suffix = list(argv[command_index:])
    return suffix[:1] + prefix + suffix[1:] + remaining


def _load_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    if bool(args.messages_file) == bool(args.messages_json):
        raise ClientCLIError("provide exactly one of --messages-file or --messages-json", code="invalid_messages")
    try:
        raw = args.messages_file.read_text(encoding="utf-8") if args.messages_file else args.messages_json
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ClientCLIError("messages must be a UTF-8 JSON array", code="invalid_messages") from exc
    if not isinstance(value, list) or not value:
        raise ClientCLIError("messages must be a non-empty JSON array", code="invalid_messages")
    if any(not isinstance(item, Mapping) for item in value):
        raise ClientCLIError("every message must be a JSON object", code="invalid_messages")
    return [dict(item) for item in value]


def _message(role: str, content: str, *, session_id: str, index: int) -> dict[str, Any]:
    return {
        "message_id": f"{session_id}-{role}-{index}",
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _print_receipt(receipt: Mapping[str, Any], *, secret: str | None = None) -> None:
    print(json.dumps(_redact(receipt, secret=secret), ensure_ascii=False, sort_keys=True))


def _error_receipt(operation: str, scope: str | None, exc: ClientCLIError) -> dict[str, Any]:
    error = {
        "code": exc.code,
        "message": str(exc),
    }
    if exc.status_code is not None:
        error["status_code"] = exc.status_code
    if exc.request_id:
        error["request_id"] = exc.request_id
    if exc.details is not None:
        error["details"] = _redact(exc.details)
    return _receipt(operation, status="failed", scope=scope, error=error)


def _job_receipt(operation: str, scope: str | None, status: str, job: Mapping[str, Any], *, key: str | None = None) -> dict[str, Any]:
    return _receipt(operation, status=status, scope=scope, data={"job": _redact(job)}, idempotency_key=key)


def _submission_status(job: Mapping[str, Any]) -> str:
    status = str(job.get("status") or "")
    if status == "succeeded":
        return "succeeded"
    if status in {"failed", "cancelled"}:
        return status
    return "submitted"


def _wait_for_job(
    client: HTTPClient,
    job: Mapping[str, Any],
    *,
    scope: str | None,
    timeout: float,
    poll_interval: float,
    operation: str,
    key: str | None = None,
) -> dict[str, Any]:
    job_value = _validate_job(job)
    deadline = time.monotonic() + timeout
    while True:
        status = str(job_value["status"])
        if status in TERMINAL_JOB_STATUSES:
            return _job_receipt(
                operation,
                scope,
                status,
                job_value,
                key=key,
            )
        if time.monotonic() >= deadline:
            return _receipt(
                operation,
                status="timeout",
                scope=scope,
                data={"job": _redact(job_value)},
                idempotency_key=key,
                error={"code": "job_wait_timeout", "message": f"job did not finish within {timeout:g}s"},
            )
        time.sleep(max(0.0, poll_interval))
        _, _, payload = client.request("GET", f"/v1/jobs/{quote(str(job_value['job_id']), safe='')}")
        job_value = _validate_job(payload)


def _run_recall(args: argparse.Namespace, client: HTTPClient) -> tuple[dict[str, Any], int]:
    try:
        _, headers, payload = client.request(
            "POST",
            f"/v1/scopes/{quote(args.scope, safe='')}/recall",
            body={
                "query": args.query,
                "evidence_mode": args.evidence_mode,
                "recall_profile": "quality",
                "response_projection": "full",
                "max_windows": 8,
                **({"wait_for_job_id": args.wait_for_job_id} if args.wait_for_job_id else {}),
            },
        )
        recall = _validate_recall(payload)
        receipt = _receipt(
            "recall",
            status="succeeded",
            scope=args.scope,
            data={"recall": _redact(recall)},
            request_id=_request_id({}, headers),
        )
        return receipt, 0
    except ClientCLIError as exc:
        receipt = _error_receipt("recall", args.scope, exc)
        if args.recall_policy == "lenient":
            receipt["status"] = "degraded"
            receipt["final_status"] = None
            receipt["final"] = False
            return receipt, 0
        return receipt, 1


def _run_ingest(args: argparse.Namespace, client: HTTPClient) -> tuple[dict[str, Any], int]:
    try:
        messages = _load_messages(args)
        body = {
            "session_id": args.session_id,
            "messages": messages,
            "consistency": args.consistency,
            "slow_policy": args.slow_policy,
            "metadata": {},
        }
        key = deterministic_idempotency_key(
            "ingest", scope=args.scope, payload=body, supplied=args.idempotency_key
        )
        _, headers, payload = client.request(
            "POST",
            f"/v1/scopes/{quote(args.scope, safe='')}/ingest",
            body=body,
            idempotency_key=key,
        )
        job = _validate_job(payload)
        if args.wait:
            receipt = _wait_for_job(
                client,
                job,
                scope=args.scope,
                timeout=args.wait_timeout,
                poll_interval=args.poll_interval,
                operation="ingest.wait",
                key=key,
            )
        else:
            receipt = _job_receipt("ingest", args.scope, _submission_status(job), job, key=key)
            request_id = _request_id({}, headers)
            if request_id:
                receipt["request_id"] = request_id
        return receipt, 0 if receipt["status"] in {"submitted", "succeeded"} else 1
    except ClientCLIError as exc:
        return _error_receipt("ingest", args.scope, exc), 1


def _run_job(args: argparse.Namespace, client: HTTPClient) -> tuple[dict[str, Any], int]:
    operation = f"job.{args.job_command}"
    try:
        _, _, payload = client.request("GET", f"/v1/jobs/{quote(args.job_id, safe='')}")
        job = _validate_job(payload)
        if args.job_command == "get":
            return _job_receipt(operation, job.get("scope_name"), job["status"], job), 0
        receipt = _wait_for_job(
            client,
            job,
            scope=job.get("scope_name"),
            timeout=args.timeout,
            poll_interval=args.poll_interval,
            operation=operation,
        )
        return receipt, 0 if receipt["status"] == "succeeded" else 1
    except ClientCLIError as exc:
        return _error_receipt(operation, None, exc), 1


def _run_turn(args: argparse.Namespace, client: HTTPClient) -> tuple[dict[str, Any], int]:
    query = args.query or args.user_message
    recall_args = argparse.Namespace(
        scope=args.scope,
        query=query,
        evidence_mode=args.evidence_mode,
        wait_for_job_id=None,
        recall_policy=args.recall_policy,
    )
    recall_receipt, recall_code = _run_recall(recall_args, client)
    if recall_code and args.recall_policy == "strict":
        return _receipt(
            "turn",
            status="failed",
            scope=args.scope,
            data={"recall": recall_receipt},
            error={"code": "strict_recall_failed", "message": "turn was not written"},
        ), 1
    body = {
        "session_id": args.session_id,
        "messages": [
            _message("user", args.user_message, session_id=args.session_id, index=0),
            _message("assistant", args.assistant_message, session_id=args.session_id, index=1),
        ],
        "consistency": args.consistency,
        "slow_policy": args.slow_policy,
        "metadata": {},
    }
    key = deterministic_idempotency_key(
        "turn", scope=args.scope, payload=body, supplied=args.idempotency_key
    )
    try:
        _, _, payload = client.request(
            "POST",
            f"/v1/scopes/{quote(args.scope, safe='')}/ingest",
            body=body,
            idempotency_key=key,
        )
        job = _validate_job(payload)
        if args.wait:
            ingest_receipt = _wait_for_job(
                client,
                job,
                scope=args.scope,
                timeout=args.wait_timeout,
                poll_interval=args.poll_interval,
                operation="turn.ingest.wait",
                key=key,
            )
        else:
            ingest_receipt = _job_receipt(
                "turn.ingest", args.scope, _submission_status(job), job, key=key
            )
        status = ingest_receipt["status"]
        if recall_receipt["status"] == "degraded" and status in {"submitted", "succeeded"}:
            status = "degraded"
        return _receipt(
            "turn",
            status=status,
            scope=args.scope,
            idempotency_key=key,
            data={"recall": recall_receipt, "ingest": ingest_receipt},
        ), 0 if status in {"submitted", "succeeded", "degraded"} else 1
    except ClientCLIError as exc:
        return _receipt(
            "turn",
            status="failed",
            scope=args.scope,
            idempotency_key=key,
            data={"recall": recall_receipt},
            error={"code": exc.code, "message": str(exc)},
        ), 1


def run(argv: Sequence[str] | None = None, *, client: HTTPClient | None = None) -> int:
    parser = build_parser()
    normalized_argv = None if argv is None else _normalize_global_options(argv)
    args = parser.parse_args(normalized_argv)
    active_client: HTTPClient | None = client
    try:
        active_client = client or HTTPClient(load_config(args))
        if args.command == "recall":
            receipt, code = _run_recall(args, active_client)
        elif args.command == "ingest":
            receipt, code = _run_ingest(args, active_client)
        elif args.command == "job":
            receipt, code = _run_job(args, active_client)
        else:
            receipt, code = _run_turn(args, active_client)
    except ClientCLIError as exc:
        receipt, code = _error_receipt(getattr(args, "command", "cli"), None, exc), 1
    _print_receipt(receipt, secret=active_client.config.api_key if active_client else None)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
