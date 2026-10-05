from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import tmcra_v4_recall_planner as core
from tmcra_v3_recall_planner import RecallPlannerError, RecallPlannerResponseError

from .writer_provider import (
    LOCAL_QWEN_PLANNER_SLOT_ID,
    validate_loopback_openai_compatible_url,
)


PLANNER_ADAPTER_ID = "qwen36-planner-v1"
LOCAL_SYSTEM_PROMPT = core.SYSTEM_PROMPT + """

Local Qwen execution rules:
- resolved_query must be understandable without recent_dialogue. Replace
  pronouns and deictic phrases such as it, that, there, this city, or the former
  with the event or entity being asked about when recent_dialogue provides it.
- Rewrite only the question. Do not answer it, disclose candidate evidence, or
  copy unrelated dialogue into resolved_query.
- temporal_focus describes the time of the fact, event, state, or decision
  requested by the query. A question about a past request is historical even
  when that request created a future recurring task.
- Include source, fast, and slow exactly once even when a layer is unavailable.
Example: dialogue says "I moved to Hangzhou last week" and the query is
"Which city is that?"; resolved_query should ask which city the user said they
moved to last week, without answering the question.
"""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_local_usage(value: Any) -> dict[str, int]:
    normalized = dict(value) if isinstance(value, Mapping) else value
    if isinstance(normalized, dict):
        details = normalized.get("prompt_tokens_details")
        if isinstance(details, Mapping) and normalized.get("cached_tokens") is None:
            normalized["cached_tokens"] = details.get("cached_tokens", 0)
    return core._normalize_usage(normalized)


class LocalQwenRecallRolePlanner:
    """Strict local OpenAI-compatible transport for RecallRolePlan v1."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_keys: Sequence[str],
        timeout: float = 60.0,
        max_tokens: int = 512,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url = _text(base_url).rstrip("/")
        self.model = _text(model)
        self.api_keys = list(dict.fromkeys(_text(key) for key in api_keys if _text(key)))
        self.timeout = max(1.0, float(timeout))
        self.max_tokens = max(128, int(max_tokens))
        self.request_index = 0
        self.user_id = ""
        self.opener = opener or urllib.request.urlopen
        try:
            validate_loopback_openai_compatible_url(
                self.base_url, name="TMCRA_RECALL_PLANNER_BASE_URL"
            )
        except ValueError as exc:
            raise RecallPlannerError("local planner route is invalid") from exc
        if not self.model or len(self.api_keys) != 1:
            raise RecallPlannerError("local Qwen planner route is not production-approved")

    def _metadata(
        self,
        *,
        physical_call_id: str,
        key_index: int,
        started: float,
        finish_reason: str,
        content: str = "",
        usage: Mapping[str, Any] | None = None,
        http_status: int | None = None,
        error_type: str | None = None,
        request_sha256: str = "",
        response_id: str = "",
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "physical_call_id": physical_call_id,
            "physical_api_call": True,
            "physical_api_calls": 1,
            "stage": "recall_planner",
            "provider": "local-qwen",
            "model": self.model,
            "api_key_index": key_index,
            "latency_seconds": round(time.time() - started, 3),
            "response_sha256": _sha256(content),
            "finish_reason": finish_reason,
            "status": "completed" if finish_reason == "stop" else finish_reason,
            "planner_version": core.PLANNER_VERSION,
            "prompt_version": core.PLANNER_PROMPT_VERSION + "+qwen36-planner-v1",
            "prompt_adapter": PLANNER_ADAPTER_ID,
            "request_sha256": request_sha256,
            "response_id": response_id,
        }
        if usage is not None:
            normalized_usage = dict(usage)
            metadata.update(normalized_usage)
            metadata["usage"] = normalized_usage
        if http_status is not None:
            metadata["http_status"] = int(http_status)
        if error_type:
            metadata["error_type"] = error_type
        return metadata

    def plan(
        self,
        *,
        query: str,
        question_date: str,
        available_layers: Mapping[str, Any],
        recent_dialogue: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        query, question_date = _text(query), _text(question_date)
        if not query or not question_date:
            raise RecallPlannerError("query and question_date are required")
        if not isinstance(available_layers, Mapping) or set(available_layers) != set(
            core.LAYER_NAMES
        ):
            raise RecallPlannerError(
                "available_layers must contain exactly source, fast, and slow summaries"
            )
        core._reject_gold(available_layers, "available_layers")
        dialogue = core._validate_recent_dialogue(recent_dialogue)
        payload = {
            "query": query,
            "question_date": question_date,
            "recent_dialogue": dialogue,
            "available_layers": dict(available_layers),
        }
        key_index = self.request_index % len(self.api_keys)
        self.request_index += 1
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": LOCAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "enable_thinking": False,
        }
        body["id_slot"] = 0 if os.getenv("TMCRA_DEPLOYMENT_MODE") == "local" else LOCAL_QWEN_PLANNER_SLOT_ID
        encoded_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request_sha256 = _sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True)
        )
        physical_call_id = "lqp_" + uuid.uuid4().hex
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=encoded_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_keys[key_index]}",
            },
            method="POST",
        )
        started = time.time()
        try:
            with self.opener(request, timeout=self.timeout) as response:
                http_status = int(response.getcode())
                raw_http = response.read().decode("utf-8")
                response_payload = json.loads(raw_http)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RecallPlannerResponseError(
                f"local recall planner HTTP {exc.code}: {detail}",
                response_content=detail,
                request_metadata=self._metadata(
                    physical_call_id=physical_call_id,
                    key_index=key_index,
                    started=started,
                    finish_reason="http_error",
                    content=detail,
                    http_status=exc.code,
                    request_sha256=request_sha256,
                ),
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raw = locals().get("raw_http", "")
            raise RecallPlannerResponseError(
                "local recall planner returned invalid HTTP JSON",
                response_content=raw,
                request_metadata=self._metadata(
                    physical_call_id=physical_call_id,
                    key_index=key_index,
                    started=started,
                    finish_reason="invalid_http_json",
                    content=raw,
                    error_type=type(exc).__name__,
                    request_sha256=request_sha256,
                    http_status=locals().get("http_status"),
                ),
            ) from exc
        except Exception as exc:
            raise RecallPlannerResponseError(
                f"local recall planner request failed: {type(exc).__name__}: {exc}",
                request_metadata=self._metadata(
                    physical_call_id=physical_call_id,
                    key_index=key_index,
                    started=started,
                    finish_reason="request_error",
                    error_type=type(exc).__name__,
                    request_sha256=request_sha256,
                ),
            ) from exc
        try:
            usage = _normalize_local_usage(
                response_payload.get("usage")
                if isinstance(response_payload, Mapping)
                else None
            )
            choices = (
                response_payload.get("choices")
                if isinstance(response_payload, Mapping)
                else None
            )
            if (
                not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(choices[0], Mapping)
            ):
                raise RecallPlannerError("response must contain exactly one choice")
            choice = choices[0]
            message = choice.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
            finish_reason = _text(choice.get("finish_reason"))
            metadata = self._metadata(
                physical_call_id=physical_call_id,
                key_index=key_index,
                started=started,
                finish_reason=finish_reason,
                content=content if isinstance(content, str) else raw_http,
                usage=usage,
                http_status=http_status,
                request_sha256=request_sha256,
                response_id=_text(response_payload.get("id")),
            )
            if finish_reason != "stop" or not isinstance(content, str):
                raise RecallPlannerError("response did not finish with a JSON string")
            plan = core.validate_recall_role_plan(json.loads(content))
        except (json.JSONDecodeError, RecallPlannerError) as exc:
            response_content = locals().get("content")
            if not isinstance(response_content, str):
                response_content = raw_http
            metadata = locals().get(
                "metadata",
                self._metadata(
                    physical_call_id=physical_call_id,
                    key_index=key_index,
                    started=started,
                    finish_reason="invalid_response",
                    content=raw_http,
                    http_status=http_status,
                    request_sha256=request_sha256,
                ),
            )
            raise RecallPlannerResponseError(
                f"local recall planner returned invalid RecallRolePlan: {exc}",
                response_content=response_content,
                request_metadata=metadata,
            ) from exc
        if plan["query_kind"] not in core.QUERY_KINDS:
            metadata["validation_warnings"] = [
                {
                    "code": "noncanonical_query_kind",
                    "query_kind": plan["query_kind"],
                    "disposition": "preserved_as_standard_query",
                }
            ]
        return plan, metadata
