"""Model-client adapter backed by the authenticated user-device task broker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .control_db import ControlDB
from .jobs import JobStore
from .usage_attribution import UNATTRIBUTED, UsageAttribution
from .user_provider_tasks import UserProviderTask, UserProviderTaskStore


USER_PROVIDER = "user-provider"
USER_PROVIDER_PRICE_VERSION = "user-provider-direct-billing-v1"


def normalize_user_provider_execution(
    value: Mapping[str, Any] | None,
    *,
    stage: str,
) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("provider execution contract must be an object")
    extras = set(value) - {"writer", "organizer", "auth_key_id"}
    if extras:
        raise ValueError("user-provider execution contract has unknown fields")
    route = str(value.get(stage) or "").strip()
    if not route:
        return None
    auth_key_id = str(value.get("auth_key_id") or "").strip()
    if route != USER_PROVIDER or not auth_key_id or len(auth_key_id) > 200:
        raise ValueError("user-provider execution contract is invalid")
    return {stage: route, "auth_key_id": auth_key_id}


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class UserProviderCallError(RuntimeError):
    def __init__(self, task: UserProviderTask) -> None:
        status = "request_error" if task.state == "unknown" else "client_error"
        metadata = UserProviderBrokerClient.task_metadata(task, status=status)
        metadata["error_code"] = task.error_code or "user_provider_failed"
        self.metadata = metadata
        super().__init__(
            f"user-provider task {task.task_id} ended as {task.state}: "
            f"{task.error_code or 'unspecified'}"
        )


class UserProviderBrokerClient:
    """Create one durable call task and wait for a locally validated JSON object."""

    def __init__(
        self,
        *,
        control_db: Path | str,
        tenant_id: str,
        scope_name: str,
        auth_key_id: str,
        job_id: str,
        stage_id: str,
        task_stage: str,
        timeout: float,
        max_tokens: int,
        usage_attribution: UsageAttribution = UNATTRIBUTED,
        record_ledger: bool = False,
    ) -> None:
        if task_stage not in {"writer", "organizer"}:
            raise ValueError("user-provider task stage is invalid")
        if timeout <= 0 or max_tokens <= 0:
            raise ValueError("user-provider timeout and max_tokens must be positive")
        identities = (tenant_id, scope_name, auth_key_id, job_id, stage_id)
        if any(not str(value).strip() for value in identities):
            raise ValueError("user-provider broker identity is incomplete")
        self.database = ControlDB(Path(control_db))
        self.tasks = UserProviderTaskStore(self.database)
        self.ledger = JobStore(self.database) if record_ledger else None
        self.tenant_id = tenant_id
        self.scope_name = scope_name
        self.auth_key_id = auth_key_id
        self.job_id = job_id
        self.stage_id = stage_id
        self.task_stage = task_stage
        self.timeout = float(timeout)
        self.max_tokens = int(max_tokens)
        self.usage_attribution = usage_attribution
        self.model = "client-selected"
        self.provider = USER_PROVIDER
        self.last_call_metadata: dict[str, Any] = {}

    @staticmethod
    def task_metadata(
        task: UserProviderTask, *, status: str | None = None
    ) -> dict[str, Any]:
        usage = dict(task.usage or {})
        prompt = int(usage.get("input_tokens", 0) or 0)
        completion = int(usage.get("output_tokens", 0) or 0)
        hit = int(usage.get("cache_hit_tokens", 0) or 0)
        miss = int(usage.get("cache_miss_tokens", max(0, prompt - hit)) or 0)
        normalized_usage = (
            {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "prompt_cache_hit_tokens": hit,
                "prompt_cache_miss_tokens": miss,
                "total_tokens": int(
                    usage.get("total_tokens", prompt + completion)
                    or prompt + completion
                ),
            }
            if task.usage is not None
            else {}
        )
        return {
            "physical_call_id": task.task_id,
            "physical_api_call": task.provider_started_at is not None,
            "physical_api_calls": 1 if task.provider_started_at is not None else 0,
            "stage": task.operation,
            "model": task.model or "client-selected",
            "provider": task.provider or USER_PROVIDER,
            "api_provider": task.provider or USER_PROVIDER,
            "execution_route": USER_PROVIDER,
            "status": status or task.state,
            "request_sha256": task.request_sha256,
            "response_sha256": task.response_sha256,
            "provider_request_id": task.provider_request_id,
            "started_at": task.provider_started_at or task.created_at,
            "completed_at": task.provider_finished_at or task.completed_at,
            "usage": normalized_usage,
            **normalized_usage,
        }

    def _record(self, task: UserProviderTask) -> None:
        if self.ledger is None:
            return
        usage = dict(task.usage or {})
        usage_state = "complete" if task.usage is not None else "missing"
        provider = task.provider or USER_PROVIDER
        model = task.model or "client-selected"
        self.ledger.record_provider_call(
            self.tenant_id,
            provider,
            model,
            scope_name=self.scope_name,
            call_id=task.task_id,
            job_id=self.job_id,
            stage_id=self.stage_id,
            operation=task.operation,
            status=task.state,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            cache_hit_tokens=usage.get("cache_hit_tokens"),
            cache_miss_tokens=usage.get("cache_miss_tokens"),
            usage_state=usage_state,
            price_version=USER_PROVIDER_PRICE_VERSION,
            usage_attribution=self.usage_attribution,
            request_sha256=task.request_sha256,
            response_sha256=task.response_sha256,
            started_at=task.provider_started_at or task.created_at,
            finished_at=task.provider_finished_at or task.completed_at,
            created_at=task.created_at,
        )

    def complete_messages(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        operation: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request = {
            "schema_version": "tmcra.openai-compatible-request.1",
            "messages": [dict(message) for message in messages],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "tmcra_structured_response",
                        "strict": True,
                        "schema": dict(response_schema),
                    },
                }
                if response_schema is not None
                else {"type": "json_object"}
            ),
        }
        task = self.tasks.create(
            tenant_id=self.tenant_id,
            scope_name=self.scope_name,
            auth_key_id=self.auth_key_id,
            job_id=self.job_id,
            stage_id=self.stage_id,
            task_stage=self.task_stage,
            operation=operation,
            request=request,
        )
        task = self.tasks.await_terminal(task.task_id, timeout=self.timeout)
        self.model = task.model or self.model
        self.provider = task.provider or self.provider
        self.last_call_metadata = self.task_metadata(task)
        self._record(task)
        if task.state != "completed" or task.output is None:
            raise UserProviderCallError(task)
        return dict(task.output), dict(self.last_call_metadata)

    def complete_prompt(
        self,
        *,
        system_prompt: str,
        payload: Mapping[str, Any],
        operation: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.complete_messages(
            messages=(
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _json(dict(payload))},
            ),
            operation=operation,
            response_schema=response_schema,
        )
