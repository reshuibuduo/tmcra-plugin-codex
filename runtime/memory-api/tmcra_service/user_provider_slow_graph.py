"""Service adapter for executing the pinned V4 Slow Graph on a user device.

The pinned algorithm remains byte-for-byte unchanged.  This module supplies
its existing Flash/Pro client boundary with the durable user-provider broker.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import tmcra_v4_slow_graph as _slow

from .usage_attribution import UsageAttribution
from .user_provider_client import (
    USER_PROVIDER,
    UserProviderBrokerClient,
    normalize_user_provider_execution,
)


def _required_environment(name: str) -> str:
    value = str(os.getenv(name) or "").strip()
    if not value:
        raise _slow.SlowGraphError(
            f"{name} is required for user-provider execution"
        )
    return value


def _usage_for_slow_graph(metadata: Mapping[str, Any]) -> dict[str, int]:
    raw = metadata.get("usage")
    usage = raw if isinstance(raw, Mapping) else {}
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    hit = int(
        usage.get(
            "prompt_cache_hit_tokens",
            usage.get("cache_hit_tokens", 0),
        )
        or 0
    )
    miss = int(
        usage.get(
            "prompt_cache_miss_tokens",
            usage.get("cache_miss_tokens", max(0, prompt - hit)),
        )
        or 0
    )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cache_read_input_tokens": hit,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "total_tokens": int(
            usage.get("total_tokens", prompt + completion)
            or prompt + completion
        ),
    }


class UserProviderTierClient(_slow._DeepSeekTierClient):
    """Preserve the pinned prompt and validation logic around a broker call."""

    def __init__(
        self,
        *,
        route: str,
        broker: UserProviderBrokerClient,
    ) -> None:
        super().__init__(
            _slow.DeepSeekTierConfig(
                base_url="https://user-provider.invalid/v1",
                key_pool=("broker",),
                max_tokens=broker.max_tokens,
                model="client-selected",
            ),
            route=route,
        )
        self.broker = broker

    def _response_metadata(
        self,
        metadata: Mapping[str, Any],
        output: Mapping[str, Any],
    ) -> dict[str, Any]:
        content = _slow._json(output)
        usage = _usage_for_slow_graph(metadata)
        provider = str(
            metadata.get("provider")
            or metadata.get("api_provider")
            or USER_PROVIDER
        )
        model = str(metadata.get("model") or "client-selected")
        synthetic_response = {
            "id": metadata.get("provider_request_id")
            or metadata.get("physical_call_id"),
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": usage,
        }
        started = metadata.get("started_at")
        completed = metadata.get("completed_at")
        latency_ms = 0.0
        if isinstance(started, (int, float)) and isinstance(
            completed, (int, float)
        ):
            latency_ms = max(0.0, (float(completed) - float(started)) * 1000)
        return {
            **dict(metadata),
            "route": self.route,
            "prompt_version": _slow.SLOW_PROMPT_VERSION,
            "provider": provider,
            "api_provider": provider,
            "model": model,
            "execution_route": USER_PROVIDER,
            "status": "response_received",
            "http_status": 200,
            "finish_reason": "stop",
            "content": content,
            "usage": usage,
            "provider_usage": usage,
            "cost_audit": {**usage, "estimated_cost": 0.0},
            "raw_response": _slow._json(synthetic_response),
            "latency_ms": round(latency_ms, 3),
        }

    def _propose(
        self,
        region: Mapping[str, Any],
        capsules: list[Mapping[str, Any]],
        *,
        correction: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        _slow._assert_no_benchmark_fields(region)
        _slow._assert_no_benchmark_fields(capsules)
        messages = self._messages(region, capsules, correction=correction)
        operation = (
            f"slow_graph_{self.route}_correction"
            if correction is not None
            else f"slow_graph_{self.route}"
        )
        try:
            raw_patch, metadata = self.broker.complete_messages(
                messages=messages,
                operation=operation,
            )
        except Exception as exc:
            error_metadata = getattr(exc, "metadata", None)
            self.last_call_metadata = {
                **(
                    dict(error_metadata)
                    if isinstance(error_metadata, Mapping)
                    else {}
                ),
                "route": self.route,
                "prompt_version": _slow.SLOW_PROMPT_VERSION,
                "execution_route": USER_PROVIDER,
            }
            raise _slow.TieredAPIError(
                f"{self.route} user-provider call failed: {exc}"
            ) from exc

        self.last_call_metadata = self._response_metadata(metadata, raw_patch)
        if self.route == "flash" and _slow._flash_escalation_patch(raw_patch):
            required_evidence_ids = region.get("required_evidence_ids")
            if not isinstance(required_evidence_ids, list) or not required_evidence_ids:
                raise _slow.TieredAPIError(
                    "flash escalation requires pending durable evidence"
                )
            self.last_call_metadata = {
                **dict(self.last_call_metadata),
                "status": "completed",
                "escalation_requested": True,
                "escalation_reason": _slow.FLASH_ESCALATION_REASON,
                "raw_patch_sha256": _slow._digest(raw_patch),
            }
            return raw_patch

        patch, transport_normalizations = _slow._normalize_transport_patch(
            raw_patch,
            capsules,
            region,
        )
        if transport_normalizations:
            self.last_call_metadata = {
                **dict(self.last_call_metadata),
                "transport_normalizations": transport_normalizations,
                "raw_patch_sha256": _slow._digest(raw_patch),
                "normalized_patch_sha256": _slow._digest(patch),
            }
        try:
            _slow.validate_patch(patch)
        except _slow.PatchValidationError as exc:
            raise _slow.TieredAPIError(
                f"{self.route} returned an invalid GraphPatch: {exc}"
            ) from exc
        self.last_call_metadata = {
            **dict(self.last_call_metadata),
            "status": "completed",
        }
        return patch


def manager_from_environment() -> Any:
    try:
        raw_execution = json.loads(
            _required_environment("TMCRA_USER_PROVIDER_EXECUTION_JSON")
        )
        execution = normalize_user_provider_execution(
            raw_execution,
            stage="organizer",
        )
        if execution is None:
            raise ValueError("organizer execution route is missing")
        attribution_raw = str(
            os.getenv("TMCRA_USAGE_ATTRIBUTION_JSON") or "{}"
        ).strip()
        usage_attribution = UsageAttribution.from_mapping(
            json.loads(attribution_raw)
        )
        control_db = Path(_required_environment("TMCRA_SERVICE_CONTROL_DB"))
        tenant_id = _required_environment("TMCRA_SERVICE_TENANT_ID")
        scope_name = _required_environment("TMCRA_SERVICE_SCOPE_NAME")
        job_id = _required_environment("TMCRA_SERVICE_JOB_ID")
        stage_id = _required_environment("TMCRA_SERVICE_STAGE_ID")
        timeout = float(os.getenv("TMCRA_USER_PROVIDER_TIMEOUT_SECONDS", "900"))
        max_tokens = int(os.getenv("TMCRA_SLOW_GRAPH_MAX_TOKENS", "16384"))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise _slow.SlowGraphError(
            "user-provider Slow Graph environment is invalid"
        ) from exc

    def broker() -> UserProviderBrokerClient:
        return UserProviderBrokerClient(
            control_db=control_db,
            tenant_id=tenant_id,
            scope_name=scope_name,
            auth_key_id=execution["auth_key_id"],
            job_id=job_id,
            stage_id=stage_id,
            task_stage="organizer",
            timeout=timeout,
            max_tokens=max_tokens,
            usage_attribution=usage_attribution,
            record_ledger=False,
        )

    manager = _slow.TieredGraphPatchManager(
        flash=UserProviderTierClient(route="flash", broker=broker()),
        pro=UserProviderTierClient(route="pro", broker=broker()),
    )
    manager.model_config = {
        **dict(manager.model_config),
        "model": "user-provider-tiered-slow-graph",
        "provider": USER_PROVIDER,
    }
    return manager


def main() -> None:
    _slow.TieredGraphPatchManager.from_env = classmethod(  # type: ignore[method-assign]
        lambda _cls: manager_from_environment()
    )
    _slow.main()


if __name__ == "__main__":
    main()
