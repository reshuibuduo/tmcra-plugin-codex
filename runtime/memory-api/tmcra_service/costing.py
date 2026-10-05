from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from .jobs import JobStore
from .usage_attribution import UNATTRIBUTED, UsageAttribution
from .writer import (
    DEEPSEEK_PRICING_SOURCE,
    DEEPSEEK_V4_PRICES_MICRO_CNY,
    DEEPSEEK_V4_PRICE_VERSION,
)
from .writer_provider import (
    DEEPSEEK_PROVIDER,
    LOCAL_QWEN_PROVIDER,
    OPENAI_COMPATIBLE_PROVIDER,
)


LOCAL_EXTERNAL_PRICE_VERSION = "tmcra-local-external-api-cost-v1"
LOCAL_EXTERNAL_PRICING_SOURCE = "self-hosted local inference; external API cost only"
LOCAL_OPENAI_COMPATIBLE_PROVIDER = "local-openai-compatible"


class ProviderMetadataError(ValueError):
    pass


def physical_call_metadata(value: Any) -> list[dict[str, Any]]:
    """Flatten direct and aggregate metadata into unique physical calls."""
    found: dict[str, dict[str, Any]] = {}

    def visit(item: Any) -> None:
        if not isinstance(item, Mapping):
            return
        for key in ("calls", "prior_calls", "tier_calls"):
            children = item.get(key)
            if isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
                for child in children:
                    visit(child)
        call_id = str(item.get("physical_call_id") or "").strip()
        if call_id:
            found.setdefault(call_id, dict(item))

    visit(value)
    return list(found.values())


def _usage(metadata: Mapping[str, Any]) -> tuple[dict[str, int], str]:
    raw = metadata.get("usage")
    value = raw if isinstance(raw, Mapping) else metadata

    def count(*names: str) -> int | None:
        raw_value = next(
            (value.get(name) for name in names if value.get(name) is not None),
            None,
        )
        if raw_value is None:
            return None
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ProviderMetadataError("provider usage is not numeric")
        result = int(raw_value)
        if result < 0:
            raise ProviderMetadataError("provider usage is negative")
        return result

    prompt = count("prompt_tokens", "input_tokens")
    completion = count("completion_tokens", "output_tokens")
    hit = count(
        "prompt_cache_hit_tokens",
        "cache_hit_tokens",
        "cache_read_input_tokens",
        "cached_tokens",
    )
    miss = count("prompt_cache_miss_tokens", "cache_miss_tokens")
    if prompt is None or completion is None:
        return {}, "missing"
    if hit is None and miss is None:
        return {}, "invalid"
    if hit is None:
        hit = prompt - int(miss or 0)
    if miss is None:
        miss = prompt - int(hit)
    if hit < 0 or miss < 0 or hit + miss != prompt:
        return {}, "invalid"
    total = count("total_tokens")
    if total is None:
        total = prompt + completion
    if total < prompt + completion:
        return {}, "invalid"
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
    }, "complete"


def _terminal_status(metadata: Mapping[str, Any]) -> str:
    status = str(metadata.get("status") or "").strip().lower()
    if status in {
        "completed",
        "response_received",
        "completed_with_neutral_empty_layer_repair",
    }:
        return "completed"
    if status in {
        "request_error",
        "transport_error",
        "timeout",
        "started",
        "response_received_unvalidated",
    }:
        return "unknown"
    return "failed"


def _cost_micro_cny(
    provider: str, model: str, usage: Mapping[str, int]
) -> int | None:
    if provider in {LOCAL_QWEN_PROVIDER, LOCAL_OPENAI_COMPATIBLE_PROVIDER} and usage:
        return 0
    rates = DEEPSEEK_V4_PRICES_MICRO_CNY.get(model)
    if rates is None or not usage:
        return None
    numerator = (
        int(usage["cache_hit_tokens"]) * rates[0]
        + int(usage["cache_miss_tokens"]) * rates[1]
        + int(usage["output_tokens"]) * rates[2]
    )
    return (numerator + 999_999) // 1_000_000


def journal_deepseek_calls(
    store: JobStore,
    metadata: Any,
    *,
    tenant_id: str,
    scope_name: str,
    job_id: str | None,
    stage_id: str,
    operation: str,
    default_model: str,
    usage_attribution: UsageAttribution = UNATTRIBUTED,
) -> int:
    """Persist physical-call metadata and return the number of unique calls."""
    calls = physical_call_metadata(metadata)
    registered = 0
    for call in calls:
        provider = str(
            call.get("provider")
            or call.get("api_provider")
            or DEEPSEEK_PROVIDER
        ).strip().lower()
        model = str(call.get("model") or default_model).strip()
        if provider not in {
            DEEPSEEK_PROVIDER,
            LOCAL_QWEN_PROVIDER,
            LOCAL_OPENAI_COMPATIBLE_PROVIDER,
            OPENAI_COMPATIBLE_PROVIDER,
        }:
            raise ProviderMetadataError(f"unsupported provider metadata: {provider}")
        if store.get_provider_call(str(call["physical_call_id"])) is not None:
            continue
        usage, usage_state = _usage(call)
        terminal = _terminal_status(call)
        if provider == DEEPSEEK_PROVIDER:
            rates = DEEPSEEK_V4_PRICES_MICRO_CNY.get(model)
            price_version = DEEPSEEK_V4_PRICE_VERSION
            pricing_source = DEEPSEEK_PRICING_SOURCE
        elif provider in {LOCAL_QWEN_PROVIDER, LOCAL_OPENAI_COMPATIBLE_PROVIDER}:
            rates = (0, 0, 0)
            price_version = LOCAL_EXTERNAL_PRICE_VERSION
            pricing_source = LOCAL_EXTERNAL_PRICING_SOURCE
        else:
            rates = None
            price_version = "operator-pricing-not-configured"
            pricing_source = "operator pricing not configured"
        if rates is not None:
            store.upsert_provider_price(
                provider,
                model,
                cache_hit_input_micro_cny_per_million=rates[0],
                cache_miss_input_micro_cny_per_million=rates[1],
                output_micro_cny_per_million=rates[2],
                effective_at=0.0,
                currency="CNY",
                metadata={
                    "price_version": price_version,
                    "source": pricing_source,
                    "unit": "micro-CNY per million tokens",
                },
            )
        started_at = call.get("started_at")
        if not isinstance(started_at, (int, float)):
            latency = call.get("latency_seconds")
            started_at = time.time() - float(latency or 0.0)
        key_id = call.get("key_id")
        if key_id is None and call.get("api_key_index") is not None:
            key_id = f"key-index:{int(call['api_key_index'])}"
        call_id = str(call["physical_call_id"])
        store.record_provider_call(
            tenant_id,
            provider,
            model,
            scope_name=scope_name,
            call_id=call_id,
            job_id=job_id,
            stage_id=stage_id,
            operation=str(call.get("stage") or operation),
            status="started",
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            cache_hit_tokens=usage.get("cache_hit_tokens"),
            cache_miss_tokens=usage.get("cache_miss_tokens"),
            usage_state=usage_state,
            price_version=price_version if rates is not None else None,
            key_id=str(key_id) if key_id is not None else None,
            usage_attribution=usage_attribution,
            request_sha256=(
                str(call["request_sha256"]) if call.get("request_sha256") else None
            ),
            started_at=float(started_at),
            created_at=float(started_at),
        )
        store.transition_provider_call(
            call_id,
            terminal,
            error=(
                None
                if terminal == "completed"
                else str(call.get("error_type") or call.get("status") or "provider_failure")
            ),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            cache_hit_tokens=usage.get("cache_hit_tokens"),
            cache_miss_tokens=usage.get("cache_miss_tokens"),
            usage_state=usage_state,
            price_version=price_version if rates is not None else None,
            cost_micro_cny=(
                _cost_micro_cny(provider, model, usage)
                if terminal != "unknown" and usage_state == "complete"
                else None
            ),
            response_sha256=(
                str(call["response_sha256"]) if call.get("response_sha256") else None
            ),
        )
        registered += 1
    return registered
