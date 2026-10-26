from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit

from .writer_provider import (
    DEEPSEEK_PROVIDER,
    LOCAL_QWEN_PROVIDER,
    validate_loopback_openai_compatible_url,
)


LOCAL_QWEN_PLANNER_ADAPTER = "qwen36-planner-v1"


@dataclass(frozen=True)
class RecallPlannerRoute:
    provider: str
    base_url: str
    model: str
    api_keys: tuple[str, ...]
    prompt_adapter: str
    paid: bool


def _value(environment: Mapping[str, str], name: str, default: str = "") -> str:
    return str(environment.get(name) or default).strip()


def _keys(environment: Mapping[str, str], name: str) -> tuple[str, ...]:
    raw = str(environment.get(name) or "")
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique non-empty keys")
    return values


def _require_https(base_url: str) -> None:
    parsed = urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username:
        raise ValueError("DeepSeek recall planner requires an HTTPS provider URL")


def recall_planner_route(
    environment: Mapping[str, str],
) -> RecallPlannerRoute:
    provider = _value(
        environment, "TMCRA_RECALL_PLANNER_PROVIDER", DEEPSEEK_PROVIDER
    )
    base_url = _value(
        environment,
        "TMCRA_RECALL_PLANNER_BASE_URL",
        _value(environment, "TMCRA_WRITER_BASE_URL"),
    )
    model = _value(
        environment,
        "TMCRA_RECALL_PLANNER_MODEL",
        _value(environment, "TMCRA_WRITER_MODEL"),
    )
    planner_key_pool = _value(
        environment,
        "TMCRA_RECALL_PLANNER_API_KEY_POOL",
        _value(environment, "TMCRA_WRITER_API_KEY_POOL"),
    )
    route_environment = {
        **dict(environment),
        "TMCRA_RECALL_PLANNER_API_KEY_POOL": planner_key_pool,
    }
    api_keys = _keys(route_environment, "TMCRA_RECALL_PLANNER_API_KEY_POOL")
    adapter = _value(
        environment, "TMCRA_RECALL_PLANNER_PROMPT_ADAPTER", "none"
    )
    if provider == DEEPSEEK_PROVIDER:
        _require_https(base_url)
        if not model or adapter != "none":
            raise ValueError("DeepSeek recall planner route drifted from its contract")
        return RecallPlannerRoute(
            provider=provider,
            base_url=base_url.rstrip("/"),
            model=model,
            api_keys=api_keys,
            prompt_adapter=adapter,
            paid=True,
        )
    if provider == LOCAL_QWEN_PROVIDER:
        validate_loopback_openai_compatible_url(
            base_url, name="TMCRA_RECALL_PLANNER_BASE_URL"
        )
        if not model or adapter != LOCAL_QWEN_PLANNER_ADAPTER or len(api_keys) != 1:
            raise ValueError("local Qwen recall planner route drifted from its contract")
        return RecallPlannerRoute(
            provider=provider,
            base_url=base_url.rstrip("/"),
            model=model,
            api_keys=api_keys,
            prompt_adapter=adapter,
            paid=False,
        )
    raise ValueError(f"unsupported recall planner provider: {provider}")
