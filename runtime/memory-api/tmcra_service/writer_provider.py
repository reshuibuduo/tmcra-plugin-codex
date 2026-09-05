from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit


DEEPSEEK_PROVIDER = "deepseek"
LOCAL_QWEN_PROVIDER = "local-qwen"
OPENAI_COMPATIBLE_PROVIDER = "openai-compatible"
LOCAL_QWEN_BASE_URL = "http://127.0.0.1:11435/v1"
LOCAL_QWEN_MODEL = "tmcra-qwen3.6-35b-a3b-iq3s"
LOCAL_QWEN_PROMPT_ADAPTER = "qwen36-v5"
LOCAL_QWEN_REVIEWER_PROMPT_ADAPTER = "qwen36-reconciliation-v1"
LOCAL_QWEN_SLOW_PROMPT_ADAPTER = "qwen36-slow-graph-v1"
LOCAL_QWEN_MIN_CONTEXT_TOKENS = 65536
LOCAL_QWEN_WRITER_SLOT_ID = 0
LOCAL_QWEN_PLANNER_SLOT_ID = 1
LOCAL_QWEN_GRAPH_SLOT_ID = 2
DESKTOP_LOCAL_QWEN_BASE_URL = "http://127.0.0.1:2010/v1"
DESKTOP_LOCAL_QWEN_MODEL = "tmcra-qwen3-4b-q4km"
DESKTOP_LOCAL_QWEN_PROMPT_ADAPTER = "qwen-local-v1"
DESKTOP_LOCAL_QWEN_REVIEWER_PROMPT_ADAPTER = "qwen-local-reconciliation-v1"
DESKTOP_LOCAL_QWEN_MIN_CONTEXT_TOKENS = 32768
OPENAI_WRITER_PROMPT_ADAPTER = "openai-memory-v1"
OPENAI_REVIEWER_PROMPT_ADAPTER = "openai-memory-reconciliation-v1"


@dataclass(frozen=True)
class WriterProviderRoute:
    provider: str
    base_url: str
    model: str
    api_keys: tuple[str, ...]
    pool_name: str
    prompt_adapter: str
    paid: bool


def _value(environment: Mapping[str, str], name: str, default: str = "") -> str:
    return str(environment.get(name) or default).strip()


def _keys(environment: Mapping[str, str], name: str) -> tuple[str, ...]:
    raw = str(environment.get(name) or "")
    parts = raw.split(",") if raw else []
    values = tuple(part.strip() for part in parts)
    if not values or any(not value for value in values):
        raise ValueError(f"{name} is missing or contains an empty entry")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate keys")
    return values


def _validate_https_provider_url(base_url: str, *, name: str) -> None:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{name} must be a credential-free HTTPS URL")


def validate_openai_compatible_url(base_url: str, *, name: str) -> None:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise ValueError(f"{name} must be a credential-free HTTP(S) /v1 URL")
    try:
        loopback = ipaddress.ip_address(str(parsed.hostname)).is_loopback
    except ValueError:
        loopback = str(parsed.hostname).lower() == "localhost"
    if parsed.scheme == "http" and not loopback:
        raise ValueError(f"{name} may use plain HTTP only on an exact loopback host")


def validate_loopback_openai_compatible_url(base_url: str, *, name: str) -> None:
    validate_openai_compatible_url(base_url, name=name)
    parsed = urlsplit(base_url)
    try:
        loopback = ipaddress.ip_address(str(parsed.hostname)).is_loopback
    except ValueError:
        loopback = str(parsed.hostname).lower() == "localhost"
    if parsed.scheme != "http" or not loopback:
        raise ValueError(f"{name} must use an exact loopback HTTP /v1 URL")


def _local_writer_identity(
    *, base_url: str, model: str, prompt_adapter: str, reviewer: bool
) -> str:
    legacy_adapter = (
        LOCAL_QWEN_REVIEWER_PROMPT_ADAPTER
        if reviewer
        else LOCAL_QWEN_PROMPT_ADAPTER
    )
    desktop_adapter = (
        DESKTOP_LOCAL_QWEN_REVIEWER_PROMPT_ADAPTER
        if reviewer
        else DESKTOP_LOCAL_QWEN_PROMPT_ADAPTER
    )
    if model == DESKTOP_LOCAL_QWEN_MODEL and prompt_adapter == desktop_adapter:
        validate_loopback_openai_compatible_url(
            base_url, name="desktop local Qwen Writer base URL"
        )
        return "desktop-qwen3"
    validate_loopback_openai_compatible_url(
        base_url, name="local Writer base URL"
    )
    if not model:
        raise ValueError("local Writer model alias is required")
    if prompt_adapter != legacy_adapter:
        raise ValueError(
            f"local Writer must use the configured {legacy_adapter} prompt adapter"
        )
    return "server-local"


def primary_writer_route(
    environment: Mapping[str, str],
) -> WriterProviderRoute:
    provider = _value(environment, "TMCRA_WRITER_PROVIDER", DEEPSEEK_PROVIDER)
    base_url = _value(environment, "TMCRA_WRITER_BASE_URL")
    model = _value(environment, "TMCRA_WRITER_MODEL")
    keys = _keys(environment, "TMCRA_WRITER_API_KEY_POOL")
    prompt_adapter = _value(environment, "TMCRA_WRITER_PROMPT_ADAPTER", "none")
    if provider == DEEPSEEK_PROVIDER:
        _validate_https_provider_url(base_url, name="TMCRA_WRITER_BASE_URL")
        if not model:
            raise ValueError("DeepSeek Writer model is required")
        if prompt_adapter != "none":
            raise ValueError("DeepSeek Writer must not use a local prompt adapter")
        return WriterProviderRoute(
            provider=provider,
            base_url=base_url,
            model=model,
            api_keys=keys,
            pool_name="deepseek-writer",
            prompt_adapter=prompt_adapter,
            paid=True,
        )
    if provider == LOCAL_QWEN_PROVIDER:
        identity = _local_writer_identity(
            base_url=base_url,
            model=model,
            prompt_adapter=prompt_adapter,
            reviewer=False,
        )
        if len(keys) != 1:
            raise ValueError("local Qwen Writer requires exactly one loopback key")
        return WriterProviderRoute(
            provider=provider,
            base_url=base_url,
            model=model,
            api_keys=keys,
            pool_name=(
                "local-qwen-writer"
                if identity == "server-local"
                else "local-qwen-writer-desktop"
            ),
            prompt_adapter=prompt_adapter,
            paid=False,
        )
    if provider == OPENAI_COMPATIBLE_PROVIDER:
        validate_openai_compatible_url(base_url, name="TMCRA_WRITER_BASE_URL")
        if not model:
            raise ValueError("OpenAI-compatible Writer model is required")
        if prompt_adapter != OPENAI_WRITER_PROMPT_ADAPTER:
            raise ValueError(
                "OpenAI-compatible Writer must use openai-memory-v1"
            )
        return WriterProviderRoute(
            provider=provider,
            base_url=base_url.rstrip("/"),
            model=model,
            api_keys=keys,
            pool_name="openai-compatible-writer",
            prompt_adapter=prompt_adapter,
            paid=True,
        )
    raise ValueError(f"unsupported Writer provider: {provider}")


def reviewer_writer_route(
    environment: Mapping[str, str],
    *,
    fallback_model: str = "deepseek-v4-pro",
) -> WriterProviderRoute:
    provider = _value(
        environment, "TMCRA_WRITER_REVIEWER_PROVIDER", DEEPSEEK_PROVIDER
    )
    primary_provider = _value(
        environment, "TMCRA_WRITER_PROVIDER", DEEPSEEK_PROVIDER
    )
    if provider == LOCAL_QWEN_PROVIDER:
        base_url = _value(
            environment,
            "TMCRA_WRITER_REVIEWER_BASE_URL",
            _value(environment, "TMCRA_WRITER_BASE_URL"),
        )
        model = _value(
            environment,
            "TMCRA_WRITER_REVIEWER_MODEL",
            _value(environment, "TMCRA_WRITER_MODEL"),
        )
        raw_keys = _value(
            environment,
            "TMCRA_WRITER_REVIEWER_API_KEY_POOL",
            _value(environment, "TMCRA_WRITER_API_KEY_POOL"),
        )
        reviewer_environment = dict(environment)
        reviewer_environment["TMCRA_WRITER_REVIEWER_API_KEY_POOL"] = raw_keys
        keys = _keys(reviewer_environment, "TMCRA_WRITER_REVIEWER_API_KEY_POOL")
        prompt_adapter = _value(
            environment,
            "TMCRA_WRITER_REVIEWER_PROMPT_ADAPTER",
            LOCAL_QWEN_REVIEWER_PROMPT_ADAPTER,
        )
        identity = _local_writer_identity(
            base_url=base_url,
            model=model,
            prompt_adapter=prompt_adapter,
            reviewer=True,
        )
        if len(keys) != 1:
            raise ValueError(
                "local Qwen Writer reviewer requires exactly one loopback key"
            )
        return WriterProviderRoute(
            provider=provider,
            base_url=base_url,
            model=model,
            api_keys=keys,
            pool_name=(
                "local-qwen-writer"
                if identity == "server-local"
                else "local-qwen-writer-desktop"
            ),
            prompt_adapter=prompt_adapter,
            paid=False,
        )
    if provider == OPENAI_COMPATIBLE_PROVIDER:
        base_url = _value(
            environment,
            "TMCRA_WRITER_REVIEWER_BASE_URL",
            _value(environment, "TMCRA_WRITER_BASE_URL"),
        )
        model = _value(
            environment,
            "TMCRA_WRITER_REVIEWER_MODEL",
            _value(environment, "TMCRA_WRITER_MODEL"),
        )
        raw_keys = _value(
            environment,
            "TMCRA_WRITER_REVIEWER_API_KEY_POOL",
            _value(environment, "TMCRA_WRITER_API_KEY_POOL"),
        )
        reviewer_environment = dict(environment)
        reviewer_environment["TMCRA_WRITER_REVIEWER_API_KEY_POOL"] = raw_keys
        keys = _keys(reviewer_environment, "TMCRA_WRITER_REVIEWER_API_KEY_POOL")
        prompt_adapter = _value(
            environment,
            "TMCRA_WRITER_REVIEWER_PROMPT_ADAPTER",
            OPENAI_REVIEWER_PROMPT_ADAPTER,
        )
        validate_openai_compatible_url(
            base_url, name="TMCRA_WRITER_REVIEWER_BASE_URL"
        )
        if not model or prompt_adapter != OPENAI_REVIEWER_PROMPT_ADAPTER:
            raise ValueError(
                "OpenAI-compatible Writer reviewer route is incomplete"
            )
        return WriterProviderRoute(
            provider=provider,
            base_url=base_url.rstrip("/"),
            model=model,
            api_keys=keys,
            pool_name="openai-compatible-writer",
            prompt_adapter=prompt_adapter,
            paid=True,
        )
    if provider != DEEPSEEK_PROVIDER:
        raise ValueError("unsupported Writer reviewer provider")
    fallback_base_url = _value(environment, "TMCRA_DEEPSEEK_WRITER_BASE_URL")
    fallback_keys = _value(environment, "TMCRA_DEEPSEEK_WRITER_KEY_POOL")
    if primary_provider == DEEPSEEK_PROVIDER:
        fallback_base_url = fallback_base_url or _value(
            environment, "TMCRA_WRITER_BASE_URL"
        )
        fallback_keys = fallback_keys or _value(
            environment, "TMCRA_WRITER_API_KEY_POOL"
        )
    base_url = _value(
        environment,
        "TMCRA_WRITER_REVIEWER_BASE_URL",
        fallback_base_url,
    )
    model = _value(
        environment, "TMCRA_WRITER_REVIEWER_MODEL", fallback_model
    )
    raw_keys = _value(
        environment,
        "TMCRA_WRITER_REVIEWER_API_KEY_POOL",
        fallback_keys,
    )
    reviewer_environment = dict(environment)
    reviewer_environment["TMCRA_WRITER_REVIEWER_API_KEY_POOL"] = raw_keys
    keys = _keys(reviewer_environment, "TMCRA_WRITER_REVIEWER_API_KEY_POOL")
    _validate_https_provider_url(
        base_url, name="TMCRA_WRITER_REVIEWER_BASE_URL"
    )
    if not model:
        raise ValueError("Writer reviewer model is required")
    return WriterProviderRoute(
        provider=provider,
        base_url=base_url,
        model=model,
        api_keys=keys,
        pool_name="deepseek-writer",
        prompt_adapter="none",
        paid=True,
    )
