from __future__ import annotations

import ipaddress
import os
import stat
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import uvicorn

from .app import create_app
from .planner_provider import LOCAL_QWEN_PLANNER_ADAPTER
from .settings import ServiceSettings
from .shared_core import SharedCoreVerificationError, verify_shared_core
from .writer_provider import (
    DEEPSEEK_PROVIDER,
    LOCAL_QWEN_BASE_URL,
    LOCAL_QWEN_MODEL,
    LOCAL_QWEN_PROMPT_ADAPTER,
    LOCAL_QWEN_REVIEWER_PROMPT_ADAPTER,
    LOCAL_QWEN_SLOW_PROMPT_ADAPTER,
    LOCAL_QWEN_PROVIDER,
)


DEFAULT_WRITER_ENV = (
    "/opt/tmcra-data/migration/legacy/"
    "tmcra_api_service/env/deepseek-writer-pool.env"
)


def _load_shell_environment(path: str | Path) -> None:
    from tmcra_local_only import enabled, read_environment
    if enabled():
        environment = read_environment(path)
        os.environ.clear()
        os.environ.update(environment)
        return
    env_path = Path(path).expanduser()
    if not env_path.is_file():
        raise RuntimeError(f"writer env file is missing or not a file: {env_path}")
    try:
        result = subprocess.run(
            [
                "bash",
                "-c",
                'set -a; source "$1"; env -0',
                "tmcra-service",
                str(env_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "writer env cannot be loaded: bash is not installed"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"writer env could not be sourced: {env_path}"
            + (f" ({detail})" if detail else "")
        ) from exc
    for entry in result.stdout.decode("utf-8").split("\0"):
        if "=" in entry:
            key, value = entry.split("=", 1)
            os.environ[key] = value


def _normalized_key_pool(raw: str, *, name: str) -> str:
    if not raw.strip():
        raise RuntimeError(f"writer configuration is invalid: set {name}")
    parts = raw.split(",")
    keys = [value.strip() for value in parts]
    if any(not value for value in keys):
        raise RuntimeError(
            f"writer key pool {name} is invalid: entries must not be empty"
        )
    if len(keys) != len(set(keys)):
        raise RuntimeError(
            f"writer key pool {name} is invalid: duplicate API keys are not allowed"
        )
    return ",".join(keys)


def _local_writer_key(path_value: str) -> str:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"local Writer API key file is missing: {path}")
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise RuntimeError(f"local Writer API key file is not private: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value or "," in value or "\n" in value or "\r" in value:
        raise RuntimeError("local Writer API key file must contain exactly one key")
    return value


def _configure_writer_aliases() -> None:
    from tmcra_local_only import enabled, validate_environment
    if enabled():
        validate_environment(os.environ)
        return
    deepseek_base_url = str(
        os.getenv("TMCRA_DEEPSEEK_WRITER_BASE_URL") or ""
    ).strip()
    if not deepseek_base_url:
        raise RuntimeError(
            "writer configuration is invalid: set TMCRA_DEEPSEEK_WRITER_BASE_URL"
        )
    deepseek_key_pool = _normalized_key_pool(
        str(os.getenv("TMCRA_DEEPSEEK_WRITER_KEY_POOL") or ""),
        name="TMCRA_DEEPSEEK_WRITER_KEY_POOL",
    )
    provider = str(
        os.getenv("TMCRA_WRITER_PROVIDER") or DEEPSEEK_PROVIDER
    ).strip()
    max_tokens = os.getenv("TMCRA_WRITER_MAX_TOKENS", "16384")
    if provider == DEEPSEEK_PROVIDER:
        primary = {
            "TMCRA_WRITER_PROVIDER": DEEPSEEK_PROVIDER,
            "TMCRA_WRITER_BASE_URL": deepseek_base_url,
            "TMCRA_WRITER_MODEL": str(
                os.getenv("TMCRA_WRITER_MODEL")
                or os.getenv("TMCRA_DEEPSEEK_FLASH_MODEL")
                or "deepseek-v4-flash"
            ).strip(),
            "TMCRA_WRITER_API_KEY_POOL": deepseek_key_pool,
            "TMCRA_WRITER_PROMPT_ADAPTER": "none",
        }
    elif provider == LOCAL_QWEN_PROVIDER:
        local_key_file = str(
            os.getenv("TMCRA_LOCAL_WRITER_API_KEY_FILE")
            or "/opt/tmcra-data/local-llm/secrets/qwen36-api.key"
        )
        primary = {
            "TMCRA_WRITER_PROVIDER": LOCAL_QWEN_PROVIDER,
            "TMCRA_WRITER_BASE_URL": str(
                os.getenv("TMCRA_LOCAL_WRITER_BASE_URL") or LOCAL_QWEN_BASE_URL
            ).strip(),
            "TMCRA_WRITER_MODEL": str(
                os.getenv("TMCRA_LOCAL_WRITER_MODEL")
                or os.getenv("TMCRA_WRITER_MODEL")
                or LOCAL_QWEN_MODEL
            ).strip(),
            "TMCRA_WRITER_API_KEY_POOL": _local_writer_key(local_key_file),
            "TMCRA_WRITER_PROMPT_ADAPTER": str(
                os.getenv("TMCRA_WRITER_PROMPT_ADAPTER")
                or LOCAL_QWEN_PROMPT_ADAPTER
            ).strip(),
        }
    else:
        raise RuntimeError(f"unsupported TMCRA_WRITER_PROVIDER: {provider}")
    planner_provider = str(
        os.getenv("TMCRA_RECALL_PLANNER_PROVIDER") or DEEPSEEK_PROVIDER
    ).strip()
    if planner_provider == DEEPSEEK_PROVIDER:
        planner = {
            "TMCRA_RECALL_PLANNER_PROVIDER": DEEPSEEK_PROVIDER,
            "TMCRA_RECALL_PLANNER_BASE_URL": deepseek_base_url,
            "TMCRA_RECALL_PLANNER_MODEL": str(
                os.getenv("TMCRA_RECALL_PLANNER_MODEL")
                or primary["TMCRA_WRITER_MODEL"]
            ).strip(),
            "TMCRA_RECALL_PLANNER_API_KEY_POOL": deepseek_key_pool,
            "TMCRA_RECALL_PLANNER_PROMPT_ADAPTER": "none",
        }
    elif planner_provider == LOCAL_QWEN_PROVIDER:
        local_planner_key_file = str(
            os.getenv("TMCRA_LOCAL_PLANNER_API_KEY_FILE")
            or os.getenv("TMCRA_LOCAL_WRITER_API_KEY_FILE")
            or "/opt/tmcra-data/local-llm/secrets/qwen36-api.key"
        )
        planner = {
            "TMCRA_RECALL_PLANNER_PROVIDER": LOCAL_QWEN_PROVIDER,
            "TMCRA_RECALL_PLANNER_BASE_URL": str(
                os.getenv("TMCRA_LOCAL_PLANNER_BASE_URL") or LOCAL_QWEN_BASE_URL
            ).strip(),
            "TMCRA_RECALL_PLANNER_MODEL": str(
                os.getenv("TMCRA_LOCAL_PLANNER_MODEL")
                or os.getenv("TMCRA_RECALL_PLANNER_MODEL")
                or primary["TMCRA_WRITER_MODEL"]
            ).strip(),
            "TMCRA_RECALL_PLANNER_API_KEY_POOL": _local_writer_key(
                local_planner_key_file
            ),
            "TMCRA_RECALL_PLANNER_PROMPT_ADAPTER": str(
                os.getenv("TMCRA_RECALL_PLANNER_PROMPT_ADAPTER")
                or LOCAL_QWEN_PLANNER_ADAPTER
            ).strip(),
        }
    else:
        raise RuntimeError(
            f"unsupported TMCRA_RECALL_PLANNER_PROVIDER: {planner_provider}"
        )
    reviewer_provider = str(
        os.getenv("TMCRA_WRITER_REVIEWER_PROVIDER")
        or (LOCAL_QWEN_PROVIDER if provider == LOCAL_QWEN_PROVIDER else DEEPSEEK_PROVIDER)
    ).strip()
    if reviewer_provider == LOCAL_QWEN_PROVIDER:
        local_reviewer_key_file = str(
            os.getenv("TMCRA_LOCAL_REVIEWER_API_KEY_FILE")
            or os.getenv("TMCRA_LOCAL_WRITER_API_KEY_FILE")
            or "/opt/tmcra-data/local-llm/secrets/qwen36-api.key"
        )
        reviewer = {
            "TMCRA_WRITER_REVIEWER_PROVIDER": LOCAL_QWEN_PROVIDER,
            "TMCRA_WRITER_REVIEWER_BASE_URL": str(
                os.getenv("TMCRA_LOCAL_REVIEWER_BASE_URL") or LOCAL_QWEN_BASE_URL
            ).strip(),
            "TMCRA_WRITER_REVIEWER_MODEL": str(
                os.getenv("TMCRA_LOCAL_REVIEWER_MODEL")
                or os.getenv("TMCRA_WRITER_REVIEWER_MODEL")
                or primary["TMCRA_WRITER_MODEL"]
            ).strip(),
            "TMCRA_WRITER_REVIEWER_API_KEY_POOL": _local_writer_key(
                local_reviewer_key_file
            ),
            "TMCRA_WRITER_REVIEWER_PROMPT_ADAPTER": str(
                os.getenv("TMCRA_WRITER_REVIEWER_PROMPT_ADAPTER")
                or LOCAL_QWEN_REVIEWER_PROMPT_ADAPTER
            ).strip(),
        }
    elif reviewer_provider == DEEPSEEK_PROVIDER:
        reviewer = {
            "TMCRA_WRITER_REVIEWER_PROVIDER": DEEPSEEK_PROVIDER,
            "TMCRA_WRITER_REVIEWER_BASE_URL": deepseek_base_url,
            "TMCRA_WRITER_REVIEWER_MODEL": str(
                os.getenv("TMCRA_WRITER_REVIEWER_MODEL")
                or os.getenv("TMCRA_DEEPSEEK_PRO_MODEL")
                or "deepseek-v4-pro"
            ).strip(),
            "TMCRA_WRITER_REVIEWER_API_KEY_POOL": deepseek_key_pool,
            "TMCRA_WRITER_REVIEWER_PROMPT_ADAPTER": "none",
        }
    else:
        raise RuntimeError(
            f"unsupported TMCRA_WRITER_REVIEWER_PROVIDER: {reviewer_provider}"
        )
    slow_provider = str(
        os.getenv("TMCRA_SLOW_GRAPH_PROVIDER")
        or (LOCAL_QWEN_PROVIDER if provider == LOCAL_QWEN_PROVIDER else DEEPSEEK_PROVIDER)
    ).strip()
    if slow_provider == LOCAL_QWEN_PROVIDER:
        local_slow_key_file = str(
            os.getenv("TMCRA_LOCAL_SLOW_GRAPH_API_KEY_FILE")
            or os.getenv("TMCRA_LOCAL_WRITER_API_KEY_FILE")
            or "/opt/tmcra-data/local-llm/secrets/qwen36-api.key"
        )
        slow = {
            "TMCRA_SLOW_GRAPH_PROVIDER": LOCAL_QWEN_PROVIDER,
            "TMCRA_SLOW_GRAPH_BASE_URL": str(
                os.getenv("TMCRA_LOCAL_SLOW_GRAPH_BASE_URL")
                or os.getenv("TMCRA_SLOW_GRAPH_BASE_URL")
                or primary["TMCRA_WRITER_BASE_URL"]
            ).strip(),
            "TMCRA_SLOW_GRAPH_MODEL": str(
                os.getenv("TMCRA_LOCAL_SLOW_GRAPH_MODEL")
                or os.getenv("TMCRA_SLOW_GRAPH_MODEL")
                or primary["TMCRA_WRITER_MODEL"]
            ).strip(),
            "TMCRA_SLOW_GRAPH_API_KEY_POOL": _local_writer_key(
                local_slow_key_file
            ),
            "TMCRA_SLOW_GRAPH_PROMPT_ADAPTER": str(
                os.getenv("TMCRA_SLOW_GRAPH_PROMPT_ADAPTER")
                or LOCAL_QWEN_SLOW_PROMPT_ADAPTER
            ).strip(),
            "TMCRA_SLOW_GRAPH_MAX_TOKENS": str(
                os.getenv("TMCRA_SLOW_GRAPH_MAX_TOKENS") or max_tokens
            ).strip(),
        }
    elif slow_provider == DEEPSEEK_PROVIDER:
        slow = {"TMCRA_SLOW_GRAPH_PROVIDER": DEEPSEEK_PROVIDER}
    else:
        raise RuntimeError(
            f"unsupported TMCRA_SLOW_GRAPH_PROVIDER: {slow_provider}"
        )
    aliases = {
        **primary,
        **planner,
        **reviewer,
        **slow,
        "TMCRA_WRITER_MAX_TOKENS": max_tokens,
        "TMCRA_DEEPSEEK_FLASH_BASE_URL": deepseek_base_url,
        "TMCRA_DEEPSEEK_FLASH_KEY_POOL": deepseek_key_pool,
        "TMCRA_DEEPSEEK_FLASH_MAX_TOKENS": max_tokens,
        "TMCRA_DEEPSEEK_PRO_BASE_URL": deepseek_base_url,
        "TMCRA_DEEPSEEK_PRO_KEY_POOL": deepseek_key_pool,
        "TMCRA_DEEPSEEK_PRO_MAX_TOKENS": max_tokens,
        "TMCRA_RECALL_PLANNER_MAX_TOKENS": os.getenv(
            "TMCRA_RECALL_PLANNER_MAX_TOKENS", "512"
        ),
        "TMCRA_RECALL_PLANNER_TIMEOUT_SECONDS": os.getenv(
            "TMCRA_RECALL_PLANNER_TIMEOUT_SECONDS", "60"
        ),
    }
    os.environ.update(aliases)


def _validate_startup(settings: ServiceSettings) -> None:
    try:
        settings.validate()
    except Exception as exc:
        raise RuntimeError(f"service startup validation failed: {exc}") from exc

    try:
        verify_shared_core(settings.v4_root)
    except SharedCoreVerificationError as exc:
        raise RuntimeError(
            f"service startup validation failed: {exc}"
        ) from exc

    try:
        bind_address = ipaddress.ip_address(settings.bind_host)
    except ValueError as exc:
        raise RuntimeError(
            "service startup validation failed: TMCRA_SERVICE_BIND_HOST must be "
            "a loopback address or an explicitly proxied wildcard address"
        ) from exc
    proxy_mode = os.getenv("TMCRA_SERVICE_TLS_PROXY_MODE", "").strip().lower()
    proxied_wildcard = bind_address.is_unspecified and proxy_mode in {
        "trusted_proxy",
        "gpuhome",  # Backward-compatible alias for earlier deployment files.
    }
    if not bind_address.is_loopback and not proxied_wildcard:
        raise RuntimeError(
            "service startup validation failed: a non-loopback bind requires "
            "TMCRA_SERVICE_TLS_PROXY_MODE=trusted_proxy"
        )

    parsed_url = urlparse(settings.public_base_url)
    from tmcra_local_only import enabled, loopback_url
    if enabled():
        if not bind_address.is_loopback:
            raise RuntimeError("full-local service must bind loopback")
        loopback_url(settings.public_base_url, port=settings.bind_port, path="")
    elif parsed_url.scheme.lower() != "https" or not parsed_url.netloc:
        raise RuntimeError(
            "service startup validation failed: TMCRA_SERVICE_PUBLIC_BASE_URL "
            "must be an HTTPS URL served by a trusted TLS reverse proxy"
        )

    os.environ["TMCRA_LEARNED_GRAPH_ENABLED"] = (
        "1" if settings.learned_graph_enabled else "0"
    )
    if settings.learned_graph_enabled:
        for name, path in (
            ("node_model", settings.node_model),
            ("path_model", settings.path_model),
        ):
            if not path.is_file():
                raise RuntimeError(
                    f"service startup validation failed: {name} model is not a file: {path}"
                )

        # The native harness reads these names when learned graph retrieval is enabled.
        os.environ.update(
            {
                "TMCRA_NODE_MODEL": str(settings.node_model),
                "TMCRA_PATH_MODEL": str(settings.path_model),
                "TMCRA_NODE_MODEL_PATH": str(settings.node_model),
                "TMCRA_PATH_MODEL_PATH": str(settings.path_model),
            }
        )
    else:
        os.environ["TMCRA_RETRIEVAL_MODE"] = "dense_fast"
        os.environ["TMCRA_FAST_PATH"] = "dense"
        for name in (
            "TMCRA_NODE_MODEL",
            "TMCRA_PATH_MODEL",
            "TMCRA_NODE_MODEL_PATH",
            "TMCRA_PATH_MODEL_PATH",
        ):
            os.environ.pop(name, None)


def main() -> int:
    os.umask(0o077)
    writer_env = os.getenv("TMCRA_WRITER_ENV", DEFAULT_WRITER_ENV)
    _load_shell_environment(writer_env)
    _configure_writer_aliases()
    from tmcra_local_only import enabled, install_network_guard
    if enabled():
        install_network_guard()
    settings = ServiceSettings.from_env()
    _validate_startup(settings)
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.bind_host,
        port=settings.bind_port,
        workers=1,
        # Public URLs are constructed from validated settings, so untrusted
        # forwarding headers are unnecessary even behind the platform proxy.
        proxy_headers=False,
        # The JSONL journal carries bounded request IDs, latency, tenant
        # attribution, and error status without raw paths or payloads.
        access_log=not settings.api_access_log_enabled,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
