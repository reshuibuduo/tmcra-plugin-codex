"""Validated request attribution for the commercial usage ledger.

Attribution is deliberately kept separate from memory payload metadata.  A
client may describe arbitrary source material inside an ingest body; those
descriptions must never silently become billing identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


CLIENT_PLATFORM_HEADER = "X-TMCRA-Client-Platform"
INTEGRATION_ID_HEADER = "X-TMCRA-Integration-ID"
AGENT_ID_HEADER = "X-TMCRA-Agent-ID"

ATTRIBUTION_SOURCES = frozenset(
    {"trusted_proxy", "client_reported", "system_derived", "unattributed"}
)
CLIENT_PLATFORMS = frozenset(
    {
        "claude_code",
        "codex",
        "deepseek_harness",
        "hermes",
        "langgraph",
        "mcp",
        "openai_agents",
        "openclaw",
        "python",
        "rest",
        "typescript",
        "vercel_ai_sdk",
        "zcode",
        "tmcra_internal",
    }
)
REQUEST_CLIENT_PLATFORMS = CLIENT_PLATFORMS - {"tmcra_internal"}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,199}$")
_INTEGRATION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TRUSTED_INTEGRATION_RE = re.compile(r"^int_[a-f0-9]{32}$")


def _none_or_text(value: Any) -> str | None:
    if value is None:
        return None
    clean = str(value).strip()
    return clean or None


def _optional_identifier(
    value: Any, pattern: re.Pattern[str], field: str
) -> str | None:
    clean = _none_or_text(value)
    if clean is not None and not pattern.fullmatch(clean):
        raise UsageAttributionError(f"invalid {field}")
    return clean


class UsageAttributionError(ValueError):
    """Raised when dedicated ledger-attribution headers are malformed."""


@dataclass(frozen=True)
class UsageAttribution:
    client_platform: str = "unattributed"
    integration_id: str | None = None
    agent_id: str | None = None
    attribution_source: str = "unattributed"

    def __post_init__(self) -> None:
        platform = str(self.client_platform or "unattributed").strip().lower()
        source = str(self.attribution_source or "unattributed").strip().lower()
        integration_id = _optional_identifier(
            self.integration_id, _INTEGRATION_RE, "integration_id"
        )
        agent_id = _optional_identifier(self.agent_id, _IDENTIFIER_RE, "agent_id")
        if platform != "unattributed" and platform not in CLIENT_PLATFORMS:
            raise UsageAttributionError("unsupported client platform")
        if source not in ATTRIBUTION_SOURCES:
            raise UsageAttributionError("unsupported attribution source")
        if platform == "unattributed":
            if integration_id is not None or agent_id is not None:
                raise UsageAttributionError(
                    "integration_id and agent_id require a client platform"
                )
            if source != "unattributed":
                raise UsageAttributionError(
                    "unattributed platform requires unattributed source"
                )
        elif source == "unattributed":
            raise UsageAttributionError(
                "attributed platform requires an attribution source"
            )
        object.__setattr__(self, "client_platform", platform)
        object.__setattr__(self, "integration_id", integration_id)
        object.__setattr__(self, "agent_id", agent_id)
        object.__setattr__(self, "attribution_source", source)

    def as_dict(self) -> dict[str, str | None]:
        return {
            "client_platform": self.client_platform,
            "integration_id": self.integration_id,
            "agent_id": self.agent_id,
            "attribution_source": self.attribution_source,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "UsageAttribution":
        source = dict(value or {})
        return cls(
            client_platform=str(source.get("client_platform") or "unattributed"),
            integration_id=_none_or_text(source.get("integration_id")),
            agent_id=_none_or_text(source.get("agent_id")),
            attribution_source=str(
                source.get("attribution_source") or "unattributed"
            ),
        )


UNATTRIBUTED = UsageAttribution()
SYSTEM_MAINTENANCE = UsageAttribution(
    client_platform="tmcra_internal",
    agent_id="memory-maintenance",
    attribution_source="system_derived",
)


def resolve_request_attribution(
    context: Any,
    headers: Mapping[str, str],
) -> UsageAttribution:
    """Resolve immutable ledger identity from dedicated request headers.

    A platform supplied by a normal SDK or scoped token is retained as
    ``client_reported``.  Only the server-side personal BFF can obtain
    ``trusted_proxy``: it must authenticate with a managing API key, act on
    behalf of a subject, and supply a registry-shaped integration ID.
    """

    platform = str(headers.get(CLIENT_PLATFORM_HEADER, "") or "").strip().lower()
    integration_id = str(headers.get(INTEGRATION_ID_HEADER, "") or "").strip()
    agent_id = str(headers.get(AGENT_ID_HEADER, "") or "").strip()
    if not platform and not integration_id and not agent_id:
        return UNATTRIBUTED
    if platform not in REQUEST_CLIENT_PLATFORMS:
        raise UsageAttributionError("unsupported client platform")
    trusted_proxy = bool(
        getattr(context, "credential_type", "") == "api_key"
        and "tokens:manage" in getattr(context, "scopes", frozenset())
        and str(getattr(context, "subject", "") or "").strip()
        and _TRUSTED_INTEGRATION_RE.fullmatch(integration_id)
    )
    if trusted_proxy:
        # Trusted server-side attribution is bound to one row in the personal
        # integration registry. Registry IDs never coexist with a missing
        # platform, and the platform is independently checked by the BFF.
        source = "trusted_proxy"
    else:
        source = "client_reported"
    return UsageAttribution(
        client_platform=platform,
        integration_id=integration_id or None,
        agent_id=agent_id or None,
        attribution_source=source,
    )
