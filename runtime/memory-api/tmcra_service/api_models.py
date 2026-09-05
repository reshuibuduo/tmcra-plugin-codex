from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .actor_provenance import ActorProvenanceError, normalize_message_actor_metadata


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MemoryMessage(StrictModel):
    message_id: str = Field(min_length=1, max_length=200)
    role: Literal["user", "assistant", "system", "tool"]
    content: str = Field(min_length=1, max_length=200_000)
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def bounded_actor_metadata(self) -> "MemoryMessage":
        if len(self.metadata) > 64:
            raise ValueError("message metadata may contain at most 64 fields")
        if any(not isinstance(key, str) or not key.strip() for key in self.metadata):
            raise ValueError("message metadata keys must be non-empty strings")
        try:
            encoded = json.dumps(
                self.metadata,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("message metadata must contain JSON values") from exc
        if len(encoded) > 8_192:
            raise ValueError("message metadata must be at most 8192 UTF-8 bytes")
        try:
            normalize_message_actor_metadata(self.role, self.metadata)
        except ActorProvenanceError as exc:
            raise ValueError(str(exc)) from exc
        return self


class IngestRequest(StrictModel):
    session_id: str = Field(min_length=1, max_length=200)
    messages: list[MemoryMessage] = Field(min_length=1, max_length=1000)
    consistency: Literal["eventual", "read_your_writes"] = "eventual"
    slow_policy: Literal["auto", "deferred", "force"] = "auto"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def unique_message_ids(cls, value: list[MemoryMessage]) -> list[MemoryMessage]:
        identifiers = [item.message_id for item in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("message_id values must be unique within one request")
        return value


class BulkIngestItem(IngestRequest):
    idempotency_key: str = Field(min_length=8, max_length=200)


class BulkIngestRequest(StrictModel):
    items: list[BulkIngestItem] = Field(min_length=1, max_length=100)

    @field_validator("items")
    @classmethod
    def validate_batch(cls, value: list[BulkIngestItem]) -> list[BulkIngestItem]:
        keys = [item.idempotency_key for item in value]
        if len(keys) != len(set(keys)):
            raise ValueError("idempotency_key values must be unique within one batch")
        if sum(len(item.messages) for item in value) > 5000:
            raise ValueError("one batch may contain at most 5000 messages")
        return value


class RecallRequest(StrictModel):
    query: str = Field(min_length=1, max_length=100_000)
    query_time: datetime | None = None
    evidence_mode: Literal["raw", "auto", "compiled"] = "auto"
    recall_profile: Literal["quality", "interactive"] = "quality"
    response_projection: Literal["full", "prompt_only"] = "full"
    max_windows: Literal[8] = 8
    wait_for_job_id: str | None = Field(default=None, max_length=100)
    debug: bool = False


class TurnRequest(StrictModel):
    session_id: str = Field(min_length=1, max_length=200)
    user_message: MemoryMessage
    query: str | None = Field(default=None, max_length=100_000)
    evidence_mode: Literal["raw", "auto", "compiled"] = "auto"
    consistency: Literal["eventual", "read_your_writes"] = "eventual"
    max_windows: Literal[8] = 8


class ConsistencyContract(StrictModel):
    mode: Literal["eventual", "read_your_writes"]
    visible_after_job_id: str
    recall_wait_for_job_id: str | None = None


ReceiptStatus = Literal[
    "submitted",
    "pending",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]
TerminalReceiptStatus = Literal["succeeded", "failed", "cancelled"]


class WatermarkView(StrictModel):
    """Searchability watermarks shared by lifecycle receipt implementations."""

    source_event_seq: int | None = Field(default=None, ge=0)
    promoted_event_seq: int | None = Field(default=None, ge=0)
    indexed_event_seq: int | None = Field(default=None, ge=0)
    source_raw_token_estimate: int | None = Field(default=None, ge=0)
    available: bool = False


class RecallReceipt(StrictModel):
    """Client-side receipt derived from one successful RecallResponse."""

    query_id: str
    scope_name: str
    index_job_id: str
    evidence_hash: str | None = None
    submitted_status: Literal["completed"] = "completed"
    final_status: Literal["completed"] = "completed"
    submitted: Literal[True] = True
    final: Literal[True] = True
    status_url: str | None = None
    watermarks: WatermarkView


class IngestReceipt(StrictModel):
    """Client-side receipt for submitted and optionally terminal ingest."""

    scope_name: str
    message_ids: list[str] = Field(min_length=1)
    idempotency_key: str
    job_id: str | None = None
    submitted_status: Literal["submitted"] = "submitted"
    observed_status: str
    final_status: TerminalReceiptStatus | None = None
    submitted: Literal[True] = True
    final: bool = False
    status_url: str | None = None
    watermarks: WatermarkView
    error: dict[str, Any] | None = None


class LifecycleTurnReceipt(StrictModel):
    """Unified recall -> inject -> ingest receipt used by client adapters."""

    session_id: str
    idempotency_key: str
    recall_receipts: list[RecallReceipt] = Field(default_factory=list)
    ingest_receipt: IngestReceipt
    message_ids: list[str] = Field(min_length=1)
    query_ids: list[str] = Field(default_factory=list)
    evidence_hashes: list[str] = Field(default_factory=list)
    submitted_status: Literal["submitted"] = "submitted"
    final_status: TerminalReceiptStatus | None = None
    job_id: str | None = None
    status_url: str | None = None
    submitted: Literal[True] = True
    final: bool = False
    watermarks: WatermarkView


class ReceiptContractAnchor(StrictModel):
    """OpenAPI-only anchor that keeps reusable receipt schemas exported.

    This field is excluded from runtime response serialization. The service
    deliberately keeps the existing recall/ingest/job endpoints and client
    adapters derive receipts from those responses.
    """

    recall: RecallReceipt | None = None
    ingest: IngestReceipt | None = None
    lifecycle: LifecycleTurnReceipt | None = None


def _add_receipt_contract_extension(schema: dict[str, Any]) -> None:
    """Publish client-derived receipt schemas without adding a runtime field."""

    schema["x-tmcra-receipt-contract"] = {
        "schema_version": "tmcra.receipts.v1",
        "runtime_response_fields": {
            "recall": [
                "query_id",
                "scope_name",
                "index_job_id",
                "evidence_route",
                "evidence",
                "prompt_evidence",
            ],
            "ingest": ["JobView"],
            "job_terminal": ["JobView.status", "JobView.result", "JobView.error"],
        },
        "derived_receipts": {
            "recall": "RecallReceipt",
            "ingest": "IngestReceipt",
            "lifecycle": "LifecycleTurnReceipt",
        },
        "client_projections": {
            "python": "snake_case fields; flat watermark values map to WatermarkView",
            "typescript": "camelCase fields; turnIdempotencyKey maps to idempotency_key",
            "mcp": "adds schema_version and receipt_type to the validated receipt envelope",
        },
        "schemas": deepcopy(_RECEIPT_CONTRACT_SCHEMAS),
        "protocol": {
            "order": ["recall", "inject", "ingest", "job_terminal"],
            "inject_source": "RecallResponse.prompt_evidence.content",
            "terminal_statuses": ["succeeded", "failed", "cancelled"],
            "submitted_is_terminal": False,
            "strict_recall": (
                "stop before answer/write when recall is unavailable or invalid"
            ),
            "degraded_recall": (
                "caller may continue only when explicitly configured; never claim injection"
            ),
        },
    }


_RECEIPT_WATERMARK_SCHEMA = {
    "additionalProperties": False,
    "description": "Searchability watermarks shared by lifecycle receipt implementations.",
    "properties": {
        "source_event_seq": {"type": ["integer", "null"], "minimum": 0},
        "promoted_event_seq": {"type": ["integer", "null"], "minimum": 0},
        "indexed_event_seq": {"type": ["integer", "null"], "minimum": 0},
        "source_raw_token_estimate": {
            "type": ["integer", "null"],
            "minimum": 0,
        },
        "available": {"type": "boolean", "default": False},
    },
    "required": ["source_event_seq", "promoted_event_seq", "indexed_event_seq", "source_raw_token_estimate", "available"],
    "title": "WatermarkView",
    "type": "object",
}

_RECEIPT_STATUS_SCHEMA = {
    "enum": ["submitted", "pending", "running", "succeeded", "failed", "cancelled"],
    "type": "string",
}

_RECEIPT_TERMINAL_STATUS_SCHEMA = {
    "enum": ["succeeded", "failed", "cancelled"],
    "type": "string",
}

_RECEIPT_RECALL_SCHEMA = {
    "additionalProperties": False,
    "description": "Client-side receipt derived from one successful RecallResponse.",
    "properties": {
        "query_id": {"type": "string"},
        "scope_name": {"type": "string"},
        "index_job_id": {"type": "string"},
        "evidence_hash": {"type": ["string", "null"]},
        "submitted_status": {"const": "completed", "default": "completed"},
        "final_status": {"const": "completed", "default": "completed"},
        "submitted": {"const": True, "default": True},
        "final": {"const": True, "default": True},
        "status_url": {"type": ["string", "null"]},
        "watermarks": deepcopy(_RECEIPT_WATERMARK_SCHEMA),
    },
    "required": [
        "query_id",
        "scope_name",
        "index_job_id",
        "evidence_hash",
        "submitted_status",
        "final_status",
        "submitted",
        "final",
        "status_url",
        "watermarks",
    ],
    "title": "RecallReceipt",
    "type": "object",
}

_RECEIPT_INGEST_SCHEMA = {
    "additionalProperties": False,
    "description": "Client-side receipt for submitted and optionally terminal ingest.",
    "properties": {
        "scope_name": {"type": "string"},
        "message_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "idempotency_key": {"type": "string"},
        "job_id": {"type": ["string", "null"]},
        "submitted_status": {"const": "submitted", "default": "submitted"},
        "observed_status": {"type": "string"},
        "final_status": deepcopy(_RECEIPT_TERMINAL_STATUS_SCHEMA) | {"type": "null"},
        "submitted": {"const": True, "default": True},
        "final": {"type": "boolean", "default": False},
        "status_url": {"type": ["string", "null"]},
        "watermarks": deepcopy(_RECEIPT_WATERMARK_SCHEMA),
        "error": {"type": ["object", "null"], "additionalProperties": True},
    },
    "required": [
        "scope_name",
        "message_ids",
        "idempotency_key",
        "job_id",
        "submitted_status",
        "observed_status",
        "final_status",
        "submitted",
        "final",
        "status_url",
        "watermarks",
        "error",
    ],
    "title": "IngestReceipt",
    "type": "object",
}

_RECEIPT_LIFECYCLE_SCHEMA = {
    "additionalProperties": False,
    "description": "Unified recall -> inject -> ingest receipt used by client adapters.",
    "properties": {
        "session_id": {"type": "string"},
        "idempotency_key": {"type": "string"},
        "recall_receipts": {
            "type": "array",
            "items": deepcopy(_RECEIPT_RECALL_SCHEMA),
        },
        "ingest_receipt": deepcopy(_RECEIPT_INGEST_SCHEMA),
        "message_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "query_ids": {"type": "array", "items": {"type": "string"}},
        "evidence_hashes": {"type": "array", "items": {"type": "string"}},
        "submitted_status": {"const": "submitted", "default": "submitted"},
        "final_status": deepcopy(_RECEIPT_TERMINAL_STATUS_SCHEMA) | {"type": "null"},
        "job_id": {"type": ["string", "null"]},
        "status_url": {"type": ["string", "null"]},
        "submitted": {"const": True, "default": True},
        "final": {"type": "boolean", "default": False},
        "watermarks": deepcopy(_RECEIPT_WATERMARK_SCHEMA),
    },
    "required": [
        "session_id",
        "idempotency_key",
        "recall_receipts",
        "ingest_receipt",
        "message_ids",
        "query_ids",
        "evidence_hashes",
        "submitted_status",
        "final_status",
        "job_id",
        "status_url",
        "submitted",
        "final",
        "watermarks",
    ],
    "title": "LifecycleTurnReceipt",
    "type": "object",
}

_RECEIPT_CONTRACT_SCHEMAS = {
    "WatermarkView": _RECEIPT_WATERMARK_SCHEMA,
    "RecallReceipt": _RECEIPT_RECALL_SCHEMA,
    "IngestReceipt": _RECEIPT_INGEST_SCHEMA,
    "LifecycleTurnReceipt": _RECEIPT_LIFECYCLE_SCHEMA,
}


class JobView(StrictModel):
    job_id: str
    tenant_id: str
    scope_name: str
    job_type: str
    status: str
    attempts: int
    created_at: float
    updated_at: float
    started_at: float | None = None
    finished_at: float | None = None
    heartbeat_at: float | None = None
    lease_expires_at: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    status_url: str
    idempotent_replay: bool | None = None
    idempotent_retry: bool | None = None
    resume_mode: str | None = None
    consistency_contract: ConsistencyContract | None = None


class MemoryDeleteRequest(StrictModel):
    memory_ids: list[str] = Field(min_length=1, max_length=100)

    @field_validator("memory_ids")
    @classmethod
    def unique_bounded_memory_ids(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 512 for item in value):
            raise ValueError("memory IDs must be 1-512 characters")
        if len(value) != len(set(value)):
            raise ValueError("memory IDs must be unique")
        return value


class MessageDeleteRequest(StrictModel):
    message_ids: list[str] = Field(min_length=1, max_length=100)

    @field_validator("message_ids")
    @classmethod
    def unique_bounded_message_ids(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 200 for item in value):
            raise ValueError("message IDs must be 1-200 characters")
        if len(value) != len(set(value)):
            raise ValueError("message IDs must be unique")
        return value


class ContentDeletionJobView(JobView):
    deletion_id: str
    deletion_status_url: str


class ContentDeletionView(StrictModel):
    deletion_id: str
    tenant_id: str
    scope_name: str
    mode: Literal["memory_ids", "session"]
    target_count: int = Field(ge=1)
    state: Literal["requested", "purging", "reindexing", "completed", "failed"]
    job_id: str | None = None
    job_status_url: str | None = None
    result: dict[str, Any] | None = None
    error_code: str | None = None
    created_at: float
    updated_at: float
    completed_at: float | None = None


class BulkIngestResponse(StrictModel):
    scope_name: str
    jobs: list[JobView]


class ScopeTokenCreateRequest(StrictModel):
    label: str = Field(min_length=1, max_length=120)
    subject: str | None = Field(default=None, min_length=1, max_length=200)
    permissions: list[str] = Field(min_length=1, max_length=10)
    scope_names: list[str] = Field(default_factory=list, max_length=100)
    scope_prefixes: list[str] = Field(default_factory=list, max_length=100)
    expires_in_seconds: int = Field(ge=300, le=31_622_400)
    provisional_delivery_seconds: int | None = Field(default=None, ge=60, le=900)

    @field_validator("permissions", "scope_names", "scope_prefixes")
    @classmethod
    def unique_strings(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("values must be unique")
        return value

    @field_validator("scope_names", "scope_prefixes")
    @classmethod
    def valid_scope_selectors(cls, value: list[str]) -> list[str]:
        pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
        if any(not pattern.fullmatch(item) for item in value):
            raise ValueError("scope selectors must use valid TMCRA scope characters")
        return value

    @model_validator(mode="after")
    def at_least_one_scope_selector(self) -> "ScopeTokenCreateRequest":
        if not self.scope_names and not self.scope_prefixes:
            raise ValueError("at least one scope name or scope prefix is required")
        return self


class ScopeTokenView(StrictModel):
    token_id: str
    tenant_id: str
    permissions: list[str]
    scope_names: list[str]
    scope_prefixes: list[str]
    label: str
    subject: str | None = None
    created_by_key_id: str | None = None
    created_at: float
    expires_at: float
    revoked_at: float | None = None
    last_used_at: float | None = None


class IssuedScopeTokenView(ScopeTokenView):
    access_token: str


class SessionServiceView(StrictModel):
    name: Literal["tmcra-memory"] = "tmcra-memory"
    version: str
    capabilities: list[str]


class SessionScopeRestrictionsView(StrictModel):
    unrestricted: bool
    names: list[str]
    prefixes: list[str]


class SessionCredentialView(StrictModel):
    type: Literal["api_key", "scope_token"]
    tenant_id: str
    principal: str
    subject: str | None = None
    permissions: list[str]
    scope_restrictions: SessionScopeRestrictionsView
    expires_at: float | None = None


class AuthenticatedSessionView(StrictModel):
    ok: Literal[True] = True
    authenticated: Literal[True] = True
    service: SessionServiceView
    credential: SessionCredentialView


class ScopeRecoveryView(StrictModel):
    state: Literal["ready", "recovering", "attention_required"]
    phase: Literal[
        "ready",
        "waiting",
        "auditing",
        "repairing",
        "consolidating",
        "indexing",
        "verifying",
        "manual_review",
    ]
    progress_percent: int = Field(ge=0, le=100)
    completed_items: int = Field(ge=0)
    total_items: int = Field(ge=0)
    pending_items: int = Field(ge=0)
    recovery_attempts: int = Field(ge=0)
    automatic: bool
    reads_available: bool
    writes_available: bool
    requires_support: bool
    started_at: float | None = None
    updated_at: float | None = None
    next_attempt_at: float | None = None


class ScopeCatalogView(StrictModel):
    scope_name: str
    created_at: float
    last_seen_at: float
    last_ingest_at: float | None = None
    last_recall_at: float | None = None
    session_count: int
    ingest_request_count: int
    recall_request_count: int
    message_count: int
    recovery: ScopeRecoveryView | None = None


class ScopeSessionView(StrictModel):
    session_id: str
    created_at: float
    last_ingest_at: float
    ingest_request_count: int
    message_count: int


class ScopeSummaryView(StrictModel):
    scope: ScopeCatalogView
    sessions: list[ScopeSessionView]
    recovery: ScopeRecoveryView | None = None


class QuotaMetricView(StrictModel):
    used: int
    limit: int | None = None
    remaining: int | None = None


class BillingQuotaGroupView(StrictModel):
    group_id: str
    display_name: str
    status: Literal["active", "suspended", "cancelled"]
    period_id: str
    period_status: Literal["scheduled", "active", "expired", "cancelled"]
    billing_interval: Literal["monthly", "yearly", "custom"]
    starts_at: float
    ends_at: float
    max_members: int = Field(ge=1)
    currency: str
    price_minor_units: int | None = Field(default=None, ge=0)


class QuotaView(StrictModel):
    tenant_id: str
    principal: str
    plan: str
    plan_version: str | None = None
    billing_group: BillingQuotaGroupView | None = None
    ingest_raw_tokens: QuotaMetricView
    recall_requests: QuotaMetricView
    member_usage: dict[str, dict[str, int]] = Field(default_factory=dict)


class BillingProfileView(StrictModel):
    tenant_id: str
    subject: str | None = None
    consumer_principal: str
    quota_principal: str
    membership: dict[str, Any] | None = None
    quota: QuotaView


class BillingPlanVersionUpsertRequest(StrictModel):
    display_name: str = Field(min_length=1, max_length=120)
    billing_interval: Literal["monthly", "yearly", "custom"]
    ingest_raw_tokens: int | None = Field(ge=0)
    recall_requests: int | None = Field(ge=0)
    max_members: int = Field(ge=1, le=100_000)
    currency: str = Field(pattern=r"^[A-Za-z]{3}$")
    price_minor_units: int | None = Field(default=None, ge=0)
    entitlements: dict[str, Any] = Field(default_factory=dict)


class BillingPlanVersionView(StrictModel):
    plan_code: str
    plan_version: str
    display_name: str
    status: Literal["active", "retired"]
    billing_interval: Literal["monthly", "yearly", "custom"]
    ingest_raw_token_limit: int | None = Field(default=None, ge=0)
    recall_request_limit: int | None = Field(default=None, ge=0)
    max_members: int = Field(ge=1)
    currency: str
    price_minor_units: int | None = Field(default=None, ge=0)
    entitlements: dict[str, Any]
    created_by: str
    created_at: float
    updated_at: float


class BillingGroupCreateRequest(StrictModel):
    tenant_id: str = Field(min_length=1, max_length=200)
    group_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    display_name: str = Field(min_length=1, max_length=120)
    owner_subject: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,199}$")
    plan_code: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    plan_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    starts_at: float = Field(ge=0)
    ends_at: float = Field(gt=0)


class BillingGroupMemberRequest(StrictModel):
    subject: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,199}$")
    role: Literal["admin", "member"] = "member"


class BillingPeriodChangeRequest(StrictModel):
    plan_code: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    plan_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    starts_at: float = Field(ge=0)
    ends_at: float = Field(gt=0)


class BillingGroupStatusRequest(StrictModel):
    status: Literal["active", "suspended", "cancelled"]


class UsageSourceView(StrictModel):
    scope_count: int = Field(ge=0)
    ingested_raw_token_estimate: int = Field(ge=0)
    ingested_user_turns: int = Field(ge=0)
    source_event_count: int = Field(ge=0)


class UsageCallTotalsView(StrictModel):
    registered_call_count: int = Field(ge=0)
    completed_call_count: int = Field(ge=0)
    failed_call_count: int = Field(ge=0)
    unknown_call_count: int = Field(ge=0)
    in_flight_call_count: int = Field(ge=0)
    unpriced_completed_call_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cache_hit_tokens: int = Field(ge=0)
    cache_miss_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    known_cost_micro_cny: int = Field(ge=0)


class UsageStageView(StrictModel):
    registered_call_count: int = Field(ge=0)
    completed_call_count: int = Field(ge=0)
    unknown_or_unpriced_call_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    known_cost_micro_cny: int = Field(ge=0)


class UsageAttributionCoverageView(StrictModel):
    provider_call_count: int = Field(ge=0)
    usage_event_count: int = Field(ge=0)
    ingest_raw_tokens: int = Field(ge=0)
    recall_requests: int = Field(ge=0)
    known_cost_micro_cny: int = Field(ge=0)


class UsageCostBucketView(UsageCallTotalsView):
    key: str
    ingest_raw_tokens: int = Field(ge=0)
    recall_requests: int = Field(ge=0)
    known_cost_cny: float = Field(ge=0)


class UsageCostsView(StrictModel):
    tenant_id: str
    scope_name: str | None = None
    scope_prefix: str | None = None
    from_timestamp: float | None = Field(default=None, ge=0)
    to_timestamp: float | None = Field(default=None, ge=0)
    currency: str
    ledger_coverage: str
    source_ledger_coverage: str
    complete_for_registered_calls: bool
    source: UsageSourceView
    calls: UsageCallTotalsView
    known_cost_cny: float = Field(ge=0)
    known_model_api_cny_per_million_ingested_raw_tokens: float | None = Field(
        default=None, ge=0
    )
    uncertain_cost_call_count: int = Field(ge=0)
    by_stage: dict[str, UsageStageView]
    quota_events: dict[str, int]
    quota_event_scope_coverage: dict[str, str]
    attribution_coverage: dict[str, UsageAttributionCoverageView]
    group_by: Literal[
        "day",
        "scope",
        "stage",
        "operation",
        "provider",
        "model",
        "platform",
        "integration",
        "agent",
        "attribution_source",
    ] | None = None
    buckets: list[UsageCostBucketView]


class ProviderCallReportRequest(StrictModel):
    """Server-to-server receipt for an end-user answer-model call.

    The payload deliberately contains accounting metadata only. Prompts,
    attachments, recalled evidence, and model responses are never accepted by
    this endpoint.
    """

    call_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,199}$")
    provider: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    model: str = Field(min_length=1, max_length=160)
    operation: str = Field(
        default="chat_answer",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$",
    )
    status: Literal["completed", "failed", "unknown"]
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cache_hit_tokens: int | None = Field(default=None, ge=0)
    error_code: str | None = Field(default=None, max_length=200)
    request_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    started_at: float | None = Field(default=None, ge=0)
    finished_at: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_usage(self) -> "ProviderCallReportRequest":
        supplied = (self.input_tokens, self.output_tokens, self.total_tokens)
        if any(value is not None for value in supplied) and (
            self.input_tokens is None or self.output_tokens is None
        ):
            raise ValueError(
                "input_tokens and output_tokens are both required when usage is reported"
            )
        if (
            self.total_tokens is not None
            and self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens < self.input_tokens + self.output_tokens
        ):
            raise ValueError("total_tokens is smaller than input plus output")
        if (
            self.cache_hit_tokens is not None
            and self.input_tokens is not None
            and self.cache_hit_tokens > self.input_tokens
        ):
            raise ValueError("cache_hit_tokens cannot exceed input_tokens")
        if self.finished_at is not None and self.started_at is not None:
            if self.finished_at < self.started_at:
                raise ValueError("finished_at cannot precede started_at")
        if self.status == "completed" and self.error_code is not None:
            raise ValueError("completed provider calls cannot include error_code")
        return self


class ProviderCallReportView(StrictModel):
    call_id: str
    scope_name: str
    provider: str
    model: str
    operation: str
    status: Literal["completed", "failed", "unknown"]
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cache_hit_tokens: int | None = Field(default=None, ge=0)
    cache_miss_tokens: int | None = Field(default=None, ge=0)
    usage_state: Literal["missing", "complete"]
    cost_micro_cny: int | None = Field(default=None, ge=0)
    price_version: str | None = None
    idempotent_replay: bool


class UserProviderTaskClaimRequest(StrictModel):
    stage: Literal["writer", "organizer"]


class UserProviderTaskLeaseView(StrictModel):
    schema_version: Literal["tmcra.user-provider-task.1"]
    task_id: str
    stage: Literal["writer", "organizer"]
    operation: str
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request: dict[str, Any]
    lease_token: str
    lease_expires_at: float = Field(ge=0)


class UserProviderTaskClaimView(StrictModel):
    task: UserProviderTaskLeaseView | None = None
    retry_after_seconds: float = Field(ge=0)


class UserProviderTaskLeaseRequest(StrictModel):
    lease_token: str = Field(min_length=32, max_length=256)


class UserProviderUsage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cache_hit_tokens: int | None = Field(default=None, ge=0)
    cache_miss_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_usage(self) -> "UserProviderUsage":
        if self.input_tokens is None or self.output_tokens is None:
            if any(
                value is not None
                for value in (
                    self.input_tokens,
                    self.output_tokens,
                    self.total_tokens,
                    self.cache_hit_tokens,
                    self.cache_miss_tokens,
                )
            ):
                raise ValueError(
                    "input_tokens and output_tokens are required with provider usage"
                )
            return self
        total = self.total_tokens
        if total is not None and total < self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens is smaller than input plus output")
        hit = self.cache_hit_tokens
        miss = self.cache_miss_tokens
        if hit is not None and hit > self.input_tokens:
            raise ValueError("cache_hit_tokens cannot exceed input_tokens")
        if miss is not None and miss > self.input_tokens:
            raise ValueError("cache_miss_tokens cannot exceed input_tokens")
        if hit is not None and miss is not None and hit + miss != self.input_tokens:
            raise ValueError("cache hit and miss tokens must equal input_tokens")
        return self


class UserProviderTaskCompleteRequest(UserProviderTaskLeaseRequest):
    provider: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    model: str = Field(min_length=1, max_length=160)
    output: dict[str, Any]
    usage: UserProviderUsage | None = None
    provider_request_id: str | None = Field(default=None, max_length=200)


class UserProviderTaskFailRequest(UserProviderTaskLeaseRequest):
    provider: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    model: str = Field(min_length=1, max_length=160)
    outcome: Literal["failed", "unknown"]
    error_code: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$",
    )


class UserProviderTaskStatusView(StrictModel):
    task_id: str
    state: Literal[
        "queued", "leased", "running", "completed", "failed", "unknown"
    ]
    lease_expires_at: float | None = Field(default=None, ge=0)
    idempotent_replay: bool = False


class EntitlementUpdateRequest(StrictModel):
    ingest_raw_tokens: int | None = Field(ge=0)
    recall_requests: int | None = Field(ge=0)


class RetentionPolicyRequest(StrictModel):
    enabled: bool
    inactive_days: int = Field(ge=1, le=3650)


class RetentionPolicyView(StrictModel):
    scope_name: str
    enabled: bool
    inactive_days: int
    created_at: float | None = None
    updated_at: float | None = None


class FeedbackRequest(StrictModel):
    query_id: str | None = Field(default=None, max_length=200)
    rating: Literal["helpful", "incorrect", "stale", "unsafe", "missing"]
    memory_ids: list[str] = Field(default_factory=list, max_length=100)
    comment: str | None = Field(default=None, max_length=4000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    action: Literal["note", "ignore", "correct", "restore"] = "note"
    replacement: str | None = Field(default=None, min_length=1, max_length=4000)

    @model_validator(mode="after")
    def targeted_action(self) -> "FeedbackRequest":
        if self.action != "note" and (not self.memory_ids or any(not item.strip() for item in self.memory_ids)):
            raise ValueError("memory_ids are required for an effective feedback action")
        if self.action == "correct" and not self.replacement:
            raise ValueError("replacement is required for correct")
        if self.action != "correct" and self.replacement is not None:
            raise ValueError("replacement is supported only by correct")
        return self


class FeedbackView(StrictModel):
    feedback_id: str
    scope_name: str
    rating: str
    created_at: float
    action: str = "note"
    effective: bool = False
    correction_job_id: str | None = None
    correction_index_status: str | None = None


WebhookEvent = Literal[
    "job.succeeded",
    "job.failed",
    "job.cancelled",
    "ingest.completed",
    "consolidation.completed",
    "index.completed",
    "export.ready",
    "scope.deleted",
]


class WebhookCreateRequest(StrictModel):
    label: str = Field(min_length=1, max_length=120)
    url: str = Field(min_length=8, max_length=2048)
    events: list[WebhookEvent] = Field(min_length=1, max_length=8)


class WebhookView(StrictModel):
    endpoint_id: str
    label: str
    url: str
    events: list[str]
    enabled: bool
    created_at: float
    updated_at: float | None = None


class IssuedWebhookView(WebhookView):
    signing_secret: str


class EvidenceRouteView(StrictModel):
    requested: str
    selected: Literal["raw", "compiled"]
    reasons: tuple[str, ...]


class PromptEvidenceView(StrictModel):
    schema_version: str
    format: Literal["text/plain", "application/json"]
    mode: Literal["raw_hierarchical", "compiled_evidence_packet"]
    content: str
    content_sha256: str
    content_character_count: int
    source_text_verbatim: bool
    trust_boundary: str
    window_count: int | None = None
    source_block_count: int | None = None
    neighbor_block_count: int | None = None
    memory_context_block_count: int | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)


class RecallResponse(StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra=_add_receipt_contract_extension,
    )

    query_id: str
    scope_name: str
    index_job_id: str
    evidence_route: EvidenceRouteView
    evidence: dict[str, Any]
    prompt_evidence: PromptEvidenceView
    debug: dict[str, Any] | None = None


GraphLayer = Literal["slow", "fast", "source"]
ActorRole = Literal["user", "assistant", "system", "tool"]


class MemoryGraphNodeView(StrictModel):
    id: str
    layer: GraphLayer
    kind: str
    category: str
    label: str
    summary: str
    relation: str
    state: str
    status: str
    confidence: float
    salience: float
    turn_index: int
    occurred_at: str | None = None
    subject_id: str | None = None
    cluster_id: str | None = None
    source_kind: str | None = None
    actor_role: ActorRole | None = None
    actor_roles: list[ActorRole] = Field(default_factory=list)
    authority: str | None = None
    provenance_source: str | None = None
    evidence_count: int
    visible_neighbor_count: int
    expandable: bool
    attributes: dict[str, Any] = Field(default_factory=dict)


class MemoryGraphEdgeView(StrictModel):
    id: str
    source: str
    target: str
    type: str
    weight: float
    origin: Literal["stored", "derived"]
    provenance: dict[str, Any] = Field(default_factory=dict)


class MemoryGraphCountsView(StrictModel):
    nodes: int
    edges: int
    slow: int
    fast: int
    source: int


class MemoryGraphPageView(StrictModel):
    limit: int
    offset: int
    truncated: bool
    next_cursor: str | None = None
    returned_neighbors: int | None = None


NarrativeKind = Literal[
    "decision",
    "milestone",
    "goal",
    "issue",
    "preference",
    "relationship",
    "fact",
]


class NarrativeGraphThreadView(StrictModel):
    id: str
    title: str
    summary: str
    kind: NarrativeKind
    status: str
    node_ids: list[str]
    memory_count: int
    evidence_count: int
    started_at: str | None = None
    updated_at: str | None = None


class NarrativeGraphSummaryView(StrictModel):
    headline: str
    summary: str
    thread_count: int
    key_moment_count: int
    evidence_count: int
    started_at: str | None = None
    updated_at: str | None = None
    focus: str
    source_schema_version: str
    source_node_count: int
    source_truncated: bool
    projection_strategy: str
    semantic_source: str


class MemoryGraphResponse(StrictModel):
    schema_version: str
    scope_name: str
    snapshot_id: str
    snapshot_state: Literal["committed", "building"] = "committed"
    provisional: bool = False
    view: Literal["overview", "neighbors", "recall_trace", "narrative"]
    requested_layers: list[GraphLayer]
    resolved_layers: list[GraphLayer]
    fallback_layer: GraphLayer | None = None
    nodes: list[MemoryGraphNodeView]
    edges: list[MemoryGraphEdgeView]
    counts: MemoryGraphCountsView
    page: MemoryGraphPageView
    root_id: str | None = None
    depth: int | None = None
    selected_memory_ids: list[str] = Field(default_factory=list)
    missing_memory_ids: list[str] = Field(default_factory=list)
    threads: list[NarrativeGraphThreadView] = Field(default_factory=list)
    narrative: NarrativeGraphSummaryView | None = None


class MemoryGraphEvidenceItem(StrictModel):
    source_record_id: str
    relationship: str
    session_id: str | None = None
    message_id: str | None = None
    role: str | None = None
    actor_role: ActorRole | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    agent_role: str | None = None
    agent_specialty: str | None = None
    agent_team: str | None = None
    target_agent_id: str | None = None
    occurred_at: str | None = None
    text: str
    text_sha256: str
    source_text_verbatim: bool
    evidence_char_start: int | None = None
    evidence_char_end: int | None = None


class MemoryGraphEvidenceResponse(StrictModel):
    schema_version: str
    scope_name: str
    snapshot_id: str
    snapshot_state: Literal["committed", "building"] = "committed"
    provisional: bool = False
    memory_id: str
    items: list[MemoryGraphEvidenceItem]
    page: MemoryGraphPageView


class MemoryGraphTraceRequest(StrictModel):
    query: str = Field(min_length=1, max_length=100_000)
    query_time: datetime | None = None
    max_windows: Literal[8] = 8
    debug: bool = False


class MemoryGraphTraceResponse(MemoryGraphResponse):
    query_id: str
    index_job_id: str
    retrieval_summary: dict[str, Any]
    debug: dict[str, Any] | None = None


class SessionMapResponse(StrictModel):
    schema_version: Literal["tmcra.session-map.1"]
    scope_name: str
    session_id: str
    snapshot_id: str
    snapshot_state: Literal["committed", "building"] = "committed"
    provisional: bool = False
    view: Literal["session_map"] = "session_map"
    projection_state: Literal["fallback", "ready"]
    generated_by: str
    prompt_version: str | None = None
    model: str | None = None
    title: str
    summary: str
    status: str
    source_app: str | None = None
    native_thread_id: str | None = None
    parent_session_id: str | None = None
    created_at: float | None = None
    updated_at: float | None = None
    message_count: int
    source_record_count: int
    semantic_record_count: int
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    threads: list[dict[str, Any]]
    counts: dict[str, int]
    time_range: dict[str, Any]
    evidence_binding: dict[str, Any]
    refresh: dict[str, Any] | None = None


class SessionAtlasResponse(StrictModel):
    schema_version: Literal["tmcra.session-atlas.1"]
    scope_name: str
    snapshot_id: str
    view: Literal["session_atlas"] = "session_atlas"
    projection_state: Literal["fallback", "ready"]
    generated_by: str
    prompt_version: str | None = None
    model: str | None = None
    session_count: int
    message_count: int
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    counts: dict[str, int]
    refresh: dict[str, Any] | None = None
    agent_enabled: bool = False


class VisualAtlasResponse(StrictModel):
    schema_version: Literal["tmcra.visual-atlas.1"]
    scope_name: str
    snapshot_id: str
    view: Literal["visual_atlas"] = "visual_atlas"
    projection_state: Literal["fallback", "ready"]
    generated_by: str
    prompt_version: str | None = None
    model: str | None = None
    full_projection: Literal[True] = True
    truncated: Literal[False] = False
    levels: list[Literal["domain", "session", "episode", "evidence"]]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    identity_manifest: dict[str, dict[str, Any]]
    addressability: dict[str, Any]
    counts: dict[str, int]
    refresh: dict[str, Any] | None = None
    agent_enabled: bool = False


class PersonalKnowledgeBaseResponse(StrictModel):
    schema_version: Literal["tmcra.personal-knowledge.1"]
    scope_name: str
    snapshot_id: str
    source_snapshot_id: str
    source_fingerprint: str
    view: Literal["personal_knowledge_base"] = "personal_knowledge_base"
    projection_state: Literal["fallback", "ready"]
    generated_by: str
    prompt_version: str | None = None
    model: str | None = None
    full_projection: Literal[True] = True
    truncated: Literal[False] = False
    domains: list[dict[str, Any]]
    pages: list[dict[str, Any]]
    evidence_catalog: dict[str, dict[str, Any]]
    counts: dict[str, int]
    refresh: dict[str, Any] | None = None
    agent_enabled: bool = False
    stale: bool = False


class SessionGraphRefreshResponse(StrictModel):
    accepted: Literal[True] = True
    projection_key: str
    source_fingerprint: str


class ProjectionBuildProgressResponse(StrictModel):
    schema_version: Literal["tmcra.projection-build-progress.1"]
    scope_name: str
    status: Literal["queued", "running", "ready", "failed"]
    stage: Literal[
        "session_maps", "session_atlas", "visual_atlas", "knowledge_base", "ready"
    ]
    progress_percent: int
    completed_units: int
    total_units: int
    session_maps: dict[str, Any]
    session_atlas: dict[str, Any]
    visual_atlas: dict[str, Any]
    knowledge_base: dict[str, Any]
    detail: str
    last_error: str | None = None
    can_retry: bool = False
    updated_at: float
    agent_enabled: bool
    resource_isolation: Literal[
        "adaptive-local-first",
        "dedicated-local-slot",
        "dedicated-provider",
        "shared-local-reserve",
        "user-provider",
        "unknown",
        "disabled",
    ]


class ErrorDetail(StrictModel):
    code: str
    message: str | None = None
    request_id: str | None = None
    details: Any = None
    retry_after_seconds: float | None = None


class ErrorResponse(StrictModel):
    error: ErrorDetail
