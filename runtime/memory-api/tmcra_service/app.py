from __future__ import annotations

import json
import logging
import hashlib
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal, Mapping

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.gzip import GZipMiddleware

from . import __version__
from .adapters.v4 import (
    ContentDeletionTargetNotFound,
    V4AdapterError,
    LocalEvidenceCompilationUnavailable,
    V4StorageAdapter,
)
from .api_models import (
    AuthenticatedSessionView,
    BillingGroupCreateRequest,
    BillingGroupMemberRequest,
    BillingGroupStatusRequest,
    BillingPeriodChangeRequest,
    BillingPlanVersionUpsertRequest,
    BillingPlanVersionView,
    BillingProfileView,
    BulkIngestRequest,
    BulkIngestResponse,
    ErrorResponse,
    EntitlementUpdateRequest,
    FeedbackRequest,
    FeedbackView,
    IngestRequest,
    IssuedScopeTokenView,
    IssuedWebhookView,
    JobView,
    ContentDeletionJobView,
    ContentDeletionView,
    MemoryDeleteRequest,
    MessageDeleteRequest,
    MemoryGraphEvidenceResponse,
    MemoryGraphResponse,
    MemoryGraphTraceRequest,
    MemoryGraphTraceResponse,
    PersonalKnowledgeBaseResponse,
    ProjectionBuildProgressResponse,
    ProviderCallReportRequest,
    ProviderCallReportView,
    UserProviderTaskClaimRequest,
    UserProviderTaskClaimView,
    UserProviderTaskCompleteRequest,
    UserProviderTaskFailRequest,
    UserProviderTaskLeaseRequest,
    UserProviderTaskStatusView,
    RecallRequest,
    RecallResponse,
    RetentionPolicyRequest,
    RetentionPolicyView,
    ScopeTokenCreateRequest,
    ScopeTokenView,
    ScopeCatalogView,
    ScopeRecoveryView,
    ScopeSummaryView,
    SessionAtlasResponse,
    SessionGraphRefreshResponse,
    SessionMapResponse,
    VisualAtlasResponse,
    QuotaView,
    UsageCostsView,
    WebhookCreateRequest,
    WebhookView,
)
from .auth import (
    APIKeyAuth,
    AuthContext,
    AuthenticationError,
    AuthorizationError,
    TokenIdempotencyConflict,
)
from .audio_asr_proxy import (
    AudioAsrProxy,
    AudioAsrProxyDisabled,
    AudioAsrProxyError,
    AudioAsrProxyTimeout,
)
from .actor_provenance import (
    ActorProvenanceError,
    enrich_evidence_actor_provenance,
)
from .api_access_log import (
    ApiAccessJournal,
    bounded_content_length,
    normalize_request_id,
    request_access_event,
)
from .commercial import (
    CommercialContractError,
    CommercialControl,
    WebhookDispatcher,
)
from .control_db import ControlDB
from .diagnostic_log import DiagnosticJournal
from .control_plane import (
    BillingAccessDenied,
    BillingConflict,
    BillingNotFound,
    MemoryControlPlane,
    QuotaExceeded,
    estimate_raw_tokens,
)
from .costing import journal_deepseek_calls
from .evidence_view import EvidenceViewError, build_prompt_evidence
from .graph_projection import (
    GraphProjectionError,
    MemoryGraphProjection,
    extract_trace_memory_ids,
    parse_layers,
)
from .narrative_graph import NARRATIVE_FOCI, NarrativeGraphError, build_narrative_graph
from .health_monitor import ContinuousReadinessMonitor
from .jobs import (
    FAILED,
    IdempotencyConflict,
    Job,
    JobQueueFull,
    JobStateError,
    JobStore,
    ResumeAuthorization,
)
from .gpu_scheduler import (
    GpuSchedulerClosedError,
    GpuSchedulerTimeoutError,
    GpuWorkload,
    GpuWorkloadScheduler,
)
from .provider_pool import ProviderCircuitBreaker, ProviderKeyPool
from .rate_limit import PressureGate
from .recall_pool import (
    RecallEnginePool,
    RecallPoolClosedError,
    RecallPoolSaturatedError,
    RecallPoolTimeoutError,
)
from .routing import select_evidence_route
from .feedback_effects import apply_feedback
from .runtime import LazyOnlineEngine, ServiceWorker
from .session_graph import (
    SessionGraphAgentRouter,
    SessionGraphError,
    SessionGraphService,
)
from .settings import ServiceSettings
from .staff_runtime import (
    LATENCY_EXCLUDED_PATHS,
    STAFF_MONITORING_HEADER,
    RequestLatencyWindow,
    StaffRuntimeStatus,
    staff_key_matches,
)
from .startup import (
    MemoryWriteAdmission,
    StartupPreflight,
    WriteAdmissionRejected,
)
from .usage_attribution import (
    AGENT_ID_HEADER,
    CLIENT_PLATFORM_HEADER,
    INTEGRATION_ID_HEADER,
    UsageAttribution,
    UsageAttributionError,
    resolve_request_attribution,
)
from .user_provider_tasks import (
    TASK_SCHEMA_VERSION as USER_PROVIDER_TASK_SCHEMA_VERSION,
    UserProviderLeaseLost,
    UserProviderTaskError,
    UserProviderTaskNotFound,
    UserProviderTaskStore,
)
from .writer_provider import LOCAL_QWEN_PROVIDER, primary_writer_route


class HealthResponse(BaseModel):
    """Stable anonymous liveness response exposed at ``/healthz``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = Field(description="The service process is running.")
    service: str = Field(description="Stable service identifier.")
    version: str = Field(description="Running TMCRA service version.")


class ReadinessResponse(BaseModel):
    """Anonymous readiness response shared by 200 and 503 responses."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "not_ready"] = Field(
        description="Whether the service is ready to accept production traffic."
    )
    service: str = Field(description="Stable service identifier.")
    version: str = Field(description="Running TMCRA service version.")
    checks: dict[str, Any] = Field(
        description="Named readiness checks and their current status."
    )
    snapshot_stale: bool = Field(
        description="Whether the monitor snapshot is older than its freshness threshold."
    )
    snapshot_age_seconds: float = Field(
        ge=0, description="Age of the readiness snapshot in seconds."
    )
    monitor_generation: int = Field(
        ge=0, description="Monotonic monitor snapshot generation."
    )
    recall_pool: dict[str, Any] = Field(
        description="Current recall-pool capacity and loading status."
    )
    write_admission: dict[str, Any] = Field(
        description="Informational write-admission state; it does not gate recall readiness."
    )


SCOPE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ON_BEHALF_SUBJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
ON_BEHALF_SUBJECT_HEADER = "X-TMCRA-On-Behalf-Of-Subject"
RECALL_ERROR_RESPONSES = {
    429: {
        "model": ErrorResponse,
        "description": "Tenant quota, rate, or recall queue limit was reached.",
        "headers": {
            "Retry-After": {
                "description": "Whole seconds to wait before retrying.",
                "schema": {"type": "string"},
            }
        },
    },
    503: {
        "model": ErrorResponse,
        "description": "Recall capacity is temporarily unavailable.",
        "headers": {
            "Retry-After": {
                "description": "Whole seconds to wait before retrying.",
                "schema": {"type": "string"},
            }
        },
    },
}


def usage_attribution_headers(
    client_platform: str | None = Header(
        default=None,
        alias=CLIENT_PLATFORM_HEADER,
        max_length=64,
        description=(
            "Calling client family, such as codex, openclaw, hermes, mcp, "
            "python, typescript, or rest. Direct-client values are reported, "
            "not trusted billing identity."
        ),
    ),
    integration_id: str | None = Header(
        default=None,
        alias=INTEGRATION_ID_HEADER,
        max_length=128,
        description=(
            "Installation or connection registry identifier. It becomes "
            "trusted attribution only inside the managing-key/on-behalf proxy boundary."
        ),
    ),
    agent_id: str | None = Header(
        default=None,
        alias=AGENT_ID_HEADER,
        max_length=200,
        description="Optional invoking Agent identifier for operational allocation.",
    ),
) -> dict[str, str]:
    return {
        **({CLIENT_PLATFORM_HEADER: client_platform} if client_platform else {}),
        **({INTEGRATION_ID_HEADER: integration_id} if integration_id else {}),
        **({AGENT_ID_HEADER: agent_id} if agent_id else {}),
    }


@dataclass
class ServiceComponents:
    settings: ServiceSettings
    database: ControlDB
    auth: APIKeyAuth
    control: MemoryControlPlane
    jobs: JobStore
    gate: PressureGate
    gpu_scheduler: GpuWorkloadScheduler
    storage: V4StorageAdapter
    online: LazyOnlineEngine
    worker: ServiceWorker
    commercial: CommercialControl
    webhooks: WebhookDispatcher
    startup: StartupPreflight
    health_monitor: ContinuousReadinessMonitor
    latency_window: RequestLatencyWindow
    staff_runtime: StaffRuntimeStatus
    provider_circuit: ProviderCircuitBreaker
    write_admission: MemoryWriteAdmission
    session_graphs: SessionGraphService
    api_access_log: ApiAccessJournal
    diagnostic_log: DiagnosticJournal
    audio_asr: AudioAsrProxy
    user_provider_tasks: UserProviderTaskStore


def build_components(settings: ServiceSettings) -> ServiceComponents:
    settings.validate()
    database = ControlDB(settings.control_db)
    auth = APIKeyAuth(database)
    control = MemoryControlPlane(database)
    control.backfill_catalog_from_jobs()
    jobs = JobStore(database)
    user_provider_tasks = UserProviderTaskStore(
        database,
        lease_seconds=max(60.0, float(settings.provider_lease_seconds)),
    )
    diagnostic_log = DiagnosticJournal(
        settings.diagnostic_log_path,
        enabled=settings.diagnostic_log_enabled,
    )
    gate = PressureGate(
        database,
        max_concurrency=settings.request_max_concurrency,
        per_minute=settings.request_per_minute,
        lease_seconds=settings.request_lease_seconds,
    )
    gpu_scheduler = GpuWorkloadScheduler.from_settings(settings)
    storage = V4StorageAdapter(settings)
    session_graphs = SessionGraphService(
        database,
        storage,
        agent=SessionGraphAgentRouter.from_env(),
        gpu_scheduler=gpu_scheduler,
    )
    online = LazyOnlineEngine(settings, gpu_scheduler=gpu_scheduler)
    commercial = CommercialControl(
        database,
        webhook_signing_key=settings.webhook_signing_key,
    )
    worker = ServiceWorker(
        settings=settings,
        database=database,
        jobs=jobs,
        storage=storage,
        online=online,
        gpu_scheduler=gpu_scheduler,
        commercial=commercial,
        on_ingest_committed=session_graphs.record_committed,
        on_generation_committed=session_graphs.record_generation_committed,
        diagnostic_log=diagnostic_log,
    )
    def projection_capacity_available() -> bool:
        return gpu_scheduler.can_start(GpuWorkload.GRAPH_BACKGROUND)

    session_graphs.set_production_capacity_guard(projection_capacity_available)
    writer_route = None
    if str(os.environ.get("TMCRA_WRITER_API_KEY_POOL") or "").strip():
        try:
            writer_route = primary_writer_route(os.environ)
        except ValueError as exc:
            raise RuntimeError(f"invalid Writer provider route: {exc}") from exc
        # Registration is local-only: ProviderKeyPool persists key hashes and
        # capacity, never secrets, and performs no provider request. Doing this
        # before lifespan admission prevents an empty bootstrap table from
        # being mistaken for available capacity.
        ProviderKeyPool(
            settings.control_db,
            pool=writer_route.pool_name,
            keys=writer_route.api_keys,
            max_concurrency_per_key=(
                1
                if writer_route.provider == LOCAL_QWEN_PROVIDER
                else settings.provider_key_concurrency
            ),
            lease_seconds=settings.provider_lease_seconds,
            billing_circuit_seconds=settings.provider_billing_circuit_seconds,
            auth_circuit_seconds=settings.provider_auth_circuit_seconds,
        )
    worker.writer_uses_local_gpu = bool(
        writer_route is not None and writer_route.provider == LOCAL_QWEN_PROVIDER
    )
    provider_circuit = ProviderCircuitBreaker(
        settings.control_db,
        pool=(writer_route.pool_name if writer_route else "deepseek-writer"),
    )
    write_admission = MemoryWriteAdmission(
        settings=settings,
        storage=storage,
        worker=worker,
        provider=provider_circuit,
    )
    webhooks = WebhookDispatcher(
        commercial,
        timeout_seconds=settings.webhook_timeout_seconds,
    )
    startup = StartupPreflight(settings)
    health_monitor = ContinuousReadinessMonitor(
        settings=settings,
        database=database,
        storage=storage,
        online=online,
        worker=worker,
    )
    latency_window = RequestLatencyWindow(
        window_seconds=settings.staff_latency_window_seconds,
        max_samples=settings.staff_latency_max_samples,
    )
    staff_runtime = StaffRuntimeStatus(
        settings=settings,
        database=database,
        startup=startup,
        health_monitor=health_monitor,
        latency_window=latency_window,
    )
    api_access_log = ApiAccessJournal(
        settings.api_access_log_path,
        enabled=settings.api_access_log_enabled,
    )
    audio_asr = AudioAsrProxy(
        base_url=settings.audio_asr_base_url,
        api_key_file=settings.audio_asr_api_key_file,
        timeout_seconds=settings.audio_asr_timeout_seconds,
        maximum_request_bytes=settings.audio_asr_max_request_bytes,
    )
    return ServiceComponents(
        settings=settings,
        database=database,
        auth=auth,
        control=control,
        jobs=jobs,
        gate=gate,
        gpu_scheduler=gpu_scheduler,
        storage=storage,
        online=online,
        worker=worker,
        commercial=commercial,
        webhooks=webhooks,
        startup=startup,
        health_monitor=health_monitor,
        latency_window=latency_window,
        staff_runtime=staff_runtime,
        provider_circuit=provider_circuit,
        write_admission=write_admission,
        session_graphs=session_graphs,
        api_access_log=api_access_log,
        diagnostic_log=diagnostic_log,
        audio_asr=audio_asr,
        user_provider_tasks=user_provider_tasks,
    )


def _scope_name(value: str) -> str:
    if not SCOPE_NAME_RE.fullmatch(value):
        raise HTTPException(status_code=422, detail="invalid scope name")
    return value


def _bounded_identifier(value: str, *, label: str, max_length: int = 512) -> str:
    normalized = str(value or "").strip()
    if not normalized or len(normalized) > max_length or any(
        ord(character) < 32 for character in normalized
    ):
        raise HTTPException(status_code=422, detail=f"invalid {label}")
    return normalized


def _content_deletion_payload(
    deletion: dict[str, Any], public_base_url: str
) -> dict[str, Any]:
    value = dict(deletion)
    value.pop("target_sha256", None)
    job_id = str(value.get("job_id") or "")
    value["job_status_url"] = (
        f"{public_base_url}/v1/jobs/{job_id}" if job_id else None
    )
    return value


def _job_payload(job: Job, public_base_url: str) -> dict[str, Any]:
    payload = dict(job.payload or {})
    error: Any = None
    if job.error:
        try:
            error = json.loads(job.error)
        except json.JSONDecodeError:
            error = {"message": job.error}
    return {
        "job_id": job.job_id,
        "tenant_id": job.tenant_id,
        "scope_name": payload.get("scope_name", "default"),
        "job_type": payload.get("job_type", ""),
        "status": job.state,
        "attempts": max(0, job.version),
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "heartbeat_at": job.heartbeat_at,
        "lease_expires_at": job.lease_expires_at,
        "result": job.result,
        "error": error,
        "status_url": f"{public_base_url}/v1/jobs/{job.job_id}",
    }


def _provider_call_report_view(
    provider_call: Any, *, idempotent_replay: bool
) -> dict[str, Any]:
    """Return the public, accounting-only view of an answer-model call."""

    return {
        "call_id": provider_call.call_id,
        "scope_name": provider_call.scope_name,
        "provider": provider_call.provider,
        "model": provider_call.model,
        "operation": provider_call.operation or "chat_answer",
        "status": provider_call.status,
        "input_tokens": provider_call.input_tokens,
        "output_tokens": provider_call.output_tokens,
        "total_tokens": provider_call.total_tokens,
        "cache_hit_tokens": provider_call.cache_hit_tokens,
        "cache_miss_tokens": provider_call.cache_miss_tokens,
        "usage_state": (
            "complete" if provider_call.usage_state == "complete" else "missing"
        ),
        "cost_micro_cny": provider_call.cost_micro_cny,
        "price_version": provider_call.price_version,
        "idempotent_replay": idempotent_replay,
    }


def _find_idempotent_job(
    database: ControlDB, tenant_id: str, idempotency_key: str
) -> Job | None:
    with database.transaction(immediate=False) as connection:
        row = connection.execute(
            "SELECT job_id FROM jobs WHERE tenant_id=? AND idempotency_key=?",
            (tenant_id, idempotency_key),
        ).fetchone()
    if row is None:
        return None
    return JobStore(database).get(str(row["job_id"]), tenant_id=tenant_id)


def create_app(settings: ServiceSettings) -> FastAPI:
    components = build_components(settings)
    bearer = HTTPBearer(auto_error=False, scheme_name="TMCRAApiKey")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            components.gpu_scheduler.start()
            await run_in_threadpool(
                components.startup.run, components.storage, components.online
            )
            await run_in_threadpool(
                components.database.reconcile_stale_scope_claims
            )
            if (
                settings.startup_preflight_mode == "basic"
                and settings.preload_online_engine
            ):
                dispatcher = await run_in_threadpool(components.online.get)
                warmup = getattr(dispatcher, "warmup", None)
                if callable(warmup):
                    snapshots = await run_in_threadpool(
                        components.storage.audit_active_indexes
                    )
                    await run_in_threadpool(warmup, snapshots)
            components.worker.start()
            components.session_graphs.start()
            components.webhooks.start()
            components.startup.record_runtime(components.worker.status())
            components.health_monitor.start()
            yield
        finally:
            components.health_monitor.stop(timeout=5.0)
            components.webhooks.stop(timeout=5.0)
            components.session_graphs.stop(timeout=5.0)
            components.worker.stop()
            components.storage.stop()
            stop_online = getattr(components.online, "stop", None)
            if callable(stop_online):
                # Leave time inside the control script's 30-second graceful
                # shutdown window for Uvicorn/supervisor teardown. Any model
                # close still running after this bound is released with the
                # process, and the verified control fallback prevents a hung
                # rollback from swapping files under a live child.
                stop_online(timeout=20.0)
            components.gpu_scheduler.stop(timeout=3.0)
            components.api_access_log.close()
            components.diagnostic_log.close()

    app = FastAPI(
        title="TMCRA Memory API",
        version=__version__,
        description=(
            "Tenant-isolated long-term memory ingestion and prompt-ready recall. "
            "TMCRA does not generate the final assistant answer."
        ),
        servers=[{"url": settings.public_base_url, "description": "Configured API"}],
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_tags=[
            {
                "name": "health",
                "description": "Anonymous liveness and readiness probes for service discovery.",
            },
            {"name": "memory", "description": "Write, consolidate, and recall memory."},
            {
                "name": "memory-graph",
                "description": "Explore committed slow, fast, and source memory projections.",
            },
            {"name": "jobs", "description": "Inspect and control asynchronous jobs."},
            {"name": "usage", "description": "Inspect registered model-API usage."},
            {
                "name": "audio",
                "description": "Authenticated speech transcription through the isolated TMCRA ASR worker.",
            },
            {"name": "access", "description": "Issue revocable persona-scoped access tokens."},
            {"name": "governance", "description": "Export, delete, retain, and review memory."},
            {"name": "webhooks", "description": "Deliver signed lifecycle event notifications."},
        ],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.state.components = components

    def record_request_diagnostic(
        request: Request,
        exc: BaseException,
        *,
        status_code: int,
        error_code: str,
        severity: str = "error",
        context: dict[str, Any] | None = None,
    ) -> None:
        route_object = request.scope.get("route")
        route = getattr(route_object, "path", None)
        if not isinstance(route, str) or not route:
            route = "__unmatched__"
        scope_name = request.path_params.get("scope_name")
        if scope_name is not None and not SCOPE_NAME_RE.fullmatch(str(scope_name)):
            scope_name = None
        path_job_id = request.path_params.get("job_id")
        job_ids = [
            str(value)
            for value in getattr(request.state, "job_ids", [])
            if str(value)
        ]
        if path_job_id and str(path_job_id) not in job_ids:
            job_ids.append(str(path_job_id))
        auth_context = getattr(request.state, "auth_context", None)
        components.diagnostic_log.record_exception(
            exc,
            component="api",
            operation=f"{request.method.upper()} {route}",
            severity=severity,
            request_id=getattr(request.state, "request_id", None),
            job_id=job_ids[0] if len(job_ids) == 1 else None,
            tenant_id=getattr(auth_context, "tenant_id", None),
            scope_name=str(scope_name) if scope_name is not None else None,
            status_code=status_code,
            error_code=error_code,
            context={
                "route": route,
                "method": request.method.upper(),
                "job_count": len(job_ids),
                **dict(context or {}),
            },
        )

    @app.middleware("http")
    async def request_contract(request: Request, call_next: Callable[..., Any]) -> Response:
        request_id = normalize_request_id(
            request.headers.get("x-request-id"), generated=uuid.uuid4().hex
        )
        request.state.request_id = request_id
        started = time.perf_counter()
        request_bytes = bounded_content_length(request.headers.get("content-length"))

        def record_access(
            *,
            response: Response | None,
            status_code: int,
            latency_ms: float,
            exception_type: str | None = None,
        ) -> None:
            route_object = request.scope.get("route")
            route = getattr(route_object, "path", None)
            unmatched_path = None
            if not isinstance(route, str) or not route:
                route = "__unmatched__"
                unmatched_path = request.url.path
            scope_name = request.path_params.get("scope_name")
            if scope_name is not None and not SCOPE_NAME_RE.fullmatch(str(scope_name)):
                scope_name = None
            path_job_id = request.path_params.get("job_id")
            job_ids = [
                str(value)
                for value in getattr(request.state, "job_ids", [])
                if str(value)
            ]
            if path_job_id and str(path_job_id) not in job_ids:
                job_ids.append(str(path_job_id))
            attribution = getattr(request.state, "usage_attribution", None)
            response_bytes = (
                bounded_content_length(response.headers.get("content-length"))
                if response is not None
                else None
            )
            components.api_access_log.record(
                request_access_event(
                    request_id=request_id,
                    method=request.method,
                    route=route,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                    auth_context=getattr(request.state, "auth_context", None),
                    auth_kind=getattr(request.state, "auth_kind", None),
                    scope_name=str(scope_name) if scope_name is not None else None,
                    job_ids=job_ids,
                    client_platform=getattr(attribution, "client_platform", None),
                    integration_id=getattr(attribution, "integration_id", None),
                    agent_id=getattr(attribution, "agent_id", None),
                    error_code=getattr(request.state, "error_code", None),
                    exception_type=exception_type,
                    unmatched_path=unmatched_path,
                )
            )

        def finalize(
            response: Response, *, exception_type: str | None = None
        ) -> Response:
            latency_ms = (time.perf_counter() - started) * 1000
            response.headers["x-request-id"] = request_id
            response.headers["x-tmcra-latency-ms"] = str(round(latency_ms, 2))
            if request.url.path not in LATENCY_EXCLUDED_PATHS:
                components.latency_window.observe(
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                )
            if (
                response.status_code >= 400
                and getattr(request.state, "error_code", None) is None
            ):
                request.state.error_code = {
                    400: "bad_request",
                    401: "unauthorized",
                    403: "forbidden",
                    404: "not_found",
                    405: "method_not_allowed",
                    409: "conflict",
                    411: "content_length_required",
                    413: "request_too_large",
                    422: "validation_error",
                    429: "rate_limited",
                    500: "internal_error",
                    502: "upstream_error",
                    503: "service_unavailable",
                    504: "upstream_timeout",
                }.get(response.status_code, "http_error")
            if (
                response.status_code in {401, 403}
                and getattr(request.state, "auth_context", None) is None
                and getattr(request.state, "auth_kind", None) is None
            ):
                request.state.auth_kind = "rejected"
            record_access(
                response=response,
                status_code=response.status_code,
                latency_ms=latency_ms,
                exception_type=exception_type,
            )
            return response

        content_length = request.headers.get("content-length")
        if request.method in {"POST", "PUT", "PATCH"} and not content_length:
            request.state.error_code = "content_length_required"
            return finalize(JSONResponse(
                status_code=411,
                content={
                    "error": {
                        "code": "content_length_required",
                        "request_id": request_id,
                    }
                },
                headers={"x-request-id": request_id},
            ))
        if content_length:
            try:
                too_large = int(content_length) > settings.request_body_limit
            except ValueError:
                too_large = True
            if too_large:
                request.state.error_code = "request_too_large"
                return finalize(JSONResponse(
                    status_code=413,
                    content={"error": {"code": "request_too_large", "request_id": request_id}},
                    headers={"x-request-id": request_id},
                ))
        try:
            response = await call_next(request)
        except Exception as exc:
            request.state.error_code = "internal_error"
            record_request_diagnostic(
                request,
                exc,
                status_code=500,
                error_code="internal_error",
            )
            return finalize(
                JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "code": "internal_error",
                            "message": "internal service error",
                            "request_id": request_id,
                        }
                    },
                    headers={"x-request-id": request_id},
                ),
                exception_type=type(exc).__name__,
            )
        return finalize(response)

    def error_content(
        request: Request,
        *,
        code: str,
        message: str | None = None,
        details: Any = None,
        retry_after_seconds: float | None = None,
    ) -> dict[str, Any]:
        request.state.error_code = code
        return {
            "error": {
                "code": code,
                "message": message,
                "request_id": getattr(request.state, "request_id", None),
                "details": details,
                "retry_after_seconds": retry_after_seconds,
            }
        }

    def recall_retry_after(exc: Any) -> tuple[float, str]:
        retry_after = max(1.0, float(getattr(exc, "retry_after", 1.0)))
        return retry_after, str(max(1, int(retry_after + 0.999)))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_content(
                request,
                code="validation_error",
                message="request validation failed",
                details=jsonable_encoder(exc.errors()),
            ),
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        code = {
            404: "not_found",
            409: "conflict",
            411: "content_length_required",
            413: "request_too_large",
            422: "validation_error",
            429: "rate_limited",
            503: "unavailable",
        }.get(exc.status_code, "request_failed")
        details: Any = None
        message: str | None = None
        if isinstance(detail, dict):
            code = str(detail.get("code") or code)
            message = str(detail.get("message") or "") or None
            details = {key: value for key, value in detail.items() if key not in {"code", "message"}}
        elif detail is not None:
            message = str(detail)
        if exc.status_code >= 500:
            record_request_diagnostic(
                request,
                exc,
                status_code=exc.status_code,
                error_code=code,
                severity="warning",
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_content(request, code=code, message=message, details=details),
            headers=exc.headers,
        )

    @app.exception_handler(AuthenticationError)
    async def authentication_error(request: Request, exc: AuthenticationError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content=error_content(request, code="unauthorized", message=str(exc)),
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(AuthorizationError)
    async def authorization_error(request: Request, exc: AuthorizationError) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content=error_content(request, code="forbidden", message=str(exc)),
        )

    @app.exception_handler(UserProviderTaskError)
    async def user_provider_task_error(
        request: Request, exc: UserProviderTaskError
    ) -> JSONResponse:
        status_code = (
            404
            if isinstance(exc, UserProviderTaskNotFound)
            else 409
            if isinstance(exc, UserProviderLeaseLost)
            else 422
        )
        return JSONResponse(
            status_code=status_code,
            content=error_content(request, code=exc.code, message=str(exc)),
        )

    @app.exception_handler(QuotaExceeded)
    async def quota_exceeded_error(request: Request, exc: QuotaExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content=error_content(
                request,
                code="quota_exceeded",
                message=str(exc),
                details={
                    "metric": exc.metric,
                    "used": exc.used,
                    "limit": exc.limit,
                    "requested": exc.requested,
                    "remaining": max(0, exc.limit - exc.used),
                },
            ),
        )

    @app.exception_handler(BillingAccessDenied)
    async def billing_access_denied_error(
        request: Request, exc: BillingAccessDenied
    ) -> JSONResponse:
        return JSONResponse(
            status_code=402,
            content=error_content(
                request,
                code="billing_inactive",
                message=str(exc),
                details={"group_id": exc.group_id},
            ),
        )

    @app.exception_handler(BillingConflict)
    async def billing_conflict_error(
        request: Request, exc: BillingConflict
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=error_content(
                request, code="billing_conflict", message=str(exc)
            ),
        )

    @app.exception_handler(BillingNotFound)
    async def billing_not_found_error(
        request: Request, exc: BillingNotFound
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=error_content(
                request, code="billing_not_found", message=str(exc.args[0])
            ),
        )

    @app.exception_handler(IdempotencyConflict)
    async def idempotency_error(request: Request, exc: IdempotencyConflict) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=error_content(request, code="idempotency_conflict", message=str(exc)),
        )

    @app.exception_handler(TokenIdempotencyConflict)
    async def token_idempotency_error(
        request: Request, exc: TokenIdempotencyConflict
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=error_content(request, code="idempotency_conflict", message=str(exc)),
        )

    @app.exception_handler(JobQueueFull)
    async def queue_full_error(request: Request, exc: JobQueueFull) -> JSONResponse:
        status_code = 429 if exc.queue_scope == "tenant" else 503
        if status_code >= 500:
            record_request_diagnostic(
                request,
                exc,
                status_code=status_code,
                error_code=f"{exc.queue_scope}_queue_full",
                severity="warning",
                context={"limit": exc.limit},
            )
        return JSONResponse(
            status_code=status_code,
            content=error_content(
                request,
                code=f"{exc.queue_scope}_queue_full",
                message=str(exc),
                details={"limit": exc.limit},
                retry_after_seconds=5,
            ),
            headers={"Retry-After": "5"},
        )

    @app.exception_handler(WriteAdmissionRejected)
    async def write_admission_error(
        request: Request, exc: WriteAdmissionRejected
    ) -> JSONResponse:
        retry_after_header = str(
            max(1, int(exc.retry_after_seconds + 0.999))
        )
        record_request_diagnostic(
            request,
            exc,
            status_code=503,
            error_code=exc.reason,
            severity="warning",
            context={"retry_after_seconds": exc.retry_after_seconds},
        )
        return JSONResponse(
            status_code=503,
            content=error_content(
                request,
                code=exc.reason,
                message="memory ingestion is temporarily unavailable",
                details={"admission": "closed"},
                retry_after_seconds=exc.retry_after_seconds,
            ),
            headers={"Retry-After": retry_after_header},
        )

    @app.exception_handler(RecallPoolSaturatedError)
    async def recall_pool_saturated_error(
        request: Request, exc: RecallPoolSaturatedError
    ) -> JSONResponse:
        retry_after, retry_after_header = recall_retry_after(exc)
        status_code = 429 if exc.scope == "tenant" else 503
        if status_code >= 500:
            record_request_diagnostic(
                request,
                exc,
                status_code=status_code,
                error_code=f"{exc.scope}_recall_queue_full",
                severity="warning",
                context={"retry_after_seconds": retry_after},
            )
        return JSONResponse(
            status_code=status_code,
            content=error_content(
                request,
                code=f"{exc.scope}_recall_queue_full",
                message=str(exc),
                details={"scope": exc.scope},
                retry_after_seconds=retry_after,
            ),
            headers={"Retry-After": retry_after_header},
        )

    @app.exception_handler(LocalEvidenceCompilationUnavailable)
    async def local_evidence_compilation_unavailable(request: Request, exc: LocalEvidenceCompilationUnavailable) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": {
            "code": "local_evidence_compilation_failed", "message": str(exc),
            "request_id": getattr(request.state, "request_id", None),
        }})

    @app.exception_handler(RecallPoolTimeoutError)
    async def recall_pool_timeout_error(
        request: Request, exc: RecallPoolTimeoutError
    ) -> JSONResponse:
        retry_after, retry_after_header = recall_retry_after(exc)
        record_request_diagnostic(
            request,
            exc,
            status_code=503,
            error_code="recall_queue_timeout",
            severity="warning",
            context={
                "retry_after_seconds": retry_after,
                "waited_seconds": exc.waited,
            },
        )
        return JSONResponse(
            status_code=503,
            content=error_content(
                request,
                code="recall_queue_timeout",
                message=str(exc),
                details={"waited_seconds": exc.waited},
                retry_after_seconds=retry_after,
            ),
            headers={"Retry-After": retry_after_header},
        )

    @app.exception_handler(RecallPoolClosedError)
    async def recall_pool_closed_error(
        request: Request, exc: RecallPoolClosedError
    ) -> JSONResponse:
        retry_after, retry_after_header = recall_retry_after(exc)
        record_request_diagnostic(
            request,
            exc,
            status_code=503,
            error_code="recall_pool_unavailable",
            severity="warning",
            context={"retry_after_seconds": retry_after},
        )
        return JSONResponse(
            status_code=503,
            content=error_content(
                request,
                code="recall_pool_unavailable",
                message=str(exc),
                retry_after_seconds=retry_after,
            ),
            headers={"Retry-After": retry_after_header},
        )

    @app.exception_handler(GraphProjectionError)
    async def graph_projection_error(
        request: Request, exc: GraphProjectionError
    ) -> JSONResponse:
        if exc.status_code >= 500:
            record_request_diagnostic(
                request,
                exc,
                status_code=exc.status_code,
                error_code=exc.code,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_content(request, code=exc.code, message=str(exc)),
        )

    @app.exception_handler(SessionGraphError)
    async def session_graph_error(
        request: Request, exc: SessionGraphError
    ) -> JSONResponse:
        if exc.status_code >= 500:
            record_request_diagnostic(
                request,
                exc,
                status_code=exc.status_code,
                error_code=exc.code,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_content(request, code=exc.code, message=str(exc)),
        )

    @app.exception_handler(CommercialContractError)
    async def commercial_contract_error(
        request: Request, exc: CommercialContractError
    ) -> JSONResponse:
        if exc.code == "scope_deleted":
            status_code = 410
        elif exc.code in {"scope_deleting", "scope_content_deleting", "feedback_idempotency_conflict"}:
            status_code = 409
        elif exc.code == "webhook_signing_not_configured":
            status_code = 503
        else:
            status_code = 422
        retry_after = 30 if exc.code == "scope_quarantined" else None
        if status_code >= 500:
            record_request_diagnostic(
                request,
                exc,
                status_code=status_code,
                error_code=exc.code,
                severity="warning",
            )
        return JSONResponse(
            status_code=status_code,
            content=error_content(
                request,
                code=exc.code,
                message=str(exc),
                retry_after_seconds=retry_after,
            ),
            headers={"Retry-After": str(retry_after)} if retry_after else None,
        )

    def trusted_subject_context(
        context: AuthContext,
        tenant_scopes: frozenset[str],
        on_behalf_subject: str | None,
    ) -> AuthContext:
        if on_behalf_subject is None:
            return context
        if (
            context.credential_type != "api_key"
            or "tokens:manage" not in context.scopes
            or "tokens:manage" not in tenant_scopes
        ):
            raise AuthorizationError(
                f"{ON_BEHALF_SUBJECT_HEADER} requires a tokens:manage API key"
            )
        clean_subject = str(on_behalf_subject).strip()
        if (
            clean_subject != on_behalf_subject
            or not ON_BEHALF_SUBJECT_RE.fullmatch(clean_subject)
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_on_behalf_subject",
                    "message": f"{ON_BEHALF_SUBJECT_HEADER} is invalid",
                },
            )
        # Only attribution changes. Tenant, permissions, credential type, and
        # exact/prefix scope selectors remain those of the authenticated key.
        return replace(context, subject=clean_subject)

    def require_permission(
        permission: str, *, api_key_only: bool = False
    ) -> Callable[..., AuthContext]:
        def dependency(
            request: Request,
            on_behalf_subject: str | None = Header(
                default=None,
                alias=ON_BEHALF_SUBJECT_HEADER,
                min_length=1,
                max_length=200,
                description=(
                    "Trusted subject attribution for server-side control-plane "
                    "proxies. Requires a tokens:manage API key."
                ),
            ),
            credentials: HTTPAuthorizationCredentials | None = Depends(bearer)
        ) -> AuthContext:
            if credentials is None or credentials.scheme.lower() != "bearer":
                raise AuthenticationError("Bearer API key is required")
            context = components.auth.authenticate(credentials.credentials)
            request.state.auth_context = context
            request.state.auth_kind = context.credential_type
            tenant_scopes = components.database.get_tenant_scopes(context.tenant_id)
            if permission not in context.scopes or permission not in tenant_scopes:
                raise AuthorizationError(f"missing permission: {permission}")
            if api_key_only and context.credential_type != "api_key":
                raise AuthorizationError("this operation requires an API key")
            scope_name = request.path_params.get("scope_name")
            if scope_name is not None and not context.allows_scope_name(str(scope_name)):
                raise AuthorizationError("access token is not valid for this scope")
            resolved = trusted_subject_context(
                context, tenant_scopes, on_behalf_subject
            )
            request.state.auth_context = resolved
            request.state.auth_kind = resolved.credential_type
            return resolved

        return dependency

    def require_any_permission(
        *permissions: str, api_key_only: bool = False
    ) -> Callable[..., AuthContext]:
        requested = frozenset(permissions)

        def dependency(
            request: Request,
            on_behalf_subject: str | None = Header(
                default=None,
                alias=ON_BEHALF_SUBJECT_HEADER,
                min_length=1,
                max_length=200,
                description=(
                    "Trusted subject attribution for server-side control-plane "
                    "proxies. Requires a tokens:manage API key."
                ),
            ),
            credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
        ) -> AuthContext:
            if credentials is None or credentials.scheme.lower() != "bearer":
                raise AuthenticationError("Bearer API key is required")
            context = components.auth.authenticate(credentials.credentials)
            request.state.auth_context = context
            request.state.auth_kind = context.credential_type
            tenant_scopes = components.database.get_tenant_scopes(context.tenant_id)
            if not (requested & context.scopes & tenant_scopes):
                raise AuthorizationError(
                    "missing permission: " + " or ".join(sorted(requested))
                )
            if api_key_only and context.credential_type != "api_key":
                raise AuthorizationError("this operation requires an API key")
            resolved = trusted_subject_context(
                context, tenant_scopes, on_behalf_subject
            )
            request.state.auth_context = resolved
            request.state.auth_kind = resolved.credential_type
            return resolved

        return dependency

    def require_authenticated(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> AuthContext:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise AuthenticationError("Bearer API key is required")
        context = components.auth.authenticate(credentials.credentials)
        request.state.auth_context = context
        request.state.auth_kind = context.credential_type
        return context

    def require_staff_monitoring(
        request: Request,
        supplied_key: str | None = Header(
            default=None,
            alias=STAFF_MONITORING_HEADER,
        ),
    ) -> None:
        if settings.staff_monitoring_key is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "not_found", "message": "not found"},
            )
        if not staff_key_matches(settings.staff_monitoring_key, supplied_key):
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "staff_unauthorized",
                    "message": "staff credentials are invalid",
                },
                headers={"WWW-Authenticate": "TMCRA-Staff-Key"},
            )
        request.state.auth_kind = "staff"

    def require_billing_staff(
        request: Request,
        supplied_key: str | None = Header(
            default=None,
            alias=STAFF_MONITORING_HEADER,
            description="Private TMCRA billing-administration credential.",
        ),
    ) -> None:
        if settings.staff_monitoring_key is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "billing_admin_disabled",
                    "message": "billing administration is not configured",
                },
            )
        if not staff_key_matches(settings.staff_monitoring_key, supplied_key):
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "billing_staff_unauthorized",
                    "message": "billing staff credentials are invalid",
                },
                headers={"WWW-Authenticate": "TMCRA-Staff-Key"},
            )
        request.state.auth_kind = "billing_staff"

    def acquire_gate(tenant_id: str) -> str:
        decision = components.gate.acquire(tenant_id)
        if not decision:
            raise HTTPException(
                status_code=429,
                detail=f"tenant pressure gate rejected request: {decision.reason}",
                headers={"Retry-After": str(max(1, int(decision.retry_after + 0.999)))},
            )
        assert decision.lease_id is not None
        return decision.lease_id

    def require_active_scope(tenant_id: str, scope_name: str) -> None:
        components.commercial.require_scope_active(tenant_id, scope_name)

    def require_recall_scope(
        tenant_id: str, scope_name: str
    ) -> dict[str, Any] | None:
        return components.commercial.require_scope_readable(tenant_id, scope_name)

    def run_online_recall(
        tenant_id: str,
        *,
        before_execute: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Route production recalls through the tenant-aware pool.

        Legacy test doubles and a pre-pool engine expose only
        ``recall(**kwargs)``. Keeping that compatibility branch explicit avoids
        teaching production code to silently drop tenant identity.
        """

        online = components.online.get()
        try:
            with components.gpu_scheduler.lease(
                GpuWorkload.RECALL_FOREGROUND,
                timeout=float(settings.recall_queue_timeout_seconds),
            ):
                if isinstance(online, RecallEnginePool):
                    return online.recall(
                        tenant_id=tenant_id,
                        before_execute=before_execute,
                        **kwargs,
                    )
                if before_execute is not None:
                    before_execute()
                return online.recall(**kwargs)
        except GpuSchedulerTimeoutError as exc:
            raise RecallPoolTimeoutError(
                waited=exc.waited_seconds,
                retry_after=1.0,
            ) from exc
        except GpuSchedulerClosedError as exc:
            raise RecallPoolClosedError(retry_after=1.0) from exc

    def online_status() -> dict[str, Any]:
        status_method = getattr(components.online, "status", None)
        if not callable(status_method):
            return {"loaded": bool(getattr(components.online, "loaded", False))}
        value = status_method()
        if not isinstance(value, dict):
            as_dict_method = getattr(value, "as_dict", None)
            value = as_dict_method() if callable(as_dict_method) else {}
        if not isinstance(value, dict):
            value = {}

        # Readiness is public. Expose only bounded scheduler/capacity counters;
        # in particular, do not surface constructor or scale failure strings.
        result = {
            key: value[key]
            for key in ("loaded", "loaded_count", "minimum_loaded", "stopped")
            if key in value
        }
        pool = value.get("pool")
        if not isinstance(pool, dict):
            pool = value
        result["pool"] = {
            key: pool[key]
            for key in (
                "min_size",
                "max_size",
                "current_size",
                "desired_size",
                "loaded",
                "fully_loaded",
                "active",
                "idle",
                "pending",
                "pending_tenants",
                "max_pending",
                "per_tenant_pending",
                "warming",
                "scaling",
                "scaling_direction",
                "closed",
            )
            if key in pool
        }
        metrics = value.get("metrics")
        if isinstance(metrics, dict):
            result["metrics"] = {
                key: metrics[key]
                for key in (
                    "submitted",
                    "started",
                    "completed",
                    "failed",
                    "saturated",
                    "timed_out",
                    "current_size",
                    "desired_size",
                    "active",
                    "pending",
                    "peak_active",
                    "peak_pending",
                    "arrival_rate_ewma",
                    "service_time_ewma_seconds",
                    "offered_load",
                    "utilization",
                    "target_utilization",
                )
                if key in metrics
            }
        capacity = value.get("gpu_capacity")
        if isinstance(capacity, dict):
            result["gpu_capacity"] = {
                key: capacity[key]
                for key in (
                    "device",
                    "total_bytes",
                    "free_bytes",
                    "allocated_bytes",
                    "reserved_bytes",
                    "reusable_reserved_bytes",
                    "effective_free_bytes",
                    "headroom_bytes",
                    "replica_estimate_bytes",
                    "can_add_replica",
                )
                if key in capacity
            }
        cache_trim = value.get("cuda_cache_trim")
        if isinstance(cache_trim, dict):
            result["cuda_cache_trim"] = {
                key: cache_trim[key]
                for key in (
                    "enabled",
                    "monitor_alive",
                    "idle_seconds",
                    "cooldown_seconds",
                    "min_reclaimable_bytes",
                    "idle_for_seconds",
                    "last_success_age_seconds",
                    "attempts",
                    "successes",
                    "failures",
                    "last_released_bytes",
                    "total_released_bytes",
                )
                if key in cache_trim
            }
        scheduler = value.get("gpu_scheduler")
        if isinstance(scheduler, dict):
            safe_scheduler = {
                key: scheduler[key]
                for key in (
                    "enabled",
                    "monitor_alive",
                    "recall_capacity",
                    "safety_free_bytes",
                    "foreground_waiting",
                    "foreground_active",
                    "active",
                    "waiting",
                )
                if key in scheduler
            }
            telemetry = scheduler.get("telemetry")
            if isinstance(telemetry, dict):
                safe_scheduler["telemetry"] = {
                    key: telemetry[key]
                    for key in (
                        "available",
                        "sample_age_seconds",
                        "utilization_percent",
                        "memory_used_bytes",
                        "memory_free_bytes",
                        "power_watts",
                        "recent_mean_utilization_percent",
                        "recent_max_utilization_percent",
                    )
                    if key in telemetry
                }
            result["gpu_scheduler"] = safe_scheduler
        return result

    def ingest_payload(
        scope_name: str,
        body: IngestRequest,
        usage_attribution: UsageAttribution,
        provider_execution: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "job_type": "ingest",
            "scope_name": scope_name,
            "session_id": body.session_id,
            "messages": [item.model_dump(mode="json") for item in body.messages],
            "consistency": body.consistency,
            "slow_policy": body.slow_policy,
            "metadata": body.metadata,
            "_usage_attribution": usage_attribution.as_dict(),
        }
        if provider_execution is not None:
            payload["_provider_execution"] = dict(provider_execution)
        return payload

    def requested_provider_execution(
        value: str | None,
        *,
        stage: Literal["writer", "organizer"],
        context: AuthContext,
    ) -> dict[str, str] | None:
        route = str(value or "").strip()
        if not route:
            return None
        if os.getenv("TMCRA_DEPLOYMENT_MODE") == "local":
            raise HTTPException(status_code=422, detail={
                "code": "local_provider_handoff_disabled",
                "message": "Full-local memory executes all model stages inside the local runtime",
            })
        if route != "user-provider":
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_provider_execution",
                    "message": f"unsupported {stage} provider execution route",
                },
            )
        return {
            stage: route,
            "auth_key_id": context.key_id,
        }

    def request_usage_attribution(
        request: Request, context: AuthContext
    ) -> UsageAttribution:
        try:
            attribution = resolve_request_attribution(context, request.headers)
            request.state.usage_attribution = attribution
            return attribution
        except UsageAttributionError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_usage_attribution",
                    "message": str(exc),
                },
            ) from exc

    @app.get(
        "/healthz",
        response_model=HealthResponse,
        tags=["health"],
        summary="Check service liveness",
        description=(
            "Returns 200 when the HTTP service is running. This endpoint is anonymous "
            "and does not validate provider, database, or worker readiness."
        ),
        operation_id="healthz",
    )
    def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok", service="tmcra-memory", version=__version__
        )

    @app.get(
        "/readyz",
        response_model=ReadinessResponse,
        responses={
            503: {
                "model": ReadinessResponse,
                "description": "The service is running but is not ready to accept production traffic.",
            }
        },
        tags=["health"],
        summary="Check service readiness",
        description=(
            "Returns the current readiness snapshot without authentication. A 200 response "
            "means the service is ready; a 503 response means one or more readiness checks "
            "are not healthy."
        ),
        operation_id="readyz",
    )
    def readyz(response: Response) -> ReadinessResponse:
        snapshot = components.health_monitor.snapshot()
        write_admission = components.write_admission.snapshot().as_dict()
        if not snapshot["ready"]:
            response.status_code = 503
        return ReadinessResponse(
            status="ready" if snapshot["ready"] else "not_ready",
            service="tmcra-memory",
            version=__version__,
            checks=snapshot["checks"],
            snapshot_stale=snapshot["stale"],
            snapshot_age_seconds=snapshot["snapshot_age_seconds"],
            monitor_generation=snapshot["generation"],
            recall_pool=online_status(),
            # Paid-write admission is intentionally informational here. A
            # billing/auth circuit must not remove a healthy read/recall
            # instance from service discovery.
            write_admission=write_admission,
        )

    @app.get("/v1/internal/runtime", include_in_schema=False)
    def staff_runtime(
        response: Response,
        _: None = Depends(require_staff_monitoring),
    ) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        snapshot = components.staff_runtime.snapshot()
        snapshot["gpu_scheduler"] = components.gpu_scheduler.status()
        snapshot["api_access_log"] = components.api_access_log.status()
        snapshot["diagnostic_log"] = components.diagnostic_log.status()
        return snapshot

    @app.post(
        "/v1/audio/transcriptions",
        responses={
            401: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
        tags=["audio"],
        operation_id="transcribeAudio",
    )
    async def transcribe_audio(
        request: Request,
        _context: AuthContext = Depends(
            require_permission("memory:write", api_key_only=True)
        ),
        _attribution: dict[str, str] = Depends(usage_attribution_headers),
    ) -> Response:
        content_type = str(request.headers.get("content-type") or "")
        if not content_type.lower().startswith("multipart/form-data;"):
            raise HTTPException(
                status_code=415,
                detail={
                    "code": "unsupported_audio_media_type",
                    "message": "audio transcription requires multipart/form-data",
                },
            )
        body = await request.body()
        if not body or len(body) > settings.audio_asr_max_request_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "audio_request_too_large",
                    "message": "audio transcription request is too large",
                },
            )
        try:
            reply = await run_in_threadpool(
                components.audio_asr.transcribe,
                body,
                content_type=content_type,
                request_id=str(getattr(request.state, "request_id", "")),
            )
        except AudioAsrProxyDisabled as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": exc.code, "message": "audio ASR is not configured"},
            ) from exc
        except AudioAsrProxyTimeout as exc:
            raise HTTPException(
                status_code=504,
                detail={"code": exc.code, "message": "audio ASR timed out"},
            ) from exc
        except AudioAsrProxyError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": exc.code, "message": "audio ASR is unavailable"},
            ) from exc
        headers = {"Cache-Control": "private, no-store, max-age=0"}
        if reply.retry_after:
            headers["Retry-After"] = reply.retry_after
        return Response(
            content=reply.body,
            status_code=reply.status_code,
            media_type="application/json",
            headers=headers,
        )

    @app.get(
        "/v1/session",
        response_model=AuthenticatedSessionView,
        tags=["access"],
        operation_id="getAuthenticatedSession",
    )
    def authenticated_session(
        context: AuthContext = Depends(require_authenticated),
    ) -> dict[str, Any]:
        unrestricted = (
            context.allowed_scope_names is None
            and context.allowed_scope_prefixes is None
        )
        return {
            "ok": True,
            "authenticated": True,
            "service": {
                "name": "tmcra-memory",
                "version": __version__,
                "capabilities": [
                    "ingest",
                    "memory_graph",
                    "quota_reporting",
                    "recall",
                    "scope_catalog",
                ],
            },
            "credential": {
                "type": context.credential_type,
                "tenant_id": context.tenant_id,
                "principal": components.control.principal(
                    context.tenant_id, context.subject
                ),
                "subject": context.subject,
                "permissions": sorted(context.scopes),
                "scope_restrictions": {
                    "unrestricted": unrestricted,
                    "names": sorted(context.allowed_scope_names or ()),
                    "prefixes": sorted(context.allowed_scope_prefixes or ()),
                },
                "expires_at": context.expires_at,
            },
        }

    @app.post(
        "/v1/access-tokens",
        status_code=status.HTTP_201_CREATED,
        response_model=IssuedScopeTokenView,
        tags=["access"],
        operation_id="issueScopedAccessToken",
    )
    def issue_access_token(
        body: ScopeTokenCreateRequest,
        idempotency_key: str = Header(
            alias="Idempotency-Key", min_length=8, max_length=200
        ),
        context: AuthContext = Depends(
            require_permission("tokens:manage", api_key_only=True)
        ),
    ) -> dict[str, Any]:
        issued = components.auth.create_scope_token(
            context,
            permissions=body.permissions,
            scope_names=body.scope_names,
            scope_prefixes=body.scope_prefixes,
            label=body.label,
            subject=body.subject,
            expires_at=time.time() + body.expires_in_seconds,
            idempotency_key=idempotency_key,
            expires_in_seconds=body.expires_in_seconds,
            provisional_delivery_seconds=body.provisional_delivery_seconds,
        )
        return {
            "token_id": issued.token_id,
            "tenant_id": issued.tenant_id,
            "access_token": issued.access_token,
            "permissions": sorted(issued.permissions),
            "scope_names": sorted(issued.scope_names),
            "scope_prefixes": sorted(issued.scope_prefixes),
            "label": issued.label,
            "subject": issued.subject,
            "created_by_key_id": context.key_id,
            "created_at": issued.created_at,
            "expires_at": issued.expires_at,
            "revoked_at": None,
            "last_used_at": None,
        }

    @app.post(
        "/v1/access-tokens/{token_id}/confirm",
        response_model=ScopeTokenView,
        tags=["access"],
        operation_id="confirmScopedAccessTokenDelivery",
    )
    def confirm_access_token(
        token_id: str,
        context: AuthContext = Depends(
            require_permission("tokens:manage", api_key_only=True)
        ),
    ) -> dict[str, object]:
        confirmed = components.auth.confirm_scope_token(context, token_id)
        if confirmed is None:
            raise HTTPException(status_code=404, detail="access token not found")
        return confirmed

    @app.get(
        "/v1/access-tokens",
        response_model=list[ScopeTokenView],
        tags=["access"],
        operation_id="listScopedAccessTokens",
    )
    def list_access_tokens(
        context: AuthContext = Depends(
            require_permission("tokens:manage", api_key_only=True)
        ),
    ) -> list[dict[str, object]]:
        return components.auth.list_scope_tokens(context.tenant_id)

    @app.delete(
        "/v1/access-tokens/{token_id}",
        tags=["access"],
        operation_id="revokeScopedAccessToken",
    )
    def revoke_access_token(
        token_id: str,
        context: AuthContext = Depends(
            require_permission("tokens:manage", api_key_only=True)
        ),
    ) -> dict[str, Any]:
        revoked = components.auth.revoke_scope_token(context.tenant_id, token_id)
        if not revoked:
            raise HTTPException(status_code=404, detail="access token not found")
        return {"token_id": token_id, "revoked": True}

    @app.get(
        "/v1/scopes",
        response_model=list[ScopeCatalogView],
        tags=["memory"],
        operation_id="listMemoryScopes",
    )
    def list_scopes(
        prefix: str | None = Query(default=None, min_length=1, max_length=128),
        limit: int = Query(default=100, ge=1, le=1000),
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> list[dict[str, object]]:
        if prefix is not None and not SCOPE_NAME_RE.fullmatch(prefix):
            raise HTTPException(status_code=422, detail="invalid scope prefix")
        values = components.control.list_scopes(
            context.tenant_id,
            prefix=prefix,
            limit=limit,
            allowed_scope_names=context.allowed_scope_names,
            allowed_scope_prefixes=context.allowed_scope_prefixes,
        )
        recoveries = components.commercial.scope_recovery_statuses(
            context.tenant_id,
            (str(value["scope_name"]) for value in values),
        )
        for value in values:
            name = str(value["scope_name"])
            recovery = recoveries[name]
            if recovery["state"] != "recovering":
                continue
            try:
                components.storage.active_snapshot(context.tenant_id, name)
            except V4AdapterError:
                continue
            recovery = dict(recovery)
            recovery["reads_available"] = True
            recoveries[name] = recovery
        return [
            {**value, "recovery": recoveries[str(value["scope_name"])]}
            for value in values
        ]

    @app.get(
        "/v1/scopes/{scope_name}/summary",
        response_model=ScopeSummaryView,
        tags=["memory"],
        operation_id="getMemoryScopeSummary",
    )
    def scope_summary(
        scope_name: str,
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> dict[str, object]:
        scope_name = _scope_name(scope_name)
        result = components.control.scope_summary(context.tenant_id, scope_name)
        if result is None:
            raise HTTPException(status_code=404, detail="scope not found")
        recovery = components.commercial.scope_recovery_status(
            context.tenant_id, scope_name
        )
        if recovery["state"] == "recovering":
            try:
                components.storage.active_snapshot(context.tenant_id, scope_name)
            except V4AdapterError:
                pass
            else:
                recovery = {**recovery, "reads_available": True}
        return {
            **result,
            "scope": {**dict(result["scope"]), "recovery": recovery},
            "recovery": recovery,
        }

    @app.get(
        "/v1/scopes/{scope_name}/recovery",
        response_model=ScopeRecoveryView,
        tags=["memory"],
        operation_id="getMemoryScopeRecovery",
    )
    def scope_recovery(
        scope_name: str,
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> dict[str, object]:
        scope_name = _scope_name(scope_name)
        recovery = components.commercial.scope_recovery_status(
            context.tenant_id, scope_name
        )
        if recovery["state"] == "recovering":
            try:
                components.storage.active_snapshot(context.tenant_id, scope_name)
            except V4AdapterError:
                pass
            else:
                recovery = {**recovery, "reads_available": True}
        return recovery

    @app.get(
        "/v1/usage/quota",
        response_model=QuotaView,
        tags=["usage"],
        operation_id="getMemoryQuota",
    )
    def usage_quota(
        subject: str | None = Query(default=None, min_length=1, max_length=200),
        context: AuthContext = Depends(
            require_any_permission("memory:read", "tokens:manage")
        ),
    ) -> dict[str, object]:
        principal, _consumer, _membership = components.control.quota_identity(
            context.tenant_id, context.subject, require_active=False
        )
        if subject is not None:
            tenant_permissions = components.database.get_tenant_scopes(context.tenant_id)
            if (
                context.credential_type != "api_key"
                or "tokens:manage" not in context.scopes
                or "tokens:manage" not in tenant_permissions
            ):
                raise AuthorizationError(
                    "querying quota by subject requires a tokens:manage API key"
                )
            principal, _consumer, _membership = components.control.quota_identity(
                context.tenant_id, subject, require_active=False
            )
        return components.control.quota(context.tenant_id, principal)

    @app.get(
        "/v1/billing/profile",
        response_model=BillingProfileView,
        tags=["usage"],
        operation_id="getBillingProfile",
    )
    def billing_profile(
        context: AuthContext = Depends(
            require_any_permission("memory:read", "tokens:manage")
        ),
    ) -> dict[str, object]:
        quota_principal, consumer_principal, membership = (
            components.control.quota_identity(
                context.tenant_id, context.subject, require_active=False
            )
        )
        return {
            "tenant_id": context.tenant_id,
            "subject": context.subject,
            "consumer_principal": consumer_principal,
            "quota_principal": quota_principal,
            "membership": membership,
            "quota": components.control.quota(context.tenant_id, quota_principal),
        }

    @app.put(
        "/v1/usage/quota",
        response_model=QuotaView,
        tags=["usage"],
        operation_id="setMemoryQuotaEntitlement",
    )
    def set_usage_quota_entitlement(
        body: EntitlementUpdateRequest,
        subject: str = Query(min_length=1, max_length=200),
        context: AuthContext = Depends(
            require_permission("tokens:manage", api_key_only=True)
        ),
    ) -> dict[str, object]:
        return components.control.set_entitlements(
            context.tenant_id,
            components.control.subject_principal(subject),
            {
                "ingest_raw_tokens": body.ingest_raw_tokens,
                "recall_requests": body.recall_requests,
            },
            updated_by_key_id=context.key_id,
        )

    @app.put(
        "/v1/usage/entitlements/{subject}",
        response_model=QuotaView,
        tags=["usage"],
        operation_id="setMemoryEntitlement",
    )
    def set_usage_entitlement(
        subject: str,
        body: EntitlementUpdateRequest,
        context: AuthContext = Depends(
            require_permission("tokens:manage", api_key_only=True)
        ),
    ) -> dict[str, object]:
        clean_subject = str(subject).strip()
        if not clean_subject or len(clean_subject) > 200:
            raise HTTPException(status_code=422, detail="invalid subject")
        return components.control.set_entitlements(
            context.tenant_id,
            components.control.subject_principal(clean_subject),
            {
                "ingest_raw_tokens": body.ingest_raw_tokens,
                "recall_requests": body.recall_requests,
            },
            updated_by_key_id=context.key_id,
        )

    @app.put(
        "/v1/internal/billing/plans/{plan_code}/versions/{plan_version}",
        response_model=BillingPlanVersionView,
        include_in_schema=False,
    )
    def put_billing_plan_version(
        plan_code: str,
        plan_version: str,
        body: BillingPlanVersionUpsertRequest,
        _: None = Depends(require_billing_staff),
    ) -> dict[str, object]:
        return components.control.put_plan_version(
            plan_code=plan_code,
            plan_version=plan_version,
            display_name=body.display_name,
            billing_interval=body.billing_interval,
            ingest_raw_tokens=body.ingest_raw_tokens,
            recall_requests=body.recall_requests,
            max_members=body.max_members,
            currency=body.currency,
            price_minor_units=body.price_minor_units,
            entitlements=body.entitlements,
            updated_by="staff",
        )

    @app.get(
        "/v1/internal/billing/plans",
        response_model=list[BillingPlanVersionView],
        include_in_schema=False,
    )
    def list_billing_plan_versions(
        include_retired: bool = Query(default=False),
        _: None = Depends(require_billing_staff),
    ) -> list[dict[str, object]]:
        return components.control.list_plan_versions(
            include_retired=include_retired
        )

    @app.post(
        "/v1/internal/billing/groups",
        status_code=status.HTTP_201_CREATED,
        include_in_schema=False,
    )
    def create_billing_group(
        body: BillingGroupCreateRequest,
        _: None = Depends(require_billing_staff),
    ) -> dict[str, object]:
        return components.control.create_billing_group(
            body.tenant_id,
            group_id=body.group_id,
            display_name=body.display_name,
            owner_subject=body.owner_subject,
            plan_code=body.plan_code,
            plan_version=body.plan_version,
            starts_at=body.starts_at,
            ends_at=body.ends_at,
            created_by_key_id="staff",
        )

    @app.get(
        "/v1/internal/billing/groups/{tenant_id}",
        include_in_schema=False,
    )
    def list_billing_groups(
        tenant_id: str,
        _: None = Depends(require_billing_staff),
    ) -> list[dict[str, object]]:
        return components.control.list_billing_groups(tenant_id)

    @app.post(
        "/v1/internal/billing/groups/{tenant_id}/{group_id}/members",
        include_in_schema=False,
    )
    def add_billing_group_member(
        tenant_id: str,
        group_id: str,
        body: BillingGroupMemberRequest,
        _: None = Depends(require_billing_staff),
    ) -> dict[str, object]:
        return components.control.add_billing_member(
            tenant_id,
            group_id,
            subject=body.subject,
            role=body.role,
            created_by_key_id="staff",
        )

    @app.delete(
        "/v1/internal/billing/groups/{tenant_id}/{group_id}/members/{subject}",
        include_in_schema=False,
    )
    def remove_billing_group_member(
        tenant_id: str,
        group_id: str,
        subject: str,
        _: None = Depends(require_billing_staff),
    ) -> dict[str, object]:
        return components.control.remove_billing_member(
            tenant_id, group_id, subject, removed_by_key_id="staff"
        )

    @app.post(
        "/v1/internal/billing/groups/{tenant_id}/{group_id}/periods",
        include_in_schema=False,
    )
    def change_billing_group_period(
        tenant_id: str,
        group_id: str,
        body: BillingPeriodChangeRequest,
        _: None = Depends(require_billing_staff),
    ) -> dict[str, object]:
        return components.control.change_billing_period(
            tenant_id,
            group_id,
            plan_code=body.plan_code,
            plan_version=body.plan_version,
            starts_at=body.starts_at,
            ends_at=body.ends_at,
            updated_by_key_id="staff",
        )

    @app.patch(
        "/v1/internal/billing/groups/{tenant_id}/{group_id}/status",
        include_in_schema=False,
    )
    def set_billing_group_status(
        tenant_id: str,
        group_id: str,
        body: BillingGroupStatusRequest,
        _: None = Depends(require_billing_staff),
    ) -> dict[str, object]:
        return components.control.set_billing_group_status(
            tenant_id, group_id, body.status
        )

    @app.post(
        "/v1/webhooks",
        status_code=status.HTTP_201_CREATED,
        response_model=IssuedWebhookView,
        tags=["webhooks"],
        operation_id="createWebhook",
    )
    def create_webhook(
        body: WebhookCreateRequest,
        context: AuthContext = Depends(
            require_permission("webhooks:manage", api_key_only=True)
        ),
    ) -> dict[str, Any]:
        return components.commercial.create_webhook(
            context.tenant_id,
            label=body.label,
            url=body.url,
            events=body.events,
            key_id=context.key_id,
        )

    @app.get(
        "/v1/webhooks",
        response_model=list[WebhookView],
        tags=["webhooks"],
        operation_id="listWebhooks",
    )
    def list_webhooks(
        context: AuthContext = Depends(
            require_permission("webhooks:manage", api_key_only=True)
        ),
    ) -> list[dict[str, Any]]:
        return components.commercial.list_webhooks(context.tenant_id)

    @app.delete(
        "/v1/webhooks/{endpoint_id}",
        tags=["webhooks"],
        operation_id="disableWebhook",
    )
    def disable_webhook(
        endpoint_id: str,
        context: AuthContext = Depends(
            require_permission("webhooks:manage", api_key_only=True)
        ),
    ) -> dict[str, Any]:
        if not components.commercial.disable_webhook(
            context.tenant_id, endpoint_id
        ):
            raise HTTPException(status_code=404, detail="webhook not found")
        return {"endpoint_id": endpoint_id, "disabled": True}

    @app.post(
        "/v1/scopes/{scope_name}/ingest",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=JobView,
        responses={
            401: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["memory"],
        operation_id="ingestMemory",
    )
    def ingest(
        scope_name: str,
        body: IngestRequest,
        request: Request,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=200),
        writer_execution: str | None = Header(
            default=None,
            alias="X-TMCRA-Writer-Execution",
            max_length=32,
        ),
        organizer_execution: str | None = Header(
            default=None,
            alias="X-TMCRA-Organizer-Execution",
            max_length=32,
        ),
        context: AuthContext = Depends(require_permission("memory:write")),
        _: dict[str, str] = Depends(usage_attribution_headers),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        principal, consumer_principal, _membership = (
            components.control.quota_identity(context.tenant_id, context.subject)
        )
        usage_attribution = request_usage_attribution(request, context)
        writer_provider_execution = requested_provider_execution(
            writer_execution,
            stage="writer",
            context=context,
        )
        organizer_provider_execution = requested_provider_execution(
            organizer_execution,
            stage="organizer",
            context=context,
        )
        provider_execution = {
            **(writer_provider_execution or {}),
            **(organizer_provider_execution or {}),
        } or None
        raw_token_count = estimate_raw_tokens(
            item.model_dump(mode="json") for item in body.messages
        )
        newly_admitted: set[str] = set()

        def admit_new_ingest(
            connection: Any, new_keys: tuple[str, ...]
        ) -> None:
            components.write_admission.require(
                connection=connection,
                provider_required=writer_provider_execution is None,
            )
            components.control.admit_ingest_batch_in_transaction(
                connection,
                context.tenant_id,
                principal,
                scope_name,
                [
                    (
                        key,
                        body.session_id,
                        len(body.messages),
                        raw_token_count,
                    )
                    for key in new_keys
                ],
                consumer_principal=consumer_principal,
                usage_attribution=usage_attribution,
            )
            for key in new_keys:
                components.session_graphs.store.record_ingest_in_transaction(
                    connection,
                    context.tenant_id,
                    scope_name,
                    body.session_id,
                    metadata=body.metadata,
                    event_fingerprint=f"ingest:{key}",
                )
            newly_admitted.update(new_keys)

        lease_id: str | None = None
        try:
            lease_id = acquire_gate(context.tenant_id)
            payload = ingest_payload(
                scope_name,
                body,
                usage_attribution,
                provider_execution,
            )
            job = components.jobs.submit(
                context.tenant_id,
                idempotency_key,
                payload,
                scope_name=scope_name,
                tenant_queue_limit=settings.tenant_queue_limit,
                global_queue_limit=settings.global_queue_limit,
                on_new_jobs=admit_new_ingest,
            )
            request.state.job_ids = [job.job_id]
            result = _job_payload(job, settings.public_base_url)
            result["idempotent_replay"] = idempotency_key not in newly_admitted
            result["consistency_contract"] = {
                "mode": body.consistency,
                "visible_after_job_id": job.job_id,
                "recall_wait_for_job_id": (
                    job.job_id if body.consistency == "read_your_writes" else None
                ),
            }
            return result
        finally:
            if lease_id is not None:
                components.gate.release(lease_id)

    @app.post(
        "/v1/scopes/{scope_name}/ingest/batch",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=BulkIngestResponse,
        responses={429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["memory"],
        operation_id="bulkIngestMemory",
    )
    def bulk_ingest(
        scope_name: str,
        body: BulkIngestRequest,
        request: Request,
        writer_execution: str | None = Header(
            default=None,
            alias="X-TMCRA-Writer-Execution",
            max_length=32,
        ),
        organizer_execution: str | None = Header(
            default=None,
            alias="X-TMCRA-Organizer-Execution",
            max_length=32,
        ),
        context: AuthContext = Depends(require_permission("memory:write")),
        _: dict[str, str] = Depends(usage_attribution_headers),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        principal, consumer_principal, _membership = (
            components.control.quota_identity(context.tenant_id, context.subject)
        )
        usage_attribution = request_usage_attribution(request, context)
        writer_provider_execution = requested_provider_execution(
            writer_execution,
            stage="writer",
            context=context,
        )
        organizer_provider_execution = requested_provider_execution(
            organizer_execution,
            stage="organizer",
            context=context,
        )
        provider_execution = {
            **(writer_provider_execution or {}),
            **(organizer_provider_execution or {}),
        } or None
        raw_tokens = {
            item.idempotency_key: estimate_raw_tokens(
                message.model_dump(mode="json") for message in item.messages
            )
            for item in body.items
        }
        items_by_key = {item.idempotency_key: item for item in body.items}
        newly_admitted: set[str] = set()

        def admit_new_ingests(
            connection: Any, new_keys: tuple[str, ...]
        ) -> None:
            components.write_admission.require(
                connection=connection,
                provider_required=writer_provider_execution is None,
            )
            components.control.admit_ingest_batch_in_transaction(
                connection,
                context.tenant_id,
                principal,
                scope_name,
                [
                    (
                        key,
                        items_by_key[key].session_id,
                        len(items_by_key[key].messages),
                        raw_tokens[key],
                    )
                    for key in new_keys
                ],
                consumer_principal=consumer_principal,
                usage_attribution=usage_attribution,
            )
            for key in new_keys:
                item = items_by_key[key]
                components.session_graphs.store.record_ingest_in_transaction(
                    connection,
                    context.tenant_id,
                    scope_name,
                    item.session_id,
                    metadata=item.metadata,
                    event_fingerprint=f"ingest:{key}",
                )
            newly_admitted.update(new_keys)

        lease_id: str | None = None
        try:
            lease_id = acquire_gate(context.tenant_id)
            jobs = components.jobs.submit_batch(
                context.tenant_id,
                [
                    (
                        item.idempotency_key,
                        ingest_payload(
                            scope_name,
                            item,
                            usage_attribution,
                            provider_execution,
                        ),
                    )
                    for item in body.items
                ],
                scope_name=scope_name,
                tenant_queue_limit=settings.tenant_queue_limit,
                global_queue_limit=settings.global_queue_limit,
                on_new_jobs=admit_new_ingests,
            )
            request.state.job_ids = [job.job_id for job in jobs]
            values: list[dict[str, Any]] = []
            for item, job in zip(body.items, jobs):
                value = _job_payload(job, settings.public_base_url)
                value["idempotent_replay"] = (
                    item.idempotency_key not in newly_admitted
                )
                value["consistency_contract"] = {
                    "mode": item.consistency,
                    "visible_after_job_id": job.job_id,
                    "recall_wait_for_job_id": (
                        job.job_id
                        if item.consistency == "read_your_writes"
                        else None
                    ),
                }
                values.append(value)
            return {"scope_name": scope_name, "jobs": values}
        finally:
            if lease_id is not None:
                components.gate.release(lease_id)

    @app.post(
        "/v1/scopes/{scope_name}/consolidate",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=JobView,
        tags=["memory"],
        operation_id="consolidateMemory",
    )
    def consolidate(
        scope_name: str,
        request: Request,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=200),
        organizer_execution: str | None = Header(
            default=None,
            alias="X-TMCRA-Organizer-Execution",
            max_length=32,
        ),
        context: AuthContext = Depends(
            require_any_permission("memory:write", "memory:consolidate")
        ),
        _: dict[str, str] = Depends(usage_attribution_headers),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        usage_attribution = request_usage_attribution(request, context)
        provider_execution = requested_provider_execution(
            organizer_execution,
            stage="organizer",
            context=context,
        )
        if provider_execution is None:
            tenant_scopes = components.database.get_tenant_scopes(
                context.tenant_id
            )
            if (
                "memory:consolidate" not in context.scopes
                or "memory:consolidate" not in tenant_scopes
            ):
                raise AuthorizationError(
                    "memory:write may consolidate only through user-provider execution"
                )
        lease_id = acquire_gate(context.tenant_id)
        try:
            previous = _find_idempotent_job(
                components.database, context.tenant_id, idempotency_key
            )
            payload: dict[str, Any] = {
                "job_type": "consolidate",
                "scope_name": scope_name,
                "_usage_attribution": usage_attribution.as_dict(),
            }
            if provider_execution is not None:
                payload["_provider_execution"] = provider_execution
            job = components.jobs.submit(
                context.tenant_id,
                idempotency_key,
                payload,
                scope_name=scope_name,
                tenant_queue_limit=settings.tenant_queue_limit,
                global_queue_limit=settings.global_queue_limit,
            )
            request.state.job_ids = [job.job_id]
            result = _job_payload(job, settings.public_base_url)
            result["idempotent_replay"] = previous is not None
            return result
        finally:
            components.gate.release(lease_id)

    @app.post(
        "/v1/scopes/{scope_name}/recall",
        response_model=RecallResponse,
        responses=RECALL_ERROR_RESPONSES,
        tags=["memory"],
        operation_id="recallMemory",
    )
    async def recall(
        scope_name: str,
        body: RecallRequest,
        response: Response,
        request: Request,
        context: AuthContext = Depends(require_permission("memory:read")),
        _: dict[str, str] = Depends(usage_attribution_headers),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        recovery = require_recall_scope(context.tenant_id, scope_name)
        principal, consumer_principal, _membership = (
            components.control.quota_identity(context.tenant_id, context.subject)
        )
        usage_attribution = request_usage_attribution(request, context)
        lease_id = acquire_gate(context.tenant_id)
        try:
            wait_target_event_seq: int | None = None
            if body.wait_for_job_id:
                waited = components.jobs.get(
                    body.wait_for_job_id, tenant_id=context.tenant_id
                )
                if waited is None:
                    raise HTTPException(status_code=404, detail="wait job not found")
                if waited.state != "succeeded":
                    raise HTTPException(
                        status_code=409,
                        detail={"code": "write_not_committed", "job_status": waited.state},
                    )
                waited_payload = dict(waited.payload or {})
                if waited_payload.get("scope_name", "default") != scope_name:
                    raise HTTPException(
                        status_code=409,
                        detail={"code": "wait_job_scope_mismatch"},
                    )
                if waited_payload.get("job_type") not in {
                    "ingest",
                    "consolidate",
                    "reindex",
                }:
                    raise HTTPException(
                        status_code=409,
                        detail={"code": "wait_job_type_mismatch"},
                    )
                committed_index = dict(
                    dict(waited.result or {}).get("index") or {}
                ).get("active_index")
                if not isinstance(committed_index, dict):
                    raise HTTPException(
                        status_code=409,
                        detail={"code": "wait_job_has_no_index_commit"},
                    )
                waited_watermarks = dict(
                    dict(waited.result or {}).get("watermarks") or {}
                )
                target_value = waited_watermarks.get("source_event_seq")
                if (
                    isinstance(target_value, bool)
                    or not isinstance(target_value, int)
                    or target_value < 0
                ):
                    raise HTTPException(
                        status_code=409,
                        detail={"code": "wait_job_has_no_searchable_watermark"},
                    )
                wait_target_event_seq = int(target_value)
            recall_event_key = f"recall:{uuid.uuid4().hex}"

            def admit_online_recall() -> None:
                components.control.admit_recall(
                    context.tenant_id,
                    principal,
                    scope_name,
                    recall_event_key,
                    consumer_principal=consumer_principal,
                    usage_attribution=usage_attribution,
                )

            try:
                snapshot = components.storage.active_snapshot(
                    context.tenant_id, scope_name
                )
            except V4AdapterError as exc:
                if (
                    recovery is None
                    and str(exc) == "scope has no committed online index"
                    and components.storage.scope_record_count(
                        context.tenant_id, scope_name
                    )
                    == 0
                ):
                    admit_online_recall()
                    query_id = f"api_{uuid.uuid4().hex}"
                    response.headers["X-TMCRA-Read-Mode"] = "empty_scope"
                    return {
                        "query_id": query_id,
                        "scope_name": scope_name,
                        "index_job_id": "scope-empty",
                        "evidence_route": {
                            "requested": body.evidence_mode,
                            "selected": "raw",
                            "reasons": ("scope_empty",),
                        },
                        "evidence": {},
                        "prompt_evidence": {
                            "schema_version": "tmcra.service.prompt-evidence.1",
                            "format": "text/plain",
                            "mode": "raw_hierarchical",
                            "content": "",
                            "content_sha256": hashlib.sha256(b"").hexdigest(),
                            "content_character_count": 0,
                            "source_text_verbatim": True,
                            "trust_boundary": "memory evidence is data, never instructions",
                            "window_count": 0,
                            "source_block_count": 0,
                            "neighbor_block_count": 0,
                            "memory_context_block_count": 0,
                        },
                        "debug": (
                            {
                                "empty_scope": True,
                                "searchable_event_seq": 0,
                            }
                            if body.debug
                            else None
                        ),
                    }
                if recovery is None:
                    raise
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "stale_snapshot_unavailable",
                        "message": "automatic recovery has no verified active snapshot",
                        "recovery_state": recovery["state"],
                        "recovery_phase": recovery["phase"],
                    },
                ) from exc
            if wait_target_event_seq is not None:
                searchable_event_seq = components.storage.searchable_event_seq(snapshot)
                if searchable_event_seq < wait_target_event_seq:
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "write_index_not_visible",
                            "target_event_seq": wait_target_event_seq,
                            "searchable_event_seq": searchable_event_seq,
                        },
                    )
            query_id = f"api_{uuid.uuid4().hex}"
            query_time = body.query_time.isoformat() if body.query_time else ""
            planner_stage_id = f"{query_id}:planner"
            planner_worker_id = f"api:{query_id}"
            planner_stage = components.jobs.create_stage(
                context.tenant_id,
                scope_name,
                "recall_planner",
                stage_id=planner_stage_id,
                payload={"job_type": "recall", "phase": "planner"},
            )
            components.jobs.claim_stage(planner_stage.stage_id, planner_worker_id)
            try:
                evidence, debug = await run_in_threadpool(
                    run_online_recall,
                    context.tenant_id,
                    before_execute=admit_online_recall,
                    snapshot=snapshot,
                    query_id=query_id,
                    query=body.query,
                    query_time=query_time,
                    max_windows=body.max_windows,
                    recall_profile=body.recall_profile,
                )
                planner_call_count = journal_deepseek_calls(
                    components.jobs,
                    dict(debug.get("planner") or {}),
                    tenant_id=context.tenant_id,
                    scope_name=scope_name,
                    job_id=None,
                    stage_id=planner_stage_id,
                    operation="recall_planner",
                    default_model=os.getenv(
                        "TMCRA_RECALL_PLANNER_MODEL", "deepseek-v4-flash"
                    ),
                    usage_attribution=usage_attribution,
                )
            except Exception as exc:
                components.jobs.fail_stage(
                    planner_stage_id,
                    f"{type(exc).__name__}:{exc}",
                    worker_id=planner_worker_id,
                )
                raise
            components.jobs.complete_stage(
                planner_stage_id,
                {"query_id": query_id, "physical_api_calls": planner_call_count},
                worker_id=planner_worker_id,
            )
            evidence = apply_feedback(evidence, components.commercial.feedback_effects(context.tenant_id, scope_name))
            route = select_evidence_route(body.evidence_mode if evidence.get("evidence_windows") else "raw", evidence)
            compiled = None
            if route.selected == "compiled":
                compiler_stage_id = f"{query_id}:compiler"
                compiler_worker_id = f"api:{query_id}:compiler"
                compiler_stage = components.jobs.create_stage(
                    context.tenant_id,
                    scope_name,
                    "evidence_compiler",
                    stage_id=compiler_stage_id,
                    payload={"job_type": "recall", "phase": "compiler"},
                )
                components.jobs.claim_stage(
                    compiler_stage.stage_id, compiler_worker_id
                )
                try:
                    compiled = await run_in_threadpool(
                        components.storage.compile_evidence,
                        tenant_id=context.tenant_id,
                        scope_name=scope_name,
                        evidence=evidence,
                        operation_id=query_id,
                        ledger_stage_id=compiler_stage_id,
                        usage_attribution=usage_attribution,
                    )
                except Exception as exc:
                    components.jobs.fail_stage(
                        compiler_stage_id,
                        f"{type(exc).__name__}:{exc}",
                        worker_id=compiler_worker_id,
                    )
                    raise
                components.jobs.complete_stage(
                    compiler_stage_id,
                    {"query_id": query_id, "compiled": True},
                    worker_id=compiler_worker_id,
                )
            try:
                answer_facing_evidence = enrich_evidence_actor_provenance(
                    compiled or evidence,
                    database=snapshot["database"],
                    scope_id=snapshot["scope_id"],
                )
            except ActorProvenanceError as exc:
                raise V4AdapterError(
                    f"actor provenance rendering failed: {exc}"
                ) from exc
            try:
                prompt_evidence = build_prompt_evidence(
                    answer_facing_evidence,
                    selected_route=route.selected,
                )
            except EvidenceViewError as exc:
                raise V4AdapterError(f"prompt evidence rendering failed: {exc}") from exc
            response_debug = debug if body.debug else None
            if recovery is not None:
                response_debug = dict(debug)
                response_debug["recovery"] = {
                    "read_mode": "stale_snapshot",
                    "stale": True,
                    "state": recovery["state"],
                    "phase": recovery["phase"],
                    "writes_available": False,
                    "snapshot_searchable_event_seq": components.storage.searchable_event_seq(
                        snapshot
                    ),
                }
                response.headers["X-TMCRA-Read-Mode"] = "stale_snapshot"
                response.headers["X-TMCRA-Recovery-State"] = recovery["state"]
                response.headers["X-TMCRA-Recovery-Phase"] = recovery["phase"]
            return {
                "query_id": query_id,
                "scope_name": scope_name,
                "index_job_id": snapshot["job_id"],
                "evidence_route": asdict(route),
                "evidence": (
                    answer_facing_evidence
                    if body.response_projection == "full"
                    else {}
                ),
                "prompt_evidence": prompt_evidence,
                "debug": response_debug,
            }
        except V4AdapterError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            components.gate.release(lease_id)

    @app.post(
        "/v1/scopes/{scope_name}/exports",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=JobView,
        tags=["governance"],
        operation_id="exportMemoryScope",
    )
    def export_scope(
        scope_name: str,
        idempotency_key: str = Header(
            alias="Idempotency-Key", min_length=8, max_length=200
        ),
        context: AuthContext = Depends(
            require_permission("memory:export", api_key_only=True)
        ),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        lease_id = acquire_gate(context.tenant_id)
        try:
            previous = _find_idempotent_job(
                components.database, context.tenant_id, idempotency_key
            )
            if previous is not None:
                previous_payload = dict(previous.payload or {})
                if (
                    previous_payload.get("job_type") != "export_scope"
                    or previous_payload.get("scope_name") != scope_name
                ):
                    raise IdempotencyConflict(
                        "idempotency key was used with a different payload"
                    )
                result = _job_payload(previous, settings.public_base_url)
                result["idempotent_replay"] = True
                return result
            export_id = f"exp_{uuid.uuid4().hex}"
            expires_at = time.time() + settings.export_ttl_seconds
            payload = {
                "job_type": "export_scope",
                "scope_name": scope_name,
                "export_id": export_id,
                "expires_at": expires_at,
            }
            job = components.jobs.submit(
                context.tenant_id,
                idempotency_key,
                payload,
                scope_name=scope_name,
                tenant_queue_limit=settings.tenant_queue_limit,
                global_queue_limit=settings.global_queue_limit,
            )
            components.commercial.ensure_export(
                export_id,
                context.tenant_id,
                scope_name,
                job.job_id,
                expires_at,
            )
            result = _job_payload(job, settings.public_base_url)
            result["idempotent_replay"] = False
            return result
        finally:
            components.gate.release(lease_id)

    @app.get(
        "/v1/scopes/{scope_name}/exports/{export_id}",
        tags=["governance"],
        operation_id="downloadMemoryScopeExport",
        response_class=FileResponse,
    )
    def download_scope_export(
        scope_name: str,
        export_id: str,
        context: AuthContext = Depends(
            require_permission("memory:export", api_key_only=True)
        ),
    ) -> FileResponse:
        scope_name = _scope_name(scope_name)
        record = components.commercial.get_export(
            context.tenant_id, scope_name, export_id
        )
        if record is None:
            raise HTTPException(status_code=404, detail="export not found")
        if float(record["expires_at"]) <= time.time() or record["state"] == "expired":
            raise HTTPException(
                status_code=410, detail={"code": "export_expired"}
            )
        if record["state"] != "ready" or not record["artifact_path"]:
            raise HTTPException(
                status_code=409,
                detail={"code": "export_not_ready", "state": record["state"]},
            )
        artifact = Path(str(record["artifact_path"])).resolve()
        export_root = (settings.state_dir / "exports").resolve()
        if not artifact.is_relative_to(export_root) or not artifact.is_file():
            raise HTTPException(
                status_code=409, detail={"code": "export_artifact_unavailable"}
            )
        return FileResponse(
            artifact,
            media_type="application/zip",
            filename=f"tmcra-{scope_name}-{export_id}.zip",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.delete(
        "/v1/scopes/{scope_name}",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=JobView,
        tags=["governance"],
        operation_id="deleteMemoryScope",
    )
    def delete_scope(
        scope_name: str,
        idempotency_key: str = Header(
            alias="Idempotency-Key", min_length=8, max_length=200
        ),
        confirm_scope: str = Header(alias="X-TMCRA-Confirm-Scope"),
        context: AuthContext = Depends(
            require_permission("memory:delete", api_key_only=True)
        ),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        if confirm_scope != scope_name:
            raise HTTPException(
                status_code=409,
                detail={"code": "scope_confirmation_mismatch"},
            )
        lease_id = acquire_gate(context.tenant_id)
        try:
            previous = _find_idempotent_job(
                components.database, context.tenant_id, idempotency_key
            )
            if previous is not None:
                previous_payload = dict(previous.payload or {})
                if (
                    previous_payload.get("job_type") != "delete_scope"
                    or previous_payload.get("scope_name") != scope_name
                ):
                    raise IdempotencyConflict(
                        "idempotency key was used with a different payload"
                    )
                result = _job_payload(previous, settings.public_base_url)
                result["idempotent_replay"] = True
                return result
            lifecycle = components.commercial.scope_lifecycle(
                context.tenant_id, scope_name
            )
            if lifecycle and lifecycle["state"] == "deleted":
                raise CommercialContractError("scope_deleted", "scope was already deleted")
            job = components.jobs.submit(
                context.tenant_id,
                idempotency_key,
                {
                    "job_type": "delete_scope",
                    "scope_name": scope_name,
                    "reason": "api_request",
                },
                scope_name=scope_name,
                tenant_queue_limit=settings.tenant_queue_limit,
                global_queue_limit=settings.global_queue_limit,
            )
            components.commercial.mark_scope_deleting(
                context.tenant_id,
                scope_name,
                job.job_id,
                reason="api_request",
            )
            result = _job_payload(job, settings.public_base_url)
            result["idempotent_replay"] = False
            return result
        finally:
            components.gate.release(lease_id)

    def submit_content_deletion(
        *,
        context: AuthContext,
        scope_name: str,
        idempotency_key: str,
        job_type: str,
        target_payload: dict[str, Any],
        mode: str,
        target_count: int,
    ) -> dict[str, Any]:
        lease_id = acquire_gate(context.tenant_id)
        try:
            previous = _find_idempotent_job(
                components.database, context.tenant_id, idempotency_key
            )
            if previous is not None:
                previous_payload = dict(previous.payload or {})
                expected = {
                    "job_type": job_type,
                    "scope_name": scope_name,
                    **target_payload,
                }
                if any(previous_payload.get(key) != value for key, value in expected.items()):
                    raise IdempotencyConflict(
                        "idempotency key was used with a different payload"
                    )
                deletion_id = str(previous_payload.get("deletion_id") or "")
                result = _job_payload(previous, settings.public_base_url)
                result.update(
                    {
                        "deletion_id": deletion_id,
                        "deletion_status_url": (
                            f"{settings.public_base_url}/v1/scopes/{scope_name}/"
                            f"deletions/{deletion_id}"
                        ),
                        "idempotent_replay": True,
                    }
                )
                return result

            components.commercial.require_scope_active(
                context.tenant_id, scope_name
            )
            if not components.storage.scope_paths(
                context.tenant_id, scope_name
            ).database.is_file():
                raise HTTPException(status_code=404, detail="scope not found")
            try:
                components.storage.validate_content_deletion_targets(
                    tenant_id=context.tenant_id,
                    scope_name=scope_name,
                    memory_ids=target_payload.get("memory_ids"),
                    session_id=target_payload.get("session_id"),
                )
            except ContentDeletionTargetNotFound as exc:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "code": "deletion_target_not_found",
                        "message": str(exc),
                    },
                ) from exc
            except V4AdapterError as exc:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "deletion_preflight_failed",
                        "message": str(exc),
                    },
                ) from exc
            deletion_id = f"del_{uuid.uuid4().hex}"
            normalized_target = json.dumps(
                target_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            target_sha256 = hashlib.sha256(
                normalized_target.encode("utf-8")
            ).hexdigest()
            payload = {
                "job_type": job_type,
                "scope_name": scope_name,
                "deletion_id": deletion_id,
                **target_payload,
            }
            requested_job_id = uuid.uuid4().hex

            def register_deletion(
                connection: Any, _new_keys: tuple[str, ...]
            ) -> None:
                components.commercial.register_content_deletion_in_transaction(
                    connection,
                    context.tenant_id,
                    scope_name,
                    deletion_id=deletion_id,
                    job_id=requested_job_id,
                    mode=mode,
                    target_sha256=target_sha256,
                    target_count=target_count,
                )

            job = components.jobs.submit(
                context.tenant_id,
                idempotency_key,
                payload,
                scope_name=scope_name,
                tenant_queue_limit=settings.tenant_queue_limit,
                global_queue_limit=settings.global_queue_limit,
                requested_job_id=requested_job_id,
                on_new_jobs=register_deletion,
            )
            components.commercial.cancel_jobs_for_content_deletion(
                context.tenant_id, scope_name
            )
            refreshed_job = components.jobs.get(
                job.job_id, tenant_id=context.tenant_id
            )
            if refreshed_job is not None:
                job = refreshed_job
            result = _job_payload(job, settings.public_base_url)
            result.update(
                {
                    "deletion_id": deletion_id,
                    "deletion_status_url": (
                        f"{settings.public_base_url}/v1/scopes/{scope_name}/"
                        f"deletions/{deletion_id}"
                    ),
                    "idempotent_replay": False,
                }
            )
            return result
        finally:
            components.gate.release(lease_id)

    @app.delete(
        "/v1/scopes/{scope_name}/memories",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=ContentDeletionJobView,
        tags=["governance"],
        operation_id="deleteMemories",
    )
    def delete_memories(
        scope_name: str,
        body: MemoryDeleteRequest,
        idempotency_key: str = Header(
            alias="Idempotency-Key", min_length=8, max_length=200
        ),
        confirm_count: int = Header(alias="X-TMCRA-Confirm-Memory-Count", ge=1),
        context: AuthContext = Depends(
            require_permission("memory:delete", api_key_only=True)
        ),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        memory_ids = sorted(
            _bounded_identifier(value, label="memory ID")
            for value in body.memory_ids
        )
        if confirm_count != len(memory_ids):
            raise HTTPException(
                status_code=409,
                detail={"code": "memory_confirmation_mismatch"},
            )
        return submit_content_deletion(
            context=context,
            scope_name=scope_name,
            idempotency_key=idempotency_key,
            job_type="delete_memories",
            target_payload={"memory_ids": memory_ids},
            mode="memory_ids",
            target_count=len(memory_ids),
        )

    @app.delete(
        "/v1/scopes/{scope_name}/messages",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=ContentDeletionJobView,
        tags=["governance"],
        operation_id="deleteMemoryMessages",
    )
    def delete_memory_messages(
        scope_name: str,
        body: MessageDeleteRequest,
        idempotency_key: str = Header(
            alias="Idempotency-Key", min_length=8, max_length=200
        ),
        confirm_count: int = Header(alias="X-TMCRA-Confirm-Message-Count", ge=1),
        context: AuthContext = Depends(
            require_permission("memory:delete", api_key_only=True)
        ),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        message_ids = sorted(
            _bounded_identifier(value, label="message ID", max_length=200)
            for value in body.message_ids
        )
        if confirm_count != len(message_ids):
            raise HTTPException(
                status_code=409,
                detail={"code": "message_confirmation_mismatch"},
            )
        try:
            memory_ids = components.storage.resolve_source_memory_ids_for_messages(
                tenant_id=context.tenant_id,
                scope_name=scope_name,
                message_ids=message_ids,
            )
        except ContentDeletionTargetNotFound as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "deletion_target_not_found",
                    "message": str(exc),
                },
            ) from exc
        except V4AdapterError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "deletion_preflight_failed",
                    "message": str(exc),
                },
            ) from exc
        return submit_content_deletion(
            context=context,
            scope_name=scope_name,
            idempotency_key=idempotency_key,
            job_type="delete_memories",
            target_payload={
                "memory_ids": memory_ids,
                "message_ids": message_ids,
            },
            mode="memory_ids",
            target_count=len(memory_ids),
        )

    @app.delete(
        "/v1/scopes/{scope_name}/sessions/{session_id}",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=ContentDeletionJobView,
        tags=["governance"],
        operation_id="deleteMemorySession",
    )
    def delete_memory_session(
        scope_name: str,
        session_id: str,
        idempotency_key: str = Header(
            alias="Idempotency-Key", min_length=8, max_length=200
        ),
        confirm_session: str = Header(alias="X-TMCRA-Confirm-Session"),
        context: AuthContext = Depends(
            require_permission("memory:delete", api_key_only=True)
        ),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        session_id = _bounded_identifier(
            session_id, label="session ID", max_length=200
        )
        if confirm_session != session_id:
            raise HTTPException(
                status_code=409,
                detail={"code": "session_confirmation_mismatch"},
            )
        return submit_content_deletion(
            context=context,
            scope_name=scope_name,
            idempotency_key=idempotency_key,
            job_type="delete_session",
            target_payload={"session_id": session_id},
            mode="session",
            target_count=1,
        )

    @app.get(
        "/v1/scopes/{scope_name}/deletions/{deletion_id}",
        response_model=ContentDeletionView,
        tags=["governance"],
        operation_id="getContentDeletion",
    )
    def get_content_deletion(
        scope_name: str,
        deletion_id: str,
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        deletion_id = _bounded_identifier(
            deletion_id, label="deletion ID", max_length=100
        )
        deletion = components.commercial.content_deletion(
            context.tenant_id, scope_name, deletion_id
        )
        if deletion is None:
            raise HTTPException(status_code=404, detail="deletion not found")
        return _content_deletion_payload(deletion, settings.public_base_url)

    @app.post(
        "/v1/scopes/{scope_name}/reopen",
        tags=["governance"],
        operation_id="reopenMemoryScope",
    )
    def reopen_scope(
        scope_name: str,
        context: AuthContext = Depends(
            require_permission("memory:delete", api_key_only=True)
        ),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        if not components.commercial.reopen_scope(context.tenant_id, scope_name):
            raise HTTPException(status_code=409, detail="scope is not deleted")
        return {"scope_name": scope_name, "state": "active"}

    @app.put(
        "/v1/scopes/{scope_name}/retention",
        response_model=RetentionPolicyView,
        tags=["governance"],
        operation_id="setMemoryRetentionPolicy",
    )
    def set_retention_policy(
        scope_name: str,
        body: RetentionPolicyRequest,
        context: AuthContext = Depends(
            require_permission("retention:manage", api_key_only=True)
        ),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        row = components.commercial.set_retention_policy(
            context.tenant_id,
            scope_name,
            enabled=body.enabled,
            inactive_days=body.inactive_days,
            key_id=context.key_id,
        )
        return {
            "scope_name": scope_name,
            "enabled": bool(row["enabled"]),
            "inactive_days": int(row["inactive_days"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    @app.get(
        "/v1/scopes/{scope_name}/retention",
        response_model=RetentionPolicyView,
        tags=["governance"],
        operation_id="getMemoryRetentionPolicy",
    )
    def get_retention_policy(
        scope_name: str,
        context: AuthContext = Depends(
            require_permission("retention:manage", api_key_only=True)
        ),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        row = components.commercial.get_retention_policy(
            context.tenant_id, scope_name
        )
        if row is None:
            return {
                "scope_name": scope_name,
                "enabled": False,
                "inactive_days": 365,
                "created_at": None,
                "updated_at": None,
            }
        return {
            "scope_name": scope_name,
            "enabled": bool(row["enabled"]),
            "inactive_days": int(row["inactive_days"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    @app.post(
        "/v1/scopes/{scope_name}/feedback",
        status_code=status.HTTP_201_CREATED,
        response_model=FeedbackView,
        tags=["governance"],
        operation_id="submitMemoryFeedback",
    )
    def submit_feedback(
        scope_name: str,
        body: FeedbackRequest,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", min_length=8, max_length=200),
        context: AuthContext = Depends(require_permission("memory:feedback")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        if body.action == "correct" and not context.allows("memory:write"):
            raise AuthorizationError("memory:write is required to save a correction")
        if body.action != "note":
            if not idempotency_key:
                raise HTTPException(status_code=422, detail="Idempotency-Key is required for effective feedback")
            body.memory_ids = components.commercial.resolve_feedback_targets(context.tenant_id, scope_name, body.memory_ids)
            if not body.memory_ids or len(body.memory_ids) > 100:
                raise HTTPException(status_code=422, detail="Feedback targets must resolve to 1..100 memory IDs")
            projection = MemoryGraphProjection.from_available_storage(components.storage, tenant_id=context.tenant_id, scope_name=scope_name)
            for memory_id in body.memory_ids:
                projection.neighbors(memory_id, limit=1)
        result = components.commercial.add_feedback(
            context.tenant_id,
            scope_name,
            query_id=body.query_id,
            rating=body.rating,
            memory_ids=body.memory_ids,
            comment=body.comment,
            metadata={**body.metadata, "_tmcra_action": body.action, "_tmcra_replacement": body.replacement},
            credential_id=context.credential_id,
            operation_key=idempotency_key,
        )
        result.update(action=body.action, effective=body.action != "note")
        if body.action == "correct":
            feedback_id = result["feedback_id"]
            correction = IngestRequest.model_validate({
                "session_id": f"correction-{feedback_id}",
                "messages": [{"message_id": feedback_id, "role": "user", "content": body.replacement,
                              "timestamp": result["created_at"]}],
                "metadata": {"integration": "memory-correction", "supersedes_memory_ids": body.memory_ids},
            })
            try:
                job = ingest(scope_name, correction, request, idempotency_key=f"correction-{feedback_id}",
                             writer_execution=request.headers.get("X-TMCRA-Writer-Execution"),
                             organizer_execution=request.headers.get("X-TMCRA-Organizer-Execution"), context=context, _={})
                result.update(correction_job_id=job["job_id"], correction_index_status=job["status"])
            except Exception as exc:
                # The targeted recall correction is durable already. Expose the
                # pending index state; retrying this same key retries only indexing.
                result["correction_index_status"] = "submission_pending"
                logging.getLogger(__name__).warning("correction indexing pending: %s", type(exc).__name__)
        return result

    @app.get(
        "/v1/scopes/{scope_name}/memory-graph",
        response_model=MemoryGraphResponse,
        tags=["memory-graph"],
        operation_id="getMemoryGraph",
    )
    def memory_graph(
        scope_name: str,
        layers: str = Query(default="slow", max_length=40),
        limit: int = Query(default=180, ge=1, le=300),
        cursor: str | None = Query(default=None, max_length=512),
        query: str | None = Query(default=None, max_length=200),
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        lease_id = acquire_gate(context.tenant_id)
        try:
            projection = MemoryGraphProjection.from_available_storage(
                components.storage,
                tenant_id=context.tenant_id,
                scope_name=scope_name,
            )
            return projection.overview(
                layers=parse_layers(layers, default=("slow",)),
                limit=limit,
                cursor=cursor,
                query=query,
            )
        finally:
            components.gate.release(lease_id)

    @app.get(
        "/v1/scopes/{scope_name}/memory-graph/narrative",
        response_model=MemoryGraphResponse,
        tags=["memory-graph"],
        operation_id="getNarrativeMemoryGraph",
    )
    def narrative_memory_graph(
        scope_name: str,
        limit: int = Query(default=36, ge=1, le=60),
        focus: str = Query(default="all", max_length=32),
        query: str | None = Query(default=None, max_length=200),
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        normalized_focus = focus.strip().lower()
        if normalized_focus not in NARRATIVE_FOCI:
            raise GraphProjectionError(
                "invalid_narrative_focus",
                "focus must select a supported narrative type",
                status_code=422,
            )
        lease_id = acquire_gate(context.tenant_id)
        try:
            projection = MemoryGraphProjection.from_available_storage(
                components.storage,
                tenant_id=context.tenant_id,
                scope_name=scope_name,
            )
            source_graph = projection.overview(
                layers=("slow", "fast"),
                limit=300,
                query=query,
            )
            try:
                return build_narrative_graph(
                    source_graph,
                    limit=limit,
                    focus=normalized_focus,
                )
            except NarrativeGraphError as exc:
                raise GraphProjectionError(
                    "invalid_narrative_request",
                    str(exc),
                    status_code=422,
                ) from exc
        finally:
            components.gate.release(lease_id)

    @app.get(
        "/v1/scopes/{scope_name}/memory-graph/visual-atlas",
        response_model=VisualAtlasResponse,
        tags=["memory-graph"],
        operation_id="getVisualMemoryAtlas",
    )
    def visual_memory_atlas(
        scope_name: str,
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        lease_id = acquire_gate(context.tenant_id)
        try:
            return components.session_graphs.visual_atlas(
                context.tenant_id, scope_name
            )
        finally:
            components.gate.release(lease_id)

    @app.post(
        "/v1/scopes/{scope_name}/memory-graph/visual-atlas/refresh",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=SessionGraphRefreshResponse,
        tags=["memory-graph"],
        operation_id="refreshVisualMemoryAtlas",
    )
    def refresh_visual_memory_atlas(
        scope_name: str,
        context: AuthContext = Depends(require_permission("memory:write")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        return components.session_graphs.request_visual_atlas_refresh(
            context.tenant_id, scope_name
        )

    @app.get(
        "/v1/scopes/{scope_name}/knowledge-base",
        response_model=PersonalKnowledgeBaseResponse,
        tags=["knowledge-base"],
        operation_id="getPersonalKnowledgeBase",
    )
    def personal_knowledge_base(
        scope_name: str,
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        lease_id = acquire_gate(context.tenant_id)
        try:
            return components.session_graphs.personal_knowledge_base(
                context.tenant_id, scope_name
            )
        finally:
            components.gate.release(lease_id)

    @app.post(
        "/v1/scopes/{scope_name}/knowledge-base/refresh",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=SessionGraphRefreshResponse,
        tags=["knowledge-base"],
        operation_id="refreshPersonalKnowledgeBase",
    )
    def refresh_personal_knowledge_base(
        scope_name: str,
        context: AuthContext = Depends(require_permission("memory:write")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        return components.session_graphs.request_personal_knowledge_refresh(
            context.tenant_id, scope_name
        )

    @app.get(
        "/v1/scopes/{scope_name}/projection-build",
        response_model=ProjectionBuildProgressResponse,
        tags=["memory-graph", "knowledge-base"],
        operation_id="getProjectionBuildProgress",
    )
    def projection_build_progress(
        scope_name: str,
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        return components.session_graphs.projection_build_status(
            context.tenant_id, scope_name
        )

    @app.post(
        "/v1/scopes/{scope_name}/projection-build",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=ProjectionBuildProgressResponse,
        tags=["memory-graph", "knowledge-base"],
        operation_id="startProjectionBuild",
    )
    def start_projection_build(
        scope_name: str,
        context: AuthContext = Depends(require_permission("memory:write")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        return components.session_graphs.request_projection_build(
            context.tenant_id, scope_name
        )

    @app.get(
        "/v1/scopes/{scope_name}/memory-graph/sessions",
        response_model=SessionAtlasResponse,
        tags=["memory-graph"],
        operation_id="getSessionMemoryAtlas",
    )
    def session_memory_atlas(
        scope_name: str,
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        lease_id = acquire_gate(context.tenant_id)
        try:
            return components.session_graphs.atlas(
                context.tenant_id, scope_name
            )
        finally:
            components.gate.release(lease_id)

    @app.get(
        "/v1/scopes/{scope_name}/memory-graph/sessions/{session_id}",
        response_model=SessionMapResponse,
        tags=["memory-graph"],
        operation_id="getSessionMemoryMap",
    )
    def session_memory_map(
        scope_name: str,
        session_id: str,
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        lease_id = acquire_gate(context.tenant_id)
        try:
            return components.session_graphs.session_map(
                context.tenant_id, scope_name, session_id
            )
        finally:
            components.gate.release(lease_id)

    @app.post(
        "/v1/scopes/{scope_name}/memory-graph/sessions/refresh",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=SessionGraphRefreshResponse,
        tags=["memory-graph"],
        operation_id="refreshSessionMemoryAtlas",
    )
    def refresh_session_memory_atlas(
        scope_name: str,
        context: AuthContext = Depends(require_permission("memory:write")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        return components.session_graphs.request_refresh(
            context.tenant_id, scope_name
        )

    @app.post(
        "/v1/scopes/{scope_name}/memory-graph/sessions/{session_id}/refresh",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=SessionGraphRefreshResponse,
        tags=["memory-graph"],
        operation_id="refreshSessionMemoryMap",
    )
    def refresh_session_memory_map(
        scope_name: str,
        session_id: str,
        context: AuthContext = Depends(require_permission("memory:write")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        return components.session_graphs.request_refresh(
            context.tenant_id, scope_name, session_id
        )

    @app.get(
        "/v1/scopes/{scope_name}/memory-graph/nodes/{memory_id}/neighbors",
        response_model=MemoryGraphResponse,
        tags=["memory-graph"],
        operation_id="getMemoryGraphNeighbors",
    )
    def memory_graph_neighbors(
        scope_name: str,
        memory_id: str,
        depth: int = Query(default=1, ge=1, le=2),
        layers: str = Query(default="slow,fast,source", max_length=40),
        limit: int = Query(default=80, ge=1, le=120),
        cursor: str | None = Query(default=None, max_length=512),
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        lease_id = acquire_gate(context.tenant_id)
        try:
            projection = MemoryGraphProjection.from_available_storage(
                components.storage,
                tenant_id=context.tenant_id,
                scope_name=scope_name,
            )
            return projection.neighbors(
                memory_id,
                depth=depth,
                layers=parse_layers(
                    layers, default=("slow", "fast", "source")
                ),
                limit=limit,
                cursor=cursor,
            )
        finally:
            components.gate.release(lease_id)

    @app.get(
        "/v1/scopes/{scope_name}/memory-graph/nodes/{memory_id}/evidence",
        response_model=MemoryGraphEvidenceResponse,
        tags=["memory-graph"],
        operation_id="getMemoryGraphEvidence",
    )
    def memory_graph_evidence(
        scope_name: str,
        memory_id: str,
        limit: int = Query(default=10, ge=1, le=25),
        cursor: str | None = Query(default=None, max_length=512),
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        lease_id = acquire_gate(context.tenant_id)
        try:
            projection = MemoryGraphProjection.from_available_storage(
                components.storage,
                tenant_id=context.tenant_id,
                scope_name=scope_name,
            )
            return projection.evidence(memory_id, limit=limit, cursor=cursor)
        finally:
            components.gate.release(lease_id)

    @app.post(
        "/v1/scopes/{scope_name}/memory-graph/trace",
        response_model=MemoryGraphTraceResponse,
        responses=RECALL_ERROR_RESPONSES,
        tags=["memory-graph"],
        operation_id="traceMemoryRecall",
    )
    async def memory_graph_trace(
        scope_name: str,
        body: MemoryGraphTraceRequest,
        request: Request,
        context: AuthContext = Depends(require_permission("memory:read")),
        _: dict[str, str] = Depends(usage_attribution_headers),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        lease_id = acquire_gate(context.tenant_id)
        usage_attribution = request_usage_attribution(request, context)
        try:
            try:
                snapshot = components.storage.active_snapshot(
                    context.tenant_id, scope_name
                )
            except V4AdapterError as exc:
                raise GraphProjectionError(
                    "graph_snapshot_unavailable",
                    "scope has no committed memory graph snapshot",
                ) from exc
            projection = MemoryGraphProjection.from_snapshot(
                components.storage,
                tenant_id=context.tenant_id,
                scope_name=scope_name,
                snapshot=snapshot,
            )
            principal, consumer_principal, _membership = (
                components.control.quota_identity(
                    context.tenant_id, context.subject
                )
            )
            recall_event_key = f"recall:{uuid.uuid4().hex}"

            def admit_graph_trace_recall() -> None:
                components.control.admit_recall(
                    context.tenant_id,
                    principal,
                    scope_name,
                    recall_event_key,
                    consumer_principal=consumer_principal,
                    usage_attribution=usage_attribution,
                )

            query_id = f"graph_{uuid.uuid4().hex}"
            query_time = body.query_time.isoformat() if body.query_time else ""
            planner_stage_id = f"{query_id}:planner"
            planner_worker_id = f"api:{query_id}"
            planner_stage = components.jobs.create_stage(
                context.tenant_id,
                scope_name,
                "graph_trace_planner",
                stage_id=planner_stage_id,
            )
            components.jobs.claim_stage(planner_stage.stage_id, planner_worker_id)
            try:
                evidence, debug = await run_in_threadpool(
                    run_online_recall,
                    context.tenant_id,
                    before_execute=admit_graph_trace_recall,
                    snapshot=snapshot,
                    query_id=query_id,
                    query=body.query,
                    query_time=query_time,
                    max_windows=body.max_windows,
                )
                planner_call_count = journal_deepseek_calls(
                    components.jobs,
                    dict(debug.get("planner") or {}),
                    tenant_id=context.tenant_id,
                    scope_name=scope_name,
                    job_id=None,
                    stage_id=planner_stage_id,
                    operation="graph_trace_planner",
                    default_model=os.getenv(
                        "TMCRA_RECALL_PLANNER_MODEL",
                        os.getenv("TMCRA_WRITER_MODEL", "deepseek-v4-flash"),
                    ),
                    usage_attribution=usage_attribution,
                )
            except Exception as exc:
                components.jobs.fail_stage(
                    planner_stage_id,
                    f"{type(exc).__name__}:{exc}",
                    worker_id=planner_worker_id,
                )
                raise
            components.jobs.complete_stage(
                planner_stage_id,
                {"query_id": query_id, "physical_api_calls": planner_call_count},
                worker_id=planner_worker_id,
            )
            selected_ids = extract_trace_memory_ids(evidence)
            result = await run_in_threadpool(projection.trace, selected_ids)
            windows = evidence.get("evidence_windows")
            result.update(
                {
                    "query_id": query_id,
                    "index_job_id": str(snapshot.get("job_id") or ""),
                    "retrieval_summary": {
                        "evidence_window_count": len(windows)
                        if isinstance(windows, list)
                        else 0,
                        "persisted_memory_id_count": len(selected_ids),
                        "projected_memory_id_count": len(
                            result["selected_memory_ids"]
                        ),
                    },
                    "debug": debug if body.debug else None,
                }
            )
            return result
        finally:
            components.gate.release(lease_id)

    @app.get(
        "/v1/jobs/{job_id}",
        response_model=JobView,
        tags=["jobs"],
        operation_id="getMemoryJob",
    )
    def get_job(
        job_id: str,
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> dict[str, Any]:
        job = components.jobs.get(job_id, tenant_id=context.tenant_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if not context.allows_scope_name(job.scope_name):
            raise AuthorizationError("access token is not valid for this scope")
        return _job_payload(job, settings.public_base_url)

    @app.get(
        "/v1/usage/costs",
        response_model=UsageCostsView,
        tags=["usage"],
        operation_id="getMemoryUsage",
    )
    def usage_costs(
        scope_name: str | None = None,
        scope_prefix: str | None = None,
        from_timestamp: float | None = Query(default=None, ge=0),
        to_timestamp: float | None = Query(default=None, ge=0),
        group_by: str | None = Query(
            default=None,
            pattern=(
                "^(day|scope|stage|operation|provider|model|platform|"
                "integration|agent|attribution_source)$"
            ),
        ),
        context: AuthContext = Depends(require_permission("memory:read")),
    ) -> dict[str, Any]:
        if scope_name is not None and scope_prefix is not None:
            raise HTTPException(
                status_code=422,
                detail="scope_name and scope_prefix are mutually exclusive",
            )
        normalized_scope = _scope_name(scope_name) if scope_name is not None else None
        normalized_prefix = (
            _scope_name(scope_prefix) if scope_prefix is not None else None
        )
        if normalized_prefix is not None:
            tenant_scopes = components.database.get_tenant_scopes(context.tenant_id)
            if (
                context.credential_type != "api_key"
                or "tokens:manage" not in context.scopes
                or "tokens:manage" not in tenant_scopes
            ):
                raise AuthorizationError(
                    "scope_prefix usage queries require a tokens:manage API key"
                )
        if (
            context.allowed_scope_names is not None
            or context.allowed_scope_prefixes is not None
        ):
            if normalized_scope is None:
                raise AuthorizationError(
                    "scoped access tokens must request usage for one allowed scope"
                )
            if not context.allows_scope_name(normalized_scope):
                raise AuthorizationError("access token is not valid for this scope")
        if (
            from_timestamp is not None
            and to_timestamp is not None
            and from_timestamp >= to_timestamp
        ):
            raise HTTPException(
                status_code=422,
                detail="from_timestamp must be earlier than to_timestamp",
            )
        return components.jobs.usage_cost_summary(
            context.tenant_id,
            scope_name=normalized_scope,
            scope_prefix=normalized_prefix,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            group_by=group_by,
        )

    def require_user_provider_stage_permission(
        context: AuthContext, stage: str
    ) -> None:
        permissions = (
            frozenset({"memory:write"})
            if stage == "writer"
            else frozenset({"memory:write", "memory:consolidate"})
        )
        tenant_scopes = components.database.get_tenant_scopes(context.tenant_id)
        if not permissions.intersection(context.scopes).intersection(tenant_scopes):
            raise AuthorizationError(
                f"missing permission: {' or '.join(sorted(permissions))}"
            )

    def require_owned_user_provider_task(
        task_id: str, context: AuthContext
    ) -> Any:
        task = components.user_provider_tasks.get(task_id)
        if (
            task is None
            or task.tenant_id != context.tenant_id
            or task.auth_key_id != context.key_id
        ):
            raise UserProviderTaskNotFound(task_id)
        if not context.allows_scope_name(task.scope_name):
            raise AuthorizationError("access token is not valid for this scope")
        require_user_provider_stage_permission(context, task.task_stage)
        return task

    @app.post(
        "/v1/provider-tasks/claim",
        response_model=UserProviderTaskClaimView,
        tags=["memory"],
        operation_id="claimUserProviderTask",
        summary="Lease one local provider task",
        description=(
            "Authenticated device endpoint for locally executing a bounded Writer or "
            "organizer model call. Model-provider credentials are never accepted."
        ),
    )
    def claim_user_provider_task(
        body: UserProviderTaskClaimRequest,
        context: AuthContext = Depends(
            require_any_permission("memory:write", "memory:consolidate")
        ),
    ) -> dict[str, Any]:
        require_user_provider_stage_permission(context, body.stage)
        if os.getenv("TMCRA_DEPLOYMENT_MODE") == "local":
            return {"task": None, "retry_after_seconds": 60.0}
        claimed = components.user_provider_tasks.claim_next(
            tenant_id=context.tenant_id,
            auth_key_id=context.key_id,
            task_stage=body.stage,
            scope_allowed=context.allows_scope_name,
        )
        if claimed is None:
            return {"task": None, "retry_after_seconds": 1.0}
        task, lease_token = claimed
        return {
            "task": {
                "schema_version": USER_PROVIDER_TASK_SCHEMA_VERSION,
                "task_id": task.task_id,
                "stage": task.task_stage,
                "operation": task.operation,
                "request_sha256": task.request_sha256,
                "request": task.request,
                "lease_token": lease_token,
                "lease_expires_at": task.lease_expires_at,
            },
            "retry_after_seconds": 0.0,
        }

    @app.post(
        "/v1/provider-tasks/{task_id}/started",
        response_model=UserProviderTaskStatusView,
        tags=["memory"],
        operation_id="startUserProviderTask",
    )
    def start_user_provider_task(
        task_id: str,
        body: UserProviderTaskLeaseRequest,
        context: AuthContext = Depends(
            require_any_permission("memory:write", "memory:consolidate")
        ),
    ) -> dict[str, Any]:
        require_owned_user_provider_task(task_id, context)
        task, replay = components.user_provider_tasks.start(
            task_id,
            tenant_id=context.tenant_id,
            auth_key_id=context.key_id,
            lease_token=body.lease_token,
        )
        return {
            "task_id": task.task_id,
            "state": task.state,
            "lease_expires_at": task.lease_expires_at,
            "idempotent_replay": replay,
        }

    @app.post(
        "/v1/provider-tasks/{task_id}/heartbeat",
        response_model=UserProviderTaskStatusView,
        tags=["memory"],
        operation_id="heartbeatUserProviderTask",
    )
    def heartbeat_user_provider_task(
        task_id: str,
        body: UserProviderTaskLeaseRequest,
        context: AuthContext = Depends(
            require_any_permission("memory:write", "memory:consolidate")
        ),
    ) -> dict[str, Any]:
        require_owned_user_provider_task(task_id, context)
        task = components.user_provider_tasks.heartbeat(
            task_id,
            tenant_id=context.tenant_id,
            auth_key_id=context.key_id,
            lease_token=body.lease_token,
        )
        return {
            "task_id": task.task_id,
            "state": task.state,
            "lease_expires_at": task.lease_expires_at,
            "idempotent_replay": False,
        }

    @app.post(
        "/v1/provider-tasks/{task_id}/complete",
        response_model=UserProviderTaskStatusView,
        tags=["memory"],
        operation_id="completeUserProviderTask",
    )
    def complete_user_provider_task(
        task_id: str,
        body: UserProviderTaskCompleteRequest,
        context: AuthContext = Depends(
            require_any_permission("memory:write", "memory:consolidate")
        ),
    ) -> dict[str, Any]:
        require_owned_user_provider_task(task_id, context)
        task, replay = components.user_provider_tasks.complete(
            task_id,
            tenant_id=context.tenant_id,
            auth_key_id=context.key_id,
            lease_token=body.lease_token,
            provider=body.provider,
            model=body.model,
            output=body.output,
            usage=(
                None
                if body.usage is None
                else body.usage.model_dump(exclude_none=True)
            ),
            provider_request_id=body.provider_request_id,
        )
        return {
            "task_id": task.task_id,
            "state": task.state,
            "lease_expires_at": task.lease_expires_at,
            "idempotent_replay": replay,
        }

    @app.post(
        "/v1/provider-tasks/{task_id}/fail",
        response_model=UserProviderTaskStatusView,
        tags=["memory"],
        operation_id="failUserProviderTask",
    )
    def fail_user_provider_task(
        task_id: str,
        body: UserProviderTaskFailRequest,
        context: AuthContext = Depends(
            require_any_permission("memory:write", "memory:consolidate")
        ),
    ) -> dict[str, Any]:
        require_owned_user_provider_task(task_id, context)
        task, replay = components.user_provider_tasks.fail(
            task_id,
            tenant_id=context.tenant_id,
            auth_key_id=context.key_id,
            lease_token=body.lease_token,
            provider=body.provider,
            model=body.model,
            outcome=body.outcome,
            error_code=body.error_code,
        )
        return {
            "task_id": task.task_id,
            "state": task.state,
            "lease_expires_at": task.lease_expires_at,
            "idempotent_replay": replay,
        }

    @app.post(
        "/v1/scopes/{scope_name}/provider-calls",
        status_code=status.HTTP_201_CREATED,
        response_model=ProviderCallReportView,
        tags=["usage"],
        operation_id="reportAnswerProviderCall",
        summary="Record one answer-model usage receipt",
        description=(
            "Server-to-server endpoint used by the TMCRA chat gateway. It accepts "
            "accounting metadata only and never accepts prompts, attachments, memory "
            "evidence, or model response content."
        ),
    )
    def report_answer_provider_call(
        scope_name: str,
        body: ProviderCallReportRequest,
        request: Request,
        context: AuthContext = Depends(
            require_permission("tokens:manage", api_key_only=True)
        ),
        _: dict[str, str] = Depends(usage_attribution_headers),
    ) -> dict[str, Any]:
        scope_name = _scope_name(scope_name)
        require_active_scope(context.tenant_id, scope_name)
        if not context.allows_scope_name(scope_name):
            raise AuthorizationError("access token is not valid for this scope")
        usage_attribution = request_usage_attribution(request, context)
        existing = components.jobs.get_provider_call(body.call_id)
        if existing is not None:
            expected = {
                "tenant_id": context.tenant_id,
                "scope_name": scope_name,
                "provider": body.provider,
                "model": body.model,
                "operation": body.operation,
                "status": body.status,
                "input_tokens": body.input_tokens,
                "output_tokens": body.output_tokens,
                "total_tokens": (
                    body.total_tokens
                    if body.total_tokens is not None
                    else (
                        body.input_tokens + body.output_tokens
                        if body.input_tokens is not None
                        and body.output_tokens is not None
                        else None
                    )
                ),
                "cache_hit_tokens": (
                    body.cache_hit_tokens or 0
                    if body.input_tokens is not None
                    else None
                ),
                "error": body.error_code,
                "request_sha256": body.request_sha256,
                "response_sha256": body.response_sha256,
                "started_at": body.started_at,
                "finished_at": body.finished_at,
                "client_platform": usage_attribution.client_platform,
                "integration_id": usage_attribution.integration_id,
                "agent_id": usage_attribution.agent_id,
                "attribution_source": usage_attribution.attribution_source,
            }
            actual = {
                key: getattr(existing, key)
                for key in expected
            }
            if actual != expected:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "provider_call_idempotency_conflict"},
                )
            return _provider_call_report_view(existing, idempotent_replay=True)

        input_tokens = body.input_tokens
        output_tokens = body.output_tokens
        total_tokens = body.total_tokens
        cache_hit_tokens = body.cache_hit_tokens
        cache_miss_tokens: int | None = None
        usage_state = "missing"
        if input_tokens is not None and output_tokens is not None:
            total_tokens = total_tokens or input_tokens + output_tokens
            cache_hit_tokens = cache_hit_tokens or 0
            cache_miss_tokens = input_tokens - cache_hit_tokens
            usage_state = "complete"
        price = components.jobs.get_provider_price(
            body.provider,
            body.model,
            at=body.finished_at,
        )
        cost_micro_cny: int | None = None
        price_version: str | None = None
        if price is not None:
            price_version = f"{price.provider}:{price.model}:{price.effective_at:g}"
        if (
            body.status == "completed"
            and usage_state == "complete"
            and price is not None
            and price.currency == "CNY"
        ):
            hit_rate = (
                price.cache_hit_input_micro_cny_per_million
                if price.cache_hit_input_micro_cny_per_million is not None
                else price.input_micro_cny_per_million
            )
            miss_rate = (
                price.cache_miss_input_micro_cny_per_million
                if price.cache_miss_input_micro_cny_per_million is not None
                else price.input_micro_cny_per_million
            )
            if (
                hit_rate is not None
                and miss_rate is not None
                and price.output_micro_cny_per_million is not None
            ):
                numerator = (
                    int(cache_hit_tokens or 0) * hit_rate
                    + int(cache_miss_tokens or 0) * miss_rate
                    + int(output_tokens or 0)
                    * price.output_micro_cny_per_million
                )
                cost_micro_cny = (numerator + 999_999) // 1_000_000

        recorded = components.jobs.record_provider_call(
            context.tenant_id,
            body.provider,
            body.model,
            scope_name=scope_name,
            call_id=body.call_id,
            operation=body.operation,
            status=body.status,
            error=body.error_code,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_micro_cny=cost_micro_cny,
            cache_hit_tokens=cache_hit_tokens,
            cache_miss_tokens=cache_miss_tokens,
            usage_state=usage_state,
            price_version=price_version,
            key_id=context.key_id,
            usage_attribution=usage_attribution,
            request_sha256=body.request_sha256,
            response_sha256=body.response_sha256,
            started_at=body.started_at,
            finished_at=body.finished_at,
        )
        return _provider_call_report_view(recorded, idempotent_replay=False)

    @app.post(
        "/v1/jobs/{job_id}/cancel",
        response_model=JobView,
        tags=["jobs"],
        operation_id="cancelMemoryJob",
    )
    def cancel_job(
        job_id: str,
        context: AuthContext = Depends(require_permission("memory:write")),
    ) -> dict[str, Any]:
        job = components.jobs.get(job_id, tenant_id=context.tenant_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if not context.allows_scope_name(job.scope_name):
            raise AuthorizationError("access token is not valid for this scope")
        if job.state != "pending":
            raise HTTPException(
                status_code=409,
                detail="only pending jobs can be cancelled safely",
            )
        try:
            cancelled = components.jobs.cancel(job_id)
        except JobStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        components.commercial.enqueue_job_events(cancelled)
        return _job_payload(cancelled, settings.public_base_url)

    @app.post(
        "/v1/jobs/{job_id}/retry",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=JobView,
        tags=["jobs"],
        operation_id="retryMemoryJob",
    )
    def retry_job(
        job_id: str,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=200),
        context: AuthContext = Depends(require_permission("memory:write")),
    ) -> dict[str, Any]:
        prior = components.jobs.get(job_id, tenant_id=context.tenant_id)
        if prior is None:
            raise HTTPException(status_code=404, detail="job not found")
        if not context.allows_scope_name(prior.scope_name):
            raise AuthorizationError("access token is not valid for this scope")

        def audit_ingest_retry(
            job: Job,
        ) -> tuple[dict[str, Any], dict[str, Any]] | None:
            payload = dict(job.payload or {})
            if str(payload.get("job_type") or "") != "ingest":
                return None
            scope_name = str(payload.get("scope_name") or "default")
            audit = dict(
                components.storage.audit_scope_recovery(
                    tenant_id=context.tenant_id,
                    scope_name=scope_name,
                )
            )
            failed_operations = {
                str(value)
                for value in audit.get("failed_operation_ids", [])
                if str(value)
            }
            plan = dict(
                components.storage.ingest_recovery_plan(
                    tenant_id=context.tenant_id,
                    scope_name=scope_name,
                    job_id=job.job_id,
                )
            )
            local_complete_writer_repair = bool(
                job.job_id in failed_operations
                and plan.get("resumable") is True
                and plan.get("mode")
                in {"complete_writer_artifacts", "committed_writer_artifacts"}
                and plan.get("parallel_safe") is True
                and plan.get("external_api_calls_expected") is False
                and plan.get("deterministic_local_repair") is True
            )
            journal_failure_repair = bool(
                job.job_id in failed_operations and plan.get("resumable") is True
            )
            if (
                not bool(audit.get("integrity_ok"))
                or not (journal_failure_repair or local_complete_writer_repair)
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "ingest retry requires a clean Source/journal "
                        "audit and a resumable failed operation"
                    ),
                )
            return audit, plan

        def wake_quarantined_ingest(job: Job, audit: Mapping[str, Any] | None) -> None:
            if audit is None:
                return
            payload = dict(job.payload or {})
            scope_name = str(payload.get("scope_name") or "default")
            if components.commercial.scope_quarantine(
                context.tenant_id, scope_name
            ) is None:
                return
            if not components.commercial.request_quarantine_recovery_after_audit(
                context.tenant_id,
                scope_name,
                audit_report=audit,
            ):
                raise HTTPException(
                    status_code=409,
                    detail="quarantine recovery cannot be restarted safely",
                )

        lease_id = acquire_gate(context.tenant_id)
        try:
            audit: dict[str, Any] | None = None
            plan: dict[str, Any] | None = None
            if prior.state in {"pending", "running", "succeeded"}:
                if prior.state == "pending":
                    if components.commercial.scope_quarantine(
                        context.tenant_id, prior.scope_name
                    ) is not None:
                        audited = audit_ingest_retry(prior)
                        if audited is not None:
                            audit, plan = audited
                    wake_quarantined_ingest(prior, audit)
                result = _job_payload(prior, settings.public_base_url)
                result["idempotent_retry"] = True
                if (
                    prior.state == "pending"
                    and audit is not None
                    and str(dict(prior.payload or {}).get("job_type") or "")
                    == "ingest"
                ):
                    result["resume_mode"] = "audited_writer_state"
                return result
            if prior.state != FAILED:
                raise HTTPException(status_code=409, detail="job cannot be retried")
            payload = dict(prior.payload or {})
            job_type = payload.get("job_type")
            resumable = job_type in {
                "reindex",
                "export_scope",
                "delete_scope",
                "delete_memories",
                "delete_session",
            }
            if job_type == "ingest":
                audited = audit_ingest_retry(prior)
                if audited is None:
                    raise HTTPException(
                        status_code=409,
                        detail="ingest retry requires an explicit Source/journal audit",
                    )
                audit, plan = audited
                resumable = True
            if not resumable:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "writer and slow-graph failures require explicit artifact audit; "
                        "no durable Writer commit was found"
                    ),
                )
            wake_quarantined_ingest(prior, audit)
            if job_type == "ingest":
                assert audit is not None and plan is not None
                evidence = {
                    "job_id": job_id,
                    "tenant_id": context.tenant_id,
                    "scope_name": str(payload.get("scope_name") or "default"),
                    "audit": audit,
                    "recovery_plan": plan,
                }
                authorization = ResumeAuthorization.from_evidence(
                    reason_code="http_ingest_retry_audited_writer_state",
                    resume_mode=str(plan.get("mode") or "audited_writer_state"),
                    evidence=evidence,
                )
            else:
                authorization = ResumeAuthorization(
                    reason_code=f"http_retry_{str(job_type or 'unknown')}",
                )
            try:
                retry = components.jobs.resume_failed(
                    job_id, authorization=authorization
                )
            except JobStateError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if job_type in {"delete_memories", "delete_session"}:
                components.commercial.resume_content_deletion(
                    context.tenant_id,
                    str(payload.get("scope_name") or "default"),
                    str(payload.get("deletion_id") or ""),
                    job_id,
                )
            result = _job_payload(retry, settings.public_base_url)
            result["idempotent_retry"] = False
            result["resume_mode"] = authorization.resume_mode or str(job_type)
            return result
        finally:
            components.gate.release(lease_id)

    return app
