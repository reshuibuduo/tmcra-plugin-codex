from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


class SettingsError(RuntimeError):
    pass


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer") from exc
    if value <= 0:
        raise SettingsError(f"{name} must be positive")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a number") from exc
    if value <= 0:
        raise SettingsError(f"{name} must be positive")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"{name} must be a boolean")


def _choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in allowed:
        choices = ",".join(sorted(allowed))
        raise SettingsError(f"{name} must be one of: {choices}")
    return value


def _optional_float(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a number") from exc


RELEASE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
RELEASE_CHANNEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class ServiceSettings:
    state_dir: Path
    control_db: Path
    bind_host: str
    bind_port: int
    public_base_url: str
    v4_root: Path
    integrated_repo: Path
    writer_env: Path
    embedding_model: Path
    native_harness: Path
    node_model: Path
    path_model: Path
    checkpoint: Path
    cross_model: Path
    device: str
    graph_device: str
    request_body_limit: int
    provider_lease_seconds: int
    provider_key_concurrency: int
    disk_free_min_bytes: int
    learned_graph_enabled: bool = False
    provider_billing_circuit_seconds: float = 900.0
    provider_auth_circuit_seconds: float = 900.0
    write_admission_retry_seconds: float = 5.0
    worker_concurrency: int = 4
    request_max_concurrency: int = 8
    request_per_minute: int = 240
    request_lease_seconds: int = 600
    tenant_queue_limit: int = 100
    global_queue_limit: int = 1000
    recall_pool_min_size: int = 1
    recall_pool_max_size: int = 1
    recall_global_queue_limit: int = 8
    recall_tenant_queue_limit: int = 2
    recall_queue_timeout_seconds: float = 30.0
    recall_scale_up_sustain_seconds: float = 2.0
    recall_scale_up_cooldown_seconds: float = 5.0
    recall_scale_down_idle_seconds: float = 600.0
    recall_scale_down_cooldown_seconds: float = 60.0
    recall_target_utilization: float = 0.70
    recall_warm_spare: int = 1
    recall_gpu_headroom_bytes: int = 6 * 1024**3
    recall_replica_estimate_bytes: int = 5 * 1024**3
    recall_scope_cache_size: int = 4
    recall_idle_cache_trim_enabled: bool = False
    recall_idle_cache_seconds: float = 60.0
    recall_cache_trim_interval_seconds: float = 5.0
    recall_cache_trim_cooldown_seconds: float = 300.0
    recall_cache_trim_min_bytes: int = 4 * 1024**3
    slow_dirty_token_threshold: int = 32_000
    slow_dirty_user_turn_threshold: int = 64
    slow_max_age_seconds: float = 86_400.0
    slow_min_token_threshold: int = 4_000
    slow_min_user_turn_threshold: int = 8
    slow_min_interval_seconds: float = 1_800.0
    slow_graph_drain_concurrency: int = 1
    # Base compaction thresholds. Online visibility is provided by the
    # per-write delta and must never wait for these values.
    index_dirty_threshold: int = 16
    index_max_age_seconds: float = 2.0
    index_claim_wait_seconds: float = 900.0
    index_generation_retention: int = 8
    scheduler_interval_seconds: float = 1.0
    quarantine_recovery_interval_seconds: float = 15.0
    quarantine_recovery_lease_seconds: float = 120.0
    quarantine_recovery_concurrency: int = 4
    quarantine_recovery_max_job_attempts: int = 3
    quarantine_recovery_max_local_repairs: int = 8
    quarantine_recovery_backoff_seconds: float = 30.0
    preload_online_engine: bool = True
    export_ttl_seconds: int = 86_400
    webhook_signing_key: str | None = None
    webhook_timeout_seconds: float = 10.0
    writer_execution_mode: str = "subprocess"
    writer_pool_size: int = 1
    writer_pool_startup_timeout_seconds: float = 120.0
    writer_pool_request_timeout_seconds: float = 900.0
    local_writer_recovery_concurrency: int = 1
    startup_preflight_mode: str = "basic"
    staff_monitoring_key: str | None = None
    staff_latency_window_seconds: float = 300.0
    staff_latency_max_samples: int = 4096
    staff_recent_error_window_seconds: float = 86_400.0
    staff_recent_error_limit: int = 20
    api_access_log_enabled: bool = False
    api_access_log_path: Path | None = None
    diagnostic_log_enabled: bool = False
    diagnostic_log_path: Path | None = None
    service_release_id: str | None = None
    service_release_sha256: str | None = None
    service_release_channel: str | None = None
    service_canary_percent: float | None = None
    service_rollback_release_id: str | None = None
    audio_asr_base_url: str | None = None
    audio_asr_api_key_file: Path | None = None
    audio_asr_timeout_seconds: float = 120.0
    audio_asr_max_request_bytes: int = 2_621_440

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        runtime_root = Path(
            os.getenv("TMCRA_V4_ROOT", str(Path(__file__).resolve().parents[1]))
        ).resolve()
        state_dir = Path(
            os.getenv("TMCRA_SERVICE_STATE_DIR", "/opt/tmcra/tmcra_service_state")
        ).resolve()
        public_base_url = os.getenv("TMCRA_SERVICE_PUBLIC_BASE_URL", "").strip().rstrip("/")
        if not public_base_url:
            raise SettingsError("TMCRA_SERVICE_PUBLIC_BASE_URL is required")
        worker_concurrency = _positive_int(
            "TMCRA_SERVICE_WORKER_CONCURRENCY", 4
        )
        return cls(
            state_dir=state_dir,
            control_db=Path(
                os.getenv(
                    "TMCRA_SERVICE_CONTROL_DB",
                    str(state_dir / "control.sqlite3"),
                )
            ).resolve(),
            bind_host=os.getenv("TMCRA_SERVICE_BIND_HOST", "0.0.0.0").strip(),
            bind_port=_positive_int("TMCRA_SERVICE_BIND_PORT", 2009),
            public_base_url=public_base_url,
            v4_root=runtime_root,
            integrated_repo=Path(
                os.getenv(
                    "TMCRA_INTEGRATED_REPO",
                    str(runtime_root),
                )
            ).resolve(),
            writer_env=Path(
                os.getenv(
                    "TMCRA_WRITER_ENV",
                    "/etc/tmcra/writer.env",
                )
            ).resolve(),
            embedding_model=Path(
                os.getenv("TMCRA_EMBEDDING_MODEL", "/opt/tmcra-models/BAAI/bge-m3")
            ).resolve(),
            native_harness=Path(
                os.getenv(
                    "TMCRA_NATIVE_HARNESS",
                    str(Path(__file__).resolve().parent / "native_harness.py"),
                )
            ).resolve(),
            node_model=Path(
                os.getenv(
                    "TMCRA_NODE_MODEL",
                    "/opt/tmcra-data/tmcra_service_assets/"
                    "tmcra_node_scorer.pt",
                )
            ).resolve(),
            path_model=Path(
                os.getenv(
                    "TMCRA_PATH_MODEL",
                    "/opt/tmcra-data/tmcra_service_assets/"
                    "tmcra_path_scorer.pt",
                )
            ).resolve(),
            checkpoint=Path(
                os.getenv(
                    "TMCRA_CHECKPOINT",
                    "/opt/tmcra-data/tmcra_service_assets/"
                    "tmcra_v3_reranker.pt",
                )
            ).resolve(),
            cross_model=Path(
                os.getenv("TMCRA_CROSS_MODEL", "/opt/tmcra-models/BAAI/bge-reranker-v2-m3")
            ).resolve(),
            device=os.getenv("TMCRA_SERVICE_DEVICE", "cuda").strip(),
            graph_device=os.getenv("TMCRA_SERVICE_GRAPH_DEVICE", "cuda").strip(),
            learned_graph_enabled=_boolean(
                "TMCRA_LEARNED_GRAPH_ENABLED", False
            ),
            request_body_limit=_positive_int(
                "TMCRA_SERVICE_REQUEST_BODY_LIMIT", 2 * 1024 * 1024
            ),
            provider_lease_seconds=_positive_int(
                "TMCRA_PROVIDER_LEASE_SECONDS", 300
            ),
            provider_key_concurrency=_positive_int(
                "TMCRA_PROVIDER_KEY_CONCURRENCY", 2
            ),
            disk_free_min_bytes=_positive_int(
                "TMCRA_SERVICE_DISK_FREE_MIN_BYTES", 5 * 1024**3
            ),
            provider_billing_circuit_seconds=_positive_float(
                "TMCRA_PROVIDER_BILLING_CIRCUIT_SECONDS", 900.0
            ),
            provider_auth_circuit_seconds=_positive_float(
                "TMCRA_PROVIDER_AUTH_CIRCUIT_SECONDS", 900.0
            ),
            write_admission_retry_seconds=_positive_float(
                "TMCRA_SERVICE_WRITE_ADMISSION_RETRY_SECONDS", 5.0
            ),
            worker_concurrency=worker_concurrency,
            request_max_concurrency=_positive_int(
                "TMCRA_SERVICE_REQUEST_MAX_CONCURRENCY", 8
            ),
            request_per_minute=_positive_int(
                "TMCRA_SERVICE_REQUESTS_PER_MINUTE", 240
            ),
            request_lease_seconds=_positive_int(
                "TMCRA_SERVICE_REQUEST_LEASE_SECONDS", 600
            ),
            tenant_queue_limit=_positive_int(
                "TMCRA_SERVICE_TENANT_QUEUE_LIMIT", 100
            ),
            global_queue_limit=_positive_int(
                "TMCRA_SERVICE_GLOBAL_QUEUE_LIMIT", 1000
            ),
            recall_pool_min_size=_positive_int(
                "TMCRA_SERVICE_RECALL_POOL_MIN_SIZE", 2
            ),
            recall_pool_max_size=_positive_int(
                "TMCRA_SERVICE_RECALL_POOL_MAX_SIZE", 2
            ),
            recall_global_queue_limit=_positive_int(
                "TMCRA_SERVICE_RECALL_GLOBAL_QUEUE_LIMIT", 8
            ),
            recall_tenant_queue_limit=_positive_int(
                "TMCRA_SERVICE_RECALL_TENANT_QUEUE_LIMIT", 2
            ),
            recall_queue_timeout_seconds=_positive_float(
                "TMCRA_SERVICE_RECALL_QUEUE_TIMEOUT_SECONDS", 30.0
            ),
            recall_scale_up_sustain_seconds=_positive_float(
                "TMCRA_SERVICE_RECALL_SCALE_UP_SUSTAIN_SECONDS", 2.0
            ),
            recall_scale_up_cooldown_seconds=_positive_float(
                "TMCRA_SERVICE_RECALL_SCALE_UP_COOLDOWN_SECONDS", 5.0
            ),
            recall_scale_down_idle_seconds=_positive_float(
                "TMCRA_SERVICE_RECALL_SCALE_DOWN_IDLE_SECONDS", 600.0
            ),
            recall_scale_down_cooldown_seconds=_positive_float(
                "TMCRA_SERVICE_RECALL_SCALE_DOWN_COOLDOWN_SECONDS", 60.0
            ),
            recall_target_utilization=_positive_float(
                "TMCRA_SERVICE_RECALL_TARGET_UTILIZATION", 0.70
            ),
            recall_warm_spare=_positive_int(
                "TMCRA_SERVICE_RECALL_WARM_SPARE", 1
            ),
            recall_gpu_headroom_bytes=_positive_int(
                "TMCRA_SERVICE_RECALL_GPU_HEADROOM_BYTES", 6 * 1024**3
            ),
            recall_replica_estimate_bytes=_positive_int(
                "TMCRA_SERVICE_RECALL_REPLICA_ESTIMATE_BYTES", 5 * 1024**3
            ),
            recall_scope_cache_size=_positive_int(
                "TMCRA_SERVICE_RECALL_SCOPE_CACHE_SIZE", 4
            ),
            recall_idle_cache_trim_enabled=_boolean(
                "TMCRA_SERVICE_RECALL_IDLE_CACHE_TRIM_ENABLED", False
            ),
            recall_idle_cache_seconds=_positive_float(
                "TMCRA_SERVICE_RECALL_IDLE_CACHE_SECONDS", 60.0
            ),
            recall_cache_trim_interval_seconds=_positive_float(
                "TMCRA_SERVICE_RECALL_CACHE_TRIM_INTERVAL_SECONDS", 5.0
            ),
            recall_cache_trim_cooldown_seconds=_positive_float(
                "TMCRA_SERVICE_RECALL_CACHE_TRIM_COOLDOWN_SECONDS", 300.0
            ),
            recall_cache_trim_min_bytes=_positive_int(
                "TMCRA_SERVICE_RECALL_CACHE_TRIM_MIN_BYTES", 4 * 1024**3
            ),
            slow_dirty_token_threshold=_positive_int(
                "TMCRA_SERVICE_SLOW_DIRTY_TOKEN_THRESHOLD", 32_000
            ),
            slow_dirty_user_turn_threshold=_positive_int(
                "TMCRA_SERVICE_SLOW_DIRTY_USER_TURN_THRESHOLD", 64
            ),
            slow_max_age_seconds=_positive_float(
                "TMCRA_SERVICE_SLOW_MAX_AGE_SECONDS", 86_400.0
            ),
            slow_min_token_threshold=_positive_int(
                "TMCRA_SERVICE_SLOW_MIN_TOKEN_THRESHOLD", 4_000
            ),
            slow_min_user_turn_threshold=_positive_int(
                "TMCRA_SERVICE_SLOW_MIN_USER_TURN_THRESHOLD", 8
            ),
            slow_min_interval_seconds=_positive_float(
                "TMCRA_SERVICE_SLOW_MIN_INTERVAL_SECONDS", 1_800.0
            ),
            slow_graph_drain_concurrency=_positive_int(
                "TMCRA_SERVICE_SLOW_GRAPH_DRAIN_CONCURRENCY", 1
            ),
            index_dirty_threshold=_positive_int(
                "TMCRA_SERVICE_INDEX_DIRTY_THRESHOLD", 16
            ),
            index_max_age_seconds=_positive_float(
                "TMCRA_SERVICE_INDEX_MAX_AGE_SECONDS", 2.0
            ),
            index_claim_wait_seconds=_positive_float(
                "TMCRA_SERVICE_INDEX_CLAIM_WAIT_SECONDS", 900.0
            ),
            index_generation_retention=_positive_int(
                "TMCRA_SERVICE_INDEX_GENERATION_RETENTION", 8
            ),
            scheduler_interval_seconds=_positive_float(
                "TMCRA_SERVICE_SCHEDULER_INTERVAL_SECONDS", 1.0
            ),
            quarantine_recovery_interval_seconds=_positive_float(
                "TMCRA_SERVICE_QUARANTINE_RECOVERY_INTERVAL_SECONDS", 15.0
            ),
            quarantine_recovery_lease_seconds=_positive_float(
                "TMCRA_SERVICE_QUARANTINE_RECOVERY_LEASE_SECONDS", 120.0
            ),
            quarantine_recovery_concurrency=_positive_int(
                "TMCRA_SERVICE_QUARANTINE_RECOVERY_CONCURRENCY", 4
            ),
            quarantine_recovery_max_job_attempts=_positive_int(
                "TMCRA_SERVICE_QUARANTINE_RECOVERY_MAX_JOB_ATTEMPTS", 3
            ),
            quarantine_recovery_max_local_repairs=_positive_int(
                "TMCRA_SERVICE_QUARANTINE_RECOVERY_MAX_LOCAL_REPAIRS", 8
            ),
            quarantine_recovery_backoff_seconds=_positive_float(
                "TMCRA_SERVICE_QUARANTINE_RECOVERY_BACKOFF_SECONDS", 30.0
            ),
            preload_online_engine=_boolean(
                "TMCRA_SERVICE_PRELOAD_ONLINE_ENGINE", True
            ),
            export_ttl_seconds=_positive_int(
                "TMCRA_SERVICE_EXPORT_TTL_SECONDS", 86_400
            ),
            webhook_signing_key=(
                os.getenv("TMCRA_WEBHOOK_SIGNING_KEY", "").strip() or None
            ),
            webhook_timeout_seconds=_positive_float(
                "TMCRA_WEBHOOK_TIMEOUT_SECONDS", 10.0
            ),
            writer_execution_mode=_choice(
                "TMCRA_SERVICE_WRITER_EXECUTION_MODE",
                "resident",
                {"resident", "subprocess"},
            ),
            writer_pool_size=_positive_int(
                "TMCRA_SERVICE_WRITER_POOL_SIZE", worker_concurrency
            ),
            writer_pool_startup_timeout_seconds=_positive_float(
                "TMCRA_SERVICE_WRITER_POOL_STARTUP_TIMEOUT_SECONDS", 120.0
            ),
            writer_pool_request_timeout_seconds=_positive_float(
                "TMCRA_SERVICE_WRITER_POOL_REQUEST_TIMEOUT_SECONDS", 900.0
            ),
            local_writer_recovery_concurrency=_positive_int(
                "TMCRA_LOCAL_WRITER_RECOVERY_CONCURRENCY", 1
            ),
            startup_preflight_mode=_choice(
                "TMCRA_SERVICE_STARTUP_PREFLIGHT_MODE",
                "full",
                {"off", "basic", "full"},
            ),
            staff_monitoring_key=(
                os.getenv("TMCRA_SERVICE_STAFF_MONITORING_KEY", "").strip()
                or None
            ),
            staff_latency_window_seconds=_positive_float(
                "TMCRA_SERVICE_STAFF_LATENCY_WINDOW_SECONDS", 300.0
            ),
            staff_latency_max_samples=_positive_int(
                "TMCRA_SERVICE_STAFF_LATENCY_MAX_SAMPLES", 4096
            ),
            staff_recent_error_window_seconds=_positive_float(
                "TMCRA_SERVICE_STAFF_RECENT_ERROR_WINDOW_SECONDS", 86_400.0
            ),
            staff_recent_error_limit=_positive_int(
                "TMCRA_SERVICE_STAFF_RECENT_ERROR_LIMIT", 20
            ),
            api_access_log_enabled=_boolean(
                "TMCRA_SERVICE_API_ACCESS_LOG_ENABLED", True
            ),
            api_access_log_path=Path(
                os.getenv(
                    "TMCRA_SERVICE_API_ACCESS_LOG_PATH",
                    str(state_dir / "api-access.jsonl"),
                )
            ).expanduser().resolve(),
            diagnostic_log_enabled=_boolean(
                "TMCRA_SERVICE_DIAGNOSTIC_LOG_ENABLED", True
            ),
            diagnostic_log_path=Path(
                os.getenv(
                    "TMCRA_SERVICE_DIAGNOSTIC_LOG_PATH",
                    str(state_dir / "api-errors.jsonl"),
                )
            ).expanduser().resolve(),
            service_release_id=(
                os.getenv("TMCRA_SERVICE_RELEASE_ID", "").strip() or None
            ),
            service_release_sha256=(
                os.getenv("TMCRA_SERVICE_RELEASE_SHA256", "").strip() or None
            ),
            service_release_channel=(
                os.getenv("TMCRA_SERVICE_RELEASE_CHANNEL", "").strip() or None
            ),
            service_canary_percent=_optional_float(
                "TMCRA_SERVICE_CANARY_PERCENT"
            ),
            service_rollback_release_id=(
                os.getenv("TMCRA_SERVICE_ROLLBACK_RELEASE_ID", "").strip()
                or None
            ),
            audio_asr_base_url=(
                os.getenv("TMCRA_AUDIO_ASR_BASE_URL", "").strip().rstrip("/")
                or None
            ),
            audio_asr_api_key_file=(
                Path(os.environ["TMCRA_AUDIO_ASR_API_KEY_FILE"])
                .expanduser()
                .resolve()
                if os.getenv("TMCRA_AUDIO_ASR_API_KEY_FILE", "").strip()
                else None
            ),
            audio_asr_timeout_seconds=_positive_float(
                "TMCRA_AUDIO_ASR_TIMEOUT_SECONDS", 120.0
            ),
            audio_asr_max_request_bytes=_positive_int(
                "TMCRA_AUDIO_ASR_MAX_REQUEST_BYTES", 2_621_440
            ),
        )

    def required_paths(self) -> dict[str, Path]:
        paths = {
            "v4_root": self.v4_root,
            "integrated_repo": self.integrated_repo,
            "writer_env": self.writer_env,
            "embedding_model": self.embedding_model,
            "native_harness": self.native_harness,
            "checkpoint": self.checkpoint,
            "cross_model": self.cross_model,
        }
        if self.learned_graph_enabled:
            paths["node_model"] = self.node_model
            paths["path_model"] = self.path_model
        if self.audio_asr_api_key_file is not None:
            paths["audio_asr_api_key_file"] = self.audio_asr_api_key_file
        return paths

    def validate(self) -> None:
        if not self.bind_host:
            raise SettingsError("TMCRA_SERVICE_BIND_HOST cannot be empty")
        if (self.audio_asr_base_url is None) != (self.audio_asr_api_key_file is None):
            raise SettingsError(
                "TMCRA_AUDIO_ASR_BASE_URL and TMCRA_AUDIO_ASR_API_KEY_FILE "
                "must be configured together"
            )
        if self.audio_asr_base_url is not None:
            parsed_asr = urlsplit(self.audio_asr_base_url)
            if (
                parsed_asr.scheme != "http"
                or parsed_asr.hostname != "127.0.0.1"
                or parsed_asr.username is not None
                or parsed_asr.password is not None
                or parsed_asr.query
                or parsed_asr.fragment
                or parsed_asr.path not in {"", "/", "/v1"}
            ):
                raise SettingsError(
                    "TMCRA_AUDIO_ASR_BASE_URL must use an exact loopback HTTP URL"
                )
        if self.webhook_signing_key is not None and len(self.webhook_signing_key) < 32:
            raise SettingsError("TMCRA_WEBHOOK_SIGNING_KEY must contain at least 32 characters")
        if self.writer_execution_mode not in {"resident", "subprocess"}:
            raise SettingsError(
                "TMCRA_SERVICE_WRITER_EXECUTION_MODE must be resident or subprocess"
            )
        if self.startup_preflight_mode not in {"off", "basic", "full"}:
            raise SettingsError(
                "TMCRA_SERVICE_STARTUP_PREFLIGHT_MODE must be off, basic, or full"
            )
        if self.api_access_log_enabled:
            if self.api_access_log_path is None:
                raise SettingsError(
                    "TMCRA_SERVICE_API_ACCESS_LOG_PATH is required when API access logging is enabled"
                )
            if not self.api_access_log_path.resolve().is_relative_to(
                self.state_dir.resolve()
            ):
                raise SettingsError(
                    "production API access log must stay inside TMCRA_SERVICE_STATE_DIR"
                )
        if self.diagnostic_log_enabled:
            if self.diagnostic_log_path is None:
                raise SettingsError(
                    "TMCRA_SERVICE_DIAGNOSTIC_LOG_PATH is required when diagnostic logging is enabled"
                )
            if not self.diagnostic_log_path.resolve().is_relative_to(
                self.state_dir.resolve()
            ):
                raise SettingsError(
                    "production diagnostic log must stay inside TMCRA_SERVICE_STATE_DIR"
                )
        if self.writer_pool_size <= 0:
            raise SettingsError("TMCRA_SERVICE_WRITER_POOL_SIZE must be positive")
        if self.writer_pool_startup_timeout_seconds <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_WRITER_POOL_STARTUP_TIMEOUT_SECONDS must be positive"
            )
        if self.writer_pool_request_timeout_seconds <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_WRITER_POOL_REQUEST_TIMEOUT_SECONDS must be positive"
            )
        if self.slow_graph_drain_concurrency > 4:
            raise SettingsError(
                "TMCRA_SERVICE_SLOW_GRAPH_DRAIN_CONCURRENCY cannot exceed 4"
            )
        if self.local_writer_recovery_concurrency > self.writer_pool_size:
            raise SettingsError(
                "TMCRA_LOCAL_WRITER_RECOVERY_CONCURRENCY cannot exceed "
                "TMCRA_SERVICE_WRITER_POOL_SIZE"
            )
        if (
            self.writer_pool_size > 1
            and self.local_writer_recovery_concurrency >= self.writer_pool_size
        ):
            raise SettingsError(
                "TMCRA_LOCAL_WRITER_RECOVERY_CONCURRENCY must reserve one "
                "resident Writer slot for online traffic"
            )
        for name, value in (
            (
                "TMCRA_PROVIDER_BILLING_CIRCUIT_SECONDS",
                self.provider_billing_circuit_seconds,
            ),
            (
                "TMCRA_PROVIDER_AUTH_CIRCUIT_SECONDS",
                self.provider_auth_circuit_seconds,
            ),
        ):
            if value <= 0 or value > 86_400:
                raise SettingsError(f"{name} must be between 0 and 86400 seconds")
        if self.write_admission_retry_seconds <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_WRITE_ADMISSION_RETRY_SECONDS must be positive"
            )
        if self.recall_pool_min_size <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_POOL_MIN_SIZE must be positive"
            )
        if self.recall_pool_max_size < self.recall_pool_min_size:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_POOL_MAX_SIZE cannot be smaller than "
                "TMCRA_SERVICE_RECALL_POOL_MIN_SIZE"
            )
        if self.recall_global_queue_limit <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_GLOBAL_QUEUE_LIMIT must be positive"
            )
        if self.recall_tenant_queue_limit <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_TENANT_QUEUE_LIMIT must be positive"
            )
        if self.recall_tenant_queue_limit > self.recall_global_queue_limit:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_TENANT_QUEUE_LIMIT cannot exceed "
                "TMCRA_SERVICE_RECALL_GLOBAL_QUEUE_LIMIT"
            )
        if self.recall_queue_timeout_seconds <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_QUEUE_TIMEOUT_SECONDS must be positive"
            )
        if self.recall_scale_up_sustain_seconds <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_SCALE_UP_SUSTAIN_SECONDS must be positive"
            )
        if self.recall_scale_up_cooldown_seconds <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_SCALE_UP_COOLDOWN_SECONDS must be positive"
            )
        if self.recall_scale_down_idle_seconds <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_SCALE_DOWN_IDLE_SECONDS must be positive"
            )
        if self.recall_scale_down_cooldown_seconds <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_SCALE_DOWN_COOLDOWN_SECONDS must be positive"
            )
        if not 0.0 < self.recall_target_utilization < 1.0:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_TARGET_UTILIZATION must be between 0 and 1"
            )
        if self.recall_warm_spare <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_WARM_SPARE must be positive"
            )
        if self.recall_gpu_headroom_bytes <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_GPU_HEADROOM_BYTES must be positive"
            )
        if self.recall_replica_estimate_bytes <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_REPLICA_ESTIMATE_BYTES must be positive"
            )
        if self.recall_scope_cache_size <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_SCOPE_CACHE_SIZE must be positive"
            )
        for name, value in (
            (
                "TMCRA_SERVICE_RECALL_IDLE_CACHE_SECONDS",
                self.recall_idle_cache_seconds,
            ),
            (
                "TMCRA_SERVICE_RECALL_CACHE_TRIM_INTERVAL_SECONDS",
                self.recall_cache_trim_interval_seconds,
            ),
            (
                "TMCRA_SERVICE_RECALL_CACHE_TRIM_COOLDOWN_SECONDS",
                self.recall_cache_trim_cooldown_seconds,
            ),
        ):
            if value <= 0:
                raise SettingsError(f"{name} must be positive")
        if self.recall_cache_trim_min_bytes <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_RECALL_CACHE_TRIM_MIN_BYTES must be positive"
            )
        if self.index_generation_retention <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_INDEX_GENERATION_RETENTION must be positive"
            )
        if self.staff_monitoring_key is not None and not (
            32 <= len(self.staff_monitoring_key) <= 512
        ):
            raise SettingsError(
                "TMCRA_SERVICE_STAFF_MONITORING_KEY must contain 32 to 512 characters"
            )
        if self.staff_latency_window_seconds <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_STAFF_LATENCY_WINDOW_SECONDS must be positive"
            )
        if not 1 <= self.staff_latency_max_samples <= 1_000_000:
            raise SettingsError(
                "TMCRA_SERVICE_STAFF_LATENCY_MAX_SAMPLES must be between 1 and 1000000"
            )
        if self.staff_recent_error_window_seconds <= 0:
            raise SettingsError(
                "TMCRA_SERVICE_STAFF_RECENT_ERROR_WINDOW_SECONDS must be positive"
            )
        if not 1 <= self.staff_recent_error_limit <= 100:
            raise SettingsError(
                "TMCRA_SERVICE_STAFF_RECENT_ERROR_LIMIT must be between 1 and 100"
            )
        for name, value in (
            ("TMCRA_SERVICE_RELEASE_ID", self.service_release_id),
            (
                "TMCRA_SERVICE_ROLLBACK_RELEASE_ID",
                self.service_rollback_release_id,
            ),
        ):
            if value is not None and not RELEASE_IDENTIFIER_RE.fullmatch(value):
                raise SettingsError(f"{name} has an invalid release identifier")
        if (
            self.service_release_channel is not None
            and not RELEASE_CHANNEL_RE.fullmatch(self.service_release_channel)
        ):
            raise SettingsError(
                "TMCRA_SERVICE_RELEASE_CHANNEL has an invalid channel name"
            )
        if (
            self.service_release_sha256 is not None
            and not SHA256_RE.fullmatch(self.service_release_sha256)
        ):
            raise SettingsError(
                "TMCRA_SERVICE_RELEASE_SHA256 must be a 64-character hexadecimal digest"
            )
        if self.service_canary_percent is not None and not (
            0.0 <= self.service_canary_percent <= 100.0
        ):
            raise SettingsError(
                "TMCRA_SERVICE_CANARY_PERCENT must be between 0 and 100"
            )
        missing = [name for name, path in self.required_paths().items() if not path.exists()]
        if missing:
            raise SettingsError("required service paths are missing: " + ",".join(missing))
