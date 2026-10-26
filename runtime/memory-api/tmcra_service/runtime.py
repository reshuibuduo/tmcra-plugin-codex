from __future__ import annotations

import json
import hashlib
import inspect
import os
import signal
import sqlite3
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .adapters.v4 import V4OnlineEngine, V4StorageAdapter
from .commercial import CommercialControl, CommercialContractError
from .control_db import ControlDB, StaleSourceAccountingRecovery
from .diagnostic_log import DiagnosticJournal
from .gpu_capacity import CudaReplicaCapacityGuard
from .gpu_scheduler import GpuWorkload, GpuWorkloadScheduler
from .jobs import (
    CANCELLED,
    FAILED,
    PENDING,
    RUNNING,
    STAGE_CANCELLED,
    STAGE_FAILED,
    STAGE_RUNNING,
    STAGE_SUCCEEDED,
    SUCCEEDED,
    Job,
    JobStateError,
    JobStore,
    ResumeAuthorization,
)
from .recall_pool import RecallEnginePool
from .settings import ServiceSettings
from .usage_attribution import SYSTEM_MAINTENANCE, UsageAttribution


class RuntimeErrorBase(RuntimeError):
    pass


class UnsupportedJob(RuntimeErrorBase):
    pass


class IncompleteWriterStage(RuntimeErrorBase):
    """Carries a durable partial result while keeping the Writer stage failed."""

    def __init__(
        self,
        report: Mapping[str, Any],
        *,
        accounting_operation_id: str,
    ) -> None:
        super().__init__(
            "writer reached the Source durability boundary but enrichment is incomplete"
        )
        self.report = dict(report)
        self.accounting_operation_id = accounting_operation_id


@dataclass(frozen=True)
class WorkerStatus:
    worker_id: str
    alive: bool
    active_job_id: str | None
    started_at: float


class LazyOnlineEngine:
    def __init__(
        self,
        settings: ServiceSettings,
        gpu_scheduler: GpuWorkloadScheduler | None = None,
    ) -> None:
        self.settings = settings
        self.gpu_scheduler = gpu_scheduler
        self._pool: RecallEnginePool[V4OnlineEngine] | None = None
        self._capacity_guard = CudaReplicaCapacityGuard(
            device=str(getattr(settings, "device", "cpu")),
            headroom_bytes=int(
                getattr(settings, "recall_gpu_headroom_bytes", 6 * 1024**3)
            ),
            replica_estimate_bytes=int(
                getattr(settings, "recall_replica_estimate_bytes", 5 * 1024**3)
            ),
        )
        self._lock = threading.Lock()
        self._stopped = False
        self._process_restart_lock = threading.Lock()
        self._process_restart_scheduled = False
        self._cache_trim_enabled = bool(
            getattr(settings, "recall_idle_cache_trim_enabled", False)
        )
        self._cache_trim_idle_seconds = float(
            getattr(settings, "recall_idle_cache_seconds", 60.0)
        )
        self._cache_trim_interval_seconds = float(
            getattr(settings, "recall_cache_trim_interval_seconds", 5.0)
        )
        self._cache_trim_cooldown_seconds = float(
            getattr(settings, "recall_cache_trim_cooldown_seconds", 300.0)
        )
        self._cache_trim_min_bytes = int(
            getattr(settings, "recall_cache_trim_min_bytes", 4 * 1024**3)
        )
        self._cache_trim_lock = threading.Lock()
        self._cache_trim_stop = threading.Event()
        self._cache_trim_thread: threading.Thread | None = None
        self._cache_trim_idle_since: float | None = None
        self._cache_trim_last_attempt_at = float("-inf")
        self._cache_trim_last_success_at: float | None = None
        self._cache_trim_attempts = 0
        self._cache_trim_successes = 0
        self._cache_trim_failures = 0
        self._cache_trim_total_released_bytes = 0
        self._cache_trim_last_released_bytes = 0
        self._cache_trim_last_error: str | None = None

    @staticmethod
    def _fatal_recall_exception(error: BaseException) -> bool:
        """Recognize failures that make one resident CUDA replica unsafe.

        The match is intentionally narrow. Provider/rate-limit errors,
        request validation, and ordinary RuntimeError/ValueError instances do
        not retire a healthy multi-gigabyte model replica.
        """

        pending: list[BaseException] = [error]
        seen: set[int] = set()
        cuda_fatal_markers = (
            "cuda out of memory",
            "cuda error: out of memory",
            "cuda error: device-side assert triggered",
            "cuda error: an illegal memory access was encountered",
            "cuda error: illegal memory access",
            "cuda error: unspecified launch failure",
            "cublas_status_internal_error",
            "cublas_status_execution_failed",
            "cudnn_status_internal_error",
            "cudnn_status_execution_failed",
        )
        while pending:
            current = pending.pop()
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            current_type = type(current)
            if current_type.__name__ == "OutOfMemoryError" and current_type.__module__.startswith(
                "torch"
            ):
                return True
            message = str(current).casefold()
            if any(marker in message for marker in cuda_fatal_markers):
                return True
            cause = current.__cause__
            context = current.__context__
            if cause is not None:
                pending.append(cause)
            if context is not None and context is not cause:
                pending.append(context)
        return False

    @staticmethod
    def _cuda_context_corrupted(error: BaseException) -> bool:
        markers = (
            "cuda error: device-side assert triggered",
            "cuda error: an illegal memory access was encountered",
            "cuda error: illegal memory access",
            "cuda error: unspecified launch failure",
            "cublas_status_internal_error",
            "cublas_status_execution_failed",
            "cudnn_status_internal_error",
            "cudnn_status_execution_failed",
        )
        pending: list[BaseException] = [error]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if any(marker in str(current).casefold() for marker in markers):
                return True
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if (
                current.__context__ is not None
                and current.__context__ is not current.__cause__
            ):
                pending.append(current.__context__)
        return False

    def _schedule_cuda_context_restart(self) -> None:
        with self._process_restart_lock:
            if self._process_restart_scheduled:
                return
            self._process_restart_scheduled = True

        def terminate() -> None:
            os.kill(os.getpid(), signal.SIGTERM)

        timer = threading.Timer(0.5, terminate)
        timer.daemon = True
        timer.name = "tmcra-cuda-context-restart"
        timer.start()

    def _pool_fatal_exception(self, error: BaseException) -> bool:
        fatal = self._fatal_recall_exception(error)
        if fatal and self._cuda_context_corrupted(error):
            # CUDA context corruption is process-wide. Quarantining one lane is
            # insufficient; terminate the child after the failed response can
            # unwind so the resident supervisor starts a clean process.
            self._schedule_cuda_context_restart()
        return fatal

    def _ensure_cache_trim_monitor(self) -> None:
        if not self._cache_trim_enabled:
            return
        if not str(getattr(self.settings, "device", "cpu")).startswith("cuda"):
            return
        with self._cache_trim_lock:
            if self._cache_trim_thread is not None or self._cache_trim_stop.is_set():
                return
            thread = threading.Thread(
                target=self._cache_trim_monitor,
                name="tmcra-recall-cuda-cache-trimmer",
                daemon=True,
            )
            self._cache_trim_thread = thread
            thread.start()

    def _cache_trim_monitor(self) -> None:
        while not self._cache_trim_stop.wait(self._cache_trim_interval_seconds):
            try:
                self._trim_cuda_cache_if_idle()
            except Exception as exc:
                now = time.monotonic()
                with self._cache_trim_lock:
                    self._cache_trim_failures += 1
                    self._cache_trim_last_attempt_at = now
                    self._cache_trim_last_error = (
                        f"{type(exc).__name__}: {str(exc).strip()}"[:500]
                    )

    @staticmethod
    def _pool_fully_idle(status: Any) -> bool:
        current_size = int(getattr(status, "current_size", 0) or 0)
        return (
            current_size > 0
            and int(getattr(status, "loaded", 0) or 0) == current_size
            and int(getattr(status, "active", 0) or 0) == 0
            and int(getattr(status, "pending", 0) or 0) == 0
            and int(getattr(status, "retiring", 0) or 0) == 0
            and int(getattr(status, "idle", 0) or 0) == current_size
            and not bool(getattr(status, "warming", False))
            and not bool(getattr(status, "scaling", False))
            and not bool(getattr(status, "closed", False))
        )

    def _trim_cuda_cache_if_idle(self, *, now: float | None = None) -> bool:
        """Release only unused allocator blocks after a sustained idle period."""

        if not self._cache_trim_enabled:
            return False
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            pool = self._pool
            stopped = self._stopped
        if stopped or pool is None:
            return False
        if not self._pool_fully_idle(pool.status()):
            with self._cache_trim_lock:
                self._cache_trim_idle_since = None
            return False

        with self._cache_trim_lock:
            if self._cache_trim_idle_since is None:
                self._cache_trim_idle_since = timestamp
                return False
            if timestamp - self._cache_trim_idle_since < self._cache_trim_idle_seconds:
                return False
            if (
                timestamp - self._cache_trim_last_attempt_at
                < self._cache_trim_cooldown_seconds
            ):
                return False

        before = self._capacity_guard.snapshot()
        reclaimable = int(before.reusable_reserved_bytes or 0)
        if reclaimable < self._cache_trim_min_bytes:
            return False

        def release_unused_blocks() -> None:
            import torch

            torch.cuda.empty_cache()

        admitted, _result = pool.run_idle_maintenance(release_unused_blocks)
        if not admitted:
            with self._cache_trim_lock:
                self._cache_trim_idle_since = None
            return False
        after = self._capacity_guard.snapshot()
        released = max(
            0,
            int(before.reserved_bytes or 0) - int(after.reserved_bytes or 0),
        )
        with self._cache_trim_lock:
            self._cache_trim_attempts += 1
            self._cache_trim_successes += 1
            self._cache_trim_last_attempt_at = timestamp
            self._cache_trim_last_success_at = timestamp
            self._cache_trim_total_released_bytes += released
            self._cache_trim_last_released_bytes = released
            self._cache_trim_last_error = None
        return True

    def _cache_trim_status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._cache_trim_lock:
            thread = self._cache_trim_thread
            idle_since = self._cache_trim_idle_since
            last_success = self._cache_trim_last_success_at
            return {
                "enabled": self._cache_trim_enabled,
                "monitor_alive": bool(thread is not None and thread.is_alive()),
                "idle_seconds": self._cache_trim_idle_seconds,
                "interval_seconds": self._cache_trim_interval_seconds,
                "cooldown_seconds": self._cache_trim_cooldown_seconds,
                "min_reclaimable_bytes": self._cache_trim_min_bytes,
                "idle_for_seconds": (
                    max(0.0, now - idle_since) if idle_since is not None else 0.0
                ),
                "last_success_age_seconds": (
                    max(0.0, now - last_success)
                    if last_success is not None
                    else None
                ),
                "attempts": self._cache_trim_attempts,
                "successes": self._cache_trim_successes,
                "failures": self._cache_trim_failures,
                "last_released_bytes": self._cache_trim_last_released_bytes,
                "total_released_bytes": self._cache_trim_total_released_bytes,
                "last_error": self._cache_trim_last_error,
            }

    def get(self) -> RecallEnginePool[V4OnlineEngine]:
        with self._lock:
            if self._stopped:
                raise RuntimeErrorBase("online recall engine pool is stopped")
            if self._pool is None:
                self._pool = RecallEnginePool(
                    lambda: V4OnlineEngine(
                        self.settings,
                        gpu_scheduler=self.gpu_scheduler,
                    ),
                    min_size=int(
                        getattr(self.settings, "recall_pool_min_size", 1)
                    ),
                    max_size=int(
                        getattr(self.settings, "recall_pool_max_size", 1)
                    ),
                    max_pending=int(
                        getattr(self.settings, "recall_global_queue_limit", 8)
                    ),
                    per_tenant_pending=int(
                        getattr(self.settings, "recall_tenant_queue_limit", 2)
                    ),
                    queue_timeout=float(
                        getattr(
                            self.settings, "recall_queue_timeout_seconds", 30.0
                        )
                    ),
                    capacity_guard=self._capacity_guard,
                    fatal_exception_predicate=self._pool_fatal_exception,
                    target_utilization=float(
                        getattr(self.settings, "recall_target_utilization", 0.70)
                    ),
                    warm_spares=int(
                        getattr(self.settings, "recall_warm_spare", 1)
                    ),
                    scale_up_sustain_seconds=float(
                        getattr(
                            self.settings,
                            "recall_scale_up_sustain_seconds",
                            2.0,
                        )
                    ),
                    scale_up_cooldown_seconds=float(
                        getattr(
                            self.settings,
                            "recall_scale_up_cooldown_seconds",
                            5.0,
                        )
                    ),
                    scale_down_idle_seconds=float(
                        getattr(
                            self.settings,
                            "recall_scale_down_idle_seconds",
                            600.0,
                        )
                    ),
                    scale_down_cooldown_seconds=float(
                        getattr(
                            self.settings,
                            "recall_scale_down_cooldown_seconds",
                            60.0,
                        )
                    ),
                    forward_tenant_as="provider_tenant_id",
                )
            pool = self._pool
        self._ensure_cache_trim_monitor()
        return pool

    def execute(
        self,
        tenant_id: str,
        operation: Callable[[V4OnlineEngine], Any],
        *,
        queue_timeout: float | None = None,
        workload: GpuWorkload = GpuWorkload.INDEX_BACKGROUND,
        scheduler_timeout: float | None = None,
    ) -> Any:
        """Run non-recall model work on an already warmed online replica."""

        lease = (
            self.gpu_scheduler.lease(workload, timeout=scheduler_timeout)
            if self.gpu_scheduler is not None
            else nullcontext()
        )
        with lease:
            return self.get().execute(
                tenant_id,
                operation,
                queue_timeout=queue_timeout,
            )

    @property
    def loaded_count(self) -> int:
        with self._lock:
            pool = self._pool
        if pool is None:
            return 0
        return int(pool.loaded_count)

    @property
    def loaded(self) -> bool:
        with self._lock:
            if self._stopped:
                return False
        minimum = int(getattr(self.settings, "recall_pool_min_size", 1))
        return self.loaded_count >= minimum

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        converter = getattr(value, "as_dict", None)
        if converter is not None and callable(converter):
            return dict(converter())
        if isinstance(value, Mapping):
            return dict(value)
        return {"value": str(value)}

    def status(self) -> dict[str, Any]:
        with self._lock:
            pool = self._pool
            stopped = self._stopped
        minimum = int(getattr(self.settings, "recall_pool_min_size", 1))
        maximum = int(getattr(self.settings, "recall_pool_max_size", minimum))
        if pool is None:
            pool_status: dict[str, Any] = {
                "min_size": minimum,
                "max_size": maximum,
                "current_size": 0,
                "desired_size": minimum,
                "loaded": 0,
                "fully_loaded": False,
                "active": 0,
                "idle": 0,
                "pending": 0,
                "scaling": False,
                "closed": stopped,
            }
            pool_metrics: dict[str, Any] = {}
        else:
            pool_status = self._as_dict(pool.status())
            pool_metrics = self._as_dict(pool.metrics())
        try:
            gpu_capacity = self._capacity_guard.snapshot().as_dict()
        except Exception:
            # Status is operational metadata; a failed probe must not make the
            # status endpoint itself fail or expose provider/config secrets.
            gpu_capacity = {
                "device": str(getattr(self.settings, "device", "cpu")),
                "can_add_replica": False,
                "reason": "capacity_probe_unavailable",
            }
        result = {
            "loaded": (
                not stopped
                and not bool(pool_status.get("closed", False))
                and int(pool_status.get("loaded", 0) or 0) >= minimum
            ),
            "loaded_count": int(pool_status.get("loaded", 0) or 0),
            "minimum_loaded": minimum,
            "stopped": stopped,
            "process_restart_scheduled": self._process_restart_scheduled,
            "pool": pool_status,
            "metrics": pool_metrics,
            "gpu_capacity": gpu_capacity,
            "cuda_cache_trim": self._cache_trim_status(),
        }
        if self.gpu_scheduler is not None:
            result["gpu_scheduler"] = self.gpu_scheduler.status()
        return result

    def stop(self, timeout: float | None = None) -> None:
        with self._lock:
            self._stopped = True
            pool = self._pool
        self._cache_trim_stop.set()
        with self._cache_trim_lock:
            cache_trim_thread = self._cache_trim_thread
        if (
            cache_trim_thread is not None
            and cache_trim_thread is not threading.current_thread()
        ):
            cache_trim_thread.join(timeout=1.0)
        if pool is None:
            return
        close = getattr(pool, "close", None)
        if close is not None and callable(close):
            close(wait=True, timeout=timeout)
            return
        stop = getattr(pool, "stop", None)
        if stop is not None and callable(stop):
            stop(timeout=timeout)


class ServiceWorker:
    def __init__(
        self,
        *,
        settings: ServiceSettings,
        database: ControlDB,
        jobs: JobStore,
        storage: V4StorageAdapter,
        online: LazyOnlineEngine | None = None,
        gpu_scheduler: GpuWorkloadScheduler | None = None,
        writer_uses_local_gpu: bool | None = None,
        slow_graph_uses_local_gpu: bool | None = None,
        commercial: CommercialControl | None = None,
        on_ingest_committed: Callable[[str, str, str, int], None] | None = None,
        on_generation_committed: Callable[[str, str, int], None] | None = None,
        diagnostic_log: DiagnosticJournal | None = None,
        poll_seconds: float = 0.5,
    ) -> None:
        self.settings = settings
        self.database = database
        self.jobs = jobs
        self.storage = storage
        self.online = online
        self.gpu_scheduler = gpu_scheduler
        self.writer_uses_local_gpu = (
            str(os.environ.get("TMCRA_WRITER_PROVIDER") or "").strip().casefold()
            == "local_qwen"
            if writer_uses_local_gpu is None
            else bool(writer_uses_local_gpu)
        )
        self.slow_graph_uses_local_gpu = (
            str(os.environ.get("TMCRA_SLOW_GRAPH_PROVIDER") or "")
            .strip()
            .casefold()
            in {"local-qwen", "local_qwen"}
            if slow_graph_uses_local_gpu is None
            else bool(slow_graph_uses_local_gpu)
        )
        self.commercial = commercial
        self.on_ingest_committed = on_ingest_committed
        self.on_generation_committed = on_generation_committed
        self.diagnostic_log = diagnostic_log
        self.poll_seconds = poll_seconds
        self.worker_id = f"tmcra-service-{os.getpid()}-{id(self):x}"
        self.started_at = time.time()
        self.active_job_id: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._futures: set[Future[Any]] = set()
        self._state_lock = threading.Lock()
        self._active_jobs: dict[str, Job] = {}
        self._active_job_lanes: dict[str, str] = {}
        # Recovery identity belongs to the claimed execution attempt. The
        # controller can change mutable scope state while that attempt runs.
        self._claimed_quarantine_recovery_jobs: set[str] = set()
        self._scope_counts: dict[tuple[str, str], int] = {}
        self._scope_lane_counts: dict[tuple[str, str], dict[str, int]] = {}
        self._scope_locks: dict[tuple[str, str, str], threading.Lock] = {}
        self._transient_db_error_count = 0
        self._last_transient_db_error_log_at = 0.0

    def _record_exception(
        self,
        exc: BaseException,
        *,
        operation: str,
        job: Job | None = None,
        stage_id: str | None = None,
        stage_name: str | None = None,
        stage_attempt: int | None = None,
        error_code: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        journal = self.diagnostic_log
        if journal is None:
            return
        payload = dict(getattr(job, "payload", None) or {}) if job is not None else {}
        journal.record_exception(
            exc,
            component="service_worker",
            operation=operation,
            job_id=getattr(job, "job_id", None),
            job_type=str(payload.get("job_type") or "") or None,
            stage_id=stage_id,
            stage_name=stage_name,
            stage_attempt=stage_attempt,
            tenant_id=getattr(job, "tenant_id", None),
            scope_name=getattr(job, "scope_name", None),
            worker_id=self.worker_id,
            error_code=error_code or type(exc).__name__,
            context=context,
        )

    def _worker_concurrency(self) -> int:
        value = getattr(self.settings, "worker_concurrency", 4)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("worker_concurrency must be positive")
        return value

    @staticmethod
    def _is_transient_db_contention(error: sqlite3.OperationalError) -> bool:
        message = str(error).casefold()
        return "database is locked" in message or "database is busy" in message

    def _control_db_operation(self, operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except sqlite3.OperationalError as error:
            if not self._is_transient_db_contention(error):
                raise
            self._transient_db_error_count += 1
            now = time.monotonic()
            if now - self._last_transient_db_error_log_at >= 30.0:
                self._record_exception(
                    error,
                    operation="control_db_operation",
                    error_code="control_db_contention",
                    context={"occurrences": self._transient_db_error_count},
                )
                traceback.print_exc()
                self._last_transient_db_error_log_at = now
            return None

    @staticmethod
    def _scope_key(job: Job) -> tuple[str, str]:
        return (
            str(getattr(job, "tenant_id", "") or ""),
            str(getattr(job, "scope_name", "") or "default"),
        )

    @staticmethod
    def _execution_lane(payload: Mapping[str, Any]) -> str:
        job_type = str(payload.get("job_type") or "")
        if job_type in {
            "ingest",
            "reindex",
            "consolidate",
            "delete_memories",
            "delete_session",
        }:
            # Mutations are serialized within one scope. Different scopes still
            # run in parallel, but Writer/delta/base activation cannot race.
            return "mutation"
        return "exclusive"

    @staticmethod
    def _index_workload(job: Job) -> GpuWorkload:
        payload = dict(getattr(job, "payload", None) or {})
        job_type = str(payload.get("job_type") or "")
        # User-visible write/delete indexing keeps foreground priority.
        # Automatic compaction and rebuilds consume spare recall capacity.
        if job_type in {"ingest", "delete_memories", "delete_session"} and not bool(
            payload.get("auto")
        ):
            return GpuWorkload.INDEX_FOREGROUND
        return GpuWorkload.INDEX_BACKGROUND

    def _is_quarantine_recovery_ingest(self, job: Job) -> bool:
        payload = getattr(job, "payload", None) or {}
        if str(payload.get("job_type") or "") != "ingest":
            return False
        if job.job_id in self._claimed_quarantine_recovery_jobs:
            return True
        return bool(
            self.commercial is not None
            and self.commercial.is_quarantine_recovery_job(
                job.tenant_id,
                job.scope_name,
                job.job_id,
            )
        )

    def _job_lane(self, job: Job) -> str:
        return self._execution_lane(getattr(job, "payload", None) or {})

    def _job_lock_key(self, job: Job) -> tuple[str, str, str]:
        return (*self._scope_key(job), self._job_lane(job))

    def _mark_active(self, job: Job) -> None:
        scope_key = self._scope_key(job)
        lane = self._job_lane(job)
        with self._state_lock:
            self._active_jobs[job.job_id] = job
            self._active_job_lanes[job.job_id] = lane
            self._scope_counts[scope_key] = self._scope_counts.get(scope_key, 0) + 1
            lanes = self._scope_lane_counts.setdefault(scope_key, {})
            lanes[lane] = lanes.get(lane, 0) + 1
            self.active_job_id = next(iter(self._active_jobs), None)

    def _unmark_active(self, job: Job) -> None:
        scope_key = self._scope_key(job)
        with self._state_lock:
            self._active_jobs.pop(job.job_id, None)
            lane = self._active_job_lanes.pop(job.job_id, None)
            if lane is None:
                lane = self._execution_lane(getattr(job, "payload", None) or {})
            self._claimed_quarantine_recovery_jobs.discard(job.job_id)
            count = self._scope_counts.get(scope_key, 0)
            if count <= 1:
                self._scope_counts.pop(scope_key, None)
            else:
                self._scope_counts[scope_key] = count - 1
            lanes = self._scope_lane_counts.get(scope_key, {})
            lane_count = lanes.get(lane, 0)
            if lane_count <= 1:
                lanes.pop(lane, None)
            else:
                lanes[lane] = lane_count - 1
            if not lanes:
                self._scope_lane_counts.pop(scope_key, None)
            self.active_job_id = next(iter(self._active_jobs), None)

    def _scope_lane_is_busy(self, scope_key: tuple[str, str], lane: str) -> bool:
        with self._state_lock:
            lanes = self._scope_lane_counts.get(scope_key, {})
            if not lanes:
                return False
            if lane == "exclusive" or "exclusive" in lanes:
                return True
            return lanes.get(lane, 0) > 0

    def _scope_lock(self, lock_key: tuple[str, str, str]) -> threading.Lock:
        with self._state_lock:
            lock = self._scope_locks.get(lock_key)
            if lock is None:
                lock = threading.Lock()
                self._scope_locks[lock_key] = lock
            return lock

    def _scope_lock_path(self, lock_key: tuple[str, str, str]) -> Path:
        if self.database is None:
            return Path(os.getcwd()) / ".tmcra-test-scope.lock"
        state_dir = Path(
            getattr(self.settings, "state_dir", None)
            or Path(self.database.path).parent
        )
        lock_dir = state_dir / "scope-mutation-locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        identity = "\0".join(lock_key).encode("utf-8")
        return lock_dir / f"{hashlib.sha256(identity).hexdigest()}.lock"

    @contextmanager
    def _scope_execution_lock(
        self,
        lock_key: tuple[str, str, str],
        *,
        blocking: bool = True,
    ) -> Any:
        """Serialize one scope lane across threads and service processes."""

        # All current lanes conflict with each other. Normalize the physical
        # lock identity so an expired mutation cannot overlap a newly claimed
        # exclusive job in another process.
        lock_key = (lock_key[0], lock_key[1], "scope")
        local_lock = self._scope_lock(lock_key)
        if not local_lock.acquire(blocking=blocking):
            raise BlockingIOError("scope mutation lane is active")
        lock_file = None
        windows_lock = None
        try:
            if os.name == "nt":
                from tmcra_local_only import process_lock
                windows_lock = process_lock(self._scope_lock_path(lock_key), timeout=600 if blocking else 0)
                try:
                    windows_lock.__enter__()
                except TimeoutError as exc:
                    windows_lock = None
                    raise BlockingIOError("scope mutation lane is active") from exc
            if os.name == "posix":
                import fcntl

                lock_file = self._scope_lock_path(lock_key).open("a+b")
                flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                try:
                    fcntl.flock(lock_file, flags)
                except BlockingIOError:
                    lock_file.close()
                    lock_file = None
                    raise
            yield
        finally:
            if windows_lock is not None:
                windows_lock.__exit__(None, None, None)
            if lock_file is not None:
                import fcntl

                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()
            local_lock.release()

    def _scheduler_interval(self) -> float:
        value = getattr(self.settings, "scheduler_interval_seconds", 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("scheduler_interval_seconds must be positive")
        return float(value)

    def _quarantine_recovery_interval(self) -> float:
        value = getattr(self.settings, "quarantine_recovery_interval_seconds", 15.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("quarantine_recovery_interval_seconds must be positive")
        return float(value)

    def _quarantine_recovery_delay(self, attempt: int) -> float:
        base = float(
            getattr(self.settings, "quarantine_recovery_backoff_seconds", 30.0)
        )
        return min(1800.0, base * (2 ** max(0, min(int(attempt) - 1, 6))))

    def _quarantine_recovery_concurrency(self) -> int:
        value = int(getattr(self.settings, "quarantine_recovery_concurrency", 4))
        if value <= 0:
            raise ValueError("quarantine_recovery_concurrency must be positive")
        # Source and graph persistence are a single-writer state machine per
        # scope. Worker concurrency remains available across different scopes.
        return 1

    @staticmethod
    def _quarantine_recovery_report(
        audit: Mapping[str, Any], *, recovery_job_count: int = 0
    ) -> dict[str, Any]:
        failed = int(audit.get("failed_source_count", 0) or 0)
        pending = int(audit.get("pending_source_count", 0) or 0)
        unaccounted = int(audit.get("unaccounted_source_count", 0) or 0)
        return {
            "phase": "repairing" if failed or pending or unaccounted else "verifying",
            "integrity_ok": bool(audit.get("integrity_ok")),
            "ready_to_release": bool(audit.get("ready_to_release")),
            "error_code": str(audit.get("error_code") or ""),
            "source_count": int(audit.get("source_count", 0) or 0),
            "record_source_count": int(audit.get("record_source_count", 0) or 0),
            "enriched_source_count": int(
                audit.get("enriched_source_count", 0) or 0
            ),
            "failed_source_count": failed,
            "pending_source_count": pending,
            "prepared_message_commit_count": int(
                audit.get("prepared_message_commit_count", 0) or 0
            ),
            "control_source_event_seq": int(
                audit.get("control_source_event_seq", 0) or 0
            ),
            "unaccounted_source_count": unaccounted,
            "unaccounted_operation_count": len(
                audit.get("unaccounted_operation_ids", []) or []
            ),
            "registered_message_count": int(
                audit.get("registered_message_count", 0) or 0
            ),
            "recovery_job_count": int(recovery_job_count),
        }

    @staticmethod
    def _job_error_code(job: Job) -> str:
        raw = str(job.error or "").strip()
        return raw.split(":", 1)[0][:120] if raw else "job_failed"

    @staticmethod
    def _is_pre_writer_quarantine_gate_failure(job: Job) -> bool:
        """Prove that the failed attempt was rejected before Writer execution."""

        if (
            job.state != FAILED
            or str((job.payload or {}).get("job_type") or "") != "ingest"
        ):
            return False
        raw = str(job.error or "").strip()
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(decoded, Mapping):
            return False
        traceback_text = str(decoded.get("traceback") or "")
        return bool(
            str(decoded.get("type") or "") == "CommercialContractError"
            and str(decoded.get("message") or "") == "scope is quarantined"
            and " in _execute" in traceback_text
            and "require_scope_active" in traceback_text
        )

    def _is_pre_stage_evolution_claim_failure(self, job: Job) -> bool:
        """Prove that a failed consolidation never entered its Slow stage."""

        if (
            job.state != FAILED
            or str((job.payload or {}).get("job_type") or "") != "consolidate"
        ):
            return False
        evidence_reader = getattr(self.jobs, "job_execution_evidence", None)
        if evidence_reader is None:
            return False
        try:
            evidence = dict(evidence_reader(job.job_id))
        except Exception:
            return False
        if int(evidence.get("stage_count", 0) or 0) != 0 or int(
            evidence.get("provider_call_count", 0) or 0
        ) != 0:
            return False

        raw = str(job.error or "").strip()
        error_type = ""
        message = raw
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, Mapping):
            error_type = str(decoded.get("type") or "")
            message = str(decoded.get("message") or "")
        return (
            error_type in {"", "RuntimeErrorBase"}
            and message == "evolution job does not own this scope"
        )

    def _slow_graph_recovery_plan(
        self, *, tenant_id: str, scope_name: str
    ) -> dict[str, Any]:
        planner = getattr(self.storage, "slow_graph_recovery_plan", None)
        if not callable(planner):
            return {"resumable": False, "reason": "slow_recovery_unsupported"}
        try:
            value = planner(tenant_id=tenant_id, scope_name=scope_name)
        except Exception as exc:
            return {
                "resumable": False,
                "reason": "slow_recovery_plan_failed",
                "error_type": type(exc).__name__,
            }
        if not isinstance(value, Mapping):
            return {"resumable": False, "reason": "slow_recovery_plan_invalid"}
        return dict(value)

    def _slow_graph_recovery_progress(
        self, *, tenant_id: str, scope_name: str
    ) -> dict[str, Any]:
        reader = getattr(self.storage, "slow_graph_recovery_status", None)
        if not callable(reader):
            return {}
        try:
            value = reader(tenant_id=tenant_id, scope_name=scope_name)
        except Exception:
            return {}
        if not isinstance(value, Mapping):
            return {}
        progress = dict(value)
        return {
            "slow_child_completed_job_count": int(
                progress.get("completed_job_count", 0) or 0
            ),
            "slow_child_pending_job_count": int(
                progress.get("pending_job_count", 0) or 0
            ),
            "slow_child_failed_job_count": int(
                progress.get("failed_job_count", 0) or 0
            ),
            "slow_child_retryable_job_count": int(
                progress.get("retryable_job_count", 0) or 0
            ),
            "slow_child_active_job_count": int(
                progress.get("active_job_count", 0) or 0
            ),
            "slow_child_total_job_count": int(
                progress.get("total_job_count", 0) or 0
            ),
            "slow_child_progress_percent": float(
                progress.get("progress_percent", 0.0) or 0.0
            ),
        }

    def _resume_quarantined_ingest(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        job: Job,
        owner: str,
        now: float,
        report: Mapping[str, Any],
        recovery_plan: Mapping[str, Any],
        audit: Mapping[str, Any] | None = None,
    ) -> int:
        if self.commercial is None:
            return False
        gate_compensation = self._is_pre_writer_quarantine_gate_failure(job)
        max_attempts = self._provider_recovery_attempt_limit(
            recovery_plan,
            job=job,
        )
        deterministic_local_repair = bool(
            recovery_plan.get("deterministic_local_repair")
        )
        attempt = self.commercial.authorize_quarantine_recovery_job(
            tenant_id,
            scope_name,
            job.job_id,
            owner,
            max_attempts=max_attempts,
            attempt_kind="local" if deterministic_local_repair else "provider",
            local_repair_fingerprint=(
                str(recovery_plan.get("recovery_fingerprint") or "")
                if deterministic_local_repair
                else None
            ),
            max_local_repairs=int(
                getattr(self.settings, "quarantine_recovery_max_local_repairs", 8)
            ),
        )
        try:
            if job.state == FAILED:
                self._resume_failed_authorized(
                    job,
                    code="automatic_quarantine_recovery",
                    authorization={
                        "source": "quarantine_recovery_audit",
                        "mode": str(
                            recovery_plan.get("mode") or "audited_writer_state"
                        ),
                        "attempt": attempt,
                        "fingerprint": str(
                            recovery_plan.get("recovery_fingerprint") or ""
                        ),
                        "pre_writer_quarantine_gate_compensation": (
                            gate_compensation
                        ),
                    },
                    evidence={
                        "tenant_id": tenant_id,
                        "scope_name": scope_name,
                        "job_id": job.job_id,
                        "audit": dict(audit or report),
                        "recovery_plan": dict(recovery_plan),
                        "runtime_authorization": {
                            "source": "quarantine_recovery_audit",
                            "attempt": attempt,
                            "pre_writer_quarantine_gate_compensation": (
                                gate_compensation
                            ),
                        },
                    },
                )
            elif job.state != PENDING:
                raise JobStateError(
                    f"cannot adopt quarantine recovery job in state {job.state!r}"
                )
            self.commercial.mark_quarantine_recovery_job(
                tenant_id, scope_name, job.job_id, state="pending"
            )
        except Exception as exc:
            self.commercial.mark_quarantine_recovery_job(
                tenant_id,
                scope_name,
                job.job_id,
                state="failed",
                error_code=type(exc).__name__,
            )
            raise
        return attempt

    def _provider_recovery_attempt_limit(
        self,
        recovery_plan: Mapping[str, Any],
        *,
        job: Job | None = None,
    ) -> int:
        max_attempts = int(
            getattr(self.settings, "quarantine_recovery_max_job_attempts", 3)
        )
        recovery_mode = str(recovery_plan.get("mode") or "")
        if recovery_mode == "schema_constrained_invalid_response":
            max_attempts += 1
        elif recovery_mode == "schema_constrained_invalid_response_prepared":
            max_attempts += 2
        if job is not None and self._is_pre_writer_quarantine_gate_failure(job):
            # A prior controller race rejected this attempt before Writer or
            # provider execution. Grant exactly one replacement authorization;
            # the same evidence cannot grow the bound beyond this single slot.
            max_attempts += 1
        return max_attempts

    def _ingest_recovery_plan(
        self, *, tenant_id: str, scope_name: str, job_id: str
    ) -> dict[str, Any]:
        planner = getattr(self.storage, "ingest_recovery_plan", None)
        if planner is None:
            resumable = bool(
                self.storage.can_resume_ingest(
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    job_id=job_id,
                )
            )
            return {
                "resumable": resumable,
                "mode": "audited_writer_state" if resumable else "manual_review",
                "parallel_safe": False,
                "external_api_calls_expected": None,
                "deterministic_local_repair": False,
            }
        value = planner(
            tenant_id=tenant_id,
            scope_name=scope_name,
            job_id=job_id,
        )
        if not isinstance(value, Mapping):
            return {
                "resumable": False,
                "mode": "manual_review",
                "parallel_safe": False,
                "external_api_calls_expected": None,
                "deterministic_local_repair": False,
                "reason": "recovery_plan_invalid",
            }
        return dict(value)

    def _recovery_plan_has_authorizable_attempt(
        self,
        *,
        job: Job | None,
        plan: Mapping[str, Any],
        recovery_job: Mapping[str, Any] | None,
        tenant_id: str | None = None,
        scope_name: str | None = None,
        audit_fingerprint: str | None = None,
        audit: Mapping[str, Any] | None = None,
    ) -> bool:
        if job is None or job.state not in {FAILED, PENDING}:
            return False
        formal_pending = bool(
            job.state == PENDING
            and self._formal_pending_retry_is_current(
                job=job,
                plan=plan,
                audit=audit,
            )
        )
        if recovery_job is None:
            # A pending attempt is only executable when the control plane has
            # published an audited mapping for the current quarantine.
            if job.state == PENDING and not formal_pending:
                return False
        elif not self._recovery_mapping_is_current(
            tenant_id or job.tenant_id,
            scope_name or job.scope_name,
            recovery_job,
        ) and not formal_pending:
            return False
        if formal_pending:
            return True
        if job.state == PENDING and recovery_job is not None and audit_fingerprint:
            mapped_fingerprint = self._mapped_audit_fingerprint(
                tenant_id or job.tenant_id,
                scope_name or job.scope_name,
                job.job_id,
            )
            if mapped_fingerprint != str(audit_fingerprint).strip():
                return False
        if job.state == PENDING:
            mapping_state = str(recovery_job.get("state") or "")
            if mapping_state not in {"authorized", "pending", "running"}:
                return False
            current_fingerprint = str(audit_fingerprint or "").strip()
            mapped_fingerprint = self._mapped_audit_fingerprint(
                tenant_id or job.tenant_id,
                scope_name or job.scope_name,
                job.job_id,
            )
            # A published attempt must be bound to the same audited durable
            # plan. Provider retries without such a binding stay quarantined.
            return bool(current_fingerprint and current_fingerprint == mapped_fingerprint)
        if recovery_job is None:
            # A resumable failed operation without a mapping has not consumed
            # a recovery authorization yet.
            return True
        if str(recovery_job.get("state") or "") not in {
            "authorized",
            "pending",
            "running",
            "failed",
        }:
            return False
        if bool(plan.get("deterministic_local_repair")):
            fingerprint = str(plan.get("recovery_fingerprint") or "").strip()
            prior_fingerprint = str(
                recovery_job.get("last_local_repair_fingerprint") or ""
            ).strip()
            if not fingerprint or fingerprint == prior_fingerprint:
                return False
            contract = fingerprint.partition(":")[0]
            prior_contract = prior_fingerprint.partition(":")[0]
            contract_upgrade = bool(
                contract and prior_contract and contract != prior_contract
            )
            max_local_repairs = int(
                getattr(self.settings, "quarantine_recovery_max_local_repairs", 8)
            )
            return bool(
                int(recovery_job.get("local_repair_attempt_count", 0) or 0)
                < max_local_repairs
                or contract_upgrade
            )
        return int(recovery_job.get("provider_attempt_count", 0) or 0) < (
            self._provider_recovery_attempt_limit(plan, job=job)
        )

    def _formal_pending_retry_is_current(
        self,
        *,
        job: Job,
        plan: Mapping[str, Any],
        audit: Mapping[str, Any] | None,
    ) -> bool:
        """Validate a formal ``/retry`` authorization before adopting it.

        The authorization is version-bound and must still agree with the
        current read-only Source audit and recovery plan. This bridges the job
        ledger into the quarantine ledger without creating another retry.
        """

        if job.state != PENDING or not isinstance(audit, Mapping):
            return False
        if audit.get("integrity_ok") is not True or not bool(plan.get("resumable")):
            return False
        failed_operation_ids = {
            str(value)
            for value in audit.get("failed_operation_ids", ())
            if str(value)
        }
        if job.job_id not in failed_operation_ids:
            return False
        try:
            lifecycle = self.database.list_job_lifecycle_audits(job.job_id)
        except (AttributeError, sqlite3.Error, TypeError, ValueError):
            return False
        latest_transition: Mapping[str, Any] | None = None
        for row in reversed(lifecycle):
            if row.get("stage_id") is None and str(row.get("to_state") or "") in {
                PENDING,
                RUNNING,
                SUCCEEDED,
                FAILED,
                CANCELLED,
            }:
                latest_transition = row
                break
        if (
            latest_transition is None
            or str(latest_transition.get("event_type") or "") != "job_recovered"
            or str(latest_transition.get("to_state") or "") != PENDING
        ):
            return False
        reason = latest_transition.get("reason")
        if not isinstance(reason, Mapping):
            return False
        try:
            previous_version = int(reason.get("previous_job_version"))
        except (TypeError, ValueError):
            return False
        if previous_version + 1 != int(job.version):
            return False
        evidence = reason.get("evidence")
        if not isinstance(evidence, Mapping):
            return False
        if (
            str(evidence.get("job_id") or "") != job.job_id
            or str(evidence.get("tenant_id") or "") != job.tenant_id
            or str(evidence.get("scope_name") or "") != job.scope_name
        ):
            return False
        authorized_audit = evidence.get("audit")
        authorized_plan = evidence.get("recovery_plan")
        if not isinstance(authorized_audit, Mapping) or not isinstance(
            authorized_plan, Mapping
        ):
            return False
        if authorized_audit.get("integrity_ok") is not True or job.job_id not in {
            str(value)
            for value in authorized_audit.get("failed_operation_ids", ())
            if str(value)
        }:
            return False
        encoded = json.dumps(
            dict(evidence),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        if str(reason.get("audit_fingerprint") or "") != hashlib.sha256(
            encoded.encode("utf-8")
        ).hexdigest():
            return False
        authorized_mode = str(reason.get("resume_mode") or "")
        if authorized_mode != str(authorized_plan.get("mode") or ""):
            return False
        exact_plan_match = all(
            authorized_plan.get(key) == plan.get(key)
            for key in (
                "mode",
                "parallel_safe",
                "external_api_calls_expected",
                "deterministic_local_repair",
            )
        )
        safe_local_downgrade = bool(
            authorized_plan.get("resumable") is True
            and authorized_plan.get("external_api_calls_expected") is True
            and authorized_plan.get("deterministic_local_repair") is False
            and plan.get("resumable") is True
            and plan.get("parallel_safe") is True
            and plan.get("external_api_calls_expected") is False
            and plan.get("deterministic_local_repair") is True
        )
        if not (exact_plan_match or safe_local_downgrade):
            return False
        try:
            with self.database.transaction(immediate=False) as connection:
                unresolved = connection.execute(
                    "SELECT COUNT(*) FROM provider_calls AS calls "
                    "LEFT JOIN provider_call_reconciliations AS reconciliation "
                    "ON reconciliation.call_id=calls.call_id "
                    "WHERE calls.job_id=? "
                    "AND calls.status IN ('started','unknown') "
                    "AND reconciliation.call_id IS NULL",
                    (job.job_id,),
                ).fetchone()[0]
        except (AttributeError, sqlite3.Error, TypeError):
            return False
        return int(unresolved or 0) == 0

    def _recovery_mapping_is_current(
        self,
        tenant_id: str,
        scope_name: str,
        recovery_job: Mapping[str, Any],
    ) -> bool:
        """Reject mappings left behind by an older quarantine generation."""

        try:
            with self.database.transaction(immediate=False) as connection:
                row = connection.execute(
                    "SELECT quarantine.quarantined_at,recovery.quarantine_started_at "
                    "FROM scope_quarantines AS quarantine "
                    "JOIN scope_quarantine_recoveries AS recovery "
                    "ON recovery.tenant_id=quarantine.tenant_id "
                    "AND recovery.scope_name=quarantine.scope_name "
                    "WHERE quarantine.tenant_id=? AND quarantine.scope_name=?",
                    (tenant_id, scope_name),
                ).fetchone()
        except (AttributeError, sqlite3.Error):
            return False
        if row is None:
            return False
        return float(row["quarantined_at"]) == float(row["quarantine_started_at"])

    def _mapped_audit_fingerprint(
        self, tenant_id: str, scope_name: str, job_id: str
    ) -> str:
        """Read the audit binding persisted for the current recovery cycle."""

        try:
            with self.database.transaction(immediate=False) as connection:
                row = connection.execute(
                    "SELECT report_json FROM scope_quarantine_recoveries "
                    "WHERE tenant_id=? AND scope_name=?",
                    (tenant_id, scope_name),
                ).fetchone()
        except (AttributeError, sqlite3.Error):
            return ""
        if row is None:
            return ""
        try:
            report = json.loads(str(row["report_json"] or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return ""
        if not isinstance(report, Mapping):
            return ""
        direct = str(report.get("scope_audit_fingerprint") or "").strip()
        if direct:
            return direct
        bindings = report.get("recovery_audit_fingerprints")
        if not isinstance(bindings, Mapping):
            bindings = {}
        bound = str(bindings.get(job_id) or "").strip()
        if bound:
            return bound
        audit_keys = {
            "source_count",
            "record_source_count",
            "enriched_source_count",
            "failed_source_count",
            "pending_source_count",
            "unaccounted_source_count",
            "prepared_message_commit_count",
            "control_source_event_seq",
            "failed_operation_ids",
            "unaccounted_operation_ids",
            "error_code",
        }
        if audit_keys.intersection(report):
            return self._scope_audit_fingerprint(report)
        return ""

    @staticmethod
    def _scope_audit_fingerprint(audit: Mapping[str, Any]) -> str:
        """Bind retries to the complete read-only audit, not a partial count."""

        normalized = {
            key: audit.get(key)
            for key in (
                "source_count",
                "record_source_count",
                "enriched_source_count",
                "failed_source_count",
                "pending_source_count",
                "unaccounted_source_count",
                "prepared_message_commit_count",
                "control_source_event_seq",
                "failed_operation_ids",
                "unaccounted_operation_ids",
                "error_code",
            )
        }
        return "tmcra.scope-audit:" + hashlib.sha256(
            json.dumps(
                normalized,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()

    def _persist_recovery_audit_report(
        self,
        tenant_id: str,
        scope_name: str,
        report: Mapping[str, Any],
    ) -> None:
        """Bind the current audit to the leased recovery cycle before retry selection."""

        encoded = json.dumps(
            dict(report), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        with self.database.transaction() as connection:
            updated = connection.execute(
                "UPDATE scope_quarantine_recoveries SET report_json=?,updated_at=? "
                "WHERE tenant_id=? AND scope_name=? AND lease_owner=? "
                "AND state IN ('auditing','repairing','verifying')",
                (encoded, time.time(), tenant_id, scope_name, self.worker_id),
            ).rowcount
        if updated != 1:
            raise CommercialContractError(
                "quarantine_recovery_lease_lost",
                "recovery audit could not be bound to the active lease",
            )

    @staticmethod
    def _recovery_plan_fingerprint(plan: Mapping[str, Any]) -> str:
        """Return a stable binding for the audited recovery plan."""

        explicit = str(
            plan.get("audit_fingerprint")
            or plan.get("recovery_fingerprint")
            or ""
        ).strip()
        if explicit:
            return explicit
        durable = {
            key: plan.get(key)
            for key in (
                "mode",
                "parallel_safe",
                "external_api_calls_expected",
                "deterministic_local_repair",
                "reason",
            )
            if key in plan
        }
        if not durable:
            return ""
        encoded = json.dumps(durable, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return "tmcra.audit-plan:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _resume_failed_authorized(
        self,
        job: Job,
        *,
        code: str,
        authorization: Mapping[str, Any],
        evidence: Mapping[str, Any] | None = None,
    ) -> Job:
        """Resume only with a durable, explicit recovery authorization."""

        auth = {
            "contract": "tmcra.runtime.safe-resume.v1",
            "job_id": job.job_id,
            "job_version": int(job.version),
            **dict(authorization),
        }
        if not auth.get("source"):
            raise JobStateError("safe resume authorization is incomplete")
        mode = str(auth.get("mode") or "").strip()
        job_type = str((job.payload or {}).get("job_type") or "")
        if job_type == "ingest":
            if not mode or not isinstance(evidence, Mapping):
                raise JobStateError("ingest resume authorization is incomplete")
            evidence_value = dict(evidence)
            audited_plan = evidence_value.get("recovery_plan")
            # The runtime planner historically called the local-only lane
            # ``validation``.  The strict JobStore contract names that same
            # proof ``deterministic_local_repair``.  Normalize only when the
            # evidence proves the operation is local-only; provider retries
            # remain on audited_writer_state and cannot take this branch.
            if (
                isinstance(audited_plan, Mapping)
                and bool(audited_plan.get("deterministic_local_repair"))
                and bool(audited_plan.get("parallel_safe"))
                and audited_plan.get("external_api_calls_expected") is False
                and mode != "deterministic_local_repair"
            ):
                normalized_plan = dict(audited_plan)
                normalized_plan["mode"] = "deterministic_local_repair"
                evidence_value["recovery_plan"] = normalized_plan
                mode = "deterministic_local_repair"
            authorization_object: Any = ResumeAuthorization.from_evidence(
                reason_code=code,
                resume_mode=mode,
                evidence=evidence_value,
            )
        else:
            authorization_object = ResumeAuthorization(reason_code=code)
        reason = {
            "code": code,
            "authorization": auth,
            "resume_mode": mode or None,
        }
        resume = self.jobs.resume_failed
        try:
            parameters = inspect.signature(resume).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "authorization" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return resume(job.job_id, authorization=authorization_object)
        return resume(job.job_id, reason=reason)

    def _recovery_plans_have_authorizable_attempt(
        self,
        *,
        operation_ids: list[str],
        plans: list[dict[str, Any]],
        recovery_jobs: list[Mapping[str, Any]],
        tenant_id: str | None = None,
        scope_name: str | None = None,
        audit_fingerprint: str | None = None,
        audit: Mapping[str, Any] | None = None,
    ) -> bool:
        """Require a genuinely new bounded action before reopening recovery."""

        if not operation_ids or len(operation_ids) != len(plans):
            return False
        rows = {str(row.get("job_id") or ""): row for row in recovery_jobs}
        candidates: list[tuple[Job, Mapping[str, Any]]] = []
        first_scope_seq_by_session: dict[str, int] = {}
        for job_id, plan in zip(operation_ids, plans):
            job = self.jobs.get(job_id)
            if job is None or job.state not in {FAILED, PENDING}:
                continue
            session_id = str((job.payload or {}).get("session_id") or "")
            if not session_id:
                continue
            candidates.append((job, plan))
            first_scope_seq_by_session[session_id] = min(
                job.scope_seq,
                first_scope_seq_by_session.get(session_id, job.scope_seq),
            )
        for job, plan in candidates:
            session_id = str((job.payload or {}).get("session_id") or "")
            if job.scope_seq != first_scope_seq_by_session[session_id]:
                # Recovery preserves Source order within a session. A later
                # operation cannot make progress while its failed frontier is
                # exhausted, so it must not wake a new audit cycle.
                continue
            if self._recovery_plan_has_authorizable_attempt(
                job=job,
                plan=plan,
                recovery_job=rows.get(job.job_id),
                tenant_id=tenant_id,
                scope_name=scope_name,
                audit_fingerprint=audit_fingerprint,
                audit=audit,
            ):
                return True
        return False

    def _reaudit_manual_quarantine_recovery(self, *, now: float) -> bool:
        """Reopen only interruption-related manual review after a fresh proof."""

        if self.commercial is None or not hasattr(
            self.commercial, "manual_quarantine_recovery_candidates"
        ):
            return False
        allowed_errors = frozenset(
            {
                "source_journal_nonterminal",
                "source_journal_not_release_ready",
                "source_operation_binding_set_mismatch",
                "recovery_job_not_safely_resumable",
                "quarantine_recovery_budget_exhausted",
                "quarantine_local_repair_budget_exhausted",
                "quarantine_recovery_frontier_blocked",
                "quarantine_reason_requires_manual_review",
                "slow_graph_retry_requires_audit",
            }
        )
        for candidate in self.commercial.manual_quarantine_recovery_candidates():
            error_code = str(candidate.get("last_error_code") or "")
            reason = str(candidate.get("reason") or "")
            if (
                error_code not in allowed_errors
                or not self.commercial.quarantine_reason_supports_auto_recovery(
                    reason
                )
            ):
                continue
            tenant_id = str(candidate["tenant_id"])
            scope_name = str(candidate["scope_name"])
            try:
                prior_report = json.loads(str(candidate.get("report_json") or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                prior_report = {}
            report: dict[str, Any] = (
                dict(prior_report) if isinstance(prior_report, Mapping) else {}
            )
            try:
                source_accounting = self._recover_scope_source_accounting(
                    tenant_id, scope_name
                )
                if source_accounting["source_count"]:
                    report.update(
                        {
                            "source_accounting_recovery_operation_count": (
                                source_accounting["operation_count"]
                            ),
                            "source_accounting_recovery_source_count": (
                                source_accounting["source_count"]
                            ),
                            "source_accounting_recovery_external_api_calls": 0,
                        }
                    )
                audit = dict(
                    self.storage.audit_scope_recovery(
                        tenant_id=tenant_id,
                        scope_name=scope_name,
                    )
                )
            except Exception:
                continue
            if not bool(audit.get("integrity_ok")):
                continue
            operation_ids = [
                str(value)
                for value in audit.get("failed_operation_ids", [])
                if str(value)
            ]
            if error_code == "slow_graph_retry_requires_audit":
                if operation_ids or not bool(audit.get("ready_to_release")):
                    continue
                state = self._state(tenant_id, scope_name) or {}
                source_event_seq = int(state.get("source_event_seq", 0) or 0)
                conflict_generation = int(
                    state.get("conflict_generation", 0) or 0
                )
                current_failures: list[Job] = []
                for row in self.commercial.quarantine_recovery_jobs(
                    tenant_id, scope_name
                ):
                    job = self.jobs.get(str(row["job_id"]))
                    payload = dict((job.payload or {}) if job is not None else {})
                    if (
                        job is not None
                        and job.state == FAILED
                        and str(payload.get("job_type") or "") == "consolidate"
                        and payload.get("target_source_event_seq") is not None
                        and int(payload["target_source_event_seq"])
                        == source_event_seq
                        and payload.get("target_conflict_generation") is not None
                        and int(payload["target_conflict_generation"])
                        == conflict_generation
                    ):
                        current_failures.append(job)
                if len(current_failures) != 1:
                    continue
                recovery_mode = "pre_stage_scope_claim_repair"
                report_updates: dict[str, Any] = {
                    "audited_pre_stage_consolidation_count": 1,
                }
                if not self._is_pre_stage_evolution_claim_failure(
                    current_failures[0]
                ):
                    plan = self._slow_graph_recovery_plan(
                        tenant_id=tenant_id,
                        scope_name=scope_name,
                    )
                    if not bool(plan.get("resumable")):
                        continue
                    preparer = getattr(
                        self.storage, "prepare_slow_graph_recovery", None
                    )
                    if not callable(preparer):
                        continue
                    prepared = preparer(
                        tenant_id=tenant_id,
                        scope_name=scope_name,
                        expected_evidence=dict(plan.get("evidence") or {}),
                    )
                    prepared_mode = (
                        str(prepared.get("mode") or "")
                        if isinstance(prepared, Mapping)
                        else ""
                    )
                    if (
                        prepared_mode
                        not in {
                            "audited_local_saved_response_revalidation_completed",
                            "audited_model_validation_retry_prepared",
                            "audited_unattempted_queue_continuation_verified",
                        }
                        or int(prepared.get("external_api_calls_performed", -1))
                        != 0
                    ):
                        continue
                    recovery_mode = (
                        "audited_slow_child_zero_call_revalidation"
                        if prepared_mode
                        == "audited_local_saved_response_revalidation_completed"
                        else (
                            "audited_slow_child_model_validation_retry"
                            if prepared_mode
                            == "audited_model_validation_retry_prepared"
                            else "audited_slow_unattempted_queue_continuation"
                        )
                    )
                    report_updates = {
                        "audited_pre_stage_consolidation_count": 0,
                        "audited_slow_child_recovery_count": 1,
                        "slow_child_completed_job_count": int(
                            prepared.get("completed_job_count", 0) or 0
                        ),
                        "slow_child_pending_job_count": int(
                            prepared.get("pending_job_count", 0) or 0
                        ),
                        "slow_child_failed_job_count": int(
                            prepared.get("failed_job_count", 0) or 0
                        ),
                        "slow_child_total_job_count": int(
                            prepared.get("total_job_count", 0) or 0
                        ),
                        "slow_child_progress_percent": float(
                            prepared.get("progress_percent", 0.0) or 0.0
                        ),
                        "slow_child_recovery_external_api_calls": 0,
                        "slow_child_prior_physical_api_calls": int(
                            dict(prepared.get("evidence") or {}).get(
                                "prior_physical_api_calls", 0
                            )
                            or 0
                        ),
                    }
                report = {
                    **report,
                    **self._quarantine_recovery_report(audit),
                }
                report.update(
                    {
                        "phase": "waiting",
                        "reaudited": True,
                        "recovery_mode": recovery_mode,
                        **report_updates,
                    }
                )
                if self.commercial.reopen_quarantine_recovery_after_audit(
                    tenant_id,
                    scope_name,
                    expected_error_code=error_code,
                    audit_report=report,
                    now=now,
                ):
                    return True
                continue
            if not operation_ids:
                continue
            plans = [
                self._ingest_recovery_plan(
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    job_id=job_id,
                )
                for job_id in operation_ids
            ]
            if not all(bool(plan.get("resumable")) for plan in plans):
                continue
            if error_code in {
                "quarantine_recovery_budget_exhausted",
                "quarantine_local_repair_budget_exhausted",
                "quarantine_recovery_frontier_blocked",
            } and not self._recovery_plans_have_authorizable_attempt(
                operation_ids=operation_ids,
                plans=plans,
                recovery_jobs=self.commercial.quarantine_recovery_jobs(
                    tenant_id, scope_name
                ),
                tenant_id=tenant_id,
                scope_name=scope_name,
                audit_fingerprint=self._scope_audit_fingerprint(audit),
                audit=audit,
            ):
                continue
            report = {**report, **self._quarantine_recovery_report(audit)}
            report.update(
                {
                    "phase": "waiting",
                    "reaudited": True,
                    "parallel_safe_operation_count": sum(
                        1 for plan in plans if bool(plan.get("parallel_safe"))
                    ),
                }
            )
            if self.commercial.reopen_quarantine_recovery_after_audit(
                tenant_id,
                scope_name,
                expected_error_code=error_code,
                audit_report=report,
                now=now,
            ):
                return True
        return False

    def _schedule_quarantine_reindex(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        source_event_seq: int,
        owner: str,
        now: float,
        report: Mapping[str, Any],
    ) -> bool:
        if self.commercial is None:
            return False
        job = self.jobs.submit(
            tenant_id,
            f"auto:quarantine-reindex:{scope_name}:{source_event_seq}",
            {
                "job_type": "reindex",
                "scope_name": scope_name,
                "auto": True,
                "quarantine_recovery": True,
                "target_source_event_seq": source_event_seq,
            },
            scope_name=scope_name,
            tenant_queue_limit=getattr(self.settings, "tenant_queue_limit", None),
            global_queue_limit=getattr(self.settings, "global_queue_limit", None),
        )
        if job.state == SUCCEEDED:
            return False
        if job.state not in {PENDING, FAILED}:
            raise RuntimeErrorBase("quarantine recovery reindex is not resumable")
        max_attempts = int(
            getattr(self.settings, "quarantine_recovery_max_job_attempts", 3)
        )
        attempt = self.commercial.authorize_quarantine_recovery_job(
            tenant_id,
            scope_name,
            job.job_id,
            owner,
            max_attempts=max_attempts,
        )
        if job.state == FAILED:
            job = self._resume_failed_authorized(
                job,
                code="automatic_quarantine_reindex",
                authorization={
                    "source": "quarantine_recovery_audit",
                    "mode": "audited_index_state",
                    "attempt": attempt,
                },
            )
        # ``authorize_quarantine_recovery_job`` makes the recovery repairable
        # while the mapping stays non-executable. Hold the index lane before
        # publishing this pending job to workers.
        if not self._claim_index(
            tenant_id, scope_name, job.job_id, job_version=job.version
        ):
            raise RuntimeErrorBase(
                "quarantine recovery reindex could not claim its scope"
            )
        try:
            self.commercial.publish_quarantine_recovery_job(
                tenant_id,
                scope_name,
                job.job_id,
                owner,
                next_attempt_at=now + self._quarantine_recovery_delay(attempt),
                report=report,
            )
        except Exception:
            self._release_index(
                tenant_id,
                scope_name,
                job.job_id,
                job_version=job.version,
            )
            raise
        return True

    def _schedule_quarantine_consolidation(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        source_event_seq: int,
        conflict_generation: int,
        raw_token_estimate: int,
        user_turns: int,
        owner: str,
        now: float,
        report: Mapping[str, Any],
    ) -> bool:
        """Coalesce recovered Source backlog into one Slow-plus-index commit."""

        if self.commercial is None:
            return False
        job = self.jobs.submit(
            tenant_id,
            (
                f"auto:quarantine-consolidate:{scope_name}:"
                f"{source_event_seq}:{conflict_generation}"
            ),
            {
                "job_type": "consolidate",
                "scope_name": scope_name,
                "auto": True,
                "quarantine_recovery": True,
                "target_source_event_seq": source_event_seq,
                "target_conflict_generation": conflict_generation,
                "target_raw_token_estimate": raw_token_estimate,
                "target_user_turns": user_turns,
            },
            scope_name=scope_name,
            tenant_queue_limit=getattr(self.settings, "tenant_queue_limit", None),
            global_queue_limit=getattr(self.settings, "global_queue_limit", None),
        )
        if job.state == SUCCEEDED:
            return False
        if job.state not in {PENDING, FAILED}:
            raise RuntimeErrorBase(
                "quarantine recovery consolidation is not resumable"
            )
        failed_job = job.state == FAILED
        max_attempts = int(
            getattr(self.settings, "quarantine_recovery_max_job_attempts", 3)
        )
        slow_recovery = (
            self._slow_graph_recovery_plan(
                tenant_id=tenant_id,
                scope_name=scope_name,
            )
            if failed_job
            and not self._is_pre_stage_evolution_claim_failure(job)
            else {}
        )
        audited_queue_continuation = bool(
            slow_recovery.get("resumable")
            and str(slow_recovery.get("mode") or "")
            == "audited_unattempted_queue_continuation"
            and int(slow_recovery.get("external_api_calls_expected", 0)) > 0
            and str(slow_recovery.get("recovery_fingerprint") or "")
        )
        deterministic_local_repair = bool(
            slow_recovery.get("resumable")
            and slow_recovery.get("deterministic_local_repair")
            and int(slow_recovery.get("external_api_calls_expected", -1)) == 0
        )
        local_authorization = deterministic_local_repair or audited_queue_continuation
        recovery_report = dict(report)
        if audited_queue_continuation:
            recovery_report.update(
                {
                    "recovery_mode": (
                        "audited_slow_unattempted_queue_continuation"
                    ),
                    "audited_slow_child_recovery_count": 1,
                    "slow_child_completed_job_count": int(
                        slow_recovery.get("completed_job_count", 0) or 0
                    ),
                    "slow_child_pending_job_count": int(
                        slow_recovery.get("pending_job_count", 0) or 0
                    ),
                    "slow_child_failed_job_count": int(
                        slow_recovery.get("failed_job_count", 0) or 0
                    ),
                    "slow_child_total_job_count": int(
                        slow_recovery.get("total_job_count", 0) or 0
                    ),
                    "slow_child_progress_percent": float(
                        slow_recovery.get("progress_percent", 0.0) or 0.0
                    ),
                    "slow_child_recovery_external_api_calls": 0,
                    "slow_child_future_model_call_budget": int(
                        slow_recovery.get("external_api_calls_expected", 0) or 0
                    ),
                }
            )
        attempt = self.commercial.authorize_quarantine_recovery_job(
            tenant_id,
            scope_name,
            job.job_id,
            owner,
            max_attempts=max_attempts,
            attempt_kind="local" if local_authorization else "provider",
            local_repair_fingerprint=(
                str(slow_recovery.get("recovery_fingerprint") or "")
                if local_authorization
                else None
            ),
            max_local_repairs=int(
                getattr(self.settings, "quarantine_recovery_max_local_repairs", 8)
            ),
        )
        if failed_job:
            job = self._resume_failed_authorized(
                job,
                code="automatic_quarantine_consolidation",
                authorization={
                    "source": "quarantine_recovery_audit",
                    "mode": str(
                        slow_recovery.get("mode") or "audited_writer_state"
                    ),
                    "attempt": attempt,
                    "fingerprint": str(
                        slow_recovery.get("recovery_fingerprint") or ""
                    ),
                },
            )
        # Keep the recovery mapping ``authorized`` until the evolution lane is
        # durable. That makes the pending job invisible to quarantined workers
        # during the control-plane handoff.
        if not self._claim_evolution(
            tenant_id, scope_name, job.job_id, job_version=job.version
        ):
            raise RuntimeErrorBase(
                "quarantine recovery consolidation could not claim its scope"
            )
        try:
            self.commercial.publish_quarantine_recovery_job(
                tenant_id,
                scope_name,
                job.job_id,
                owner,
                next_attempt_at=now + self._quarantine_recovery_delay(attempt),
                report=recovery_report,
            )
        except Exception:
            self._release_evolution(
                tenant_id,
                scope_name,
                job.job_id,
                job_version=job.version,
            )
            raise
        return True

    def recover_quarantined_scopes(self, *, now: float | None = None) -> int:
        """Advance one fail-closed scope through audited automatic recovery."""

        if self.commercial is None or not hasattr(
            self.storage, "audit_scope_recovery"
        ):
            return 0
        moment = time.time() if now is None else float(now)

        def due_after(delay: float) -> float:
            # Production audits can take longer than their polling interval.
            # Anchor the next cycle at completion, while preserving deterministic
            # synthetic clocks used by recovery tests.
            base = moment if now is not None else time.time()
            return base + float(delay)
        lease_seconds = float(
            getattr(self.settings, "quarantine_recovery_lease_seconds", 120.0)
        )
        recovery = self.commercial.claim_due_quarantine_recovery(
            self.worker_id,
            now=moment,
            lease_seconds=lease_seconds,
        )
        if recovery is None and self._reaudit_manual_quarantine_recovery(now=moment):
            recovery = self.commercial.claim_due_quarantine_recovery(
                self.worker_id,
                now=moment,
                lease_seconds=lease_seconds,
            )
        if recovery is None:
            return 0
        tenant_id = str(recovery["tenant_id"])
        scope_name = str(recovery["scope_name"])
        reason = str(recovery.get("reason") or "")
        try:
            persisted_report = json.loads(str(recovery.get("report_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            persisted_report = {}
        report: dict[str, Any] = (
            dict(persisted_report)
            if isinstance(persisted_report, Mapping)
            else {}
        )
        if not self.commercial.quarantine_reason_supports_auto_recovery(reason):
            self.commercial.finish_quarantine_recovery_cycle(
                tenant_id,
                scope_name,
                self.worker_id,
                state="manual_review",
                next_attempt_at=moment,
                error_code="quarantine_reason_requires_manual_review",
                report={"integrity_ok": False, "ready_to_release": False},
            )
            return 1
        try:
            source_accounting = self._recover_scope_source_accounting(
                tenant_id, scope_name
            )
            if source_accounting["source_count"]:
                report["source_accounting_repaired"] = True
                report.update(
                    {
                        "source_accounting_recovery_operation_count": (
                            source_accounting["operation_count"]
                        ),
                        "source_accounting_recovery_source_count": (
                            source_accounting["source_count"]
                        ),
                        "source_accounting_recovery_external_api_calls": 0,
                    }
                )
            audit = dict(
                self.storage.audit_scope_recovery(
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                )
            )
            report = {
                **report,
                **self._quarantine_recovery_report(audit),
            }
            if not bool(audit.get("integrity_ok")):
                self.commercial.finish_quarantine_recovery_cycle(
                    tenant_id,
                    scope_name,
                    self.worker_id,
                    state="manual_review",
                    next_attempt_at=moment,
                    error_code=str(audit.get("error_code") or "integrity_audit_failed"),
                    report=report,
                )
                return 1

            audit_fingerprint = self._scope_audit_fingerprint(audit)
            report["scope_audit_fingerprint"] = audit_fingerprint
            self._persist_recovery_audit_report(
                tenant_id,
                scope_name,
                report,
            )

            current_state = self._state(tenant_id, scope_name) or {}
            current_source_event_seq = int(
                current_state.get("source_event_seq", 0) or 0
            )
            report.update(
                {
                    "source_event_seq": current_source_event_seq,
                    "promoted_event_seq": int(
                        current_state.get("promoted_event_seq", 0) or 0
                    ),
                    "searchable_event_seq": max(
                        int(current_state.get("indexed_event_seq", 0) or 0),
                        int(
                            current_state.get("delta_indexed_event_seq", 0) or 0
                        ),
                    ),
                }
            )

            # A committed ingest can retain an ``unknown`` provider outcome
            # after a crash even though its immutable Source projection and
            # watermark are complete. Derived scope claims intentionally stay
            # closed until that side effect is reconciled. Do the proof-backed
            # reconciliation before adopting or scheduling Slow/index work so
            # recovery cannot deadlock behind its own scheduler gate.
            if bool(audit.get("ready_to_release")) and int(
                audit.get("source_count", 0) or 0
            ) > 0:
                reconciler = getattr(
                    self.jobs,
                    "reconcile_committed_ingest_uncertain_calls",
                    None,
                )
                if callable(reconciler):
                    reconciled_calls = tuple(
                        reconciler(
                            tenant_id,
                            scope_name,
                            audit=audit,
                            reconciled_by=self.worker_id,
                        )
                    )
                    report["provider_side_effect_reconciliation_count"] = len(
                        reconciled_calls
                    )

            failed_operation_ids = [
                str(value)
                for value in audit.get("failed_operation_ids", [])
                if str(value)
            ]
            failed_operation_id_set = set(failed_operation_ids)
            source_accounting_repaired = bool(
                report.get("source_accounting_repaired")
            )
            provider_retry_suppressed = False
            recovery_jobs = self.commercial.quarantine_recovery_jobs(
                tenant_id, scope_name
            )
            report["recovery_job_count"] = len(recovery_jobs)
            mapped_ids = {str(row["job_id"]) for row in recovery_jobs}
            ignored_historical_ingest_count = 0
            ignored_historical_ingest_job_ids: set[str] = set()
            active_job_ids: list[str] = []
            active_job_types: set[str] = set()
            active_session_ids: set[str] = set()
            active_parallel_safe = True
            retry_candidates: list[tuple[Job, dict[str, Any]]] = []
            for row in recovery_jobs:
                mapped_job_id = str(row["job_id"])
                job = self.jobs.get(mapped_job_id)
                if job is None:
                    if mapped_job_id not in failed_operation_id_set:
                        ignored_historical_ingest_count += 1
                        continue
                    self.commercial.finish_quarantine_recovery_cycle(
                        tenant_id,
                        scope_name,
                        self.worker_id,
                        state="manual_review",
                        next_attempt_at=moment,
                        error_code="recovery_job_missing",
                        report=report,
                    )
                    return 1
                if job.state == PENDING and str(
                    (job.payload or {}).get("job_type") or ""
                ) == "ingest":
                    plan = self._ingest_recovery_plan(
                        tenant_id=tenant_id,
                        scope_name=scope_name,
                        job_id=job.job_id,
                    )
                    if not bool(plan.get("resumable")):
                        self.commercial.finish_quarantine_recovery_cycle(
                            tenant_id,
                            scope_name,
                            self.worker_id,
                            state="manual_review",
                            next_attempt_at=moment,
                            error_code="recovery_job_not_safely_resumable",
                            report=report,
                        )
                        return 1
                    mapping_state = str(row.get("state") or "")
                    explicitly_audited = (
                        mapping_state in {"authorized", "pending", "running"}
                        and self._recovery_plan_has_authorizable_attempt(
                            job=job,
                            plan=plan,
                            recovery_job=row,
                            tenant_id=tenant_id,
                            scope_name=scope_name,
                            audit_fingerprint=audit_fingerprint,
                            audit=audit,
                        )
                    )
                    if (
                        source_accounting_repaired
                        and int(plan.get("external_api_calls_expected", 0) or 0) > 0
                        and not explicitly_audited
                    ):
                        provider_retry_suppressed = True
                        continue
                    # A pending job has not started a model call. Keep every
                    # formally retried candidate authorized until scope order and
                    # the recovery concurrency gate select it for claiming.
                    self.commercial.mark_quarantine_recovery_job(
                        tenant_id, scope_name, job.job_id, state="authorized"
                    )
                    retry_candidates.append((job, plan))
                    continue
                if job.state == PENDING and str(
                    (job.payload or {}).get("job_type") or ""
                ) in {"consolidate", "reindex"}:
                    job_type = str((job.payload or {}).get("job_type") or "")
                    mapping_state = str(row.get("state") or "")
                    if mapping_state == "authorized":
                        self.commercial.prepare_quarantine_recovery_job(
                            tenant_id,
                            scope_name,
                            job.job_id,
                            self.worker_id,
                            next_attempt_at=due_after(
                                self._quarantine_recovery_interval()
                            ),
                        )
                        claim = (
                            self._claim_evolution
                            if job_type == "consolidate"
                            else self._claim_index
                        )
                        if not claim(tenant_id, scope_name, job.job_id):
                            self.commercial.finish_quarantine_recovery_cycle(
                                tenant_id,
                                scope_name,
                                self.worker_id,
                                state="waiting",
                                next_attempt_at=due_after(
                                    self._quarantine_recovery_interval()
                                ),
                                error_code="derived_recovery_scope_claim_unavailable",
                                report=report,
                            )
                            return 1
                        try:
                            self.commercial.publish_quarantine_recovery_job(
                                tenant_id,
                                scope_name,
                                job.job_id,
                                self.worker_id,
                                next_attempt_at=due_after(
                                    self._quarantine_recovery_interval()
                                ),
                                report={**report, "phase": (
                                    "consolidating"
                                    if job_type == "consolidate"
                                    else "indexing"
                                )},
                            )
                        except Exception:
                            (
                                self._release_evolution
                                if job_type == "consolidate"
                                else self._release_index
                            )(tenant_id, scope_name, job.job_id)
                            raise
                    else:
                        self.commercial.mark_quarantine_recovery_job(
                            tenant_id,
                            scope_name,
                            job.job_id,
                            state="pending",
                        )
                    active_job_ids.append(job.job_id)
                    active_job_types.add(job_type)
                    active_parallel_safe = False
                    continue
                if job.state == RUNNING or job.state == PENDING:
                    self.commercial.mark_quarantine_recovery_job(
                        tenant_id,
                        scope_name,
                        job.job_id,
                        state="pending" if job.state == PENDING else "running",
                    )
                    active_job_ids.append(job.job_id)
                    active_job_types.add(
                        str((job.payload or {}).get("job_type") or "")
                    )
                    session_id = str((job.payload or {}).get("session_id") or "")
                    if session_id:
                        active_session_ids.add(session_id)
                    if str((job.payload or {}).get("job_type") or "") == "ingest":
                        active_plan = self._ingest_recovery_plan(
                            tenant_id=tenant_id,
                            scope_name=scope_name,
                            job_id=job.job_id,
                        )
                        active_parallel_safe = active_parallel_safe and bool(
                            active_plan.get("parallel_safe")
                        )
                    else:
                        active_parallel_safe = False
                    if str((job.payload or {}).get("job_type") or "") == "consolidate":
                        report.update(
                            self._slow_graph_recovery_progress(
                                tenant_id=tenant_id,
                                scope_name=scope_name,
                            )
                        )
                    continue
                if job.state == SUCCEEDED:
                    self.commercial.mark_quarantine_recovery_job(
                        tenant_id, scope_name, job.job_id, state="succeeded"
                    )
                    continue
                if job.state == FAILED:
                    self.commercial.mark_quarantine_recovery_job(
                        tenant_id,
                        scope_name,
                        job.job_id,
                        state="failed",
                        error_code=self._job_error_code(job),
                    )
                    job_type = str((job.payload or {}).get("job_type") or "")
                    if (
                        job_type == "ingest"
                        and job.job_id not in failed_operation_id_set
                    ):
                        # Recovery mappings are append-only audit history. A
                        # failed ingest from an older cycle must not block the
                        # current, freshly audited Source failure set.
                        ignored_historical_ingest_count += 1
                        ignored_historical_ingest_job_ids.add(job.job_id)
                        continue
                    if job_type == "reindex":
                        # A base-index rebuild is deterministic at the frozen
                        # Source watermark and can resume with the same job id.
                        continue
                    if job_type == "consolidate":
                        if self._is_pre_stage_evolution_claim_failure(job):
                            report["audited_pre_stage_consolidation_count"] = int(
                                report.get(
                                    "audited_pre_stage_consolidation_count", 0
                                )
                                or 0
                            ) + 1
                            continue
                        slow_stage = self.jobs.get_stage(
                            f"{job.job_id}:slow"
                        )
                        if (
                            slow_stage is not None
                            and slow_stage.state == STAGE_SUCCEEDED
                        ):
                            # Slow has a durable result; only deterministic
                            # index/promotion work remains.
                            continue
                        slow_recovery = self._slow_graph_recovery_plan(
                            tenant_id=tenant_id,
                            scope_name=scope_name,
                        )
                        slow_evidence = dict(
                            slow_recovery.get("evidence") or {}
                        )
                        if bool(slow_recovery.get("resumable")) and (
                            bool(slow_evidence.get("already_prepared"))
                            or bool(slow_evidence.get("already_completed"))
                        ):
                            report[
                                "audited_slow_child_recovery_prepared_count"
                            ] = 1
                            report.update(
                                self._slow_graph_recovery_progress(
                                    tenant_id=tenant_id,
                                    scope_name=scope_name,
                                )
                            )
                            continue
                        self.commercial.finish_quarantine_recovery_cycle(
                            tenant_id,
                            scope_name,
                            self.worker_id,
                            state="manual_review",
                            next_attempt_at=moment,
                            error_code="slow_graph_retry_requires_audit",
                            report=report,
                        )
                        return 1
                    plan = self._ingest_recovery_plan(
                        tenant_id=tenant_id,
                        scope_name=scope_name,
                        job_id=job.job_id,
                    )
                    if job_type == "ingest" and bool(plan.get("resumable")):
                        if (
                            source_accounting_repaired
                            and int(plan.get("external_api_calls_expected", 0) or 0) > 0
                            and job.state != PENDING
                        ):
                            provider_retry_suppressed = True
                            continue
                        retry_candidates.append((job, plan))
                        continue
                    self.commercial.finish_quarantine_recovery_cycle(
                        tenant_id,
                        scope_name,
                        self.worker_id,
                        state="manual_review",
                        next_attempt_at=moment,
                        error_code="recovery_job_not_safely_resumable",
                        report=report,
                    )
                    return 1

            report["ignored_historical_ingest_count"] = (
                ignored_historical_ingest_count
            )
            for job_id in failed_operation_ids:
                if job_id in mapped_ids:
                    continue
                job = self.jobs.get(job_id)
                if (
                    job is None
                    or job.tenant_id != tenant_id
                    or str((job.payload or {}).get("scope_name") or "default")
                    != scope_name
                    or str((job.payload or {}).get("job_type") or "") != "ingest"
                    or job.state not in {FAILED, PENDING}
                ):
                    self.commercial.finish_quarantine_recovery_cycle(
                        tenant_id,
                        scope_name,
                        self.worker_id,
                        state="manual_review",
                        next_attempt_at=moment,
                        error_code="failed_source_operation_not_safely_resumable",
                        report=report,
                    )
                    return 1
                plan = self._ingest_recovery_plan(
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    job_id=job_id,
                )
                if not bool(plan.get("resumable")):
                    self.commercial.finish_quarantine_recovery_cycle(
                        tenant_id,
                        scope_name,
                        self.worker_id,
                        state="manual_review",
                        next_attempt_at=moment,
                        error_code="failed_source_operation_not_safely_resumable",
                        report=report,
                    )
                    return 1
                if (
                    source_accounting_repaired
                    and int(plan.get("external_api_calls_expected", 0) or 0) > 0
                ):
                    provider_retry_suppressed = True
                    continue
                retry_candidates.append((job, plan))

            # A formal or already-published pending retry is an existing
            # authorization. Adopt it before spending budget on any failed
            # candidate, while retaining source order within each state.
            recovery_jobs_by_id = {
                str(row.get("job_id") or ""): row for row in recovery_jobs
            }
            retry_candidates.sort(
                key=lambda item: (
                    not self._recovery_plan_has_authorizable_attempt(
                        job=item[0],
                        plan=item[1],
                        recovery_job=recovery_jobs_by_id.get(item[0].job_id),
                        tenant_id=tenant_id,
                        scope_name=scope_name,
                        audit_fingerprint=audit_fingerprint,
                        audit=audit,
                    ),
                    item[0].state != PENDING,
                    item[0].scope_seq,
                )
            )
            concurrency = self._quarantine_recovery_concurrency()
            if active_job_ids and not active_parallel_safe:
                available_slots = 0
            else:
                available_slots = max(0, concurrency - len(active_job_ids))
            resumed_job_ids: list[str] = []
            resumed_attempts: list[int] = []
            adopted_pending_job_count = 0
            queued_candidates: list[tuple[Job, dict[str, Any]]] = []
            scheduled_non_parallel = False
            first_retry_scope_seq_by_session: dict[str, int] = {}
            for candidate_job, _candidate_plan in retry_candidates:
                candidate_session = str(
                    (candidate_job.payload or {}).get("session_id") or ""
                )
                if candidate_session:
                    first_retry_scope_seq_by_session[candidate_session] = min(
                        candidate_job.scope_seq,
                        first_retry_scope_seq_by_session.get(
                            candidate_session, candidate_job.scope_seq
                        ),
                    )
            for job, plan in retry_candidates:
                authorizable = self._recovery_plan_has_authorizable_attempt(
                    job=job,
                    plan=plan,
                    recovery_job=recovery_jobs_by_id.get(job.job_id),
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    audit_fingerprint=audit_fingerprint,
                    audit=audit,
                )
                session_id = str((job.payload or {}).get("session_id") or "")
                has_earlier_same_session_candidate = bool(
                    session_id
                    and job.scope_seq
                    != first_retry_scope_seq_by_session.get(
                        session_id, job.scope_seq
                    )
                )
                if (
                    not authorizable
                    or has_earlier_same_session_candidate
                    or len(resumed_job_ids) >= available_slots
                    or not session_id
                    or session_id in active_session_ids
                    or scheduled_non_parallel
                    or (
                        not bool(plan.get("parallel_safe"))
                        and (active_job_ids or resumed_job_ids)
                    )
                ):
                    queued_candidates.append((job, plan))
                    continue
                resumed_attempts.append(self._resume_quarantined_ingest(
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    job=job,
                    owner=self.worker_id,
                    now=moment,
                    report=report,
                    recovery_plan=plan,
                    audit=audit,
                ))
                resumed_job_ids.append(job.job_id)
                active_job_ids.append(job.job_id)
                active_session_ids.add(session_id)
                if job.state == PENDING:
                    adopted_pending_job_count += 1
                if not bool(plan.get("parallel_safe")):
                    scheduled_non_parallel = True
            if retry_candidates and not active_job_ids and not resumed_job_ids:
                report.update(
                    {
                        "phase": "manual_review",
                        "queued_recovery_job_count": len(queued_candidates),
                        "blocked_recovery_frontier": True,
                    }
                )
                self.commercial.finish_quarantine_recovery_cycle(
                    tenant_id,
                    scope_name,
                    self.worker_id,
                    state="manual_review",
                    next_attempt_at=moment,
                    error_code="quarantine_recovery_frontier_blocked",
                    report=report,
                )
                return 1
            if provider_retry_suppressed:
                report.update(
                    {
                        "provider_retry_suppressed": True,
                        "source_accounting_repaired": True,
                    }
                )
                self.commercial.finish_quarantine_recovery_cycle(
                    tenant_id,
                    scope_name,
                    self.worker_id,
                    state="manual_review",
                    next_attempt_at=moment,
                    error_code="source_accounting_repair_requires_audit",
                    report=report,
                )
                return 1
            if active_job_ids or retry_candidates:
                active_phase = (
                    "consolidating"
                    if "consolidate" in active_job_types
                    else "indexing"
                    if "reindex" in active_job_types
                    else "repairing"
                )
                report.update(
                    {
                        "phase": active_phase,
                        "recovery_concurrency": concurrency,
                        "active_recovery_job_count": len(active_job_ids),
                        "scheduled_recovery_job_count": len(resumed_job_ids),
                        "adopted_pending_recovery_job_count": (
                            adopted_pending_job_count
                        ),
                        "queued_recovery_job_count": max(
                            0, len(queued_candidates)
                        ),
                        "parallel_safe_recovery_job_count": sum(
                            1
                            for _job, plan in retry_candidates
                            if bool(plan.get("parallel_safe"))
                        ),
                    }
                )
                self.commercial.finish_quarantine_recovery_cycle(
                    tenant_id,
                    scope_name,
                    self.worker_id,
                    state="repairing",
                    active_job_id=active_job_ids[0] if active_job_ids else None,
                    next_attempt_at=due_after(min(
                        self._quarantine_recovery_interval(),
                        self._quarantine_recovery_delay(min(resumed_attempts))
                        if resumed_attempts
                        else self._quarantine_recovery_interval(),
                    )),
                    report=report,
                )
                return 1

            if not bool(audit.get("ready_to_release")):
                self.commercial.finish_quarantine_recovery_cycle(
                    tenant_id,
                    scope_name,
                    self.worker_id,
                    state="manual_review",
                    next_attempt_at=moment,
                    error_code="source_journal_not_release_ready",
                    report=report,
                )
                return 1
            state = self._state(tenant_id, scope_name) or {}
            source_event_seq = int(state.get("source_event_seq", 0) or 0)
            promoted_event_seq = int(state.get("promoted_event_seq", 0) or 0)
            conflict_generation = int(state.get("conflict_generation", 0) or 0)
            promoted_conflict_generation = int(
                state.get("promoted_conflict_generation", 0) or 0
            )
            raw_token_estimate = int(
                state.get("source_raw_token_estimate", 0) or 0
            )
            promoted_raw_token_estimate = int(
                state.get("promoted_raw_token_estimate", 0) or 0
            )
            user_turns = int(state.get("source_user_turns", 0) or 0)
            promoted_user_turns = int(
                state.get("promoted_user_turns", 0) or 0
            )
            searchable_event_seq = max(
                int(state.get("indexed_event_seq", 0) or 0),
                int(state.get("delta_indexed_event_seq", 0) or 0),
            )
            slow_backlog = bool(
                promoted_event_seq < source_event_seq
                or promoted_conflict_generation < conflict_generation
                or promoted_raw_token_estimate < raw_token_estimate
                or promoted_user_turns < user_turns
            )
            if slow_backlog:
                report.update(
                    {
                        "phase": "consolidating",
                        "source_event_seq": source_event_seq,
                        "promoted_event_seq": promoted_event_seq,
                        "searchable_event_seq": searchable_event_seq,
                    }
                )
                if self._schedule_quarantine_consolidation(
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    source_event_seq=source_event_seq,
                    conflict_generation=conflict_generation,
                    raw_token_estimate=raw_token_estimate,
                    user_turns=user_turns,
                    owner=self.worker_id,
                    now=(moment if now is not None else time.time()),
                    report=report,
                ):
                    return 1
                state = self._state(tenant_id, scope_name) or {}
                promoted_event_seq = int(
                    state.get("promoted_event_seq", 0) or 0
                )
                promoted_conflict_generation = int(
                    state.get("promoted_conflict_generation", 0) or 0
                )
                promoted_raw_token_estimate = int(
                    state.get("promoted_raw_token_estimate", 0) or 0
                )
                promoted_user_turns = int(
                    state.get("promoted_user_turns", 0) or 0
                )
                if (
                    promoted_event_seq < source_event_seq
                    or promoted_conflict_generation < conflict_generation
                    or promoted_raw_token_estimate < raw_token_estimate
                    or promoted_user_turns < user_turns
                ):
                    self.commercial.finish_quarantine_recovery_cycle(
                        tenant_id,
                        scope_name,
                        self.worker_id,
                        state="manual_review",
                        next_attempt_at=moment,
                        error_code="slow_graph_watermark_not_recoverable",
                        report=report,
                    )
                    return 1
                searchable_event_seq = max(
                    int(state.get("indexed_event_seq", 0) or 0),
                    int(state.get("delta_indexed_event_seq", 0) or 0),
                )
            if searchable_event_seq < source_event_seq:
                report.update(
                    {
                        "phase": "indexing",
                        "source_event_seq": source_event_seq,
                        "searchable_event_seq": searchable_event_seq,
                    }
                )
                if self._schedule_quarantine_reindex(
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    source_event_seq=source_event_seq,
                    owner=self.worker_id,
                    now=(moment if now is not None else time.time()),
                    report=report,
                ):
                    return 1
                state = self._state(tenant_id, scope_name) or {}
                searchable_event_seq = max(
                    int(state.get("indexed_event_seq", 0) or 0),
                    int(state.get("delta_indexed_event_seq", 0) or 0),
                )
                if searchable_event_seq < source_event_seq:
                    self.commercial.finish_quarantine_recovery_cycle(
                        tenant_id,
                        scope_name,
                        self.worker_id,
                        state="manual_review",
                        next_attempt_at=moment,
                        error_code="search_index_watermark_not_recoverable",
                        report=report,
                    )
                    return 1
            released = self.commercial.complete_quarantine_recovery(
                tenant_id,
                scope_name,
                self.worker_id,
                report={
                    **report,
                    "phase": "verifying",
                    "source_event_seq": source_event_seq,
                    "promoted_event_seq": promoted_event_seq,
                    "searchable_event_seq": searchable_event_seq,
                },
                audited_historical_failed_ingest_job_ids=(
                    ignored_historical_ingest_job_ids
                ),
            )
            if not released:
                self.commercial.finish_quarantine_recovery_cycle(
                    tenant_id,
                    scope_name,
                    self.worker_id,
                    state="verifying",
                    next_attempt_at=due_after(
                        self._quarantine_recovery_interval()
                    ),
                    error_code="release_precondition_changed",
                    report=report,
                )
            return 1
        except CommercialContractError as exc:
            try:
                self.commercial.finish_quarantine_recovery_cycle(
                    tenant_id,
                    scope_name,
                    self.worker_id,
                    state="manual_review",
                    next_attempt_at=moment,
                    error_code=exc.code,
                    report=report,
                )
            except CommercialContractError:
                pass
            return 1
        except Exception as exc:
            attempt = int(recovery.get("cycle_count", 1) or 1)
            self.commercial.finish_quarantine_recovery_cycle(
                tenant_id,
                scope_name,
                self.worker_id,
                state="waiting",
                next_attempt_at=due_after(
                    self._quarantine_recovery_delay(attempt)
                ),
                error_code=type(exc).__name__,
                report=report,
            )
            return 1

    def _state(self, tenant_id: str, scope_name: str) -> dict[str, object] | None:
        return self.database.get_scope_evolution_state(tenant_id, scope_name)

    def _claim_evolution(
        self,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        *,
        job_version: int | None = None,
    ) -> bool:
        method = getattr(self.jobs, "claim_scope_evolution_job", None)
        if method is not None:
            try:
                supports_version = "job_version" in inspect.signature(method).parameters
            except (TypeError, ValueError):
                supports_version = False
            if not supports_version and self.database is not None:
                return bool(
                    self.database.claim_evolution_job(
                        tenant_id,
                        scope_name,
                        job_id,
                        job_version=job_version,
                    )
                )
            try:
                return bool(
                    method(
                        tenant_id,
                        scope_name,
                        job_id,
                        job_version=job_version,
                    )
                )
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
                return bool(method(tenant_id, scope_name, job_id))
        return bool(
            self.database.claim_evolution_job(
                tenant_id,
                scope_name,
                job_id,
                job_version=job_version,
            )
        )

    def _release_evolution(
        self,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        *,
        job_version: int | None = None,
    ) -> None:
        method = getattr(self.jobs, "release_scope_evolution_job", None)
        if method is not None:
            try:
                supports_version = "job_version" in inspect.signature(method).parameters
            except (TypeError, ValueError):
                supports_version = False
            if not supports_version and self.database is not None:
                self.database.release_evolution_job(
                    tenant_id,
                    scope_name,
                    job_id,
                    job_version=job_version,
                    reason={
                        "code": "version_fenced_scope_claim_release",
                        "job_version": job_version,
                    },
                )
                return
            try:
                method(
                    tenant_id,
                    scope_name,
                    job_id,
                    job_version=job_version,
                    reason={
                        "code": "version_fenced_scope_claim_release",
                        "job_version": job_version,
                    },
                )
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
                if self.database is not None and hasattr(
                    self.database, "release_evolution_job"
                ):
                    self.database.release_evolution_job(
                        tenant_id,
                        scope_name,
                        job_id,
                        job_version=job_version,
                        reason={
                            "code": "version_fenced_scope_claim_release",
                            "job_version": job_version,
                        },
                    )
                else:
                    method(tenant_id, scope_name, job_id)
        elif self.database is not None and hasattr(self.database, "release_evolution_job"):
            self.database.release_evolution_job(
                tenant_id,
                scope_name,
                job_id,
                job_version=job_version,
                reason={
                    "code": "version_fenced_scope_claim_release",
                    "job_version": job_version,
                },
            )

    def _claim_index(
        self,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        *,
        job_version: int | None = None,
    ) -> bool:
        method = getattr(self.jobs, "claim_scope_index_job", None)
        if method is not None:
            try:
                supports_version = "job_version" in inspect.signature(method).parameters
            except (TypeError, ValueError):
                supports_version = False
            if not supports_version and self.database is not None:
                return bool(
                    self.database.claim_index_job(
                        tenant_id,
                        scope_name,
                        job_id,
                        job_version=job_version,
                    )
                )
            try:
                return bool(
                    method(
                        tenant_id,
                        scope_name,
                        job_id,
                        job_version=job_version,
                    )
                )
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
                return bool(method(tenant_id, scope_name, job_id))
        return bool(
            self.database.claim_index_job(
                tenant_id,
                scope_name,
                job_id,
                job_version=job_version,
            )
        )

    def _release_index(
        self,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        *,
        job_version: int | None = None,
    ) -> None:
        method = getattr(self.jobs, "release_scope_index_job", None)
        if method is not None:
            try:
                supports_version = "job_version" in inspect.signature(method).parameters
            except (TypeError, ValueError):
                supports_version = False
            if not supports_version and self.database is not None:
                self.database.release_index_job(
                    tenant_id,
                    scope_name,
                    job_id,
                    job_version=job_version,
                    reason={
                        "code": "version_fenced_scope_claim_release",
                        "job_version": job_version,
                    },
                )
                return
            try:
                method(
                    tenant_id,
                    scope_name,
                    job_id,
                    job_version=job_version,
                    reason={
                        "code": "version_fenced_scope_claim_release",
                        "job_version": job_version,
                    },
                )
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
                if self.database is not None and hasattr(
                    self.database, "release_index_job"
                ):
                    self.database.release_index_job(
                        tenant_id,
                        scope_name,
                        job_id,
                        job_version=job_version,
                        reason={
                            "code": "version_fenced_scope_claim_release",
                            "job_version": job_version,
                        },
                    )
                else:
                    method(tenant_id, scope_name, job_id)
        elif self.database is not None and hasattr(self.database, "release_index_job"):
            self.database.release_index_job(
                tenant_id,
                scope_name,
                job_id,
                job_version=job_version,
                reason={
                    "code": "version_fenced_scope_claim_release",
                    "job_version": job_version,
                },
            )

    def _advance_index(
        self, tenant_id: str, scope_name: str, *, target_event_seq: int, job_id: str
    ) -> dict[str, object]:
        method = getattr(self.jobs, "advance_index_watermark", None)
        kwargs = {
            "indexed_event_seq": target_event_seq,
            "index_succeeded": True,
            "index_job_id": job_id,
        }
        if method is not None:
            return dict(method(tenant_id, scope_name, **kwargs))
        return dict(self.database.advance_index_watermark(tenant_id, scope_name, **kwargs))

    def _advance_delta_index(
        self, tenant_id: str, scope_name: str, *, target_event_seq: int, job_id: str
    ) -> dict[str, object]:
        method = getattr(self.jobs, "advance_delta_index_watermark", None)
        kwargs = {
            "delta_indexed_event_seq": target_event_seq,
            "index_job_id": job_id,
        }
        if method is not None:
            return dict(method(tenant_id, scope_name, **kwargs))
        return dict(
            self.database.advance_delta_index_watermark(
                tenant_id, scope_name, **kwargs
            )
        )

    def _advance_evolution(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        target_event_seq: int,
        target_conflict_generation: int,
        target_raw_token_estimate: int,
        target_user_turns: int,
        job_id: str,
    ) -> dict[str, object]:
        method = getattr(self.jobs, "advance_evolution_watermarks", None)
        kwargs = {
            "source_event_seq": target_event_seq,
            "conflict_generation": target_conflict_generation,
            "slow_succeeded": True,
            "index_activated": True,
            "evolution_job_id": job_id,
            "raw_token_estimate": target_raw_token_estimate,
            "user_turns": target_user_turns,
        }
        try:
            if method is not None:
                return dict(method(tenant_id, scope_name, **kwargs))
            return dict(self.database.advance_promoted_watermarks(tenant_id, scope_name, **kwargs))
        except TypeError as exc:
            if "unexpected keyword" not in str(exc) and "positional" not in str(exc):
                raise
            kwargs.pop("raw_token_estimate")
            kwargs.pop("user_turns")
            if method is not None:
                return dict(method(tenant_id, scope_name, **kwargs))
            return dict(self.database.advance_promoted_watermarks(tenant_id, scope_name, **kwargs))

    def _record_ingest_source(
        self,
        tenant_id: str,
        scope_name: str,
        operation_id: str,
        writer: Mapping[str, Any],
        messages: list[Mapping[str, Any]],
        *,
        required_failed_job_id: str | None = None,
        required_failed_stage_id: str | None = None,
        required_failed_stage_attempt: int | None = None,
    ) -> dict[str, object]:
        state = self._state(tenant_id, scope_name) or {}
        durable_sources = writer.get("durable_sources")
        if durable_sources is not None:
            if not isinstance(durable_sources, list) or any(
                not isinstance(item, Mapping) for item in durable_sources
            ):
                raise RuntimeErrorBase("writer durable_sources must be an object array")
            commit_options: dict[str, Any] = {
                "conflict_generation": int(
                    state.get("conflict_generation", 0) or 0
                )
            }
            if required_failed_job_id is not None:
                commit_options.update(
                    {
                        "required_failed_job_id": required_failed_job_id,
                        "required_failed_stage_id": required_failed_stage_id,
                        "required_failed_stage_attempt": required_failed_stage_attempt,
                    }
                )
            return dict(
                self.database.record_committed_source_records(
                    tenant_id,
                    scope_name,
                    operation_id,
                    durable_sources,
                    **commit_options,
                )
            )
        current_seq = int(state.get("source_event_seq", 0) or 0)
        new_count = int(writer.get("new_message_count", 0) or 0)
        if new_count < 0:
            raise RuntimeErrorBase("writer new_message_count must be non-negative")
        if new_count == 0:
            return state
        raw_token_estimate = writer.get("new_raw_token_estimate")
        if raw_token_estimate is None:
            raw_token_estimate = writer.get("raw_token_estimate")
        if raw_token_estimate is None:
            raw_token_estimate = sum(
                self._estimate_raw_tokens(str(message.get("content") or ""))
                for message in messages
            )
        user_turns = writer.get("new_user_turns")
        if user_turns is None:
            user_turns = writer.get("new_user_turn_count")
        if user_turns is None:
            user_turns = sum(
                1 for message in messages if str(message.get("role") or "") == "user"
            )
        raw_token_estimate = int(raw_token_estimate or 0)
        user_turns = int(user_turns or 0)
        if raw_token_estimate < 0 or user_turns < 0:
            raise RuntimeErrorBase("ingest evolution metrics must be non-negative")
        return dict(
            self.database.record_committed_source_events(
                tenant_id,
                scope_name,
                current_seq + new_count,
                conflict_generation=int(state.get("conflict_generation", 0) or 0),
                operation_id=operation_id,
                new_message_count=new_count,
                raw_token_estimate=raw_token_estimate,
                user_turns=user_turns,
            )
        )

    def _prepare_ingest_source_accounting(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        session_id: str,
        job_id: str,
        messages: list[Mapping[str, Any]],
        writer: Mapping[str, Any],
        default_accounting_operation_id: str,
    ) -> tuple[dict[str, Any], str]:
        preparer = getattr(self.storage, "prepare_writer_source_accounting", None)
        if not callable(preparer):
            return dict(writer), default_accounting_operation_id
        prepared = preparer(
            tenant_id=tenant_id,
            scope_name=scope_name,
            session_id=session_id,
            job_id=job_id,
            messages=messages,
            writer=writer,
            default_accounting_operation_id=default_accounting_operation_id,
        )
        if not isinstance(prepared, Mapping):
            raise RuntimeErrorBase("writer Source accounting preparation is invalid")
        prepared_writer = prepared.get("writer")
        operation_id = str(prepared.get("accounting_operation_id") or "").strip()
        if not isinstance(prepared_writer, Mapping) or not operation_id:
            raise RuntimeErrorBase("writer Source accounting preparation is invalid")
        return dict(prepared_writer), operation_id

    def _recover_ingest_source_accounting(
        self,
        *,
        tenant_id: str,
        scope_name: str,
        session_id: str,
        job_id: str,
        messages: list[Mapping[str, Any]],
        default_accounting_operation_id: str,
    ) -> int:
        """Account a proven Source prefix after the Writer loses its report."""

        recoverer = getattr(self.storage, "recover_writer_source_accounting", None)
        if not callable(recoverer):
            return 0
        guard_factory = getattr(
            self.storage, "source_accounting_recovery_guard", None
        )
        guard = (
            guard_factory(tenant_id=tenant_id, scope_name=scope_name)
            if callable(guard_factory)
            else nullcontext()
        )
        with guard:
            prepared = recoverer(
                tenant_id=tenant_id,
                scope_name=scope_name,
                session_id=session_id,
                job_id=job_id,
                messages=messages,
                default_accounting_operation_id=default_accounting_operation_id,
            )
            if not isinstance(prepared, Mapping):
                raise RuntimeErrorBase(
                    "writer Source recovery preparation is invalid"
                )
            writer = prepared.get("writer")
            operation_id = str(
                prepared.get("accounting_operation_id") or ""
            ).strip()
            recovered_source_count = int(
                prepared.get("recovered_source_count", 0) or 0
            )
            if recovered_source_count < 0:
                raise RuntimeErrorBase("writer Source recovery count is invalid")
            if recovered_source_count == 0:
                return 0
            if not isinstance(writer, Mapping) or not operation_id:
                raise RuntimeErrorBase(
                    "writer Source recovery preparation is invalid"
                )
            durable_sources = writer.get("durable_sources")
            if (
                not isinstance(durable_sources, list)
                or len(durable_sources) != recovered_source_count
            ):
                raise RuntimeErrorBase("writer Source recovery set is invalid")
            self._record_ingest_source(
                tenant_id=tenant_id,
                scope_name=scope_name,
                operation_id=operation_id,
                writer=writer,
                messages=[],
            )
        return recovered_source_count

    def _recover_scope_source_accounting(
        self, tenant_id: str, scope_name: str
    ) -> dict[str, int]:
        """Apply read-only-proven Source ledger plans without resuming Writers."""

        with self._scope_execution_lock((tenant_id, scope_name, "mutation")):
            return self._recover_scope_source_accounting_locked(
                tenant_id, scope_name
            )

    def _recover_scope_source_accounting_locked(
        self, tenant_id: str, scope_name: str
    ) -> dict[str, int]:
        """Recover one static scope while its in-process mutation lane is held."""

        planner = getattr(self.storage, "source_accounting_recovery_plans", None)
        if not callable(planner):
            return {"operation_count": 0, "source_count": 0}
        plans = planner(tenant_id=tenant_id, scope_name=scope_name)
        if not isinstance(plans, list) or any(
            not isinstance(plan, Mapping) for plan in plans
        ):
            raise RuntimeErrorBase("scope Source recovery plans are invalid")
        operation_count = 0
        source_count = 0
        prior_scope_seq = -1
        validator = getattr(
            self.storage, "validate_source_accounting_recovery_plan", None
        )
        guard_factory = getattr(
            self.storage, "source_accounting_recovery_guard", None
        )
        for plan in plans:
            scope_seq = int(plan.get("scope_seq", 0) or 0)
            job_id = str(plan.get("job_id") or "").strip()
            stage_id = str(plan.get("writer_stage_id") or "").strip()
            stage_attempt = int(plan.get("writer_stage_attempt", 0) or 0)
            operation_id = str(
                plan.get("accounting_operation_id") or ""
            ).strip()
            writer = plan.get("writer")
            planned_source_count = int(plan.get("source_count", 0) or 0)
            if (
                scope_seq < prior_scope_seq
                or not job_id
                or stage_id != f"{job_id}:writer"
                or stage_attempt <= 0
                or not operation_id
                or not isinstance(writer, Mapping)
                or planned_source_count <= 0
                or not isinstance(writer.get("durable_sources"), list)
                or len(writer["durable_sources"]) != planned_source_count
            ):
                raise RuntimeErrorBase("scope Source recovery plan is invalid")
            prior_scope_seq = scope_seq
            guard = (
                guard_factory(tenant_id=tenant_id, scope_name=scope_name)
                if callable(guard_factory)
                else nullcontext()
            )
            with guard:
                if callable(validator):
                    validator(
                        tenant_id=tenant_id,
                        scope_name=scope_name,
                        plan=plan,
                    )
                before = self._state(tenant_id, scope_name) or {}
                before_seq = int(before.get("source_event_seq", 0) or 0)
                try:
                    after = self._record_ingest_source(
                        tenant_id=tenant_id,
                        scope_name=scope_name,
                        operation_id=operation_id,
                        writer=writer,
                        messages=[],
                        required_failed_job_id=job_id,
                        required_failed_stage_id=stage_id,
                        required_failed_stage_attempt=stage_attempt,
                    )
                except StaleSourceAccountingRecovery:
                    # The Writer was requeued while the read-only graph scan was
                    # running. Its live attempt exclusively owns Source accounting.
                    continue
            after_seq = int(after.get("source_event_seq", 0) or 0)
            if after_seq < before_seq:
                raise RuntimeErrorBase("scope Source recovery moved its watermark back")
            operation_count += int(after_seq > before_seq)
            source_count += after_seq - before_seq
        return {
            "operation_count": operation_count,
            "source_count": source_count,
        }

    @staticmethod
    def _writer_accounting_operation_id(
        report: Mapping[str, Any],
        *,
        expected_stage_id: str,
        current_stage_attempt: int,
    ) -> str:
        """Bind Source accounting to the Writer attempt that produced the report."""

        reported_stage_id = report.get("stage_id")
        reported_attempt = report.get("stage_attempt")
        if (
            reported_stage_id != expected_stage_id
            or isinstance(reported_attempt, bool)
            or not isinstance(reported_attempt, int)
            or reported_attempt <= 0
            or reported_attempt > current_stage_attempt
        ):
            raise RuntimeErrorBase("writer accounting attempt identity is invalid")
        return f"{reported_stage_id}:attempt:{reported_attempt}"

    @staticmethod
    def _estimate_raw_tokens(content: str) -> int:
        if not content:
            return 0
        characters = [character for character in content if not character.isspace()]
        cjk = sum(
            1
            for character in characters
            if any(
                start <= ord(character) <= end
                for start, end in (
                    (0x3400, 0x4DBF),
                    (0x4E00, 0x9FFF),
                    (0xF900, 0xFAFF),
                )
            )
        )
        other = len(characters) - cjk
        return cjk + (other + 3) // 4

    def _has_active_generation(self, tenant_id: str, scope_name: str) -> bool:
        try:
            snapshot = self.storage.active_snapshot(tenant_id, scope_name)
        except Exception:
            return False
        return isinstance(snapshot, Mapping) and bool(snapshot)

    def _run_stage(
        self,
        job: Job,
        stage_name: str,
        stage_seq: int,
        action: Callable[[str, int], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Run one durable side-effect stage, replaying only incomplete work."""
        stage_id = f"{job.job_id}:{stage_name}"
        stage = self.jobs.create_stage(
            job_id=job.job_id,
            stage_name=stage_name,
            stage_seq=stage_seq,
            stage_id=stage_id,
        )
        if stage.state == STAGE_SUCCEEDED:
            return dict(stage.result or {})
        if stage.state == STAGE_CANCELLED:
            raise RuntimeErrorBase(f"stage {stage_name} was cancelled")
        if stage.state == STAGE_RUNNING:
            lease_expired = (
                stage.lease_expires_at is not None
                and float(stage.lease_expires_at) <= time.time()
            )
            if not lease_expired:
                raise RuntimeErrorBase(f"stage {stage_name} is owned by another worker")
            if not self.jobs.fail_expired_stage(
                stage_id,
                str(stage.worker_id or ""),
                "stage_lease_expired",
                stage_version=stage.version,
            ):
                raise RuntimeErrorBase(
                    f"stage {stage_name} lease ownership changed during recovery"
                )
            stage = self.jobs.retry_stage(stage_id)
        elif stage.state == STAGE_FAILED:
            stage = self.jobs.retry_stage(stage_id)
        enforce_attempt = job.state == RUNNING
        if enforce_attempt:
            self.jobs.assert_running_attempt(job.job_id, self.worker_id, job.version)
        stage = self.jobs.claim_stage(
            stage.stage_id,
            self.worker_id,
            job_version=job.version if enforce_attempt else None,
        )
        stage_heartbeat_stop = threading.Event()

        def keep_stage_lease() -> None:
            interval = max(0.1, min(30.0, self.jobs.lease_seconds / 3.0))
            while not stage_heartbeat_stop.wait(interval):
                try:
                    if not self.jobs.stage_heartbeat(
                        stage.stage_id,
                        self.worker_id,
                        stage_version=stage.version,
                    ):
                        return
                except Exception:
                    traceback.print_exc()

        stage_heartbeat = threading.Thread(
            target=keep_stage_lease,
            name=f"{self.worker_id}-stage-lease-{stage.stage_id}",
            daemon=True,
        )
        stage_heartbeat.start()
        try:
            result = dict(action(stage.stage_id, stage.attempt))
            if enforce_attempt:
                self.jobs.assert_running_attempt(job.job_id, self.worker_id, job.version)
        except Exception as exc:
            self._record_exception(
                exc,
                operation="stage_failed",
                job=job,
                stage_id=stage.stage_id,
                stage_name=stage_name,
                stage_attempt=stage.attempt,
            )
            try:
                self.jobs.fail_stage(
                    stage.stage_id,
                    f"{type(exc).__name__}:{exc}",
                    worker_id=self.worker_id,
                    stage_version=stage.version,
                )
            except JobStateError:
                pass
            raise
        finally:
            stage_heartbeat_stop.set()
            stage_heartbeat.join(timeout=2.0)
        self.jobs.complete_stage(
            stage.stage_id,
            result,
            worker_id=self.worker_id,
            stage_version=stage.version,
        )
        return result

    def _resident_base_builder(
        self,
        tenant_id: str,
        *,
        workload: GpuWorkload = GpuWorkload.INDEX_BACKGROUND,
    ) -> Callable[..., Mapping[str, Any]]:
        if self.online is None:
            raise RuntimeErrorBase("resident online index engine is unavailable")

        def build(**kwargs: Any) -> Mapping[str, Any]:
            return dict(
                self.online.execute(
                    tenant_id,
                    lambda engine: engine.build_base_index(**kwargs),
                    queue_timeout=float(
                        getattr(
                            self.settings,
                            "recall_queue_timeout_seconds",
                            30.0,
                        )
                    ),
                    workload=workload,
                )
            )

        return build

    def _run_index(
        self,
        job: Job,
        *,
        target_event_seq: int,
    ) -> dict[str, Any]:
        scope_name = str((job.payload or {}).get("scope_name") or "default")
        tenant_id = job.tenant_id
        job_type = str((job.payload or {}).get("job_type") or "")
        workload = self._index_workload(job)
        stage_seq = 1 if job_type in {
            "ingest",
            "delete_memories",
            "delete_session",
        } else 0

        def execute(_stage_id: str, _stage_attempt: int) -> Mapping[str, Any]:
            if not self._claim_index(tenant_id, scope_name, job.job_id):
                raise RuntimeErrorBase("index job does not own this scope")
            try:
                index = self.storage.build_index(
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    job_id=f"{job.job_id}_index",
                    source_event_seq=target_event_seq,
                    builder=self._resident_base_builder(
                        tenant_id,
                        workload=workload,
                    ),
                )
                watermarks = self._advance_index(
                    tenant_id,
                    scope_name,
                    target_event_seq=target_event_seq,
                    job_id=job.job_id,
                )
                return {"index": index, "watermarks": watermarks}
            except Exception:
                self._release_index(tenant_id, scope_name, job.job_id)
                raise

        return self._run_stage(job, "index", stage_seq, execute)

    def _run_delta_index(
        self,
        job: Job,
        *,
        target_event_seq: int,
    ) -> dict[str, Any]:
        scope_name = str((job.payload or {}).get("scope_name") or "default")
        tenant_id = job.tenant_id
        workload = self._index_workload(job)
        if self.online is None:
            raise RuntimeErrorBase("resident online index engine is unavailable")

        def execute(_stage_id: str, _stage_attempt: int) -> Mapping[str, Any]:
            if not self._claim_index(tenant_id, scope_name, job.job_id):
                raise RuntimeErrorBase("delta index job does not own this scope")
            try:
                def resident_builder(**kwargs: Any) -> Mapping[str, Any]:
                    return dict(
                        self.online.execute(
                            tenant_id,
                            lambda engine: engine.build_delta_index(**kwargs),
                            queue_timeout=float(
                                getattr(
                                    self.settings,
                                    "recall_queue_timeout_seconds",
                                    30.0,
                                )
                            ),
                            workload=workload,
                        )
                    )

                index = self.storage.build_delta_index(
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    job_id=f"{job.job_id}_delta",
                    source_event_seq=target_event_seq,
                    builder=resident_builder,
                )
                watermarks = self._advance_delta_index(
                    tenant_id,
                    scope_name,
                    target_event_seq=target_event_seq,
                    job_id=job.job_id,
                )
                return {"index": index, "watermarks": watermarks}
            except Exception:
                self._release_index(tenant_id, scope_name, job.job_id)
                raise

        return self._run_stage(job, "delta_index", 1, execute)

    def _claim_index_after_slow(
        self,
        tenant_id: str,
        scope_name: str,
        job_id: str,
    ) -> None:
        wait_seconds = float(
            getattr(self.settings, "index_claim_wait_seconds", 900.0)
        )
        if wait_seconds <= 0:
            raise ValueError("index_claim_wait_seconds must be positive")
        deadline = time.monotonic() + wait_seconds
        while not self._stop.is_set():
            if self._claim_index(tenant_id, scope_name, job_id):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                state = self._state(tenant_id, scope_name) or {}
                owner = str(state.get("active_index_job_id") or "")
                raise RuntimeErrorBase(
                    "timed out waiting for the post-Slow index lane"
                    + (f" owned by {owner}" if owner else "")
                )
            self._stop.wait(min(0.5, remaining))
        raise RuntimeErrorBase("service stopped while waiting for the post-Slow index lane")

    def _run_consolidation(self, job: Job) -> tuple[dict[str, Any], dict[str, Any], dict[str, object]]:
        scope_name = str((job.payload or {}).get("scope_name") or "default")
        tenant_id = job.tenant_id
        payload = dict(job.payload or {})
        usage_attribution = UsageAttribution.from_mapping(
            payload.get("_usage_attribution")
            if isinstance(payload.get("_usage_attribution"), Mapping)
            else None
        )
        raw_provider_execution = payload.get("_provider_execution")
        organizer_execution = (
            raw_provider_execution
            if isinstance(raw_provider_execution, Mapping)
            and str(raw_provider_execution.get("organizer") or "").strip()
            else None
        )
        state = self._state(tenant_id, scope_name) or {}
        target_event_seq = int(
            payload.get("target_source_event_seq", state.get("source_event_seq", 0)) or 0
        )
        target_conflict_generation = int(
            payload.get(
                "target_conflict_generation", state.get("conflict_generation", 0)
            )
            or 0
        )
        target_raw_token_estimate = int(
            payload.get(
                "target_raw_token_estimate", state.get("source_raw_token_estimate", 0)
            )
            or 0
        )
        target_user_turns = int(
            payload.get("target_user_turns", state.get("source_user_turns", 0)) or 0
        )
        is_ingest = str((job.payload or {}).get("job_type") or "") == "ingest"
        base_seq = 1 if is_ingest else 0
        if not self._claim_evolution(tenant_id, scope_name, job.job_id):
            raise RuntimeErrorBase("evolution job does not own this scope")
        index_claimed = False
        try:
            def execute_slow(
                stage_id: str, _stage_attempt: int
            ) -> Mapping[str, Any]:
                graph_lease = (
                    self.gpu_scheduler.lease(GpuWorkload.GRAPH_BACKGROUND)
                    if self.gpu_scheduler is not None
                    and self.slow_graph_uses_local_gpu
                    else nullcontext()
                )
                with graph_lease:
                    return self.storage.consolidate_slow(
                        tenant_id=tenant_id,
                        scope_name=scope_name,
                        job_id=f"{job.job_id}_slow",
                        ledger_job_id=job.job_id,
                        ledger_stage_id=stage_id,
                        usage_attribution=usage_attribution,
                        provider_execution=organizer_execution,
                    )

            slow = self._run_stage(
                job,
                "slow",
                base_seq,
                execute_slow,
            )
            self._claim_index_after_slow(tenant_id, scope_name, job.job_id)
            index_claimed = True
            index_stage = self._run_stage(
                job,
                "index",
                base_seq + 1,
                lambda _stage_id, _stage_attempt: {
                    "index": self.storage.build_index(
                        tenant_id=tenant_id,
                        scope_name=scope_name,
                        job_id=f"{job.job_id}_index",
                        source_event_seq=target_event_seq,
                        builder=self._resident_base_builder(
                            tenant_id,
                            workload=self._index_workload(job),
                        ),
                    )
                },
            )
            index = dict(index_stage.get("index") or {})
            promoted = self._run_stage(
                job,
                "promote",
                base_seq + 2,
                lambda _stage_id, _stage_attempt: {
                    "watermarks": self._advance_evolution(
                        tenant_id,
                        scope_name,
                        target_event_seq=target_event_seq,
                        target_conflict_generation=target_conflict_generation,
                        target_raw_token_estimate=target_raw_token_estimate,
                        target_user_turns=target_user_turns,
                        job_id=job.job_id,
                    )
                },
            )
            watermarks = dict(promoted.get("watermarks") or {})
            self._release_index(tenant_id, scope_name, job.job_id)
            index_claimed = False
            if self.on_generation_committed is not None:
                try:
                    self.on_generation_committed(
                        tenant_id,
                        scope_name,
                        int(watermarks.get("promoted_event_seq", target_event_seq) or 0),
                    )
                except Exception:
                    traceback.print_exc()
            return slow, index, watermarks
        except Exception:
            if index_claimed:
                self._release_index(tenant_id, scope_name, job.job_id)
            self._release_evolution(tenant_id, scope_name, job.job_id)
            raise

    def _due_scopes(self, *, evolution: bool, now: float) -> list[dict[str, object]]:
        if evolution:
            method = getattr(self.jobs, "list_due_evolution_scopes", None)
            dirty_token_threshold = int(
                getattr(self.settings, "slow_dirty_token_threshold", 32_000)
            )
            dirty_user_turn_threshold = int(
                getattr(self.settings, "slow_dirty_user_turn_threshold", 64)
            )
            max_age = float(getattr(self.settings, "slow_max_age_seconds", 86_400.0))
            min_token_threshold = int(
                getattr(self.settings, "slow_min_token_threshold", 4_000)
            )
            min_user_turn_threshold = int(
                getattr(self.settings, "slow_min_user_turn_threshold", 8)
            )
            min_success_interval = float(
                getattr(self.settings, "slow_min_interval_seconds", 1_800.0)
            )
        else:
            method = getattr(self.jobs, "list_due_index_scopes", None)
            dirty_threshold = int(getattr(self.settings, "index_dirty_threshold", 256))
            max_age = float(getattr(self.settings, "index_max_age_seconds", 300.0))
        if method is None:
            method = (
                self.database.list_due_scopes
                if evolution
                else self.database.list_due_index_scopes
            )
        if evolution:
            try:
                return list(
                    method(
                        dirty_token_threshold=dirty_token_threshold,
                        dirty_user_turn_threshold=dirty_user_turn_threshold,
                        max_age_seconds=max_age,
                        min_token_threshold=min_token_threshold,
                        min_user_turn_threshold=min_user_turn_threshold,
                        min_success_interval_seconds=min_success_interval,
                        now=now,
                        include_conflicts=False,
                    )
                )
            except TypeError as exc:
                if "unexpected keyword" not in str(exc) and "positional" not in str(exc):
                    raise
                # Old schemas cannot evaluate the token/turn policy safely.
                return []
        return list(method(dirty_threshold=dirty_threshold, max_age_seconds=max_age, now=now))

    def _schedule_auto_job(self, row: Mapping[str, object], job_type: str) -> bool:
        tenant_id = str(row["tenant_id"])
        scope_name = str(row["scope_name"])
        source_seq = int(row.get("source_event_seq", 0) or 0)
        conflict_generation = int(row.get("conflict_generation", 0) or 0)
        if job_type == "consolidate":
            key = f"auto:consolidate:{scope_name}:{source_seq}:{conflict_generation}"
        else:
            key = f"auto:reindex:{scope_name}:{source_seq}"
        payload: dict[str, object] = {
            "job_type": job_type,
            "scope_name": scope_name,
            "auto": True,
            "target_source_event_seq": source_seq,
            "target_conflict_generation": conflict_generation,
            "target_raw_token_estimate": int(
                row.get("source_raw_token_estimate", 0) or 0
            ),
            "target_user_turns": int(row.get("source_user_turns", 0) or 0),
        }
        if job_type == "consolidate":
            # Automatic graph maintenance is a real tenant-scoped cost, but it
            # is not truthfully owned by the last client that touched a scope.
            # Keep it in the same ledger under an explicit internal cost center.
            payload["_usage_attribution"] = SYSTEM_MAINTENANCE.as_dict()
        try:
            job = self.jobs.submit(
                tenant_id,
                key,
                payload,
                scope_name=scope_name,
                tenant_queue_limit=getattr(self.settings, "tenant_queue_limit", None),
                global_queue_limit=getattr(self.settings, "global_queue_limit", None),
            )
            # A failed derived job requires an explicit audited retry. Replaying
            # it here can repeat a corrupt projection or an uncertain provider
            # outcome on every scheduler tick.
            if getattr(job, "state", PENDING) == FAILED:
                return False
            if getattr(job, "state", PENDING) in {SUCCEEDED, CANCELLED}:
                return False
            claim = (
                self.jobs.claim_scope_evolution_job
                if job_type == "consolidate"
                else self.jobs.claim_scope_index_job
            )
            if not claim(tenant_id, scope_name, job.job_id):
                if getattr(job, "state", PENDING) == PENDING:
                    cancel = getattr(self.jobs, "cancel", None)
                    if cancel is not None:
                        cancel(job.job_id)
                return False
            return True
        except Exception:
            return False

    def _schedule_due_jobs(self, *, now: float | None = None) -> int:
        if self.database is None:
            return 0
        moment = time.time() if now is None else float(now)
        try:
            evolution_due = self._due_scopes(evolution=True, now=moment)
            index_due = self._due_scopes(evolution=False, now=moment)
        except Exception:
            return 0
        index_by_scope = {
            (str(row["tenant_id"]), str(row["scope_name"])): row
            for row in index_due
        }
        evolution_by_scope = {
            (str(row["tenant_id"]), str(row["scope_name"])): row
            for row in evolution_due
        }
        scheduled = 0
        for key in dict.fromkeys([*evolution_by_scope, *index_by_scope]):
            row = evolution_by_scope.get(key) or index_by_scope[key]
            if self.commercial is not None:
                try:
                    self.commercial.require_scope_active(*key)
                except CommercialContractError:
                    continue
            evolution_row = evolution_by_scope.get(key)
            if evolution_row is not None and not evolution_row.get(
                "active_evolution_job_id"
            ):
                if self._schedule_auto_job(evolution_row, "consolidate"):
                    scheduled += 1
            index_row = index_by_scope.get(key)
            if index_row is not None and not index_row.get("active_index_job_id"):
                if self._schedule_auto_job(index_row, "reindex"):
                    scheduled += 1
        if self.commercial is not None:
            for row in self.commercial.due_retention_scopes(now=moment):
                tenant_id = str(row["tenant_id"])
                scope_name = str(row["scope_name"])
                last_ingest_at = int(float(row["last_ingest_at"]))
                idempotency_key = (
                    f"auto:retention-delete:{scope_name}:{last_ingest_at}:"
                    f"{int(row['inactive_days'])}"
                )
                try:
                    job = self.jobs.submit(
                        tenant_id,
                        idempotency_key,
                        {
                            "job_type": "delete_scope",
                            "scope_name": scope_name,
                            "reason": "retention_policy",
                            "auto": True,
                        },
                        scope_name=scope_name,
                        tenant_queue_limit=getattr(
                            self.settings, "tenant_queue_limit", None
                        ),
                        global_queue_limit=getattr(
                            self.settings, "global_queue_limit", None
                        ),
                    )
                    if job.state not in {SUCCEEDED, CANCELLED}:
                        self.commercial.mark_scope_deleting(
                            tenant_id,
                            scope_name,
                            job.job_id,
                            reason="retention_policy",
                        )
                        scheduled += 1
                except Exception:
                    continue
        return scheduled

    def _claim_next(self) -> Job | None:
        """Claim the oldest pending job whose scope is not locally occupied.

        The direct selection path keeps a still-global-FIFO JobStore from
        blocking unrelated scopes. The fallback is retained for small test
        doubles and older stores; the execution lock still serializes scopes.
        """
        if isinstance(self.database, ControlDB) and isinstance(self.jobs, JobStore):
            with self.database.transaction(immediate=False) as connection:
                rows = connection.execute(
                    "SELECT job_id, tenant_id, scope_name, payload_json FROM jobs "
                    "WHERE state=? ORDER BY created_at, job_id",
                    (PENDING,),
                ).fetchall()
                quarantined = {
                    (str(row["tenant_id"]), str(row["scope_name"]))
                    for row in connection.execute(
                        "SELECT tenant_id,scope_name FROM scope_quarantines"
                    ).fetchall()
                }
                inactive = {
                    (str(row["tenant_id"]), str(row["scope_name"]))
                    for row in connection.execute(
                        "SELECT tenant_id,scope_name FROM scope_lifecycle "
                        "WHERE state<>'active'"
                    ).fetchall()
                }
                content_deleting = {
                    (str(row["tenant_id"]), str(row["scope_name"]))
                    for row in connection.execute(
                        "SELECT tenant_id,scope_name FROM content_deletions "
                        "WHERE state IN ('requested','purging','reindexing','failed')"
                    ).fetchall()
                }
                blocked_scopes = quarantined | inactive | content_deleting
            rows = sorted(
                rows,
                key=lambda row: (
                    str(json.loads(row["payload_json"]).get("job_type") or "")
                    in {"reindex", "consolidate"},
                ),
            )
            for row in rows:
                payload = json.loads(row["payload_json"])
                job_type = str(payload.get("job_type") or "")
                if self.gpu_scheduler is not None and job_type in {
                    "reindex",
                    "consolidate",
                }:
                    scheduler_workload = (
                        GpuWorkload.GRAPH_BACKGROUND
                        if job_type == "consolidate"
                        else GpuWorkload.INDEX_BACKGROUND
                    )
                    if not self.gpu_scheduler.can_start(scheduler_workload):
                        continue
                scope_key = (
                    str(row["tenant_id"] or ""),
                    str(row["scope_name"] or "default"),
                )
                if (
                    scope_key in blocked_scopes
                    and str(payload.get("job_type") or "")
                    not in {
                        "export_scope",
                        "delete_scope",
                        "delete_memories",
                        "delete_session",
                    }
                    and not (
                        self.commercial is not None
                        and self.commercial.is_quarantine_recovery_job(
                            scope_key[0], scope_key[1], str(row["job_id"])
                        )
                    )
                ):
                    continue
                is_quarantine_recovery = bool(
                    self.commercial is not None
                    and str(payload.get("job_type") or "") == "ingest"
                    and self.commercial.is_quarantine_recovery_job(
                        scope_key[0], scope_key[1], str(row["job_id"])
                    )
                )
                lane = self._execution_lane(payload)
                if self._scope_lane_is_busy(scope_key, lane):
                    continue
                try:
                    claimed = self.jobs.claim(row["job_id"], self.worker_id)
                    if is_quarantine_recovery:
                        with self._state_lock:
                            self._claimed_quarantine_recovery_jobs.add(
                                claimed.job_id
                            )
                    return claimed
                except JobStateError:
                    continue
            return None
        return self.jobs.claim_next(self.worker_id)

    def recover_abandoned_jobs(self) -> int:
        recovered = 0
        for job in self.jobs.expired_running():
            payload = dict(job.payload or {})
            job_type = str(payload.get("job_type") or "")
            safe_to_resume = job_type in {
                "reindex",
                "consolidate",
                "export_scope",
                "delete_scope",
                "delete_memories",
                "delete_session",
            }
            scope_name = job.scope_name
            resume_evidence: dict[str, Any] | None = None
            if job_type == "ingest":
                safe_to_resume = self.storage.can_resume_ingest(
                    tenant_id=job.tenant_id,
                    scope_name=scope_name,
                    job_id=job.job_id,
                )
                audit_reader = getattr(self.storage, "audit_scope_recovery", None)
                if safe_to_resume and callable(audit_reader):
                    audit = dict(
                        audit_reader(
                            tenant_id=job.tenant_id,
                            scope_name=scope_name,
                            job_id=job.job_id,
                        )
                    )
                    plan = self._ingest_recovery_plan(
                        tenant_id=job.tenant_id,
                        scope_name=scope_name,
                        job_id=job.job_id,
                    )
                    safe_to_resume = bool(
                        audit.get("integrity_ok")
                        and job.job_id
                        in {
                            str(value)
                            for value in audit.get("failed_operation_ids", ())
                            if str(value)
                        }
                        and plan.get("resumable")
                    )
                    if safe_to_resume:
                        resume_evidence = {
                            "tenant_id": job.tenant_id,
                            "scope_name": scope_name,
                            "job_id": job.job_id,
                            "audit": audit,
                            "recovery_plan": plan,
                        }
                else:
                    safe_to_resume = False
            try:
                scope_guard = self._scope_execution_lock(
                    self._job_lock_key(job), blocking=False
                )
                with scope_guard:
                    expired = self.jobs.fail_expired(
                        job.job_id,
                        str(job.worker_id or ""),
                        "worker_lease_expired_after_committed_stage"
                        if safe_to_resume
                        else "process_lost_requires_explicit_artifact_audit",
                        job_version=job.version,
                    )
                    # ``scope_evolution_state`` still stores the version that
                    # created the old claim. The job version increments in
                    # ``fail_expired``, so release with the pre-transition
                    # version captured above, never with the new attempt's
                    # version.
                    if expired and job_type == "consolidate":
                        self._release_evolution(
                            job.tenant_id,
                            scope_name,
                            job.job_id,
                            job_version=job.version,
                        )
                        self._release_index(
                            job.tenant_id,
                            scope_name,
                            job.job_id,
                            job_version=job.version,
                        )
                    elif expired and job_type == "reindex":
                        self._release_index(
                            job.tenant_id,
                            scope_name,
                            job.job_id,
                            job_version=job.version,
                        )
                    if expired and safe_to_resume:
                        failed_job = self.jobs.get(job.job_id)
                        if failed_job is None:
                            raise JobStateError(
                                "expired job disappeared before safe resume"
                            )
                        recovery_mode = "committed_stage_state"
                        if job_type == "ingest":
                            recovery_plan = (
                                resume_evidence.get("recovery_plan")
                                if isinstance(resume_evidence, Mapping)
                                else None
                            )
                            recovery_mode = (
                                str(recovery_plan.get("mode") or "").strip()
                                if isinstance(recovery_plan, Mapping)
                                else ""
                            ) or "audited_writer_state"
                        self._resume_failed_authorized(
                            failed_job,
                            code="requeue_after_worker_lease_expiry",
                            authorization={
                                "source": "expired_job_artifact_audit",
                                "mode": recovery_mode,
                                "attempt": 1,
                                "fingerprint": "lease-expiry:" + str(job.version),
                            },
                            evidence=resume_evidence,
                        )
            except BlockingIOError:
                # A live process still owns this scope. Do not create a second
                # attempt merely because its control heartbeat was delayed.
                continue
            if not expired:
                continue
            recovered += 1
        recover_stages = getattr(self.jobs, "fail_abandoned_stages", None)
        recovered_stages = int(recover_stages()) if callable(recover_stages) else 0
        if recovered_stages and self.diagnostic_log is not None:
            self.diagnostic_log.record(
                {
                    "event": "abandoned_stages_recovered",
                    "severity": "warning",
                    "component": "service_worker",
                    "operation": "recover_abandoned_jobs",
                    "worker_id": self.worker_id,
                    "process_id": os.getpid(),
                    "thread_name": threading.current_thread().name,
                    "recovered_stage_count": recovered_stages,
                }
            )
        return recovered

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._control_db_operation(self.recover_abandoned_jobs)
        self._control_db_operation(self.recover_quarantined_scopes)
        self._stop.clear()
        with self._state_lock:
            self._active_jobs.clear()
            self._active_job_lanes.clear()
            self._claimed_quarantine_recovery_jobs.clear()
            self._scope_counts.clear()
            self._scope_lane_counts.clear()
            self.active_job_id = None
        self._thread = threading.Thread(
            target=self._run, name=self.worker_id, daemon=False
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def status(self) -> WorkerStatus:
        with self._state_lock:
            active_job_id = self.active_job_id
        return WorkerStatus(
            worker_id=self.worker_id,
            alive=bool(self._thread and self._thread.is_alive()),
            active_job_id=active_job_id,
            started_at=self.started_at,
        )

    def _run(self) -> None:
        concurrency = self._worker_concurrency()
        executor = ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix=f"{self.worker_id}-job",
        )
        self._executor = executor
        last_recovery = 0.0
        last_quarantine_recovery = 0.0
        last_scheduler = 0.0
        try:
            while not self._stop.is_set():
                done = {future for future in self._futures if future.done()}
                self._futures.difference_update(done)
                for future in done:
                    try:
                        future.result()
                    except Exception as exc:
                        self._record_exception(
                            exc,
                            operation="worker_future_failed",
                        )
                        traceback.print_exc()

                now = time.monotonic()
                if now - last_recovery >= max(5.0, self.jobs.lease_seconds / 2.0):
                    self._control_db_operation(self.recover_abandoned_jobs)
                    last_recovery = now
                if now - last_quarantine_recovery >= self._quarantine_recovery_interval():
                    self._control_db_operation(self.recover_quarantined_scopes)
                    last_quarantine_recovery = now
                if now - last_scheduler >= self._scheduler_interval():
                    self._control_db_operation(self._schedule_due_jobs)
                    last_scheduler = now

                while not self._stop.is_set() and len(self._futures) < concurrency:
                    job = self._control_db_operation(self._claim_next)
                    if job is None:
                        break
                    self._mark_active(job)
                    heartbeat_stop, heartbeat = self._start_heartbeat(job)
                    self._futures.add(
                        executor.submit(
                            self._run_job, job, heartbeat_stop, heartbeat
                        )
                    )

                if self._futures:
                    done, _ = wait(
                        tuple(self._futures),
                        timeout=self.poll_seconds,
                        return_when=FIRST_COMPLETED,
                    )
                    self._futures.difference_update(done)
                    for future in done:
                        try:
                            future.result()
                        except Exception as exc:
                            self._record_exception(
                                exc,
                                operation="worker_future_failed",
                            )
                            traceback.print_exc()
                else:
                    self._stop.wait(self.poll_seconds)
        finally:
            executor.shutdown(wait=True)
            self._executor = None

    def _start_heartbeat(
        self, job: Job
    ) -> tuple[threading.Event, threading.Thread]:
        heartbeat_stop = threading.Event()
        attempt_version = getattr(job, "version", None)

        def keep_lease() -> None:
            # Keep every claimed job leased, including jobs waiting for a
            # same-scope predecessor in a compatibility fallback.
            interval = max(0.1, min(30.0, self.jobs.lease_seconds / 3.0))
            while not heartbeat_stop.wait(interval):
                try:
                    heartbeat_kwargs = (
                        {"job_version": attempt_version}
                        if attempt_version is not None
                        else {}
                    )
                    if not self.jobs.heartbeat(
                        job.job_id, self.worker_id, **heartbeat_kwargs
                    ):
                        return
                except Exception:
                    traceback.print_exc()

        heartbeat = threading.Thread(
            target=keep_lease,
            name=f"{self.worker_id}-lease-{job.job_id}",
            daemon=True,
        )
        heartbeat.start()
        return heartbeat_stop, heartbeat

    def _run_job(
        self,
        job: Job,
        heartbeat_stop: threading.Event,
        heartbeat: threading.Thread,
    ) -> None:
        try:
            with self._scope_execution_lock(self._job_lock_key(job)):
                assert_attempt = getattr(self.jobs, "assert_running_attempt", None)
                try:
                    if callable(assert_attempt):
                        assert_attempt(job.job_id, self.worker_id, job.version)
                    result = self._execute(job)
                    if callable(assert_attempt):
                        assert_attempt(job.job_id, self.worker_id, job.version)
                except Exception as exc:
                    self._record_exception(
                        exc,
                        operation="job_failed",
                        job=job,
                    )
                    error = json.dumps(
                        {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(limit=20),
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    try:
                        fail_kwargs: dict[str, Any] = {"worker_id": self.worker_id}
                        if hasattr(job, "version"):
                            fail_kwargs["job_version"] = job.version
                        failed = self.jobs.fail(job.job_id, error, **fail_kwargs)
                        if self.commercial is not None:
                            payload = dict(job.payload or {})
                            job_type = str(payload.get("job_type") or "")
                            if job_type == "export_scope":
                                self.commercial.fail_export(
                                    str(
                                        payload.get("export_id") or ""
                                    )
                                )
                            elif job_type in {"delete_memories", "delete_session"}:
                                try:
                                    self.commercial.update_content_deletion(
                                        job.tenant_id,
                                        job.scope_name,
                                        str(payload.get("deletion_id") or ""),
                                        job.job_id,
                                        state="failed",
                                        error_code=f"{type(exc).__name__}:{exc}"[:500],
                                    )
                                except Exception:
                                    traceback.print_exc()
                            self.commercial.enqueue_job_events(failed)
                    except JobStateError:
                        pass
                else:
                    try:
                        succeed_kwargs: dict[str, Any] = {
                            "worker_id": self.worker_id
                        }
                        if hasattr(job, "version"):
                            succeed_kwargs["job_version"] = job.version
                        succeeded = self.jobs.succeed(
                            job.job_id, result, **succeed_kwargs
                        )
                        if self.commercial is not None:
                            self.commercial.enqueue_job_events(succeeded)
                    except JobStateError:
                        pass
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=2.0)
            self._unmark_active(job)

    def _execute(self, job: Job) -> Mapping[str, Any]:
        payload = dict(job.payload or {})
        usage_attribution = UsageAttribution.from_mapping(
            payload.get("_usage_attribution")
            if isinstance(payload.get("_usage_attribution"), Mapping)
            else None
        )
        job_type = str(payload.get("job_type") or "")
        scope_name = job.scope_name
        quarantine_recovery_ingest = self._is_quarantine_recovery_ingest(job)
        if self.commercial is not None and job_type not in {
            "delete_scope",
            "delete_memories",
            "delete_session",
        }:
            is_derived_recovery = bool(
                job_type != "ingest"
                and self.commercial.is_quarantine_recovery_job(
                    job.tenant_id, scope_name, job.job_id
                )
            )
            if not quarantine_recovery_ingest and not is_derived_recovery:
                self.commercial.require_scope_active(job.tenant_id, scope_name)
        if job_type == "ingest":
            def execute_writer(
                stage_id: str, stage_attempt: int
            ) -> Mapping[str, Any]:
                try:
                    writer_lease = (
                        self.gpu_scheduler.lease(GpuWorkload.WRITER_FOREGROUND)
                        if self.gpu_scheduler is not None
                        and self.writer_uses_local_gpu
                        else nullcontext()
                    )
                    with writer_lease:
                        result = dict(
                            self.storage.ingest(
                                tenant_id=job.tenant_id,
                                scope_name=scope_name,
                                session_id=str(payload["session_id"]),
                                messages=list(payload["messages"]),
                                job_id=job.job_id,
                                stage_id=stage_id,
                                stage_attempt=stage_attempt,
                                usage_attribution=usage_attribution,
                                provider_execution=(
                                    payload.get("_provider_execution")
                                    if isinstance(
                                        payload.get("_provider_execution"), Mapping
                                    )
                                    else None
                                ),
                            )
                        )
                except Exception as writer_error:
                    try:
                        self._recover_ingest_source_accounting(
                            tenant_id=job.tenant_id,
                            scope_name=scope_name,
                            session_id=str(payload["session_id"]),
                            job_id=job.job_id,
                            messages=list(payload["messages"]),
                            default_accounting_operation_id=(
                                f"{stage_id}:attempt:{stage_attempt}"
                            ),
                        )
                    except Exception as accounting_error:
                        raise RuntimeErrorBase(
                            "writer failed and its durable Source boundary could not "
                            f"be reconciled: {type(accounting_error).__name__}:"
                            f"{accounting_error}"
                        ) from writer_error
                    raise
                accounting_operation_id = self._writer_accounting_operation_id(
                    result,
                    expected_stage_id=stage_id,
                    current_stage_attempt=stage_attempt,
                )
                incomplete = (
                    result.get("input_complete") is False
                    or result.get("degraded") is True
                    or str(result.get("status") or "").strip().lower()
                    == "degraded"
                )
                if incomplete:
                    raise IncompleteWriterStage(
                        result,
                        accounting_operation_id=accounting_operation_id,
                    )
                return result

            try:
                writer = self._run_stage(
                    job,
                    "writer",
                    0,
                    execute_writer,
                )
            except IncompleteWriterStage as exc:
                prepared_writer, accounting_operation_id = (
                    self._prepare_ingest_source_accounting(
                        tenant_id=job.tenant_id,
                        scope_name=scope_name,
                        session_id=str(payload["session_id"]),
                        job_id=job.job_id,
                        messages=list(payload["messages"]),
                        writer=exc.report,
                        default_accounting_operation_id=(
                            exc.accounting_operation_id
                        ),
                    )
                )
                self._record_ingest_source(
                    tenant_id=job.tenant_id,
                    scope_name=scope_name,
                    operation_id=accounting_operation_id,
                    writer=prepared_writer,
                    messages=list(payload["messages"]),
                )
                raise RuntimeErrorBase(str(exc)) from exc
            if not isinstance(writer, Mapping):
                raise RuntimeErrorBase("writer result must be an object")
            stage = self.jobs.get_stage(f"{job.job_id}:writer")
            if stage is None or stage.attempt <= 0:
                raise RuntimeErrorBase("writer stage attempt is unavailable")
            accounting_operation_id = self._writer_accounting_operation_id(
                writer,
                expected_stage_id=stage.stage_id,
                current_stage_attempt=stage.attempt,
            )
            writer, accounting_operation_id = self._prepare_ingest_source_accounting(
                tenant_id=job.tenant_id,
                scope_name=scope_name,
                session_id=str(payload["session_id"]),
                job_id=job.job_id,
                messages=list(payload["messages"]),
                writer=writer,
                default_accounting_operation_id=accounting_operation_id,
            )
            state = self._record_ingest_source(
                job.tenant_id,
                scope_name,
                accounting_operation_id,
                writer,
                list(payload["messages"]),
            )
            slow = None
            policy = str(payload.get("slow_policy") or "auto")
            if policy == "force":
                slow, index, watermarks = self._run_consolidation(job)
            else:
                if quarantine_recovery_ingest:
                    index_result = self._run_stage(
                        job,
                        "delta_index",
                        1,
                        lambda _stage_id, _stage_attempt: {
                            "index": {
                                "deferred": True,
                                "reason": "scope_recovery_coalesced_index",
                                "target_source_event_seq": int(
                                    state.get("source_event_seq", 0) or 0
                                ),
                            },
                            "watermarks": dict(state),
                        },
                    )
                elif not self._has_active_generation(job.tenant_id, scope_name):
                    index_result = self._run_index(
                        job,
                        target_event_seq=int(state.get("source_event_seq", 0) or 0),
                    )
                else:
                    index_result = self._run_delta_index(
                        job,
                        target_event_seq=int(state.get("source_event_seq", 0) or 0),
                    )
                index = index_result["index"]
                watermarks = index_result["watermarks"]
            if self.on_ingest_committed is not None:
                self.on_ingest_committed(
                    job.tenant_id,
                    scope_name,
                    str(payload["session_id"]),
                    int(state.get("source_event_seq", 0) or 0),
                )
            return {
                "job_type": job_type,
                "writer": writer,
                "slow": slow,
                "index": index,
                "watermarks": watermarks,
            }
        if job_type == "consolidate":
            slow, index, watermarks = self._run_consolidation(job)
            return {
                "job_type": job_type,
                "slow": slow,
                "index": index,
                "watermarks": watermarks,
            }
        if job_type == "reindex":
            state = self._state(job.tenant_id, scope_name) or {}
            return {
                "job_type": job_type,
                **self._run_index(
                    job,
                    target_event_seq=int(state.get("source_event_seq", 0) or 0),
                ),
            }
        if job_type == "export_scope":
            if self.commercial is None:
                raise UnsupportedJob("commercial export control is unavailable")
            export_id = str(payload["export_id"])
            expires_at = float(payload["expires_at"])
            self.commercial.ensure_export(
                export_id,
                job.tenant_id,
                scope_name,
                job.job_id,
                expires_at,
            )
            result = self.storage.export_scope(
                tenant_id=job.tenant_id,
                scope_name=scope_name,
                export_id=export_id,
                job_id=job.job_id,
                expires_at=expires_at,
            )
            self.commercial.complete_export(
                export_id,
                artifact_path=Path(str(result["artifact_path"])),
                artifact_sha256=str(result["artifact_sha256"]),
                size_bytes=int(result["size_bytes"]),
            )
            return {
                "job_type": job_type,
                "export_id": export_id,
                "expires_at": expires_at,
                "artifact_sha256": str(result["artifact_sha256"]),
                "size_bytes": int(result["size_bytes"]),
            }
        if job_type == "delete_scope":
            if self.commercial is None:
                raise UnsupportedJob("commercial deletion control is unavailable")
            self.commercial.mark_scope_deleting(
                job.tenant_id,
                scope_name,
                job.job_id,
                reason=str(payload.get("reason") or "api_request"),
            )
            result = self.storage.delete_scope(
                tenant_id=job.tenant_id,
                scope_name=scope_name,
                job_id=job.job_id,
            )
            self.commercial.complete_scope_deletion(
                job.tenant_id,
                scope_name,
                job.job_id,
                scope_id=str(result["scope_id"]),
            )
            return {
                "job_type": job_type,
                "scope_name": scope_name,
                "deleted": True,
                "scope_removed": bool(result["scope_removed"]),
                "exports_removed": bool(result["exports_removed"]),
            }
        if job_type in {"delete_memories", "delete_session"}:
            if self.commercial is None:
                raise UnsupportedJob("commercial deletion control is unavailable")
            deletion_id = str(payload.get("deletion_id") or "")
            self.commercial.update_content_deletion(
                job.tenant_id,
                scope_name,
                deletion_id,
                job.job_id,
                state="purging",
            )
            commit = self.storage.content_deletion_commit(
                tenant_id=job.tenant_id,
                scope_name=scope_name,
                job_id=job.job_id,
            )
            if commit is None:
                purge = self.storage.delete_memories(
                    tenant_id=job.tenant_id,
                    scope_name=scope_name,
                    job_id=job.job_id,
                    memory_ids=(
                        list(payload.get("memory_ids") or [])
                        if job_type == "delete_memories"
                        else None
                    ),
                    session_id=(
                        str(payload.get("session_id") or "")
                        if job_type == "delete_session"
                        else None
                    ),
                )
            else:
                committed_result = dict(commit.get("result") or {})
                purge = {
                    "scope_id": self.storage.scope_paths(
                        job.tenant_id, scope_name
                    ).scope_id,
                    "mode": str(commit.get("mode") or ""),
                    "requested_memory_count": len(
                        list(payload.get("memory_ids") or [])
                    ),
                    "matched_source_memory_count": len(
                        list(committed_result.get("deleted_source_record_ids") or [])
                    ),
                    "deleted_memory_count": int(
                        committed_result.get("deleted_memory_count", 0) or 0
                    ),
                    "deleted_message_count": int(
                        committed_result.get("deleted_message_count", 0) or 0
                    ),
                    "invalidated_slow_memory_count": int(
                        committed_result.get("invalidated_slow_memory_count", 0)
                        or 0
                    ),
                    "slow_rebuild_required": bool(
                        committed_result.get("invalidated_slow_memory_count", 0)
                    ),
                    "job_id": job.job_id,
                    "_deleted_source_record_ids": list(
                        committed_result.get("deleted_source_record_ids") or []
                    ),
                    "_deleted_message_ids": [],
                    "_deleted_session_message_counts": dict(
                        committed_result.get("deleted_session_message_counts") or {}
                    ),
                    "resumed_from_deletion_commit": True,
                }
            self.commercial.apply_content_deletion_control_cleanup(
                job.tenant_id,
                scope_name,
                deleted_source_record_ids=list(
                    purge.pop("_deleted_source_record_ids", []) or []
                ),
                deleted_session_message_counts=dict(
                    purge.pop("_deleted_session_message_counts", {}) or {}
                ),
                deleted_session_id=(
                    str(payload.get("session_id") or "")
                    if job_type == "delete_session"
                    else None
                ),
            )
            purge.pop("_deleted_message_ids", None)
            self.commercial.update_content_deletion(
                job.tenant_id,
                scope_name,
                deletion_id,
                job.job_id,
                state="reindexing",
                result=purge,
            )
            state = self._state(job.tenant_id, scope_name) or {}
            index_result = self._run_index(
                job,
                target_event_seq=int(state.get("source_event_seq", 0) or 0),
            )
            result = {
                "job_type": job_type,
                "deletion_id": deletion_id,
                "scope_name": scope_name,
                "deleted": True,
                **purge,
                "index": index_result["index"],
                "watermarks": index_result["watermarks"],
            }
            self.commercial.update_content_deletion(
                job.tenant_id,
                scope_name,
                deletion_id,
                job.job_id,
                state="completed",
                result=result,
            )
            return result
        raise UnsupportedJob(f"unsupported job type: {job_type}")
