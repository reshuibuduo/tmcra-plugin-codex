from __future__ import annotations

import math
import os
import shutil
import subprocess
import threading
import time
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Iterator, Mapping


class GpuWorkload(str, Enum):
    RECALL_FOREGROUND = "recall_foreground"
    WRITER_FOREGROUND = "writer_foreground"
    PLANNER_FOREGROUND = "planner_foreground"
    INDEX_FOREGROUND = "index_foreground"
    INDEX_BACKGROUND = "index_background"
    GRAPH_BACKGROUND = "graph_background"
    GRAPH_BORROWED_PLANNER = "graph_borrowed_planner"


FOREGROUND_WORKLOADS = frozenset(
    {
        GpuWorkload.RECALL_FOREGROUND,
        GpuWorkload.WRITER_FOREGROUND,
        GpuWorkload.PLANNER_FOREGROUND,
        GpuWorkload.INDEX_FOREGROUND,
    }
)
BACKGROUND_WORKLOADS = frozenset(
    {
        GpuWorkload.INDEX_BACKGROUND,
        GpuWorkload.GRAPH_BACKGROUND,
        GpuWorkload.GRAPH_BORROWED_PLANNER,
    }
)
RECALL_LANE_WORKLOADS = frozenset(
    {
        GpuWorkload.RECALL_FOREGROUND,
        GpuWorkload.INDEX_FOREGROUND,
        GpuWorkload.INDEX_BACKGROUND,
    }
)

_PRIORITY = {
    GpuWorkload.RECALL_FOREGROUND: 0,
    GpuWorkload.PLANNER_FOREGROUND: 1,
    GpuWorkload.WRITER_FOREGROUND: 2,
    GpuWorkload.INDEX_FOREGROUND: 3,
    GpuWorkload.INDEX_BACKGROUND: 10,
    GpuWorkload.GRAPH_BACKGROUND: 20,
    GpuWorkload.GRAPH_BORROWED_PLANNER: 20,
}


class GpuSchedulerError(RuntimeError):
    pass


class GpuSchedulerClosedError(GpuSchedulerError):
    pass


class GpuSchedulerTimeoutError(GpuSchedulerError):
    def __init__(self, workload: GpuWorkload, waited_seconds: float) -> None:
        self.workload = workload
        self.waited_seconds = max(0.0, float(waited_seconds))
        super().__init__(
            f"timed out after {self.waited_seconds:.3f}s waiting for {workload.value}"
        )


@dataclass(frozen=True)
class GpuTelemetry:
    sampled_at: float
    utilization_percent: float
    memory_used_bytes: int
    memory_free_bytes: int
    power_watts: float | None = None

    def as_dict(self, *, now: float) -> dict[str, Any]:
        return {
            "available": True,
            "sample_age_seconds": round(max(0.0, now - self.sampled_at), 3),
            "utilization_percent": round(self.utilization_percent, 3),
            "memory_used_bytes": self.memory_used_bytes,
            "memory_free_bytes": self.memory_free_bytes,
            "power_watts": (
                round(self.power_watts, 3) if self.power_watts is not None else None
            ),
        }


class NvidiaSmiTelemetryProbe:
    """Read one bounded aggregate GPU sample without importing CUDA libraries."""

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("nvidia-smi") or "nvidia-smi"

    def __call__(self) -> Mapping[str, Any]:
        completed = subprocess.run(
            [
                self.executable,
                "--query-gpu=utilization.gpu,memory.used,memory.free,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not rows:
            raise RuntimeError("nvidia-smi returned no GPU rows")
        parsed: list[tuple[float, float, float, float | None]] = []
        for row in rows:
            fields = [field.strip() for field in row.split(",")]
            if len(fields) != 4:
                raise RuntimeError("nvidia-smi returned an unexpected row")
            power = None if fields[3] in {"", "N/A", "[N/A]"} else float(fields[3])
            parsed.append((float(fields[0]), float(fields[1]), float(fields[2]), power))
        used_mib = sum(item[1] for item in parsed)
        free_mib = sum(item[2] for item in parsed)
        powers = [item[3] for item in parsed if item[3] is not None]
        return {
            "utilization_percent": sum(item[0] for item in parsed) / len(parsed),
            "memory_used_bytes": int(used_mib * 1024**2),
            "memory_free_bytes": int(free_mib * 1024**2),
            "power_watts": sum(powers) if powers else None,
        }


@dataclass
class _Waiter:
    sequence: int
    workload: GpuWorkload
    enqueued_at: float


@dataclass
class _WorkloadMetrics:
    started: int = 0
    completed: int = 0
    failed: int = 0
    timed_out: int = 0
    total_wait_seconds: float = 0.0
    max_wait_seconds: float = 0.0
    total_runtime_seconds: float = 0.0
    max_runtime_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "total_wait_seconds",
            "max_wait_seconds",
            "total_runtime_seconds",
            "max_runtime_seconds",
        ):
            value[key] = round(float(value[key]), 6)
        value["average_wait_seconds"] = round(
            self.total_wait_seconds / self.started if self.started else 0.0, 6
        )
        finished = self.completed + self.failed
        value["average_runtime_seconds"] = round(
            self.total_runtime_seconds / finished if finished else 0.0, 6
        )
        return value


class GpuLease:
    def __init__(
        self,
        scheduler: "GpuWorkloadScheduler",
        workload: GpuWorkload,
        *,
        started_at: float,
        wait_seconds: float,
    ) -> None:
        self.scheduler = scheduler
        self.workload = workload
        self.started_at = started_at
        self.wait_seconds = wait_seconds
        self._released = False
        self._lock = threading.Lock()

    def release(self, error: BaseException | None = None) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self.scheduler._release(self, error=error)

    def __enter__(self) -> "GpuLease":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        self.release(error=exc)
        return False


class GpuWorkloadScheduler:
    """Coordinate TMCRA GPU consumers with work-conserving admission.

    The scheduler controls task starts. CUDA kernels already running cannot be
    pre-empted safely, so every workload still respects its physical resource
    capacity.  Available lanes are filled while telemetry remains below the
    configured saturation and memory-safety boundaries. Priority orders waiters
    only after a real resource conflict exists; foreground activity by itself is
    not a reason to leave a different GPU lane idle. The recall pool remains the
    final owner of its resident model replicas.
    """

    def __init__(
        self,
        *,
        recall_capacity: int,
        safety_free_bytes: int = 1024**3,
        background_utilization_limit: float = 70.0,
        background_overlap_utilization_limit: float = 35.0,
        foreground_quiet_seconds: float = 0.5,
        borrowed_slot_quiet_seconds: float = 30.0,
        telemetry_interval_seconds: float = 1.0,
        telemetry_stale_seconds: float = 5.0,
        telemetry_probe: Callable[[], Mapping[str, Any]] | None = None,
        dedicated_graph_slot: bool = False,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(recall_capacity, bool) or recall_capacity <= 0:
            raise ValueError("recall_capacity must be positive")
        if isinstance(safety_free_bytes, bool) or safety_free_bytes < 0:
            raise ValueError("safety_free_bytes must be non-negative")
        for name, value in (
            ("background_utilization_limit", background_utilization_limit),
            (
                "background_overlap_utilization_limit",
                background_overlap_utilization_limit,
            ),
        ):
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 100:
                raise ValueError(f"{name} must be between 0 and 100")
        for name, value, allow_zero in (
            ("foreground_quiet_seconds", foreground_quiet_seconds, True),
            ("borrowed_slot_quiet_seconds", borrowed_slot_quiet_seconds, True),
            ("telemetry_interval_seconds", telemetry_interval_seconds, False),
            ("telemetry_stale_seconds", telemetry_stale_seconds, False),
        ):
            number = float(value)
            if not math.isfinite(number) or number < 0 or (not allow_zero and number == 0):
                raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if telemetry_probe is not None and not callable(telemetry_probe):
            raise TypeError("telemetry_probe must be callable")

        self.recall_capacity = int(recall_capacity)
        self.safety_free_bytes = int(safety_free_bytes)
        self.background_utilization_limit = float(background_utilization_limit)
        self.background_overlap_utilization_limit = float(
            background_overlap_utilization_limit
        )
        self.foreground_quiet_seconds = float(foreground_quiet_seconds)
        self.borrowed_slot_quiet_seconds = float(borrowed_slot_quiet_seconds)
        self.telemetry_interval_seconds = float(telemetry_interval_seconds)
        self.telemetry_stale_seconds = float(telemetry_stale_seconds)
        self.telemetry_probe = telemetry_probe
        self.dedicated_graph_slot = bool(dedicated_graph_slot)
        self.enabled = bool(enabled)
        self._clock = clock
        self._condition = threading.Condition()
        self._waiters: list[_Waiter] = []
        self._sequence = 0
        self._active: Counter[GpuWorkload] = Counter()
        self._metrics = {
            workload: _WorkloadMetrics() for workload in GpuWorkload
        }
        self._last_foreground_activity_at = float("-inf")
        self._telemetry: GpuTelemetry | None = None
        self._recent_utilization: deque[float] = deque(maxlen=5)
        self._telemetry_failures = 0
        self._telemetry_last_error_type: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        environment: Mapping[str, str] | None = None,
        telemetry_probe: Callable[[], Mapping[str, Any]] | None = None,
    ) -> "GpuWorkloadScheduler":
        env = os.environ if environment is None else environment

        def number(name: str, default: float) -> float:
            raw = str(env.get(name) or "").strip()
            return default if not raw else float(raw)

        def boolean(name: str, default: bool) -> bool:
            raw = str(env.get(name) or "").strip().casefold()
            if not raw:
                return default
            if raw in {"1", "true", "yes", "on"}:
                return True
            if raw in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"{name} must be a boolean")

        device = str(getattr(settings, "device", "cpu"))
        if telemetry_probe is None and device.startswith("cuda"):
            telemetry_probe = NvidiaSmiTelemetryProbe()
        return cls(
            recall_capacity=int(getattr(settings, "recall_pool_max_size", 1)),
            safety_free_bytes=int(
                number("TMCRA_GPU_SCHEDULER_SAFETY_FREE_BYTES", 1024**3)
            ),
            background_utilization_limit=number(
                "TMCRA_GPU_SCHEDULER_BACKGROUND_UTILIZATION_LIMIT", 70.0
            ),
            background_overlap_utilization_limit=number(
                "TMCRA_GPU_SCHEDULER_BACKGROUND_OVERLAP_UTILIZATION_LIMIT", 35.0
            ),
            foreground_quiet_seconds=number(
                "TMCRA_GPU_SCHEDULER_FOREGROUND_QUIET_SECONDS", 0.5
            ),
            borrowed_slot_quiet_seconds=number(
                "TMCRA_GPU_SCHEDULER_BORROWED_SLOT_QUIET_SECONDS", 30.0
            ),
            telemetry_interval_seconds=number(
                "TMCRA_GPU_SCHEDULER_TELEMETRY_INTERVAL_SECONDS", 1.0
            ),
            telemetry_stale_seconds=number(
                "TMCRA_GPU_SCHEDULER_TELEMETRY_STALE_SECONDS", 5.0
            ),
            telemetry_probe=telemetry_probe,
            dedicated_graph_slot=boolean(
                "TMCRA_GPU_SCHEDULER_DEDICATED_GRAPH_SLOT", False
            ),
            enabled=boolean("TMCRA_GPU_SCHEDULER_ENABLED", True),
        )

    def start(self) -> None:
        if not self.enabled or self.telemetry_probe is None:
            return
        with self._condition:
            if self._closed:
                raise GpuSchedulerClosedError("GPU workload scheduler is closed")
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._monitor,
                name="tmcra-gpu-telemetry",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))

    def _monitor(self) -> None:
        while not self._stop.is_set():
            self.sample_telemetry()
            if self._stop.wait(self.telemetry_interval_seconds):
                return

    def sample_telemetry(self) -> GpuTelemetry | None:
        probe = self.telemetry_probe
        if probe is None:
            return None
        try:
            value = probe()
            utilization = float(value["utilization_percent"])
            used = int(value["memory_used_bytes"])
            free = int(value["memory_free_bytes"])
            raw_power = value.get("power_watts")
            power = float(raw_power) if raw_power is not None else None
            if not math.isfinite(utilization) or not 0 <= utilization <= 100:
                raise ValueError("GPU utilization is invalid")
            if used < 0 or free < 0:
                raise ValueError("GPU memory counters are invalid")
            sample = GpuTelemetry(
                sampled_at=self._clock(),
                utilization_percent=utilization,
                memory_used_bytes=used,
                memory_free_bytes=free,
                power_watts=power,
            )
        except Exception as exc:
            with self._condition:
                self._telemetry_failures += 1
                self._telemetry_last_error_type = type(exc).__name__
                self._condition.notify_all()
            return None
        with self._condition:
            self._telemetry = sample
            self._recent_utilization.append(utilization)
            self._telemetry_last_error_type = None
            self._condition.notify_all()
        return sample

    @staticmethod
    def _resource(workload: GpuWorkload) -> str:
        if workload in RECALL_LANE_WORKLOADS:
            return "recall"
        if workload == GpuWorkload.WRITER_FOREGROUND:
            return "writer"
        if workload in {
            GpuWorkload.PLANNER_FOREGROUND,
            GpuWorkload.GRAPH_BORROWED_PLANNER,
        }:
            return "planner"
        return "graph"

    def _active_for_resource_locked(self, resource: str) -> int:
        return sum(
            count
            for workload, count in self._active.items()
            if self._resource(workload) == resource
        )

    def _capacity_for_resource(self, resource: str) -> int:
        return self.recall_capacity if resource == "recall" else 1

    def _foreground_waiting_locked(self) -> bool:
        return any(waiter.workload in FOREGROUND_WORKLOADS for waiter in self._waiters)

    def _foreground_active_locked(self) -> bool:
        return any(self._active[workload] > 0 for workload in FOREGROUND_WORKLOADS)

    def _telemetry_allows_background_locked(self, *, overlap: bool) -> bool:
        sample = self._telemetry
        if sample is None:
            # CPU/non-CUDA deployments intentionally omit a probe. A configured
            # CUDA probe that has not produced a valid sample fails closed so a
            # startup race or nvidia-smi failure cannot bypass the safety line.
            return self.telemetry_probe is None
        now = self._clock()
        if now - sample.sampled_at > self.telemetry_stale_seconds:
            return False
        if sample.memory_free_bytes < self.safety_free_bytes:
            return False
        limit = (
            self.background_overlap_utilization_limit
            if overlap
            else self.background_utilization_limit
        )
        recent = list(self._recent_utilization)
        if not recent:
            recent = [sample.utilization_percent]
        mean = sum(recent) / len(recent)
        return mean <= limit and max(recent) <= min(100.0, limit + 25.0)

    def _telemetry_has_memory_headroom_locked(self) -> bool:
        sample = self._telemetry
        if sample is None:
            return self.telemetry_probe is None
        now = self._clock()
        return bool(
            now - sample.sampled_at <= self.telemetry_stale_seconds
            and sample.memory_free_bytes >= self.safety_free_bytes
        )

    def _admissible_locked(self, workload: GpuWorkload) -> bool:
        if self._closed:
            return False
        if not self.enabled:
            return True
        resource = self._resource(workload)
        if self._active_for_resource_locked(resource) >= self._capacity_for_resource(
            resource
        ):
            return False
        if workload == GpuWorkload.INDEX_BACKGROUND and self._active[workload] >= 1:
            return False
        if workload not in BACKGROUND_WORKLOADS:
            return True
        foreground_overlap = (
            self._foreground_waiting_locked() or self._foreground_active_locked()
        )
        now = self._clock()
        if workload == GpuWorkload.GRAPH_BORROWED_PLANNER:
            # Slot 1 belongs to the foreground recall planner. It may be
            # borrowed only after the whole foreground has stayed quiet. A
            # newly queued planner shares this resource and therefore waits at
            # most for the current bounded projection batch. Writer remains on
            # its dedicated slot 0 and is never borrowed.
            if foreground_overlap:
                return False
            if (
                now - self._last_foreground_activity_at
                < self.borrowed_slot_quiet_seconds
            ):
                return False
            return self._telemetry_has_memory_headroom_locked()
        if (
            not self.dedicated_graph_slot
            or workload != GpuWorkload.GRAPH_BACKGROUND
        ) and now - self._last_foreground_activity_at < self.foreground_quiet_seconds:
            return False
        other_background_active = any(
            self._active[item] > 0
            for item in BACKGROUND_WORKLOADS
            if item != workload
        )
        return self._telemetry_allows_background_locked(
            overlap=foreground_overlap or other_background_active
        )

    def _has_turn_locked(self, waiter: _Waiter) -> bool:
        resource = self._resource(waiter.workload)
        eligible = [
            item
            for item in self._waiters
            if self._resource(item.workload) == resource
        ]
        if not eligible:
            return True
        first = min(eligible, key=lambda item: (_PRIORITY[item.workload], item.sequence))
        return first is waiter

    def can_start(self, workload: GpuWorkload | str) -> bool:
        resolved = GpuWorkload(workload)
        with self._condition:
            if any(
                self._resource(waiter.workload) == self._resource(resolved)
                and (
                    _PRIORITY[waiter.workload] < _PRIORITY[resolved]
                    or _PRIORITY[waiter.workload] == _PRIORITY[resolved]
                )
                for waiter in self._waiters
            ):
                return False
            return self._admissible_locked(resolved)

    def try_acquire(self, workload: GpuWorkload | str) -> GpuLease | None:
        resolved = GpuWorkload(workload)
        now = self._clock()
        with self._condition:
            if not self.can_start(resolved):
                return None
            return self._admit_locked(resolved, enqueued_at=now)

    def acquire(
        self,
        workload: GpuWorkload | str,
        *,
        timeout: float | None = None,
    ) -> GpuLease:
        resolved = GpuWorkload(workload)
        if timeout is not None and (
            not math.isfinite(float(timeout)) or float(timeout) <= 0
        ):
            raise ValueError("GPU scheduler timeout must be positive and finite")
        enqueued_at = self._clock()
        deadline = None if timeout is None else enqueued_at + float(timeout)
        with self._condition:
            if self._closed:
                raise GpuSchedulerClosedError("GPU workload scheduler is closed")
            self._sequence += 1
            waiter = _Waiter(self._sequence, resolved, enqueued_at)
            self._waiters.append(waiter)
            try:
                while True:
                    if self._closed:
                        raise GpuSchedulerClosedError(
                            "GPU workload scheduler is closed"
                        )
                    if self._has_turn_locked(waiter) and self._admissible_locked(resolved):
                        self._waiters.remove(waiter)
                        return self._admit_locked(resolved, enqueued_at=enqueued_at)
                    now = self._clock()
                    remaining = None if deadline is None else deadline - now
                    if remaining is not None and remaining <= 0:
                        self._metrics[resolved].timed_out += 1
                        raise GpuSchedulerTimeoutError(
                            resolved, max(0.0, now - enqueued_at)
                        )
                    self._condition.wait(
                        timeout=(
                            0.25 if remaining is None else min(0.25, remaining)
                        )
                    )
            finally:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()

    def _admit_locked(
        self, workload: GpuWorkload, *, enqueued_at: float
    ) -> GpuLease:
        now = self._clock()
        waited = max(0.0, now - enqueued_at)
        self._active[workload] += 1
        metrics = self._metrics[workload]
        metrics.started += 1
        metrics.total_wait_seconds += waited
        metrics.max_wait_seconds = max(metrics.max_wait_seconds, waited)
        if workload in FOREGROUND_WORKLOADS:
            self._last_foreground_activity_at = now
        self._condition.notify_all()
        return GpuLease(
            self,
            workload,
            started_at=now,
            wait_seconds=waited,
        )

    def _release(self, lease: GpuLease, *, error: BaseException | None) -> None:
        now = self._clock()
        with self._condition:
            workload = lease.workload
            if self._active[workload] <= 0:
                raise GpuSchedulerError(
                    f"released an inactive GPU workload: {workload.value}"
                )
            self._active[workload] -= 1
            metrics = self._metrics[workload]
            runtime = max(0.0, now - lease.started_at)
            metrics.total_runtime_seconds += runtime
            metrics.max_runtime_seconds = max(metrics.max_runtime_seconds, runtime)
            if error is None:
                metrics.completed += 1
            else:
                metrics.failed += 1
            if workload in FOREGROUND_WORKLOADS:
                self._last_foreground_activity_at = now
            self._condition.notify_all()

    @contextmanager
    def lease(
        self,
        workload: GpuWorkload | str,
        *,
        timeout: float | None = None,
    ) -> Iterator[GpuLease]:
        acquired = self.acquire(workload, timeout=timeout)
        with acquired:
            yield acquired

    def status(self) -> dict[str, Any]:
        now = self._clock()
        with self._condition:
            telemetry = self._telemetry
            active = {
                workload.value: int(self._active[workload])
                for workload in GpuWorkload
            }
            waiting_counts = Counter(waiter.workload for waiter in self._waiters)
            waiting = {
                workload.value: int(waiting_counts[workload])
                for workload in GpuWorkload
            }
            metrics = {
                workload.value: self._metrics[workload].as_dict()
                for workload in GpuWorkload
            }
            telemetry_value = (
                telemetry.as_dict(now=now)
                if telemetry is not None
                else {
                    "available": False,
                    "sample_age_seconds": None,
                    "utilization_percent": None,
                    "memory_used_bytes": None,
                    "memory_free_bytes": None,
                    "power_watts": None,
                }
            )
            recent = list(self._recent_utilization)
            telemetry_value["recent_mean_utilization_percent"] = (
                round(sum(recent) / len(recent), 3) if recent else None
            )
            telemetry_value["recent_max_utilization_percent"] = (
                round(max(recent), 3) if recent else None
            )
            telemetry_value["failures"] = self._telemetry_failures
            telemetry_value["last_error_type"] = self._telemetry_last_error_type
            return {
                "schema_version": "tmcra.gpu-scheduler.1",
                "enabled": self.enabled,
                "dedicated_graph_slot": self.dedicated_graph_slot,
                "closed": self._closed,
                "monitor_alive": bool(self._thread and self._thread.is_alive()),
                "recall_capacity": self.recall_capacity,
                "safety_free_bytes": self.safety_free_bytes,
                "background_utilization_limit": self.background_utilization_limit,
                "background_overlap_utilization_limit": (
                    self.background_overlap_utilization_limit
                ),
                "foreground_quiet_seconds": self.foreground_quiet_seconds,
                "borrowed_slot_quiet_seconds": self.borrowed_slot_quiet_seconds,
                "foreground_waiting": self._foreground_waiting_locked(),
                "foreground_active": self._foreground_active_locked(),
                "active": active,
                "waiting": waiting,
                "metrics": metrics,
                "telemetry": telemetry_value,
            }
