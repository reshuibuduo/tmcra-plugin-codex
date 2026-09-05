from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Deque, Generic, Mapping, TypeVar, cast


EngineT = TypeVar("EngineT")
ResultT = TypeVar("ResultT")


class RecallPoolError(RuntimeError):
    """Base class for errors raised by the recall scheduler itself."""


class RecallPoolAdmissionError(RecallPoolError):
    """Base class for failures that happen before recall execution starts."""

    def __init__(self, message: str, *, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = float(retry_after)

    @property
    def retry_after_seconds(self) -> float:
        """Alias suitable for APIs that use an explicit unit in field names."""

        return self.retry_after


class RecallPoolSaturatedError(RecallPoolAdmissionError):
    """The global or per-tenant pending queue has reached its hard limit."""

    def __init__(self, *, scope: str, retry_after: float) -> None:
        if scope not in {"global", "tenant"}:
            raise ValueError("saturation scope must be 'global' or 'tenant'")
        self.scope = scope
        super().__init__(
            f"recall pool {scope} pending queue is saturated",
            retry_after=retry_after,
        )


class RecallPoolTimeoutError(RecallPoolAdmissionError):
    """A request left the pending queue before an engine became available."""

    def __init__(self, *, waited: float, retry_after: float) -> None:
        self.waited = max(0.0, float(waited))
        super().__init__(
            f"timed out after {self.waited:.3f}s waiting for a recall engine",
            retry_after=retry_after,
        )


class RecallPoolClosedError(RecallPoolAdmissionError):
    """The pool is shutting down and cannot accept more work."""

    def __init__(self, *, retry_after: float) -> None:
        super().__init__("recall pool is closed", retry_after=retry_after)


class RecallPoolWarmupError(RecallPoolError):
    """At least one engine replica failed its warmup probe."""

    def __init__(
        self,
        *,
        failures: tuple[tuple[int, Exception], ...],
        results: tuple[Any | None, ...],
    ) -> None:
        self.failures = failures
        self.results = results
        indexes = ", ".join(str(index) for index, _error in failures)
        super().__init__(f"recall engine warmup failed for replica(s): {indexes}")


# Short aliases keep endpoint integration readable while retaining explicit
# canonical exception names for logs and introspection.
RecallPoolSaturated = RecallPoolSaturatedError
RecallPoolTimeout = RecallPoolTimeoutError


@dataclass(frozen=True)
class RecallPoolStatus:
    configured: int
    min_size: int
    max_size: int
    current_size: int
    desired_size: int
    loaded: int
    fully_loaded: bool
    active: int
    retiring: int
    idle: int
    pending: int
    pending_tenants: int
    max_pending: int
    per_tenant_pending: int
    warming: bool
    scaling: bool
    scaling_direction: str | None
    replacement_pending: bool
    repair_target_size: int
    closed: bool
    last_scale_error: str | None

    def as_dict(self) -> dict[str, int | bool | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class RecallPoolMetrics:
    submitted: int
    started: int
    completed: int
    failed: int
    saturated: int
    timed_out: int
    engine_load_failures: int
    warmup_runs: int
    warmup_failures: int
    scale_successes: int
    scale_failures: int
    scale_up_successes: int
    scale_up_failures: int
    scale_down_successes: int
    scale_down_failures: int
    fatal_operation_failures: int
    quarantined_replicas: int
    replacement_attempts: int
    replacement_successes: int
    replacement_failures: int
    current_size: int
    desired_size: int
    active: int
    pending: int
    peak_active: int
    peak_pending: int
    arrival_rate_ewma: float
    service_time_ewma_seconds: float
    offered_load: float
    utilization: float
    target_utilization: float
    total_queue_wait_seconds: float
    average_queue_wait_seconds: float
    max_queue_wait_seconds: float
    total_execution_seconds: float
    average_execution_seconds: float
    max_execution_seconds: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


_UNSET = object()


class _EngineSlot(Generic[EngineT]):
    """A lazy replica with a lock that serializes every engine operation."""

    def __init__(
        self,
        index: int,
        factory: Callable[[], EngineT],
        close_callback: Callable[[EngineT], None] | None,
    ) -> None:
        self.index = index
        self._factory = factory
        self._close_callback = close_callback
        self._engine: EngineT | object = _UNSET
        self._initialization_lock = threading.Lock()
        self._operation_lock = threading.Lock()
        self._loaded = threading.Event()
        self._load_failures = 0

    @property
    def loaded(self) -> bool:
        # Health/status probes must not wait behind a potentially expensive
        # model constructor. Event state is safe to read across threads.
        return self._loaded.is_set()

    @property
    def load_failures(self) -> int:
        with self._initialization_lock:
            return self._load_failures

    def _get(self) -> EngineT:
        with self._initialization_lock:
            if self._engine is _UNSET:
                try:
                    engine = self._factory()
                except Exception:
                    self._load_failures += 1
                    raise
                if engine is None:
                    self._load_failures += 1
                    raise RecallPoolError("recall engine factory returned None")
                self._engine = engine
                self._loaded.set()
            return cast(EngineT, self._engine)

    def run(self, operation: Callable[[EngineT], ResultT]) -> ResultT:
        # The pool already leases a slot to only one caller at a time. This
        # second boundary deliberately keeps the engine serialized even if a
        # future maintenance path invokes the slot outside normal scheduling.
        with self._operation_lock:
            return operation(self._get())

    def close(self) -> None:
        """Close a constructed replica without ever constructing a lazy one."""

        with self._operation_lock:
            with self._initialization_lock:
                if self._engine is _UNSET:
                    return
                engine = cast(EngineT, self._engine)
                # Publish retirement before invoking user code. Even a failing
                # closer cannot make this engine eligible for reuse.
                self._engine = _UNSET
                self._loaded.clear()
            if self._close_callback is not None:
                self._close_callback(engine)
                return
            method = getattr(engine, "close", None)
            if method is not None and callable(method):
                method()


@dataclass
class _Waiter(Generic[EngineT]):
    tenant_id: str
    enqueued_at: float
    slot: _EngineSlot[EngineT] | None = None


class RecallEnginePool(Generic[EngineT]):
    """Bounded, tenant-fair scheduler for serialized recall engine replicas.

    ``execute`` is deliberately synchronous so it can run unchanged inside a
    Starlette/FastAPI threadpool. Initial engines become schedulable only after
    explicit startup ``warmup``; elastic replicas are built and warmed only by
    the background autoscaler. A user request never owns model cold start.
    """

    def __init__(
        self,
        replica_factory: Callable[[], EngineT],
        *,
        size: int | None = None,
        min_size: int | None = None,
        max_size: int | None = None,
        max_pending: int = 8,
        per_tenant_pending: int = 2,
        queue_timeout: float = 10.0,
        retry_after: float = 1.0,
        capacity_guard: Callable[[int, int], bool] | None = None,
        close_callback: Callable[[EngineT], None] | None = None,
        fatal_exception_predicate: Callable[[BaseException], bool] | None = None,
        warmup_snapshots: Any | None = None,
        target_utilization: float = 0.70,
        warm_spares: int = 1,
        scale_up_sustain_seconds: float = 0.25,
        scale_down_idle_seconds: float = 600.0,
        scaling_cooldown_seconds: float | None = None,
        scale_up_cooldown_seconds: float | None = None,
        scale_down_cooldown_seconds: float | None = None,
        monitor_interval_seconds: float = 0.25,
        ewma_alpha: float = 0.20,
        arrival_decay_seconds: float = 10.0,
        forward_tenant_as: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(replica_factory):
            raise TypeError("replica_factory must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if forward_tenant_as is not None and (
            not isinstance(forward_tenant_as, str)
            or not forward_tenant_as.isidentifier()
        ):
            raise ValueError("forward_tenant_as must be a valid Python identifier")
        if size is not None:
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("size must be a positive integer")
            if min_size is not None and min_size != size:
                raise ValueError("size and min_size disagree")
            if max_size is not None and max_size != size:
                raise ValueError("size and max_size disagree")
            min_size = max_size = size
        if min_size is None:
            min_size = 2
        if max_size is None:
            max_size = min_size
        for name, value in (("min_size", min_size), ("max_size", max_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_size < min_size:
            raise ValueError("max_size must be greater than or equal to min_size")
        for name, value in (
            ("max_pending", max_pending),
            ("per_tenant_pending", per_tenant_pending),
            ("warm_spares", warm_spares),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if capacity_guard is not None and not callable(capacity_guard):
            raise TypeError("capacity_guard must be callable")
        if close_callback is not None and not callable(close_callback):
            raise TypeError("close_callback must be callable")
        if fatal_exception_predicate is not None and not callable(
            fatal_exception_predicate
        ):
            raise TypeError("fatal_exception_predicate must be callable")
        if not math.isfinite(float(queue_timeout)) or queue_timeout <= 0:
            raise ValueError("queue_timeout must be a positive finite number")
        if not math.isfinite(float(retry_after)) or retry_after <= 0:
            raise ValueError("retry_after must be a positive finite number")
        if (
            not math.isfinite(float(target_utilization))
            or target_utilization <= 0
            or target_utilization > 1
        ):
            raise ValueError("target_utilization must be in the interval (0, 1]")
        for name, value, allow_zero in (
            ("scale_up_sustain_seconds", scale_up_sustain_seconds, True),
            ("scale_down_idle_seconds", scale_down_idle_seconds, False),
            ("monitor_interval_seconds", monitor_interval_seconds, False),
            ("arrival_decay_seconds", arrival_decay_seconds, False),
        ):
            number = float(value)
            if not math.isfinite(number) or number < 0 or (not allow_zero and number == 0):
                raise ValueError(f"{name} must be a {'non-negative' if allow_zero else 'positive'} finite number")
        if not math.isfinite(float(ewma_alpha)) or not 0 < ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in the interval (0, 1]")

        # ``scaling_cooldown_seconds`` was the original single-knob API.  Keep
        # it as a compatibility default while allowing production to tune
        # scale-up and scale-down independently.
        legacy_cooldown = (
            30.0 if scaling_cooldown_seconds is None else scaling_cooldown_seconds
        )
        if (
            not math.isfinite(float(legacy_cooldown))
            or float(legacy_cooldown) < 0
        ):
            raise ValueError(
                "scaling_cooldown_seconds must be a non-negative finite number"
            )
        if scale_up_cooldown_seconds is None:
            scale_up_cooldown_seconds = legacy_cooldown
        if scale_down_cooldown_seconds is None:
            scale_down_cooldown_seconds = legacy_cooldown
        for name, value in (
            ("scale_up_cooldown_seconds", scale_up_cooldown_seconds),
            ("scale_down_cooldown_seconds", scale_down_cooldown_seconds),
        ):
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{name} must be a non-negative finite number")

        self.min_size = min_size
        self.max_size = max_size
        self.max_pending = max_pending
        self.per_tenant_pending = per_tenant_pending
        self.queue_timeout = float(queue_timeout)
        self.retry_after = float(retry_after)
        self.target_utilization = float(target_utilization)
        self.warm_spares = warm_spares
        self.scale_up_sustain_seconds = float(scale_up_sustain_seconds)
        self.scale_down_idle_seconds = float(scale_down_idle_seconds)
        self.scale_up_cooldown_seconds = float(scale_up_cooldown_seconds)
        self.scale_down_cooldown_seconds = float(scale_down_cooldown_seconds)
        # Attribute retained for callers that expose the legacy setting.
        self.scaling_cooldown_seconds = float(legacy_cooldown)
        self.monitor_interval_seconds = float(monitor_interval_seconds)
        self.ewma_alpha = float(ewma_alpha)
        self.arrival_decay_seconds = float(arrival_decay_seconds)
        self.forward_tenant_as = forward_tenant_as
        self._replica_factory = replica_factory
        self._capacity_guard = capacity_guard
        self._close_callback = close_callback
        self._fatal_exception_predicate = fatal_exception_predicate
        self._warmup_args: tuple[Any, ...] = (
            (warmup_snapshots,) if warmup_snapshots is not None else ()
        )
        self._warmup_kwargs: dict[str, Any] = {}
        self._clock = clock

        initial_slots = tuple(
            _EngineSlot(index, replica_factory, close_callback)
            for index in range(min_size)
        )
        self._slots: dict[int, _EngineSlot[EngineT]] = {
            slot.index: slot for slot in initial_slots
        }
        self._next_slot_index = min_size
        # Initial slots are intentionally not schedulable until startup
        # ``warmup`` has constructed and probed them.  This is the hard
        # boundary that prevents the first user request from paying cold-start
        # latency on its request thread.
        self._available: Deque[_EngineSlot[EngineT]] = deque()
        self._tenant_queues: dict[str, Deque[_Waiter[EngineT]]] = {}
        self._tenant_order: Deque[str] = deque()
        self._pending = 0
        self._active = 0
        self._retiring = 0
        self._warming = False
        self._scaling = False
        self._scaling_direction: str | None = None
        self._desired_size = min_size
        self._startup_ready = False
        self._repair_target_size = 0
        self._repair_next_attempt_at = -math.inf
        self._last_scale_error: str | None = None
        self._scale_up_needed_since: float | None = None
        self._scale_down_needed_since: float | None = None
        self._last_scale_up_finished_at = -math.inf
        self._last_scale_down_finished_at = -math.inf
        self._closed = False
        self._condition = threading.Condition(threading.Lock())
        self._warmup_lock = threading.Lock()
        self._monitor_stop = threading.Event()
        self._monitor_wakeup = threading.Event()
        self._monitor_thread: threading.Thread | None = None

        self._last_arrival_at: float | None = None
        self._arrival_rate_ewma = 0.0
        self._arrival_rate_updated_at = self._clock()
        self._arrival_trend = 0.0
        self._service_time_ewma = 0.0
        self._last_instantaneous_demand = 0

        self._submitted = 0
        self._started = 0
        self._completed = 0
        self._failed = 0
        self._saturated = 0
        self._timed_out = 0
        self._observed_engine_load_failures = 0
        self._warmup_runs = 0
        self._warmup_failures = 0
        self._scale_successes = 0
        self._scale_failures = 0
        self._scale_up_successes = 0
        self._scale_up_failures = 0
        self._scale_down_successes = 0
        self._scale_down_failures = 0
        self._fatal_operation_failures = 0
        self._quarantined_replicas = 0
        self._replacement_attempts = 0
        self._replacement_successes = 0
        self._replacement_failures = 0
        self._peak_active = 0
        self._peak_pending = 0
        self._total_queue_wait = 0.0
        self._max_queue_wait = 0.0
        self._total_execution = 0.0
        self._max_execution = 0.0

    @staticmethod
    def _tenant(tenant_id: str) -> str:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string")
        return tenant_id

    @staticmethod
    def _timeout(value: float | None, default: float) -> float:
        timeout = default if value is None else float(value)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("queue timeout must be a finite non-negative number")
        return timeout

    @property
    def size(self) -> int:
        """Current replica count; fixed pools retain their configured size."""

        with self._condition:
            return len(self._slots)

    def _ensure_monitor(self) -> None:
        # A fixed-size pool still needs a background repair worker when fatal
        # engine isolation is enabled. Pools using the compatibility default
        # (no fatal predicate) retain the old zero-monitor behavior.
        if (
            self.min_size == self.max_size
            and self._fatal_exception_predicate is None
        ):
            return
        with self._condition:
            if self._closed or self._monitor_thread is not None:
                return
            thread = threading.Thread(
                target=self._monitor,
                name="tmcra-recall-pool-autoscaler",
                daemon=True,
            )
            self._monitor_thread = thread
            thread.start()

    def _monitor(self) -> None:
        while not self._monitor_stop.is_set():
            self._monitor_wakeup.wait(self.monitor_interval_seconds)
            self._monitor_wakeup.clear()
            if self._monitor_stop.is_set():
                break
            try:
                self.reconcile(background=False)
            except Exception as exc:
                # Autoscaling must never terminate request service. Expose the
                # error through status/metrics and retry only after cooldown.
                with self._condition:
                    self._last_scale_error = self._error_text(exc)

    @staticmethod
    def _error_text(error: BaseException) -> str:
        text = str(error).replace("\n", " ").strip()
        return f"{type(error).__name__}: {text}"[:500]

    def _decayed_arrival_rate_locked(self, now: float) -> float:
        elapsed = max(0.0, now - self._arrival_rate_updated_at)
        if elapsed:
            self._arrival_rate_ewma *= math.exp(
                -elapsed / self.arrival_decay_seconds
            )
            self._arrival_rate_updated_at = now
        return self._arrival_rate_ewma

    def _record_arrival_locked(self, now: float) -> None:
        previous = self._decayed_arrival_rate_locked(now)
        if self._last_arrival_at is not None:
            interval = max(0.001, now - self._last_arrival_at)
            sample_rate = 1.0 / interval
            updated = (
                self.ewma_alpha * sample_rate
                + (1.0 - self.ewma_alpha) * previous
            )
            self._arrival_trend = updated - previous
            self._arrival_rate_ewma = updated
        else:
            self._arrival_trend = 0.0
        self._arrival_rate_updated_at = now
        self._last_arrival_at = now

    def _record_service_time_locked(self, value: float) -> None:
        value = max(0.0, value)
        if self._service_time_ewma == 0.0:
            self._service_time_ewma = value
        else:
            self._service_time_ewma = (
                self.ewma_alpha * value
                + (1.0 - self.ewma_alpha) * self._service_time_ewma
            )

    def _desired_capacity_locked(self, now: float) -> int:
        current = len(self._slots)
        arrival_rate = self._decayed_arrival_rate_locked(now)
        offered_load = arrival_rate * self._service_time_ewma
        instantaneous = self._active + self._pending
        demand = max(float(instantaneous), offered_load)
        desired = (
            math.ceil(demand / self.target_utilization)
            if demand > 0
            else self.min_size
        )
        utilization = instantaneous / current if current else 0.0
        rising = (
            self._arrival_trend > 0
            or instantaneous > self._last_instantaneous_demand
        )
        if instantaneous and (rising or utilization >= self.target_utilization):
            desired = max(desired, instantaneous + self.warm_spares)
        self._last_instantaneous_demand = instantaneous
        return min(self.max_size, max(self.min_size, desired))

    def _plan_reconcile_locked(
        self, now: float
    ) -> tuple[
        str,
        _EngineSlot[EngineT] | int,
        tuple[tuple[Any, ...], dict[str, Any]] | None,
    ] | None:
        current = len(self._slots)
        repair_target = min(
            self.max_size,
            max(self.min_size, self._repair_target_size),
        ) if self._repair_target_size else 0
        desired = max(self._desired_capacity_locked(now), repair_target)
        self._desired_size = desired
        if desired > current:
            if self._scale_up_needed_since is None:
                self._scale_up_needed_since = now
            self._scale_down_needed_since = None
        elif desired < current:
            if self._scale_down_needed_since is None:
                self._scale_down_needed_since = now
            self._scale_up_needed_since = None
        else:
            self._scale_up_needed_since = None
            self._scale_down_needed_since = None

        if self._closed or self._warming or self._scaling:
            return None

        # A quarantined resident slot is repaired before normal elasticity.
        # This path deliberately bypasses startup-ready, fixed-size, sustain,
        # and normal scale-up cooldown gates. Failed repairs still use an
        # explicit retry deadline so a broken constructor cannot busy-loop.
        if repair_target > current:
            if now < self._repair_next_attempt_at:
                return None
            index = self._next_slot_index
            self._next_slot_index += 1
            self._scaling = True
            self._scaling_direction = "repair"
            self._replacement_attempts += 1
            warmup_call = (self._warmup_args, dict(self._warmup_kwargs))
            return "repair", index, warmup_call

        if not self._startup_ready or self.min_size == self.max_size:
            return None

        if (
            desired > current
            and self._scale_up_needed_since is not None
            and now - self._scale_up_needed_since
            >= self.scale_up_sustain_seconds
            and now - self._last_scale_up_finished_at
            >= self.scale_up_cooldown_seconds
        ):
            index = self._next_slot_index
            self._next_slot_index += 1
            self._scaling = True
            self._scaling_direction = "up"
            # Capture an immutable view of the successful startup probe. The
            # scaler thread will build and warm the replica entirely off the
            # user request path.
            warmup_call = (self._warmup_args, dict(self._warmup_kwargs))
            return "up", index, warmup_call

        if (
            desired < current
            and current > self.min_size
            and self._scale_down_needed_since is not None
            and now - self._scale_down_needed_since
            >= self.scale_down_idle_seconds
            and now - self._last_scale_down_finished_at
            >= self.scale_down_cooldown_seconds
        ):
            candidates = [
                slot
                for slot in self._available
                if slot.index in self._slots
            ]
            if not candidates:
                return None
            slot = max(candidates, key=lambda item: item.index)
            self._available.remove(slot)
            del self._slots[slot.index]
            self._scaling = True
            self._scaling_direction = "down"
            return "down", slot, None
        return None

    def reconcile(
        self, *, now: float | None = None, background: bool = True
    ) -> int:
        """Recompute desired capacity and, when due, perform one scale step.

        ``now`` plus ``background=False`` makes autoscaling deterministic in
        tests. Production callers normally use the defaults; construction and
        warmup then happen on a daemon thread rather than a request thread.
        """

        timestamp = self._clock() if now is None else float(now)
        if not math.isfinite(timestamp):
            raise ValueError("reconcile time must be finite")
        with self._condition:
            action = self._plan_reconcile_locked(timestamp)
            desired = self._desired_size
        if action is None:
            return desired
        if background:
            thread = threading.Thread(
                target=self._perform_scale_action,
                args=(action,),
                name=f"tmcra-recall-scale-{action[0]}",
                daemon=True,
            )
            thread.start()
        else:
            self._perform_scale_action(action)
        return desired

    def _perform_scale_action(
        self,
        action: tuple[
            str,
            _EngineSlot[EngineT] | int,
            tuple[tuple[Any, ...], dict[str, Any]] | None,
        ],
    ) -> None:
        direction, value, warmup_call = action
        if direction in {"up", "repair"}:
            if warmup_call is None:
                raise RecallPoolError("missing scale-up warmup call")
            self._scale_up(
                int(cast(int, value)),
                warmup_call[0],
                warmup_call[1],
                replacement=direction == "repair",
            )
        else:
            self._scale_down(cast(_EngineSlot[EngineT], value))

    def _scale_up(
        self,
        index: int,
        warmup_args: tuple[Any, ...],
        warmup_kwargs: Mapping[str, Any],
        *,
        replacement: bool = False,
    ) -> None:
        slot = _EngineSlot(index, self._replica_factory, self._close_callback)
        error: Exception | None = None
        try:
            with self._condition:
                current = len(self._slots)
                if self._closed:
                    raise RecallPoolClosedError(retry_after=self.retry_after)
            if self._capacity_guard is not None and not bool(
                self._capacity_guard(current, current + 1)
            ):
                raise RecallPoolError(
                    f"capacity guard rejected replica count {current + 1}"
                )
            slot.run(
                lambda engine: self._warm_engine(
                    engine,
                    warmup_args,
                    warmup_kwargs,
                )
            )
        except Exception as exc:
            error = exc

        close_error: Exception | None = None
        with self._condition:
            closed = self._closed
        if error is not None or closed:
            close_error = self._close_slot_safely(slot)
            if error is None:
                error = RecallPoolClosedError(retry_after=self.retry_after)
        with self._condition:
            if error is None:
                self._slots[slot.index] = slot
                self._available.append(slot)
                self._scale_successes += 1
                self._scale_up_successes += 1
                if replacement:
                    self._replacement_successes += 1
                self._last_scale_error = None
                if (
                    self._repair_target_size
                    and len(self._slots) >= self._repair_target_size
                ):
                    self._repair_target_size = 0
                self._repair_next_attempt_at = -math.inf
                loaded = sum(
                    1 for candidate in self._slots.values() if candidate.loaded
                )
                self._startup_ready = loaded >= self.min_size
                self._dispatch_locked()
            else:
                self._observed_engine_load_failures += slot.load_failures
                self._scale_failures += 1
                self._scale_up_failures += 1
                if replacement:
                    self._replacement_failures += 1
                self._last_scale_error = self._error_text(error)
                if close_error is not None:
                    self._last_scale_error += "; close: " + self._error_text(
                        close_error
                    )
            self._scaling = False
            self._scaling_direction = None
            self._last_scale_up_finished_at = self._clock()
            if replacement and error is not None:
                self._repair_next_attempt_at = (
                    self._last_scale_up_finished_at
                    + max(
                        self.monitor_interval_seconds,
                        self.scale_up_cooldown_seconds,
                    )
                )
            self._scale_up_needed_since = None
            self._desired_size = max(
                self._desired_capacity_locked(
                    self._last_scale_up_finished_at
                ),
                min(
                    self.max_size,
                    max(self.min_size, self._repair_target_size),
                )
                if self._repair_target_size
                else 0,
            )
            self._condition.notify_all()
        self._monitor_wakeup.set()

    def _scale_down(self, slot: _EngineSlot[EngineT]) -> None:
        close_error = self._close_slot_safely(slot)
        with self._condition:
            if close_error is None:
                self._scale_successes += 1
                self._scale_down_successes += 1
                self._last_scale_error = None
            else:
                self._scale_failures += 1
                self._scale_down_failures += 1
                self._last_scale_error = self._error_text(close_error)
            self._scaling = False
            self._scaling_direction = None
            self._last_scale_down_finished_at = self._clock()
            self._desired_size = self._desired_capacity_locked(
                self._last_scale_down_finished_at
            )
            # Preserve the original low-load timestamp while more than one
            # idle replica still needs retirement.  Subsequent removals are
            # then paced by the scale-down cooldown instead of charging a new
            # ten-minute idle window for every replica.
            if self._desired_size >= len(self._slots):
                self._scale_down_needed_since = None
            self._condition.notify_all()
        self._monitor_wakeup.set()

    @staticmethod
    def _close_slot_safely(slot: _EngineSlot[EngineT]) -> Exception | None:
        try:
            slot.close()
        except Exception as exc:
            return exc
        return None

    def _close_tracked_slot(
        self, slot: _EngineSlot[EngineT]
    ) -> Exception | None:
        """Close one slot whose retirement was registered under the lock."""

        error = self._close_slot_safely(slot)
        with self._condition:
            if self._retiring <= 0:
                raise RecallPoolError("recall engine pool retiring count underflow")
            self._retiring -= 1
            if error is not None:
                self._last_scale_error = self._error_text(error)
            self._condition.notify_all()
        return error

    def _start_tracked_closes(
        self, slots: tuple[_EngineSlot[EngineT], ...]
    ) -> None:
        if not slots:
            return

        def close_all() -> None:
            # CUDA cache release is intentionally sequential even though this
            # worker is detached from the caller's timeout budget.
            for slot in slots:
                self._close_tracked_slot(slot)

        thread = threading.Thread(
            target=close_all,
            name="tmcra-recall-retire-idle",
            daemon=True,
        )
        thread.start()

    def _dispatch_locked(self) -> None:
        if self._warming or self._closed:
            return
        assigned = False
        while self._available and self._tenant_order:
            tenant_id = self._tenant_order.popleft()
            tenant_queue = self._tenant_queues.get(tenant_id)
            if not tenant_queue:
                self._tenant_queues.pop(tenant_id, None)
                continue
            waiter = tenant_queue.popleft()
            if tenant_queue:
                self._tenant_order.append(tenant_id)
            else:
                del self._tenant_queues[tenant_id]
            waiter.slot = self._available.popleft()
            self._pending -= 1
            self._active += 1
            self._peak_active = max(self._peak_active, self._active)
            assigned = True
        if assigned:
            self._condition.notify_all()

    def _remove_waiter_locked(self, waiter: _Waiter[EngineT]) -> bool:
        tenant_queue = self._tenant_queues.get(waiter.tenant_id)
        if tenant_queue is None:
            return False
        try:
            tenant_queue.remove(waiter)
        except ValueError:
            return False
        self._pending -= 1
        if not tenant_queue:
            del self._tenant_queues[waiter.tenant_id]
            try:
                self._tenant_order.remove(waiter.tenant_id)
            except ValueError:
                pass
        return True

    def _checkout(
        self, tenant_id: str, queue_timeout: float | None
    ) -> tuple[_EngineSlot[EngineT], float]:
        self._ensure_monitor()
        tenant_id = self._tenant(tenant_id)
        timeout = self._timeout(queue_timeout, self.queue_timeout)
        submitted_at = self._clock()
        deadline = submitted_at + timeout

        with self._condition:
            self._submitted += 1
            self._record_arrival_locked(submitted_at)
            if self._closed:
                raise RecallPoolClosedError(retry_after=self.retry_after)
            # Do not make an idle engine pass through queue accounting. This
            # permits a useful zero-length pending queue configuration.
            if self._available and not self._tenant_order and not self._warming:
                slot = self._available.popleft()
                self._active += 1
                self._peak_active = max(self._peak_active, self._active)
                queue_wait = self._clock() - submitted_at
                self._record_start_locked(queue_wait)
                self._monitor_wakeup.set()
                return slot, queue_wait

            tenant_queue = self._tenant_queues.get(tenant_id)
            tenant_pending = len(tenant_queue) if tenant_queue is not None else 0
            if tenant_pending >= self.per_tenant_pending:
                self._saturated += 1
                raise RecallPoolSaturatedError(
                    scope="tenant", retry_after=self.retry_after
                )
            if self._pending >= self.max_pending:
                self._saturated += 1
                raise RecallPoolSaturatedError(
                    scope="global", retry_after=self.retry_after
                )

            waiter = _Waiter[EngineT](tenant_id, submitted_at)
            if tenant_queue is None:
                tenant_queue = deque()
                self._tenant_queues[tenant_id] = tenant_queue
                self._tenant_order.append(tenant_id)
            tenant_queue.append(waiter)
            self._pending += 1
            self._peak_pending = max(self._peak_pending, self._pending)
            self._dispatch_locked()
            self._monitor_wakeup.set()

            while waiter.slot is None:
                if self._closed:
                    self._remove_waiter_locked(waiter)
                    raise RecallPoolClosedError(retry_after=self.retry_after)
                remaining = deadline - self._clock()
                if remaining <= 0:
                    # Assignment and cancellation are both performed under the
                    # same condition lock, so no engine can be lost in a race.
                    if self._remove_waiter_locked(waiter):
                        self._timed_out += 1
                        waited = self._clock() - submitted_at
                        self._dispatch_locked()
                        self._condition.notify_all()
                        raise RecallPoolTimeoutError(
                            waited=waited, retry_after=self.retry_after
                        )
                    raise RecallPoolError(
                        "recall pool lost a pending waiter during timeout"
                    )
                self._condition.wait(timeout=remaining)

            queue_wait = self._clock() - submitted_at
            self._record_start_locked(queue_wait)
            return waiter.slot, queue_wait

    def _record_start_locked(self, queue_wait: float) -> None:
        self._started += 1
        self._total_queue_wait += queue_wait
        self._max_queue_wait = max(self._max_queue_wait, queue_wait)

    def _is_fatal_exception(self, error: BaseException) -> bool:
        predicate = self._fatal_exception_predicate
        if predicate is None:
            return False
        try:
            return bool(predicate(error))
        except Exception as predicate_error:
            # Never mask the operation's original exception or destroy a
            # usable replica because an observability policy is itself broken.
            with self._condition:
                self._last_scale_error = (
                    "fatal predicate failed: "
                    + type(predicate_error).__name__
                )
            return False

    def _return(
        self,
        slot: _EngineSlot[EngineT],
        *,
        succeeded: bool,
        fatal: bool,
        execution_seconds: float,
        load_failures_before: int,
    ) -> None:
        retire = False
        repair_after_retire = False
        with self._condition:
            if self._active <= 0:
                raise RecallPoolError("recall engine pool active count underflow")
            self._active -= 1
            if self._closed:
                self._slots.pop(slot.index, None)
                retire = True
            elif fatal:
                target = min(
                    self.max_size,
                    max(self.min_size, self._desired_size),
                )
                self._slots.pop(slot.index, None)
                retire = True
                self._fatal_operation_failures += 1
                self._quarantined_replicas += 1
                if len(self._slots) < target:
                    self._repair_target_size = max(
                        self._repair_target_size, target
                    )
                    # Do not construct a replacement until the failed slot's
                    # close hook has had a chance to release its GPU memory.
                    self._repair_next_attempt_at = math.inf
                    repair_after_retire = True
                loaded = sum(
                    1 for candidate in self._slots.values() if candidate.loaded
                )
                self._startup_ready = loaded >= self.min_size
            else:
                self._available.append(slot)
            if retire:
                self._retiring += 1
            if succeeded:
                self._completed += 1
            else:
                self._failed += 1
            self._total_execution += execution_seconds
            self._max_execution = max(self._max_execution, execution_seconds)
            self._record_service_time_locked(execution_seconds)
            self._observed_engine_load_failures += max(
                0, slot.load_failures - load_failures_before
            )
            repair_target = (
                min(
                    self.max_size,
                    max(self.min_size, self._repair_target_size),
                )
                if self._repair_target_size
                else 0
            )
            self._desired_size = max(
                self._desired_capacity_locked(self._clock()), repair_target
            )
            self._dispatch_locked()
            self._condition.notify_all()
        if retire:
            self._close_tracked_slot(slot)
            with self._condition:
                if (
                    repair_after_retire
                    and not self._closed
                    and self._repair_target_size > len(self._slots)
                ):
                    self._repair_next_attempt_at = self._clock()
                self._condition.notify_all()
        self._monitor_wakeup.set()

    def execute(
        self,
        tenant_id: str,
        operation: Callable[[EngineT], ResultT],
        *,
        queue_timeout: float | None = None,
        before_execute: Callable[[], None] | None = None,
    ) -> ResultT:
        """Execute one blocking operation on a fairly scheduled engine slot."""

        if not callable(operation):
            raise TypeError("recall operation must be callable")
        if before_execute is not None and not callable(before_execute):
            raise TypeError("before_execute must be callable")
        slot, _queue_wait = self._checkout(tenant_id, queue_timeout)
        load_failures_before = slot.load_failures
        started_at = self._clock()
        succeeded = False
        operation_started = False
        failure: BaseException | None = None
        try:
            if before_execute is not None:
                # Admission/quota work runs only after an engine has been
                # reserved, but outside the engine boundary. Its failure must
                # return the slot without classifying the engine as corrupt.
                before_execute()
            operation_started = True
            result = slot.run(operation)
            succeeded = True
            return result
        except BaseException as exc:
            failure = exc
            raise
        finally:
            self._return(
                slot,
                succeeded=succeeded,
                fatal=(
                    operation_started
                    and failure is not None
                    and self._is_fatal_exception(failure)
                ),
                execution_seconds=max(0.0, self._clock() - started_at),
                load_failures_before=load_failures_before,
            )

    def run_idle_maintenance(
        self, operation: Callable[[], ResultT]
    ) -> tuple[bool, ResultT | None]:
        """Run a short process-wide maintenance action only while every lane is idle.

        The maintenance gate prevents a request, startup warmup, or autoscale
        action from beginning between the idle check and ``operation``.  New
        requests may enter the bounded pending queue and are dispatched as
        soon as the operation completes.  The callback must therefore remain
        short; CUDA allocator cache release is the intended production use.
        """

        if not callable(operation):
            raise TypeError("idle maintenance operation must be callable")
        if not self._warmup_lock.acquire(blocking=False):
            return False, None
        try:
            with self._condition:
                current_size = len(self._slots)
                fully_idle = (
                    not self._closed
                    and self._startup_ready
                    and not self._warming
                    and not self._scaling
                    and self._active == 0
                    and self._pending == 0
                    and self._retiring == 0
                    and current_size > 0
                    and len(self._available) == current_size
                )
                if not fully_idle:
                    return False, None
                # Reuse the existing warming gate: checkout may enqueue while
                # maintenance is in progress, but it cannot lease a lane.
                self._warming = True
            try:
                return True, operation()
            finally:
                with self._condition:
                    self._warming = False
                    self._dispatch_locked()
                    self._condition.notify_all()
                self._monitor_wakeup.set()
        finally:
            self._warmup_lock.release()

    def recall(
        self,
        tenant_id: str,
        *,
        queue_timeout: float | None = None,
        before_execute: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call ``engine.recall(**kwargs)`` through the blocking scheduler."""

        call_kwargs = dict(kwargs)
        if self.forward_tenant_as is not None:
            if self.forward_tenant_as in call_kwargs:
                raise ValueError(
                    f"recall keyword conflicts with forwarded tenant field: "
                    f"{self.forward_tenant_as}"
                )
            call_kwargs[self.forward_tenant_as] = tenant_id

        def invoke(engine: EngineT) -> Any:
            method = getattr(engine, "recall", None)
            if method is None or not callable(method):
                raise RecallPoolError("recall engine has no callable recall method")
            return method(**call_kwargs)

        return self.execute(
            tenant_id,
            invoke,
            queue_timeout=queue_timeout,
            before_execute=before_execute,
        )

    def warmup(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Exclusively construct and warm every replica in stable index order.

        Warmup is intended for application startup. New requests may enter the
        bounded queue while it runs, but no recall operation can overlap it.
        Every replica is attempted even when an earlier replica fails.
        """

        self._ensure_monitor()
        with self._warmup_lock:
            with self._condition:
                if self._closed:
                    raise RecallPoolClosedError(retry_after=self.retry_after)
                self._warming = True
                self._warmup_runs += 1
                while self._active or self._scaling:
                    self._condition.wait()
                slots = tuple(
                    sorted(self._slots.values(), key=lambda item: item.index)
                )
                current_size = len(self._slots)
                # No slot is schedulable while startup/re-warm is running.
                self._available.clear()
                self._active += len(slots)

            results: list[Any | None] = [None] * current_size
            failures: list[tuple[int, Exception]] = []
            successful: list[_EngineSlot[EngineT]] = []
            load_failures_before = [slot.load_failures for slot in slots]
            retire_after_warmup: tuple[_EngineSlot[EngineT], ...] = ()
            try:
                for offset, slot in enumerate(slots):
                    try:
                        results[offset] = slot.run(
                            lambda engine: self._warm_engine(engine, args, kwargs)
                        )
                        successful.append(slot)
                    except Exception as exc:
                        failures.append((slot.index, exc))
                        close_error = self._close_slot_safely(slot)
                        if close_error is not None:
                            exc.add_note(
                                "replica close also failed: "
                                + self._error_text(close_error)
                            )
            finally:
                with self._condition:
                    self._active -= len(slots)
                    if self._closed:
                        retire_after_warmup = slots
                        for slot in slots:
                            self._slots.pop(slot.index, None)
                        self._retiring += len(retire_after_warmup)
                    else:
                        # A failed or half-warmed engine is never published to
                        # request scheduling. A later explicit warmup may retry
                        # the retained empty slot.
                        self._available.extend(successful)
                    self._observed_engine_load_failures += sum(
                        max(0, slot.load_failures - load_failures_before[offset])
                        for offset, slot in enumerate(slots)
                    )
                    self._warmup_failures += len(failures)
                    loaded = sum(1 for slot in self._slots.values() if slot.loaded)
                    self._startup_ready = (
                        not self._closed and loaded >= self.min_size
                    )
                    if self._startup_ready:
                        self._warmup_args = tuple(args)
                        self._warmup_kwargs = dict(kwargs)
                    self._warming = False
                    self._dispatch_locked()
                    self._condition.notify_all()
                for slot in retire_after_warmup:
                    self._close_tracked_slot(slot)
                self._monitor_wakeup.set()

            result_tuple = tuple(results)
            if failures:
                raise RecallPoolWarmupError(
                    failures=tuple(failures), results=result_tuple
                ) from failures[0][1]
            return cast(tuple[Any, ...], result_tuple)

    @staticmethod
    def _warm_engine(
        engine: EngineT, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> Any:
        method = getattr(engine, "warmup", None)
        if method is None or not callable(method):
            raise RecallPoolError("recall engine has no callable warmup method")
        return method(*args, **dict(kwargs))

    def close(
        self,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Stop admission/autoscaling and retire every replica safely.

        Idle replicas are detached before their user-defined close hook runs.
        Leased replicas are retired by ``_return`` after the in-flight request
        finishes.  A replica being warmed or scaled is likewise closed by the
        owning background path, so this method never races a close against an
        engine operation.
        """

        if not isinstance(wait, bool):
            raise TypeError("wait must be a boolean")
        if timeout is not None:
            timeout = float(timeout)
            if not math.isfinite(timeout) or timeout < 0:
                raise ValueError("close timeout must be finite and non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout

        immediately_retired: list[_EngineSlot[EngineT]] = []
        with self._condition:
            self._closed = True
            self._startup_ready = False
            self._repair_target_size = 0
            self._repair_next_attempt_at = math.inf
            self._monitor_stop.set()
            self._monitor_wakeup.set()

            # Unassigned waiters observe ``closed`` after this notification.
            # Clearing scheduler ownership here makes pending status converge
            # immediately and cannot lose a leased slot (leased waiters have
            # already been removed by ``_dispatch_locked``).
            self._tenant_queues.clear()
            self._tenant_order.clear()
            self._pending = 0

            if not self._warming:
                available_ids = {slot.index for slot in self._available}
                immediately_retired.extend(self._available)
                self._available.clear()
                # Empty startup slots have never been leased and are safe to
                # retire even while other, loaded slots remain active.
                immediately_retired.extend(
                    slot
                    for slot in self._slots.values()
                    if slot.index not in available_ids and not slot.loaded
                )
                for slot in immediately_retired:
                    self._slots.pop(slot.index, None)
                self._retiring += len(immediately_retired)
            monitor_thread = self._monitor_thread
            self._condition.notify_all()

        # User-defined/model close hooks may block. Run idle retirement in a
        # daemon worker so ``timeout`` is a real upper bound for this caller.
        self._start_tracked_closes(tuple(immediately_retired))

        if (
            wait
            and monitor_thread is not None
            and monitor_thread is not threading.current_thread()
        ):
            remaining = (
                None
                if deadline is None
                else max(0.0, deadline - time.monotonic())
            )
            monitor_thread.join(timeout=remaining)

        if wait:
            with self._condition:
                while (
                    self._active
                    or self._warming
                    or self._scaling
                    or self._retiring
                ):
                    remaining = (
                        None
                        if deadline is None
                        else deadline - time.monotonic()
                    )
                    if remaining is not None and remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)

    def stop(
        self,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Lifecycle alias used by the service runtime."""

        self.close(wait=wait, timeout=timeout)

    @property
    def loaded(self) -> bool:
        """Whether the configured minimum resident pool is ready."""

        with self._condition:
            return self._startup_ready and not self._closed

    @property
    def loaded_count(self) -> int:
        with self._condition:
            return sum(1 for slot in self._slots.values() if slot.loaded)

    def status(self) -> RecallPoolStatus:
        with self._condition:
            current_size = len(self._slots)
            loaded = sum(1 for slot in self._slots.values() if slot.loaded)
            return RecallPoolStatus(
                configured=self.min_size,
                min_size=self.min_size,
                max_size=self.max_size,
                current_size=current_size,
                desired_size=self._desired_size,
                loaded=loaded,
                fully_loaded=self._startup_ready and not self._closed,
                active=self._active,
                retiring=self._retiring,
                idle=len(self._available),
                pending=self._pending,
                pending_tenants=len(self._tenant_queues),
                max_pending=self.max_pending,
                per_tenant_pending=self.per_tenant_pending,
                warming=self._warming,
                scaling=self._scaling,
                scaling_direction=self._scaling_direction,
                replacement_pending=(
                    not self._closed
                    and self._repair_target_size > current_size
                ),
                repair_target_size=self._repair_target_size,
                closed=self._closed,
                last_scale_error=self._last_scale_error,
            )

    def metrics(self) -> RecallPoolMetrics:
        with self._condition:
            terminal = self._completed + self._failed
            now = self._clock()
            arrival_rate = self._decayed_arrival_rate_locked(now)
            offered_load = arrival_rate * self._service_time_ewma
            current_size = len(self._slots)
            return RecallPoolMetrics(
                submitted=self._submitted,
                started=self._started,
                completed=self._completed,
                failed=self._failed,
                saturated=self._saturated,
                timed_out=self._timed_out,
                engine_load_failures=self._observed_engine_load_failures,
                warmup_runs=self._warmup_runs,
                warmup_failures=self._warmup_failures,
                scale_successes=self._scale_successes,
                scale_failures=self._scale_failures,
                scale_up_successes=self._scale_up_successes,
                scale_up_failures=self._scale_up_failures,
                scale_down_successes=self._scale_down_successes,
                scale_down_failures=self._scale_down_failures,
                fatal_operation_failures=self._fatal_operation_failures,
                quarantined_replicas=self._quarantined_replicas,
                replacement_attempts=self._replacement_attempts,
                replacement_successes=self._replacement_successes,
                replacement_failures=self._replacement_failures,
                current_size=current_size,
                desired_size=self._desired_size,
                active=self._active,
                pending=self._pending,
                peak_active=self._peak_active,
                peak_pending=self._peak_pending,
                arrival_rate_ewma=arrival_rate,
                service_time_ewma_seconds=self._service_time_ewma,
                offered_load=offered_load,
                utilization=(
                    self._active / current_size if current_size else 0.0
                ),
                target_utilization=self.target_utilization,
                total_queue_wait_seconds=self._total_queue_wait,
                average_queue_wait_seconds=(
                    self._total_queue_wait / self._started if self._started else 0.0
                ),
                max_queue_wait_seconds=self._max_queue_wait,
                total_execution_seconds=self._total_execution,
                average_execution_seconds=(
                    self._total_execution / terminal if terminal else 0.0
                ),
                max_execution_seconds=self._max_execution,
            )


__all__ = [
    "RecallEnginePool",
    "RecallPoolAdmissionError",
    "RecallPoolClosedError",
    "RecallPoolError",
    "RecallPoolMetrics",
    "RecallPoolSaturated",
    "RecallPoolSaturatedError",
    "RecallPoolStatus",
    "RecallPoolTimeout",
    "RecallPoolTimeoutError",
    "RecallPoolWarmupError",
]
