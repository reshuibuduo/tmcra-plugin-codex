from __future__ import annotations

import json
import os
import shutil
import threading
import time
import urllib.request
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from .writer_provider import primary_writer_route


CHECK_NAMES = (
    "control_db",
    "state_disk",
    "gpu",
    "online_engine",
    "writer_pool",
    "service_worker",
    "adapter_compatibility",
    "active_indexes",
    "provider",
)

_MAX_PROVIDER_RESPONSE_BYTES = 1024 * 1024


def _positive_environment_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Any,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        return None


def _open_without_redirects(request: Any, *, timeout: float) -> Any:
    opener = urllib.request.build_opener(_NoRedirectHandler())
    return opener.open(request, timeout=timeout)


class ProviderModelListProbe:
    """Use only the provider's non-billable model-list endpoint."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        environment: Mapping[str, str] | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("provider probe timeout must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.environment = environment if environment is not None else os.environ
        self.opener = opener or _open_without_redirects
        self._key_index = 0
        self._lock = threading.Lock()

    def __call__(self) -> bool:
        try:
            route = primary_writer_route(self.environment)
        except ValueError:
            return False
        models_url = route.base_url.rstrip("/") + "/models"
        with self._lock:
            api_key = route.api_keys[self._key_index % len(route.api_keys)]
            self._key_index += 1
        request = urllib.request.Request(
            models_url,
            method="GET",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "tmcra-readiness/1",
            },
        )
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                status_code = int(getattr(response, "status", response.getcode()))
                if status_code != 200:
                    return False
                raw = response.read(_MAX_PROVIDER_RESPONSE_BYTES + 1)
        except Exception:
            return False
        if len(raw) > _MAX_PROVIDER_RESPONSE_BYTES:
            return False
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(payload, Mapping):
            return False
        entries = payload.get("data")
        if not isinstance(entries, list):
            return False
        model_ids = {
            str(entry.get("id") or "")
            for entry in entries
            if isinstance(entry, Mapping)
        }
        return route.model in model_ids


@dataclass(frozen=True)
class _CheckObservation:
    ok: bool
    checked_at: float


class ContinuousReadinessMonitor:
    """Periodically refresh a fail-closed, secret-free readiness snapshot."""

    def __init__(
        self,
        *,
        settings: Any,
        database: Any,
        storage: Any,
        online: Any,
        worker: Any,
        interval_seconds: float | None = None,
        snapshot_ttl_seconds: float | None = None,
        active_index_interval_seconds: float | None = None,
        provider_interval_seconds: float | None = None,
        provider_timeout_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        check_overrides: Mapping[str, Callable[[], bool]] | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.storage = storage
        self.online = online
        self.worker = worker
        self.interval_seconds = float(
            interval_seconds
            if interval_seconds is not None
            else _positive_environment_float(
                "TMCRA_SERVICE_READINESS_INTERVAL_SECONDS", 10.0
            )
        )
        self.snapshot_ttl_seconds = float(
            snapshot_ttl_seconds
            if snapshot_ttl_seconds is not None
            else _positive_environment_float(
                "TMCRA_SERVICE_READINESS_TTL_SECONDS", 30.0
            )
        )
        self.active_index_interval_seconds = float(
            active_index_interval_seconds
            if active_index_interval_seconds is not None
            else _positive_environment_float(
                "TMCRA_SERVICE_ACTIVE_INDEX_HEALTH_INTERVAL_SECONDS", 60.0
            )
        )
        self.provider_interval_seconds = float(
            provider_interval_seconds
            if provider_interval_seconds is not None
            else _positive_environment_float(
                "TMCRA_SERVICE_PROVIDER_HEALTH_INTERVAL_SECONDS", 300.0
            )
        )
        provider_timeout = float(
            provider_timeout_seconds
            if provider_timeout_seconds is not None
            else _positive_environment_float(
                "TMCRA_SERVICE_PROVIDER_HEALTH_TIMEOUT_SECONDS", 3.0
            )
        )
        for name, value in (
            ("interval_seconds", self.interval_seconds),
            ("snapshot_ttl_seconds", self.snapshot_ttl_seconds),
            ("active_index_interval_seconds", self.active_index_interval_seconds),
            ("provider_interval_seconds", self.provider_interval_seconds),
            ("provider_timeout_seconds", provider_timeout),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.clock = clock
        provider_probe = ProviderModelListProbe(timeout_seconds=provider_timeout)
        self._checks: dict[str, Callable[[], bool]] = {
            "control_db": self._check_control_db,
            "state_disk": self._check_state_disk,
            "gpu": self._check_gpu,
            "online_engine": self._check_online_engine,
            "writer_pool": self._check_writer_pool,
            "service_worker": self._check_service_worker,
            "adapter_compatibility": self._check_adapter_compatibility,
            "active_indexes": self._check_active_indexes,
            "provider": provider_probe,
        }
        if check_overrides:
            unknown = sorted(set(check_overrides) - set(CHECK_NAMES))
            if unknown:
                raise ValueError("unknown readiness checks: " + ",".join(unknown))
            self._checks.update(check_overrides)
        self._cadences = {
            name: self.interval_seconds for name in CHECK_NAMES
        }
        self._cadences["active_indexes"] = self.active_index_interval_seconds
        self._cadences["provider"] = self.provider_interval_seconds
        self._observations: dict[str, _CheckObservation] = {}
        self._updated_at: float | None = None
        self._generation = 0
        self._monitor_failed = True
        self._running = False
        self._state_lock = threading.Lock()
        self._cycle_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._running

    @property
    def thread_alive(self) -> bool:
        with self._state_lock:
            thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def start(self, *, background: bool = True) -> None:
        with self._lifecycle_lock:
            with self._state_lock:
                if self._running:
                    return
                self._running = True
                self._monitor_failed = True
                self._stop_event.clear()
            self.run_once(force=True)
            if not background:
                return
            thread = threading.Thread(
                target=self._run_loop,
                name="tmcra-readiness-monitor",
                daemon=True,
            )
            with self._state_lock:
                self._thread = thread
            thread.start()

    def stop(self, *, timeout: float | None = None) -> bool:
        with self._lifecycle_lock:
            with self._state_lock:
                self._running = False
                thread = self._thread
            self._stop_event.set()
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=timeout)
            stopped = thread is None or not thread.is_alive()
            if stopped:
                with self._state_lock:
                    if self._thread is thread:
                        self._thread = None
            return stopped

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.run_once()
            except Exception:
                with self._state_lock:
                    self._monitor_failed = True
                    self._updated_at = self.clock()

    def run_once(self, *, force: bool = False) -> None:
        with self._cycle_lock:
            with self._state_lock:
                if not self._running:
                    return
                observations = dict(self._observations)
            started_at = self.clock()
            updates: dict[str, _CheckObservation] = {}
            for name in CHECK_NAMES:
                previous = observations.get(name)
                cadence = self._cadences[name] if previous and previous.ok else self.interval_seconds
                due = (
                    force
                    or previous is None
                    or started_at - previous.checked_at >= cadence
                )
                if not due:
                    continue
                try:
                    ok = bool(self._checks[name]())
                except Exception:
                    ok = False
                updates[name] = _CheckObservation(ok=ok, checked_at=self.clock())
            completed_at = self.clock()
            with self._state_lock:
                self._observations.update(updates)
                self._updated_at = completed_at
                self._generation += 1
                self._monitor_failed = False

    def snapshot(self) -> dict[str, Any]:
        now = self.clock()
        with self._state_lock:
            running = self._running
            observations = dict(self._observations)
            updated_at = self._updated_at
            generation = self._generation
            monitor_failed = self._monitor_failed
        age = None if updated_at is None else max(0.0, now - updated_at)
        stale = age is None or age > self.snapshot_ttl_seconds
        checks = {
            name: bool(observations.get(name) and observations[name].ok)
            for name in CHECK_NAMES
        }
        ready = (
            running
            and not monitor_failed
            and not stale
            and all(checks.values())
        )
        return {
            "ready": ready,
            "stale": stale,
            "running": running,
            "generation": generation,
            "snapshot_age_seconds": None if age is None else round(age, 3),
            "checks": checks,
        }

    def _check_control_db(self) -> bool:
        with closing(self.database.connect()) as connection:
            quick = connection.execute("PRAGMA quick_check(1)").fetchone()
            if not quick or str(quick[0]).lower() != "ok":
                return False
            return connection.execute("SELECT 1").fetchone() is not None

    def _check_state_disk(self) -> bool:
        usage = shutil.disk_usage(self.settings.state_dir)
        return int(usage.free) >= int(self.settings.disk_free_min_bytes)

    def _check_gpu(self) -> bool:
        import torch

        for configured in dict.fromkeys(
            [self.settings.device, self.settings.graph_device]
        ):
            device = torch.device(configured)
            if device.type == "cuda" and not torch.cuda.is_available():
                return False
            with torch.inference_mode():
                value = (torch.ones(4, device=device) + 1).sum()
                if not bool(torch.isfinite(value)):
                    return False
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        return True

    def _check_online_engine(self) -> bool:
        minimum = int(getattr(self.settings, "recall_pool_min_size", 1))
        loaded_count = getattr(self.online, "loaded_count", None)
        if isinstance(loaded_count, int) and not isinstance(loaded_count, bool):
            return int(loaded_count) >= minimum
        # Preserve compatibility with lightweight readiness fakes and older
        # embedding applications. Availability/busy state is deliberately not
        # part of readiness: a fully loaded pool remains healthy while busy.
        return bool(getattr(self.online, "loaded", False))

    def _check_writer_pool(self) -> bool:
        status = self.storage.writer_status()
        if not bool(status.get("alive")):
            return False
        if status.get("mode") != "resident":
            return True
        configured = int(status.get("configured", 0) or 0)
        ready = int(status.get("ready", 0) or 0)
        return configured > 0 and ready == configured

    def _check_service_worker(self) -> bool:
        return bool(self.worker.status().alive)

    def _check_adapter_compatibility(self) -> bool:
        compatibility = self.storage.compatibility()
        return bool(compatibility) and all(bool(value) for value in compatibility.values())

    def _check_active_indexes(self) -> bool:
        self.storage.audit_active_indexes()
        states = self.database.list_scope_evolution_states()
        audit = self.storage.audit_searchable_watermarks(
            states,
            # Runtime readiness describes whether the service can safely answer
            # requests.  A committed active index remains safe while its delta
            # index catches up with a normal write.  Requiring zero event lag
            # here removed the whole API from service discovery after every
            # ingest, even though read-your-writes already waits on the job
            # watermark at the request boundary.
            require_fresh=False,
        )
        missing = int(audit.get("missing_index_scope_count", 0) or 0)
        return bool(audit.get("ready")) and missing == 0
