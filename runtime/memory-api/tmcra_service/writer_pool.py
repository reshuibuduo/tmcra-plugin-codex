from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, TextIO

from .writer_daemon import (
    PROTOCOL_VERSION,
    REQUEST_STATUS_SCHEMA_VERSION,
    read_request_status,
    request_identity,
    request_status_path,
    write_request_status,
)


class WriterPoolError(RuntimeError):
    pass


class WriterPoolOperationError(WriterPoolError):
    def __init__(self, message: str, *, error_type: str = "WriterError") -> None:
        super().__init__(message)
        self.error_type = error_type


class WriterPoolOutcomeUnknown(WriterPoolError):
    pass


@dataclass(frozen=True)
class WriterPoolStatus:
    configured: int
    ready: int
    alive: bool
    pids: tuple[int, ...]
    protocol: str
    available: int = 0
    leased: int = 0


@dataclass
class _RequestGate:
    lock: threading.Lock
    users: int = 0


def _readline(stream: TextIO, timeout: float) -> str:
    result: queue.Queue[object] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            result.put(stream.readline())
        except BaseException as exc:
            result.put(exc)

    thread = threading.Thread(target=read, name="tmcra-writer-protocol-read", daemon=True)
    thread.start()
    try:
        value = result.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError("resident Writer protocol timed out") from exc
    if isinstance(value, BaseException):
        raise value
    return str(value)


class _WriterProcess:
    def __init__(
        self,
        *,
        index: int,
        python: Path,
        v4_root: Path,
        repo: Path,
        log_root: Path,
        environment: Mapping[str, str],
    ) -> None:
        self.index = index
        self.python = python
        self.v4_root = v4_root
        self.repo = repo
        self.log_root = log_root
        self.environment = dict(environment)
        self.process: subprocess.Popen[str] | None = None
        self.log: TextIO | None = None
        self.hello: dict[str, Any] = {}
        self._request_lock = threading.Lock()

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    @property
    def alive(self) -> bool:
        return bool(self.process is not None and self.process.poll() is None and self.hello.get("ok"))

    def launch(self) -> None:
        if self.process is not None:
            raise WriterPoolError("resident Writer worker was already launched")
        self.log_root.mkdir(parents=True, exist_ok=True)
        log_path = self.log_root / f"worker-{self.index}.stderr.log"
        self.log = log_path.open("a", encoding="utf-8")
        command = [
            str(self.python),
            "-u",
            "-m",
            "tmcra_service.writer_daemon",
            "--repo",
            str(self.repo),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=str(self.v4_root),
            env=self.environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def await_ready(self, timeout: float) -> None:
        process = self.process
        if process is None or process.stdout is None:
            raise WriterPoolError("resident Writer worker is not launched")
        line = _readline(process.stdout, timeout)
        if not line:
            code = process.poll()
            raise WriterPoolError(f"resident Writer exited during preload (code={code})")
        try:
            hello = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WriterPoolError("resident Writer emitted an invalid handshake") from exc
        if (
            not isinstance(hello, dict)
            or hello.get("type") != "hello"
            or hello.get("protocol") != PROTOCOL_VERSION
            or not hello.get("ok")
        ):
            raise WriterPoolError(
                "resident Writer preload or protocol validation failed"
            )
        self.hello = hello

    def request(self, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        with self._request_lock:
            process = self.process
            if (
                process is None
                or process.stdin is None
                or process.stdout is None
                or process.poll() is not None
            ):
                raise WriterPoolError("resident Writer is not alive")
            process.stdin.write(json.dumps(dict(payload), ensure_ascii=True) + "\n")
            process.stdin.flush()
            line = _readline(process.stdout, timeout)
            if not line:
                raise WriterPoolError("resident Writer exited before acknowledging operation")
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WriterPoolError("resident Writer emitted an invalid response") from exc
            if (
                not isinstance(response, dict)
                or response.get("type") != "result"
                or response.get("request_id") != payload.get("request_id")
                or response.get("request_sha256") != payload.get("request_sha256")
            ):
                raise WriterPoolError("resident Writer response identity mismatch")
            if not response.get("ok"):
                error_type = str(response.get("error_type") or "WriterError")
                if response.get("request_state") == "outcome_unknown":
                    raise WriterPoolOutcomeUnknown(
                        "resident Writer operation outcome is unknown"
                    )
                raise WriterPoolOperationError(
                    f"resident Writer operation failed ({error_type})",
                    error_type=error_type,
                )
            return response

    def stop(self, timeout: float = 5.0) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(
                        json.dumps(
                            {
                                "type": "shutdown",
                                "request_id": "shutdown-" + uuid.uuid4().hex,
                            }
                        )
                        + "\n"
                    )
                    process.stdin.flush()
                process.wait(timeout=timeout)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=timeout)
        if self.log is not None:
            self.log.close()
            self.log = None


class ResidentWriterPool:
    def __init__(
        self,
        *,
        size: int,
        python: Path,
        v4_root: Path,
        repo: Path,
        state_dir: Path,
        startup_timeout: float,
        request_timeout: float,
        control_db: Path,
        provider_key_concurrency: int,
        provider_lease_seconds: int,
    ) -> None:
        if size <= 0:
            raise WriterPoolError("resident Writer pool size must be positive")
        self.size = size
        self.python = python.absolute()
        self.v4_root = v4_root.resolve()
        self.repo = repo.resolve()
        self.log_root = state_dir.resolve() / "writer_pool"
        self.request_state_root = state_dir.resolve() / "writer_requests"
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        environment = dict(os.environ)
        environment["TMCRA_SERVICE_CONTROL_DB"] = str(control_db.resolve())
        environment["TMCRA_PROVIDER_KEY_CONCURRENCY"] = str(provider_key_concurrency)
        environment["TMCRA_PROVIDER_LEASE_SECONDS"] = str(provider_lease_seconds)
        environment["TMCRA_WRITER_REQUEST_STATE_DIR"] = str(self.request_state_root)
        python_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(self.v4_root) + (
            os.pathsep + python_path if python_path else ""
        )
        self.environment = environment
        self._available: queue.Queue[_WriterProcess] = queue.Queue(maxsize=size)
        self._available_workers: set[_WriterProcess] = set()
        self._workers: list[_WriterProcess] = []
        self._lock = threading.Lock()
        self._leased: set[_WriterProcess] = set()
        self._started = False
        self._stopping = False
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._request_gates: dict[str, _RequestGate] = {}
        self._request_gates_lock = threading.Lock()

    @contextmanager
    def _request_gate(self, request_id: str) -> Iterator[None]:
        with self._request_gates_lock:
            gate = self._request_gates.get(request_id)
            if gate is None:
                gate = _RequestGate(lock=threading.Lock())
                self._request_gates[request_id] = gate
            gate.users += 1
        gate.lock.acquire()
        try:
            yield
        finally:
            gate.lock.release()
            with self._request_gates_lock:
                gate.users -= 1
                if gate.users == 0:
                    self._request_gates.pop(request_id, None)

    def _status(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        path = request_status_path(
            self.request_state_root, str(payload["request_id"])
        )
        return read_request_status(
            path,
            request_id=str(payload["request_id"]),
            request_sha256=str(payload["request_sha256"]),
        )

    @staticmethod
    def _terminal_response(status: Mapping[str, Any]) -> dict[str, Any] | None:
        state = str(status.get("state") or "")
        if state == "succeeded":
            response = status.get("response")
            if not isinstance(response, Mapping) or not response.get("ok"):
                raise WriterPoolError("succeeded Writer receipt lacks a valid response")
            result = dict(response)
            result["recovered_from_receipt"] = True
            return result
        if state == "failed":
            response = status.get("response")
            error_type = (
                str(response.get("error_type") or "WriterError")
                if isinstance(response, Mapping)
                else "WriterError"
            )
            raise WriterPoolOperationError(
                f"resident Writer operation previously failed ({error_type})",
                error_type=error_type,
            )
        if state in {"running", "outcome_unknown"}:
            raise WriterPoolOutcomeUnknown(
                f"resident Writer request is {state}; successful provider calls will not be replayed"
            )
        raise WriterPoolError("writer request receipt has an invalid state")

    def _mark_outcome_unknown(
        self, payload: Mapping[str, Any], prior: Mapping[str, Any] | None
    ) -> None:
        if prior is not None and str(prior.get("state") or "") in {
            "succeeded",
            "failed",
            "outcome_unknown",
        }:
            return
        now = time.time()
        response = {
            "type": "result",
            "request_id": payload["request_id"],
            "request_sha256": payload["request_sha256"],
            "request_state": "outcome_unknown",
            "ok": False,
            "error_type": "WriterProtocolOutcomeUnknown",
        }
        write_request_status(
            request_status_path(
                self.request_state_root, str(payload["request_id"])
            ),
            {
                "schema_version": REQUEST_STATUS_SCHEMA_VERSION,
                "protocol": PROTOCOL_VERSION,
                "request_id": payload["request_id"],
                "request_sha256": payload["request_sha256"],
                "state": "outcome_unknown",
                "contract": dict(payload["request_contract"]),
                "worker_pid": prior.get("worker_pid") if prior else None,
                "accepted_at": prior.get("accepted_at", now) if prior else now,
                "updated_at": now,
                "completed_at": now,
                "response": response,
            },
        )

    def _recover_protocol_failure(
        self, worker: _WriterProcess, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        deadline = time.monotonic() + min(5.0, max(0.25, self.request_timeout))
        status: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            status = self._status(payload)
            if status is not None and str(status.get("state") or "") != "running":
                result = self._terminal_response(status)
                if result is not None:
                    return result
            process = getattr(worker, "process", None)
            if process is not None and process.poll() is not None:
                break
            time.sleep(0.05)

        # Stop first, then inspect the durable receipt. This ordering prevents
        # overwriting a success that the daemon was concurrently committing.
        worker.stop()
        status = self._status(payload)
        if status is not None and str(status.get("state") or "") != "running":
            result = self._terminal_response(status)
            if result is not None:
                return result
        self._mark_outcome_unknown(payload, status)
        raise WriterPoolOutcomeUnknown(
            "resident Writer connection failed before a terminal receipt; "
            "successful provider calls will not be replayed"
        )

    def _new_worker(self, index: int) -> _WriterProcess:
        return _WriterProcess(
            index=index,
            python=self.python,
            v4_root=self.v4_root,
            repo=self.repo,
            log_root=self.log_root,
            environment=self.environment,
        )

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._stopping = False
            workers = [self._new_worker(index) for index in range(self.size)]
            self._workers = workers
        try:
            for worker in workers:
                worker.launch()
            deadline = time.monotonic() + self.startup_timeout
            for worker in workers:
                worker.await_ready(max(0.1, deadline - time.monotonic()))
            with self._lock:
                self._started = True
            for worker in workers:
                self._enqueue_available(worker)
            self._monitor_stop.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor,
                name="tmcra-resident-writer-monitor",
                daemon=True,
            )
            self._monitor_thread.start()
        except Exception:
            for worker in workers:
                worker.stop()
            with self._lock:
                self._workers = []
                self._started = False
            raise

    def _enqueue_available(self, worker: _WriterProcess) -> None:
        with self._lock:
            current = (
                self._workers[worker.index]
                if worker.index < len(self._workers)
                else None
            )
            if (
                self._stopping
                or not self._started
                or current is not worker
                or worker in self._leased
                or worker in self._available_workers
                or not worker.alive
            ):
                return
            self._available_workers.add(worker)
        try:
            self._available.put_nowait(worker)
        except queue.Full:
            with self._lock:
                self._available_workers.discard(worker)
            raise WriterPoolError("resident Writer availability queue overflow")

    def execute(
        self,
        values: Mapping[str, Any],
        *,
        operation_timeout: float | None = None,
    ) -> dict[str, Any]:
        if not self._started:
            raise WriterPoolError("resident Writer pool is not started")
        payload = dict(values)
        payload["type"] = "execute"
        request_id, request_sha256, contract = request_identity(payload)
        payload["request_id"] = request_id
        payload["request_sha256"] = request_sha256
        payload["request_contract"] = contract
        with self._request_gate(request_id):
            prior = self._status(payload)
            if prior is not None:
                recovered = self._terminal_response(prior)
                if recovered is not None:
                    return recovered

            deadline = time.monotonic() + self.request_timeout
            while True:
                try:
                    worker = self._available.get(
                        timeout=max(0.01, deadline - time.monotonic())
                    )
                except queue.Empty as exc:
                    raise WriterPoolError("resident Writer pool is saturated") from exc
                with self._lock:
                    self._available_workers.discard(worker)
                    current = (
                        self._workers[worker.index]
                        if worker.index < len(self._workers)
                        else None
                    )
                    if current is worker and worker not in self._leased:
                        self._leased.add(worker)
                        break
                if time.monotonic() >= deadline:
                    raise WriterPoolError("resident Writer pool is saturated")
            healthy = True
            try:
                timeout = (
                    self.request_timeout
                    if operation_timeout is None
                    else float(operation_timeout)
                )
                if timeout <= 0:
                    raise WriterPoolError(
                        "resident Writer operation timeout must be positive"
                    )
                wire_payload = dict(payload)
                wire_payload.pop("request_contract", None)
                return worker.request(wire_payload, timeout)
            except (WriterPoolOperationError, WriterPoolOutcomeUnknown):
                raise
            except Exception:
                healthy = False
                return self._recover_protocol_failure(worker, payload)
            finally:
                with self._lock:
                    self._leased.discard(worker)
                if healthy and worker.alive and not self._stopping:
                    self._enqueue_available(worker)
                elif not healthy:
                    try:
                        self._replace_worker(worker)
                    except Exception:
                        # Readiness goes false if replacement also fails. The
                        # original operation is never replayed.
                        pass

    def _discard_available(self, target: _WriterProcess) -> None:
        retained: list[_WriterProcess] = []
        while True:
            try:
                worker = self._available.get_nowait()
            except queue.Empty:
                break
            self._available_workers.discard(worker)
            if worker is not target:
                retained.append(worker)
        for worker in retained:
            self._available_workers.add(worker)
            self._available.put_nowait(worker)

    def _replace_worker(self, prior: _WriterProcess) -> None:
        with self._lock:
            if self._stopping or not self._started:
                return
            if prior in self._leased:
                return
            if self._workers[prior.index] is not prior:
                return
            self._discard_available(prior)
            replacement = self._new_worker(prior.index)
            self._workers[prior.index] = replacement
        try:
            replacement.launch()
            replacement.await_ready(self.startup_timeout)
        except Exception:
            replacement.stop()
            raise
        if not self._stopping:
            self._enqueue_available(replacement)

    def _repair_dead_workers(self) -> None:
        with self._lock:
            candidates = [
                worker
                for worker in self._workers
                if not worker.alive and worker not in self._leased
            ]
        for worker in candidates:
            try:
                self._replace_worker(worker)
            except Exception:
                # Keep retrying at the monitor interval. Readiness remains false
                # until all configured workers have completed a fresh handshake.
                pass

    def _repair_lost_available_workers(self) -> None:
        with self._lock:
            candidates = [
                worker
                for worker in self._workers
                if worker.alive
                and worker not in self._leased
                and worker not in self._available_workers
            ]
        for worker in candidates:
            self._enqueue_available(worker)

    def _monitor(self) -> None:
        while not self._monitor_stop.wait(1.0):
            self._repair_dead_workers()
            self._repair_lost_available_workers()

    def status(self) -> WriterPoolStatus:
        with self._lock:
            workers = tuple(self._workers)
            available = sum(worker in self._available_workers for worker in workers)
            leased = sum(worker in self._leased for worker in workers)
        pids = tuple(worker.pid for worker in workers if worker.alive and worker.pid is not None)
        return WriterPoolStatus(
            configured=self.size,
            ready=len(pids),
            alive=(
                self._started
                and len(pids) == self.size
                and available + leased == self.size
            ),
            pids=pids,
            protocol=PROTOCOL_VERSION,
            available=available,
            leased=leased,
        )

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            self._monitor_stop.set()
            workers = list(self._workers)
            self._started = False
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=max(2.0, self.startup_timeout + 1.0))
            self._monitor_thread = None
        for worker in workers:
            worker.stop()
        with self._lock:
            self._workers = []
            self._leased.clear()
            self._available_workers.clear()
            self._request_gates.clear()
            while True:
                try:
                    self._available.get_nowait()
                except queue.Empty:
                    break
