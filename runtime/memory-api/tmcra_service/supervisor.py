from __future__ import annotations

import argparse
import errno
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .__main__ import (
    DEFAULT_WRITER_ENV,
    _configure_writer_aliases,
    _load_shell_environment,
    _validate_startup,
)
from .settings import ServiceSettings


DEFAULT_SERVICE_ENV = "/opt/tmcra/deploy/tmcra-service.env"
PROCESS_GROUP_TERM_TIMEOUT_SECONDS = 5.0
PROCESS_GROUP_KILL_TIMEOUT_SECONDS = 5.0
PROCESS_GROUP_POLL_INTERVAL_SECONDS = 0.05
SERVICE_WAIT_POLL_INTERVAL_SECONDS = 0.25
PROCESS_GROUP_KILL_SIGNAL = getattr(signal, "SIGKILL", 9)


class Supervisor:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir.resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.stop_requested = False
        self.child: subprocess.Popen[bytes] | None = None
        self._term_sent_child: subprocess.Popen[bytes] | None = None

    def request_stop(self, signum: int, frame: object) -> None:
        self.stop_requested = True
        # Do not wait from a signal handler: the main thread may already hold
        # Popen's wait lock. The bounded group cleanup runs in the main path.
        if self.child is not None:
            self._send_term(self.child)

    @staticmethod
    def _safe_child_group_id(child: subprocess.Popen[bytes]) -> int | None:
        try:
            group_id = int(child.pid)
        except (TypeError, ValueError):
            return None
        if group_id <= 1 or group_id == os.getpid():
            return None

        getpgrp = getattr(os, "getpgrp", None)
        if getpgrp is not None:
            try:
                if group_id == int(getpgrp()):
                    return None
            except (OSError, TypeError, ValueError):
                pass
        return group_id

    @staticmethod
    def _group_exists(group_id: int, killpg: object) -> bool:
        try:
            killpg(group_id, 0)  # type: ignore[operator]
        except ProcessLookupError:
            return False
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return False
        return True

    @staticmethod
    def _signal_group(group_id: int, signum: int, killpg: object) -> bool:
        try:
            killpg(group_id, signum)  # type: ignore[operator]
        except ProcessLookupError:
            return False
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return False
            raise
        return True

    def _signal_process_group(
        self, child: subprocess.Popen[bytes], signum: int
    ) -> bool:
        group_id = self._safe_child_group_id(child)
        if group_id is None:
            return False

        killpg = getattr(os, "killpg", None)
        if killpg is not None:
            return self._signal_group(group_id, signum, killpg)

        # Windows does not expose os.killpg. The fallback still gives the
        # child a bounded terminate/kill lifecycle; POSIX uses the full group.
        try:
            if signum == signal.SIGTERM:
                child.terminate()
            elif signum == PROCESS_GROUP_KILL_SIGNAL:
                child.kill()
        except ProcessLookupError:
            return False
        return True

    def _send_term(self, child: subprocess.Popen[bytes]) -> None:
        if self._term_sent_child is child:
            return
        self._signal_process_group(child, signal.SIGTERM)
        self._term_sent_child = child

    @staticmethod
    def _wait_for_process(process: subprocess.Popen[bytes], timeout: float) -> None:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return

    def _wait_for_group_exit(
        self,
        child: subprocess.Popen[bytes],
        group_id: int,
        killpg: object,
        timeout: float,
    ) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if child.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                try:
                    child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    return False

            if not self._group_exists(group_id, killpg):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(PROCESS_GROUP_POLL_INTERVAL_SECONDS, remaining))

    @staticmethod
    def _cleanup_single_process(
        child: subprocess.Popen[bytes],
    ) -> None:
        if child.poll() is not None:
            return
        try:
            child.terminate()
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=PROCESS_GROUP_TERM_TIMEOUT_SECONDS)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            child.kill()
        except ProcessLookupError:
            return
        Supervisor._wait_for_process(child, PROCESS_GROUP_KILL_TIMEOUT_SECONDS)

    def _cleanup_process_group(
        self,
        child: subprocess.Popen[bytes] | None,
    ) -> None:
        if child is None:
            return
        group_id = self._safe_child_group_id(child)
        if group_id is None:
            return

        killpg = getattr(os, "killpg", None)
        if killpg is None:
            self._cleanup_single_process(child)
            return
        if not self._group_exists(group_id, killpg):
            return

        self._send_term(child)
        if self._wait_for_group_exit(
            child,
            group_id,
            killpg,
            PROCESS_GROUP_TERM_TIMEOUT_SECONDS,
        ):
            return

        self._signal_process_group(child, PROCESS_GROUP_KILL_SIGNAL)
        self._wait_for_group_exit(
            child,
            group_id,
            killpg,
            PROCESS_GROUP_KILL_TIMEOUT_SECONDS,
        )

    def _start_service(self) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [sys.executable, "-m", "tmcra_service"],
            cwd=str(Path(__file__).resolve().parent.parent),
            start_new_session=True,
        )

    @staticmethod
    def _return_code_after_cleanup(child: subprocess.Popen[bytes]) -> int:
        code = child.poll()
        if code is not None:
            return code
        return -PROCESS_GROUP_KILL_SIGNAL

    def _run_service_once(self, service_pid: Path) -> int:
        child = self._start_service()
        self.child = child
        try:
            service_pid.write_text(str(child.pid) + "\n", encoding="utf-8")
            while True:
                try:
                    return child.wait(timeout=SERVICE_WAIT_POLL_INTERVAL_SECONDS)
                except subprocess.TimeoutExpired:
                    if self.stop_requested:
                        self._cleanup_process_group(child)
                        return self._return_code_after_cleanup(child)
        finally:
            try:
                self._cleanup_process_group(child)
            finally:
                if self.child is child:
                    self.child = None
                if self._term_sent_child is child:
                    self._term_sent_child = None
                service_pid.unlink(missing_ok=True)

    def run(self) -> int:
        import fcntl

        lock_path = self.state_dir / "supervisor.lock"
        with lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("TMCRA service supervisor is already running") from exc
            supervisor_pid = self.state_dir / "supervisor.pid"
            service_pid = self.state_dir / "service.pid"
            supervisor_pid.write_text(str(os.getpid()) + "\n", encoding="utf-8")
            signal.signal(signal.SIGTERM, self.request_stop)
            signal.signal(signal.SIGINT, self.request_stop)
            backoff = 1.0
            try:
                while not self.stop_requested:
                    started = time.monotonic()
                    code = self._run_service_once(service_pid)
                    if self.stop_requested:
                        return 0
                    uptime = time.monotonic() - started
                    if uptime >= 300:
                        backoff = 1.0
                    next_backoff = min(30.0, backoff * 2.0)
                    with (self.state_dir / "supervisor_restarts.log").open(
                        "a", encoding="utf-8"
                    ) as log:
                        log.write(
                            f"at={time.time():.3f} exit_code={code} uptime={uptime:.3f} "
                            f"next_backoff={next_backoff:.3f}\n"
                        )
                    time.sleep(backoff)
                    backoff = next_backoff
            finally:
                service_pid.unlink(missing_ok=True)
                supervisor_pid.unlink(missing_ok=True)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervise the TMCRA Memory API")
    parser.add_argument(
        "--env-file",
        default=os.getenv("TMCRA_SERVICE_ENV_FILE", DEFAULT_SERVICE_ENV),
        help="shell environment file for the service deployment",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.umask(0o077)
    _load_shell_environment(args.env_file)
    writer_env = os.getenv("TMCRA_WRITER_ENV", DEFAULT_WRITER_ENV)
    _load_shell_environment(writer_env)
    _configure_writer_aliases()
    settings = ServiceSettings.from_env()
    _validate_startup(settings)
    return Supervisor(settings.state_dir).run()


if __name__ == "__main__":
    raise SystemExit(main())
