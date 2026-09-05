"""Explicit full-local runtime boundary, also used by Python worker processes.

Installation/downloads run outside this boundary. This is a process-level guard,
not an OS firewall or a sandbox for untrusted plugins/native code.
"""
from __future__ import annotations

import ipaddress
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

_guard_installed = False
ROLES = ("WRITER", "WRITER_REVIEWER", "RECALL_PLANNER", "SLOW_GRAPH",
         "SUBJECT_ATTRIBUTION", "EVIDENCE_COMPILER", "SESSION_GRAPH")


def enabled(environment=None):
    return (os.environ if environment is None else environment).get("TMCRA_DEPLOYMENT_MODE") == "local"


def loopback_url(value, *, port=None, path=None):
    parsed = urlsplit(value)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        valid_port = parsed.port
    except ValueError as exc:
        raise ValueError("local URLs require a numeric loopback address and valid port") from exc
    if (parsed.scheme != "http" or not address.is_loopback or not valid_port
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment
            or (port is not None and valid_port != int(port))
            or (path is not None and parsed.path.rstrip("/") != path)):
        raise ValueError("full-local URLs must use the configured loopback HTTP endpoint")
    return parsed


def validate_environment(environment):
    if not enabled(environment):
        raise ValueError("full-local deployment mode is required")
    generation = loopback_url(environment["TMCRA_LOCAL_WRITER_BASE_URL"], path="/v1")
    public = loopback_url(environment["TMCRA_SERVICE_PUBLIC_BASE_URL"],
                          port=environment["TMCRA_SERVICE_BIND_PORT"], path="")
    if public.hostname != environment.get("TMCRA_SERVICE_BIND_HOST"):
        raise ValueError("local service URL and bind host differ")
    for role in ROLES:
        loopback_url(environment[f"TMCRA_{role}_BASE_URL"], port=generation.port, path="/v1")
        if environment[f"TMCRA_{role}_BASE_URL"] != environment["TMCRA_LOCAL_WRITER_BASE_URL"]:
            raise ValueError(f"local model route differs: {role}")
        if not environment.get(f"TMCRA_{role}_MODEL"):
            raise ValueError(f"local model identity is missing: {role}")
    for name, value in environment.items():
        if name.lower() in {"http_proxy", "https_proxy", "all_proxy"} and value:
            raise ValueError("proxies must be cleared in full-local mode")
        if name.startswith("TMCRA_") and name.endswith(("_BASE_URL", "_ENDPOINT", "_URL")) and value:
            loopback_url(value)
    return environment


def configure_routes(environment, *, base_url, model, key):
    """Return a private child environment; never mutate host/cloud credentials."""
    loopback_url(base_url, path="/v1")
    if not model or not key or any(c in key for c in ",\r\n"):
        raise ValueError("one local model alias and API key are required")
    clean = {}
    for name, value in environment.items():
        upper = name.upper()
        if (upper.startswith(("TMCRA_", "OPENAI_", "ANTHROPIC_", "DEEPSEEK_", "ARK_", "DASHSCOPE_"))
                or upper in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}):
            continue
        clean[name] = value
    # Keep service settings explicitly selected by the local installer only.
    clean.update({name: value for name, value in environment.items()
                  if name.startswith(("TMCRA_SERVICE_", "TMCRA_LOCAL_", "TMCRA_EMBEDDING_"))})
    clean.update({"TMCRA_DEPLOYMENT_MODE": "local", "HF_HUB_OFFLINE": "1",
                  "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                  "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "NO_PROXY": "*",
                  "TMCRA_LOCAL_WRITER_BASE_URL": base_url, "TMCRA_LOCAL_WRITER_MODEL": model})
    for role in ROLES:
        clean.update({f"TMCRA_{role}_BASE_URL": base_url, f"TMCRA_{role}_MODEL": model,
                      f"TMCRA_{role}_API_KEY_POOL": key, f"TMCRA_{role}_KEY_POOL": key,
                      f"TMCRA_{role}_PROVIDER": "local-qwen"})
    clean.update({"TMCRA_WRITER_PROMPT_ADAPTER": "qwen36-v5",
                  "TMCRA_WRITER_REVIEWER_PROMPT_ADAPTER": "qwen36-reconciliation-v1",
                  "TMCRA_RECALL_PLANNER_PROMPT_ADAPTER": "qwen36-planner-v1",
                  "TMCRA_SLOW_GRAPH_PROMPT_ADAPTER": "qwen36-slow-graph-v1",
                  "TMCRA_WRITER_MAX_TOKENS": "16384",
                  "TMCRA_SLOW_GRAPH_MAX_TOKENS": "16384",
                  "TMCRA_RECALL_PLANNER_MAX_TOKENS": "512",
                  "TMCRA_RECALL_PLANNER_TIMEOUT_SECONDS": "600"})
    # Older core modules read these aliases. They contain ONLY the local endpoint
    # and its generated key, with zero cloud prices and no inherited cloud pool.
    for role in ("DEEPSEEK_WRITER", "DEEPSEEK_FLASH", "DEEPSEEK_PRO"):
        clean.update({f"TMCRA_{role}_BASE_URL": base_url, f"TMCRA_{role}_MODEL": model,
                      f"TMCRA_{role}_KEY_POOL": key, f"TMCRA_{role}_MAX_TOKENS": "16384"})
        for cost in ("PROMPT", "COMPLETION", "CACHE"):
            clean[f"TMCRA_{role}_{cost}_COST_PER_MILLION"] = "0"
    return clean


def read_environment(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != "tmcra.local-environment.1":
        raise ValueError("invalid local environment schema")
    values = data.get("environment")
    if not isinstance(values, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in values.items()):
        raise ValueError("local environment must be a string mapping")
    # No shell expansion or executable configuration.
    result = dict(os.environ)
    operation_names = {"TMCRA_SERVICE_TENANT_ID", "TMCRA_SERVICE_SCOPE_NAME", "TMCRA_SERVICE_JOB_ID",
                       "TMCRA_SERVICE_STAGE_ID", "TMCRA_SERVICE_STAGE_ATTEMPT", "TMCRA_USAGE_ATTRIBUTION_JSON"}
    operation = {name: result[name] for name in operation_names if name in result}
    for name in list(result):
        if name.upper().startswith(("TMCRA_", "OPENAI_", "ANTHROPIC_", "DEEPSEEK_", "ARK_", "DASHSCOPE_")) or name.upper() in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
            result.pop(name)
    result.update(values)
    result.update(operation)
    validate_environment(result)
    return result


def install_network_guard():
    global _guard_installed
    if _guard_installed:
        return

    def check_host(host):
        if host in (None, ""):
            return
        if isinstance(host, bytes):
            host = host.decode("ascii", errors="strict")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            raise PermissionError("TMCRA full-local mode blocks DNS and non-numeric network hosts") from None
        if not address.is_loopback:
            raise PermissionError("TMCRA full-local mode blocks external network connections")

    def audit(event, args):
        if event == "socket.getaddrinfo":
            check_host(args[0])
        elif event in {"socket.gethostbyname", "socket.gethostbyaddr"}:
            check_host(args[0])
        elif event in {"socket.connect", "socket.bind"}:
            address = args[1]
            if isinstance(address, tuple):
                check_host(address[0])
        elif event == "socket.sendto":
            address = args[-1]
            if isinstance(address, tuple):
                check_host(address[0])

    sys.addaudithook(audit)
    _guard_installed = True


@contextmanager
def process_lock(path, *, timeout=60):
    """Cooperating process lock on Windows and POSIX; state survives crashes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        if os.name == "nt":
            import msvcrt
            deadline = time.monotonic() + timeout
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("another local process holds the state lock") from None
                    time.sleep(0.1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
