"""Local model installer and supervisor; uses the production memory API unchanged.

Downloads require the network only during `prepare`. `run` installs the full-local
Python boundary before starting the API and verifies all pinned model weights.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from tmcra_local_models import profile_by_id, profiles, sha256_file, signature, verify_weights
from tmcra_local_only import configure_routes, process_lock, read_environment, validate_environment

API_ROOT = Path(__file__).resolve().parents[1]
GENERATION = {
    "repo_id": "Qwen/Qwen3-4B-GGUF", "revision": "bc640142c66e1fdd12af0bd68f40445458f3869b",
    "license": "Apache-2.0", "model": "tmcra-qwen3-4b-q4km",
    "weights": [{"file": "Qwen3-4B-Q4_K_M.gguf", "bytes": 2497280256,
                 "sha256": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"}],
}
LLAMA = {
    "version": "b10276", "archive": "llama-b10276-bin-win-cpu-x64.zip",
    "sha256": "b1db7fc5b3d2728dcead5b792b0565da045dec688df81c9272ce5aef5f55a3e8",
    "url": "https://github.com/ggml-org/llama.cpp/releases/download/b10276/llama-b10276-bin-win-cpu-x64.zip",
}


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + secrets.token_hex(4))
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def private_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        result = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], check=True,
                                capture_output=True, text=True)
        sid = list(csv.reader(io.StringIO(result.stdout)))[0][-1]
        if not sid.startswith("S-1-"):
            raise RuntimeError("cannot determine current Windows user SID")
        subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r",
                        f"*{sid}:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F"],
                       check=True, capture_output=True)
    else:
        path.chmod(0o700)


def hardware():
    import psutil
    import torch
    ram = psutil.virtual_memory()
    gpu = torch.cuda.is_available()
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu else 0
    recommended = "lite-cpu"
    if ram.total / 1024**3 >= 30 and vram >= 6:
        recommended = "balanced-bge"
    if ram.total / 1024**3 >= 60 and vram >= 16:
        recommended = "quality-qwen"
    return {"ram_gib": round(ram.total / 1024**3, 1), "available_ram_gib": round(ram.available / 1024**3, 1),
            "cuda_available": gpu, "vram_gib": round(vram, 1), "recommended_profile": recommended,
            "recommendation_basis": "capacity_estimate_not_quality_benchmark"}


def download_model(model, directory):
    directory.mkdir(parents=True, exist_ok=True)
    try:
        verify_weights(model, directory)
        if model is GENERATION or (directory / "config.json").is_file():
            emit("model_cached", repo=model["repo_id"])
            return
    except RuntimeError:
        pass
    hf = shutil.which("hf")
    if not hf:
        raise RuntimeError("Hugging Face CLI missing; run the local setup script first")
    includes = [item["file"] for item in model["weights"]]
    if model is not GENERATION:
        includes += ["*.json", "*.model", "vocab.txt", "merges.txt", "README.md", "LICENSE*"]
    emit("downloading", repo=model["repo_id"], revision=model["revision"])
    subprocess.run([hf, "download", model["repo_id"], "--revision", model["revision"],
                    "--local-dir", str(directory), "--include", *includes], check=True)
    verify_weights(model, directory)


def install_llama(root):
    if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("this portable launcher currently supports Windows x64; other platforms remain unvalidated")
    directory = root / "runtime" / LLAMA["version"]
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / LLAMA["archive"]
    if not archive.is_file() or sha256_file(archive) != LLAMA["sha256"]:
        emit("downloading_runtime", version=LLAMA["version"])
        partial = archive.with_suffix(".partial")
        with urllib.request.urlopen(LLAMA["url"], timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if sha256_file(partial) != LLAMA["sha256"]:
            raise RuntimeError("llama.cpp archive failed SHA-256 verification")
        os.replace(partial, archive)
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            target = (directory / item.filename).resolve()
            if not target.is_relative_to(directory.resolve()) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError("unsafe path in llama.cpp archive")
        zipped.extractall(directory)
    servers = list(directory.rglob("llama-server.exe"))
    if len(servers) != 1:
        raise RuntimeError("llama.cpp archive must contain exactly one server")
    return servers[0]


def initialize_identity(state):
    from .auth import APIKeyAuth
    from .cli import DEFAULT_SCOPES
    from .control_db import ControlDB
    private_directory(state)
    secrets_dir = state / "secrets"
    private_directory(secrets_dir)
    auth = APIKeyAuth(ControlDB(state / "control.sqlite3"))
    credentials = secrets_dir / "client.json"
    if credentials.exists():
        saved = json.loads(credentials.read_text(encoding="utf-8"))
        auth.authenticate(saved["api_key"])
    else:
        tenant = "local-" + secrets.token_hex(12)
        auth.set_tenant_scopes(tenant, frozenset(DEFAULT_SCOPES))
        issued = auth.create_key(tenant)
        saved = {"schema_version": "tmcra.local-client.1", "tenant_id": tenant,
                 "api_key": issued.api_key, "key_id": issued.key_id, "scope": "personal"}
        atomic_json(credentials, saved)
    key_file = secrets_dir / "generation.key"
    if not key_file.exists():
        with key_file.open("x", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(48))
    return key_file


def prepare(root, profile_id, *, device="auto", api_port=2009, model_port=2010, auto_ports=False):
    if api_port == model_port or not all(1024 <= port <= 65535 for port in (api_port, model_port)):
        raise ValueError("choose two different non-privileged local ports")
    root = root.resolve()
    private_directory(root)
    with process_lock(root / "run.lock", timeout=0), process_lock(root / "install.lock", timeout=0):
        if auto_ports:
            # Hold both reservations while selecting distinct ports; run() checks again.
            with socket.socket() as first, socket.socket() as second:
                previous = root / "installation.json"
                ports = json.loads(previous.read_text(encoding="utf-8")) if previous.is_file() else {}
                for probe, name in ((first, "api_port"), (second, "model_port")):
                    preferred = ports.get(name, 0)
                    if not isinstance(preferred, int) or not 1024 <= preferred <= 65535:
                        preferred = 0
                    try:
                        probe.bind(("127.0.0.1", preferred))
                    except OSError:
                        probe.bind(("127.0.0.1", 0))
                api_port, model_port = first.getsockname()[1], second.getsockname()[1]
        profile = profile_by_id(profile_id)
        resources = hardware()
        minimum = profile["system_ram_gib_min"]
        if resources["ram_gib"] < minimum * 0.95:
            raise RuntimeError(f"profile requires at least {minimum} GiB physical RAM")
        free = shutil.disk_usage(root).free
        required = profile["weights_bytes"] + GENERATION["weights"][0]["bytes"] + 5 * 1024**3
        if free < required:
            raise RuntimeError("insufficient disk space for models, verification and memory state")
        device = ("cuda" if resources["cuda_available"] else "cpu") if device == "auto" else device
        if device == "cuda" and not resources["cuda_available"]:
            raise RuntimeError("CUDA requested but this Python runtime has no usable CUDA device")
        model_root = root / "models" / profile_id
        for role in ("embedding", "reranker"):
            download_model(profile[role], model_root / role)
            atomic_json(model_root / role / "TMCRA_MODEL_MANIFEST.json", profile[role])
        download_model(GENERATION, root / "models" / "generation")
        server = install_llama(root)
        state = root / "state" / profile_id
        key_file = initialize_identity(state)
        client = json.loads((state / "secrets/client.json").read_text(encoding="utf-8"))
        atomic_json(state / "secrets/client-plugin.json", {
            "schemaVersion": 2, "authMode": "api-key", "baseUrl": f"http://127.0.0.1:{api_port}",
            "apiKey": client["api_key"], "tokenType": "Bearer", "scopeNamespace": "local",
            "globalScope": "local-global", "projectScopePrefix": "local-project", "timeoutMs": 180000,
            "defaultScope": "personal",
            "deploymentMode": "local", "integrationIds": {},
        })
        config = state / "local-environment.json"
        settings = {
            "TMCRA_SERVICE_STATE_DIR": str(state), "TMCRA_SERVICE_CONTROL_DB": str(state / "control.sqlite3"),
            "TMCRA_SERVICE_BIND_HOST": "127.0.0.1", "TMCRA_SERVICE_BIND_PORT": str(api_port),
            "TMCRA_SERVICE_PUBLIC_BASE_URL": f"http://127.0.0.1:{api_port}",
            "TMCRA_SERVICE_DEVICE": device, "TMCRA_SERVICE_GRAPH_DEVICE": device,
            "TMCRA_SERVICE_WORKER_CONCURRENCY": "1", "TMCRA_SERVICE_REQUEST_MAX_CONCURRENCY": "4",
            "TMCRA_SERVICE_RECALL_POOL_MIN_SIZE": "1", "TMCRA_SERVICE_RECALL_POOL_MAX_SIZE": "1",
            "TMCRA_SERVICE_WRITER_EXECUTION_MODE": "resident", "TMCRA_SERVICE_WRITER_POOL_SIZE": "1",
            "TMCRA_SERVICE_LOCAL_WRITER_RECOVERY_CONCURRENCY": "1",
            "TMCRA_SERVICE_WRITER_POOL_REQUEST_TIMEOUT_SECONDS": "1800",
            "TMCRA_SERVICE_STARTUP_PREFLIGHT_MODE": "full", "TMCRA_SERVICE_RECALL_QUEUE_TIMEOUT_SECONDS": "600",
            "TMCRA_LOCAL_PROFILE": profile_id, "TMCRA_LOCAL_WRITER_API_KEY_FILE": str(key_file),
            "TMCRA_EMBEDDING_MODEL": str(model_root / "embedding"),
        }
        environment = configure_routes(settings, base_url=f"http://127.0.0.1:{model_port}/v1",
                                       model=GENERATION["model"], key=key_file.read_text().strip())
        environment.update({"TMCRA_V4_ROOT": str(API_ROOT), "TMCRA_INTEGRATED_REPO": str(API_ROOT),
                            "TMCRA_WRITER_ENV": str(config), "TMCRA_CROSS_MODEL": str(model_root / "reranker"),
                            "TMCRA_CHECKPOINT": str(API_ROOT / "models" / "tmcra_v3_reranker.pt"),
                            "TMCRA_LEARNED_GRAPH_ENABLED": "0", "PYTHONUTF8": "1",
                            "TMCRA_LOCAL_LLM_PARALLEL": "1", "TMCRA_PROJECTION_RESERVED_PRODUCTION_SLOTS": "0",
                            "TMCRA_SESSION_GRAPH_AGENT_TIMEOUT_SECONDS": "600",
                            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                            "PYTHONPATH": os.pathsep.join([str(API_ROOT / "deploy" / "local-bootstrap"), str(API_ROOT)])})
        validate_environment(environment)
        atomic_json(config, {"schema_version": "tmcra.local-environment.1", "environment": environment})
        # Receipt is public metadata only; credentials live under private state.
        receipt = {"schema_version": "tmcra.local-installation.1", "profile": profile_id,
                   "api_root": str(API_ROOT),
                   "embedding_signature": signature(profile), "environment_file": str(config),
                   "llama_server": str(server), "llama_sha256": sha256_file(server),
                   "model_port": model_port, "api_port": api_port, "hardware": resources,
                   "generation": GENERATION, "runtime": LLAMA,
                   "status": "installed_runtime_validation_required", "external_network_at_runtime": "python_loopback_only"}
        atomic_json(root / "installation.json", receipt)
        emit("installed", profile=profile_id, root=str(root), full_pipeline_verified=False)
        return receipt


def port_free(port):
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"local port {port} is occupied; existing services were left running") from exc


def run(root):
    import psutil
    root = root.resolve()
    with process_lock(root / "run.lock", timeout=0):
        receipt = json.loads((root / "installation.json").read_text(encoding="utf-8"))
        profile = profile_by_id(receipt["profile"])
        if receipt["embedding_signature"] != signature(profile):
            raise RuntimeError("installed embedding contract changed; prepare a new profile/state first")
        for role in ("embedding", "reranker"):
            verify_weights(profile[role], root / "models" / profile["id"] / role)
        verify_weights(GENERATION, root / "models" / "generation")
        if sha256_file(receipt["llama_server"]) != receipt["llama_sha256"]:
            raise RuntimeError("local generation executable changed")
        env = read_environment(receipt["environment_file"])
        for port in (receipt["api_port"], receipt["model_port"]):
            port_free(port)
        available = psutil.virtual_memory().available
        # Reserve OS/application headroom as well as model + KV + Python memory.
        required_available = max(6 * 1024**3, int(profile["weights_bytes"] * 1.4 + 5 * 1024**3))
        if available < required_available:
            raise RuntimeError(f"insufficient available memory: {available / 1024**3:.1f} GiB available; "
                               f"{required_available / 1024**3:.1f} GiB required before starting; close other workloads and retry")
        state = Path(env["TMCRA_SERVICE_STATE_DIR"])
        logs = state / "logs"
        logs.mkdir(exist_ok=True)
        children = []
        handles = []
        run_id = secrets.token_hex(16)
        try:
            commands = [
                [receipt["llama_server"], "--model", str(root / "models" / "generation" / GENERATION["weights"][0]["file"]),
                 "--alias", GENERATION["model"], "--host", "127.0.0.1", "--port", str(receipt["model_port"]),
                 "--api-key-file", env["TMCRA_LOCAL_WRITER_API_KEY_FILE"], "--ctx-size", "32768",
                 "--parallel", "1", "--threads", str(min(8, os.cpu_count() or 4)),
                 "--batch-size", "128", "--ubatch-size", "64", "--cache-ram", "0",
                 "--n-gpu-layers", "0", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                 "--flash-attn", "on", "--jinja", "--reasoning-budget", "0"],
                [sys.executable, "-m", "tmcra_service"],
            ]
            for index, command in enumerate(commands):
                handle = (logs / ("generation.log" if index == 0 else "api.log")).open("ab")
                handles.append(handle)
                child = subprocess.Popen(command, cwd=API_ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=handle, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                children.append(child)
                if index == 0:
                    deadline = time.monotonic() + 180
                    while True:
                        if child.poll() is not None:
                            raise RuntimeError("generation service exited; inspect its local log")
                        try:
                            request = urllib.request.Request(f"http://127.0.0.1:{receipt['model_port']}/health")
                            with urllib.request.urlopen(request, timeout=2) as response:
                                if response.status == 200:
                                    break
                        except (OSError, TimeoutError):
                            pass
                        if time.monotonic() >= deadline:
                            raise RuntimeError("generation service did not become healthy within 180 seconds")
                        time.sleep(0.5)
            atomic_json(root / "running.json", {"run_id": run_id, "supervisor_pid": os.getpid(),
                        "supervisor_created": psutil.Process().create_time(), "api_port": receipt["api_port"],
                        "profile": profile["id"], "pids": [child.pid for child in children]})
            emit("started", api_url=f"http://127.0.0.1:{receipt['api_port']}", run_id=run_id)
            while all(child.poll() is None for child in children):
                stop = root / "stop-request.json"
                if stop.exists() and json.loads(stop.read_text(encoding="utf-8"))["run_id"] == run_id:
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("a local service exited; inspect private logs")
        finally:
            for child in reversed(children):
                if child.poll() is None:
                    descendants = psutil.Process(child.pid).children(recursive=True)
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                    for descendant in reversed(descendants):
                        try:
                            descendant.terminate()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    _, survivors = psutil.wait_procs(descendants, timeout=5)
                    for descendant in survivors:
                        try:
                            descendant.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            for handle in handles:
                handle.close()
            atomic_json(root / "running.json", {"run_id": run_id, "stopped": True})


def main(argv=None):
    parser = argparse.ArgumentParser(description="TMCRA full-local Windows deployment")
    parser.add_argument("command", choices=["recommend", "prepare", "run", "stop", "status"])
    parser.add_argument("--root", type=Path, default=Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "TMCRA" / "local")
    parser.add_argument("--profile", choices=[p["id"] for p in profiles()], default="lite-cpu")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--api-port", type=int, default=2009)
    parser.add_argument("--model-port", type=int, default=2010)
    parser.add_argument("--auto-ports", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "recommend":
        emit("recommendation", **hardware(), profiles=profiles())
    elif args.command == "prepare":
        prepare(args.root, args.profile, device=args.device, api_port=args.api_port, model_port=args.model_port, auto_ports=args.auto_ports)
    elif args.command == "run":
        try:
            run(args.root)
        except Exception as exc:
            atomic_json(args.root / "launch-error.json", {"error_type": type(exc).__name__,
                        "detail": str(exc), "at": time.time()})
            raise
    else:
        running = args.root / "running.json"
        data = json.loads(running.read_text(encoding="utf-8")) if running.exists() else {"stopped": True}
        if args.command == "stop" and not data.get("stopped"):
            import psutil
            process = psutil.Process(data["supervisor_pid"])
            if abs(process.create_time() - data["supervisor_created"]) > 0.01:
                raise RuntimeError("stored supervisor PID was reused; refusing to stop")
            atomic_json(args.root / "stop-request.json", {"run_id": data["run_id"]})
            emit("stop_requested")
        else:
            emit("status", **data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
