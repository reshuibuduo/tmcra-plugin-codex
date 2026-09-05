"""Loaded by every Python worker in the explicit full-local launch environment."""
import os

if os.environ.get("TMCRA_DEPLOYMENT_MODE") == "local":
    try:
        from tmcra_local_only import install_network_guard, validate_environment
        validate_environment(os.environ)
        install_network_guard()
    except Exception:
        # CPython normally ignores sitecustomize exceptions. Fail closed instead.
        os.write(2, b"TMCRA local network boundary failed; refusing to start.\n")
        os._exit(78)
