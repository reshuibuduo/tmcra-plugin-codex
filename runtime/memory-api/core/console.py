from __future__ import annotations

import os
import sys


def configure_stdio_utf8() -> None:
    """Prefer UTF-8 stdio so Windows terminals don't garble Chinese output."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue
