from __future__ import annotations

import http.client
import json
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


class AudioAsrProxyError(RuntimeError):
    code = "audio_asr_unavailable"


class AudioAsrProxyDisabled(AudioAsrProxyError):
    code = "audio_asr_not_configured"


class AudioAsrProxyTimeout(AudioAsrProxyError):
    code = "audio_asr_timeout"


@dataclass(frozen=True)
class AudioAsrReply:
    status_code: int
    body: bytes
    content_type: str
    retry_after: str | None = None


class AudioAsrProxy:
    """Bounded loopback proxy for the isolated GPU ASR worker."""

    def __init__(
        self,
        *,
        base_url: str | None,
        api_key_file: Path | None,
        timeout_seconds: float,
        maximum_request_bytes: int,
        maximum_response_bytes: int = 512 * 1024,
    ) -> None:
        self.base_url = str(base_url or "").strip().rstrip("/") or None
        self.api_key_file = api_key_file.resolve() if api_key_file else None
        self.timeout_seconds = float(timeout_seconds)
        self.maximum_request_bytes = int(maximum_request_bytes)
        self.maximum_response_bytes = int(maximum_response_bytes)
        if self.timeout_seconds <= 0:
            raise ValueError("audio ASR timeout must be positive")
        if self.maximum_request_bytes <= 0 or self.maximum_response_bytes <= 0:
            raise ValueError("audio ASR byte limits must be positive")
        if (self.base_url is None) != (self.api_key_file is None):
            raise ValueError("audio ASR endpoint and key file must be configured together")
        if self.base_url is not None:
            parsed = urlsplit(self.base_url)
            if (
                parsed.scheme != "http"
                or parsed.hostname != "127.0.0.1"
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or parsed.path not in {"", "/", "/v1"}
            ):
                raise ValueError("audio ASR endpoint must be an exact loopback HTTP URL")

    @property
    def enabled(self) -> bool:
        return self.base_url is not None and self.api_key_file is not None

    def _api_key(self) -> str:
        path = self.api_key_file
        if path is None:
            raise AudioAsrProxyDisabled("audio ASR is not configured")
        try:
            if path.is_symlink() or not path.is_file():
                raise OSError("unsafe key path")
            key = path.read_text(encoding="ascii").strip()
        except OSError as exc:
            raise AudioAsrProxyDisabled("audio ASR credential is unavailable") from exc
        if not 32 <= len(key) <= 512 or any(ord(character) < 33 for character in key):
            raise AudioAsrProxyDisabled("audio ASR credential is invalid")
        return key

    def transcribe(
        self,
        body: bytes,
        *,
        content_type: str,
        request_id: str,
    ) -> AudioAsrReply:
        if not self.enabled:
            raise AudioAsrProxyDisabled("audio ASR is not configured")
        if not body or len(body) > self.maximum_request_bytes:
            raise ValueError("audio ASR request size is invalid")
        if not content_type.lower().startswith("multipart/form-data;"):
            raise ValueError("audio ASR content type is invalid")
        assert self.base_url is not None
        parsed = urlsplit(self.base_url)
        connection = http.client.HTTPConnection(
            parsed.hostname,
            parsed.port or 80,
            timeout=self.timeout_seconds,
        )
        base_path = parsed.path.rstrip("/")
        path = (
            f"{base_path}/audio/transcriptions"
            if base_path == "/v1"
            else "/v1/audio/transcriptions"
        )
        try:
            connection.request(
                "POST",
                path,
                body=body,
                headers={
                    "Authorization": f"Bearer {self._api_key()}",
                    "Content-Type": content_type,
                    "Content-Length": str(len(body)),
                    "Accept": "application/json",
                    "X-Request-ID": request_id,
                },
            )
            response = connection.getresponse()
            announced = response.getheader("Content-Length")
            if announced:
                try:
                    if int(announced) > self.maximum_response_bytes:
                        raise AudioAsrProxyError("audio ASR response is too large")
                except ValueError as exc:
                    raise AudioAsrProxyError("audio ASR response length is invalid") from exc
            body_bytes = response.read(self.maximum_response_bytes + 1)
            if len(body_bytes) > self.maximum_response_bytes:
                raise AudioAsrProxyError("audio ASR response is too large")
            content_type_value = str(
                response.getheader("Content-Type") or "application/json"
            )
            if "application/json" not in content_type_value.lower():
                raise AudioAsrProxyError("audio ASR returned an invalid content type")
            try:
                parsed_body = json.loads(body_bytes)
            except (TypeError, ValueError) as exc:
                raise AudioAsrProxyError("audio ASR returned invalid JSON") from exc
            if not isinstance(parsed_body, dict):
                raise AudioAsrProxyError("audio ASR returned invalid JSON")
            if response.status in {401, 403}:
                raise AudioAsrProxyDisabled("audio ASR worker authentication failed")
            status_code = (
                response.status
                if response.status in {200, 413, 422, 429, 503}
                else 502
            )
            return AudioAsrReply(
                status_code=status_code,
                body=body_bytes,
                content_type="application/json",
                retry_after=response.getheader("Retry-After"),
            )
        except (TimeoutError, socket.timeout) as exc:
            raise AudioAsrProxyTimeout("audio ASR request timed out") from exc
        except (ConnectionError, OSError, http.client.HTTPException) as exc:
            raise AudioAsrProxyError("audio ASR worker is unavailable") from exc
        finally:
            connection.close()
