from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tmcra_v3_recall_planner import RecallPlannerError, RecallPlannerResponseError
from tmcra_v4_recall_planner import (
    DeepSeekFlashRecallRolePlanner,
    LAYER_NAMES,
    ROLE_PLAN_SCHEMA,
    validate_recall_role_plan,
)

from .planner_provider import recall_planner_route
from .gpu_scheduler import GpuWorkload, GpuWorkloadScheduler
from .qwen36_planner_adapter import LocalQwenRecallRolePlanner
from .writer_provider import LOCAL_QWEN_PROVIDER


_AUDIT_LOCKS_GUARD = threading.Lock()
_AUDIT_LOCKS: dict[Path, threading.Lock] = {}
_MAX_INTERACTIVE_QUERY_CHARS = 2000


def _bounded_interactive_query(query: str) -> tuple[str, bool]:
    value = str(query or "").strip()
    if not value:
        raise ValueError("interactive recall query is empty")
    if len(value) <= _MAX_INTERACTIVE_QUERY_CHARS:
        return value, False
    separator = "\n...[middle omitted for interactive retrieval]...\n"
    available = _MAX_INTERACTIVE_QUERY_CHARS - len(separator)
    head = available // 2
    tail = available - head
    return f"{value[:head]}{separator}{value[-tail:]}", True


def _shared_audit_lock(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _AUDIT_LOCKS_GUARD:
        lock = _AUDIT_LOCKS.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _AUDIT_LOCKS[resolved] = lock
        return lock


def interactive_recall_plan(query: str) -> dict[str, Any]:
    """Return a neutral, provider-free plan for latency-bounded clients.

    Every available local retrieval layer still runs. Equal weights avoid a
    brittle keyword router while the answer model retains the full evidence
    window needed to interpret the current query.
    """

    resolved_query, _ = _bounded_interactive_query(query)
    return validate_recall_role_plan(
        {
            "schema_version": ROLE_PLAN_SCHEMA,
            "resolved_query": resolved_query,
            "query_kind": "unknown",
            "temporal_focus": "unknown",
            "conflict_policy": "surface_uncertainty",
            "layers": {
                layer: {"role": "evidence", "weight": 1.0}
                for layer in LAYER_NAMES
            },
        }
    )


def recall_planner_from_env() -> Any:
    route = recall_planner_route(os.environ)
    try:
        timeout = float(os.getenv("TMCRA_RECALL_PLANNER_TIMEOUT_SECONDS", "60"))
        max_tokens = int(os.getenv("TMCRA_RECALL_PLANNER_MAX_TOKENS", "512"))
    except ValueError as exc:
        raise ValueError("recall planner timeout or max tokens is invalid") from exc
    kwargs = {
        "base_url": route.base_url,
        "model": route.model,
        "api_keys": list(route.api_keys),
        "timeout": timeout,
        "max_tokens": max_tokens,
    }
    if route.provider == LOCAL_QWEN_PROVIDER:
        return LocalQwenRecallRolePlanner(**kwargs)
    return DeepSeekFlashRecallRolePlanner(**kwargs)


class ScheduledRecallPlanner:
    """Serialize physical planner calls on the shared Qwen planner lane."""

    def __init__(self, delegate: Any, scheduler: GpuWorkloadScheduler) -> None:
        self.delegate = delegate
        self.scheduler = scheduler

    def set_provider_user_id(self, value: str) -> None:
        setter = getattr(self.delegate, "set_provider_user_id", None)
        if callable(setter):
            setter(value)
            return
        setattr(self.delegate, "user_id", str(value or ""))

    def plan(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        with self.scheduler.lease(GpuWorkload.PLANNER_FOREGROUND):
            result = self.delegate.plan(**kwargs)
        return result


class AuditedRecallPlanner:
    """Accept one provably neutral schema omission and audit the repair."""

    def __init__(self, delegate: Any, audit_path: Path) -> None:
        self.delegate = delegate
        self.audit_path = audit_path.resolve()
        self._audit_lock = _shared_audit_lock(self.audit_path)

    @staticmethod
    def _repair_unavailable_layers(
        content: str, available_layers: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        value = json.loads(content)
        if not isinstance(value, Mapping):
            raise ValueError("planner response is not an object")
        repaired = dict(value)
        raw_layers = repaired.get("layers")
        if not isinstance(raw_layers, Mapping):
            raise ValueError("planner response has no layer object")
        layers = {str(name): dict(entry) for name, entry in raw_layers.items()}
        expected = set(LAYER_NAMES)
        if set(layers) - expected:
            raise ValueError("planner response contains an unknown layer")
        missing = sorted(expected - set(layers))
        if not missing:
            raise ValueError("planner failure is not a missing-layer omission")
        for layer in missing:
            summary = available_layers.get(layer)
            if not isinstance(summary, Mapping) or bool(summary.get("available")):
                raise ValueError("planner omitted a layer that has candidates")
            layers[layer] = {"role": "context", "weight": 0.0}
        repaired["layers"] = layers
        return validate_recall_role_plan(repaired), missing

    @staticmethod
    def _repair_overlong_resolved_query(
        content: str, original_query: str
    ) -> dict[str, Any]:
        value = json.loads(content)
        if not isinstance(value, Mapping):
            raise ValueError("planner response is not an object")
        resolved_query = value.get("resolved_query")
        if not isinstance(resolved_query, str) or len(resolved_query.strip()) <= 2000:
            raise ValueError("planner failure is not an overlong resolved query")
        fallback_query, _ = _bounded_interactive_query(original_query)
        repaired = dict(value)
        repaired["resolved_query"] = fallback_query
        return validate_recall_role_plan(repaired)

    @staticmethod
    def _neutral_plan(query: str) -> dict[str, Any]:
        bounded_query, _ = _bounded_interactive_query(query)
        return validate_recall_role_plan(
            {
                "schema_version": ROLE_PLAN_SCHEMA,
                "resolved_query": bounded_query,
                "query_kind": "unknown",
                "temporal_focus": "unknown",
                "conflict_policy": "surface_uncertainty",
                "layers": {
                    layer: {"role": "evidence", "weight": 1.0}
                    for layer in LAYER_NAMES
                },
            }
        )

    @staticmethod
    def _digest(value: str) -> dict[str, Any]:
        text = str(value or "")
        return {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "length": len(text),
        }

    @staticmethod
    def _safe_request_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
        """Keep only metadata fields that cannot contain user/provider text."""

        allowed = {
            "physical_call_id",
            "physical_api_call",
            "physical_api_calls",
            "provider",
            "model",
            "api_key_index",
            "latency_seconds",
            "response_sha256",
            "finish_reason",
            "status",
            "planner_version",
            "prompt_version",
            "prompt_adapter",
            "request_sha256",
            "response_id",
            "http_status",
            "error_type",
            "prompt_tokens",
            "completion_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "total_tokens",
        }
        safe: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name not in allowed:
                continue
            if item is None or isinstance(item, (bool, int, float, str)):
                safe[name] = item
        return safe

    def _degrade_to_neutral_plan(
        self, exc: RecallPlannerError, *, query: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response_content = getattr(exc, "response_content", "")
        request_metadata = getattr(exc, "request_metadata", {})
        if not isinstance(request_metadata, Mapping):
            request_metadata = {}
        plan = self._neutral_plan(query)
        try:
            physical_api_calls = int(request_metadata.get("physical_api_calls", 0) or 0)
        except (TypeError, ValueError):
            physical_api_calls = 0
        metadata = {
            **self._safe_request_metadata(request_metadata),
            "physical_api_call": bool(
                request_metadata.get("physical_api_call", False)
            ),
            "physical_api_calls": max(0, physical_api_calls),
            "status": "degraded_with_neutral_plan",
            "planner_degraded": True,
            "fallback": "neutral_all_layers",
            "error_type": type(exc).__name__,
            "error_sha256": self._digest(str(exc))["sha256"],
            "error_length": len(str(exc)),
            "query_bounded": len(str(query or "").strip()) > len(plan["resolved_query"]),
        }
        audit = {
            "schema_version": "tmcra.service.recall-planner-repair.1",
            "created_at": time.time(),
            "repair_kind": "neutral_plan_after_invalid_output",
            "error_type": type(exc).__name__,
            "error": self._digest(str(exc)),
            "query": self._digest(query),
            "response": self._digest(response_content),
            "request_metadata": self._safe_request_metadata(request_metadata),
        }
        self._append_audit(audit)
        return plan, metadata

    def _append_audit(self, row: Mapping[str, Any]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n"
        with self._audit_lock:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(self.audit_path, 0o600)

    def set_provider_user_id(self, value: str) -> None:
        """Set the privacy-safe business identity for the next planner call.

        V4OnlineEngine serializes access to one planner replica, so this
        per-call transport attribute cannot race with another tenant.
        """

        setattr(self.delegate, "user_id", str(value or ""))

    def plan(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            result = self.delegate.plan(**kwargs)
            if (
                not isinstance(result, tuple)
                or len(result) != 2
                or not isinstance(result[0], Mapping)
                or not isinstance(result[1], Mapping)
            ):
                raise RecallPlannerError("planner returned an invalid result envelope")
            return validate_recall_role_plan(result[0]), dict(result[1])
        except RecallPlannerError as exc:
            available_layers = kwargs.get("available_layers") or {}
            response_content = getattr(exc, "response_content", "")
            if isinstance(exc, RecallPlannerResponseError) and response_content:
                try:
                    repaired, missing = self._repair_unavailable_layers(
                        response_content,
                        available_layers,
                    )
                except Exception:
                    try:
                        repaired = self._repair_overlong_resolved_query(
                            response_content,
                            str(kwargs.get("query") or ""),
                        )
                    except Exception:
                        return self._degrade_to_neutral_plan(
                            exc, query=str(kwargs.get("query") or "")
                        )
                    request_metadata = getattr(exc, "request_metadata", {})
                    request_metadata = (
                        request_metadata
                        if isinstance(request_metadata, Mapping)
                        else {}
                    )
                    metadata = {
                        **self._safe_request_metadata(request_metadata),
                        "status": "completed_with_bounded_query_repair",
                        "structural_repair": True,
                        "repaired_resolved_query": True,
                    }
                    self._append_audit(
                        {
                            "schema_version": "tmcra.service.recall-planner-repair.1",
                            "created_at": time.time(),
                            "repair_kind": "bounded_resolved_query",
                            "query": self._digest(str(kwargs.get("query") or "")),
                            "response": self._digest(response_content),
                            "request_metadata": self._safe_request_metadata(
                                request_metadata
                            ),
                        }
                    )
                    return repaired, metadata
                request_metadata = getattr(exc, "request_metadata", {})
                request_metadata = (
                    request_metadata
                    if isinstance(request_metadata, Mapping)
                    else {}
                )
                metadata = {
                    **self._safe_request_metadata(request_metadata),
                    "status": "completed_with_neutral_empty_layer_repair",
                    "structural_repair": True,
                    "repaired_missing_layers": missing,
                }
                self._append_audit(
                    {
                        "schema_version": "tmcra.service.recall-planner-repair.1",
                        "created_at": time.time(),
                        "repair_kind": "neutral_empty_layer",
                        "missing_layers": missing,
                        "available_layers": {
                            layer: bool(
                                isinstance(available_layers.get(layer), Mapping)
                                and available_layers[layer].get("available")
                            )
                            for layer in LAYER_NAMES
                        },
                        "query": self._digest(str(kwargs.get("query") or "")),
                        "response": self._digest(response_content),
                        "request_metadata": self._safe_request_metadata(
                            request_metadata
                        ),
                    }
                )
                return repaired, metadata
            return self._degrade_to_neutral_plan(
                exc, query=str(kwargs.get("query") or "")
            )
