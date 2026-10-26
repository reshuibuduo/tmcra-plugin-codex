from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, TypeVar


_StoreType = TypeVar("_StoreType", bound=type)
_REQUIRED_GRAPH_TABLES = frozenset(
    {"records", "memory_edges", "slot_heads", "slot_history", "meta"}
)


def _read_only_store_class(base_store: _StoreType) -> _StoreType:
    """Wrap the integrated SQLite store without running its write-side setup."""
    if getattr(base_store, "_tmcra_production_read_only", False):
        return base_store

    class ProductionReadOnlySQLiteStore(base_store):  # type: ignore[valid-type, misc]
        _tmcra_production_read_only = True

        def __init__(
            self, storage_path: str | Path, *, audit_retention: int = 256
        ) -> None:
            self.storage_path = Path(storage_path).expanduser().resolve()
            self.audit_retention = max(1, int(audit_retention))
            if not self.storage_path.is_file():
                raise FileNotFoundError(self.storage_path)
            connection = self._connect()
            try:
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            finally:
                connection.close()
            missing = sorted(_REQUIRED_GRAPH_TABLES - tables)
            if missing:
                raise RuntimeError(
                    "production graph snapshot lacks required tables: "
                    + ",".join(missing)
                )

        def _connect(self) -> sqlite3.Connection:
            connection = sqlite3.connect(
                self.storage_path.as_uri() + "?mode=ro&immutable=1",
                uri=True,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            return connection

    ProductionReadOnlySQLiteStore.__name__ = (
        f"ProductionReadOnly{base_store.__name__}"
    )
    return ProductionReadOnlySQLiteStore  # type: ignore[return-value]


def _redirect_audit_persistence(adapter: Any, database: Any | None = None) -> Any:
    """Persist mutable retrieval audits outside immutable index generations."""
    if database is None:
        from tmcra_service.control_db import ControlDB

        control_path = os.getenv("TMCRA_SERVICE_CONTROL_DB", "").strip()
        if not control_path:
            state_dir = os.getenv("TMCRA_SERVICE_STATE_DIR", "").strip()
            if state_dir:
                control_path = str(Path(state_dir) / "control.sqlite3")
        if not control_path:
            raise RuntimeError("TMCRA_SERVICE_CONTROL_DB is required by production harness")
        database = ControlDB(control_path)

    original_reload = adapter._reload_graph
    audit_fields = ("retrieval_log", "answer_support_log")

    def reload_with_runtime_audits(self: Any) -> None:
        original_reload()
        base_state: dict[str, tuple[int, int]] = {}
        for field_name in audit_fields:
            base_events = list(getattr(self.graph, field_name))
            base_total = max(
                len(base_events),
                int(self.graph.audit_event_totals.get(field_name, 0) or 0),
            )
            base_trimmed = max(
                int(self.graph.audit_trimmed_counts.get(field_name, 0) or 0),
                base_total - len(base_events),
            )
            runtime = database.graph_runtime_audits(self.scope_id, field_name)
            runtime_payloads = [dict(item) for item in runtime["payloads"]]
            combined_total = base_total + int(runtime["event_total"])
            combined = [*base_events, *runtime_payloads]
            if len(combined) > self.audit_retention:
                combined = combined[-self.audit_retention :]
            setattr(self.graph, field_name, combined)
            self.graph.audit_event_totals[field_name] = combined_total
            self.graph.audit_trimmed_counts[field_name] = max(
                base_trimmed + int(runtime["trimmed_total"]),
                combined_total - len(combined),
            )
            base_state[field_name] = (base_total, base_trimmed)
        self._tmcra_generation_audit_base = base_state

    def persist_runtime_audit(self: Any, field_name: str) -> dict[str, Any]:
        if field_name not in audit_fields:
            raise RuntimeError(
                f"read-only production graph cannot persist audit field: {field_name}"
            )
        events = getattr(self.graph, field_name)
        if not events:
            raise RuntimeError(f"cannot persist empty audit field: {field_name}")
        base_total, base_trimmed = self._tmcra_generation_audit_base[field_name]
        persisted = database.append_graph_runtime_audit(
            self.scope_id,
            field_name,
            dict(events[-1]),
            retention=self.audit_retention,
            base_event_total=base_total,
            base_trimmed_total=base_trimmed,
        )
        events[-1] = dict(persisted["payload"])
        self.graph.audit_event_totals[field_name] = int(persisted["event_total"])
        self.graph.audit_trimmed_counts[field_name] = int(
            persisted["trimmed_total"]
        )
        return dict(events[-1])

    adapter._reload_graph = MethodType(reload_with_runtime_audits, adapter)
    adapter._persist_latest_audit = MethodType(persist_runtime_audit, adapter)
    adapter._reload_graph()
    return adapter


def build_adapter(scope_id: str, storage_path: Path) -> Any:
    import experiments.replacement.adapters.memory_adapters as memory_adapters

    memory_adapters.SQLiteSessionMemoryStore = _read_only_store_class(
        memory_adapters.SQLiteSessionMemoryStore
    )

    adapter = memory_adapters.GraphSessionMemoryAdapter(
        auto_extract=False,
        storage_backend="sqlite",
        storage_path=str(storage_path),
        scope_id=scope_id,
        retrieval_mode=os.getenv("TMCRA_RETRIEVAL_MODE", "hybrid_node_scored"),
        node_model_path=os.getenv("TMCRA_NODE_MODEL_PATH", ""),
        path_model_path=os.getenv("TMCRA_PATH_MODEL_PATH", ""),
        node_model_device=os.getenv("TMCRA_NODE_MODEL_DEVICE", "cpu"),
        candidate_event_k=int(os.getenv("TMCRA_CANDIDATE_EVENT_K", "24")),
        support_path_k=int(os.getenv("TMCRA_SUPPORT_PATH_K", "3")),
        path_tunnel_rescue_k=int(os.getenv("TMCRA_PATH_TUNNEL_RESCUE_K", "2")),
        path_tunnel_rescue_score_floor=float(
            os.getenv("TMCRA_PATH_TUNNEL_RESCUE_SCORE_FLOOR", "0.0")
        ),
        path_tunnel_rescue_min_age=int(
            os.getenv("TMCRA_PATH_TUNNEL_RESCUE_MIN_AGE", "0")
        ),
        path_tunnel_rescue_min_score_margin=float(
            os.getenv("TMCRA_PATH_TUNNEL_RESCUE_MIN_SCORE_MARGIN", "0.0")
        ),
    )
    return _redirect_audit_persistence(adapter)


def disable_topic_bucket_runtime() -> None:
    import experiments.replacement.adapters.memory_adapters as memory_adapters

    def empty_bucket(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def no_apply(records: list[Any], topic_bucket: Mapping[str, Any]) -> None:
        return None

    def disabled_edges(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "topic_bridge_disabled": True,
            "dialogue_tunnel_disabled": True,
            "disabled_reason": "tmcra_production_no_topic_bucket",
        }

    def no_rerank(
        graph: Any, query: str, hits: list[Any], *, top_k: int
    ) -> dict[str, Any]:
        limit = max(1, int(top_k or 1))
        selected = list(hits)[:limit]
        return {
            "hits": selected,
            "metadata": {
                "topic_bucket_rerank_enabled": False,
                "topic_bucket_disabled": True,
                "topic_bucket_disable_reason": "tmcra_production_no_topic_bucket",
                "topic_bucket_candidate_count": len(list(hits)),
                "topic_bucket_final_count": len(selected),
            },
        }

    memory_adapters._assign_topic_bucket_for_text = empty_bucket
    memory_adapters._apply_topic_bucket_to_records = no_apply
    memory_adapters._last_topic_turn = empty_bucket
    memory_adapters._add_topic_bridge_edges = disabled_edges
    memory_adapters._add_dialogue_tunnel_edges = disabled_edges
    memory_adapters._topic_bucket_rerank_hits = no_rerank
