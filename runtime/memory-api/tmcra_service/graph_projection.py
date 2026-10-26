"""Read-only, tenant-bound projections of committed TMCRA memory graphs."""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .adapters.v4 import V4AdapterError, V4StorageAdapter


GRAPH_SCHEMA_VERSION = "tmcra.memory-graph.1"
GRAPH_LAYERS = frozenset({"slow", "fast", "source"})
ACTOR_ROLES = frozenset({"user", "assistant", "system", "tool"})
AGENT_PROVENANCE_FIELDS = (
    "agent_id",
    "agent_name",
    "agent_role",
    "agent_specialty",
    "agent_team",
    "target_agent_id",
)
DEFAULT_STATES = frozenset({"active", "challenged", "evidence"})
MAX_OVERVIEW_NODES = 300
MAX_NEIGHBOR_NODES = 120
MAX_EVIDENCE_ITEMS = 25
MAX_CURSOR_OFFSET = 10_000

_LAYER_SQL = """
CASE
  WHEN json_extract(metadata_json, '$.memory_layer') = 'slow'
    OR json_extract(metadata_json, '$.content_variant') = 'slow_memory_capsule'
    THEN 'slow'
  WHEN json_extract(metadata_json, '$.content_variant') = 'source_message'
    OR json_extract(metadata_json, '$.node_kind') = 'immutable_source_message'
    THEN 'source'
  ELSE 'fast'
END
"""


class GraphProjectionError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class SnapshotBinding:
    scope_name: str
    scope_id: str
    snapshot_id: str
    database_sha256: str | None
    database: Path
    snapshot_state: str = "committed"
    provisional: bool = False
    database_immutable: bool = False


@dataclass(frozen=True)
class GraphRecord:
    memory_id: str
    category: str
    slot_key: str
    value: str
    relation: str
    evidence_anchors: tuple[str, ...]
    salience: float
    confidence: float
    source_kind: str
    turn_index: int
    state: str
    supersedes: tuple[str, ...]
    metadata: Mapping[str, Any]

    @property
    def layer(self) -> str:
        metadata = self.metadata
        variant = _text(metadata.get("content_variant")).lower()
        layer = _text(metadata.get("memory_layer")).lower()
        node_kind = _text(metadata.get("node_kind")).lower()
        if layer == "slow" or variant == "slow_memory_capsule":
            return "slow"
        if variant == "source_message" or node_kind == "immutable_source_message":
            return "source"
        return "fast"


def parse_layers(value: str | Sequence[str] | None, *, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        layers = list(default)
    elif isinstance(value, str):
        layers = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        layers = [str(item).strip().lower() for item in value if str(item).strip()]
    unique = tuple(dict.fromkeys(layers))
    if not unique or any(item not in GRAPH_LAYERS for item in unique):
        raise GraphProjectionError(
            "invalid_graph_layers",
            "layers must contain slow, fast, or source",
            status_code=422,
        )
    return unique


def extract_trace_memory_ids(evidence: Mapping[str, Any]) -> list[str]:
    """Extract persisted memory identities from answer-facing recall evidence."""

    identifiers: list[str] = []

    def add(value: Any) -> None:
        identifier = _text(value)
        if identifier and identifier not in identifiers:
            identifiers.append(identifier)

    for window in _sequence(evidence.get("evidence_windows")):
        if not isinstance(window, Mapping):
            continue
        add(window.get("source_record_id"))
        for identifier in _sequence(window.get("semantic_record_ids")):
            add(identifier)
        for attachment in _sequence(window.get("attachments")):
            if isinstance(attachment, Mapping):
                add(attachment.get("memory_id"))
                parent = attachment.get("source_parent")
                if isinstance(parent, Mapping):
                    add(parent.get("source_record_id"))
        for context in _sequence(window.get("memory_contexts")):
            if not isinstance(context, Mapping):
                continue
            provenance = context.get("provenance")
            if isinstance(provenance, Mapping):
                add(provenance.get("memory_id"))
                add(provenance.get("semantic_memory_id"))
            capsule_id = _text(context.get("capsule_id"))
            revision = _integer(context.get("revision"), 0)
            if capsule_id and revision > 0:
                add(f"slow.{capsule_id}.r{revision}")
            for field in ("support", "counterevidence"):
                for identifier in _sequence(context.get(field)):
                    add(identifier)
            for parent in _sequence(context.get("source_parents")):
                if isinstance(parent, Mapping):
                    add(parent.get("source_record_id"))
    return identifiers


class MemoryGraphProjection:
    def __init__(self, binding: SnapshotBinding) -> None:
        self.binding = binding

    @classmethod
    def from_storage(
        cls,
        storage: V4StorageAdapter,
        *,
        tenant_id: str,
        scope_name: str,
    ) -> "MemoryGraphProjection":
        try:
            snapshot = storage.active_snapshot(tenant_id, scope_name)
        except V4AdapterError as exc:
            raise GraphProjectionError(
                "graph_snapshot_unavailable",
                "scope has no committed memory graph snapshot",
            ) from exc
        return cls.from_snapshot(
            storage,
            tenant_id=tenant_id,
            scope_name=scope_name,
            snapshot=snapshot,
        )

    @classmethod
    def from_available_storage(
        cls,
        storage: V4StorageAdapter,
        *,
        tenant_id: str,
        scope_name: str,
    ) -> "MemoryGraphProjection":
        """Use the committed snapshot, or a clearly marked live read-only preview."""

        try:
            return cls.from_storage(
                storage,
                tenant_id=tenant_id,
                scope_name=scope_name,
            )
        except GraphProjectionError as exc:
            if exc.code != "graph_snapshot_unavailable":
                raise
        return cls.from_live_storage(
            storage,
            tenant_id=tenant_id,
            scope_name=scope_name,
        )

    @classmethod
    def from_live_storage(
        cls,
        storage: V4StorageAdapter,
        *,
        tenant_id: str,
        scope_name: str,
    ) -> "MemoryGraphProjection":
        paths = storage.scope_paths(tenant_id, scope_name)
        database = paths.database.resolve()
        try:
            database_stat = database.stat()
        except OSError as exc:
            raise GraphProjectionError(
                "graph_snapshot_unavailable",
                "scope has neither a committed graph nor written memory to preview",
            ) from exc
        if not database.is_file() or database_stat.st_size <= 0:
            raise GraphProjectionError(
                "graph_snapshot_unavailable",
                "scope has neither a committed graph nor written memory to preview",
            )
        wal = Path(f"{database}-wal")
        try:
            wal_stat = wal.stat()
            wal_fingerprint = f"{wal_stat.st_size}:{wal_stat.st_mtime_ns}"
        except FileNotFoundError:
            wal_fingerprint = "none"
        seed = (
            f"{paths.scope_id}:{database_stat.st_size}:{database_stat.st_mtime_ns}:"
            f"{wal_fingerprint}"
        )
        binding = SnapshotBinding(
            scope_name=scope_name,
            scope_id=paths.scope_id,
            snapshot_id=f"building-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}",
            database_sha256=None,
            database=database,
            snapshot_state="building",
            provisional=True,
        )
        projection = cls(binding)
        projection._validate_schema()
        return projection

    @classmethod
    def from_snapshot(
        cls,
        storage: V4StorageAdapter,
        *,
        tenant_id: str,
        scope_name: str,
        snapshot: Mapping[str, Any],
    ) -> "MemoryGraphProjection":
        paths = storage.scope_paths(tenant_id, scope_name)
        if _text(snapshot.get("scope_id")) != paths.scope_id:
            raise GraphProjectionError(
                "graph_snapshot_scope_mismatch",
                "committed memory graph belongs to a different scope",
            )
        generation_id = _text(snapshot.get("generation_id"))
        snapshot_id = generation_id or _text(snapshot.get("job_id"))
        database_sha256 = _text(snapshot.get("database_sha256")) or None
        if not snapshot_id:
            seed = database_sha256 or hashlib.sha256(
                str(snapshot.get("database", "")).encode("utf-8")
            ).hexdigest()
            snapshot_id = f"legacy-{seed[:16]}"
        binding = SnapshotBinding(
            scope_name=scope_name,
            scope_id=paths.scope_id,
            snapshot_id=snapshot_id,
            database_sha256=database_sha256,
            database=Path(str(snapshot["database"])).resolve(),
            # Non-legacy generations have already passed the storage adapter's
            # integrity validation and durable sealing boundary. A legacy
            # manifest still targets the mutable live database and may depend
            # on WAL contents, even though it is presented as a committed view.
            database_immutable=bool(generation_id),
        )
        projection = cls(binding)
        projection._validate_schema()
        return projection

    def overview(
        self,
        *,
        layers: Sequence[str] = ("slow",),
        limit: int = 180,
        cursor: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        normalized_layers = parse_layers(layers, default=("slow",))
        limit = _bounded(limit, 1, MAX_OVERVIEW_NODES, "limit")
        offset = self._decode_cursor(cursor)
        records, has_more = self._page_records(
            layers=normalized_layers,
            limit=limit,
            offset=offset,
            query=query,
        )
        fallback_layer: str | None = None
        if not records and offset == 0 and normalized_layers == ("slow",) and not query:
            for candidate in ("fast", "source"):
                records, has_more = self._page_records(
                    layers=(candidate,),
                    limit=limit,
                    offset=0,
                    query=None,
                )
                if records:
                    normalized_layers = (candidate,)
                    fallback_layer = candidate
                    break
        return self._graph_response(
            view="overview",
            records=records,
            requested_layers=tuple(layers),
            resolved_layers=normalized_layers,
            limit=limit,
            offset=offset,
            has_more=has_more,
            fallback_layer=fallback_layer,
        )

    def session_overview(
        self,
        session_id: str,
        *,
        semantic_limit: int = 240,
        source_limit: int = 1000,
        include_source_text: bool = False,
    ) -> dict[str, Any]:
        """Return records whose immutable evidence belongs to one Session.

        Session ownership is derived from Source records rather than copied
        from mutable semantic labels. This keeps the user-facing projection
        tenant-bound and prevents an LLM-created topic label from moving a
        memory between conversations.
        """

        session_id = _text(session_id)
        if not session_id or len(session_id) > 200 or "\x00" in session_id:
            raise GraphProjectionError(
                "invalid_session_id", "session id is invalid", status_code=422
            )
        semantic_limit = _bounded(
            semantic_limit, 1, 20_000, "semantic_limit"
        )
        source_limit = _bounded(source_limit, 1, 100_000, "source_limit")
        columns = """
            memory_id,category,slot_key,value,relation,evidence_anchors_json,
            salience,confidence,source_kind,turn_index,state,supersedes_json,
            metadata_json
        """
        with closing(self._connect()) as connection:
            source_rows = connection.execute(
                f"""
                SELECT {columns}
                FROM records
                WHERE scope_id=? AND ({_LAYER_SQL})='source'
                  AND json_extract(metadata_json, '$.session_id')=?
                  AND state IN ({_marks(DEFAULT_STATES)})
                ORDER BY turn_index ASC,memory_id ASC
                LIMIT ?
                """,
                [
                    self.binding.scope_id,
                    session_id,
                    *sorted(DEFAULT_STATES),
                    source_limit + 1,
                ],
            ).fetchall()
            semantic_rows = connection.execute(
                f"""
                SELECT {columns}
                FROM records
                WHERE scope_id=? AND ({_LAYER_SQL})!='source'
                  AND state IN ({_marks(DEFAULT_STATES)})
                ORDER BY turn_index ASC,memory_id ASC
                """,
                [self.binding.scope_id, *sorted(DEFAULT_STATES)],
            ).fetchall()

        source_has_more = len(source_rows) > source_limit
        sources = [_record(row) for row in source_rows[:source_limit]]
        source_ids = {record.memory_id for record in sources}
        semantic = [_record(row) for row in semantic_rows]
        known_ids = set(source_ids)
        selected: dict[str, GraphRecord] = {}
        unresolved = list(semantic)
        # Fast records normally point to Source, while Slow records point to
        # Fast records and/or Source parents. Iterate to preserve that chain.
        for _ in range(8):
            changed = False
            remaining: list[GraphRecord] = []
            for record in unresolved:
                references = set(_record_link_ids(record))
                if references & known_ids:
                    selected[record.memory_id] = record
                    known_ids.add(record.memory_id)
                    changed = True
                else:
                    remaining.append(record)
            unresolved = remaining
            if not changed:
                break

        ranked = sorted(
            selected.values(),
            key=lambda item: (
                item.turn_index,
                -item.salience,
                item.memory_id,
            ),
        )
        has_more = source_has_more or len(ranked) > semantic_limit
        ranked = ranked[:semantic_limit]
        records = [*sources, *ranked]
        response = self._graph_response(
            view="overview",
            records=records,
            requested_layers=("slow", "fast", "source"),
            resolved_layers=("slow", "fast", "source"),
            limit=len(records) or 1,
            offset=0,
            has_more=has_more,
        )

        selected_map = {record.memory_id: record for record in ranked}
        resolved_sources: dict[str, set[str]] = {}
        for record in ranked:
            direct = {
                identifier
                for identifier in _record_link_ids(record)
                if identifier in source_ids
            }
            resolved_sources[record.memory_id] = direct
        for _ in range(8):
            changed = False
            for record in ranked:
                values = resolved_sources[record.memory_id]
                before = len(values)
                for identifier in _record_link_ids(record):
                    if identifier in selected_map:
                        values.update(resolved_sources.get(identifier, ()))
                changed = changed or len(values) != before
            if not changed:
                break
        source_text_by_id = (
            {record.memory_id: record.value for record in sources}
            if include_source_text
            else {}
        )
        for node in response["nodes"]:
            attributes = dict(node.get("attributes") or {})
            attributes["session_id"] = session_id
            if node["id"] in resolved_sources:
                attributes["source_record_ids"] = sorted(
                    resolved_sources[node["id"]]
                )
            node["attributes"] = attributes
            if node["id"] in source_text_by_id:
                # Internal projection input for the human-memory Atlas. Public
                # graph callers keep the default False and never receive raw
                # Source text through this endpoint.
                node["_source_text"] = source_text_by_id[node["id"]]
        response["session_id"] = session_id
        response["source_record_count"] = len(sources)
        response["semantic_record_count"] = len(ranked)
        return response

    def neighbors(
        self,
        memory_id: str,
        *,
        depth: int = 1,
        layers: Sequence[str] = ("slow", "fast", "source"),
        limit: int = 80,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        memory_id = _memory_id(memory_id)
        depth = _bounded(depth, 1, 2, "depth")
        normalized_layers = parse_layers(
            layers, default=("slow", "fast", "source")
        )
        limit = _bounded(limit, 1, MAX_NEIGHBOR_NODES, "limit")
        offset = self._decode_cursor(cursor)
        root = self._record(memory_id)
        if root is None:
            raise GraphProjectionError(
                "memory_node_not_found", "memory node not found", status_code=404
            )

        discovered: list[str] = []
        seen = {root.memory_id}
        frontier = [root]
        hard_limit = min(MAX_CURSOR_OFFSET + MAX_NEIGHBOR_NODES + 1, offset + limit + 1)
        for _ in range(depth):
            next_ids: list[str] = []
            adjacency = self._adjacent_ids_many(frontier)
            for record in frontier:
                for identifier in adjacency.get(record.memory_id, ()):
                    if identifier in seen:
                        continue
                    seen.add(identifier)
                    next_ids.append(identifier)
            next_records = self._records(next_ids)
            frontier = []
            for identifier in next_ids:
                record = next_records.get(identifier)
                if record is None or record.layer not in normalized_layers:
                    continue
                discovered.append(identifier)
                frontier.append(record)
                if len(discovered) >= hard_limit:
                    break
            if len(discovered) >= hard_limit or not frontier:
                break

        page_ids = discovered[offset : offset + limit]
        selected = self._records(page_ids)
        page_records = [root, *[selected[item] for item in page_ids if item in selected]]
        has_more = len(discovered) > offset + limit
        response = self._graph_response(
            view="neighbors",
            records=page_records,
            requested_layers=tuple(layers),
            resolved_layers=normalized_layers,
            limit=limit,
            offset=offset,
            has_more=has_more,
            root_id=root.memory_id,
            depth=depth,
        )
        response["page"]["returned_neighbors"] = max(0, len(page_records) - 1)
        return response

    def evidence(
        self,
        memory_id: str,
        *,
        limit: int = 10,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        memory_id = _memory_id(memory_id)
        limit = _bounded(limit, 1, MAX_EVIDENCE_ITEMS, "limit")
        offset = self._decode_cursor(cursor)
        root = self._record(memory_id)
        if root is None:
            raise GraphProjectionError(
                "memory_node_not_found", "memory node not found", status_code=404
            )

        references = self._source_references(root)
        page_refs = references[offset : offset + limit]
        source_ids = [item[0] for item in page_refs]
        source_records = self._records(source_ids)
        external_message_ids = self._external_message_ids(
            _text(record.metadata.get("message_id"))
            for record in source_records.values()
        )
        items: list[dict[str, Any]] = []
        for source_id, relationship, offsets in page_refs:
            record = source_records.get(source_id)
            if record is None or record.layer != "source":
                continue
            text = record.value
            metadata = record.metadata
            stored_message_id = _text(metadata.get("message_id"))
            items.append(
                {
                    "source_record_id": record.memory_id,
                    "relationship": relationship,
                    "session_id": _text(metadata.get("session_id")) or None,
                    "message_id": (
                        external_message_ids.get(stored_message_id, stored_message_id)
                        or None
                    ),
                    "role": _text(metadata.get("role") or metadata.get("speaker")) or None,
                    "actor_role": _actor_role(metadata),
                    "agent_id": _text(metadata.get("agent_id")) or None,
                    "agent_name": _text(metadata.get("agent_name")) or None,
                    "agent_role": _text(metadata.get("agent_role")) or None,
                    "agent_specialty": _text(metadata.get("agent_specialty")) or None,
                    "agent_team": _text(metadata.get("agent_team")) or None,
                    "target_agent_id": _text(metadata.get("target_agent_id")) or None,
                    "occurred_at": _occurred_at(metadata),
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "source_text_verbatim": True,
                    "evidence_char_start": offsets[0],
                    "evidence_char_end": offsets[1],
                }
            )
        has_more = len(references) > offset + limit
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "scope_name": self.binding.scope_name,
            "snapshot_id": self.binding.snapshot_id,
            "snapshot_state": self.binding.snapshot_state,
            "provisional": self.binding.provisional,
            "memory_id": root.memory_id,
            "items": items,
            "page": self._page(limit=limit, offset=offset, has_more=has_more),
        }

    def trace(self, memory_ids: Sequence[str]) -> dict[str, Any]:
        ordered = list(dict.fromkeys(_memory_id(item) for item in memory_ids if _text(item)))
        records_by_id = self._records(ordered)
        records = [records_by_id[item] for item in ordered if item in records_by_id]
        response = self._graph_response(
            view="recall_trace",
            records=records,
            requested_layers=("slow", "fast", "source"),
            resolved_layers=("slow", "fast", "source"),
            limit=max(1, len(records)),
            offset=0,
            has_more=False,
        )
        response["selected_memory_ids"] = [record.memory_id for record in records]
        response["missing_memory_ids"] = [
            item for item in ordered if item not in records_by_id
        ]
        return response

    def _validate_schema(self) -> None:
        required = {
            "records": {
                "scope_id",
                "memory_id",
                "category",
                "slot_key",
                "value",
                "relation",
                "evidence_anchors_json",
                "salience",
                "confidence",
                "source_kind",
                "turn_index",
                "state",
                "supersedes_json",
                "metadata_json",
            },
            "memory_edges": {
                "scope_id",
                "edge_id",
                "source_memory_id",
                "target_memory_id",
                "edge_type",
                "score",
                "metadata_json",
            },
        }
        with closing(self._connect()) as connection:
            for table, columns in required.items():
                rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
                actual = {str(row["name"]) for row in rows}
                if not columns <= actual:
                    raise GraphProjectionError(
                        "graph_schema_unavailable",
                        "committed memory graph does not expose the production projection schema",
                    )

    def _connect(self) -> sqlite3.Connection:
        try:
            query = (
                "mode=ro&immutable=1"
                if self.binding.database_immutable
                else "mode=ro"
            )
            connection = sqlite3.connect(
                self.binding.database.as_uri() + f"?{query}",
                uri=True,
                timeout=5.0,
            )
        except (OSError, sqlite3.Error) as exc:
            raise GraphProjectionError(
                "graph_snapshot_unavailable", "committed memory graph is unavailable"
            ) from exc
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _decode_cursor(self, cursor: str | None) -> int:
        if not cursor:
            return 0
        try:
            raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
            value = json.loads(raw.decode("utf-8"))
            offset = int(value["offset"])
            snapshot_id = str(value["snapshot_id"])
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GraphProjectionError(
                "invalid_graph_cursor", "graph cursor is invalid", status_code=422
            ) from exc
        if snapshot_id != self.binding.snapshot_id:
            raise GraphProjectionError(
                "stale_graph_cursor", "graph cursor belongs to a different snapshot", status_code=409
            )
        if offset < 0 or offset > MAX_CURSOR_OFFSET:
            raise GraphProjectionError(
                "invalid_graph_cursor", "graph cursor is outside the supported range", status_code=422
            )
        return offset

    def _encode_cursor(self, offset: int) -> str:
        payload = json.dumps(
            {"snapshot_id": self.binding.snapshot_id, "offset": offset},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    def _page(self, *, limit: int, offset: int, has_more: bool) -> dict[str, Any]:
        return {
            "limit": limit,
            "offset": offset,
            "truncated": has_more,
            "next_cursor": self._encode_cursor(offset + limit) if has_more else None,
        }

    def _page_records(
        self,
        *,
        layers: Sequence[str],
        limit: int,
        offset: int,
        query: str | None,
    ) -> tuple[list[GraphRecord], bool]:
        clauses = ["scope_id=?", f"({_LAYER_SQL}) IN ({_marks(layers)})"]
        parameters: list[Any] = [self.binding.scope_id, *layers]
        clauses.append(f"state IN ({_marks(DEFAULT_STATES)})")
        parameters.extend(sorted(DEFAULT_STATES))
        clean_query = _text(query)
        if clean_query:
            if len(clean_query) > 200:
                raise GraphProjectionError(
                    "graph_query_too_long", "graph query is too long", status_code=422
                )
            escaped = clean_query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append(
                "(value LIKE ? ESCAPE '\\' OR slot_key LIKE ? ESCAPE '\\' "
                "OR category LIKE ? ESCAPE '\\')"
            )
            pattern = f"%{escaped}%"
            parameters.extend([pattern, pattern, pattern])
        sql = f"""
            SELECT memory_id,category,slot_key,value,relation,evidence_anchors_json,
                   salience,confidence,source_kind,turn_index,state,supersedes_json,
                   metadata_json
            FROM records
            WHERE {' AND '.join(clauses)}
            ORDER BY salience DESC, turn_index DESC, memory_id ASC
            LIMIT ? OFFSET ?
        """
        parameters.extend([limit + 1, offset])
        with closing(self._connect()) as connection:
            rows = connection.execute(sql, parameters).fetchall()
        records = [_record(row) for row in rows[:limit]]
        return records, len(rows) > limit

    def _record(self, memory_id: str) -> GraphRecord | None:
        return self._records([memory_id]).get(memory_id)

    def _records(self, memory_ids: Iterable[str]) -> dict[str, GraphRecord]:
        unique = list(dict.fromkeys(_text(item) for item in memory_ids if _text(item)))
        if not unique:
            return {}
        result: dict[str, GraphRecord] = {}
        with closing(self._connect()) as connection:
            for chunk in _chunks(unique, 400):
                rows = connection.execute(
                    f"""
                    SELECT memory_id,category,slot_key,value,relation,evidence_anchors_json,
                           salience,confidence,source_kind,turn_index,state,supersedes_json,
                           metadata_json
                    FROM records
                    WHERE scope_id=? AND memory_id IN ({_marks(chunk)})
                    """,
                    [self.binding.scope_id, *chunk],
                ).fetchall()
                for row in rows:
                    record = _record(row)
                    result[record.memory_id] = record
        return result

    def _external_message_ids(
        self, stored_message_ids: Iterable[str]
    ) -> dict[str, str]:
        """Resolve graph-internal source IDs to immutable caller message IDs.

        Databases written before the service identity table existed keep the
        caller ID directly in source metadata, so an absent table deliberately
        falls back to that stored value. Once the table exists, malformed
        schema or read failures are treated as unavailable evidence rather than
        risking an incorrect cross-scope identity.
        """

        unique = list(
            dict.fromkeys(
                _text(item) for item in stored_message_ids if _text(item)
            )
        )
        if not unique:
            return {}
        result: dict[str, str] = {}
        with closing(self._connect()) as connection:
            try:
                table = connection.execute(
                    "SELECT type FROM sqlite_master WHERE name=? COLLATE BINARY",
                    ("tmcra_service_messages",),
                ).fetchone()
                if table is None:
                    return {}
                if _text(table["type"]) != "table":
                    raise GraphProjectionError(
                        "graph_schema_unavailable",
                        "committed memory graph message identity schema is invalid",
                    )
                columns = {
                    _text(row["name"])
                    for row in connection.execute(
                        "PRAGMA table_info(tmcra_service_messages)"
                    ).fetchall()
                }
                if not {"scope_id", "message_id", "internal_message_id"} <= columns:
                    raise GraphProjectionError(
                        "graph_schema_unavailable",
                        "committed memory graph message identity schema is invalid",
                    )
                for chunk in _chunks(unique, 400):
                    rows = connection.execute(
                        f"""
                        SELECT internal_message_id,message_id
                        FROM tmcra_service_messages
                        WHERE scope_id=? AND internal_message_id IN ({_marks(chunk)})
                        """,
                        [self.binding.scope_id, *chunk],
                    ).fetchall()
                    for row in rows:
                        internal_message_id = _text(row["internal_message_id"])
                        external_message_id = _text(row["message_id"])
                        prior = result.get(internal_message_id)
                        if (
                            not internal_message_id
                            or not external_message_id
                            or (prior is not None and prior != external_message_id)
                        ):
                            raise GraphProjectionError(
                                "graph_schema_unavailable",
                                "committed memory graph message identity is ambiguous",
                            )
                        result[internal_message_id] = external_message_id
                if set(result) != set(unique):
                    raise GraphProjectionError(
                        "graph_schema_unavailable",
                        "committed memory graph message identity is incomplete",
                    )
            except GraphProjectionError:
                raise
            except sqlite3.Error as exc:
                raise GraphProjectionError(
                    "graph_snapshot_unavailable",
                    "committed memory graph message identity is unavailable",
                ) from exc
        return result

    def _adjacent_ids_many(
        self, records: Sequence[GraphRecord]
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for record in records:
            values = [*record.supersedes, *record.evidence_anchors]
            source_id = _text(record.metadata.get("source_record_id"))
            if source_id:
                values.append(source_id)
            values.extend(_claim_memory_ids(record.metadata))
            values.extend(_source_parent_ids(record.metadata))
            result[record.memory_id] = values
        identifiers = list(result)
        if not identifiers:
            return result

        with closing(self._connect()) as connection:
            for chunk in _chunks(identifiers, 400):
                marks = _marks(chunk)
                rows = connection.execute(
                    f"""
                    SELECT source_memory_id,target_memory_id
                    FROM memory_edges
                    WHERE scope_id=? AND (
                        source_memory_id IN ({marks}) OR target_memory_id IN ({marks})
                    )
                    ORDER BY score DESC, edge_id ASC
                    LIMIT 5000
                    """,
                    [self.binding.scope_id, *chunk, *chunk],
                ).fetchall()
                for row in rows:
                    source = str(row["source_memory_id"])
                    target = str(row["target_memory_id"])
                    if source in result:
                        result[source].append(target)
                    if target in result:
                        result[target].append(source)

            for chunk in _chunks(identifiers, 250):
                marks = _marks(chunk)
                rows = connection.execute(
                    f"""
                    SELECT memory_id,evidence_anchors_json,supersedes_json,
                           json_extract(metadata_json, '$.source_record_id') AS source_record_id
                    FROM records
                    WHERE scope_id=? AND (
                        EXISTS (
                            SELECT 1 FROM json_each(records.evidence_anchors_json)
                            WHERE value IN ({marks})
                        ) OR EXISTS (
                            SELECT 1 FROM json_each(records.supersedes_json)
                            WHERE value IN ({marks})
                        ) OR json_extract(metadata_json, '$.source_record_id') IN ({marks})
                    )
                    ORDER BY turn_index DESC, memory_id ASC
                    LIMIT 5000
                    """,
                    [self.binding.scope_id, *chunk, *chunk, *chunk],
                ).fetchall()
                chunk_set = set(chunk)
                for row in rows:
                    child = str(row["memory_id"])
                    targets = [
                        *_strings(_json(row["evidence_anchors_json"], [])),
                        *_strings(_json(row["supersedes_json"], [])),
                    ]
                    source = _text(row["source_record_id"])
                    if source:
                        targets.append(source)
                    for target in dict.fromkeys(targets):
                        if target in chunk_set:
                            result[target].append(child)

        return {
            identifier: [
                item
                for item in dict.fromkeys(values)
                if item and item != identifier
            ]
            for identifier, values in result.items()
        }

    def _source_references(
        self, record: GraphRecord
    ) -> list[tuple[str, str, tuple[int | None, int | None]]]:
        references: list[tuple[str, str, tuple[int | None, int | None]]] = []

        def add(
            source_id: Any,
            relationship: str,
            start: Any = None,
            end: Any = None,
        ) -> None:
            identifier = _text(source_id)
            if not identifier:
                return
            item = (
                identifier,
                relationship,
                (_optional_integer(start), _optional_integer(end)),
            )
            if all(existing[0] != identifier for existing in references):
                references.append(item)

        if record.layer == "source":
            add(record.memory_id, "self")
        source_id = record.metadata.get("source_record_id")
        add(
            source_id,
            "direct_source",
            record.metadata.get("evidence_char_start"),
            record.metadata.get("evidence_char_end"),
        )
        for parent in _source_parents(record.metadata):
            add(
                parent.get("source_record_id"),
                "slow_graph_source",
                parent.get("evidence_char_start"),
                parent.get("evidence_char_end"),
            )

        memory_ids = [
            *record.evidence_anchors,
            *_claim_memory_ids(record.metadata),
        ]
        related = self._records(memory_ids)
        for identifier in memory_ids:
            item = related.get(identifier)
            if item is None:
                continue
            if item.layer == "source":
                add(item.memory_id, "evidence_anchor")
            else:
                add(
                    item.metadata.get("source_record_id"),
                    "semantic_source",
                    item.metadata.get("evidence_char_start"),
                    item.metadata.get("evidence_char_end"),
                )
        return references

    def _graph_response(
        self,
        *,
        view: str,
        records: Sequence[GraphRecord],
        requested_layers: Sequence[str],
        resolved_layers: Sequence[str],
        limit: int,
        offset: int,
        has_more: bool,
        fallback_layer: str | None = None,
        root_id: str | None = None,
        depth: int | None = None,
    ) -> dict[str, Any]:
        record_map = {record.memory_id: record for record in records}
        edges = self._edges(record_map)
        degrees = {identifier: 0 for identifier in record_map}
        for edge in edges:
            degrees[edge["source"]] = degrees.get(edge["source"], 0) + 1
            degrees[edge["target"]] = degrees.get(edge["target"], 0) + 1
        nodes = [
            _node(
                record,
                records=record_map,
                visible_neighbor_count=degrees.get(record.memory_id, 0),
            )
            for record in records
        ]
        response: dict[str, Any] = {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "scope_name": self.binding.scope_name,
            "snapshot_id": self.binding.snapshot_id,
            "snapshot_state": self.binding.snapshot_state,
            "provisional": self.binding.provisional,
            "view": view,
            "requested_layers": list(requested_layers),
            "resolved_layers": list(resolved_layers),
            "fallback_layer": fallback_layer,
            "nodes": nodes,
            "edges": edges,
            "counts": {
                "nodes": len(nodes),
                "edges": len(edges),
                "slow": sum(node["layer"] == "slow" for node in nodes),
                "fast": sum(node["layer"] == "fast" for node in nodes),
                "source": sum(node["layer"] == "source" for node in nodes),
            },
            "page": self._page(limit=limit, offset=offset, has_more=has_more),
        }
        if root_id is not None:
            response["root_id"] = root_id
        if depth is not None:
            response["depth"] = depth
        return response

    def _edges(self, records: Mapping[str, GraphRecord]) -> list[dict[str, Any]]:
        identifiers = list(records)
        if not identifiers:
            return []
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        def add(
            edge_id: str,
            source: str,
            target: str,
            relation: str,
            weight: float,
            origin: str,
            provenance: Mapping[str, Any],
        ) -> None:
            if source not in records or target not in records or source == target:
                return
            key = (source, target, relation)
            if key in seen:
                return
            seen.add(key)
            edges.append(
                {
                    "id": edge_id,
                    "source": source,
                    "target": target,
                    "type": relation or "related",
                    "weight": max(0.0, min(1.0, float(weight))),
                    "origin": origin,
                    "provenance": dict(provenance),
                }
            )

        with closing(self._connect()) as connection:
            for chunk in _chunks(identifiers, 400):
                marks = _marks(chunk)
                rows = connection.execute(
                    f"""
                    SELECT edge_id,source_memory_id,target_memory_id,edge_type,score
                    FROM memory_edges
                    WHERE scope_id=? AND source_memory_id IN ({marks})
                    ORDER BY score DESC, edge_id ASC
                    """,
                    [self.binding.scope_id, *chunk],
                ).fetchall()
                for row in rows:
                    add(
                        str(row["edge_id"]),
                        str(row["source_memory_id"]),
                        str(row["target_memory_id"]),
                        _text(row["edge_type"]) or "related",
                        _number(row["score"], 0.5),
                        "stored",
                        {
                            "source": "memory_edges",
                            "edge_id": str(row["edge_id"]),
                        },
                    )
        for record in records.values():
            for target in record.supersedes:
                add(
                    f"derived:supersedes:{record.memory_id}:{target}",
                    record.memory_id,
                    target,
                    "supersedes",
                    1.0,
                    "derived",
                    {
                        "source": "record_metadata",
                        "record_id": record.memory_id,
                        "field": "supersedes",
                    },
                )
            for target in record.evidence_anchors:
                add(
                    f"derived:supports:{target}:{record.memory_id}",
                    target,
                    record.memory_id,
                    "supports",
                    0.9,
                    "derived",
                    {
                        "source": "record_evidence_anchors",
                        "record_id": record.memory_id,
                    },
                )
            source_id = _text(record.metadata.get("source_record_id"))
            if source_id:
                add(
                    f"derived:source:{source_id}:{record.memory_id}",
                    source_id,
                    record.memory_id,
                    "derived_from",
                    1.0,
                    "derived",
                    {
                        "source": "record_metadata",
                        "record_id": record.memory_id,
                        "field": "source_record_id",
                    },
                )
            for target in _claim_memory_ids(record.metadata):
                add(
                    f"derived:claim:{target}:{record.memory_id}",
                    target,
                    record.memory_id,
                    "supports",
                    0.9,
                    "derived",
                    {
                        "source": "record_metadata",
                        "record_id": record.memory_id,
                        "field": "claims",
                    },
                )
            for target in _source_parent_ids(record.metadata):
                add(
                    f"derived:source-parent:{target}:{record.memory_id}",
                    target,
                    record.memory_id,
                    "derived_from",
                    1.0,
                    "derived",
                    {
                        "source": "record_metadata",
                        "record_id": record.memory_id,
                        "field": "source_parents",
                    },
                )
        return edges


def _record(row: sqlite3.Row) -> GraphRecord:
    return GraphRecord(
        memory_id=str(row["memory_id"]),
        category=_text(row["category"]),
        slot_key=_text(row["slot_key"]),
        value=_text(row["value"]),
        relation=_text(row["relation"]),
        evidence_anchors=tuple(_strings(_json(row["evidence_anchors_json"], []))),
        salience=_number(row["salience"], 0.0),
        confidence=_number(row["confidence"], 0.0),
        source_kind=_text(row["source_kind"]),
        turn_index=_integer(row["turn_index"], 0),
        state=_text(row["state"]) or "unknown",
        supersedes=tuple(_strings(_json(row["supersedes_json"], []))),
        metadata=_mapping(_json(row["metadata_json"], {})),
    )


def _node(
    record: GraphRecord,
    *,
    records: Mapping[str, GraphRecord],
    visible_neighbor_count: int,
) -> dict[str, Any]:
    metadata = record.metadata
    variant = _text(metadata.get("content_variant"))
    status = _text(metadata.get("status")) or record.state
    actor_roles = _record_actor_roles(record, records)
    actor_role = actor_roles[0] if len(actor_roles) == 1 else None
    agent_values = _record_agent_values(record, records)
    authority = _text(metadata.get("authority"))
    if not authority and record.layer == "source" and actor_role:
        authority = f"{actor_role}_source"
    elif not authority and record.layer == "slow" and actor_roles == ["user"]:
        authority = "derived_user_memory"
    provenance_source = _text(
        metadata.get("provenance_source")
        or metadata.get("source")
        or record.source_kind
    )
    if record.layer == "source":
        role = actor_role or "message"
        occurred = _occurred_at(metadata)
        label = f"{role} source" + (f" - {occurred[:10]}" if occurred else "")
        summary = "Immutable source message. Open evidence to inspect verbatim text."
    else:
        label = _short(record.value, 96)
        summary = _short(record.value, 1_200)
    if record.layer == "slow":
        cluster_id = _text(metadata.get("region_key") or metadata.get("capsule_id"))
    elif record.layer == "source":
        cluster_id = _text(metadata.get("session_id"))
    else:
        cluster_id = _text(
            metadata.get("graph_entity_key")
            or metadata.get("memory_family")
            or metadata.get("subject_signature")
        )
    subject_id = _text(
        metadata.get("subject_signature")
        or metadata.get("graph_entity_key")
        or metadata.get("subject")
        or metadata.get("region_key")
    )
    evidence_count = len(record.evidence_anchors)
    if record.layer == "source":
        evidence_count = 1
    elif metadata.get("source_record_id"):
        evidence_count = max(1, evidence_count)
    source_parent_count = len(_source_parents(metadata))
    evidence_count = max(evidence_count, source_parent_count)
    return {
        "id": record.memory_id,
        "layer": record.layer,
        "kind": _text(metadata.get("node_kind")) or variant or record.category or "memory",
        "category": record.category or "memory",
        "label": label,
        "summary": summary,
        "relation": record.relation or "related",
        "state": record.state,
        "status": status,
        "confidence": max(0.0, min(1.0, record.confidence)),
        "salience": max(0.0, min(1.0, record.salience)),
        "turn_index": record.turn_index,
        "occurred_at": _occurred_at(metadata),
        "subject_id": subject_id or None,
        "cluster_id": cluster_id or None,
        "source_kind": record.source_kind or None,
        "actor_role": actor_role,
        "actor_roles": actor_roles,
        "authority": authority or None,
        "provenance_source": provenance_source or None,
        "evidence_count": evidence_count,
        "visible_neighbor_count": visible_neighbor_count,
        "expandable": bool(evidence_count or visible_neighbor_count),
        "attributes": {
            key: value
            for key, value in {
                "memory_type": metadata.get("memory_type"),
                "memory_family": metadata.get("memory_family"),
                "graph_entity_key": metadata.get("graph_entity_key"),
                "capsule_id": metadata.get("capsule_id"),
                "revision": metadata.get("revision"),
                "canonical_slots": metadata.get("canonical_slots"),
                **{
                    field: values[0] if len(values) == 1 else values
                    for field, values in agent_values.items()
                    if values
                },
            }.items()
            if value not in (None, "", [], {})
        },
    }


def _actor_role(metadata: Mapping[str, Any]) -> str | None:
    roles = _metadata_actor_roles(metadata)
    return roles[0] if len(roles) == 1 else None


def _metadata_actor_roles(metadata: Mapping[str, Any]) -> list[str]:
    roles: list[str] = []

    def add(value: Any) -> None:
        for item in _sequence(value) if isinstance(value, (list, tuple)) else (value,):
            role = _text(item).lower()
            if role in ACTOR_ROLES and role not in roles:
                roles.append(role)

    for field in ("actor_role", "actor_roles", "message_role", "role", "speaker"):
        add(metadata.get(field))
    if _text(metadata.get("authority")).lower() == "user_assertion":
        add("user")
    for parent in _source_parents(metadata):
        for field in ("actor_role", "message_role", "role", "speaker"):
            add(parent.get(field))
    return roles


def _record_actor_roles(
    record: GraphRecord,
    records: Mapping[str, GraphRecord],
    visited: set[str] | None = None,
) -> list[str]:
    seen = set(visited or ())
    if record.memory_id in seen:
        return []
    seen.add(record.memory_id)
    roles = _metadata_actor_roles(record.metadata)
    references = [
        *record.evidence_anchors,
        _text(record.metadata.get("source_record_id")),
        *_claim_memory_ids(record.metadata),
        *_source_parent_ids(record.metadata),
    ]
    for identifier in dict.fromkeys(item for item in references if item):
        parent = records.get(identifier)
        if parent is None:
            continue
        for role in _record_actor_roles(parent, records, seen):
            if role not in roles:
                roles.append(role)
    return roles


def _metadata_agent_values(metadata: Mapping[str, Any]) -> dict[str, list[str]]:
    values = {field: [] for field in AGENT_PROVENANCE_FIELDS}

    def add(field: str, raw: Any) -> None:
        for item in _sequence(raw) if isinstance(raw, (list, tuple)) else (raw,):
            value = _text(item)
            if value and value not in values[field]:
                values[field].append(value)

    plural_fields = {
        "agent_id": "agent_ids",
        "agent_name": "agent_names",
        "agent_role": "agent_roles",
        "agent_specialty": "agent_specialties",
        "agent_team": "agent_teams",
        "target_agent_id": "target_agent_ids",
    }
    for field in AGENT_PROVENANCE_FIELDS:
        add(field, metadata.get(field))
        add(field, metadata.get(plural_fields[field]))
    for parent in _source_parents(metadata):
        for field in AGENT_PROVENANCE_FIELDS:
            add(field, parent.get(field))
            add(field, parent.get(plural_fields[field]))
    return values


def _record_agent_values(
    record: GraphRecord,
    records: Mapping[str, GraphRecord],
    visited: set[str] | None = None,
) -> dict[str, list[str]]:
    seen = set(visited or ())
    if record.memory_id in seen:
        return {field: [] for field in AGENT_PROVENANCE_FIELDS}
    seen.add(record.memory_id)
    values = _metadata_agent_values(record.metadata)
    references = [
        *record.evidence_anchors,
        _text(record.metadata.get("source_record_id")),
        *_claim_memory_ids(record.metadata),
        *_source_parent_ids(record.metadata),
    ]
    for identifier in dict.fromkeys(item for item in references if item):
        parent = records.get(identifier)
        if parent is None:
            continue
        nested = _record_agent_values(parent, records, seen)
        for field, items in nested.items():
            for item in items:
                if item not in values[field]:
                    values[field].append(item)
    return values


def _source_parents(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    for item in _sequence(metadata.get("source_parents")):
        if isinstance(item, Mapping):
            values.append(item)
    for claim in _sequence(metadata.get("claims")):
        if not isinstance(claim, Mapping):
            continue
        for item in _sequence(claim.get("source_parents")):
            if isinstance(item, Mapping):
                values.append(item)
    return values


def _source_parent_ids(metadata: Mapping[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            _text(item.get("source_record_id"))
            for item in _source_parents(metadata)
            if _text(item.get("source_record_id"))
        )
    )


def _claim_memory_ids(metadata: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for claim in _sequence(metadata.get("claims")):
        if not isinstance(claim, Mapping):
            continue
        for field in ("support", "counterevidence"):
            for identifier in _sequence(claim.get(field)):
                clean = _text(identifier)
                if clean and clean not in values:
                    values.append(clean)
    return values


def _record_link_ids(record: GraphRecord) -> list[str]:
    values = [*record.evidence_anchors, *record.supersedes]
    source_id = _text(record.metadata.get("source_record_id"))
    if source_id:
        values.append(source_id)
    values.extend(_source_parent_ids(record.metadata))
    values.extend(_claim_memory_ids(record.metadata))
    return list(dict.fromkeys(item for item in values if item))


def _occurred_at(metadata: Mapping[str, Any]) -> str | None:
    return _text(metadata.get("timestamp") or metadata.get("historical_date")) or None


def _memory_id(value: Any) -> str:
    identifier = _text(value)
    if not identifier or len(identifier) > 512 or "\x00" in identifier:
        raise GraphProjectionError(
            "invalid_memory_node_id", "memory node id is invalid", status_code=422
        )
    return identifier


def _bounded(value: Any, minimum: int, maximum: int, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise GraphProjectionError(
            f"invalid_{field}", f"{field} is invalid", status_code=422
        ) from exc
    if result < minimum or result > maximum:
        raise GraphProjectionError(
            f"invalid_{field}",
            f"{field} must be between {minimum} and {maximum}",
            status_code=422,
        )
    return result


def _json(value: Any, fallback: Any) -> Any:
    if not isinstance(value, str):
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, (list, tuple)) else ()


def _strings(value: Any) -> list[str]:
    return [_text(item) for item in _sequence(value) if _text(item)]


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _integer(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _optional_integer(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _short(value: str, maximum: int) -> str:
    clean = " ".join(value.split())
    if len(clean) <= maximum:
        return clean
    return clean[: max(1, maximum - 3)].rstrip() + "..."


def _marks(values: Sequence[Any] | Iterable[Any]) -> str:
    return ",".join("?" for _ in values)


def _chunks(values: Sequence[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])
