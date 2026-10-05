"""Evidence-bound Session Atlas and per-conversation memory maps.

These projections are deliberately separate from TMCRA's retrieval graph.
They may be regenerated, relabelled, or deleted without changing Writer,
Fast/Slow graph, Source journal, or index state.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from .adapters.v4 import V4AdapterError, V4StorageAdapter
from .control_db import ControlDB
from .costing import journal_deepseek_calls
from .graph_projection import GraphProjectionError, MemoryGraphProjection
from .gpu_scheduler import GpuWorkload, GpuWorkloadScheduler
from .jobs import JobStore
from .narrative_graph import build_narrative_graph
from .personal_knowledge import (
    PERSONAL_KNOWLEDGE_DOMAIN_SCHEMA_VERSION,
    PERSONAL_KNOWLEDGE_MAX_OUTPUT_TOKENS,
    PERSONAL_KNOWLEDGE_PROMPT_VERSION,
    PERSONAL_KNOWLEDGE_REPAIR_SYSTEM_PROMPT,
    PERSONAL_KNOWLEDGE_SCHEMA_VERSION,
    PERSONAL_KNOWLEDGE_SYSTEM_PROMPT,
    PersonalKnowledgeError,
    build_personal_knowledge_batches,
    build_personal_knowledge_fallback,
    merge_personal_knowledge_batches,
    personal_knowledge_source_fingerprint,
    sanitize_personal_knowledge_grounding,
    validate_personal_knowledge_batch,
)
from .visual_atlas import (
    VISUAL_ATLAS_EPISODE_BATCH_PROMPT_VERSION,
    VISUAL_ATLAS_EPISODE_BATCH_REPAIR_SYSTEM_PROMPT,
    VISUAL_ATLAS_EPISODE_BATCH_SYSTEM_PROMPT,
    VISUAL_ATLAS_MAX_RELATIONS_PER_BATCH,
    VISUAL_ATLAS_MAX_RELATIONS_PER_PATCH,
    VISUAL_ATLAS_PROMPT_VERSION,
    VISUAL_ATLAS_SCHEMA_VERSION,
    VISUAL_ATLAS_TAXONOMY_PROMPT_VERSION,
    VISUAL_ATLAS_TAXONOMY_REPAIR_SYSTEM_PROMPT,
    VISUAL_ATLAS_TAXONOMY_SYSTEM_PROMPT,
    VisualAtlasError,
    apply_visual_atlas_patch,
    apply_visual_atlas_taxonomy,
    build_visual_atlas,
    build_visual_atlas_episode_batches,
    build_visual_atlas_taxonomy_payload,
    merge_visual_atlas_episode_batch_patches,
    prepare_visual_atlas_patch_validation,
    sanitize_visual_atlas_episode_batch_patch,
    validate_visual_atlas_episode_batch_patch,
    validate_visual_atlas_episode_batch_patch_with_relation_rejections,
    validate_visual_atlas_taxonomy,
)
from .writer_provider import (
    DEEPSEEK_PROVIDER,
    LOCAL_QWEN_BASE_URL,
    LOCAL_QWEN_GRAPH_SLOT_ID,
    LOCAL_QWEN_MODEL,
    LOCAL_QWEN_PLANNER_SLOT_ID,
    LOCAL_QWEN_PROVIDER,
    OPENAI_COMPATIBLE_PROVIDER,
    validate_openai_compatible_url,
    validate_loopback_openai_compatible_url,
)

try:
    from .writer_provider import (
        DESKTOP_LOCAL_QWEN_BASE_URL,
        DESKTOP_LOCAL_QWEN_MODEL,
        DESKTOP_LOCAL_QWEN_MODELS,
    )
except ImportError:  # Backward-compatible during rolling server upgrades.
    DESKTOP_LOCAL_QWEN_BASE_URL = "http://127.0.0.1:2010/v1"
    DESKTOP_LOCAL_QWEN_MODEL = "tmcra-qwen3-4b-q4km"
    DESKTOP_LOCAL_QWEN_MODELS = frozenset({DESKTOP_LOCAL_QWEN_MODEL})


SESSION_GRAPH_PROVIDER_LOCAL = "local-qwen"
SESSION_GRAPH_PROVIDER_DEDICATED = "dedicated-deepseek"
SESSION_GRAPH_PROVIDER_OPENAI = "openai-compatible"
SESSION_GRAPH_PROVIDER_LOCAL_FIRST = "local-first"
DEDICATED_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEDICATED_DEEPSEEK_MODEL = "deepseek-chat"
SESSION_ATLAS_SCHEMA_VERSION = "tmcra.session-atlas.1"
SESSION_MAP_SCHEMA_VERSION = "tmcra.session-map.1"
SESSION_GRAPH_PROMPT_VERSION = "tmcra-session-graph-agent-v5"
SESSION_GRAPH_MAX_OUTPUT_TOKENS = 12288
SESSION_ATLAS_MAX_OUTPUT_TOKENS = 8192
SESSION_ATLAS_MAX_NODE_UPDATES = 16
SESSION_ATLAS_MAX_EDGE_ADDITIONS = 16
# Production slot 2 has a verified 65,536-token context.  Visual batch prompts
# observed in production remain below 17K input tokens.  A 24K output ceiling
# keeps the request below the slot boundary and prevents valid, schema-bound
# bilingual batches from being cut off at the former 12K ceiling.
VISUAL_ATLAS_TAXONOMY_MAX_OUTPUT_TOKENS = 16384
VISUAL_ATLAS_EPISODE_BATCH_MAX_OUTPUT_TOKENS = 24576
SESSION_GRAPH_ALIAS_SCHEME = "tmcra-request-local-id-alias.1"
ATLAS_KEY = "atlas"
VISUAL_ATLAS_KEY = "visual-atlas"
PERSONAL_KNOWLEDGE_KEY = "knowledge-base"
VISUAL_ATLAS_TAXONOMY_CHECKPOINT_KEY = "visual-atlas-taxonomy"
VISUAL_ATLAS_RUN_CHECKPOINT_KEY = "visual-atlas-run-snapshot"
VISUAL_ATLAS_BATCH_CHECKPOINT_PREFIX = "visual-atlas-batch:"
PERSONAL_KNOWLEDGE_BATCH_CHECKPOINT_PREFIX = "knowledge-base-batch:"
MANUAL_VISUAL_REFRESH_PREFIX = "manual-visual:"
VISUAL_ATLAS_TAXONOMY_CHECKPOINT_SCHEMA_VERSION = (
    "tmcra.visual-atlas-taxonomy-checkpoint.1"
)
VISUAL_ATLAS_BATCH_CHECKPOINT_SCHEMA_VERSION = "tmcra.visual-atlas-batch-checkpoint.1"
VISUAL_ATLAS_RUN_CHECKPOINT_SCHEMA_VERSION = "tmcra.visual-atlas-run-checkpoint.1"
PERSONAL_KNOWLEDGE_BATCH_CHECKPOINT_SCHEMA_VERSION = (
    "tmcra.personal-knowledge-batch-checkpoint.1"
)
SESSION_KEY_PREFIX = "session:"
SESSION_MAP_PROGRESS_WEIGHT = 40
SESSION_ATLAS_PROGRESS_WEIGHT = 15
VISUAL_ATLAS_PROGRESS_WEIGHT = 25
PERSONAL_KNOWLEDGE_PROGRESS_WEIGHT = 20
SESSION_STATUS = frozenset({"active", "paused", "completed", "archived"})
SESSION_NODE_KINDS = frozenset(
    {
        "decision",
        "milestone",
        "goal",
        "issue",
        "preference",
        "relationship",
        "fact",
    }
)
SESSION_EDGE_TYPES = frozenset(
    {
        "related",
        "explains",
        "blocks",
        "enables",
        "contrasts",
        "depends_on",
        "continues",
        "causes",
        "resolves",
        "followed_by",
    }
)
ATLAS_EDGE_TYPES = frozenset({"parent", "continues", "related", "forked_from"})

SESSION_MAP_SYSTEM_PROMPT = """You are TMCRA Session Map Agent.
Transform one evidence-bound conversation projection into a concise user-readable map.

Hard rules:
1. The input node set is immutable. Never add, delete, merge, or rename node IDs.
2. Every statement must be supported by the supplied node summaries and source_record_ids.
3. Preserve speaker and authority boundaries. Assistant progress is not a user fact.
4. Do not output layout coordinates. The client owns layout.
5. Added edges may connect only existing node IDs. Use only the allowed edge types.
6. nodes[].id values are compact request-local aliases. evidence_ids are those
   same aliases, NOT source_record_ids. Copy aliases exactly from nodes[].id;
   the service restores immutable memory IDs before strict validation.
7. Copy every edge type exactly from allowed_edge_types. If an edge cannot meet
   both the node-ID and evidence-ID rules, omit it instead of guessing.
8. Keep uncertainty explicit. Do not invent people, dates, decisions, or outcomes.
9. Do not repeat an edge already present in existing_edges. Add an edge only
   when the supplied summaries genuinely support it. Add at most 24 new edges
   and omit weak or speculative relations.
10. Emit at most one node_update per supplied node. Write labels in the dominant
    language of the conversation. Keep a label below 18 Chinese characters or
    12 English words; avoid opaque IDs and field names. Keep node/thread
    summaries under 80 words and the conversation summary under 160 words.
11. Return one compact JSON object and no prose.

Return exactly:
{
  "title": "short conversation title",
  "summary": "one paragraph summary",
  "node_updates": [
    {"id":"existing id","label":"short label","summary":"grounded summary","kind":"allowed kind","thread_id":"stable short thread label","tags":["short tag"]}
  ],
  "edge_additions": [
    {"source":"existing id","target":"existing id","type":"allowed edge type","weight":0.0,"evidence_ids":["existing id"]}
  ],
  "threads": [
    {"id":"short stable id","title":"short title","summary":"grounded summary","node_ids":["existing id"]}
  ]
}
"""

SESSION_MAP_REPAIR_SYSTEM_PROMPT = """You repair one invalid TMCRA Session Map patch.
Return the complete corrected patch as one JSON object and no prose.

Hard rules:
1. Use only compact aliases copied exactly from nodes[].id for node updates,
   edge endpoints, evidence_ids, and thread node_ids.
2. evidence_ids are request-local node aliases, never source_record_ids. The
   service restores immutable memory IDs before strict validation.
3. Use only allowed_node_kinds and allowed_edge_types supplied in the payload.
4. Remove an invalid or weak edge instead of inventing a replacement.
5. Do not add, delete, merge, or rename memory nodes.
6. Resolve the supplied validation_error. A second invalid patch is rejected.
7. Do not repeat existing_edges. Keep at most 24 edge_additions and use the
   same concise limits as the original Session Map contract.

Return exactly the same JSON shape requested by the Session Map prompt.
"""

SESSION_ATLAS_SYSTEM_PROMPT = """You are TMCRA Global Session Atlas Agent.
Organize an existing list of conversation Sessions into a readable global map.

Hard rules:
1. Sessions are immutable catalog entries. session_id values are compact
   request-local aliases. Never add, delete, merge, or change an alias; the
   service restores immutable Session IDs before strict validation.
2. Parent/fork relationships supplied as trusted metadata are immutable.
3. Cross-session edges may connect only supplied session IDs and must be justified by both session summaries.
4. Do not infer a parent relationship from topical similarity.
5. Do not output layout coordinates. The client owns layout.
6. Existing Session titles and summaries already come from evidence-bound
   Session Maps. Emit a node_update only when it makes one of them materially
   clearer. Respect output_limits exactly: at most 16 node_updates and 16
   edge_additions. Prioritize the strongest cross-session continuations.
   Keep titles under 12 words, summaries under 60 words, and edge reasons under
   24 words.
7. Return one compact JSON object and no prose.

Return exactly:
{
  "node_updates": [
    {"session_id":"existing session id","title":"short title","summary":"grounded summary","topic_tags":["short tag"]}
  ],
  "edge_additions": [
    {"source_session_id":"existing id","target_session_id":"existing id","type":"continues or related","weight":0.0,"reason":"short grounded reason"}
  ]
}
"""

SESSION_ATLAS_REPAIR_SYSTEM_PROMPT = """You repair one invalid TMCRA Session Atlas patch.
Return the complete corrected patch as one JSON object and no prose.

Hard rules:
1. Copy compact Session aliases exactly from sessions[].session_id. The service
   restores immutable Session IDs before strict validation.
2. Use only the allowed_edge_types supplied in the payload.
3. Every added edge needs a non-empty reason grounded in both Session summaries.
4. Remove an invalid or weak edge instead of inventing a replacement.
5. Do not add, delete, merge, or rename Sessions.
6. Resolve the supplied validation_error. A second invalid patch is rejected.
7. Keep at most 16 node_updates and 16 edge_additions, using the same concise
   limits as the original Session Atlas contract.

Return exactly the same JSON shape requested by the Session Atlas prompt.
"""


class SessionGraphError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _text(value: Any, maximum: int = 0) -> str:
    clean = value.strip() if isinstance(value, str) else ""
    if maximum and len(clean) > maximum:
        return clean[:maximum].rstrip()
    return clean


def _items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def projection_progress_percent(
    *,
    total_sessions: int,
    ready_sessions: int,
    atlas_ready: bool,
    graph_ready: bool,
    knowledge_ready: bool,
    all_ready: bool,
) -> int:
    """Return milestone progress without treating every Session as a stage.

    This is intentionally not a time estimate.  A running LLM stage keeps its
    last completed milestone percentage while the API exposes the exact active
    stage separately.
    """

    session_fraction = (
        min(max(0, ready_sessions), total_sessions) / total_sessions
        if total_sessions > 0
        else 0.0
    )
    value = round(session_fraction * SESSION_MAP_PROGRESS_WEIGHT)
    value += SESSION_ATLAS_PROGRESS_WEIGHT if atlas_ready else 0
    value += VISUAL_ATLAS_PROGRESS_WEIGHT if graph_ready else 0
    value += PERSONAL_KNOWLEDGE_PROGRESS_WEIGHT if knowledge_ready else 0
    return 100 if all_ready else min(99, max(0, value))


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}.{digest}"


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _visual_atlas_batch_evidence_payload(value: Any) -> Any:
    """Remove taxonomy-only fields from one immutable evidence batch.

    Visual batch patches cannot update Domain nodes or Session/ Episode domain
    ownership.  Re-running the model because a taxonomy label or domain ID was
    regenerated therefore wastes capacity when every evidence-bearing field is
    unchanged.  The strict patch validator still runs on every reuse.
    """

    if isinstance(value, Mapping):
        return {
            key: _visual_atlas_batch_evidence_payload(item)
            for key, item in value.items()
            if key not in {"batch_id", "domain", "domain_id", "domain_key"}
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_visual_atlas_batch_evidence_payload(item) for item in value]
    return value


def visual_atlas_batch_checkpoint_fingerprint(
    batch: Mapping[str, Any], *, model: str
) -> str:
    return _fingerprint(
        {
            "schema": VISUAL_ATLAS_EPISODE_BATCH_PROMPT_VERSION,
            "model": _text(model, 512),
            "evidence_batch": _visual_atlas_batch_evidence_payload(batch),
        }
    )


def _json_object(value: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SessionGraphError(
            "session_graph_agent_invalid_json",
            "the Session Graph Agent returned invalid JSON",
        ) from exc
    if not isinstance(result, dict):
        raise SessionGraphError(
            "session_graph_agent_invalid_json",
            "the Session Graph Agent response must be a JSON object",
        )
    return result


def session_projection_key(session_id: str) -> str:
    clean = _text(session_id)
    if not clean or len(clean) > 200 or "\x00" in clean:
        raise SessionGraphError("invalid_session_id", "session id is invalid", status_code=422)
    return SESSION_KEY_PREFIX + clean


class SessionGraphStore:
    def __init__(self, database: ControlDB) -> None:
        self.database = database

    @staticmethod
    def _metadata_value(metadata: Mapping[str, Any], names: Sequence[str], maximum: int) -> str | None:
        for name in names:
            value = _text(metadata.get(name), maximum)
            if value:
                return value
        return None

    def record_ingest_in_transaction(
        self,
        connection: Any,
        tenant_id: str,
        scope_name: str,
        session_id: str,
        *,
        metadata: Mapping[str, Any],
        event_fingerprint: str,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else float(now)
        title = self._metadata_value(
            metadata,
            ("session_title", "conversation_title", "thread_title", "title"),
            160,
        )
        source_app = self._metadata_value(
            metadata,
            ("source_app", "integration", "platform", "connector", "client"),
            80,
        )
        native_thread_id = self._metadata_value(
            metadata,
            (
                "native_thread_id",
                "thread_id",
                "conversation_id",
                "source_session_id_hash",
                "source_session_hash",
            ),
            200,
        )
        parent_session_id = self._metadata_value(
            metadata,
            ("parent_session_id", "forked_from_session_id", "source_session_id"),
            200,
        )
        if parent_session_id == session_id:
            parent_session_id = None
        status = self._metadata_value(
            metadata, ("session_status",), 32
        ) or "active"
        if status not in SESSION_STATUS:
            status = "active"
        public_metadata = {
            key: metadata[key]
            for key in ("project", "workspace", "language", "tags")
            if key in metadata
        }
        encoded = json.dumps(
            public_metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(encoded.encode("utf-8")) > 4096:
            encoded = "{}"
        connection.execute(
            """
            INSERT INTO session_graph_metadata(
                tenant_id,scope_name,session_id,title,source_app,native_thread_id,
                parent_session_id,session_status,metadata_json,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(tenant_id,scope_name,session_id) DO UPDATE SET
                title=COALESCE(excluded.title,session_graph_metadata.title),
                source_app=COALESCE(excluded.source_app,session_graph_metadata.source_app),
                native_thread_id=COALESCE(
                    excluded.native_thread_id,session_graph_metadata.native_thread_id
                ),
                parent_session_id=COALESCE(
                    excluded.parent_session_id,session_graph_metadata.parent_session_id
                ),
                session_status=excluded.session_status,
                metadata_json=CASE WHEN excluded.metadata_json='{}'
                    THEN session_graph_metadata.metadata_json
                    ELSE excluded.metadata_json END,
                updated_at=excluded.updated_at
            """,
            (
                tenant_id,
                scope_name,
                session_id,
                title,
                source_app,
                native_thread_id,
                parent_session_id,
                status,
                encoded,
                now,
                now,
            ),
        )
        # Admission only records trusted Session metadata. Agent work is
        # scheduled after Writer commit and subject to the message-delta gate.

    @staticmethod
    def enqueue_in_transaction(
        connection: Any,
        tenant_id: str,
        scope_name: str,
        projection_key: str,
        *,
        source_fingerprint: str,
        due_at: float,
        now: float,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_graph_refresh_queue(
                tenant_id,scope_name,projection_key,state,source_fingerprint,
                pending_source_fingerprint,
                due_at,attempts,created_at,updated_at
            ) VALUES(?,?,?,'dirty',?,NULL,?,0,?,?)
            ON CONFLICT(tenant_id,scope_name,projection_key) DO UPDATE SET
                state=CASE
                    WHEN memory_graph_refresh_queue.state='running'
                    THEN memory_graph_refresh_queue.state ELSE 'dirty' END,
                source_fingerprint=CASE
                    WHEN memory_graph_refresh_queue.state='running'
                    THEN memory_graph_refresh_queue.source_fingerprint
                    ELSE excluded.source_fingerprint END,
                pending_source_fingerprint=CASE
                    WHEN memory_graph_refresh_queue.state='running'
                     AND memory_graph_refresh_queue.source_fingerprint
                         <> excluded.source_fingerprint
                    THEN excluded.source_fingerprint
                    WHEN memory_graph_refresh_queue.state='running'
                    THEN memory_graph_refresh_queue.pending_source_fingerprint
                    ELSE NULL END,
                due_at=CASE
                    WHEN memory_graph_refresh_queue.state='running'
                    THEN memory_graph_refresh_queue.due_at ELSE excluded.due_at END,
                attempts=CASE
                    WHEN memory_graph_refresh_queue.state='running'
                    THEN memory_graph_refresh_queue.attempts ELSE 0 END,
                claimed_at=CASE
                    WHEN memory_graph_refresh_queue.state='running'
                    THEN memory_graph_refresh_queue.claimed_at ELSE NULL END,
                last_error=CASE
                    WHEN memory_graph_refresh_queue.state='running'
                    THEN memory_graph_refresh_queue.last_error ELSE NULL END,
                heartbeat_at=CASE
                    WHEN memory_graph_refresh_queue.state='running'
                    THEN memory_graph_refresh_queue.heartbeat_at ELSE NULL END,
                progress_stage=CASE
                    WHEN memory_graph_refresh_queue.state='running'
                    THEN memory_graph_refresh_queue.progress_stage ELSE NULL END,
                progress_completed=CASE
                    WHEN memory_graph_refresh_queue.state='running'
                    THEN memory_graph_refresh_queue.progress_completed ELSE NULL END,
                progress_total=CASE
                    WHEN memory_graph_refresh_queue.state='running'
                    THEN memory_graph_refresh_queue.progress_total ELSE NULL END,
                updated_at=CASE
                    WHEN memory_graph_refresh_queue.state='running'
                    THEN memory_graph_refresh_queue.updated_at ELSE excluded.updated_at END
            """,
            (
                tenant_id,
                scope_name,
                projection_key,
                source_fingerprint,
                due_at,
                now,
                now,
            ),
        )

    def enqueue(
        self,
        tenant_id: str,
        scope_name: str,
        projection_key: str,
        *,
        source_fingerprint: str,
        delay_seconds: float = 0.0,
    ) -> None:
        now = time.time()
        with self.database.transaction() as connection:
            self.enqueue_in_transaction(
                connection,
                tenant_id,
                scope_name,
                projection_key,
                source_fingerprint=source_fingerprint,
                due_at=now + max(0.0, delay_seconds),
                now=now,
            )

    def requeue_superseded(
        self,
        task: Mapping[str, Any],
        *,
        source_fingerprint: str,
    ) -> bool:
        """Finish a stale attempt and immediately queue its newest source.

        ``enqueue`` deliberately preserves an in-flight snapshot and records a
        different source as pending.  A worker that discovers before doing any
        work that its claimed source is already stale must take the opposite
        path: release the old lease now.  Leaving that row in ``running`` makes
        the new source wait for stale-lease recovery even though no worker owns
        it anymore.
        """

        now = time.time()
        with self.database.transaction() as connection:
            updated = connection.execute(
                """
                UPDATE memory_graph_refresh_queue
                SET state='dirty',
                    source_fingerprint=COALESCE(
                        NULLIF(pending_source_fingerprint,source_fingerprint),
                        ?
                    ),
                    pending_source_fingerprint=NULL,
                    due_at=MIN(due_at,?),
                    attempts=0,claimed_at=NULL,heartbeat_at=NULL,
                    progress_stage='queued_after_source_change',
                    progress_completed=0,progress_total=NULL,
                    last_error=NULL,updated_at=?
                WHERE tenant_id=? AND scope_name=? AND projection_key=?
                  AND state='running' AND source_fingerprint=?
                  AND attempts=?
                """,
                (
                    source_fingerprint,
                    now,
                    now,
                    task["tenant_id"],
                    task["scope_name"],
                    task["projection_key"],
                    task["source_fingerprint"],
                    int(task.get("attempts") or -1),
                ),
            )
        return updated.rowcount == 1

    def cancel_dirty_refresh(
        self,
        tenant_id: str,
        scope_name: str,
        projection_key: str,
        *,
        stage: str,
    ) -> bool:
        """Cancel one unclaimed refresh while retaining its ready view."""

        now = time.time()
        with self.database.transaction() as connection:
            updated = connection.execute(
                """
                UPDATE memory_graph_refresh_queue
                SET state='clean',
                    source_fingerprint=COALESCE(
                        pending_source_fingerprint,source_fingerprint
                    ),
                    pending_source_fingerprint=NULL,
                    due_at=?,attempts=0,claimed_at=NULL,heartbeat_at=NULL,
                    progress_stage=?,progress_completed=0,progress_total=NULL,
                    last_error=NULL,updated_at=?
                WHERE tenant_id=? AND scope_name=? AND projection_key=?
                  AND state='dirty'
                """,
                (
                    now,
                    _text(stage, 80),
                    now,
                    tenant_id,
                    scope_name,
                    projection_key,
                ),
            )
        return updated.rowcount == 1

    def sessions(self, tenant_id: str, scope_name: str) -> list[dict[str, Any]]:
        with self.database.transaction(immediate=False) as connection:
            rows = connection.execute(
                """
                SELECT sessions.session_id,sessions.created_at,sessions.last_ingest_at,
                       sessions.ingest_request_count,sessions.message_count,
                       metadata.title,metadata.source_app,metadata.native_thread_id,
                       metadata.parent_session_id,metadata.session_status,
                       metadata.metadata_json
                FROM scope_sessions AS sessions
                LEFT JOIN session_graph_metadata AS metadata
                  ON metadata.tenant_id=sessions.tenant_id
                 AND metadata.scope_name=sessions.scope_name
                 AND metadata.session_id=sessions.session_id
                WHERE sessions.tenant_id=? AND sessions.scope_name=?
                ORDER BY sessions.last_ingest_at DESC,sessions.session_id
                LIMIT 5000
                """,
                (tenant_id, scope_name),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except (TypeError, ValueError):
                metadata = {}
            result.append(
                {
                    "session_id": str(row["session_id"]),
                    "created_at": float(row["created_at"]),
                    "last_ingest_at": float(row["last_ingest_at"]),
                    "ingest_request_count": int(row["ingest_request_count"]),
                    "message_count": int(row["message_count"]),
                    "title": _text(row["title"], 160) or None,
                    "source_app": _text(row["source_app"], 80) or None,
                    "native_thread_id": _text(row["native_thread_id"], 200) or None,
                    "parent_session_id": _text(row["parent_session_id"], 200) or None,
                    "status": _text(row["session_status"], 32) or "active",
                    "metadata": metadata if isinstance(metadata, dict) else {},
                }
            )
        return result

    def session_projection_message_watermark(
        self, tenant_id: str, scope_name: str
    ) -> int:
        """Return the evidence-bound watermark of completed Session maps.

        The admission counter can be inflated by retries, so heavy projections
        follow the Agent checkpoints stored in completed Session maps.
        """

        with self.database.transaction(immediate=False) as connection:
            row = connection.execute(
                """
                SELECT COALESCE(SUM(
                    CAST(COALESCE(
                        json_extract(
                            projection_json,
                            '$.agent_checkpoint.message_count'
                        ),
                        json_extract(projection_json,'$.message_count'),
                        0
                    ) AS INTEGER)
                ),0) AS message_watermark
                FROM memory_graph_views
                WHERE tenant_id=? AND scope_name=?
                  AND projection_key LIKE 'session:%'
                  AND schema_version=?
                  AND json_valid(projection_json)=1
                """,
                (tenant_id, scope_name, SESSION_MAP_SCHEMA_VERSION),
            ).fetchone()
        return max(0, int(row["message_watermark"] if row else 0))

    def scopes_with_sessions(self) -> list[dict[str, Any]]:
        """Return persisted scopes which need projection reconciliation.

        This query is intentionally limited to the control database.  Startup
        reconciliation must never open every Source graph before the API can
        become ready.
        """

        with self.database.transaction(immediate=False) as connection:
            rows = connection.execute(
                """
                SELECT tenant_id,scope_name,COUNT(*) AS session_count,
                       SUM(message_count) AS message_count,
                       MAX(last_ingest_at) AS last_ingest_at
                FROM scope_sessions
                GROUP BY tenant_id,scope_name
                ORDER BY MAX(last_ingest_at) DESC
                """
            ).fetchall()
        return [
            {
                "tenant_id": str(row["tenant_id"]),
                "scope_name": str(row["scope_name"]),
                "session_count": int(row["session_count"] or 0),
                "message_count": int(row["message_count"] or 0),
                "last_ingest_at": float(row["last_ingest_at"] or 0.0),
            }
            for row in rows
        ]

    def session(
        self, tenant_id: str, scope_name: str, session_id: str
    ) -> dict[str, Any] | None:
        for item in self.sessions(tenant_id, scope_name):
            if item["session_id"] == session_id:
                return item
        return None

    def get_view(
        self, tenant_id: str, scope_name: str, projection_key: str
    ) -> dict[str, Any] | None:
        with self.database.transaction(immediate=False) as connection:
            row = connection.execute(
                """
                SELECT schema_version,source_snapshot_id,source_fingerprint,
                       generator,model,prompt_version,projection_json,
                       created_at,updated_at
                FROM memory_graph_views
                WHERE tenant_id=? AND scope_name=? AND projection_key=?
                """,
                (tenant_id, scope_name, projection_key),
            ).fetchone()
        if row is None:
            return None
        try:
            projection = json.loads(str(row["projection_json"]))
        except (TypeError, ValueError):
            return None
        if not isinstance(projection, dict):
            return None
        return {
            "projection": projection,
            "schema_version": str(row["schema_version"]),
            "source_snapshot_id": (
                None if row["source_snapshot_id"] is None else str(row["source_snapshot_id"])
            ),
            "source_fingerprint": str(row["source_fingerprint"]),
            "generator": str(row["generator"]),
            "model": None if row["model"] is None else str(row["model"]),
            "prompt_version": (
                None if row["prompt_version"] is None else str(row["prompt_version"])
            ),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def delete_views_by_prefix_except(
        self,
        tenant_id: str,
        scope_name: str,
        prefix: str,
        keep: Sequence[str],
    ) -> int:
        """Remove obsolete durable batch checkpoints after a complete publish."""

        keep_keys = {str(item) for item in keep if str(item)}
        with self.database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT projection_key
                FROM memory_graph_views
                WHERE tenant_id=? AND scope_name=? AND projection_key LIKE ?
                """,
                (tenant_id, scope_name, prefix + "%"),
            ).fetchall()
            stale = [
                (tenant_id, scope_name, str(row["projection_key"]))
                for row in rows
                if str(row["projection_key"]) not in keep_keys
            ]
            if stale:
                connection.executemany(
                    """
                    DELETE FROM memory_graph_views
                    WHERE tenant_id=? AND scope_name=? AND projection_key=?
                    """,
                    stale,
                )
        return len(stale)

    def put_view(
        self,
        tenant_id: str,
        scope_name: str,
        projection_key: str,
        projection: Mapping[str, Any],
        *,
        source_snapshot_id: str | None,
        source_fingerprint: str,
        generator: str,
        model: str | None = None,
        prompt_version: str | None = None,
        mark_clean: bool = False,
        expected_queue_fingerprint: str | None = None,
        expected_queue_attempts: int | None = None,
    ) -> bool:
        now = time.time()
        encoded = json.dumps(
            projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.database.transaction() as connection:
            if expected_queue_fingerprint is not None:
                queue_row = connection.execute(
                    """
                    SELECT state,source_fingerprint,attempts
                    FROM memory_graph_refresh_queue
                    WHERE tenant_id=? AND scope_name=? AND projection_key=?
                    """,
                    (tenant_id, scope_name, projection_key),
                ).fetchone()
                if (
                    queue_row is None
                    or str(queue_row["state"]) != "running"
                    or str(queue_row["source_fingerprint"])
                    != expected_queue_fingerprint
                    or (
                        expected_queue_attempts is not None
                        and int(queue_row["attempts"] or 0)
                        != int(expected_queue_attempts)
                    )
                ):
                    return False
            connection.execute(
                """
                INSERT INTO memory_graph_views(
                    tenant_id,scope_name,projection_key,schema_version,
                    source_snapshot_id,source_fingerprint,generator,model,
                    prompt_version,projection_json,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(tenant_id,scope_name,projection_key) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    source_snapshot_id=excluded.source_snapshot_id,
                    source_fingerprint=excluded.source_fingerprint,
                    generator=excluded.generator,model=excluded.model,
                    prompt_version=excluded.prompt_version,
                    projection_json=excluded.projection_json,
                    updated_at=excluded.updated_at
                """,
                (
                    tenant_id,
                    scope_name,
                    projection_key,
                    str(projection.get("schema_version") or "unknown"),
                    source_snapshot_id,
                    source_fingerprint,
                    generator,
                    model,
                    prompt_version,
                    encoded,
                    now,
                    now,
                ),
            )
            if mark_clean:
                updated = connection.execute(
                    """
                    UPDATE memory_graph_refresh_queue
                    SET state=CASE
                            WHEN pending_source_fingerprint IS NOT NULL
                             AND pending_source_fingerprint<>source_fingerprint
                            THEN 'dirty' ELSE 'clean' END,
                        source_fingerprint=CASE
                            WHEN pending_source_fingerprint IS NOT NULL
                             AND pending_source_fingerprint<>source_fingerprint
                            THEN pending_source_fingerprint ELSE source_fingerprint END,
                        due_at=CASE
                            WHEN pending_source_fingerprint IS NOT NULL
                             AND pending_source_fingerprint<>source_fingerprint
                            THEN ? ELSE due_at END,
                        pending_source_fingerprint=NULL,
                        claimed_at=NULL,heartbeat_at=NULL,
                        progress_stage=CASE
                            WHEN pending_source_fingerprint IS NOT NULL
                             AND pending_source_fingerprint<>source_fingerprint
                            THEN 'queued_after_snapshot' ELSE 'ready' END,
                        progress_completed=CASE
                            WHEN pending_source_fingerprint IS NOT NULL
                             AND pending_source_fingerprint<>source_fingerprint
                            THEN 0 ELSE COALESCE(progress_total,1) END,
                        progress_total=CASE
                            WHEN pending_source_fingerprint IS NOT NULL
                             AND pending_source_fingerprint<>source_fingerprint
                            THEN NULL ELSE COALESCE(progress_total,1) END,
                        last_error=NULL,updated_at=?
                    WHERE tenant_id=? AND scope_name=? AND projection_key=?
                      AND state='running' AND source_fingerprint=?
                      AND (? IS NULL OR attempts=?)
                    """,
                    (
                        now,
                        now,
                        tenant_id,
                        scope_name,
                        projection_key,
                        expected_queue_fingerprint or source_fingerprint,
                        expected_queue_attempts,
                        expected_queue_attempts,
                    ),
                )
                if updated.rowcount != 1:
                    return False
        return True

    def claim(self, *, stale_after_seconds: float = 300.0) -> dict[str, Any] | None:
        now = time.time()
        with self.database.transaction() as connection:
            connection.execute(
                """
                UPDATE memory_graph_refresh_queue
                SET state='dirty',claimed_at=NULL,heartbeat_at=NULL,due_at=?,updated_at=?
                WHERE state='running' AND claimed_at IS NOT NULL
                  AND COALESCE(heartbeat_at,claimed_at)<?
                """,
                (now, now, now - stale_after_seconds),
            )
            row = connection.execute(
                """
                SELECT candidate.tenant_id,candidate.scope_name,
                       candidate.projection_key,candidate.source_fingerprint,
                       candidate.attempts
                FROM memory_graph_refresh_queue AS candidate
                WHERE candidate.state='dirty' AND candidate.due_at<=?
                  AND (
                    candidate.projection_key LIKE 'session:%'
                    OR (
                      candidate.projection_key='atlas'
                      AND NOT EXISTS (
                        SELECT 1 FROM memory_graph_refresh_queue AS dependency
                        WHERE dependency.tenant_id=candidate.tenant_id
                          AND dependency.scope_name=candidate.scope_name
                          AND dependency.projection_key LIKE 'session:%'
                          AND dependency.state IN ('dirty','running')
                      )
                    )
                    OR (
                      candidate.projection_key='visual-atlas'
                      AND NOT EXISTS (
                        SELECT 1 FROM memory_graph_refresh_queue AS dependency
                        WHERE dependency.tenant_id=candidate.tenant_id
                          AND dependency.scope_name=candidate.scope_name
                          AND dependency.state IN ('dirty','running')
                          AND (
                            dependency.projection_key LIKE 'session:%'
                            OR dependency.projection_key='atlas'
                          )
                      )
                    )
                    OR (
                      candidate.projection_key='knowledge-base'
                      AND NOT EXISTS (
                        SELECT 1 FROM memory_graph_refresh_queue AS dependency
                        WHERE dependency.tenant_id=candidate.tenant_id
                          AND dependency.scope_name=candidate.scope_name
                          AND dependency.state IN ('dirty','running')
                           AND (
                             dependency.projection_key LIKE 'session:%'
                             OR dependency.projection_key='atlas'
                             OR (
                               dependency.projection_key='visual-atlas'
                               AND NOT EXISTS (
                                 SELECT 1 FROM memory_graph_views AS ready_visual
                                 WHERE ready_visual.tenant_id=candidate.tenant_id
                                   AND ready_visual.scope_name=candidate.scope_name
                                   AND ready_visual.projection_key='visual-atlas'
                                   AND ready_visual.schema_version=?
                                   AND json_extract(
                                         ready_visual.projection_json,
                                         '$.projection_state'
                                       )='ready'
                                   AND json_extract(
                                         ready_visual.projection_json,
                                         '$.full_projection'
                                       )=1
                                   AND COALESCE(
                                         json_extract(
                                           ready_visual.projection_json,
                                           '$.truncated'
                                         ),
                                         0
                                       )=0
                               )
                             )
                           )
                       )
                    )
                    OR candidate.projection_key NOT LIKE 'session:%'
                       AND candidate.projection_key NOT IN (
                         'atlas','visual-atlas','knowledge-base'
                       )
                  )
                ORDER BY candidate.due_at,candidate.updated_at,
                         CASE
                           WHEN candidate.projection_key='atlas' THEN 0
                           WHEN candidate.projection_key='visual-atlas' THEN 1
                           WHEN candidate.projection_key='knowledge-base' THEN 2
                           WHEN candidate.projection_key LIKE 'session:%' THEN 3
                           ELSE 4
                         END
                LIMIT 1
                """,
                (now, VISUAL_ATLAS_SCHEMA_VERSION),
            ).fetchone()
            if row is None:
                return None
            updated = connection.execute(
                """
                UPDATE memory_graph_refresh_queue
                SET state='running',attempts=attempts+1,claimed_at=?,heartbeat_at=?,
                    progress_stage=projection_key,progress_completed=0,
                    progress_total=NULL,last_error=NULL,updated_at=?
                WHERE tenant_id=? AND scope_name=? AND projection_key=?
                  AND state='dirty'
                """,
                (
                    now,
                    now,
                    now,
                    row["tenant_id"],
                    row["scope_name"],
                    row["projection_key"],
                ),
            )
            if updated.rowcount != 1:
                return None
        return {
            "tenant_id": str(row["tenant_id"]),
            "scope_name": str(row["scope_name"]),
            "projection_key": str(row["projection_key"]),
            "source_fingerprint": str(row["source_fingerprint"]),
            "attempts": int(row["attempts"]) + 1,
        }

    def recover_interrupted_refreshes(self) -> int:
        """Return tasks left running by a previous service process to the queue."""

        now = time.time()
        with self.database.transaction() as connection:
            updated = connection.execute(
                """
                UPDATE memory_graph_refresh_queue
                SET state='dirty',due_at=?,claimed_at=NULL,heartbeat_at=NULL,
                    last_error='interrupted by service restart',updated_at=?
                WHERE state='running'
                """,
                (now, now),
            )
        return max(0, int(updated.rowcount))

    def heartbeat(
        self,
        task: Mapping[str, Any],
        *,
        stage: str,
        completed: int,
        total: int | None,
    ) -> bool:
        """Persist exact projection progress and renew the worker lease."""

        now = time.time()
        safe_total = None if total is None else max(0, int(total))
        safe_completed = max(0, int(completed))
        if safe_total is not None:
            safe_completed = min(safe_completed, safe_total)
        with self.database.transaction() as connection:
            updated = connection.execute(
                """
                UPDATE memory_graph_refresh_queue
                SET heartbeat_at=?,progress_stage=?,progress_completed=?,
                    progress_total=?,updated_at=?
                WHERE tenant_id=? AND scope_name=? AND projection_key=?
                  AND state='running' AND source_fingerprint=?
                  AND attempts=?
                """,
                (
                    now,
                    _text(stage, 80),
                    safe_completed,
                    safe_total,
                    now,
                    task["tenant_id"],
                    task["scope_name"],
                    task["projection_key"],
                    task["source_fingerprint"],
                    int(task.get("attempts") or -1),
                ),
            )
        return updated.rowcount == 1

    def renew(self, task: Mapping[str, Any]) -> bool:
        """Renew only the active attempt lease without changing progress."""

        now = time.time()
        with self.database.transaction() as connection:
            updated = connection.execute(
                """
                UPDATE memory_graph_refresh_queue
                SET heartbeat_at=?,updated_at=?
                WHERE tenant_id=? AND scope_name=? AND projection_key=?
                  AND state='running' AND source_fingerprint=?
                  AND attempts=?
                """,
                (
                    now,
                    now,
                    task["tenant_id"],
                    task["scope_name"],
                    task["projection_key"],
                    task["source_fingerprint"],
                    int(task.get("attempts") or -1),
                ),
            )
        return updated.rowcount == 1

    def defer(self, task: Mapping[str, Any], *, seconds: float, reason: str) -> None:
        now = time.time()
        with self.database.transaction() as connection:
            connection.execute(
                """
                UPDATE memory_graph_refresh_queue
                SET state='dirty',due_at=?,claimed_at=NULL,heartbeat_at=NULL,
                    last_error=?,updated_at=?
                WHERE tenant_id=? AND scope_name=? AND projection_key=?
                  AND state='running' AND source_fingerprint=?
                  AND attempts=?
                """,
                (
                    now + max(1.0, seconds),
                    _text(reason, 1000),
                    now,
                    task["tenant_id"],
                    task["scope_name"],
                    task["projection_key"],
                    task["source_fingerprint"],
                    int(task.get("attempts") or -1),
                ),
            )

    def fail(self, task: Mapping[str, Any], error: BaseException) -> None:
        now = time.time()
        message = f"{type(error).__name__}:{error}"[:2000]
        with self.database.transaction() as connection:
            connection.execute(
                """
                UPDATE memory_graph_refresh_queue
                SET state='failed',claimed_at=NULL,heartbeat_at=NULL,
                    last_error=?,updated_at=?
                WHERE tenant_id=? AND scope_name=? AND projection_key=?
                  AND state='running' AND source_fingerprint=?
                  AND attempts=?
                """,
                (
                    message,
                    now,
                    task["tenant_id"],
                    task["scope_name"],
                    task["projection_key"],
                    task["source_fingerprint"],
                    int(task.get("attempts") or -1),
                ),
            )

    def refresh_state(
        self, tenant_id: str, scope_name: str, projection_key: str
    ) -> dict[str, Any] | None:
        with self.database.transaction(immediate=False) as connection:
            row = connection.execute(
                """
                SELECT state,source_fingerprint,attempts,due_at,claimed_at,heartbeat_at,
                       progress_stage,progress_completed,progress_total,
                       pending_source_fingerprint,last_error,created_at,updated_at
                FROM memory_graph_refresh_queue
                WHERE tenant_id=? AND scope_name=? AND projection_key=?
                """,
                (tenant_id, scope_name, projection_key),
            ).fetchone()
        if row is None:
            return None
        return {
            "state": str(row["state"]),
            "source_fingerprint": str(row["source_fingerprint"]),
            "attempts": int(row["attempts"]),
            "due_at": float(row["due_at"]),
            "claimed_at": None if row["claimed_at"] is None else float(row["claimed_at"]),
            "heartbeat_at": (
                None if row["heartbeat_at"] is None else float(row["heartbeat_at"])
            ),
            "progress_stage": (
                None if row["progress_stage"] is None else str(row["progress_stage"])
            ),
            "progress_completed": (
                None
                if row["progress_completed"] is None
                else int(row["progress_completed"])
            ),
            "progress_total": (
                None if row["progress_total"] is None else int(row["progress_total"])
            ),
            "pending_source_fingerprint": (
                None
                if row["pending_source_fingerprint"] is None
                else str(row["pending_source_fingerprint"])
            ),
            "last_error": None if row["last_error"] is None else str(row["last_error"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def refresh_states(self, tenant_id: str, scope_name: str) -> dict[str, dict[str, Any]]:
        with self.database.transaction(immediate=False) as connection:
            rows = connection.execute(
                """
                SELECT projection_key,state,source_fingerprint,attempts,due_at,claimed_at,heartbeat_at,
                       progress_stage,progress_completed,progress_total,
                       pending_source_fingerprint,last_error,created_at,updated_at
                FROM memory_graph_refresh_queue
                WHERE tenant_id=? AND scope_name=?
                """,
                (tenant_id, scope_name),
            ).fetchall()
        return {
            str(row["projection_key"]): {
                "state": str(row["state"]),
                "source_fingerprint": str(row["source_fingerprint"]),
                "attempts": int(row["attempts"]),
                "due_at": float(row["due_at"]),
                "claimed_at": (
                    None if row["claimed_at"] is None else float(row["claimed_at"])
                ),
                "heartbeat_at": (
                    None
                    if row["heartbeat_at"] is None
                    else float(row["heartbeat_at"])
                ),
                "progress_stage": (
                    None
                    if row["progress_stage"] is None
                    else str(row["progress_stage"])
                ),
                "progress_completed": (
                    None
                    if row["progress_completed"] is None
                    else int(row["progress_completed"])
                ),
                "progress_total": (
                    None
                    if row["progress_total"] is None
                    else int(row["progress_total"])
                ),
                "pending_source_fingerprint": (
                    None
                    if row["pending_source_fingerprint"] is None
                    else str(row["pending_source_fingerprint"])
                ),
                "last_error": (
                    None if row["last_error"] is None else str(row["last_error"])
                ),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
            }
            for row in rows
        }

    def scope_has_pending_sessions(self, tenant_id: str, scope_name: str) -> bool:
        with self.database.transaction(immediate=False) as connection:
            row = connection.execute(
                """
                SELECT 1 FROM memory_graph_refresh_queue
                WHERE tenant_id=? AND scope_name=?
                  AND projection_key LIKE 'session:%'
                  AND state IN ('dirty','running')
                LIMIT 1
                """,
                (tenant_id, scope_name),
            ).fetchone()
        return row is not None

    def has_due_refresh(self, *, now: float | None = None) -> bool:
        moment = time.time() if now is None else float(now)
        with self.database.transaction(immediate=False) as connection:
            row = connection.execute(
                """
                SELECT 1 FROM memory_graph_refresh_queue
                WHERE state='dirty' AND due_at<=?
                LIMIT 1
                """,
                (moment,),
            ).fetchone()
        return row is not None

    def production_work_pending(self) -> bool:
        """Return whether a user-facing write/index/Slow job needs capacity."""

        with self.database.transaction(immediate=False) as connection:
            row = connection.execute(
                """
                SELECT 1 FROM jobs
                WHERE state IN ('pending','running')
                LIMIT 1
                """
            ).fetchone()
        return row is not None


def build_session_map(
    source_graph: Mapping[str, Any],
    session: Mapping[str, Any],
    *,
    limit: int = 60,
) -> dict[str, Any]:
    session_id = _text(session.get("session_id"), 200)
    if not session_id:
        raise SessionGraphError("invalid_session_id", "session id is required", status_code=422)
    narrative = build_narrative_graph(source_graph, limit=max(1, min(60, limit)))
    nodes = [dict(item) for item in _items(narrative.get("nodes"))]
    edges = [dict(item) for item in _items(narrative.get("edges"))]
    for node in nodes:
        attributes = dict(node.get("attributes")) if isinstance(node.get("attributes"), Mapping) else {}
        attributes["session_id"] = session_id
        node["attributes"] = attributes
    source_nodes = [
        item for item in _items(source_graph.get("nodes")) if _text(item.get("layer")) == "source"
    ]
    first_label = _text(nodes[0].get("label"), 96) if nodes else ""
    title = _text(session.get("title"), 160) or first_label or f"Session {session_id[:12]}"
    summaries = [_text(item.get("summary"), 240) for item in nodes[:3]]
    summary = " ".join(item for item in summaries if item)
    if not summary:
        summary = "This conversation has not produced a semantic memory map yet."
    started = min(
        (_text(item.get("occurred_at")) for item in source_nodes if _text(item.get("occurred_at"))),
        default=None,
    )
    updated = max(
        (_text(item.get("occurred_at")) for item in source_nodes if _text(item.get("occurred_at"))),
        default=None,
    )
    result = {
        "schema_version": SESSION_MAP_SCHEMA_VERSION,
        "scope_name": _text(source_graph.get("scope_name")),
        "session_id": session_id,
        "snapshot_id": _text(source_graph.get("snapshot_id")),
        "snapshot_state": _text(source_graph.get("snapshot_state")) or "committed",
        "provisional": bool(source_graph.get("provisional")),
        "view": "session_map",
        "projection_state": "fallback",
        "generated_by": "deterministic-evidence-projection",
        "prompt_version": None,
        "model": None,
        "title": title,
        "summary": summary,
        "status": _text(session.get("status"), 32) or "active",
        "source_app": _text(session.get("source_app"), 80) or None,
        "native_thread_id": _text(session.get("native_thread_id"), 200) or None,
        "parent_session_id": _text(session.get("parent_session_id"), 200) or None,
        "created_at": session.get("created_at"),
        "updated_at": session.get("last_ingest_at"),
        "message_count": int(session.get("message_count") or 0),
        "source_record_count": int(source_graph.get("source_record_count") or len(source_nodes)),
        "semantic_record_count": int(source_graph.get("semantic_record_count") or len(nodes)),
        "nodes": nodes,
        "edges": edges,
        "threads": list(narrative.get("threads") or []),
        "counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "threads": len(list(narrative.get("threads") or [])),
            "source_records": int(source_graph.get("source_record_count") or len(source_nodes)),
        },
        "time_range": {"started_at": started, "updated_at": updated},
        "evidence_binding": {
            "strategy": "immutable_source_session_id",
            "source_text_exposed": False,
            "source_record_ids": [str(item.get("id")) for item in source_nodes if item.get("id")],
        },
    }
    return result


def apply_session_map_patch(
    base: Mapping[str, Any], patch: Mapping[str, Any]
) -> dict[str, Any]:
    result = json.loads(json.dumps(base, ensure_ascii=False))
    nodes = {str(item["id"]): item for item in result.get("nodes", []) if isinstance(item, dict) and item.get("id")}
    if not nodes and (_items(patch.get("node_updates")) or _items(patch.get("edge_additions"))):
        raise SessionGraphError("session_graph_agent_invalid_patch", "an empty map cannot be expanded by the Agent")
    title = _text(patch.get("title"), 160)
    summary = _text(patch.get("summary"), 1200)
    if title:
        result["title"] = title
    if summary:
        result["summary"] = summary
    for update in _items(patch.get("node_updates")):
        identifier = _text(update.get("id"), 512)
        if identifier not in nodes:
            raise SessionGraphError("session_graph_agent_invalid_patch", "Agent referenced an unknown memory node")
        node = nodes[identifier]
        label = _text(update.get("label"), 96)
        node_summary = _text(update.get("summary"), 1200)
        kind = _text(update.get("kind"), 40).lower()
        if kind and kind not in SESSION_NODE_KINDS:
            raise SessionGraphError("session_graph_agent_invalid_patch", "Agent used an unsupported node kind")
        if label:
            node["label"] = label
        if node_summary:
            node["summary"] = node_summary
        if kind:
            node["kind"] = kind
        attributes = dict(node.get("attributes") or {})
        thread_id = _text(update.get("thread_id"), 80)
        if thread_id:
            attributes["thread_id"] = thread_id
        tags = [_text(item, 40) for item in update.get("tags", []) if _text(item, 40)] if isinstance(update.get("tags"), list) else []
        if tags:
            attributes["topic_tags"] = list(dict.fromkeys(tags))[:8]
        node["attributes"] = attributes

    existing_edges = {
        (str(item.get("source")), str(item.get("target")), str(item.get("type")))
        for item in _items(result.get("edges"))
    }
    for edge in _items(patch.get("edge_additions")):
        source = _text(edge.get("source"), 512)
        target = _text(edge.get("target"), 512)
        relation = _text(edge.get("type"), 40).lower()
        evidence_ids = [
            _text(item, 512)
            for item in edge.get("evidence_ids", [])
            if _text(item, 512)
        ] if isinstance(edge.get("evidence_ids"), list) else []
        if source not in nodes or target not in nodes or source == target:
            raise SessionGraphError("session_graph_agent_invalid_patch", "Agent edge referenced an invalid node")
        if relation not in SESSION_EDGE_TYPES:
            raise SessionGraphError("session_graph_agent_invalid_patch", "Agent used an unsupported edge type")
        if not evidence_ids or any(item not in nodes for item in evidence_ids):
            raise SessionGraphError("session_graph_agent_invalid_patch", "Agent edge lacks valid evidence IDs")
        key = (source, target, relation)
        if key in existing_edges:
            continue
        existing_edges.add(key)
        result.setdefault("edges", []).append(
            {
                "id": _stable_id("session-edge", "|".join(key)),
                "source": source,
                "target": target,
                "type": relation,
                "weight": max(0.0, min(1.0, _number(edge.get("weight"), 0.6))),
                "origin": "derived",
                "provenance": {
                    "source": "session_map_agent",
                    "prompt_version": SESSION_GRAPH_PROMPT_VERSION,
                    "evidence_ids": list(dict.fromkeys(evidence_ids)),
                },
            }
        )

    threads: list[dict[str, Any]] = []
    seen_thread_ids: set[str] = set()
    for thread in _items(patch.get("threads")):
        identifier = _text(thread.get("id"), 80)
        node_ids = [
            _text(item, 512)
            for item in thread.get("node_ids", [])
            if _text(item, 512)
        ] if isinstance(thread.get("node_ids"), list) else []
        if not identifier or identifier in seen_thread_ids or not node_ids:
            continue
        if any(item not in nodes for item in node_ids):
            raise SessionGraphError("session_graph_agent_invalid_patch", "Agent thread referenced an unknown node")
        seen_thread_ids.add(identifier)
        thread_nodes = [nodes[item] for item in dict.fromkeys(node_ids)]
        kinds = Counter(_text(item.get("kind")) or "fact" for item in thread_nodes)
        times = [_text(item.get("occurred_at")) for item in thread_nodes if _text(item.get("occurred_at"))]
        threads.append(
            {
                "id": identifier,
                "title": _text(thread.get("title"), 120) or identifier,
                "summary": _text(thread.get("summary"), 600),
                "node_ids": list(dict.fromkeys(node_ids)),
                "kind": kinds.most_common(1)[0][0] if kinds else "fact",
                "status": "active",
                "memory_count": len(thread_nodes),
                "evidence_count": sum(int(item.get("evidence_count") or 0) for item in thread_nodes),
                "started_at": min(times) if times else None,
                "updated_at": max(times) if times else None,
            }
        )
    if threads:
        result["threads"] = threads
    result["projection_state"] = "ready"
    result["generated_by"] = "local-session-map-agent"
    result["prompt_version"] = SESSION_GRAPH_PROMPT_VERSION
    result["counts"] = {
        **dict(result.get("counts") or {}),
        "nodes": len(result.get("nodes") or []),
        "edges": len(result.get("edges") or []),
        "threads": len(result.get("threads") or []),
    }
    return result


def build_session_atlas(
    scope_name: str,
    sessions: Sequence[Mapping[str, Any]],
    session_views: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    session_ids = {_text(item.get("session_id"), 200) for item in sessions}
    session_ids.discard("")
    nodes: list[dict[str, Any]] = []
    for session in sessions:
        session_id = _text(session.get("session_id"), 200)
        if not session_id:
            continue
        view = session_views.get(session_id) or {}
        title = _text(view.get("title"), 160) or _text(session.get("title"), 160) or f"Session {session_id[:12]}"
        summary = _text(view.get("summary"), 800) or "Conversation memory is waiting for semantic projection."
        thread_titles = [
            _text(item.get("title"), 80)
            for item in _items(view.get("threads"))[:5]
            if _text(item.get("title"), 80)
        ]
        nodes.append(
            {
                "id": "session:" + session_id,
                "session_id": session_id,
                "kind": "session",
                "title": title,
                "summary": summary,
                "status": _text(session.get("status"), 32) or "active",
                "source_app": _text(session.get("source_app"), 80) or None,
                "native_thread_id": _text(session.get("native_thread_id"), 200) or None,
                "parent_session_id": _text(session.get("parent_session_id"), 200) or None,
                "created_at": session.get("created_at"),
                "updated_at": session.get("last_ingest_at"),
                "message_count": int(session.get("message_count") or 0),
                "ingest_request_count": int(session.get("ingest_request_count") or 0),
                "memory_node_count": len(view.get("nodes") or []),
                "thread_count": len(view.get("threads") or []),
                "thread_titles": thread_titles,
                "topic_tags": [],
                "projection_state": _text(view.get("projection_state")) or "pending",
            }
        )
    edges: list[dict[str, Any]] = []
    for session in sessions:
        session_id = _text(session.get("session_id"), 200)
        parent = _text(session.get("parent_session_id"), 200)
        if not session_id or not parent or parent not in session_ids or parent == session_id:
            continue
        edges.append(
            {
                "id": _stable_id("atlas-edge", f"{parent}|{session_id}|parent"),
                "source": "session:" + parent,
                "target": "session:" + session_id,
                "type": "parent",
                "weight": 1.0,
                "origin": "trusted_session_metadata",
                "reason": "Explicit parent Session metadata",
            }
        )
    snapshot_ids = sorted(
        {
            _text(view.get("snapshot_id"))
            for view in session_views.values()
            if _text(view.get("snapshot_id"))
        }
    )
    return {
        "schema_version": SESSION_ATLAS_SCHEMA_VERSION,
        "scope_name": scope_name,
        "snapshot_id": snapshot_ids[-1] if snapshot_ids else "catalog-only",
        "view": "session_atlas",
        "projection_state": "fallback",
        "generated_by": "deterministic-session-catalog",
        "prompt_version": None,
        "model": None,
        "session_count": len(nodes),
        "message_count": sum(int(item.get("message_count") or 0) for item in sessions),
        "nodes": nodes,
        "edges": edges,
        "counts": {"sessions": len(nodes), "edges": len(edges)},
    }


def apply_session_atlas_patch(
    base: Mapping[str, Any], patch: Mapping[str, Any]
) -> dict[str, Any]:
    result = json.loads(json.dumps(base, ensure_ascii=False))
    nodes = {
        str(item["session_id"]): item
        for item in result.get("nodes", [])
        if isinstance(item, dict) and item.get("session_id")
    }
    for update in _items(patch.get("node_updates")):
        session_id = _text(update.get("session_id"), 200)
        if session_id not in nodes:
            raise SessionGraphError("session_atlas_agent_invalid_patch", "Agent referenced an unknown Session")
        node = nodes[session_id]
        title = _text(update.get("title"), 160)
        summary = _text(update.get("summary"), 800)
        tags = [_text(item, 40) for item in update.get("topic_tags", []) if _text(item, 40)] if isinstance(update.get("topic_tags"), list) else []
        if title:
            node["title"] = title
        if summary:
            node["summary"] = summary
        if tags:
            node["topic_tags"] = list(dict.fromkeys(tags))[:8]
    existing = {
        (str(item.get("source")), str(item.get("target")), str(item.get("type")))
        for item in _items(result.get("edges"))
    }
    for edge in _items(patch.get("edge_additions")):
        source = _text(edge.get("source_session_id"), 200)
        target = _text(edge.get("target_session_id"), 200)
        relation = _text(edge.get("type"), 40).lower()
        reason = _text(edge.get("reason"), 240)
        if source not in nodes or target not in nodes or source == target:
            raise SessionGraphError("session_atlas_agent_invalid_patch", "Agent edge referenced an invalid Session")
        if relation not in ATLAS_EDGE_TYPES - {"parent", "forked_from"}:
            raise SessionGraphError("session_atlas_agent_invalid_patch", "Agent used an unsupported Atlas edge type")
        if not reason:
            raise SessionGraphError("session_atlas_agent_invalid_patch", "Agent edge lacks a grounded reason")
        key = ("session:" + source, "session:" + target, relation)
        if key in existing:
            continue
        existing.add(key)
        result.setdefault("edges", []).append(
            {
                "id": _stable_id("atlas-edge", "|".join(key)),
                "source": key[0],
                "target": key[1],
                "type": relation,
                "weight": max(0.0, min(1.0, _number(edge.get("weight"), 0.55))),
                "origin": "session_atlas_agent",
                "reason": reason,
            }
        )
    result["projection_state"] = "ready"
    result["generated_by"] = "local-session-atlas-agent"
    result["prompt_version"] = SESSION_GRAPH_PROMPT_VERSION
    result["session_count"] = len(result.get("nodes") or [])
    result["counts"] = {
        "sessions": len(result.get("nodes") or []),
        "edges": len(result.get("edges") or []),
    }
    return result


class LocalSessionGraphAgent:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        provider: str = SESSION_GRAPH_PROVIDER_LOCAL,
        timeout_seconds: float = 120.0,
        reserved_production_slots: int = 2,
        opener: Callable[..., Any] | None = None,
        gpu_scheduler: GpuWorkloadScheduler | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.provider = _text(provider, 64).lower()
        self.timeout_seconds = max(5.0, timeout_seconds)
        self.reserved_production_slots = max(0, int(reserved_production_slots))
        self.opener = opener or urllib.request.urlopen
        self.gpu_scheduler = gpu_scheduler
        approved_local = False
        if self.provider == SESSION_GRAPH_PROVIDER_LOCAL and self.model:
            try:
                validate_loopback_openai_compatible_url(
                    self.base_url, name="TMCRA_SESSION_GRAPH_BASE_URL"
                )
                approved_local = True
            except ValueError:
                approved_local = False
        approved_openai = False
        if self.provider == SESSION_GRAPH_PROVIDER_OPENAI:
            try:
                validate_openai_compatible_url(
                    self.base_url, name="TMCRA_SESSION_GRAPH_BASE_URL"
                )
                approved_openai = bool(self.model)
            except ValueError:
                approved_openai = False
        allowed_route = (
            self.provider == SESSION_GRAPH_PROVIDER_LOCAL and approved_local
        ) or (
            self.provider == SESSION_GRAPH_PROVIDER_DEDICATED
            and self.base_url == DEDICATED_DEEPSEEK_BASE_URL
            and bool(self.model)
        ) or approved_openai
        if not allowed_route:
            raise SessionGraphError(
                "session_graph_agent_route_invalid",
                "Session Graph Agent must use an approved isolated route",
            )

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> "LocalSessionGraphAgent | None":
        env = dict(os.environ if environment is None else environment)
        enabled = _text(env.get("TMCRA_SESSION_GRAPH_AGENT_ENABLED") or "1").lower()
        if enabled in {"0", "false", "no", "off"}:
            return None
        provider = _text(
            env.get("TMCRA_SESSION_GRAPH_PROVIDER")
            or SESSION_GRAPH_PROVIDER_LOCAL,
            64,
        ).lower()
        key = _text(env.get("TMCRA_SESSION_GRAPH_API_KEY"), 512)
        key_file_default = (
            env.get("TMCRA_LOCAL_WRITER_API_KEY_FILE")
            or "/opt/tmcra-data/local-llm/secrets/qwen36-server-lanes.key"
            if provider == SESSION_GRAPH_PROVIDER_LOCAL
            else ""
        )
        key_file = _text(
            env.get("TMCRA_SESSION_GRAPH_API_KEY_FILE") or key_file_default
        )
        if not key and key_file:
            path = Path(key_file)
            if path.is_file():
                key = next(
                    (
                        _text(line, 512)
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if _text(line, 512)
                    ),
                    "",
                )
        if not key:
            return None
        if provider == SESSION_GRAPH_PROVIDER_DEDICATED:
            foreground_pools = (
                "TMCRA_DEEPSEEK_WRITER_KEY_POOL",
                "TMCRA_WRITER_API_KEY_POOL",
                "TMCRA_WRITER_REVIEWER_API_KEY_POOL",
                "TMCRA_RECALL_PLANNER_API_KEY_POOL",
                "TMCRA_SLOW_GRAPH_API_KEY_POOL",
            )
            for name in foreground_pools:
                values = {
                    item.strip()
                    for item in str(env.get(name) or "").split(",")
                    if item.strip()
                }
                if key in values:
                    raise SessionGraphError(
                        "projection_provider_key_not_isolated",
                        "the projection provider credential overlaps a foreground pool",
                    )
        timeout = _number(env.get("TMCRA_SESSION_GRAPH_AGENT_TIMEOUT_SECONDS"), 120.0)
        if provider == SESSION_GRAPH_PROVIDER_DEDICATED:
            default_base_url = DEDICATED_DEEPSEEK_BASE_URL
            default_model = DEDICATED_DEEPSEEK_MODEL
        elif provider == SESSION_GRAPH_PROVIDER_OPENAI:
            default_base_url = ""
            default_model = ""
        else:
            default_base_url = _text(
                env.get("TMCRA_WRITER_BASE_URL")
                or env.get("TMCRA_LOCAL_WRITER_BASE_URL")
                or LOCAL_QWEN_BASE_URL
            )
            default_model = _text(
                env.get("TMCRA_WRITER_MODEL")
                or env.get("TMCRA_LOCAL_WRITER_MODEL")
                or LOCAL_QWEN_MODEL
            )
        return cls(
            base_url=_text(env.get("TMCRA_SESSION_GRAPH_BASE_URL") or default_base_url),
            model=_text(env.get("TMCRA_SESSION_GRAPH_MODEL") or default_model),
            api_key=key,
            provider=provider,
            timeout_seconds=timeout,
            reserved_production_slots=max(
                0,
                int(
                    _number(
                        env.get("TMCRA_PROJECTION_RESERVED_PRODUCTION_SLOTS"),
                        (
                            0
                            if provider == SESSION_GRAPH_PROVIDER_OPENAI
                            or _text(env.get("TMCRA_SESSION_GRAPH_BASE_URL"))
                            == DESKTOP_LOCAL_QWEN_BASE_URL
                            else _number(env.get("TMCRA_LOCAL_LLM_PARALLEL"), 2)
                        ),
                    )
                ),
            ),
        )

    def _call(
        self,
        system_prompt: str,
        payload: Mapping[str, Any],
        *,
        max_tokens: int,
        response_schema: Mapping[str, Any] | None = None,
        response_schema_name: str = "tmcra_projection",
        slot_id: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        use_local_schema = (
            self.provider == SESSION_GRAPH_PROVIDER_LOCAL
            and self.model in DESKTOP_LOCAL_QWEN_MODELS
            and response_schema is not None
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        (
                            ""
                            if self.provider == SESSION_GRAPH_PROVIDER_DEDICATED
                            or self.base_url == LOCAL_QWEN_BASE_URL
                            else "/no_think\n"
                        )
                        + json.dumps(
                            payload, ensure_ascii=False, separators=(",", ":")
                        )
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_schema_name,
                        "strict": True,
                        "schema": dict(response_schema),
                    },
                }
                if use_local_schema
                else {"type": "json_object"}
            ),
        }
        local_slot_id: int | None = None
        if (
            self.provider == SESSION_GRAPH_PROVIDER_LOCAL
            and self.base_url == LOCAL_QWEN_BASE_URL
        ):
            local_slot_id = (
                LOCAL_QWEN_GRAPH_SLOT_ID if slot_id is None else int(slot_id)
            )
            if local_slot_id not in {
                LOCAL_QWEN_GRAPH_SLOT_ID,
                LOCAL_QWEN_PLANNER_SLOT_ID,
            }:
                raise SessionGraphError(
                    "projection_slot_invalid",
                    "projection work may use only the graph slot or the borrowed planner slot",
                )
            body["id_slot"] = 0 if os.getenv("TMCRA_DEPLOYMENT_MODE") == "local" else local_slot_id
        if (
            self.provider == SESSION_GRAPH_PROVIDER_DEDICATED
            or self.base_url == LOCAL_QWEN_BASE_URL
        ):
            body.update(
                {
                    "thinking": {"type": "disabled"},
                    "enable_thinking": False,
                }
            )
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=encoded,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        started = time.time()
        gpu_workload = (
            GpuWorkload.GRAPH_BORROWED_PLANNER
            if local_slot_id == LOCAL_QWEN_PLANNER_SLOT_ID
            else GpuWorkload.GRAPH_BACKGROUND
        )
        gpu_lease = (
            self.gpu_scheduler.lease(gpu_workload)
            if self.gpu_scheduler is not None
            and self.provider == SESSION_GRAPH_PROVIDER_LOCAL
            and self.base_url == LOCAL_QWEN_BASE_URL
            else nullcontext()
        )
        try:
            with gpu_lease:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    status = int(response.getcode())
                    raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise SessionGraphError(
                "session_graph_agent_http_error",
                f"Session Graph Agent HTTP {exc.code}: {detail}",
            ) from exc
        except Exception as exc:
            raise SessionGraphError(
                "session_graph_agent_unavailable",
                f"Session Graph Agent request failed: {type(exc).__name__}: {exc}",
            ) from exc
        response_payload = _json_object(raw)
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
            raise SessionGraphError("session_graph_agent_invalid_response", "Agent response must contain exactly one choice")
        choice = choices[0]
        message = choice.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        finish_reason = _text(choice.get("finish_reason"))
        if finish_reason != "stop" or not isinstance(content, str):
            reason = finish_reason or "missing"
            raise SessionGraphError(
                "session_graph_agent_invalid_response",
                f"Agent did not finish with a JSON object (finish_reason={reason})",
            )
        response_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        metadata = {
            "physical_call_id": _text(response_payload.get("id"), 256)
            or f"tmcra-projection-{response_sha256[:32]}",
            "provider": (
                DEEPSEEK_PROVIDER
                if self.provider == SESSION_GRAPH_PROVIDER_DEDICATED
                else OPENAI_COMPATIBLE_PROVIDER
                if self.provider == SESSION_GRAPH_PROVIDER_OPENAI
                else LOCAL_QWEN_PROVIDER
            ),
            "model": _text(response_payload.get("model"), 160) or self.model,
            "status": "completed",
            "http_status": status,
            "latency_seconds": round(time.time() - started, 3),
            "usage": response_payload.get("usage") if isinstance(response_payload.get("usage"), Mapping) else {},
            "request_sha256": hashlib.sha256(encoded).hexdigest(),
            "response_sha256": response_sha256,
            "started_at": started,
        }
        if use_local_schema:
            metadata["response_schema_sha256"] = hashlib.sha256(
                json.dumps(
                    dict(response_schema),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        return _json_object(content), metadata

    def capacity_available(self) -> bool:
        # The dedicated provider uses a key removed from every foreground pool;
        # it therefore has no local GPU lane to steal from Writer, planner, or
        # Slow graph.  Foreground queue guards still pause its next batch.
        if self.provider in {
            SESSION_GRAPH_PROVIDER_DEDICATED,
            SESSION_GRAPH_PROVIDER_OPENAI,
        }:
            return True
        server_root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        request = urllib.request.Request(
            f"{server_root}/slots",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with self.opener(request, timeout=5.0) as response:
                slots = json.loads(response.read().decode("utf-8"))
        except Exception:
            return False
        if not isinstance(slots, list):
            return False
        if (
            self.provider == SESSION_GRAPH_PROVIDER_LOCAL
            and self.base_url == LOCAL_QWEN_BASE_URL
        ):
            for index, slot in enumerate(slots):
                if not isinstance(slot, Mapping):
                    continue
                try:
                    slot_id = int(slot.get("id", index))
                except (TypeError, ValueError):
                    continue
                if slot_id == LOCAL_QWEN_GRAPH_SLOT_ID:
                    return not bool(slot.get("is_processing"))
            return False
        idle = sum(
            1
            for slot in slots
            if isinstance(slot, Mapping) and not bool(slot.get("is_processing"))
        )
        # The projection agent may consume at most one shared-model slot.  A
        # fixed reserve remains immediately available to Writer/Slow work.
        return idle >= self.reserved_production_slots + 1

    def borrowed_planner_slot_available(self) -> bool:
        """Return whether slot 1 can serve one bounded projection batch.

        The scheduler supplies the cross-role lock and quiet-period policy. The
        llama.cpp slot probe is the final race-resistant admission check.
        Writer slot 0 is intentionally never considered here.
        """

        if not (
            self.provider == SESSION_GRAPH_PROVIDER_LOCAL
            and self.base_url == LOCAL_QWEN_BASE_URL
            and self.gpu_scheduler is not None
            and self.gpu_scheduler.can_start(
                GpuWorkload.GRAPH_BORROWED_PLANNER
            )
        ):
            return False
        server_root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        request = urllib.request.Request(
            f"{server_root}/slots",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with self.opener(request, timeout=5.0) as response:
                slots = json.loads(response.read().decode("utf-8"))
        except Exception:
            return False
        if not isinstance(slots, list):
            return False
        for index, slot in enumerate(slots):
            if not isinstance(slot, Mapping):
                continue
            try:
                candidate = int(slot.get("id", index))
            except (TypeError, ValueError):
                continue
            if candidate == LOCAL_QWEN_PLANNER_SLOT_ID:
                return not bool(slot.get("is_processing"))
        return False

    @property
    def resource_isolation(self) -> str:
        return (
            "dedicated-provider"
            if self.provider == SESSION_GRAPH_PROVIDER_DEDICATED
            else "user-provider"
            if self.provider == SESSION_GRAPH_PROVIDER_OPENAI
            else "dedicated-local-slot"
            if self.base_url == LOCAL_QWEN_BASE_URL
            else "shared-local-reserve"
        )


    @staticmethod
    def _identifier_aliases(
        identifiers: Sequence[str], prefix: str
    ) -> tuple[dict[str, str], dict[str, str]]:
        values = sorted({_text(item, 512) for item in identifiers if _text(item, 512)})
        width = max(2, len(str(len(values))))
        real_to_alias = {
            identifier: f"{prefix}{index:0{width}d}"
            for index, identifier in enumerate(values, start=1)
        }
        return real_to_alias, {alias: real for real, alias in real_to_alias.items()}

    @staticmethod
    def _alias_call_metadata(
        call: Mapping[str, Any], real_to_alias: Mapping[str, str]
    ) -> dict[str, Any]:
        encoded = json.dumps(
            sorted(real_to_alias.items()),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            **dict(call),
            "identifier_alias_scheme": SESSION_GRAPH_ALIAS_SCHEME,
            "identifier_alias_count": len(real_to_alias),
            "identifier_alias_binding_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    @staticmethod
    def _translate_session_map_patch(
        patch: Mapping[str, Any], identifiers: Mapping[str, str]
    ) -> dict[str, Any]:
        def translated(value: Any) -> str:
            identifier = _text(value, 512)
            return identifiers.get(identifier, identifier)

        result = dict(patch)
        result["node_updates"] = [
            {**dict(item), "id": translated(item.get("id"))}
            for item in _items(patch.get("node_updates"))
        ]
        result["edge_additions"] = [
            {
                **dict(item),
                "source": translated(item.get("source")),
                "target": translated(item.get("target")),
                "evidence_ids": [translated(value) for value in item.get("evidence_ids", [])]
                if isinstance(item.get("evidence_ids"), list)
                else item.get("evidence_ids"),
            }
            for item in _items(patch.get("edge_additions"))
        ]
        result["threads"] = [
            {
                **dict(item),
                "node_ids": [translated(value) for value in item.get("node_ids", [])]
                if isinstance(item.get("node_ids"), list)
                else item.get("node_ids"),
            }
            for item in _items(patch.get("threads"))
        ]
        return result

    @staticmethod
    def _session_map_response_schema(payload: Mapping[str, Any]) -> dict[str, Any]:
        node_ids = sorted(
            {
                _text(item.get("id"), 512)
                for item in _items(payload.get("nodes"))
                if _text(item.get("id"), 512)
            }
        )
        node_id: dict[str, Any] = {"type": "string"}
        if node_ids:
            node_id["enum"] = node_ids
        node_update = {
            "type": "object",
            "properties": {
                "id": node_id,
                "label": {"type": "string"},
                "summary": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": sorted(SESSION_NODE_KINDS),
                },
                "thread_id": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "id",
                "label",
                "summary",
                "kind",
                "thread_id",
                "tags",
            ],
            "additionalProperties": False,
        }
        edge_addition = {
            "type": "object",
            "properties": {
                "source": node_id,
                "target": node_id,
                "type": {
                    "type": "string",
                    "enum": sorted(SESSION_EDGE_TYPES),
                },
                "weight": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence_ids": {
                    "type": "array",
                    "items": node_id,
                    "minItems": 1,
                },
            },
            "required": [
                "source",
                "target",
                "type",
                "weight",
                "evidence_ids",
            ],
            "additionalProperties": False,
        }
        thread = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "node_ids": {
                    "type": "array",
                    "items": node_id,
                    "minItems": 1,
                },
            },
            "required": ["id", "title", "summary", "node_ids"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "node_updates": {
                    "type": "array",
                    "items": node_update,
                    "maxItems": len(node_ids),
                },
                "edge_additions": {
                    "type": "array",
                    "items": edge_addition,
                    "maxItems": 24,
                },
                "threads": {"type": "array", "items": thread},
            },
            "required": [
                "title",
                "summary",
                "node_updates",
                "edge_additions",
                "threads",
            ],
            "additionalProperties": False,
        }

    @classmethod
    def _session_map_payload(
        cls, graph: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
        nodes = _items(graph.get("nodes"))[:80]
        real_to_alias, alias_to_real = cls._identifier_aliases(
            [_text(item.get("id"), 512) for item in nodes], "n"
        )
        payload = {
            "session": {
                key: graph.get(key)
                for key in ("title", "status", "message_count", "source_app")
            },
            "allowed_node_kinds": sorted(SESSION_NODE_KINDS),
            "allowed_edge_types": sorted(SESSION_EDGE_TYPES),
            "nodes": [
                {
                    "id": real_to_alias[_text(item.get("id"), 512)],
                    "kind": item.get("kind"),
                    "label": item.get("label"),
                    "summary": item.get("summary"),
                    "occurred_at": item.get("occurred_at"),
                    "actor_role": item.get("actor_role"),
                    "authority": item.get("authority"),
                    "source_record_count": len(
                        dict(item.get("attributes") or {}).get("source_record_ids", [])
                    ),
                }
                for item in nodes
                if _text(item.get("id"), 512) in real_to_alias
            ],
            "existing_edges": [
                {
                    "source": real_to_alias[_text(item.get("source"), 512)],
                    "target": real_to_alias[_text(item.get("target"), 512)],
                    "type": item.get("type"),
                }
                for item in _items(graph.get("edges"))[:160]
                if _text(item.get("source"), 512) in real_to_alias
                and _text(item.get("target"), 512) in real_to_alias
            ],
        }
        return payload, real_to_alias, alias_to_real

    def session_map(self, graph: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        payload, real_to_alias, alias_to_real = self._session_map_payload(graph)
        patch, call = self._call(
            SESSION_MAP_SYSTEM_PROMPT,
            payload,
            max_tokens=SESSION_GRAPH_MAX_OUTPUT_TOKENS,
            response_schema=self._session_map_response_schema(payload),
            response_schema_name="tmcra_session_map",
        )
        return (
            self._translate_session_map_patch(patch, alias_to_real),
            self._alias_call_metadata(call, real_to_alias),
        )

    def repair_session_map(
        self,
        graph: Mapping[str, Any],
        invalid_patch: Mapping[str, Any],
        *,
        validation_error: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload, real_to_alias, alias_to_real = self._session_map_payload(graph)
        payload["invalid_patch"] = self._translate_session_map_patch(
            invalid_patch, real_to_alias
        )
        payload["validation_error"] = dict(validation_error)
        patch, call = self._call(
            SESSION_MAP_REPAIR_SYSTEM_PROMPT,
            payload,
            max_tokens=SESSION_GRAPH_MAX_OUTPUT_TOKENS,
            response_schema=self._session_map_response_schema(payload),
            response_schema_name="tmcra_session_map_repair",
        )
        return (
            self._translate_session_map_patch(patch, alias_to_real),
            self._alias_call_metadata(call, real_to_alias),
        )

    @staticmethod
    def _translate_atlas_patch(
        patch: Mapping[str, Any], identifiers: Mapping[str, str]
    ) -> dict[str, Any]:
        def translated(value: Any) -> str:
            identifier = _text(value, 512)
            return identifiers.get(identifier, identifier)

        result = dict(patch)
        result["node_updates"] = [
            {**dict(item), "session_id": translated(item.get("session_id"))}
            for item in _items(patch.get("node_updates"))
        ]
        result["edge_additions"] = [
            {
                **dict(item),
                "source_session_id": translated(item.get("source_session_id")),
                "target_session_id": translated(item.get("target_session_id")),
            }
            for item in _items(patch.get("edge_additions"))
        ]
        return result

    @classmethod
    def _atlas_payload(
        cls, graph: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
        sessions = _items(graph.get("nodes"))[:160]
        real_to_alias, alias_to_real = cls._identifier_aliases(
            [_text(item.get("session_id"), 512) for item in sessions], "s"
        )
        payload = {
            "session_count": graph.get("session_count"),
            "allowed_edge_types": sorted(ATLAS_EDGE_TYPES - {"parent", "forked_from"}),
            "output_limits": {
                "node_updates": SESSION_ATLAS_MAX_NODE_UPDATES,
                "edge_additions": SESSION_ATLAS_MAX_EDGE_ADDITIONS,
            },
            "sessions": [
                {
                    "session_id": real_to_alias[_text(item.get("session_id"), 512)],
                    "title": item.get("title"),
                    "summary": item.get("summary"),
                    "status": item.get("status"),
                    "source_app": item.get("source_app"),
                    "parent_session_id": real_to_alias.get(
                        _text(item.get("parent_session_id"), 512)
                    ),
                    "message_count": item.get("message_count"),
                    "thread_titles": item.get("thread_titles"),
                }
                for item in sessions
                if _text(item.get("session_id"), 512) in real_to_alias
            ],
        }
        return payload, real_to_alias, alias_to_real

    def atlas(self, graph: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        payload, real_to_alias, alias_to_real = self._atlas_payload(graph)
        patch, call = self._call(
            SESSION_ATLAS_SYSTEM_PROMPT,
            payload,
            max_tokens=SESSION_ATLAS_MAX_OUTPUT_TOKENS,
        )
        return (
            self._translate_atlas_patch(patch, alias_to_real),
            self._alias_call_metadata(call, real_to_alias),
        )

    def repair_atlas(
        self,
        graph: Mapping[str, Any],
        invalid_patch: Mapping[str, Any],
        *,
        validation_error: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload, real_to_alias, alias_to_real = self._atlas_payload(graph)
        payload["invalid_patch"] = self._translate_atlas_patch(
            invalid_patch, real_to_alias
        )
        payload["validation_error"] = dict(validation_error)
        patch, call = self._call(
            SESSION_ATLAS_REPAIR_SYSTEM_PROMPT,
            payload,
            max_tokens=SESSION_ATLAS_MAX_OUTPUT_TOKENS,
        )
        return (
            self._translate_atlas_patch(patch, alias_to_real),
            self._alias_call_metadata(call, real_to_alias),
        )

    @staticmethod
    def _translate_visual_taxonomy(
        taxonomy: Mapping[str, Any], identifiers: Mapping[str, str]
    ) -> dict[str, Any]:
        # Provider repair responses sometimes echo the supplied catalog next to
        # the corrected object.  Only the two contracted output fields cross
        # the validation boundary.
        result = {
            "domains": [dict(item) for item in _items(taxonomy.get("domains"))]
        }
        result["session_assignments"] = [
            {
                **dict(item),
                "session_id": identifiers.get(
                    _text(item.get("session_id"), 512),
                    _text(item.get("session_id"), 512),
                ),
            }
            for item in _items(taxonomy.get("session_assignments"))
        ]
        return result

    @classmethod
    def _visual_taxonomy_payload(
        cls,
        sessions: Sequence[Mapping[str, Any]],
        session_views: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
        payload = build_visual_atlas_taxonomy_payload(sessions, session_views)
        catalog = _items(payload.get("sessions"))
        real_to_alias, alias_to_real = cls._identifier_aliases(
            [_text(item.get("session_id"), 512) for item in catalog], "s"
        )
        payload["sessions"] = [
            {
                **dict(item),
                "session_id": real_to_alias[_text(item.get("session_id"), 512)],
                "parent_session_id": real_to_alias.get(
                    _text(item.get("parent_session_id"), 512)
                ),
            }
            for item in catalog
        ]
        return payload, real_to_alias, alias_to_real

    def visual_taxonomy(
        self,
        sessions: Sequence[Mapping[str, Any]],
        session_views: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload, real_to_alias, alias_to_real = self._visual_taxonomy_payload(
            sessions, session_views
        )
        taxonomy, call = self._call(
            VISUAL_ATLAS_TAXONOMY_SYSTEM_PROMPT,
            payload,
            max_tokens=VISUAL_ATLAS_TAXONOMY_MAX_OUTPUT_TOKENS,
        )
        return (
            self._translate_visual_taxonomy(taxonomy, alias_to_real),
            self._alias_call_metadata(call, real_to_alias),
        )

    def repair_visual_taxonomy(
        self,
        sessions: Sequence[Mapping[str, Any]],
        session_views: Mapping[str, Mapping[str, Any]],
        invalid_taxonomy: Mapping[str, Any],
        *,
        validation_error: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload, real_to_alias, alias_to_real = self._visual_taxonomy_payload(
            sessions, session_views
        )
        payload["invalid_taxonomy"] = self._translate_visual_taxonomy(
            invalid_taxonomy, real_to_alias
        )
        payload["validation_error"] = dict(validation_error)
        taxonomy, call = self._call(
            VISUAL_ATLAS_TAXONOMY_REPAIR_SYSTEM_PROMPT,
            payload,
            max_tokens=VISUAL_ATLAS_TAXONOMY_MAX_OUTPUT_TOKENS,
        )
        return (
            self._translate_visual_taxonomy(taxonomy, alias_to_real),
            self._alias_call_metadata(call, real_to_alias),
        )

    @staticmethod
    def _translate_visual_patch(
        patch: Mapping[str, Any], identifiers: Mapping[str, str]
    ) -> dict[str, Any]:
        def translated(value: Any) -> str:
            identifier = _text(value, 512)
            return identifiers.get(identifier, identifier)

        result = dict(patch)
        result["domain_updates"] = [
            {**dict(item), "domain_id": translated(item.get("domain_id"))}
            for item in _items(patch.get("domain_updates"))
        ]
        result["episode_updates"] = [
            {**dict(item), "episode_id": translated(item.get("episode_id"))}
            for item in _items(patch.get("episode_updates"))
        ]
        result["memory_updates"] = [
            {**dict(item), "evidence_id": translated(item.get("evidence_id"))}
            for item in _items(patch.get("memory_updates"))
        ]
        result["relations"] = [
            {
                **dict(item),
                "source_id": translated(item.get("source_id")),
                "target_id": translated(item.get("target_id")),
                "evidence_ids": [
                    translated(value)
                    for value in item.get("evidence_ids", [])
                    if _text(value, 512)
                ]
                if isinstance(item.get("evidence_ids"), list)
                else [],
            }
            for item in _items(patch.get("relations"))
        ]
        return result

    @classmethod
    def _visual_episode_batch_payload(
        cls, batch: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
        payload = json.loads(json.dumps(batch, ensure_ascii=False))
        identifiers: list[str] = []

        def collect(value: Any) -> None:
            identifier = _text(value, 512)
            if identifier:
                identifiers.append(identifier)

        for value in payload.get("expected_episode_ids", []):
            collect(value)
        for value in payload.get("expected_memory_evidence_ids", []):
            collect(value)
        for value in payload.get("relation_candidate_memory_ids", []):
            collect(value)
        domain = payload.get("domain")
        if isinstance(domain, Mapping):
            collect(domain.get("domain_id"))
        session = payload.get("session")
        if isinstance(session, Mapping):
            for key in ("session_id", "domain_id", "parent_session_id"):
                collect(session.get(key))
        for session_item in _items(payload.get("sessions")):
            for key in ("session_id", "domain_id", "parent_session_id"):
                collect(session_item.get(key))
        for episode in _items(payload.get("episodes")):
            for key in ("episode_id", "session_id", "domain_id"):
                collect(episode.get(key))
            for value in episode.get("evidence_ids", []):
                collect(value)
        for evidence in _items(payload.get("evidence")):
            collect(evidence.get("id"))
            for value in evidence.get("episode_ids", []):
                collect(value)
        for relation in _items(payload.get("existing_relations")):
            collect(relation.get("source_id"))
            collect(relation.get("target_id"))
            for value in relation.get("evidence_ids", []):
                collect(value)

        real_to_alias, alias_to_real = cls._identifier_aliases(identifiers, "v")

        def alias(value: Any) -> str:
            identifier = _text(value, 512)
            return real_to_alias.get(identifier, identifier)

        payload["expected_episode_ids"] = [
            alias(value) for value in payload.get("expected_episode_ids", [])
        ]
        payload["expected_memory_evidence_ids"] = [
            alias(value)
            for value in payload.get("expected_memory_evidence_ids", [])
        ]
        payload["relation_candidate_memory_ids"] = [
            alias(value)
            for value in payload.get("relation_candidate_memory_ids", [])
        ]
        if isinstance(domain, dict):
            domain["domain_id"] = alias(domain.get("domain_id"))
        if isinstance(session, dict):
            for key in ("session_id", "domain_id", "parent_session_id"):
                if session.get(key):
                    session[key] = alias(session.get(key))
        for session_item in _items(payload.get("sessions")):
            for key in ("session_id", "domain_id", "parent_session_id"):
                if session_item.get(key):
                    session_item[key] = alias(session_item.get(key))
        for episode in _items(payload.get("episodes")):
            for key in ("episode_id", "session_id", "domain_id"):
                episode[key] = alias(episode.get(key))
            episode["evidence_ids"] = [
                alias(value) for value in episode.get("evidence_ids", [])
            ]
        for evidence in _items(payload.get("evidence")):
            evidence["id"] = alias(evidence.get("id"))
            evidence["episode_ids"] = [
                alias(value) for value in evidence.get("episode_ids", [])
            ]
        for relation in _items(payload.get("existing_relations")):
            relation["source_id"] = alias(relation.get("source_id"))
            relation["target_id"] = alias(relation.get("target_id"))
            relation["evidence_ids"] = [
                alias(value) for value in relation.get("evidence_ids", [])
            ]
        return payload, real_to_alias, alias_to_real

    @staticmethod
    def _visual_episode_response_schema(
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        def text(maximum: int = 240) -> dict[str, Any]:
            return {"type": "string", "minLength": 1, "maxLength": maximum}

        def bilingual(fields: Sequence[str]) -> dict[str, Any]:
            localized = {
                "type": "object",
                "properties": {
                    field: text(80 if field == "label" else 140)
                    for field in fields
                },
                "required": list(fields),
                "additionalProperties": False,
            }
            return {
                "type": "object",
                "properties": {"zh": localized, "en": localized},
                "required": ["zh", "en"],
                "additionalProperties": False,
            }

        episode_ids = [
            _text(value, 512)
            for value in payload.get("expected_episode_ids", [])
            if _text(value, 512)
        ]
        memory_ids = [
            _text(value, 512)
            for value in payload.get("expected_memory_evidence_ids", [])
            if _text(value, 512)
        ]
        candidate_ids = sorted(
            {
                _text(value, 512)
                for value in payload.get("relation_candidate_memory_ids", [])
                if _text(value, 512)
            }
        )
        memory_types = sorted(
            {
                _text(value, 40)
                for value in payload.get("allowed_memory_types", [])
                if _text(value, 40)
            }
        )
        relation_types = sorted(
            {
                _text(value, 40)
                for value in payload.get("allowed_relation_types", [])
                if _text(value, 40)
            }
        )
        episode_updates = [
            {
                "type": "object",
                "properties": {
                    "episode_id": {"type": "string", "enum": [identifier]},
                    "label": text(80),
                    "summary": text(140),
                    "chapter_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                    },
                    "display": bilingual(("label", "summary")),
                },
                "required": [
                    "episode_id",
                    "label",
                    "summary",
                    "chapter_tags",
                    "display",
                ],
                "additionalProperties": False,
            }
            for identifier in episode_ids
        ]
        memory_updates = [
            {
                "type": "object",
                "properties": {
                    "evidence_id": {"type": "string", "enum": [identifier]},
                    "label": text(80),
                    "summary": text(140),
                    "memory_type": {"type": "string", "enum": memory_types},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 40},
                        "maxItems": 4,
                    },
                    "display": bilingual(("label", "summary")),
                },
                "required": [
                    "evidence_id",
                    "label",
                    "summary",
                    "memory_type",
                    "keywords",
                    "display",
                ],
                "additionalProperties": False,
            }
            for identifier in memory_ids
        ]
        relation = {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "enum": candidate_ids},
                "target_id": {"type": "string", "enum": candidate_ids},
                "type": {"type": "string", "enum": relation_types},
                "weight": {"type": "number", "minimum": 0, "maximum": 1},
                "label": text(100),
                "reason": text(160),
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": candidate_ids},
                    "minItems": 2,
                    "maxItems": len(candidate_ids),
                },
                "display": bilingual(("label", "reason")),
            },
            "required": [
                "source_id",
                "target_id",
                "type",
                "weight",
                "label",
                "reason",
                "evidence_ids",
                "display",
            ],
            "additionalProperties": False,
        }
        maximum_relations = max(0, int(_number(payload.get("max_relations"), 0)))
        relation_array: dict[str, Any] = {
            "type": "array",
            "maxItems": maximum_relations,
        }
        if maximum_relations and len(candidate_ids) >= 2 and relation_types:
            relation_array["items"] = relation
        else:
            # An empty candidate catalogue cannot produce a valid relation.  Do
            # not emit an `items` schema containing empty enums because local
            # grammar compilers reject it before the model is called.
            relation_array["maxItems"] = 0
        return {
            "type": "object",
            "properties": {
                "domain_updates": {"type": "array", "maxItems": 0},
                "episode_updates": {
                    "type": "array",
                    "prefixItems": episode_updates,
                    "minItems": len(episode_updates),
                    "maxItems": len(episode_updates),
                },
                "memory_updates": {
                    "type": "array",
                    "prefixItems": memory_updates,
                    "minItems": len(memory_updates),
                    "maxItems": len(memory_updates),
                },
                "relations": relation_array,
            },
            "required": [
                "domain_updates",
                "episode_updates",
                "memory_updates",
                "relations",
            ],
            "additionalProperties": False,
        }

    def visual_atlas_batch(
        self,
        graph: Mapping[str, Any],
        batch: Mapping[str, Any],
        *,
        batch_index: int = 0,
        slot_id: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate one bounded batch so the service can checkpoint it."""

        payload, real_to_alias, alias_to_real = self._visual_episode_batch_payload(
            batch
        )
        recoverable_response_codes = {
            "session_graph_agent_invalid_json",
            "session_graph_agent_invalid_response",
        }
        initial_response_error: dict[str, Any] | None = None
        try:
            patch, call = self._call(
                VISUAL_ATLAS_EPISODE_BATCH_SYSTEM_PROMPT,
                payload,
                max_tokens=VISUAL_ATLAS_EPISODE_BATCH_MAX_OUTPUT_TOKENS,
                response_schema=self._visual_episode_response_schema(payload),
                response_schema_name="tmcra_visual_atlas_episode_batch",
                slot_id=slot_id,
            )
        except SessionGraphError as exc:
            if exc.code not in recoverable_response_codes:
                raise
            patch = {}
            initial_response_error = {"code": exc.code, "message": str(exc)}
            call = {"failed": True, "error": initial_response_error}
        translated = self._translate_visual_patch(patch, alias_to_real)
        validation_error = initial_response_error
        if validation_error is None:
            try:
                normalized, relation_rejections = (
                    validate_visual_atlas_episode_batch_patch_with_relation_rejections(
                        graph, batch, translated
                    )
                )
            except VisualAtlasError as exc:
                validation_error = {"code": exc.code, "message": str(exc)}
        if validation_error is not None:
            payload["invalid_patch"] = self._translate_visual_patch(
                translated, real_to_alias
            )
            payload["validation_error"] = validation_error
            try:
                repaired, repair_call = self._call(
                    VISUAL_ATLAS_EPISODE_BATCH_REPAIR_SYSTEM_PROMPT,
                    payload,
                    max_tokens=VISUAL_ATLAS_EPISODE_BATCH_MAX_OUTPUT_TOKENS,
                    response_schema=self._visual_episode_response_schema(payload),
                    response_schema_name="tmcra_visual_atlas_episode_batch_repair",
                    slot_id=slot_id,
                )
            except SessionGraphError as repair_exc:
                if repair_exc.code not in recoverable_response_codes:
                    raise
                normalized = sanitize_visual_atlas_episode_batch_patch(
                    graph, batch, translated
                )
                relation_rejections = [
                    {
                        "index": -1,
                        "code": repair_exc.code,
                        "message": str(repair_exc),
                    }
                ]
                repair_validation_error = {
                    "code": repair_exc.code,
                    "message": str(repair_exc),
                }
                sanitizer_applied = True
                repair_call = {
                    "failed": True,
                    "error": repair_validation_error,
                }
            else:
                translated = self._translate_visual_patch(repaired, alias_to_real)
                try:
                    normalized, relation_rejections = (
                        validate_visual_atlas_episode_batch_patch_with_relation_rejections(
                            graph, batch, translated
                        )
                    )
                    repair_validation_error = None
                    sanitizer_applied = False
                except VisualAtlasError as repair_exc:
                    normalized = sanitize_visual_atlas_episode_batch_patch(
                        graph, batch, translated
                    )
                    relation_rejections = [
                        {
                            "index": -1,
                            "code": repair_exc.code,
                            "message": str(repair_exc),
                        }
                    ]
                    repair_validation_error = {
                        "code": repair_exc.code,
                        "message": str(repair_exc),
                    }
                    sanitizer_applied = True
            call = {
                "repair_attempted": True,
                "sanitizer_applied": sanitizer_applied,
                "validation_error": validation_error,
                "repair_validation_error": repair_validation_error,
                "initial": call,
                "repair": repair_call,
            }
        return normalized, {
            "batch_index": batch_index,
            "batch_id": batch.get("batch_id"),
            "episode_count": len(batch.get("expected_episode_ids", [])),
            "evidence_count": len(batch.get("evidence", [])),
            "call": self._alias_call_metadata(call, real_to_alias),
            "rejected_relation_count": len(relation_rejections),
            "relation_rejections": relation_rejections,
        }

    def visual_atlas_batches(
        self, graph: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        batches = build_visual_atlas_episode_batches(graph)
        patches: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(batches):
            normalized, call = self.visual_atlas_batch(
                graph, batch, batch_index=batch_index
            )
            patches.append(normalized)
            calls.append(call)
        merged = merge_visual_atlas_episode_batch_patches(graph, batches, patches)
        return merged, {
            "strategy": "domain-local-human-memory-batches",
            "batch_count": len(batches),
            "calls": calls,
        }

    @staticmethod
    def _translate_personal_knowledge_result(
        result: Mapping[str, Any], identifiers: Mapping[str, str]
    ) -> dict[str, Any]:
        def translated(value: Any) -> str:
            identifier = _text(value, 512)
            return identifiers.get(identifier, identifier)

        translated_result = dict(result)
        translated_result["domain_id"] = translated(result.get("domain_id"))
        translated_pages: list[dict[str, Any]] = []
        for page in _items(result.get("pages")):
            value = dict(page)
            value["claims"] = [
                {
                    **dict(claim),
                    "evidence_ids": [
                        translated(item)
                        for item in claim.get("evidence_ids", [])
                        if _text(item, 512)
                    ]
                    if isinstance(claim.get("evidence_ids"), list)
                    else [],
                }
                for claim in _items(page.get("claims"))
            ]
            value["sections"] = [
                {
                    **dict(section),
                    "evidence_ids": [
                        translated(item)
                        for item in section.get("evidence_ids", [])
                        if _text(item, 512)
                    ]
                    if isinstance(section.get("evidence_ids"), list)
                    else [],
                }
                for section in _items(page.get("sections"))
            ]
            translated_pages.append(value)
        translated_result["pages"] = translated_pages
        translated_result["excluded_evidence_ids"] = [
            translated(item)
            for item in result.get("excluded_evidence_ids", [])
            if _text(item, 512)
        ] if isinstance(result.get("excluded_evidence_ids"), list) else []
        return translated_result

    @classmethod
    def _personal_knowledge_payload(
        cls, batch: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
        catalog = [
            dict(batch.get("domain") or {}),
            *_items(batch.get("sessions")),
            *_items(batch.get("episodes")),
            *_items(batch.get("evidence")),
        ]
        real_to_alias, alias_to_real = cls._identifier_aliases(
            [_text(item.get("id"), 512) for item in catalog], "k"
        )

        def alias(value: Any) -> str:
            identifier = _text(value, 512)
            return real_to_alias.get(identifier, identifier)

        def compact(item: Mapping[str, Any]) -> dict[str, Any]:
            return {
                key: value
                for key, value in {
                    "id": alias(item.get("id")),
                    "level": item.get("level"),
                    "domain_id": alias(item.get("domain_id")),
                    "session_id": item.get("session_id"),
                    "episode_id": alias(item.get("episode_id")),
                    "label": item.get("label"),
                    "summary": item.get("summary"),
                    "status": item.get("status"),
                    "source_app": item.get("source_app"),
                    "parent_session_id": item.get("parent_session_id"),
                    "first_turn": item.get("first_turn"),
                    "last_turn": item.get("last_turn"),
                    "turn_index": item.get("turn_index"),
                    "occurred_at": item.get("occurred_at"),
                    "actor_role": item.get("actor_role"),
                    "state": item.get("state"),
                    "confidence": item.get("confidence"),
                    "evidence_kind": item.get("evidence_kind"),
                    "tags": item.get("tags"),
                    "topic_tags": item.get("topic_tags"),
                    "evidence_ids": [
                        alias(value)
                        for value in item.get("evidence_ids", [])
                        if _text(value, 512)
                    ]
                    if isinstance(item.get("evidence_ids"), list)
                    else None,
                    "episode_ids": [
                        alias(value)
                        for value in item.get("episode_ids", [])
                        if _text(value, 512)
                    ]
                    if isinstance(item.get("episode_ids"), list)
                    else None,
                }.items()
                if value not in (None, [], {})
            }

        domain = dict(batch.get("domain") or {})
        payload = {
            "schema_version": batch.get("schema_version"),
            "domain_id": alias(batch.get("domain_id")),
            "batch_id": batch.get("batch_id"),
            "batch_index": batch.get("batch_index"),
            "batch_count": batch.get("batch_count"),
            "complete_episode_batch": batch.get("complete_episode_batch"),
            "no_evidence_truncation": batch.get("no_evidence_truncation"),
            "allowed_page_types": batch.get("allowed_page_types"),
            "allowed_collections": batch.get("allowed_collections"),
            "collection_page_types": batch.get("collection_page_types"),
            "allowed_statuses": batch.get("allowed_statuses"),
            "domain": compact(domain),
            "sessions": [compact(item) for item in _items(batch.get("sessions"))],
            "episodes": [compact(item) for item in _items(batch.get("episodes"))],
            "evidence": [compact(item) for item in _items(batch.get("evidence"))],
            "expected_episode_ids": [
                alias(value) for value in batch.get("expected_episode_ids", [])
            ],
            "expected_evidence_ids": [
                alias(value) for value in batch.get("expected_evidence_ids", [])
            ],
        }
        return payload, real_to_alias, alias_to_real

    @staticmethod
    def _personal_knowledge_response_schema(
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        evidence_ids = sorted(
            {
                _text(item.get("id"), 512)
                for item in _items(payload.get("evidence"))
                if _text(item.get("id"), 512)
            }
        )
        evidence_id: dict[str, Any] = {"type": "string"}
        if evidence_ids:
            evidence_id["enum"] = evidence_ids

        def text() -> dict[str, Any]:
            return {"type": "string", "minLength": 1}

        def bilingual(fields: Sequence[str]) -> dict[str, Any]:
            localized = {
                "type": "object",
                "properties": {field: text() for field in fields},
                "required": list(fields),
                "additionalProperties": False,
            }
            return {
                "type": "object",
                "properties": {"zh": localized, "en": localized},
                "required": ["zh", "en"],
                "additionalProperties": False,
            }

        claim = {
            "type": "object",
            "properties": {
                "text": text(),
                "status": {
                    "type": "string",
                    "enum": sorted(
                        {
                            _text(item, 32)
                            for item in payload.get("allowed_statuses", [])
                            if _text(item, 32)
                        }
                    ),
                },
                "evidence_ids": {
                    "type": "array",
                    "items": evidence_id,
                    "minItems": 1,
                },
                "display": bilingual(("text",)),
            },
            "required": ["text", "status", "evidence_ids", "display"],
            "additionalProperties": False,
        }
        section = {
            "type": "object",
            "properties": {
                "heading": text(),
                "body": text(),
                "evidence_ids": {
                    "type": "array",
                    "items": evidence_id,
                    "minItems": 1,
                },
                "display": bilingual(("heading", "body")),
            },
            "required": ["heading", "body", "evidence_ids", "display"],
            "additionalProperties": False,
        }
        collection_page_types = payload.get("collection_page_types")
        collection_pairs = []
        if isinstance(collection_page_types, Mapping):
            for collection in sorted(collection_page_types):
                page_types = sorted(
                    {
                        _text(item, 40)
                        for item in collection_page_types.get(collection, [])
                        if _text(item, 40)
                    }
                )
                if page_types:
                    collection_pairs.append(
                        {
                            "properties": {
                                "collection": {
                                    "type": "string",
                                    "enum": [_text(collection, 32)],
                                },
                                "page_type": {
                                    "type": "string",
                                    "enum": page_types,
                                },
                            },
                            "required": ["collection", "page_type"],
                        }
                    )

        page = {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "enum": sorted(
                        {
                            _text(item, 32)
                            for item in payload.get("allowed_collections", [])
                            if _text(item, 32)
                        }
                    ),
                },
                "page_type": {
                    "type": "string",
                    "enum": sorted(
                        {
                            _text(item, 40)
                            for item in payload.get("allowed_page_types", [])
                            if _text(item, 40)
                        }
                    ),
                },
                "title": text(),
                "abstract": text(),
                "display": bilingual(("title", "abstract")),
                "claims": {
                    "type": "array",
                    "items": claim,
                    "maxItems": 3,
                },
                "sections": {
                    "type": "array",
                    "items": section,
                    "maxItems": 2,
                },
            },
            "required": [
                "collection",
                "page_type",
                "title",
                "abstract",
                "display",
                "claims",
                "sections",
            ],
            "additionalProperties": False,
            "allOf": [
                {"oneOf": collection_pairs},
                {
                    "anyOf": [
                        {
                            "properties": {
                                "claims": {
                                    "type": "array",
                                    "items": claim,
                                    "minItems": 1,
                                    "maxItems": 3,
                                }
                            }
                        },
                        {
                            "properties": {
                                "sections": {
                                    "type": "array",
                                    "items": section,
                                    "minItems": 1,
                                    "maxItems": 2,
                                }
                            }
                        },
                    ]
                },
            ],
        }
        return {
            "type": "object",
            "properties": {
                "schema_version": {
                    "type": "string",
                    "enum": [PERSONAL_KNOWLEDGE_DOMAIN_SCHEMA_VERSION],
                },
                "domain_id": {
                    "type": "string",
                    "enum": [_text(payload.get("domain_id"), 512)],
                },
                "batch_id": {
                    "type": "string",
                    "enum": [_text(payload.get("batch_id"), 512)],
                },
                "title": text(),
                "description": text(),
                "display": bilingual(("title", "description")),
                "pages": {
                    "type": "array",
                    "items": page,
                    "minItems": 1,
                    "maxItems": 4,
                },
                "excluded_evidence_ids": {
                    "type": "array",
                    "items": evidence_id,
                },
            },
            "required": [
                "schema_version",
                "domain_id",
                "batch_id",
                "title",
                "description",
                "display",
                "pages",
                "excluded_evidence_ids",
            ],
            "additionalProperties": False,
        }

    def personal_knowledge_batch(
        self, batch: Mapping[str, Any], *, slot_id: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload, real_to_alias, alias_to_real = self._personal_knowledge_payload(batch)
        try:
            result, call = self._call(
                PERSONAL_KNOWLEDGE_SYSTEM_PROMPT,
                payload,
                max_tokens=PERSONAL_KNOWLEDGE_MAX_OUTPUT_TOKENS,
                response_schema=self._personal_knowledge_response_schema(payload),
                response_schema_name="tmcra_personal_knowledge",
                slot_id=slot_id,
            )
        except SessionGraphError as exc:
            if exc.code not in {
                "session_graph_agent_invalid_json",
                "session_graph_agent_invalid_response",
            }:
                raise
            validation_error = {"code": exc.code, "message": str(exc)}
            repaired, repair_call = self.repair_personal_knowledge_batch(
                batch,
                {},
                validation_error=validation_error,
                slot_id=slot_id,
            )
            return repaired, self._alias_call_metadata(
                {
                    "repair_attempted": True,
                    "validation_error": validation_error,
                    "initial": {"failed": True, "error": validation_error},
                    "repair": repair_call,
                },
                real_to_alias,
            )
        return (
            self._translate_personal_knowledge_result(result, alias_to_real),
            self._alias_call_metadata(call, real_to_alias),
        )

    def repair_personal_knowledge_batch(
        self,
        batch: Mapping[str, Any],
        invalid_result: Mapping[str, Any],
        *,
        validation_error: Mapping[str, Any],
        slot_id: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload, real_to_alias, alias_to_real = self._personal_knowledge_payload(batch)
        payload["invalid_result"] = self._translate_personal_knowledge_result(
            invalid_result, real_to_alias
        )
        payload["validation_error"] = dict(validation_error)
        result, call = self._call(
            PERSONAL_KNOWLEDGE_REPAIR_SYSTEM_PROMPT,
            payload,
            max_tokens=PERSONAL_KNOWLEDGE_MAX_OUTPUT_TOKENS,
            response_schema=self._personal_knowledge_response_schema(payload),
            response_schema_name="tmcra_personal_knowledge_repair",
            slot_id=slot_id,
        )
        return (
            self._translate_personal_knowledge_result(result, alias_to_real),
            self._alias_call_metadata(call, real_to_alias),
        )


class SessionGraphAgentRouter:
    """Choose one projection provider for the lifetime of a queue task.

    Local inference owns the primary route.  The dedicated DeepSeek credential
    is selected only when the local server cannot offer a slot beyond its
    production reserve.  Selection happens before a task is claimed so a
    multi-call projection never mixes providers halfway through its output.
    """

    def __init__(
        self,
        *,
        local_agent: LocalSessionGraphAgent | None,
        fallback_agent: LocalSessionGraphAgent | None,
        local_failure_cooldown_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if local_agent is None and fallback_agent is None:
            raise SessionGraphError(
                "session_graph_agent_disabled",
                "no projection provider is configured",
            )
        if local_agent is not None and local_agent.provider != SESSION_GRAPH_PROVIDER_LOCAL:
            raise SessionGraphError(
                "session_graph_agent_route_invalid",
                "the primary projection provider must be local",
            )
        if (
            fallback_agent is not None
            and fallback_agent.provider != SESSION_GRAPH_PROVIDER_DEDICATED
        ):
            raise SessionGraphError(
                "session_graph_agent_route_invalid",
                "the fallback projection provider must be dedicated",
            )
        self.local_agent = local_agent
        self.fallback_agent = fallback_agent
        self.local_failure_cooldown_seconds = max(
            5.0, float(local_failure_cooldown_seconds)
        )
        self._clock = clock
        self._local_unavailable_until = 0.0

    @staticmethod
    def _with_route(
        environment: Mapping[str, str],
        *,
        provider: str,
        base_url: str,
        model: str,
        api_key: str,
        api_key_file: str,
        timeout_seconds: str,
    ) -> dict[str, str]:
        result = {str(key): str(value) for key, value in environment.items()}
        result["TMCRA_SESSION_GRAPH_PROVIDER"] = provider
        result["TMCRA_SESSION_GRAPH_BASE_URL"] = base_url
        result["TMCRA_SESSION_GRAPH_MODEL"] = model
        result["TMCRA_SESSION_GRAPH_API_KEY"] = api_key
        result["TMCRA_SESSION_GRAPH_API_KEY_FILE"] = api_key_file
        result["TMCRA_SESSION_GRAPH_AGENT_TIMEOUT_SECONDS"] = timeout_seconds
        return result

    @classmethod
    def from_env(
        cls, environment: Mapping[str, str] | None = None
    ) -> "SessionGraphAgentRouter | LocalSessionGraphAgent | None":
        env = dict(os.environ if environment is None else environment)
        enabled = _text(env.get("TMCRA_SESSION_GRAPH_AGENT_ENABLED") or "1").lower()
        if enabled in {"0", "false", "no", "off"}:
            return None
        provider = _text(
            env.get("TMCRA_SESSION_GRAPH_PROVIDER") or SESSION_GRAPH_PROVIDER_LOCAL,
            64,
        ).lower()
        if provider != SESSION_GRAPH_PROVIDER_LOCAL_FIRST:
            return LocalSessionGraphAgent.from_env(env)

        local_key = _text(env.get("TMCRA_SESSION_GRAPH_LOCAL_API_KEY"), 512)
        local_key_file = _text(
            env.get("TMCRA_SESSION_GRAPH_LOCAL_API_KEY_FILE")
            or env.get("TMCRA_LOCAL_WRITER_API_KEY_FILE")
            or "/opt/tmcra-data/local-llm/secrets/qwen36-server-lanes.key"
        )
        local_environment = cls._with_route(
            env,
            provider=SESSION_GRAPH_PROVIDER_LOCAL,
            base_url=_text(
                env.get("TMCRA_SESSION_GRAPH_LOCAL_BASE_URL") or LOCAL_QWEN_BASE_URL
            ),
            model=_text(
                env.get("TMCRA_SESSION_GRAPH_LOCAL_MODEL")
                or env.get("TMCRA_WRITER_MODEL")
                or env.get("TMCRA_LOCAL_WRITER_MODEL")
                or LOCAL_QWEN_MODEL
            ),
            api_key=local_key,
            api_key_file=local_key_file,
            timeout_seconds=_text(
                env.get("TMCRA_SESSION_GRAPH_LOCAL_TIMEOUT_SECONDS") or "900"
            ),
        )
        local_agent = LocalSessionGraphAgent.from_env(local_environment)

        fallback_key = _text(
            env.get("TMCRA_SESSION_GRAPH_FALLBACK_API_KEY")
            or env.get("TMCRA_SESSION_GRAPH_API_KEY"),
            512,
        )
        fallback_key_file = _text(
            env.get("TMCRA_SESSION_GRAPH_FALLBACK_API_KEY_FILE")
            or env.get("TMCRA_SESSION_GRAPH_API_KEY_FILE")
        )
        fallback_environment = cls._with_route(
            env,
            provider=SESSION_GRAPH_PROVIDER_DEDICATED,
            base_url=_text(
                env.get("TMCRA_SESSION_GRAPH_FALLBACK_BASE_URL")
                or DEDICATED_DEEPSEEK_BASE_URL
            ),
            model=_text(
                env.get("TMCRA_SESSION_GRAPH_FALLBACK_MODEL")
                or DEDICATED_DEEPSEEK_MODEL
            ),
            api_key=fallback_key,
            api_key_file=fallback_key_file,
            timeout_seconds=_text(
                env.get("TMCRA_SESSION_GRAPH_FALLBACK_TIMEOUT_SECONDS")
                or env.get("TMCRA_SESSION_GRAPH_AGENT_TIMEOUT_SECONDS")
                or "180"
            ),
        )
        fallback_agent = LocalSessionGraphAgent.from_env(fallback_environment)
        return cls(
            local_agent=local_agent,
            fallback_agent=fallback_agent,
            local_failure_cooldown_seconds=_number(
                env.get("TMCRA_SESSION_GRAPH_LOCAL_FAILURE_COOLDOWN_SECONDS"),
                60.0,
            ),
        )

    @property
    def model(self) -> str:
        if self.local_agent is not None:
            return self.local_agent.model
        assert self.fallback_agent is not None
        return self.fallback_agent.model

    @property
    def resource_isolation(self) -> str:
        return "adaptive-local-first"

    def select_agent(self) -> LocalSessionGraphAgent | None:
        local = self.local_agent
        if (
            local is not None
            and self._clock() >= self._local_unavailable_until
            and local.capacity_available()
        ):
            return local
        return self.fallback_agent

    def report_success(self, agent: LocalSessionGraphAgent | Any) -> None:
        if agent is self.local_agent:
            self._local_unavailable_until = 0.0

    def report_failure(self, agent: LocalSessionGraphAgent | Any) -> None:
        if agent is self.local_agent:
            self._local_unavailable_until = max(
                self._local_unavailable_until,
                self._clock() + self.local_failure_cooldown_seconds,
            )


class SessionGraphService:
    def __init__(
        self,
        database: ControlDB,
        storage: V4StorageAdapter,
        *,
        agent: LocalSessionGraphAgent | SessionGraphAgentRouter | None = None,
        poll_seconds: float = 2.0,
        initial_agent_messages: int | None = None,
        agent_message_delta: int | None = None,
        heavy_projection_message_delta: int | None = None,
        refresh_policy: str | None = None,
        knowledge_workers: int | None = None,
        idle_borrow_enabled: bool | None = None,
        production_capacity_guard: Callable[[], bool] | None = None,
        gpu_scheduler: GpuWorkloadScheduler | None = None,
    ) -> None:
        self.store = SessionGraphStore(database)
        self.jobs = JobStore(database)
        self.storage = storage
        self.agent = agent
        self.poll_seconds = max(0.5, poll_seconds)
        self.initial_agent_messages = max(
            1,
            int(
                initial_agent_messages
                if initial_agent_messages is not None
                else _number(os.environ.get("TMCRA_SESSION_GRAPH_INITIAL_MESSAGES"), 2)
            ),
        )
        self.agent_message_delta = max(
            2,
            int(
                agent_message_delta
                if agent_message_delta is not None
                else _number(os.environ.get("TMCRA_SESSION_GRAPH_MESSAGE_DELTA"), 16)
            ),
        )
        self.heavy_projection_message_delta = max(
            16,
            int(
                heavy_projection_message_delta
                if heavy_projection_message_delta is not None
                else _number(
                    os.environ.get("TMCRA_HEAVY_PROJECTION_MESSAGE_DELTA"),
                    256,
                )
            ),
        )
        self.refresh_policy = _text(
            refresh_policy
            if refresh_policy is not None
            else os.environ.get("TMCRA_SESSION_GRAPH_REFRESH_POLICY", "generation"),
            32,
        ).lower()
        if self.refresh_policy not in {"generation", "message"}:
            raise ValueError("session graph refresh policy must be generation or message")
        if idle_borrow_enabled is None:
            raw_idle_borrow = _text(
                os.environ.get("TMCRA_PROJECTION_IDLE_BORROW_ENABLED") or "0"
            ).lower()
            if raw_idle_borrow in {"1", "true", "yes", "on"}:
                idle_borrow_enabled = True
            elif raw_idle_borrow in {"0", "false", "no", "off"}:
                idle_borrow_enabled = False
            else:
                raise ValueError(
                    "TMCRA_PROJECTION_IDLE_BORROW_ENABLED must be a boolean"
                )
        self.idle_borrow_enabled = bool(idle_borrow_enabled)
        requested_knowledge_workers = max(
            1,
            int(
                knowledge_workers
                if knowledge_workers is not None
                else _number(
                    os.environ.get("TMCRA_PERSONAL_KNOWLEDGE_WORKERS"), 2
                )
            ),
        )
        # Slot 0 remains reserved for Writer. At most slot 1 (Planner) and slot
        # 2 (Graph) may serve derived projections, and only when idle borrowing
        # is explicitly enabled.
        self.knowledge_workers = (
            min(2, requested_knowledge_workers)
            if self.idle_borrow_enabled
            else 1
        )
        self.production_capacity_guard = production_capacity_guard
        self.gpu_scheduler = gpu_scheduler
        if gpu_scheduler is not None:
            if isinstance(agent, SessionGraphAgentRouter):
                if agent.local_agent is not None:
                    agent.local_agent.gpu_scheduler = gpu_scheduler
            elif isinstance(agent, LocalSessionGraphAgent):
                agent.gpu_scheduler = gpu_scheduler
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_production_capacity_guard(self, guard: Callable[[], bool] | None) -> None:
        if guard is not None and not callable(guard):
            raise TypeError("production capacity guard must be callable")
        self.production_capacity_guard = guard

    def _select_background_agent(self) -> LocalSessionGraphAgent | Any | None:
        configured = self.agent
        if configured is None:
            return None
        if isinstance(configured, SessionGraphAgentRouter):
            # A burst recall replica and the local projection model time-share
            # the GPU.  While recall owns that spare capacity, route derived
            # knowledge work to the isolated fallback without probing /slots;
            # /slots is an ordinary llama-server task and would wake a sleeping
            # local model just to perform admission.
            guard = self.production_capacity_guard
            if guard is not None:
                try:
                    if not bool(guard()):
                        return configured.fallback_agent
                except Exception:
                    return configured.fallback_agent
            # Once recall has retired its burst lane, the local route performs
            # the remaining model-slot reserve check.
            return configured.select_agent()
        # The production Qwen role contract pins projection work to slot 2.
        # Foreground Jobs use other model roles, so their durable Job state is
        # not by itself a reason to idle this slot.  GPU telemetry remains the
        # admission boundary through production_capacity_guard.
        if (
            not self._uses_local_gpu(configured)
            and self.store.production_work_pending()
        ):
            return None
        guard = self.production_capacity_guard
        if guard is not None:
            try:
                if not bool(guard()):
                    return None
            except Exception:
                return None
        capacity_available = getattr(configured, "capacity_available", None)
        if callable(capacity_available) and not bool(capacity_available()):
            return None
        return configured

    def _background_capacity_available(self) -> bool:
        return self._select_background_agent() is not None

    def _borrowed_planner_slot_available(self, agent: Any) -> bool:
        if not self.idle_borrow_enabled or not isinstance(
            agent, LocalSessionGraphAgent
        ):
            return False
        available = getattr(agent, "borrowed_planner_slot_available", None)
        return bool(callable(available) and available())

    @staticmethod
    def _uses_local_gpu(agent: LocalSessionGraphAgent | Any) -> bool:
        return bool(
            isinstance(agent, LocalSessionGraphAgent)
            and agent.provider == SESSION_GRAPH_PROVIDER_LOCAL
            and agent.base_url == LOCAL_QWEN_BASE_URL
        )

    def _journal_agent_call(
        self,
        tenant_id: str,
        scope_name: str,
        projection_key: str,
        operation: str,
        metadata: Any,
        *,
        default_model: str | None = None,
    ) -> None:
        """Record every physical projection call in the shared usage ledger."""

        seen: set[str] = set()

        def visit(item: Any) -> None:
            if isinstance(item, Mapping):
                call_id = _text(item.get("physical_call_id"), 256)
                if call_id and call_id not in seen:
                    seen.add(call_id)
                    journal_deepseek_calls(
                        self.jobs,
                        item,
                        tenant_id=tenant_id,
                        scope_name=scope_name,
                        job_id=None,
                        stage_id=f"projection:{projection_key}",
                        operation=operation,
                        default_model=default_model
                        or (
                            self.agent.model
                            if self.agent is not None
                            else "unknown"
                        ),
                    )
                for value in item.values():
                    visit(value)
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                for value in item:
                    visit(value)

        visit(metadata)

    def start(self) -> None:
        if self.agent is None or (self._thread is not None and self._thread.is_alive()):
            return
        # No projection worker exists before this service instance starts, so
        # every persisted running row belongs to an interrupted predecessor.
        self.store.recover_interrupted_refreshes()
        # A restart must not revive a heavy refresh that is still below its
        # evidence-bound waterline. Explicit manual requests remain queued.
        self._suppress_sub_waterline_visual_refreshes()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="tmcra-session-graph-agent",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, timeout))
        self._thread = None

    def _start_projection_heartbeat(
        self, task: Mapping[str, Any]
    ) -> tuple[threading.Event, threading.Thread]:
        """Keep a long projection attempt leased without rewriting progress."""

        heartbeat_stop = threading.Event()

        def keep_lease() -> None:
            while not heartbeat_stop.wait(30.0):
                try:
                    if not self.store.renew(task):
                        return
                except Exception:
                    # A transient database failure must not crash the service
                    # thread. If renewal cannot resume, normal stale recovery
                    # will reclaim the attempt and attempt fencing prevents the
                    # old worker from publishing over its successor.
                    return

        heartbeat = threading.Thread(
            target=keep_lease,
            name=(
                "tmcra-projection-lease-"
                + _text(task.get("projection_key"), 80).replace(":", "-")
            ),
            daemon=True,
        )
        heartbeat.start()
        return heartbeat_stop, heartbeat

    def _run(self) -> None:
        # Old imports may predate the projection agent.  Reconcile missing
        # projections in the worker thread so service startup remains fast and
        # users do not have to open the graph page to begin organization.
        try:
            self.reconcile_projection_backlog()
        except Exception:
            # One malformed historical scope must not stop the queue worker.
            pass
        while not self._stop.is_set():
            if not self.store.has_due_refresh():
                self._stop.wait(self.poll_seconds)
                continue
            task_agent = self._select_background_agent()
            if task_agent is None:
                self._stop.wait(max(2.0, self.poll_seconds))
                continue
            task = self.store.claim()
            if task is None:
                self._stop.wait(self.poll_seconds)
                continue
            heartbeat_stop, heartbeat = self._start_projection_heartbeat(task)
            try:
                try:
                    self._refresh_task(task, task_agent=task_agent)
                except (SessionGraphError, VisualAtlasError, PersonalKnowledgeError) as exc:
                    if isinstance(self.agent, SessionGraphAgentRouter):
                        self.agent.report_failure(task_agent)
                    self.store.defer(task, seconds=15.0, reason=str(exc))
                except (GraphProjectionError, V4AdapterError) as exc:
                    self.store.defer(task, seconds=15.0, reason=str(exc))
                except Exception as exc:
                    self.store.fail(task, exc)
                else:
                    if isinstance(self.agent, SessionGraphAgentRouter):
                        self.agent.report_success(task_agent)
            finally:
                heartbeat_stop.set()
                heartbeat.join(timeout=5.0)

    def _projection(self, tenant_id: str, scope_name: str) -> MemoryGraphProjection:
        return MemoryGraphProjection.from_available_storage(
            self.storage, tenant_id=tenant_id, scope_name=scope_name
        )

    @staticmethod
    def _source_bound_session(
        session: Mapping[str, Any], source_graph: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Bind display/agent counts to committed Source evidence.

        ``scope_sessions.message_count`` is an admission counter. Historical
        retries can therefore make it larger than the unique Source records
        that were actually committed. Session Graph and Personal Knowledge
        operate on immutable committed Source evidence, so their completeness
        watermark must use the graph count rather than the admission counter.
        """

        result = dict(session)
        result["catalog_message_count"] = max(
            0, int(_number(session.get("message_count"), 0))
        )
        result["message_count"] = max(
            0, int(_number(source_graph.get("source_record_count"), 0))
        )
        return result

    @staticmethod
    def _public_refresh_state(state: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if state is None:
            return None
        return {
            "state": _text(state.get("state"), 32),
            "attempts": max(0, int(_number(state.get("attempts"), 0))),
            "stage": _text(state.get("progress_stage"), 80) or None,
            "completed": (
                None
                if state.get("progress_completed") is None
                else max(0, int(_number(state.get("progress_completed"), 0)))
            ),
            "total": (
                None
                if state.get("progress_total") is None
                else max(0, int(_number(state.get("progress_total"), 0)))
            ),
            "updated_at": _number(state.get("updated_at"), 0.0),
        }

    @staticmethod
    def _session_agent_checkpoint(stored: Mapping[str, Any] | None) -> int:
        if not stored:
            return 0
        projection = stored.get("projection")
        if not isinstance(projection, Mapping):
            return 0
        checkpoint = projection.get("agent_checkpoint")
        if isinstance(checkpoint, Mapping):
            return max(0, int(_number(checkpoint.get("message_count"), 0)))
        if stored.get("generator") == "local-session-map-agent":
            return max(0, int(_number(projection.get("message_count"), 0)))
        return 0

    def _session_agent_due(
        self,
        session: Mapping[str, Any],
        stored: Mapping[str, Any] | None,
    ) -> bool:
        if self.agent is None:
            return False
        message_count = max(0, int(_number(session.get("message_count"), 0)))
        checkpoint = self._session_agent_checkpoint(stored)
        if checkpoint == 0:
            return message_count >= self.initial_agent_messages
        if _text(session.get("status"), 32) in {"completed", "archived"}:
            return message_count > checkpoint
        return message_count - checkpoint >= self.agent_message_delta

    @staticmethod
    def _atlas_agent_checkpoint(stored: Mapping[str, Any] | None) -> list[str]:
        if not stored:
            return []
        projection = stored.get("projection")
        if not isinstance(projection, Mapping):
            return []
        checkpoint = projection.get("agent_checkpoint")
        values = checkpoint.get("session_ids") if isinstance(checkpoint, Mapping) else None
        if not isinstance(values, list) and stored.get("generator") == "local-session-atlas-agent":
            values = [item.get("session_id") for item in _items(projection.get("nodes"))]
        if not isinstance(values, list):
            return []
        return sorted({_text(item, 200) for item in values if _text(item, 200)})

    def _atlas_agent_due(
        self,
        sessions: Sequence[Mapping[str, Any]],
        stored: Mapping[str, Any] | None,
    ) -> bool:
        if self.agent is None or not sessions:
            return False
        current = sorted(
            {
                _text(item.get("session_id"), 200)
                for item in sessions
                if _text(item.get("session_id"), 200)
            }
        )
        return current != self._atlas_agent_checkpoint(stored)

    @staticmethod
    def _valid_projection(
        stored: Mapping[str, Any] | None,
        schema_version: str,
        *,
        require_full: bool = False,
    ) -> bool:
        projection = (stored or {}).get("projection")
        if not isinstance(projection, Mapping):
            return False
        if projection.get("schema_version") != schema_version:
            return False
        if require_full and (
            projection.get("full_projection") is not True
            or projection.get("truncated") is not False
        ):
            return False
        return True

    @classmethod
    def _ready_projection(
        cls,
        stored: Mapping[str, Any] | None,
        schema_version: str,
        *,
        require_full: bool = False,
    ) -> bool:
        return cls._valid_projection(
            stored, schema_version, require_full=require_full
        ) and (stored or {}).get("projection", {}).get("projection_state") == "ready"

    @classmethod
    def _current_projection_ready(
        cls,
        stored: Mapping[str, Any] | None,
        schema_version: str,
        refresh_state: Mapping[str, Any] | None,
        *,
        require_full: bool = False,
    ) -> bool:
        queue_state = _text((refresh_state or {}).get("state"), 32)
        return queue_state not in {"dirty", "running", "failed"} and cls._ready_projection(
            stored,
            schema_version,
            require_full=require_full,
        )

    def _projection_pipeline_complete(self, tenant_id: str, scope_name: str) -> bool:
        visual = self.store.get_view(tenant_id, scope_name, VISUAL_ATLAS_KEY)
        knowledge = self.store.get_view(
            tenant_id, scope_name, PERSONAL_KNOWLEDGE_KEY
        )
        return self._ready_projection(
            visual, VISUAL_ATLAS_SCHEMA_VERSION, require_full=True
        ) and self._ready_projection(
            knowledge, PERSONAL_KNOWLEDGE_SCHEMA_VERSION, require_full=True
        )

    @staticmethod
    def _published_visual_message_watermark(
        stored: Mapping[str, Any] | None,
    ) -> int:
        projection = (stored or {}).get("projection")
        if not isinstance(projection, Mapping):
            return 0
        checkpoint = projection.get("agent_checkpoint")
        if isinstance(checkpoint, Mapping) and checkpoint.get("message_count") is not None:
            return max(0, int(_number(checkpoint.get("message_count"), 0)))
        return sum(
            max(0, int(_number(item.get("message_count"), 0)))
            for item in _items(projection.get("nodes"))
            if _text(item.get("level"), 32) == "session"
        )

    @staticmethod
    def _manual_visual_refresh_pending(state: Mapping[str, Any] | None) -> bool:
        if not state:
            return False
        return any(
            _text(state.get(name), 512).startswith(MANUAL_VISUAL_REFRESH_PREFIX)
            for name in ("source_fingerprint", "pending_source_fingerprint")
        )

    def _visual_atlas_auto_refresh_due(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        stored: Mapping[str, Any] | None = None,
    ) -> bool:
        visual = stored or self.store.get_view(
            tenant_id, scope_name, VISUAL_ATLAS_KEY
        )
        if not self._ready_projection(
            visual, VISUAL_ATLAS_SCHEMA_VERSION, require_full=True
        ):
            return True
        published = self._published_visual_message_watermark(visual)
        current = self.store.session_projection_message_watermark(
            tenant_id, scope_name
        )
        return current - published >= self.heavy_projection_message_delta

    def _suppress_sub_waterline_visual_refreshes(self) -> int:
        suppressed = 0
        for item in self.store.scopes_with_sessions():
            tenant_id = str(item["tenant_id"])
            scope_name = str(item["scope_name"])
            state = self.store.refresh_state(
                tenant_id, scope_name, VISUAL_ATLAS_KEY
            )
            if _text((state or {}).get("state"), 32) != "dirty":
                continue
            if self._manual_visual_refresh_pending(state):
                continue
            if self._visual_atlas_auto_refresh_due(tenant_id, scope_name):
                continue
            if self.store.cancel_dirty_refresh(
                tenant_id,
                scope_name,
                VISUAL_ATLAS_KEY,
                stage="waiting_for_message_waterline",
            ):
                suppressed += 1
        return suppressed

    @staticmethod
    def _build_seed(
        scope_name: str,
        sessions: Sequence[Mapping[str, Any]],
        *,
        projection_key: str,
    ) -> str:
        return _fingerprint(
            {
                "schema": "tmcra-projection-build-v1",
                "scope_name": scope_name,
                "projection_key": projection_key,
                "sessions": [
                    {
                        "session_id": item.get("session_id"),
                        "message_count": item.get("message_count"),
                        "last_ingest_at": item.get("last_ingest_at"),
                    }
                    for item in sessions
                ],
            }
        )

    def _enqueue_build_step(
        self,
        tenant_id: str,
        scope_name: str,
        projection_key: str,
        *,
        source_fingerprint: str,
        retry_failed: bool,
        delay_seconds: float = 0.0,
    ) -> bool:
        state = self.store.refresh_state(tenant_id, scope_name, projection_key)
        current = _text((state or {}).get("state"), 32)
        if current in {"dirty", "running"}:
            return False
        if current == "failed" and not retry_failed:
            return False
        self.store.enqueue(
            tenant_id,
            scope_name,
            projection_key,
            source_fingerprint=source_fingerprint,
            delay_seconds=delay_seconds,
        )
        return True

    def ensure_projection_build(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        retry_failed: bool = False,
    ) -> dict[str, Any]:
        """Queue every missing projection without opening the Source graph.

        The expensive evidence projection and all LLM work stay in the
        background worker.  This makes first-open latency independent of the
        amount of imported history.
        """

        if self.agent is None:
            raise SessionGraphError(
                "session_graph_agent_disabled",
                "the projection agent is not configured",
                status_code=503,
            )
        sessions = self.store.sessions(tenant_id, scope_name)
        if not sessions:
            raise SessionGraphError(
                "projection_build_no_sessions",
                "the Scope has no committed Sessions yet",
                status_code=404,
            )

        for index, session in enumerate(sessions):
            session_id = str(session["session_id"])
            key = session_projection_key(session_id)
            stored = self.store.get_view(tenant_id, scope_name, key)
            if self._ready_projection(stored, SESSION_MAP_SCHEMA_VERSION):
                continue
            fingerprint = self._build_seed(
                scope_name, [session], projection_key=key
            )
            self._enqueue_build_step(
                tenant_id,
                scope_name,
                key,
                source_fingerprint=fingerprint,
                retry_failed=retry_failed,
                delay_seconds=min(2.0, index * 0.02),
            )

        steps = (
            (ATLAS_KEY, SESSION_ATLAS_SCHEMA_VERSION, False, 1.0),
            (VISUAL_ATLAS_KEY, VISUAL_ATLAS_SCHEMA_VERSION, True, 2.0),
            (
                PERSONAL_KNOWLEDGE_KEY,
                PERSONAL_KNOWLEDGE_SCHEMA_VERSION,
                True,
                3.0,
            ),
        )
        for key, schema, require_full, delay in steps:
            stored = self.store.get_view(tenant_id, scope_name, key)
            if self._ready_projection(
                stored, schema, require_full=require_full
            ):
                continue
            self._enqueue_build_step(
                tenant_id,
                scope_name,
                key,
                source_fingerprint=self._build_seed(
                    scope_name, sessions, projection_key=key
                ),
                retry_failed=retry_failed,
                delay_seconds=delay,
            )
        return self.projection_build_status(tenant_id, scope_name)

    def reconcile_projection_backlog(self) -> dict[str, int]:
        """Schedule historical scopes which never received derived views."""

        checked = 0
        scheduled = 0
        for item in self.store.scopes_with_sessions():
            if self._stop.is_set():
                break
            if int(item.get("message_count") or 0) < self.initial_agent_messages:
                continue
            checked += 1
            tenant_id = str(item["tenant_id"])
            scope_name = str(item["scope_name"])
            if self._projection_pipeline_complete(tenant_id, scope_name):
                continue
            try:
                self.ensure_projection_build(
                    tenant_id, scope_name, retry_failed=True
                )
            except SessionGraphError:
                continue
            scheduled += 1
        return {"checked_scopes": checked, "scheduled_scopes": scheduled}

    def projection_build_status(
        self, tenant_id: str, scope_name: str
    ) -> dict[str, Any]:
        sessions = self.store.sessions(tenant_id, scope_name)
        states = self.store.refresh_states(tenant_id, scope_name)
        total_sessions = len(sessions)
        ready_sessions = 0
        session_state_counts = Counter()
        updated_at_values: list[float] = []

        for session in sessions:
            key = session_projection_key(str(session["session_id"]))
            stored = self.store.get_view(tenant_id, scope_name, key)
            state = states.get(key)
            if self._current_projection_ready(
                stored,
                SESSION_MAP_SCHEMA_VERSION,
                state,
            ):
                ready_sessions += 1
            state_name = _text((state or {}).get("state"), 32)
            if state_name:
                session_state_counts[state_name] += 1
            if stored:
                updated_at_values.append(float(stored.get("updated_at") or 0.0))

        atlas = self.store.get_view(tenant_id, scope_name, ATLAS_KEY)
        visual = self.store.get_view(tenant_id, scope_name, VISUAL_ATLAS_KEY)
        knowledge = self.store.get_view(
            tenant_id, scope_name, PERSONAL_KNOWLEDGE_KEY
        )
        atlas_available = self._ready_projection(
            atlas, SESSION_ATLAS_SCHEMA_VERSION
        )
        atlas_ready = self._current_projection_ready(
            atlas,
            SESSION_ATLAS_SCHEMA_VERSION,
            states.get(ATLAS_KEY),
        )
        graph_available = self._valid_projection(
            visual, VISUAL_ATLAS_SCHEMA_VERSION, require_full=True
        )
        graph_ready = self._current_projection_ready(
            visual,
            VISUAL_ATLAS_SCHEMA_VERSION,
            states.get(VISUAL_ATLAS_KEY),
            require_full=True,
        )
        knowledge_available = self._valid_projection(
            knowledge, PERSONAL_KNOWLEDGE_SCHEMA_VERSION, require_full=True
        )
        knowledge_ready = self._current_projection_ready(
            knowledge,
            PERSONAL_KNOWLEDGE_SCHEMA_VERSION,
            states.get(PERSONAL_KNOWLEDGE_KEY),
            require_full=True,
        )
        for item in (atlas, visual, knowledge):
            if item:
                updated_at_values.append(float(item.get("updated_at") or 0.0))
        updated_at_values.extend(
            float(item.get("updated_at") or 0.0) for item in states.values()
        )

        active_states = [_text(item.get("state"), 32) for item in states.values()]
        all_ready = bool(total_sessions) and (
            ready_sessions == total_sessions
            and atlas_ready
            and graph_ready
            and knowledge_ready
            and not any(item in {"dirty", "running", "failed"} for item in active_states)
        )
        if all_ready:
            status = "ready"
            stage = "ready"
        elif any(item == "running" for item in active_states):
            status = "running"
            stage = "session_maps"
        elif any(item == "dirty" for item in active_states):
            status = "queued"
            stage = "session_maps"
        elif any(item == "failed" for item in active_states):
            status = "failed"
            stage = "session_maps"
        else:
            status = "queued"
            stage = "session_maps"

        if ready_sessions < total_sessions:
            stage = "session_maps"
        elif not atlas_ready:
            stage = "session_atlas"
        elif not graph_ready:
            stage = "visual_atlas"
        elif not knowledge_ready:
            stage = "knowledge_base"
        elif all_ready:
            stage = "ready"

        total_units = max(1, total_sessions + 3)
        completed_units = (
            ready_sessions
            + int(atlas_ready)
            + int(graph_ready)
            + int(knowledge_ready)
        )
        progress_percent = projection_progress_percent(
            total_sessions=total_sessions,
            ready_sessions=ready_sessions,
            atlas_ready=atlas_ready,
            graph_ready=graph_ready,
            knowledge_ready=knowledge_ready,
            all_ready=all_ready,
        )
        visual_state = states.get(VISUAL_ATLAS_KEY) or {}
        knowledge_state = states.get(PERSONAL_KNOWLEDGE_KEY) or {}

        def stage_fraction(state: Mapping[str, Any]) -> float:
            total = max(0, int(_number(state.get("progress_total"), 0)))
            completed = max(0, int(_number(state.get("progress_completed"), 0)))
            return min(1.0, completed / total) if total else 0.0

        if not all_ready and ready_sessions == total_sessions and atlas_ready:
            if not graph_ready:
                progress_percent = min(
                    79,
                    SESSION_MAP_PROGRESS_WEIGHT
                    + SESSION_ATLAS_PROGRESS_WEIGHT
                    + round(VISUAL_ATLAS_PROGRESS_WEIGHT * stage_fraction(visual_state)),
                )
            elif not knowledge_ready:
                progress_percent = min(
                    99,
                    SESSION_MAP_PROGRESS_WEIGHT
                    + SESSION_ATLAS_PROGRESS_WEIGHT
                    + VISUAL_ATLAS_PROGRESS_WEIGHT
                    + round(
                        PERSONAL_KNOWLEDGE_PROGRESS_WEIGHT
                        * stage_fraction(knowledge_state)
                    ),
                )

        failed = [
            item for item in states.values() if _text(item.get("state"), 32) == "failed"
        ]
        stage_details = {
            "session_maps": f"Organizing conversations ({ready_sessions}/{total_sessions})",
            "session_atlas": "Linking conversation maps",
            "visual_atlas": "Building the visual memory graph",
            "knowledge_base": "Writing the personal knowledge base",
            "ready": "Memory graph and knowledge base are ready",
        }
        stage_state = (
            visual_state
            if stage == "visual_atlas"
            else knowledge_state
            if stage == "knowledge_base"
            else {}
        )
        stage_completed = stage_state.get("progress_completed")
        stage_total = stage_state.get("progress_total")
        detail = stage_details[stage]
        if (
            stage_completed is not None
            and stage_total is not None
            and int(stage_total) > 0
        ):
            detail = f"{detail} ({int(stage_completed)}/{int(stage_total)})"
        return {
            "schema_version": "tmcra.projection-build-progress.1",
            "scope_name": scope_name,
            "status": status,
            "stage": stage,
            "progress_percent": max(0, progress_percent),
            "completed_units": completed_units,
            "total_units": total_units,
            "session_maps": {
                "total": total_sessions,
                "ready": ready_sessions,
                "queued": int(session_state_counts.get("dirty", 0)),
                "running": int(session_state_counts.get("running", 0)),
                "failed": int(session_state_counts.get("failed", 0)),
            },
            "session_atlas": {
                "available": atlas_available,
                "ready": atlas_ready,
                "state": _text((states.get(ATLAS_KEY) or {}).get("state"), 32)
                or ("ready" if atlas_ready else "waiting"),
            },
            "visual_atlas": {
                "available": graph_available,
                "ready": graph_ready,
                "state": _text(
                    (states.get(VISUAL_ATLAS_KEY) or {}).get("state"), 32
                )
                or ("ready" if graph_ready else "waiting"),
                "stage": _text(visual_state.get("progress_stage"), 80) or None,
                "completed": visual_state.get("progress_completed"),
                "total": visual_state.get("progress_total"),
            },
            "knowledge_base": {
                "available": knowledge_available,
                "ready": knowledge_ready,
                "state": _text(
                    (states.get(PERSONAL_KNOWLEDGE_KEY) or {}).get("state"), 32
                )
                or ("ready" if knowledge_ready else "waiting"),
                "stage": _text(knowledge_state.get("progress_stage"), 80) or None,
                "completed": knowledge_state.get("progress_completed"),
                "total": knowledge_state.get("progress_total"),
            },
            "detail": detail,
            "last_error": _text((failed[0] if failed else {}).get("last_error"), 1000)
            or None,
            "can_retry": bool(failed),
            "updated_at": max(updated_at_values, default=0.0),
            "agent_enabled": self.agent is not None,
            "resource_isolation": (
                getattr(self.agent, "resource_isolation", "unknown")
                if self.agent is not None
                else "disabled"
            ),
        }

    def request_projection_build(
        self, tenant_id: str, scope_name: str
    ) -> dict[str, Any]:
        return self.ensure_projection_build(
            tenant_id, scope_name, retry_failed=True
        )

    def _base_session_map(
        self, tenant_id: str, scope_name: str, session_id: str
    ) -> tuple[dict[str, Any], str]:
        session = self.store.session(tenant_id, scope_name, session_id)
        if session is None:
            raise SessionGraphError("session_not_found", "Session was not found", status_code=404)
        source_graph = self._projection(tenant_id, scope_name).session_overview(session_id)
        if int(source_graph.get("source_record_count") or 0) < int(
            session.get("message_count") or 0
        ):
            try:
                live_graph = MemoryGraphProjection.from_live_storage(
                    self.storage,
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                ).session_overview(session_id)
            except GraphProjectionError:
                live_graph = None
            if live_graph is not None and int(
                live_graph.get("source_record_count") or 0
            ) > int(source_graph.get("source_record_count") or 0):
                source_graph = live_graph
        session = self._source_bound_session(session, source_graph)
        graph = build_session_map(source_graph, session)
        fingerprint = _fingerprint(
            {
                "snapshot_id": graph.get("snapshot_id"),
                "session_id": session_id,
                "message_count": session.get("message_count"),
                "source_record_ids": dict(graph.get("evidence_binding") or {}).get("source_record_ids", []),
                "semantic_ids": [item.get("id") for item in graph.get("nodes", [])],
            }
        )
        return graph, fingerprint

    def record_committed(
        self,
        tenant_id: str,
        scope_name: str,
        session_id: str,
        source_event_seq: int,
    ) -> None:
        if self.agent is None or self.refresh_policy != "message":
            return
        session = self.store.session(tenant_id, scope_name, session_id)
        if session is None:
            return
        key = session_projection_key(session_id)
        stored = self.store.get_view(tenant_id, scope_name, key)
        try:
            base, fingerprint = self._base_session_map(
                tenant_id, scope_name, session_id
            )
        except (GraphProjectionError, SessionGraphError):
            return
        source_bound = dict(session)
        source_bound["message_count"] = int(base.get("message_count") or 0)
        if self._session_agent_due(source_bound, stored):
            self.store.enqueue(
                tenant_id,
                scope_name,
                key,
                source_fingerprint=(
                    fingerprint
                    or f"source-event:{max(0, int(source_event_seq))}"
                ),
                delay_seconds=2.0,
            )

    def record_generation_committed(
        self,
        tenant_id: str,
        scope_name: str,
        promoted_event_seq: int,
    ) -> None:
        if self.agent is None:
            return
        sessions = self.store.sessions(tenant_id, scope_name)
        refresh_needed = False
        eligible_for_initial_build = False
        for session in sessions:
            session_id = str(session["session_id"])
            stored = self.store.get_view(
                tenant_id, scope_name, session_projection_key(session_id)
            )
            try:
                base, session_fingerprint = self._base_session_map(
                    tenant_id, scope_name, session_id
                )
            except (GraphProjectionError, SessionGraphError):
                continue
            source_bound = dict(session)
            source_bound["message_count"] = int(base.get("message_count") or 0)
            eligible_for_initial_build = eligible_for_initial_build or (
                int(source_bound["message_count"]) >= self.initial_agent_messages
            )
            if not self._session_agent_due(source_bound, stored):
                continue
            self.store.enqueue(
                tenant_id,
                scope_name,
                session_projection_key(session_id),
                source_fingerprint=session_fingerprint,
            )
            refresh_needed = True
        if refresh_needed:
            self.store.enqueue(
                tenant_id,
                scope_name,
                ATLAS_KEY,
                source_fingerprint=(
                    f"promoted-generation:{max(0, int(promoted_event_seq))}"
                ),
                delay_seconds=2.0,
            )
        elif eligible_for_initial_build and not self._projection_pipeline_complete(
            tenant_id, scope_name
        ):
            self.ensure_projection_build(tenant_id, scope_name)

    def _base_atlas(
        self, tenant_id: str, scope_name: str
    ) -> tuple[dict[str, Any], str]:
        sessions = self.store.sessions(tenant_id, scope_name)
        views: dict[str, Mapping[str, Any]] = {}
        for session in sessions:
            session_id = str(session["session_id"])
            stored = self.store.get_view(
                tenant_id, scope_name, session_projection_key(session_id)
            )
            if stored:
                views[session_id] = stored["projection"]
        graph = build_session_atlas(scope_name, sessions, views)
        fingerprint = _fingerprint(
            {
                "sessions": [
                    {
                        "session_id": item.get("session_id"),
                        "message_count": item.get("message_count"),
                        "last_ingest_at": item.get("last_ingest_at"),
                        "parent_session_id": item.get("parent_session_id"),
                        "view": (
                            self.store.get_view(
                                tenant_id,
                                scope_name,
                                session_projection_key(str(item["session_id"])),
                            )
                            or {}
                        ).get("source_fingerprint"),
                    }
                    for item in sessions
                ]
            }
        )
        return graph, fingerprint

    def _visual_atlas_inputs(
        self, tenant_id: str, scope_name: str
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        sessions = [dict(item) for item in self.store.sessions(tenant_id, scope_name)]
        if not sessions:
            raise SessionGraphError(
                "visual_atlas_no_sessions",
                "the Scope has no committed Sessions yet",
                status_code=404,
            )
        views: dict[str, dict[str, Any]] = {}
        atlas_view = self.store.get_view(tenant_id, scope_name, ATLAS_KEY)
        atlas_projection = (atlas_view or {}).get("projection")
        if not isinstance(atlas_projection, Mapping):
            atlas_projection = {}
        atlas_nodes = {
            _text(item.get("session_id"), 512): item
            for item in _items(atlas_projection.get("nodes"))
            if _text(item.get("session_id"), 512)
        }
        projection = self._projection(tenant_id, scope_name)
        live_projection: MemoryGraphProjection | None = None
        source_graphs: dict[str, dict[str, Any]] = {}
        for index, session in enumerate(sessions):
            session_id = _text(session.get("session_id"), 512)
            stored = self.store.get_view(
                tenant_id, scope_name, session_projection_key(session_id)
            )
            view = dict((stored or {}).get("projection") or {})
            atlas_node = atlas_nodes.get(session_id)
            if atlas_node:
                if atlas_node.get("title"):
                    view["title"] = atlas_node.get("title")
                if atlas_node.get("summary"):
                    view["summary"] = atlas_node.get("summary")
                if isinstance(atlas_node.get("topic_tags"), list):
                    view["topic_tags"] = list(atlas_node.get("topic_tags") or [])
            views[session_id] = view
            graph = projection.session_overview(
                session_id,
                semantic_limit=20_000,
                source_limit=100_000,
                include_source_text=True,
            )
            expected_sources = max(0, int(_number(session.get("message_count"), 0)))
            if int(graph.get("source_record_count") or 0) < expected_sources:
                if live_projection is None:
                    try:
                        live_projection = MemoryGraphProjection.from_live_storage(
                            self.storage,
                            tenant_id=tenant_id,
                            scope_name=scope_name,
                        )
                    except GraphProjectionError:
                        live_projection = None
                if live_projection is not None:
                    candidate = live_projection.session_overview(
                        session_id,
                        semantic_limit=20_000,
                        source_limit=100_000,
                        include_source_text=True,
                    )
                    if int(candidate.get("source_record_count") or 0) > int(
                        graph.get("source_record_count") or 0
                    ):
                        graph = candidate
            sessions[index] = self._source_bound_session(session, graph)
            if bool(dict(graph.get("page") or {}).get("truncated")):
                raise SessionGraphError(
                    "visual_atlas_full_projection_required",
                    f"Session {session_id} exceeds the full visual projection boundary",
                )
            source_graphs[session_id] = graph
        return sessions, views, source_graphs

    def _base_visual_atlas(
        self, tenant_id: str, scope_name: str
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        sessions, views, source_graphs = self._visual_atlas_inputs(
            tenant_id, scope_name
        )
        try:
            graph = build_visual_atlas(scope_name, sessions, views, source_graphs)
        except VisualAtlasError as exc:
            raise SessionGraphError(exc.code, str(exc)) from exc
        fingerprint = _fingerprint(
            {
                "schema": VISUAL_ATLAS_PROMPT_VERSION,
                "sessions": [
                    {
                        "session_id": item.get("session_id"),
                        "message_count": item.get("message_count"),
                        "last_ingest_at": item.get("last_ingest_at"),
                        "parent_session_id": item.get("parent_session_id"),
                        "source_snapshot": source_graphs[str(item["session_id"])].get("snapshot_id"),
                        "source_ids": [
                            node.get("id")
                            for node in source_graphs[str(item["session_id"])].get("nodes", [])
                        ],
                        "view": views.get(str(item["session_id"])),
                    }
                    for item in sessions
                ],
            }
        )
        return graph, fingerprint, sessions, views, source_graphs

    def session_map(
        self, tenant_id: str, scope_name: str, session_id: str
    ) -> dict[str, Any]:
        key = session_projection_key(session_id)
        base, fingerprint = self._base_session_map(
            tenant_id, scope_name, session_id
        )
        stored = self.store.get_view(tenant_id, scope_name, key)
        if stored and stored["source_fingerprint"] == fingerprint:
            result = dict(stored["projection"])
        else:
            checkpoint = self._session_agent_checkpoint(stored)
            if checkpoint:
                base["agent_checkpoint"] = {"message_count": checkpoint}
            self.store.put_view(
                tenant_id,
                scope_name,
                key,
                base,
                source_snapshot_id=_text(base.get("snapshot_id")) or None,
                source_fingerprint=fingerprint,
                generator="deterministic-evidence-projection",
            )
            result = base
            session = self.store.session(tenant_id, scope_name, session_id)
            source_bound = (
                self._source_bound_session(session, base)
                if session is not None
                else None
            )
            if source_bound is not None and self._session_agent_due(source_bound, stored):
                self.store.enqueue(
                    tenant_id,
                    scope_name,
                    key,
                    source_fingerprint=fingerprint,
                )
        result["refresh"] = self._public_refresh_state(
            self.store.refresh_state(tenant_id, scope_name, key)
        )
        result.pop("agent_checkpoint", None)
        result.pop("agent_call", None)
        return result

    def atlas(self, tenant_id: str, scope_name: str) -> dict[str, Any]:
        sessions = self.store.sessions(tenant_id, scope_name)
        base, fingerprint = self._base_atlas(tenant_id, scope_name)
        stored = self.store.get_view(tenant_id, scope_name, ATLAS_KEY)
        if stored and stored["source_fingerprint"] == fingerprint:
            result = dict(stored["projection"])
        else:
            checkpoint = self._atlas_agent_checkpoint(stored)
            if checkpoint:
                base["agent_checkpoint"] = {"session_ids": checkpoint}
            self.store.put_view(
                tenant_id,
                scope_name,
                ATLAS_KEY,
                base,
                source_snapshot_id=_text(base.get("snapshot_id")) or None,
                source_fingerprint=fingerprint,
                generator="deterministic-session-catalog",
            )
            result = base
            if self._atlas_agent_due(sessions, stored):
                self.store.enqueue(
                    tenant_id,
                    scope_name,
                    ATLAS_KEY,
                    source_fingerprint=fingerprint,
                    delay_seconds=2.0,
                )
        result["refresh"] = self._public_refresh_state(
            self.store.refresh_state(tenant_id, scope_name, ATLAS_KEY)
        )
        result["agent_enabled"] = self.agent is not None
        result.pop("agent_checkpoint", None)
        result.pop("agent_call", None)
        return result

    def visual_atlas(self, tenant_id: str, scope_name: str) -> dict[str, Any]:
        stored = self.store.get_view(
            tenant_id, scope_name, VISUAL_ATLAS_KEY
        )
        projection = (stored or {}).get("projection")
        if (
            isinstance(projection, Mapping)
            and projection.get("schema_version") == VISUAL_ATLAS_SCHEMA_VERSION
            and projection.get("full_projection") is True
            and projection.get("truncated") is False
        ):
            result = dict(projection)
            result["refresh"] = self._public_refresh_state(
                self.store.refresh_state(tenant_id, scope_name, VISUAL_ATLAS_KEY)
            )
            result["agent_enabled"] = self.agent is not None
            result.pop("agent_checkpoint", None)
            result.pop("agent_call", None)
            return result

        # A first projection can traverse thousands of Source records.  Never
        # make the request connection own that work; queue it and let clients
        # follow the projection-build progress resource.
        if self.agent is not None:
            self.ensure_projection_build(tenant_id, scope_name)
        raise SessionGraphError(
            "projection_build_pending",
            "the visual memory graph is being organized in the background",
            status_code=409,
        )

    def _personal_knowledge_atlas(
        self, tenant_id: str, scope_name: str
    ) -> tuple[dict[str, Any], str, bool]:
        stored = self.store.get_view(tenant_id, scope_name, VISUAL_ATLAS_KEY)
        projection = (stored or {}).get("projection")
        if (
            stored
            and isinstance(projection, Mapping)
            and projection.get("schema_version") == VISUAL_ATLAS_SCHEMA_VERSION
            and projection.get("full_projection") is True
            and projection.get("truncated") is False
        ):
            return (
                dict(projection),
                str(stored["source_fingerprint"]),
                projection.get("projection_state") == "ready",
            )

        base, atlas_fingerprint, _, _, _ = self._base_visual_atlas(
            tenant_id, scope_name
        )
        ready = bool(
            stored
            and stored.get("source_fingerprint") == atlas_fingerprint
            and isinstance(projection, Mapping)
            and projection.get("projection_state") == "ready"
        )
        return (dict(projection) if ready else base), atlas_fingerprint, ready

    @staticmethod
    def _public_personal_knowledge(
        projection: Mapping[str, Any],
        *,
        refresh: Mapping[str, Any] | None,
        agent_enabled: bool,
        stale: bool,
    ) -> dict[str, Any]:
        result = dict(projection)
        result["refresh"] = refresh
        result["agent_enabled"] = agent_enabled
        result["stale"] = stale
        result.pop("agent_batches", None)
        result.pop("agent_call", None)
        return result

    def personal_knowledge_base(
        self, tenant_id: str, scope_name: str
    ) -> dict[str, Any]:
        visual_stored = self.store.get_view(
            tenant_id, scope_name, VISUAL_ATLAS_KEY
        )
        if not self._valid_projection(
            visual_stored, VISUAL_ATLAS_SCHEMA_VERSION, require_full=True
        ):
            stored = self.store.get_view(
                tenant_id, scope_name, PERSONAL_KNOWLEDGE_KEY
            )
            if self._valid_projection(
                stored, PERSONAL_KNOWLEDGE_SCHEMA_VERSION, require_full=True
            ):
                return self._public_personal_knowledge(
                    dict(stored["projection"]),
                    refresh=self._public_refresh_state(
                        self.store.refresh_state(
                            tenant_id, scope_name, PERSONAL_KNOWLEDGE_KEY
                        )
                    ),
                    agent_enabled=self.agent is not None,
                    stale=True,
                )
            if self.agent is not None:
                self.ensure_projection_build(tenant_id, scope_name)
            raise SessionGraphError(
                "projection_build_pending",
                "the personal knowledge base is being organized in the background",
                status_code=409,
            )
        atlas, atlas_fingerprint, visual_ready = self._personal_knowledge_atlas(
            tenant_id, scope_name
        )
        desired_fingerprint = personal_knowledge_source_fingerprint(atlas)
        stored = self.store.get_view(tenant_id, scope_name, PERSONAL_KNOWLEDGE_KEY)
        current = bool(
            stored
            and stored.get("source_fingerprint") == desired_fingerprint
            and isinstance(stored.get("projection"), Mapping)
            and stored["projection"].get("projection_state") == "ready"
        )
        if current:
            projection = dict(stored["projection"])
        elif stored and isinstance(stored.get("projection"), Mapping) and stored["projection"].get("projection_state") == "ready":
            projection = dict(stored["projection"])
        else:
            projection = build_personal_knowledge_fallback(atlas)
            self.store.put_view(
                tenant_id,
                scope_name,
                PERSONAL_KNOWLEDGE_KEY,
                projection,
                source_snapshot_id=_text(atlas.get("snapshot_id")) or None,
                source_fingerprint=desired_fingerprint,
                generator="deterministic-knowledge-catalog",
            )
        if self.agent is not None and not current:
            if not visual_ready:
                visual_state = self.store.refresh_state(
                    tenant_id, scope_name, VISUAL_ATLAS_KEY
                )
                if not visual_state or _text(visual_state.get("state"), 32) not in {
                    "dirty",
                    "running",
                }:
                    self.store.enqueue(
                        tenant_id,
                        scope_name,
                        VISUAL_ATLAS_KEY,
                        source_fingerprint=atlas_fingerprint,
                    )
            self.store.enqueue(
                tenant_id,
                scope_name,
                PERSONAL_KNOWLEDGE_KEY,
                source_fingerprint=desired_fingerprint,
                delay_seconds=1.0,
            )
        refresh_state = self.store.refresh_state(
            tenant_id, scope_name, PERSONAL_KNOWLEDGE_KEY
        )
        refresh_pending = bool(
            refresh_state
            and _text(refresh_state.get("state"), 32)
            in {"dirty", "running", "failed"}
        )
        return self._public_personal_knowledge(
            projection,
            refresh=self._public_refresh_state(refresh_state),
            agent_enabled=self.agent is not None,
            stale=not current or refresh_pending,
        )

    def request_personal_knowledge_refresh(
        self, tenant_id: str, scope_name: str
    ) -> dict[str, Any]:
        if self.agent is None:
            raise SessionGraphError(
                "session_graph_agent_disabled",
                "the Personal Knowledge Agent is not configured",
                status_code=503,
            )
        sessions = self.store.sessions(tenant_id, scope_name)
        self.ensure_projection_build(tenant_id, scope_name, retry_failed=True)
        visual_fingerprint = MANUAL_VISUAL_REFRESH_PREFIX + self._build_seed(
            scope_name, sessions, projection_key=VISUAL_ATLAS_KEY
        )
        self.store.enqueue(
            tenant_id,
            scope_name,
            VISUAL_ATLAS_KEY,
            source_fingerprint=visual_fingerprint,
        )
        fingerprint = self._build_seed(
            scope_name, sessions, projection_key=PERSONAL_KNOWLEDGE_KEY
        )
        # A manual refresh is an explicit rebuild request even when the current
        # projection is already complete.  Upstream missing steps were queued
        # above; this task will yield until their new snapshots are ready.
        self.store.enqueue(
            tenant_id,
            scope_name,
            PERSONAL_KNOWLEDGE_KEY,
            source_fingerprint=fingerprint,
        )
        return {
            "accepted": True,
            "projection_key": PERSONAL_KNOWLEDGE_KEY,
            "source_fingerprint": fingerprint,
        }

    def request_refresh(
        self, tenant_id: str, scope_name: str, session_id: str | None = None
    ) -> dict[str, Any]:
        if self.agent is None:
            raise SessionGraphError(
                "session_graph_agent_disabled",
                "the Session Graph Agent is not configured",
                status_code=503,
            )
        if session_id:
            key = session_projection_key(session_id)
            base, fingerprint = self._base_session_map(
                tenant_id, scope_name, session_id
            )
            self.store.put_view(
                tenant_id,
                scope_name,
                key,
                base,
                source_snapshot_id=_text(base.get("snapshot_id")) or None,
                source_fingerprint=fingerprint,
                generator="deterministic-evidence-projection",
            )
        else:
            key = ATLAS_KEY
            _, fingerprint = self._base_atlas(tenant_id, scope_name)
        self.store.enqueue(
            tenant_id,
            scope_name,
            key,
            source_fingerprint=fingerprint,
        )
        return {"accepted": True, "projection_key": key, "source_fingerprint": fingerprint}

    def request_visual_atlas_refresh(
        self, tenant_id: str, scope_name: str
    ) -> dict[str, Any]:
        if self.agent is None:
            raise SessionGraphError(
                "session_graph_agent_disabled",
                "the Visual Atlas Agent is not configured",
                status_code=503,
            )
        sessions = self.store.sessions(tenant_id, scope_name)
        self.ensure_projection_build(tenant_id, scope_name, retry_failed=True)
        fingerprint = self._build_seed(
            scope_name, sessions, projection_key=VISUAL_ATLAS_KEY
        )
        fingerprint = MANUAL_VISUAL_REFRESH_PREFIX + fingerprint
        self.store.enqueue(
            tenant_id,
            scope_name,
            VISUAL_ATLAS_KEY,
            source_fingerprint=fingerprint,
        )
        return {
            "accepted": True,
            "projection_key": VISUAL_ATLAS_KEY,
            "source_fingerprint": fingerprint,
        }

    def _refresh_task(
        self,
        task: Mapping[str, Any],
        *,
        task_agent: LocalSessionGraphAgent | Any | None = None,
    ) -> None:
        if self.agent is None:
            raise SessionGraphError("session_graph_agent_disabled", "Session Graph Agent is disabled")
        if task_agent is None:
            if isinstance(self.agent, SessionGraphAgentRouter):
                task_agent = self.agent.select_agent()
            else:
                task_agent = self.agent
        if task_agent is None:
            raise SessionGraphError(
                "session_graph_agent_unavailable",
                "no projection provider currently has capacity",
            )
        tenant_id = str(task["tenant_id"])
        scope_name = str(task["scope_name"])
        key = str(task["projection_key"])
        if key.startswith(SESSION_KEY_PREFIX):
            session_id = key[len(SESSION_KEY_PREFIX) :]
            base, fingerprint = self._base_session_map(
                tenant_id, scope_name, session_id
            )
            if base.get("snapshot_state") != "committed" or base.get("provisional"):
                attempts = max(1, int(task.get("attempts") or 1))
                retry_delay = min(
                    1800.0,
                    max(30.0, 15.0 * (2 ** min(attempts - 1, 7))),
                )
                self.store.defer(
                    task,
                    seconds=retry_delay,
                    reason="Session Graph Agent waits for a committed index snapshot",
                )
                return
            patch, call = task_agent.session_map(base)
            self._journal_agent_call(
                tenant_id,
                scope_name,
                key,
                "session_map",
                call,
                default_model=task_agent.model,
            )
            try:
                result = apply_session_map_patch(base, patch)
            except SessionGraphError as exc:
                if exc.code != "session_graph_agent_invalid_patch":
                    raise
                repaired, repair_call = task_agent.repair_session_map(
                    base,
                    patch,
                    validation_error={"code": exc.code, "message": str(exc)},
                )
                self._journal_agent_call(
                    tenant_id,
                    scope_name,
                    key,
                    "session_map_repair",
                    repair_call,
                    default_model=task_agent.model,
                )
                result = apply_session_map_patch(base, repaired)
                call = {
                    "repair_attempted": True,
                    "validation_error": {"code": exc.code, "message": str(exc)},
                    "initial": call,
                    "repair": repair_call,
                }
            result["model"] = task_agent.model
            result["agent_call"] = call
            result["agent_checkpoint"] = {
                "message_count": int(base.get("message_count") or 0)
            }
            stored = self.store.put_view(
                tenant_id,
                scope_name,
                key,
                result,
                source_snapshot_id=_text(base.get("snapshot_id")) or None,
                source_fingerprint=fingerprint,
                generator="local-session-map-agent",
                model=task_agent.model,
                prompt_version=SESSION_GRAPH_PROMPT_VERSION,
                mark_clean=True,
                expected_queue_fingerprint=str(task["source_fingerprint"]),
                expected_queue_attempts=int(task["attempts"]),
            )
            if not stored:
                return
            self.store.enqueue(
                tenant_id,
                scope_name,
                ATLAS_KEY,
                source_fingerprint=fingerprint,
                delay_seconds=1.0,
            )
            return
        if key == PERSONAL_KNOWLEDGE_KEY:
            if self.store.scope_has_pending_sessions(tenant_id, scope_name):
                self.store.defer(
                    task,
                    seconds=5.0,
                    reason="Personal Knowledge waits for Session maps",
                )
                return
            atlas, atlas_fingerprint, visual_ready = self._personal_knowledge_atlas(
                tenant_id, scope_name
            )
            if not visual_ready:
                visual_state = self.store.refresh_state(
                    tenant_id, scope_name, VISUAL_ATLAS_KEY
                )
                if not visual_state or _text(visual_state.get("state"), 32) not in {
                    "dirty",
                    "running",
                }:
                    self.store.enqueue(
                        tenant_id,
                        scope_name,
                        VISUAL_ATLAS_KEY,
                        source_fingerprint=atlas_fingerprint,
                    )
                self.store.defer(
                    task,
                    seconds=5.0,
                    reason="Personal Knowledge waits for a ready Visual Atlas",
                )
                return
            fingerprint = personal_knowledge_source_fingerprint(atlas)
            if str(task["source_fingerprint"]) != fingerprint:
                self.store.requeue_superseded(
                    task,
                    source_fingerprint=fingerprint,
                )
                return
            batches = build_personal_knowledge_batches(atlas)
            if not self.store.heartbeat(
                task,
                stage="knowledge_batches",
                completed=0,
                total=len(batches),
            ):
                return
            previous = self.store.get_view(
                tenant_id, scope_name, PERSONAL_KNOWLEDGE_KEY
            )
            if not self._valid_projection(
                previous, PERSONAL_KNOWLEDGE_SCHEMA_VERSION, require_full=True
            ):
                fallback = build_personal_knowledge_fallback(atlas)
                if not self.store.put_view(
                    tenant_id,
                    scope_name,
                    PERSONAL_KNOWLEDGE_KEY,
                    fallback,
                    source_snapshot_id=_text(atlas.get("snapshot_id")) or None,
                    source_fingerprint=fingerprint,
                    generator="deterministic-knowledge-catalog",
                    expected_queue_fingerprint=str(task["source_fingerprint"]),
                    expected_queue_attempts=int(task["attempts"]),
                ):
                    return
                previous = self.store.get_view(
                    tenant_id, scope_name, PERSONAL_KNOWLEDGE_KEY
                )
            previous_projection = (previous or {}).get("projection")
            previous_batches = (
                previous_projection.get("agent_batches")
                if isinstance(previous_projection, Mapping)
                and isinstance(previous_projection.get("agent_batches"), Mapping)
                else {}
            )
            previous_model = _text((previous or {}).get("model"), 160)
            results: dict[str, dict[str, Any]] = {}
            calls: list[dict[str, Any]] = []
            pending: list[dict[str, Any]] = []
            checkpoint_keys: set[str] = set()
            for batch in batches:
                batch_id = _text(batch.get("batch_id"), 512)
                checkpoint_key = PERSONAL_KNOWLEDGE_BATCH_CHECKPOINT_PREFIX + batch_id
                checkpoint_keys.add(checkpoint_key)
                checkpoint_fingerprint = _fingerprint(
                    {
                        "schema": PERSONAL_KNOWLEDGE_PROMPT_VERSION,
                        "model": task_agent.model,
                        "batch": batch,
                    }
                )
                prior = previous_batches.get(batch_id)
                if (
                    isinstance(prior, Mapping)
                    and prior.get("source_fingerprint")
                    == batch.get("source_fingerprint")
                    and previous_model == task_agent.model
                ):
                    results[batch_id] = dict(prior)
                    calls.append(
                        {
                            "batch_id": batch_id,
                            "domain_id": batch.get("domain_id"),
                            "source_fingerprint": batch.get("source_fingerprint"),
                            "reused": True,
                            "checkpoint": "published_projection",
                        }
                    )
                    continue
                checkpoint = self.store.get_view(
                    tenant_id, scope_name, checkpoint_key
                )
                checkpoint_projection = (checkpoint or {}).get("projection")
                checkpoint_result = (
                    checkpoint_projection.get("result")
                    if isinstance(checkpoint_projection, Mapping)
                    else None
                )
                if (
                    checkpoint
                    and checkpoint.get("source_fingerprint")
                    == checkpoint_fingerprint
                    and checkpoint.get("model") == task_agent.model
                    and checkpoint.get("prompt_version")
                    == PERSONAL_KNOWLEDGE_PROMPT_VERSION
                    and isinstance(checkpoint_result, Mapping)
                ):
                    try:
                        normalized_checkpoint = validate_personal_knowledge_batch(
                            batch, checkpoint_result
                        )
                    except PersonalKnowledgeError:
                        normalized_checkpoint = None
                    if normalized_checkpoint is not None:
                        results[batch_id] = normalized_checkpoint
                        calls.append(
                            {
                                "batch_id": batch_id,
                                "domain_id": batch.get("domain_id"),
                                "source_fingerprint": batch.get(
                                    "source_fingerprint"
                                ),
                                "reused": True,
                                "checkpoint": "durable_batch",
                            }
                        )
                        continue
                pending.append(batch)

            def generate(
                batch: Mapping[str, Any], slot_id: int
            ) -> tuple[
                dict[str, Any],
                dict[str, Any],
                list[tuple[str, dict[str, Any]]],
            ]:
                if isinstance(task_agent, LocalSessionGraphAgent):
                    raw, call = task_agent.personal_knowledge_batch(
                        batch, slot_id=slot_id
                    )
                else:
                    raw, call = task_agent.personal_knowledge_batch(batch)
                journal_calls = [("personal_knowledge_batch", call)]
                try:
                    normalized = validate_personal_knowledge_batch(batch, raw)
                except PersonalKnowledgeError as exc:
                    repair_kwargs = {
                        "validation_error": {
                            "code": exc.code,
                            "message": str(exc),
                        }
                    }
                    if isinstance(task_agent, LocalSessionGraphAgent):
                        repaired, repair_call = (
                            task_agent.repair_personal_knowledge_batch(
                                batch,
                                raw,
                                slot_id=slot_id,
                                **repair_kwargs,
                            )
                        )
                    else:
                        repaired, repair_call = (
                            task_agent.repair_personal_knowledge_batch(
                                batch,
                                raw,
                                **repair_kwargs,
                            )
                        )
                    journal_calls.append(
                        ("personal_knowledge_batch_repair", repair_call)
                    )
                    sanitizer_applied = False
                    try:
                        normalized = validate_personal_knowledge_batch(batch, repaired)
                    except PersonalKnowledgeError as repair_exc:
                        if repair_exc.code not in {
                            "personal_knowledge_claim_invalid",
                            "personal_knowledge_section_invalid",
                            "personal_knowledge_excluded_invalid",
                        }:
                            raise
                        sanitized = sanitize_personal_knowledge_grounding(
                            batch, repaired
                        )
                        normalized = validate_personal_knowledge_batch(
                            batch, sanitized
                        )
                        sanitizer_applied = True
                    call = {
                        "repair_attempted": True,
                        "grounding_sanitizer_applied": sanitizer_applied,
                        "validation_error": {"code": exc.code, "message": str(exc)},
                        "initial": call,
                        "repair": repair_call,
                    }
                return normalized, call, journal_calls

            effective_workers = 0
            if pending:
                with ThreadPoolExecutor(
                    max_workers=min(2, self.knowledge_workers, len(pending)),
                    thread_name_prefix="tmcra-knowledge-domain",
                ) as executor:
                    cursor = 0
                    futures: dict[Any, tuple[Mapping[str, Any], int]] = {}

                    def submit_next(slot_id: int) -> bool:
                        nonlocal cursor, effective_workers
                        if cursor >= len(pending):
                            return False
                        batch = pending[cursor]
                        cursor += 1
                        futures[executor.submit(generate, batch, slot_id)] = (
                            batch,
                            slot_id,
                        )
                        effective_workers = max(effective_workers, len(futures))
                        return True

                    submit_next(LOCAL_QWEN_GRAPH_SLOT_ID)
                    if (
                        self.knowledge_workers > 1
                        and cursor < len(pending)
                        and self.idle_borrow_enabled
                        and isinstance(task_agent, LocalSessionGraphAgent)
                    ):
                        first_future = next(iter(futures))
                        scheduler = task_agent.gpu_scheduler
                        deadline = time.monotonic() + 5.0
                        while (
                            scheduler is not None
                            and not first_future.done()
                            and time.monotonic() < deadline
                        ):
                            status = scheduler.status()
                            if (
                                status.get("active", {}).get(
                                    GpuWorkload.GRAPH_BACKGROUND.value, 0
                                )
                                > 0
                            ):
                                break
                            time.sleep(0.01)
                        if (
                            not first_future.done()
                            and self._borrowed_planner_slot_available(task_agent)
                        ):
                            submit_next(LOCAL_QWEN_PLANNER_SLOT_ID)

                    while futures:
                        completed, _ = wait(
                            tuple(futures), return_when=FIRST_COMPLETED
                        )
                        for future in completed:
                            batch, slot_id = futures.pop(future)
                            normalized, call, journal_calls = future.result()
                            for operation, metadata in journal_calls:
                                self._journal_agent_call(
                                    tenant_id,
                                    scope_name,
                                    PERSONAL_KNOWLEDGE_KEY,
                                    operation,
                                    metadata,
                                    default_model=task_agent.model,
                                )
                            batch_id = _text(batch.get("batch_id"), 512)
                            results[batch_id] = normalized
                            checkpoint_key = (
                                PERSONAL_KNOWLEDGE_BATCH_CHECKPOINT_PREFIX + batch_id
                            )
                            checkpoint_fingerprint = _fingerprint(
                                {
                                    "schema": PERSONAL_KNOWLEDGE_PROMPT_VERSION,
                                    "model": task_agent.model,
                                    "batch": batch,
                                }
                            )
                            self.store.put_view(
                                tenant_id,
                                scope_name,
                                checkpoint_key,
                                {
                                    "schema_version": (
                                        PERSONAL_KNOWLEDGE_BATCH_CHECKPOINT_SCHEMA_VERSION
                                    ),
                                    "batch_id": batch_id,
                                    "result": normalized,
                                },
                                source_snapshot_id=_text(
                                    atlas.get("snapshot_id")
                                )
                                or None,
                                source_fingerprint=checkpoint_fingerprint,
                                generator="local-personal-knowledge-checkpoint",
                                model=task_agent.model,
                                prompt_version=PERSONAL_KNOWLEDGE_PROMPT_VERSION,
                            )
                            calls.append(
                                {
                                    "batch_id": batch_id,
                                    "domain_id": batch.get("domain_id"),
                                    "source_fingerprint": batch.get(
                                        "source_fingerprint"
                                    ),
                                    "reused": False,
                                    "call": call,
                                }
                            )
                            if not self.store.heartbeat(
                                task,
                                stage="knowledge_batches",
                                completed=len(results),
                                total=len(batches),
                            ):
                                return
                            if slot_id == LOCAL_QWEN_GRAPH_SLOT_ID:
                                submit_next(LOCAL_QWEN_GRAPH_SLOT_ID)
                            elif self._borrowed_planner_slot_available(task_agent):
                                submit_next(LOCAL_QWEN_PLANNER_SLOT_ID)

                        borrowed_active = any(
                            active_slot == LOCAL_QWEN_PLANNER_SLOT_ID
                            for _batch, active_slot in futures.values()
                        )
                        if (
                            not borrowed_active
                            and cursor < len(pending)
                            and self.knowledge_workers > 1
                            and self._borrowed_planner_slot_available(task_agent)
                        ):
                            submit_next(LOCAL_QWEN_PLANNER_SLOT_ID)
            if not self.store.heartbeat(
                task,
                stage="knowledge_merge",
                completed=len(results),
                total=len(batches),
            ):
                return
            merged = merge_personal_knowledge_batches(
                atlas,
                batches,
                [results[_text(batch.get("batch_id"), 512)] for batch in batches],
                model=task_agent.model,
                agent_call={
                    "strategy": "complete-domain-batches",
                    "worker_count": effective_workers,
                    "batch_count": len(batches),
                    "generated_batch_count": len(pending),
                    "reused_batch_count": len(batches) - len(pending),
                    "calls": calls,
                },
            )
            stored = self.store.put_view(
                tenant_id,
                scope_name,
                PERSONAL_KNOWLEDGE_KEY,
                merged,
                source_snapshot_id=_text(atlas.get("snapshot_id")) or None,
                source_fingerprint=fingerprint,
                generator="local-personal-knowledge-agent",
                model=task_agent.model,
                prompt_version=PERSONAL_KNOWLEDGE_PROMPT_VERSION,
                mark_clean=True,
                expected_queue_fingerprint=str(task["source_fingerprint"]),
                expected_queue_attempts=int(task["attempts"]),
            )
            if stored:
                self.store.delete_views_by_prefix_except(
                    tenant_id,
                    scope_name,
                    PERSONAL_KNOWLEDGE_BATCH_CHECKPOINT_PREFIX,
                    sorted(checkpoint_keys),
                )
            return
        if key == VISUAL_ATLAS_KEY:
            if self.store.scope_has_pending_sessions(tenant_id, scope_name):
                self.store.defer(
                    task,
                    seconds=5.0,
                    reason="Session maps are still refreshing",
                )
                return
            atlas_state = self.store.refresh_state(tenant_id, scope_name, ATLAS_KEY)
            if atlas_state and _text(atlas_state.get("state"), 32) in {"dirty", "running"}:
                self.store.defer(
                    task,
                    seconds=5.0,
                    reason="Session Atlas taxonomy is still refreshing",
                )
                return
            task_fingerprint = str(task["source_fingerprint"])
            run_checkpoint = self.store.get_view(
                tenant_id, scope_name, VISUAL_ATLAS_RUN_CHECKPOINT_KEY
            )
            run_projection = (run_checkpoint or {}).get("projection")
            frozen_base = (
                run_projection.get("base")
                if isinstance(run_projection, Mapping)
                else None
            )
            reuse_run = bool(
                run_checkpoint
                and isinstance(run_projection, Mapping)
                and run_projection.get("schema_version")
                == VISUAL_ATLAS_RUN_CHECKPOINT_SCHEMA_VERSION
                and run_projection.get("queue_source_fingerprint")
                == task_fingerprint
                and run_checkpoint.get("model") == task_agent.model
                and run_checkpoint.get("prompt_version") == VISUAL_ATLAS_PROMPT_VERSION
                and isinstance(frozen_base, Mapping)
            )
            if reuse_run:
                try:
                    base = dict(frozen_base)
                    # Full validation and deterministic batch construction prove
                    # that the durable input snapshot is still readable.
                    build_visual_atlas_episode_batches(base)
                    fingerprint = _text(
                        run_projection.get("source_fingerprint"), 256
                    )
                    session_ids = [
                        _text(value, 512)
                        for value in run_projection.get("session_ids", [])
                        if _text(value, 512)
                    ]
                    if not fingerprint or not session_ids:
                        reuse_run = False
                except VisualAtlasError:
                    reuse_run = False
            if reuse_run:
                taxonomy_call = {
                    "reused": True,
                    "checkpoint": "durable_run_snapshot",
                }
            else:
                base, fingerprint, sessions, views, source_graphs = (
                    self._base_visual_atlas(tenant_id, scope_name)
                )
                published_visual = self.store.get_view(
                    tenant_id, scope_name, VISUAL_ATLAS_KEY
                )
                if not self._ready_projection(
                    published_visual,
                    VISUAL_ATLAS_SCHEMA_VERSION,
                    require_full=True,
                ):
                    if not self.store.put_view(
                        tenant_id,
                        scope_name,
                        VISUAL_ATLAS_KEY,
                        base,
                        source_snapshot_id=_text(base.get("snapshot_id")) or None,
                        source_fingerprint=fingerprint,
                        generator="deterministic-visual-atlas-fallback",
                        expected_queue_fingerprint=task_fingerprint,
                        expected_queue_attempts=int(task["attempts"]),
                    ):
                        return
                taxonomy_payload = build_visual_atlas_taxonomy_payload(sessions, views)
                taxonomy_fingerprint = _fingerprint(
                    {
                        "schema": VISUAL_ATLAS_TAXONOMY_PROMPT_VERSION,
                        "model": task_agent.model,
                        "payload": taxonomy_payload,
                    }
                )
                if not self.store.heartbeat(
                    task,
                    stage="visual_taxonomy",
                    completed=0,
                    total=1,
                ):
                    return
                normalized_taxonomy: dict[str, Any] | None = None
                taxonomy_call: dict[str, Any]
                taxonomy_checkpoint = self.store.get_view(
                    tenant_id, scope_name, VISUAL_ATLAS_TAXONOMY_CHECKPOINT_KEY
                )
                taxonomy_checkpoint_projection = (taxonomy_checkpoint or {}).get(
                    "projection"
                )
                checkpoint_taxonomy = (
                    taxonomy_checkpoint_projection.get("taxonomy")
                    if isinstance(taxonomy_checkpoint_projection, Mapping)
                    else None
                )
                if (
                    taxonomy_checkpoint
                    and taxonomy_checkpoint.get("source_fingerprint")
                    == taxonomy_fingerprint
                    and taxonomy_checkpoint.get("model") == task_agent.model
                    and taxonomy_checkpoint.get("prompt_version")
                    == VISUAL_ATLAS_TAXONOMY_PROMPT_VERSION
                    and isinstance(checkpoint_taxonomy, Mapping)
                ):
                    try:
                        normalized_taxonomy = validate_visual_atlas_taxonomy(
                            sessions, checkpoint_taxonomy
                        )
                    except VisualAtlasError:
                        normalized_taxonomy = None
                if normalized_taxonomy is not None:
                    taxonomy_call = {
                        "reused": True,
                        "checkpoint": "durable_taxonomy",
                    }
                else:
                    taxonomy, taxonomy_call = task_agent.visual_taxonomy(sessions, views)
                    self._journal_agent_call(
                        tenant_id,
                        scope_name,
                        VISUAL_ATLAS_KEY,
                        "visual_atlas_taxonomy",
                        taxonomy_call,
                        default_model=task_agent.model,
                    )
                    try:
                        normalized_taxonomy = validate_visual_atlas_taxonomy(
                            sessions, taxonomy
                        )
                    except VisualAtlasError as exc:
                        repaired, repair_call = task_agent.repair_visual_taxonomy(
                            sessions,
                            views,
                            taxonomy,
                            validation_error={"code": exc.code, "message": str(exc)},
                        )
                        self._journal_agent_call(
                            tenant_id,
                            scope_name,
                            VISUAL_ATLAS_KEY,
                            "visual_atlas_taxonomy_repair",
                            repair_call,
                            default_model=task_agent.model,
                        )
                        normalized_taxonomy = validate_visual_atlas_taxonomy(
                            sessions, repaired
                        )
                        taxonomy_call = {
                            "repair_attempted": True,
                            "validation_error": {
                                "code": exc.code,
                                "message": str(exc),
                            },
                            "initial": taxonomy_call,
                            "repair": repair_call,
                        }
                    self.store.put_view(
                        tenant_id,
                        scope_name,
                        VISUAL_ATLAS_TAXONOMY_CHECKPOINT_KEY,
                        {
                            "schema_version": (
                                VISUAL_ATLAS_TAXONOMY_CHECKPOINT_SCHEMA_VERSION
                            ),
                            "taxonomy": normalized_taxonomy,
                        },
                        source_snapshot_id=_text(base.get("snapshot_id")) or None,
                        source_fingerprint=taxonomy_fingerprint,
                        generator="local-visual-atlas-taxonomy-checkpoint",
                        model=task_agent.model,
                        prompt_version=VISUAL_ATLAS_TAXONOMY_PROMPT_VERSION,
                    )
                if not self.store.heartbeat(
                    task,
                    stage="visual_taxonomy",
                    completed=1,
                    total=1,
                ):
                    return
                classified_sessions = apply_visual_atlas_taxonomy(
                    sessions, normalized_taxonomy
                )
                base = build_visual_atlas(
                    scope_name, classified_sessions, views, source_graphs
                )
                session_ids = sorted(
                    _text(item.get("session_id"), 512) for item in sessions
                )
                run_source_fingerprint = _fingerprint(
                    {
                        "schema": VISUAL_ATLAS_RUN_CHECKPOINT_SCHEMA_VERSION,
                        "queue_source_fingerprint": task_fingerprint,
                        "source_fingerprint": fingerprint,
                        "model": task_agent.model,
                    }
                )
                self.store.put_view(
                    tenant_id,
                    scope_name,
                    VISUAL_ATLAS_RUN_CHECKPOINT_KEY,
                    {
                        "schema_version": VISUAL_ATLAS_RUN_CHECKPOINT_SCHEMA_VERSION,
                        "queue_source_fingerprint": task_fingerprint,
                        "source_fingerprint": fingerprint,
                        "session_ids": session_ids,
                        "base": base,
                    },
                    source_snapshot_id=_text(base.get("snapshot_id")) or None,
                    source_fingerprint=run_source_fingerprint,
                    generator="local-visual-atlas-run-checkpoint",
                    model=task_agent.model,
                    prompt_version=VISUAL_ATLAS_PROMPT_VERSION,
                )
            batches = build_visual_atlas_episode_batches(base)
            (
                validated_batch_base,
                batch_nodes,
                batch_descendants,
                batch_existing_edges,
            ) = prepare_visual_atlas_patch_validation(base)
            patch_relation_limit = VISUAL_ATLAS_MAX_RELATIONS_PER_PATCH
            if not self.store.heartbeat(
                task,
                stage="visual_batches",
                completed=0,
                total=len(batches),
            ):
                return
            patches: dict[str, dict[str, Any]] = {}
            batch_calls: list[dict[str, Any]] = []
            checkpoint_keys: set[str] = set()
            batch_method = getattr(task_agent, "visual_atlas_batch", None)
            if not callable(batch_method):
                # Compatibility for externally supplied test/integration agents.
                patch, legacy_call = task_agent.visual_atlas_batches(base)
                self._journal_agent_call(
                    tenant_id,
                    scope_name,
                    VISUAL_ATLAS_KEY,
                    "visual_atlas_episode_batches",
                    legacy_call,
                    default_model=task_agent.model,
                )
                atlas_call = {
                    "strategy": "legacy-complete-atlas",
                    "batch_count": len(batches),
                    "generated_batch_count": len(batches),
                    "reused_batch_count": 0,
                    "call": legacy_call,
                }
            else:
                pending_batches: list[
                    tuple[int, Mapping[str, Any], str, str, str]
                ] = []
                for batch_index, batch in enumerate(batches):
                    batch_id = _text(batch.get("batch_id"), 512)
                    batch_fingerprint = visual_atlas_batch_checkpoint_fingerprint(
                        batch, model=task_agent.model
                    )
                    checkpoint_key = (
                        VISUAL_ATLAS_BATCH_CHECKPOINT_PREFIX
                        + "evidence."
                        + batch_fingerprint
                    )
                    checkpoint_keys.add(checkpoint_key)
                    normalized_patch: dict[str, Any] | None = None
                    checkpoint = self.store.get_view(
                        tenant_id, scope_name, checkpoint_key
                    )
                    checkpoint_projection = (checkpoint or {}).get("projection")
                    checkpoint_patch = (
                        checkpoint_projection.get("patch")
                        if isinstance(checkpoint_projection, Mapping)
                        else None
                    )
                    if (
                        checkpoint
                        and checkpoint.get("source_fingerprint")
                        == batch_fingerprint
                        and checkpoint.get("model") == task_agent.model
                        and checkpoint.get("prompt_version")
                        == VISUAL_ATLAS_EPISODE_BATCH_PROMPT_VERSION
                        and isinstance(checkpoint_patch, Mapping)
                    ):
                        try:
                            normalized_patch = (
                                validate_visual_atlas_episode_batch_patch(
                                    base,
                                    batch,
                                    checkpoint_patch,
                                    _validated_base=validated_batch_base,
                                    _nodes=batch_nodes,
                                    _descendants=batch_descendants,
                                    _existing_edges=batch_existing_edges,
                                )
                            )
                        except VisualAtlasError:
                            normalized_patch = None
                    if normalized_patch is not None:
                        batch_call = {
                            "batch_id": batch_id,
                            "batch_index": batch_index,
                            "reused": True,
                            "checkpoint": "durable_batch",
                        }
                        patches[batch_id] = normalized_patch
                        batch_calls.append(batch_call)
                        if not self.store.heartbeat(
                            task,
                            stage="visual_batches",
                            completed=len(patches),
                            total=len(batches),
                        ):
                            return
                        continue
                    pending_batches.append(
                        (
                            batch_index,
                            batch,
                            batch_id,
                            checkpoint_key,
                            batch_fingerprint,
                        )
                    )

                def generate_visual_batch(
                    entry: tuple[int, Mapping[str, Any], str, str, str],
                    slot_id: int,
                ) -> tuple[dict[str, Any], dict[str, Any]]:
                    batch_index, batch, _batch_id, _checkpoint_key, _fingerprint = (
                        entry
                    )
                    if isinstance(task_agent, LocalSessionGraphAgent):
                        raw_patch, raw_call = batch_method(
                            base,
                            batch,
                            batch_index=batch_index,
                            slot_id=slot_id,
                        )
                    else:
                        raw_patch, raw_call = batch_method(
                            base, batch, batch_index=batch_index
                        )
                    return (
                        validate_visual_atlas_episode_batch_patch(
                            base,
                            batch,
                            raw_patch,
                            _validated_base=validated_batch_base,
                            _nodes=batch_nodes,
                            _descendants=batch_descendants,
                            _existing_edges=batch_existing_edges,
                        ),
                        raw_call,
                    )

                cursor = 0
                with ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="tmcra-visual-atlas-batch",
                ) as executor:
                    futures: dict[
                        Any,
                        tuple[
                            tuple[int, Mapping[str, Any], str, str, str],
                            int,
                        ],
                    ] = {}

                    def submit_next_visual(slot_id: int) -> bool:
                        nonlocal cursor
                        if cursor >= len(pending_batches):
                            return False
                        entry = pending_batches[cursor]
                        cursor += 1
                        futures[
                            executor.submit(
                                generate_visual_batch,
                                entry,
                                slot_id,
                            )
                        ] = (entry, slot_id)
                        return True

                    submit_next_visual(LOCAL_QWEN_GRAPH_SLOT_ID)
                    if (
                        cursor < len(pending_batches)
                        and self.idle_borrow_enabled
                        and isinstance(task_agent, LocalSessionGraphAgent)
                    ):
                        first_future = next(iter(futures))
                        scheduler = task_agent.gpu_scheduler
                        deadline = time.monotonic() + 5.0
                        while (
                            scheduler is not None
                            and not first_future.done()
                            and time.monotonic() < deadline
                        ):
                            status = scheduler.status()
                            if (
                                status.get("active", {}).get(
                                    GpuWorkload.GRAPH_BACKGROUND.value, 0
                                )
                                > 0
                            ):
                                break
                            time.sleep(0.01)
                        if (
                            not first_future.done()
                            and self._borrowed_planner_slot_available(task_agent)
                        ):
                            submit_next_visual(LOCAL_QWEN_PLANNER_SLOT_ID)

                    while futures:
                        completed, _ = wait(
                            tuple(futures), return_when=FIRST_COMPLETED
                        )
                        for future in completed:
                            entry, slot_id = futures.pop(future)
                            (
                                batch_index,
                                batch,
                                batch_id,
                                checkpoint_key,
                                batch_fingerprint,
                            ) = entry
                            normalized_patch, raw_batch_call = future.result()
                            self._journal_agent_call(
                                tenant_id,
                                scope_name,
                                VISUAL_ATLAS_KEY,
                                "visual_atlas_episode_batch",
                                raw_batch_call,
                                default_model=task_agent.model,
                            )
                            self.store.put_view(
                                tenant_id,
                                scope_name,
                                checkpoint_key,
                                {
                                    "schema_version": (
                                        VISUAL_ATLAS_BATCH_CHECKPOINT_SCHEMA_VERSION
                                    ),
                                    "batch_id": batch_id,
                                    "patch": normalized_patch,
                                },
                                source_snapshot_id=(
                                    _text(base.get("snapshot_id")) or None
                                ),
                                source_fingerprint=batch_fingerprint,
                                generator="local-visual-atlas-batch-checkpoint",
                                model=task_agent.model,
                                prompt_version=(
                                    VISUAL_ATLAS_EPISODE_BATCH_PROMPT_VERSION
                                ),
                            )
                            batch_call = {
                                "batch_id": batch_id,
                                "batch_index": batch_index,
                                "reused": False,
                                "rejected_relation_count": int(
                                    raw_batch_call.get("rejected_relation_count")
                                    or 0
                                )
                                if isinstance(raw_batch_call, Mapping)
                                else 0,
                            }
                            patches[batch_id] = normalized_patch
                            batch_calls.append(batch_call)
                            if not self.store.heartbeat(
                                task,
                                stage="visual_batches",
                                completed=len(patches),
                                total=len(batches),
                            ):
                                return
                            if slot_id == LOCAL_QWEN_GRAPH_SLOT_ID:
                                submit_next_visual(LOCAL_QWEN_GRAPH_SLOT_ID)
                            elif self._borrowed_planner_slot_available(task_agent):
                                submit_next_visual(LOCAL_QWEN_PLANNER_SLOT_ID)

                        borrowed_active = any(
                            active_slot == LOCAL_QWEN_PLANNER_SLOT_ID
                            for _entry, active_slot in futures.values()
                        )
                        if (
                            not borrowed_active
                            and cursor < len(pending_batches)
                            and self._borrowed_planner_slot_available(task_agent)
                        ):
                            submit_next_visual(LOCAL_QWEN_PLANNER_SLOT_ID)
                if not self.store.heartbeat(
                    task,
                    stage="visual_merge",
                    completed=len(patches),
                    total=len(batches),
                ):
                    return
                patch = merge_visual_atlas_episode_batch_patches(
                    base,
                    batches,
                    [patches[_text(batch.get("batch_id"), 512)] for batch in batches],
                )
                patch_relation_limit = max(
                    VISUAL_ATLAS_MAX_RELATIONS_PER_PATCH,
                    len(batches) * VISUAL_ATLAS_MAX_RELATIONS_PER_BATCH,
                )
                atlas_call = {
                    "strategy": "durable-domain-local-human-memory-batches",
                    "batch_count": len(batches),
                    "generated_batch_count": sum(
                        not bool(call.get("reused")) for call in batch_calls
                    ),
                    "reused_batch_count": sum(
                        bool(call.get("reused")) for call in batch_calls
                    ),
                    "calls": batch_calls,
                }
            result = apply_visual_atlas_patch(
                base,
                patch,
                model=task_agent.model,
                max_relations=patch_relation_limit,
            )
            result["agent_call"] = {
                "taxonomy": taxonomy_call,
                "atlas": atlas_call,
            }
            result["agent_checkpoint"] = {
                "session_ids": sorted(session_ids),
                "source_fingerprint": fingerprint,
                "message_count": sum(
                    max(0, int(_number(item.get("message_count"), 0)))
                    for item in _items(result.get("nodes"))
                    if _text(item.get("level"), 32) == "session"
                ),
            }
            stored = self.store.put_view(
                tenant_id,
                scope_name,
                VISUAL_ATLAS_KEY,
                result,
                source_snapshot_id=_text(result.get("snapshot_id")) or None,
                source_fingerprint=fingerprint,
                generator="local-visual-atlas-agent",
                model=task_agent.model,
                prompt_version=VISUAL_ATLAS_PROMPT_VERSION,
                mark_clean=True,
                expected_queue_fingerprint=str(task["source_fingerprint"]),
                expected_queue_attempts=int(task["attempts"]),
            )
            if stored:
                queued_state = self.store.refresh_state(
                    tenant_id, scope_name, VISUAL_ATLAS_KEY
                )
                if (
                    _text((queued_state or {}).get("state"), 32) == "dirty"
                    and not self._manual_visual_refresh_pending(queued_state)
                    and not self._visual_atlas_auto_refresh_due(
                        tenant_id, scope_name
                    )
                ):
                    self.store.cancel_dirty_refresh(
                        tenant_id,
                        scope_name,
                        VISUAL_ATLAS_KEY,
                        stage="waiting_for_message_waterline",
                    )
                self.store.delete_views_by_prefix_except(
                    tenant_id,
                    scope_name,
                    VISUAL_ATLAS_BATCH_CHECKPOINT_PREFIX,
                    sorted(checkpoint_keys),
                )
                self.store.delete_views_by_prefix_except(
                    tenant_id,
                    scope_name,
                    VISUAL_ATLAS_RUN_CHECKPOINT_KEY,
                    [],
                )
                final_state = self.store.refresh_state(
                    tenant_id, scope_name, VISUAL_ATLAS_KEY
                )
                final_current = _text((final_state or {}).get("state"), 32) == "clean"
                if final_current and self.store.scope_has_pending_sessions(
                    tenant_id, scope_name
                ):
                    final_current = False
                if final_current:
                    upstream_state = self.store.refresh_state(
                        tenant_id, scope_name, ATLAS_KEY
                    )
                    if upstream_state and _text(
                        upstream_state.get("state"), 32
                    ) in {"dirty", "running"}:
                        final_current = False
                    else:
                        try:
                            _, latest_fingerprint, _, _, _ = self._base_visual_atlas(
                                tenant_id, scope_name
                            )
                        except (GraphProjectionError, SessionGraphError):
                            latest_fingerprint = fingerprint
                        if (
                            latest_fingerprint != fingerprint
                            and self._visual_atlas_auto_refresh_due(
                                tenant_id, scope_name
                            )
                        ):
                            self.store.enqueue(
                                tenant_id,
                                scope_name,
                                VISUAL_ATLAS_KEY,
                                source_fingerprint=latest_fingerprint,
                                delay_seconds=1.0,
                            )
                            final_current = False
                if final_current:
                    self.store.enqueue(
                        tenant_id,
                        scope_name,
                        PERSONAL_KNOWLEDGE_KEY,
                        source_fingerprint=personal_knowledge_source_fingerprint(result),
                        delay_seconds=1.0,
                    )
            return
        if key != ATLAS_KEY:
            raise SessionGraphError("invalid_projection_key", "unknown Session Graph projection")
        if self.store.scope_has_pending_sessions(tenant_id, scope_name):
            self.store.defer(
                task,
                seconds=5.0,
                reason="Session maps are still refreshing",
            )
            return
        base, fingerprint = self._base_atlas(tenant_id, scope_name)
        patch, call = task_agent.atlas(base)
        self._journal_agent_call(
            tenant_id,
            scope_name,
            ATLAS_KEY,
            "session_atlas",
            call,
            default_model=task_agent.model,
        )
        try:
            result = apply_session_atlas_patch(base, patch)
        except SessionGraphError as exc:
            if exc.code != "session_atlas_agent_invalid_patch":
                raise
            repaired, repair_call = task_agent.repair_atlas(
                base,
                patch,
                validation_error={"code": exc.code, "message": str(exc)},
            )
            self._journal_agent_call(
                tenant_id,
                scope_name,
                ATLAS_KEY,
                "session_atlas_repair",
                repair_call,
                default_model=task_agent.model,
            )
            result = apply_session_atlas_patch(base, repaired)
            call = {
                "repair_attempted": True,
                "validation_error": {"code": exc.code, "message": str(exc)},
                "initial": call,
                "repair": repair_call,
            }
        result["model"] = task_agent.model
        result["agent_call"] = call
        result["agent_checkpoint"] = {
            "session_ids": sorted(
                {
                    _text(item.get("session_id"), 200)
                    for item in _items(base.get("nodes"))
                    if _text(item.get("session_id"), 200)
                }
            )
        }
        stored = self.store.put_view(
            tenant_id,
            scope_name,
            ATLAS_KEY,
            result,
            source_snapshot_id=_text(base.get("snapshot_id")) or None,
            source_fingerprint=fingerprint,
            generator="local-session-atlas-agent",
            model=task_agent.model,
            prompt_version=SESSION_GRAPH_PROMPT_VERSION,
            mark_clean=True,
            expected_queue_fingerprint=str(task["source_fingerprint"]),
            expected_queue_attempts=int(task["attempts"]),
        )
        if stored and self._visual_atlas_auto_refresh_due(
            tenant_id, scope_name
        ):
            self.store.enqueue(
                tenant_id,
                scope_name,
                VISUAL_ATLAS_KEY,
                source_fingerprint=fingerprint,
                delay_seconds=1.0,
            )
