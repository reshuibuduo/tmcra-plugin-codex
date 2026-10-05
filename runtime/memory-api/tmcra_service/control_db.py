"""SQLite-backed control-plane storage.

Every public write in the service modules uses this database's explicit
transaction helper.  SQLite WAL is enabled once and remains a property of the
database file, which permits concurrent readers while writers serialize.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


_GRAPH_AUDIT_FIELDS = frozenset(
    {"turn_log", "retrieval_log", "answer_support_log"}
)


class StaleSourceAccountingRecovery(RuntimeError):
    """A read-only recovery plan no longer owns the failed Writer attempt."""


class ControlDB:
    """Small connection-per-operation SQLite database wrapper."""

    def __init__(self, path: os.PathLike[str] | str, *, timeout: float = 10.0) -> None:
        self.path = os.fspath(path)
        self.timeout = timeout
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {max(1, int(self.timeout * 1000))}")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with closing(self.connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tenant_scopes (
                    tenant_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, scope)
                );

                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    secret_hash TEXT NOT NULL,
                    scopes_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    revoked_at REAL
                );
                CREATE INDEX IF NOT EXISTS api_keys_tenant_idx
                    ON api_keys (tenant_id, revoked_at);

                CREATE TABLE IF NOT EXISTS scope_tokens (
                    token_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    secret_hash TEXT NOT NULL,
                    permissions_json TEXT NOT NULL,
                    scope_names_json TEXT NOT NULL,
                    scope_prefixes_json TEXT NOT NULL DEFAULT '[]',
                    label TEXT NOT NULL,
                    subject TEXT,
                    created_by_key_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    revoked_at REAL,
                    last_used_at REAL
                );
                CREATE INDEX IF NOT EXISTS scope_tokens_tenant_idx
                    ON scope_tokens (tenant_id, revoked_at, expires_at);

                CREATE TABLE IF NOT EXISTS scope_token_issuances (
                    tenant_id TEXT NOT NULL,
                    created_by_key_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    token_id TEXT NOT NULL UNIQUE,
                    token_replay_hash TEXT NOT NULL,
                    final_expires_at REAL NOT NULL,
                    confirmed_at REAL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, created_by_key_id, idempotency_key),
                    FOREIGN KEY (token_id) REFERENCES scope_tokens(token_id)
                );
                CREATE INDEX IF NOT EXISTS scope_token_issuances_created_idx
                    ON scope_token_issuances (tenant_id, created_at);

                CREATE TABLE IF NOT EXISTS scope_catalog (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    last_ingest_at REAL,
                    last_recall_at REAL,
                    ingest_request_count INTEGER NOT NULL DEFAULT 0,
                    recall_request_count INTEGER NOT NULL DEFAULT 0,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (tenant_id, scope_name)
                );
                CREATE INDEX IF NOT EXISTS scope_catalog_seen_idx
                    ON scope_catalog (tenant_id, last_seen_at DESC, scope_name);

                CREATE TABLE IF NOT EXISTS scope_sessions (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_ingest_at REAL NOT NULL,
                    ingest_request_count INTEGER NOT NULL DEFAULT 0,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (tenant_id, scope_name, session_id)
                );
                CREATE INDEX IF NOT EXISTS scope_sessions_recent_idx
                    ON scope_sessions (tenant_id, scope_name, last_ingest_at DESC, session_id);

                CREATE TABLE IF NOT EXISTS session_graph_metadata (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    title TEXT,
                    source_app TEXT,
                    native_thread_id TEXT,
                    parent_session_id TEXT,
                    session_status TEXT NOT NULL DEFAULT 'active',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, scope_name, session_id)
                );
                CREATE INDEX IF NOT EXISTS session_graph_metadata_parent_idx
                    ON session_graph_metadata (
                        tenant_id, scope_name, parent_session_id, updated_at DESC
                    );

                CREATE TABLE IF NOT EXISTS memory_graph_views (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    projection_key TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    source_snapshot_id TEXT,
                    source_fingerprint TEXT NOT NULL,
                    generator TEXT NOT NULL,
                    model TEXT,
                    prompt_version TEXT,
                    projection_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, scope_name, projection_key)
                );
                CREATE INDEX IF NOT EXISTS memory_graph_views_updated_idx
                    ON memory_graph_views (
                        tenant_id, scope_name, updated_at DESC, projection_key
                    );

                CREATE TABLE IF NOT EXISTS memory_graph_refresh_queue (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    projection_key TEXT NOT NULL,
                    state TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    pending_source_fingerprint TEXT,
                    due_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    claimed_at REAL,
                    heartbeat_at REAL,
                    progress_stage TEXT,
                    progress_completed INTEGER,
                    progress_total INTEGER,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, scope_name, projection_key),
                    CHECK (state IN ('dirty', 'running', 'clean', 'failed'))
                );
                CREATE INDEX IF NOT EXISTS memory_graph_refresh_ready_idx
                    ON memory_graph_refresh_queue (state, due_at, updated_at);

                CREATE TABLE IF NOT EXISTS scope_ingest_events (
                    tenant_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    raw_token_count INTEGER NOT NULL,
                    client_platform TEXT NOT NULL DEFAULT 'unattributed',
                    integration_id TEXT,
                    agent_id TEXT,
                    attribution_source TEXT NOT NULL DEFAULT 'unattributed',
                    created_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, idempotency_key)
                );

                CREATE TABLE IF NOT EXISTS usage_entitlements (
                    tenant_id TEXT NOT NULL,
                    principal TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    limit_units INTEGER,
                    updated_by_key_id TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, principal, metric),
                    CHECK (metric IN ('ingest_raw_tokens', 'recall_requests')),
                    CHECK (limit_units IS NULL OR limit_units >= 0)
                );

                CREATE TABLE IF NOT EXISTS usage_totals (
                    tenant_id TEXT NOT NULL,
                    principal TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    used_units INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, principal, metric),
                    CHECK (metric IN ('ingest_raw_tokens', 'recall_requests')),
                    CHECK (used_units >= 0)
                );

                CREATE TABLE IF NOT EXISTS usage_events (
                    tenant_id TEXT NOT NULL,
                    principal TEXT NOT NULL,
                    consumer_principal TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    units INTEGER NOT NULL,
                    scope_name TEXT,
                    client_platform TEXT NOT NULL DEFAULT 'unattributed',
                    integration_id TEXT,
                    agent_id TEXT,
                    attribution_source TEXT NOT NULL DEFAULT 'unattributed',
                    created_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, principal, metric, event_key),
                    CHECK (metric IN ('ingest_raw_tokens', 'recall_requests')),
                    CHECK (units >= 0)
                );

                CREATE TABLE IF NOT EXISTS billing_plan_versions (
                    plan_code TEXT NOT NULL,
                    plan_version TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    billing_interval TEXT NOT NULL,
                    ingest_raw_token_limit INTEGER,
                    recall_request_limit INTEGER,
                    max_members INTEGER NOT NULL,
                    currency TEXT NOT NULL,
                    price_minor_units INTEGER,
                    entitlements_json TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (plan_code, plan_version),
                    CHECK (status IN ('active', 'retired')),
                    CHECK (billing_interval IN ('monthly', 'yearly', 'custom')),
                    CHECK (ingest_raw_token_limit IS NULL OR ingest_raw_token_limit >= 0),
                    CHECK (recall_request_limit IS NULL OR recall_request_limit >= 0),
                    CHECK (max_members >= 1),
                    CHECK (price_minor_units IS NULL OR price_minor_units >= 0)
                );
                CREATE INDEX IF NOT EXISTS billing_plan_versions_status_idx
                    ON billing_plan_versions (status, plan_code, updated_at);

                CREATE TABLE IF NOT EXISTS billing_groups (
                    tenant_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    active_period_id TEXT NOT NULL,
                    created_by_key_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, group_id),
                    CHECK (status IN ('active', 'suspended', 'cancelled'))
                );
                CREATE INDEX IF NOT EXISTS billing_groups_status_idx
                    ON billing_groups (tenant_id, status, updated_at);

                CREATE TABLE IF NOT EXISTS billing_group_periods (
                    tenant_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    period_id TEXT NOT NULL,
                    usage_principal TEXT NOT NULL,
                    plan_code TEXT NOT NULL,
                    plan_version TEXT NOT NULL,
                    billing_interval TEXT NOT NULL,
                    starts_at REAL NOT NULL,
                    ends_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    ingest_raw_token_limit INTEGER,
                    recall_request_limit INTEGER,
                    max_members INTEGER NOT NULL,
                    currency TEXT NOT NULL,
                    price_minor_units INTEGER,
                    entitlement_snapshot_json TEXT NOT NULL,
                    created_by_key_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, group_id, period_id),
                    UNIQUE (tenant_id, usage_principal),
                    FOREIGN KEY (tenant_id, group_id)
                        REFERENCES billing_groups(tenant_id, group_id)
                        ON DELETE CASCADE,
                    CHECK (billing_interval IN ('monthly', 'yearly', 'custom')),
                    CHECK (status IN ('scheduled', 'active', 'expired', 'cancelled')),
                    CHECK (ends_at > starts_at),
                    CHECK (ingest_raw_token_limit IS NULL OR ingest_raw_token_limit >= 0),
                    CHECK (recall_request_limit IS NULL OR recall_request_limit >= 0),
                    CHECK (max_members >= 1),
                    CHECK (price_minor_units IS NULL OR price_minor_units >= 0)
                );
                CREATE INDEX IF NOT EXISTS billing_group_periods_status_idx
                    ON billing_group_periods (
                        tenant_id, group_id, status, starts_at, ends_at
                    );

                CREATE TABLE IF NOT EXISTS billing_group_members (
                    tenant_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_by_key_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, subject),
                    FOREIGN KEY (tenant_id, group_id)
                        REFERENCES billing_groups(tenant_id, group_id)
                        ON DELETE CASCADE,
                    CHECK (role IN ('owner', 'admin', 'member'))
                );
                CREATE INDEX IF NOT EXISTS billing_group_members_group_idx
                    ON billing_group_members (tenant_id, group_id, role, created_at);

                CREATE TABLE IF NOT EXISTS billing_group_member_events (
                    event_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    role TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_by_key_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (tenant_id, group_id)
                        REFERENCES billing_groups(tenant_id, group_id)
                        ON DELETE CASCADE,
                    CHECK (role IN ('owner', 'admin', 'member')),
                    CHECK (event_type IN ('added', 'removed'))
                );
                CREATE INDEX IF NOT EXISTS billing_group_member_events_group_idx
                    ON billing_group_member_events (
                        tenant_id, group_id, created_at, event_id
                    );

                CREATE TABLE IF NOT EXISTS control_migrations (
                    migration_id TEXT PRIMARY KEY,
                    applied_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scope_lifecycle (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    deletion_job_id TEXT,
                    reason TEXT,
                    updated_at REAL NOT NULL,
                    deleted_at REAL,
                    PRIMARY KEY (tenant_id, scope_name),
                    CHECK (state IN ('active', 'deleting', 'deleted'))
                );
                CREATE INDEX IF NOT EXISTS scope_lifecycle_state_idx
                    ON scope_lifecycle (state, updated_at);

                CREATE TABLE IF NOT EXISTS content_deletions (
                    deletion_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    target_sha256 TEXT NOT NULL,
                    target_count INTEGER NOT NULL,
                    job_id TEXT UNIQUE,
                    state TEXT NOT NULL,
                    result_json TEXT,
                    error_code TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    CHECK (mode IN ('memory_ids', 'session')),
                    CHECK (target_count >= 1),
                    CHECK (state IN (
                        'requested', 'purging', 'reindexing', 'completed', 'failed'
                    ))
                );
                CREATE INDEX IF NOT EXISTS content_deletions_scope_idx
                    ON content_deletions (
                        tenant_id, scope_name, state, updated_at
                    );

                CREATE TABLE IF NOT EXISTS scope_quarantines (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    quarantined_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, scope_name)
                );
                CREATE INDEX IF NOT EXISTS scope_quarantines_updated_idx
                    ON scope_quarantines (updated_at);

                CREATE TABLE IF NOT EXISTS scope_quarantine_recoveries (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    quarantine_started_at REAL NOT NULL,
                    state TEXT NOT NULL,
                    cycle_count INTEGER NOT NULL DEFAULT 0,
                    resumed_job_count INTEGER NOT NULL DEFAULT 0,
                    active_job_id TEXT,
                    next_attempt_at REAL NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    last_error_code TEXT,
                    report_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    recovered_at REAL,
                    PRIMARY KEY (tenant_id, scope_name),
                    CHECK (state IN (
                        'waiting', 'auditing', 'repairing', 'verifying',
                        'manual_review', 'recovered'
                    )),
                    CHECK (cycle_count >= 0),
                    CHECK (resumed_job_count >= 0)
                );
                CREATE INDEX IF NOT EXISTS scope_quarantine_recoveries_due_idx
                    ON scope_quarantine_recoveries (
                        state, next_attempt_at, lease_expires_at, updated_at
                    );

                CREATE TABLE IF NOT EXISTS scope_quarantine_recovery_jobs (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    provider_attempt_count INTEGER NOT NULL DEFAULT 0,
                    local_repair_attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_local_repair_fingerprint TEXT,
                    last_error_code TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, scope_name, job_id),
                    CHECK (state IN (
                        'authorized', 'pending', 'running', 'succeeded',
                        'failed', 'manual_review'
                    )),
                    CHECK (attempt_count >= 0),
                    CHECK (provider_attempt_count >= 0),
                    CHECK (local_repair_attempt_count >= 0)
                );
                CREATE INDEX IF NOT EXISTS scope_quarantine_recovery_jobs_state_idx
                    ON scope_quarantine_recovery_jobs (
                        tenant_id, scope_name, state, updated_at
                    );

                CREATE TABLE IF NOT EXISTS scope_exports (
                    export_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    artifact_path TEXT,
                    artifact_sha256 TEXT,
                    size_bytes INTEGER,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    completed_at REAL,
                    CHECK (state IN ('pending', 'ready', 'failed', 'expired'))
                );
                CREATE UNIQUE INDEX IF NOT EXISTS scope_exports_job_uq
                    ON scope_exports (job_id);
                CREATE INDEX IF NOT EXISTS scope_exports_lookup_idx
                    ON scope_exports (tenant_id, scope_name, expires_at);

                CREATE TABLE IF NOT EXISTS scope_retention_policies (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    inactive_days INTEGER NOT NULL,
                    updated_by_key_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, scope_name),
                    CHECK (enabled IN (0, 1)),
                    CHECK (inactive_days BETWEEN 1 AND 3650)
                );

                CREATE TABLE IF NOT EXISTS memory_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    query_id TEXT,
                    rating TEXT NOT NULL,
                    memory_ids_json TEXT NOT NULL,
                    comment TEXT,
                    metadata_json TEXT NOT NULL,
                    created_by_credential_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    CHECK (rating IN ('helpful', 'incorrect', 'stale', 'unsafe', 'missing'))
                );
                CREATE INDEX IF NOT EXISTS memory_feedback_scope_idx
                    ON memory_feedback (tenant_id, scope_name, created_at);

                CREATE TABLE IF NOT EXISTS webhook_endpoints (
                    endpoint_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    url TEXT NOT NULL,
                    events_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_by_key_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    disabled_at REAL,
                    CHECK (enabled IN (0, 1))
                );
                CREATE INDEX IF NOT EXISTS webhook_endpoints_tenant_idx
                    ON webhook_endpoints (tenant_id, enabled, created_at);

                CREATE TABLE IF NOT EXISTS webhook_events (
                    event_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS webhook_events_tenant_idx
                    ON webhook_events (tenant_id, created_at);

                CREATE TABLE IF NOT EXISTS webhook_deliveries (
                    delivery_id TEXT PRIMARY KEY,
                    endpoint_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL,
                    last_error TEXT,
                    last_status_code INTEGER,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    delivered_at REAL,
                    UNIQUE (endpoint_id, event_id),
                    CHECK (state IN ('pending', 'delivering', 'delivered', 'dead'))
                );
                CREATE INDEX IF NOT EXISTS webhook_deliveries_due_idx
                    ON webhook_deliveries (state, next_attempt_at);

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    state TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    worker_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    heartbeat_at REAL,
                    lease_expires_at REAL,
                    version INTEGER NOT NULL DEFAULT 0,
                    scope_name TEXT NOT NULL DEFAULT 'default',
                    scope_seq INTEGER,
                    UNIQUE (tenant_id, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS jobs_pending_idx
                    ON jobs (tenant_id, state, created_at);

                CREATE TABLE IF NOT EXISTS rate_limit_minute (
                    tenant_id TEXT NOT NULL,
                    bucket_start INTEGER NOT NULL,
                    request_count INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, bucket_start)
                );

                CREATE TABLE IF NOT EXISTS rate_limit_leases (
                    lease_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS rate_limit_leases_active_idx
                    ON rate_limit_leases (tenant_id, expires_at);

                CREATE TABLE IF NOT EXISTS scope_heads (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    next_seq INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, scope_name)
                );

                CREATE TABLE IF NOT EXISTS scope_evolution_state (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    source_event_seq INTEGER NOT NULL DEFAULT 0,
                    promoted_event_seq INTEGER NOT NULL DEFAULT 0,
                    indexed_event_seq INTEGER NOT NULL DEFAULT 0,
                    delta_indexed_event_seq INTEGER NOT NULL DEFAULT 0,
                    conflict_generation INTEGER NOT NULL DEFAULT 0,
                    promoted_conflict_generation INTEGER NOT NULL DEFAULT 0,
                    last_ingest_at REAL,
                    last_slow_success_at REAL,
                    last_index_success_at REAL,
                    last_delta_index_success_at REAL,
                    active_evolution_job_id TEXT,
                    active_evolution_job_version INTEGER,
                    active_index_job_id TEXT,
                    active_index_job_version INTEGER,
                    source_raw_token_estimate INTEGER NOT NULL DEFAULT 0,
                    promoted_raw_token_estimate INTEGER NOT NULL DEFAULT 0,
                    source_user_turns INTEGER NOT NULL DEFAULT 0,
                    promoted_user_turns INTEGER NOT NULL DEFAULT 0,
                    dirty_since_at REAL,
                    index_dirty_since_at REAL,
                    reserved_cost_micro_cny INTEGER NOT NULL DEFAULT 0,
                    spent_cost_micro_cny INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, scope_name)
                );
                CREATE INDEX IF NOT EXISTS scope_evolution_due_idx
                    ON scope_evolution_state (source_event_seq, promoted_event_seq, last_ingest_at);

                CREATE TABLE IF NOT EXISTS scope_ingest_watermark_commits (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    source_event_seq INTEGER NOT NULL,
                    new_message_count INTEGER NOT NULL,
                    raw_token_estimate INTEGER NOT NULL,
                    user_turns INTEGER NOT NULL,
                    committed_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, scope_name, operation_id)
                );
                CREATE INDEX IF NOT EXISTS scope_ingest_watermark_commits_scope_idx
                    ON scope_ingest_watermark_commits (tenant_id, scope_name, source_event_seq);

                CREATE TABLE IF NOT EXISTS scope_source_event_commits (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    source_record_id TEXT NOT NULL,
                    origin_operation_id TEXT NOT NULL,
                    accounting_operation_id TEXT NOT NULL,
                    raw_token_estimate INTEGER NOT NULL,
                    user_turns INTEGER NOT NULL,
                    committed_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, scope_name, source_record_id)
                );
                CREATE INDEX IF NOT EXISTS scope_source_event_commits_operation_idx
                    ON scope_source_event_commits(
                        tenant_id, scope_name, accounting_operation_id
                    );

                CREATE TABLE IF NOT EXISTS scope_ingest_source_sets (
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    source_set_sha256 TEXT NOT NULL,
                    source_count INTEGER NOT NULL,
                    committed_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, scope_name, operation_id)
                );

                CREATE TABLE IF NOT EXISTS operation_stages (
                    stage_id TEXT PRIMARY KEY,
                    job_id TEXT,
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    scope_seq INTEGER,
                    stage_name TEXT NOT NULL,
                    stage_seq INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT,
                    result_json TEXT,
                    error TEXT,
                    worker_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    heartbeat_at REAL,
                    lease_expires_at REAL,
                    version INTEGER NOT NULL DEFAULT 0,
                    UNIQUE (job_id, stage_name)
                );
                CREATE INDEX IF NOT EXISTS operation_stages_ready_idx
                    ON operation_stages (tenant_id, scope_name, state, scope_seq, stage_seq, created_at);
                CREATE INDEX IF NOT EXISTS operation_stages_job_idx
                    ON operation_stages (job_id, stage_seq);

                CREATE TABLE IF NOT EXISTS job_lifecycle_audits (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    scope_seq INTEGER,
                    stage_id TEXT,
                    stage_name TEXT,
                    event_type TEXT NOT NULL,
                    from_state TEXT,
                    to_state TEXT,
                    reason_code TEXT NOT NULL,
                    reason_json TEXT NOT NULL,
                    worker_id TEXT,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS job_lifecycle_audits_job_idx
                    ON job_lifecycle_audits (job_id, audit_id);
                CREATE INDEX IF NOT EXISTS job_lifecycle_audits_scope_idx
                    ON job_lifecycle_audits (
                        tenant_id, scope_name, scope_seq, audit_id
                    );

                CREATE TABLE IF NOT EXISTS graph_runtime_audits (
                    scope_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    event_index INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (scope_id, field_name, event_index)
                );
                CREATE INDEX IF NOT EXISTS graph_runtime_audits_scope_idx
                    ON graph_runtime_audits (scope_id, field_name, event_index);

                CREATE TABLE IF NOT EXISTS provider_calls (
                    call_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    job_id TEXT,
                    stage_id TEXT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    operation TEXT,
                    status TEXT NOT NULL,
                    request_json TEXT,
                    response_json TEXT,
                    error TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    cost_micros INTEGER,
                    cache_hit_tokens INTEGER,
                    cache_miss_tokens INTEGER,
                    usage_state TEXT NOT NULL DEFAULT 'missing',
                    price_version TEXT,
                    key_id TEXT,
                    client_platform TEXT NOT NULL DEFAULT 'unattributed',
                    integration_id TEXT,
                    agent_id TEXT,
                    attribution_source TEXT NOT NULL DEFAULT 'unattributed',
                    request_sha256 TEXT,
                    response_sha256 TEXT,
                    started_at REAL,
                    finished_at REAL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS provider_calls_scope_idx
                    ON provider_calls (tenant_id, scope_name, created_at);
                CREATE INDEX IF NOT EXISTS provider_calls_stage_idx
                    ON provider_calls (stage_id, created_at);

                CREATE TABLE IF NOT EXISTS user_provider_tasks (
                    task_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    auth_key_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    stage_id TEXT NOT NULL,
                    task_stage TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL,
                    lease_token_sha256 TEXT,
                    lease_expires_at REAL,
                    provider TEXT,
                    model TEXT,
                    output_json TEXT,
                    response_sha256 TEXT,
                    usage_json TEXT,
                    provider_request_id TEXT,
                    error_code TEXT,
                    provider_started_at REAL,
                    provider_finished_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    version INTEGER NOT NULL DEFAULT 0,
                    CHECK (task_stage IN ('writer', 'organizer')),
                    CHECK (state IN (
                        'queued', 'leased', 'running', 'completed', 'failed', 'unknown'
                    )),
                    UNIQUE (job_id, stage_id, operation, request_sha256)
                );
                CREATE INDEX IF NOT EXISTS user_provider_tasks_claim_idx
                    ON user_provider_tasks (
                        tenant_id, auth_key_id, task_stage, state, created_at
                    );
                CREATE INDEX IF NOT EXISTS user_provider_tasks_job_idx
                    ON user_provider_tasks (job_id, stage_id, created_at);

                CREATE TABLE IF NOT EXISTS provider_call_reconciliations (
                    call_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    scope_name TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    reconciliation_kind TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    evidence_sha256 TEXT NOT NULL,
                    reconciled_at REAL NOT NULL,
                    FOREIGN KEY (call_id) REFERENCES provider_calls(call_id)
                );
                CREATE INDEX IF NOT EXISTS provider_call_reconciliations_scope_idx
                    ON provider_call_reconciliations (
                        tenant_id, scope_name, reconciled_at
                    );

                CREATE TABLE IF NOT EXISTS provider_prices (
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    currency TEXT NOT NULL DEFAULT 'USD',
                    input_micros_per_million INTEGER,
                    cache_hit_input_micros_per_million INTEGER,
                    cache_miss_input_micros_per_million INTEGER,
                    output_micros_per_million INTEGER,
                    effective_at REAL NOT NULL,
                    metadata_json TEXT,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (provider, model, effective_at)
                );
                CREATE INDEX IF NOT EXISTS provider_prices_lookup_idx
                    ON provider_prices (provider, model, effective_at DESC);
                """
            )
            scope_token_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(scope_tokens)")
            }
            if "scope_prefixes_json" not in scope_token_columns:
                connection.execute(
                    "ALTER TABLE scope_tokens "
                    "ADD COLUMN scope_prefixes_json TEXT NOT NULL DEFAULT '[]'"
                )
            projection_queue_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(memory_graph_refresh_queue)"
                )
            }
            for column, definition in {
                "heartbeat_at": "REAL",
                "progress_stage": "TEXT",
                "progress_completed": "INTEGER",
                "progress_total": "INTEGER",
                "pending_source_fingerprint": "TEXT",
            }.items():
                if column not in projection_queue_columns:
                    connection.execute(
                        "ALTER TABLE memory_graph_refresh_queue "
                        f"ADD COLUMN {column} {definition}"
                    )
            recovery_job_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(scope_quarantine_recovery_jobs)"
                )
            }
            if "provider_attempt_count" not in recovery_job_columns:
                connection.execute(
                    "ALTER TABLE scope_quarantine_recovery_jobs "
                    "ADD COLUMN provider_attempt_count INTEGER NOT NULL DEFAULT 0"
                )
            if "local_repair_attempt_count" not in recovery_job_columns:
                connection.execute(
                    "ALTER TABLE scope_quarantine_recovery_jobs "
                    "ADD COLUMN local_repair_attempt_count INTEGER NOT NULL DEFAULT 0"
                )
            if "last_local_repair_fingerprint" not in recovery_job_columns:
                connection.execute(
                    "ALTER TABLE scope_quarantine_recovery_jobs "
                    "ADD COLUMN last_local_repair_fingerprint TEXT"
                )
            connection.execute("BEGIN IMMEDIATE")
            try:
                split_ledger_migration = connection.execute(
                    "SELECT 1 FROM control_migrations WHERE migration_id=?",
                    ("quarantine_recovery_split_attempt_ledgers_v1",),
                ).fetchone()
                if split_ledger_migration is None:
                    # Legacy attempts all authorized a Writer/provider path.
                    # Keep that spent budget when the ledgers are split.
                    connection.execute(
                        "UPDATE scope_quarantine_recovery_jobs "
                        "SET provider_attempt_count=attempt_count "
                        "WHERE provider_attempt_count=0 "
                        "AND local_repair_attempt_count=0 "
                        "AND last_local_repair_fingerprint IS NULL"
                    )
                    connection.execute(
                        "INSERT INTO control_migrations(migration_id,applied_at) "
                        "VALUES(?,?)",
                        (
                            "quarantine_recovery_split_attempt_ledgers_v1",
                            time.time(),
                        ),
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            scope_token_issuance_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(scope_token_issuances)"
                )
            }
            legacy_scope_token_issuances = "confirmed_at" not in scope_token_issuance_columns
            if "token_replay_hash" not in scope_token_issuance_columns:
                connection.execute(
                    "ALTER TABLE scope_token_issuances ADD COLUMN token_replay_hash TEXT"
                )
            if "final_expires_at" not in scope_token_issuance_columns:
                connection.execute(
                    "ALTER TABLE scope_token_issuances ADD COLUMN final_expires_at REAL"
                )
            if "confirmed_at" not in scope_token_issuance_columns:
                connection.execute(
                    "ALTER TABLE scope_token_issuances ADD COLUMN confirmed_at REAL"
                )
            connection.execute(
                """
                UPDATE scope_token_issuances
                SET final_expires_at=COALESCE(
                    final_expires_at,
                    (SELECT expires_at FROM scope_tokens
                     WHERE scope_tokens.token_id=scope_token_issuances.token_id)
                )
                WHERE final_expires_at IS NULL
                """
            )
            if legacy_scope_token_issuances:
                # Rows from the pre-provisional protocol were fully active at
                # issue time.  Never run this backfill after the column exists:
                # NULL then means a new provisional Token is awaiting ACK.
                connection.execute(
                    "UPDATE scope_token_issuances "
                    "SET confirmed_at=created_at WHERE confirmed_at IS NULL"
                )
            # Early control-plane builds accidentally made quota idempotency
            # tenant-wide.  The table primary key already has the correct
            # principal-aware identity, so remove the extra legacy index.
            connection.execute("DROP INDEX IF EXISTS usage_events_tenant_event_uq")
            scope_ingest_event_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(scope_ingest_events)"
                )
            }
            usage_attribution_migrations = {
                "client_platform": "TEXT NOT NULL DEFAULT 'unattributed'",
                "integration_id": "TEXT",
                "agent_id": "TEXT",
                "attribution_source": "TEXT NOT NULL DEFAULT 'unattributed'",
            }
            for column, definition in usage_attribution_migrations.items():
                if column not in scope_ingest_event_columns:
                    connection.execute(
                        "ALTER TABLE scope_ingest_events "
                        f"ADD COLUMN {column} {definition}"
                    )
            usage_event_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(usage_events)")
            }
            if "consumer_principal" not in usage_event_columns:
                connection.execute(
                    "ALTER TABLE usage_events ADD COLUMN consumer_principal TEXT"
                )
                connection.execute(
                    "UPDATE usage_events SET consumer_principal=principal "
                    "WHERE consumer_principal IS NULL"
                )
            if "scope_name" not in usage_event_columns:
                connection.execute("ALTER TABLE usage_events ADD COLUMN scope_name TEXT")
            for column, definition in usage_attribution_migrations.items():
                if column not in usage_event_columns:
                    connection.execute(
                        f"ALTER TABLE usage_events ADD COLUMN {column} {definition}"
                    )
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS usage_events_scope_time_idx
                    ON usage_events (tenant_id, scope_name, created_at);
                CREATE INDEX IF NOT EXISTS usage_events_platform_time_idx
                    ON usage_events (tenant_id, client_platform, created_at);
                CREATE INDEX IF NOT EXISTS usage_events_integration_time_idx
                    ON usage_events (tenant_id, integration_id, created_at);
                CREATE INDEX IF NOT EXISTS usage_events_agent_time_idx
                    ON usage_events (tenant_id, agent_id, created_at);
                CREATE INDEX IF NOT EXISTS usage_events_consumer_time_idx
                    ON usage_events (tenant_id, consumer_principal, created_at);

                CREATE TRIGGER IF NOT EXISTS usage_events_consumer_insert_guard
                BEFORE INSERT ON usage_events
                WHEN NEW.consumer_principal IS NULL
                  OR trim(NEW.consumer_principal)=''
                BEGIN
                    SELECT RAISE(ABORT, 'usage_events.consumer_principal is required');
                END;

                CREATE TRIGGER IF NOT EXISTS usage_events_consumer_update_guard
                BEFORE UPDATE OF consumer_principal ON usage_events
                WHEN NEW.consumer_principal IS NULL
                  OR trim(NEW.consumer_principal)=''
                BEGIN
                    SELECT RAISE(ABORT, 'usage_events.consumer_principal is required');
                END;
                """
            )
            # Historical ingest events have a provable scope through the
            # immutable ingest admission row. Historical recall events do not,
            # so they intentionally remain NULL/unattributed.
            connection.execute(
                """
                UPDATE usage_events
                SET scope_name=(
                    SELECT ingest.scope_name
                    FROM scope_ingest_events AS ingest
                    WHERE ingest.tenant_id=usage_events.tenant_id
                      AND ingest.idempotency_key=usage_events.event_key
                )
                WHERE metric='ingest_raw_tokens' AND scope_name IS NULL
                """
            )
            connection.execute("BEGIN IMMEDIATE")
            try:
                principal_migration = connection.execute(
                    "SELECT 1 FROM control_migrations WHERE migration_id=?",
                    ("usage_principal_namespace_v1",),
                ).fetchone()
                if principal_migration is None:
                    for table in (
                        "usage_entitlements",
                        "usage_totals",
                        "usage_events",
                    ):
                        connection.execute(
                            f"""
                            UPDATE {table}
                            SET principal=CASE
                                WHEN principal=tenant_id THEN 'tenant:' || principal
                                ELSE 'subject:' || principal
                            END
                            """
                        )
                    # ``consumer_principal`` did not exist before this
                    # namespace migration.  When both migrations run on the
                    # same legacy database it was initially copied from the
                    # unqualified principal, so align it with the newly
                    # qualified value before recording the migration marker.
                    connection.execute(
                        "UPDATE usage_events SET consumer_principal=principal"
                    )
                    connection.execute(
                        "INSERT INTO control_migrations(migration_id,applied_at) "
                        "VALUES(?,?)",
                        ("usage_principal_namespace_v1", time.time()),
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            provider_call_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(provider_calls)")
            }
            provider_call_migrations = {
                "cache_hit_tokens": "INTEGER",
                "cache_miss_tokens": "INTEGER",
                "usage_state": "TEXT NOT NULL DEFAULT 'missing'",
                "price_version": "TEXT",
                "key_id": "TEXT",
                **usage_attribution_migrations,
                "request_sha256": "TEXT",
                "response_sha256": "TEXT",
            }
            for column, definition in provider_call_migrations.items():
                if column not in provider_call_columns:
                    connection.execute(
                        f"ALTER TABLE provider_calls ADD COLUMN {column} {definition}"
                    )
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS provider_calls_platform_time_idx
                    ON provider_calls (tenant_id, client_platform, created_at);
                CREATE INDEX IF NOT EXISTS provider_calls_integration_time_idx
                    ON provider_calls (tenant_id, integration_id, created_at);
                CREATE INDEX IF NOT EXISTS provider_calls_agent_time_idx
                    ON provider_calls (tenant_id, agent_id, created_at);
                """
            )
            provider_price_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(provider_prices)")
            }
            provider_price_migrations = {
                "cache_hit_input_micros_per_million": "INTEGER",
                "cache_miss_input_micros_per_million": "INTEGER",
            }
            for column, definition in provider_price_migrations.items():
                if column not in provider_price_columns:
                    connection.execute(
                        f"ALTER TABLE provider_prices ADD COLUMN {column} {definition}"
                    )
            job_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(jobs)")
            }
            if "heartbeat_at" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN heartbeat_at REAL")
            if "lease_expires_at" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN lease_expires_at REAL")
            if "scope_name" not in job_columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN scope_name TEXT NOT NULL DEFAULT 'default'"
                )
            if "scope_seq" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN scope_seq INTEGER")
            connection.execute(
                "UPDATE jobs SET scope_name='default' "
                "WHERE scope_name IS NULL OR trim(scope_name)=''"
            )
            connection.execute(
                """
                UPDATE jobs
                SET scope_name = json_extract(payload_json, '$.scope_name')
                WHERE json_valid(payload_json)
                  AND json_type(payload_json, '$.scope_name') = 'text'
                  AND trim(json_extract(payload_json, '$.scope_name')) <> ''
                """
            )
            # Backfill legacy rows before installing the database-level
            # identity contract.  From this point onward the column is the
            # durable scope identity; payload scope is only a checked echo.
            connection.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS jobs_scope_payload_insert_json_guard
                BEFORE INSERT ON jobs
                WHEN COALESCE(json_valid(NEW.payload_json), 0) = 0
                BEGIN
                    SELECT RAISE(ABORT, 'jobs.payload_json must be valid JSON');
                END;

                CREATE TRIGGER IF NOT EXISTS jobs_scope_payload_update_json_guard
                BEFORE UPDATE ON jobs
                WHEN COALESCE(json_valid(NEW.payload_json), 0) = 0
                BEGIN
                    SELECT RAISE(ABORT, 'jobs.payload_json must be valid JSON');
                END;

                CREATE TRIGGER IF NOT EXISTS jobs_scope_payload_insert_identity_guard
                BEFORE INSERT ON jobs
                WHEN json_type(
                    CASE
                        WHEN COALESCE(json_valid(NEW.payload_json), 0) = 1
                        THEN NEW.payload_json
                        ELSE '{}'
                    END,
                    '$.scope_name'
                ) = 'text'
                AND trim(json_extract(
                    CASE
                        WHEN COALESCE(json_valid(NEW.payload_json), 0) = 1
                        THEN NEW.payload_json
                        ELSE '{}'
                    END,
                    '$.scope_name'
                )) <> ''
                AND json_extract(
                    CASE
                        WHEN COALESCE(json_valid(NEW.payload_json), 0) = 1
                        THEN NEW.payload_json
                        ELSE '{}'
                    END,
                    '$.scope_name'
                ) IS NOT NEW.scope_name
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'jobs.scope_name must match payload_json.scope_name'
                    );
                END;

                CREATE TRIGGER IF NOT EXISTS jobs_scope_payload_update_identity_guard
                BEFORE UPDATE ON jobs
                WHEN json_type(
                    CASE
                        WHEN COALESCE(json_valid(NEW.payload_json), 0) = 1
                        THEN NEW.payload_json
                        ELSE '{}'
                    END,
                    '$.scope_name'
                ) = 'text'
                AND trim(json_extract(
                    CASE
                        WHEN COALESCE(json_valid(NEW.payload_json), 0) = 1
                        THEN NEW.payload_json
                        ELSE '{}'
                    END,
                    '$.scope_name'
                )) <> ''
                AND json_extract(
                    CASE
                        WHEN COALESCE(json_valid(NEW.payload_json), 0) = 1
                        THEN NEW.payload_json
                        ELSE '{}'
                    END,
                    '$.scope_name'
                ) IS NOT NEW.scope_name
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'jobs.scope_name must match payload_json.scope_name'
                    );
                END;
                """
            )
            evolution_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(scope_evolution_state)")
            }
            if "indexed_event_seq" not in evolution_columns:
                connection.execute(
                    "ALTER TABLE scope_evolution_state ADD COLUMN indexed_event_seq INTEGER NOT NULL DEFAULT 0"
                )
                # Before this ledger column existed, every successful Slow
                # promotion completed by activating the matching full index.
                # Backfill only that proven coverage; later Source events remain
                # dirty and must be made searchable by the online delta path.
                connection.execute(
                    "UPDATE scope_evolution_state "
                    "SET indexed_event_seq=promoted_event_seq"
                )
            if "last_index_success_at" not in evolution_columns:
                connection.execute(
                    "ALTER TABLE scope_evolution_state ADD COLUMN last_index_success_at REAL"
                )
            if "delta_indexed_event_seq" not in evolution_columns:
                connection.execute(
                    "ALTER TABLE scope_evolution_state ADD COLUMN delta_indexed_event_seq INTEGER NOT NULL DEFAULT 0"
                )
                connection.execute(
                    "UPDATE scope_evolution_state SET delta_indexed_event_seq=indexed_event_seq"
                )
            if "last_delta_index_success_at" not in evolution_columns:
                connection.execute(
                    "ALTER TABLE scope_evolution_state ADD COLUMN last_delta_index_success_at REAL"
                )
            if "active_index_job_id" not in evolution_columns:
                connection.execute(
                    "ALTER TABLE scope_evolution_state ADD COLUMN active_index_job_id TEXT"
                )
            evolution_migrations = {
                "source_raw_token_estimate": "INTEGER NOT NULL DEFAULT 0",
                "promoted_raw_token_estimate": "INTEGER NOT NULL DEFAULT 0",
                "source_user_turns": "INTEGER NOT NULL DEFAULT 0",
                "promoted_user_turns": "INTEGER NOT NULL DEFAULT 0",
                "dirty_since_at": "REAL",
                "index_dirty_since_at": "REAL",
                "active_evolution_job_version": "INTEGER",
                "active_index_job_version": "INTEGER",
            }
            for column, definition in evolution_migrations.items():
                if column not in evolution_columns:
                    connection.execute(
                        f"ALTER TABLE scope_evolution_state ADD COLUMN {column} {definition}"
                    )
            connection.execute(
                """
                UPDATE scope_evolution_state
                SET dirty_since_at=COALESCE(dirty_since_at, last_ingest_at)
                WHERE source_event_seq>promoted_event_seq AND dirty_since_at IS NULL
                """
            )
            connection.execute(
                """
                UPDATE scope_evolution_state
                SET index_dirty_since_at=COALESCE(index_dirty_since_at, last_ingest_at)
                WHERE source_event_seq>indexed_event_seq AND index_dirty_since_at IS NULL
                """
            )
            connection.execute(
                """
                UPDATE jobs AS current
                SET scope_seq = (
                    SELECT COUNT(*)
                    FROM jobs AS prior
                    WHERE prior.tenant_id = current.tenant_id
                      AND prior.scope_name = current.scope_name
                      AND (
                          prior.created_at < current.created_at
                          OR (prior.created_at = current.created_at AND prior.job_id <= current.job_id)
                      )
                )
                WHERE current.scope_seq IS NULL
                """
            )
            connection.execute(
                """
                INSERT INTO scope_heads(tenant_id, scope_name, next_seq, updated_at)
                SELECT tenant_id, scope_name, MAX(scope_seq) + 1, strftime('%s', 'now')
                FROM jobs
                GROUP BY tenant_id, scope_name
                ON CONFLICT(tenant_id, scope_name) DO UPDATE SET
                    next_seq = CASE
                        WHEN scope_heads.next_seq < excluded.next_seq THEN excluded.next_seq
                        ELSE scope_heads.next_seq
                    END,
                    updated_at = excluded.updated_at
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_lease_idx "
                "ON jobs(state, lease_expires_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_scope_order_idx "
                "ON jobs(tenant_id, scope_name, state, scope_seq, created_at)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS jobs_scope_seq_idx "
                "ON jobs(tenant_id, scope_name, scope_seq)"
            )

    @contextlib.contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        """Yield a connection with an explicit commit/rollback boundary."""

        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _validate_graph_audit_identity(scope_id: str, field_name: str) -> None:
        if not scope_id or not scope_id.strip():
            raise ValueError("scope_id is required")
        if field_name not in _GRAPH_AUDIT_FIELDS:
            raise ValueError(f"unsupported graph audit field: {field_name}")

    def graph_runtime_audits(
        self, scope_id: str, field_name: str
    ) -> dict[str, Any]:
        self._validate_graph_audit_identity(scope_id, field_name)
        with self.transaction(immediate=False) as connection:
            rows = connection.execute(
                "SELECT event_index,payload_json FROM graph_runtime_audits "
                "WHERE scope_id=? AND field_name=? ORDER BY event_index",
                (scope_id, field_name),
            ).fetchall()
        payloads = [json.loads(str(row["payload_json"])) for row in rows]
        event_total = int(rows[-1]["event_index"]) + 1 if rows else 0
        return {
            "payloads": payloads,
            "event_total": event_total,
            "trimmed_total": max(0, event_total - len(payloads)),
        }

    def append_graph_runtime_audit(
        self,
        scope_id: str,
        field_name: str,
        payload: Mapping[str, Any],
        *,
        retention: int,
        base_event_total: int = 0,
        base_trimmed_total: int = 0,
    ) -> dict[str, Any]:
        self._validate_graph_audit_identity(scope_id, field_name)
        if retention <= 0:
            raise ValueError("retention must be positive")
        if base_event_total < 0 or base_trimmed_total < 0:
            raise ValueError("base audit counters cannot be negative")
        stored_payload = dict(payload)
        now = time.time()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(event_index), -1) AS maximum "
                "FROM graph_runtime_audits WHERE scope_id=? AND field_name=?",
                (scope_id, field_name),
            ).fetchone()
            event_index = int(row["maximum"]) + 1
            event_total = base_event_total + event_index + 1
            if field_name == "retrieval_log":
                stored_payload["query_id"] = f"query:{event_total}"
            connection.execute(
                "INSERT INTO graph_runtime_audits"
                "(scope_id,field_name,event_index,payload_json,created_at) "
                "VALUES(?,?,?,?,?)",
                (
                    scope_id,
                    field_name,
                    event_index,
                    self.encode_json(stored_payload),
                    now,
                ),
            )
            retained = int(
                connection.execute(
                    "SELECT COUNT(*) FROM graph_runtime_audits "
                    "WHERE scope_id=? AND field_name=?",
                    (scope_id, field_name),
                ).fetchone()[0]
            )
            overflow = max(0, retained - retention)
            if overflow:
                connection.execute(
                    "DELETE FROM graph_runtime_audits WHERE rowid IN ("
                    "SELECT rowid FROM graph_runtime_audits "
                    "WHERE scope_id=? AND field_name=? "
                    "ORDER BY event_index LIMIT ?)",
                    (scope_id, field_name, overflow),
                )
                retained -= overflow
        external_total = event_index + 1
        return {
            "payload": stored_payload,
            "event_total": event_total,
            "trimmed_total": max(
                base_trimmed_total,
                event_total - (base_event_total - base_trimmed_total + retained),
            ),
            "external_event_total": external_total,
            "appended": True,
        }

    def get_tenant_scopes(self, tenant_id: str) -> frozenset[str]:
        with self.transaction(immediate=False) as connection:
            rows = connection.execute(
                "SELECT scope FROM tenant_scopes WHERE tenant_id = ? ORDER BY scope",
                (tenant_id,),
            ).fetchall()
        return frozenset(row["scope"] for row in rows)

    def set_tenant_scopes(self, tenant_id: str, scopes: set[str] | frozenset[str]) -> None:
        if not tenant_id or any(not scope or not scope.strip() for scope in scopes):
            raise ValueError("tenant_id and scopes must be non-empty")
        with self.transaction() as connection:
            connection.execute("DELETE FROM tenant_scopes WHERE tenant_id = ?", (tenant_id,))
            connection.executemany(
                "INSERT INTO tenant_scopes (tenant_id, scope, created_at) VALUES (?, ?, strftime('%s', 'now'))",
                [(tenant_id, scope) for scope in sorted(scopes)],
            )

    @staticmethod
    def encode_json(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def journal_mode(self) -> str:
        with closing(self.connect()) as connection:
            return str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    @staticmethod
    def _validate_scope(tenant_id: str, scope_name: str) -> None:
        if not tenant_id or not scope_name or not scope_name.strip():
            raise ValueError("tenant_id and scope_name are required")

    @staticmethod
    def _allocate_scope_seq(connection: sqlite3.Connection, tenant_id: str, scope_name: str, now: float) -> int:
        """Allocate a monotonically increasing sequence while holding the write lock."""
        row = connection.execute(
            "SELECT next_seq FROM scope_heads WHERE tenant_id=? AND scope_name=?",
            (tenant_id, scope_name),
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO scope_heads(tenant_id, scope_name, next_seq, updated_at) VALUES (?, ?, ?, ?)",
                (tenant_id, scope_name, 2, now),
            )
            return 1
        sequence = int(row[0])
        connection.execute(
            "UPDATE scope_heads SET next_seq=?, updated_at=? WHERE tenant_id=? AND scope_name=?",
            (sequence + 1, now, tenant_id, scope_name),
        )
        return sequence

    def allocate_scope_seq(self, tenant_id: str, scope_name: str) -> int:
        """Atomically allocate the next sequence for a tenant/scope pair."""
        self._validate_scope(tenant_id, scope_name)
        with self.transaction() as connection:
            return self._allocate_scope_seq(connection, tenant_id, scope_name, time.time())

    @staticmethod
    def _evolution_row(row: sqlite3.Row | None) -> dict[str, object] | None:
        if row is None:
            return None
        return {key: row[key] for key in row.keys()}

    def get_scope_evolution_state(
        self, tenant_id: str, scope_name: str
    ) -> dict[str, object] | None:
        self._validate_scope(tenant_id, scope_name)
        with self.transaction(immediate=False) as connection:
            row = connection.execute(
                "SELECT * FROM scope_evolution_state WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
        return self._evolution_row(row)

    def list_scope_evolution_states(
        self, *, include_inactive: bool = False
    ) -> list[dict[str, object]]:
        """Return watermark ledgers eligible for production readiness.

        Deleted, deleting, and explicitly quarantined scopes cannot serve user
        traffic and therefore do not make the whole service unready. Their
        immutable artifacts are still covered by the separate active-index
        integrity audit.
        """

        with self.transaction(immediate=False) as connection:
            if include_inactive:
                rows = connection.execute(
                    "SELECT evolution.* FROM scope_evolution_state AS evolution "
                    "ORDER BY evolution.tenant_id, evolution.scope_name"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT evolution.* FROM scope_evolution_state AS evolution "
                    "LEFT JOIN scope_lifecycle AS lifecycle "
                    "ON lifecycle.tenant_id=evolution.tenant_id "
                    "AND lifecycle.scope_name=evolution.scope_name "
                    "LEFT JOIN scope_quarantines AS quarantine "
                    "ON quarantine.tenant_id=evolution.tenant_id "
                    "AND quarantine.scope_name=evolution.scope_name "
                    "LEFT JOIN content_deletions AS content_deletion "
                    "ON content_deletion.tenant_id=evolution.tenant_id "
                    "AND content_deletion.scope_name=evolution.scope_name "
                    "AND content_deletion.state IN "
                    "('requested','purging','reindexing','failed') "
                    "WHERE quarantine.tenant_id IS NULL "
                    "AND content_deletion.deletion_id IS NULL "
                    "AND (lifecycle.state IS NULL OR lifecycle.state='active') "
                    "ORDER BY evolution.tenant_id, evolution.scope_name"
                ).fetchall()
        return [dict(self._evolution_row(row) or {}) for row in rows]

    def count_quarantined_scopes(self) -> int:
        with self.transaction(immediate=False) as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM scope_quarantines"
            ).fetchone()
        return int(row["total"] or 0)

    @staticmethod
    def _append_job_lifecycle_audit(
        connection: sqlite3.Connection,
        *,
        job_id: str,
        tenant_id: str,
        scope_name: str,
        scope_seq: int | None,
        event_type: str,
        reason: Mapping[str, Any],
        from_state: str | None = None,
        to_state: str | None = None,
        stage_id: str | None = None,
        stage_name: str | None = None,
        worker_id: str | None = None,
        created_at: float | None = None,
    ) -> None:
        """Append one structured state-machine decision inside its transaction."""

        code = str(reason.get("code") or "").strip()
        if not code:
            raise ValueError("lifecycle audit reason requires a code")
        moment = time.time() if created_at is None else float(created_at)
        connection.execute(
            """
            INSERT INTO job_lifecycle_audits(
                job_id,tenant_id,scope_name,scope_seq,stage_id,stage_name,
                event_type,from_state,to_state,reason_code,reason_json,
                worker_id,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                job_id,
                tenant_id,
                scope_name,
                scope_seq,
                stage_id,
                stage_name,
                event_type,
                from_state,
                to_state,
                code,
                ControlDB.encode_json(dict(reason)),
                worker_id,
                moment,
            ),
        )

    def list_job_lifecycle_audits(self, job_id: str) -> list[dict[str, Any]]:
        if not job_id:
            raise ValueError("job_id is required")
        with self.transaction(immediate=False) as connection:
            rows = connection.execute(
                "SELECT * FROM job_lifecycle_audits WHERE job_id=? ORDER BY audit_id",
                (job_id,),
            ).fetchall()
        return [
            {
                **{key: row[key] for key in row.keys() if key != "reason_json"},
                "reason": json.loads(str(row["reason_json"])),
            }
            for row in rows
        ]

    @staticmethod
    def _job_type_from_row(row: sqlite3.Row) -> str:
        try:
            payload = json.loads(str(row["payload_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return ""
        return str(payload.get("job_type") or "") if isinstance(payload, Mapping) else ""

    def _scope_scheduler_gate(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        scope_name: str,
        *,
        candidate_job_id: str | None = None,
        include_candidate_ingest: bool = False,
        target_source_event_seq: int | None = None,
    ) -> dict[str, Any]:
        """Prove that every ingest visible to a derived job is durably closed.

        The proof is intentionally control-plane-only. A successful ingest must
        have an immutable watermark commit, unresolved job states block derived
        work, and provider calls in ``started``/``unknown`` remain uncertain.
        Terminal failed/cancelled attempts do not block forever: the storage
        projection remains the authoritative Source/journal integrity gate when
        the derived stage actually opens the scope database.
        """

        quarantine = connection.execute(
            "SELECT quarantined_at FROM scope_quarantines "
            "WHERE tenant_id=? AND scope_name=?",
            (tenant_id, scope_name),
        ).fetchone()
        recovery_authorized = None
        if quarantine is not None and candidate_job_id is not None:
            recovery_authorized = connection.execute(
                """
                SELECT 1
                FROM scope_quarantine_recovery_jobs AS recovery_job
                JOIN scope_quarantine_recoveries AS recovery
                  ON recovery.tenant_id=recovery_job.tenant_id
                 AND recovery.scope_name=recovery_job.scope_name
                WHERE recovery_job.tenant_id=?
                  AND recovery_job.scope_name=?
                  AND recovery_job.job_id=?
                  AND recovery_job.state IN ('authorized','pending','running')
                  AND recovery.state='repairing'
                  AND recovery.quarantine_started_at=?
                """,
                (
                    tenant_id,
                    scope_name,
                    candidate_job_id,
                    float(quarantine["quarantined_at"]),
                ),
            ).fetchone()
        if quarantine is not None and recovery_authorized is None:
            return {
                "ready": False,
                "reason_code": "scope_quarantined",
                "tenant_id": tenant_id,
                "scope_name": scope_name,
                "candidate_job_id": candidate_job_id,
                "blockers": ({"code": "scope_quarantined"},),
            }
        lifecycle = connection.execute(
            "SELECT state FROM scope_lifecycle WHERE tenant_id=? AND scope_name=?",
            (tenant_id, scope_name),
        ).fetchone()
        if lifecycle is not None and str(lifecycle["state"]) != "active":
            state = str(lifecycle["state"])
            return {
                "ready": False,
                "reason_code": f"scope_{state}",
                "tenant_id": tenant_id,
                "scope_name": scope_name,
                "candidate_job_id": candidate_job_id,
                "blockers": ({"code": f"scope_{state}"},),
            }
        content_deletion = connection.execute(
            "SELECT state,job_id FROM content_deletions "
            "WHERE tenant_id=? AND scope_name=? "
            "AND state IN ('requested','purging','reindexing','failed') "
            "ORDER BY created_at LIMIT 1",
            (tenant_id, scope_name),
        ).fetchone()
        if (
            content_deletion is not None
            and str(content_deletion["job_id"] or "") != str(candidate_job_id or "")
        ):
            return {
                "ready": False,
                "reason_code": "scope_content_deleting",
                "tenant_id": tenant_id,
                "scope_name": scope_name,
                "candidate_job_id": candidate_job_id,
                "blockers": ({"code": "scope_content_deleting"},),
            }

        candidate = None
        cutoff_scope_seq: int | None = None
        if candidate_job_id is not None:
            candidate = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (candidate_job_id,)
            ).fetchone()
            if candidate is None:
                return {
                    "ready": False,
                    "reason_code": "candidate_job_missing",
                    "tenant_id": tenant_id,
                    "scope_name": scope_name,
                    "candidate_job_id": candidate_job_id,
                    "blockers": ({"code": "candidate_job_missing"},),
                }
            if (
                str(candidate["tenant_id"]) != tenant_id
                or str(candidate["scope_name"]) != scope_name
            ):
                return {
                    "ready": False,
                    "reason_code": "candidate_scope_mismatch",
                    "tenant_id": tenant_id,
                    "scope_name": scope_name,
                    "candidate_job_id": candidate_job_id,
                    "blockers": ({"code": "candidate_scope_mismatch"},),
                }
            cutoff_scope_seq = int(candidate["scope_seq"])

        # A content-deletion job deliberately removes the Source-accounting
        # rows that proved its earlier ingests were committed.  Once that same
        # job has reached ``reindexing``, re-running the ordinary ingest barrier
        # would therefore reject the exact cleanup it just performed.  Keep the
        # exception narrow: the registered deletion must own the candidate,
        # the durable deletion state must already be ``reindexing``, and the
        # candidate must still be one of the two content-deletion job types.
        if (
            content_deletion is not None
            and candidate is not None
            and str(content_deletion["job_id"] or "") == str(candidate_job_id or "")
            and str(content_deletion["state"] or "") == "reindexing"
            and self._job_type_from_row(candidate)
            in {"delete_memories", "delete_session"}
        ):
            return {
                "ready": True,
                "reason_code": "content_deletion_reindex_owner",
                "tenant_id": tenant_id,
                "scope_name": scope_name,
                "candidate_job_id": candidate_job_id,
                "checked_before_scope_seq": cutoff_scope_seq,
                "checked_ingest_count": 0,
                "state_counts": {},
                "committed_source_event_seq": None,
                "target_source_event_seq": target_source_event_seq,
                "blockers": (),
            }

        query = "SELECT * FROM jobs WHERE tenant_id=? AND scope_name=?"
        parameters: list[Any] = [tenant_id, scope_name]
        if cutoff_scope_seq is not None:
            query += " AND scope_seq<?"
            parameters.append(cutoff_scope_seq)
        query += " ORDER BY scope_seq, job_id"
        prior_rows = list(connection.execute(query, parameters).fetchall())
        if (
            include_candidate_ingest
            and candidate is not None
            and self._job_type_from_row(candidate) == "ingest"
        ):
            prior_rows.append(candidate)

        ingest_rows = [row for row in prior_rows if self._job_type_from_row(row) == "ingest"]
        job_ids = [str(row["job_id"]) for row in ingest_rows]
        commits: set[str] = set()
        stage_summaries: dict[str, dict[str, int]] = {}
        provider_summaries: dict[str, dict[str, int]] = {}
        if job_ids:
            job_id_set = set(job_ids)
            for row in connection.execute(
                "SELECT operation_id FROM scope_ingest_watermark_commits "
                "WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchall():
                operation_id = str(row["operation_id"])
                if operation_id in job_id_set:
                    commits.add(operation_id)
                    continue
                job_id, marker, attempt = operation_id.rpartition(
                    ":writer:attempt:"
                )
                if marker and job_id in job_id_set and attempt.isdigit():
                    commits.add(job_id)
            for row in connection.execute(
                "SELECT job_id,COUNT(*) AS total,"
                "SUM(CASE WHEN state='running' THEN 1 ELSE 0 END) AS running,"
                "SUM(CASE WHEN state='failed' THEN 1 ELSE 0 END) AS failed "
                "FROM operation_stages WHERE tenant_id=? AND scope_name=? "
                "AND job_id IS NOT NULL GROUP BY job_id",
                (tenant_id, scope_name),
            ).fetchall():
                if str(row["job_id"]) not in job_id_set:
                    continue
                stage_summaries[str(row["job_id"])] = {
                    "total": int(row["total"] or 0),
                    "running": int(row["running"] or 0),
                    "failed": int(row["failed"] or 0),
                }
            for row in connection.execute(
                "SELECT calls.job_id AS job_id,COUNT(*) AS total,"
                "SUM(CASE WHEN calls.status IN ('started','unknown') "
                "AND reconciliation.call_id IS NULL THEN 1 ELSE 0 END) AS uncertain "
                "FROM provider_calls AS calls "
                "LEFT JOIN provider_call_reconciliations AS reconciliation "
                "ON reconciliation.call_id=calls.call_id "
                "WHERE calls.tenant_id=? AND calls.scope_name=? "
                "AND calls.job_id IS NOT NULL GROUP BY calls.job_id",
                (tenant_id, scope_name),
            ).fetchall():
                if str(row["job_id"]) not in job_id_set:
                    continue
                provider_summaries[str(row["job_id"])] = {
                    "total": int(row["total"] or 0),
                    "uncertain": int(row["uncertain"] or 0),
                }

        blockers: list[dict[str, Any]] = []
        state_counts: dict[str, int] = {}
        for row in ingest_rows:
            job_id = str(row["job_id"])
            state = str(row["state"])
            state_counts[state] = state_counts.get(state, 0) + 1
            is_candidate = job_id == candidate_job_id and include_candidate_ingest
            committed = job_id in commits
            stages = stage_summaries.get(job_id, {"total": 0, "running": 0, "failed": 0})
            providers = provider_summaries.get(job_id, {"total": 0, "uncertain": 0})
            codes: list[str] = []
            if providers["uncertain"]:
                codes.append("provider_call_uncertain")
            if not is_candidate and state in {"pending", "running"}:
                codes.append(f"ingest_{state}")
            elif is_candidate and state not in {"pending", "running"}:
                codes.append("candidate_ingest_not_active")
            if state == "succeeded" and not committed:
                codes.append("journal_readiness_uncommitted")
            if is_candidate and not committed:
                codes.append("journal_readiness_uncommitted")
            if stages["running"] and not is_candidate:
                codes.append("ingest_stage_running")
            if codes:
                blockers.append(
                    {
                        "job_id": job_id,
                        "scope_seq": int(row["scope_seq"]),
                        "state": state,
                        "codes": tuple(dict.fromkeys(codes)),
                        "journal_committed": committed,
                        "stage_count": stages["total"],
                        "uncertain_provider_call_count": providers["uncertain"],
                    }
                )

        evolution = connection.execute(
            "SELECT source_event_seq FROM scope_evolution_state "
            "WHERE tenant_id=? AND scope_name=?",
            (tenant_id, scope_name),
        ).fetchone()
        committed_source_event_seq = 0 if evolution is None else int(evolution["source_event_seq"])
        if (
            target_source_event_seq is not None
            and int(target_source_event_seq) > committed_source_event_seq
        ):
            blockers.append(
                {
                    "code": "target_source_watermark_uncommitted",
                    "target_source_event_seq": int(target_source_event_seq),
                    "committed_source_event_seq": committed_source_event_seq,
                }
            )
        return {
            "ready": not blockers,
            "reason_code": "ready" if not blockers else "scope_ingest_barrier_not_ready",
            "tenant_id": tenant_id,
            "scope_name": scope_name,
            "candidate_job_id": candidate_job_id,
            "checked_before_scope_seq": cutoff_scope_seq,
            "checked_ingest_count": len(ingest_rows),
            "state_counts": state_counts,
            "committed_source_event_seq": committed_source_event_seq,
            "target_source_event_seq": target_source_event_seq,
            "blockers": tuple(blockers),
        }

    def scope_scheduler_gate(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        candidate_job_id: str | None = None,
        include_candidate_ingest: bool = False,
        target_source_event_seq: int | None = None,
    ) -> dict[str, Any]:
        self._validate_scope(tenant_id, scope_name)
        with self.transaction(immediate=False) as connection:
            return self._scope_scheduler_gate(
                connection,
                tenant_id,
                scope_name,
                candidate_job_id=candidate_job_id,
                include_candidate_ingest=include_candidate_ingest,
                target_source_event_seq=target_source_event_seq,
            )

    def record_committed_source_events(
        self,
        tenant_id: str,
        scope_name: str,
        source_event_seq: int,
        *,
        conflict_generation: int = 0,
        ingested_at: float | None = None,
        operation_id: str | None = None,
        new_message_count: int | None = None,
        raw_token_estimate: int = 0,
        user_turns: int = 0,
    ) -> dict[str, object]:
        """Record one committed ingest exactly once.

        ``operation_id`` is mandatory for production callers.  The legacy
        absolute-watermark path remains for migrations and tests, but only an
        operation identity can make a crash replay provably idempotent.
        """
        self._validate_scope(tenant_id, scope_name)
        values = (source_event_seq, conflict_generation, raw_token_estimate, user_turns)
        if any(value < 0 for value in values):
            raise ValueError("event, conflict, token, and turn values must be non-negative")
        if new_message_count is not None and new_message_count < 0:
            raise ValueError("new_message_count must be non-negative")
        if operation_id is not None and not operation_id.strip():
            raise ValueError("operation_id cannot be empty")
        if operation_id is not None and new_message_count is None:
            raise ValueError("new_message_count is required with operation_id")
        now = time.time() if ingested_at is None else float(ingested_at)
        with self.transaction() as connection:
            existing = None
            if operation_id is not None:
                existing = connection.execute(
                    """
                    SELECT * FROM scope_ingest_watermark_commits
                    WHERE tenant_id=? AND scope_name=? AND operation_id=?
                    """,
                    (tenant_id, scope_name, operation_id),
                ).fetchone()
            if existing is not None:
                expected = (
                    int(new_message_count or 0),
                    int(raw_token_estimate),
                    int(user_turns),
                )
                actual = (
                    int(existing["new_message_count"]),
                    int(existing["raw_token_estimate"]),
                    int(existing["user_turns"]),
                )
                if actual != expected:
                    raise ValueError("ingest operation replay changed committed metrics")
                row = connection.execute(
                    "SELECT * FROM scope_evolution_state WHERE tenant_id=? AND scope_name=?",
                    (tenant_id, scope_name),
                ).fetchone()
                return self._evolution_row(row) or {}

            current = connection.execute(
                "SELECT * FROM scope_evolution_state WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            current_source = 0 if current is None else int(current["source_event_seq"])
            if operation_id is not None:
                expected_source = current_source + int(new_message_count or 0)
                if source_event_seq != expected_source:
                    raise ValueError(
                        "source_event_seq must equal the current watermark plus new_message_count"
                    )
                advances_source = int(new_message_count or 0) > 0
            else:
                advances_source = source_event_seq > current_source

            token_delta = int(raw_token_estimate) if advances_source else 0
            turn_delta = int(user_turns) if advances_source else 0
            connection.execute(
                """
                INSERT INTO scope_evolution_state(
                    tenant_id, scope_name, source_event_seq, conflict_generation,
                    source_raw_token_estimate, source_user_turns,
                    dirty_since_at, index_dirty_since_at, last_ingest_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, scope_name) DO UPDATE SET
                    source_event_seq = MAX(scope_evolution_state.source_event_seq, excluded.source_event_seq),
                    conflict_generation = MAX(scope_evolution_state.conflict_generation, excluded.conflict_generation),
                    source_raw_token_estimate = scope_evolution_state.source_raw_token_estimate + excluded.source_raw_token_estimate,
                    source_user_turns = scope_evolution_state.source_user_turns + excluded.source_user_turns,
                    dirty_since_at = CASE
                        WHEN excluded.source_event_seq>scope_evolution_state.source_event_seq
                        THEN COALESCE(scope_evolution_state.dirty_since_at, excluded.dirty_since_at)
                        ELSE scope_evolution_state.dirty_since_at END,
                    index_dirty_since_at = CASE
                        WHEN excluded.source_event_seq>scope_evolution_state.source_event_seq
                        THEN COALESCE(scope_evolution_state.index_dirty_since_at, excluded.index_dirty_since_at)
                        ELSE scope_evolution_state.index_dirty_since_at END,
                    last_ingest_at = MAX(COALESCE(scope_evolution_state.last_ingest_at, 0), excluded.last_ingest_at),
                    updated_at = excluded.updated_at
                """,
                (
                    tenant_id,
                    scope_name,
                    source_event_seq,
                    conflict_generation,
                    token_delta,
                    turn_delta,
                    now if advances_source else None,
                    now if advances_source else None,
                    now,
                    now,
                ),
            )
            if operation_id is not None:
                connection.execute(
                    """
                    INSERT INTO scope_ingest_watermark_commits(
                        tenant_id, scope_name, operation_id, source_event_seq,
                        new_message_count, raw_token_estimate, user_turns, committed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant_id,
                        scope_name,
                        operation_id,
                        source_event_seq,
                        int(new_message_count or 0),
                        int(raw_token_estimate),
                        int(user_turns),
                        now,
                    ),
                )
            row = connection.execute(
                "SELECT * FROM scope_evolution_state WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
        return self._evolution_row(row) or {}

    def record_committed_source_records(
        self,
        tenant_id: str,
        scope_name: str,
        operation_id: str,
        source_records: Sequence[Mapping[str, Any]],
        *,
        conflict_generation: int = 0,
        ingested_at: float | None = None,
        required_failed_job_id: str | None = None,
        required_failed_stage_id: str | None = None,
        required_failed_stage_attempt: int | None = None,
    ) -> dict[str, object]:
        """Account immutable Sources once, independent of job replays.

        A Writer attempt may terminate after only part of its input crossed the
        Source durability boundary. Source identity, rather than the attempt's
        message count, is therefore the only safe increment key.
        """

        self._validate_scope(tenant_id, scope_name)
        if not operation_id or not operation_id.strip():
            raise ValueError("operation_id is required")
        if conflict_generation < 0:
            raise ValueError("conflict_generation must be non-negative")
        guarded_recovery = required_failed_job_id is not None
        if guarded_recovery:
            required_failed_job_id = str(required_failed_job_id or "").strip()
            required_failed_stage_id = str(required_failed_stage_id or "").strip()
            if (
                not required_failed_job_id
                or not required_failed_stage_id
                or required_failed_stage_attempt is None
                or int(required_failed_stage_attempt) <= 0
            ):
                raise ValueError("failed Writer recovery identity is invalid")
            required_failed_stage_attempt = int(required_failed_stage_attempt)
        elif (
            required_failed_stage_id is not None
            or required_failed_stage_attempt is not None
        ):
            raise ValueError("failed Writer recovery identity must be complete")
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in source_records:
            source_record_id = str(record.get("source_record_id") or "").strip()
            origin_operation_id = str(
                record.get("origin_operation_id") or operation_id
            ).strip()
            raw_token_estimate = int(record.get("raw_token_estimate", 0) or 0)
            user_turns = int(record.get("user_turns", 0) or 0)
            if not source_record_id or not origin_operation_id:
                raise ValueError("source and origin operation identities are required")
            if source_record_id in seen:
                raise ValueError("source record IDs must be unique within an operation")
            if raw_token_estimate < 0 or user_turns not in {0, 1}:
                raise ValueError("source token and user-turn metrics are invalid")
            seen.add(source_record_id)
            normalized.append(
                {
                    "source_record_id": source_record_id,
                    "origin_operation_id": origin_operation_id,
                    "raw_token_estimate": raw_token_estimate,
                    "user_turns": user_turns,
                }
            )
        normalized.sort(key=lambda item: item["source_record_id"])
        encoded = json.dumps(
            normalized, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        source_set_sha256 = hashlib.sha256(encoded).hexdigest()
        now = time.time() if ingested_at is None else float(ingested_at)

        with self.transaction() as connection:
            if guarded_recovery:
                recovery_owner = connection.execute(
                    "SELECT jobs.state AS job_state,stages.state AS stage_state,"
                    "stages.attempt AS stage_attempt "
                    "FROM jobs JOIN operation_stages AS stages "
                    "ON stages.job_id=jobs.job_id "
                    "WHERE jobs.job_id=? AND jobs.tenant_id=? AND jobs.scope_name=? "
                    "AND stages.stage_id=? AND stages.stage_name='writer'",
                    (
                        required_failed_job_id,
                        tenant_id,
                        scope_name,
                        required_failed_stage_id,
                    ),
                ).fetchone()
                if (
                    recovery_owner is None
                    or str(recovery_owner["job_state"] or "") != "failed"
                    or str(recovery_owner["stage_state"] or "") != "failed"
                    or int(recovery_owner["stage_attempt"] or 0)
                    != required_failed_stage_attempt
                ):
                    raise StaleSourceAccountingRecovery(
                        "failed Writer recovery plan is stale"
                    )
                scope_has_live_work = connection.execute(
                    "SELECT 1 FROM jobs WHERE tenant_id=? AND scope_name=? "
                    "AND state='running' LIMIT 1",
                    (tenant_id, scope_name),
                ).fetchone()
                scope_has_live_stage = connection.execute(
                    "SELECT 1 FROM operation_stages WHERE tenant_id=? "
                    "AND scope_name=? AND state='running' LIMIT 1",
                    (tenant_id, scope_name),
                ).fetchone()
                if scope_has_live_work is not None or scope_has_live_stage is not None:
                    raise StaleSourceAccountingRecovery(
                        "failed Writer recovery scope has live work"
                    )
                lifecycle = connection.execute(
                    "SELECT state FROM scope_lifecycle "
                    "WHERE tenant_id=? AND scope_name=?",
                    (tenant_id, scope_name),
                ).fetchone()
                if lifecycle is not None and str(lifecycle["state"] or "") != "active":
                    raise StaleSourceAccountingRecovery(
                        "failed Writer recovery scope is not active"
                    )
                content_deletion = connection.execute(
                    "SELECT 1 FROM content_deletions "
                    "WHERE tenant_id=? AND scope_name=? "
                    "AND state IN ('requested','purging','reindexing','failed') "
                    "LIMIT 1",
                    (tenant_id, scope_name),
                ).fetchone()
                if content_deletion is not None:
                    raise StaleSourceAccountingRecovery(
                        "failed Writer recovery scope has an active content deletion"
                    )
            source_set = connection.execute(
                "SELECT * FROM scope_ingest_source_sets "
                "WHERE tenant_id=? AND scope_name=? AND operation_id=?",
                (tenant_id, scope_name, operation_id),
            ).fetchone()
            operation_commit = connection.execute(
                "SELECT * FROM scope_ingest_watermark_commits "
                "WHERE tenant_id=? AND scope_name=? AND operation_id=?",
                (tenant_id, scope_name, operation_id),
            ).fetchone()
            if source_set is not None or operation_commit is not None:
                if source_set is None or operation_commit is None:
                    raise ValueError("source accounting operation is only partially committed")
                if (
                    str(source_set["source_set_sha256"]) != source_set_sha256
                    or int(source_set["source_count"]) != len(normalized)
                ):
                    raise ValueError("source accounting replay changed its immutable set")
                row = connection.execute(
                    "SELECT * FROM scope_evolution_state "
                    "WHERE tenant_id=? AND scope_name=?",
                    (tenant_id, scope_name),
                ).fetchone()
                return self._evolution_row(row) or {}

            new_count = 0
            token_delta = 0
            turn_delta = 0
            for record in normalized:
                existing = connection.execute(
                    "SELECT * FROM scope_source_event_commits "
                    "WHERE tenant_id=? AND scope_name=? AND source_record_id=?",
                    (tenant_id, scope_name, record["source_record_id"]),
                ).fetchone()
                if existing is not None:
                    expected = (
                        record["origin_operation_id"],
                        record["raw_token_estimate"],
                        record["user_turns"],
                    )
                    actual = (
                        str(existing["origin_operation_id"]),
                        int(existing["raw_token_estimate"]),
                        int(existing["user_turns"]),
                    )
                    if actual != expected:
                        raise ValueError("committed Source accounting metadata changed")
                    continue
                connection.execute(
                    """
                    INSERT INTO scope_source_event_commits(
                        tenant_id,scope_name,source_record_id,origin_operation_id,
                        accounting_operation_id,raw_token_estimate,user_turns,committed_at
                    ) VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (
                        tenant_id,
                        scope_name,
                        record["source_record_id"],
                        record["origin_operation_id"],
                        operation_id,
                        record["raw_token_estimate"],
                        record["user_turns"],
                        now,
                    ),
                )
                new_count += 1
                token_delta += int(record["raw_token_estimate"])
                turn_delta += int(record["user_turns"])

            current = connection.execute(
                "SELECT * FROM scope_evolution_state "
                "WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            current_source = 0 if current is None else int(current["source_event_seq"])
            source_event_seq = current_source + new_count
            connection.execute(
                """
                INSERT INTO scope_evolution_state(
                    tenant_id,scope_name,source_event_seq,conflict_generation,
                    source_raw_token_estimate,source_user_turns,
                    dirty_since_at,index_dirty_since_at,last_ingest_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(tenant_id,scope_name) DO UPDATE SET
                    source_event_seq=excluded.source_event_seq,
                    conflict_generation=MAX(
                        scope_evolution_state.conflict_generation,
                        excluded.conflict_generation
                    ),
                    source_raw_token_estimate=
                        scope_evolution_state.source_raw_token_estimate
                        + excluded.source_raw_token_estimate,
                    source_user_turns=scope_evolution_state.source_user_turns
                        + excluded.source_user_turns,
                    dirty_since_at=CASE WHEN excluded.source_event_seq>
                        scope_evolution_state.source_event_seq THEN COALESCE(
                            scope_evolution_state.dirty_since_at,
                            excluded.dirty_since_at
                        ) ELSE scope_evolution_state.dirty_since_at END,
                    index_dirty_since_at=CASE WHEN excluded.source_event_seq>
                        scope_evolution_state.source_event_seq THEN COALESCE(
                            scope_evolution_state.index_dirty_since_at,
                            excluded.index_dirty_since_at
                        ) ELSE scope_evolution_state.index_dirty_since_at END,
                    last_ingest_at=MAX(
                        COALESCE(scope_evolution_state.last_ingest_at,0),
                        excluded.last_ingest_at
                    ),
                    updated_at=excluded.updated_at
                """,
                (
                    tenant_id,
                    scope_name,
                    source_event_seq,
                    conflict_generation,
                    token_delta,
                    turn_delta,
                    now if new_count else None,
                    now if new_count else None,
                    now,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO scope_ingest_source_sets VALUES(?,?,?,?,?,?)",
                (
                    tenant_id,
                    scope_name,
                    operation_id,
                    source_set_sha256,
                    len(normalized),
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO scope_ingest_watermark_commits(
                    tenant_id,scope_name,operation_id,source_event_seq,
                    new_message_count,raw_token_estimate,user_turns,committed_at
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    tenant_id,
                    scope_name,
                    operation_id,
                    source_event_seq,
                    new_count,
                    token_delta,
                    turn_delta,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM scope_evolution_state "
                "WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
        return self._evolution_row(row) or {}

    def list_due_scopes(
        self,
        *,
        dirty_threshold: int | None = None,
        dirty_token_threshold: int = 32_000,
        dirty_user_turn_threshold: int = 64,
        max_age_seconds: float | None = None,
        min_token_threshold: int = 4_000,
        min_user_turn_threshold: int = 8,
        min_success_interval_seconds: float = 1_800.0,
        now: float | None = None,
        include_conflicts: bool = False,
    ) -> list[dict[str, object]]:
        """List scopes eligible for one batched slow-graph promotion."""
        thresholds = (
            dirty_token_threshold,
            dirty_user_turn_threshold,
            min_token_threshold,
            min_user_turn_threshold,
        )
        if any(value < 1 for value in thresholds):
            raise ValueError("slow token and turn thresholds must be positive")
        if dirty_threshold is not None and dirty_threshold < 1:
            raise ValueError("dirty_threshold must be positive when provided")
        if max_age_seconds is not None and max_age_seconds < 0:
            raise ValueError("max_age_seconds must be non-negative")
        if min_success_interval_seconds < 0:
            raise ValueError("min_success_interval_seconds must be non-negative")
        moment = time.time() if now is None else float(now)
        with self.transaction(immediate=False) as connection:
            rows = connection.execute(
                "SELECT evolution.* FROM scope_evolution_state AS evolution "
                "LEFT JOIN scope_lifecycle AS lifecycle "
                "ON lifecycle.tenant_id=evolution.tenant_id "
                "AND lifecycle.scope_name=evolution.scope_name "
                "LEFT JOIN scope_quarantines AS quarantine "
                "ON quarantine.tenant_id=evolution.tenant_id "
                "AND quarantine.scope_name=evolution.scope_name "
                "LEFT JOIN content_deletions AS content_deletion "
                "ON content_deletion.tenant_id=evolution.tenant_id "
                "AND content_deletion.scope_name=evolution.scope_name "
                "AND content_deletion.state IN "
                "('requested','purging','reindexing','failed') "
                "WHERE quarantine.tenant_id IS NULL "
                "AND content_deletion.deletion_id IS NULL "
                "AND (lifecycle.state IS NULL OR lifecycle.state='active') "
                "ORDER BY evolution.tenant_id, evolution.scope_name"
            ).fetchall()
        due: list[dict[str, object]] = []
        for row in rows:
            item = self._evolution_row(row) or {}
            dirty_events = int(item["source_event_seq"]) - int(item["promoted_event_seq"])
            dirty_tokens = int(item["source_raw_token_estimate"]) - int(
                item["promoted_raw_token_estimate"]
            )
            dirty_turns = int(item["source_user_turns"]) - int(item["promoted_user_turns"])
            conflict = int(item["conflict_generation"]) > int(item["promoted_conflict_generation"])
            dirty_since = item["dirty_since_at"]
            last_success = item["last_slow_success_at"]
            age = None if dirty_since is None else max(0.0, moment - float(dirty_since))
            cooldown_remaining = 0.0
            if last_success is not None:
                cooldown_remaining = max(
                    0.0,
                    min_success_interval_seconds - max(0.0, moment - float(last_success)),
                )
            if cooldown_remaining > 0:
                continue
            batch_due = (
                dirty_tokens >= dirty_token_threshold
                or dirty_turns >= dirty_user_turn_threshold
            )
            aged = (
                max_age_seconds is not None
                and dirty_events > 0
                and age is not None
                and age >= max_age_seconds
                and (
                    dirty_tokens >= min_token_threshold
                    or dirty_turns >= min_user_turn_threshold
                )
            )
            reasons: list[str] = []
            if batch_due:
                reasons.append("batch_threshold")
            if dirty_threshold is not None and dirty_events >= dirty_threshold:
                reasons.append("dirty_threshold")
            if aged:
                reasons.append("max_age")
            if include_conflicts and conflict:
                reasons.append("conflict")
            if reasons:
                item["dirty_events"] = dirty_events
                item["dirty_raw_token_estimate"] = dirty_tokens
                item["dirty_user_turns"] = dirty_turns
                item["age_seconds"] = age
                item["cooldown_remaining_seconds"] = cooldown_remaining
                item["due_reasons"] = tuple(reasons)
                due.append(item)
        return due

    def list_due_index_scopes(
        self,
        *,
        dirty_threshold: int = 1,
        max_age_seconds: float | None = None,
        now: float | None = None,
    ) -> list[dict[str, object]]:
        """List coalesced index work without treating every ingest as immediately due."""
        if dirty_threshold < 1:
            raise ValueError("dirty_threshold must be positive")
        if max_age_seconds is not None and max_age_seconds < 0:
            raise ValueError("max_age_seconds must be non-negative")
        moment = time.time() if now is None else float(now)
        with self.transaction(immediate=False) as connection:
            rows = connection.execute(
                "SELECT evolution.* FROM scope_evolution_state AS evolution "
                "LEFT JOIN scope_lifecycle AS lifecycle "
                "ON lifecycle.tenant_id=evolution.tenant_id "
                "AND lifecycle.scope_name=evolution.scope_name "
                "LEFT JOIN scope_quarantines AS quarantine "
                "ON quarantine.tenant_id=evolution.tenant_id "
                "AND quarantine.scope_name=evolution.scope_name "
                "LEFT JOIN content_deletions AS content_deletion "
                "ON content_deletion.tenant_id=evolution.tenant_id "
                "AND content_deletion.scope_name=evolution.scope_name "
                "AND content_deletion.state IN "
                "('requested','purging','reindexing','failed') "
                "WHERE quarantine.tenant_id IS NULL "
                "AND content_deletion.deletion_id IS NULL "
                "AND (lifecycle.state IS NULL OR lifecycle.state='active') "
                "ORDER BY evolution.tenant_id, evolution.scope_name"
            ).fetchall()
        due: list[dict[str, object]] = []
        for row in rows:
            item = self._evolution_row(row) or {}
            dirty_events = int(item["source_event_seq"]) - int(item["indexed_event_seq"])
            dirty_since = item["index_dirty_since_at"]
            age = None if dirty_since is None else max(0.0, moment - float(dirty_since))
            aged = (
                max_age_seconds is not None
                and dirty_events > 0
                and age is not None
                and age >= max_age_seconds
            )
            reasons: list[str] = []
            if dirty_events >= dirty_threshold:
                reasons.append("dirty_threshold")
            if aged:
                reasons.append("max_age")
            if reasons:
                item["dirty_events"] = dirty_events
                item["age_seconds"] = age
                item["due_reasons"] = tuple(reasons)
                due.append(item)
        return due

    def _reconcile_stale_scope_claim(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        scope_name: str,
        *,
        claim_kind: str,
        now: float,
    ) -> bool:
        id_column = f"active_{claim_kind}_job_id"
        version_column = f"active_{claim_kind}_job_version"
        state = connection.execute(
            "SELECT * FROM scope_evolution_state WHERE tenant_id=? AND scope_name=?",
            (tenant_id, scope_name),
        ).fetchone()
        if state is None or state[id_column] is None:
            return False
        active_job_id = str(state[id_column])
        active_version = state[version_column]
        active = connection.execute(
            "SELECT * FROM jobs WHERE job_id=?", (active_job_id,)
        ).fetchone()
        reason_code = None
        if active is None:
            reason_code = "stale_claim_job_missing"
        elif (
            str(active["tenant_id"]) != tenant_id
            or str(active["scope_name"]) != scope_name
        ):
            reason_code = "stale_claim_scope_mismatch"
        elif str(active["state"]) in {"succeeded", "failed", "cancelled"}:
            reason_code = "stale_claim_terminal_job"
        if reason_code is None:
            return False
        cursor = connection.execute(
            f"""
            UPDATE scope_evolution_state
            SET {id_column}=NULL,{version_column}=NULL,updated_at=?
            WHERE tenant_id=? AND scope_name=? AND {id_column}=?
              AND ({version_column} IS ? OR {version_column}=?)
            """,
            (
                now,
                tenant_id,
                scope_name,
                active_job_id,
                active_version,
                active_version,
            ),
        )
        if cursor.rowcount != 1:
            return False
        self._append_job_lifecycle_audit(
            connection,
            job_id=active_job_id,
            tenant_id=tenant_id,
            scope_name=scope_name,
            scope_seq=None if active is None else int(active["scope_seq"]),
            event_type="scope_claim_released",
            stage_name=f"{claim_kind}_claim",
            reason={
                "code": reason_code,
                "claim_kind": claim_kind,
                "claim_job_version": active_version,
            },
            from_state=None if active is None else str(active["state"]),
            to_state=None if active is None else str(active["state"]),
            created_at=now,
        )
        return True

    def reconcile_stale_scope_claims(self) -> dict[str, int]:
        """Release only orphaned or terminal scope claims after a restart."""
        now = time.time()
        released = {"evolution": 0, "index": 0}
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT tenant_id,scope_name FROM scope_evolution_state "
                "WHERE active_evolution_job_id IS NOT NULL "
                "OR active_index_job_id IS NOT NULL "
                "ORDER BY tenant_id,scope_name"
            ).fetchall()
            for row in rows:
                tenant_id = str(row["tenant_id"])
                scope_name = str(row["scope_name"])
                for claim_kind in ("evolution", "index"):
                    released[claim_kind] += int(
                        self._reconcile_stale_scope_claim(
                            connection,
                            tenant_id,
                            scope_name,
                            claim_kind=claim_kind,
                            now=now,
                        )
                    )
        return released

    def _claim_scope_job(
        self,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        *,
        claim_kind: str,
        job_version: int | None,
    ) -> bool:
        self._validate_scope(tenant_id, scope_name)
        if not job_id:
            raise ValueError("job_id is required")
        if claim_kind not in {"evolution", "index"}:
            raise ValueError("claim_kind must be evolution or index")
        now = time.time()
        id_column = f"active_{claim_kind}_job_id"
        version_column = f"active_{claim_kind}_job_version"

        # The scheduler proof can scan thousands of historical ingest jobs.
        # Compute it under a WAL read transaction so ordinary API writes,
        # heartbeats, and billing updates are never serialized behind that
        # scan. Reuse the same connection and fence the short write phase with
        # SQLite's per-connection data_version: if any other connection commits
        # after the proof snapshot, discard the proof and retry later.
        connection = self.connect()
        cursor: sqlite3.Cursor | None = None
        try:
            connection.execute("BEGIN")
            proof_data_version = int(
                connection.execute("PRAGMA data_version").fetchone()[0]
            )
            job = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                connection.rollback()
                return False
            if (
                str(job["tenant_id"]) != tenant_id
                or str(job["scope_name"]) != scope_name
                or str(job["state"]) not in {"pending", "running"}
            ):
                connection.rollback()
                return False
            current_version = int(job["version"])
            if job_version is not None and int(job_version) != current_version:
                connection.rollback()
                return False
            payload = json.loads(str(job["payload_json"]))
            job_type = str(payload.get("job_type") or "") if isinstance(payload, Mapping) else ""
            target_source_event_seq = (
                int(payload["target_source_event_seq"])
                if isinstance(payload, Mapping)
                and payload.get("target_source_event_seq") is not None
                else None
            )
            gate = self._scope_scheduler_gate(
                connection,
                tenant_id,
                scope_name,
                candidate_job_id=job_id,
                include_candidate_ingest=job_type == "ingest",
                target_source_event_seq=target_source_event_seq,
            )
            if not bool(gate["ready"]):
                connection.rollback()
                return False

            proven_scope_seq = int(job["scope_seq"])
            proven_state = str(job["state"])
            connection.commit()

            connection.execute("BEGIN IMMEDIATE")
            if int(connection.execute("PRAGMA data_version").fetchone()[0]) != proof_data_version:
                connection.rollback()
                return False
            current = connection.execute(
                "SELECT tenant_id,scope_name,state,version,scope_seq "
                "FROM jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if (
                current is None
                or str(current["tenant_id"]) != tenant_id
                or str(current["scope_name"]) != scope_name
                or str(current["state"]) != proven_state
                or int(current["version"]) != current_version
                or int(current["scope_seq"]) != proven_scope_seq
            ):
                connection.rollback()
                return False
            connection.execute(
                """
                INSERT INTO scope_evolution_state(tenant_id,scope_name,updated_at)
                VALUES(?,?,?) ON CONFLICT(tenant_id,scope_name) DO NOTHING
                """,
                (tenant_id, scope_name, now),
            )
            self._reconcile_stale_scope_claim(
                connection,
                tenant_id,
                scope_name,
                claim_kind=claim_kind,
                now=now,
            )
            cursor = connection.execute(
                f"""
                UPDATE scope_evolution_state
                SET {id_column}=?,{version_column}=?,updated_at=?
                WHERE tenant_id=? AND scope_name=?
                  AND ({id_column} IS NULL OR {id_column}=?)
                """,
                (
                    job_id,
                    current_version,
                    now,
                    tenant_id,
                    scope_name,
                    job_id,
                ),
            )
            if cursor.rowcount == 1:
                self._append_job_lifecycle_audit(
                    connection,
                    job_id=job_id,
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    scope_seq=proven_scope_seq,
                    event_type="scope_claim_acquired",
                    stage_name=f"{claim_kind}_claim",
                    reason={
                        "code": "scope_claim_acquired",
                        "claim_kind": claim_kind,
                        "job_version": current_version,
                    },
                    from_state=proven_state,
                    to_state=proven_state,
                    created_at=now,
                )
            connection.commit()
            return cursor.rowcount == 1
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def claim_evolution_job(
        self,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        *,
        job_version: int | None = None,
    ) -> bool:
        return self._claim_scope_job(
            tenant_id,
            scope_name,
            job_id,
            claim_kind="evolution",
            job_version=job_version,
        )

    def release_evolution_job(
        self,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        *,
        job_version: int | None = None,
        reason: Mapping[str, Any] | None = None,
    ) -> bool:
        self._validate_scope(tenant_id, scope_name)
        now = time.time()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE scope_evolution_state
                SET active_evolution_job_id=NULL,active_evolution_job_version=NULL,updated_at=?
                WHERE tenant_id=? AND scope_name=? AND active_evolution_job_id=?
                  AND (? IS NULL OR active_evolution_job_version=?)
                """,
                (now, tenant_id, scope_name, job_id, job_version, job_version),
            )
            if cursor.rowcount == 1:
                job = connection.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                self._append_job_lifecycle_audit(
                    connection,
                    job_id=job_id,
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    scope_seq=None if job is None else int(job["scope_seq"]),
                    event_type="scope_claim_released",
                    stage_name="evolution_claim",
                    reason=dict(reason or {"code": "scope_claim_released", "claim_kind": "evolution"}),
                    from_state=None if job is None else str(job["state"]),
                    to_state=None if job is None else str(job["state"]),
                    created_at=now,
                )
        return cursor.rowcount == 1

    def claim_index_job(
        self,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        *,
        job_version: int | None = None,
    ) -> bool:
        """Claim the one coalesced index rebuild allowed for a scope."""
        return self._claim_scope_job(
            tenant_id,
            scope_name,
            job_id,
            claim_kind="index",
            job_version=job_version,
        )

    def release_index_job(
        self,
        tenant_id: str,
        scope_name: str,
        job_id: str,
        *,
        job_version: int | None = None,
        reason: Mapping[str, Any] | None = None,
    ) -> bool:
        self._validate_scope(tenant_id, scope_name)
        now = time.time()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE scope_evolution_state
                SET active_index_job_id=NULL,active_index_job_version=NULL,updated_at=?
                WHERE tenant_id=? AND scope_name=? AND active_index_job_id=?
                  AND (? IS NULL OR active_index_job_version=?)
                """,
                (now, tenant_id, scope_name, job_id, job_version, job_version),
            )
            if cursor.rowcount == 1:
                job = connection.execute(
                    "SELECT * FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                self._append_job_lifecycle_audit(
                    connection,
                    job_id=job_id,
                    tenant_id=tenant_id,
                    scope_name=scope_name,
                    scope_seq=None if job is None else int(job["scope_seq"]),
                    event_type="scope_claim_released",
                    stage_name="index_claim",
                    reason=dict(reason or {"code": "scope_claim_released", "claim_kind": "index"}),
                    from_state=None if job is None else str(job["state"]),
                    to_state=None if job is None else str(job["state"]),
                    created_at=now,
                )
        return cursor.rowcount == 1

    def advance_index_watermark(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        indexed_event_seq: int,
        index_succeeded: bool = True,
        index_job_id: str | None = None,
        index_job_version: int | None = None,
        succeeded_at: float | None = None,
    ) -> dict[str, object]:
        """Advance the immutable base watermark only after activation succeeds."""
        self._validate_scope(tenant_id, scope_name)
        if not index_succeeded:
            raise ValueError("index watermark requires successful index activation")
        if indexed_event_seq < 0:
            raise ValueError("indexed_event_seq must be non-negative")
        now = time.time() if succeeded_at is None else float(succeeded_at)
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM scope_evolution_state WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            if row is None:
                raise KeyError((tenant_id, scope_name))
            if index_job_id is not None and row["active_index_job_id"] != index_job_id:
                raise ValueError("index job does not own this scope")
            if (
                index_job_version is not None
                and row["active_index_job_version"] != index_job_version
            ):
                raise ValueError("index job attempt does not own this scope")
            if indexed_event_seq > int(row["source_event_seq"]):
                raise ValueError("cannot index events that are not committed")
            connection.execute(
                """
                UPDATE scope_evolution_state
                SET indexed_event_seq=MAX(indexed_event_seq, ?),
                    delta_indexed_event_seq=MAX(delta_indexed_event_seq, ?),
                    last_index_success_at=?,
                    index_dirty_since_at=CASE
                        WHEN MAX(indexed_event_seq, ?) >= source_event_seq THEN NULL
                        ELSE index_dirty_since_at END,
                    active_index_job_id=CASE WHEN ? IS NULL OR active_index_job_id=?
                        THEN NULL ELSE active_index_job_id END,
                    active_index_job_version=CASE
                        WHEN ? IS NULL OR active_index_job_version=?
                        THEN NULL ELSE active_index_job_version END,
                    updated_at=?
                WHERE tenant_id=? AND scope_name=?
                """,
                (
                    indexed_event_seq,
                    indexed_event_seq,
                    now,
                    indexed_event_seq,
                    index_job_id,
                    index_job_id,
                    index_job_version,
                    index_job_version,
                    now,
                    tenant_id,
                    scope_name,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM scope_evolution_state WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            return self._evolution_row(updated) or {}

    def advance_delta_index_watermark(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        delta_indexed_event_seq: int,
        index_job_id: str | None = None,
        index_job_version: int | None = None,
        succeeded_at: float | None = None,
    ) -> dict[str, object]:
        """Advance the cumulative base-plus-delta searchable watermark."""

        self._validate_scope(tenant_id, scope_name)
        if delta_indexed_event_seq < 0:
            raise ValueError("delta_indexed_event_seq must be non-negative")
        now = time.time() if succeeded_at is None else float(succeeded_at)
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM scope_evolution_state WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            if row is None:
                raise KeyError((tenant_id, scope_name))
            if index_job_id is not None and row["active_index_job_id"] != index_job_id:
                raise ValueError("index job does not own this scope")
            if (
                index_job_version is not None
                and row["active_index_job_version"] != index_job_version
            ):
                raise ValueError("index job attempt does not own this scope")
            if delta_indexed_event_seq > int(row["source_event_seq"]):
                raise ValueError("cannot index events that are not committed")
            if delta_indexed_event_seq < int(row["indexed_event_seq"]):
                raise ValueError("delta watermark cannot precede the active base")
            connection.execute(
                """
                UPDATE scope_evolution_state
                SET delta_indexed_event_seq=MAX(delta_indexed_event_seq, ?),
                    last_delta_index_success_at=?,
                    active_index_job_id=CASE WHEN ? IS NULL OR active_index_job_id=?
                        THEN NULL ELSE active_index_job_id END,
                    active_index_job_version=CASE
                        WHEN ? IS NULL OR active_index_job_version=?
                        THEN NULL ELSE active_index_job_version END,
                    updated_at=?
                WHERE tenant_id=? AND scope_name=?
                """,
                (
                    delta_indexed_event_seq,
                    now,
                    index_job_id,
                    index_job_id,
                    index_job_version,
                    index_job_version,
                    now,
                    tenant_id,
                    scope_name,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM scope_evolution_state WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            return self._evolution_row(updated) or {}

    def advance_promoted_watermarks(
        self,
        tenant_id: str,
        scope_name: str,
        *,
        source_event_seq: int,
        conflict_generation: int,
        slow_succeeded: bool,
        index_activated: bool,
        raw_token_estimate: int | None = None,
        user_turns: int | None = None,
        evolution_job_id: str | None = None,
        evolution_job_version: int | None = None,
        succeeded_at: float | None = None,
        spent_cost_micro_cny: int = 0,
    ) -> dict[str, object]:
        """Advance promotion only after both durable activation steps succeeded."""
        self._validate_scope(tenant_id, scope_name)
        if not slow_succeeded or not index_activated:
            raise ValueError("promotion requires successful slow and index activation")
        if source_event_seq < 0 or conflict_generation < 0 or spent_cost_micro_cny < 0:
            raise ValueError("watermarks and cost must be non-negative")
        if raw_token_estimate is not None and raw_token_estimate < 0:
            raise ValueError("raw_token_estimate must be non-negative")
        if user_turns is not None and user_turns < 0:
            raise ValueError("user_turns must be non-negative")
        now = time.time() if succeeded_at is None else float(succeeded_at)
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM scope_evolution_state WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
            if row is None:
                raise KeyError((tenant_id, scope_name))
            if evolution_job_id is not None and row["active_evolution_job_id"] != evolution_job_id:
                raise ValueError("evolution job does not own this scope")
            if (
                evolution_job_version is not None
                and row["active_evolution_job_version"] != evolution_job_version
            ):
                raise ValueError("evolution job attempt does not own this scope")
            if source_event_seq > int(row["source_event_seq"]):
                raise ValueError("cannot promote events that are not committed")
            if conflict_generation > int(row["conflict_generation"]):
                raise ValueError("cannot promote conflicts that are not committed")
            promoted_tokens = (
                int(row["source_raw_token_estimate"])
                if raw_token_estimate is None and source_event_seq == int(row["source_event_seq"])
                else int(raw_token_estimate or 0)
            )
            promoted_turns = (
                int(row["source_user_turns"])
                if user_turns is None and source_event_seq == int(row["source_event_seq"])
                else int(user_turns or 0)
            )
            if promoted_tokens > int(row["source_raw_token_estimate"]):
                raise ValueError("cannot promote token estimates that are not committed")
            if promoted_turns > int(row["source_user_turns"]):
                raise ValueError("cannot promote user turns that are not committed")
            connection.execute(
                """
                UPDATE scope_evolution_state
                SET promoted_event_seq=MAX(promoted_event_seq, ?),
                    promoted_conflict_generation=MAX(promoted_conflict_generation, ?),
                    promoted_raw_token_estimate=MAX(promoted_raw_token_estimate, ?),
                    promoted_user_turns=MAX(promoted_user_turns, ?),
                    indexed_event_seq=CASE WHEN ? THEN MAX(indexed_event_seq, ?) ELSE indexed_event_seq END,
                    delta_indexed_event_seq=CASE WHEN ? THEN MAX(delta_indexed_event_seq, ?) ELSE delta_indexed_event_seq END,
                    last_slow_success_at=?,
                    last_index_success_at=CASE WHEN ? THEN ? ELSE last_index_success_at END,
                    dirty_since_at=CASE
                        WHEN MAX(promoted_event_seq, ?) >= source_event_seq
                         AND MAX(promoted_raw_token_estimate, ?) >= source_raw_token_estimate
                         AND MAX(promoted_user_turns, ?) >= source_user_turns
                        THEN NULL ELSE dirty_since_at END,
                    index_dirty_since_at=CASE
                        WHEN ? AND MAX(indexed_event_seq, ?) >= source_event_seq
                        THEN NULL ELSE index_dirty_since_at END,
                    active_evolution_job_id=CASE WHEN ? IS NULL OR active_evolution_job_id=?
                        THEN NULL ELSE active_evolution_job_id END,
                    active_evolution_job_version=CASE
                        WHEN ? IS NULL OR active_evolution_job_version=?
                        THEN NULL ELSE active_evolution_job_version END,
                    reserved_cost_micro_cny=MAX(0, reserved_cost_micro_cny-?),
                    spent_cost_micro_cny=spent_cost_micro_cny+?,
                    updated_at=?
                WHERE tenant_id=? AND scope_name=?
                """,
                (
                    source_event_seq,
                    conflict_generation,
                    promoted_tokens,
                    promoted_turns,
                    index_activated,
                    source_event_seq,
                    index_activated,
                    source_event_seq,
                    now,
                    index_activated,
                    now,
                    source_event_seq,
                    promoted_tokens,
                    promoted_turns,
                    index_activated,
                    source_event_seq,
                    evolution_job_id,
                    evolution_job_id,
                    evolution_job_version,
                    evolution_job_version,
                    spent_cost_micro_cny,
                    spent_cost_micro_cny,
                    now,
                    tenant_id,
                    scope_name,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM scope_evolution_state WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
        return self._evolution_row(updated) or {}

    def reserve_evolution_cost(
        self, tenant_id: str, scope_name: str, amount_micro_cny: int
    ) -> dict[str, object]:
        self._validate_scope(tenant_id, scope_name)
        if amount_micro_cny < 0:
            raise ValueError("amount_micro_cny must be non-negative")
        now = time.time()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO scope_evolution_state(tenant_id, scope_name, reserved_cost_micro_cny, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tenant_id, scope_name) DO UPDATE SET
                    reserved_cost_micro_cny=scope_evolution_state.reserved_cost_micro_cny+excluded.reserved_cost_micro_cny,
                    updated_at=excluded.updated_at
                """,
                (tenant_id, scope_name, amount_micro_cny, now),
            )
            row = connection.execute(
                "SELECT * FROM scope_evolution_state WHERE tenant_id=? AND scope_name=?",
                (tenant_id, scope_name),
            ).fetchone()
        return self._evolution_row(row) or {}
