#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmcra_service.auth import APIKeyAuth
from tmcra_service.cli import DEFAULT_SCOPES
from tmcra_service.control_db import ControlDB


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the commercial API contract")
    parser.add_argument("--base-url", default="http://127.0.0.1:2009")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(
            os.getenv(
                "TMCRA_SERVICE_CONTROL_DB",
                "/opt/tmcra/tmcra_service_state/control.sqlite3",
            )
        ),
    )
    parser.add_argument("--tenant-id", default="deploy-contract-smoke")
    return parser


def _cleanup_previous(database: ControlDB, auth: APIKeyAuth, tenant_id: str) -> None:
    now = time.time()
    with database.transaction() as connection:
        key_ids = [
            str(row[0])
            for row in connection.execute(
                "SELECT key_id FROM api_keys WHERE tenant_id = ? AND revoked_at IS NULL",
                (tenant_id,),
            ).fetchall()
        ]
        connection.execute(
            "UPDATE scope_tokens SET revoked_at = COALESCE(revoked_at, ?) "
            "WHERE tenant_id = ?",
            (now, tenant_id),
        )
    for key_id in key_ids:
        auth.revoke_key(key_id)


def run(base_url: str, database_path: Path, tenant_id: str) -> dict[str, Any]:
    database = ControlDB(database_path.resolve())
    auth = APIKeyAuth(database)
    _cleanup_previous(database, auth, tenant_id)
    auth.set_tenant_scopes(tenant_id, frozenset(DEFAULT_SCOPES))
    issued = auth.create_key(tenant_id)
    scope = f"{tenant_id}-scope"
    statuses: dict[str, int] = {}
    try:
        root_headers = {"Authorization": f"Bearer {issued.api_key}"}
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=20.0) as client:
            for path in ("/healthz", "/readyz"):
                response = client.get(path)
                response.raise_for_status()
                statuses[path] = response.status_code

            response = client.post(
                "/v1/access-tokens",
                headers=root_headers,
                json={
                    "label": "deploy smoke",
                    "subject": "deployment-verifier",
                    "permissions": ["memory:read"],
                    "scope_names": [scope],
                    "expires_in_seconds": 900,
                },
            )
            response.raise_for_status()
            scoped = response.json()
            statuses["scope_token_create"] = response.status_code

            denied = client.get(
                "/v1/access-tokens",
                headers={"Authorization": f"Bearer {scoped['access_token']}"},
            )
            if denied.status_code != 403:
                raise RuntimeError(
                    f"terminal scoped token received unexpected status {denied.status_code}"
                )
            statuses["terminal_token_admin_denied"] = denied.status_code

            listed = client.get("/v1/access-tokens", headers=root_headers)
            listed.raise_for_status()
            if not any(
                item.get("token_id") == scoped["token_id"] for item in listed.json()
            ):
                raise RuntimeError("issued scoped token was absent from the token inventory")
            statuses["scope_token_list"] = listed.status_code

            revoked = client.delete(
                f"/v1/access-tokens/{scoped['token_id']}", headers=root_headers
            )
            revoked.raise_for_status()
            statuses["scope_token_revoke"] = revoked.status_code

            retention = client.put(
                f"/v1/scopes/{scope}/retention",
                headers=root_headers,
                json={"enabled": False, "inactive_days": 365},
            )
            retention.raise_for_status()
            statuses["retention"] = retention.status_code

            feedback = client.post(
                f"/v1/scopes/{scope}/feedback",
                headers=root_headers,
                json={
                    "rating": "helpful",
                    "memory_ids": [],
                    "metadata": {"surface": "deploy_smoke"},
                },
            )
            feedback.raise_for_status()
            statuses["feedback"] = feedback.status_code

            invalid_batch = client.post(
                f"/v1/scopes/{scope}/ingest/batch",
                headers={
                    **root_headers,
                    "Idempotency-Key": f"smoke-{uuid.uuid4()}",
                },
                json={"items": []},
            )
            if invalid_batch.status_code != 422:
                raise RuntimeError(
                    f"empty batch received unexpected status {invalid_batch.status_code}"
                )
            statuses["batch_validation"] = invalid_batch.status_code
    finally:
        auth.revoke_key(issued.key_id)
        with database.transaction() as connection:
            connection.execute(
                "UPDATE scope_tokens SET revoked_at = COALESCE(revoked_at, ?) "
                "WHERE tenant_id = ?",
                (time.time(), tenant_id),
            )

    return {"commercial_contract_smoke": "passed", "statuses": statuses}


def main() -> int:
    args = _parser().parse_args()
    print(json.dumps(run(args.base_url, args.database, args.tenant_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
