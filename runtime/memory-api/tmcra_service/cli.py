from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .auth import APIKeyAuth
from .control_db import ControlDB


DEFAULT_SCOPES = (
    "memory:read",
    "memory:write",
    "memory:consolidate",
    "memory:feedback",
    "memory:export",
    "memory:delete",
    "tokens:manage",
    "webhooks:manage",
    "retention:manage",
)


def _database(value: str | None) -> Path:
    raw = value or os.getenv(
        "TMCRA_SERVICE_CONTROL_DB", "/opt/tmcra/tmcra_service_state/control.sqlite3"
    )
    return Path(raw).resolve()


CLIENT_COMMANDS = frozenset({"recall", "ingest", "job", "turn"})
CLIENT_OPTION_VALUES = frozenset({"--base-url", "--api-key-env", "--request-timeout"})


def _is_client_invocation(argv: list[str]) -> bool:
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in CLIENT_COMMANDS:
            return True
        if token in CLIENT_OPTION_VALUES:
            index += 2
        else:
            index += 1
    return False


def main(argv: list[str] | None = None) -> int:
    os.umask(0o077)
    argv = list(sys.argv[1:] if argv is None else argv)
    if _is_client_invocation(argv):
        from .client_cli import main as client_main

        return client_main(argv)
    parser = argparse.ArgumentParser(description="TMCRA production service administration")
    parser.add_argument("--database")
    sub = parser.add_subparsers(dest="command", required=True)
    tenant = sub.add_parser("tenant-create")
    tenant.add_argument("--tenant-id", required=True)
    tenant.add_argument("--scopes", default=",".join(DEFAULT_SCOPES))
    tenant.add_argument("--no-key", action="store_true")
    issue = sub.add_parser("key-issue")
    issue.add_argument("--tenant-id", required=True)
    issue.add_argument("--scopes", default="")
    revoke = sub.add_parser("key-revoke")
    revoke.add_argument("--key-id", required=True)
    status = sub.add_parser("status")
    args = parser.parse_args(argv)

    database = ControlDB(_database(args.database))
    auth = APIKeyAuth(database)
    if args.command == "tenant-create":
        scopes = frozenset(item.strip() for item in args.scopes.split(",") if item.strip())
        auth.set_tenant_scopes(args.tenant_id, scopes)
        result: dict[str, object] = {
            "tenant_id": args.tenant_id,
            "scopes": sorted(scopes),
        }
        if not args.no_key:
            issued = auth.create_key(args.tenant_id)
            result.update({"key_id": issued.key_id, "api_key": issued.api_key})
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "key-issue":
        scopes = (
            frozenset(item.strip() for item in args.scopes.split(",") if item.strip())
            if args.scopes
            else None
        )
        issued = auth.create_key(args.tenant_id, scopes)
        print(
            json.dumps(
                {
                    "tenant_id": issued.tenant_id,
                    "key_id": issued.key_id,
                    "api_key": issued.api_key,
                    "scopes": sorted(issued.scopes),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "key-revoke":
        print(json.dumps({"revoked": auth.revoke_key(args.key_id)}, sort_keys=True))
        return 0
    with database.transaction(immediate=False) as connection:
        tenants = int(connection.execute("SELECT COUNT(DISTINCT tenant_id) FROM tenant_scopes").fetchone()[0])
        keys = int(connection.execute("SELECT COUNT(*) FROM api_keys WHERE revoked_at IS NULL").fetchone()[0])
        jobs = dict(
            connection.execute(
                "SELECT state, COUNT(*) AS count FROM jobs GROUP BY state"
            ).fetchall()
        )
    print(
        json.dumps(
            {
                "database": str(_database(args.database)),
                "journal_mode": database.journal_mode(),
                "tenant_count": tenants,
                "active_key_count": keys,
                "jobs_by_state": jobs,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
