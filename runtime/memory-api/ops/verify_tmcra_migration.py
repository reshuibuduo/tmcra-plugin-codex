#!/usr/bin/env python3
"""Read-only verification for the migrated TMCRA production assets."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path


RUN = Path("/opt/tmcra/runs/v4_writer100_frozen_20260713_011551")
REQUIRED = (
    Path("/opt/tmcra-models/BAAI/bge-m3"),
    Path("/opt/tmcra-models/BAAI/bge-reranker-v2-m3"),
    Path(
        "/opt/tmcra-data/tmcra_latest_training_model_architecture_20260607/"
        "runs/set_c_temporal_hardneg_train_20260607_231735/node_scorer.pt"
    ),
    Path(
        "/opt/tmcra-data/tmcra_latest_training_model_architecture_20260607/"
        "runs/set_c_temporal_hardneg_train_20260607_231735/path_scorer.pt"
    ),
    Path(
        "/opt/tmcra/runs/v3_s500_only_multiseed_20260710_101625/"
        "seed_31/tmcra_v3_reranker.pt"
    ),
    Path(
        "/opt/tmcra-data/migration/legacy/"
        "tmcra_longmemeval/data/longmemeval_s_cleaned.json"
    ),
    Path(
        "/opt/tmcra-data/migration/legacy/"
        "tmcra_longmemeval/scripts/run_lme_s10_native_tmcra.py"
    ),
    Path(
        "/opt/tmcra-data/migration/legacy/"
        "tmcra_api_service/private/tmcra-integrated"
    ),
)
SECRET_FILES = (
    Path(
        "/opt/tmcra-data/migration/legacy/"
        "tmcra_api_service/env/deepseek-writer-pool.env"
    ),
    Path(
        "/opt/tmcra-data/migration/legacy/"
        "tmcra_longmemeval/env/answer-vectorengine-gpt54.env"
    ),
)


def main() -> int:
    missing = [str(path) for path in (RUN, *REQUIRED, *SECRET_FILES) if not path.exists()]
    databases = sorted(RUN.glob("writer/worker_*/native_memory.sqlite3"))
    sqlite_failures: list[dict[str, str]] = []
    source_rows = 0
    record_rows = 0
    for path in databases:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            check = str(con.execute("PRAGMA quick_check").fetchone()[0])
            if check != "ok":
                sqlite_failures.append({"database": str(path), "error": check})
                continue
            tables = {
                str(row[0])
                for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "records" not in tables:
                sqlite_failures.append(
                    {"database": str(path), "error": "records table missing"}
                )
                continue
            record_rows += int(con.execute("SELECT count(*) FROM records").fetchone()[0])
            if "source_message_records" in tables:
                source_rows += int(
                    con.execute("SELECT count(*) FROM source_message_records").fetchone()[0]
                )
            elif "memory_source_records" in tables:
                source_rows += int(
                    con.execute("SELECT count(*) FROM memory_source_records").fetchone()[0]
                )
            else:
                source_rows += int(
                    con.execute(
                        "SELECT count(*) FROM records WHERE category = 'source'"
                    ).fetchone()[0]
                )
        finally:
            con.close()
    secret_modes = {
        str(path): oct(os.stat(path).st_mode & 0o777) if path.exists() else None
        for path in SECRET_FILES
    }
    report = {
        "schema_version": "tmcra.v4.migration-verification.1",
        "read_only": True,
        "status": (
            "passed"
            if not missing
            and len(databases) == 100
            and not sqlite_failures
            and source_rows > 0
            and all(mode == "0o600" for mode in secret_modes.values())
            else "failed"
        ),
        "missing_paths": missing,
        "database_count": len(databases),
        "sqlite_failures": sqlite_failures,
        "source_row_count": source_rows,
        "record_row_count": record_rows,
        "secret_file_modes": secret_modes,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
