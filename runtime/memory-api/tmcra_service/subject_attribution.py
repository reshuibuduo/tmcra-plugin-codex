"""Production single-scope subject-attribution gate.

The benchmark module owns the subject-attribution contract.  This module keeps
that deterministic implementation shared and adds the durable, single-scope
service boundary around it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

# Keep `python tmcra_service/subject_attribution.py ...` usable from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ops import audit_tmcra_v4_subject_attribution as _benchmark
from tmcra_service.usage_attribution import UsageAttribution
from tmcra_service.user_provider_client import (
    UserProviderBrokerClient,
    normalize_user_provider_execution,
)

__all__ = [
    "AttributionError",
    "DeepSeekProAttributionClient",
    "UserProviderAttributionClient",
    "document_route_reasons",
    "execute_job",
    "main",
    "run",
    "run_subject_attribution",
    "scan_database",
    "validate_decisions",
]


# These aliases deliberately expose the benchmark contract without copying or
# modifying its deterministic routing, validation, and application logic.
AttributionError = _benchmark.AttributionError
AttributionClient = _benchmark.AttributionClient
DeepSeekProAttributionClient = _benchmark.DeepSeekProAttributionClient
CURRENT_STATES = _benchmark.CURRENT_STATES
DECISIONS = _benchmark.DECISIONS
MODEL = _benchmark.MODEL
PROMPT_VERSION = _benchmark.PROMPT_VERSION
SYSTEM_PROMPT = _benchmark.SYSTEM_PROMPT

document_route_reasons = _benchmark.document_route_reasons
scan_database = _benchmark.scan_database
validate_decisions = _benchmark.validate_decisions
execute_job = _benchmark.execute_job


def _required_environment(name: str) -> str:
    value = str(os.getenv(name) or "").strip()
    if not value:
        raise AttributionError(f"{name} is required for user-provider execution")
    return value


class UserProviderAttributionClient:
    """Route one attribution request through the authenticated local executor."""

    def __init__(self) -> None:
        try:
            raw_execution = json.loads(
                _required_environment("TMCRA_USER_PROVIDER_EXECUTION_JSON")
            )
            execution = normalize_user_provider_execution(
                raw_execution,
                stage="organizer",
            )
            if execution is None:
                raise ValueError("organizer execution route is missing")
            raw_attribution = str(
                os.getenv("TMCRA_USAGE_ATTRIBUTION_JSON") or "{}"
            ).strip()
            usage_attribution = UsageAttribution.from_mapping(
                json.loads(raw_attribution)
            )
            self.broker = UserProviderBrokerClient(
                control_db=Path(_required_environment("TMCRA_SERVICE_CONTROL_DB")),
                tenant_id=_required_environment("TMCRA_SERVICE_TENANT_ID"),
                scope_name=_required_environment("TMCRA_SERVICE_SCOPE_NAME"),
                auth_key_id=execution["auth_key_id"],
                job_id=_required_environment("TMCRA_SERVICE_JOB_ID"),
                stage_id=_required_environment("TMCRA_SERVICE_STAGE_ID"),
                task_stage="organizer",
                timeout=float(
                    os.getenv("TMCRA_USER_PROVIDER_TIMEOUT_SECONDS", "900")
                ),
                max_tokens=int(
                    os.getenv("TMCRA_SUBJECT_ATTRIBUTION_MAX_TOKENS", "16384")
                ),
                usage_attribution=usage_attribution,
                record_ledger=False,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise AttributionError(
                "user-provider attribution environment is invalid"
            ) from exc

    def complete(self, payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
        output, metadata = self.broker.complete_prompt(
            system_prompt=SYSTEM_PROMPT,
            payload=payload,
            operation="subject_attribution_pro",
        )
        return json.dumps(
            output,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ), metadata


def _default_attribution_client() -> AttributionClient:
    if str(os.getenv("TMCRA_USER_PROVIDER_EXECUTION_JSON") or "").strip():
        return UserProviderAttributionClient()
    return DeepSeekProAttributionClient()


def _usage_totals(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
    }
    for result in results:
        metadata = result.get("call_metadata")
        if not isinstance(metadata, Mapping):
            continue
        usage = metadata.get("usage")
        usage = usage if isinstance(usage, Mapping) else metadata
        for name in totals:
            totals[name] += int(usage.get(name, 0) or 0)
    return totals


def _cost_metadata(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "currency": "CNY",
        "prompt_cost_per_million": float(
            os.getenv("TMCRA_DEEPSEEK_PRO_PROMPT_COST_PER_MILLION", "3")
        ),
        "completion_cost_per_million": float(
            os.getenv("TMCRA_DEEPSEEK_PRO_COMPLETION_COST_PER_MILLION", "6")
        ),
        "cache_cost_per_million": float(
            os.getenv("TMCRA_DEEPSEEK_PRO_CACHE_COST_PER_MILLION", "0.025")
        ),
        "estimated_cost_cny": round(
            sum(float(item.get("estimated_cost_cny", 0.0) or 0.0) for item in results),
            8,
        ),
    }


def _resolved_memory_ids(
    job: Mapping[str, Any], result: Mapping[str, Any]
) -> set[str]:
    decisions = result.get("decisions")
    if not isinstance(decisions, list):
        return set()
    ids = {
        str(item.get("memory_id"))
        for item in decisions
        if isinstance(item, Mapping) and item.get("memory_id")
    }
    expected = {
        str(item["memory_id"])
        for item in job["payload"]["candidates"]
        if isinstance(item, Mapping) and item.get("memory_id")
    }
    if len(ids) != len(decisions) or len(ids) != len(expected):
        return set()
    return ids if ids == expected else set()


def _write_report(output: Path, report: Mapping[str, Any]) -> None:
    """Replace the report atomically and make the replacement durable."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
        try:
            directory_fd = os.open(output.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _routed_report_entry(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "database": job["database"],
        "scope_id": job["scope_id"],
        "message_id": job["message_id"],
        "session_index": job["session_index"],
        "message_index": job["message_index"],
        "source_turn_index": job["source_turn_index"],
        "route_reasons": job["route_reasons"],
        "request_sha256": job["request_sha256"],
        "candidate_count": len(job["payload"]["candidates"]),
        "candidates": job["review_candidates"],
    }


def run_subject_attribution(
    database: Path | str,
    scope_id: str,
    output: Path | str,
    *,
    apply: bool = False,
    client: AttributionClient | None = None,
) -> dict[str, Any]:
    """Scan or apply subject attribution for exactly one database scope.

    Scan-only never constructs a provider client.  Apply writes a failed report
    before raising when any routed candidate remains unresolved, so callers and
    operators cannot mistake a partial run for a successful gate.
    """
    database_path = Path(database).resolve()
    output_path = Path(output).resolve()
    scope = str(scope_id).strip()
    if not scope:
        raise AttributionError("scope_id must not be empty")

    scanned = scan_database(database_path, scope)
    results: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    if apply and scanned:
        try:
            active_client = (
                client if client is not None else _default_attribution_client()
            )
        except Exception as exc:
            active_client = None
            for job in scanned:
                unresolved.append(
                    {
                        "message_id": job["message_id"],
                        "memory_ids": [
                            item["memory_id"] for item in job["payload"]["candidates"]
                        ],
                        "error": str(exc),
                    }
                )
        if active_client is not None:
            for job in scanned:
                try:
                    result = execute_job(database_path, job, active_client)
                except Exception as exc:
                    result = {
                        "audit_id": "",
                        "status": "failed",
                        "physical_api_calls": 0,
                        "message_id": job["message_id"],
                        "error": str(exc),
                    }
                    unresolved.append(
                        {
                            "message_id": job["message_id"],
                            "memory_ids": [
                                item["memory_id"]
                                for item in job["payload"]["candidates"]
                            ],
                            "error": str(exc),
                        }
                    )
                else:
                    result = {"message_id": job["message_id"], **result}
                    if (
                        result.get("status") not in {"completed", "reused"}
                        or not _resolved_memory_ids(job, result)
                    ):
                        unresolved.append(
                            {
                                "message_id": job["message_id"],
                                "memory_ids": [
                                    item["memory_id"]
                                    for item in job["payload"]["candidates"]
                                ],
                                "error": "routed candidates were not fully resolved",
                            }
                        )
                results.append(result)
    elif not apply:
        for job in scanned:
            unresolved.append(
                {
                    "message_id": job["message_id"],
                    "memory_ids": [
                        item["memory_id"] for item in job["payload"]["candidates"]
                    ],
                    "error": "scan_only did not apply a decision",
                }
            )

    physical_api_calls = sum(
        int(item.get("physical_api_calls", 0) or 0) for item in results
    )
    resolved_routed_candidate_count = sum(
        len(item.get("decisions", []))
        for item in results
        if item.get("status") in {"completed", "reused"}
    )
    decision_quarantined_count = sum(
        len(item.get("quarantined_memory_ids", [])) for item in results
    )
    cascaded_quarantined_count = sum(
        len(item.get("cascaded_quarantined_memory_ids", [])) for item in results
    )
    cost = _cost_metadata(results)
    report: dict[str, Any] = {
        "schema_version": "tmcra.v4.subject-attribution-service-report.1",
        "status": "complete" if not (apply and unresolved) else "failed",
        "gate_passed": bool(apply and not unresolved),
        "mode": "apply" if apply else "scan_only",
        "database": str(database_path),
        "scope_id": scope,
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "routed_message_count": len(scanned),
        "routed_candidate_count": sum(
            len(job["payload"]["candidates"]) for job in scanned
        ),
        "resolved_routed_message_count": len(scanned) - len(unresolved)
        if apply
        else 0,
        "resolved_routed_candidate_count": resolved_routed_candidate_count,
        "unresolved_routed_message_count": len(unresolved),
        "unresolved_routed_candidate_count": sum(
            len(item["memory_ids"]) for item in unresolved
        ),
        "physical_api_calls": physical_api_calls,
        "estimated_cost_cny": cost["estimated_cost_cny"],
        "decision_quarantined_count": decision_quarantined_count,
        "cascaded_quarantined_count": cascaded_quarantined_count,
        "quarantined_count": decision_quarantined_count
        + cascaded_quarantined_count,
        "usage": _usage_totals(results),
        "cost": cost,
        "routed": [_routed_report_entry(job) for job in scanned],
        "results": results,
        "unresolved": unresolved,
    }
    _write_report(output_path, report)
    if apply and unresolved:
        raise AttributionError(
            f"subject attribution gate failed: {len(unresolved)} routed message(s) unresolved"
        )
    return report


# A short alias for callers that use the gate name rather than the operation name.
run = run_subject_attribution


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--scope-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_subject_attribution(
            args.database,
            args.scope_id,
            args.output,
            apply=args.apply,
        )
    except AttributionError as exc:
        if args.output.is_file():
            report = json.loads(args.output.read_text(encoding="utf-8"))
            report["error"] = str(exc)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        else:
            print(str(exc))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
