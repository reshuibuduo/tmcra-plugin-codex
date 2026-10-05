# Changelog

## 1.0.0-rc.1 - 2026-09-06

- Add the redesigned memory workspace, local Writer/organizer settings, knowledge and graph views, session controls, task continuity, and bounded recall budgets.
- Require interactive host confirmation for conversational memory corrections; protect the correction discussion from automatic capture and replay.
- Add three pinned embedding/reranker profiles and Windows local-runtime preview controls. Local identity disables inherited cloud-provider task handoff.
- Preserve binary image bytes in release ZIPs and verify packaged assets against their sources.
- Bundle the verified backend, automatic private Python bootstrap and shared local identity discovery. `Install-Local.cmd` installs without TMCRA servers/accounts. Runtime files survive plugin-cache updates; stale cloud connections are blocked after local selection.
- Full-local acceptance remains partial: CPU ingest/raw recall passed; complex compilation timed out, with organizer and restart recovery still pending. Production is unchanged.

## 0.3.0-rc.10 - 2026-09-04

- Include the icon and overview image referenced by the Codex marketplace manifest in every release archive.

## 0.3.0-rc.9 - 2026-09-04

- Add complete Codex marketplace presentation metadata so the current plugin bundle can be mirrored and refreshed by community marketplaces.

## 0.3.0-rc.8 - 2026-09-04

- Execute configured Writer and background-organizer calls from the authenticated local Codex process.
- Add a durable, credential-bound task lease protocol with heartbeat and idempotent terminal receipts.
- Return only parsed JSON, normalized usage, and provider request identity to TMCRA; keep API keys and raw provider envelopes local.
- Add explicit `tmcra_consolidate` routing, bounded response handling, fair stage scheduling, and quiet exponential recovery.

## 0.3.0-rc.7 - 2026-09-04

- Add a loopback-only local settings UI for separate Writer and background-organizer model providers.
- Keep provider credentials in the local user configuration and redact them from every MCP response.
- Prevent a saved credential from being reused after its provider or Base URL changes.

## 0.3.0-rc.6 - 2026-09-04

- Canonicalize ZIP creator and file-mode metadata so Windows and Linux now produce the same archive and SHA-256 digest.

## 0.3.0-rc.5 - 2026-09-04

- Normalize every release entry to UTF-8 with LF line endings.
- Remove a race in the Claude Code ingestion contract on slower CI runners.
- Scope plugin security scanning to shipped code while retaining high-severity enforcement.

## 0.3.0-rc.4 - 2026-09-04

- Replace the obsolete owner-local four-Hook bridge with the current nine-Hook TMCRA lifecycle implementation.
- Add browser device authorization, protected credential storage, explicit recall inspection, durable asynchronous capture, long-task checkpoints, history migration, and repository bootstrap.
- Bundle the MCP server and the complete `manage-tmcra-memory` skill in the self-contained release ZIP.
- Add deterministic Codex, Claude Code, device-authorization, security, and cross-platform release contracts.
- Align the Claude Code manifest with the Apache-2.0 repository license and verify sensitive `userConfig` injection.

## 0.1.1 - 2026-08-16

- Initial owner-local Codex plugin preview.
