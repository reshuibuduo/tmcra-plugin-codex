# Changelog

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
