# TMCRA Codex end-to-end tests

Run the complete deterministic suite from the plugin root:

```powershell
node .\scripts\test_codex_e2e.mjs
```

Run the device-authorization and Windows installer flow separately:

```powershell
node .\tests\device_login_mock.mjs
```

This local test exercises PKCE, `authorization_pending`, `slow_down`, explicit mock-user approval, denial, expiry, invalid verifier rejection, one-time device-code consumption, persistent installation identity, protected credential writing, the default PowerShell installer path, and authenticated configuration checking. It asserts that the access token, device code and PKCE verifier never appear in captured output. The normal end-to-end command above also includes this as its device-authorization group.

The default mode starts a local mock TMCRA HTTP API and isolated temporary Codex/project directories. It does not contact the public API and it does not read or modify real Codex history. The test covers:

- configuration writing and authenticated configuration checking;
- Codex plugin manifest, lifecycle hook, and bundled MCP configuration contracts;
- marker, Git-origin, configured, and path-based project identity plus global/project scope partitioning;
- separate session identifiers with cross-session recall inside one project;
- `SessionStart` and `UserPromptSubmit` context injection;
- `Stop` asynchronous ingestion and duplicate-Stop protection;
- cross-project isolation with a shared user-global layer;
- fail-open hooks when a token is revoked;
- all four bundled MCP tools and MCP error behavior;
- read-only history preview, explicit confirmed import, content filtering, and repeat-import idempotency;
- read-only repository bootstrap preview, explicit confirmed import, file filtering, and repeat-import idempotency;
- invalid/revoked token rejection and output checks that prevent API-key disclosure.

## Public API smoke

Public smoke is opt-in because it writes a uniquely named test turn to an isolated project scope. Use a short-lived test credential when possible. Configuration can come from the normal local TMCRA config file, or from environment variables supplied by a secret-aware shell/CI runner.

```powershell
$env:TMCRA_TEST_MODE = "real"
node .\scripts\test_codex_e2e.mjs
Remove-Item Env:TMCRA_TEST_MODE
```

To override the configured endpoint or credential, inject `TMCRA_BASE_URL` and `TMCRA_API_KEY` through the CI secret store. Do not paste a production API key into a committed script, command example, test snapshot, or issue log.

Run both deterministic and public checks in one invocation with `TMCRA_TEST_MODE=all`. The runner captures child output, never prints the API key, places real-smoke state in a temporary directory, and removes that directory after completion.

## Expected result

The command exits nonzero on the first failed assertion. A successful mock result has `ok: true`, the core Codex groups, a `deviceAuthorization` group, and every field under `assertions` set to `true`. A successful real result contains both `real.lifecycle.ok: true` and `real.mcp.ok: true`.
