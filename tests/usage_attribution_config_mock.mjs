import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadConfig, request } from "../scripts/tmcra_client.mjs";

const root = await mkdtemp(join(tmpdir(), "tmcra-codex-ledger-"));
const configPath = join(root, "config.json");
const integrationId = `int_${"c".repeat(32)}`;
await writeFile(configPath, JSON.stringify({
  baseUrl: "https://api.tmcra.com",
  accessToken: "device-token",
  expiresAt: "2999-01-01T00:00:00.000Z",
  scopeNamespace: "personal-1",
  globalScope: "personal-1-global",
  projectScopePrefix: "personal-1-project",
  integrationIds: { codex: integrationId },
  codexAgentId: "codex-primary",
}));
process.env.TMCRA_CONFIG_FILE = configPath;
const config = await loadConfig();
assert.equal(config.integrationId, integrationId);
assert.equal(config.agentId, "codex-primary");

let captured;
globalThis.fetch = async (_url, init) => {
  captured = init;
  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
};
await request("/v1/session", { config, attempts: 1 });
assert.equal(captured.headers["X-TMCRA-Client-Platform"], "codex");
assert.equal(captured.headers["X-TMCRA-Integration-ID"], integrationId);
assert.equal(captured.headers["X-TMCRA-Agent-ID"], "codex-primary");
process.stdout.write(`${JSON.stringify({ ok: true })}\n`);
