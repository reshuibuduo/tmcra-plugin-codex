import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";

const apiKey = String(process.env.TMCRA_SETUP_API_KEY || "").trim();
if (!apiKey) throw new Error("TMCRA_SETUP_API_KEY is required");

const path = resolve(
  process.env.TMCRA_CONFIG_FILE || join(homedir(), ".config", "tmcra", "config.json"),
);
const namespace = String(process.env.TMCRA_SCOPE_NAMESPACE || "tmcra").trim();
let integrationIds = {};
if (existsSync(path)) {
  const previous = JSON.parse(await readFile(path, "utf8"));
  if (!previous || typeof previous !== "object" || Array.isArray(previous)) {
    throw new Error("TMCRA config must contain a JSON object");
  }
  if (previous.integrationIds !== undefined) {
    if (!previous.integrationIds || typeof previous.integrationIds !== "object" || Array.isArray(previous.integrationIds)) {
      throw new Error("TMCRA device integration bindings are invalid");
    }
    const allowed = new Set(["codex", "openclaw", "hermes", "claude_code", "mcp"]);
    for (const [platformName, integrationId] of Object.entries(previous.integrationIds)) {
      if (!allowed.has(platformName) || !/^int_[a-f0-9]{32}$/u.test(String(integrationId))) {
        throw new Error("TMCRA device integration bindings are invalid");
      }
      integrationIds[platformName] = String(integrationId);
    }
  }
}
const value = {
  schemaVersion: 2,
  authMode: "api-key",
  baseUrl: String(
    process.env.TMCRA_BASE_URL ||
      "https://api.tmcra.com",
  ).replace(/\/+$/u, ""),
  apiKey,
  tokenType: "Bearer",
  scopeNamespace: namespace,
  globalScope: String(process.env.TMCRA_GLOBAL_SCOPE || `${namespace}-global`),
  projectScopePrefix: String(process.env.TMCRA_PROJECT_SCOPE_PREFIX || `${namespace}-project`),
  timeoutMs: Number(process.env.TMCRA_REQUEST_TIMEOUT_MS || 120000),
  integrationIds,
};

await mkdir(dirname(path), { recursive: true });
await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
process.stdout.write(`${JSON.stringify({ ok: true, configPath: path, apiKeyStored: true })}\n`);
