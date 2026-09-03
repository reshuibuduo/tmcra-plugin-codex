import { existsSync } from "node:fs";
import { readFile, readdir } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

import {
  getSession,
  LIFECYCLE_CONTRACT_VERSION,
  loadConfig,
  PLUGIN_VERSION,
  pluginDataDir,
  resolveMemoryScopes,
} from "./tmcra_client.mjs";

const requireLifecycle = process.argv.includes("--require-lifecycle");
const apiOnly = process.argv.includes("--api-only");

async function findCodex() {
  const requested = process.env.TMCRA_CODEX_CLI;
  if (requested && existsSync(requested)) return requested;
  const home = process.env.CODEX_HOME || join(homedir(), ".codex");
  const direct = join(home, "plugins", ".plugin-appserver", "codex.exe");
  if (existsSync(direct)) return direct;
  if (process.platform !== "win32") return "codex";
  const root = join(process.env.LOCALAPPDATA || "", "OpenAI", "Codex", "bin");
  if (!existsSync(root)) throw new Error("Codex CLI was not found");
  const children = await readdir(root, { withFileTypes: true });
  const matches = children
    .filter((item) => item.isDirectory())
    .map((item) => join(root, item.name, "codex.exe"))
    .filter(existsSync);
  if (!matches.length) throw new Error("Codex CLI was not found");
  return matches.at(-1);
}

function runCodex(codex, args, { json = false } = {}) {
  const result = spawnSync(codex, args, {
    encoding: "utf8",
    windowsHide: true,
    timeout: 30_000,
  });
  if (result.error || result.status !== 0) {
    throw new Error(`Codex diagnostic command failed: ${args.join(" ")}`);
  }
  const output = String(result.stdout || "").trim();
  return json ? JSON.parse(output) : output;
}

async function readLifecycle() {
  const path = join(pluginDataDir(), "logs", "events.jsonl");
  if (!existsSync(path)) return { observed: false, path, eventCount: 0, latestEvent: null };
  const rows = (await readFile(path, "utf8"))
    .split(/\r?\n/u)
    .filter(Boolean)
    .slice(-500)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean)
    .filter(
      (row) =>
        row.pluginVersion === PLUGIN_VERSION &&
        row.lifecycleContractVersion === LIFECYCLE_CONTRACT_VERSION &&
        ["session_started", "recall_completed", "recall_degraded", "ingest_submitted", "ingest_succeeded", "ingest_failed"].includes(row.event),
    );
  const latest = rows.at(-1) || null;
  const eventNames = new Set(rows.map((row) => row.event));
  const checks = {
    sessionStarted: eventNames.has("session_started"),
    recallCompleted: eventNames.has("recall_completed") || eventNames.has("recall_degraded"),
    ingestSubmitted: eventNames.has("ingest_submitted"),
    ingestSucceeded: eventNames.has("ingest_succeeded"),
  };
  return {
    observed: Object.values(checks).every(Boolean),
    path,
    eventCount: rows.length,
    checks,
    latestEvent: latest ? { at: latest.at || null, event: latest.event } : null,
  };
}

const started = Date.now();
const codex = apiOnly ? null : await findCodex();
const config = await loadConfig();
if (config.configSource !== "device_config" && process.env.TMCRA_ALLOW_ENV_CREDENTIAL !== "1") {
  throw new Error("TMCRA must use the device credential file; implicit legacy MCP credentials are not accepted");
}
const scopes = await resolveMemoryScopes({ cwd: process.cwd(), config });
const [response, lifecycle] = await Promise.all([getSession(config), readLifecycle()]);
if (response?.ok !== true || response?.authenticated !== true) {
  throw new Error("TMCRA session verification returned an invalid response");
}

let hooksEnabled = null;
let tmcraServers = [];
if (!apiOnly) {
  const features = runCodex(codex, ["features", "list"]);
  hooksEnabled = /^hooks\s+\S+\s+true\s*$/mu.test(features);
  if (!hooksEnabled) throw new Error("Codex lifecycle hooks are not enabled");

  const mcpServers = runCodex(codex, ["mcp", "list", "--json"], { json: true });
  tmcraServers = mcpServers.filter((server) =>
    String(server.name || "")
      .toLowerCase()
      .replaceAll("_", "-")
      .startsWith("tmcra-memory"),
  );
  if (tmcraServers.length !== 1 || tmcraServers[0].name !== "tmcra-memory") {
    throw new Error("Codex must have exactly one TMCRA MCP server named tmcra-memory");
  }

  const plugins = runCodex(codex, ["plugin", "list", "--json"], { json: true });
  const plugin = (plugins.installed || []).find(
    (item) => item.pluginId === "tmcra-memory@tmcra-local" && item.enabled === true,
  );
  if (!plugin || plugin.version !== PLUGIN_VERSION) {
    throw new Error(`Codex is not running TMCRA Memory ${PLUGIN_VERSION}`);
  }
}
if (requireLifecycle && !lifecycle.observed) {
  throw new Error("No real Codex lifecycle event has been observed yet");
}

process.stdout.write(
  `${JSON.stringify(
    {
      ok: true,
      state: lifecycle.observed ? "ready" : "awaiting_hook_trust_and_real_task",
      pluginVersion: PLUGIN_VERSION,
      baseUrl: config.baseUrl,
      authorizationSource: config.configSource,
      credentialPath: config.configPath ? resolve(config.configPath) : null,
      authenticated: true,
      apiKeyExposed: false,
      scopeReady: null,
      queryId: null,
      hooksEnabled,
      lifecycle,
      mcpServers: tmcraServers.map((server) => server.name),
      globalScope: scopes.globalScope,
      projectScope: scopes.projectScope,
      projectId: scopes.projectId,
      projectIdentitySource: scopes.projectIdentitySource,
      serviceVersion: response?.service?.version || null,
      remoteCapabilities: Array.isArray(response?.service?.capabilities)
        ? response.service.capabilities
        : [],
      elapsedMs: Date.now() - started,
    },
    null,
    2,
  )}\n`,
);
