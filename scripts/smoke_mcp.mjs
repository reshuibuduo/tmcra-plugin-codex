import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const child = spawn(process.execPath, [
  join(scriptDir, "..", "hooks", "run_hook.mjs"),
  "mcp_server.mjs",
], {
  stdio: ["pipe", "pipe", "pipe"],
  windowsHide: true,
});
const pending = new Map();
let stderr = "";
child.stderr.on("data", (chunk) => (stderr += chunk));
createInterface({ input: child.stdout }).on("line", (line) => {
  const message = JSON.parse(line);
  const handler = pending.get(message.id);
  if (handler) {
    pending.delete(message.id);
    handler(message);
  }
});

let nextId = 1;
function request(method, params) {
  const id = nextId++;
  child.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`);
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`MCP ${method} timed out; ${stderr}`));
    }, 180_000);
    pending.set(id, (message) => {
      clearTimeout(timer);
      if (message.error) reject(new Error(message.error.message));
      else resolve(message.result);
    });
  });
}

const started = Date.now();
const initialized = await request("initialize", {
  protocolVersion: "2025-03-26",
  capabilities: {},
  clientInfo: { name: "tmcra-plugin-smoke", version: "0.3.0-rc.10" },
});
child.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" })}\n`);
const listed = await request("tools/list", {});
const resources = await request("resources/list", {});
const statusResource = await request("resources/read", {
  uri: "ui://tmcra/memory-status-v1.html",
});
const recalled = await request("tools/call", {
  name: "tmcra_recall",
  arguments: {
    query: "What project memory is relevant to the current TMCRA integration?",
    memory_layer: "auto",
    project_path: process.cwd(),
  },
});
child.stdin.end();

if (initialized.serverInfo?.name !== "TMCRA Memory") throw new Error("unexpected MCP server identity");
const expectedTools = [
  "tmcra_get_job",
  "tmcra_ingest",
  "tmcra_last_recall",
  "tmcra_recall",
  "tmcra_status",
  "tmcra_wait_job",
];
const actualTools = Array.isArray(listed.tools)
  ? listed.tools.map((tool) => tool.name).sort()
  : [];
if (JSON.stringify(actualTools) !== JSON.stringify(expectedTools)) {
  throw new Error(`unexpected MCP tools: ${actualTools.join(", ")}`);
}
if (recalled.isError) throw new Error(recalled.content?.[0]?.text || "recall tool failed");
if (initialized.capabilities?.resources?.listChanged !== false) {
  throw new Error("MCP resource capability is missing");
}
if (!listed.tools.find((tool) => tool.name === "tmcra_status")?._meta?.ui?.resourceUri) {
  throw new Error("TMCRA status UI resource is not bound to its tool");
}
if (!listed.tools.find((tool) => tool.name === "tmcra_last_recall")?._meta?.ui?.resourceUri) {
  throw new Error("TMCRA recall UI resource is not bound to its tool");
}
if (!Array.isArray(resources.resources) || resources.resources.length !== 2) {
  throw new Error("unexpected MCP app resource list");
}
if (!statusResource.contents?.[0]?.text?.includes("TMCRA / MEMORY STATUS")) {
  throw new Error("TMCRA status UI resource is unreadable");
}

process.stdout.write(
  `${JSON.stringify(
    {
      ok: true,
      protocolVersion: initialized.protocolVersion,
      tools: listed.tools.map((tool) => tool.name),
      recallReturnedStructuredContent: Boolean(recalled.structuredContent),
      resources: resources.resources.map((resource) => resource.uri),
      elapsedMs: Date.now() - started,
    },
    null,
    2,
  )}\n`,
);
