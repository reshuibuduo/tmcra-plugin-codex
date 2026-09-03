import { execFileSync, spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";


const HOOK_ROOT = dirname(fileURLToPath(import.meta.url));
const TARGETS = new Map([
  ["session_start.mjs", { path: join(HOOK_ROOT, "session_start.mjs"), failOpen: true }],
  ["subagent_start.mjs", { path: join(HOOK_ROOT, "subagent_start.mjs"), failOpen: true }],
  ["user_prompt_submit.mjs", { path: join(HOOK_ROOT, "user_prompt_submit.mjs"), failOpen: true }],
  ["post_tool_use.mjs", { path: join(HOOK_ROOT, "post_tool_use.mjs"), failOpen: true }],
  ["pre_compact.mjs", { path: join(HOOK_ROOT, "pre_compact.mjs"), failOpen: true }],
  ["post_compact.mjs", { path: join(HOOK_ROOT, "post_compact.mjs"), failOpen: true }],
  ["stop.mjs", { path: join(HOOK_ROOT, "stop.mjs"), failOpen: true }],
  ["subagent_stop.mjs", { path: join(HOOK_ROOT, "subagent_stop.mjs"), failOpen: true }],
  ["mcp_server.mjs", { path: join(HOOK_ROOT, "..", "scripts", "mcp_server.mjs"), failOpen: false }],
]);


export function normalizeProxyServer(value, protocol = "https") {
  const text = String(value || "").trim();
  if (!text) return null;
  const assignments = text
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const separator = part.indexOf("=");
      return separator > 0
        ? [part.slice(0, separator).trim().toLowerCase(), part.slice(separator + 1).trim()]
        : ["all", part];
    });
  const selected =
    assignments.find(([name]) => name === protocol)?.[1] ||
    assignments.find(([name]) => name === "http")?.[1] ||
    assignments.find(([name]) => name === "all")?.[1] ||
    null;
  if (!selected) return null;
  return /^[a-z][a-z0-9+.-]*:\/\//iu.test(selected) ? selected : `http://${selected}`;
}


export function appendNoProxy(current) {
  const entries = String(current || "")
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  for (const loopback of ["127.0.0.1", "localhost", "::1"]) {
    if (!entries.includes(loopback)) entries.push(loopback);
  }
  return entries.join(",");
}


function windowsSystemProxy() {
  if (process.platform !== "win32") return null;
  try {
    const output = execFileSync(
      "reg.exe",
      [
        "query",
        "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings",
        "/v",
        "ProxyEnable",
      ],
      { encoding: "utf8", windowsHide: true, timeout: 1_000 },
    );
    if (!/ProxyEnable\s+REG_DWORD\s+0x1\b/iu.test(output)) return null;
    const serverOutput = execFileSync(
      "reg.exe",
      [
        "query",
        "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings",
        "/v",
        "ProxyServer",
      ],
      { encoding: "utf8", windowsHide: true, timeout: 1_000 },
    );
    return serverOutput.match(/ProxyServer\s+REG_SZ\s+(.+)$/imu)?.[1]?.trim() || null;
  } catch {
    return null;
  }
}


export function proxyEnvironment(environment = process.env) {
  const existingHttps = environment.HTTPS_PROXY || environment.https_proxy;
  const existingHttp = environment.HTTP_PROXY || environment.http_proxy;
  const existingAll = environment.ALL_PROXY || environment.all_proxy;
  const system = existingHttps || existingHttp || existingAll ? null : windowsSystemProxy();
  const httpsProxy = existingHttps || existingAll || normalizeProxyServer(system, "https");
  const httpProxy = existingHttp || existingAll || normalizeProxyServer(system, "http");
  if (!httpsProxy && !httpProxy) return { ...environment };
  const noProxy = appendNoProxy(environment.NO_PROXY || environment.no_proxy);
  return {
    ...environment,
    ...(httpsProxy ? { HTTPS_PROXY: httpsProxy } : {}),
    ...(httpProxy ? { HTTP_PROXY: httpProxy } : {}),
    NO_PROXY: noProxy,
    NODE_USE_ENV_PROXY: "1",
  };
}


export async function run() {
  const targetName = String(process.argv[2] || "");
  const target = TARGETS.get(targetName);
  if (!target) {
    throw new Error("TMCRA runtime launcher received an invalid target name");
  }
  const environment = proxyEnvironment();
  const proxyConfigured = Boolean(
    environment.HTTPS_PROXY || environment.https_proxy ||
    environment.HTTP_PROXY || environment.http_proxy ||
    environment.ALL_PROXY || environment.all_proxy
  );
  const canUseEnvironmentProxy = process.allowedNodeEnvironmentFlags.has("--use-env-proxy");
  const args = [
    ...(proxyConfigured && canUseEnvironmentProxy ? ["--use-env-proxy"] : []),
    target.path,
  ];
  const child = spawn(process.execPath, args, {
    env: environment,
    stdio: "inherit",
    windowsHide: true,
  });
  const terminate = () => {
    if (!child.killed) child.kill();
  };
  process.once("exit", terminate);
  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.once(signal, () => {
      terminate();
      process.exitCode = 1;
    });
  }
  const exitCode = await new Promise((accept, reject) => {
    child.once("error", reject);
    child.once("close", (code) => accept(code));
  });
  process.removeListener("exit", terminate);
  process.exitCode = exitCode ?? 1;
}


if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    await run();
  } catch (error) {
    const target = TARGETS.get(String(process.argv[2] || ""));
    if (target?.failOpen) {
      process.stderr.write(`TMCRA hook launcher failed open: ${error.message}\n`);
      process.stdout.write('{"continue":true}\n');
    } else {
      process.stderr.write(`TMCRA runtime launcher failed: ${error.message}\n`);
      process.exitCode = 1;
    }
  }
}
