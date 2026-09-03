import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const pluginRoot = resolve(scriptDir, "..");
const mode = String(
  process.env.TMCRA_TEST_MODE || (process.env.TMCRA_TEST_REAL === "1" ? "real" : "mock"),
).toLowerCase();
const allowedModes = new Set(["mock", "real", "all"]);

if (!allowedModes.has(mode)) {
  throw new Error("TMCRA_TEST_MODE must be mock, real, or all");
}

function redacted(value) {
  const secret = String(process.env.TMCRA_API_KEY || "");
  return secret ? String(value || "").replaceAll(secret, "[REDACTED]") : String(value || "");
}

function run(script, { env = process.env, timeoutMs = 60_000 } = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(process.execPath, [script], {
      cwd: pluginRoot,
      env,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`${basename(script)} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.once("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(`${basename(script)} exited ${code}: ${redacted(stderr || stdout)}`));
        return;
      }
      resolvePromise(JSON.parse(stdout.trim()));
    });
  });
}

const output = { ok: true, requestedMode: mode };

if (mode === "mock" || mode === "all") {
  const providerSetup = await run(join(pluginRoot, "tests", "provider_setup_contract.mjs"), {
    timeoutMs: 60_000,
  });
  const providerExecutor = await run(join(pluginRoot, "tests", "provider_executor_contract.mjs"), {
    timeoutMs: 60_000,
  });
  const coreMock = await run(join(pluginRoot, "tests", "codex_e2e_mock.mjs"), {
    timeoutMs: 90_000,
  });
  const deviceAuthorization = await run(join(pluginRoot, "tests", "device_login_mock.mjs"), {
    timeoutMs: 90_000,
  });
  const claudeCode = await run(join(pluginRoot, "tests", "claude_code_contract.mjs"), {
    timeoutMs: 90_000,
  });
  output.mock = {
    ...coreMock,
    providerSetup,
    providerExecutor,
    deviceAuthorization,
    claudeCode,
  };
}

if (mode === "real" || mode === "all") {
  const tempPrefix = "tmcra-codex-real-smoke-";
  const tempRoot = await mkdtemp(join(tmpdir(), tempPrefix));
  try {
    const realEnv = {
      ...process.env,
      TMCRA_SMOKE_DATA_DIR: join(tempRoot, "plugin-data"),
    };
    const lifecycle = await run(join(scriptDir, "smoke_lifecycle.mjs"), {
      env: realEnv,
      timeoutMs: 360_000,
    });
    const mcp = await run(join(scriptDir, "smoke_mcp.mjs"), {
      env: realEnv,
      timeoutMs: 180_000,
    });
    delete lifecycle.dataDir;
    output.real = {
      ok: Boolean(lifecycle.ok && mcp.ok),
      lifecycle,
      mcp,
    };
  } finally {
    const resolvedTemp = resolve(tempRoot);
    if (dirname(resolvedTemp) === resolve(tmpdir()) && basename(resolvedTemp).startsWith(tempPrefix)) {
      await rm(resolvedTemp, { recursive: true, force: true });
    }
  }
}

process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
