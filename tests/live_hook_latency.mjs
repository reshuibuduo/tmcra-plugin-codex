import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";


const testRoot = dirname(fileURLToPath(import.meta.url));
const pluginRoot = resolve(testRoot, "..");
const defaultHook = join(pluginRoot, "hooks", "user_prompt_submit.mjs");
const defaultConfig = join(process.env.USERPROFILE || process.env.HOME, ".config", "tmcra", "config.json");
const LIVE_PROMPTS = [
  "What did we decide about the Codex lifecycle Hook timeout?",
  "Which server currently hosts the production TMCRA API?",
  "What local model is now used by the recall planner?",
  "How was the Windows system-proxy compatibility issue diagnosed?",
  "Which canary proves the local Writer can complete a formal write?",
  "What remains external after moving the Writer and Planner to the local model?",
  "Which evidence mode should the interactive Codex recall Hook request?",
  "Why must global and project memory remain separate during recall?",
  "What latency percentile should govern the Hook timeout decision?",
  "Which production checks must pass before calling the memory service ready?",
  "How does the plugin preserve a task before automatic context compaction?",
  "What is the current TMCRA Codex plugin integration version?",
];


function percentile(values, quantile) {
  const ordered = [...values].sort((left, right) => left - right);
  const rank = (ordered.length - 1) * quantile;
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  if (lower === upper) return ordered[lower];
  return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower);
}


function systemProxy() {
  if (process.env.TMCRA_LIVE_TEST_NO_EXPLICIT_PROXY === "1") return null;
  if (process.env.HTTPS_PROXY || process.env.https_proxy) {
    return process.env.HTTPS_PROXY || process.env.https_proxy;
  }
  if (process.platform !== "win32") return null;
  return process.env.TMCRA_LIVE_TEST_PROXY || "http://127.0.0.1:7890";
}


async function runHook({ hook, hookArgs, config, dataDir, sequence, cwd }) {
  const proxy = systemProxy();
  const nodeArgs = Number(process.versions.node.split(".")[0]) >= 24
    ? ["--use-env-proxy", hook, ...hookArgs]
    : [hook, ...hookArgs];
  const input = JSON.stringify({
    hook_event_name: "UserPromptSubmit",
    session_id: `tmcra-live-latency-${process.pid}-${sequence}`,
    turn_id: `turn-${sequence}`,
    cwd,
    model: "gpt-5.4",
    prompt: LIVE_PROMPTS[(sequence - 1) % LIVE_PROMPTS.length],
  });
  const started = process.hrtime.bigint();
  const result = await new Promise((accept, reject) => {
    const child = spawn(process.execPath, nodeArgs, {
      cwd,
      env: {
        ...process.env,
        PLUGIN_DATA: dataDir,
        TMCRA_CONFIG_FILE: config,
        ...(proxy ? { HTTPS_PROXY: proxy, HTTP_PROXY: proxy } : {}),
      },
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8").on("data", (chunk) => { stdout += chunk; });
    child.stderr.setEncoding("utf8").on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => accept({ code, stdout, stderr }));
    child.stdin.end(input);
  });
  const elapsedMs = Number(process.hrtime.bigint() - started) / 1_000_000;
  let output = null;
  try {
    output = JSON.parse(result.stdout.trim());
  } catch {
    // The report below exposes no hook output or memory content.
  }
  return {
    sequence,
    exit_code: result.code,
    elapsed_ms: elapsedMs,
    continue: output?.continue === true,
    context_present: Boolean(output?.hookSpecificOutput?.additionalContext),
    context_characters: String(output?.hookSpecificOutput?.additionalContext || "").length,
    stderr_present: Boolean(result.stderr.trim()),
  };
}


const args = process.argv.slice(2);
const valueAfter = (name, fallback) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : fallback;
};
const samples = Number(valueAfter("--samples", "6"));
const warmups = Number(valueAfter("--warmups", "1"));
const hook = resolve(valueAfter("--hook", defaultHook));
const hookArg = valueAfter("--hook-arg", null);
const hookArgs = hookArg ? [hookArg] : [];
const config = resolve(valueAfter("--config", defaultConfig));
const cwd = resolve(valueAfter("--cwd", process.cwd()));
if (!Number.isInteger(samples) || samples < 1 || !Number.isInteger(warmups) || warmups < 0) {
  throw new Error("samples must be positive and warmups must be non-negative integers");
}
await readFile(config, "utf8");
const dataDir = await mkdtemp(join(tmpdir(), "tmcra-live-hook-"));
try {
  const rows = [];
  for (let sequence = 1; sequence <= samples + warmups; sequence += 1) {
    rows.push(await runHook({ hook, hookArgs, config, dataDir, sequence, cwd }));
  }
  const measured = rows.slice(warmups);
  const elapsed = measured.map((row) => row.elapsed_ms);
  process.stdout.write(`${JSON.stringify({
    schema_version: "tmcra.live-hook-latency.1",
    sample_count: measured.length,
    warmup_count: warmups,
    success_count: measured.filter((row) => row.exit_code === 0 && row.continue).length,
    context_success_count: measured.filter((row) => row.context_present).length,
    stderr_count: measured.filter((row) => row.stderr_present).length,
    elapsed_ms: {
      mean: elapsed.reduce((sum, value) => sum + value, 0) / elapsed.length,
      p50: percentile(elapsed, 0.50),
      p95: percentile(elapsed, 0.95),
      max: Math.max(...elapsed),
    },
    rows: measured,
  })}\n`);
} finally {
  await rm(dataDir, { recursive: true, force: true });
}
