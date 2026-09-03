import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdir, readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  loadConfig,
  promptEvidenceContent,
  recall,
  resolveMemoryScopes,
  waitJob,
} from "./tmcra_client.mjs";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const pluginRoot = resolve(scriptDir, "..");
const smokeId = `smoke-${Date.now()}-${randomUUID().slice(0, 8)}`;
const marker = `TMCRA_CODEX_LIFECYCLE_${randomUUID().replaceAll("-", "").slice(0, 16)}`;
const toolMarker = `TMCRA_COMPACT_TOOL_${randomUUID().replaceAll("-", "").slice(0, 16)}`;
const secretMarker = `TMCRA_SECRET_${randomUUID().replaceAll("-", "").slice(0, 16)}`;
const dataDir = resolve(process.env.TMCRA_SMOKE_DATA_DIR || join(pluginRoot, ".smoke-data", smokeId));
const env = {
  ...process.env,
  PLUGIN_ROOT: pluginRoot,
  CLAUDE_PLUGIN_ROOT: pluginRoot,
  PLUGIN_DATA: dataDir,
  CLAUDE_PLUGIN_DATA: dataDir,
  TMCRA_PROJECT_ID: smokeId,
};

await mkdir(dataDir, { recursive: true });

function runHook(name, input) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(process.execPath, [join(pluginRoot, "hooks", name)], {
      env,
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`${name} exited ${code}: ${stderr}`));
        return;
      }
      try {
        resolvePromise(JSON.parse(stdout.trim() || "{}"));
      } catch (error) {
        reject(new Error(`${name} returned invalid JSON: ${error.message}; ${stdout}`));
      }
    });
    child.stdin.end(JSON.stringify(input));
  });
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function readEvents(path) {
  try {
    return (await readFile(path, "utf8"))
      .trim()
      .split(/\r?\n/u)
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  } catch (error) {
    if (error?.code === "ENOENT") return [];
    throw error;
  }
}

async function waitForSubmittedJob(path, projectId, cursor, timeoutMs = 60_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const events = await readEvents(path);
    const submitted = events
      .slice(cursor)
      .findLast((event) => (
        event.event === "ingest_submitted" &&
        event.projectId === projectId &&
        event.jobId
      ));
    if (submitted) return submitted;
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 250));
  }
  return null;
}

const sessionId = `codex-session-${smokeId}`;
const turnId = `turn-${randomUUID()}`;
const common = {
  session_id: sessionId,
  turn_id: turnId,
  cwd: process.cwd(),
  model: "smoke-test",
  permission_mode: "default",
};

const before = Date.now();
await runHook("user_prompt_submit.mjs", {
  ...common,
  hook_event_name: "UserPromptSubmit",
  prompt: `Remember this project-only integration marker: ${marker}`,
});
const recalledBeforeIngestMs = Date.now() - before;

process.env.PLUGIN_DATA = dataDir;
process.env.TMCRA_PROJECT_ID = smokeId;
const config = await loadConfig();
const currentScopes = await resolveMemoryScopes({ cwd: process.cwd(), projectId: smokeId, config });
const logPath = join(dataDir, "logs", "events.jsonl");
const checkpointCursor = (await readEvents(logPath)).length;

await runHook("post_tool_use.mjs", {
  ...common,
  hook_event_name: "PostToolUse",
  tool_name: "shell_command",
  tool_input: { command: `verify-memory --password=${secretMarker}` },
  tool_response: { exit_code: 0, output: `${toolMarker} completed` },
});
await runHook("pre_compact.mjs", {
  ...common,
  hook_event_name: "PreCompact",
  trigger: "auto",
});
await runHook("post_compact.mjs", {
  ...common,
  hook_event_name: "PostCompact",
  trigger: "auto",
});
const checkpointSubmitEvent = await waitForSubmittedJob(
  logPath,
  currentScopes.projectId,
  checkpointCursor,
  Number(process.env.TMCRA_OUTBOX_WAIT_MS || 60_000),
);
assert(checkpointSubmitEvent?.jobId, "PreCompact hook did not record a checkpoint ingestion job ID");
const checkpointJob = await waitJob(checkpointSubmitEvent.jobId, {
  timeoutMs: Number(process.env.TMCRA_SMOKE_TIMEOUT_MS || 1_800_000),
  pollMs: 1500,
  config,
});
assert(checkpointJob.status === "succeeded", `checkpoint job ended with ${checkpointJob.status}`);

const compactStarted = Date.now();
const compactResume = await runHook("session_start.mjs", {
  ...common,
  hook_event_name: "SessionStart",
  source: "compact",
});
const compactResumeMs = Date.now() - compactStarted;
const compactContext = compactResume?.hookSpecificOutput?.additionalContext || "";
assert(compactContext.includes(marker), "compact resume omitted the current user objective");
assert(compactContext.includes(toolMarker), "compact resume omitted the latest tool progress");
assert(!compactContext.includes(secretMarker), "compact resume exposed a redacted credential marker");

const eventCursor = (await readEvents(logPath)).length;

const submittedAt = Date.now();
await runHook("stop.mjs", {
  ...common,
  hook_event_name: "Stop",
  stop_hook_active: false,
  last_assistant_message: `Confirmed the project-only integration marker ${marker}.`,
});
const submitMs = Date.now() - submittedAt;
const submitEvent = await waitForSubmittedJob(
  logPath,
  currentScopes.projectId,
  eventCursor,
  Number(process.env.TMCRA_OUTBOX_WAIT_MS || 60_000),
);
assert(submitEvent?.jobId, "Stop hook did not record an ingestion job ID");
const jobStartedAt = Date.now();
const job = await waitJob(submitEvent.jobId, {
  timeoutMs: Number(process.env.TMCRA_SMOKE_TIMEOUT_MS || 1_800_000),
  pollMs: 1500,
  config,
});
const jobMs = Date.now() - jobStartedAt;
assert(job.status === "succeeded", `ingestion job ended with ${job.status}`);

const recallTurnId = `turn-${randomUUID()}`;
const recalledAt = Date.now();
const recalled = await runHook("user_prompt_submit.mjs", {
  ...common,
  turn_id: recallTurnId,
  hook_event_name: "UserPromptSubmit",
  prompt: "What is the project-only TMCRA lifecycle marker?",
});
const recallMs = Date.now() - recalledAt;
const injected = recalled?.hookSpecificOutput?.additionalContext || "";
assert(injected.includes(marker), "current project recall did not inject the marker");
let answerAgent = { verified: false };
if (process.env.TMCRA_ANSWER_AGENT_URL) {
  const question = "What is the project-only TMCRA lifecycle marker?";
  const answerResponse = await fetch(process.env.TMCRA_ANSWER_AGENT_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ question, memory_context: injected }),
  });
  const answered = await answerResponse.json();
  assert(answerResponse.ok && answered.ok, "Codex answer agent returned an error");
  assert(String(answered.answer || "").includes(marker), "Codex answer agent did not use recalled memory");
  answerAgent = { verified: true, model: String(answered.model || "unknown") };
}

const otherScopes = await resolveMemoryScopes({
  cwd: process.cwd(),
  projectId: `${smokeId}-other-project`,
  config,
});
const isolationQuery = `Find the exact marker ${marker}`;

let globalContent = "";
try {
  globalContent = promptEvidenceContent(
    await recall({ query: isolationQuery, scope: currentScopes.globalScope, config }),
  );
} catch {
  globalContent = "";
}
let otherProjectContent = "";
try {
  otherProjectContent = promptEvidenceContent(
    await recall({ query: isolationQuery, scope: otherScopes.projectScope, config }),
  );
} catch {
  otherProjectContent = "";
}
assert(!globalContent.includes(marker), "project marker leaked into the user-global memory layer");
assert(!otherProjectContent.includes(marker), "project marker leaked into another project scope");

process.stdout.write(
  `${JSON.stringify(
    {
      ok: true,
      projectId: currentScopes.projectId,
      projectScope: currentScopes.projectScope,
      globalScope: currentScopes.globalScope,
      sessionId,
      jobId: submitEvent.jobId,
      jobStatus: job.status,
      checkpointJobId: checkpointSubmitEvent.jobId,
      checkpointJobStatus: checkpointJob.status,
      timingsMs: {
        initialRecall: recalledBeforeIngestMs,
        compactResume: compactResumeMs,
        ingestSubmit: submitMs,
        ingestJob: jobMs,
        recallAndInject: recallMs,
      },
      assertions: {
        currentProjectRecall: true,
        compactObjectiveRestored: true,
        compactToolProgressRestored: true,
        compactCredentialRedaction: true,
        globalIsolation: true,
        otherProjectIsolation: true,
        rolesWritten: ["user", "assistant"],
      },
      answerAgent,
      dataDir,
    },
    null,
    2,
  )}\n`,
);
