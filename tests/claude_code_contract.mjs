import { strict as assert } from "node:assert";
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { MockTmcraServer } from "./mock_tmcra_server.mjs";
import {
  loadConfig,
  loadRecallReceiptForTurn,
  resolveMemoryScopes,
} from "../scripts/tmcra_client.mjs";
import { pairingTurnId } from "../hooks/hook_common.mjs";

const testDir = dirname(fileURLToPath(import.meta.url));
const pluginRoot = resolve(testDir, "..");
const hooksDir = join(pluginRoot, "hooks");
const token = "tmcra-claude-contract-token";
const tempRoot = await mkdtemp(join(tmpdir(), "tmcra-claude-contract-"));
const project = join(tempRoot, "project");
const dataDir = join(tempRoot, "claude-plugin-data");
const transcript = join(project, "transcript.jsonl");
const server = new MockTmcraServer({ validTokens: [token] });
const testEnvironmentKeys = [
  "TMCRA_API_KEY",
  "TMCRA_ACCESS_TOKEN",
  "TMCRA_BASE_URL",
  "TMCRA_CLIENT_PLATFORM",
  "TMCRA_SCOPE_NAMESPACE",
  "CLAUDE_PLUGIN_ROOT",
  "CLAUDE_PLUGIN_DATA",
  "CLAUDE_PLUGIN_OPTION_API_ENDPOINT",
  "CLAUDE_PLUGIN_OPTION_API_TOKEN",
  "PLUGIN_DATA",
];
const originalEnvironment = Object.fromEntries(
  testEnvironmentKeys.map((key) => [key, process.env[key]]),
);

function runHook(script, input, env) {
  const targetByEvent = {
    SessionStart: "session_start.mjs",
    UserPromptSubmit: "user_prompt_submit.mjs",
    PreCompact: "pre_compact.mjs",
    PostCompact: "post_compact.mjs",
    Stop: "stop.mjs",
    StopFailure: "stop.mjs",
  };
  const target = targetByEvent[input.hook_event_name] || script;
  return new Promise((resolvePromise, reject) => {
    const child = spawn(process.execPath, [join(hooksDir, script), target], {
      cwd: project,
      env,
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.once("error", reject);
    child.once("close", (code) => {
      if (code !== 0) {
        reject(new Error(`${basename(script)} exited ${code}: ${stderr || stdout}`));
        return;
      }
      resolvePromise(JSON.parse(stdout.trim()));
    });
    child.stdin.end(`${JSON.stringify(input)}\n`);
  });
}

async function waitFor(predicate, timeoutMs = 8_000) {
  const deadline = Date.now() + timeoutMs;
  while (!(await predicate())) {
    if (Date.now() >= deadline) throw new Error("condition did not become true");
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 40));
  }
}

try {
  await server.start();
  const env = {
    ...process.env,
    TMCRA_CLIENT_PLATFORM: "claude-code",
    TMCRA_SCOPE_NAMESPACE: "claude-contract",
    CLAUDE_PLUGIN_ROOT: pluginRoot,
    CLAUDE_PLUGIN_DATA: dataDir,
    CLAUDE_PLUGIN_OPTION_API_ENDPOINT: server.baseUrl,
    CLAUDE_PLUGIN_OPTION_API_TOKEN: token,
    PLUGIN_DATA: dataDir,
  };
  delete env.TMCRA_API_KEY;
  delete env.TMCRA_ACCESS_TOKEN;
  delete env.TMCRA_BASE_URL;
  Object.assign(process.env, env);
  delete process.env.TMCRA_API_KEY;
  delete process.env.TMCRA_ACCESS_TOKEN;
  delete process.env.TMCRA_BASE_URL;
  await mkdir(project, { recursive: true });
  await mkdir(dataDir, { recursive: true });
  await writeFile(transcript, "", "utf8");

  const base = {
    session_id: "claude-contract-session",
    transcript_path: transcript,
    cwd: project,
    permission_mode: "default",
  };

  assert.deepEqual(await runHook("claude_run_hook.mjs", {
    ...base,
    hook_event_name: "SessionStart",
    source: "startup",
    model: "claude-sonnet-test",
  }, env), { continue: true });

  const first = "CLAUDE_CONTRACT_FIRST_PROMPT";
  const firstRecall = await runHook("claude_run_hook.mjs", {
    ...base,
    hook_event_name: "UserPromptSubmit",
    prompt: first,
  }, env);
  assert.equal(firstRecall.continue, true);
  assert.equal(firstRecall.hookSpecificOutput?.hookEventName, "UserPromptSubmit");

  await runHook("claude_run_hook.mjs", {
    ...base,
    hook_event_name: "Stop",
    stop_hook_active: false,
    last_assistant_message: "CLAUDE_CONTRACT_FIRST_RESPONSE",
    background_tasks: [{ id: "background-1", description: "pending test task" }],
    session_crons: [],
  }, env);
  await waitFor(() => server.records.length === 1);
  assert.equal(server.records[0].metadata.integration, "claude-code");
  assert.match(server.records[0].metadata.integration_version, /\+claude\./u);
  assert.equal(server.records[0].messages[0].content, first);
  assert(server.requests.some((request) => request.clientPlatform === "claude-code"));
  await waitFor(() => server.records.some((record) =>
    record.metadata.checkpoint_reason === "stop_with_background_work",
  ));
  assert(server.records.some((record) => record.metadata.checkpoint_reason === "stop_with_background_work"));

  const config = await loadConfig();
  assert.equal(config.configSource, "claude_plugin_config");
  assert.equal(config.baseUrl, server.baseUrl);
  const scopes = await resolveMemoryScopes({ cwd: project, config });
  const firstTurnId = pairingTurnId({ ...base, prompt: first });
  await waitFor(async () => {
    const receipt = await loadRecallReceiptForTurn(
      scopes.projectId,
      base.session_id,
      firstTurnId,
      "completed",
    );
    return receipt?.ingest?.state === "succeeded";
  });

  const secondRecall = await runHook("claude_run_hook.mjs", {
    ...base,
    hook_event_name: "UserPromptSubmit",
    prompt: "CLAUDE_CONTRACT_SECOND_RECALL",
  }, env);
  assert(secondRecall.hookSpecificOutput.additionalContext.includes("CLAUDE_CONTRACT_FIRST_RESPONSE"));

  await runHook("claude_run_hook.mjs", {
    ...base,
    hook_event_name: "PreCompact",
    trigger: "auto",
    custom_instructions: "",
  }, env);
  await runHook("claude_run_hook.mjs", {
    ...base,
    hook_event_name: "PostCompact",
    trigger: "auto",
    compact_summary: "CLAUDE_COMPACT_SUMMARY",
  }, env);

  await runHook("claude_run_hook.mjs", {
    ...base,
    hook_event_name: "StopFailure",
    error: "rate_limit",
    error_details: "provider detail must stay out of memory",
    last_assistant_message: "API Error: provider detail must stay out of memory",
  }, env);
  await waitFor(() => server.records.some((record) => record.metadata.checkpoint_reason === "stop_failure"));
  const failureCheckpoint = server.records.find((record) => record.metadata.checkpoint_reason === "stop_failure");
  assert(failureCheckpoint);
  assert(!JSON.stringify(failureCheckpoint).includes("provider detail must stay out of memory"));

  const ambiguousBase = { ...base, session_id: "claude-contract-ambiguous" };
  await Promise.all([
    runHook("claude_run_hook.mjs", {
      ...ambiguousBase,
      hook_event_name: "UserPromptSubmit",
      prompt: "CLAUDE_AMBIGUOUS_ONE",
    }, env),
    runHook("claude_run_hook.mjs", {
      ...ambiguousBase,
      hook_event_name: "UserPromptSubmit",
      prompt: "CLAUDE_AMBIGUOUS_TWO",
    }, env),
  ]);
  await runHook("claude_run_hook.mjs", {
    ...ambiguousBase,
    hook_event_name: "Stop",
    last_assistant_message: "MUST_NOT_BE_PAIRED",
  }, env);
  assert.equal(server.records.filter((record) => record.messages.some((message) => message.content === "MUST_NOT_BE_PAIRED")).length, 0);

  const claudeConfig = JSON.parse(await readFile(join(pluginRoot, ".claude-plugin", "plugin.json"), "utf8"));
  const claudeHooks = JSON.parse(await readFile(join(hooksDir, "claude-hooks.json"), "utf8"));
  assert.equal(claudeConfig.hooks, "./hooks/claude-hooks.json");
  assert.equal(claudeConfig.mcpServers, "./claude-mcp.json");
  assert.equal(claudeConfig.userConfig.api_token.sensitive, true);
  assert.equal(claudeConfig.license, "Apache-2.0");
  assert.match(claudeConfig.version, /^0\.3\.0-rc\.4\+claude\./u);
  assert.match(claudeHooks.hooks.UserPromptSubmit[0].hooks[0].args[0], /CLAUDE_PLUGIN_ROOT/u);
  assert.match(claudeHooks.hooks.UserPromptSubmit[0].hooks[0].args[0], /claude_run_hook\.mjs/u);

  process.stdout.write(`${JSON.stringify({
    ok: true,
    integration: "claude-code",
    assertions: {
      officialPayloadFields: true,
      explicitPlatformRouting: true,
      recallInjection: true,
      stopIngestReceiptBinding: true,
      stopFailureIsolation: true,
      compactionLifecycle: true,
      ambiguousNoTurnIdFailsClosed: true,
      separatePluginContract: true,
      sensitiveUserConfigInjection: true,
    },
    storedIngestCount: server.records.length,
  }, null, 2)}\n`);
} finally {
  for (const key of testEnvironmentKeys) {
    if (originalEnvironment[key] === undefined) delete process.env[key];
    else process.env[key] = originalEnvironment[key];
  }
  await server.stop();
  await rm(tempRoot, { recursive: true, force: true });
}
