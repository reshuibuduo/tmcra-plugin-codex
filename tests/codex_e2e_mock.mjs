import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { createInterface } from "node:readline";
import {
  appendFile,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { MockTmcraServer } from "./mock_tmcra_server.mjs";
import {
  getJob,
  resolveMemoryScopes,
  waitJob,
} from "../scripts/tmcra_client.mjs";
import { boundedHookConfig } from "../hooks/hook_common.mjs";

const testsDir = dirname(fileURLToPath(import.meta.url));
const pluginRoot = resolve(testsDir, "..");
const scriptsDir = join(pluginRoot, "scripts");
const hooksDir = join(pluginRoot, "hooks");
const validToken = "tmcra-test-valid-token";
const invalidToken = "tmcra-test-invalid-token";
const sensitiveValues = new Set([validToken, invalidToken]);
const capturedOutput = [];

assert.equal(boundedHookConfig({ timeoutMs: 120_000 }, {}).timeoutMs, 9_000);
assert.equal(
  boundedHookConfig({ timeoutMs: 120_000 }, { TMCRA_HOOK_REQUEST_TIMEOUT_MS: "8000" }).timeoutMs,
  8_000,
);
assert.equal(boundedHookConfig({ timeoutMs: 1_000 }, {}).timeoutMs, 1_000);

function redact(value) {
  let text = String(value || "");
  for (const secret of sensitiveValues) text = text.replaceAll(secret, "[REDACTED]");
  return text;
}

function cleanEnvironment(overrides = {}) {
  const env = { ...process.env };
  for (const name of Object.keys(env)) {
    if (name.startsWith("TMCRA_") || name === "PLUGIN_DATA" || name === "CLAUDE_PLUGIN_DATA") {
      delete env[name];
    }
  }
  return { ...env, ...overrides };
}

class ChildFailure extends Error {
  constructor(script, code, stdout, stderr) {
    super(`${basename(script)} exited ${code}: ${redact(stderr || stdout)}`);
    this.name = "ChildFailure";
    this.code = code;
    this.stdout = stdout;
    this.stderr = stderr;
  }
}

function runNode(script, args = [], { cwd = pluginRoot, env = process.env, input, timeoutMs = 30_000 } = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(process.execPath, [script, ...args], {
      cwd,
      env,
      stdio: ["pipe", "pipe", "pipe"],
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
      capturedOutput.push(stdout, stderr);
      if (code !== 0) {
        reject(new ChildFailure(script, code, stdout, stderr));
        return;
      }
      resolvePromise({ stdout, stderr, code });
    });
    child.stdin.end(input === undefined ? undefined : String(input));
  });
}

function parseJson(result) {
  return JSON.parse(result.stdout.trim());
}

async function expectChildFailure(callback, pattern) {
  let error = null;
  try {
    await callback();
  } catch (caught) {
    error = caught;
  }
  assert(error instanceof ChildFailure, "expected the child command to fail");
  if (pattern) assert.match(`${error.stdout}\n${error.stderr}`, pattern);
  return error;
}

class McpClient {
  constructor(env, cwd) {
    this.child = spawn(
      process.execPath,
      [join(pluginRoot, "hooks", "run_hook.mjs"), "mcp_server.mjs"],
      {
      cwd,
      env,
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
      },
    );
    this.nextId = 1;
    this.pending = new Map();
    this.stderr = "";
    this.child.stderr.on("data", (chunk) => (this.stderr += chunk));
    createInterface({ input: this.child.stdout }).on("line", (line) => {
      let message;
      try {
        message = JSON.parse(line);
      } catch (error) {
        for (const pending of this.pending.values()) pending.reject(error);
        this.pending.clear();
        return;
      }
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      clearTimeout(pending.timer);
      if (message.error) pending.reject(new Error(message.error.message));
      else pending.resolve(message.result);
    });
    this.child.once("exit", (code) => {
      for (const pending of this.pending.values()) {
        clearTimeout(pending.timer);
        pending.reject(new Error(`MCP server exited ${code}: ${redact(this.stderr)}`));
      }
      this.pending.clear();
    });
  }

  request(method, params = {}) {
    const id = this.nextId++;
    this.child.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`);
    return new Promise((resolvePromise, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP ${method} timed out: ${redact(this.stderr)}`));
      }, 15_000);
      this.pending.set(id, { resolve: resolvePromise, reject, timer });
    });
  }

  async initialize() {
    const initialized = await this.request("initialize", {
      protocolVersion: "2025-03-26",
      capabilities: {},
      clientInfo: { name: "tmcra-e2e", version: "0.2.8" },
    });
    this.child.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" })}\n`);
    return initialized;
  }

  call(name, args) {
    return this.request("tools/call", { name, arguments: args });
  }

  async close() {
    if (this.child.exitCode !== null) return;
    this.child.stdin.end();
    await new Promise((resolvePromise) => {
      const timer = setTimeout(() => {
        this.child.kill();
        resolvePromise();
      }, 2000);
      this.child.once("close", () => {
        clearTimeout(timer);
        capturedOutput.push(this.stderr);
        resolvePromise();
      });
    });
  }
}

function message(messageId, role, content) {
  return {
    message_id: messageId,
    role,
    content,
    timestamp: "2026-07-16T00:00:00.000Z",
  };
}

const results = [];
async function test(name, callback) {
  const started = Date.now();
  await callback();
  results.push({ name, ok: true, elapsedMs: Date.now() - started });
}

async function waitFor(predicate, { timeoutMs = 8_000, intervalMs = 50 } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (!(await predicate())) {
    if (Date.now() >= deadline) throw new Error(`condition did not become true within ${timeoutMs}ms`);
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

const tempPrefix = "tmcra-codex-e2e-";
const tempRoot = await mkdtemp(join(tmpdir(), tempPrefix));
const projectA = join(tempRoot, "project-a");
const projectANested = join(projectA, "src", "nested");
const projectB = join(tempRoot, "project-b");
const gitProject = join(tempRoot, "git-project");
const gitProjectNested = join(gitProject, "packages", "child");
const pathProject = join(tempRoot, "path-project");
const bootstrapProject = join(tempRoot, "legacy-bootstrap");
const codexHome = join(tempRoot, "codex-home");
const dataDir = join(tempRoot, "plugin-data");
const configPath = join(tempRoot, "config", "tmcra.json");
const server = new MockTmcraServer({ validTokens: [validToken] });

let env;
let config;
let scopesA;
let scopesB;
let mcpProjectJobId;

try {
  await Promise.all([
    mkdir(join(projectA, ".tmcra"), { recursive: true }),
    mkdir(projectANested, { recursive: true }),
    mkdir(join(projectB, ".tmcra"), { recursive: true }),
    mkdir(join(gitProject, ".git"), { recursive: true }),
    mkdir(gitProjectNested, { recursive: true }),
    mkdir(pathProject, { recursive: true }),
    mkdir(join(bootstrapProject, ".tmcra"), { recursive: true }),
    mkdir(join(codexHome, "sessions", "2026", "07", "16"), { recursive: true }),
    mkdir(dataDir, { recursive: true }),
  ]);
  await Promise.all([
    writeFile(join(projectA, ".tmcra", "project.json"), JSON.stringify({
      schemaVersion: 2,
      projectId: "e2e-project-a",
      name: "E2E Project A",
      scopeName: "personal-e2e-project-memory-os-1111111111111111",
    })),
    writeFile(join(projectB, ".tmcra", "project.json"), JSON.stringify({
      schemaVersion: 1,
      projectId: "e2e-project-b",
      name: "E2E Project B",
    })),
    writeFile(join(gitProject, ".git", "config"), [
      '[remote "origin"]',
      "  url = https://example.test/tmcra/e2e-git-project.git",
      "",
    ].join("\n")),
    writeFile(join(bootstrapProject, ".tmcra", "project.json"), JSON.stringify({
      schemaVersion: 1,
      projectId: "e2e-bootstrap",
      name: "E2E Bootstrap",
    })),
    writeFile(join(bootstrapProject, "README.md"), "# Legacy bootstrap\n\nBOOTSTRAP_VISIBLE_MARKER\n"),
    writeFile(join(bootstrapProject, ".env"), "MUST_NOT_UPLOAD_BOOTSTRAP\n"),
  ]);
  await server.start();

  const baseEnv = cleanEnvironment({
    TMCRA_CONFIG_FILE: configPath,
    CODEX_HOME: codexHome,
    PLUGIN_ROOT: pluginRoot,
    CLAUDE_PLUGIN_ROOT: pluginRoot,
    PLUGIN_DATA: dataDir,
    CLAUDE_PLUGIN_DATA: dataDir,
  });

  await test("configuration write and authenticated check", async () => {
    const configured = await runNode(join(scriptsDir, "configure.mjs"), [], {
      env: {
        ...baseEnv,
        TMCRA_SETUP_API_KEY: validToken,
        TMCRA_BASE_URL: server.baseUrl,
        TMCRA_SCOPE_NAMESPACE: "codex-e2e",
      },
    });
    const output = parseJson(configured);
    assert.equal(output.ok, true);
    assert.equal(output.apiKeyStored, true);
    assert(!configured.stdout.includes(validToken), "configure output exposed the API token");
    config = JSON.parse(await readFile(configPath, "utf8"));
    assert.equal(config.apiKey, validToken);
    assert.equal(config.baseUrl, server.baseUrl);
    assert.equal(config.globalScope, "codex-e2e-global");
    env = { ...baseEnv };

    const recallsBeforeCheck = server.requests.filter((item) => item.pathname.endsWith("/recall")).length;
    const checked = await runNode(join(scriptsDir, "check_config.mjs"), ["--api-only"], {
      cwd: projectA,
      env,
    });
    const check = parseJson(checked);
    assert.equal(check.ok, true);
    assert.equal(check.authenticated, true);
    assert.equal(check.apiKeyExposed, false);
    assert.equal(check.scopeReady, null);
    assert.equal(check.queryId, null);
    assert.equal(check.serviceVersion, "0.2.0-mock");
    assert(check.remoteCapabilities.includes("quota_reporting"));
    assert.equal(
      server.requests.filter((item) => item.pathname === "/v1/session").length,
      1,
    );
    assert.equal(
      server.requests.filter((item) => item.pathname.endsWith("/recall")).length,
      recallsBeforeCheck,
      "config check must not consume a recall request",
    );
    assert(!checked.stdout.includes(validToken), "config check output exposed the API token");
  });

  await test("Codex plugin manifest, lifecycle hooks, and bundled MCP contract", async () => {
    const manifest = JSON.parse(await readFile(join(pluginRoot, ".codex-plugin", "plugin.json"), "utf8"));
    const hooks = JSON.parse(await readFile(join(hooksDir, "hooks.json"), "utf8"));
    const mcp = JSON.parse(await readFile(join(pluginRoot, ".mcp.json"), "utf8"));
    assert.equal(manifest.name, "tmcra-memory");
    assert.equal(manifest.mcpServers, "./.mcp.json");
    assert.deepEqual(Object.keys(hooks.hooks).sort(), [
      "PostCompact",
      "PostToolUse",
      "PreCompact",
      "SessionStart",
      "Stop",
      "StopFailure",
      "SubagentStart",
      "SubagentStop",
      "UserPromptSubmit",
    ]);
    assert.match(hooks.hooks.PostCompact[0].hooks[0].command, /post_compact\.mjs/u);
    assert.match(hooks.hooks.PostToolUse[0].hooks[0].command, /post_tool_use\.mjs/u);
    assert.match(hooks.hooks.PreCompact[0].hooks[0].command, /pre_compact\.mjs/u);
    assert.match(hooks.hooks.SessionStart[0].hooks[0].command, /session_start\.mjs/u);
    assert.match(hooks.hooks.SubagentStart[0].hooks[0].command, /subagent_start\.mjs/u);
    assert.match(hooks.hooks.UserPromptSubmit[0].hooks[0].command, /user_prompt_submit\.mjs/u);
    assert.match(hooks.hooks.Stop[0].hooks[0].command, /stop\.mjs/u);
    assert.match(hooks.hooks.StopFailure[0].hooks[0].command, /stop\.mjs/u);
    assert.match(hooks.hooks.SubagentStop[0].hooks[0].command, /subagent_stop\.mjs/u);

    const releaseScript = await readFile(
      join(pluginRoot, "scripts", "build_release.ps1"),
      "utf8",
    );
    const referencedHookScripts = Object.values(hooks.hooks)
      .flatMap((groups) => groups)
      .flatMap((group) => group.hooks || [])
      .map((hook) => /run_hook\.mjs" "([^"]+\.mjs)/u.exec(String(hook.command || ""))?.[1])
      .filter(Boolean);
    assert.match(releaseScript, /plugins\/tmcra-memory\/hooks\/run_hook\.mjs/u);
    for (const script of referencedHookScripts) {
      assert.match(
        releaseScript,
        new RegExp(`plugins/tmcra-memory/hooks/${script.replaceAll(".", "\\.")}`, "u"),
      );
    }
    assert.equal(mcp.mcpServers["tmcra-memory"].command, "node");
    assert.deepEqual(mcp.mcpServers["tmcra-memory"].args, [
      "./hooks/run_hook.mjs",
      "mcp_server.mjs",
    ]);
  });

  await test("legacy lifecycle logs cannot satisfy the current Hook contract", async () => {
    const logPath = join(dataDir, "logs", "events.jsonl");
    await mkdir(dirname(logPath), { recursive: true });
    const legacyEvents = ["session_started", "recall_completed", "ingest_submitted", "ingest_succeeded"]
      .map((event) => JSON.stringify({
        at: new Date().toISOString(),
        event,
        pluginVersion: "0.2.8+codex.20260805155645",
      }))
      .join("\n");
    await appendFile(logPath, `${legacyEvents}\n`, "utf8");
    const client = new McpClient(env, projectA);
    try {
      await client.initialize();
      const status = await client.call("tmcra_status", {});
      assert.equal(status.isError, false);
      assert.equal(status.structuredContent.lifecycle.observed, false);
      assert.equal(status.structuredContent.lifecycle.eventCount, 0);
    } finally {
      await client.close();
    }
  });

  await test("Codex subagent lifecycle is isolated and captured through SubagentStop", async () => {
    const parentSessionId = "codex-subagent-parent-session";
    const parentTurnId = "codex-parent-turn";
    const agentId = "codex-subagent-id";
    const subagentTurnId = "codex-subagent-turn";
    const parentInput = {
      hook_event_name: "UserPromptSubmit",
      session_id: parentSessionId,
      turn_id: parentTurnId,
      cwd: projectA,
      model: "e2e",
      prompt: "PARENT_OBJECTIVE_MUST_REMAIN_ISOLATED",
    };
    await runNode(join(hooksDir, "user_prompt_submit.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify(parentInput),
    });

    const parentStateName = `${createHash("sha256").update(parentSessionId).digest("hex")}.json`;
    const parentStatePath = join(dataDir, "task-state", parentStateName);
    assert.equal(JSON.parse(await readFile(parentStatePath, "utf8")).objective, parentInput.prompt);

    const subagentBase = {
      session_id: parentSessionId,
      turn_id: subagentTurnId,
      agent_id: agentId,
      agent_type: "worker",
      cwd: projectA,
      model: "e2e",
    };
    const started = parseJson(await runNode(join(hooksDir, "subagent_start.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify({ ...subagentBase, hook_event_name: "SubagentStart" }),
    }));
    assert.deepEqual(started, { continue: true });

    const subagentPrompt = "SUBAGENT_USER_PROMPT_MUST_BE_CAPTURED";
    await runNode(join(hooksDir, "user_prompt_submit.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify({
        ...subagentBase,
        hook_event_name: "UserPromptSubmit",
        prompt: subagentPrompt,
      }),
    });
    assert.equal(
      JSON.parse(await readFile(parentStatePath, "utf8")).objective,
      parentInput.prompt,
      "subagent prompt overwrote the parent task objective",
    );
    const agentKey = createHash("sha256").update(agentId).digest("hex").slice(0, 24);
    const lifecycleSessionId = `${parentSessionId}:subagent:${agentKey}`;
    const subagentStateName = `${createHash("sha256").update(lifecycleSessionId).digest("hex")}.json`;
    assert.equal(
      JSON.parse(await readFile(join(dataDir, "task-state", subagentStateName), "utf8")).objective,
      subagentPrompt,
    );

    const recordsBeforeSubagentStop = server.records.length;
    const stopped = parseJson(await runNode(join(hooksDir, "subagent_stop.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify({
        ...subagentBase,
        hook_event_name: "SubagentStop",
        last_assistant_message: "SUBAGENT_ASSISTANT_RESPONSE_MUST_BE_CAPTURED",
      }),
    }));
    assert.deepEqual(stopped, { continue: true });
    await waitFor(() => server.records.length === recordsBeforeSubagentStop + 1);
    const subagentRecord = server.records.at(-1);
    assert.deepEqual(
      subagentRecord.messages.map((item) => item.content),
      [subagentPrompt, "SUBAGENT_ASSISTANT_RESPONSE_MUST_BE_CAPTURED"],
    );
    assert.equal(subagentRecord.sessionId, `codex-${createHash("sha256").update(lifecycleSessionId).digest("hex").slice(0, 40)}`);
    assert(!(await readdir(join(dataDir, "task-state"))).includes(subagentStateName));
    const bindingName = `${createHash("sha256").update(`${parentSessionId}:${subagentTurnId}`).digest("hex")}.json`;
    assert(!(await readdir(join(dataDir, "subagent-bindings"))).includes(bindingName));

    const recordsBeforeParentStop = server.records.length;
    await runNode(join(hooksDir, "stop.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify({
        ...parentInput,
        hook_event_name: "Stop",
        last_assistant_message: "PARENT_ASSISTANT_RESPONSE_REMAINS_SEPARATE",
      }),
    });
    await waitFor(() => server.records.length === recordsBeforeParentStop + 1);
    const parentRecord = server.records.at(-1);
    assert.notEqual(parentRecord.sessionId, subagentRecord.sessionId);
    assert.deepEqual(
      parentRecord.messages.map((item) => item.content),
      [parentInput.prompt, "PARENT_ASSISTANT_RESPONSE_REMAINS_SEPARATE"],
    );
  });

  await test("Claude-style lifecycle pairs concurrent prompts without turn_id", async () => {
    const claudeEnv = { ...env };
    delete claudeEnv.CODEX_HOME;
    const transcriptPath = join(tempRoot, "claude-session.jsonl");
    const sessionId = "claude-concurrent-no-turn-id";
    const baseInput = {
      session_id: sessionId,
      transcript_path: transcriptPath,
      cwd: projectA,
      permission_mode: "default",
    };
    const firstPrompt = "CLAUDE_CONCURRENT_FIRST_PROMPT";
    const secondPrompt = "CLAUDE_CONCURRENT_SECOND_PROMPT";
    await Promise.all([
      runNode(join(hooksDir, "user_prompt_submit.mjs"), [], {
        cwd: projectA,
        env: claudeEnv,
        input: JSON.stringify({
          ...baseInput,
          hook_event_name: "UserPromptSubmit",
          prompt_id: "claude-prompt-first",
          prompt: firstPrompt,
        }),
      }),
      runNode(join(hooksDir, "user_prompt_submit.mjs"), [], {
        cwd: projectA,
        env: claudeEnv,
        input: JSON.stringify({
          ...baseInput,
          hook_event_name: "UserPromptSubmit",
          prompt_id: "claude-prompt-second",
          prompt: secondPrompt,
        }),
      }),
    ]);
    await writeFile(transcriptPath, [
      JSON.stringify({ role: "user", content: firstPrompt }),
      JSON.stringify({ role: "user", content: secondPrompt }),
    ].join("\n") + "\n", "utf8");
    const before = server.records.length;
    await runNode(join(hooksDir, "stop.mjs"), [], {
      cwd: projectA,
      env: claudeEnv,
      input: JSON.stringify({
        ...baseInput,
        hook_event_name: "Stop",
        prompt_id: "claude-prompt-second",
        last_assistant_message: "CLAUDE_SECOND_RESPONSE",
      }),
    });
    await waitFor(() => server.records.length === before + 1);
    const secondRecord = server.records.at(-1);
    assert.deepEqual(
      secondRecord.messages.map((message) => message.content),
      [secondPrompt, "CLAUDE_SECOND_RESPONSE"],
    );

    const firstStop = await runNode(join(hooksDir, "stop.mjs"), [], {
      cwd: projectA,
      env: claudeEnv,
      input: JSON.stringify({
        ...baseInput,
        hook_event_name: "Stop",
        prompt_id: "claude-prompt-first",
        last_assistant_message: "CLAUDE_FIRST_RESPONSE",
      }),
    });
    assert.deepEqual(parseJson(firstStop), { continue: true });
    await waitFor(() => server.records.length === before + 2);
    assert.deepEqual(
      server.records.slice(-2).map((record) => record.messages.map((message) => message.content)),
      [
        [secondPrompt, "CLAUDE_SECOND_RESPONSE"],
        [firstPrompt, "CLAUDE_FIRST_RESPONSE"],
      ],
    );

    const crossTurnPrompt = "CLAUDE_NO_TURN_ID_CROSS_TURN_RECALL";
    const crossTurn = parseJson(await runNode(join(hooksDir, "user_prompt_submit.mjs"), [], {
      cwd: projectA,
      env: claudeEnv,
      input: JSON.stringify({
        ...baseInput,
        hook_event_name: "UserPromptSubmit",
        prompt_id: "claude-prompt-cross-turn",
        prompt: crossTurnPrompt,
      }),
    }));
    assert(crossTurn.hookSpecificOutput.additionalContext.includes("CLAUDE_FIRST_RESPONSE"));
    await runNode(join(hooksDir, "stop.mjs"), [], {
      cwd: projectA,
      env: claudeEnv,
      input: JSON.stringify({
        ...baseInput,
        hook_event_name: "Stop",
        prompt_id: "claude-prompt-cross-turn",
        last_assistant_message: "CLAUDE_CROSS_TURN_RESPONSE",
      }),
    });
    await waitFor(() => server.records.length === before + 3);

    const ambiguousSessionId = "claude-ambiguous-no-stable-event";
    const ambiguousInput = {
      ...baseInput,
      session_id: ambiguousSessionId,
    };
    await Promise.all([
      runNode(join(hooksDir, "user_prompt_submit.mjs"), [], {
        cwd: projectA,
        env: claudeEnv,
        input: JSON.stringify({
          ...ambiguousInput,
          hook_event_name: "UserPromptSubmit",
          prompt: "CLAUDE_AMBIGUOUS_FIRST",
        }),
      }),
      runNode(join(hooksDir, "user_prompt_submit.mjs"), [], {
        cwd: projectA,
        env: claudeEnv,
        input: JSON.stringify({
          ...ambiguousInput,
          hook_event_name: "UserPromptSubmit",
          prompt: "CLAUDE_AMBIGUOUS_SECOND",
        }),
      }),
    ]);
    await runNode(join(hooksDir, "stop.mjs"), [], {
      cwd: projectA,
      env: claudeEnv,
      input: JSON.stringify({
        ...ambiguousInput,
        hook_event_name: "Stop",
        last_assistant_message: "CLAUDE_AMBIGUOUS_MUST_NOT_BE_PERSISTED",
      }),
    });
    assert.equal(server.records.length, before + 3);
    const ambiguousStateName = `${createHash("sha256").update(ambiguousSessionId).digest("hex")}.json`;
    const ambiguousStateKey = ambiguousStateName.slice(0, -5);
    await Promise.all([
      rm(join(dataDir, "task-state", ambiguousStateName), { force: true }),
      rm(join(dataDir, "task-checkpoints", ambiguousStateName), { force: true }),
      rm(join(dataDir, "pending-index", ambiguousStateName), { force: true }),
      rm(join(dataDir, "task-events", ambiguousStateKey), { recursive: true, force: true }),
    ]);
  });

  await test("StopFailure checkpoints the user prompt without persisting provider error text", async () => {
    const claudeEnv = { ...env };
    delete claudeEnv.CODEX_HOME;
    const transcriptPath = join(tempRoot, "claude-failure-session.jsonl");
    const sessionId = "claude-stop-failure-session";
    const prompt = "CLAUDE_FAILURE_USER_PROMPT_MUST_SURVIVE";
    const baseInput = {
      session_id: sessionId,
      transcript_path: transcriptPath,
      cwd: projectA,
      permission_mode: "default",
    };
    await runNode(join(hooksDir, "user_prompt_submit.mjs"), [], {
      cwd: projectA,
      env: claudeEnv,
      input: JSON.stringify({
        ...baseInput,
        hook_event_name: "UserPromptSubmit",
        prompt,
      }),
    });
    await writeFile(transcriptPath, `${JSON.stringify({ role: "user", content: prompt })}\n`, "utf8");
    const failureResult = await runNode(join(hooksDir, "stop.mjs"), [], {
      cwd: projectA,
      env: claudeEnv,
      input: JSON.stringify({
        ...baseInput,
        hook_event_name: "StopFailure",
        error: "rate_limit",
        error_details: "429 Too Many Requests",
        last_assistant_message: "API Error: RATE_LIMIT_PROVIDER_TEXT_MUST_NOT_BE_MEMORY",
      }),
    });
    assert.deepEqual(parseJson(failureResult), { continue: true });
    await waitFor(() => server.records.some((record) =>
      record.metadata.checkpoint_reason === "stop_failure" &&
      record.messages.some((message) => message.content.includes(prompt)),
    ));
    const checkpoint = server.records.find((record) =>
      record.metadata.checkpoint_reason === "stop_failure" &&
      record.messages.some((message) => message.content.includes(prompt)),
    );
    assert.equal(checkpoint.metadata.checkpoint_reason, "stop_failure");
    assert(checkpoint.messages[0].content.includes(prompt));
    assert(!checkpoint.messages[0].content.includes("RATE_LIMIT_PROVIDER_TEXT_MUST_NOT_BE_MEMORY"));
    assert(!JSON.stringify(checkpoint).includes("429 Too Many Requests"));
    const failureStateName = `${createHash("sha256").update(sessionId).digest("hex")}.json`;
    await Promise.all([
      rm(join(dataDir, "task-state", failureStateName), { force: true }),
      rm(join(dataDir, "task-checkpoints", failureStateName), { force: true }),
      rm(join(dataDir, "pending-index", failureStateName), { force: true }),
      rm(join(dataDir, "task-events", failureStateName.slice(0, -5)), { recursive: true, force: true }),
    ]);
  });

  await test("stable project identity and global/project scope partition", async () => {
    scopesA = await resolveMemoryScopes({ cwd: projectA, config });
    const nestedScopes = await resolveMemoryScopes({ cwd: projectANested, config });
    scopesB = await resolveMemoryScopes({ cwd: projectB, config });
    const gitScopes = await resolveMemoryScopes({ cwd: gitProject, config });
    const gitNestedScopes = await resolveMemoryScopes({ cwd: gitProjectNested, config });
    const pathScopes = await resolveMemoryScopes({ cwd: pathProject, config });
    const configuredScopes = await resolveMemoryScopes({
      cwd: projectA,
      projectId: "explicit-project-identity",
      config,
    });
    assert.equal(scopesA.projectIdentitySource, "marker");
    assert.equal(scopesA.projectScope, "personal-e2e-project-memory-os-1111111111111111");
    assert.equal(scopesA.projectScope, nestedScopes.projectScope);
    assert.equal(scopesA.projectId, nestedScopes.projectId);
    assert.equal(scopesA.globalScope, scopesB.globalScope);
    assert.notEqual(scopesA.projectScope, scopesB.projectScope);
    assert.notEqual(scopesA.projectId, scopesB.projectId);
    assert.equal(gitScopes.projectIdentitySource, "git-origin");
    assert.equal(gitScopes.projectScope, gitNestedScopes.projectScope);
    assert.equal(pathScopes.projectIdentitySource, "path");
    assert.equal(configuredScopes.projectIdentitySource, "configured");
    assert.equal(configuredScopes.projectScope, scopesA.projectScope);
  });

  await test("MCP tools, async jobs, scopes, and idempotent ingest", async () => {
    const client = new McpClient(env, projectA);
    try {
      const initialized = await client.initialize();
      assert.equal(initialized.serverInfo.name, "TMCRA Memory");
      assert.equal(initialized.capabilities.resources.listChanged, false);
      const listed = await client.request("tools/list");
      assert.deepEqual(
        listed.tools.map((tool) => tool.name).sort(),
        ["tmcra_consolidate", "tmcra_get_job", "tmcra_ingest", "tmcra_last_recall", "tmcra_memory_control", "tmcra_open_local_install", "tmcra_open_local_model_settings", "tmcra_open_memory_center", "tmcra_recall", "tmcra_status", "tmcra_wait_job"],
      );
      assert.equal(
        listed.tools.find((tool) => tool.name === "tmcra_status")._meta.ui.resourceUri,
        "ui://tmcra/memory-status-v1.html",
      );
      assert.equal(
        listed.tools.find((tool) => tool.name === "tmcra_last_recall")._meta.ui.resourceUri,
        "ui://tmcra/recall-inspector-v1.html",
      );
      const resources = await client.request("resources/list");
      assert.deepEqual(
        resources.resources.map((resource) => resource.uri).sort(),
        ["ui://tmcra/memory-status-v1.html", "ui://tmcra/recall-inspector-v1.html"],
      );
      const statusWidget = await client.request("resources/read", {
        uri: "ui://tmcra/memory-status-v1.html",
      });
      assert.equal(statusWidget.contents[0].mimeType, "text/html;profile=mcp-app");
      assert.match(statusWidget.contents[0].text, /TMCRA \/ MEMORY STATUS/u);

      const statusResult = await client.call("tmcra_status", {});
      assert.equal(statusResult.isError, false);
      assert.equal(statusResult._meta["openai/outputTemplate"], "ui://tmcra/memory-status-v1.html");
      assert.equal(statusResult.structuredContent.ok, true);
      assert.equal(statusResult.structuredContent.serviceOnline, true);
      assert.equal(statusResult.structuredContent.authorization.source, "device_config");
      assert.equal(statusResult.structuredContent.authorization.credentialPresent, true);

      const offlineConfigPath = join(tempRoot, "config", "tmcra-offline.json");
      await writeFile(offlineConfigPath, JSON.stringify({
        baseUrl: "http://localhost:1",
        accessToken: validToken,
        scopeNamespace: "codex-e2e",
        timeoutMs: 200,
      }));
      const offlineClient = new McpClient({
        ...env,
        TMCRA_CONFIG_FILE: offlineConfigPath,
        TMCRA_REQUEST_TIMEOUT_MS: "200",
      }, projectA);
      try {
        await offlineClient.initialize();
        const offlineStatus = await offlineClient.call("tmcra_status", {});
        assert.equal(offlineStatus.isError, false, JSON.stringify(offlineStatus));
        assert.equal(offlineStatus.structuredContent.ok, false);
        assert.equal(offlineStatus.structuredContent.serviceOnline, false);
        assert.equal(offlineStatus.structuredContent.authorization.credentialPresent, true);
        assert.equal(offlineStatus.structuredContent.authorization.authenticated, false);
        assert.equal(offlineStatus.structuredContent.recovery.state, "unknown");
        assert.match(offlineStatus.structuredContent.nextAction, /local durable queue/iu);
        assert.doesNotMatch(
          JSON.stringify(offlineStatus.structuredContent),
          /plugin-data|config\\tmcra|apiKey/iu,
        );
      } finally {
        await offlineClient.close();
      }

      const globalResult = await client.call("tmcra_ingest", {
        session_id: "global-profile-session",
        memory_layer: "global",
        project_path: projectA,
        idempotency_key: "e2e-global-idempotency",
        messages: [message("global-1", "user", "GLOBAL_PROFILE_MARKER")],
      });
      assert.equal(globalResult.isError, false);
      assert.equal(globalResult.structuredContent.memory_layer, "global");
      assert.equal(Object.hasOwn(globalResult.structuredContent, "scope_name"), false);

      const projectArgs = {
        session_id: "mcp-project-session",
        memory_layer: "project",
        project_path: projectA,
        idempotency_key: "e2e-project-idempotency",
        messages: [message("project-1", "user", "PROJECT_A_MCP_MARKER")],
      };
      const projectResult = await client.call("tmcra_ingest", projectArgs);
      assert.equal(projectResult.isError, false);
      mcpProjectJobId = projectResult.structuredContent.job_id;
      assert.equal(projectResult.structuredContent.status, "queued");

      const duplicate = await client.call("tmcra_ingest", projectArgs);
      assert.equal(duplicate.isError, false);
      assert.equal(duplicate.structuredContent.job_id, mcpProjectJobId);
      assert.equal(duplicate.structuredContent.duplicate, true);

      const inspected = await client.call("tmcra_get_job", { job_id: mcpProjectJobId });
      assert.equal(inspected.isError, false);
      assert.equal(inspected.structuredContent.status, "succeeded");
      assert.equal(inspected.structuredContent.writer.input_messages, 1);
      assert.match(inspected.structuredContent.finished_at, /^\d{4}-\d{2}-\d{2}T/u);
      assert.doesNotMatch(
        JSON.stringify(inspected.structuredContent),
        /db_path|tenant_id|scope_name|\/root\/private/u,
      );
      const waited = await client.call("tmcra_wait_job", {
        job_id: mcpProjectJobId,
        timeout_seconds: 2,
        poll_interval_seconds: 0.1,
      });
      assert.equal(waited.isError, false);
      assert.equal(waited.structuredContent.status, "succeeded");
      assert.match(waited.structuredContent.finished_at, /^\d{4}-\d{2}-\d{2}T/u);

      const recalledA = await client.call("tmcra_recall", {
        query: "profile and project markers",
        memory_layer: "auto",
        project_path: projectA,
        include_structured_evidence: true,
      });
      const contextA = recalledA.structuredContent.prompt_evidence.content;
      assert(contextA.includes("GLOBAL_PROFILE_MARKER"));
      assert(contextA.includes("PROJECT_A_MCP_MARKER"));
      assert(contextA.includes('tmcra_memory trust="untrusted"'));
      assert.doesNotMatch(
        JSON.stringify(recalledA.structuredContent),
        /db_path|scope_id|retrieval_metadata|\/root\/private/u,
      );

      const recalledB = await client.call("tmcra_recall", {
        query: "profile and project markers",
        memory_layer: "auto",
        project_path: projectB,
      });
      const contextB = recalledB.structuredContent.prompt_evidence.content;
      assert(contextB.includes("GLOBAL_PROFILE_MARKER"));
      assert(!contextB.includes("PROJECT_A_MCP_MARKER"));

      const invalidArguments = await client.call("tmcra_ingest", {
        session_id: "empty-messages",
        messages: [],
      });
      assert.equal(invalidArguments.isError, true);
    } finally {
      await client.close();
    }
    assert.equal(
      server.records.filter((record) => record.idempotencyKey === "e2e-project-idempotency").length,
      1,
      "idempotent MCP ingest stored duplicate data",
    );
  });

  await test("Session controls reach real hooks and preserve continuation queries", async () => {
    const client = new McpClient(env, projectA);
    try {
      await client.initialize();
      for (const mode of ["off", "recall_only"]) {
        const sessionId = `controls-${mode}`;
        const control = await client.call("tmcra_memory_control", { operation: "mode", mode, session_id: sessionId, project_path: projectA });
        assert.equal(control.isError, false);
        const before = server.requests.filter((item) => item.pathname.endsWith("/recall")).length;
        const input = { session_id: sessionId, turn_id: "private-turn", cwd: projectA, prompt: `DO_NOT_BACKFILL_${mode}` };
        parseJson(await runNode(join(hooksDir, "user_prompt_submit.mjs"), [], { cwd: projectA, env, input: JSON.stringify(input) }));
        assert.equal(server.requests.filter((item) => item.pathname.endsWith("/recall")).length - before, mode === "off" ? 0 : 2);
        await client.call("tmcra_memory_control", { operation: "mode", mode: "normal", session_id: sessionId, project_path: projectA });
        await runNode(join(hooksDir, "stop.mjs"), [], { cwd: projectA, env, input: JSON.stringify({ ...input, last_assistant_message: "PRIVATE_RESULT_MUST_NOT_BACKFILL" }) });
      }
      assert(!JSON.stringify(server.records).includes("DO_NOT_BACKFILL_"));
      assert(!JSON.stringify(server.records).includes("PRIVATE_RESULT_MUST_NOT_BACKFILL"));
      await client.call("tmcra_memory_control", { operation: "task", session_id: "controls-continuation", project_path: projectA,
        objective: "CONTINUITY_TARGET_AUTHENTICATION", nextStep: "VERIFY_TOKEN_EXPIRY" });
      const continued = parseJson(await runNode(join(hooksDir, "user_prompt_submit.mjs"), [], { cwd: projectA, env,
        input: JSON.stringify({ session_id: "controls-continuation", turn_id: "continue-one", cwd: projectA, prompt: "继续" }) }));
      assert.match(continued.hookSpecificOutput.additionalContext, /CONTINUITY_TARGET_AUTHENTICATION/u);
      assert(server.requests.filter((item) => item.pathname.endsWith("/recall")).slice(-2).every((item) => item.query.includes("VERIFY_TOKEN_EXPIRY")));
    } finally { await client.close(); }
  });

  await test("Codex hooks recall, Stop ingest, sessions, and cross-session memory", async () => {
    const recallsBeforeSessionStart = server.requests.filter(
      (item) => item.pathname.endsWith("/recall"),
    ).length;
    const sessionStart = parseJson(await runNode(join(hooksDir, "session_start.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify({
        hook_event_name: "SessionStart",
        session_id: "codex-session-start",
        cwd: projectA,
        model: "e2e",
      }),
    }));
    assert.deepEqual(sessionStart, { continue: true });
    assert.equal(
      server.requests.filter((item) => item.pathname.endsWith("/recall")).length,
      recallsBeforeSessionStart,
      "SessionStart must initialize scope without recalling or injecting memory",
    );

    const firstInput = {
      hook_event_name: "UserPromptSubmit",
      session_id: "codex-session-one",
      turn_id: "turn-one",
      cwd: projectA,
      model: "e2e",
      prompt: "USER_REQUIREMENT_MARKER: Build the Codex integration with actor-safe recall. password: HOOK_USER_SECRET_1234",
    };
    const safeFirstPrompt = "USER_REQUIREMENT_MARKER: Build the Codex integration with actor-safe recall. password: [REDACTED]";
    const promptResult = parseJson(await runNode(join(hooksDir, "user_prompt_submit.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify(firstInput),
    }));
    assert.equal(promptResult.continue, true);
    assert.equal(
      server.requests.filter((item) => item.pathname.endsWith("/recall")).length,
      recallsBeforeSessionStart + 2,
      "UserPromptSubmit must recall global and project memory for the current prompt",
    );
    assert(
      server.requests
        .filter((item) => item.pathname.endsWith("/recall"))
        .slice(-2)
        .every(
          (item) =>
            item.query === safeFirstPrompt &&
            item.evidenceMode === "raw" &&
            item.recallProfile === "interactive" &&
            item.responseProjection === "prompt_only",
        ),
      "UserPromptSubmit recall must use the current prompt and compact interactive response",
    );
    const promptContext = promptResult.hookSpecificOutput.additionalContext;
    assert(promptContext.includes("GLOBAL_PROFILE_MARKER"));
    assert(promptContext.includes("PROJECT_A_MCP_MARKER"));
    const receiptClient = new McpClient(env, projectA);
    try {
      await receiptClient.initialize();
      const receipt = await receiptClient.call("tmcra_last_recall", {
        project_path: projectA,
        view: "current_prompt",
        session_id: firstInput.session_id,
        turn_id: firstInput.turn_id,
      });
      assert.equal(receipt.isError, false);
      assert.equal(receipt.structuredContent.found, true);
      assert.equal(receipt.structuredContent.view, "current_prompt");
      assert.equal(receipt.structuredContent.binding, "session_turn");
      assert.equal(receipt.structuredContent.answer_completed, false);
      assert.equal(receipt.structuredContent.query, safeFirstPrompt);
      assert(receipt.structuredContent.counts.total >= 2);
      assert(receipt.structuredContent.evidence.project.content.includes("PROJECT_A_MCP_MARKER"));
      assert.doesNotMatch(
        JSON.stringify(receipt.structuredContent),
        /projectId|scope_name|db_path|tenant_id|\/root\//u,
      );
    } finally {
      await receiptClient.close();
    }

    const beforeStop = server.records.length;
    const stopResult = parseJson(await runNode(join(hooksDir, "stop.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify({
        ...firstInput,
        hook_event_name: "Stop",
        stop_hook_active: false,
        last_assistant_message: "ASSISTANT_ONLY_PROGRESS_MARKER: Completed the Codex integration and passed actor/session tests. api_key=HOOK_ASSISTANT_SECRET_5678",
      }),
    }));
    assert.equal(stopResult.continue, true);
    await waitFor(() => server.records.length === beforeStop + 1);
    assert.equal(server.records.length, beforeStop + 1);
    const firstHookRecord = server.records.at(-1);
    assert.equal(firstHookRecord.metadata.integration, "codex");
    assert.equal(firstHookRecord.scope, scopesA.projectScope);
    assert.notEqual(firstHookRecord.sessionId, firstInput.session_id);
    assert(firstHookRecord.sessionId.startsWith("codex-"));
    assert.deepEqual(firstHookRecord.messages.map((item) => item.role), ["user", "assistant"]);
    assert(firstHookRecord.messages[0].content.includes("USER_REQUIREMENT_MARKER"));
    assert(firstHookRecord.messages[0].content.includes("password: [REDACTED]"));
    assert(!firstHookRecord.messages[0].content.includes("HOOK_USER_SECRET_1234"));
    assert(!firstHookRecord.messages[0].content.includes("ASSISTANT_ONLY_PROGRESS_MARKER"));
    assert(!firstHookRecord.messages[1].content.includes("USER_REQUIREMENT_MARKER"));
    assert(firstHookRecord.messages[1].content.includes("ASSISTANT_ONLY_PROGRESS_MARKER"));
    assert(firstHookRecord.messages[1].content.includes("api_key=[REDACTED]"));
    assert(!firstHookRecord.messages[1].content.includes("HOOK_ASSISTANT_SECRET_5678"));
    const submittedJob = await getJob(firstHookRecord.jobId, config);
    assert.equal(submittedJob.status, "succeeded");
    const waitedJob = await waitJob(firstHookRecord.jobId, { timeoutMs: 2000, pollMs: 50, config });
    assert.equal(waitedJob.status, "succeeded");

    const completedReceiptClient = new McpClient(env, projectA);
    try {
      await completedReceiptClient.initialize();
      const completedReceipt = await completedReceiptClient.call("tmcra_last_recall", {
        project_path: projectA,
        session_id: firstInput.session_id,
        turn_id: firstInput.turn_id,
      });
      assert.equal(completedReceipt.isError, false);
      assert.equal(completedReceipt.structuredContent.view, "latest_answer");
      assert.equal(completedReceipt.structuredContent.binding, "session_turn");
      assert.equal(completedReceipt.structuredContent.answer_completed, true);
      assert.equal(completedReceipt.structuredContent.query, safeFirstPrompt);
      assert(completedReceipt.structuredContent.completed_at);
    } finally {
      await completedReceiptClient.close();
    }

    await runNode(join(hooksDir, "stop.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify({
        ...firstInput,
        hook_event_name: "Stop",
        last_assistant_message: "duplicate stop must be ignored",
      }),
    });
    assert.equal(server.records.length, beforeStop + 1, "duplicate Stop created a second ingest");

    const secondInput = {
      hook_event_name: "UserPromptSubmit",
      session_id: "codex-session-two",
      turn_id: "turn-two",
      cwd: projectA,
      model: "e2e",
      prompt: "What implementation progress did Codex complete previously, and which user requirement guided it?",
    };
    const crossSession = parseJson(await runNode(join(hooksDir, "user_prompt_submit.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify(secondInput),
    }));
    const overwrittenReceiptClient = new McpClient(env, projectA);
    try {
      await overwrittenReceiptClient.initialize();
      const latestAnswer = await overwrittenReceiptClient.call("tmcra_last_recall", {
        project_path: projectA,
      });
      const currentPrompt = await overwrittenReceiptClient.call("tmcra_last_recall", {
        project_path: projectA,
        view: "current_prompt",
        session_id: secondInput.session_id,
        turn_id: secondInput.turn_id,
      });
      const ambiguousCurrentPrompt = await overwrittenReceiptClient.call("tmcra_last_recall", {
        project_path: projectA,
        view: "current_prompt",
      });
      const staleSessionCurrentPrompt = await overwrittenReceiptClient.call("tmcra_last_recall", {
        project_path: projectA,
        view: "current_prompt",
        session_id: firstInput.session_id,
        turn_id: firstInput.turn_id,
      });
      assert.equal(latestAnswer.structuredContent.view, "latest_answer");
      assert.equal(latestAnswer.structuredContent.query, safeFirstPrompt);
      assert.equal(currentPrompt.structuredContent.view, "current_prompt");
      assert.equal(currentPrompt.structuredContent.binding, "session_turn");
      assert.equal(currentPrompt.structuredContent.query, secondInput.prompt);
      assert.equal(ambiguousCurrentPrompt.structuredContent.found, false);
      assert.equal(ambiguousCurrentPrompt.structuredContent.reason, "session_and_turn_context_required");
      assert.equal(staleSessionCurrentPrompt.structuredContent.found, false);
      assert.equal(staleSessionCurrentPrompt.structuredContent.reason, "receipt_not_found");
    } finally {
      await overwrittenReceiptClient.close();
    }
    const crossSessionContext = crossSession.hookSpecificOutput.additionalContext;
    const projectBlockStart = crossSessionContext.indexOf("Memory layer: project (");
    assert(projectBlockStart >= 0, "project memory block is missing");
    const projectContext = crossSessionContext.slice(projectBlockStart);
    const userSection = projectContext.indexOf("User requirements and facts");
    const userMarker = projectContext.indexOf("USER_REQUIREMENT_MARKER");
    const assistantSection = projectContext.indexOf("Codex work progress and results");
    const assistantMarker = projectContext.indexOf("ASSISTANT_ONLY_PROGRESS_MARKER");
    assert(userSection >= 0 && userSection < userMarker);
    assert(userMarker < assistantSection && assistantSection < assistantMarker);
    assert(projectContext.includes("actor=user | authority=user_statement"));
    assert(projectContext.includes("actor=assistant | authority=assistant_source"));
    assert(crossSessionContext.includes("current user instruction has highest authority"));
    assert(crossSessionContext.includes("must never be promoted into user statements"));
    assert(!crossSessionContext.includes("actor=assistant | authority=user_statement"));
    assert(!crossSessionContext.includes("actor=user | authority=assistant_source"));
    await runNode(join(hooksDir, "stop.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify({
        ...secondInput,
        hook_event_name: "Stop",
        last_assistant_message: "Cross-session recall worked in HOOK_SESSION_TWO_MARKER.",
      }),
    });

    const isCurrentHookRecord = (record) => record.metadata.integration === "codex" &&
      record.messages.some((item) =>
        item.content.includes("USER_REQUIREMENT_MARKER") ||
        item.content.includes("HOOK_SESSION_TWO_MARKER")
      );
    await waitFor(() => server.records.filter(isCurrentHookRecord).length === 2);
    const hookRecords = server.records.filter(isCurrentHookRecord);
    assert.equal(hookRecords.length, 2);
    assert.notEqual(hookRecords[0].sessionId, hookRecords[1].sessionId);
    assert(hookRecords.every((record) => record.scope === scopesA.projectScope));
    assert.equal(
      server.recordsForScope(scopesA.globalScope).some((record) => record.metadata.integration === "codex"),
      false,
      "automatic Stop write leaked into global memory",
    );

    const projectBPrompt = parseJson(await runNode(join(hooksDir, "user_prompt_submit.mjs"), [], {
      cwd: projectB,
      env,
      input: JSON.stringify({
        hook_event_name: "UserPromptSubmit",
        session_id: "codex-project-b-session",
        turn_id: "project-b-turn",
        cwd: projectB,
        model: "e2e",
        prompt: "Recall project state.",
      }),
    }));
    const projectBContext = projectBPrompt.hookSpecificOutput.additionalContext;
    assert(projectBContext.includes("GLOBAL_PROFILE_MARKER"));
    assert(!projectBContext.includes("PROJECT_A_MCP_MARKER"));
    assert(!projectBContext.includes("HOOK_CROSS_SESSION_MARKER"));
  });

  await test("long-task checkpoints survive automatic context compaction", async () => {
    const input = {
      hook_event_name: "UserPromptSubmit",
      session_id: "codex-long-task-session",
      turn_id: "long-task-turn",
      cwd: projectA,
      model: "e2e",
      prompt: "LONG_TASK_OBJECTIVE_MARKER: complete a multi-stage implementation across context compaction.",
    };
    await runNode(join(hooksDir, "user_prompt_submit.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify(input),
    });

    const checkpointEnv = { ...env, TMCRA_CHECKPOINT_EVENT_THRESHOLD: "2" };
    const toolEvents = [
      {
        tool_name: "shell_command",
        tool_input: { command: "build --password=LONG_TASK_SECRET_MUST_NOT_LEAK" },
        tool_response: { exit_code: 0, output: "LONG_TASK_BUILD_MARKER completed" },
      },
      {
        tool_name: "apply_patch",
        tool_input: { path: "src/continuity.mjs" },
        tool_response: { ok: true, output: "LONG_TASK_PATCH_MARKER applied" },
      },
    ];
    for (const event of toolEvents) {
      const recorded = parseJson(await runNode(join(hooksDir, "post_tool_use.mjs"), [], {
        cwd: projectA,
        env: checkpointEnv,
        input: JSON.stringify({
          ...input,
          ...event,
          hook_event_name: "PostToolUse",
        }),
      }));
      assert.deepEqual(recorded, { continue: true });
    }

    await waitFor(
      () => server.records.some(
        (record) => record.metadata.integration === "codex-long-task-checkpoint" &&
          record.messages.some((messageValue) => messageValue.content.includes("LONG_TASK_PATCH_MARKER")),
      ),
    );
    const periodicCheckpoint = server.records.find(
      (record) => record.metadata.integration === "codex-long-task-checkpoint" &&
        record.messages.some((messageValue) => messageValue.content.includes("LONG_TASK_PATCH_MARKER")),
    );
    const periodicText = periodicCheckpoint.messages.map((item) => item.content).join("\n");
    assert(periodicText.includes("LONG_TASK_OBJECTIVE_MARKER"));
    assert(periodicText.includes("LONG_TASK_BUILD_MARKER"));
    assert(!periodicText.includes("LONG_TASK_SECRET_MUST_NOT_LEAK"));
    assert(periodicText.includes("[REDACTED]"));

    await runNode(join(hooksDir, "post_tool_use.mjs"), [], {
      cwd: projectA,
      env: checkpointEnv,
      input: JSON.stringify({
        ...input,
        hook_event_name: "PostToolUse",
        tool_name: "shell_command",
        tool_input: { command: "node tests/continuity.mjs" },
        tool_response: { exit_code: 0, output: "LONG_TASK_FINAL_TOOL_MARKER passed" },
      }),
    });
    const preCompact = parseJson(await runNode(join(hooksDir, "pre_compact.mjs"), [], {
      cwd: projectA,
      env: checkpointEnv,
      input: JSON.stringify({
        ...input,
        hook_event_name: "PreCompact",
        trigger: "auto",
      }),
    }));
    assert.deepEqual(preCompact, { continue: true });
    const postCompact = parseJson(await runNode(join(hooksDir, "post_compact.mjs"), [], {
      cwd: projectA,
      env: checkpointEnv,
      input: JSON.stringify({
        ...input,
        hook_event_name: "PostCompact",
        trigger: "auto",
      }),
    }));
    assert.deepEqual(postCompact, { continue: true });

    await waitFor(
      () => server.records.some(
        (record) => record.metadata.integration === "codex-long-task-checkpoint" &&
          record.messages.some((messageValue) => messageValue.content.includes("LONG_TASK_FINAL_TOOL_MARKER")),
      ),
    );
    const resumed = parseJson(await runNode(join(hooksDir, "session_start.mjs"), [], {
      cwd: projectA,
      env: checkpointEnv,
      input: JSON.stringify({
        ...input,
        hook_event_name: "SessionStart",
        source: "compact",
      }),
    }));
    assert.equal(resumed.continue, true);
    const resumedContext = resumed.hookSpecificOutput.additionalContext;
    assert(resumedContext.includes("tmcra_task_handoff"));
    assert(resumedContext.includes("LONG_TASK_OBJECTIVE_MARKER"));
    assert(resumedContext.includes("LONG_TASK_BUILD_MARKER"));
    assert(resumedContext.includes("LONG_TASK_PATCH_MARKER"));
    assert(resumedContext.includes("LONG_TASK_FINAL_TOOL_MARKER"));
    assert(!resumedContext.includes("LONG_TASK_SECRET_MUST_NOT_LEAK"));
    assert(resumedContext.length <= 11_000);

    const checkpointFiles = (await readdir(join(dataDir, "task-checkpoints")))
      .filter((name) => name.endsWith(".json"));
    assert.equal(checkpointFiles.length, 1);
    const checkpoint = JSON.parse(await readFile(
      join(dataDir, "task-checkpoints", checkpointFiles[0]),
      "utf8",
    ));
    assert.equal(checkpoint.reason, "pre_compact_auto");
    assert.equal(checkpoint.sequence, 2);
    const beforeFinal = server.records.length;
    await runNode(join(hooksDir, "stop.mjs"), [], {
      cwd: projectA,
      env: checkpointEnv,
      input: JSON.stringify({
        ...input,
        hook_event_name: "Stop",
        last_assistant_message: "LONG_TASK_FINAL_ANSWER_MARKER: implementation completed after compaction.",
      }),
    });
    await waitFor(() => server.records.length === beforeFinal + 1);
    const taskStateName = `${createHash("sha256").update(input.session_id).digest("hex")}.json`;
    assert(!(await readdir(join(dataDir, "task-state"))).includes(taskStateName));
    assert(!(await readdir(join(dataDir, "task-checkpoints"))).includes(taskStateName));
  });

  await test("Stop durably queues a complete turn before slow remote ingestion", async () => {
    const input = {
      hook_event_name: "UserPromptSubmit",
      session_id: "codex-durable-outbox",
      turn_id: "durable-turn",
      cwd: projectA,
      model: "e2e",
      prompt: "DURABLE_USER_TURN_MARKER must survive a slow ingestion acknowledgement.",
    };
    await runNode(join(hooksDir, "user_prompt_submit.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify(input),
    });
    server.ingestDelayMs = 3_000;
    const before = server.records.length;
    const started = Date.now();
    const stopped = parseJson(await runNode(join(hooksDir, "stop.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify({
        ...input,
        hook_event_name: "Stop",
        last_assistant_message: "DURABLE_ASSISTANT_TURN_MARKER was queued locally first.",
      }),
    }));
    assert.equal(stopped.continue, true);
    const stopElapsedMs = Date.now() - started;
    assert(
      stopElapsedMs < server.ingestDelayMs / 2,
      `Stop took ${stopElapsedMs}ms and appears to have waited for remote ingestion`,
    );
    await waitFor(async () => (await readdir(join(dataDir, "outbox"))).some((name) => name.endsWith(".json")));
    await waitFor(() => server.records.length === before + 1, { timeoutMs: 6_000 });
    await waitFor(async () => !(await readdir(join(dataDir, "outbox"))).some((name) => name.endsWith(".json")));
    const record = server.records.at(-1);
    assert(record.messages[0].content.includes("DURABLE_USER_TURN_MARKER"));
    assert(record.messages[1].content.includes("DURABLE_ASSISTANT_TURN_MARKER"));
    server.ingestDelayMs = 0;
  });

  await test("Codex history preview, explicit confirmation, filtering, and idempotency", async () => {
    const historyPath = join(codexHome, "sessions", "2026", "07", "16", "history-e2e.jsonl");
    const records = [
      {
        type: "session_meta",
        timestamp: "2026-07-01T10:00:00.000Z",
        payload: { id: "historical-session-one", cwd: projectA },
      },
      {
        type: "response_item",
        timestamp: "2026-07-01T10:00:01.000Z",
        payload: { type: "message", role: "developer", content: [{ type: "input_text", text: "HISTORY_DEVELOPER_MUST_NOT_UPLOAD" }] },
      },
      {
        type: "response_item",
        timestamp: "2026-07-01T10:00:01.100Z",
        payload: { type: "message", role: "user", content: [{ type: "input_text", text: "<recommended_plugins>HISTORY_PLUGIN_CONTEXT_MUST_NOT_UPLOAD</recommended_plugins>\n<environment_context>HISTORY_ENV_CONTEXT_MUST_NOT_UPLOAD</environment_context>" }] },
      },
      {
        type: "response_item",
        timestamp: "2026-07-01T10:00:01.200Z",
        payload: { type: "message", role: "user", content: [{ type: "input_text", text: "<subagent_notification>HISTORY_SUBAGENT_MUST_NOT_UPLOAD</subagent_notification>" }] },
      },
      {
        type: "response_item",
        timestamp: "2026-07-01T10:00:01.300Z",
        payload: { type: "message", role: "user", content: [{ type: "input_text", text: "<turn_aborted>HISTORY_ABORT_MUST_NOT_UPLOAD</turn_aborted>" }] },
      },
      {
        type: "response_item",
        timestamp: "2026-07-01T10:00:01.400Z",
        payload: { type: "message", role: "user", content: [{ type: "input_text", text: "<environment_context>HISTORY_STANDALONE_ENV_MUST_NOT_UPLOAD</environment_context>" }] },
      },
      {
        type: "response_item",
        timestamp: "2026-07-01T10:00:02.000Z",
        payload: { type: "message", role: "user", content: [{ type: "input_text", text: "<in-app-browser-context source=\"ambient-ui-state\">https://ambient.invalid</in-app-browser-context>\n\nHISTORY_USER_MARKER" }] },
      },
      {
        type: "response_item",
        timestamp: "2026-07-01T10:00:02.500Z",
        payload: { type: "message", role: "user", content: [{ type: "input_text", text: "password: HISTORY_SECRET_MUST_NOT_UPLOAD" }] },
      },
      {
        type: "response_item",
        timestamp: "2026-07-01T10:00:02.750Z",
        payload: { type: "message", role: "user", content: [{ type: "input_text", text: "密码HISTORY_CHINESE_SECRET_MUST_NOT_UPLOAD" }] },
      },
      {
        type: "response_item",
        timestamp: "2026-07-01T10:00:03.000Z",
        payload: { type: "reasoning", summary: "HISTORY_REASONING_MUST_NOT_UPLOAD" },
      },
      {
        type: "response_item",
        timestamp: "2026-07-01T10:00:03.250Z",
        payload: { type: "message", role: "user", content: [{ type: "input_text", text: "<heartbeat><automation_id>history-noise</automation_id></heartbeat>" }] },
      },
      {
        type: "response_item",
        timestamp: "2026-07-01T10:00:03.500Z",
        payload: { type: "message", role: "assistant", phase: "commentary", content: [{ type: "output_text", text: "HISTORY_COMMENTARY_MUST_NOT_UPLOAD" }] },
      },
      {
        type: "response_item",
        timestamp: "2026-07-01T10:00:04.000Z",
        payload: { type: "message", role: "assistant", phase: "final_answer", content: [{ type: "output_text", text: "HISTORY_ASSISTANT_MARKER" }] },
      },
      {
        type: "response_item",
        timestamp: "2026-07-01T10:00:05.000Z",
        payload: { type: "function_call_output", output: "HISTORY_TOOL_MUST_NOT_UPLOAD" },
      },
      {
        type: "response_item",
        timestamp: "2026-07-01T10:00:06.000Z",
        payload: { type: "message", role: "user", content: [{ type: "input_text", text: "HISTORY_INCOMPLETE_TAIL_MUST_NOT_UPLOAD" }] },
      },
    ];
    await writeFile(historyPath, `${records.map((record) => JSON.stringify(record)).join("\n")}\n`);

    const requestsBeforePreview = server.requests.length;
    const preview = parseJson(await runNode(join(scriptsDir, "history_import.mjs"), ["preview"], {
      env,
    }));
    assert.equal(preview.ok, true);
    assert.equal(preview.uploadPerformed, false);
    assert.equal(preview.projectCount, 1);
    assert.equal(preview.projects[0].sessions, 1);
    assert.equal(server.requests.length, requestsBeforePreview, "history preview contacted the API");

    const projectPreview = parseJson(await runNode(
      join(scriptsDir, "history_import.mjs"),
      ["preview", "--project", projectA, "--session", "historical-session-one"],
      { env },
    ));
    assert.equal(projectPreview.sessions, 1);
    assert.equal(projectPreview.messages, 4);
    assert.equal(projectPreview.redactedSensitive, 2);
    assert.equal(projectPreview.excludedSensitive, 0);
    assert.equal(projectPreview.excludedNonConversation, 7);
    assert.equal(projectPreview.incompleteTailMessages, 1);
    assert.equal(projectPreview.cacheHit, false);
    const cachedPreview = parseJson(await runNode(
      join(scriptsDir, "history_import.mjs"),
      ["preview", "--project", projectA, "--session", "historical-session-one"],
      { env },
    ));
    assert.equal(cachedPreview.cacheHit, true);
    assert.equal(cachedPreview.messages, projectPreview.messages);
    assert.equal(cachedPreview.redactedSensitive, projectPreview.redactedSensitive);

    await expectChildFailure(
      () => runNode(join(scriptsDir, "history_import.mjs"), ["import", "--project", projectA], { env }),
      /--confirm/u,
    );

    const importArgs = ["import", "--project", projectA, "--session", "historical-session-one", "--confirm", "--wait"];
    const firstImport = parseJson(await runNode(join(scriptsDir, "history_import.mjs"), importArgs, {
      env,
      timeoutMs: 20_000,
    }));
    assert.equal(firstImport.ok, true);
    assert.equal(firstImport.sessions, 1);
    assert.equal(firstImport.messages, 4);
    assert.equal(firstImport.redactedSensitive, 2);
    assert.equal(firstImport.excludedSensitive, 0);
    assert.equal(firstImport.excludedNonConversation, 7);
    assert.equal(firstImport.incompleteTailMessages, 1);
    assert.equal(firstImport.waited, true);
    const historyRecords = server.records.filter(
      (record) => record.metadata.integration === "codex-history-import",
    );
    assert.equal(historyRecords.length, 1);
    assert.deepEqual(historyRecords[0].messages.map((item) => item.role), ["user", "user", "user", "assistant"]);
    assert.equal(
      historyRecords[0].messages[0].message_id,
      `codex-history-${createHash("sha256")
        .update("historical-session-one:4:user:HISTORY_USER_MARKER")
        .digest("hex")
        .slice(0, 40)}`,
      "filtering internal envelopes changed the legacy position of later messages",
    );
    const importedText = historyRecords[0].messages.map((item) => item.content).join("\n");
    assert(importedText.includes("HISTORY_USER_MARKER"));
    assert(importedText.includes("HISTORY_ASSISTANT_MARKER"));
    assert(!importedText.includes("HISTORY_DEVELOPER_MUST_NOT_UPLOAD"));
    assert(!importedText.includes("HISTORY_REASONING_MUST_NOT_UPLOAD"));
    assert(!importedText.includes("HISTORY_TOOL_MUST_NOT_UPLOAD"));
    assert(!importedText.includes("HISTORY_COMMENTARY_MUST_NOT_UPLOAD"));
    assert(!importedText.includes("HISTORY_INCOMPLETE_TAIL_MUST_NOT_UPLOAD"));
    assert(!importedText.includes("HISTORY_PLUGIN_CONTEXT_MUST_NOT_UPLOAD"));
    assert(!importedText.includes("HISTORY_ENV_CONTEXT_MUST_NOT_UPLOAD"));
    assert(!importedText.includes("HISTORY_SUBAGENT_MUST_NOT_UPLOAD"));
    assert(!importedText.includes("HISTORY_ABORT_MUST_NOT_UPLOAD"));
    assert(!importedText.includes("HISTORY_STANDALONE_ENV_MUST_NOT_UPLOAD"));
    assert(!importedText.includes("ambient.invalid"));
    assert(!importedText.includes("HISTORY_SECRET_MUST_NOT_UPLOAD"));
    assert(!importedText.includes("HISTORY_CHINESE_SECRET_MUST_NOT_UPLOAD"));
    assert(importedText.includes("password: [REDACTED]"));
    assert(importedText.includes("密码[REDACTED]"));
    assert(!JSON.stringify(historyRecords[0].metadata).includes(projectA));
    assert(!("source_cwd" in historyRecords[0].metadata));
    assert(!("source_session_id" in historyRecords[0].metadata));

    const secondImport = parseJson(await runNode(join(scriptsDir, "history_import.mjs"), importArgs, {
      env,
      timeoutMs: 20_000,
    }));
    assert.deepEqual(secondImport.jobs, firstImport.jobs);
    assert.equal(
      server.records.filter((record) => record.metadata.integration === "codex-history-import").length,
      1,
      "repeated history import stored duplicate data",
    );
  });

  await test("repository bootstrap preview, explicit confirmation, filtering, and idempotency", async () => {
    const requestsBeforePreview = server.requests.length;
    const previewArgs = ["preview", "--project", bootstrapProject];
    const preview = parseJson(await runNode(join(scriptsDir, "project_bootstrap.mjs"), previewArgs, {
      env,
    }));
    assert.equal(preview.ok, true);
    assert.equal(preview.uploadPerformed, false);
    assert(preview.includedFiles.some((item) => item.path === "README.md"));
    assert(!preview.includedFiles.some((item) => item.path === ".env"));
    assert.equal(server.requests.length, requestsBeforePreview, "bootstrap preview contacted the API");

    await expectChildFailure(
      () => runNode(
        join(scriptsDir, "project_bootstrap.mjs"),
        ["import", "--project", bootstrapProject],
        { env },
      ),
      /--confirm/u,
    );

    const importArgs = ["import", "--project", bootstrapProject, "--confirm", "--wait"];
    const firstImport = parseJson(await runNode(join(scriptsDir, "project_bootstrap.mjs"), importArgs, {
      env,
      timeoutMs: 20_000,
    }));
    assert.equal(firstImport.ok, true);
    assert.equal(firstImport.jobStatus, "succeeded");
    const bootstrapRecords = server.records.filter(
      (record) => record.metadata.integration === "codex-project-bootstrap",
    );
    assert.equal(bootstrapRecords.length, 1);
    assert.equal(bootstrapRecords[0].messages[0].role, "system");
    assert(bootstrapRecords[0].messages[0].content.includes("BOOTSTRAP_VISIBLE_MARKER"));
    assert(!bootstrapRecords[0].messages[0].content.includes("MUST_NOT_UPLOAD_BOOTSTRAP"));
    assert(!bootstrapRecords[0].messages[0].content.includes(bootstrapProject));

    const secondImport = parseJson(await runNode(join(scriptsDir, "project_bootstrap.mjs"), importArgs, {
      env,
      timeoutMs: 20_000,
    }));
    assert.equal(secondImport.jobId, firstImport.jobId);
    assert.equal(
      server.records.filter((record) => record.metadata.integration === "codex-project-bootstrap").length,
      1,
      "repeated bootstrap import stored duplicate data",
    );
  });

  await test("revoked and invalid tokens fail open in hooks and fail closed in API tools", async () => {
    const recordCountBefore = server.records.length;
    server.revoke(validToken);

    const revokedInput = {
      hook_event_name: "UserPromptSubmit",
      session_id: "revoked-session",
      turn_id: "revoked-turn",
      cwd: projectA,
      model: "e2e",
      prompt: "This prompt must not be blocked by a revoked memory token.",
    };
    const recalled = parseJson(await runNode(join(hooksDir, "user_prompt_submit.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify(revokedInput),
    }));
    assert.deepEqual(recalled, { continue: true });

    const stopped = parseJson(await runNode(join(hooksDir, "stop.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify({
        ...revokedInput,
        hook_event_name: "Stop",
        last_assistant_message: "Codex continued normally.",
      }),
    }));
    assert.deepEqual(stopped, { continue: true });
    assert.equal(server.records.length, recordCountBefore, "revoked token still wrote memory");

    const sessionStarted = parseJson(await runNode(join(hooksDir, "session_start.mjs"), [], {
      cwd: projectA,
      env,
      input: JSON.stringify({
        hook_event_name: "SessionStart",
        session_id: "revoked-session-start",
        cwd: projectA,
        model: "e2e",
      }),
    }));
    assert.deepEqual(sessionStarted, { continue: true });

    const configError = await expectChildFailure(
      () => runNode(join(scriptsDir, "check_config.mjs"), ["--api-only"], { cwd: projectA, env }),
      /invalid or revoked/u,
    );
    assert(!`${configError.stdout}\n${configError.stderr}`.includes(validToken));

    const invalidEnv = { ...env, TMCRA_API_KEY: invalidToken };
    const client = new McpClient(invalidEnv, projectA);
    try {
      await client.initialize();
      const result = await client.call("tmcra_recall", {
        query: "must fail",
        memory_layer: "auto",
        project_path: projectA,
      });
      assert.equal(result.isError, true);
      assert.match(result.content[0].text, /authorization is no longer valid/u);
      assert.doesNotMatch(result.content[0].text, /invalid or revoked|API token/u);
      assert(!result.content[0].text.includes(invalidToken));
    } finally {
      await client.close();
    }
  });

  for (const output of capturedOutput) {
    assert(!output.includes(validToken), "a child process exposed the configured API token");
    assert(!output.includes(invalidToken), "a child process exposed an invalid API token");
  }

  process.stdout.write(`${JSON.stringify({
    ok: true,
    mode: "mock",
    tests: results,
    assertions: {
      configuration: true,
      pluginContract: true,
      lifecycleContractIsolation: true,
      projectIdentity: true,
      globalProjectScopes: true,
      sessionIsolation: true,
      subagentLifecycleIsolation: true,
      sessionStartRecallZero: true,
      recallInjection: true,
      actorProvenance: true,
      stopIngest: true,
      asyncJobs: true,
      crossSessionRecall: true,
      crossProjectIsolation: true,
      failOpen: true,
      mcpTools: true,
      historyImport: true,
      repositoryBootstrap: true,
      idempotency: true,
      revokedAndInvalidTokens: true,
      secretRedaction: true,
    },
    mock: {
      requestCount: server.requests.length,
      storedIngestCount: server.records.length,
      projectACount: server.recordsForScope(scopesA.projectScope).length,
      projectBCount: server.recordsForScope(scopesB.projectScope).length,
      globalCount: server.recordsForScope(scopesA.globalScope).length,
    },
  }, null, 2)}\n`);
} finally {
  await server.stop();
  const resolvedTemp = resolve(tempRoot);
  if (dirname(resolvedTemp) === resolve(tmpdir()) && basename(resolvedTemp).startsWith(tempPrefix)) {
    await rm(resolvedTemp, { recursive: true, force: true });
  }
}
