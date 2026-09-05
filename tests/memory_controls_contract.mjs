import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { controlKey, memoryPolicy, mayWrite, legacyWriteAllowed, setMemoryMode, updateTask, taskContext,
  budgetEvidence, recordMemoryActivity, memoryDashboard, finishObservedTurn } from "../scripts/memory_controls.mjs";
import { createMemoryActions, startMemoryCenter } from "../scripts/memory_center.mjs";
import { saveOutboxTurn, submitOutboxTurn, listOutboxTurns } from "../scripts/tmcra_client.mjs";

const root = await mkdtemp(join(tmpdir(), "tmcra-controls-contract-"));
process.env.TMCRA_MEMORY_STATE_DIR = join(root, "controls");
process.env.PLUGIN_DATA = join(root, "plugin");
const config = { apiKey: "isolated-test-credential", baseUrl: "http://localhost:1" };
const key = controlKey(config, "project-a");
let center;
try {
  assert.notEqual(key, controlKey({ ...config, apiKey: "different-account" }, "project-a"));
  assert.notEqual(key, controlKey(config, "project-b"));
  const first = await memoryPolicy(key, "session-a");
  assert.equal(await mayWrite(first), true);
  await setMemoryMode(key, "session-a", "recall_only");
  const readonly = await memoryPolicy(key, "session-a");
  assert.equal(readonly.read, true); assert.equal(readonly.write, false);
  await recordMemoryActivity(readonly, { query: "PRIVATE-READONLY-TURN" });
  assert.equal(await finishObservedTurn(readonly, "PRIVATE-OFF-TURN", "private result"), null);
  await setMemoryMode(key, "session-a", "off");
  assert.equal((await memoryPolicy(key, "session-a")).read, false);
  await setMemoryMode(key, "session-a", "normal");
  assert.equal(await mayWrite(first), false, "reenabling never authorizes an old generation");
  const child = await memoryPolicy(key, "parent:subagent:one");
  await setMemoryMode(key, "parent", "off");
  assert.equal((await memoryPolicy(key, "parent:subagent:one")).read, false);
  await setMemoryMode(key, "parent", "normal");
  assert.equal(await mayWrite(child), false, "parent mode generations also invalidate subagent capture");
  assert.equal(await legacyWriteAllowed(key, { sessionId: "parent:subagent:one" }), false, "parent generations also protect legacy subagent queues");
  const policy = await memoryPolicy(key, "session-a");
  const task = await updateTask(key, "session-a", { objective: "Finish the authentication flow", nextStep: "Test expired tokens" });
  assert.equal((await memoryDashboard(key, 'session-a')).currentTaskId, task.id);
  await finishObservedTurn(policy, "继续", "The login UI is done");
  assert.match((await taskContext(key, "fresh-session", "继续")).query, /authentication flow/u);
  await updateTask(key, "parallel-session", { objective: "Build the billing screen" });
  const ambiguous = await taskContext(key, "third-session", "continue");
  assert.equal(ambiguous.task, null); assert.equal(ambiguous.candidates.length, 2);
  assert.equal((await taskContext(key, "session-a", "继续")).task.id, task.id);
  await updateTask(key, "session-a", { id: task.id, status: "completed" });
  assert.equal((await taskContext(key, "third-session", "continue")).task.objective, "Build the billing screen");
  const source = "[Immutable source window 1 | actor=user | memory_id=m1]\nThis is a real source.";
  const selected = budgetEvidence([{ scope: "global", content: source }, { scope: "project", content: source }]);
  assert.equal(selected.included.length, 1); assert.equal(selected.omitted[0].reason, "duplicate");
  assert.equal(budgetEvidence([{ scope: "p", content: source }], { visibleText: source }).included.length, 0);
  assert.equal(budgetEvidence([{ scope: "p", content: source }], { visibleText: "compacted summary" }).included.length, 1);
  const over = budgetEvidence([{ scope: "p", content: source + "x".repeat(2000) }], { budgetChars: 1000 });
  assert.equal(over.content, ""); assert.equal(over.omitted[0].reason, "budget");
  const queued = await saveOutboxTurn({ scope: "project-a", sessionId: "stable-session", capture: policy,
    messages: [{ message_id: "one", role: "user", content: "capture while enabled" }], idempotencyKey: "test-generation-queue" });
  await setMemoryMode(key, "session-a", "off");
  await setMemoryMode(key, "session-a", "normal");
  assert.equal((await submitOutboxTurn(queued, config)).skipped, true);
  assert.equal((await listOutboxTurns()).length, 0);
  const legacy = await saveOutboxTurn({ scope: "project-a", sessionId: "stable-session", receiptBinding: { projectId: "aaaaaaaaaaaaaaaa", sessionId: "session-a", turnId: "old" },
    messages: [{ message_id: "old", role: "user", content: "legacy queued" }], idempotencyKey: "legacy-generation-queue" });
  assert.equal((await submitOutboxTurn(legacy, config)).skipped, true);
  const calls = [];
  const invoke = createMemoryActions({ config, scope: "project-a", sessionId: "session-a",
    request: async (path, options) => { calls.push({ path, options }); return { effective: true }; } });
  await assert.rejects(invoke("feedback", { scope: "project-b", action: "ignore", memory_ids: ["m1"] }), /outside/u);
  await invoke("feedback", { action: "correct", memory_ids: ["m1"], replacement: "Current correct fact", idempotency_key: "feedback-contract-one" });
  assert.equal(calls[0].options.body.replacement, "Current correct fact");
  center = await startMemoryCenter({ invoke, open: false, idleTimeoutMs: 10000 });
  const post = (headers, body) => fetch(`${center.baseUrl}/api/action`, { method: "POST", headers: { "Content-Type": "application/json", ...headers }, body: JSON.stringify(body) });
  assert.equal((await post({}, { action: "dashboard" })).status, 403);
  assert.equal((await post({ "X-TMCRA-Token": center.token, Origin: "https://attacker.invalid" }, { action: "mode", args: { mode: "off" } })).status, 403);
  const response = await post({ "X-TMCRA-Token": center.token, Origin: center.baseUrl }, { action: "dashboard" });
  const text = await response.text(); assert.equal(response.status, 200); assert.ok(!text.includes(config.apiKey));
  assert.equal((await memoryDashboard(key, "session-a")).policy.mode, "normal");
  for (const path of await readdir(process.env.TMCRA_MEMORY_STATE_DIR)) {
    assert.doesNotMatch(await readFile(join(process.env.TMCRA_MEMORY_STATE_DIR, path), "utf8"), /PRIVATE-READONLY-TURN|PRIVATE-OFF-TURN|isolated-test-credential/u);
  }
  console.log(JSON.stringify({ ok: true, controls: true, continuation: true, budget: true, noBackfill: true, outbox: true, loopback: true }));
} finally {
  if (center) await new Promise((resolve) => center.server.close(resolve));
  await rm(root, { recursive: true, force: true });
}
