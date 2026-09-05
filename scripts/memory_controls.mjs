import { createHash, randomUUID } from "node:crypto";
import { mkdir, open, readFile, rename, unlink } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

// Shared by the Codex and DSH distributions. Never store credentials in this file.
export const MODES = Object.freeze(["normal", "recall_only", "off"]);
const hash = (value) => createHash("sha256").update(String(value)).digest("hex");
const clean = (text, limit = 4000) => String(text || "").trim().slice(0, limit);
export function controlKey(config, scope) {
  if (!config?.apiKey || !scope) throw new Error("Authenticated scope is required for memory controls");
  return hash(`${String(config.baseUrl).replace(/\/+$/u, "")}\0${config.apiKey}\0${scope}`);
}
export function controlsRoot() {
  return process.env.TMCRA_MEMORY_STATE_DIR || (process.env.PLUGIN_DATA
    ? join(process.env.PLUGIN_DATA, "memory-controls")
    : join(homedir(), ".config", "tmcra", "memory-controls"));
}
function blank() { return { schemaVersion: 1, sessions: {}, tasks: {}, recent: [], budgetChars: 12000 }; }
async function read(key) {
  if (!/^[a-f0-9]{64}$/u.test(key)) throw new Error("Invalid memory control key");
  try {
    const state = JSON.parse(await readFile(join(controlsRoot(), `${key}.json`), "utf8"));
    if (state.schemaVersion !== 1) throw new Error("Unsupported memory controls version");
    return state;
  } catch (error) { if (error.code === "ENOENT") return blank(); throw error; }
}
async function edit(key, fn) {
  if (!/^[a-f0-9]{64}$/u.test(key)) throw new Error("Invalid memory control key");
  await mkdir(controlsRoot(), { recursive: true, mode: 0o700 });
  const lockPath = join(controlsRoot(), `${key}.lock`);
  let lock;
  for (let i = 0; i < 60; i++) {
    try { lock = await open(lockPath, "wx", 0o600); break; }
    catch (error) { if (error.code !== "EEXIST") throw error; await new Promise((r) => setTimeout(r, 25)); }
  }
  // Fail closed for writes if another process/crash owns the lock. No unsafe stale-lock takeover.
  if (!lock) throw new Error("Memory controls busy; retry or inspect the local lock");
  let temporary;
  try {
    const state = await read(key);
    const result = await fn(state);
    temporary = join(controlsRoot(), `${key}.${randomUUID()}.tmp`);
    const file = await open(temporary, "wx", 0o600);
    try { await file.writeFile(JSON.stringify(state)); await file.sync(); } finally { await file.close(); }
    await rename(temporary, join(controlsRoot(), `${key}.json`));
    return result;
  } finally {
    if (temporary) await unlink(temporary).catch(() => {});
    await lock.close(); await unlink(lockPath);
  }
}
function session(state, id) {
  if (!clean(id, 500)) throw new Error("An exact session_id is required");
  return state.sessions[hash(id)] ||= { mode: "normal", generation: 0, taskId: null };
}
export async function memoryPolicy(key, sessionId) {
  const state = await read(key);
  const row = session(state, sessionId);
  const parentId = sessionId.includes(":subagent:") ? sessionId.split(":subagent:")[0] : null;
  const parent = parentId ? session(state, parentId) : null;
  return { key, sessionId, mode: row.mode, generation: row.generation,
    turnHash: row.currentTurnHash || null, parentTurnHash: parent?.currentTurnHash || null,
    parentGeneration: parent?.generation ?? null,
    read: row.mode !== "off" && parent?.mode !== "off",
    write: row.mode === "normal" && (!parent || parent.mode === "normal") };
}
export async function mayWrite(capture) {
  if (!capture?.write) return false;
  const current = await memoryPolicy(capture.key, capture.sessionId);
  const state = await read(capture.key);
  const allowed = (id, turnHash) => {
    const row = session(state, id);
    return turnHash ? !row.suppressedTurns?.[turnHash] : !row.suppressLegacyCapture;
  };
  return current.write && current.generation === capture.generation && current.parentGeneration === (capture.parentGeneration ?? null)
    && allowed(capture.sessionId, capture.turnHash)
    && (!capture.sessionId.includes(":subagent:") || allowed(capture.sessionId.split(":subagent:")[0], capture.parentTurnHash));
}
// Host lifecycle IDs identify the vetoed capture; raw prompts are never stored here.
export async function beginMemoryTurn(key, sessionId, turnId) {
  if (typeof turnId !== "string" || !turnId.trim()) throw new Error("An exact host turn ID is required");
  if ((await memoryPolicy(key, sessionId)).write) await edit(key, (state) => {
    session(state, sessionId).currentTurnHash = hash(turnId);
  });
  return memoryPolicy(key, sessionId);
}
export async function suppressMemoryTurn(key, sessionId) {
  return edit(key, (state) => {
    const row = session(state, sessionId);
    if (row.currentTurnHash) (row.suppressedTurns ||= {})[row.currentTurnHash] = true;
    row.suppressLegacyCapture = true;
    // Permanent hashes prevent an old offline queue from backfilling the discussion.
    return { automaticCapture: "suppressed", turnIdentified: Boolean(row.currentTurnHash), originalMemoryChanged: false };
  });
}
export async function legacyWriteAllowed(key, { sessionId, sessionHash } = {}) {
  const state = await read(key);
  if (sessionId) {
    const policy = await memoryPolicy(key, sessionId);
    return policy.generation === 0 && (policy.parentGeneration ?? 0) === 0 && await mayWrite({ ...policy, turnHash: null, parentTurnHash: null });
  }
  if (sessionHash) return Object.entries(state.sessions).every(([id, row]) => !id.startsWith(sessionHash) || (row.generation === 0 && !row.suppressLegacyCapture));
  return true;
}
export async function setMemoryMode(key, sessionId, mode) {
  if (!MODES.includes(mode)) throw new Error("mode must be normal, recall_only or off");
  return edit(key, (state) => {
    const row = session(state, sessionId);
    if (row.mode !== mode) row.generation++;
    row.mode = mode;
    row.changedAt = new Date().toISOString();
    // Do not preserve a hidden turn or later bind it as the task to continue.
    if (mode !== "normal") row.taskId = null;
    return { mode, generation: row.generation, alreadySubmittedWrites: "cannot_be_recalled",
      pendingOlderGeneration: "discard_on_delivery", disabledContentBackfill: false };
  });
}
export function isContinuation(prompt) {
  return /^(?:好[的]?[,，\s]*)?(?:继续|接着[做来]?|往下[做走]?|补齐这些|完成这些|continue|resume|go on|carry on)[。.!！\s]*$/iu.test(clean(prompt));
}
export async function taskContext(key, sessionId, prompt, { capture = null } = {}) {
  const state = await read(key);
  const binding = session(state, sessionId);
  const active = Object.values(state.tasks).filter((row) => row.status === "active");
  let task = state.tasks[binding.taskId];
  if (task?.status !== "active") task = null;
  if (isContinuation(prompt) && !task && active.length === 1) task = active[0];
  if (!isContinuation(prompt)) return { query: prompt, task: null, candidates: [] };
  if (!task) return { query: prompt, task: null, candidates: active.map(({ id, objective }) => ({ id, objective })) };
  if (capture && await mayWrite(capture)) await edit(key, (current) => { session(current, sessionId).taskId = task.id; });
  return { query: `${prompt}\nCurrent task: ${task.objective}\nLast observed result: ${task.summary || ""}\nNext step: ${task.nextStep || ""}`.slice(0, 12000),
    task, candidates: [] };
}
export async function updateTask(key, sessionId, { id, objective, summary, nextStep, status = "active" } = {}) {
  const policy = await memoryPolicy(key, sessionId);
  if (!policy.write) throw new Error("Task capture is disabled in this session");
  if (!["active", "completed", "blocked"].includes(status)) throw new Error("Invalid task status");
  return edit(key, (state) => {
    const binding = session(state, sessionId);
    if (binding.generation !== policy.generation || binding.mode !== "normal") throw new Error("Memory mode changed");
    if (id && !state.tasks[id]) throw new Error("Unknown task_id in this account and project");
    const taskId = id || `task_${randomUUID()}`;
    const previous = state.tasks[taskId] || {};
    const task = { ...previous, id: taskId, objective: clean(objective ?? previous.objective),
      summary: clean(summary ?? previous.summary, 6000), nextStep: clean(nextStep ?? previous.nextStep), status,
      updatedAt: new Date().toISOString() };
    if (!task.objective) throw new Error("A task objective is required");
    state.tasks[taskId] = task;
    binding.taskId = status === "active" ? taskId : null;
    // Completed history is capped; active tasks are never silently evicted.
    const done = Object.values(state.tasks).filter((item) => item.status !== "active").sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
    for (const item of done.slice(100)) delete state.tasks[item.id];
    return task;
  });
}
export async function finishObservedTurn(capture, prompt, answer) {
  if (!await mayWrite(capture)) return null;
  const continuation = await taskContext(capture.key, capture.sessionId, prompt, { capture });
  if (isContinuation(prompt) && !continuation.task) return null; // Ambiguous tasks require selection.
  return updateTask(capture.key, capture.sessionId, { id: continuation.task?.id,
    objective: continuation.task?.objective || prompt, summary: answer });
}
export function budgetEvidence(layers, { budgetChars = 12000, visibleText = "" } = {}) {
  if (!Number.isInteger(budgetChars) || budgetChars < 1000 || budgetChars > 64000) throw new Error("budgetChars must be 1000..64000");
  const seen = new Set(); const included = []; const omitted = []; let used = 0;
  for (const layer of layers) {
    const text = clean(layer.content, 200000);
    if (!text) continue;
    // Split only at renderer-provided source boundaries; never slice a source midway.
    const blocks = text.split(/\n\n(?=\[(?:Immutable |Slow memory |Fast memory |TMCRA actor section))/u);
    for (const content of blocks) {
      const identity = hash(content);
      const reason = seen.has(identity) || (visibleText && visibleText.includes(content)) ? "duplicate"
        : used + content.length + 128 > budgetChars ? "budget" : null;
      seen.add(identity);
      if (reason) { omitted.push({ scope: layer.scope, hash: identity, reason, characters: content.length }); continue; }
      included.push({ scope: layer.scope, label: layer.label, content, hash: identity }); used += content.length + 128;
    }
  }
  return { content: included.map((row) => `${row.label || `Memory scope: ${row.scope}`}\n${row.content}`).join("\n\n"),
    included, omitted, characters: used, estimatedTokens: Math.ceil(used / 3), tokenEstimateOnly: true, budgetChars };
}
export async function recordMemoryActivity(capture, activity) {
  if (!await mayWrite(capture)) return;
  await edit(capture.key, (state) => {
    const row = session(state, capture.sessionId);
    if (row.mode !== "normal" || row.generation !== capture.generation) return;
    if (activity.kind === "write") {
      const existing = state.recent.find((item) => item.kind === "write"
        && ((activity.outboxId && item.outboxId === activity.outboxId) || (activity.jobId && item.jobId === activity.jobId)));
      if (existing) { Object.assign(existing, activity, { updatedAt: new Date().toISOString() }); return; }
    }
    state.recent.unshift({ ...activity, sessionKey: hash(capture.sessionId), at: new Date().toISOString() });
    state.recent = state.recent.slice(0, 20);
  });
}
export async function memoryDashboard(key, sessionId) {
  const state = await read(key);
  return { policy: await memoryPolicy(key, sessionId), currentTaskId: state.sessions[hash(sessionId)]?.taskId || null, tasks: Object.values(state.tasks),
    recent: state.recent.filter((row) => row.sessionKey === hash(sessionId)), budgetChars: state.budgetChars };
}
export async function setMemoryBudget(key, budgetChars) {
  budgetEvidence([], { budgetChars });
  return edit(key, (state) => { state.budgetChars = budgetChars; return { budgetChars }; });
}
