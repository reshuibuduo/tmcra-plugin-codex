import { createHash } from "node:crypto";
import { createReadStream, existsSync } from "node:fs";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, relative, resolve, sep } from "node:path";
import { createInterface } from "node:readline";

import {
  consolidate,
  deterministicKey,
  ingest,
  loadConfig,
  pluginDataDir,
  redactSensitiveText,
  resolveMemoryScopes,
  waitJob,
  PLUGIN_VERSION,
} from "./tmcra_client.mjs";

const command = process.argv[2] || "preview";
const args = process.argv.slice(3);
const PREVIEW_SCHEMA_VERSION = 3;
const INTERNAL_USER_ENVELOPE_TAGS = Object.freeze([
  "environment_context",
  "recommended_plugins",
  "subagent_notification",
  "turn_aborted",
]);

function valueAfter(flag) {
  const index = args.indexOf(flag);
  return index >= 0 ? args[index + 1] : null;
}

function has(flag) {
  return args.includes(flag);
}

function normalizedPath(value) {
  const withoutExtendedPrefix = String(value || "").replace(/^\\\\\?\\/u, "");
  const path = resolve(withoutExtendedPrefix).replaceAll("\\", "/");
  return process.platform === "win32" ? path.toLowerCase() : path;
}

function isInside(path, root) {
  const target = normalizedPath(path);
  const parent = normalizedPath(root).replace(/\/+$/u, "");
  return target === parent || target.startsWith(`${parent}/`);
}

async function walkJsonl(root, output = []) {
  if (!existsSync(root)) return output;
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const path = join(root, entry.name);
    if (entry.isDirectory()) await walkJsonl(path, output);
    else if (entry.isFile() && entry.name.endsWith(".jsonl")) output.push(path);
  }
  return output;
}

async function sessionFiles() {
  const codexHome = resolve(process.env.CODEX_HOME || join(homedir(), ".codex"));
  return [
    ...(await walkJsonl(join(codexHome, "sessions"))),
    ...(await walkJsonl(join(codexHome, "archived_sessions"))),
  ];
}

async function discoverFromStateDatabase() {
  const codexHome = resolve(process.env.CODEX_HOME || join(homedir(), ".codex"));
  const databasePath = join(codexHome, "state_5.sqlite");
  if (!existsSync(databasePath)) return null;
  let database;
  try {
    const { DatabaseSync } = await import("node:sqlite");
    database = new DatabaseSync(databasePath, { readOnly: true });
    const rows = database.prepare(`
      SELECT id, rollout_path, cwd, created_at
      FROM threads
      WHERE rollout_path <> '' AND cwd <> ''
    `).all();
    const metas = [];
    for (const row of rows) {
      const path = String(row.rollout_path || "");
      if (!path || !existsSync(path)) continue;
      const info = await stat(path);
      metas.push({
        path,
        sessionId: String(row.id || ""),
        cwd: String(row.cwd || ""),
        timestamp: Number.isFinite(Number(row.created_at))
          ? new Date(Number(row.created_at) * 1000).toISOString()
          : "",
        bytes: info.size,
        mtimeMs: Math.trunc(info.mtimeMs),
      });
    }
    return metas;
  } catch {
    return null;
  } finally {
    database?.close();
  }
}

async function sessionMeta(path) {
  const input = createInterface({ input: createReadStream(path, { encoding: "utf8" }) });
  try {
    for await (const line of input) {
      if (!line.trim()) continue;
      const record = JSON.parse(line);
      if (record.type !== "session_meta") continue;
      const payload = record.payload || {};
      const info = await stat(path);
      return {
        path,
        sessionId: String(payload.session_id || payload.id || ""),
        cwd: String(payload.cwd || ""),
        timestamp: String(payload.timestamp || record.timestamp || ""),
        bytes: info.size,
        mtimeMs: Math.trunc(info.mtimeMs),
      };
    }
  } finally {
    input.close();
  }
  return null;
}

function contentText(content) {
  if (!Array.isArray(content)) return "";
  return content
    .filter((item) => item && ["input_text", "output_text", "text"].includes(item.type))
    .map((item) => String(item.text || ""))
    .join("\n")
    .trim();
}

function stripAmbientContext(text) {
  return String(text || "")
    .replace(
      /<in-app-browser-context\b[^>]*>[\s\S]*?<\/in-app-browser-context>\s*/giu,
      "",
    )
    .trim();
}

function stripInternalUserEnvelopes(text) {
  let output = String(text || "");
  for (const tag of INTERNAL_USER_ENVELOPE_TAGS) {
    const pattern = new RegExp(
      `<${tag}\\b[^>]*>[\\s\\S]*?<\\/${tag}>\\s*`,
      "giu",
    );
    output = output.replace(pattern, "");
  }
  return output.trim();
}

function isHeartbeat(text) {
  return /^\s*<heartbeat\b[^>]*>[\s\S]*<\/heartbeat>\s*$/iu.test(String(text || ""));
}

function isConversationMessage(payload) {
  if (payload.role === "user") return true;
  if (payload.role !== "assistant") return false;
  const phase = String(payload.phase || "").trim().toLowerCase();
  return !phase || phase === "final" || phase === "final_answer";
}

async function readMessages(meta) {
  const messages = [];
  const redactedFlags = [];
  const seen = new Set();
  let excludedNonConversation = 0;
  const input = createInterface({ input: createReadStream(meta.path, { encoding: "utf8" }) });
  let index = 0;
  try {
    for await (const line of input) {
      if (!line.trim()) continue;
      if (!line.includes('"type":"response_item"') || !line.includes('"type":"message"')) {
        continue;
      }
      let record;
      try {
        record = JSON.parse(line);
      } catch {
        continue;
      }
      if (record.type !== "response_item" || record.payload?.type !== "message") continue;
      if (!isConversationMessage(record.payload)) {
        excludedNonConversation += 1;
        continue;
      }
      const role = record.payload.role;
      const rawContent = stripAmbientContext(contentText(record.payload.content));
      if (isHeartbeat(rawContent)) {
        excludedNonConversation += 1;
        continue;
      }
      const legacyContent = redactSensitiveText(rawContent).trim();
      if (!legacyContent) continue;
      const timestamp = String(record.timestamp || meta.timestamp || new Date(0).toISOString());
      const duplicateKey = createHash("sha256")
        .update(`${timestamp}:${role}:${legacyContent}`)
        .digest("hex");
      if (seen.has(duplicateKey)) continue;
      seen.add(duplicateKey);
      const legacyMessageIndex = index;
      index += 1;
      const conversationRawContent = role === "user"
        ? stripInternalUserEnvelopes(rawContent)
        : rawContent;
      if (!conversationRawContent) {
        excludedNonConversation += 1;
        continue;
      }
      const content = redactSensitiveText(conversationRawContent).trim();
      if (!content) continue;
      const wasRedacted = content !== conversationRawContent;
      messages.push({
        message_id: `codex-history-${createHash("sha256")
          .update(`${meta.sessionId}:${legacyMessageIndex}:${role}:${content}`)
          .digest("hex")
          .slice(0, 40)}`,
        role,
        content: content.slice(0, 200_000),
        timestamp,
      });
      redactedFlags.push(wasRedacted);
    }
  } finally {
    input.close();
  }
  const lastCompletedIndex = messages.findLastIndex((message) => message.role === "assistant");
  const completedMessages = lastCompletedIndex >= 0 ? messages.slice(0, lastCompletedIndex + 1) : [];
  return {
    messages: completedMessages,
    redactedSensitive: redactedFlags.slice(0, completedMessages.length).filter(Boolean).length,
    excludedNonConversation,
    incompleteTailMessages: messages.length - completedMessages.length,
  };
}

async function discover() {
  const indexed = await discoverFromStateDatabase();
  if (indexed) return indexed;
  const metas = [];
  for (const path of await sessionFiles()) {
    try {
      const meta = await sessionMeta(path);
      if (meta?.sessionId && meta.cwd) metas.push(meta);
    } catch {
      // A damaged historical rollout must not prevent other projects from being discovered.
    }
  }
  return metas;
}

async function selectedProjectSessions(sourceRoot) {
  const requestedSession = String(valueAfter("--session") || "").trim();
  const selected = (await discover()).filter(
    (meta) => isInside(meta.cwd, sourceRoot) &&
      (!requestedSession || meta.sessionId === requestedSession),
  );
  if (requestedSession && selected.length === 0) {
    throw new Error(`Codex session ${requestedSession} was not found under ${sourceRoot}`);
  }
  return { selected, requestedSession };
}

function previewFingerprint(selected, targetRoot, sourceRoot) {
  const payload = selected
    .map((meta) => `${normalizedPath(meta.path)}:${meta.bytes}:${meta.mtimeMs || 0}`)
    .sort()
    .join("\n");
  return createHash("sha256")
    .update(`${PREVIEW_SCHEMA_VERSION}\n${normalizedPath(targetRoot)}\n${normalizedPath(sourceRoot)}\n${payload}`)
    .digest("hex");
}

function previewCachePath(targetRoot, sourceRoot, requestedSession) {
  const key = createHash("sha256")
    .update(`${normalizedPath(targetRoot)}\n${normalizedPath(sourceRoot)}\n${requestedSession}`)
    .digest("hex");
  return join(pluginDataDir(), "imports", "preview-cache", `${key}.json`);
}

async function readPreviewCache(path, fingerprint) {
  try {
    const cached = JSON.parse(await readFile(path, "utf8"));
    if (
      cached?.fingerprint === fingerprint &&
      Number.isSafeInteger(cached.messages) &&
      Number.isSafeInteger(cached.redactedSensitive) &&
      Number.isSafeInteger(cached.excludedNonConversation) &&
      Number.isSafeInteger(cached.incompleteTailMessages)
    ) {
      return cached;
    }
  } catch {
    // A missing or damaged local cache only makes the next preview slower.
  }
  return null;
}

async function writePreviewCache(path, report) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(report)}\n`, "utf8");
}

async function preview() {
  const projectPath = valueAfter("--project");
  if (projectPath) {
    const targetRoot = resolve(projectPath);
    const sourceRoot = resolve(valueAfter("--source") || targetRoot);
    const { selected, requestedSession } = await selectedProjectSessions(sourceRoot);
    const fingerprint = previewFingerprint(selected, targetRoot, sourceRoot);
    const cachePath = previewCachePath(targetRoot, sourceRoot, requestedSession);
    const cached = await readPreviewCache(cachePath, fingerprint);
    let messages = 0;
    let redactedSensitive = 0;
    let excludedNonConversation = 0;
    let incompleteTailMessages = 0;
    const bytes = selected.reduce((sum, meta) => sum + meta.bytes, 0);
    if (cached) {
      messages = cached.messages;
      redactedSensitive = cached.redactedSensitive;
      excludedNonConversation = cached.excludedNonConversation;
      incompleteTailMessages = cached.incompleteTailMessages;
    } else {
      for (const meta of selected) {
        const result = await readMessages(meta);
        messages += result.messages.length;
        redactedSensitive += result.redactedSensitive;
        excludedNonConversation += result.excludedNonConversation;
        incompleteTailMessages += result.incompleteTailMessages;
      }
      await writePreviewCache(cachePath, {
        fingerprint,
        messages,
        redactedSensitive,
        excludedNonConversation,
        incompleteTailMessages,
      });
    }
    process.stdout.write(
      `${JSON.stringify(
        {
          ok: true,
          uploadPerformed: false,
          targetRoot,
          sourceRoot,
          requestedSession: requestedSession || null,
          sessions: selected.length,
          messages,
          redactedSensitive,
          excludedSensitive: 0,
          excludedNonConversation,
          incompleteTailMessages,
          bytes,
          cacheHit: Boolean(cached),
        },
        null,
        2,
      )}\n`,
    );
    return;
  }
  const projects = new Map();
  for (const meta of await discover()) {
    const key = normalizedPath(meta.cwd);
    const existing = projects.get(key) || {
      cwd: meta.cwd,
      sessions: 0,
      bytes: 0,
      firstSeen: meta.timestamp,
      lastSeen: meta.timestamp,
    };
    existing.sessions += 1;
    existing.bytes += meta.bytes;
    if (meta.timestamp && (!existing.firstSeen || meta.timestamp < existing.firstSeen)) {
      existing.firstSeen = meta.timestamp;
    }
    if (meta.timestamp && (!existing.lastSeen || meta.timestamp > existing.lastSeen)) {
      existing.lastSeen = meta.timestamp;
    }
    projects.set(key, existing);
  }
  const rows = [...projects.values()].sort((a, b) => b.lastSeen.localeCompare(a.lastSeen));
  const output = { ok: true, uploadPerformed: false, projectCount: rows.length, projects: rows };
  process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
}

async function importProject() {
  if (!has("--confirm")) {
    throw new Error("Historical import uploads prior user/assistant messages. Re-run with --confirm after reviewing preview output.");
  }
  const projectPath = valueAfter("--project");
  if (!projectPath) throw new Error("--project <current-project-path> is required");
  if (has("--consolidate") && !has("--wait")) {
    throw new Error("--consolidate requires --wait so every ingest job is verified first");
  }
  const targetRoot = resolve(projectPath);
  const sourceRoot = resolve(valueAfter("--source") || targetRoot);
  const { selected, requestedSession } = await selectedProjectSessions(sourceRoot);
  if (selected.length === 0) {
    throw new Error(`No Codex history was found under ${sourceRoot}`);
  }
  const config = await loadConfig();
  const scopes = await resolveMemoryScopes({ cwd: targetRoot, config });
  const jobs = [];
  let messageCount = 0;
  let redactedSensitive = 0;
  let excludedNonConversation = 0;
  let incompleteTailMessages = 0;
  for (const meta of selected) {
    const parsed = await readMessages(meta);
    const messages = parsed.messages;
    redactedSensitive += parsed.redactedSensitive;
    excludedNonConversation += parsed.excludedNonConversation;
    incompleteTailMessages += parsed.incompleteTailMessages;
    for (let offset = 0; offset < messages.length; offset += 250) {
      const batch = messages.slice(offset, offset + 250);
      if (!batch.length) continue;
      const bodyIdentity = {
        source: meta.path,
        sessionId: meta.sessionId,
        offset,
        messageIds: batch.map((item) => item.message_id),
      };
      const ingestResult = await ingest({
        config,
        scope: scopes.projectScope,
        sessionId: `codex-history-${createHash("sha256")
          .update(meta.sessionId)
          .digest("hex")
          .slice(0, 40)}`,
        messages: batch,
        idempotencyKey: deterministicKey(bodyIdentity),
        metadata: {
          integration: "codex-history-import",
          integration_version: PLUGIN_VERSION,
          source_session_hash: createHash("sha256")
            .update(meta.sessionId)
            .digest("hex")
            .slice(0, 24),
          source_file_hash: createHash("sha256").update(meta.path).digest("hex").slice(0, 24),
          project_id: scopes.projectId,
          project_name: scopes.projectName,
        },
      });
      const jobId = ingestResult.job_id || ingestResult.id;
      jobs.push(jobId);
      messageCount += batch.length;
      if (has("--wait")) {
        const job = await waitJob(jobId, { timeoutMs: 3_600_000, pollMs: 2000, config });
        if (job.status !== "succeeded") {
          throw new Error(`history job ${jobId} ended with ${job.status}`);
        }
      }
    }
  }

  let consolidationJob = null;
  if (has("--consolidate")) {
    const consolidation = await consolidate({
      scope: scopes.projectScope,
      config,
      idempotencyKey: deterministicKey({
        action: "codex-history-consolidate",
        scope: scopes.projectScope,
        sessions: selected.map((meta) => meta.sessionId).sort(),
      }),
    });
    const jobId = consolidation.job_id || consolidation.id;
    const job = await waitJob(jobId, { timeoutMs: 3_600_000, pollMs: 3000, config });
    if (job.status !== "succeeded") {
      throw new Error(`history consolidation job ${jobId} ended with ${job.status}`);
    }
    consolidationJob = { id: jobId, status: job.status };
  }
  const receipt = {
    importedAt: new Date().toISOString(),
    targetRoot,
    sourceRoot,
    requestedSession: requestedSession || null,
    projectId: scopes.projectId,
    projectScope: scopes.projectScope,
    sessions: selected.length,
    messages: messageCount,
    redactedSensitive,
    excludedSensitive: 0,
    excludedNonConversation,
    incompleteTailMessages,
    jobs,
    waited: has("--wait"),
    consolidationJob,
  };
  const receiptPath = join(pluginDataDir(), "imports", `${Date.now()}-${scopes.projectId}.json`);
  await mkdir(dirname(receiptPath), { recursive: true });
  await writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, "utf8");
  process.stdout.write(`${JSON.stringify({ ok: true, ...receipt, receiptPath }, null, 2)}\n`);
}

if (command === "preview") await preview();
else if (command === "import") await importProject();
else throw new Error("Usage: history_import.mjs preview | import --project <path> [--source <old-path>] [--session <thread-id>] --confirm [--wait] [--consolidate]");
