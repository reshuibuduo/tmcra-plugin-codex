import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const root = await mkdtemp(join(tmpdir(), "tmcra-closed-loop-"));
const previousEnvironment = new Map();
const testEnvironment = {
  PLUGIN_DATA: root,
  TMCRA_BASE_URL: "http://localhost:0",
  TMCRA_API_KEY: "regression-test-token",
  TMCRA_GLOBAL_SCOPE: "global-test",
  TMCRA_PROJECT_SCOPE: "project-test",
  TMCRA_PROJECT_SCOPE_PREFIX: "project",
};
for (const [name, value] of Object.entries(testEnvironment)) {
  previousEnvironment.set(name, process.env[name]);
  process.env[name] = value;
}

const client = await import("../scripts/tmcra_client.mjs");
const hooks = await import("../hooks/hook_common.mjs");

function checkpointEntry({ outboxSlot, outboxGroupKey, sequence, queueKind = "checkpoint", key }) {
  return {
    scope: "project-test",
    projectId: "project-id",
    sessionId: "codex-session",
    messages: [{
      message_id: `checkpoint-${sequence}`,
      role: "assistant",
      content: `checkpoint-${sequence}`,
      timestamp: new Date().toISOString(),
    }],
    metadata: {
      integration: "codex-long-task-checkpoint",
      checkpoint_sequence: sequence,
      source_session_id_hash: "session-hash",
    },
    queueKind,
    outboxSlot,
    outboxGroupKey,
    idempotencyKey: key,
  };
}

try {
  const userTurn = await client.saveOutboxTurn({
    scope: "project-test",
    projectId: "project-id",
    sessionId: "codex-session",
    messages: [{
      message_id: "user-turn",
      role: "user",
      content: "ordinary user turn must survive checkpoint cleanup",
      timestamp: new Date().toISOString(),
    }],
    metadata: { integration: "codex" },
    queueKind: "turn",
    idempotencyKey: "ordinary-user-turn",
  });

  const legacyOne = await client.saveOutboxTurn(
    checkpointEntry({
      outboxSlot: "legacy-one",
      outboxGroupKey: "same-periodic-group",
      sequence: 1,
      key: "legacy-checkpoint-one",
    }),
  );
  const legacyLatest = await client.saveOutboxTurn(
    checkpointEntry({
      outboxSlot: "legacy-two",
      outboxGroupKey: "same-periodic-group",
      sequence: 2,
      key: "legacy-checkpoint-two",
    }),
  );
  const finalCheckpoint = await client.saveOutboxTurn(
    checkpointEntry({
      outboxSlot: "final-slot",
      outboxGroupKey: "final-group",
      queueKind: "final_checkpoint",
      sequence: 1,
      key: "final-checkpoint",
    }),
  );

  const stableFirst = await client.saveOutboxTurn(
    checkpointEntry({
      outboxSlot: "stable-periodic-slot",
      outboxGroupKey: "stable-periodic-slot",
      sequence: 3,
      key: "stable-checkpoint-three",
    }),
  );
  const stableLatest = await client.saveOutboxTurn(
    checkpointEntry({
      outboxSlot: "stable-periodic-slot",
      outboxGroupKey: "stable-periodic-slot",
      sequence: 4,
      key: "stable-checkpoint-four",
    }),
  );
  assert.equal(stableFirst.outboxId, stableLatest.outboxId, "periodic checkpoints must use one stable slot");
  assert.equal(
    await client.removeOutboxTurn(stableFirst.outboxId, {
      expectedIdempotencyKey: stableFirst.idempotencyKey,
    }),
    false,
    "an old drain must not delete a newer stable-slot entry",
  );

  const beforeCoalesce = await client.listOutboxTurns();
  const afterCoalesce = await client.coalesceOutboxCheckpoints(beforeCoalesce);
  assert(afterCoalesce.some((entry) => entry.outboxId === userTurn.outboxId));
  assert(afterCoalesce.some((entry) => entry.outboxId === legacyLatest.outboxId));
  assert(!afterCoalesce.some((entry) => entry.outboxId === legacyOne.outboxId));
  assert(afterCoalesce.some((entry) => entry.outboxId === finalCheckpoint.outboxId));
  assert(afterCoalesce.some((entry) => entry.outboxId === stableLatest.outboxId));

  const sorted = client.sortOutboxEntries([
    afterCoalesce.find((entry) => entry.outboxId === stableLatest.outboxId),
    afterCoalesce.find((entry) => entry.outboxId === finalCheckpoint.outboxId),
    afterCoalesce.find((entry) => entry.outboxId === userTurn.outboxId),
  ]);
  assert.deepEqual(sorted.map((entry) => entry.outboxId), [
    userTurn.outboxId,
    finalCheckpoint.outboxId,
    stableLatest.outboxId,
  ], "ordinary turns must drain before checkpoints");

  let recallMode = "success";
  const server = createServer(async (request, response) => {
    if (!request.url?.endsWith("/recall")) {
      response.writeHead(404);
      response.end();
      return;
    }
    let body = "";
    for await (const chunk of request) body += chunk;
    const scope = decodeURIComponent(String(request.url).split("/scopes/")[1].split("/recall")[0]);
    const shouldFail = recallMode === "failed" || (recallMode === "partial" && scope === "global-test");
    if (shouldFail) {
      const requestId = `request-${scope}-${recallMode}`;
      response.writeHead(503, {
        "content-type": "application/json",
        "x-request-id": requestId,
      });
      response.end(JSON.stringify({
        error: { code: "temporary_unavailable", message: "mock unavailable", request_id: requestId },
      }));
      return;
    }
    response.writeHead(200, {
      "content-type": "application/json",
      "x-request-id": `request-${scope}-${recallMode}`,
    });
    response.end(JSON.stringify({
      query_id: `query-${scope}-${recallMode}`,
      prompt_evidence: {
        content: `memory content for ${scope}`,
        window_count: 1,
      },
    }));
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "localhost", resolve);
  });
  const address = server.address();
  process.env.TMCRA_BASE_URL = `http://localhost:${address.port}`;

  const input = {
    session_id: "recall-session",
    turn_id: "recall-turn-success",
    cwd: root,
    model: "regression",
  };
  const completed = await hooks.recallForContext(input, "successful recall");
  const completedReceipt = await client.loadRecallReceipt(completed.scopes.projectId);
  assert.equal(completed.status, "completed");
  assert.equal(completedReceipt.status, "completed");
  assert.equal(completedReceipt.global.queryId, "query-global-test-success");
  assert.equal(completedReceipt.project.requestId, "request-project-test-success");
  const exactSuccessReceipt = await client.loadRecallReceiptForTurn(
    completed.scopes.projectId,
    input.session_id,
    input.turn_id,
    "current",
  );
  assert.equal(exactSuccessReceipt.query, "successful recall");
  assert.equal(
    await client.loadRecallReceiptForTurn(completed.scopes.projectId, "other-session", input.turn_id, "current"),
    null,
    "a same-project receipt must not cross session boundaries",
  );

  recallMode = "partial";
  const degraded = await hooks.recallForContext({ ...input, turn_id: "recall-turn-partial" }, "partial recall");
  const degradedReceipt = await client.loadRecallReceipt(degraded.scopes.projectId);
  assert.equal(degraded.status, "degraded");
  assert.equal(degradedReceipt.status, "degraded");
  assert.equal(degradedReceipt.global.status, "failed");
  assert.equal(degradedReceipt.project.status, "success");
  assert.equal(degradedReceipt.global.requestId, "request-global-test-partial");
  const exactDegradedReceipt = await client.loadRecallReceiptForTurn(
    degraded.scopes.projectId,
    input.session_id,
    "recall-turn-partial",
    "current",
  );
  assert.equal(exactDegradedReceipt.query, "partial recall");

  recallMode = "failed";
  const failed = await hooks.recallForContext({ ...input, turn_id: "recall-turn-failed" }, "failed recall");
  const failedReceipt = await client.loadRecallReceipt(failed.scopes.projectId);
  assert.equal(failed.status, "failed");
  assert.equal(failedReceipt.status, "failed");
  assert.equal(failed.context, "");
  const exactFailedReceipt = await client.loadRecallReceiptForTurn(
    failed.scopes.projectId,
    input.session_id,
    "recall-turn-failed",
    "current",
  );
  assert.equal(exactFailedReceipt.status, "failed");

  const logLines = (await readFile(join(root, "logs", "events.jsonl"), "utf8"))
    .split(/\r?\n/u)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  assert(logLines.some((row) => row.event === "recall_completed"));
  assert(logLines.some((row) => row.event === "recall_degraded"));
  assert(logLines.some((row) => row.event === "recall_failed"));
  await new Promise((resolve) => server.close(resolve));
  process.stdout.write("closed-loop regression mock passed\n");
} finally {
  for (const [name, value] of previousEnvironment) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
  await rm(root, { recursive: true, force: true });
}
