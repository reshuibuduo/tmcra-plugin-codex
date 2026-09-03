import assert from "node:assert/strict";
import { mkdtemp, readdir, rm } from "node:fs/promises";
import http from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const root = await mkdtemp(join(tmpdir(), "tmcra-outbox-quarantine-"));
process.env.PLUGIN_DATA = root;
const client = await import("../scripts/tmcra_client.mjs");

let ingestRequests = 0;
let recoveryRequests = 0;
let supportRecoveryMode = false;
let supportRecoveryRequests = 0;
let jobRequests = 0;
const requestOrder = [];
const server = http.createServer((request, response) => {
  request.resume();
  if (request.method === "GET" && request.url?.includes("/jobs/")) {
    jobRequests += 1;
    requestOrder.push("job");
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({
      job_id: request.url.split("/").at(-1),
      status: "succeeded",
    }));
    return;
  }
  if (request.url?.endsWith("/recovery")) {
    recoveryRequests += 1;
    requestOrder.push("recovery");
    if (supportRecoveryMode) {
      supportRecoveryRequests += 1;
      const writesAvailable = supportRecoveryRequests >= 2;
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({
        state: writesAvailable ? "healthy" : "attention_required",
        phase: writesAvailable ? "complete" : "manual_review",
        progress_percent: writesAvailable ? 100 : 50,
        completed_items: writesAvailable ? 2 : 1,
        total_items: 2,
        pending_items: writesAvailable ? 0 : 1,
        recovery_attempts: 1,
        automatic: !writesAvailable,
        reads_available: writesAvailable,
        writes_available: writesAvailable,
        requires_support: !writesAvailable,
        started_at: Date.now() / 1000,
        updated_at: Date.now() / 1000,
        next_attempt_at: null,
      }));
      return;
    }
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({
      state: "recovering",
      phase: "repairing",
      progress_percent: 50,
      completed_items: 1,
      total_items: 2,
      pending_items: 1,
      recovery_attempts: 1,
      automatic: true,
      reads_available: false,
      writes_available: false,
      requires_support: false,
      started_at: Date.now() / 1000,
      updated_at: Date.now() / 1000,
      next_attempt_at: Date.now() / 1000 + 0.05,
    }));
    return;
  }
  ingestRequests += 1;
  requestOrder.push("ingest");
  if (!supportRecoveryMode && ingestRequests === 1) {
    response.writeHead(422, {
      "content-type": "application/json",
      "retry-after": "0.05",
    });
    response.end(
      JSON.stringify({
        error: {
          code: "scope_quarantined",
          message: "scope is quarantined",
          request_id: "test-request",
        },
      }),
    );
    return;
  }
  response.writeHead(202, { "content-type": "application/json" });
  response.end(JSON.stringify({ job_id: "recovered-job", status: "queued" }));
});
await new Promise((resolve) => server.listen(0, "localhost", resolve));
const address = server.address();
assert(address && typeof address === "object");

const entry = await client.saveOutboxTurn({
  scope: "project-test",
  projectId: "project-test",
  sessionId: "session-test",
  messages: [
    {
      message_id: "message-test",
      role: "user",
      content: "queued while repair is active",
      timestamp: new Date().toISOString(),
    },
  ],
  metadata: { integration: "test" },
  consistency: "eventual",
  slowPolicy: "auto",
  idempotencyKey: "outbox-quarantine-test",
});

const drainScript = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "scripts",
  "drain_outbox.mjs",
);

async function runDrain() {
  await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [drainScript], {
      env: {
        ...process.env,
        PLUGIN_DATA: root,
        TMCRA_BASE_URL: `http://localhost:${address.port}`,
        TMCRA_API_KEY: "test-key",
        TMCRA_OUTBOX_CIRCUIT_MIN_MS: "50",
        TMCRA_OUTBOX_CIRCUIT_MAX_MS: "1000",
        TMCRA_OUTBOX_SUPPORT_RECHECK_MS: "50",
      },
      stdio: "inherit",
    });
    child.once("error", reject);
    child.once("exit", (code) =>
      code === 0 ? resolve() : reject(new Error(`drain exited ${code}`)),
    );
  });
}

try {
  await runDrain();
  assert.equal(ingestRequests, 2, "the same drain process must resume after quarantine");
  assert.equal(recoveryRequests, 1, "the drain must read server recovery state once");
  assert.equal(
    (await readdir(join(root, "outbox"))).filter((name) => name.endsWith(".json")).length,
    0,
  );
  assert.equal(jobRequests, 1, "accepted ingest must be verified to a remote terminal state");
  const cleared = await client.outboxStatus();
  assert.equal(cleared.state, "healthy");
  assert.equal(cleared.queuedCount, 0);

  supportRecoveryMode = true;
  supportRecoveryRequests = 0;
  jobRequests = 0;
  ingestRequests = 0;
  requestOrder.length = 0;
  const supportEntry = await client.saveOutboxTurn({
    scope: "project-support-recovered",
    projectId: "project-support-recovered",
    sessionId: "session-support-recovered",
    messages: [
      {
        message_id: "message-support-recovered",
        role: "user",
        content: "queued while manual review status later becomes healthy",
        timestamp: new Date().toISOString(),
      },
    ],
    metadata: { integration: "test" },
    consistency: "eventual",
    slowPolicy: "auto",
    idempotencyKey: "outbox-support-recovery-test",
  });
  await client.saveOutboxTurn({
    scope: "project-support-recovered",
    projectId: "project-support-recovered",
    sessionId: "session-support-recovered",
    messages: [
      {
        message_id: "message-support-recovered-2",
        role: "assistant",
        content: "a second queued turn shares the same recovery circuit",
        timestamp: new Date().toISOString(),
      },
    ],
    metadata: { integration: "test" },
    consistency: "eventual",
    slowPolicy: "auto",
    idempotencyKey: "outbox-support-recovery-test-2",
  });
  await client.openOutboxCircuit(supportEntry, {
    status: 422,
    code: "scope_quarantined",
    recovery: {
      state: "attention_required",
      phase: "manual_review",
      writes_available: false,
      requires_support: true,
      next_attempt_at: null,
    },
  });

  await runDrain();
  assert.equal(
    supportRecoveryRequests,
    2,
    "a stale support circuit must be rechecked without probing ingest",
  );
  assert.equal(ingestRequests, 2, "each queued turn must ingest only after writes are available");
  assert.deepEqual(
    requestOrder,
    ["recovery", "recovery", "ingest", "ingest", "job", "job"],
    "one shared recovery check per pass must authorize queued writes",
  );
  assert.equal(jobRequests, 2, "both accepted jobs must reach a verified remote terminal state");
  const supportCleared = await client.outboxStatus();
  assert.equal(supportCleared.state, "healthy");
  assert.equal(supportCleared.queuedCount, 0);
} finally {
  await new Promise((resolve) => server.close(resolve));
  await rm(root, { recursive: true, force: true });
}

process.stdout.write("outbox quarantine circuit smoke passed\n");
