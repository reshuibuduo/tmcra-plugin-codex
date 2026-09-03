import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import http from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";

const root = await mkdtemp(join(tmpdir(), "tmcra-outbox-formal-recovery-"));
process.env.PLUGIN_DATA = root;
const client = await import("../scripts/tmcra_client.mjs");
let getRequests = 0;
let retryRequests = 0;
let ingestRequests = 0;
let succeededJobId = null;

const server = http.createServer((request, response) => {
  request.resume();
  response.setHeader("content-type", "application/json");
  const unsafe = request.url?.includes("unsafe-job");
  if (request.method === "GET" && request.url?.includes("/jobs/")) {
    getRequests += 1;
    const jobId = unsafe ? "unsafe-job" : request.url?.includes("already-job")
      ? "already-job"
      : "recover-job";
    response.writeHead(200);
    response.end(JSON.stringify({
      job_id: jobId,
      status: jobId === succeededJobId ? "succeeded" : "failed",
    }));
    return;
  }
  if (request.method === "POST" && request.url?.endsWith("/retry")) {
    retryRequests += 1;
    response.writeHead(200);
    response.end(JSON.stringify({
      job_id: unsafe ? "unsafe-job" : "recover-job",
      status: "pending",
      idempotent_retry: false,
      ...(unsafe ? {} : { resume_mode: "audited_writer_state" }),
    }));
    return;
  }
  ingestRequests += 1;
  response.writeHead(500);
  response.end(JSON.stringify({ error: { code: "unexpected_ingest" } }));
});
await new Promise((resolve) => server.listen(0, "localhost", resolve));
const address = server.address();
assert(address && typeof address === "object");
const config = {
  apiKey: "test-key",
  baseUrl: `http://localhost:${address.port}`,
  timeoutMs: 5_000,
};

async function failedEntry(name, jobId) {
  const entry = await client.saveOutboxTurn({
    scope: "project-test",
    projectId: "project-test",
    sessionId: `session-${name}`,
    messages: [{
      message_id: `message-${name}`,
      role: "user",
      content: "formal recovery must not submit a second ingest",
      timestamp: new Date().toISOString(),
    }],
    metadata: { integration: "test" },
    consistency: "eventual",
    slowPolicy: "auto",
    idempotencyKey: `outbox-formal-recovery-${name}`,
  });
  const submitted = await client.markOutboxSubmitted(entry, {
    job_id: jobId,
    status: "pending",
  });
  await client.completeOutboxTurn(submitted, {
    job_id: jobId,
    status: "failed",
    error: { code: "writer_validation_failed" },
  });
  return submitted;
}

try {
  const recoverable = await failedEntry("recover", "recover-job");
  const recovered = await client.retryOutboxTurn(recoverable.outboxId, { config });
  assert.equal(recovered.jobId, "recover-job");
  assert.equal(recovered.resumeMode, "audited_writer_state");
  assert.equal(
    existsSync(join(root, "outbox-failed", `${recoverable.outboxId}.json`)),
    false,
  );
  const active = (await client.listOutboxTurns()).find(
    (entry) => entry.outboxId === recoverable.outboxId,
  );
  assert.equal(active?.jobId, "recover-job");
  assert.equal(active?.resumeMode, "audited_writer_state");
  await client.completeOutboxTurn(active, { job_id: "recover-job", status: "succeeded" });
  const receipt = JSON.parse(
    await readFile(join(root, "outbox-receipts", `${recoverable.outboxId}.json`), "utf8"),
  );
  assert.equal(receipt.state, "succeeded");
  assert.equal(receipt.resumeMode, "audited_writer_state");

  const unsafe = await failedEntry("unsafe", "unsafe-job");
  await assert.rejects(
    client.retryOutboxTurn(unsafe.outboxId, { config }),
    /audited Writer resume contract/u,
  );
  assert.equal(
    existsSync(join(root, "outbox-failed", `${unsafe.outboxId}.json`)),
    true,
  );

  const already = await failedEntry("already", "already-job");
  succeededJobId = "already-job";
  const retryCountBeforeReconcile = retryRequests;
  const reconciled = await client.retryOutboxTurn(already.outboxId, { config });
  assert.equal(reconciled.reconciled, true);
  assert.equal(reconciled.remoteStatus, "succeeded");
  assert.equal(retryRequests, retryCountBeforeReconcile);
  assert.equal(
    existsSync(join(root, "outbox-failed", `${already.outboxId}.json`)),
    false,
  );
  const reconciledReceipt = JSON.parse(
    await readFile(join(root, "outbox-receipts", `${already.outboxId}.json`), "utf8"),
  );
  assert.equal(reconciledReceipt.state, "succeeded");
  assert.equal(reconciledReceipt.remoteStatus, "succeeded");
  assert.equal(ingestRequests, 0, "formal recovery must never call ingest");
  assert.equal(getRequests, 3);
  assert.equal(retryRequests, 2);
} finally {
  await new Promise((resolve) => server.close(resolve));
  await rm(root, { recursive: true, force: true });
}

process.stdout.write("outbox formal recovery smoke passed\n");
