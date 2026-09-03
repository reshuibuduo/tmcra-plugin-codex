import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import http from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const root = await mkdtemp(join(tmpdir(), "tmcra-outbox-terminal-failure-"));
process.env.PLUGIN_DATA = root;
const client = await import("../scripts/tmcra_client.mjs");
let ingestRequests = 0;
let jobRequests = 0;

const server = http.createServer((request, response) => {
  request.resume();
  response.setHeader("content-type", "application/json");
  if (request.method === "GET" && request.url?.includes("/jobs/")) {
    jobRequests += 1;
    response.writeHead(200);
    response.end(JSON.stringify({
      job_id: "failed-job",
      status: "failed",
      error: { code: "writer_validation_failed" },
    }));
    return;
  }
  ingestRequests += 1;
  response.writeHead(202);
  response.end(JSON.stringify({ job_id: "failed-job", status: "queued" }));
});
await new Promise((resolve) => server.listen(0, "localhost", resolve));
const address = server.address();
assert(address && typeof address === "object");

const entry = await client.saveOutboxTurn({
  scope: "project-test",
  projectId: "project-test",
  sessionId: "session-test",
  messages: [{
    message_id: "message-test",
    role: "user",
    content: "a failed remote job must not be silently replayed",
    timestamp: new Date().toISOString(),
  }],
  metadata: { integration: "test" },
  consistency: "eventual",
  slowPolicy: "auto",
  idempotencyKey: "outbox-terminal-failure-test",
});

const drainScript = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "scripts",
  "drain_outbox.mjs",
);

try {
  await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [drainScript], {
      env: {
        ...process.env,
        PLUGIN_DATA: root,
        TMCRA_BASE_URL: `http://localhost:${address.port}`,
        TMCRA_API_KEY: "test-key",
        TMCRA_OUTBOX_JOB_POLL_MS: "50",
      },
      stdio: "inherit",
    });
    child.once("error", reject);
    child.once("exit", (code) => code === 0
      ? resolve()
      : reject(new Error(`drain exited ${code}`)));
  });

  assert.equal(ingestRequests, 1, "a failed accepted job must not be submitted again");
  assert.equal(jobRequests, 1, "the accepted job must be inspected exactly once");
  assert.equal(
    (await readdir(join(root, "outbox"))).filter((name) => name.endsWith(".json")).length,
    0,
  );
  const failed = JSON.parse(await readFile(join(root, "outbox-failed", `${entry.outboxId}.json`), "utf8"));
  assert.equal(failed.remoteStatus, "failed");
  assert.equal(failed.errorCode, "writer_validation_failed");
  const receipt = JSON.parse(await readFile(join(root, "outbox-receipts", `${entry.outboxId}.json`), "utf8"));
  assert.equal(receipt.state, "failed");
  assert.equal(receipt.remoteStatus, "failed");
  const status = await client.outboxStatus();
  assert.equal(status.queuedCount, 0);
  assert.equal(status.failedCount, 1);
  assert.equal(status.state, "attention_required");
} finally {
  await new Promise((resolve) => server.close(resolve));
  await rm(root, { recursive: true, force: true });
}

process.stdout.write("outbox terminal failure smoke passed\n");
