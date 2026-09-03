import assert from "node:assert/strict";
import { mkdtemp, readdir, rm } from "node:fs/promises";
import http from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const root = await mkdtemp(join(tmpdir(), "tmcra-outbox-transient-"));
process.env.PLUGIN_DATA = root;
const client = await import("../scripts/tmcra_client.mjs");

let ingestRequests = 0;
let jobRequests = 0;
const server = http.createServer((request, response) => {
  request.resume();
  if (request.method === "GET" && request.url?.includes("/jobs/")) {
    jobRequests += 1;
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({ job_id: "accepted-job", status: "succeeded" }));
    return;
  }
  ingestRequests += 1;
  if (ingestRequests <= 2) {
    response.writeHead(429, {
      "content-type": "application/json",
      "retry-after": "0.05",
    });
    response.end(JSON.stringify({
      error: {
        code: "tenant_queue_full",
        message: "queue is full",
        request_id: "transient-test-request",
      },
    }));
    return;
  }
  response.writeHead(202, { "content-type": "application/json" });
  response.end(JSON.stringify({ job_id: "accepted-job", status: "queued" }));
});
await new Promise((resolve) => server.listen(0, "localhost", resolve));
const address = server.address();
assert(address && typeof address === "object");

await client.saveOutboxTurn({
  scope: "project-test",
  projectId: "project-test",
  sessionId: "session-test",
  messages: [{
    message_id: "message-test",
    role: "user",
    content: "queued through temporary saturation",
    timestamp: new Date().toISOString(),
  }],
  metadata: { integration: "test" },
  consistency: "eventual",
  slowPolicy: "auto",
  idempotencyKey: "outbox-transient-test",
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
      },
      stdio: "inherit",
    });
    child.once("error", reject);
    child.once("exit", (code) =>
      code === 0 ? resolve() : reject(new Error(`drain exited ${code}`)),
    );
  });

  assert.equal(ingestRequests, 3, "the same drain process must retry after client retries exhaust");
  assert.equal(jobRequests, 1, "accepted ingest must be verified to a remote terminal state");
  assert.equal(
    (await readdir(join(root, "outbox"))).filter((name) => name.endsWith(".json")).length,
    0,
  );
  assert.equal((await client.outboxStatus()).state, "healthy");
} finally {
  await new Promise((resolve) => server.close(resolve));
  await rm(root, { recursive: true, force: true });
}

process.stdout.write("outbox transient retry smoke passed\n");
