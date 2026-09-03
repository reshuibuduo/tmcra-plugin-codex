import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { executeAvailableProviderTasks } from "../scripts/provider_executor.mjs";

const root = await mkdtemp(join(tmpdir(), "tmcra-provider-executor-"));
process.env.PLUGIN_DATA = root;

assert.deepEqual(
  await executeAvailableProviderTasks({ maxTasks: 1 }),
  { executed: 0 },
  "an unconfigured local provider must stay idle without resolving service credentials",
);

async function readJson(request) {
  let text = "";
  for await (const chunk of request) text += chunk;
  return text ? JSON.parse(text) : {};
}

function listen(server) {
  return new Promise((resolvePromise, reject) => {
    server.once("error", reject);
    server.listen(0, "localhost", resolvePromise);
  });
}

function close(server) {
  return new Promise((resolvePromise, reject) => {
    server.close((error) => (error ? reject(error) : resolvePromise()));
  });
}

function send(response, status, value) {
  response.writeHead(status, { "Content-Type": "application/json" });
  response.end(JSON.stringify(value));
}

function task(stage, sequence) {
  const request = {
    schema_version: "tmcra.openai-compatible-request.1",
    messages: [
      { role: "system", content: "Return one JSON object." },
      { role: "user", content: JSON.stringify({ stage, sequence }) },
    ],
    temperature: 0,
    max_tokens: 128,
    response_format: { type: "json_object" },
  };
  return {
    schema_version: "tmcra.user-provider-task.1",
    task_id: `upt_${stage}_${sequence}`,
    stage,
    operation: `${stage}_contract_test`,
    request_sha256: createHash("sha256").update(JSON.stringify(request)).digest("hex"),
    request,
    lease_token: `lease-${stage}-${"x".repeat(40)}`,
    lease_expires_at: Date.now() / 1000 + 60,
  };
}

const queued = {
  writer: [task("writer", 1)],
  organizer: [task("organizer", 1)],
};
const serviceRequests = [];
const completions = [];
const service = createServer((request, response) => {
  void (async () => {
    assert.equal(request.headers.authorization, "Bearer service-device-token");
    const body = await readJson(request);
    serviceRequests.push({ url: request.url, body });
    if (request.url === "/v1/provider-tasks/claim") {
      send(response, 200, {
        task: queued[body.stage].shift() || null,
        retry_after_seconds: 0,
      });
      return;
    }
    if (request.url?.endsWith("/started") || request.url?.endsWith("/heartbeat")) {
      send(response, 200, { task_id: "task", state: "running", idempotent_replay: false });
      return;
    }
    if (request.url?.endsWith("/complete")) {
      completions.push(body);
      send(response, 200, { task_id: "task", state: "completed", idempotent_replay: false });
      return;
    }
    if (request.url?.endsWith("/fail")) {
      send(response, 200, { task_id: "task", state: "failed", idempotent_replay: false });
      return;
    }
    send(response, 404, {});
  })().catch((error) => send(response, 500, { error: { message: error.message } }));
});

const providerRequests = [];
const provider = createServer((request, response) => {
  void (async () => {
    assert.equal(request.url, "/v1/chat/completions");
    assert.equal(request.headers.authorization, "Bearer provider-secret-key");
    const body = await readJson(request);
    providerRequests.push(body);
    const requested = JSON.parse(body.messages[1].content);
    send(response, 200, {
      id: `provider-${requested.stage}`,
      choices: [
        {
          finish_reason: "stop",
          message: {
            role: "assistant",
            content: JSON.stringify({ ok: true, stage: requested.stage }),
          },
        },
      ],
      usage: {
        prompt_tokens: 12,
        completion_tokens: 4,
        total_tokens: 16,
        prompt_cache_hit_tokens: 2,
        prompt_cache_miss_tokens: 10,
      },
      provider_private_envelope: "must-stay-local",
    });
  })().catch((error) => send(response, 500, { error: { message: error.message } }));
});

await Promise.all([listen(service), listen(provider)]);
const serviceAddress = service.address();
const providerAddress = provider.address();
assert(serviceAddress && typeof serviceAddress !== "string");
assert(providerAddress && typeof providerAddress !== "string");

try {
  const result = await executeAvailableProviderTasks({
    config: {
      baseUrl: `http://localhost:${serviceAddress.port}`,
      apiKey: "service-device-token",
      tokenType: "Bearer",
      timeoutMs: 5_000,
      integrationId: "",
      agentId: "",
    },
    providerConfig: {
      writer: {
        provider: "openai-compatible",
        baseUrl: `http://localhost:${providerAddress.port}/v1`,
        model: "writer-model",
        apiKey: "provider-secret-key",
      },
      organizer: {
        inheritWriter: false,
        provider: "openai-compatible",
        baseUrl: `http://localhost:${providerAddress.port}/v1`,
        model: "organizer-model",
        apiKey: "provider-secret-key",
      },
    },
    maxTasks: 2,
  });
  assert.equal(result.executed, 2);
  assert.deepEqual(providerRequests.map((item) => item.model), ["writer-model", "organizer-model"]);
  assert.equal(completions.length, 2);
  assert.deepEqual(completions.map((item) => item.output.stage), ["writer", "organizer"]);
  assert.equal(JSON.stringify(completions).includes("must-stay-local"), false);
  assert.equal(JSON.stringify(serviceRequests).includes("provider-secret-key"), false);
  assert.equal(completions.every((item) => item.usage.total_tokens === 16), true);
} finally {
  await Promise.all([close(service), close(provider)]);
}

process.stdout.write(`${JSON.stringify({
  ok: true,
  writer: true,
  organizer: true,
  parsedResultOnly: true,
  providerCredentialLocal: true,
})}\n`);
