import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, stat } from "node:fs/promises";
import { createServer, request as httpRequest } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  normalizeProviderBaseUrl,
  probeProvider,
  publicProviderConfig,
  readProviderConfig,
} from "../scripts/provider_config.mjs";
import { startProviderSetupServer } from "../scripts/provider_setup.mjs";

const root = await mkdtemp(join(tmpdir(), "tmcra-provider-setup-"));
const configPath = join(root, "用户 配置", "local-providers.json");
const provider = createServer((request, response) => {
  assert.equal(request.url, "/v1/models");
  assert.equal(request.headers.authorization, "Bearer local-probe-credential");
  response.writeHead(200, { "Content-Type": "application/json" });
  response.end(JSON.stringify({ data: [{ id: "writer-model" }, { id: "organizer-model" }] }));
});
await new Promise((resolvePromise, reject) => {
  provider.once("error", reject);
  provider.listen(0, "127.0.0.1", resolvePromise);
});
const providerAddress = provider.address();
assert(providerAddress && typeof providerAddress !== "string");
const providerBaseUrl = `http://127.0.0.1:${providerAddress.port}/v1`;

const setup = await startProviderSetupServer({ configPath, open: false, idleTimeoutMs: 60_000 });
const headers = { "X-TMCRA-Setup-Token": setup.token };

try {
  const unauthorized = await fetch(`${setup.baseUrl}/api/config`);
  assert.equal(unauthorized.status, 403);
  const hostileHostStatus = await new Promise((resolvePromise, reject) => {
    const request = httpRequest(`${setup.baseUrl}/api/config`, {
      headers: { ...headers, Host: `${new URL(setup.baseUrl).host}.invalid` },
    }, (response) => {
      response.resume();
      response.once("end", () => resolvePromise(response.statusCode));
    });
    request.once("error", reject);
    request.end();
  });
  assert.equal(hostileHostStatus, 421);

  const initial = await fetch(`${setup.baseUrl}/api/config`, { headers }).then((response) => response.json());
  assert.equal(initial.ok, true);
  assert.equal(initial.config.configured, false);

  const savedResponse = await fetch(`${setup.baseUrl}/api/config`, {
    method: "PUT",
    headers: { ...headers, "Content-Type": "application/json" },
    body: JSON.stringify({
      writer: {
        provider: "local-openai-compatible",
        baseUrl: providerBaseUrl,
        model: "writer-model",
        apiKey: "local-probe-credential",
      },
      organizer: { inheritWriter: true },
    }),
  });
  const saved = await savedResponse.json();
  assert.equal(savedResponse.status, 200, JSON.stringify(saved));
  assert.equal(saved.config.writer.credentialPresent, true);
  assert.equal(JSON.stringify(saved).includes("local-probe-credential"), false);

  const testedResponse = await fetch(`${setup.baseUrl}/api/test`, {
    method: "POST",
    headers: { ...headers, "Content-Type": "application/json" },
    body: JSON.stringify({
      stage: "writer",
      config: {
        writer: {
          provider: "local-openai-compatible",
          baseUrl: providerBaseUrl,
          model: "writer-model",
          apiKey: "",
        },
        organizer: { inheritWriter: true },
      },
    }),
  });
  assert.equal(testedResponse.status, 200);
  const tested = await testedResponse.json();
  assert.equal(tested.ok, true);
  assert.equal(tested.modelVisible, true);

  const stored = await readProviderConfig(configPath);
  assert.equal(stored.writer.apiKey, "local-probe-credential");
  assert.equal(stored.organizer.inheritWriter, true);
  assert.equal(JSON.stringify(publicProviderConfig(stored)).includes("local-probe-credential"), false);
  assert.equal((await readFile(configPath, "utf8")).includes("local-probe-credential"), true);
  if (process.platform === "win32") {
    const acl = spawnSync("icacls.exe", [configPath], { encoding: "utf8", windowsHide: true });
    assert.equal(acl.status, 0, acl.stderr || acl.stdout);
    assert.doesNotMatch(acl.stdout, /\(I\)/u, "provider credential file must not inherit broad ACL entries");
  } else {
    assert.equal((await stat(configPath)).mode & 0o777, 0o600);
  }

  await assert.rejects(
    probeProvider("writer", {
      writer: {
        provider: "openai-compatible",
        baseUrl: "https://provider.example/v1",
        model: "writer-model",
        apiKey: "",
      },
      organizer: { inheritWriter: true },
    }, {
      path: configPath,
      fetchImpl: async () => { throw new Error("stored credential was sent to a changed endpoint"); },
    }),
    /API key is required/u,
  );

  assert.throws(() => normalizeProviderBaseUrl("http://example.com/v1"), /HTTPS/u);
  assert.throws(() => normalizeProviderBaseUrl("https://user:secret@example.com/v1"), /must not contain/u);
  assert.throws(() => normalizeProviderBaseUrl("https://example.com/v1?token=secret"), /must not contain/u);

  const clearedResponse = await fetch(`${setup.baseUrl}/api/credentials/writer`, {
    method: "DELETE",
    headers,
  });
  assert.equal(clearedResponse.status, 200);
  const cleared = await clearedResponse.json();
  assert.equal(cleared.config.writer.credentialPresent, false);
  assert.equal((await readProviderConfig(configPath)).writer.apiKey, undefined);

  const closed = new Promise((resolvePromise) => setup.server.once("close", resolvePromise));
  await fetch(`${setup.baseUrl}/api/close`, { method: "POST", headers });
  await closed;
} finally {
  if (setup.server.listening) await new Promise((resolvePromise) => setup.server.close(resolvePromise));
  await new Promise((resolvePromise) => provider.close(resolvePromise));
}

process.stdout.write(`${JSON.stringify({ ok: true, providerSetup: true, secretRedaction: true, loopbackOnly: true, credentialFileProtected: true })}\n`);
