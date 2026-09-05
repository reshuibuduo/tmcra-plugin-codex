import assert from "node:assert/strict";
import { mkdtemp, mkdir, copyFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig, localProviderExecutionHeaders } from "../scripts/tmcra_client.mjs";
import { executeAvailableProviderTasks } from "../scripts/provider_executor.mjs";
import { assertActiveMemoryConnection, assertCloudProvidersAllowed } from "../scripts/local_binding.mjs";

const root = await mkdtemp(join(tmpdir(), "tmcra-local-contract-"));
const previous = { ...process.env };
try {
  const path = join(root, "local.json");
  await writeFile(path, JSON.stringify({ deploymentMode: "local", baseUrl: "http://127.0.0.1:2009",
    apiKey: "synthetic-local-key", globalScope: "local-global", projectScopePrefix: "local-project" }));
  process.env.TMCRA_CONFIG_FILE = path;
  process.env.TMCRA_BASE_URL = "https://cloud.example.invalid";
  process.env.TMCRA_API_KEY = "synthetic-cloud-key";
  const config = await loadConfig();
  assert.equal(config.baseUrl, "http://127.0.0.1:2009");
  assert.equal(config.apiKey, "synthetic-local-key");
  assert.deepEqual(await localProviderExecutionHeaders("writer", config), {});
  assert.deepEqual(await localProviderExecutionHeaders("organizer", config), {});
  let calls = 0;
  const result = await executeAvailableProviderTasks({ config, providerConfig: { writer: {} },
    fetchImpl: async () => { calls++; throw Error("cloud call attempted"); } });
  assert.equal(result.executed, 0);
  assert.equal(calls, 0);
  delete process.env.TMCRA_CONFIG_FILE;
  process.env.TMCRA_LOCAL_BINDING_FILE = join(root, "local-memory.json");
  const secrets = join(root, "state/lite-cpu/secrets");
  await mkdir(secrets, { recursive: true });
  await writeFile(process.env.TMCRA_LOCAL_BINDING_FILE, JSON.stringify({ schemaVersion: 1, mode: "local", dataRoot: root, profile: "lite-cpu" }));
  await assert.rejects(() => loadConfig(), /ENOENT/);
  await copyFile(path, join(secrets, "client-plugin.json"));
  assert.equal((await loadConfig()).apiKey, "synthetic-local-key");
  await assertActiveMemoryConnection(config);
  await assert.rejects(() => assertActiveMemoryConnection({ baseUrl: "https://cloud.example.invalid", apiKey: "synthetic-cloud-key" }), /blocked/);
  await assert.rejects(() => assertCloudProvidersAllowed(), /blocked/);
  process.env.TMCRA_CONFIG_FILE = path;
  await writeFile(path, JSON.stringify({ deploymentMode: "local", baseUrl: "https://cloud.example.invalid" }));
  await assert.rejects(() => loadConfig(), /numeric loopback/);
  console.log(JSON.stringify({ ok: true, localIdentity: true, inheritedCloudConfigIgnored: true, cloudWorkerCalls: 0 }));
} finally {
  for (const key of Object.keys(process.env)) if (!(key in previous)) delete process.env[key];
  Object.assign(process.env, previous);
  await rm(root, { recursive: true, force: true });
}
