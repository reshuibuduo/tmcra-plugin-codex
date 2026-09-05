import assert from "node:assert/strict";
import { fork } from "node:child_process";
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { MockTmcraServer } from "./mock_tmcra_server.mjs";

const plugin = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const root = await mkdtemp(join(tmpdir(), "tmcra-wakeup-race-"));
const previous = { ...process.env };
const workers = new Set();
const token = randomUUID();
const server = new MockTmcraServer({ validTokens: [token] });
const bounded = (promise, label, ms = 8000) => new Promise((resolve, reject) => {
  const timer = setTimeout(() => reject(Error(`Timed out: ${label}`)), ms);
  promise.then(value => { clearTimeout(timer); resolve(value); }, error => { clearTimeout(timer); reject(error); });
});

// Pause real fs operations in a child process only. Production source and
// scheduler timings remain unchanged; IPC selects the exact race window.
const preload = join(root, "gate.mjs");
await writeFile(preload, `
import fs from 'node:fs';
import { syncBuiltinESMExports } from 'node:module';
const op = process.env.RACE_OPERATION;
const original = fs.promises[op];
let paused = false;
fs.promises[op] = async (...args) => {
  if (!paused && String(args[0]).endsWith('.drain.lock')) {
    paused = true;
    const value = op === 'stat' ? await original(...args) : undefined;
    process.send({ gated: true });
    await new Promise(resolve => process.once('message', resolve));
    process.disconnect();
    return op === 'stat' ? value : original(...args);
  }
  return original(...args);
};
syncBuiltinESMExports();
`);
const producer = join(root, "producer.mjs");
await writeFile(producer, `import { startOutboxDrain } from ${JSON.stringify(pathToFileURL(join(plugin, "hooks/hook_common.mjs")).href)}; await startOutboxDrain();`);

function launch(script, operation) {
  const child = fork(script, [], { env: { ...process.env, RACE_OPERATION: operation },
    execArgv: operation ? ["--import", pathToFileURL(preload).href] : [],
    silent: true, windowsHide: true });
  workers.add(child);
  let errors = "";
  child.stderr.on("data", value => { errors += value; });
  const exited = new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", code => { workers.delete(child); code === 0 ? resolve() : reject(Error(`Worker failed (${code}): ${errors.replaceAll(token, "[REDACTED]")}`)); });
  });
  exited.catch(() => {});
  const gated = new Promise(resolve => child.once("message", resolve));
  return { child, exited, gated: bounded(gated, `${operation} gate`), release: () => child.send({ release: true }) };
}

try {
  for (const key of Object.keys(process.env)) if (key.startsWith("TMCRA_") || key === "PLUGIN_DATA" || key === "CLAUDE_PLUGIN_DATA") delete process.env[key];
  await server.start();
  process.env.TMCRA_CONFIG_FILE = join(root, "config.json");
  process.env.TMCRA_LOCAL_BINDING_FILE = join(root, "no-local-binding.json");
  process.env.TMCRA_PROVIDER_CONFIG_FILE = join(root, "no-providers.json");
  await writeFile(process.env.TMCRA_CONFIG_FILE, JSON.stringify({ baseUrl: server.baseUrl, apiKey: token }));
  const { saveOutboxTurn } = await import("../scripts/tmcra_client.mjs");
  const { startOutboxDrain } = await import("../hooks/hook_common.mjs");
  const enqueue = async name => saveOutboxTurn({ scope: "race-project", projectId: "race-project",
    sessionId: name, messages: [{ message_id: name, role: "user", content: name, timestamp: new Date().toISOString() }],
    metadata: { integration: "race-test" }, consistency: "eventual", slowPolicy: "auto", idempotencyKey: name });

  // The worker has decided the queue is empty but still owns the drain lock.
  // A new entry signals it during that interval; no subsequent hook may be needed.
  process.env.PLUGIN_DATA = join(root, "consumer-exit");
  let before = server.records.length;
  const exiting = launch(join(plugin, "scripts/drain_outbox.mjs"), "rm");
  await exiting.gated;
  await enqueue("queued-during-worker-exit");
  await startOutboxDrain();
  exiting.release();
  await bounded(exiting.exited, "consumer handoff");
  assert.equal(server.records.length, before + 1, "a signal during worker exit must drain without another host event");
  assert.equal((await readdir(join(process.env.PLUGIN_DATA, "outbox"))).filter(name => name.endsWith(".json")).length, 0);

  // The producer observed the old lock, then the worker released it and finished
  // its final request check before the producer actually wrote its signal.
  process.env.PLUGIN_DATA = join(root, "producer-observation");
  before = server.records.length;
  const oldWorker = launch(join(plugin, "scripts/drain_outbox.mjs"), "rm");
  await oldWorker.gated;
  await enqueue("queued-after-stale-observation");
  const staleProducer = launch(producer, "stat");
  await staleProducer.gated;
  oldWorker.release();
  await bounded(oldWorker.exited, "old worker exit");
  staleProducer.release();
  await bounded(staleProducer.exited, "producer handoff");
  const deadline = Date.now() + 8000;
  while (server.records.length !== before + 1 || (await readdir(join(process.env.PLUGIN_DATA, "outbox"))).some(name => name.endsWith(".json") || name === ".drain.lock")) {
    assert(Date.now() < deadline, "producer must launch a replacement after observing a released lock");
    await new Promise(resolve => setTimeout(resolve, 25));
  }

  process.env.PLUGIN_DATA = join(root, "concurrent-signals");
  const outbox = join(process.env.PLUGIN_DATA, "outbox");
  await mkdir(outbox, { recursive: true });
  await writeFile(join(outbox, ".drain.lock"), "test-owned-lock");
  await Promise.all(Array.from({ length: 20 }, () => startOutboxDrain()));
  assert((await readdir(outbox)).includes(".drain.request"));
  const events = await readFile(join(process.env.PLUGIN_DATA, "logs/events.jsonl"), "utf8").catch(error => {
    if (error.code === "ENOENT") return ""; throw error;
  });
  assert(!events.includes("outbox_drain_launch_failed"), "concurrent signals must be idempotent");
  console.log(JSON.stringify({ ok: true, consumerExitWakeup: true, staleProducerObservation: true, concurrentSignals: 20 }));
} finally {
  for (const child of workers) child.kill();
  await server.stop();
  for (const key of Object.keys(process.env)) if (!(key in previous)) delete process.env[key];
  Object.assign(process.env, previous);
  // Only this test's explicit mkdtemp directory is eligible for cleanup.
  if (dirname(root) === resolve(tmpdir()) && root.startsWith(join(resolve(tmpdir()), "tmcra-wakeup-race-")))
    await rm(root, { recursive: true, force: true });
}
