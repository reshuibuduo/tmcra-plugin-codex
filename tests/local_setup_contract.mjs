import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const root = await mkdtemp(join(tmpdir(), "tmcra-account-free-"));
const env = { ...process.env, TMCRA_LOCAL_BINDING_FILE: join(root, "binding.json"),
  TMCRA_LOCAL_DATA_ROOT: join(root, "data"), TMCRA_MEMORY_STATE_DIR: join(root, "controls"),
  TMCRA_PROVIDER_CONFIG_FILE: join(root, "providers.json") };
for (const key of ["TMCRA_API_KEY", "TMCRA_ACCESS_TOKEN", "TMCRA_CONFIG_FILE"]) delete env[key];
const child = spawn(process.execPath, [...(process.argv.length > 2 ? process.argv.slice(2) : ["scripts/local_setup.mjs"]), "--no-open"],
  { windowsHide: true, env, stdio: ["ignore", "pipe", "pipe"] });
const exit = new Promise(resolve => child.once("exit", resolve));
let output = "", errors = "";
try {
  const launch = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(Error(`Account-free setup failed to start: ${errors}`)), 15000);
    child.on("error", reject);
    child.stderr.on("data", chunk => { errors += chunk; });
    child.stdout.on("data", chunk => {
      output += chunk;
      if (output.includes("\n")) { clearTimeout(timer); try { resolve(JSON.parse(output.trim())); } catch (error) { reject(error); } }
    });
    child.once("exit", code => { clearTimeout(timer); reject(Error(`Setup exited early: ${code}`)); });
  });
  assert.equal(launch.accountRequired, false);
  const url = new URL(launch.url);
  assert.equal(url.hostname, "127.0.0.1");
  const action = async name => fetch(`${url.origin}/api/action`, { method: "POST",
    headers: { "Content-Type": "application/json", "X-TMCRA-Token": url.hash.slice(1) },
    body: JSON.stringify({ action: name }) });
  const state = await (await action("local_deployment_status")).json();
  assert.equal(state.result.missing, null);
  assert.equal(state.result.profiles.length, 3);
  assert.equal(state.result.available, process.platform === "win32" && process.arch === "x64");
  assert.equal((await action("knowledge")).status, 400);
  assert.match(await (await fetch(url.origin)).text(), /记忆工作台/u);
  await action("close");
  assert.equal(await exit, 0);
  console.log(JSON.stringify({ ok: true, accountFreeSetup: true, bundledBackend: true, threeProfiles: true }));
} finally {
  child.kill(); await exit;
  await rm(root, { recursive: true, force: true });
}
