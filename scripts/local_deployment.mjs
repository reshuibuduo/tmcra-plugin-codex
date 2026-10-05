import { existsSync, createWriteStream } from "node:fs";
import { readFile, mkdir } from "node:fs/promises";
import { spawn } from "node:child_process";
import { homedir, totalmem } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const pluginRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const apiRoot = [process.env.TMCRA_LOCAL_API_ROOT, join(pluginRoot, "runtime/memory-api"),
  resolve(pluginRoot, "../tmcra-source/02-tmcra-memory-api")].filter(Boolean)
  .find(path => existsSync(join(path, "deploy/Install-TmcraLocal.ps1")));
const dataRoot = resolve(process.env.TMCRA_LOCAL_DATA_ROOT || join(process.env.LOCALAPPDATA || homedir(), "TMCRA/local"));
let operation = { state: "idle" };

async function readJson(path, fallback) {
  try { return JSON.parse(await readFile(path, "utf8")); }
  catch (error) { if (error.code === "ENOENT") return fallback; throw error; }
}

async function installedApiRoot() {
  const receipt = await readJson(join(dataRoot, "installation.json"), null);
  return receipt?.api_root || apiRoot;
}

export async function localDeploymentStatus() {
  const catalog = await readJson(new URL("../resources/local-model-profiles.json", import.meta.url), { profiles: [] });
  const installed = await readJson(join(dataRoot, "installation.json"), null);
  const running = await readJson(join(dataRoot, "running.json"), null);
  const launchError = await readJson(join(dataRoot, "launch-error.json"), null);
  if (operation.state === "starting" && launchError?.at * 1000 >= Date.parse(operation.startedAt))
    operation = { ...operation, state: "failed", error: launchError.detail };
  let ready = false;
  if (installed && Number.isInteger(installed.api_port) && installed.api_port > 1023 && installed.api_port < 65536) {
    try {
      const response = await fetch(`http://127.0.0.1:${installed.api_port}/readyz`, { signal: AbortSignal.timeout(1500), redirect: "error" });
      const body = await response.json();
      ready = response.ok && (body.status === "ready" || body.ready === true);
    } catch {}
  }
  if (ready && operation.state === "starting") operation.state = "ready";
  return { available: process.platform === "win32" && process.arch === "x64" && Boolean(apiRoot),
    requirement: "Windows x64；自动准备 Python，无需 TMCRA 账号；首次下载需要联网",
    missing: apiRoot ? null : "本地运行包不完整，请重新下载安装包。",
    profiles: catalog.profiles, dataRoot, ramGiB: Math.round(totalmem() / 1024 ** 3),
    recommendedProfile: installed?.hardware?.recommended_profile || "lite-cpu",
    installedProfile: installed?.profile || null, ready,
    running: Boolean(running && !running.stopped), operation: { ...operation },
    connectionConfig: installed ? join(dataRoot, "state", installed.profile, "secrets/client-plugin.json") : null,
    automaticLocalBinding: Boolean(installed) };
}

export async function installLocalDeployment(profile) {
  const state = await localDeploymentStatus();
  if (!state.available) throw Error(state.missing || state.requirement);
  const selected = state.profiles.find(p => p.id === profile);
  if (!selected) throw Error("请选择已登记的本地模型档位。");
  if (["installing", "starting"].includes(operation.state) || state.running) throw Error("本地任务正在运行，请先查看状态或停止实例。");
  if (state.ramGiB < selected.system_ram_gib_min) throw Error(`此档位至少需要 ${selected.system_ram_gib_min}GB 内存。`);
  operation = { state: "installing", profile, startedAt: new Date().toISOString(), event: "正在准备独立运行环境" };
  await mkdir(dataRoot, { recursive: true });
  const log = createWriteStream(join(dataRoot, "installation.log"), { flags: "a", mode: 0o600 });
  log.on("error", () => {});
  // Executables/paths come from installed code, never from browser request fields.
  const child = spawn("powershell.exe", ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
    join(apiRoot, "deploy/Install-TmcraLocal.ps1"), "-Profile", profile, "-DataDir", dataRoot,
    "-Device", profile === "lite-cpu" ? "cpu" : "auto"], { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
  let pending = "";
  child.stdout.on("data", chunk => {
    log.write(chunk);
    pending = (pending + chunk.toString()).slice(-12000);
    const lines = pending.split(/\r?\n/u); pending = lines.pop();
    for (const line of lines) {
      try { const value = JSON.parse(line); if (typeof value.event === "string") operation.event = value.event; }
      catch { /* Dependency output is kept out of the browser. */ }
    }
  });
  child.stderr.on("data", chunk => log.write(chunk));
  child.on("error", error => { operation = { ...operation, state: "failed", error: `无法启动安装程序：${error.code || "unknown"}` }; });
  child.on("exit", code => { log.end(); operation = { ...operation, state: code === 0 ? "starting" : "failed",
    ...(code !== 0 ? { error: "安装未完成。已下载文件保留；请查看本地安装日志后重试。" } : {}) }; });
  return { ...operation };
}

export async function stopLocalDeployment() {
  const state = await localDeploymentStatus();
  if (!apiRoot || !state.running) return { stopped: true };
  const runtimeRoot = await installedApiRoot();
  return new Promise((resolveStop, reject) => {
    const child = spawn(join(dataRoot, "venv/Scripts/python.exe"),
      ["-m", "tmcra_service.local_deployment", "stop", "--root", dataRoot],
      { cwd: runtimeRoot, windowsHide: true, stdio: "ignore" });
    child.on("error", reject);
    child.on("exit", code => code === 0 ? (operation = { state: "idle" }, resolveStop({ stopRequested: true }))
      : reject(Error("停止请求失败；现有本地数据保持原样。")));
  });
}

export async function startLocalDeployment() {
  const state = await localDeploymentStatus();
  if (!state.available || !state.installedProfile) throw Error("请先完成本地运行包和模型安装。");
  if (state.running || ["installing", "starting"].includes(operation.state)) throw Error("本地实例正在运行或启动中。");
  operation = { state: "starting", startedAt: new Date().toISOString() };
  const runtimeRoot = await installedApiRoot();
  const child = spawn(join(dataRoot, "venv/Scripts/python.exe"),
    ["-m", "tmcra_service.local_deployment", "run", "--root", dataRoot],
    { cwd: runtimeRoot, windowsHide: true, detached: true, stdio: "ignore" });
  child.on("error", error => { operation = { ...operation, state: "failed", error: `启动失败：${error.code || "unknown"}` }; });
  child.unref();
  return { ...operation };
}
