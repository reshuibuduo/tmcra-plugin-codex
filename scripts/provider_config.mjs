import { randomUUID } from "node:crypto";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { chmod, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";

export const PROVIDER_CONFIG_SCHEMA_VERSION = 1;
export const PROVIDER_STAGES = Object.freeze(["writer", "organizer"]);
export const PROVIDER_KINDS = Object.freeze([
  "deepseek",
  "openai-compatible",
  "local-openai-compatible",
]);

const MAX_SECRET_LENGTH = 4_096;
const MAX_TEXT_LENGTH = 512;

async function protectCredentialFile(path) {
  if (process.platform !== "win32") {
    await chmod(path, 0o600);
    return;
  }
  const identity = spawnSync("whoami.exe", ["/user", "/fo", "csv", "/nh"], {
    encoding: "utf8",
    windowsHide: true,
  });
  const sid = String(identity.stdout || "").match(/S-1-5-(?:\d+-)+\d+/u)?.[0];
  if (identity.status !== 0 || !sid) throw new Error("could not identify the Windows user for credential protection");
  const protectedAcl = spawnSync("icacls.exe", [
    path,
    "/inheritance:r",
    "/grant:r",
    `*${sid}:F`,
    "/grant:r",
    "*S-1-5-18:F",
  ], { windowsHide: true, stdio: "ignore" });
  if (protectedAcl.status !== 0) throw new Error("could not protect the local provider credential file");
}

export function resolveProviderConfigPath(value = process.env.TMCRA_LOCAL_PROVIDER_CONFIG) {
  return resolve(value?.trim() || join(homedir(), ".config", "tmcra", "local-providers.json"));
}

function boundedText(value, field, { required = true, maximum = MAX_TEXT_LENGTH } = {}) {
  const normalized = String(value ?? "").trim();
  if (required && !normalized) throw new Error(`${field} is required`);
  if (normalized.length > maximum) throw new Error(`${field} is too long`);
  if (/\r|\n|\0/u.test(normalized)) throw new Error(`${field} contains unsupported characters`);
  return normalized;
}

export function loopbackHost(hostname) {
  return ["localhost", "127.0.0.1", "::1", "[::1]"].includes(hostname.toLowerCase());
}

export function normalizeProviderBaseUrl(value, field = "provider base URL") {
  const normalized = boundedText(value, field, { maximum: 2_048 });
  let parsed;
  try {
    parsed = new URL(normalized);
  } catch {
    throw new Error(`${field} must be a valid URL`);
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error(`${field} must not contain credentials, query parameters, or fragments`);
  }
  if (parsed.protocol !== "https:" && !(parsed.protocol === "http:" && loopbackHost(parsed.hostname))) {
    throw new Error(`${field} must use HTTPS; exact loopback hosts may use HTTP`);
  }
  return parsed.toString().replace(/\/+$/u, "");
}

function normalizeSecret(value, field) {
  const normalized = String(value ?? "").trim();
  if (normalized.length > MAX_SECRET_LENGTH) throw new Error(`${field} is too long`);
  if (/\r|\n|\0/u.test(normalized)) throw new Error(`${field} contains unsupported characters`);
  return normalized;
}

function normalizeStage(stage, input, previous = {}) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new Error(`${stage} configuration must be an object`);
  }
  const provider = boundedText(input.provider, `${stage} provider`);
  if (!PROVIDER_KINDS.includes(provider)) throw new Error(`${stage} provider is unsupported`);
  const baseUrl = normalizeProviderBaseUrl(input.baseUrl, `${stage} base URL`);
  const model = boundedText(input.model, `${stage} model`);
  let reusablePreviousKey = "";
  try {
    const sameCredentialTarget = String(previous.provider ?? "").trim() === provider
      && normalizeProviderBaseUrl(previous.baseUrl, `${stage} previous base URL`) === baseUrl;
    if (sameCredentialTarget) reusablePreviousKey = String(previous.apiKey ?? "").trim();
  } catch {
    reusablePreviousKey = "";
  }
  const suppliedKey = input.clearApiKey === true
    ? ""
    : String(input.apiKey ?? "").trim() || reusablePreviousKey;
  const apiKey = normalizeSecret(suppliedKey, `${stage} API key`);
  return {
    provider,
    baseUrl,
    model,
    ...(apiKey ? { apiKey } : {}),
  };
}

export function normalizeProviderConfig(input, previous = undefined) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new Error("provider configuration must be an object");
  }
  const prior = previous && typeof previous === "object" && !Array.isArray(previous) ? previous : {};
  const writer = normalizeStage("writer", input.writer, prior.writer);
  const organizerInput = input.organizer;
  if (!organizerInput || typeof organizerInput !== "object" || Array.isArray(organizerInput)) {
    throw new Error("organizer configuration must be an object");
  }
  const inheritWriter = organizerInput.inheritWriter !== false;
  const organizer = inheritWriter
    ? { inheritWriter: true }
    : { inheritWriter: false, ...normalizeStage("organizer", organizerInput, prior.organizer) };
  return {
    schemaVersion: PROVIDER_CONFIG_SCHEMA_VERSION,
    execution: "local",
    writer,
    organizer,
    updatedAt: new Date().toISOString(),
  };
}

function assertStoredConfig(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("local provider configuration must contain a JSON object");
  }
  if (value.schemaVersion !== PROVIDER_CONFIG_SCHEMA_VERSION || value.execution !== "local") {
    throw new Error("local provider configuration version is unsupported");
  }
  const normalized = normalizeProviderConfig(value, value);
  const updatedAt = boundedText(value.updatedAt, "updatedAt", { maximum: 64 });
  if (Number.isNaN(Date.parse(updatedAt))) throw new Error("updatedAt must be an ISO timestamp");
  normalized.updatedAt = updatedAt;
  return normalized;
}

export async function readProviderConfig(path = resolveProviderConfigPath()) {
  if (!existsSync(path)) return null;
  const parsed = JSON.parse(await readFile(path, "utf8"));
  return assertStoredConfig(parsed);
}

export function publicProviderConfig(value) {
  if (!value) {
    return {
      schemaVersion: PROVIDER_CONFIG_SCHEMA_VERSION,
      execution: "local",
      configured: false,
      writer: null,
      organizer: { inheritWriter: true },
    };
  }
  const publicStage = (stage) => ({
    provider: stage.provider,
    baseUrl: stage.baseUrl,
    model: stage.model,
    credentialPresent: Boolean(stage.apiKey),
  });
  return {
    schemaVersion: PROVIDER_CONFIG_SCHEMA_VERSION,
    execution: "local",
    configured: true,
    writer: publicStage(value.writer),
    organizer: value.organizer.inheritWriter
      ? { inheritWriter: true, credentialPresent: Boolean(value.writer.apiKey) }
      : { inheritWriter: false, ...publicStage(value.organizer) },
    updatedAt: value.updatedAt,
  };
}

export async function writeProviderConfig(input, path = resolveProviderConfigPath()) {
  const previous = await readProviderConfig(path).catch((error) => {
    if (error?.code === "ENOENT") return null;
    throw error;
  });
  const value = normalizeProviderConfig(input, previous ?? undefined);
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  const temporary = join(dirname(path), `.${randomUUID().replaceAll("-", "")}.tmp`);
  try {
    await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o600,
      flag: "wx",
    });
    await protectCredentialFile(temporary);
    await rename(temporary, path);
    await protectCredentialFile(path);
  } finally {
    await rm(temporary, { force: true }).catch(() => undefined);
  }
  return publicProviderConfig(value);
}

export async function clearProviderCredential(stage, path = resolveProviderConfigPath()) {
  if (!PROVIDER_STAGES.includes(stage)) throw new Error("provider stage is unsupported");
  const current = await readProviderConfig(path);
  if (!current) return publicProviderConfig(null);
  if (stage === "writer") delete current.writer.apiKey;
  else if (!current.organizer.inheritWriter) delete current.organizer.apiKey;
  current.updatedAt = new Date().toISOString();
  return writeProviderConfig({
    ...current,
    writer: { ...current.writer, apiKey: current.writer.apiKey ?? "", clearApiKey: !current.writer.apiKey },
    organizer: current.organizer.inheritWriter
      ? { inheritWriter: true }
      : {
          ...current.organizer,
          apiKey: current.organizer.apiKey ?? "",
          clearApiKey: !current.organizer.apiKey,
        },
  }, path);
}

export function resolvedProviderStage(config, stage) {
  if (stage === "writer") return config.writer;
  if (stage === "organizer") {
    return config.organizer.inheritWriter ? config.writer : config.organizer;
  }
  throw new Error("provider stage is unsupported");
}

export function providerStageReady(config, stage) {
  const target = resolvedProviderStage(config, stage);
  return Boolean(target && (target.apiKey || loopbackHost(new URL(target.baseUrl).hostname)));
}

export async function probeProvider(stage, input, {
  path = resolveProviderConfigPath(),
  fetchImpl = fetch,
  timeoutMs = 15_000,
  mode = "models",
} = {}) {
  if (!["models", "inference"].includes(mode)) throw new Error("Unknown provider test mode");
  const previous = await readProviderConfig(path);
  const normalized = normalizeProviderConfig(input, previous ?? undefined);
  const target = resolvedProviderStage(normalized, stage);
  if (!target.apiKey && !loopbackHost(new URL(target.baseUrl).hostname)) {
    throw new Error(`${stage} API key is required for a remote provider`);
  }
  const url = `${target.baseUrl}/${mode === "inference" ? "chat/completions" : "models"}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  const started = performance.now();
  try {
    const response = await fetchImpl(url, {
      method: mode === "inference" ? "POST" : "GET",
      headers: {
        Accept: "application/json",
        ...(mode === "inference" ? { "Content-Type": "application/json" } : {}),
        ...(target.apiKey ? { Authorization: `Bearer ${target.apiKey}` } : {}),
      },
      redirect: "error",
      signal: controller.signal,
      ...(mode === "inference" ? { body: JSON.stringify({ model: target.model,
        messages: [{ role: "user", content: `Synthetic TMCRA ${stage} connectivity test. Reply with the JSON object {"ok":true,"stage":"${stage}"} and nothing else.` }],
        max_tokens: 2048, temperature: 0, response_format: { type: "json_object" },
        ...(target.provider === "deepseek" ? { thinking: { type: "disabled" }, enable_thinking: false } : {}),
      }) } : {}),
    });
    const text = await response.text();
    if (!response.ok) throw new Error(`provider returned HTTP ${response.status}`);
    let payload;
    try {
      payload = text ? JSON.parse(text) : {};
    } catch {
      throw new Error("provider returned a non-JSON model response");
    }
    const modelIds = Array.isArray(payload?.data)
      ? payload.data.map((item) => String(item?.id ?? "")).filter(Boolean)
      : [];
    if (mode === "inference") {
      let answer;
      try { answer = JSON.parse(payload?.choices?.[0]?.message?.content || ""); } catch { throw new Error("模型响应未通过 JSON 结构校验，请检查模型和输出参数。"); }
      if (answer.ok !== true || answer.stage !== stage || payload?.choices?.[0]?.finish_reason !== "stop") throw new Error("模型响应不完整或未通过测试样本校验。");
    }
    return {
      ok: true,
      stage,
      endpoint: new URL(target.baseUrl).origin,
      model: target.model,
      testMode: mode,
      ...(mode === "inference" ? { servedModel: String(payload.model || target.model).slice(0, 512), inferenceValidated: true, syntheticDataOnly: true } : {}),
      modelVisible: modelIds.length === 0 ? null : modelIds.includes(target.model),
      latencyMs: Math.max(0, Math.round(performance.now() - started)),
    };
  } catch (error) {
    if (error?.name === "AbortError") throw new Error("provider connection timed out");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}
