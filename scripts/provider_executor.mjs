import { createHash } from "node:crypto";

import { appendLog, loadConfig, request } from "./tmcra_client.mjs";
import {
  providerStageReady,
  readProviderConfig,
  resolvedProviderStage,
} from "./provider_config.mjs";

const TASK_SCHEMA_VERSION = "tmcra.user-provider-task.1";
const REQUEST_SCHEMA_VERSION = "tmcra.openai-compatible-request.1";
const STAGES = Object.freeze(["writer", "organizer"]);
const MAX_PROVIDER_RESPONSE_BYTES = 8 * 1024 * 1024;
const MAX_TASK_OUTPUT_BYTES = 4 * 1024 * 1024;
const DEFAULT_IDLE_MS = 1_000;
const DEFAULT_PROVIDER_TIMEOUT_MS = 180_000;

function delay(milliseconds, signal) {
  if (signal?.aborted) return Promise.resolve();
  return new Promise((resolvePromise) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener("abort", finish);
      resolvePromise();
    };
    const timer = setTimeout(finish, milliseconds);
    if (typeof timer.unref === "function") timer.unref();
    signal?.addEventListener("abort", finish, { once: true });
  });
}

function boundedString(value, label, maximum) {
  const text = String(value ?? "").trim();
  if (!text || text.length > maximum || /[\r\n\0]/u.test(text)) {
    throw new Error(`${label} is invalid`);
  }
  return text;
}

function validateTask(task, expectedStage) {
  if (!task || typeof task !== "object" || Array.isArray(task)) {
    throw new Error("provider task must be an object");
  }
  if (task.schema_version !== TASK_SCHEMA_VERSION || task.stage !== expectedStage) {
    throw new Error("provider task contract is unsupported");
  }
  const taskId = boundedString(task.task_id, "provider task ID", 200);
  const leaseToken = boundedString(task.lease_token, "provider task lease", 256);
  if (leaseToken.length < 32) throw new Error("provider task lease is invalid");
  const requestSha256 = boundedString(task.request_sha256, "provider request digest", 64);
  if (!/^[0-9a-f]{64}$/u.test(requestSha256)) {
    throw new Error("provider request digest is invalid");
  }
  const modelRequest = task.request;
  if (
    !modelRequest ||
    typeof modelRequest !== "object" ||
    Array.isArray(modelRequest) ||
    modelRequest.schema_version !== REQUEST_SCHEMA_VERSION ||
    !Array.isArray(modelRequest.messages) ||
    modelRequest.messages.length < 2 ||
    modelRequest.messages.length > 64
  ) {
    throw new Error("provider model request is invalid");
  }
  const allowed = new Set([
    "schema_version",
    "messages",
    "temperature",
    "max_tokens",
    "response_format",
  ]);
  if (Object.keys(modelRequest).some((key) => !allowed.has(key))) {
    throw new Error("provider model request contains unsupported fields");
  }
  const messages = modelRequest.messages.map((message) => {
    if (!message || typeof message !== "object" || Array.isArray(message)) {
      throw new Error("provider message is invalid");
    }
    const role = String(message.role || "");
    if (!new Set(["system", "user", "assistant"]).has(role)) {
      throw new Error("provider message role is invalid");
    }
    if (typeof message.content !== "string" || message.content.length > 8_000_000) {
      throw new Error("provider message content is invalid");
    }
    return { role, content: message.content };
  });
  const maxTokens = Number(modelRequest.max_tokens);
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 131_072) {
    throw new Error("provider max_tokens is invalid");
  }
  if (modelRequest.temperature !== 0) {
    throw new Error("provider temperature contract is invalid");
  }
  const responseFormat = modelRequest.response_format;
  if (
    !responseFormat ||
    typeof responseFormat !== "object" ||
    !["json_object", "json_schema"].includes(String(responseFormat.type || ""))
  ) {
    throw new Error("provider response format is invalid");
  }
  const leaseExpiresAt = Number(task.lease_expires_at);
  if (!Number.isFinite(leaseExpiresAt) || leaseExpiresAt < 0) {
    throw new Error("provider task lease expiry is invalid");
  }
  return {
    taskId,
    leaseToken,
    requestSha256,
    operation: boundedString(task.operation, "provider task operation", 80),
    request: {
      messages,
      temperature: 0,
      max_tokens: maxTokens,
      response_format: responseFormat,
    },
  };
}

function normalizeUsage(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const count = (...names) => {
    const found = names.map((name) => value[name]).find((item) => item !== undefined);
    if (found === undefined) return null;
    const number = Number(found);
    return Number.isSafeInteger(number) && number >= 0 ? number : null;
  };
  const input = count("prompt_tokens", "input_tokens");
  const output = count("completion_tokens", "output_tokens");
  if (input === null || output === null) return null;
  const details = value.prompt_tokens_details;
  const cachedFromDetails = details && typeof details === "object" && !Array.isArray(details)
    ? (() => {
        const number = Number(details.cached_tokens);
        return Number.isSafeInteger(number) && number >= 0 ? number : null;
      })()
    : null;
  const hit = count("prompt_cache_hit_tokens", "cache_read_input_tokens", "cached_tokens")
    ?? cachedFromDetails;
  const miss = count("prompt_cache_miss_tokens", "cache_miss_input_tokens");
  const normalizedHit = hit === null ? (miss === null ? 0 : input - miss) : hit;
  const normalizedMiss = miss === null ? input - normalizedHit : miss;
  if (
    normalizedHit < 0 ||
    normalizedMiss < 0 ||
    normalizedHit + normalizedMiss !== input
  ) return null;
  const total = count("total_tokens") ?? input + output;
  if (total < input + output) return null;
  return {
    input_tokens: input,
    output_tokens: output,
    total_tokens: total,
    cache_hit_tokens: normalizedHit,
    cache_miss_tokens: normalizedMiss,
  };
}

class ProviderExecutionError extends Error {
  constructor(message, { code, outcome = "failed", providerRequestId = null } = {}) {
    super(message);
    this.code = code || "provider_execution_failed";
    this.outcome = outcome;
    this.providerRequestId = providerRequestId;
  }
}

async function boundedResponseText(response) {
  const reader = response.body?.getReader?.();
  if (!reader) {
    const text = await response.text();
    if (Buffer.byteLength(text, "utf8") > MAX_PROVIDER_RESPONSE_BYTES) {
      throw new ProviderExecutionError("provider response is too large", {
        code: "provider_response_too_large",
      });
    }
    return text;
  }
  const chunks = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = Buffer.from(value);
    size += chunk.byteLength;
    if (size > MAX_PROVIDER_RESPONSE_BYTES) {
      await reader.cancel().catch(() => undefined);
      throw new ProviderExecutionError("provider response is too large", {
        code: "provider_response_too_large",
      });
    }
    chunks.push(chunk);
  }
  return Buffer.concat(chunks, size).toString("utf8");
}

async function providerCompletion(target, task, { fetchImpl = fetch, signal } = {}) {
  const configuredTimeout = Number(
    process.env.TMCRA_LOCAL_PROVIDER_TIMEOUT_MS || DEFAULT_PROVIDER_TIMEOUT_MS,
  );
  const timeoutMs = Number.isFinite(configuredTimeout)
    ? Math.max(1_000, Math.min(15 * 60 * 1_000, configuredTimeout))
    : DEFAULT_PROVIDER_TIMEOUT_MS;
  const controller = new AbortController();
  const abortFromParent = () => controller.abort(signal?.reason);
  if (signal?.aborted) controller.abort(signal.reason);
  else signal?.addEventListener("abort", abortFromParent, { once: true });
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const body = { ...task.request, model: target.model };
    if (target.provider === "deepseek") {
      if (body.response_format?.type === "json_schema") {
        body.response_format = { type: "json_object" };
      }
      body.thinking = { type: "disabled" };
      body.enable_thinking = false;
    }
    let response;
    try {
      response = await fetchImpl(`${target.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          ...(target.apiKey ? { Authorization: `Bearer ${target.apiKey}` } : {}),
        },
        redirect: "error",
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (error) {
      throw new ProviderExecutionError("provider transport outcome is unresolved", {
        code: error?.name === "AbortError" ? "provider_timeout" : "provider_transport_error",
        outcome: "unknown",
      });
    }
    const declaredLength = Number(response.headers.get("content-length"));
    if (Number.isFinite(declaredLength) && declaredLength > MAX_PROVIDER_RESPONSE_BYTES) {
      throw new ProviderExecutionError("provider response is too large", {
        code: "provider_response_too_large",
      });
    }
    let text;
    try {
      text = await boundedResponseText(response);
    } catch (error) {
      if (error instanceof ProviderExecutionError) throw error;
      throw new ProviderExecutionError("provider response outcome is unresolved", {
        code: error?.name === "AbortError" ? "provider_timeout" : "provider_response_error",
        outcome: "unknown",
      });
    }
    let payload;
    try {
      payload = text ? JSON.parse(text) : {};
    } catch {
      throw new ProviderExecutionError("provider returned non-JSON HTTP content", {
        code: "provider_invalid_http_json",
      });
    }
    const providerRequestId = typeof payload?.id === "string"
      ? payload.id.slice(0, 200)
      : null;
    if (!response.ok) {
      throw new ProviderExecutionError(`provider returned HTTP ${response.status}`, {
        code: `provider_http_${response.status}`,
        outcome: response.status >= 500 || [408, 425].includes(response.status)
          ? "unknown"
          : "failed",
        providerRequestId,
      });
    }
    const choices = payload?.choices;
    if (!Array.isArray(choices) || choices.length !== 1) {
      throw new ProviderExecutionError("provider response choice count is invalid", {
        code: "provider_invalid_choices",
        providerRequestId,
      });
    }
    const choice = choices[0];
    if (choice?.finish_reason !== "stop") {
      throw new ProviderExecutionError("provider response did not finish cleanly", {
        code: "provider_incomplete_response",
        providerRequestId,
      });
    }
    const content = choice?.message?.content;
    let output;
    try {
      output = typeof content === "string" ? JSON.parse(content) : content;
    } catch {
      throw new ProviderExecutionError("provider completion is not valid JSON", {
        code: "provider_invalid_completion_json",
        providerRequestId,
      });
    }
    if (!output || typeof output !== "object" || Array.isArray(output)) {
      throw new ProviderExecutionError("provider completion must be one JSON object", {
        code: "provider_completion_not_object",
        providerRequestId,
      });
    }
    const serialized = JSON.stringify(output);
    if (Buffer.byteLength(serialized, "utf8") > MAX_TASK_OUTPUT_BYTES) {
      throw new ProviderExecutionError("provider completion is too large", {
        code: "provider_completion_too_large",
        providerRequestId,
      });
    }
    return {
      output,
      usage: normalizeUsage(payload.usage),
      providerRequestId,
      responseSha256: createHash("sha256").update(serialized).digest("hex"),
    };
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", abortFromParent);
  }
}

async function taskRequest(path, body, config, attempts = 2) {
  return request(path, { method: "POST", body, config, attempts });
}

async function executeTask(stage, rawTask, target, config, options = {}) {
  const task = validateTask(rawTask, stage);
  await taskRequest(
    `/v1/provider-tasks/${encodeURIComponent(task.taskId)}/started`,
    { lease_token: task.leaseToken },
    config,
  );
  let heartbeatInFlight = false;
  const heartbeat = setInterval(() => {
    if (heartbeatInFlight) return;
    heartbeatInFlight = true;
    void taskRequest(
      `/v1/provider-tasks/${encodeURIComponent(task.taskId)}/heartbeat`,
      { lease_token: task.leaseToken },
      config,
      1,
    ).catch(() => undefined).finally(() => { heartbeatInFlight = false; });
  }, 30_000);
  if (typeof heartbeat.unref === "function") heartbeat.unref();
  try {
    const result = await providerCompletion(target, task, options);
    await taskRequest(
      `/v1/provider-tasks/${encodeURIComponent(task.taskId)}/complete`,
      {
        lease_token: task.leaseToken,
        provider: target.provider,
        model: target.model,
        output: result.output,
        usage: result.usage,
        provider_request_id: result.providerRequestId,
      },
      config,
      3,
    );
    await appendLog("local_provider_task_completed", {
      taskId: task.taskId,
      stage,
      operation: task.operation,
      provider: target.provider,
      model: target.model,
      requestSha256: task.requestSha256,
      responseSha256: result.responseSha256,
    });
    return { taskId: task.taskId, state: "completed" };
  } catch (error) {
    if (!(error instanceof ProviderExecutionError)) throw error;
    await taskRequest(
      `/v1/provider-tasks/${encodeURIComponent(task.taskId)}/fail`,
      {
        lease_token: task.leaseToken,
        provider: target.provider,
        model: target.model,
        outcome: error.outcome,
        error_code: error.code,
      },
      config,
      3,
    );
    await appendLog("local_provider_task_failed", {
      taskId: task.taskId,
      stage,
      operation: task.operation,
      provider: target.provider,
      model: target.model,
      outcome: error.outcome,
      code: error.code,
    });
    return { taskId: task.taskId, state: error.outcome };
  } finally {
    clearInterval(heartbeat);
  }
}

export async function executeAvailableProviderTasks({
  config,
  providerConfig,
  maxTasks = 4,
  fetchImpl = fetch,
  signal,
} = {}) {
  const local = providerConfig === undefined
    ? await readProviderConfig()
    : providerConfig;
  if (!local) return { executed: 0 };
  const serviceConfig = config || await loadConfig();
  let executed = 0;
  let stageCursor = 0;
  for (let index = 0; index < maxTasks; index += 1) {
    if (signal?.aborted) break;
    let claimed = false;
    for (let offset = 0; offset < STAGES.length; offset += 1) {
      const stageIndex = (stageCursor + offset) % STAGES.length;
      const stage = STAGES[stageIndex];
      if (!providerStageReady(local, stage)) continue;
      const response = await taskRequest(
        "/v1/provider-tasks/claim",
        { stage },
        serviceConfig,
        1,
      );
      if (!response?.task) continue;
      claimed = true;
      stageCursor = (stageIndex + 1) % STAGES.length;
      await executeTask(
        stage,
        response.task,
        resolvedProviderStage(local, stage),
        serviceConfig,
        { fetchImpl, signal },
      );
      executed += 1;
      break;
    }
    if (!claimed) break;
  }
  return { executed };
}

export async function runProviderExecutor({ signal, idleMs = DEFAULT_IDLE_MS } = {}) {
  let failureDelay = idleMs;
  let lastFailure = "";
  while (!signal?.aborted) {
    try {
      const result = await executeAvailableProviderTasks({ maxTasks: 8, signal });
      failureDelay = idleMs;
      lastFailure = "";
      if (result.executed > 0) continue;
    } catch (error) {
      if (signal?.aborted) break;
      const fingerprint = `${error?.code || "error"}:${error?.status || ""}`;
      if (fingerprint !== lastFailure) {
        await appendLog("local_provider_executor_waiting", {
          code: error?.code || null,
          status: error?.status || null,
        });
        lastFailure = fingerprint;
      }
      await delay(failureDelay, signal);
      failureDelay = Math.min(30_000, Math.max(idleMs, failureDelay * 2));
      continue;
    }
    await delay(idleMs, signal);
  }
}
