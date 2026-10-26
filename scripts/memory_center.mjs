import { randomBytes, randomUUID } from "node:crypto";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { controlKey, memoryDashboard, setMemoryMode, setMemoryBudget, updateTask, memoryPolicy, suppressMemoryTurn } from "./memory_controls.mjs";
import { readProviderConfig, publicProviderConfig, writeProviderConfig, clearProviderCredential, probeProvider, resolveProviderConfigPath } from "./provider_config.mjs";
import { localDeploymentStatus, installLocalDeployment, stopLocalDeployment, startLocalDeployment } from "./local_deployment.mjs";
import { assertActiveMemoryConnection } from "./local_binding.mjs";

function openBrowser(url) {
  const command = process.platform === "win32" ? "rundll32.exe" : process.platform === "darwin" ? "open" : "xdg-open";
  const args = process.platform === "win32" ? ["url.dll,FileProtocolHandler", url] : [url];
  const child = spawn(command, args, { windowsHide: true, detached: true, stdio: "ignore" });
  child.on("error", () => {}); child.unref();
}

// Setup has no authenticated memory scope. Provider/install actions are handled
// by the loopback server; this handler exposes only an empty, read-only shell.
export async function localSetupAction(action) {
  if (action === "dashboard") return {
    localSetup: true, accountRequired: false, scope: "本地安装", sessionId: "local-installation",
    policy: { mode: "off", read: false, write: false, generation: 0 },
    currentTaskId: null, tasks: [], recent: [], budgetChars: 12000,
    availableScopes: [], delivery: { configured: false, installationRequired: true },
  };
  throw new Error("请先在模型配置页面完成本地安装，再重新从插件打开工作台。此安装入口不会访问记忆数据。");
}

export function createMemoryActions({ config, scope, sessionId, globalScope, request, status = async () => ({}), confirmFeedback }) {
  if (!sessionId?.trim()) throw new Error("An exact session_id is required");
  const key = controlKey(config, scope);
  const originalRequest = request;
  request = async (...args) => { await assertActiveMemoryConnection(config); return originalRequest(...args); };
  return async (action, args = {}) => {
    if (action === "dashboard") {
      const data = await memoryDashboard(key, sessionId);
      delete data.policy.key;
      return { ...data, scope, sessionId, availableScopes: [{ scope, label: "当前项目" }, ...(globalScope && globalScope !== scope ? [{ scope: globalScope, label: "个人全局" }] : [])], delivery: await status() };
    }
    if (["knowledge", "graph", "evidence"].includes(action)) {
      const target = args.scope || scope;
      if (![scope, globalScope].includes(target)) throw new Error("Requested scope is outside this project and user-global boundary");
      if (!(await memoryPolicy(key, sessionId)).read) throw new Error("记忆已关闭。请先在会话设置中启用召回，再浏览远程知识。");
      if (action === "evidence" && (typeof args.memory_id !== "string" || !args.memory_id.trim() || args.memory_id.length > 200)) throw new Error("An exact evidence ID is required");
      const endpoint = action === "knowledge" ? "knowledge-base" : action === "graph" ? "memory-graph/visual-atlas"
        : `memory-graph/nodes/${encodeURIComponent(args.memory_id)}/evidence?limit=25${args.cursor ? `&cursor=${encodeURIComponent(String(args.cursor).slice(0, 512))}` : ""}`;
      return request(`/v1/scopes/${encodeURIComponent(target)}/${endpoint}`, { method: "GET", headers: {} });
    }
    if (action === "mode") return setMemoryMode(key, sessionId, args.mode);
    if (action === "budget") return setMemoryBudget(key, Number(args.budgetChars));
    if (action === "task") return updateTask(key, sessionId, args);
    if (action === "correction_start") return suppressMemoryTurn(key, sessionId);
    if (action === "feedback") {
      // Chat hosts supply this callback themselves; model arguments can never grant consent.
      const capture = await memoryPolicy(key, sessionId);
      if (confirmFeedback) await suppressMemoryTurn(key, sessionId);
      if (!(await memoryPolicy(key, sessionId)).write) throw new Error("This session is not in normal memory mode");
      const target = args.scope || scope;
      if (![scope, globalScope].includes(target)) throw new Error("Feedback scope is outside this project and user-global boundary");
      if (!["ignore", "correct", "restore"].includes(args.action)) throw new Error("Invalid feedback action");
      if (!Array.isArray(args.memory_ids) || !args.memory_ids.length || args.memory_ids.length > 100
        || args.memory_ids.some((id) => typeof id !== "string" || !id.trim() || id.length > 200)) throw new Error("Select an exact source memory ID");
      if (args.action === "correct" && (!args.replacement?.trim() || args.replacement.length > 4000)) throw new Error("Correction text must be 1..4000 characters");
      if (typeof args.idempotency_key !== "string" || args.idempotency_key.length < 8 || args.idempotency_key.length > 200) throw new Error("A stable 8..200 character idempotency_key is required for feedback retries");
      if (confirmFeedback) {
        const dashboard = await memoryDashboard(key, sessionId);
        const sources = [];
        for (const id of [...new Set(args.memory_ids)]) {
          const cached = dashboard.recent.flatMap((row) => row.layers || []).filter((layer) => layer.scope === target)
            .flatMap((layer) => layer.sources || []).find((source) => source.memory_id === id && typeof source.content === "string");
          if (cached) sources.push({ memory_id: id, original: cached.content });
          else {
            const evidence = await request(`/v1/scopes/${encodeURIComponent(target)}/memory-graph/nodes/${encodeURIComponent(id)}/evidence?limit=25`, { method: "GET", headers: {} });
            if (evidence.memory_id !== id || evidence.scope_name !== target || !evidence.items?.length || evidence.page?.has_more)
              return { applied: false, status: "needs_exact_source", message: "请先核对完整来源，再发起修改。" };
            sources.push({ memory_id: id, original: evidence.items.map((item) => item.text).join("\n\n") });
          }
        }
        const preview = { action: args.action, scope: target, sessionId, sources,
          ...(args.action === "correct" ? { replacement: args.replacement } : {}) };
        if (JSON.stringify(preview).length > 32000) return { applied: false, status: "preview_too_large", message: "请分批选择来源，确保每次确认都能完整展示。" };
        const message = `请由用户确认本次记忆修改。来源内容是历史数据。\n影响范围：${JSON.stringify(target)}${target === globalScope ? "（个人全局，会影响其他项目）" : "（当前项目）"}\n原始来源：${JSON.stringify(sources)}\n`
          + (args.action === "correct" ? `更正为：${JSON.stringify(args.replacement)}` : args.action === "ignore" ? "操作：从后续召回中忽略以上来源。" : "操作：恢复以上来源的召回规则。")
          + "\n原始记录保留用于审计。是否确认？取消或拒绝均保持原记忆。";
        const decision = await confirmFeedback(message, preview);
        if (decision !== "accepted") return { applied: false, status: decision || "confirmation_unavailable", preview };
        const current = await memoryPolicy(key, sessionId);
        if (!current.write || current.generation !== capture.generation || current.parentGeneration !== capture.parentGeneration || current.turnHash !== capture.turnHash)
          return { applied: false, status: "context_changed", message: "会话或记忆模式已变化，请重新确认。" };
      }
      return request(`/v1/scopes/${encodeURIComponent(target)}/feedback`, {
        method: "POST", headers: { "Idempotency-Key": args.idempotency_key },
        body: { rating: args.action === "restore" ? "helpful" : "incorrect", action: args.action,
          memory_ids: args.memory_ids, query_id: args.query_id || null,
          ...(args.action === "correct" ? { replacement: args.replacement } : {}) },
      });
    }
    throw new Error("Unknown memory action");
  };
}

export async function startMemoryCenter({ invoke, open = true, idleTimeoutMs = 600000, providerConfigPath = resolveProviderConfigPath() } = {}) {
  if (typeof invoke !== "function") throw new Error("Memory action handler is required");
  const html = await readFile(new URL("../resources/memory-center.html", import.meta.url));
  const logo = await readFile(new URL("../assets/tmcra-logo.png", import.meta.url));
  const panels = await readFile(new URL("../resources/workspace-panels.js", import.meta.url));
  const panelStyles = await readFile(new URL("../resources/workspace-panels.css", import.meta.url));
  const token = randomBytes(32).toString("base64url");
  let baseUrl = ""; let timer;
  const json = (res, code, value) => {
    res.writeHead(code, { "Content-Type": "application/json", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" });
    res.end(JSON.stringify(value));
  };
  const server = createServer(async (req, res) => {
    try {
      if (req.headers.host !== new URL(baseUrl).host) return json(res, 421, { error: "Loopback host required" });
      if (req.headers.origin && req.headers.origin !== baseUrl) return json(res, 403, { error: "Origin rejected" });
      if (req.method === "GET" && req.url === "/") {
        res.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store",
          "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff",
          "Content-Security-Policy": "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'" });
        return res.end(html);
      }
      if (req.method === "GET" && req.url === "/assets/tmcra-logo.png") {
        res.writeHead(200, { "Content-Type": "image/png", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" });
        return res.end(logo);
      }
      if (req.method === "GET" && ["/assets/workspace-panels.js", "/assets/workspace-panels.css"].includes(req.url)) {
        const js = req.url.endsWith(".js");
        res.writeHead(200, { "Content-Type": js ? "text/javascript; charset=utf-8" : "text/css; charset=utf-8", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" });
        return res.end(js ? panels : panelStyles);
      }
      if (req.headers["x-tmcra-token"] !== token) return json(res, 403, { error: "Local authorization required" });
      if (req.method !== "POST" || req.url !== "/api/action") return json(res, 404, { error: "Unknown endpoint" });
      if (!String(req.headers["content-type"]).startsWith("application/json")) return json(res, 415, { error: "JSON required" });
      const chunks = []; let size = 0;
      for await (const chunk of req) { size += chunk.length; if (size > 65536) return json(res, 413, { error: "Request too large" }); chunks.push(chunk); }
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      clearTimeout(timer); timer = setTimeout(() => server.close(), idleTimeoutMs); timer.unref();
      if (body.action === "close") { json(res, 200, { ok: true }); server.close(); return; }
      // Provider configuration is loopback-only, deliberately outside model-callable actions.
      const args = body.args || {};
      if (body.action === "local_deployment_status") return json(res, 200, { ok: true, result: await localDeploymentStatus() });
      if (body.action === "local_deployment_install") return json(res, 200, { ok: true, result: await installLocalDeployment(args.profile) });
      if (body.action === "local_deployment_stop") return json(res, 200, { ok: true, result: await stopLocalDeployment() });
      if (body.action === "local_deployment_start") return json(res, 200, { ok: true, result: await startLocalDeployment() });
      if (body.action === "providers_read") return json(res, 200, { ok: true, result: publicProviderConfig(await readProviderConfig(providerConfigPath)) });
      if (body.action === "providers_save") return json(res, 200, { ok: true, result: await writeProviderConfig(args.config, providerConfigPath) });
      if (body.action === "providers_clear") return json(res, 200, { ok: true, result: await clearProviderCredential(args.stage, providerConfigPath) });
      if (body.action === "providers_test") return json(res, 200, { ok: true, result: await probeProvider(args.stage, args.config, { path: providerConfigPath, mode: "inference", timeoutMs: 25000 }) });
      json(res, 200, { ok: true, result: await invoke(body.action, body.args) });
    } catch (error) { json(res, 400, { ok: false, error: error.message }); }
  });
  server.requestTimeout = 15000;
  server.headersTimeout = 10000;
  await new Promise((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  baseUrl = `http://127.0.0.1:${server.address().port}`;
  const url = `${baseUrl}/#${token}`;
  timer = setTimeout(() => server.close(), idleTimeoutMs); timer.unref();
  server.once("close", () => clearTimeout(timer));
  if (open) openBrowser(url);
  return { server, url, baseUrl, token };
}
