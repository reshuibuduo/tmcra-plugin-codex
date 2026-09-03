#!/usr/bin/env node

import { randomBytes } from "node:crypto";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";

import {
  clearProviderCredential,
  probeProvider,
  publicProviderConfig,
  readProviderConfig,
  resolveProviderConfigPath,
  writeProviderConfig,
} from "./provider_config.mjs";

const MAX_BODY_BYTES = 64 * 1024;
const DEFAULT_IDLE_TIMEOUT_MS = 10 * 60 * 1000;
const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const htmlPath = resolve(scriptDirectory, "..", "resources", "provider-settings.html");

function json(response, status, value) {
  const body = Buffer.from(`${JSON.stringify(value)}\n`, "utf8");
  response.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": body.length,
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
  });
  response.end(body);
}

async function bodyJson(request) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) throw Object.assign(new Error("request body is too large"), { status: 413 });
    chunks.push(chunk);
  }
  if (!chunks.length) return {};
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw Object.assign(new Error("request body must be valid JSON"), { status: 400 });
  }
}

function openBrowser(url) {
  let command;
  let args;
  if (process.platform === "win32") {
    command = "rundll32.exe";
    args = ["url.dll,FileProtocolHandler", url];
  } else if (process.platform === "darwin") {
    command = "open";
    args = [url];
  } else {
    command = "xdg-open";
    args = [url];
  }
  try {
    const child = spawn(command, args, { detached: true, stdio: "ignore", windowsHide: true });
    child.on("error", () => undefined);
    child.unref();
    return true;
  } catch {
    return false;
  }
}

export async function startProviderSetupServer({
  configPath = resolveProviderConfigPath(),
  host = "127.0.0.1",
  port = 0,
  idleTimeoutMs = DEFAULT_IDLE_TIMEOUT_MS,
  open = true,
} = {}) {
  if (host !== "127.0.0.1" && host !== "::1") throw new Error("provider setup must bind to loopback");
  const html = await readFile(htmlPath);
  const token = randomBytes(32).toString("base64url");
  let idleTimer;
  let baseUrl = "";
  const server = createServer(async (request, response) => {
    const resetIdle = () => {
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => server.close(), idleTimeoutMs);
      idleTimer.unref?.();
    };
    resetIdle();
    try {
      const hostHeader = String(request.headers.host || "");
      if (hostHeader !== new URL(baseUrl).host) {
        return json(response, 421, { ok: false, error: "loopback host required" });
      }
      const url = new URL(request.url || "/", baseUrl || "http://127.0.0.1");
      if (request.method === "GET" && url.pathname === "/") {
        response.writeHead(200, {
          "Content-Type": "text/html; charset=utf-8",
          "Content-Length": html.length,
          "Cache-Control": "no-store",
          "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
          "Referrer-Policy": "no-referrer",
          "X-Content-Type-Options": "nosniff",
          "X-Frame-Options": "DENY",
        });
        return response.end(html);
      }
      const suppliedToken = String(request.headers["x-tmcra-setup-token"] || "");
      if (suppliedToken !== token) return json(response, 403, { ok: false, error: "invalid setup session" });
      const origin = String(request.headers.origin || "");
      if (origin && origin !== baseUrl) return json(response, 403, { ok: false, error: "invalid request origin" });

      if (request.method === "GET" && url.pathname === "/api/config") {
        const current = await readProviderConfig(configPath);
        return json(response, 200, { ok: true, config: publicProviderConfig(current), configPath });
      }
      if (request.method === "PUT" && url.pathname === "/api/config") {
        const current = await writeProviderConfig(await bodyJson(request), configPath);
        return json(response, 200, { ok: true, config: current, configPath });
      }
      if (request.method === "POST" && url.pathname === "/api/test") {
        const value = await bodyJson(request);
        const result = await probeProvider(String(value.stage || ""), value.config, { path: configPath });
        return json(response, 200, result);
      }
      if (request.method === "DELETE" && url.pathname.startsWith("/api/credentials/")) {
        const stage = decodeURIComponent(url.pathname.slice("/api/credentials/".length));
        const current = await clearProviderCredential(stage, configPath);
        return json(response, 200, { ok: true, config: current });
      }
      if (request.method === "POST" && url.pathname === "/api/close") {
        json(response, 200, { ok: true });
        setImmediate(() => server.close());
        return;
      }
      return json(response, 404, { ok: false, error: "not found" });
    } catch (error) {
      return json(response, Number(error?.status || 422), {
        ok: false,
        error: error instanceof Error ? error.message : "provider setup failed",
      });
    }
  });

  await new Promise((resolvePromise, reject) => {
    server.once("error", reject);
    server.listen(port, host, resolvePromise);
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("provider setup did not obtain a TCP port");
  baseUrl = `http://${host === "::1" ? "[::1]" : host}:${address.port}`;
  const url = `${baseUrl}/#${token}`;
  idleTimer = setTimeout(() => server.close(), idleTimeoutMs);
  idleTimer.unref?.();
  const opened = open ? openBrowser(url) : false;
  return { server, url, token, baseUrl, configPath, opened, openPage: () => openBrowser(url) };
}

function option(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

export async function main() {
  const jsonOutput = process.argv.includes("--json");
  const result = await startProviderSetupServer({
    configPath: resolveProviderConfigPath(option("--config-file")),
    port: Number(option("--port") || 0),
    open: !process.argv.includes("--no-open"),
  });
  const status = { ok: true, url: result.url, configPath: result.configPath, browserOpened: result.opened };
  if (jsonOutput) process.stdout.write(`${JSON.stringify(status)}\n`);
  else {
    process.stdout.write(`TMCRA local model settings: ${result.url}\n`);
    process.stdout.write(`Credentials stay in ${result.configPath}. Close the page when finished.\n`);
  }
  await new Promise((resolvePromise) => result.server.once("close", resolvePromise));
}

const invokedPath = process.argv[1] ? pathToFileURL(process.argv[1]).href : "";
if (import.meta.url === invokedPath) {
  await main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : "TMCRA provider setup failed"}\n`);
    process.exitCode = 1;
  });
}
