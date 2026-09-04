import { createHash, randomBytes, randomUUID } from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { chmod, mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { homedir, platform } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_AUTH_BASE_URL = "https://tmcra.com";
const CLIENT_ID = "tmcra-codex";
const REQUEST_TIMEOUT_MS = 30_000;

function option(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : null;
}

const jsonOutput = process.argv.includes("--json");
const progressJson = process.argv.includes("--progress-json");
const noOpen = process.argv.includes("--no-open") || process.env.TMCRA_DEVICE_NO_OPEN === "1";
const configPath = resolve(
  option("--config") ||
    process.env.TMCRA_CONFIG_FILE ||
    join(homedir(), ".config", "tmcra", "config.json"),
);
const installationPath = resolve(
  option("--installation-file") ||
    process.env.TMCRA_INSTALLATION_FILE ||
    join(dirname(configPath), "installation.json"),
);
const authBaseUrl = String(
  option("--auth-base-url") || process.env.TMCRA_AUTH_BASE_URL || DEFAULT_AUTH_BASE_URL,
).replace(/\/+$/u, "");

function emitProgress(event, details = {}) {
  if (!progressJson) return;
  process.stderr.write(`${JSON.stringify({ event, ...details })}\n`);
}

function writeHumanProgress(message) {
  process.stderr.write(message);
}

function assertWebUrl(value, label) {
  const url = new URL(value);
  const local = ["localhost", "127.0.0.1", "::1"].includes(url.hostname);
  if (url.protocol !== "https:" && !(url.protocol === "http:" && local)) {
    throw new Error(`${label} must use HTTPS (or localhost for development)`);
  }
  if (url.username || url.password) {
    throw new Error(`${label} must not contain embedded credentials`);
  }
  return url.toString().replace(/\/$/u, "");
}

async function atomicJson(path, value) {
  const directory = dirname(path);
  await mkdir(directory, { recursive: true, mode: 0o700 });
  await protectLocalCredentialPath(directory, true);
  const temporary = `${path}.${process.pid}.${randomUUID()}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, {
    encoding: "utf8",
    mode: 0o600,
  });
  await protectLocalCredentialPath(temporary, false);
  await rename(temporary, path);
  await protectLocalCredentialPath(path, false);
}

async function protectLocalCredentialPath(path, directory) {
  if (process.platform !== "win32") {
    await chmod(path, directory ? 0o700 : 0o600);
    return;
  }
  const identity = spawnSync("whoami.exe", ["/user", "/fo", "csv", "/nh"], {
    encoding: "utf8",
    windowsHide: true,
  });
  const sid = String(identity.stdout || "").match(/S-1-5-(?:\d+-)+\d+/u)?.[0];
  if (identity.status !== 0 || !sid) {
    throw new Error("TMCRA could not determine the Windows account for credential protection.");
  }
  const permission = directory ? "(OI)(CI)F" : "F";
  const protectedAcl = spawnSync(
    "icacls.exe",
    [
      path,
      "/inheritance:r",
      "/grant:r",
      `*${sid}:${permission}`,
      "/grant:r",
      `*S-1-5-18:${permission}`,
    ],
    { encoding: "utf8", windowsHide: true, stdio: "ignore" },
  );
  if (protectedAcl.status !== 0) {
    throw new Error("TMCRA could not protect the local credential file.");
  }
}

async function installationId() {
  if (existsSync(installationPath)) {
    try {
      const value = JSON.parse(await readFile(installationPath, "utf8"));
      if (/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/iu.test(value.installationId)) {
        return value.installationId;
      }
    } catch {
      // Replace an unreadable installation marker with a new stable identifier.
    }
  }
  const value = randomUUID();
  await atomicJson(installationPath, {
    schemaVersion: 1,
    installationId: value,
    createdAt: new Date().toISOString(),
  });
  return value;
}

async function preservedIntegrationIds() {
  if (!existsSync(configPath)) return {};
  let value;
  try {
    value = JSON.parse(await readFile(configPath, "utf8"));
  } catch (error) {
    throw new Error("TMCRA device config is unreadable; refusing to discard ledger bindings.", {
      cause: error,
    });
  }
  const integrationIds = value?.integrationIds;
  if (integrationIds === undefined) return {};
  if (!integrationIds || typeof integrationIds !== "object" || Array.isArray(integrationIds)) {
    throw new Error("TMCRA device integration bindings are invalid.");
  }
  const allowed = new Set(["codex", "openclaw", "hermes", "claude_code", "mcp"]);
  const result = {};
  for (const [platformName, integrationId] of Object.entries(integrationIds)) {
    if (!allowed.has(platformName) || !/^int_[a-f0-9]{32}$/u.test(String(integrationId))) {
      throw new Error("TMCRA device integration bindings are invalid.");
    }
    result[platformName] = String(integrationId);
  }
  return result;
}

async function clientVersion() {
  const pluginRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  try {
    const manifest = JSON.parse(
      await readFile(join(pluginRoot, ".codex-plugin", "plugin.json"), "utf8"),
    );
    return String(manifest.version || "0.3.0-rc.10");
  } catch {
    return "0.3.0-rc.10";
  }
}

function pkcePair() {
  const verifier = randomBytes(48).toString("base64url");
  const challenge = createHash("sha256").update(verifier).digest("base64url");
  return { verifier, challenge };
}

async function postJson(url, body) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        "User-Agent": `tmcra-memory-plugin/${await clientVersion()}`,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await response.text();
    let payload = {};
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        throw new Error(`TMCRA authorization returned non-JSON HTTP ${response.status}`);
      }
    }
    return { response, payload };
  } finally {
    clearTimeout(timer);
  }
}

function errorCode(payload) {
  if (typeof payload?.error === "string") return payload.error;
  if (payload?.error && typeof payload.error === "object") return payload.error.code;
  return payload?.code || null;
}

function safeAuthorizationError(response, payload) {
  const code = String(errorCode(payload) || "authorization_failed").replace(
    /[^A-Za-z0-9_.-]/gu,
    "_",
  );
  const error = new Error(`TMCRA device authorization failed (${code}, HTTP ${response.status}).`);
  error.code = code;
  return error;
}

function openBrowser(url) {
  if (noOpen) return false;
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
    child.on("error", () => {});
    child.unref();
    return true;
  } catch {
    return false;
  }
}

function delay(milliseconds) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds));
}

const TRANSIENT_NETWORK_CODES = new Set([
  "ECONNABORTED",
  "ECONNREFUSED",
  "ECONNRESET",
  "EHOSTUNREACH",
  "ENETDOWN",
  "ENETUNREACH",
  "ENOTFOUND",
  "EPIPE",
  "ETIMEDOUT",
  "EAI_AGAIN",
  "UND_ERR_CONNECT_TIMEOUT",
  "UND_ERR_HEADERS_TIMEOUT",
  "UND_ERR_SOCKET",
]);

function transientNetworkError(error) {
  if (!error || typeof error !== "object") return false;
  if (error.name === "AbortError") return true;
  const code = String(error.code || error.cause?.code || "");
  if (TRANSIENT_NETWORK_CODES.has(code)) return true;
  return error.name === "TypeError" && /fetch failed/iu.test(String(error.message || ""));
}

async function postJsonWithTransientRetry(url, body, { attempts = 5 } = {}) {
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await postJson(url, body);
    } catch (error) {
      if (!transientNetworkError(error)) throw error;
      lastError = error;
      if (attempt === attempts - 1) break;
      writeHumanProgress("TMCRA authorization network retry in progress...\n");
      await delay(Math.min(8_000, 500 * 2 ** attempt));
    }
  }
  throw lastError;
}

async function acknowledgeDelivery(baseUrl, pending) {
  let acknowledgementError = null;
  for (let attempt = 0; attempt < 6; attempt += 1) {
    if (attempt > 0) await delay(Math.min(8_000, 500 * 2 ** (attempt - 1)));
    try {
      const result = await postJson(`${baseUrl}/api/device/v1/token`, {
        deviceCode: pending.deviceCode,
        codeVerifier: pending.codeVerifier,
        deliveryReceipt: pending.deliveryReceipt,
      });
      if (result.response.ok && result.payload?.claimed === true) return result.payload;
      acknowledgementError = safeAuthorizationError(result.response, result.payload);
      const code = errorCode(result.payload);
      if (
        result.response.status < 500 &&
        code !== "authorization_pending" &&
        code !== "slow_down"
      ) break;
    } catch (error) {
      acknowledgementError = error;
    }
  }
  const error = new Error(
    "TMCRA saved a short-lived provisional credential but could not confirm delivery. Run the installer again; it will retry the saved confirmation before starting a new authorization.",
    { cause: acknowledgementError },
  );
  error.code = "delivery_acknowledgement_failed";
  throw error;
}

async function recoverPendingDelivery(expectedBaseUrl) {
  if (!existsSync(configPath)) return null;
  let config;
  try {
    config = JSON.parse(await readFile(configPath, "utf8"));
  } catch {
    return null;
  }
  const pending = config?.pendingDelivery;
  if (!pending || typeof pending !== "object" || Array.isArray(pending)) return null;
  const valid =
    pending.authorizationBaseUrl === expectedBaseUrl &&
    typeof pending.deviceCode === "string" &&
    pending.deviceCode.length <= 512 &&
    typeof pending.codeVerifier === "string" &&
    /^[A-Za-z0-9._~-]{43,128}$/u.test(pending.codeVerifier) &&
    typeof pending.deliveryReceipt === "string" &&
    /^[A-Za-z0-9_-]{43}$/u.test(pending.deliveryReceipt) &&
    Number.isFinite(Date.parse(pending.expiresAt));
  if (!valid || Date.parse(pending.expiresAt) <= Date.now()) {
    delete config.pendingDelivery;
    await atomicJson(configPath, config);
    return null;
  }
  const acknowledgement = await acknowledgeDelivery(expectedBaseUrl, pending);
  const confirmedExpiresIn = Number(acknowledgement.expiresIn);
  if (
    !Number.isFinite(confirmedExpiresIn) ||
    confirmedExpiresIn <= 0 ||
    confirmedExpiresIn > 400 * 24 * 60 * 60
  ) {
    throw new Error("TMCRA token acknowledgement response is incomplete.");
  }
  config.expiresAt = new Date(Date.now() + confirmedExpiresIn * 1000).toISOString();
  delete config.pendingDelivery;
  await atomicJson(configPath, config);
  return config;
}

async function main() {
  const safeAuthBaseUrl = assertWebUrl(authBaseUrl, "TMCRA authorization URL");
  const parsedAuthBaseUrl = new URL(safeAuthBaseUrl);
  if (parsedAuthBaseUrl.search || parsedAuthBaseUrl.hash) {
    throw new Error("TMCRA authorization URL must not contain a query or fragment");
  }
  const recovered = await recoverPendingDelivery(safeAuthBaseUrl);
  if (recovered) {
    const result = {
      ok: true,
      configPath,
      authorizationBaseUrl: safeAuthBaseUrl,
      apiBaseUrl: recovered.baseUrl,
      scopeNamespace: recovered.scopeNamespace,
      expiresAt: recovered.expiresAt,
      credentialStored: true,
      deliveryRecovered: true,
      browserOpened: false,
    };
    emitProgress("tmcra.install.progress", {
      step: "authorize",
      status: "completed",
      message: "TMCRA authorization completed.",
    });
    if (jsonOutput) process.stdout.write(`${JSON.stringify(result)}\n`);
    else process.stdout.write(`TMCRA authorization recovered. Credential saved to ${configPath}.\n`);
    return;
  }
  const id = await installationId();
  const version = await clientVersion();
  const { verifier, challenge } = pkcePair();
  const created = await postJsonWithTransientRetry(`${safeAuthBaseUrl}/api/device/v1/authorizations`, {
    clientId: CLIENT_ID,
    installationId: id,
    clientVersion: version,
    platform: `${platform()}-${process.arch}`,
    codeChallenge: challenge,
    codeChallengeMethod: "S256",
  });
  if (!created.response.ok) throw safeAuthorizationError(created.response, created.payload);

  const deviceCode = String(created.payload.deviceCode || "");
  const userCode = String(created.payload.userCode || "");
  const verificationUrl = String(
    created.payload.verificationUriComplete || created.payload.verificationUri || "",
  );
  const expiresIn = Number(created.payload.expiresIn);
  let intervalSeconds = Number(created.payload.interval || 5);
  if (
    !deviceCode ||
    deviceCode.length > 512 ||
    /[\u0000-\u001f\u007f]/u.test(deviceCode) ||
    !/^[A-Za-z0-9-]{4,32}$/u.test(userCode) ||
    !verificationUrl ||
    verificationUrl.length > 2048 ||
    !Number.isFinite(expiresIn) ||
    expiresIn <= 0 ||
    expiresIn > 1800
  ) {
    throw new Error("TMCRA authorization response is incomplete.");
  }
  const safeVerificationUrl = assertWebUrl(verificationUrl, "TMCRA verification URL");
  if (new URL(safeVerificationUrl).origin !== parsedAuthBaseUrl.origin) {
    throw new Error("TMCRA verification URL must use the authorization service origin.");
  }
  const localAuthorizationService = ["localhost", "127.0.0.1", "::1"].includes(
    parsedAuthBaseUrl.hostname,
  );
  const minimumInterval = localAuthorizationService ? 0.05 : 1;
  if (
    !Number.isFinite(intervalSeconds) ||
    intervalSeconds < minimumInterval ||
    intervalSeconds > 60
  ) {
    intervalSeconds = 5;
  }

  const authorizationExpiresAt = new Date(Date.now() + expiresIn * 1000).toISOString();
  emitProgress("tmcra.authorization.required", {
    userCode,
    verificationUrl: safeVerificationUrl,
    expiresAt: authorizationExpiresAt,
  });
  writeHumanProgress(`TMCRA user code: ${userCode}\n`);
  writeHumanProgress(`Open this verification page: ${safeVerificationUrl}\n`);
  const opened = openBrowser(safeVerificationUrl);
  if (!opened && !noOpen) writeHumanProgress("The browser could not be opened automatically. Use the link above.\n");
  emitProgress("tmcra.install.progress", {
    step: "authorize",
    status: "running",
    message: "Waiting for approval in the TMCRA console.",
  });
  writeHumanProgress("Waiting for approval in the TMCRA console...\n");

  const deadline = Date.parse(authorizationExpiresAt);
  let tokenPayload = null;
  let transientFailures = 0;
  while (Date.now() < deadline) {
    await delay(Math.max(50, intervalSeconds * 1000));
    let token;
    try {
      token = await postJson(`${safeAuthBaseUrl}/api/device/v1/token`, {
        deviceCode,
        codeVerifier: verifier,
      });
      transientFailures = 0;
    } catch (error) {
      if (!transientNetworkError(error)) throw error;
      transientFailures += 1;
      intervalSeconds = Math.min(15, Math.max(intervalSeconds, 2 ** Math.min(transientFailures, 4)));
      writeHumanProgress("TMCRA authorization network retry in progress...\n");
      continue;
    }
    if (token.response.ok) {
      tokenPayload = token.payload;
      break;
    }
    const code = errorCode(token.payload);
    if (code === "authorization_pending") continue;
    if (code === "slow_down") {
      intervalSeconds += 5;
      continue;
    }
    if (code === "access_denied") throw new Error("TMCRA authorization was denied in the browser.");
    if (code === "expired_token") throw new Error("TMCRA authorization expired. Run the installer again.");
    throw safeAuthorizationError(token.response, token.payload);
  }
  if (!tokenPayload) throw new Error("TMCRA authorization expired. Run the installer again.");

  const accessToken = String(tokenPayload.accessToken || "");
  const deliveryReceipt = String(tokenPayload.deliveryReceipt || "");
  const tokenType = String(tokenPayload.tokenType || "Bearer");
  const tokenExpiresIn = Number(tokenPayload.expiresIn);
  const apiBaseUrl = assertWebUrl(String(tokenPayload.baseUrl || ""), "TMCRA API base URL");
  const scopeNamespace = String(tokenPayload.scopeNamespace || "").trim();
  if (
    !accessToken ||
    accessToken.length > 4096 ||
    /[\u0000-\u001f\u007f]/u.test(accessToken) ||
    tokenType.toLowerCase() !== "bearer" ||
    tokenPayload.deliveryAcknowledgementRequired !== true ||
    !/^[A-Za-z0-9_-]{43}$/u.test(deliveryReceipt) ||
    !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$/u.test(scopeNamespace) ||
    !Number.isFinite(tokenExpiresIn) ||
    tokenExpiresIn <= 0 ||
    tokenExpiresIn > 400 * 24 * 60 * 60
  ) {
    throw new Error("TMCRA token response is incomplete.");
  }
  let expiresAt = new Date(Date.now() + tokenExpiresIn * 1000).toISOString();
  const config = {
    schemaVersion: 2,
    authMode: "device",
    baseUrl: apiBaseUrl,
    accessToken,
    tokenType: "Bearer",
    expiresAt,
    scopeNamespace,
    globalScope: `${scopeNamespace}-global`,
    projectScopePrefix: `${scopeNamespace}-project`,
    timeoutMs: 120000,
    integrationIds: await preservedIntegrationIds(),
    pendingDelivery: {
      schemaVersion: 1,
      authorizationBaseUrl: safeAuthBaseUrl,
      deviceCode,
      codeVerifier: verifier,
      deliveryReceipt,
      expiresAt,
    },
  };
  // Persist the provisional credential before acknowledging delivery. If the
  // process stops after the acknowledgement request, the confirmed Token is
  // still recoverable from the user's protected config file.
  await atomicJson(configPath, config);

  const acknowledgement = await acknowledgeDelivery(safeAuthBaseUrl, config.pendingDelivery);
  const confirmedExpiresIn = Number(acknowledgement.expiresIn);
  if (
    !Number.isFinite(confirmedExpiresIn) ||
    confirmedExpiresIn <= 0 ||
    confirmedExpiresIn > 400 * 24 * 60 * 60
  ) {
    throw new Error("TMCRA token acknowledgement response is incomplete.");
  }
  expiresAt = new Date(Date.now() + confirmedExpiresIn * 1000).toISOString();
  config.expiresAt = expiresAt;
  delete config.pendingDelivery;
  await atomicJson(configPath, config);

  const result = {
    ok: true,
    configPath,
    authorizationBaseUrl: safeAuthBaseUrl,
    apiBaseUrl,
    scopeNamespace,
    expiresAt,
    credentialStored: true,
    browserOpened: opened,
  };
  emitProgress("tmcra.install.progress", {
    step: "authorize",
    status: "completed",
    message: "TMCRA authorization completed.",
  });
  if (jsonOutput) process.stdout.write(`${JSON.stringify(result)}\n`);
  else process.stdout.write(`TMCRA authorization complete. Credential saved to ${configPath}.\n`);
}

await main();
