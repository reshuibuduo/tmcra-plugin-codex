import assert from "node:assert/strict";
import { createHash, randomBytes, randomUUID } from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import { createServer } from "node:http";
import { copyFile, cp, mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const testDir = dirname(fileURLToPath(import.meta.url));
const pluginRoot = resolve(testDir, "..");
const scriptPath = join(pluginRoot, "scripts", "device_login.mjs");
const installPath = join(pluginRoot, "scripts", "install.ps1");
const rootInstallTemplatePath = join(pluginRoot, "packaging", "Install-TMCRA.ps1");
const powershellCommand = process.env.TMCRA_TEST_POWERSHELL || "powershell.exe";
const secretValues = [];
const authorizations = [];
const validTokens = new Set();
let authorizationCreateTransportFailures = 1;
let authorizationCreateTransportFailuresSeen = 0;
let baseUrl;

async function readJson(request) {
  let text = "";
  for await (const chunk of request) text += chunk;
  return text ? JSON.parse(text) : {};
}

function send(response, status, value) {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  response.end(`${JSON.stringify(value)}\n`);
}

const server = createServer((request, response) => {
  void (async () => {
    const url = new URL(request.url || "/", baseUrl || "http://localhost");
    if (request.method === "POST" && url.pathname === "/api/device/v1/authorizations") {
      if (authorizationCreateTransportFailures > 0) {
        authorizationCreateTransportFailures -= 1;
        authorizationCreateTransportFailuresSeen += 1;
        response.destroy();
        return;
      }
      const body = await readJson(request);
      assert.equal(body.clientId, "tmcra-codex");
      assert.match(body.installationId, /^[0-9a-f-]{36}$/iu);
      assert.equal(typeof body.clientVersion, "string");
      assert.ok(body.clientVersion.length > 0);
      assert.equal(typeof body.platform, "string");
      assert.ok(body.platform.length > 0);
      assert.equal(body.codeChallengeMethod, "S256");
      assert.match(body.codeChallenge, /^[A-Za-z0-9_-]{43}$/u);
      const index = authorizations.length + 1;
      const deviceCode = `device-code-secret-${index}-${randomUUID()}`;
      const userCode = `TEST-${String(index).padStart(4, "0")}`;
      authorizations.push({
        installationId: body.installationId,
        codeChallenge: body.codeChallenge,
        deviceCode,
        userCode,
        attempts: 0,
        codeVerifier: null,
        status: "pending",
        approvedBy: null,
        slowDown: authorizations.length === 0,
        accessToken: null,
        deliveryReceipt: null,
        acknowledgements: 0,
        transportFailures: authorizations.length === 0 ? 1 : 0,
        transportFailuresSeen: 0,
      });
      secretValues.push(deviceCode);
      send(response, 200, {
        deviceCode,
        userCode,
        verificationUriComplete: `${baseUrl}/activate?user_code=${encodeURIComponent(userCode)}`,
        expiresIn: 30,
        interval: 0.05,
      });
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/device/v1/token") {
      const body = await readJson(request);
      const authorization = authorizations.find((value) => value.deviceCode === body.deviceCode);
      assert.ok(authorization, "device code must belong to an active authorization");
      if (authorization.transportFailures > 0) {
        authorization.transportFailures -= 1;
        authorization.transportFailuresSeen += 1;
        response.destroy();
        return;
      }
      authorization.attempts += 1;
      const submittedVerifier = String(body.codeVerifier || "");
      const actualChallenge = createHash("sha256")
        .update(submittedVerifier)
        .digest("base64url");
      if (actualChallenge !== authorization.codeChallenge) {
        send(response, 400, { error: "invalid_grant" });
        return;
      }
      authorization.codeVerifier = submittedVerifier;
      if (!secretValues.includes(submittedVerifier)) secretValues.push(submittedVerifier);
      if (body.deliveryReceipt !== undefined) {
        if (
          authorization.status === "consumed" &&
          body.deliveryReceipt === authorization.deliveryReceipt
        ) {
          send(response, 200, { claimed: true, expiresIn: 3600 });
          return;
        }
        if (
          authorization.status !== "approved" ||
          body.deliveryReceipt !== authorization.deliveryReceipt
        ) {
          send(response, 400, { error: "invalid_grant" });
          return;
        }
        authorization.acknowledgements += 1;
        authorization.status = "consumed";
        send(response, 200, { claimed: true, expiresIn: 3600 });
        return;
      }
      if (authorization.status === "consumed") {
        send(response, 400, { error: "expired_token" });
        return;
      }
      if (authorization.status === "denied") {
        send(response, 400, { error: "access_denied" });
        return;
      }
      if (authorization.status === "expired") {
        send(response, 400, { error: "expired_token" });
        return;
      }
      if (authorization.status === "pending" && authorization.attempts === 1) {
        send(response, 400, { error: "authorization_pending" });
        return;
      }
      if (authorization.status === "pending" && authorization.slowDown && authorization.attempts === 2) {
        send(response, 400, { error: "slow_down" });
        return;
      }
      if (authorization.status === "pending") {
        send(response, 400, { error: "authorization_pending" });
        return;
      }
      if (!authorization.accessToken) {
        authorization.accessToken = `access-token-secret-${authorizations.indexOf(authorization) + 1}-${randomUUID()}`;
        authorization.deliveryReceipt = randomBytes(32).toString("base64url");
        validTokens.add(authorization.accessToken);
        secretValues.push(authorization.accessToken, authorization.deliveryReceipt);
      }
      send(response, 200, {
        accessToken: authorization.accessToken,
        deliveryReceipt: authorization.deliveryReceipt,
        deliveryAcknowledgementRequired: true,
        tokenType: "Bearer",
        expiresIn: 3600,
        baseUrl,
        scopeNamespace: "device-test-user",
      });
      return;
    }

    if (
      request.method === "POST" &&
      url.pathname === "/v1/scopes/device-test-user-global/recall"
    ) {
      const token = String(request.headers.authorization || "").replace(/^Bearer\s+/iu, "");
      if (!validTokens.has(token)) {
        send(response, 401, { error: { code: "invalid_token", message: "Invalid token" } });
        return;
      }
      await readJson(request);
      send(response, 200, { query_id: "device-login-check", prompt_evidence: { content: "" } });
      return;
    }

    if (request.method === "GET" && url.pathname === "/v1/session") {
      const token = String(request.headers.authorization || "").replace(/^Bearer\s+/iu, "");
      if (!validTokens.has(token)) {
        send(response, 401, { error: { code: "invalid_token", message: "Invalid token" } });
        return;
      }
      send(response, 200, {
        ok: true,
        authenticated: true,
        service: {
          version: "device-login-mock",
          capabilities: [],
        },
      });
      return;
    }

    if (request.method === "GET" && url.pathname === "/activate") {
      response.writeHead(200, { "content-type": "text/plain" });
      response.end("Approved by mock console");
      return;
    }
    send(response, 404, { error: "not_found" });
  })().catch((error) => send(response, 500, { error: "mock_failure", message: error.message }));
});

function run(command, args, { env, timeoutMs = 30_000 } = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd: pluginRoot,
      env: { ...process.env, ...env },
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`command timed out: ${command}`));
    }, timeoutMs);
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.once("error", reject);
    child.once("close", (code) => {
      clearTimeout(timer);
      resolvePromise({ code, stdout, stderr });
    });
  });
}

function progressEvents(stderr) {
  const events = [];
  for (const line of String(stderr || "").split(/\r?\n/u)) {
    const start = line.indexOf("{");
    const end = line.lastIndexOf("}");
    if (start < 0 || end <= start) continue;
    try {
      const value = JSON.parse(line.slice(start, end + 1));
      if (typeof value?.event === "string" && value.event.startsWith("tmcra.")) events.push(value);
    } catch {
      // PowerShell may add its own diagnostic lines around native stderr.
    }
  }
  return events;
}

async function waitFor(predicate, message, timeoutMs = 5_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 20));
  }
  throw new Error(message);
}

async function waitForProcessReady(processPromise, predicate, message, timeoutMs = 5_000) {
  const result = await Promise.race([
    waitFor(predicate, message, timeoutMs).then(() => null),
    processPromise,
  ]);
  if (predicate()) return;
  if (result) {
    throw new Error(
      `${message}; installer exited ${result.code}: ${result.stderr || result.stdout || "no output"}`,
    );
  }
}

function approveAsMockUser(authorization) {
  authorization.status = "approved";
  authorization.approvedBy = "mock-console-user";
}

async function createManualAuthorization(verifier) {
  const response = await fetch(`${baseUrl}/api/device/v1/authorizations`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      clientId: "tmcra-codex",
      installationId: randomUUID(),
      clientVersion: "0.1.1-test",
      platform: `${process.platform}-${process.arch}`,
      codeChallenge: createHash("sha256").update(verifier).digest("base64url"),
      codeChallengeMethod: "S256",
    }),
  });
  assert.equal(response.status, 200);
  return response.json();
}

async function exchangeToken(deviceCode, codeVerifier) {
  const response = await fetch(`${baseUrl}/api/device/v1/token`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ deviceCode, codeVerifier }),
  });
  return { status: response.status, payload: await response.json() };
}

async function acknowledgeToken(deviceCode, codeVerifier, deliveryReceipt) {
  const response = await fetch(`${baseUrl}/api/device/v1/token`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ deviceCode, codeVerifier, deliveryReceipt }),
  });
  return { status: response.status, payload: await response.json() };
}

await new Promise((resolvePromise, reject) => {
  server.once("error", reject);
  server.listen(0, "localhost", resolvePromise);
});
baseUrl = `http://localhost:${server.address().port}`;

const tempRoot = await mkdtemp(join(tmpdir(), "tmcra device flow 测试-"));
const configPath = join(tempRoot, "用户 config", "tmcra config.json");
const installationPath = join(tempRoot, "用户 config", "installation.json");
const assertions = {};

try {
  const directPromise = run(
    process.execPath,
    [scriptPath, "--no-open", "--json"],
    {
      env: {
        TMCRA_AUTH_BASE_URL: baseUrl,
        TMCRA_CONFIG_FILE: configPath,
        TMCRA_INSTALLATION_FILE: installationPath,
      },
      timeoutMs: 20_000,
    },
  );
  await waitFor(() => authorizations.length === 1, "device authorization was not created");
  await waitFor(
    () => authorizations[0].attempts >= 2,
    "device client did not exercise pending and slow_down",
  );
  approveAsMockUser(authorizations[0]);
  const direct = await directPromise;
  assert.equal(direct.code, 0, direct.stderr || direct.stdout);
  const directResult = JSON.parse(direct.stdout.trim());
  const directConfig = JSON.parse(await readFile(configPath, "utf8"));
  const installation = JSON.parse(await readFile(installationPath, "utf8"));
  assert.equal(directResult.ok, true);
  assert.equal(directResult.credentialStored, true);
  assert.equal(directResult.browserOpened, false);
  assert.equal(directConfig.authMode, "device");
  assert.equal(directConfig.scopeNamespace, "device-test-user");
  assert.equal(authorizations[0].installationId, installation.installationId);
  assert.match(direct.stderr, /TMCRA user code: TEST-0001/u);
  assertions.pkcePendingSlowDownAndSuccess = authorizations[0].attempts >= 4;
  assertions.transientNetworkFailureRetried = authorizations[0].transportFailuresSeen === 1;
  assertions.authorizationCreationNetworkFailureRetried =
    authorizationCreateTransportFailuresSeen === 1;
  assertions.tokenDeliveryAcknowledgedAfterConfigWrite =
    authorizations[0].acknowledgements === 1;
  assertions.explicitMockUserApproval = authorizations[0].approvedBy === "mock-console-user";
  assertions.unicodeAndSpaceConfigPath = directResult.configPath === resolve(configPath);
  if (process.platform === "win32") {
    const acl = spawnSync("icacls.exe", [configPath], { encoding: "utf8", windowsHide: true });
    assert.equal(acl.status, 0, acl.stderr || acl.stdout);
    assert.doesNotMatch(acl.stdout, /\(I\)/u, "credential file must not inherit broad ACL entries");
    assertions.firstInstallCredentialAclProtected = true;
  } else {
    assert.equal((await stat(configPath)).mode & 0o077, 0);
    assertions.firstInstallCredentialAclProtected = true;
  }

  let installer = { stdout: "", stderr: "" };
  let desktopInstaller = { stdout: "", stderr: "" };
  let rejectedNode = { stdout: "", stderr: "" };
  if (process.platform === "win32") {
    const installerPromise = run(
      powershellCommand,
      [
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        installPath,
        "-SkipPluginInstall",
        "-ApiOnlyCheck",
        "-NoBrowser",
        "-AuthorizationUrl",
        baseUrl,
      ],
      {
        env: {
          TMCRA_CONFIG_FILE: configPath,
          TMCRA_INSTALLATION_FILE: installationPath,
        },
        timeoutMs: 30_000,
      },
    );
    await waitForProcessReady(
      installerPromise,
      () => authorizations.length === 2,
      "installer authorization was not created",
      20_000,
    );
    await waitFor(() => authorizations[1].attempts >= 1, "installer did not poll for approval");
    approveAsMockUser(authorizations[1]);
    installer = await installerPromise;
    assert.equal(installer.code, 0, installer.stderr || installer.stdout);
    const updatedConfig = JSON.parse(await readFile(configPath, "utf8"));
    assert.equal(updatedConfig.authMode, "device");
    assert.equal(authorizations[1].installationId, installation.installationId);
    assert.match(installer.stdout, /TMCRA Memory is installed and authorized/u);
    assertions.windowsInstallerDefaultDeviceFlow = true;
    assertions.persistentInstallationId =
      authorizations[0].installationId === authorizations[1].installationId;

    const desktopPackageRoot = join(tempRoot, "桌面 安装包");
    const desktopPackagePluginRoot = join(desktopPackageRoot, "plugins", "tmcra-memory");
    const rootInstallPath = join(desktopPackageRoot, "Install-TMCRA.ps1");
    await mkdir(join(desktopPackageRoot, "plugins"), { recursive: true });
    await copyFile(rootInstallTemplatePath, rootInstallPath);
    await cp(pluginRoot, desktopPackagePluginRoot, { recursive: true });

    const desktopRuntimeDirectory = join(tempRoot, "桌面 应用 runtime");
    const desktopNodePath = join(desktopRuntimeDirectory, "TMCRA Desktop.exe");
    await mkdir(desktopRuntimeDirectory, { recursive: true });
    await copyFile(process.execPath, desktopNodePath);
    const desktopAuthorizationIndex = authorizations.length;
    const desktopInstallerPromise = run(
      powershellCommand,
      [
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        rootInstallPath,
        "-SkipPluginInstall",
        "-ApiOnlyCheck",
        "-NoBrowser",
        "-ProgressJson",
        "-NodePath",
        desktopNodePath,
        "-AuthorizationUrl",
        baseUrl,
      ],
      {
        env: {
          ELECTRON_RUN_AS_NODE: "1",
          TMCRA_CONFIG_FILE: configPath,
          TMCRA_INSTALLATION_FILE: installationPath,
        },
        timeoutMs: 30_000,
      },
    );
    await waitForProcessReady(
      desktopInstallerPromise,
      () => authorizations.length > desktopAuthorizationIndex,
      "desktop installer authorization was not created",
      20_000,
    );
    await waitFor(
      () => authorizations[desktopAuthorizationIndex].attempts >= 1,
      "desktop installer did not poll for approval",
    );
    approveAsMockUser(authorizations[desktopAuthorizationIndex]);
    desktopInstaller = await desktopInstallerPromise;
    assert.equal(desktopInstaller.code, 0, desktopInstaller.stderr || desktopInstaller.stdout);
    const events = progressEvents(desktopInstaller.stderr);
    assert.deepEqual(
      events.map((value) => value.event),
      [
        "tmcra.authorization.required",
        "tmcra.install.progress",
        "tmcra.install.progress",
      ],
    );
    const required = events[0];
    assert.deepEqual(
      Object.keys(required).sort(),
      ["event", "expiresAt", "userCode", "verificationUrl"],
    );
    assert.equal(required.userCode, authorizations[desktopAuthorizationIndex].userCode);
    assert.equal(required.verificationUrl, `${baseUrl}/activate?user_code=${encodeURIComponent(required.userCode)}`);
    assert.ok(Date.parse(required.expiresAt) > Date.now());
    assert.deepEqual(events[1], {
      event: "tmcra.install.progress",
      step: "authorize",
      status: "running",
      message: "Waiting for approval in the TMCRA console.",
    });
    assert.deepEqual(events[2], {
      event: "tmcra.install.progress",
      step: "authorize",
      status: "completed",
      message: "TMCRA authorization completed.",
    });
    assert.doesNotMatch(
      JSON.stringify(events),
      /deviceCode|codeVerifier|accessToken|deliveryReceipt|receipt/iu,
    );
    assert.match(desktopInstaller.stderr, /TMCRA user code:/u);
    assert.match(desktopInstaller.stdout, /TMCRA Memory is installed and authorized/u);
    assertions.explicitElectronNodePath = true;
    assertions.rootWrapperForwardsDesktopParameters = true;
    assertions.progressJsonLifecycleEvents = true;
    assertions.progressJsonContainsNoSensitiveFields = true;
    assertions.unicodeAndSpaceNodePath = desktopNodePath.includes("桌面 应用");

    const oldNodeDirectory = join(tempRoot, "旧 Node runtime");
    const oldNodePath = join(oldNodeDirectory, "node 16.cmd");
    await mkdir(oldNodeDirectory, { recursive: true });
    await writeFile(oldNodePath, "@echo off\r\necho v16.20.0\r\n", "utf8");
    rejectedNode = await run(
      "powershell.exe",
      [
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        installPath,
        "-SkipPluginInstall",
        "-SkipConfigure",
        "-NodePath",
        oldNodePath,
      ],
      { timeoutMs: 10_000 },
    );
    assert.notEqual(rejectedNode.code, 0);
    assert.match(
      `${rejectedNode.stdout}\n${rejectedNode.stderr}`,
      /explicit NodePath must run Node\.js 18 or newer/u,
    );
    assertions.explicitNodePathRejectsNodeBefore18 = true;
  } else {
    assertions.windowsInstallerDefaultDeviceFlow = "skipped-non-windows";
    assertions.persistentInstallationId = true;
    assertions.explicitElectronNodePath = "skipped-non-windows";
    assertions.rootWrapperForwardsDesktopParameters = "skipped-non-windows";
    assertions.progressJsonLifecycleEvents = "skipped-non-windows";
    assertions.progressJsonContainsNoSensitiveFields = "skipped-non-windows";
    assertions.unicodeAndSpaceNodePath = "skipped-non-windows";
    assertions.explicitNodePathRejectsNodeBefore18 = "skipped-non-windows";
  }

  const manualVerifier = randomBytes(48).toString("base64url");
  const invalidVerifier = randomBytes(48).toString("base64url");
  secretValues.push(manualVerifier, invalidVerifier);
  const manual = await createManualAuthorization(manualVerifier);
  const manualAuthorization = authorizations.find((value) => value.deviceCode === manual.deviceCode);
  approveAsMockUser(manualAuthorization);
  const invalid = await exchangeToken(manual.deviceCode, invalidVerifier);
  assert.equal(invalid.status, 400);
  assert.equal(invalid.payload.error, "invalid_grant");
  const firstExchange = await exchangeToken(manual.deviceCode, manualVerifier);
  assert.equal(firstExchange.status, 200);
  secretValues.push(firstExchange.payload.accessToken);
  const replay = await exchangeToken(manual.deviceCode, manualVerifier);
  assert.equal(replay.status, 200);
  assert.equal(replay.payload.accessToken, firstExchange.payload.accessToken);
  assert.equal(replay.payload.deliveryReceipt, firstExchange.payload.deliveryReceipt);
  const claimed = await acknowledgeToken(
    manual.deviceCode,
    manualVerifier,
    firstExchange.payload.deliveryReceipt,
  );
  assert.equal(claimed.status, 200);
  assert.equal(claimed.payload.claimed, true);
  const afterClaim = await exchangeToken(manual.deviceCode, manualVerifier);
  assert.equal(afterClaim.status, 400);
  assert.equal(afterClaim.payload.error, "expired_token");
  assertions.invalidPkceVerifierRejected = true;
  assertions.responseLossRetryReturnsSameCredential = true;
  assertions.deviceCodeOneTimeConsumptionAfterAcknowledgement = true;

  const recoveryVerifier = randomBytes(48).toString("base64url");
  secretValues.push(recoveryVerifier);
  const recovery = await createManualAuthorization(recoveryVerifier);
  const recoveryAuthorization = authorizations.find(
    (value) => value.deviceCode === recovery.deviceCode,
  );
  approveAsMockUser(recoveryAuthorization);
  const recoveryToken = await exchangeToken(recovery.deviceCode, recoveryVerifier);
  assert.equal(recoveryToken.status, 200);
  const beforeRecoveryCount = authorizations.length;
  await writeFile(configPath, `${JSON.stringify({
    schemaVersion: 2,
    authMode: "device",
    baseUrl,
    accessToken: recoveryToken.payload.accessToken,
    tokenType: "Bearer",
    expiresAt: new Date(Date.now() + 30_000).toISOString(),
    scopeNamespace: "device-test-user",
    globalScope: "device-test-user-global",
    projectScopePrefix: "device-test-user-project",
    timeoutMs: 120000,
    pendingDelivery: {
      schemaVersion: 1,
      authorizationBaseUrl: baseUrl,
      deviceCode: recovery.deviceCode,
      codeVerifier: recoveryVerifier,
      deliveryReceipt: recoveryToken.payload.deliveryReceipt,
      expiresAt: new Date(Date.now() + 30_000).toISOString(),
    },
  }, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
  const recovered = await run(
    process.execPath,
    [scriptPath, "--no-open", "--json", "--progress-json"],
    {
    env: {
      TMCRA_AUTH_BASE_URL: baseUrl,
      TMCRA_CONFIG_FILE: configPath,
      TMCRA_INSTALLATION_FILE: installationPath,
    },
    },
  );
  assert.equal(recovered.code, 0, recovered.stderr || recovered.stdout);
  assert.equal(JSON.parse(recovered.stdout).deliveryRecovered, true);
  assert.deepEqual(progressEvents(recovered.stderr), [
    {
      event: "tmcra.install.progress",
      step: "authorize",
      status: "completed",
      message: "TMCRA authorization completed.",
    },
  ]);
  assert.equal(authorizations.length, beforeRecoveryCount);
  assert.equal(
    Object.hasOwn(JSON.parse(await readFile(configPath, "utf8")), "pendingDelivery"),
    false,
  );
  assertions.interruptedAcknowledgementRecoversWithoutNewToken = true;
  assertions.progressJsonPreservesFinalJsonOutput = true;

  const deniedIndex = authorizations.length;
  const deniedPromise = run(process.execPath, [scriptPath, "--no-open", "--json"], {
    env: {
      TMCRA_AUTH_BASE_URL: baseUrl,
      TMCRA_CONFIG_FILE: join(tempRoot, "denied", "config.json"),
      TMCRA_INSTALLATION_FILE: installationPath,
    },
  });
  await waitFor(() => authorizations.length > deniedIndex, "denied authorization was not created");
  await waitFor(() => authorizations[deniedIndex].attempts >= 1, "denied flow did not poll");
  authorizations[deniedIndex].status = "denied";
  const denied = await deniedPromise;
  assert.notEqual(denied.code, 0);
  assert.match(denied.stderr, /authorization was denied/iu);
  assertions.accessDeniedHandled = true;

  const expiredIndex = authorizations.length;
  const expiredPromise = run(process.execPath, [scriptPath, "--no-open", "--json"], {
    env: {
      TMCRA_AUTH_BASE_URL: baseUrl,
      TMCRA_CONFIG_FILE: join(tempRoot, "expired", "config.json"),
      TMCRA_INSTALLATION_FILE: installationPath,
    },
  });
  await waitFor(() => authorizations.length > expiredIndex, "expired authorization was not created");
  await waitFor(() => authorizations[expiredIndex].attempts >= 1, "expired flow did not poll");
  authorizations[expiredIndex].status = "expired";
  const expired = await expiredPromise;
  assert.notEqual(expired.code, 0);
  assert.match(expired.stderr, /authorization expired/iu);
  assertions.expiredTokenHandled = true;

  const capturedOutput = [
    direct.stdout,
    direct.stderr,
    installer.stdout,
    installer.stderr,
    desktopInstaller.stdout,
    desktopInstaller.stderr,
    rejectedNode.stdout,
    rejectedNode.stderr,
    recovered.stdout,
    recovered.stderr,
    denied.stdout,
    denied.stderr,
    expired.stdout,
    expired.stderr,
  ].join("\n");
  for (const secret of secretValues) {
    assert.ok(secret && !capturedOutput.includes(secret), "credential material must not appear in output");
  }
  assertions.noDeviceCodeVerifierOrAccessTokenInOutput = true;
  assertions.finalCredentialAuthenticates = validTokens.has(
    JSON.parse(await readFile(configPath, "utf8")).accessToken,
  );
  assert.ok(Object.values(assertions).every((value) => value === true || value === "skipped-non-windows"));
  process.stdout.write(
    `${JSON.stringify({ ok: true, authorizationFlows: authorizations.length, assertions }, null, 2)}\n`,
  );
} finally {
  await new Promise((resolvePromise) => server.close(resolvePromise));
  await rm(tempRoot, { recursive: true, force: true });
}
