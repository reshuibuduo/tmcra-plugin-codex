import assert from "node:assert/strict";
import { createHash, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { mkdir, readFile, readdir, rename, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const testDir = dirname(fileURLToPath(import.meta.url));
const pluginRoot = resolve(testDir, "..");
const repoRoot = resolve(pluginRoot, "..", "..");
const sourceManifest = JSON.parse(
  await readFile(join(pluginRoot, ".codex-plugin", "plugin.json"), "utf8"),
);
const archivePath = resolve(
  process.env.TMCRA_TEST_RELEASE_ARCHIVE ||
    join(
      repoRoot,
      "03-tmcra-web-console",
      "public",
      "downloads",
      `tmcra-codex-${sourceManifest.version}.zip`,
    ),
);

function run(command, args, { cwd, env, timeoutMs = 60_000 } = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd,
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

async function findCodexCli() {
  if (process.env.TMCRA_TEST_CODEX_CLI) return resolve(process.env.TMCRA_TEST_CODEX_CLI);
  const binRoot = join(process.env.LOCALAPPDATA || "", "OpenAI", "Codex", "bin");
  const candidates = [];
  if (existsSync(binRoot)) {
    for (const entry of await readdir(binRoot, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const candidate = join(binRoot, entry.name, "codex.exe");
      if (existsSync(candidate)) candidates.push(candidate);
    }
  }
  if (candidates.length === 0) throw new Error("A runnable Codex CLI was not found.");
  return candidates.sort().at(-1);
}

async function findBundledNode() {
  const runtimeRoot = join(process.env.LOCALAPPDATA || "", "OpenAI", "Codex", "runtimes");
  const candidates = [];
  if (existsSync(runtimeRoot)) {
    for (const runtime of await readdir(runtimeRoot, { withFileTypes: true })) {
      if (!runtime.isDirectory()) continue;
      const runtimeDir = join(runtimeRoot, runtime.name);
      for (const revision of await readdir(runtimeDir, { withFileTypes: true })) {
        if (!revision.isDirectory()) continue;
        const candidate = join(runtimeDir, revision.name, "bin", "node.exe");
        if (existsSync(candidate)) candidates.push(candidate);
      }
    }
  }
  if (candidates.length === 0) throw new Error("The Codex bundled Node.js runtime was not found.");
  return candidates.sort().at(-1);
}

function readRequestJson(request) {
  return new Promise((resolvePromise, reject) => {
    let text = "";
    request.on("data", (chunk) => (text += chunk));
    request.once("error", reject);
    request.once("end", () => resolvePromise(text ? JSON.parse(text) : {}));
  });
}

function sendJson(response, status, value) {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  response.end(`${JSON.stringify(value)}\n`);
}

if (process.platform !== "win32") {
  process.stdout.write(`${JSON.stringify({ ok: true, skipped: "windows-only" }, null, 2)}\n`);
  process.exit(0);
}

assert.ok(existsSync(archivePath), `release archive does not exist: ${archivePath}`);
const releaseManifest = sourceManifest;
assert.match(
  releaseManifest.version,
  /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/u,
);

const testRoot = join(repoRoot, ".tmcra", `release-install-${randomUUID()}`);
const firstRoot = join(testRoot, "首次 安装 包");
const movedRoot = join(testRoot, "移动后的 中文 路径");
const codexHome = join(testRoot, "Codex 用户 home");
const configPath = join(testRoot, "用户 配置", "tmcra config.json");
const installationPath = join(testRoot, "用户 配置", "installation.json");
const authorizations = [];
const secrets = [];
let baseUrl = "";
let sessionChecks = 0;

const server = createServer((request, response) => {
  void (async () => {
    const url = new URL(request.url || "/", baseUrl || "http://localhost");
    if (request.method === "POST" && url.pathname === "/api/device/v1/authorizations") {
      const body = await readRequestJson(request);
      assert.equal(body.clientId, "tmcra-codex");
      assert.equal(body.codeChallengeMethod, "S256");
      assert.match(body.codeChallenge, /^[A-Za-z0-9_-]{43}$/u);
      const number = authorizations.length + 1;
      const deviceCode = `release-device-${number}-${randomUUID()}`;
      const userCode = `REL-${String(number).padStart(4, "0")}`;
      authorizations.push({
        installationId: body.installationId,
        challenge: body.codeChallenge,
        deviceCode,
        attempts: 0,
        acknowledgements: 0,
        verifier: "",
        accessToken: "",
        deliveryReceipt: "",
      });
      secrets.push(deviceCode);
      sendJson(response, 201, {
        deviceCode,
        userCode,
        verificationUriComplete: `${baseUrl}/activate?user_code=${encodeURIComponent(userCode)}`,
        expiresIn: 60,
        interval: 0.05,
      });
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/device/v1/token") {
      const body = await readRequestJson(request);
      const current = authorizations.find((entry) => entry.deviceCode === body.deviceCode);
      assert.ok(current);
      current.verifier = String(body.codeVerifier || "");
      if (createHash("sha256").update(current.verifier).digest("base64url") !== current.challenge) {
        sendJson(response, 400, { error: "invalid_grant" });
        return;
      }
      if (body.deliveryReceipt !== undefined) {
        assert.match(String(body.deliveryReceipt), /^[A-Za-z0-9_-]{43}$/u);
        assert.equal(body.deliveryReceipt, current.deliveryReceipt);
        current.acknowledgements += 1;
        sendJson(response, 200, { claimed: true, expiresIn: 3600 });
        return;
      }
      current.attempts += 1;
      if (current.attempts === 1) {
        sendJson(response, 400, { error: "authorization_pending" });
        return;
      }
      if (!current.accessToken) {
        current.accessToken = `release-access-${authorizations.indexOf(current) + 1}-${randomUUID()}`;
        current.deliveryReceipt = createHash("sha256")
          .update(`release-receipt-${randomUUID()}`)
          .digest("base64url");
        secrets.push(current.verifier, current.accessToken, current.deliveryReceipt);
      }
      sendJson(response, 200, {
        accessToken: current.accessToken,
        deliveryReceipt: current.deliveryReceipt,
        deliveryAcknowledgementRequired: true,
        tokenType: "Bearer",
        expiresIn: 600,
        baseUrl,
        scopeNamespace: "release-test-user",
      });
      return;
    }
    if (request.method === "GET" && url.pathname === "/v1/session") {
      const token = String(request.headers.authorization || "").replace(/^Bearer\s+/iu, "");
      assert.ok(authorizations.some((entry) => entry.accessToken === token));
      sessionChecks += 1;
      sendJson(response, 200, {
        ok: true,
        authenticated: true,
        service: {
          name: "tmcra-memory-api",
          version: "release-test",
          capabilities: ["memory.read", "memory.write"],
        },
        credential: {
          type: "scoped_token",
          tenant_id: "release-test",
          principal: "subject:release-test-user",
          subject: "release-test-user",
          permissions: ["memory:read", "memory:write"],
          scope_names: [],
          scope_prefixes: ["release-test-user-"],
          unrestricted: false,
          expires_at: null,
        },
      });
      return;
    }
    if (request.method === "GET" && url.pathname === "/activate") {
      response.writeHead(200, { "content-type": "text/plain; charset=utf-8" });
      response.end("Mock TMCRA approval page\n");
      return;
    }
    sendJson(response, 404, { error: "not_found" });
  })().catch((error) => sendJson(response, 500, { error: "mock_failure", message: error.message }));
});

await new Promise((resolvePromise, reject) => {
  server.once("error", reject);
  server.listen(0, "localhost", resolvePromise);
});
baseUrl = `http://localhost:${server.address().port}`;

const codexCli = await findCodexCli();
const bundledNode = await findBundledNode();
const bundledNodeVersion = await run(bundledNode, ["--version"], { cwd: repoRoot });
assert.equal(bundledNodeVersion.code, 0, bundledNodeVersion.stderr || bundledNodeVersion.stdout);
const bundledNodeMajor = Number(/^v(\d+)/u.exec(bundledNodeVersion.stdout.trim())?.[1]);
assert.ok(Number.isInteger(bundledNodeMajor) && bundledNodeMajor >= 18);
const outputs = [];

async function installFrom(root, { useBundledNodeFallback = false } = {}) {
  const childPath = useBundledNodeFallback
    ? `${process.env.SystemRoot}\\System32;${process.env.SystemRoot}`
    : process.env.PATH;
  const powershellPath = join(
    process.env.SystemRoot,
    "System32",
    "WindowsPowerShell",
    "v1.0",
    "powershell.exe",
  );
  const result = await run(
    powershellPath,
    [
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      join(root, "Install-TMCRA.ps1"),
      "-ProgressJson",
      "-NoBrowser",
      "-AuthorizationUrl",
      baseUrl,
    ],
    {
      cwd: root,
      env: {
        CODEX_HOME: codexHome,
        TMCRA_CONFIG_FILE: configPath,
        TMCRA_INSTALLATION_FILE: installationPath,
        PATH: childPath,
      },
      timeoutMs: 90_000,
    },
  );
  outputs.push(result.stdout, result.stderr);
  assert.equal(result.code, 0, `${result.stdout}\n${result.stderr}`);
  assert.match(
    `${result.stdout}\n${result.stderr}`,
    /TMCRA Memory is installed and authorized/u,
  );
}

try {
  await mkdir(firstRoot, { recursive: true });
  await mkdir(codexHome, { recursive: true });
  const extracted = await run("tar.exe", ["-xf", archivePath, "-C", firstRoot], {
    cwd: repoRoot,
  });
  assert.equal(extracted.code, 0, extracted.stderr || extracted.stdout);

  await installFrom(firstRoot, { useBundledNodeFallback: true });
  const legacyAcl = await run(
    "icacls.exe",
    [configPath, "/inheritance:r", "/grant:r", `${process.env.USERNAME}:(R,W)`],
    { cwd: repoRoot },
  );
  assert.equal(legacyAcl.code, 0, legacyAcl.stderr || legacyAcl.stdout);
  await installFrom(firstRoot);
  await rename(firstRoot, movedRoot);
  await installFrom(movedRoot);

  const packageRootEntries = await readdir(movedRoot);
  assert.equal(
    packageRootEntries.some((entry) => entry.startsWith("System.Management.Automation.PSReference")),
    false,
    "config backup must not be written into the package directory",
  );
  const configBackups = (await readdir(codexHome)).filter((entry) =>
    entry.startsWith("config.toml.tmcra-backup-"),
  );
  assert.ok(configBackups.length >= 1, "installer must back up config.toml beside the source file");

  const list = await run(codexCli, ["plugin", "list", "--json"], {
    cwd: movedRoot,
    env: { CODEX_HOME: codexHome },
  });
  assert.equal(list.code, 0, list.stderr || list.stdout);
  const installed = JSON.parse(list.stdout).installed.find(
    (entry) => entry.pluginId === "tmcra-memory@tmcra-local",
  );
  assert.ok(installed);
  assert.equal(installed.version, releaseManifest.version);
  const sourceRoot = resolve(String(installed.marketplaceSource.source).replace(/^\\\\\?\\/u, ""));
  assert.equal(sourceRoot.toLowerCase(), resolve(movedRoot).toLowerCase());

  const cacheParent = join(codexHome, "plugins", "cache", "tmcra-local", "tmcra-memory");
  const cacheVersions = (await readdir(cacheParent, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
  assert.deepEqual(cacheVersions, [releaseManifest.version]);
  const cachedRoot = join(cacheParent, releaseManifest.version);
  const cachedManifest = JSON.parse(
    await readFile(join(cachedRoot, ".codex-plugin", "plugin.json"), "utf8"),
  );
  assert.equal(cachedManifest.name, "tmcra-memory");
  assert.equal(cachedManifest.version, releaseManifest.version);
  assert.equal(existsSync(join(cachedRoot, "tests")), false);
  assert.equal(existsSync(join(cachedRoot, ".smoke-data")), false);
  assert.equal(existsSync(join(cachedRoot, "scripts", "build_release.ps1")), false);

  assert.equal(authorizations.length, 3);
  assert.equal(sessionChecks, 3);
  assert.ok(authorizations.every((entry) => entry.attempts === 2));
  assert.ok(authorizations.every((entry) => entry.acknowledgements === 1));
  assert.equal(new Set(authorizations.map((entry) => entry.installationId)).size, 1);
  const finalConfig = JSON.parse(await readFile(configPath, "utf8"));
  assert.equal(Object.hasOwn(finalConfig, "pendingDelivery"), false);
  assert.ok(Date.parse(finalConfig.expiresAt) > Date.now() + 3_000_000);
  const allOutput = outputs.join("\n");
  for (const secret of secrets) assert.ok(!allOutput.includes(secret));

  process.stdout.write(
    `${JSON.stringify(
      {
        ok: true,
        pluginVersion: releaseManifest.version,
        cleanInstall: true,
        repeatedInstall: true,
        configBackupPath: true,
        legacyReadWriteAclRecovered: true,
        movedUnicodeSpaceDirectory: true,
        bundledNodeFallback: true,
        bundledNodeMajor,
        cacheVersions,
        authorizationFlows: authorizations.length,
        sessionChecks,
        deliveryAcknowledgements: authorizations.reduce(
          (total, entry) => total + entry.acknowledgements,
          0,
        ),
        secretsRedacted: true,
      },
      null,
      2,
    )}\n`,
  );
} finally {
  await new Promise((resolvePromise) => server.close(resolvePromise));
  const safeRoot = resolve(testRoot);
  const safeParent = resolve(repoRoot, ".tmcra");
  if (dirname(safeRoot) === safeParent && safeRoot.startsWith(`${safeParent}\\`)) {
    await run("icacls.exe", [safeRoot, "/reset", "/T", "/C"], {
      cwd: repoRoot,
      timeoutMs: 30_000,
    });
    await rm(safeRoot, { recursive: true, force: true });
  }
}
