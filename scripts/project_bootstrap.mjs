import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { readFile, readdir } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

import {
  deterministicKey,
  ingest,
  loadConfig,
  PLUGIN_VERSION,
  resolveMemoryScopes,
  waitJob,
} from "./tmcra_client.mjs";

const command = process.argv[2] || "preview";
const args = process.argv.slice(3);
const MAX_FILE_BYTES = 80_000;
const MAX_SNAPSHOT_BYTES = 260_000;

function valueAfter(flag) {
  const index = args.indexOf(flag);
  return index >= 0 ? args[index + 1] : null;
}

function has(flag) {
  return args.includes(flag);
}

function git(root, gitArgs) {
  const result = spawnSync("git", ["-C", root, ...gitArgs], {
    encoding: "utf8",
    windowsHide: true,
    timeout: 10_000,
  });
  return result.status === 0 ? String(result.stdout || "").trim() : "";
}

function sanitizedGitOrigin(value) {
  const origin = String(value || "").trim();
  if (!origin) return "not detected";
  try {
    const parsed = new URL(origin);
    if (!["http:", "https:", "ssh:", "git:"].includes(parsed.protocol)) {
      return "local repository";
    }
    const port = parsed.port ? `:${parsed.port}` : "";
    return `${parsed.protocol}//${parsed.hostname}${port}${parsed.pathname}`;
  } catch {
    const scpLike = origin.match(/^(?:[^@/\s]+@)?([^:/\s]+):(.+)$/u);
    if (scpLike && !/^[A-Za-z]$/u.test(scpLike[1])) {
      return `${scpLike[1]}/${scpLike[2].replace(/^\/+/, "")}`;
    }
    return "local repository";
  }
}

const candidateFiles = [
  ".tmcra/project.json",
  "AGENTS.md",
  "README.md",
  "README",
  "package.json",
  "pyproject.toml",
  "requirements.txt",
  "Cargo.toml",
  "go.mod",
  "compose.yaml",
  "docker-compose.yml",
];

async function collectSnapshot(projectRoot) {
  const sections = [];
  const includedFiles = [];
  let remaining = MAX_SNAPSHOT_BYTES;
  const rootMarkdown = (await readdir(projectRoot, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && /\.md$/iu.test(entry.name))
    .map((entry) => entry.name)
    .sort()
    .slice(0, 12);
  const selectedFiles = [...new Set([...candidateFiles, ...rootMarkdown])];

  for (const relativePath of selectedFiles) {
    const path = join(projectRoot, relativePath);
    if (!existsSync(path) || remaining <= 0) continue;
    let content;
    try {
      content = await readFile(path, "utf8");
    } catch {
      continue;
    }
    const trimmed = content.trim().slice(0, Math.min(MAX_FILE_BYTES, remaining));
    if (!trimmed) continue;
    sections.push(`## ${relativePath}\n\n${trimmed}`);
    includedFiles.push({ path: relativePath, characters: trimmed.length });
    remaining -= trimmed.length;
  }

  const branch = git(projectRoot, ["branch", "--show-current"]);
  const remote = sanitizedGitOrigin(git(projectRoot, ["remote", "get-url", "origin"]));
  const log = git(projectRoot, [
    "log",
    "-30",
    "--date=short",
    "--pretty=format:%ad %h %s",
  ]);
  if (log) sections.push(`## Recent Git history\n\n${log}`);

  const header = [
    "# TMCRA project bootstrap snapshot",
    "",
    `Project: ${basename(projectRoot)}`,
    `Git branch: ${branch || "not detected"}`,
    `Git origin: ${remote}`,
    "",
    "This snapshot is repository evidence captured at onboarding time. It is not a reconstructed conversation.",
  ].join("\n");
  const content = `${header}\n\n${sections.join("\n\n")}\n`;
  return {
    content,
    includedFiles,
    git: { branch: branch || null, origin: remote, commitsIncluded: log ? log.split("\n").length : 0 },
    contentHash: createHash("sha256").update(content).digest("hex"),
  };
}

const projectRoot = resolve(valueAfter("--project") || process.cwd());
if (!existsSync(projectRoot)) throw new Error(`Project path does not exist: ${projectRoot}`);
const snapshot = await collectSnapshot(projectRoot);

if (command === "preview") {
  process.stdout.write(`${JSON.stringify({
    ok: true,
    uploadPerformed: false,
    projectRoot,
    includedFiles: snapshot.includedFiles,
    git: snapshot.git,
    characters: snapshot.content.length,
    contentHash: snapshot.contentHash,
  }, null, 2)}\n`);
} else if (command === "import") {
  if (!has("--confirm")) {
    throw new Error("Project bootstrap uploads selected repository documents and Git history. Review preview output, then re-run with --confirm.");
  }
  if (snapshot.includedFiles.length === 0 && snapshot.git.commitsIncluded === 0) {
    throw new Error("No supported repository documents or Git history were found to bootstrap.");
  }
  const config = await loadConfig();
  const scopes = await resolveMemoryScopes({ cwd: projectRoot, config });
  const timestamp = new Date().toISOString();
  const sessionId = `codex-bootstrap-${snapshot.contentHash.slice(0, 32)}`;
  const result = await ingest({
    config,
    scope: scopes.projectScope,
    sessionId,
    messages: [{
      message_id: `codex-bootstrap-${snapshot.contentHash.slice(0, 40)}`,
      role: "system",
      content: snapshot.content,
      timestamp,
    }],
    idempotencyKey: deterministicKey({
      kind: "codex-project-bootstrap",
      projectId: scopes.projectId,
      contentHash: snapshot.contentHash,
    }),
    metadata: {
      integration: "codex-project-bootstrap",
      integration_version: PLUGIN_VERSION,
      project_id: scopes.projectId,
      project_name: scopes.projectName,
      content_hash: snapshot.contentHash,
      evidence_kind: "repository-snapshot",
    },
  });
  const jobId = result.job_id || result.id;
  let jobStatus = "queued";
  if (has("--wait")) {
    const job = await waitJob(jobId, { timeoutMs: 900_000, pollMs: 2000, config });
    jobStatus = job.status;
    if (jobStatus !== "succeeded") throw new Error(`bootstrap job ${jobId} ended with ${jobStatus}`);
  }
  process.stdout.write(`${JSON.stringify({
    ok: true,
    projectRoot,
    projectId: scopes.projectId,
    projectScope: scopes.projectScope,
    contentHash: snapshot.contentHash,
    includedFiles: snapshot.includedFiles,
    git: snapshot.git,
    jobId,
    jobStatus,
    waited: has("--wait"),
  }, null, 2)}\n`);
} else {
  throw new Error("Usage: project_bootstrap.mjs preview --project <path> | import --project <path> --confirm [--wait]");
}
