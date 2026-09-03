import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

const args = process.argv.slice(2);
function valueAfter(flag) {
  const index = args.indexOf(flag);
  return index >= 0 ? args[index + 1] : null;
}

const root = resolve(valueAfter("--path") || process.cwd());
const directory = join(root, ".tmcra");
const path = join(directory, "project.json");
if (existsSync(path) && !args.includes("--force")) {
  throw new Error(`${path} already exists; use --force only when intentionally replacing its identity`);
}
const value = {
  schemaVersion: 1,
  projectId: valueAfter("--id") || randomUUID(),
  name: valueAfter("--name") || basename(root),
};
await mkdir(directory, { recursive: true });
await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
process.stdout.write(`${JSON.stringify({ ok: true, path, ...value }, null, 2)}\n`);
