import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { isAbsolute, join } from "node:path";

export async function activeLocalConfigPath() {
  if (process.env.TMCRA_CONFIG_FILE) return null;
  const path = process.env.TMCRA_LOCAL_BINDING_FILE || join(homedir(), ".config", "tmcra", "local-memory.json");
  let binding;
  try { binding = JSON.parse(await readFile(path, "utf8")); }
  catch (error) { if (error.code === "ENOENT") return null; throw error; }
  if (binding?.schemaVersion !== 1 || binding.mode !== "local" || !isAbsolute(binding.dataRoot || "")
    || !["lite-cpu", "balanced-bge", "quality-qwen"].includes(binding.profile))
    throw Error("Invalid local memory selection; cloud fallback is disabled until this selection is repaired.");
  const configPath = join(binding.dataRoot, "state", binding.profile, "secrets", "client-plugin.json");
  // A selected local install must never fall through to a previous cloud identity.
  const config = JSON.parse(await readFile(configPath, "utf8"));
  if (config.deploymentMode !== "local") throw Error("The selected memory installation is not a local identity.");
  return configPath;
}

export async function assertActiveMemoryConnection(config) {
  const path = await activeLocalConfigPath();
  if (!path) return;
  const selected = JSON.parse(await readFile(path, "utf8"));
  if (config.baseUrl !== selected.baseUrl || config.apiKey !== selected.apiKey)
    throw Error("Memory connection changed to local. Restart the host to use the selected local identity; previous cloud requests are blocked.");
}

export async function assertCloudProvidersAllowed() {
  if (await activeLocalConfigPath())
    throw Error("Local memory is selected; background cloud model requests are blocked.");
}
