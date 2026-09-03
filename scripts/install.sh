#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PLUGIN_ROOT=$(dirname "$SCRIPT_DIR")
REPO_ROOT=$(dirname "$(dirname "$PLUGIN_ROOT")")
BASE_URL=${TMCRA_BASE_URL:-https://api.tmcra.com}
AUTH_BASE_URL=${TMCRA_AUTH_BASE_URL:-https://tmcra.com}
SCOPE_NAMESPACE=${TMCRA_SCOPE_NAMESPACE:-tmcra}
CODEX_HOME=${CODEX_HOME:-"$HOME/.codex"}
NODE_PATH=${TMCRA_NODE_PATH:-}
CODEX_PATH=${TMCRA_CODEX_CLI:-}
PROGRESS_JSON=0
NO_BROWSER=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --authorization-url)
      [ "$#" -ge 2 ] || { echo "--authorization-url requires a value" >&2; exit 2; }
      AUTH_BASE_URL=$2
      shift 2
      ;;
    --node-path)
      [ "$#" -ge 2 ] || { echo "--node-path requires a value" >&2; exit 2; }
      NODE_PATH=$2
      shift 2
      ;;
    --codex-path)
      [ "$#" -ge 2 ] || { echo "--codex-path requires a value" >&2; exit 2; }
      CODEX_PATH=$2
      shift 2
      ;;
    --no-browser)
      NO_BROWSER=1
      shift
      ;;
    --progress-json)
      PROGRESS_JSON=1
      shift
      ;;
    *)
      echo "Unknown installer option: $1" >&2
      exit 2
      ;;
  esac
done

emit_progress() {
  [ "$PROGRESS_JSON" -eq 1 ] || return 0
  printf '{"event":"tmcra.install.progress","step":"%s","status":"%s","message":"%s"}\n' "$1" "$2" "$3"
}

emit_failure() {
  status=$?
  trap - EXIT
  if [ "$status" -ne 0 ] && [ "$PROGRESS_JSON" -eq 1 ]; then
    printf '%s\n' '{"event":"tmcra.install.error","code":"installer_failed","message":"TMCRA setup did not complete."}'
  fi
  exit "$status"
}
trap emit_failure EXIT

if [ -z "$NODE_PATH" ]; then
  NODE_PATH=$(command -v node 2>/dev/null || true)
fi
if [ -z "$NODE_PATH" ] || [ ! -x "$NODE_PATH" ]; then
  echo "Node.js 18 or newer is required." >&2
  exit 1
fi
NODE_MAJOR=$($NODE_PATH -p 'Number(process.versions.node.split(".")[0])')
[ "$NODE_MAJOR" -ge 18 ] || { echo "Node.js 18 or newer is required." >&2; exit 1; }

if [ -z "$CODEX_PATH" ]; then
  CODEX_PATH=$(command -v codex 2>/dev/null || true)
fi
if [ -z "$CODEX_PATH" ] || [ ! -x "$CODEX_PATH" ]; then
  echo "Codex CLI is required." >&2
  exit 1
fi
"$CODEX_PATH" plugin marketplace --help >/dev/null 2>&1 || {
  echo "This Codex version does not support plugins. Update Codex first." >&2
  exit 1
}
"$CODEX_PATH" plugin add --help >/dev/null 2>&1 || {
  echo "This Codex version does not support plugin installation. Update Codex first." >&2
  exit 1
}

EXPECTED_ROOT=$(CDPATH= cd -- "$REPO_ROOT" && pwd -P)
EXPECTED_VERSION=$($NODE_PATH -p 'JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")).version' "$PLUGIN_ROOT/.codex-plugin/plugin.json")

mkdir -p "$CODEX_HOME"
if [ -f "$CODEX_HOME/config.toml" ]; then
  BACKUP="$CODEX_HOME/config.toml.tmcra-backup-$(date +%Y%m%d-%H%M%S)"
  cp "$CODEX_HOME/config.toml" "$BACKUP"
  echo "Codex configuration backup: $BACKUP"
fi

emit_progress install_plugin running "Installing TMCRA Memory $EXPECTED_VERSION."

# Removing the known local marketplace also recovers a package that was moved
# after installation. It does not uninstall the active plugin cache.
"$CODEX_PATH" plugin marketplace remove tmcra-local --json >/dev/null 2>&1 || true
MARKETPLACE_ADD_JSON=$("$CODEX_PATH" plugin marketplace add "$EXPECTED_ROOT" --json)
printf '%s' "$MARKETPLACE_ADD_JSON" | "$NODE_PATH" -e '
  let data=""; process.stdin.on("data", c => data += c).on("end", () => {
    const value=JSON.parse(data);
    if (value.marketplaceName !== "tmcra-local") throw new Error("The package does not declare the expected marketplace.");
  });
'

INSTALLED_JSON=$("$CODEX_PATH" plugin list --json)
PLUGIN_HEALTHY=$(printf '%s' "$INSTALLED_JSON" | EXPECTED_VERSION="$EXPECTED_VERSION" EXPECTED_ROOT="$EXPECTED_ROOT" "$NODE_PATH" -e '
  const path=require("path"); let data="";
  process.stdin.on("data", c => data += c).on("end", () => {
    const value=JSON.parse(data); const plugin=(value.installed || []).find(x => x.pluginId === "tmcra-memory@tmcra-local");
    const source=String(plugin?.marketplaceSource?.source || "").replace(/^\\\\\?\\/, "");
    const healthy=plugin?.version === process.env.EXPECTED_VERSION && path.resolve(source) === path.resolve(process.env.EXPECTED_ROOT);
    process.stdout.write(healthy ? "1" : "0");
  });
')
if [ "$PLUGIN_HEALTHY" != "1" ]; then
  PLUGIN_ADD_JSON=$("$CODEX_PATH" plugin add tmcra-memory@tmcra-local --json)
  printf '%s' "$PLUGIN_ADD_JSON" | EXPECTED_VERSION="$EXPECTED_VERSION" "$NODE_PATH" -e '
    let data=""; process.stdin.on("data", c => data += c).on("end", () => {
      const value=JSON.parse(data);
      if (value.pluginId !== "tmcra-memory@tmcra-local" || value.version !== process.env.EXPECTED_VERSION) {
        throw new Error("Codex installed a stale TMCRA plugin version.");
      }
    });
  '
fi
emit_progress install_plugin completed "TMCRA Memory $EXPECTED_VERSION is installed."

"$CODEX_PATH" features enable hooks >/dev/null
LEGACY_NAMES=$("$CODEX_PATH" mcp list --json | "$NODE_PATH" -e '
  let data=""; process.stdin.on("data", c => data += c).on("end", () => {
    const rows=JSON.parse(data); for (const row of rows) {
      const name=String(row.name || ""); const normalized=name.toLowerCase().replaceAll("_", "-");
      if (normalized.startsWith("tmcra-memory") && name !== "tmcra-memory") process.stdout.write(`${name}\n`);
    }
  });
')
printf '%s\n' "$LEGACY_NAMES" | while IFS= read -r legacy_name; do
  [ -z "$legacy_name" ] || "$CODEX_PATH" mcp remove "$legacy_name" >/dev/null
done

if [ -n "${TMCRA_SETUP_API_KEY:-}" ]; then
  TMCRA_BASE_URL="$BASE_URL" TMCRA_SCOPE_NAMESPACE="$SCOPE_NAMESPACE" \
    "$NODE_PATH" "$SCRIPT_DIR/configure.mjs" >/dev/null
  unset TMCRA_SETUP_API_KEY
else
  DEVICE_ARGUMENTS=""
  if [ "$PROGRESS_JSON" -eq 1 ]; then DEVICE_ARGUMENTS="$DEVICE_ARGUMENTS --progress-json"; fi
  if [ "$NO_BROWSER" -eq 1 ]; then DEVICE_ARGUMENTS="$DEVICE_ARGUMENTS --no-open"; fi
  # DEVICE_ARGUMENTS contains only the two fixed flags above.
  # shellcheck disable=SC2086
  TMCRA_AUTH_BASE_URL="$AUTH_BASE_URL" "$NODE_PATH" "$SCRIPT_DIR/device_login.mjs" $DEVICE_ARGUMENTS
fi

CONFIG_FILE=${TMCRA_CONFIG_FILE:-"$HOME/.config/tmcra/config.json"}
chmod 600 "$CONFIG_FILE"
TMCRA_CODEX_CLI="$CODEX_PATH" "$NODE_PATH" "$SCRIPT_DIR/check_config.mjs"

trap - EXIT
if [ "$PROGRESS_JSON" -eq 1 ]; then
  printf '%s\n' '{"event":"tmcra.install.complete"}'
fi
echo "TMCRA Memory $EXPECTED_VERSION is installed and authorized in Codex."
echo "Restart Codex, confirm TMCRA Memory is enabled in Plugins, run /hooks, and trust the TMCRA lifecycle hooks."
echo "Hook trust requires explicit user review and cannot be granted safely by this installer."
