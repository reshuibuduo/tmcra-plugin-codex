#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INSTALLER="$ROOT/plugins/tmcra-memory/scripts/install.sh"
if [ ! -f "$INSTALLER" ]; then
  echo "TMCRA package is incomplete: plugins/tmcra-memory/scripts/install.sh is missing." >&2
  exit 1
fi

exec sh "$INSTALLER" "$@"
