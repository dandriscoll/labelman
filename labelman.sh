#!/usr/bin/env bash
# Shell wrapper for labelman.
# Handles uv sync, venv creation, and forwards all arguments to the labelman CLI.
#
# Usage: ./labelman.sh [labelman arguments...]
#   e.g. ./labelman.sh init
#        ./labelman.sh check
#        ./labelman.sh descriptor clip

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

# Ensure uv is available
if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

# Create venv and sync dependencies if needed
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment..." >&2
    uv venv "$VENV_DIR" >/dev/null
fi

if [[ ! -f "$VENV_DIR/.labelman_synced" ]] || [[ "$REPO_DIR/pyproject.toml" -nt "$VENV_DIR/.labelman_synced" ]]; then
    echo "Syncing dependencies..." >&2
    uv pip install -e "$REPO_DIR[dev]" --python "$VENV_DIR/bin/python" >/dev/null
    touch "$VENV_DIR/.labelman_synced"
fi

exec "$VENV_DIR/bin/labelman" "$@"
