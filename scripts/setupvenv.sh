#!/usr/bin/env bash
# ArtAgents: create ./venv in the project root and install dependencies (Linux/macOS).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "ERROR: Python 3.11+ is required (found: $("$PYTHON" --version 2>&1))." >&2
    exit 1
fi

"$PYTHON" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Patched Pillow: gradio still declares pillow<13, so it is installed on its own.
python -m pip install --no-deps -r requirements-pillow.txt

echo
echo "Environment ready. Start Ollama, then run: scripts/govenv.sh"
