#!/usr/bin/env bash
# ArtAgents: run the app from the project's ./venv (Linux/macOS).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f venv/bin/activate ]; then
    echo "ERROR: ./venv not found. Run scripts/setupvenv.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source venv/bin/activate
echo "Starting ArtAgents (make sure Ollama is running)..."
python app.py
