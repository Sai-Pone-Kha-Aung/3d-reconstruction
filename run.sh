#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtualenv
source "$SCRIPT_DIR/.venv/bin/activate"

echo "Environment is activated"

