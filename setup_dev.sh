#!/usr/bin/env bash
# =============================================================================
# setup_dev.sh — One-shot dev environment setup for flouds-onnx (Linux/macOS)
# Run from the repo root: ./setup_dev.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "=== flouds-onnx dev setup ==="

# 1. Create venv if missing
if [ ! -d "$ROOT/.venv" ]; then
    echo "Creating virtual environment..."
    "${PYTHON:-python3.14}" -m venv "$ROOT/.venv"  # Flouds standard: Python 3.14
fi

# 2. Activate
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

# 3. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# 4. Install deps
echo "Installing dependencies..."
# Editable, so scripts and the flouds-onnx command use this working copy.
pip install -e "$ROOT[dev,export,seq2seq,server]" --quiet

# 5. Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type pre-push

# 6. Verify
echo ""
echo "Running pre-commit on all files..."
pre-commit run --all-files

echo ""
echo "Running unit tests..."
pytest tests/unit/ -q

echo ""
echo "=== Setup complete ==="
echo "Pre-commit hooks are installed and will run on every commit."
