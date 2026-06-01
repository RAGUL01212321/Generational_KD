#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_on_server.sh /path/to/venv (optional)
# If no venv path provided, creates ./venv in repo root.
VENV_DIR=${1:-./venv}
PYTHON=${VENV_DIR}/bin/python

echo "Creating venv at ${VENV_DIR} if missing..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt

# Validate setup
python setup_local.py

# Run tests (optional)
if [ "$RUN_TESTS" = "1" ] || [ "$1" = "--test" ]; then
  python -m pip install pytest
  python -m pytest -q
fi

echo "Server setup steps complete."
