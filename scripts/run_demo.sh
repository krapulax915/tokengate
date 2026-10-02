#!/usr/bin/env bash
# One-command demo (bash wrapper). Windows users: python scripts/run_demo.py
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/run_demo.py --open "$@"
