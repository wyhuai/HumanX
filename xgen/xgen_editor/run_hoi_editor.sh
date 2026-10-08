#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec python xgen_editor/hoi_editor/server.py "$@"
