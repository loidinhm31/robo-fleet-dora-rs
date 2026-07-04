#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Legacy wrapper: ensuring repo-local ONNX Runtime instead of /usr/local/lib"
"$SCRIPT_DIR/setup-models.sh" ensure >/dev/null
printf 'ROVER_ORT_DYLIB_PATH=%s\n' "$("$SCRIPT_DIR/setup-models.sh" print-ort-path)"
