#!/usr/bin/env bash
set -euo pipefail

exec node "$(dirname "${BASH_SOURCE[0]}")/benchmark-edge-voice-x86.mjs" "$@"
