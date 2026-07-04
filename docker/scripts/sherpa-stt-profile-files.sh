#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/model-manifest.sh" ]; then
    # shellcheck source=../../models/scripts/model-manifest.sh
    . "$SCRIPT_DIR/model-manifest.sh"
else
    # shellcheck source=../../models/scripts/model-manifest.sh
    . "$SCRIPT_DIR/../../models/scripts/model-manifest.sh"
fi
