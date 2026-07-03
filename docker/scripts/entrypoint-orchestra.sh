#!/bin/bash
set -e

. /app/scripts/sherpa-stt-profile-files.sh

echo "==================================================================="
echo "  Robo-Fleet Orchestra Container Starting"
echo "==================================================================="

STT_PROFILE="${STT_PROFILE:-en-vad-offline}"
STT_MODEL_ROOT="${STT_MODEL_ROOT:-/models/sherpa-onnx/asr}"
export ORCHESTRA_ZENOH_CONFIG="${ORCHESTRA_ZENOH_CONFIG:-/app/config/zenoh_config.json5}"
required_files="$(mktemp /tmp/orchestra-stt-required-files.XXXXXX)"
trap 'rm -f "$required_files"' EXIT

echo "Checking required Sherpa STT models..."
if ! required_stt_files "$STT_PROFILE" >"$required_files"; then
    echo "ERROR: invalid STT_PROFILE '$STT_PROFILE'"
    echo "Valid values: en-vad-offline, vi-vad-offline"
    exit 1
fi
while IFS= read -r relative_path; do
    [ -n "$relative_path" ] || continue
    if [ ! -f "$STT_MODEL_ROOT/$relative_path" ]; then
        echo "ERROR: required STT model file missing: $relative_path"
        echo "Expected under: $STT_MODEL_ROOT"
        echo ""
        echo "Download models with:"
        echo "  make models"
        echo "  or"
        echo "  ./docker/scripts/download-models.sh"
        exit 1
    fi
done < "$required_files"

echo "✓ Sherpa STT profile '$STT_PROFILE' is available"

# Create a modified dataflow YAML with updated paths
echo "Updating dataflow YAML paths for container environment..."
cp /app/dataflow/orchestra-dataflow.yml /tmp/orchestra-dataflow.yml

# Update binary paths from ../target/release/ to /app/bin/
sed -i 's|path: ../target/release/|path: /app/bin/|g' /tmp/orchestra-dataflow.yml

echo "✓ Dataflow YAML updated"

# Display configuration
echo ""
echo "Configuration:"
echo "  - Entity ID: ${ENTITY_ID:-orchestra}"
echo "  - Zenoh Mode: ${ZENOH_MODE:-peer}"
echo "  - Zenoh Config: ${ORCHESTRA_ZENOH_CONFIG}"
echo "  - Socket.IO Port: ${SOCKET_IO_PORT:-3030}"
echo "  - Active Rovers: ${ACTIVE_ROVERS:-rover-kiwi}"
echo "  - STT Profile: ${STT_PROFILE}"
echo "  - STT Model Root: ${STT_MODEL_ROOT}"
echo ""

echo ""
echo "Starting Orchestra dataflow..."
echo "==================================================================="

# Start the dataflow locally as dora-rs best practice in docker
exec dora run /tmp/orchestra-dataflow.yml
