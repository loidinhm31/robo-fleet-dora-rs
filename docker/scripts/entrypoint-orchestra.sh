#!/bin/bash
set -e

echo "==================================================================="
echo "  Robo-Fleet Orchestra Container Starting"
echo "==================================================================="

# Verify models exist
echo "Checking for required models..."
if [ ! -f "/models/ggml/ggml-base.bin" ]; then
    echo "ERROR: Whisper model not found at /models/ggml/ggml-base.bin"
    echo ""
    echo "Please ensure the models directory is mounted correctly:"
    echo "  - Host: ./models/.cache/ggml/ggml-base.bin"
    echo "  - Container: /models/ggml/ggml-base.bin"
    echo ""
    echo "To download the model, run:"
    echo "  make models"
    echo "  or"
    echo "  ./docker/scripts/download-models.sh"
    exit 1
fi

echo "✓ Whisper model found"

# Create a modified dataflow YAML with updated paths
echo "Updating dataflow YAML paths for container environment..."
cp /app/dataflow/orchestra-dataflow.yml /tmp/orchestra-dataflow.yml

# Update binary paths from ../target/release/ to /app/bin/
sed -i 's|path: ../target/release/|path: /app/bin/|g' /tmp/orchestra-dataflow.yml

# Update Zenoh config path
sed -i "s|ZENOH_CONFIG: \"/home/loidinh/ws/robo-fleet-dora-rs/orchestra/zenoh_bridge/zenoh_config.json5\"|ZENOH_CONFIG: \"/app/config/zenoh_config.json5\"|g" /tmp/orchestra-dataflow.yml

# Update Whisper model path
sed -i 's|WHISPER_MODEL_PATH: "../models/.cache/ggml/ggml-base.bin"|WHISPER_MODEL_PATH: "/models/ggml/ggml-base.bin"|g' /tmp/orchestra-dataflow.yml

echo "✓ Dataflow YAML updated"

# Display configuration
echo ""
echo "Configuration:"
echo "  - Entity ID: ${ENTITY_ID:-orchestra}"
echo "  - Zenoh Mode: ${ZENOH_MODE:-peer}"
echo "  - Socket.IO Port: ${SOCKET_IO_PORT:-3030}"
echo "  - Active Rovers: ${ACTIVE_ROVERS:-rover-kiwi}"
echo "  - Whisper Model: ${WHISPER_MODEL_PATH:-/models/ggml/ggml-base.bin}"
echo ""

echo ""
echo "Starting Orchestra dataflow..."
echo "==================================================================="

# Start the dataflow locally as dora-rs best practice in docker
exec dora run /tmp/orchestra-dataflow.yml
