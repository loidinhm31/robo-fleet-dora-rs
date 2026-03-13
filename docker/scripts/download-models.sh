#!/bin/bash
# Download all required ML models for robo-fleet

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models/.cache"

echo "==================================================================="
echo "  Robo-Fleet Model Download Script"
echo "==================================================================="
echo "Models will be downloaded to: $MODELS_DIR"
echo ""

# Create model directories
mkdir -p "$MODELS_DIR/ggml"
mkdir -p "$MODELS_DIR/yolo"
mkdir -p "$MODELS_DIR/reid"
mkdir -p "$MODELS_DIR/sherpa-onnx"

# =============================================================================
# 1. Whisper GGML Model (for Orchestra speech recognition)
# =============================================================================
echo "[1/4] Downloading Whisper GGML tiny model (~75 MB)..."
if [ -f "$MODELS_DIR/ggml/ggml-tiny.bin" ]; then
    echo "  ✓ Whisper model already exists, skipping download"
else
    wget -O "$MODELS_DIR/ggml/ggml-tiny.bin" \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
    echo "  ✓ Whisper model downloaded"
fi

# =============================================================================
# 2. Sherpa-ONNX TTS Model (for Rover TTS)
# =============================================================================
echo ""
echo "[2/4] Downloading Sherpa-ONNX VITS TTS model (~21 MB)..."
if [ -d "$MODELS_DIR/sherpa-onnx/vits-piper-en_US-lessac-medium" ]; then
    echo "  ✓ Sherpa-ONNX model already exists, skipping download"
else
    cd "$MODELS_DIR/sherpa-onnx"
    wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2
    tar xf vits-piper-en_US-lessac-medium.tar.bz2
    rm vits-piper-en_US-lessac-medium.tar.bz2
    cd "$PROJECT_ROOT"
    echo "  ✓ Sherpa-ONNX model downloaded and extracted"
fi

# =============================================================================
# 3. YOLO Model (requires PyTorch export)
# =============================================================================
echo ""
echo "[3/4] Checking YOLO model..."
if [ -f "$MODELS_DIR/yolo/yolo12n.onnx" ]; then
    echo "  ✓ YOLO model already exists"
else
    echo "  ⚠ YOLO model NOT found"
    echo ""
    echo "  The YOLO model needs to be exported manually from PyTorch weights."
    echo "  To export the YOLO model:"
    echo ""
    echo "    cd $PROJECT_ROOT/models/scripts"
    echo "    python3 -m venv venv"
    echo "    source venv/bin/activate"
    echo "    pip install ultralytics"
    echo "    python3 export_yolo_to_onnx.py"
    echo ""
    echo "  The exported model will be saved to:"
    echo "    $MODELS_DIR/yolo/yolo12n.onnx"
    echo ""
fi

# =============================================================================
# 4. OSNet ReID Model (requires PyTorch export)
# =============================================================================
echo ""
echo "[4/4] Checking OSNet ReID model..."
if [ -f "$MODELS_DIR/reid/osnet_x0_25.onnx" ]; then
    echo "  ✓ OSNet ReID model already exists"
else
    echo "  ⚠ OSNet ReID model NOT found"
    echo ""
    echo "  The OSNet model needs to be downloaded and exported."
    echo "  To download and export the OSNet model:"
    echo ""
    echo "    cd $PROJECT_ROOT/models/scripts"
    echo "    ./download_osnet_model.sh"
    echo ""
    echo "  The script will create a Python venv, download the model,"
    echo "  and export it to ONNX format at:"
    echo "    $MODELS_DIR/reid/osnet_x0_25.onnx"
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "==================================================================="
echo "  Model Download Summary"
echo "==================================================================="

MODELS_READY=true

if [ -f "$MODELS_DIR/ggml/ggml-tiny.bin" ]; then
    echo "  ✓ Whisper (Orchestra): $MODELS_DIR/ggml/ggml-tiny.bin"
else
    echo "  ✗ Whisper (Orchestra): MISSING"
    MODELS_READY=false
fi

if [ -d "$MODELS_DIR/sherpa-onnx/vits-piper-en_US-lessac-medium" ]; then
    echo "  ✓ Sherpa-ONNX TTS (Rover): $MODELS_DIR/sherpa-onnx/vits-piper-en_US-lessac-medium"
else
    echo "  ✗ Sherpa-ONNX TTS (Rover): MISSING"
    MODELS_READY=false
fi

if [ -f "$MODELS_DIR/yolo/yolo12n.onnx" ]; then
    echo "  ✓ YOLO (Rover): $MODELS_DIR/yolo/yolo12n.onnx"
else
    echo "  ✗ YOLO (Rover): MISSING - requires manual export"
    MODELS_READY=false
fi

if [ -f "$MODELS_DIR/reid/osnet_x0_25.onnx" ]; then
    echo "  ✓ OSNet ReID (Rover): $MODELS_DIR/reid/osnet_x0_25.onnx"
else
    echo "  ✗ OSNet ReID (Rover): MISSING - requires manual export"
    MODELS_READY=false
fi

echo ""
if [ "$MODELS_READY" = true ]; then
    echo "All models are ready! You can now build and run the Docker containers."
    echo ""
    echo "Next steps:"
    echo "  make build-orchestra    # Build orchestra image"
    echo "  make build-rover        # Build rover image"
    echo "  make up-orchestra       # Start orchestra"
    echo "  make up-rover           # Start rover"
else
    echo "Some models are missing. Please follow the instructions above to"
    echo "export the required models before running the Docker containers."
fi

echo "==================================================================="
