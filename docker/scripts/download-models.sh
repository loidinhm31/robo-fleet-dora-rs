#!/bin/bash
# Download all required ML models for robo-fleet

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models/.cache"
export MODELS_DIR

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
echo "[1/4] Downloading Whisper GGML base model (~142 MB)..."
if [ -f "$MODELS_DIR/ggml/ggml-base.bin" ]; then
    echo "  ✓ Whisper model already exists, skipping download"
else
    wget -O "$MODELS_DIR/ggml/ggml-base.bin" \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
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
# Architecture check — PyTorch model export runs on x86_64 only.
# On ARM (Raspberry Pi) the models must be pre-exported on the workstation
# and copied over (or volume-mounted via Docker).
# =============================================================================
ARCH=$(uname -m)
CAN_EXPORT=true
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "armv7l" ]; then
    CAN_EXPORT=false
fi

# =============================================================================
# 3. YOLO Model (requires PyTorch export on x86_64)
# =============================================================================
echo ""
echo "[3/4] Exporting YOLO model..."
if [ -f "$MODELS_DIR/yolo/yolo12n.onnx" ]; then
    echo "  ✓ YOLO model already exists, skipping export"
elif [ "$CAN_EXPORT" = false ]; then
    echo "  ⚠ ARM architecture detected ($ARCH) — PyTorch export must run on x86_64"
    echo "    On your workstation: make models"
    echo "    Then copy to Pi:     rsync -av models/.cache/yolo/ raspb4@<pi-ip>:~/WS/robo-fleet-dora-rs/models/.cache/yolo/"
elif ! command -v python3 &>/dev/null; then
    echo "  ✗ python3 not found — cannot export YOLO model"
    echo "    Install Python 3 and re-run, or export manually: cd $PROJECT_ROOT/models/scripts && python3 export_yolo_to_onnx.py"
else
    echo "  Setting up Python venv for YOLO export..."
    VENV_DIR="$PROJECT_ROOT/models/scripts/venv"
    [ -d "$VENV_DIR" ] || python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet ultralytics
    "$VENV_DIR/bin/python" "$PROJECT_ROOT/models/scripts/export_yolo_to_onnx.py"
    echo "  ✓ YOLO model exported"
fi

# =============================================================================
# 4. OSNet ReID Model (requires PyTorch export on x86_64)
# =============================================================================
echo ""
echo "[4/4] Exporting OSNet ReID model..."
if [ -f "$MODELS_DIR/reid/osnet_x0_25.onnx" ]; then
    echo "  ✓ OSNet ReID model already exists, skipping export"
elif [ "$CAN_EXPORT" = false ]; then
    echo "  ⚠ ARM architecture detected ($ARCH) — PyTorch export must run on x86_64"
    echo "    On your workstation: make models"
    echo "    Then copy to Pi:     rsync -av models/.cache/reid/ raspb4@<pi-ip>:~/WS/robo-fleet-dora-rs/models/.cache/reid/"
elif ! command -v python3 &>/dev/null; then
    echo "  ✗ python3 not found — cannot export OSNet model"
    echo "    Install Python 3 and re-run, or export manually: cd $PROJECT_ROOT/models/scripts && ./download_osnet_model.sh"
else
    echo "  Setting up Python venv for OSNet export..."
    VENV_DIR="$PROJECT_ROOT/models/scripts/venv"
    [ -d "$VENV_DIR" ] || python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet torch torchvision torchreid onnx gdown tensorboard onnxscript

    echo "  Downloading and exporting OSNet x0.25..."
    "$VENV_DIR/bin/python" - <<'PYTHON'
import torch
import torchreid
import onnx
from pathlib import Path
import os, sys

OUTPUT_DIR = Path(os.environ["MODELS_DIR"]) / "reid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ONNX_PATH = OUTPUT_DIR / "osnet_x0_25.onnx"

print("  Loading osnet_x0_25 pretrained model...")
model = torchreid.models.build_model(
    name="osnet_x0_25", num_classes=1000, pretrained=True, loss="softmax"
)
model.eval()

dummy_input = torch.randn(1, 3, 256, 128)
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        export_params=True,
        opset_version=12,       # opset 12 → ONNX IR v7 (ORT 1.16+ compatible)
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,           # disable dynamo to avoid IR v10 output
    )

onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)
print(f"  ✓ Exported to {ONNX_PATH} (IR version {onnx_model.ir_version})")
PYTHON
    echo "  ✓ OSNet ReID model exported"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "==================================================================="
echo "  Model Download Summary"
echo "==================================================================="

MODELS_READY=true

if [ -f "$MODELS_DIR/ggml/ggml-base.bin" ]; then
    echo "  ✓ Whisper (Orchestra): $MODELS_DIR/ggml/ggml-base.bin"
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
