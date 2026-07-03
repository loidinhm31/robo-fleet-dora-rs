#!/bin/bash
# Download all required ML models for robo-fleet

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models/.cache"
export MODELS_DIR

. "$SCRIPT_DIR/sherpa-stt-profile-files.sh"

echo "==================================================================="
echo "  Robo-Fleet Model Download Script"
echo "==================================================================="
echo "Models will be downloaded to: $MODELS_DIR"
echo ""

# Create model directories
mkdir -p "$MODELS_DIR/yolo"
mkdir -p "$MODELS_DIR/reid"
mkdir -p "$MODELS_DIR/sherpa-onnx"
ASR_MODELS_DIR="$MODELS_DIR/sherpa-onnx/asr"
mkdir -p "$ASR_MODELS_DIR/silero"

have_required_files() {
    local base_dir="$1"
    shift
    local relative_path
    for relative_path in "$@"; do
        [ -s "$base_dir/$relative_path" ] || return 1
    done
}

download_file() {
    local url="$1"
    local destination="$2"
    local temporary="${destination}.part"

    if [ -s "$destination" ]; then
        echo "  ✓ $(basename "$destination") already exists, skipping download"
        return
    fi

    rm -f "$temporary"
    wget -q --show-progress -O "$temporary" "$url"
    [ -s "$temporary" ] || { rm -f "$temporary"; return 1; }
    mv "$temporary" "$destination"
}

download_asr_bundle() {
    local name="$1"
    local url="$2"
    shift 2
    local destination="$ASR_MODELS_DIR/$name"
    local archive="$ASR_MODELS_DIR/.${name}.tar.bz2.part"
    local extract_dir="$ASR_MODELS_DIR/.${name}.extract.$$"

    if have_required_files "$destination" "$@"; then
        echo "  ✓ $name already valid, skipping download"
        return
    fi

    rm -rf "$extract_dir"
    rm -f "$archive"
    mkdir -p "$extract_dir"
    if ! wget -q --show-progress -O "$archive" "$url"; then
        rm -rf "$extract_dir"
        rm -f "$archive"
        return 1
    fi
    if ! tar xjf "$archive" -C "$extract_dir"; then
        rm -rf "$extract_dir"
        rm -f "$archive"
        return 1
    fi
    if ! have_required_files "$extract_dir/$name" "$@"; then
        echo "  ✗ $name archive did not contain the expected files"
        rm -rf "$extract_dir"
        rm -f "$archive"
        return 1
    fi

    rm -rf "$destination"
    mv "$extract_dir/$name" "$destination"
    rm -rf "$extract_dir"
    rm -f "$archive"
    echo "  ✓ $name downloaded and validated"
}

# =============================================================================
# 1. Sherpa-ONNX Silero VAD (for Orchestra speech recognition)
# =============================================================================
echo ""
echo "[1/6] Downloading Sherpa-ONNX Silero VAD..."
download_file \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" \
    "$ASR_MODELS_DIR/silero/silero_vad.onnx"

# =============================================================================
# 2. English offline Zipformer ASR bundle
# =============================================================================
echo ""
echo "[2/6] Downloading English offline Zipformer ASR bundle..."
download_asr_bundle "$EN_ASR_BUNDLE" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${EN_ASR_BUNDLE}.tar.bz2" \
    "exp/encoder-epoch-30-avg-4.int8.onnx" \
    "exp/decoder-epoch-30-avg-4.onnx" \
    "exp/joiner-epoch-30-avg-4.int8.onnx" \
    "data/lang_bpe_500/tokens.txt"

# =============================================================================
# 3. Vietnamese offline Zipformer ASR bundle
# =============================================================================
echo ""
echo "[3/6] Downloading Vietnamese offline Zipformer ASR bundle..."
download_asr_bundle "$VI_ASR_BUNDLE" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${VI_ASR_BUNDLE}.tar.bz2" \
    "encoder.int8.onnx" \
    "decoder.onnx" \
    "joiner.int8.onnx" \
    "tokens.txt"

# =============================================================================
# 4. Sherpa-ONNX TTS Model (for Rover TTS)
# =============================================================================
echo ""
echo "[4/6] Downloading Sherpa-ONNX VITS TTS model (~21 MB)..."
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
# 5. YOLO Model (requires PyTorch export on x86_64)
# =============================================================================
echo ""
echo "[5/6] Exporting YOLO model..."
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
# 6. OSNet ReID Model (requires PyTorch export on x86_64)
# =============================================================================
echo ""
echo "[6/6] Exporting OSNet ReID model..."
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

if [ -d "$MODELS_DIR/sherpa-onnx/vits-piper-en_US-lessac-medium" ]; then
    echo "  ✓ Sherpa-ONNX TTS (Rover): $MODELS_DIR/sherpa-onnx/vits-piper-en_US-lessac-medium"
else
    echo "  ✗ Sherpa-ONNX TTS (Rover): MISSING"
    MODELS_READY=false
fi

if [ -s "$ASR_MODELS_DIR/silero/silero_vad.onnx" ]; then
    echo "  ✓ Silero VAD (Orchestra): $ASR_MODELS_DIR/silero/silero_vad.onnx"
else
    echo "  ✗ Silero VAD (Orchestra): MISSING"
    MODELS_READY=false
fi

if have_required_files "$ASR_MODELS_DIR/$EN_ASR_BUNDLE" \
    "exp/encoder-epoch-30-avg-4.int8.onnx" "exp/decoder-epoch-30-avg-4.onnx" \
    "exp/joiner-epoch-30-avg-4.int8.onnx" "data/lang_bpe_500/tokens.txt"; then
    echo "  ✓ English ASR (Orchestra): $ASR_MODELS_DIR/$EN_ASR_BUNDLE"
else
    echo "  ✗ English ASR (Orchestra): MISSING"
    MODELS_READY=false
fi

if have_required_files "$ASR_MODELS_DIR/$VI_ASR_BUNDLE" \
    "encoder.int8.onnx" "decoder.onnx" "joiner.int8.onnx" "tokens.txt"; then
    echo "  ✓ Vietnamese ASR (Orchestra): $ASR_MODELS_DIR/$VI_ASR_BUNDLE"
else
    echo "  ✗ Vietnamese ASR (Orchestra): MISSING"
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
