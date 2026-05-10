#!/bin/bash
# Download and convert OSNet ReID model to ONNX format

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${HOME}/.cache/reid"
MODEL_NAME="osnet_x0_25"
ONNX_FILE="${CACHE_DIR}/${MODEL_NAME}.onnx"

echo "========================================="
echo "OSNet ReID Model Download & Export"
echo "========================================="
echo ""

# Create cache directory
mkdir -p "${CACHE_DIR}"

# Check if model already exists
if [ -f "${ONNX_FILE}" ]; then
    echo "✓ OSNet model already exists at: ${ONNX_FILE}"
    echo ""
    echo "Model info:"
    ls -lh "${ONNX_FILE}"
    exit 0
fi

echo "Setting up Python virtual environment..."
cd "${SCRIPT_DIR}"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --quiet torch torchvision torchreid onnx gdown tensorboard onnxscript

echo ""
echo "Creating export script..."

# Create Python script to download and export OSNet model
cat > export_osnet.py << 'PYTHON_SCRIPT'
import torch
import torchreid
import onnx
from pathlib import Path
import sys

# Model configuration
MODEL_NAME = "osnet_x0_25"
OUTPUT_DIR = Path.home() / ".cache/reid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ONNX_PATH = OUTPUT_DIR / f"{MODEL_NAME}.onnx"

print(f"Loading {MODEL_NAME} model...")
model = torchreid.models.build_model(
    name=MODEL_NAME,
    num_classes=1000,  # Default number of classes
    pretrained=True,
    loss="softmax"
)

# Set to evaluation mode
model.eval()

print("Exporting to ONNX format...")

# Create dummy input (batch_size=1, channels=3, height=256, width=128)
# Standard input size for person ReID models
dummy_input = torch.randn(1, 3, 256, 128)

# Export using simpler method compatible with ORT 1.16.3
print("Using simplified export for compatibility with ONNX Runtime 1.16.3...")
with torch.no_grad():
    # Disable dynamo export (new default in PyTorch 2.x)
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        export_params=True,
        opset_version=12,  # Opset 12 -> IR version 7 (compatible with ORT 1.16.3)
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        dynamo=False  # Disable new dynamo-based export
    )

print(f"✓ Model exported successfully to: {ONNX_PATH}")

# Verify the ONNX model
try:
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verification passed")

    # Print model info
    file_size = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"\nModel info:")
    print(f"  - Size: {file_size:.2f} MB")
    print(f"  - Input shape: (batch, 3, 256, 128)")
    print(f"  - Output shape: (batch, 512)")
    print(f"  - Feature dimension: 512")

except Exception as e:
    print(f"✗ ONNX model verification failed: {e}")
    sys.exit(1)

print("")
print("========================================")
print("OSNet Model Download Complete!")
print("========================================")
print("")
print("Usage in rover-kiwi dataflow:")
print("  MODEL_PATH: ${HOME}/.cache/reid/osnet_x0_25.onnx")
print("  INPUT_SIZE: 256x128 (HxW)")
print("  OUTPUT_DIM: 512 features")
print("")
PYTHON_SCRIPT

echo "Running export script..."
python export_osnet.py

# Clean up
rm export_osnet.py

echo ""
echo "Done! OSNet model is ready at: ${ONNX_FILE}"
