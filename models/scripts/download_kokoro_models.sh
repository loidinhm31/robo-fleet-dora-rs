#!/bin/bash
# Download Kokoro TTS models for offline use

set -e

CACHE_DIR="$HOME/.cache/kokoro"

echo "=== Kokoro TTS Model Downloader ==="
echo ""

# Create cache directory
mkdir -p "$CACHE_DIR"

# Model URLs
ONNX_MODEL_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v1.0.onnx"
VOICES_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices-v1.0.bin"

```
ONNX_MODEL_PATH="$CACHE_DIR/kokoro-v1.0.onnx"
VOICES_PATH="$CACHE_DIR/voices-v1.0.bin"

# Download ONNX model if not exists
if [ -f "$ONNX_MODEL_PATH" ]; then
    echo "✓ ONNX model already exists: $ONNX_MODEL_PATH"
else
    echo "Downloading ONNX model (~87MB)..."
    wget -O "$ONNX_MODEL_PATH" "$ONNX_MODEL_URL"
    echo "✓ Downloaded: $ONNX_MODEL_PATH"
fi

# Download voices if not exists
if [ -f "$VOICES_PATH" ]; then
    echo "✓ Voices file already exists: $VOICES_PATH"
else
    echo "Downloading voices data..."
    wget -O "$VOICES_PATH" "$VOICES_URL"
    echo "✓ Downloaded: $VOICES_PATH"
fi

echo ""
echo "=== Download Complete ==="
echo ""
echo "Model files installed to:"
echo "  - $ONNX_MODEL_PATH"
echo "  - $VOICES_PATH"
echo ""
echo "You can now run the TTS node without internet connection!"
