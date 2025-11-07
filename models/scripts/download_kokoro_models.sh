#!/bin/bash
# Download Kokoro TTS models for offline use

set -e

MODEL_DIR="models/obj"
CACHE_DIR="$HOME/.cache/kokoros"

echo "=== Kokoro TTS Model Downloader ==="
echo ""

# Create directories
mkdir -p "$MODEL_DIR"
mkdir -p "$CACHE_DIR"

# Model URLs
ONNX_MODEL_URL="https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1.0.onnx"
VOICES_URL="https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices-v1.0.bin"

ONNX_MODEL_PATH="$MODEL_DIR/kokoro-v1.0.onnx"
VOICES_PATH="$MODEL_DIR/voices-v1.0.bin"

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

# Copy to cache directory for kokoro-tiny to use
echo ""
echo "Installing models to cache directory..."
cp "$ONNX_MODEL_PATH" "$CACHE_DIR/"
cp "$VOICES_PATH" "$CACHE_DIR/"
echo "✓ Models installed to: $CACHE_DIR"

echo ""
echo "=== Download Complete ==="
echo "Model files:"
echo "  - $ONNX_MODEL_PATH"
echo "  - $VOICES_PATH"
echo ""
echo "Cache location:"
echo "  - $CACHE_DIR/kokoro-v1.0.onnx"
echo "  - $CACHE_DIR/voices-v1.0.bin"
echo ""
echo "You can now run the TTS node without internet connection!"
