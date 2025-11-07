#!/bin/bash
# Download Sherpa-ONNX VITS TTS models for offline use on edge devices

set -e

CACHE_DIR="$HOME/.cache/sherpa-onnx"
MODEL_NAME="vits-piper-en_US-lessac-medium"
MODEL_DIR="$CACHE_DIR/$MODEL_NAME"

echo "=== Sherpa-ONNX VITS TTS Model Downloader ==="
echo ""
echo "Downloading lightweight VITS model for edge device TTS"
echo "Model: $MODEL_NAME"
echo ""

# Create directories
mkdir -p "$CACHE_DIR"

# Model URL
MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/${MODEL_NAME}.tar.bz2"
TAR_FILE="$CACHE_DIR/${MODEL_NAME}.tar.bz2"

# Check if model already exists
if [ -d "$MODEL_DIR" ]; then
    echo "✓ Model directory already exists: $MODEL_DIR"
    echo ""
    echo "Checking for required files..."

    required_files=("model.onnx" "tokens.txt" "lexicon.txt")
    all_exist=true

    for file in "${required_files[@]}"; do
        if [ -f "$MODEL_DIR/$file" ]; then
            echo "  ✓ $file"
        else
            echo "  ✗ $file (missing)"
            all_exist=false
        fi
    done

    if [ "$all_exist" = true ]; then
        echo ""
        echo "All model files are present. No download needed."
        echo ""
        echo "Model location: $MODEL_DIR"
        echo ""
        echo "You can now run the Sherpa TTS node!"
        exit 0
    else
        echo ""
        echo "Some files are missing. Re-downloading..."
        rm -rf "$MODEL_DIR"
    fi
fi

# Download model
echo "Downloading VITS model (~21MB compressed)..."
wget -O "$TAR_FILE" "$MODEL_URL"
echo "✓ Downloaded: $TAR_FILE"

# Extract model
echo ""
echo "Extracting model files..."
cd "$CACHE_DIR"
tar xf "${MODEL_NAME}.tar.bz2"
echo "✓ Extracted to: $MODEL_DIR"

# Rename model file if needed (sherpa-onnx releases sometimes use different names)
if [ -f "$MODEL_DIR/en_US-lessac-medium.onnx" ]; then
    mv "$MODEL_DIR/en_US-lessac-medium.onnx" "$MODEL_DIR/model.onnx"
    echo "✓ Renamed model file to model.onnx"
fi

# Clean up tar file
rm -f "$TAR_FILE"
echo "✓ Cleaned up archive file"

echo ""
echo "=== Download Complete ==="
echo ""
echo "Model files installed to:"
echo "  $MODEL_DIR/"
echo ""
echo "Contents:"
ls -lh "$MODEL_DIR"
echo ""
echo "Model details:"
echo "  - Type: VITS (Piper)"
echo "  - Language: English (US)"
echo "  - Voice: lessac"
echo "  - Quality: Medium (~21MB)"
echo "  - Optimized for: Edge devices, Raspberry Pi, limited resources"
echo ""
echo "You can now run the Sherpa TTS node without internet connection!"
echo ""
echo "Environment variable to use:"
echo "  TTS_MODEL_DIR=$MODEL_DIR"
