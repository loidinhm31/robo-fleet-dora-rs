#!/bin/bash
# Install the repo's pinned ONNX Runtime shared library into /usr/local/lib.

set -e

echo "=========================================="
echo "Install ONNX Runtime"
echo "=========================================="
echo ""

# Current rover vision crates use Rust ort 1.16.3, so keep the runtime on 1.16.x
# unless a caller intentionally overrides it.
VERSION="${ORT_VERSION:-1.16.3}"
ARCH="linux-x64"
DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-${ARCH}-${VERSION}.tgz"
CACHE_DIR="$HOME/.cache/onnxruntime-downloads"
TAR_FILE="$CACHE_DIR/onnxruntime-${ARCH}-${VERSION}.tgz"
EXTRACT_DIR="$CACHE_DIR/onnxruntime-${ARCH}-${VERSION}"

# Create cache directory
mkdir -p "$CACHE_DIR"

echo "Downloading ONNX Runtime ${VERSION}..."
wget -O "$TAR_FILE" "$DOWNLOAD_URL"

echo "Extracting..."
rm -rf "$EXTRACT_DIR"
tar -xzf "$TAR_FILE" -C "$CACHE_DIR"

echo "Removing old ONNX Runtime library..."
sudo rm -f /usr/local/lib/libonnxruntime.so*

echo "Installing new ONNX Runtime library..."
sudo cp "$EXTRACT_DIR"/lib/libonnxruntime.so* /usr/local/lib/
sudo ldconfig

echo "Cleaning up..."
rm -rf "$TAR_FILE" "$EXTRACT_DIR"

echo ""
echo "=========================================="
echo "Install Complete!"
echo "=========================================="
echo ""
echo "ONNX Runtime ${VERSION} installed"
