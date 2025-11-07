#!/bin/bash
# Upgrade ONNX Runtime to version 1.19 (supports ONNX IR version 10)

set -e

echo "=========================================="
echo "Reinstall ONNX Runtime 1.16.3"
echo "=========================================="
echo ""

# Download ONNX Runtime 1.16.3
VERSION="1.19.2"
ARCH="linux-x64"
DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-${ARCH}-${VERSION}.tgz"
TAR_FILE="onnxruntime-${ARCH}-${VERSION}.tgz"
EXTRACT_DIR="onnxruntime-${ARCH}-${VERSION}"

echo "Downloading ONNX Runtime ${VERSION}..."
wget -O "$TAR_FILE" "$DOWNLOAD_URL"

echo "Extracting..."
tar -xzf "$TAR_FILE"

echo "Removing old ONNX Runtime library..."
sudo rm -f /usr/local/lib/libonnxruntime.so*

echo "Installing new ONNX Runtime library..."
sudo cp "$EXTRACT_DIR"/lib/libonnxruntime.so* /usr/local/lib/
sudo ldconfig

echo "Cleaning up..."
rm -rf "$TAR_FILE" "$EXTRACT_DIR"

echo ""
echo "=========================================="
echo "Upgrade Complete!"
echo "=========================================="
echo ""
echo "ONNX Runtime ${VERSION} installed"
echo ""
echo "Now export the OSNet model with opset 18 (IR v9 compatible):"
echo "  python3 export_osnet.py"
