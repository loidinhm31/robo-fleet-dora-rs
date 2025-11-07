#!/usr/bin/env python3
"""
Export YOLOv12n PyTorch model to ONNX format.

This script requires ultralytics package to be installed:
    pip install ultralytics

Usage:
    python export_to_onnx.py
"""

import os
from pathlib import Path
from ultralytics import YOLO

def main():
    # Setup output directory
    cache_dir = Path.home() / ".cache" / "yolo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = cache_dir / "yolo12n.onnx"

    print("Loading YOLOv12n model...")
    model = YOLO('yolo12n.pt')

    print(f"Exporting to ONNX format with opset 14 (compatible with ONNX Runtime 1.16)...")
    print(f"Output directory: {cache_dir}")

    # Use opset 14 for ONNX IR version 9 compatibility
    model.export(format='onnx', simplify=True, opset=14)

    # Move the exported file to cache directory
    if Path("yolo12n.onnx").exists():
        Path("yolo12n.onnx").rename(output_path)
        print(f"Export complete! Model saved to: {output_path}")
    else:
        print(f"Warning: yolo12n.onnx not found in current directory")

    print("Note: Exported with opset 14 for ONNX Runtime 1.16 compatibility")

if __name__ == "__main__":
    main()
