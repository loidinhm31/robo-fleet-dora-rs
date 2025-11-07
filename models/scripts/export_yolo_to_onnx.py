#!/usr/bin/env python3
"""
Export YOLOv12n PyTorch model to ONNX format.

This script requires ultralytics package to be installed:
    pip install ultralytics

Usage:
    python export_to_onnx.py
"""

from ultralytics import YOLO

def main():
    print("Loading YOLOv12n model...")
    model = YOLO('yolo12n.pt')

    print("Exporting to ONNX format with opset 14 (compatible with ONNX Runtime 1.16)...")
    # Use opset 14 for ONNX IR version 9 compatibility
    model.export(format='onnx', simplify=True, opset=14)

    print("Export complete! Model saved as: yolo12n.onnx")
    print("Note: Exported with opset 14 for ONNX Runtime 1.16 compatibility")

if __name__ == "__main__":
    main()
