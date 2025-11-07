#!/usr/bin/env python3
"""Check IR and opset versions of all ONNX models"""
import sys
from pathlib import Path

try:
    import onnx
except ImportError:
    print("ERROR: onnx package not installed")
    print("Install with: pip install onnx")
    sys.exit(1)

# Define all project models in ${HOME}/.cache
home = Path.home()
models = {
    "YOLO": home / ".cache/yolo/yolo12n.onnx",
    "OSNet": home / ".cache/reid/osnet_x0_25.onnx",
    "Kokoro TTS": home / ".cache/kokoro/kokoro-v1.0.onnx",
    "Sherpa VITS TTS": home / ".cache/sherpa-onnx/vits-piper-en_US-lessac-medium/model.onnx",
}

# Non-ONNX model files to check
other_files = {
    "Kokoro Voices": home / ".cache/kokoro/voices-v1.0.bin",
    "Sherpa Tokens": home / ".cache/sherpa-onnx/vits-piper-en_US-lessac-medium/tokens.txt",
    "Sherpa Lexicon": home / ".cache/sherpa-onnx/vits-piper-en_US-lessac-medium/lexicon.txt",
}

print("="*60)
print("ONNX Model Version Check")
print("="*60)
print()

for name, path in models.items():
    try:
        model = onnx.load(path)
        ir_version = model.ir_version
        opset_version = model.opset_import[0].version

        print(f"{name} ({path}):")
        print(f"  IR version: {ir_version}")
        print(f"  Opset version: {opset_version}")

        # Check compatibility
        if ir_version <= 9:
            print(f"  ✓ Compatible with ONNX Runtime 1.16.3")
        else:
            print(f"  ✗ Requires ONNX Runtime 1.17+ (IR version {ir_version})")

        print()
    except FileNotFoundError:
        print(f"{name}: Model not found at {path}")
        print()
    except Exception as e:
        print(f"{name}: Error loading model - {e}")
        print()

print("="*60)
print("Other Model Files Check")
print("="*60)
print()

for name, path in other_files.items():
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{name}:")
        print(f"  Path: {path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  ✓ File exists")
    else:
        print(f"{name}:")
        print(f"  Path: {path}")
        print(f"  ✗ File not found")
    print()

print("="*60)
print("ONNX Runtime Compatibility")
print("="*60)
print()
print("ONNX Runtime 1.16.3: Supports IR ≤ 9, Opset ≤ 18")
print("ONNX Runtime 1.19.0: Supports IR ≤ 10, Opset ≤ 21")
print()
print("Recommendation:")
print("  - If all models have IR ≤ 9: Keep 1.16.3")
print("  - If any model has IR = 10: Upgrade to 1.19.0")
print()
print("="*60)
print("Download Scripts")
print("="*60)
print()
print("To download missing models, run:")
print("  - YOLO: python models/scripts/export_yolo_to_onnx.py")
print("  - Kokoro TTS: ./models/scripts/download_kokoro_models.sh")
print("  - Sherpa VITS TTS: ./models/scripts/download_sherpa_vits_models.sh")
