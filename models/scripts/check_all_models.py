#!/usr/bin/env python3
"""Check IR and opset versions of all ONNX models"""
import sys

try:
    import onnx
except ImportError:
    print("ERROR: onnx package not installed")
    print("Install with: pip install onnx")
    sys.exit(1)

models = {
    "YOLO": "../.cache/yolo/yolo12n.onnx",
    "OSNet": "../.cache/reid/osnet_x0_25.onnx",
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
print("ONNX Runtime Compatibility")
print("="*60)
print()
print("ONNX Runtime 1.16.3: Supports IR ≤ 9, Opset ≤ 18")
print("ONNX Runtime 1.19.0: Supports IR ≤ 10, Opset ≤ 21")
print()
print("Recommendation:")
print("  - If all models have IR ≤ 9: Keep 1.16.3")
print("  - If any model has IR = 10: Upgrade to 1.19.0")
