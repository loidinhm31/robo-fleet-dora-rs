# Models Directory

This directory contains AI models used by various Dora nodes:
- **YOLO models** for object detection (`object_detector` node)
- **Sherpa-ONNX models** for central VAD and offline speech-to-text
- **Whisper models** retained temporarily as rollback artifacts

---

# Sherpa-ONNX Models for Central Speech-to-Text

Run `make models` from the repository root. Downloads are pinned, extracted into
temporary directories, validated, then moved into place. Re-running the command
skips every complete model bundle.

The central recognizer uses this layout:

```text
models/.cache/sherpa-onnx/asr/
├── silero/silero_vad.onnx
├── icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/
│   ├── data/lang_bpe_500/tokens.txt
│   └── exp/
│       ├── encoder-epoch-30-avg-4.int8.onnx
│       ├── decoder-epoch-30-avg-4.onnx
│       └── joiner-epoch-30-avg-4.int8.onnx
└── sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/
    ├── encoder.int8.onnx
    ├── decoder.onnx
    ├── joiner.int8.onnx
    └── tokens.txt
```

Select exactly one startup profile:

| Profile | Language | Bundle |
|---|---|---|
| `en-vad-offline` | English | multidataset Zipformer; default |
| `vi-vad-offline` | Vietnamese | 30M int8 Zipformer |

Configuration is server-owned. Clients cannot submit model paths or change the
profile at runtime.

| Variable | Default | Valid values |
|---|---|---|
| `STT_PROFILE` | `en-vad-offline` | one of the two profiles above |
| `STT_MODEL_ROOT` | `models/.cache/sherpa-onnx/asr` | directory containing the documented layout |
| `STT_NUM_THREADS` | `2` | `1..=64` |
| `STT_VAD_THRESHOLD` | `0.5` | greater than `0`, less than `1` |
| `STT_VAD_MIN_SILENCE_SECONDS` | `0.25` | finite seconds in `0..=120` |
| `STT_VAD_MIN_SPEECH_SECONDS` | `0.25` | finite seconds, greater than `0` and at most `120` |
| `STT_VAD_MAX_SPEECH_SECONDS` | `8.0` | greater than minimum speech duration and at most `120` |
| `STT_DECODE_QUEUE_CAPACITY` | `8` | `1..=1024` |
| `STT_SAMPLE_RATE` | `16000` | fixed; incompatible overrides are rejected |
| `STT_VAD_WINDOW_SIZE` | `512` | fixed; incompatible overrides are rejected |

Use `make check-models` to report every required file. Startup validates the
selected profile before creating native Sherpa objects and returns concise errors
without exposing configured absolute paths.

Native smoke test after downloading models:

```bash
STT_PROFILE=en-vad-offline STT_MODEL_ROOT="$PWD/models/.cache/sherpa-onnx/asr" \
  cargo test -p central_speech_recognizer --test model_loading -- --ignored
# Repeat with STT_PROFILE=vi-vad-offline.
```

---

# Whisper Rollback Models

`models/.cache/ggml/ggml-base.bin` remains downloadable during the staged
migration so operators can roll back to the Phase 01 artifact. The current
`central_speech_recognizer` package and dataflow do not load Whisper. Phase 08
removes this artifact after the dual-profile validation gate passes.

---

# 🔍 YOLOv12 Models for Object Detection

## Quick Start

### Option 1: Download Pre-trained PyTorch Model and Export to ONNX (Recommended)

1. **Download YOLOv12n PyTorch model:**
```bash
cd models
curl -L -o yolo12n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt
```

2. **Install ultralytics (in a virtual environment):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install ultralytics
```

3. **Export to ONNX format:**
```bash
python export_yolo_to_onnx.py
```

This will create `yolo12n.onnx` in the current directory.

### Option 2: Using Python Directly

```python
from ultralytics import YOLO

# Load YOLOv12n model (will auto-download if not present)
model = YOLO('yolo12n.pt')

# Export to ONNX format with opset 14 for ONNX Runtime 1.16 compatibility
model.export(format='onnx', simplify=True, opset=14)
```

**Important**: The `opset=14` parameter is required for compatibility with ONNX Runtime 1.16.0. Without it, the model will use a newer ONNX IR version that isn't supported.

## Available YOLOv12 Models

All models are available from the Ultralytics assets repository:

| Model | Size | mAP | Speed (ms) | Download Link |
|-------|------|-----|------------|---------------|
| YOLOv12n | 6 MB | 39.8 | 1.4 | [yolo12n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt) |
| YOLOv12s | 12 MB | 47.0 | 2.2 | [yolo12s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt) |
| YOLOv12m | 28 MB | 51.6 | 4.5 | [yolo12m.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt) |
| YOLOv12l | 45 MB | 53.3 | 6.8 | [yolo12l.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt) |
| YOLOv12x | 62 MB | 54.3 | 9.2 | [yolo12x.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt) |

**Note:** Speed measured on NVIDIA T4 GPU with TensorRT FP16 precision.

## Model Information

- **YOLOv12** is an attention-centric object detection framework
- Supports 80 COCO dataset classes (person, car, dog, cat, etc.)
- Input size: 640×640 pixels
- Output format: `[batch, num_features, num_detections]` where `num_features = 4 (bbox) + 80 (classes)`

## ONNX Runtime Requirements

The rover vision crates in this repo are currently pinned to Rust `ort` `1.16.3`,
so use an ONNX Runtime `1.16.x` shared library.

### Install ONNX Runtime

**Automatic system-wide install** (recommended when you want the rover dataflow
default `/usr/local/lib/libonnxruntime.so` to work without setting
`ROVER_ORT_DYLIB_PATH`):

```bash
./models/scripts/download_onnxruntime.sh
```

This installs `libonnxruntime.so*` into `/usr/local/lib/` and runs `ldconfig`.
It requires `sudo`.

**Linux:**
```bash
# Download ONNX Runtime 1.16.3
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz

# Extract
tar -xzf onnxruntime-linux-x64-1.16.3.tgz

# Use the shared library path for the Rust rover nodes
export ROVER_ORT_DYLIB_PATH=/path/to/onnxruntime-linux-x64-1.16.3/lib/libonnxruntime.so
```

**Python-only alternative:**
```bash
pip install onnxruntime
```

For this repo's Rust rover nodes, `pip install onnxruntime` is not enough by
itself; they need a real `libonnxruntime.so` and an `ORT_DYLIB_PATH`
or `ROVER_ORT_DYLIB_PATH` pointing to it.

## Usage in Dora Dataflow

Add to your dataflow YAML (e.g., `web-dataflow.yml`):

```yaml
- id: object-detector
  build: cargo build --release -p object_detector
  path: target/release/object_detector
  inputs:
    frame: gst-camera/frame
  outputs:
    - detections
  env:
    MODEL_PATH: "models/yolo12n.onnx"
    CONFIDENCE_THRESHOLD: "0.5"
    NMS_THRESHOLD: "0.4"
    TARGET_CLASSES: "person,dog,cat"  # Optional: filter specific classes
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/yolo12n.onnx` | Path to ONNX model file |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence score (0.0-1.0) |
| `NMS_THRESHOLD` | `0.4` | Non-maximum suppression threshold (0.0-1.0) |
| `TARGET_CLASSES` | (empty) | Comma-separated class names to detect (e.g., "person,car,dog") |

## COCO Classes

The YOLOv12 models detect 80 object classes from the COCO dataset:

```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat,
dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack,
umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball,
kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair,
couch, potted plant, bed, dining table, toilet, tv, laptop, mouse,
remote, keyboard, cell phone, microwave, oven, toaster, sink,
refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
```

## Troubleshooting

### Model not found
- Ensure the model file is in the correct path specified by `MODEL_PATH`
- Check file permissions

### ONNX Runtime version mismatch
```
ort 2.0.0-rc.10 is not compatible with the ONNX Runtime binary found
```
**Solution:** Install an ONNX Runtime `1.16.x` shared library that matches the
repo's pinned `ort` crate (see ONNX Runtime Requirements above)

### Out of memory
- Use a smaller model (yolo12n instead of yolo12x)
- Reduce input resolution (requires model re-export)
- Enable CPU-only mode if GPU memory is limited

## References

- [Ultralytics YOLOv12 Documentation](https://docs.ultralytics.com/models/yolo12/)
- [ONNX Export Guide](https://docs.ultralytics.com/integrations/onnx/)
- [YOLOv12 GitHub Repository](https://github.com/sunsmarterjie/yolov12)
- [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
