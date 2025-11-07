# Models Directory

This directory contains AI models used by various Dora nodes:
- **YOLO models** for object detection (`object_detector` node)
- **Whisper models** for speech-to-text (`speech_recognizer` node)

---

# 🎤 Whisper Models for Speech-to-Text

## Quick Setup (Raspberry Pi 5 Optimized)

### Prerequisites

1. **Install CMake** (required to build whisper-rs):
```bash
# Arch/Manjaro
sudo pacman -S cmake

# Ubuntu/Debian
sudo apt install cmake build-essential

# Check installation
cmake --version
```

### Download Whisper Models

The `speech_recognizer` node uses Whisper.cpp quantized models for efficient on-device speech recognition.

**Recommended for Raspberry Pi 5:**

```bash
# Download Whisper tiny model (75 MB, ~1-2s inference for 5s audio)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin -O models/ggml-tiny.bin

# Verify download
ls -lh models/ggml-tiny.bin
```

**Alternative models:**

```bash
# Base model (142 MB, ~3-4s inference, better accuracy)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin -O models/ggml-base.bin

# Small model (466 MB, ~10s inference, even better accuracy - NOT recommended for RPi5)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin -O models/ggml-small.bin
```

## Whisper Model Comparison

| Model | Size | Speed (RPi5) | Accuracy | RAM Usage | Recommended |
|-------|------|--------------|----------|-----------|-------------|
| **ggml-tiny.bin** | 75 MB | 1-2s | Good | ~200 MB | ✅ Best for RPi5 |
| **ggml-base.bin** | 142 MB | 3-4s | Better | ~250 MB | ⚠️ Usable but slower |
| ggml-small.bin | 466 MB | ~10s | Excellent | ~500 MB | ❌ Too slow |
| ggml-medium.bin | 1.5 GB | ~30s | Excellent | ~1.5 GB | ❌ Too slow |
| ggml-large-v3.bin | 3.1 GB | ~60s | Best | ~3 GB | ❌ Too slow |

**Note:** Speed measured for 5 seconds of audio on Raspberry Pi 5 (4 cores at 2.4 GHz).

## Quantized Models

Whisper.cpp models are quantized to reduce size and improve inference speed:

| Model Type | Description | Size Reduction | Speed |
|------------|-------------|----------------|-------|
| **ggml-*.bin** | Standard quantization | 3-4x smaller | 3-4x faster |
| ggml-*-q5_0.bin | 5-bit quantization | 5x smaller | 5x faster |
| ggml-*-q8_0.bin | 8-bit quantization | 2x smaller | 2x faster |

For Raspberry Pi 5, the standard quantized models (ggml-*.bin) provide the best balance.

## Usage in Dora Dataflow

Add to your dataflow YAML (e.g., `web-dataflow.yml`):

```yaml
- id: speech-recognizer
  build: cargo build --release -p speech_recognizer
  path: target/release/speech_recognizer
  inputs:
    audio: audio-capture/audio
  outputs:
    - transcription
  env:
    WHISPER_MODEL_PATH: "models/ggml-tiny.bin"
    SAMPLE_RATE: "16000"
    BUFFER_DURATION_MS: "5000"
    CONFIDENCE_THRESHOLD: "0.5"
    ENERGY_THRESHOLD: "0.02"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_PATH` | `models/ggml-tiny.bin` | Path to Whisper model file |
| `SAMPLE_RATE` | `16000` | Audio sample rate (must match audio-capture) |
| `BUFFER_DURATION_MS` | `5000` | Buffer audio for X ms before transcription |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence to output transcription |
| `ENERGY_THRESHOLD` | `0.02` | Voice Activity Detection (VAD) threshold |

## Supported Languages

Whisper supports 99 languages. To use a specific language:

```yaml
# In speech_recognizer/src/main.rs, modify line 116:
params.set_language(Some("en"));  # English (default)
# params.set_language(Some("es"));  # Spanish
# params.set_language(Some("fr"));  # French
# params.set_language(Some("de"));  # German
# params.set_language(Some("vi"));  # Vietnamese
```

**Note:** Setting a specific language improves accuracy and speed by ~20%.

## Performance Tips

1. **Use tiny model** for real-time transcription on RPi5
2. **Reduce buffer duration** to 3-4 seconds for faster response (at cost of accuracy)
3. **Increase energy threshold** to 0.03-0.05 if background noise causes false triggers
4. **Set language explicitly** rather than auto-detect for 20% speed boost
5. **Use 4 threads** (default) - matches RPi5 CPU cores

## Troubleshooting

### Build fails with "cmake not found"
```
Solution: Install CMake (see Prerequisites above)
```

### Model not found error
```
ERROR: Whisper model not found at: models/ggml-tiny.bin
Solution: Download the model (see Download Whisper Models above)
```

### Slow transcription (>10s for 5s audio)
```
Solution: Use ggml-tiny.bin instead of larger models
```

### Low quality transcriptions
```
Solution:
- Check microphone quality (arecord -l)
- Reduce background noise
- Use ggml-base.bin for better accuracy (slower)
- Adjust ENERGY_THRESHOLD to filter out noise
```

### Audio format mismatch
```
Error: Expected Float32Array from audio_capture
Solution: Ensure audio-capture outputs Float32 at 16kHz mono
```

## References

- [Whisper.cpp GitHub](https://github.com/ggerganov/whisper.cpp)
- [Whisper.cpp Models](https://huggingface.co/ggerganov/whisper.cpp)
- [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)
- [whisper-rs Rust Bindings](https://github.com/tazz4843/whisper-rs)

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

The `object_detector` node requires ONNX Runtime 1.22.x or later.

### Install ONNX Runtime

**Linux:**
```bash
# Download ONNX Runtime 1.22.0
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz

# Extract
tar -xzf onnxruntime-linux-x64-1.22.0.tgz

# Set library path
export LD_LIBRARY_PATH=/path/to/onnxruntime-linux-x64-1.22.0/lib:$LD_LIBRARY_PATH
```

**Or install via pip:**
```bash
pip install onnxruntime
```

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
**Solution:** Install ONNX Runtime 1.22.x or later (see ONNX Runtime Requirements above)

### Out of memory
- Use a smaller model (yolo12n instead of yolo12x)
- Reduce input resolution (requires model re-export)
- Enable CPU-only mode if GPU memory is limited

## References

- [Ultralytics YOLOv12 Documentation](https://docs.ultralytics.com/models/yolo12/)
- [ONNX Export Guide](https://docs.ultralytics.com/integrations/onnx/)
- [YOLOv12 GitHub Repository](https://github.com/sunsmarterjie/yolov12)
- [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
