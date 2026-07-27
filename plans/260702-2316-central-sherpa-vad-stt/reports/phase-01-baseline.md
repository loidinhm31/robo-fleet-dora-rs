# Phase 01 Whisper Baseline

Date: 2026-07-03

## Scope

Capture a small pre-change English baseline for the current `orchestra/central_speech_recognizer` Whisper path before Sherpa replacement work.

## Environment

- Repo: `/mnt/data/ws/sharing/robo-fleet-dora-rs`
- Model: `models/.cache/ggml/ggml-base.bin`
- Runtime: local CPU, 4 Whisper decode threads
- Corpus source: Google Translate TTS MP3 fetched at run time, converted to 16 kHz mono WAV with `ffmpeg`

## Corpus

| Label | Prompt | Duration ms |
|---|---|---:|
| A | `move forward` | 1296 |
| B | `turn left` | 1080 |
| C | `stop` | 840 |

## Method

1. Fetch a short English TTS clip per prompt.
2. Convert each clip to 16 kHz mono WAV.
3. Run a throwaway local benchmark harness that loads `ggml-base.bin`, decodes one clip, prints the recognized text, and records load/decode timing.
4. Wrap each run with `/usr/bin/time` to capture elapsed time, CPU %, and peak RSS.

## Results

| Label | Output text | Confidence | Model load ms | Decode ms | Elapsed s | CPU % | Peak RSS KiB |
|---|---|---:|---:|---:|---:|---:|---:|
| A | `Move forward.` | 0.90 | 93 | 1903 | 2.03 | 374 | 389240 |
| B | `Turn left.` | 0.90 | 93 | 1845 | 1.99 | 371 | 390304 |
| C | `` | 0.00 | 111 | 30 | 0.17 | 101 | 212996 |

## Notes

- Case C exposes a current Whisper constraint, not a harness defect. The engine reports: `input is too short - 830 ms < 1000 ms`.
- Current confidence is fabricated by the recognizer code (`0.9` for any non-empty segment), so it is not suitable as a future Sherpa contract field.
- These numbers include a cold model load per utterance because the harness benchmarks one file per process. Decode timing is separated explicitly so later Sherpa comparisons can use both cold and per-utterance figures.

## Reproduction

Audio generation:

```bash
curl -L -A 'Mozilla/5.0' \
  'https://translate.googleapis.com/translate_tts?ie=UTF-8&client=gtx&q=move%20forward&tl=en' \
  -o /tmp/phase01-whisper-move-forward.mp3
ffmpeg -y -i /tmp/phase01-whisper-move-forward.mp3 -ar 16000 -ac 1 /tmp/phase01-whisper-move-forward.wav
```

Benchmark:

```bash
cargo build --manifest-path /tmp/phase01-whisper-bench/Cargo.toml
/usr/bin/time -f 'elapsed_s=%e\nuser_s=%U\nsystem_s=%S\npeak_rss_kb=%M\ncpu_pct=%P' \
  /tmp/phase01-whisper-bench/target/debug/phase01-whisper-bench \
  /mnt/data/ws/sharing/robo-fleet-dora-rs/models/.cache/ggml/ggml-base.bin \
  /tmp/phase01-whisper-move-forward.wav
```
