---
title: "Audio Capture: Silent Frames + arecord-l Auto-Detection"
description: "Root-cause: audio_capture emits valid-looking frames (count/bytes ok) but they are SILENCE. cpal default_input_device picks the wrong (muted HD-Audio) card; CHANNELS=1 is incompatible with the 2-channel-only USB camera mic. Capture metrics can't see silence. Fix: port arecord-l auto-detect from check-usb-microphone.py, add channel capability probe + downmix, add signal-level observability."
status: completed
priority: P1
effort: ~6h
branch: main
tags: [audio, capture, debugging, observability, rover-kiwi]
created: 2026-07-02
updated: 2026-07-02
---

# Audio Capture: Silent Frames + arecord-l Auto-Detection

## Symptom

`audio_capture` log looks healthy but no real audio plays in the browser:

```
audio-capture: 2026-07-01T17:35:31.754452Z INFO metric="audio_pipeline" stage="capture"
  stream_id=33be1888-... count=100 bytes=320000 drops=0 errors=0
  p50_us=111 p95_us=148 p99_us=172 max_us=181
```

Frames flow end-to-end (capture -> converter -> web_bridge -> browser) but the
browser plays silence.

## TL;DR / Verdict

The capture log proves **bytes are moving**, not that **sound is captured**.
Two compounding defects produce silent-but-valid frames:

1. **Wrong device** -- `cpal::default_input_device()` likely resolves to the
   HD-Audio card (muted built-in mic), not the USB camera mic.
2. **Channel mismatch** -- the USB camera mic is **2-channel-only**, but the
   dataflow requests `CHANNELS=1`; a raw `hw` open at mono **fails**
   (`Channels count non available`).

The Python test (`scripts/check-usb-microphone.py`, run live) **proves the
hardware is good** at `48000Hz/2ch` via `hw:0,0`. The Rust node just never
targets it correctly.

## Pipeline Traced (transport is correct -- problem is upstream)

```
audio_capture (cpal->ringbuf->Float32Array, f32le 16k mono 800samp, 50ms tick)
  -> audio_converter (f32le->s16le, BinaryArray "audio_output")
  -> web_bridge     (validates S16LE len, emits binary "audio_frame" to browser)
```

- `rover-kiwi/rover-kiwi-direct-dataflow.yml`:
  `audio-converter/audio_output` -> `web-bridge/audio_frame` OK
- `common/web_bridge/src/main.rs`: `ClientState.audio_enabled` defaults `true`
  and is **never** disabled -> frames **are** emitted to the browser.
- `validate_browser_pcm_payload` requires `S16Le` + exact length; the converter
  produces exactly that. **Transport is not the failure.**

## Evidence (3 live probes)

### Probe 1 -- capture log decoded

- `count=100` over 5s window = 20 Hz = matches `tick: dora/timer/millis/50` OK
- `bytes=320000` = 100 x 800 x 4B (f32) OK -- 800 samples/frame = 50ms @16k OK
- `drops=0 errors=0` -> ringbuf never overflowed, stream opened.
- **But these metrics count bytes, not signal.** `grep rms|silence|peak` in
  `rover-kiwi/audio_capture/src/main.rs` = **none**. Only `debug`-level
  min/max for the first 5 frames. At `RUST_LOG=info` there is **zero signal
  observability**.

### Probe 2 -- `arecord -l` (live)

```
card 0: Camera [PC-LM1E Camera], device 0: USB Audio [USB Audio]   <- the good mic
card 2: Generic_1 [HD-Audio Generic], device 0: ALC245 Analog       <- likely cpal's "default"
```

No card 1. ALSA `default` commonly maps to the HD-Audio card -> built-in mic,
often muted at the mixer -> **silence**.

### Probe 3 -- `arecord --dump-hw-params -D hw:0,0 -r 16000 -c 1` (live)

```
CHANNELS: 2          <- USB mic is STEREO ONLY
RATE: [8000 48000]   <- 16k is within range OK
-> arecord: set_params:1404: Channels count non available   <- mono request FAILS on raw hw
```

The dataflow sets `CHANNELS: "1"`. On the raw `hw` device this **cannot open**.
cpal either (a) fails -> silent mode (but then `count` would be 0,
contradicting the log), or (b) silently opens a *different* device that
supports 1ch (the HD-Audio card) -> frames flow **with silence**. Scenario (b)
matches the symptom exactly.

## Root Causes

### RC-1 -- Device selection has no auto-detection

`try_open_input_stream` falls back to `default_input_device()` when
`AUDIO_DEVICE` is unset. That picks the wrong card. The existing
`AUDIO_DEVICE` substring-match is **manual** and the dataflow leaves it
**commented out** (`# AUDIO_DEVICE: "USB Audio"`). No `arecord -l`-style
enumeration/preference.

### RC-2 -- Channel config incompatible with the USB mic

`CHANNELS=1` vs hardware `CHANNELS=2`. Needs either ALSA `plug` (software
downmix) or capture at 2ch and downmix in Rust. cpal's ALSA backend may not
insert `plug` automatically for a named `hw:` device.

### RC-3 -- Capture metrics can't see silence (observability gap)

`MetricWindow` records count/bytes/drops/errors/p50-p99 latency. **No
RMS/peak/silence-ratio.** An operator cannot tell "healthy stream of silence"
from "healthy stream of audio." This is why the bug hid behind a green log.

### Browser side (not verifiable here)

`audio-timeline-scheduler.ts` / `useAudioStream` live in a separate
`packages/ui` repo **not present** in this workspace. Even if the browser
scheduler were perfect, silent capture => no sound. Fix capture first.

## Why the log lied

`count=100 bytes=320000 drops=0 errors=0` => "the pipe is full." It says
**nothing** about acoustic content. A 16kHz stream of `0.0f` samples produces
identical metrics to a stream of speech. The node has no signal-level check,
so silence is indistinguishable from success.

## Enhancement Design (NOT yet implemented)

Mirror `scripts/check-usb-microphone.py::detect_usb_mic()`.

### A. Device auto-selection (replaces blind `default_input_device()`)

1. Enumerate `host.input_devices()`; collect `(name, supported_config)`.
2. Preference order (matches the Python heuristic):
   - name contains `USB` **and** `Camera` (or `PC-LM1E`) -> top pick
   - name contains `USB` -> second
   - else fall back to default, but **log every candidate name** so mispicks
     are visible.
3. `AUDIO_DEVICE` env (substring match) still **overrides** auto-detect for
   manual control.
4. On no match: log all candidates at `warn` (the Python script does exactly
   this) and degrade to silent mode -- never cascade.

### B. Channel/rate capability probe (the real fix for RC-2)

- For the chosen device, read `device.supported_input_configs()`; pick a config
  whose `channels` and `sample_rate` the hardware actually supports.
- If hardware is **2ch-only** and config wants **1ch**: capture at **2ch**,
  then **downmix to mono** in the cpal callback (`(l+r)*0.5`) before pushing to
  the ringbuf. Keeps downstream `CHANNELS=1` contract intact.
- If hardware rate != 16000: either let ALSA `plug` resample (use a
  `plughw:`/`default`-style device name) or capture native and resample in
  Rust. Simplest: prefer a `plug`-backed device so 16k/1ch is satisfiable in
  ALSA.

### C. Signal-level observability (fixes RC-3)

- Add a rolling **RMS + peak + silence-ratio** over each metric window (cheap:
  accumulate sum-of-squares + abs-max per frame).
- Emit in the existing `audio_pipeline` log: `rms_dbfs= peak_dbfs=
  silence_pct=`.
- On startup, log a **1-second pre-flight capture** RMS (like the Python
  `analyse()` verdict) at `info`: `pre_flight_rms_dbfs=-33.8
  signal=OK|SILENT`. This single line would have caught this bug instantly.

### D. Config alignment (dataflow, not code)

- Set `CHANNELS: "2"` **or** keep `1` and rely on the downmix from B.
- Uncomment + set `AUDIO_DEVICE: "USB Audio"` (or auto-detect makes this
  optional).
- Keep `SAMPLE_RATE: "16000"` (within `[8000 48000]` OK).

## Immediate Verification (no code change)

Confirm RC-1/RC-2 on the box in 30 seconds:

```bash
# does cpal's "default" = the wrong (HD-Audio) card?
arecord -D default -d 1 -r 16000 -c 1 -f S16_LE /tmp/t.wav   # if this works, default supports 1ch -> HD-Audio card

# does the USB mic work at the node's requested format?
arecord -D hw:0,0 -d 1 -r 16000 -c 1 -f S16_LE /tmp/u.wav    # FAILS (2ch-only) <- RC-2 confirmed

arecord -D plughw:0,0 -d 1 -r 16000 -c 1 -f S16_LE /tmp/v.wav # if this works -> plug is the bridge cpal is missing
```

Then set `AUDIO_DEVICE="USB Audio"` + `CHANNELS="2"` (or add downmix) and
re-check the capture log for non-zero RMS.

## Files Implicated

| File | Role | Issue |
|---|---|---|
| `rover-kiwi/audio_capture/src/main.rs` | capture | RC-1 (device select), RC-2 (channels), RC-3 (no signal metrics) -- **enhancement target** |
| `rover-kiwi/rover-kiwi-direct-dataflow.yml` | config | `CHANNELS:"1"` + commented `AUDIO_DEVICE` |
| `rover-kiwi/rover-kiwi-dataflow.yml` | config | same `CHANNELS:"1"` + commented `AUDIO_DEVICE` |
| `rover-kiwi/audio_converter/src/main.rs` | f32->s16 | OK (passes through correct dims) |
| `common/web_bridge/src/main.rs` | browser emit | OK (`audio_enabled=true`, S16LE validation correct) |
| `scripts/check-usb-microphone.py` | reference test | **reference implementation** for the auto-detect heuristic to port |

## Implementation Checklist (when ready to fix)

- [x] **A.** Port `detect_usb_mic()` heuristic into `try_open_input_stream`
      (enumerate + prefer `USB`+`Camera` + log all candidates).
- [x] **B.** Add `supported_input_configs()` probe; if 2ch-only, capture at 2ch
      + downmix `(l+r)*0.5` in the cpal callback before ringbuf push.
- [x] **C.** Add rolling RMS/peak/silence-ratio to `MetricWindow` snapshot;
      add startup 1-second pre-flight RMS `info` log line.
- [x] **D.** Update both dataflow yml files: set `CHANNELS` correctly and/or
      uncomment `AUDIO_DEVICE`.
- [x] **E.** Add unit tests for: auto-detect preference ordering (mocked device
      names), downmix arithmetic, silence-detection threshold.
- [ ] **F.** Run `scripts/check-usb-microphone.py` before and after to confirm
      the node targets the same device the Python script succeeds with.
- [ ] **G.** Verify capture log shows `rms_dbfs` well above silence floor
      (target >= -40 dBFS, like the Python verdict -33.8 dBFS).

## Implementation Outcome

- Completed in code and user-approved on 2026-07-02.
- Added Linux input-device auto-detection with manual-override fallback,
  capability-aware channel/rate selection, mono downmix, and native-rate
  resampling fallback for incompatible hardware defaults.
- Added signal-level observability (`rms_dbfs`, `peak_dbfs`, `silence_pct`) plus
  a startup pre-flight signal probe so silent-but-valid capture streams are
  visible at `info`.
- Updated rover dataflow comments to document deterministic `AUDIO_DEVICE`
  override behavior while preserving the downstream mono contract.
- Validation completed:
  - `cargo test -p robo_rover_lib`
  - `cargo test -p audio_capture`
  - `cargo check -p audio_capture -p web_bridge -p rover_zenoh_bridge -p orchestra_zenoh_bridge -p audio_converter -p video_encoder -p kornia_capture`
- Remaining manual hardware verification is tracked in checklist items F-G.

## Reference: Python test result (proof hardware is good)

`scripts/check-usb-microphone.py` run `2026-07-02 00:38:46 +0700`:

```
- arecord device: hw:0,0
- card number: 0
- description: card 0: Camera [PC-LM1E Camera], device 0: USB Audio [USB Audio]
- duration: 10s, rate: 48000 Hz, channels: 2
- peak: 6902/32768  (-13.5 dBFS)
- RMS:  666.3/32768  (-33.8 dBFS)
- silence: 50.7% of samples (below ~-40 dBFS)
- clipping: 0.000%
- est. SNR: 12.3 dB
- VERDICT: OK (usable signal)
- playback: OK (pw-play via PipeWire)
- full mic->speaker loop: YES
```

Key difference: the Python script records at **48000 Hz / 2ch** (the hardware's
native format) and auto-detects `hw:0,0`. The Rust node records at **16000 Hz /
1ch** (a format the raw `hw` device rejects) and uses the cpal default device.
That single format + device mismatch is the entire bug.
