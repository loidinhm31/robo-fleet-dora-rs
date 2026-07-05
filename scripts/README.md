# Scripts Workspace

This directory is a repo-level utility workspace for local validation, benchmarking, and operator-side diagnostics.

It is **not** part of `robo-control-app/` application code and it is **not** a frontend package. Some scripts talk to the web bridge over Socket.IO, but they exist to validate the full Robo-Fleet stack, not to ship with the web UI.

## Why `package.json` exists here

`benchmark-edge-voice-x86.mjs` uses `socket.io-client` as a lightweight test client against `http://127.0.0.1:3030`.

That dependency is intentionally scoped to `scripts/` so the benchmark tooling does not leak into `robo-control-app/`.

## Main files

- `benchmark-edge-voice-x86.mjs`: Phase 8 native x86 benchmark harness for edge voice, walkie preemption, and concurrent vision/TTS checks.
- `benchmark-edge-voice-x86.sh`: Thin shell wrapper for the Node benchmark.
- `fixtures/edge-voice-corpus.json`: Benchmark corpus and case definitions used by the phase 8 harness.
- `benchmark-audio-video-stream*.sh`: Earlier transport and stream benchmark helpers.
- `benchmark-rover-video-pipeline.sh`: Rover video pipeline helper.
- `check-usb-microphone.py`: Local microphone inspection helper.

## Install

From repo root:

```bash
cd scripts
pnpm install
```

## Run the phase 8 benchmark

Prerequisites:

- Dora coordinator and daemon running
- Orchestra dataflow running with web bridge on port `3030`
- Rover dataflow running

Run:

```bash
cd scripts
node benchmark-edge-voice-x86.mjs
```

Or:

```bash
./benchmark-edge-voice-x86.sh
```

## Outputs

The phase 8 benchmark writes artifacts to:

`plans/260704-1318-edge-voice-supertonic-x86/reports/`

Current outputs:

- `phase-08-native-x86-benchmark.json`
- `phase-08-native-x86-evidence.log`

## Keep or remove later

Keep this workspace if the team wants repeatable local runtime benchmarks outside the frontend app.

If these scripts are one-off plan artifacts only, the directory can be retired after phase 10, but the generated benchmark reports should remain with the plan.
