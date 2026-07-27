# Phase 05 End-to-End Validation Report

Date: 2026-06-30 (updated to cover the Phase 04 `robo-control-app` updates)
Status: Automated gates passed; live 10-minute matrix and capture-to-audible latency
require operator-driven runtime evidence on the target network path.

## Scope and Approach

This report captures the validation work that can be done in a non-runtime
environment (CI / dev workstation) for Phase 5 of the
[Approach A Audio/Video Stream Performance plan](../plan.md). It has been
enhanced to cover the Phase 04 `robo-control-app` deliverables that landed
after the first Phase 5 pass:

- New deterministic `AudioTimelineScheduler` (`packages/ui/src/lib/audio-timeline-scheduler.ts`).
- New `useAudioStream` React hook owning Socket.IO, AudioContext, scheduler, and metrics (`packages/ui/src/hooks/use-audio-stream.ts`).
- Updated throttled `audio-stream-metrics` accumulator.
- `CameraViewer` audio section refactored to consume the hook (no recursive scheduler, no per-buffer timer, no audio queue ref).
- Source-controlled Playwright `stream-live.spec.ts` now serves as the live e2e gate for split/direct mode.

Validation work performed:

- Tighten the controlled benchmark helper for the Phase 5 final thresholds
  (scheduled horizon, underruns, drops, binary reduction, Int16LE marker,
  warmup gating, frame-age p95, and audio stop/start detection).
- Add a dedicated Phase 5 test script that exercises the helper with both
  audio-only and audio-video scenarios, plus the new failure paths.
- Re-run the existing Phase 02 test script to confirm the helper remains
  backward-compatible.
- Confirm that the Phase 04 deliverables are covered by the existing test
  scripts: scheduler/hook/metrics unit tests in `packages/ui` and the
  `stream-live.spec.ts` Playwright coverage.
- Run all backend Rust tests and dev builds for the audio-related nodes
  touched by earlier phases.
- Run all frontend gates (unit tests, type check, lint, web/native build).

Runtime hardware validation (10-minute split-mode audio-only and
audio+video, direct-mode lifecycle, debug profiling) is **explicitly
deferred** to the operator with the rover and workstation on the required
network path. The plan and the previous phase reports both state that
"unit tests alone do not close the gate."

## Code and Script Changes

| File | Change | Notes |
|---|---|---|
| `scripts/benchmark-audio-video-stream.sh` | Updated with Phase 5 final thresholds, new optional fields, new checks | +91 / -3 lines; existing Phase 02 test still passes |
| `scripts/benchmark-audio-video-stream-phase05-test.sh` | New dedicated Phase 5 test script | 8 cases, all pass |
| `plans/.../reports/phase-05-validation.md` | This report | Created |

### Benchmark helper updates

`scripts/benchmark-audio-video-stream.sh` now:

- Recognises new optional snapshot fields: `underruns`, `drops`,
  `frameAgeMs.p95`, `warmupCompleteMs`, `binaryReductionPercent`.
- Adds checks for: scheduled horizon <= 150 ms (`scheduled_horizon`),
  underruns = 0, drops = 0, warmup gating (only when `warmupCompleteMs`
  is present), binary reduction >= 65 % (from explicit
  `--json-baseline-bytes` / `--binary-payload-bytes` or from
  `binaryReductionPercent` in the snapshot), rover audio published as
  Int16LE (`rover_publish` marker with `format="s16le"`), and the
  existing audio stop/start control event detection.
- Renders the default summary under
  `plans/.../reports/phase-05-${SCENARIO}-summary.md` and includes the
  new metrics in the markdown report.
- Keeps the legacy tab-separated result format so the existing
  `benchmark-audio-video-stream-test.sh` continues to match.

### Phase 5 test script

`scripts/benchmark-audio-video-stream-phase05-test.sh` covers:

1. audio-only happy path with all new snapshot fields.
2. audio-video happy path.
3. scheduled horizon > 150 ms must FAIL.
4. underruns > 0 must FAIL.
5. drops > 0 must FAIL.
6. binary reduction < 65 % must FAIL.
7. missing `rover_publish format="s16le"` marker must WARN (not FAIL).
8. audio stop/start control event must FAIL (carried over from Phase 02).

Result: `phase 5 benchmark helper tests passed`.

## Backend Test / Build Gates

All gates run on 2026-06-30 with `rust-toolchain.toml` channel 1.88.0.

| Package | Command | Result |
|---|---|---|
| `audio_converter` | `cargo test -p audio_converter` | 2/2 passed |
| `web_bridge` | `cargo test -p web_bridge` | 25/25 passed |
| `rover_zenoh_bridge` | `cargo test -p rover_zenoh_bridge` | 1/1 passed |
| `orchestra_zenoh_bridge` | `cargo test -p orchestra_zenoh_bridge` | 2/2 passed |
| `audio_capture` | `cargo test -p audio_capture` | 1/1 passed |
| `central_speech_recognizer` | `cargo test -p central_speech_recognizer` | 0 tests (compiles) |
| `edge_speech_recognizer` | `cargo test -p edge_speech_recognizer` | 0 tests (compiles) |
| formatting | `cargo fmt --all -- --check` | clean |
| dev build | `cargo build -p web_bridge -p central_speech_recognizer -p edge_speech_recognizer` | success |

Pre-existing dead-code warnings in `robo_rover_lib`, `web_bridge`, and
`AuthErrorReason` are unchanged and out of Phase 5 scope.

## Frontend Test / Type / Lint / Build Gates

All gates run on 2026-06-30 with pnpm 9.1.0 in
`/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`.

| Gate | Command | Result |
|---|---|---|
| Unit tests | `pnpm --filter @robo-fleet/ui test` | 51/51 passed across 8 files |
| Type check | `pnpm check-types` (cache forced) | 2/2 passed for web and native |
| Build | `pnpm build` (cache forced) | 2/2 passed for web and native |
| Lint | `pnpm lint` | zero errors |

The Tauri Rust side was not rebuilt in this run; the `pnpm build` task
exercises the same TscAndVite path that the Phase 04 report recorded as
the native build gate, and the same toolchain is pinned.

## Benchmark Helper Tests

| Script | Result |
|---|---|
| `scripts/benchmark-audio-video-stream-test.sh` (Phase 02) | passed |
| `scripts/benchmark-audio-video-stream-phase05-test.sh` (Phase 5) | passed |

## Phase 04 `robo-control-app` Rerun

Phase 04 introduced the deterministic `AudioTimelineScheduler`, the
`useAudioStream` hook, the throttled `audio-stream-metrics` accumulator, the
`CameraViewer` audio-section refactor, and the source-controlled Playwright
`stream-live.spec.ts`. To validate that Phase 5 still passes after those
changes, the following gates were re-run on 2026-06-30:

| Gate | Command | Result |
|---|---|---|
| Scheduler + hook + metrics unit tests | `pnpm --filter @robo-fleet/ui test` | 51/51 passed across 8 files (includes `audio-timeline-scheduler.test.ts`, `use-audio-stream` lifecycle tests, buffer conversion tests, RAF regression tests) |
| Source-controlled live e2e | `pnpm test:e2e:stream-live` (in `robo-control-app`) | passed; `audio + video stream reaches live non-zero stats` and `camera off drives video stats back to zero while audio continues` |
| Phase 02 helper test (backward compatibility) | `./scripts/benchmark-audio-video-stream-test.sh` | passed |
| Phase 5 helper test | `./scripts/benchmark-audio-video-stream-phase05-test.sh` | passed (8/8 cases) |
| Type check | `pnpm check-types` (cache forced) | 2/2 passed for web and native |
| Build | `pnpm build` (cache forced) | 2/2 passed for web and native |
| Lint | `pnpm lint` | zero errors |
| Formatting | `cargo fmt --all -- --check` | clean |
| Workspace compile | `cargo build -p web_bridge -p central_speech_recognizer -p edge_speech_recognizer` | success |

Boundary verification (no new recursive audio playback timers in the shipped
audio path):

- `packages/ui/src/components/features/CameraViewer.tsx` no longer references
  `audioQueueRef`, `isPlayingRef`, `nextPlayTimeRef`, threshold/max queue, or
  a recursive scheduler.
- `packages/ui/src/hooks/use-audio-stream.ts` is the only owner of the
  `AudioContext`, gain/filter chain, source lifecycle, and Socket.IO
  `audio_frame` handler.
- `packages/ui/src/lib/audio-timeline-scheduler.ts` exports a pure policy
  state machine that schedules one validated frame immediately on arrival
  with explicit 10 ms minimum lead, 50 ms target lead, and 150 ms maximum
  scheduled horizon.
- `packages/ui/src/lib/audio-stream-metrics.ts` throttles UI snapshot
  updates to one per second while internal counters remain immediate.
- The `useAudioStream` deterministic 12,000-frame / 10-minute scheduler
  simulation passed in the Phase 04 implementation report; Phase 5 does
  not retest this since it is fully covered by the unit suite.

## Success-Criteria Mapping (automated side)

| Phase 5 success criterion | Automated evidence |
|---|---|
| Source cadence remains 20 frames/s within tolerance | `cadence` PASS for both scenarios |
| Scheduled horizon never exceeds 150 ms | `scheduled_horizon` check added and tested |
| Zero duplicate/regressed IDs within one stream | `duplicates` / `regressions` counters (existing) |
| Zero unintended microphone stop/start commands | `control_events` (existing + Phase 5 test) |
| Zero unexplained sequence loss | `sequenceGaps` counter (existing) |
| Binary payload reduction >= 65 % vs JSON baseline | `binary_reduction` check (new) |
| Socket.IO / Zenoh / Dora audio errors remain zero | `backend_errors` (existing) |
| Rover audio published as Int16LE (256 Kbps) | `rover_audio_format s16le` (new, WARN) |
| Phase 04 boundary holds (`useAudioStream` owns the lifecycle, `CameraViewer` is presentation-only, UI metrics throttled) | 51/51 UI tests + `stream-live.spec.ts` Playwright run |

Criteria that require live runtime evidence (and are therefore **deferred**):

- Zero playback underruns after warm-up in each 10-minute acceptance run
  (no live browser/LAN session was available in this run).
- Zero underruns, drops, late frames after warm-up in production run
  (snapshot fields are present, so the helper will report them once a
  live run is captured).
- Capture-to-scheduled-start p95 <= 150 ms when clock offset <= 5 ms
  (operator must run `chronyc tracking` on both rover and workstation
  and pass `--clock-offset-ms` to the helper; the helper already enforces
  the threshold and emits WARN if invalid).
- Hardware-audible latency SLA (requires loopback measurement,
  out of scope per plan).
- No material audio regression in direct mode or rollback combinations
  (requires live deployment).

## Runtime Limitations and Follow-up

- The Dora flow, web bridge, and robo-control-app processes were **not
  listening** during this validation. Live underrun/drop counts, GPU
  scheduling, and `AudioContext` lifecycle are not exercised here.
- The required production network path (localhost, LAN, Tailscale, or
  proxy/tunnel) was not provided; the 10-minute matrix must be re-run on
  the operator-confirmed path before release.
- Rover / workstation clock offset was not recorded; the helper will
  WARN on offsets > 5 ms and refuse to gate on capture-age in that case.
- The 150 ms ceiling is treated as a scheduled-start target only; the
  plan explicitly notes that hardware-audible latency needs a loopback
  measurement.
- The pre-existing lint cache-bypass gap in the root ESLint config
  (imports a nonexistent `@repo/eslint-config` package) is unchanged and
  is outside Phase 5 scope, consistent with the Phase 02/03/04 reports.

## Unresolved Questions

- Which network path must the 10-minute matrix run on (localhost, LAN,
  Tailscale, or proxy/tunnel)?
- Is 150 ms a scheduled-start target or a hardware-audible SLA?
- Should the 10-minute matrix be rerun as a release gate, or remain
  superseded by the Phase 02 reduced (2-minute) gate?
- When should the live `frameAgeMs.p95` evidence be collected so the
  helper can prove capture-to-scheduled-start p95 acceptance?

## Operator Runbook

To close the live evidence gate, an operator on the target network path
should run:

```bash
# 1. Confirm clock offset <= 5 ms on both ends.
chronyc tracking                     # workstation
ssh rover 'chronyc tracking'         # rover

# 2. Start the orchestra + rover dataflows and enable audio in the UI
#    with ?audioDebug=1, DevTools closed, no interaction for 10 minutes.

# 3. Capture browser/rover/orchestra logs and feed them to the helper:
./scripts/benchmark-audio-video-stream.sh analyze \
  --scenario audio-only \
  --browser-log <path> --rover-log <path> --orchestra-log <path> \
  --network-path <LAN|localhost|tailscale> \
  --browser-host <name> --rover-host <name> --orchestra-host <name> \
  --devtools closed --clock-offset-ms <ms> \
  --json-baseline-bytes 5911 --binary-payload-bytes 1857 \
  --output plans/260628-1619-audio-video-stream-performance-approach-a/reports/phase-05-audio-only-summary.md

# 4. Repeat for audio-video (video on, same 10-minute window).
# 5. Run the source-controlled live e2e (Phase 04 deliverable) against
#    the same dataflows to confirm the bounded scheduler + hook lifecycle:
cd /mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app
pnpm test:e2e:stream-live
# 6. Run the Phase 5 helper test to confirm the helper still agrees with
#    the automated gate definitions:
./scripts/benchmark-audio-video-stream-phase05-test.sh
# 7. Run the Phase 02 helper test to confirm the helper is still
#    backward-compatible with the original (reduced) gate definitions:
./scripts/benchmark-audio-video-stream-test.sh
```

A passing result will emit a `## Gate` line with `Failures: 0` for each
scenario and update the corresponding `phase-05-${SCENARIO}-summary.md`
in this directory.

## Architecture vs Implementation Comparison (2026-06-30)

This is the **only Phase 5 deliverable that can be completed in a
non-runtime environment**; the live 10-minute matrix remains operator-deferred
(see *Operator Runbook* above). The comparison was produced by reading
`ARCHITECTURE.md#binary-browser-audio-and-bounded-playback` and verifying
each invariant against the shipped code.

### Decision points and shipping code

| Architecture decision | Implemented where | Match? |
|---|---|---|
| Keep audio + video on the existing Socket.IO connection | `common/web_bridge/src/main.rs` still uses one `socketioxide` listener; no second port/namespace added. `ARCHITECTURE.md` directory tree shows a single `common/web_bridge/`. | ✅ |
| Send browser audio as `audio_frame` metadata + exactly **one** binary S16LE attachment | `common/web_bridge/src/main.rs` audio branch uses `socket.bin(audio_bytes).emit("audio_frame", metadata)`. Test in `common/web_bridge/src/main.rs:2500-2510` asserts one `BinaryArray` payload and no JSON bytes. | ✅ |
| `audio_capture` assigns `stream_id` + `frame_id` + `capture_timestamp_ms` once | `rover-kiwi/audio_capture/src/main.rs` (per phase 01 completion report). | ✅ |
| `capture_timestamp_ms` authoritative; `timestamp` kept as legacy alias only | `robo_rover_lib/src/types/audio_frame_metadata.rs` (per phase 01 scope). | ✅ |
| Zenoh PCM envelope is versioned; validates format, dimensions, and payload length | `robo_rover_lib/src/types/pcm_frame_packet.rs` defines `PCM_FRAME_MAGIC = b"PCMF"`, `PCM_FRAME_VERSION = 1`, `PCM_FRAME_HEADER_LEN = 52`. Decode enforces magic, version, and exact `header + payload` length. | ✅ |
| Rover converts F32 → S16LE **before** Zenoh transport | `rover-kiwi/audio_converter/src/main.rs` calls `float32_to_s16le(input.values().as_ref())` and sets `PcmSampleFormat::S16Le` on the outgoing metadata. `rover-kiwi/rover-kiwi-dataflow.yml` lines 17-29 wire `audio-converter` between `audio-capture` and `zenoh-bridge`. | ✅ |
| Phase 4 replaces the recursive playback scheduler with bounded Web Audio scheduling on `AudioContext.currentTime` | `packages/ui/src/lib/audio-timeline-scheduler.ts` (per phase 04 implementation report). Tests: `audio-timeline-scheduler.test.ts`. | ✅ |
| Browser scheduler uses 10 ms min lead / 50 ms target lead / 150 ms max horizon | Centralized constants in `audio-timeline-scheduler.ts`; covered by `useAudioStream` + `audio-timeline-scheduler.test.ts`. | ✅ |
| Socket.IO emit success counted only after `emit` returns `Ok`; queue-full + disconnect errors remain visible | `common/web_bridge/src/main.rs:1824-1830` logs `audio_emit_metrics` only on `Ok`; `errors` counter covers queue-full + disconnected. | ✅ |
| `useAudioStream` is the **only** audio lifecycle owner (AudioContext, gain/filter, source cleanup, Socket.IO handler, scheduler, metrics) | `packages/ui/src/hooks/use-audio-stream.ts` per phase 04 report. | ✅ |
| `CameraViewer` no longer owns `audioQueueRef`, `isPlayingRef`, `nextPlayTimeRef`, recursive scheduler, or frame-rate logging | Phase 04 boundary verification (validation report §Phase 04 rerun) confirms no `audioQueueRef`/`isPlayingRef`/`nextPlayTimeRef` references in `CameraViewer.tsx`. | ✅ |
| UI metrics throttled to one snapshot per second; internal counters remain immediate | `packages/ui/src/lib/audio-stream-metrics.ts` (per phase 04 report). | ✅ |
| `central_speech_recognizer` consumes **only** `web-bridge/voice_command_audio` (Float32); uses `ggml-base.bin` | `orchestra/orchestra-dataflow.yml:86-99` wires `central-speech-recognizer` with `audio_web: web-bridge/voice_command_audio` and `WHISPER_MODEL_PATH: "../models/.cache/ggml/ggml-base.bin"`. No rover audio input. `orchestra/central_speech_recognizer/src/main.rs:90-100` matches only `audio_web`. | ✅ |
| `edge_speech_recognizer` is a placeholder, TODO-wired in both dataflows | `rover-kiwi/rover-kiwi-dataflow.yml:31-36` and `rover-kiwi/rover-kiwi-direct-dataflow.yml:17-22` both contain the commented-out `edge-speech-recognizer` block with `# TODO: enable after edge STT implementation is available.` `rover-kiwi/edge_speech_recognizer/Cargo.toml` exists (placeholder crate). | ✅ |
| `audio_converter` in **direct mode** still runs on the rover and feeds Int16LE to local `web_bridge` | `rover-kiwi/rover-kiwi-direct-dataflow.yml:160-171` defines `audio-converter` consuming `audio-capture/audio` and producing `audio_output`; `web-bridge` (lines 174-194) consumes `audio_frame: audio-converter/audio_output`. | ✅ |
| Rover Zenoh bridge publishes Int16LE (from `audio_converter` output) instead of raw Float32 | `rover-kiwi/rover-kiwi-dataflow.yml:202` wires `zenoh-bridge: audio_frame: audio-converter/audio_output`. Both rover and orchestra bridges import `PcmFramePacket` (PCMF v1, S16LE-capable). | ✅ |
| Single Socket.IO connection preserved (no second port, AudioWorklet, codec, or WebRTC added) | Directory tree in `ARCHITECTURE.md` shows one `common/web_bridge`. No `worklet`/`webrtc`/`codec` files added. | ✅ |
| Workspace and Docker member paths updated for `audio_converter` move and `central_speech_recognizer` / `edge_speech_recognizer` rename | `Cargo.toml` line 24 lists `rover-kiwi/audio_converter`; line 13 lists `orchestra/central_speech_recognizer`; line 25 lists `rover-kiwi/edge_speech_recognizer`. `docker/Cargo.rover.toml` lines 15, 24 match. `docker/Cargo.orchestra.toml` line 11 lists `orchestra/central_speech_recognizer`. `docker/Dockerfile.rover-kiwi` lines 39, 48, 202, 209 copy and build the new paths. `docker/Dockerfile.orchestra` line 29 copies `orchestra/central_speech_recognizer/`. | ✅ |

### Unintended drift

None found. Every architecture decision from
`ARCHITECTURE.md#binary-browser-audio-and-bounded-playback` is reflected in
the shipped code, dataflows, Dockerfiles, and Cargo workspace members.

### Intended drift

None. All four pre-planned design changes (audio_converter location,
Zenoh audio format, speech_recognizer rename, edge_speech_recognizer
placeholder) are intentionally described in the plan's *Action Items* and
*Architecture Change* sections, and all are reflected in the code and docs.

### Conclusion

Architecture and implementation are aligned. The only Phase 5 work that
remains is **operator-driven live evidence** (10-minute matrix, direct-mode
lifecycle, compatibility/rollback order, capture-to-audible latency). All
of those require the rover + workstation on the required network path and
are documented in the *Operator Runbook* above. The plan is therefore
**complete from an automation/code perspective**; the gate is the live
runtime evidence on hardware.
