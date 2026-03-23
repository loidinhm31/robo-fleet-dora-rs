# Vision Pipeline Consolidation Plan

**Date:** 2026-03-23
**Issue:** Object detector + tracker auto-run on camera start, exhausting Pi resources
**Goal:** Consolidate detection/tracking into kornia_capture as library crates with control layer

## Status

| Phase | Status | Completed |
|-------|--------|-----------|
| Phase 1: Lib crates | DONE | 2026-03-23 |
| Phase 2: VisionPipeline | DONE | 2026-03-23 |
| Phase 3: Dataflow wiring | DONE | 2026-03-23 |
| Phase 4: Verify + docs | PENDING | — |

## Problem

Current dataflow spawns 4 separate processes for vision:

```
gst-camera (process) → object-detector (process) → reid-extractor (process) → object-tracker (process)
```

- **object-detector**: YOLO inference on EVERY frame (~100ms+ on Pi 5)
- **reid-extractor**: OSNet inference on EVERY detection (~50ms on Pi 5)
- **object-tracker**: CMC optical flow on EVERY frame
- Each process: ~50-100MB RAM overhead, 900KB frame copy via IPC
- Total wasted IPC bandwidth at 30fps: ~81MB/s
- **No way to disable** — detector has zero gating mechanism

## Solution

Consolidate into single process with control layer:

```
kornia_capture (single process, zero-copy)
  ├── Camera Layer      (always active when camera on)
  ├── Control Layer     (gates detection/tracking via single toggle)
  ├── Detection Layer   (YOLO — lib call, lazy-loaded)
  ├── ReID Layer        (OSNet — lib call, lazy-loaded)
  └── Tracker Layer     (BoTSORT+CMC — lib call, lightweight)
```

### Control: Tracking On/Off (Single Toggle)

```
CAMERA ONLY ──tracking:enable──→ FULL PIPELINE (YOLO + ReID + CMC + BoTSORT)
 (default)  ←─tracking:disable─
```

- **Camera only (default):** zero ML overhead, video streaming only
- **Tracking on:** full pipeline — YOLO → ReID → CMC → BoTSORT → visual servo
- Reuses existing `TrackingCommand::Enable/Disable` — no new bridge code
- `DetectionControl` type added for future granularity (detection without tracking)

### Resource Impact

| Metric | Before (4 processes) | After (1 process) |
|--------|---------------------|-------------------|
| RAM baseline | ~200-400MB | ~50MB (camera only) |
| RAM full pipeline | always loaded | ~150MB (lazy on first enable) |
| Frame IPC | 900KB × 3 × 30fps = 81MB/s | 0 (zero-copy &[u8]) |
| First enable | instant (already loaded) | 2-3s cold start (model load) |
| Subsequent enables | N/A | instant (models stay loaded) |

---

## Decisions

### D1: Split object_tracker lib → YES

Split into 3 files to meet 200-line guideline:
- `kalman.rs` (~130 lines): KalmanFilter struct
- `tracked_object.rs` (~170 lines): TrackedObject + InternalTrackState
- `lib.rs` (~250 lines): ObjectTracker struct + association logic + `pub mod`

### D2: Shared ort::Environment → YES

ort 1.16.3 enforces one global ORT environment per process. Explicit `Arc<Environment>` sharing:
- Both constructors accept `Arc<Environment>` parameter
- VisionPipeline creates env once during lazy init
- Stored as `ort_env: Option<Arc<Environment>>`

### D3: Output mode → Tracking only (no detection-only output)

- Only `tracked_detections` and `tracking_telemetry` outputs when pipeline active
- Single on/off toggle via existing `TrackingCommand::Enable/Disable`
- No bridge code changes needed — existing tracking_command flow fully wired

---

## Architecture

See phase files for detailed implementation. Crate structure:

```
rover-kiwi/
  object_detector/src/lib.rs          # pub struct YoloDetector
  reid_extractor/src/lib.rs           # pub struct ReIdExtractor
  object_tracker/src/{lib,kalman,tracked_object,cmc}.rs  # pub struct ObjectTracker
  kornia_capture/src/{main,vision_pipeline}.rs            # VisionPipeline + Dora node
```

## Phases

Sequential execution (each depends on previous):

| Phase | File | Description | Status |
|-------|------|-------------|--------|
| [Phase 1](phase-1-convert-to-lib-crates.md) | Lib crates | Extract detector/reid/tracker from bin → lib | DONE 2026-03-23 |
| [Phase 2](phase-2-vision-pipeline.md) | VisionPipeline | Build control layer in kornia_capture | DONE 2026-03-23 |
| [Phase 3](phase-3-dataflow-wiring.md) | Dataflow | Update YAML, remove old nodes, rewire | DONE 2026-03-23 |
| [Phase 4](phase-4-verification.md) | Verify + docs | Build test, Pi test, update CLAUDE.md | PENDING |

## File Change Summary

| File | Action | Est. Lines |
|------|--------|-----------|
| `object_detector/src/lib.rs` | NEW | ~200 |
| `object_detector/src/main.rs` | DELETE | -389 |
| `object_detector/Cargo.toml` | EDIT | ~3 |
| `reid_extractor/src/lib.rs` | NEW | ~170 |
| `reid_extractor/src/main.rs` | DELETE | -343 |
| `reid_extractor/Cargo.toml` | EDIT | ~3 |
| `object_tracker/src/lib.rs` | NEW | ~250 |
| `object_tracker/src/kalman.rs` | NEW | ~130 |
| `object_tracker/src/tracked_object.rs` | NEW | ~170 |
| `object_tracker/src/main.rs` | DELETE | -795 |
| `object_tracker/Cargo.toml` | EDIT | ~3 |
| `kornia_capture/src/vision_pipeline.rs` | NEW | ~180 |
| `kornia_capture/src/main.rs` | REWRITE | ~200 |
| `kornia_capture/Cargo.toml` | EDIT | ~8 |
| `robo_rover_lib/src/types/detection_types.rs` | EDIT | +15 |
| `rover-kiwi/rover-kiwi-dataflow.yml` | REWRITE | ~100 |
| `CLAUDE.md` | EDIT | ~20 |

**Net:** ~+1350 new, -1527 deleted = -177 net lines | **Bridge changes:** 0 Rust files | **Total:** 17 files

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| ONNX panic kills camera | Medium | Catch with Result, log + auto-disable pipeline |
| Cold start 2-3s on first enable | Low | Acceptable; log "loading models..." |
| Build time increase | Low | Cross-compile; ort cached |
| ort Environment singleton | Low | Explicit `Arc<Environment>` sharing |
| Frame rate from in-process ML | Medium | Back-pressure: if YOLO >33ms, frames drop naturally |
