---
title: "Web UI Detection/Tracking Controls + Socket.IO Type Alignment"
description: "Split detection/tracking into independent toggles (B1: extend TrackingCommand), fix socket.ts types to match web_bridge"
status: done
priority: P1
effort: 3h
branch: main
tags: [web-bridge, socket-io, typescript, tracking, detection, vision-pipeline, type-safety]
created: 2026-03-23
---

# Web UI Detection/Tracking Controls + Socket.IO Type Alignment

## Problem

1. **No detection-only mode** — enabling tracking runs full pipeline (YOLO+ReID+BoTSORT). No way to preview detections without autonomous following. Wastes Pi 5 resources when user only wants to see bboxes.
2. **socket.ts incomplete** — 4 of 6 typed events are stale. ~15 active events untyped.

## Decision: B1 — Extend TrackingCommand

Reuse existing `tracking_command` Dora channel. Add `EnableDetection`/`DisableDetection` variants. Avoids new Dora wiring, new bridge handlers, new dataflow changes.

### State Machine

```
CAMERA ONLY ──enable_detection──→ DETECTION ONLY ──enable──→ FULL PIPELINE
  (default)  ←──disable──────────  (YOLO only)   ←─disable─  (YOLO+ReID+BoTSORT)
                                   ←────────────disable──────┘
```

| State | ML Load | RAM (Pi 5) | Per-frame | Output |
|-------|---------|------------|-----------|--------|
| Camera Only | none | ~50MB | 0ms | `CameraOnly` |
| Detection Only | YOLO | ~100MB | ~100ms | `DetectionOnly { detections }` |
| Full Pipeline | YOLO+ReID+BoTSORT | ~150MB | ~150ms | `FullTracking { tracked_detections, tracking_telemetry }` |

## Phases

| Phase | Description | Status | Effort |
|-------|-------------|--------|--------|
| [Phase 1](phase-01-vision-pipeline-split.md) | Split VisionPipeline state: detection_enabled + tracking_enabled | done | 45m |
| [Phase 2](phase-02-command-types-bridge.md) | Extend TrackingCommand + web_bridge conversion + dataflow output | done | 45m |
| [Phase 3](phase-03-fix-socket-types.md) | Fix socket.ts — web_bridge as source of truth, remove stale | done | 30m |
| [Phase 4](phase-04-camera-ui-controls.md) | CameraViewer: detection toggle button + state feedback | done | 45m |

## File Change Summary

| File | Action | Layer |
|------|--------|-------|
| `robo_rover_lib/src/types/detection_types.rs` | +2 TrackingCommand variants, +DetectionOnly TrackingState, remove DetectionControl | Rust types |
| `rover-kiwi/kornia_capture/src/vision_pipeline.rs` | Split state, new PipelineOutput::DetectionOnly, graceful degradation | Rover |
| `rover-kiwi/kornia_capture/src/main.rs` | Handle DetectionOnly output → `detections` Dora output | Rover |
| `rover-kiwi/rover-kiwi-dataflow.yml` | Add `detections` output to gst-camera | Dataflow |
| `orchestra/orchestra-dataflow.yml` | Add `detections` input to web-bridge | Dataflow |
| `rover-kiwi/rover-kiwi-direct-dataflow.yml` | Add `detections` output/routing | Dataflow |
| `common/web_bridge/src/main.rs` | +2 cases in convert_web_command_to_tracking_command() | Bridge |
| `packages/shared/src/types/socket.ts` | Rewrite: web_bridge as source of truth | Frontend types |
| `packages/shared/src/types/commands.ts` | Extend WebTrackingCommand.command_type | Frontend types |
| `packages/shared/src/types/tracking.ts` | Add DetectionOnly to TrackingState | Frontend types |
| `packages/ui/.../CameraViewer.tsx` | Detection toggle button, state indicator | Frontend UI |

**Net: ~140 lines added across 11 files**

## Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| State desync UI ↔ rover | Medium | TrackingTelemetry carries state, UI reconciles on every tick |
| Breaking existing "enable tracking" flow | Medium | `Enable` variant unchanged — identical behavior |
| ReID error kills detection | Medium | Graceful degradation: ReID error → fallback to detection-only |
| `detections` Dora output not routed through zenoh | Medium | Phase 2 includes zenoh bridge audit sub-step |

## Validation Summary

**Validated:** 2026-03-23
**Questions asked:** 4

### Confirmed Decisions
- **UX: tracking disable behavior** → Progressive (tracking off → detection-only, not camera-only). Avoids 2s YOLO cold start on re-enable.
- **Dead code: DetectionControl/DetectionAction** → Remove. TrackingCommand::EnableDetection/DisableDetection replaces them.
- **Zenoh bridge `detections` topic** → Needs investigation. Added audit sub-step to Phase 2 before implementing.
- **Stale socket.ts events** → Remove types AND component listeners. Clean break — telemetry wiring is a separate task.

### Action Items
- [x] Phase 4 already reflects progressive UX (confirmed)
- [x] Phase 1 already removes DetectionControl (confirmed)
- [x] Phase 2: add explicit zenoh bridge audit sub-step (read both bridge source files) — DONE
- [ ] Phase 3: also remove `arm_telemetry`/`rover_core_telemetry` listeners from RoboRoverControl.tsx
