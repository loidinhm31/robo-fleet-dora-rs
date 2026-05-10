# Phase 3: Fix Socket.IO Types — Web Bridge as Source of Truth

> Parent: [plan.md](plan.md) | Depends on: [Phase 2](phase-02-command-types-bridge.md)

## Overview

- **Priority:** P1
- **Status:** done
- **Effort:** 30m

Rewrite `socket.ts` to match actual web_bridge events. Remove all stale events. Add all missing events.

## Source of Truth: web_bridge main.rs

### Bridge emits (ServerToClientEvents)

| Event | Type | Keep/Add/Remove |
|-------|------|-----------------|
| `video_frame` | `VideoFrame` | keep |
| `audio_frame` | binary | add |
| `detections` | `DetectionFrame` | add |
| `tracked_detections` | `DetectionFrame` | add |
| `tracking_telemetry` | `TrackingTelemetry` | add |
| `servo_telemetry` | `TrackingTelemetry` | add |
| `transcription` | `SpeechTranscription` | add |
| `performance_metrics` | `SystemMetrics` | add |
| `fleet_status` | `FleetStatus` | add |
| `active_rovers_status` | `ActiveRoversStatus` | add |
| ~~`video_stats`~~ | — | **remove (stale)** |
| ~~`video_status`~~ | — | **remove (stale)** |
| ~~`rover_telemetry`~~ | — | **remove (stale — bridge doesn't emit)** |
| ~~`arm_telemetry`~~ | — | **remove (stale — bridge doesn't emit)** |

### Bridge handles (ClientToServerEvents)

| Event | Type | Keep/Add/Remove |
|-------|------|-----------------|
| `arm_command` | `WebArmCommand` | keep |
| `rover_command` | `WebRoverCommand` | keep |
| `tracking_command` | `WebTrackingCommand` | add |
| `camera_control` | `{ command: string }` | add |
| `audio_control` | `{ command: string }` | add |
| `tts_command` | `{ text: string }` | add |
| `audio_stream` | `{ audio_data: number[] }` | add |
| `performance_control` | `{ enabled: boolean }` | add |
| `fleet_select` | `FleetSelectCommand` | add |
| ~~`video_control`~~ | — | **remove (stale — bridge uses camera_control)** |

### Frontend types to update

**commands.ts** — extend `WebTrackingCommand`:
```typescript
export interface WebTrackingCommand {
  command_type: "enable" | "disable" | "enable_detection" | "disable_detection" | "select_target" | "clear_target";
  tracking_id?: number;
  detection_index?: number;
}
```

**tracking.ts** — add `DetectionOnly` to `TrackingState`:
```typescript
export type TrackingState = "Disabled" | "DetectionOnly" | "Enabled" | "Tracking" | "TargetLost";
```

## Related Files

| File | Repo | Action |
|------|------|--------|
| `packages/shared/src/types/socket.ts` | robo-control-app | Rewrite |
| `packages/shared/src/types/commands.ts` | robo-control-app | Extend WebTrackingCommand |
| `packages/shared/src/types/tracking.ts` | robo-control-app | Add DetectionOnly state |
| `packages/shared/src/types/index.ts` | robo-control-app | Verify re-exports |
| `packages/ui/.../RoboRoverControl.tsx` | robo-control-app | Remove stale listeners (VALIDATED) |

## Implementation Steps

- [ ] Rewrite `socket.ts` ServerToClientEvents (10 events)
- [ ] Rewrite `socket.ts` ClientToServerEvents (9 events)
- [ ] Remove stale events + unused imports (VideoStats, VideoControl, etc.)
- [ ] Add `"enable_detection" | "disable_detection"` to WebTrackingCommand
- [ ] Add `"DetectionOnly"` to TrackingState union
- [ ] Add missing imports in socket.ts
- [ ] Remove dead `VideoControl` type from commands.ts if unused elsewhere
- [ ] **Remove stale listeners from RoboRoverControl.tsx** (VALIDATED: clean break)
  - Remove `socket.on("arm_telemetry", ...)` — bridge never emits
  - Remove `socket.on("rover_core_telemetry", ...)` — bridge never emits
  - Remove associated state/handlers if now unused
- [ ] Run `pnpm check-types`

## Success Criteria

- `pnpm check-types` passes
- Every event in socket.ts matches a real bridge emit/handler
- No stale events remain
- No component listeners for events the bridge doesn't emit
