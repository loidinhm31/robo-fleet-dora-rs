# Phase 4: CameraViewer Detection Toggle + State Feedback

> Parent: [plan.md](plan.md) | Depends on: [Phase 3](phase-03-fix-socket-types.md)

## Overview

- **Priority:** P1
- **Status:** done
- **Effort:** 45m

Add detection toggle to CameraViewer. Show pipeline state feedback. Existing tracking toggle unchanged.

## UI Design

### Control Buttons (right sidebar, existing pattern)

Current: `[View Mode] [Tracking Toggle]`
New: `[View Mode] [Detection Toggle] [Tracking Toggle]`

| Button | States | Colors |
|--------|--------|--------|
| Detection toggle | Off (gray) / On (yellow) | Matches view mode color pattern |
| Tracking toggle | Off (gray) / On (green) / Tracking (blue) / Lost (red) | Unchanged |

### State Indicator (top center badge, existing pattern)

Current: shows view mode only.
New: show pipeline state derived from `trackingTelemetry.state`:

| TrackingState | Badge |
|---------------|-------|
| `Disabled` | "Camera Only" (blue) |
| `DetectionOnly` | "Detection Active" (yellow) |
| `Enabled` | "Tracking Ready" (green) |
| `Tracking` | "Following [target]" (green pulse) |
| `TargetLost` | "Target Lost" (red) |

### Interaction Logic

```typescript
const toggleDetection = () => {
  const state = trackingTelemetry?.state ?? "Disabled";
  if (state === "Disabled") {
    // Off → Detection
    sendTrackingCommand({ command_type: "enable_detection" });
  } else {
    // Any active state → Off
    sendTrackingCommand({ command_type: "disable" });
  }
};

const toggleTracking = () => {
  const state = trackingTelemetry?.state ?? "Disabled";
  if (state === "Disabled" || state === "DetectionOnly") {
    // Off/Detection → Full tracking
    sendTrackingCommand({ command_type: "enable" });
  } else {
    // Tracking → Detection only (not full disable)
    sendTrackingCommand({ command_type: "enable_detection" });
  }
};
```

**Key UX decision:** Disabling tracking drops to detection-only (not camera-only). User explicitly uses detection toggle to go fully off. Progressive: Detection → Tracking → Detection → Off.

### Auto view mode sync (optional)

When detection enabled and view mode is "camera", auto-switch to "camera_with_detections" so user sees bboxes immediately. Debatable — could be annoying. Recommend: don't auto-switch, let user control.

## Related Files

| File | Repo | Action |
|------|------|--------|
| `packages/ui/src/components/features/CameraViewer.tsx` | robo-control-app | Add detection toggle, update state badge |

## Implementation Steps

- [ ] Add `toggleDetection()` handler
- [ ] Update `toggleTracking()` — disable drops to detection-only
- [ ] Add detection toggle button in sidebar (before tracking toggle)
- [ ] Update top-center state badge to show pipeline state from telemetry
- [ ] Update button disabled states (tracking toggle disabled when detection off)
- [ ] Test: detection on → see bboxes, tracking on → see tracking IDs, tracking off → still see bboxes, detection off → camera only
- [ ] Run `pnpm check-types` + `pnpm lint`

## Success Criteria

- Detection toggle: Off → DetectionOnly, DetectionOnly → Off
- Tracking toggle: DetectionOnly → FullPipeline, FullPipeline → DetectionOnly
- State badge reflects actual rover pipeline state via telemetry
- View mode + pipeline state are independent (no auto-switching)
- Existing click-to-select-target still works in full tracking mode

## Edge Cases

| Scenario | Expected UI behavior |
|----------|---------------------|
| Enable detection, view mode = "camera" | Bboxes not visible (user must switch view mode) |
| Pipeline auto-disables from error | Badge updates to reflect fallback state |
| Network disconnect during tracking | Last known state shown, reconnect re-syncs |
| Enable tracking with no detections visible | Tracking active but no targets to select until objects appear |
