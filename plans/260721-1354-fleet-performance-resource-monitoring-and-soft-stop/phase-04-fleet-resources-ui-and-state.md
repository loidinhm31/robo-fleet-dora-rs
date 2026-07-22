# Phase 04 — Fleet Resources UI and Authoritative State

## Context links

- [Parent plan](./plan.md)
- UI: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/`
- [Resource research](./research/researcher-02-resource-metrics-ui.md)

## Overview

- Date: 2026-07-21
- Priority: P1
- Implementation status: Done (2026-07-22T03:04:10+07:00)
- Review status: Approved (2026-07-22T03:04:10+07:00)
- Description: replace Fleet Performance with resource-only monitoring and safe node controls.

## Key Insights

- Current panel optimistically toggles a global backend forwarding flag.
- Last samples never expire and selected panel rover can diverge from fleet selection.
- Pause must show pending until authoritative applied status arrives.

## Requirements

- Rename panel to `FLEET RESOURCES`; show CPU and memory only, with Host/Container scope.
- Remove FPS from Fleet panel and FleetSelector; leave CameraViewer local diagnostic intact.
- Mark snapshots stale after `max(3 × interval, 15 s)` and unavailable values as `—`.
- Add Orchestra/Rover selector tied to explicit target, not map insertion order.
- Show node presence, pause capability, lifecycle state, last-seen age, and CPU/RSS.
- Pause/resume controls only for advertised supported nodes; locked label for always-on/unsupported.
- Pending/result/error state from request ID and authoritative status; no optimistic applied state.
- Show user desired state separately from effective/applied state when a schedule wake lease temporarily runs paused media.

## Architecture

Maintain normalized maps by `(role, entity_id)` for resource snapshots and `(role, entity_id, node_id)` for lifecycle status. On reconnect/auth loss/target switch: clear pending requests, request snapshots/status, and reject stale `{manager_epoch, revision, transition_id}` regressions. A local display-freeze control is unnecessary; monitoring stays visible during pause. Lifecycle rows distinguish desired Paused from effective Running during an authorized scheduled wake.

## Related code files

- Rename/modify: `packages/ui/src/components/features/FloatingMetrics.tsx` → resource-focused component.
- Modify: `RoboRoverControl.tsx`, `FleetSelector.tsx`, hooks/services/adapters and exports.
- Modify: `packages/shared/src/types/{performance.ts,socket.ts,index.ts}` to resource/lifecycle types.
- Modify/create: Vitest component/store tests and fake-Socket.IO Playwright fixtures/specs.
- Remove: FPS/latency tabs, color logic for removed metrics, `performance_control` client emit.

## Implementation Steps

1. Add shared ResourceSnapshot/Lifecycle types and typed Socket events.
2. Extract normalized resource/lifecycle hook/store from page-local maps.
3. Handle sequence/revision/freshness and reconnect reset deterministically.
4. Build collapsed fleet summary and expanded CPU/memory node views with scope labels.
5. Add capability-aware pause/resume, request correlation, `Cancelling…`, `Quiescing…`, `Temporarily active for scheduled recording`, `Degraded`, `Failed`, and `Superseded` feedback; keep pending through terminal applied status.
6. Remove fleet FPS from FloatingMetrics/FleetSelector and obsolete metric color types.
7. Add keyboard/focus labels, mobile layout, and no-data/stale/locked states.

## Todo list

- [x] Shared UI contracts
- [x] Normalized resource/lifecycle state
- [x] Fleet Resources component
- [x] Pause/resume UX
- [x] Fleet FPS removed
- [x] Vitest and Playwright coverage

## Success Criteria

- No Fleet Resources/FleetSelector FPS or inferred latency appears.
- One client cannot hide another client's telemetry.
- UI never labels accepted as applied; partial/degraded states stay visible.
- Selected Orchestra/Rover/node target remains explicit through async commands.
- Any signed-in user can control advertised safe nodes, while concurrent stale revisions surface a conflict and current state.

## Risk Assessment

- Separate UI repository release drift: contract fixtures and controlled rollout.
- Stale snapshots: visible stale badge and controls disabled until authoritative state.
- Double clicks: request-local pending lock plus server idempotency.

## Security Considerations

- Browser cannot claim actor, capability, or arbitrary node ID.
- Sanitize reason text and avoid exposing process/path details.

## Next steps

- Phase 05 validates native, container, Zenoh, and direct-mode behavior.

## Unresolved Questions

- Confirm product wording for `Quiesced` versus user-facing `Paused`.
