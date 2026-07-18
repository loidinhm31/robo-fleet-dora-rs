# Phase 02: Render live recording elapsed time

## Context links

- [Parent plan](./plan.md)
- [Elapsed-time research](./research/researcher-02-elapsed-time.md)
- `ARCHITECTURE.md` — manual recording UI contract

## Overview

- Date: 2026-07-18
- Description: Make active recording time progress without periodic backend events.
- Priority: P1
- Implementation status: Done
- Review status: Approved
- Completed at: 2026-07-18T17:19:58Z

## Key Insights

- Active recorder status duration is not continuously updated.
- The bridge correctly deduplicates unchanged statuses.
- `started_at_ms` is server-wall-clock data and cannot safely be subtracted from
  browser time.

## Requirements

- Display a live elapsed value for `recording` and `stopping` states.
- Anchor to the last received `duration_ms` using `performance.now()`.
- Re-anchor on recording ID, state, or authoritative duration changes.
- Freeze terminal `completed` and `failed` durations exactly as received.
- Clean up timers on status change and component unmount.
- Do not add a Rust/TypeScript protocol field or periodic status traffic.

## Architecture

The shared UI owns only presentation time. The recorder remains authoritative
for final media duration. A reconnect receives the bridge-cached status and
starts a fresh local monotonic anchor, with no cross-machine clock assumption.

## Validation

- Focused timer tests: 5/5 passed.
- Full `@robo-fleet/ui` tests: 141/141 passed.
- `check-types`: passed.
- `lint`: passed.
- Code review: 8/10 approved.

## Related code files

- Create: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/use-recording-elapsed.ts` — reusable local monotonic timer.
- Modify: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/recording-session-control.tsx` — consume the hook and retain terminal formatting.
- Modify: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/recording-session-control.test.tsx` — fake-timer lifecycle coverage.
- Modify if package convention requires it: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/index.ts` — hook export.

## Implementation Steps

1. Add a focused hook holding base duration and a `performance.now()` anchor
   per active status identity.
2. Schedule a low-frequency display update (one second is sufficient); avoid
   state writes after cleanup.
3. Return static backend duration for idle, starting, completed, and failed;
   advance only confirmed active states.
4. Replace the component's direct `duration_ms` formatter with the hook output.
5. Add fake-timer tests for ticking without another socket event, re-anchoring,
   terminal freeze, and unmount cleanup.

## Todo list

- [x] Add local elapsed hook.
- [x] Integrate it into the recording status card.
- [x] Add deterministic timer tests.

## Success Criteria

- A `recording` status at zero displays `00:01` after one fake second without a
  new event.
- Terminal status displays the exact backend duration and no longer advances.
- Existing Socket.IO event mappings and shared types remain unchanged.

## Risk Assessment

- Browser background tab timer throttling can delay a paint; calculate from
  monotonic elapsed on each render so the next paint catches up.
- Reconnect begins a new local anchor because active status has no live backend
  duration; final recorder duration remains exact.

## Security Considerations

- No credentials, tickets, server clock values, or new client-controlled input.

## Next steps

Execute cross-repository validation in Phase 03.

## Unresolved questions

- None blocking.
