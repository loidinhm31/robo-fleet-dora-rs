# Phase 05 — Scheduler UI and Client State

## Context links

- [Parent plan](./plan.md)
- [Phase 01](./phase-01-contracts-and-decision-freeze.md)
- [Phase 03](./phase-03-web-bridge-coordinator-and-recorder-reconciliation.md)
- Existing page: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/pages/RoboRoverControl.tsx`
- Existing recording page: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/pages/media-recording-page.tsx`

## Overview

- Date: 2026-07-20
- Description: Add Scheduler tab, typed store, editor/list/status UI, CAS conflicts, authenticated access, reconnect, and deterministic tests in linked app.
- Priority: P2
- Implementation status: Done — 2026-07-20 22:56 +07 (UTC+0700)
- Review status: Approved — 2026-07-20 22:56 +07 (UTC+0700)
- Effort: 9h

## Key Insights

- `RoboRoverControl.tsx` is already large; add routing/props only and keep scheduler logic focused.
- Existing recording store demonstrates authoritative Socket.IO status/reconnect handling.
- Server owns recurrence resolution. Client preview is informational; saved next-run comes from snapshot.
- Draft device override hook uses obsolete lease contract and must not become scheduler store.

## Requirements

- Third top-level `SCHEDULER` view beside CONTROL/RECORDINGS for web/Tauri shared UI.
- List schedules for the authenticated user with enabled state, recurrence, timezone, next run, active/retry/missed/error state.
- Editor supports one-time/daily/weekly, weekdays, local date/time, duration, IANA zone, safe relative directory, title, enabled.
- Create/edit/enable/disable/delete use request IDs and expected revisions.
- Conflict replaces stale local record and prompts reapply.
- Reconnect/auth/entity switch resyncs authoritatively and clears stale pending state.
- Every logged-in user sees the same schedule CRUD controls, including delete; logged-out users cannot query or mutate scheduler state.
- Responsive keyboard-accessible form with live status announcements.
- Never expose absolute paths or compute recorder ownership in UI.

## Architecture

- `useRecordingScheduleStore`: normalized schedules/occurrences, readiness, pending, errors.
- Event hook attaches once, handles snapshots/status/results monotonically, detaches on socket change.
- Action hook emits validated commands; no optimistic durable mutation, only pending indicators.
- Focused page/list/editor/status/conflict components; server snapshot is source of truth.
- Timezone selector shows zone and resolved absolute timestamp; server resolves DST.

## Related code files

- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/recording-schedule.ts` — finalized types/helpers.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/socket.ts` — scheduler events.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/recording-schedule-store-types.ts` — state types.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/use-recording-schedule-store.ts` — composition.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/use-recording-schedule-events.ts` — listeners/reconnect.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/use-recording-schedule-actions.ts` — CRUD.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/pages/recording-scheduler-page.tsx` — page composition.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/recording-schedule-list.tsx` — cards.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/recording-schedule-editor.tsx` — form.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/recording-occurrence-status.tsx` — status.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/pages/RoboRoverControl.tsx` — third view only.
- Extend `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/apps/web/src/recording-e2e-harness.tsx` and its scheduler spec — deterministic browser flows.

## Implementation Steps

1. Finalize shared types/fixtures from Phase 1.
2. Implement pure field validation/formatting; keep DST authoritative server-side.
3. Build normalized store with monotonic revision/status and bounded pending requests.
4. Implement listener lifecycle for connect/disconnect/auth/entity/unmount.
5. Implement CRUD/query actions with expected revision and safe request IDs.
6. Build page/list/status empty/loading/degraded/error states.
7. Build recurrence/output editor and authoritative next-run preview.
8. Add conflict, delete confirmation, login-required, retry/missed/suppressed messaging. Do not add retry alert delivery.
9. Add third tab without moving state into oversized page.
10. Add component/store/mobile/accessibility/harness tests.
11. Deprecate lease draft only after import search proves unused.

## Todo list

- [x] Third view works in web/Tauri shared component.
- [x] CRUD/enable/disable/delete use CAS.
- [x] Recurrence validation and IANA display covered.
- [x] Reconnect/entity switch cannot leak state.
- [x] Conflict/retry/missed/suppressed/degraded states accessible.
- [x] Logged-in CRUD/delete and logged-out denial tested; no scheduler role variants.
- [x] No absolute path or actor field in payload.
- [x] Occurrence-status backend route supplies the UI with authoritative occurrence state.

## Success Criteria

- `pnpm check-types`, `pnpm lint`, and `pnpm build` pass in linked app.
- Vitest covers event ordering, reconnect, conflicts, validation, authentication, entity isolation.
- Playwright covers CRUD/mobile with fake Socket.IO.
- Keyboard-only and axe/WCAG checks pass; live changes do not steal focus.
- Existing manual recording UI tests remain green.

## Risk Assessment

- Cross-repo drift: canonical fixtures/versioned cutover.
- Client DST divergence: display server resolution.
- Large page regression: route only; hooks/components modular.
- Stale optimistic state: pending indicators then authoritative result.

## Security Considerations

- UI authentication checks are UX only; backend validates the session for every query and mutation.
- Escape titles/errors, never render detail as HTML, never log JWT/path.
- Directory helper mirrors server constraints but does not replace server validation.
- Clear state on logout/token expiry and prevent cross-user cache reuse.

## Next steps

- Run complete gates in [Phase 06](./phase-06-end-to-end-fault-and-rollout-verification.md).

## Unresolved questions

1. Show all fleet rovers or only selected rover to a logged-in user in v1?
2. Timezone selector fallback if `Intl.supportedValuesOf` is absent?
