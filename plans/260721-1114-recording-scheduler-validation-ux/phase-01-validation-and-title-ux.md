# Phase 01 — Shared Validation and Title UX

## Context links

- [Parent plan](./plan.md)
- `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/recording-schedule-helpers.ts`
- `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/recording-schedule-editor.tsx`
- `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/recording-schedule-editor.test.tsx`
- `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/recording_schedule_validation.rs`

## Overview

- Date: 2026-07-21
- Description: mirror the server's stale one-time rejection early enough for useful UI feedback and make title entry explicit.
- Status: Completed (2026-07-21 12:27 +07, Asia/Ho_Chi_Minh)
- Priority: P2

## Key insights

- The web bridge rejects an old one-time instant before it reaches Dora, so missing Dora logs are expected.
- New drafts already use `title: ""`; preserve that behavior and prevent browser autofill rather than inventing a title.
- Server resolves IANA wall-clock values using gap-forward and earlier-fold rules. Browser validation must compare the selected zone, not the workstation zone.

## Requirements

- Extend the shared browser-safe validation to reject only clearly stale `one_time` starts with actionable text; keep max-future enforcement server-authoritative.
- Resolve IANA local date/time using browser `Intl` round trips, selecting the earliest matching fold and gracefully deferring unusual unsupported cases to the server.
- Keep daily/weekly recurrence anchors valid even when their displayed date is in the past.
- Add title `required`, `autoComplete="off"`, field-specific `aria-invalid`/`aria-describedby`, and visible accessible feedback; retain form-level validation behavior.
- Preserve all existing payloads, server validation, and scheduler diagnostics.

## Related code files

- Modify `packages/shared/src/types/recording-schedule-helpers.ts`.
- Add helper tests under `packages/shared/src/types/recording-schedule-helpers.test.ts` (or the package's established test location).
- Modify `packages/ui/src/components/features/recording-schedule-editor.tsx`.
- Extend `packages/ui/src/components/features/recording-schedule-editor.test.tsx`.
- Optionally extend `apps/web/e2e/recording-scheduler.spec.ts` with explicit past/future values.

## Architecture

Keep validation in the shared pure helper so action hooks and the editor share one decision. The editor owns only presentation/accessibility state; it must not create a second timestamp policy. The server remains the final authority.

## Implementation steps

1. Add an optional deterministic `nowMs` seam or equivalent internal clock seam without breaking current callers.
2. Parse and round-trip local wall-clock fields in the selected IANA timezone via `Intl.DateTimeFormat`; handle folds/gaps consistently with the Rust contract and return no client error when resolution cannot be proven.
3. Add the stale one-time error after existing structural checks; leave non-one-time recurrences unchanged.
4. Add title accessibility attributes/help and keep the initial title empty with no generated/default value.
5. Add frozen-clock tests for UTC and offset zones, DST fold/gap behavior, invalid fields, recurrence exceptions, title blank/whitespace, and future submission.

## Todo list

- [x] Implement timezone-aware stale-start helper without new dependencies.
- [x] Add explicit title-required/autofill-resistant field UX.
- [x] Add shared and editor regression tests.
- [x] Run focused and repository quality gates.

## Implementation record

Completed in the linked UI repository at `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app` on 2026-07-21 12:27 +07. The implementation touched the shared validation helper, scheduler editor, editor tests, and the helper regression test at `packages/ui/src/lib/recording-schedule-helpers.test.ts`. It preserves the existing wire payloads and recurrence behavior while adding selected-timezone stale-start validation and accessible title/start feedback.

## Success criteria

- Stale one-time create/update is blocked locally and displays a future-start message.
- A valid future one-time schedule submits; daily/weekly schedules with past anchors still submit.
- New title is empty, cannot be submitted blank, and is announced through accessible field/form feedback.
- Existing directory and recurrence tests remain green.

## Risks and mitigations

- `Intl` timezone data can differ: verify by formatting candidates back; defer ambiguous unsupported cases to the server.
- Clock advances between validation and submit: server remains authoritative.
- Browser autofill behavior varies: use `autoComplete="off"`, no default title, and explicit required semantics.

## Security considerations

No auth, authorization, persistence, protocol, logging, or sensitive-data changes.

## Testing strategy

Use a frozen clock or explicit `nowMs` in shared tests. Exercise the selected IANA timezone (including an offset zone and DST fold/gap), malformed fields, and recurrence-specific behavior. Render the editor to assert accessible title/stale-start errors, disabled/enabled submit state, and unchanged directory/weekly behavior.

## Next steps

Phase implementation is complete; verification and handoff are recorded in Phase 02.
