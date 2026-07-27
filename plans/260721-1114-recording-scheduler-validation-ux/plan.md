# Recording Scheduler Validation UX

## Overview

- Date: 2026-07-21
- Priority: P2
- Status: Completed (2026-07-21 12:27 +07, Asia/Ho_Chi_Minh)
- Scope: browser validation and form UX in the linked `robo-control-app` repository

## Outcome

Prevent avoidable `invalid_request` submissions for stale one-time starts, show an actionable field-level error, and keep a new schedule title blank and explicitly user-entered.

## Phases

1. [Phase 01 — shared validation and editor UX](./phase-01-validation-and-title-ux.md) — completed
2. [Phase 02 — verification and handoff](./phase-02-verification.md) — completed

## Completion record

- Implementation and verification completed: 2026-07-21 12:27 +07 (Asia/Ho_Chi_Minh).
- Linked UI repository: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`.
- Scoped linked UI changes are limited to:
  - `packages/shared/src/types/recording-schedule-helpers.ts`
  - `packages/ui/src/components/features/recording-schedule-editor.tsx`
  - `packages/ui/src/components/features/recording-schedule-editor.test.tsx`
  - `packages/ui/src/lib/recording-schedule-helpers.test.ts`
- The shared helper now rejects stale one-time starts when the selected IANA timezone can be resolved, while preserving server authority for unsupported cases and future limits. The editor keeps titles blank by default and adds required, autofill-resistant, field-level accessible feedback.
- No Dora, web-bridge, protocol, persistence, auth, or unrelated repository files were changed for this plan. Linked UI changes remain uncommitted for the owning UI-repository workflow.

## Verification gates

- `pnpm --dir packages/ui exec vitest run --config vitest.config.ts src/lib/recording-schedule-helpers.test.ts src/components/features/recording-schedule-editor.test.tsx` — passed (2 files, 9 tests).
- `pnpm check-types` — passed (web and native typecheck tasks successful).
- `pnpm lint` — passed.
- `pnpm --dir apps/web test:e2e:recording` — passed (7 Playwright tests, including the scheduler create, mobile, keyboard, edit, enablement, and delete flows).

## Preflight contract

- **Output:** updated shared validation, accessible scheduler title/start-time feedback, and deterministic unit/component coverage in `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`.
- **Acceptance criteria:** stale one-time values are blocked before Socket.IO emission; future one-time values can submit; daily/weekly past anchors remain valid; title starts empty, is not autofilled, is required, and shows an accessible error while submission is blocked; existing directory/recurrence behavior remains green.
- **Scope boundary:** UI/shared helper and tests only. No Dora scheduler, web-bridge, Socket.IO protocol, persistence, auth, or server-policy changes.
- **Public contract/risk:** retain `RecordingScheduleDefinition` and wire payloads; server remains authoritative for clock races, DST policy, and configurable max-future limits.
- **Affected systems:** shared recording-schedule helper; shared scheduler editor; helper/editor tests; optional scheduler E2E assertion.
- **Testing:** targeted Vitest helper/component tests, UI typecheck/lint, then relevant browser E2E if available.
- **Open questions:** none blocking; use browser-native `Intl` timezone resolution with graceful server-authoritative fallback for unsupported DST edge cases.

## Side-effect review

- Auth/session/permissions: unchanged; no new mutation path.
- API/data/migrations: unchanged wire types and persistence.
- Security/privacy: no new user data, logging, or secrets.
- Performance: one bounded local timezone conversion during validation; no network work.
- Compatibility: keep existing helper callers valid and avoid adding a date-time dependency.
- Docs/config/deployment: no changes expected.
