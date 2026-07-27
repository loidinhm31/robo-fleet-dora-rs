# Phase 03 — Validation, review, and rollout

## Context links

- Phase 01 and Phase 02 success criteria
- Existing recorder workflow, demand-arbitration, playback safety, and recording UI tests
- `apps/web/e2e/recording-control.spec.ts` and `recording-e2e-harness.tsx`

## Validation steps — complete

1. Run focused Rust recorder/web-bridge tests, then workspace `cargo test`/`cargo check` as practical. Run FFmpeg smoke generation, ffprobe, and frame extraction with timestamp assertion.
2. Run UI Vitest suites for store, clip browser/page, CameraViewer, and existing session-control tests; run `pnpm check-types`, `pnpm lint`, and `pnpm build` from the UI checkout.
3. Run the narrowest Playwright recording spec with real user selectors and auto-waiting. Capture failure artifacts; include keyboard/focus and delete-confirmation checks. Do not use arbitrary sleeps.
4. Exercise a manual two-consumer scenario: show CameraViewer, start recording, stop/start viewer demand, switch tabs, stop recording, list/play/delete, then verify the playback ticket is rejected after deletion.
5. Request tester, debugger-on-failure, and code-reviewer subagents. Fix critical/important findings, rerun failed gates, then request explicit user approval before finalization.

## Side-effect checklist

- Auth/session: delete follows existing authenticated socket and rate limit; no client-only authorization.
- API compatibility: versioned events, bounded reason codes, old clients continue list/play/control behavior.
- Data: no DB migration; manifest/MP4 pair and catalog consistency tested.
- Security/privacy: path containment, no-follow checks, no raw path/ticket logging.
- Performance: one capture path, bounded queues, optional burn-in cost measured.
- Deployment: FFmpeg drawtext/font validation and timestamp env documented; direct and Orchestra modes tested.
- Docs/project: update architecture/plan status after approval; do not overwrite unrelated dirty files.

## Release decision — approved

Ready only with fresh command output proving each selected gate, a code-review score/findings record, and no unresolved questions. If browser infrastructure is unavailable, report the exact blocker and the cheapest concrete alternative; do not claim browser readiness.
