# Media control regression fixes

Status: Completed · Date: 2026-07-18

## Preflight contract

- Output: UI-only fix for reliable Camera Off and cross-origin recording playback, with regression tests.
- Acceptance: Camera Off stops an active browser stream and sends camera stop; a relative ticket targets the configured bridge origin; a media error becomes stable until the user retries.
- Scope: `CameraViewer`, recording page/panel, parent URL wiring, and their tests. No recorder, bridge protocol, database, Docker, scheduler contract, or visual redesign.
- Risk/contracts: keep authenticated ticket issuance/expiry unchanged; only accept HTTP(S) configured playback origins; preserve same-origin deployments.
- Test strategy: focused Vitest UI tests, TypeScript check, then authenticated Socket.IO start/stop/clip/playback smoke against the Docker stack if available.
- Open questions: none.

## Phases

1. [Phase 1 — Fix and verify media controls](phase-01-fix-and-verify-media-controls.md) — Completed

## Side-effect review

- Auth/session: playback ticket still comes only from the authenticated socket flow.
- Compatibility: backend continues emitting relative URLs; the UI resolves only against its configured bridge origin.
- Data: no migrations or persisted-data changes.
- Security: reject protocol-relative, malformed, and non-HTTP(S) playback origins/paths; do not trust a ticket to select an arbitrary origin.
- Resources: Camera Off releases stream demand before camera demand; no retry storm after video failure.
- Docs/config/deploy: no new configuration; existing Socket.IO URL is reused.
