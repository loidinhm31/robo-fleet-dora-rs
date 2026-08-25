# Phase 06: End-to-End Verification and Rollout

## Context links

- [Parent plan](./plan.md)
- [Phases 01-05](./plan.md#phases)
- [Architecture contract](../../ARCHITECTURE.md#manual-fleet-media-recording-and-playback)
- Depends on: all implementation phases. Final acceptance gate.

## Overview

- Date: 2026-07-18
- Description: Prove media correctness, isolation, security, UI playback, and native/Podman deployment before rollout; observed run covered rover-kiwi only.
- Priority: P1 acceptance
- Implementation status: Verified for rover-kiwi-only amd64 scope
- Review status: Approved with live two-rover limitation
- Effort: 5h

## Key Insights

- A successful build does not prove playable synchronized output, demand isolation, range seeking, or host persistence.
- Acceptance must inspect the filesystem and media streams, not trust recorder status alone.
- Two-repository coordination needs a frozen contract fixture and explicit revision/commit handoff.

## Requirements

- Run focused unit/integration suites, full relevant Cargo gates, UI type/lint/build/tests, and live A/V smoke.
- Use synthetic deterministic frames first, then real rover/bridge traffic when available.
- Verify two concurrent rover sessions, viewer/recorder demand overlap, failures, reconnect, and shutdown.
- Validate MP4 with ffprobe and actual browser playback/range seeking.
- Confirm output root contains finalized MP4/manifests only; no raw JPEG files anywhere under it.
- Validate native x86_64 and current amd64 Docker-compatible Podman flow; do not claim ARM acceptance.

## Architecture

- Contract gate: Rust golden JSON equals TypeScript fixtures/event maps.
- Media gate: capture timestamps, duration, codecs, resolution, audio presence/silence, gaps, and skew remain within documented tolerances.
- Isolation gate: pressure/failure in one recorder never stalls another rover, live view, STT, tracking, or controls.
- Deployment gate: host bind, non-root ownership, SELinux, restart catalog, and range playback work end to end.

## Related code files

- Modify tests created in Phases 01-05 based on failures; do not add a second implementation path.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docs/codebase-summary.md` after implementation — actual recorder/dataflow/deployment state.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/ARCHITECTURE.md` only if implementation intentionally differs; otherwise fix drift in code.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/README.md` and relevant root docs with verified commands/config.
- Add concise acceptance evidence under the active plan's `/mnt/data/ws/sharing/robo-fleet-dora-rs/plans/260717-0856-rover-media-recording-control/reports/` directory.

## Implementation Steps

1. Run formatting/check/tests for shared library, recorder, Orchestra bridge, and web bridge; run full affected workspace tests.
2. In UI checkout, run shared/UI Vitest, type checks, lint, builds for web/Tauri packages, and Playwright recording workflow.
3. Generate deterministic JPEG and 16 kHz mono PCM for two entity IDs; record overlapping sessions with a viewer demand present; stop in reverse order.
4. Assert demand transitions, no cross-routing, independent filenames/statuses, duration/size limits, and continued live/control responsiveness.
5. Run ffprobe for container/codecs/streams/duration and browser seek tests for initial, middle, suffix, invalid, and expired-ticket ranges.
6. Search the configured root recursively for `.jpg`/`.jpeg`; inspect partial/final/manifest states across crash, SIGTERM, FFmpeg failure, disk-full/unwritable, missing audio, and restart.
7. Export `XDG_RUNTIME_DIR=/run/user/$(id -u)`; run `docker info`, real Docker-compatible Podman smoke, Orchestra build/up, writable bind/readiness, record/play, restart, and ownership checks.
8. Run an architecture drift review and code review. Update docs/evidence only with observed results; note physical rover or ARM checks separately if unavailable.
9. Roll out behind signed-in-session access and conservative concurrency/duration limits; monitor encoder exits, queue drops, A/V gaps, disk free space, ticket failures, and finalization latency.

## Todo list

- [x] Pass backend and UI quality gates.
- [ ] Pass two-rover deterministic A/V test. Live two-rover isolation remains unverified because only rover-kiwi was active.
- [x] Pass auth/path/range/ticket security tests.
- [x] Prove no raw JPEG persistence.
- [x] Pass native and Podman amd64 smoke.
- [x] Review architecture drift and update docs/evidence; physical/ARM and live two-rover scope remain open.

## Success Criteria

- Admin path/record/stop/list/play workflow succeeds in web and Tauri against the same backend contract.
- Concurrent rover MP4s are playable/seekable, contain H.264 + AAC, and meet documented duration/skew tolerance.
- Viewer/recorder overlap releases resources correctly; one session failure does not affect another or live/control paths.
- Traversal, symlink, unauthorized, expired ticket, malformed range, full disk, missing encoder, and restart cases fail safely.
- Recursive output scan finds zero raw JPEG files; only verified finalized clips appear in catalog.
- Evidence clearly labels workstation amd64 acceptance and any remaining physical/ARM gap.
- Observed acceptance evidence currently covers rover-kiwi only; live two-rover isolation remains a required follow-up.

## Risk Assessment

- Risk: synthetic success hides network jitter. Mitigation: add real Zenoh/Dora soak when hardware is available and record observed counters.
- Risk: codec/container differs across environments. Mitigation: pin/assert runtime and compare ffprobe outputs native/container.
- Risk: separate repo versions drift. Mitigation: freeze event fixtures and record tested revisions in acceptance report.

## Security Considerations

- Include negative tests for authentication bypass, replay, ticket guessing/expiry, traversal/symlink races, oversized ranges, and log leakage.
- Sanitize reports: no JWTs, playback tickets, absolute private paths, credentials, or user media content.
- Rollback disables recording admission without disabling rover control/live media.

## Next steps

- After all gates, implementer may mark phases complete and hand off verified runbooks.
- Future schedules, retention, protection/delete, quotas, cloud storage, or ARM tuning require separate plans.

## Unresolved questions

- Physical multi-rover acceptance still needs a live run with both rovers active; workstation amd64 remains the required current target.
