# Phase 09 — Live Web E2E

## Context Links

- [Parent plan](./plan.md)
- [Phase 06 UI](./phase-06-web-ui-controls-and-alerts.md)
- [Phase 08 native stack](./phase-08-native-x86-integration-and-benchmark.md)
- Existing spec: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/apps/web/e2e/stream-live.spec.ts`
- Depends on: Phases 06 and 08

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-04 |
| Description | Add and run live Playwright coverage against the native Orchestra/Rover stack. |
| Priority | P1 |
| Implementation status | Completed |
| Review status | Approved |
| Recommended model | GPT-5.4; GPT-5.4-mini for Playwright traces/log summaries |
| Estimated effort | 5h |

## Key Insights

- Existing Playwright setup already seeds socket URL and local test credentials.
- E2E must exercise live backend behavior, not inject test-only status.
- Fake Chromium media can deterministically produce walkie frames.
- Physical audibility is covered in Phase 08; browser E2E verifies observable lifecycle and UX.

## Requirements

### Functional

- Add a dedicated `@edge-voice-live` suite and root/package scripts.
- Authenticate and connect to native web bridge.
- Verify defaults, config convergence, TTS lifecycle, preemption alert, reconnect, and browser STT availability.
- Retain failure screenshots, traces, and video.

### Non-functional

- Serial suite; no parallel global-config mutations.
- Prefer test IDs and typed helper functions over text-only selectors.
- No arbitrary fixed sleep except bounded settling where server event lacks a direct UI condition.
- Tests must fail on backend rejection, timeout, or partial convergence.

## Architecture

```text
Playwright Chromium -> Vite web app -> Socket.IO web bridge
                                      -> native Orchestra/Rover dataflows
                                      -> real Supertonic/status/playback path
```

Launch Chromium with fake media device/UI flags. Use a sufficiently long TTS phrase, wait for `speaking`, start walkie, then assert `interrupted_by_walkie` alert and result.

## Related Code Files

| Action | Absolute path | Purpose |
|---|---|---|
| Create | `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/apps/web/e2e/edge-voice-live.spec.ts` | Live voice suite |
| Modify | `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/apps/web/playwright.config.ts` | Fake media options if env-enabled |
| Modify | `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/apps/web/package.json` | Package test command; merge dirty changes |
| Modify | `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/package.json` | Root test command |

## Implementation Steps

1. Review existing package diff before editing scripts.
2. Add shared auth/open-voice-panel helpers without changing existing live stream suite behavior.
3. Add test: server defaults render as English/M1/Balanced/0.8 and 1/1 applied.
4. Add test: update Vietnamese/config; pending appears; applied revision converges.
5. Add test: send TTS; ack accepted; status reaches speaking then completed/ready.
6. Add test: long TTS + fake-media walkie; visible interruption alert and result reason.
7. Add test: TTS submission disabled while local walkie active.
8. Add test: reconnect restores authoritative config/status without local persistence.
9. Add test: browser STT control remains enabled during rover playback when server STT is ready.
10. Run suite three consecutive times against native stack.
11. On failure, give one bounded trace/log slice to GPT-5.4-mini; main inspects artifact and verifies cause.

## Todo List

- [x] Package diff merged safely
- [x] New E2E script added
- [x] Default/convergence test passes
- [x] TTS lifecycle test passes
- [x] Preemption alert test passes
- [x] Disabled-state test passes
- [x] Reconnect test passes
- [x] Browser STT independence test passes
- [x] Three-run stability passes

## Success Criteria

- `pnpm test:e2e:edge-voice-live` passes against live native stack.
- No test-only backend hooks, mocked Socket.IO, or injected voice status.
- Preemption alert is visible, accessible, and tied to correct rover/command.
- Existing `test:e2e:stream-live` remains passing.
- Type check, lint, package build, and focused component tests pass.

## Completion Notes

- Added `apps/web/e2e/edge-voice-live.spec.ts` and shared `apps/web/e2e/helpers/live-session.ts`.
- Added fake-media Playwright launch support and stable suite-specific output directories.
- Added UI test hooks for voice config, TTS controls, server settings, and panel toggles.
- Fixed `DraggablePanel` viewport overflow so tall live panels scroll instead of rendering controls off-screen.
- Verification passed on 2026-07-05:
  - `pnpm --dir /mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app check-types`
  - `pnpm --dir /mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app lint`
  - `pnpm --dir /mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app build`
  - `pnpm --dir /mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app test:e2e:stream-live`
  - `pnpm --dir /mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app test:e2e:edge-voice-live`
  - Focused `@edge-voice-live` suite passed three consecutive runs against the live native stack.

## Risk Assessment

- Risk: headless fake microphone timing is flaky. Mitigation: event-driven waits and bounded retry only around media startup.
- Risk: TTS completes before preemption. Mitigation: long Vietnamese/English fixture and wait for speaking before walkie.
- Risk: local auth DB reused. Mitigation: dedicated database and deterministic seed credentials.
- Risk: E2E artifacts large. Mitigation: retain only failures; summarize with mini agent.

## Security Considerations

- E2E credentials are local defaults only and supplied through environment/default test fixture.
- Never upload traces containing tokens without sanitization.
- App renders alerts as text, not HTML.

## Next Steps

- Stop native dataflows, keep models, and proceed to [Phase 10](./phase-10-amd64-docker-verification-and-documentation.md).
