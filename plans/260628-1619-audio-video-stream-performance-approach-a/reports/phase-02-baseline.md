# Phase 02 Baseline Evidence

Date: 2026-06-30
Status: Approved with reduced runtime gate

## Implementation Evidence

- Frontend validation, bounded metrics, throttled debug summaries, and long-task observation: implemented.
- Controlled-log analysis helper: `scripts/benchmark-audio-video-stream.sh`.
- Existing recursive playback scheduler and queue thresholds remain unchanged for baseline fidelity.

## Local Verification

- Focused Vitest: 18/18 passed across validator, metrics, and long-task observer tests.
- TypeScript: `pnpm check-types` passed for web and native.
- Builds: `pnpm build` passed for web and native.
- Repository `pnpm lint`: exit 0 but ran zero tasks; not accepted as substantive lint evidence.
- Targeted ESLint: zero errors; two pre-existing `CameraViewer.tsx` warnings outside the Phase 02 audio change.
- Benchmark helper: Bash syntax, valid 20 Hz synthetic run, and invalid-input rejection passed.
- Follow-up UI/runtime verification: `pnpm --filter @robo-fleet/ui test`, `pnpm check-types`, and `pnpm test:e2e:stream-live` passed after the camera stats caveat fix and source-controlled Playwright migration.

## Runtime Environment

- Active Dora flow: `robo-rover-dev` on local temp DB auth-bypass run.
- Temp MongoDB database: `gleanOak_phase02_temp_20260630_0752`.
- Browser app used for live verification: local `robo-control-app` web app.
- Socket.IO backend used for live verification: `http://127.0.0.1:3030`.
- Local workstation NTP offset: 0.474 ms at earlier inspection; rover offset unavailable.
- Required production network path: not provided.
- SLA interpretation: not provided.

## Required Runs

| Scenario | Duration | DevTools | Status |
|---|---:|---|---|
| Audio only | 2 minutes | Headless live browser | Passed |
| Audio and video | 2 minutes | Headless live browser | Passed |
| Source-controlled live e2e | ~30 seconds per spec | Playwright | Passed |
| Debug profiling reproduction | Recorded separately | Open, `?audioDebug=1` | Not run |

## Runtime Evidence

- Temp DB auth-bypass flow created working seeded admin credentials and allowed browser verification against the live rover path.
- 2-minute audio-only run passed on the temp-DB flow with continuous online state and sustained audio frame growth.
- 2-minute audio+video run passed on the temp-DB flow with sustained audio plus live video metrics around 9 fps / ~1.9-2.0 Mbps.
- Original UI caveat was real: camera-off could leave stale non-zero video stats displayed. That was fixed after the reduced runtime run.
- Source-controlled live verification now exists in `robo-control-app/apps/web/e2e/stream-live.spec.ts` and passed:
  - audio + video stream reaches live non-zero stats
  - camera off drives video stats back to zero while audio continues

## Gate Decision

Approved by user on a reduced gate. Phase 03 may proceed.

Scope note:

- This approval is based on reduced 2-minute runtime runs, not the original 10-minute matrix required by the initial phase contract.
- The reduced gate was explicitly accepted by the user in order to close the phase and proceed.
- If stricter release evidence is needed later, rerun the original 10-minute matrix against the same source-controlled live e2e path.

## Unresolved Questions

- Which network path must pass: localhost, LAN, Tailscale, or proxy/tunnel?
- Is 150 ms a scheduled-start target or a hardware-audible SLA?
- When needed, should the original 10-minute matrix be rerun as a release gate or remain superseded by this reduced approved gate?
