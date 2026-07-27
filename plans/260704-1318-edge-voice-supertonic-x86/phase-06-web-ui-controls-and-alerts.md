# Phase 06 — Web UI Controls and Alerts

## Context Links

- [Parent plan](./plan.md)
- [Locked decisions](./reports/01-locked-decisions-and-model-routing.md)
- [Phase 05 transport](./phase-05-fleet-transport-and-runtime-authority.md)
- UI instructions: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/CLAUDE.md`
- Depends on: Phases 01 and 05

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-04 |
| Description | Add authoritative global TTS controls, convergence display, and clear walkie-preemption alerts. |
| Priority | P1 |
| Implementation status | Done |
| Review status | Approved |
| Approved at | 2026-07-05 09:40 +07 |
| Recommended model | GPT-5.4; GPT-5.4-mini for test-output classification |
| Estimated effort | 6h |

## Key Insights

- `VoiceControls` already owns TTS, walkie, and browser STT interactions.
- No external state library is needed; server state remains authoritative.
- User specifically requires visible feedback when walkie blocks/intercepts TTS.
- `apps/web/package.json` is already user-modified and must be merged carefully.

## Requirements

### Functional

- Controls: English/Vietnamese, F1–F5/M1–M5, speed, quality steps, volume.
- Show desired revision and `applied/active` rover count.
- Send full config with current `base_revision`; no optimistic success state.
- Disable local TTS submit while local walkie is active.
- Show accessible alert for rejected/interrupted/failed TTS, including rover and reason.
- Preserve browser STT and current walkie/browser-capture mutual exclusion.

### Non-functional

- No localStorage authority or persisted config.
- Debounce sliders; send final values, not every pointer event.
- Reconnect requests/receives current server state.
- Shared TypeScript event maps remain the single UI wire contract.

## Architecture

```text
RoboRoverControl socket
   ├── tts_config_state -> VoiceControls desired/applied state
   ├── voice_status -> rover status map
   ├── tts_command_ack/result -> command feedback + alert
   └── VoiceControls -> tts_config_update / tts_command
```

Alert behavior:

- `walkie_active`: "TTS not started: live walkie-talkie has priority."
- `interrupted_by_walkie`: "Rover speech stopped because live walkie-talkie started."
- Use `role="alert"`, visible until dismissed or replaced; also call existing `onLog`.

## Related Code Files

| Action | Absolute path | Purpose |
|---|---|---|
| Modify | `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/VoiceControls.tsx` | Controls/state/alerts |
| Modify | `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/VoiceControls.test.tsx` | Component regressions |
| Modify | `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/pages/RoboRoverControl.tsx` | Socket subscriptions/props |
| Modify | `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/voice.ts` | TTS domain types |
| Modify | `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/socket.ts` | Event maps |

## Implementation Steps

1. Inspect UI dirty diff, especially package script changes; preserve it.
2. Add typed TTS config/status/result state in page-level socket owner.
3. Subscribe/unsubscribe using stable handlers; request state after authentication/reconnect.
4. Pass authoritative state and update callbacks into `VoiceControls`.
5. Add compact global/fleet-labeled controls using existing atoms/molecules.
6. Map quality labels: Fast=5, Balanced=8, Quality=12.
7. Debounce speed/volume changes; use current revision in update request.
8. Track pending revision until server state confirms or rejects.
9. Add alert component/state and log integration for rejection/interruption/failure.
10. Disable TTS send during walkie and explain disabled state.
11. Add component/socket tests; run type check, focused unit tests, and lint.

## Todo List

- [x] Dirty UI diff reviewed
- [x] Socket state handlers added
- [x] Config controls added
- [x] Pending/convergence state added
- [x] Walkie priority alert added
- [x] Reconnect behavior added
- [x] Component tests added
- [x] Type check/lint pass

## Success Criteria

- UI starts from server defaults, never stale browser storage.
- Config update displays pending then exact convergence.
- Walkie-active TTS button is disabled with explanation.
- Remote-client walkie preemption still produces visible alert through server result.
- Existing browser STT and walkie regression tests continue passing.
- Both web and Tauri builds consume the same shared component/types.

## Risk Assessment

- Risk: VoiceControls grows beyond maintainable size. Mitigation: extract TTS config panel and alert into focused components.
- Risk: event listener leaks on reconnect. Mitigation: explicit cleanup tests.
- Risk: controls flood updates. Mitigation: debounce and base revision.
- Risk: transient status missed. Mitigation: result events plus current status state.

## Security Considerations

- Treat all server error strings as text; never render HTML.
- Keep authentication gating for both TTS and config controls.
- Do not expose model path/provider/debug controls.

## Next Steps

- Proceed to [Phase 07](./phase-07-docker-cleanup-and-local-mongodb.md).
- Keep the reconnect voice-state replay contract explicit in future protocol docs/tests.
