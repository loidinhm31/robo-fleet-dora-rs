# Phase 05 — Fleet Transport and Runtime Authority

## Context Links

- [Parent plan](./plan.md)
- [Phase 01 contracts](./phase-01-architecture-and-contract-gate.md)
- [Phase 03 engine](./phase-03-edge-voice-engine.md)
- [Phase 04 playback](./phase-04-source-aware-playback-and-capture-suppression.md)
- Depends on: Phases 01, 03, and 04

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-04 |
| Description | Add global config fan-out, per-rover convergence, and command lifecycle transport. |
| Priority | P1 |
| Implementation status | Completed |
| Review status | Approved |
| User approval | Approved |
| Recommended model | GPT-5.4; GPT-5.4-mini for bounded bridge test logs |
| Estimated effort | 8h |

## Key Insights

- Text commands stay selected-rover targeted; config must bypass selected entity.
- `web_bridge` is desired-state authority; Orchestra bridge is only a runtime delivery cache.
- Offline rovers need config replay on activation.
- UI must never infer fleet success from publish success alone.

## Requirements

### Functional

- Web bridge initializes defaults/revision 0 and validates config updates.
- Client update includes `base_revision`; stale updates return current authoritative state.
- Orchestra bridge fans accepted config to every active rover and caches latest command.
- Rover applies only newer/equal valid revisions and emits status.
- Command ack/result/status traverse both Zenoh and direct modes.
- Late-active rover receives cached config immediately.

### Non-functional

- No MongoDB/localStorage/config-file persistence.
- Bounded update queues and rate limiting.
- No bridge blocks on a slow/offline rover.
- Existing selected-rover TTS path remains backward compatible.

## Architecture

```text
UI tts_config_update(base_revision, config)
  -> web_bridge validate + revision++
  -> orchestra bridge cache + fan-out(active rovers)
  -> rover bridge -> edge_voice
  <- rover voice_status(applied_revision)
  <- orchestra bridge aggregate
  <- web_bridge tts_config_state(desired + per-rover)
```

Direct mode uses the same Socket.IO and Dora contracts. It reports one local rover and skips Zenoh/cache fan-out.

## Related Code Files

| Action | Absolute path | Purpose |
|---|---|---|
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/main.rs` | Desired state, validation, Socket.IO events |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/zenoh_bridge/src/main.rs` | Global fan-out/cache/status subscriptions |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/zenoh_bridge/src/main.rs` | Rover config/status/result mapping |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/orchestra-dataflow.yml` | New inputs/outputs; remove local playback consumer |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/rover-kiwi-dataflow.yml` | Edge voice transport wiring |

## Implementation Steps

1. Extract web-bridge TTS state/validation into focused modules instead of expanding the existing large main file.
2. Initialize desired config from deployment defaults and revision 0.
3. Add authenticated/rate-limited config update handler with compare-and-set `base_revision`.
4. Generate server-side UUID for every accepted text command; emit immediate ack.
5. Add Dora outputs/inputs for config, status, and result in web bridge.
6. Add Orchestra bridge topic helpers and fan-out over active-rover snapshot.
7. Cache only latest config command; replay on inactive→active transition.
8. Add Rover bridge subscriber/output and status/result publisher inputs.
9. Wire default and direct dataflows; delete Orchestra `kokoro-tts` node/edge.
10. Add unit tests for topic names, selected targeting, fan-out, late rover replay, stale updates, and partial convergence.
11. Run bridge tests; delegate long output to GPT-5.4-mini using the required structured report.

## Todo List

- [ ] Web authority module added
- [ ] CAS revision validation added
- [ ] Command UUID/ack added
- [ ] Orchestra fan-out/cache added
- [ ] Rover config/status/result transport added
- [ ] Default dataflow wired
- [ ] Direct dataflow wired
- [ ] Kokoro dataflow edge removed
- [x] Transport tests pass

## Success Criteria

- Text reaches only selected rover.
- Config reaches all active rovers and UI reports exact applied count.
- Newly active rover converges without another client update.
- Offline rover does not falsely count as applied.
- Restart resets desired config to defaults/revision 0.
- Direct mode exposes identical client events.

## Risk Assessment

- Risk: current Orchestra bridge has user modifications. Mitigation: inspect diff and patch narrowly.
- Risk: config update races between clients. Mitigation: `base_revision` compare-and-set.
- Risk: status reordering. Mitigation: ignore applied revisions older than stored rover revision.
- Risk: fan-out holds lock during publish. Mitigation: snapshot IDs/config, release lock, then publish.

## Security Considerations

- Require authenticated socket for config and TTS.
- Use stricter config-update rate limit than audio/command stream.
- Never accept a rover ID list from config-update client payload.
- Sanitize per-rover errors before browser broadcast.

## Next Steps

- Proceed to [Phase 06](./phase-06-web-ui-controls-and-alerts.md) after live Socket.IO payloads are stable.
