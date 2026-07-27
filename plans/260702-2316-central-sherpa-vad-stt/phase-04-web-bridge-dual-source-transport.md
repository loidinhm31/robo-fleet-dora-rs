# Phase 04 — Web Bridge Dual-Source Transport

## Context Links

- Parent: [plan.md](./plan.md)
- Depends on: [Phase 01](./phase-01-architecture-contracts-baseline.md), [Phase 03](./phase-03-central-vad-recognizer.md)
- Current bridge: `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/main.rs`
- Dataflow: `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/orchestra-dataflow.yml`

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-02 |
| Description | Make browser voice transport bounded, source-aware, target-stable, private, and status-aware. |
| Priority | P1 |
| Implementation status | Pending |
| Review status | Pending |
| Effort | 10h |

## Key Insights

- Existing authenticated handler already accepts `voice_command_audio`, but the UI never emits it.
- Current `Vec<WebAudioStream>` queue is unbounded and has no stream/sequence/owner metadata.
- Browser transcripts must return only to the socket that owns the stream.
- Selected rover can change during speech; target must be captured once at start.
- Newly connected UI needs authoritative status even if it missed central startup events.

## Requirements

- Use explicit browser start/audio/stop lifecycle.
- Server, not browser, assigns target from current `FleetStatus.selected_entity`.
- Enforce one owner per stream UUID and monotonic frame IDs.
- Bound queued browser audio and account for drops.
- Keep closing ownership long enough to route flushed final results.
- Route rover results to all authenticated clients and browser results only to owner.

## Architecture

```text
Socket start -> ownership {stream -> socket,target,format,last_frame}
Socket audio -> validate owner/sequence -> bounded Dora queue -> central/audio_browser
Socket stop/disconnect -> central/browser_control flush -> closing ownership
central result -> source_kind switch -> owner emit OR authenticated broadcast
central status -> cache -> emit on connect/reconnect
```

## Related Code Files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/main.rs`: compose new STT bridge module and outputs/inputs.
- Create focused STT transport/state modules under `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/` to reduce `main.rs` growth.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/security.rs`: browser voice validation helpers.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/orchestra-dataflow.yml`: central/web/bridge edges.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/zenoh_bridge/src/main.rs`: remove unused `voice_audio_web` path.

## Implementation Steps

1. Define server-side browser stream state: owner socket, target entity, sample rate, channels, next frame, state `active|closing`, and expiry.
2. Add bounded `VecDeque` or bounded channel for browser audio/control messages; select one capacity constant derived from the central queue and document drop policy.
3. On authenticated `voice_command_control:start`, validate UUID/rate/mono, reject duplicate ownership, snapshot selected rover, verify it is active, store state, and enqueue start metadata.
4. On `voice_command_audio`, require active ownership by calling socket, exact stream metadata, expected frame ID, finite values, and existing maximum samples.
5. Reject browser-provided entity/target fields even if present.
6. On queue full, drop newest frame, increment source-specific metrics, and preserve sequence behavior by terminating/resetting the affected stream rather than silently creating a gap.
7. On explicit stop, mark stream closing, enqueue stop, and retain owner mapping until final result or timeout.
8. On disconnect/session expiry/idle sweep, enqueue stop for every owned active stream and mark mappings closing.
9. Emit browser Float32Array to `voice_command_audio` Dora output with parameters: source kind, stream/frame identity, sample dimensions/format, and server-assigned target.
10. Emit browser start/stop JSON to `voice_command_control` Dora output.
11. Add `stt_status` input handling and cached `Option<SttStatus>` in shared state.
12. After authentication, emit cached status; if absent, enqueue one internal `stt_status_request` Dora output.
13. Parse central transcription once. For browser source, resolve stream owner and emit `voice_command_transcription` only to that socket; remove mapping after final result when closing.
14. For rover source, broadcast `transcription` to authenticated namespace clients with `entity_id` intact.
15. Sweep expired closing mappings without leaking socket IDs or retaining abandoned streams indefinitely.
16. Update Orchestra dataflow: central gets rover audio directly from Orchestra bridge and browser audio/control/status request from Web bridge; Web bridge gets central status/transcription.
17. Remove `voice_audio_web` input from Orchestra bridge because browser command audio is local to Orchestra.
18. Add tests for auth, ownership spoofing, duplicate UUID, sequence gaps, full queue, target snapshot, disconnect flush, status reconnect, private browser emit, and rover broadcast.

## Todo List

- [ ] Extract STT bridge state/modules.
- [ ] Implement ownership lifecycle.
- [ ] Bound queues and define drop/reset policy.
- [ ] Implement start/audio/stop handlers.
- [ ] Implement status cache/request.
- [ ] Implement source-specific transcript routing.
- [ ] Update Dora dataflow.
- [ ] Remove unused Orchestra bridge voice input.
- [ ] Add transport/privacy tests.

## Success Criteria

- A socket cannot write to or receive another socket's browser stream.
- Target stays fixed through fleet selection changes.
- Disconnect and stop flush central state without orphan ownership.
- Queue overload is bounded, observable, and resets the affected stream.
- New clients obtain current STT status.
- Rover transcriptions remain broadcast and source-labeled.

## Risk Assessment

- Risk: Result arrives after ownership timeout. Mitigation: timeout exceeds maximum speech plus worst decode latency and logs late results.
- Risk: Shared-state lock contention. Mitigation: short critical sections; never emit Socket.IO/Dora while holding ownership lock.
- Risk: JSON F32 overhead. Mitigation: 50 ms bounded frames; defer binary protocol until measurements justify it.

## Security Considerations

- Apply existing authentication, session expiry, command rate limit, activity tracking, and audio finite-value validation.
- Validate stream ownership on every message.
- Derive target only from authoritative server state.
- Browser transcript privacy is mandatory, not a UI-only filter.

## Next Steps

Proceed to Phase 05 after backend tests demonstrate private routing and stable command targeting.
