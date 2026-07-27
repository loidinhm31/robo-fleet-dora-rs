# Phase 01 — Web Bridge Dual-Source Transport

## Context Links

- Parent: [plan.md](./plan.md)
- Research: [current-state report](./research/current-state-and-sherpa-report.md)
- Completed central runtime: `orchestra/central_speech_recognizer/`
- Architecture: `ARCHITECTURE.md#stt-runtime-and-contract`

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-03 |
| Description | Add bounded authenticated browser stream ownership, status recovery, and source-specific transcript delivery. |
| Priority | P1 |
| Implementation status | Complete — 2026-07-03 |
| Review status | Approved — 2026-07-03 |
| Progress | 100% |
| Effort | 10h |

## Key Insights

- Central already accepts `audio_browser`, `browser_control`, and `stt_status_request`; dataflow/web bridge do not provide the full contract.
- Legacy browser audio buffering is unbounded and has no authoritative owner, target snapshot, or sequence lifecycle.
- Browser results require server-side privacy routing; client-side filtering is insufficient.
- Existing `common/web_bridge/src/main.rs` is large. STT state and validation need focused modules.

## Requirements

- Authenticated start/audio/stop lifecycle with one socket owner per stream UUID.
- Target derived from authoritative selected and active rover state at start.
- Bounded queue, monotonic frame IDs, finite sample validation, explicit overload reset.
- Closing ownership retained until final result or bounded timeout.
- Cached authoritative STT status emitted on every authenticated connection.
- Browser results private; rover results broadcast to authenticated fleet clients.

## Architecture

```text
socket start -> stream registry(owner,target,format,next frame)
socket audio -> validate -> bounded Dora queue -> central/audio_browser
socket stop/disconnect -> central/browser_control -> flush -> closing registry
central transcription -> browser owner OR authenticated rover broadcast
central status -> cache -> connect/reconnect replay
```

## Related Code Files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/main.rs`: compose STT bridge and route Dora/Socket.IO events.
- Create focused `common/web_bridge/src/stt_*` modules: bridge/ingress queueing, Dora protocol conversion, Socket.IO delivery, ownership lifecycle/state, and transcript routing.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/security.rs`: validation/rate-limit helpers.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/orchestra-dataflow.yml`: browser control, status request/status, and transcription edges.
- Verify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/zenoh_bridge/src/main.rs` has no browser voice forwarding path; remove the obsolete dataflow edge.

## Implementation Steps

1. Reconcile current user edits in `orchestra-dataflow.yml`; preserve unrelated CORS/comment changes.
2. Define stream registry entry: stream ID, owner socket, target rover, sample rate, channels, expected frame, active/closing state, expiry.
3. Bound active/closing stream counts and queued audio/control messages. Document drop-newest plus stream-reset behavior.
4. Add authenticated `voice_command_control:start`; validate UUID, mono sample rate, unique ownership, selected active rover, and rate limit.
5. Snapshot target server-side. Reject all client target/entity fields.
6. Add `voice_command_audio`; verify owner, active state, exact format/count, finite samples, expected frame, and maximum payload.
7. Convert accepted frames to Dora `Float32Array` plus central-required metadata.
8. On gap, overload, stop, disconnect, session expiry, or idle timeout, emit one stop control and transition to closing.
9. Retain closing owner until final browser result or expiry; metric-count late/orphan results.
10. Parse central transcription once. Emit browser source only to owner; broadcast rover source only to authenticated clients.
11. Cache `SttStatus`; replay after auth. Emit one `stt_status_request` when cache is absent.
12. Wire `browser_control`, `stt_status_request`, `stt_status`, and final transcription edges in Orchestra dataflow.
13. Remove `voice_audio_web` from Orchestra bridge; browser command audio remains local to Orchestra.
14. Add unit/integration tests for spoofing, duplicate streams, sequence faults, capacity, disconnect flush, target snapshot, privacy, rover broadcast, and reconnect status.

## Todo List

- [x] Extract STT registry/transport modules.
- [x] Implement bounded ownership lifecycle.
- [x] Add authenticated start/audio/stop handlers.
- [x] Add status cache/request/replay.
- [x] Add private browser and fleet rover routing.
- [x] Complete Dora dataflow edges.
- [x] Remove obsolete Zenoh browser-audio path.
- [x] Add backend transport/privacy tests.

## Success Criteria

- One socket cannot write to or receive another socket's browser stream.
- Fleet selection changes do not retarget an active browser utterance.
- Queue/state growth is bounded and overload is observable.
- Stop/disconnect flushes central state without leaked owners.
- Reconnecting clients receive current status.
- Rover transcriptions retain source identity and fleet visibility.

## Risk Assessment

- Late decode after ownership expiry can drop a valid result. Size timeout above maximum speech plus measured P99 decode latency.
- Shared locks can stall media delivery. Never emit Socket.IO or Dora data while holding registry locks.
- Current dirty dataflow edits can be lost. Review and preserve them before patching.

## Security Considerations

- Apply authentication, session expiry, per-socket ownership, size/rate limits, and finite-value checks on every event.
- Never trust target/entity fields from browsers.
- Do not expose socket IDs, model paths, or raw audio in public events/logs.

## Next Steps

Phase 01 implementation and review complete. Proceed to [Phase 02 — Source-Aware Command Routing](./phase-02-source-aware-command-routing.md).

## Unresolved Questions

None.
