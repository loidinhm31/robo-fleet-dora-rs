# Phase 01 — Architecture and Contract Gate

## Context Links

- [Parent plan](./plan.md)
- [Locked decisions](./reports/01-locked-decisions-and-model-routing.md)
- [Repository scout](./scout/01-repository-integration-surface.md)
- [Current architecture](../../ARCHITECTURE.md)
- Depends on: none

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-04 |
| Description | Freeze cross-process contracts and update planned architecture before code changes. |
| Priority | P1 |
| Implementation status | Completed |
| Review status | Approved |
| Recommended model | GPT-5.5 |
| Estimated effort | 4h |

## Key Insights

- TTS text routing exists; configuration, status, result, and correct audio contracts do not.
- Desired global config and per-rover applied state must remain distinct.
- Command acceptance cannot imply playback completion.
- Dora metadata plus Arrow F32 payloads already solve the PCM envelope without JSON samples.
- Architecture must show walkie preemption and microphone suppression as safety behavior.

## Requirements

### Functional

- Define version-stable Rust and TypeScript contracts.
- Define Dora inputs/outputs and Zenoh/Socket.IO event names.
- Define lifecycle, revisions, command IDs, and error semantics.
- Update architecture as a planned/target design, clearly separated from current-state notes.

### Non-functional

- Use bounded enums and values; no arbitrary model path/provider from clients.
- Preserve backward-compatible `tts_command { text }` Socket.IO input.
- Keep TTS runtime state process-local and non-persistent.
- Keep contract modules focused and under repository size guidance.

## Architecture

### Rust contracts

```rust
enum TtsLanguage { En, Vi }

struct TtsRuntimeConfig {
    language: TtsLanguage,
    speaker_id: u8,
    speed: f32,
    num_steps: u8,
    volume: f32,
}

struct TtsConfigCommand { revision: u64, config: TtsRuntimeConfig }

struct TtsCommand {
    command_id: String,
    text: String,
    timestamp: u64,
    priority: TtsPriority,
}
```

Add `VoiceState`, `VoiceStatus`, `TtsAckState`, `TtsCommandAck`, `TtsResultState`, `TtsCommandResult`, `PlaybackSource`, and `PlaybackState`. Every error exposed outside a node is a bounded reason code plus sanitized optional detail.

### Event contracts

```text
Socket client -> server: tts_command, tts_config_update
Socket server -> client: tts_command_ack, tts_command_result,
                        tts_config_state, voice_status

Zenoh: rover/{id}/cmd/tts
       rover/{id}/cmd/voice/config
       rover/{id}/voice/status
       rover/{id}/voice/result
```

### PCM contract

Payload is `Float32Array`. Metadata includes `source_kind`, `command_id` or `stream_id`, `frame_id`, `capture_timestamp_ms`, `sample_rate`, `channels`, `sample_count`, `format=f32le`, and priority. Reuse `AudioFrameMetadata` validation; source-specific fields stay in Dora parameters.

## Related Code Files

| Action | Absolute path | Purpose |
|---|---|---|
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/ARCHITECTURE.md` | Target dataflow, invariants, lifecycle, deployment |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/tts_types.rs` | Shared Rust contracts |
| Modify | `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/voice.ts` | Mirrored UI contracts |
| Modify | `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/socket.ts` | Typed Socket.IO events |

## Implementation Steps

1. Inspect dirty diffs in both repositories; record overlap before edits.
2. Add target architecture flow: UI → web bridge → Orchestra bridge → rover bridge → edge voice → audio playback.
3. Add global config fan-out and desired/applied revision sequence.
4. Add playback/suppression sequence and walkie-preemption state transitions.
5. Implement Rust types with serde casing and validation helpers.
6. Mirror exact wire shapes in TypeScript; add compile-time fixtures.
7. Add serialization golden tests for all enums and optional fields.
8. Review contract names against both default and direct rover dataflows.

## Todo List

- [x] Dirty diff reviewed
- [x] Architecture updated
- [x] Rust contracts added
- [x] TypeScript contracts mirrored
- [x] Serialization tests added
- [x] GPT-5.5 architecture review complete

## Success Criteria

- Architecture shows every owner and cross-machine hop.
- Rust/TypeScript payload fixtures match byte-for-byte JSON shapes.
- No unresolved naming or lifecycle decision remains for later phases.
- Existing `tts_command { text }` clients remain valid.

## Risk Assessment

- Risk: too many status types. Mitigation: separate immediate ack, command result, and long-lived voice status only.
- Risk: type drift between repositories. Mitigation: golden fixtures and shared event-name table.
- Risk: architecture doc claims unimplemented behavior. Mitigation: label section as target until final phase changes it to current.

## Security Considerations

- Client never supplies filesystem paths, providers, entity-wide publish topics, or revision authority.
- Validate finite floats before serialization.
- Error details must not reveal absolute model paths.

## Next Steps

- Completed 2026-07-04 17:59 ICT after review fixes, local validation pass, and user approval.
- Proceed to [Phase 02](./phase-02-model-cache-reset-and-bootstrap.md) after contract review passes.
