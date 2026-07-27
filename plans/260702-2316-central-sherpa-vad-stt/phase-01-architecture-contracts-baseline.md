# Phase 01 — Architecture, Contracts, and Baseline

## Context Links

- Parent: [plan.md](./plan.md)
- Research: [research synthesis](./research/research-synthesis.md)
- Brainstorm: [source report](../reports/brainstorm-260702-1750-central-sherpa-stt.md)
- Architecture: `/mnt/data/ws/sharing/robo-fleet-dora-rs/ARCHITECTURE.md`
- UI architecture: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/docs/architecture.md`

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-02 |
| Description | Lock dual-source STT architecture, public contracts, invariants, and pre-change baseline. |
| Priority | P1 |
| Implementation status | Complete |
| Review status | Approved (2026-07-03) |
| Completed | 2026-07-03 |
| Effort | 5h |

## Key Insights

- Existing `SpeechTranscription.confidence: f32` cannot represent Sherpa's lack of confidence.
- Browser and rover sources need one contract but different visibility and targeting policies.
- Architecture docs still describe Whisper and selected-rover routing.
- Contract changes span Rust, Socket.IO TypeScript, UI state, parser, and bridge.

## Requirements

- Define source, target, stream, utterance, profile, and status fields before backend implementation.
- Keep contract final-only; do not add `is_final` or partial-result events.
- Make new fields explicit; allow only confidence to be absent.
- Document browser privacy and command target invariants.
- Capture baseline English behavior and resource measurements before deleting Whisper.

## Architecture

```text
browser mic -> web bridge -> central STT -> parser -> captured selected rover
rover mic -> rover/orchestra bridges -> central STT -> parser -> source rover
central STT -> web bridge -> origin browser OR fleet rover transcription panel
```

Status is global because one startup profile serves every source. Audio/VAD state remains per stream.

## Related Code Files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/ARCHITECTURE.md`: intended STT dataflow, contracts, invariants.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/speech_types.rs`: Rust contracts.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/voice.ts`: matching TypeScript contracts.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/socket.ts`: Socket.IO event types.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/docs/architecture.md`: UI event/dataflow.

## Implementation Steps

1. Record current Whisper output, latency, CPU, and RSS using a short labeled English command corpus. Store measurements in the phase completion report; do not commit large audio fixtures.
2. Update both architecture documents before implementation. Add dual-source flow, source isolation, status lifecycle, UI visibility, and target routing.
3. Add `SttProfile` with only `en-vad-offline` and `vi-vad-offline`.
4. Add `SttState` with `loading`, `ready`, and `error`.
5. Add `SttSourceKind` with `browser` and `rover`.
6. Extend `SpeechTranscription` with `confidence: Option<f32>`, `utterance_id`, `stream_id`, `source_kind`, optional `entity_id`, required `target_entity_id`, and `profile`.
7. Add `SttStatus { state, profile, language, timestamp, error }`.
8. Mirror exact JSON names and nullability in TypeScript.
9. Define browser client events `voice_command_control` and `voice_command_audio`; define server events `voice_command_transcription` and `stt_status`.
10. Add Rust serde round-trip tests and TypeScript compile-time fixtures for browser and rover examples.
11. Confirm command parser remains the only actuator interpretation path; explicitly defer AI interpretation.

## Todo List

- [x] Capture Whisper baseline.
- [x] Update root architecture.
- [x] Update UI architecture.
- [x] Add Rust STT contracts.
- [x] Add TypeScript STT contracts.
- [x] Add serialization tests.
- [x] Review invariants across both repositories.

## Completion Evidence

- Completed: 2026-07-03.
- Contracts and invariants locked for browser and rover sources.
- Baseline and target routing documented before implementation cutover.
- Rust/TypeScript shape parity and serializer coverage reviewed.
- Validation passed: `cargo test -p robo_rover_lib -p central_speech_recognizer -p command_parser -p web_bridge`.
- Validation passed: `pnpm --dir /mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app check-types`.

## Success Criteria

- Rust and TypeScript contracts describe identical wire shapes.
- Browser transcript has no rover `entity_id` but has stream and target.
- Rover transcript has matching `entity_id` and target.
- Missing confidence serializes as `null` or omitted according to one documented rule and UI accepts both.
- Architecture shows no runtime profile control or partial result path.
- Baseline results are reproducible for rollback comparison.

## Risk Assessment

- Risk: Coordinated contract deployment breaks older consumers. Mitigation: land Rust and UI changes in one release and keep backward parsing only for optional confidence.
- Risk: Source and target become conflated. Mitigation: separate `source_kind`, `entity_id`, and `target_entity_id` fields with fixtures.

## Security Considerations

- Never trust browser-supplied target identity; web bridge assigns it from authoritative fleet state.
- Do not include Socket.IO socket IDs in public transcript payloads.
- Do not expose absolute model paths in UI errors.

## Next Steps

Proceed to Phase 02. Keep field names stable unless both architecture docs and wire fixtures update together.
