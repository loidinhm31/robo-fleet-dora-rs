# Phase 02 — Source-Aware Command Routing

## Context Links

- Parent: [plan.md](./plan.md)
- Depends on: [Phase 01](./phase-01-web-bridge-dual-source-transport.md)
- Contract: `robo_rover_lib/src/types/speech_types.rs`
- Parser: `orchestra/command_parser/src/main.rs`
- Bridge: `orchestra/zenoh_bridge/src/main.rs`

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-03 |
| Description | Preserve authoritative transcript target through parsing and reject unsafe selected-rover fallback. |
| Priority | P1 |
| Implementation status | Complete |
| Review status | Approved |
| Effort | 6h |

## Key Insights

- `SpeechTranscription.target_entity_id` already exists, but parser output parameters are empty.
- Orchestra bridge parser channels still route to the UI-selected rover.
- Dora metadata is the common target envelope across heterogeneous command payloads.
- Automatic parser TTS can feed rover speaker output back into rover STT.

## Requirements

- Deterministic parser remains the only speech-to-actuator interpreter.
- Every parser-derived actuator output carries required target metadata.
- Parser channels reject missing, malformed, or inactive targets without fallback.
- Manual web commands continue to use current selected rover.
- Remove automatic parser TTS; preserve manual UI TTS.

## Architecture

```text
SpeechTranscription.target_entity_id
  -> parser validates and copies Dora metadata
  -> orchestra bridge validates active rover
  -> rover/{target}/cmd/*
```

## Related Code Files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/command_parser/src/main.rs`: target helper, metadata propagation, TTS removal.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/zenoh_bridge/src/main.rs`: parser-target enforcement and metrics.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/orchestra-dataflow.yml`: remove parser TTS edges/output.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/command_parser/Cargo.toml`: test dependencies only if required.

## Implementation Steps

1. Deserialize the full transcription contract; reject empty target, stream, and utterance identity before intent parsing.
2. Keep parser intent-confidence policy separate from absent STT confidence.
3. Build one target metadata helper containing `target_entity_id`, `utterance_id`, `source_kind`, and optional source `entity_id`.
4. Route every parser output through one send helper so rover, tracking, camera, and future actuator paths cannot omit metadata.
5. Remove parser-generated TTS feedback, its output declaration, Kokoro/parser input, and bridge parser-TTS match arm.
6. Preserve web/manual TTS and other manual command paths.
7. In Orchestra bridge, classify parser and web inputs explicitly.
8. For parser input, require target metadata and validate target against active rover subscriptions before topic construction.
9. Reject missing/inactive targets with structured metrics and warnings. Never substitute selected rover.
10. Keep web inputs on selected-rover routing.
11. Add table-driven parser tests for every supported intent/output.
12. Add bridge tests for rover A while B selected, browser target after selection change, inactive target rejection, and manual web behavior.

## Todo List

- [x] Separate intent confidence from STT confidence.
- [x] Add common target metadata helper.
- [x] Apply helper to every parser actuator output.
- [x] Remove automatic parser TTS.
- [x] Enforce active target on parser bridge channels.
- [x] Preserve selected-rover routing for manual web channels.
- [x] Add parser/bridge routing tests.

## Success Criteria

- Rover A speech can publish only to rover A command topics.
- Browser speech stays on target captured at stream start.
- Missing/inactive target yields no Zenoh publication.
- Manual UI commands still follow current selected rover.
- No automatic parser TTS command is emitted.

## Risk Assessment

- One output path can miss metadata. Centralize sends and test all output variants.
- Active roster can change between capture and publication. Reject stale target rather than retarget.
- Removing audible feedback changes UX. Preserve textual feedback and manual TTS.

## Security Considerations

- Validate target again at the final network boundary.
- Do not accept a JSON payload target as a substitute for Dora metadata.
- Log IDs and rejection reason, never audio or credentials.

## Next Steps

Completed 2026-07-03. Proceed to Phase 03 UI delivery with source-target isolation and manual-command regressions preserved.

## Unresolved Questions

None.
