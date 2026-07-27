# Phase 05 — Source-Aware Command Routing

## Context Links

- Parent: [plan.md](./plan.md)
- Depends on: [Phase 01](./phase-01-architecture-contracts-baseline.md), [Phase 04](./phase-04-web-bridge-dual-source-transport.md)
- Parser: `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/command_parser/src/main.rs`
- Orchestra bridge: `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/zenoh_bridge/src/main.rs`

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-02 |
| Description | Preserve transcript target through parser outputs and reject unsafe selected-rover fallback. |
| Priority | P1 |
| Implementation status | Pending |
| Review status | Pending |
| Effort | 6h |

## Key Insights

- Parser currently discards transcription origin and emits commands with empty Dora parameters.
- Orchestra bridge routes every parser command to the current UI selection.
- Different command types have inconsistent JSON wrappers, so Dora metadata is the least invasive common target envelope.
- Parser TTS feedback would be heard by the rover microphone and is unnecessary because transcripts are visible.

## Requirements

- Both browser and rover transcripts continue through deterministic parsing.
- Every parser-derived actuator output carries required `target_entity_id` Dora metadata.
- Parser channels never fall back to selected rover.
- Web/manual command channels retain selected-rover behavior.
- Remove all automatic parser TTS output while preserving manual Web UI TTS.

## Architecture

```text
SpeechTranscription.target_entity_id
  -> command parser validates/preserves target
  -> Dora output parameter target_entity_id
  -> Orchestra bridge validates active subscription
  -> rover/{target}/cmd/*
```

## Related Code Files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/command_parser/src/main.rs`: target propagation and TTS removal.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/zenoh_bridge/src/main.rs`: parser target routing.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/orchestra-dataflow.yml`: remove parser TTS edge/output.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/speech_types.rs`: only if Phase 01 review changes validation helpers.

## Implementation Steps

1. Deserialize the enriched transcription and reject empty target IDs before parsing.
2. Keep the parser's intent-confidence threshold; remove STT-confidence logging/gating when confidence is absent.
3. Build one Dora parameter map containing validated `target_entity_id`, `utterance_id`, `source_kind`, and optional `entity_id` for observability.
4. Attach this map to every parser-derived rover, tracking, camera, feedback, and future command output.
5. Refactor duplicate send calls through a small output helper so no command type omits target metadata.
6. Remove `create_voice_feedback`, parser `tts_command` output, related `DataId`, dataflow declaration, and Orchestra bridge `tts_command_parser` input/match arm.
7. Keep `tts_command_web` and manual TTS conversion unchanged.
8. Update Orchestra bridge input match to inspect Dora parameters for every `_parser` channel.
9. Require target metadata for parser channels; reject and metric-count missing/invalid targets.
10. Confirm target exists in current `active_rovers` before topic construction. Never substitute `selected_entity`.
11. Keep `_web` command channels routed to selected entity to preserve manual UI behavior.
12. Add parser tests for target preservation across each supported intent/output type.
13. Add bridge tests proving rover A speech cannot route to B, browser target survives a selection change, inactive target rejects, and web commands still follow current selection.

## Todo List

- [ ] Remove STT confidence assumptions from parser.
- [ ] Add common target parameter helper.
- [ ] Attach target to all parser outputs.
- [ ] Remove parser TTS feedback.
- [ ] Enforce target in Orchestra bridge.
- [ ] Preserve manual selected-rover routing.
- [ ] Add parser and bridge routing tests.

## Success Criteria

- Rover A transcript always publishes commands to `rover/A/cmd/*`.
- Browser utterance targets the rover captured at stream start.
- Parser command without target produces no Zenoh publication.
- Inactive target produces warning/metric, not fallback.
- Manual UI commands still use current fleet selection.
- No automatic parser TTS command is produced.

## Risk Assessment

- Risk: One command output misses target metadata. Mitigation: central send helper plus table-driven tests for all parser outputs.
- Risk: Removing TTS affects expected audible feedback. Mitigation: transcript and textual command feedback remain; decision explicitly approved.

## Security Considerations

- Treat target as trusted only because central/web bridge constructs it; still validate nonempty and active at final bridge boundary.
- Never accept a parser JSON payload field as a substitute for Dora target metadata.
- Log target/source IDs but not transcript audio.

## Next Steps

Proceed to Phase 06 after routing tests pass for browser, rover, inactive-target, and manual-command scenarios.
