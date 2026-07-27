# Phase 01 — Contract and Architecture

## Context Links

- [Parent plan](./plan.md)
- [Root-cause report](../reports/brainstorm-260706-0107-audio-playback-tts-fix.md)
- Backend architecture: `/mnt/data/ws/sharing/robo-fleet-dora-rs/ARCHITECTURE.md`
- UI architecture: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/docs/architecture.md`

## Overview

- Date: 2026-07-06
- Priority: P1
- Description: Freeze the cross-repository media, queue, lifecycle, and latency contracts before runtime changes.
- Implementation status: Complete
- Review status: Complete
- Approval note: Approved after review cycle fixes on 2026-07-06.

## Key Insights

- The UI requests 16 kHz but the current wire payload does not declare the actual `AudioContext.sampleRate`.
- Dora input queues default to one and drop the oldest event; this is unsafe for PCM and lifecycle edges.
- Existing `AudioFrameMetadata` and `PcmFramePacket` already define the authoritative transport dimensions.
- `PlaybackState` needs ordering because queue enlargement alone cannot reject stale state.

## Requirements

- Define one versioned walkie contract shared by TypeScript and Rust.
- Standardize browser and TTS media frames at 20 ms.
- Document queue budgets, pacing ownership, completion ordering, preemption, and suppression invariants.
- Preserve current accepted TTS text/configuration limits.

## Architecture

Socket.IO `audio_stream` uses two arguments:

```text
metadata = {
  protocol_version: 1,
  stream_id: UUID,
  frame_id: non-negative safe integer,
  capture_timestamp_ms: non-negative safe integer,
  sample_rate: 8000..192000,
  channels: 1,
  sample_count: samples across all channels,
  format: "f32le"
}
payload = exactly one binary attachment of sample_count * 4 bytes
```

The server converts this envelope to Dora metadata plus `Float32Array`; orchestra wraps the same metadata and PCM bytes in `PcmFramePacket`. Rover decoding restores the identical Dora envelope. `PlaybackState.sequence_id` orders each producer lifetime; consumers ignore duplicate or lower IDs.

## Related code files

- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/ARCHITECTURE.md` — paced media flow and invariants.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/tts_types/lifecycle.rs` — add `sequence_id`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/tts_types/lifecycle_validation.rs` — validate sequence field with existing state variants.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/docs/architecture.md` — binary walkie contract and capture flow.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/socket.ts` and `voice-tts.ts` — mirrored public types.

## Implementation Steps

1. Update both architecture documents first with the final data flow and fixed budgets.
2. Define `WalkieAudioFrameMetadata` in the UI shared package; type `audio_stream` as `(metadata, Float32Array | ArrayBuffer) => void`.
3. Add required `sequence_id: u64` to Rust `PlaybackState` and required `sequence_id: number` to its TypeScript mirror.
4. Update Rust and TypeScript golden fixtures for idle, TTS-active, walkie-active, and unavailable states.
5. Document hard rejection of legacy `{ audio_data }`; do not add an indefinite compatibility branch.
6. Record environment defaults: web queue 40 ms, walkie playback 80 ms, TTS playback 1,000 ms, TTS stall 60 ms, metrics interval 5,000 ms.

## Todo list

- [x] Backend architecture updated
- [x] UI architecture updated
- [x] Socket.IO metadata type added
- [x] Playback state sequence added and mirrored
- [x] Golden contract tests updated
- [x] Contract review completed before Phase 02

## Documentation Status

- Phase 01 docs updated in `ARCHITECTURE.md` and `robo-control-app/docs/architecture.md`.
- Summary: walkie audio contract, pacing, queue budgets, lifecycle ordering, and suppression rules are now documented consistently across both repos.
- No extra plan/report file added; status captured here to keep scope minimal.

## Success Criteria

- Both repositories describe the same field names, units, frame duration, payload encoding, and lifecycle ordering.
- Contract tests reject wrong versions, missing/extra attachments, malformed UUIDs, non-mono data, invalid dimensions, and payload-length mismatches.
- No architecture text claims TTS is burst-produced or that walkie is always 16 kHz.

## Risk Assessment

- Required state fields break mixed-version rover nodes. Mitigation: coordinated image deployment and full-stack restart.
- JavaScript numbers cannot represent arbitrary `u64`. Restrict browser-origin IDs/timestamps to safe integers; Rust still uses `u64` internally.

## Security Considerations

- Browser cannot provide rover routing authority; backend continues using authenticated fleet selection.
- Reject non-finite PCM before Dora publication.
- Retain authentication and command rate limits before audio admission.

## Next steps

Proceed to Phase 02 only after contract/golden tests agree across both repositories.
