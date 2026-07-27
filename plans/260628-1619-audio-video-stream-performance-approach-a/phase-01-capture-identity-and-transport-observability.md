# Phase 01 — Capture Identity and Transport Observability

## Context links

- [Parent plan](./plan.md)
- [Research](./research/researcher-01-report.md)
- [Architecture](../../ARCHITECTURE.md#binary-browser-audio-and-bounded-playback)
- [Source reassessment](../reports/brainstorm-260628-1244-audio-video-stream-performance-reassessment.md)

## Overview

- Date: 2026-06-28
- Description: preserve capture identity and expose loss/error metrics without changing browser playback.
- Priority: P1
- Implementation status: Done (2026-06-29)
- Review status: Done; approved at 8.8/10 (2026-06-29)
- Effort: 7h

## Key Insights

- Web-generated frame IDs/timestamps cannot locate upstream loss or measure end-to-end age.
- Raw Zenoh audio drops all Dora metadata and uses unsafe native-endian pointer casts.
- Process restart needs `stream_id`; frame sequence alone cannot distinguish restart from regression.
- Existing video packet/metric patterns are directly reusable.

## Requirements

- Capture creates one `stream_id` per process plus monotonic `frame_id` per emitted chunk.
- `capture_timestamp_ms` means UTC chunk-ready time at the capture node.
- Versioned PCM envelope carries identity, sample format, rate, channels, sample count, payload length.
- Decoder rejects unknown version/format, overflow, inconsistent lengths, and oversized frames.
- Orchestra temporarily accepts legacy raw F32 payloads for rover-first rollback safety.
- Direct mode retains Dora metadata without using the Zenoh envelope.
- Every ignored result becomes success/error/drop accounting; fatal Dora errors still fail the node.

## Architecture

```text
capture metadata -> audio_converter F32→S16LE on rover -> v1 S16LE Zenoh packet
  -> restored Dora metadata -> web_bridge -> browser
```

- Use explicit `to_le_bytes`/`from_le_bytes`; remove raw byte-to-f32 unsafe casts.
- Track sequences per entity and stream. A new stream resets expectation without counting regression.
- Reuse five-second `MetricWindow` logs with `metric="audio_pipeline"` and stable stage names.

## Related code files

- Create — `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/audio_types.rs`: PCM metadata, packet codec, validation.
- Modify — `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/mod.rs`: export audio module.
- Modify — `/mnt/data/ws/sharing/robo-fleet-dora-rs/robo_rover_lib/src/types/video_types.rs`: move existing audio-only types; keep public re-exports stable.
- Modify — `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_capture/src/main.rs`: identity, overflow count, metadata.
- Modify — `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/zenoh_bridge/src/main.rs`: safe v1 encode of Int16LE output from audio_converter, publish metrics/errors.
- Modify — `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/zenoh_bridge/src/main.rs`: v1/legacy decode of Int16LE payload, restore Dora metadata, per-rover metrics.
- Modify — `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_converter/src/main.rs`: runs on rover, converts F32→S16LE before Zenoh transport, strict metadata validation and conversion metrics.
- Rename — `orchestra/speech_recognizer` → `orchestra/central_speech_recognizer`: package name becomes `central_speech_recognizer`, uses whisper base model. Only processes web UI audio (`web-bridge/voice_command_audio`, Float32). Remove rover audio input (`orchestra-bridge/audio_frame`).
- Create — `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/edge_speech_recognizer/src/main.rs`: TODO placeholder crate. Will receive Float32 directly from `audio_capture`.
- Modify — `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/main.rs`: preserve origin identity in current JSON event and count emit results.

## Implementation Steps

1. Add `PcmSampleFormat`, `AudioFrameMetadata`, and borrowed `PcmFramePacket` types.
2. Define fixed magic/version and bounded header; encode/decode without serde JSON.
3. Unit-test round trip, truncated header, unknown version, invalid dimensions, payload mismatch, and maximum size.
4. Move current unused audio structs from `video_types.rs` into the focused module without changing public names.
5. In capture, make `write_audio_data` return rejected sample count; aggregate once per callback.
6. Stamp each complete chunk before Dora send. Increment frame ID even if send fails so downstream observes a gap.
7. In rover audio_converter, receive F32 from capture, convert to S16LE, preserve identity, derive output byte size, and record conversion duration/errors.
8. In rover bridge, read required metadata, encode S16LE samples explicitly little-endian, publish v1 packet, and record result.
9. In orchestra bridge, decode S16LE packet safely, restore Dora metadata plus `entity_id`, track gaps per `(entity_id, stream_id)`, and retain bounded legacy decode.
10. Add `audio-converter` node to `rover-kiwi-dataflow.yml` between `audio-capture` and `zenoh-bridge`.
11. Remove `audio-converter` node from `orchestra-dataflow.yml`; Int16LE audio goes directly from orchestra-bridge to web-bridge.
12. Create `edge_speech_recognizer` TODO placeholder crate at `rover-kiwi/edge_speech_recognizer/`.
13. Rename `speech_recognizer` to `central_speech_recognizer` at `orchestra/central_speech_recognizer/`; update package name.
14. Update `central_speech_recognizer`: remove rover audio input (`audio_rover`), keep only web UI audio input (`audio_web`); switch to whisper base model.
15. Add commented-out `edge-speech-recognizer` node to both rover dataflow YAMLs.
16. In web bridge, use origin identity in existing JSON shape. Count a frame sent only after successful Socket.IO emit.
17. Add structured shutdown totals and five-second windows for all stages.

## Todo list

- [x] Add and test shared PCM packet contract.
- [x] Add capture identity and ring-drop accounting.
- [x] Add rover-side audio_converter (F32→S16LE) and update rover-kiwi-dataflow.yml.
- [x] Remove audio-converter node from orchestra-dataflow.yml.
- [x] Upgrade rover publisher (S16LE) and orchestra decoder (S16LE with Dora metadata restore).
- [x] Create edge_speech_recognizer TODO placeholder crate.
- [x] Rename speech_recognizer → central_speech_recognizer; update package name and whisper model to base.
- [x] Update central_speech_recognizer: remove rover audio input, keep only web UI audio input (Float32).
- [x] Add commented-out edge-speech-recognizer to both rover dataflow YAMLs.
- [x] Preserve identity through current web event.
- [x] Add backend sequence/error/age metrics.
- [x] Run focused Rust tests and formatting.

## Completion Evidence

- Completed: 2026-06-29
- Tests: 51/51 passed.
- Rust formatting checks: 14/14 passed.
- YAML checks: 4/4 passed.
- Unsafe audio-path casts: 0.
- Review: 8.8/10; user approved.
- Runtime hardware validation not run; scheduled for Phase 02 evidence gate.

## Deferred Backlog

- Web shutdown totals sum per-client counters only for connected clients. Disconnect cleanup loses historical audio delivery/drop counts. Add process-level cumulative counters and a disconnect-retention test in future work.

## Success Criteria

- One origin identity appears unchanged in logs and browser event across all stages.
- No unsafe audio byte casts remain on the rover-to-orchestra path.
- Malformed packets cannot panic or allocate from untrusted lengths.
- Legacy raw F32 rover payload remains accepted during migration.
- Emit/publish failures are visible and not counted as successful sends.
- `cargo test -p robo_rover_lib -p audio_capture -p audio_converter -p rover_zenoh_bridge -p orchestra_zenoh_bridge -p web_bridge -p central_speech_recognizer` passes.

## Risk Assessment

- Packet cutover can break mixed rover/orchestra versions. Mitigate with orchestra legacy decoder and orchestra-first deploy.
- UTC clock skew distorts age metrics. Record offset; do not use negative/future age as valid latency.
- Moving public audio types could break unknown consumers. Preserve root re-exports and compile whole workspace.

## Security Considerations

- Treat Zenoh payload/header as untrusted input.
- Bound sample rate, channels, sample count, duration, and total bytes before allocation.
- Avoid logging PCM contents or secrets.

## Next steps

- Proceed to Phase 02 only after capture identity is observable end-to-end.
- Do not change scheduler or binary browser event in this phase.

## Unresolved Questions

- None.
