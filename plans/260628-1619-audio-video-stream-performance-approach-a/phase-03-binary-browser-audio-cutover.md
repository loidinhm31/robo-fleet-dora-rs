# Phase 03 — Binary Browser Audio Cutover

## Context links

- [Parent plan](./plan.md)
- [Phase 02 evidence gate](./phase-02-controlled-baseline-and-evidence-gate.md)
- [Existing binary video decision](../../ARCHITECTURE.md#phase-3-binary-browser-delivery)

## Overview

- Date: 2026-06-28
- Description: replace JSON PCM arrays with one Socket.IO binary S16LE attachment.
- Priority: P1
- Updated: 2026-06-30
- Implementation status: Done
- Review status: Approved
- Effort: 4h
- Validation: automated validation passed; live playback deferred to Phase 5.

## Key Insights

- `socketioxide 0.12` binary behavior is proven by video and its protocol test.
- New frontend can support old JSON backend, enabling safe frontend-first rollout.
- Backend must not include both JSON bytes and binary bytes; that preserves the waste.
- Send counters currently record attempts, hiding queue-full/disconnect failures.

## Requirements

- Event remains `audio_frame` with metadata first and one binary attachment second.
- Metadata includes protocol version, stream/frame identity, capture timestamp, sample rate, channels, format, sample count, and duration.
- Metadata contains no `data` after backend cutover.
- Accept S16LE only at browser boundary; validate exact `sample_count * channels * 2` bytes.
- Frontend normalizes ArrayBuffer, typed-array views, Blob, and transitional legacy number array.
- Binary emit errors increment metrics; successful count changes only on `Ok`.
- No new port, namespace, Manager, feature flag, or duplicate audio connection.

## Architecture

```text
orchestra-bridge/audio_frame (Int16LE from rover) -> web_bridge: metadata JSON + binary[0]=S16LE
    -> Socket.IO existing connection
    -> audio-frame validator/decoder
```

> **Note:** Audio arrives as Int16LE directly from the rover (converted by `rover-kiwi/audio_converter`). The orchestra-side conversion step is eliminated — `web_bridge` receives Int16LE from `orchestra-bridge/audio_frame` instead of a separate `audio-converter/audio_output` node.

- Rollout: frontend dual decoder -> orchestra/web bridge binary emitter -> rover audio_converter + Int16LE Zenoh publisher.
- Rollback: old backend JSON remains consumable by new frontend; orchestra handles both Float32 (old rover) and Int16LE (new rover) during transition.

## Related code files

- Modify — `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/main.rs`: binary payload helper, strict validation, emit result handling/tests.
- Modify — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/socket.ts`: binary second argument and transitional fallback.
- Modify — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/lib/audio-frame.ts`: binary normalization and S16LE conversion.
- Modify — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/lib/audio-frame.test.ts`: payload variants and rejection cases.
- Modify — `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/CameraViewer.tsx`: handler accepts second argument.
- Create — `/mnt/data/ws/sharing/robo-fleet-dora-rs/plans/260628-1619-audio-video-stream-performance-approach-a/reports/phase-03-binary-smoke.md`: wire shape/size evidence.

## Implementation Steps

1. Add `browser_audio_frame_payload()` returning metadata only and `validate_browser_pcm_payload()` with checked arithmetic.
2. Replace JSON `data` field with `socket.bin(vec![audio_bytes]).emit("audio_frame", metadata)`.
3. Resolve client socket ID safely; count parse/missing-socket/emit errors separately.
4. Call `mark_audio_sent()` and record bytes/duration only after successful emit.
5. Add `Packet::bin_event` test asserting attachment placeholder, exact bytes, origin identity, and absent JSON `data`.
6. Add malformed/odd-length/sample-count mismatch tests.
7. Update shared TypeScript event signature to `(metadata, binaryData?)`; retain explicit legacy metadata union.
8. Normalize binary without unnecessary copies where safe; decode S16LE via `DataView`/typed bytes with endianness explicit.
9. Deploy/test frontend first against JSON backend, then test binary backend.
10. Measure representative wire payload and browser parse/GC time against Phase 02 baseline.

## Todo list

- [x] Add strict backend browser PCM payload contract.
- [x] Switch Socket.IO emission to binary attachment.
- [x] Correct send accounting.
- [x] Update shared event types and decoder.
- [x] Add Rust and TypeScript protocol tests.
- [x] Verify frontend-first compatibility.
- [x] Record binary smoke evidence.

## Success Criteria

- Browser receives metadata plus exactly one 1,600-byte attachment for standard frames.
- Socket.IO event metadata has no PCM byte array.
- Representative audio payload bytes fall by at least 65% versus Phase 02 JSON baseline.
- Audio continues under the old scheduler before Phase 04.
- Invalid payloads are rejected without playback or panic.
- Focused Rust tests, frontend tests, type checks, lint, and builds pass.

## Risk Assessment

- Old frontend cannot consume new backend. Mitigate with enforced frontend-first deployment.
- Socket.IO polling fallback may have different packet limits. Record selected transport and smoke both when supported.
- Blob normalization is async; preserve event ordering by sequencing normalization or use socket-client native ArrayBuffer path.

## Security Considerations

- Validate metadata and payload length before allocation/sample conversion.
- Cap frame bytes and duration; never trust client/runtime-reported array shape.
- Do not log raw PCM.

## Next steps

- Proceed to Phase 04 after binary audio is continuous with the unchanged scheduler.
- Keep transitional JSON frontend fallback through rollout; schedule cleanup separately after rollback window.
- Live playback stays deferred to Phase 5.

## Unresolved Questions

- None; rollout order resolves mixed-version behavior.
