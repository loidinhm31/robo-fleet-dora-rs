# Phase 02 — Walkie Ingress and Transport

## Context Links

- [Parent plan](./plan.md)
- [Phase 01 contract](./phase-01-contract-and-architecture.md)
- UI capture: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/VoiceControls.tsx`
- Backend ingress: `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/main.rs`

## Overview

- Date: 2026-07-06
- Priority: P1
- Description: Produce honest 20 ms browser PCM frames and preserve their identity through bounded transport.
- Implementation status: Complete
- Review status: Complete
- Approval note: Approved after review and verification on 2026-07-06.

## Key Insights

- `AudioContext({ sampleRate: 16000 })` is a request; `audioContext.sampleRate` is authoritative.
- The current unbounded `Vec` and `remove(0)` add latency and copying.
- Stable stream identity and frame IDs allow every bridge to detect gaps rather than infer continuity.
- Live audio should discard the oldest queued frame when its latency budget is exceeded.

## Requirements

- Emit 20 ms mono Float32 frames with one UUID per walkie session and frame IDs starting at zero.
- Preserve current resource teardown behavior.
- Bound web ingress to 40 ms and Dora media inputs to four frames/80 ms.
- Count invalid frames, duplicates, gaps, overflow drops, forwards, and queue high-water duration.
- Keep remote and direct dataflows behaviorally identical.

## Architecture

```text
AudioWorklet (actual rate, 20 ms)
  -> Socket.IO metadata + F32LE attachment
  -> authenticated web ingress validator
  -> duration-bounded VecDeque
  -> Dora Float32Array + original metadata
  -> orchestra PcmFramePacket / direct playback
  -> rover sequence validator and resampler
```

The UI requests 16 kHz mono for efficiency but never relabels the actual output. Each server-side stream tracks stable rate/channels/format and strictly increasing frame IDs. Walkie overflow drops oldest frames and increments both frame and sample counters.

## Related code files

- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/use-walkie-capture.ts` — capture/worklet/session ownership.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/VoiceControls.tsx` — consume the hook.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/walkie-audio.rs` — ingress parser, stream validator, Dora metadata.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/main.rs` — `Bin` extraction and bounded queue processor.
- Modify the remote, direct, and orchestra dataflow YAML files — explicit queue sizes.

## Implementation Steps

1. Extract walkie capture from `VoiceControls`; retain the existing mode exclusion, pending-start cancellation, analyser level, and exact cleanup semantics.
2. Calculate `frameSize = round(audioContext.sampleRate * 0.020)` after context creation.
3. Generate `crypto.randomUUID()` at successful session start; emit frame ID, current capture timestamp, actual rate, mono count, and exactly one transferred `Float32Array` buffer.
4. Change the Rust listener to `Data<WalkieAudioFrameMetadata>, Bin`; reject legacy JSON arrays, attachment counts other than one, or byte-length mismatch.
5. Decode little-endian floats, normalize no values, and reject the entire frame if any sample is non-finite.
6. Move stream validation and queue accounting out of `main.rs` into `walkie-audio.rs`; key validation by socket/session and expire inactive state after the existing 250 ms authority window.
7. Replace `Arc<Mutex<Vec<_>>>` with `VecDeque<QueuedWalkieFrame>` tracking queued milliseconds. Drop oldest until the total is at most 40 ms.
8. Forward original stream metadata rather than assigning server UUIDs, timestamps, frame IDs, or sample rates.
9. Set media `queue_size: 4`, state/control `queue_size: 8`, and tick `queue_size: 1` on every relevant edge in all three dataflows.
10. Add structured metrics every five seconds and at shutdown; include queue duration and high-water duration, not only element count.

## Todo list

- [x] Walkie hook extracted
- [x] 20 ms actual-rate frames emitted
- [x] Binary Socket.IO ingress implemented
- [x] Legacy frames rejected
- [x] Bounded `VecDeque` implemented
- [x] Metadata preserved through both deployment paths
- [x] Explicit Dora queue policies applied
- [x] UI and Rust tests passing

## Verification Status

- Rust transport suites: 94 tests passed across `web_bridge` and both Zenoh bridges.
- UI suite: 91 tests passed; type checks, lint, and production builds passed.
- Playback/resampling suite: 23 tests passed, including 16, 44.1, and 48 kHz duration coverage.
- All three Dora dataflows parsed with explicit audio media, lifecycle/control, and tick queue policies.

## Success Criteria

- 16, 44.1, and 48 kHz fixtures retain correct duration after rover resampling.
- No bridge invents or overwrites stream identity, frame ID, timestamp, rate, channels, count, or format.
- Web queue cannot exceed 40 ms; Dora walkie backlog cannot exceed 80 ms.
- Under overload, only oldest walkie media is dropped and every dropped frame/sample is observable.

## Risk Assessment

- Browser implementations may ignore requested 16 kHz. Mitigation: actual-rate contract and rover resampling.
- Coordinated cutover can temporarily stop audio if only one repository is deployed. Mitigation: build and stage both artifacts before restart.

## Security Considerations

- Keep maximum payload at 64 KiB and validate dimensions before allocation/copy.
- Maintain per-socket authentication and rate-limit checks.
- Do not accept client-supplied target rover IDs.

## Next steps

Run contract and transport tests, then implement source pacing in Phase 03.
