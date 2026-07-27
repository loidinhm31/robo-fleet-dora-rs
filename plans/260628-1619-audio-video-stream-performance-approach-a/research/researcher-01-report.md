# Approach A Research Report

Date: 2026-06-28  
Method: local report/code analysis; no implementation  
Source: [audio/video reassessment](../../reports/brainstorm-260628-1244-audio-video-stream-performance-reassessment.md)

## Verdict

- Implement revised Approach A.
- Highest-confidence defect: browser consumes 50 ms frames every 40 ms. Five-frame reserve drains in ~1 s.
- Binary Socket.IO audio is low risk. Current video path proves `socketioxide 0.12` binary attachments.
- Do not add another connection, AudioWorklet, codec, or WebRTC before Approach A measurements.
- Report's runtime evidence gate is not satisfied. Make controlled baseline a blocking phase before cutover.

## Verified Current Path

```text
audio_capture F32/800 samples
  -> rover Zenoh bridge raw native-endian bytes
  -> orchestra Zenoh bridge F32 Dora array
  -> audio_converter S16LE BinaryArray
  -> web_bridge JSON byte array
  -> CameraViewer queue + recursive setTimeout
```

- Source cadence: 16 kHz mono, 800 samples, 50 ms/frame, 20 frames/s.
- Browser payload: 1,600 raw bytes/frame; representative JSON expands ~3.6x.
- `audio_converter` already clones Dora metadata and changes format/size.
- Capture creates sample rate/channels/format metadata but no origin frame identity.
- Both Zenoh bridges discard audio metadata; web bridge synthesizes frame ID/time.
- Rover publish and orchestra Dora send ignore errors.
- Web bridge ignores Socket.IO emit errors and counts attempted sends as successes.
- Browser currently logs and updates React state per frame, amplifying main-thread load.
- Existing frontend is a separate Git repository and `CameraViewer.tsx` has unrelated dirty video changes.

## Design Conclusions

1. Add versioned PCM packet envelope for the rover-to-orchestra Zenoh leg.
2. Capture assigns `stream_id`, `frame_id`, and `capture_timestamp_ms`; no downstream regeneration.
3. Use explicit little-endian encoding/decoding. Remove unaligned/native-endian unsafe casts.
4. Preserve capture identity through Dora metadata and conversion.
5. New frontend accepts legacy JSON or binary. Deploy it before binary backend.
6. Backend emits metadata plus one S16LE binary attachment; never duplicate bytes in JSON.
7. Browser schedules on frame arrival against `AudioContext.currentTime`.
8. Default scheduler policy: 10 ms minimum lead, 50 ms restart/target lead, 150 ms maximum end horizon.
9. Drop an incoming burst frame if scheduling it would exceed maximum horizon; count the drop.
10. Reset timeline only on real underrun/context transition; cancel tracked sources on disable/unmount.
11. Update UI metrics at most once/second. Detailed structured logs only behind `?audioDebug=1`.

## Measurement Contract

- Backend: capture ring drops, Dora/Zenoh/Socket.IO errors, bytes, frames, sequence gaps, frame age.
- Browser: inter-arrival p95/max, gaps/duplicates, invalid/late frames, underruns, horizon, long tasks.
- Primary test: DevTools closed. Profiling run: DevTools open and debug telemetry enabled.
- Compare 10-minute audio-only and audio+video runs on recorded network path.
- Cross-host capture latency requires synchronized clocks. Record clock offset before acceptance run.
- Browser metric is capture-to-scheduled-start estimate. Hardware capture-to-audible needs loopback.

## Compatibility

- Frontend dual decoder gives rollback to old JSON backend.
- Orchestra decoder temporarily accepts legacy raw F32 Zenoh payloads, then accepts v1 packet after rover upgrade.
- No old-frontend/new-backend compatibility. Enforce frontend-first rollout.
- Direct mode bypasses Zenoh envelope but retains Dora capture metadata and binary browser event.

## Unresolved Questions

- Is 150 ms a hard hardware-audible SLA or a browser scheduling target?
- Which path must pass: localhost, LAN, Tailscale, or proxy/tunnel?
- Were observed audio stop/start commands intentional?
- Can rover and workstation clocks remain within 5 ms during the validation run?
