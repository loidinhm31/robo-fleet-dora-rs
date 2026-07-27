# Audio/Video Stream Performance Reassessment

Date: 2026-06-28  
Scope: diagnosis and feasibility only; no implementation plan  
Repositories inspected:

- Backend: `/mnt/data/ws/sharing/robo-fleet-dora-rs`
- Frontend: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`

## Executive Verdict

The supplied report points in the right direction but overstates unmeasured hypotheses as root causes.

Binary audio plus Web Audio timeline scheduling is feasible and remains highest-ROI. However, the strongest code-level defect is more specific than generic main-thread timer delay: the client consumes each 50 ms audio buffer on a 40 ms timer. Under prompt timer execution, playback dequeues at 25 buffers/s while audio arrives at 20 buffers/s. A five-buffer reserve can therefore drain in about one second. This directly fits the observed on/off cadence.

The proposed replacement scheduler is directionally correct but incomplete. Scheduling on each frame arrival removes the recursive timer dependency, yet it still needs a bounded scheduled horizon, late-frame handling, sequence/loss detection, and latency control. It is resilient to main-thread stalls only while enough audio has already been scheduled; it is not immune to main-thread or transport delay.

Binary transfer is already proven in this codebase by the video path using the same `socketioxide 0.12` API. Focused protocol test passed. No infrastructure or new port required for this change.

## Current Data Path

```text
Rover microphone
  16 kHz mono Float32, 800 samples/chunk, 50 ms/chunk
  -> Dora audio_capture
  -> Zenoh audio/raw (3,200 bytes/chunk)
  -> Orchestra Zenoh bridge
  -> audio_converter S16LE (1,600 bytes/chunk)
  -> web_bridge JSON number array (currently)
  -> Socket.IO / Engine.IO connection shared with binary JPEG events
  -> CameraViewer PCM conversion
  -> AudioBuffer queue
  -> recursive setTimeout scheduler
  -> Web Audio output
```

Normal split deployment places `audio_converter` and `web_bridge` on Orchestra, not Rover. Therefore browser-facing JSON elimination reduces Orchestra/browser CPU and last-hop bandwidth. It does not reduce Rover CPU, and it does not reduce rover-to-orchestra audio bytes because that leg remains Float32.

## Evidence

### Confirmed configuration

- Audio source: 16,000 Hz, mono, 800 samples per chunk.
- One chunk represents exactly 50 ms.
- S16LE browser payload: 1,600 bytes, 20 chunks/s, 256 kbit/s raw PCM.
- Video: 640x480 JPEG, nominal 15 FPS, quality 80.
- `web_bridge` uses `socketioxide = 0.12`; lockfile resolves 0.12.0.
- Browser uses `socket.io-client 4.8.1`, WebSocket preferred with polling fallback.
- Video already uses `.bin(vec![jpeg_data]).emit(...)`; browser receives metadata plus binary second argument.

### Representative JSON size

Serializing a representative 1,600-byte `Vec<u8>` frame with current metadata produced:

| Form | Bytes/frame | Payload rate at 20 Hz |
|---|---:|---:|
| Raw/binary PCM | 1,600 | 256 kbit/s |
| JSON number array event | 5,766 | 923 kbit/s |

Expansion measured about 3.6x for representative byte values. Exact size varies with sample bytes. The supplied report's 4x estimate is reasonable as an approximation, but `Vec<u8>` serializes values from 0 through 255; negative examples such as `-45` are inaccurate.

### Binary API feasibility

Existing backend code and current `socketioxide 0.12.0` source both support:

```rust
socket.bin(vec![bytes]).emit("event", metadata)
```

Existing test `tests::video_frame_packet_uses_binary_attachment_not_json_byte_array` passed on 2026-06-28. Audio can use the same event shape. Frontend binary normalization patterns already exist for video.

This proves API and protocol feasibility. It does not prove audio continuity.

### Runtime evidence limitations

Available June 27 direct-run logs show valid 1,600-byte audio frames and active video emission. The same run also contains repeated explicit microphone `Stop`/`Start` commands, including several approximately one second apart. Current code emits those commands only from UI controls, so this run is confounded by operator/control activity and cannot certify steady-state audio cadence.

No captured browser console timeline, Socket.IO packet trace, audio frame inter-arrival histogram, long-task trace, or end-to-end audio timestamp exists. Thus TCP HoL, browser saturation, and server-side queue overflow remain plausible but unproven.

## Root-Cause Reassessment

### 1. Client scheduler rate mismatch — high confidence, directly evidenced

`CameraViewer.tsx` calculates:

```text
audioBuffer.duration = 800 / 16000 = 50 ms
schedulingTime = 50 ms - 10 ms = 40 ms
```

Each timer invocation removes one 50 ms buffer. Idealized rates:

```text
producer = 20 buffers/s
consumer = 25 buffers/s
net queue loss = 5 buffers/s
initial reserve = 5 buffers
time to empty ~= 1 s
```

The scheduler also schedules audio increasingly far ahead because it advances the Web Audio timeline by 50 ms every 40 ms of wall time.

Its drift correction is ineffective:

1. `playTime` captures the old `nextPlayTimeRef.current`.
2. A large gap sets `nextPlayTimeRef.current = currentTime + 0.05`.
3. `source.start(playTime)` still uses the old value.
4. `nextPlayTimeRef.current = playTime + duration` overwrites the correction.

When the queue becomes empty, already-scheduled sources are not tracked or cancelled. Restart can therefore create a gap or overlap, depending on scheduled horizon and arrival timing.

This defect exists without video. Audio-only can still sound acceptable if timer delay, batching, or machine headroom accidentally keeps effective dequeue cadence near 50 ms. Video changes timing enough to expose the unstable scheduler.

### 2. Main-thread timing vulnerability — high confidence mechanism, contribution unmeasured

Only one buffer is scheduled per timer callback. If the main thread cannot run before the scheduled horizon ends, Web Audio has nothing queued next and produces a gap even when buffers remain in the JavaScript queue.

Video adds main-thread work:

- Socket.IO event dispatch and metadata handling.
- Blob and object URL creation per JPEG.
- `Image` load callbacks and canvas drawing.
- Detection overlays and statistics.
- React state updates on frame receipt and render.

Audio adds avoidable work:

- JSON parsing into a 1,600-element JavaScript array.
- Allocation/copy into `Uint8Array`.
- Per-sample PCM conversion.
- React statistics updates for every frame.
- Potentially two detailed console logs per audio frame due to a stale `stats` closure; the effect does not depend on `stats`, so the `< 5` condition can remain true for the effect lifetime. This is especially expensive with DevTools open.

The Web Audio API can schedule a source at an `AudioContext.currentTime` timestamp, so pre-scheduled buffers play on its timeline without another JavaScript timer callback. Official reference: [MDN `AudioScheduledSourceNode.start()`](https://developer.mozilla.org/en-US/docs/Web/API/AudioScheduledSourceNode/start).

### 3. JSON payload bloat — confirmed inefficiency, causal weight unmeasured

Binary audio removes roughly 3.6x last-hop payload expansion in the representative measurement and avoids constructing/parsing a large JSON number array. It is worthwhile regardless of whether it is the primary stutter trigger.

Claims of a measurable CPU reduction need profiling. No current profile isolates Rust serialization, browser parsing, PCM conversion, React rendering, JPEG decode, or canvas time.

### 4. Socket.IO/TCP HoL — plausible, not demonstrated

Audio and video events are ordered on one Engine.IO transport. Socket.IO guarantees message ordering using its underlying transport, including TCP for WebSocket; a delayed video payload can therefore delay later audio on the same connection. Official reference: [Socket.IO delivery guarantees](https://socket.io/docs/v4/delivery-guarantees/).

Important qualifications:

- Video is already binary, so JPEG JSON parsing is not current behavior.
- `web_bridge` to a local workstation browser may have ample bandwidth; HoL severity depends on actual deployment path, packet loss, proxy/tunnel use, and socket buffering.
- `socketioxide` has a bounded per-connection internal packet queue. Audio emit errors are discarded, then `mark_audio_sent()` still runs. Queue-full audio loss would be invisible in current metrics.
- Server-to-client Socket.IO delivery is at-most-once by default. A dropped server event is not retried.
- A separate browser connection only isolates the Orchestra-to-browser leg. Rover audio and video also share the Zenoh TCP path with no explicit media-specific QoS in current publisher declarations. Browser socket separation cannot eliminate that upstream contention.

### 5. Silent drop points — confirmed observability gap

Audio errors are ignored at several boundaries:

- Audio capture ignores ring-buffer `try_push` failure. Its comment says "drop oldest," but failed `try_push` drops the new sample.
- Rover Zenoh bridge ignores `audio_pub.put(...)` result.
- Orchestra Zenoh bridge ignores Dora `send_output(...)` result.
- Web bridge ignores Socket.IO audio `emit(...)` result and still counts the frame as sent.

Without sequence identity originating at capture and counters at each boundary, an underrun cannot be assigned to capture, Dora, Zenoh, Socket.IO, or browser playback.

### 6. Control-plane interruptions — observed, cause unresolved

One available runtime log contains repeated explicit audio stop/start commands. This could be expected operator testing, accidental control interaction, or a separate UI behavior. It is not evidence of automatic toggling because current code paths are click handlers. Reproduction must hold capture continuously active before evaluating media transport.

## Assessment of Proposed Approaches

### Approach A: binary audio plus bounded Web Audio timeline scheduler

Verdict: recommended, with correction.

Pros:

- Binary transport already proven by current video implementation.
- No new server, port, firewall rule, codec, or signaling protocol.
- Removes confirmed payload/allocation waste.
- Removes deterministic 40 ms/50 ms scheduler mismatch.
- Pre-scheduling provides main-thread stall tolerance equal to the scheduled horizon.

Required design properties:

- Accept binary attachment as primary payload; temporary JSON fallback only if rollout requires mixed versions.
- Preserve capture-origin sequence/timestamp, not a new ID created only at `web_bridge`.
- Schedule against `AudioContext.currentTime` without a per-buffer recursive timer.
- Maintain a bounded target horizon, e.g. enough to absorb expected jitter while meeting latency SLA.
- Detect duplicates, gaps, late frames, queue overflow, and context suspend/resume.
- Cap scheduled-ahead audio; do not let burst delivery create unbounded latency.
- Reset timeline carefully after a real underrun; avoid overlap with already-scheduled sources.
- Reduce per-frame logging and UI state churn.

The supplied sample's unconditional reset when `nextPlayTime < currentTime + 20 ms` is insufficient by itself. It can create discontinuities, and it has no maximum-horizon or sequence policy.

### Approach B: separate physical Socket.IO connections

Verdict: conditional fallback, not first action.

Pros:

- Isolates browser-leg TCP/socket queueing between video and audio.
- Useful if measured audio inter-arrival spikes correlate with video payload transmission or Socket.IO queue depth.

Qualifications and costs:

- A new TCP connection does not inherently require a second port. Socket.IO clients can use separate Manager/Engine.IO connections on the same origin, but server routing/subscription must ensure audio-only and video-only sockets receive only their intended media.
- Merely using another namespace may still multiplex over one Engine.IO connection unless a separate Manager is forced.
- Current backend defaults every connected client to `audio_enabled = true` and has no effective per-client audio demand update, so isolation requires server subscription changes.
- Does not isolate the shared rover-to-orchestra Zenoh TCP leg.
- Duplicates authentication, reconnect, heartbeat, lifecycle, and metrics concerns.

Do not add this complexity without transport evidence after Approach A behavior is measured.

### Approach C: AudioWorklet jitter buffer

Verdict: viable second-line browser architecture, currently YAGNI.

An `AudioWorklet` can own continuous sample consumption on the audio rendering thread and is better suited to network-fed PCM when strict timing is required. It also introduces buffer protocol, worker lifecycle, browser compatibility, and possibly cross-origin-isolation constraints if `SharedArrayBuffer` is used.

Use only if bounded `AudioBufferSourceNode` timeline scheduling still shows glitches caused by main-thread stalls.

### Approach D: WebRTC

Verdict: defer.

WebRTC gives codec negotiation, jitter buffering, congestion control, and low-latency media transport. Cost remains high: signaling, ICE, STUN/TURN, Rust integration, browser lifecycle, observability, and codec handling. UDP is preferred but not universally guaranteed; TURN can relay over TCP/TLS.

This is justified only if product requirements expand to internet-grade, adaptive, multi-client real-time media and simpler fixes fail measured acceptance criteria.

## Recommended Solution Boundary

Endorse a revised Approach A, not the exact sample implementation:

```text
Socket.IO binary S16LE
  + capture-origin sequence/timestamp
  + bounded browser jitter/scheduled horizon
  + Web Audio timeline scheduling without recursive setTimeout
  + explicit drop/late/underrun metrics
  + reduced per-frame logging and React updates
```

Do not endorse a second socket or WebRTC yet. Current evidence does not prove last-hop TCP HoL is the limiting factor.

## Risks and Compatibility

- Mixed frontend/backend versions: binary-first event shape needs coordinated types or a temporary fallback.
- Browser transport fallback: polling behavior and payload limits differ from WebSocket; runtime should record selected Engine.IO transport.
- Latency versus resilience: current five-frame reserve plus 100 ms delayed start already implies about 350 ms startup latency, incompatible with a strict under-150 ms target.
- Scheduled source cleanup: one-shot `AudioBufferSourceNode`s need lifecycle handling across disable, reconnect, and AudioContext suspension.
- Format assumptions: client currently decodes every frame as S16LE regardless of `format`; invalid format/channels/sample-rate must be rejected or supported explicitly.
- Timestamp contract: current `web_bridge` timestamp is not microphone capture time, so true end-to-end latency cannot currently be measured.
- Frontend repository boundary: backend documentation claims `robo-control-app/` is in this repository, but active UI is a separate adjacent Git repository. Coordinated changes and validation must account for both.

## Success Metrics and Validation Criteria

### Continuity

- Zero playback underruns after initial warm-up during a 10-minute audio+video run.
- Zero unintended microphone stop/start commands.
- Zero duplicate or regressed audio sequence IDs.
- Explicit counts for lost frames at every transport boundary.

### Latency

- Capture-to-audible p95 at or below 150 ms, measured from a capture-origin timestamp.
- Scheduled horizon bounded below the latency SLA.
- No sustained queue growth; no burst-induced latency accumulation.

### Timing

- Source cadence: 20 frames/s while capture active.
- Browser inter-arrival p95 and max recorded separately for audio-only and audio+video.
- Main-thread long tasks and scheduled-horizon exhaustion correlated by timestamp.

### Resource use

- Compare actual Socket.IO audio wire bytes before/after; expected payload reduction roughly 3.6x for representative frames.
- Compare Orchestra `web_bridge` CPU and browser scripting/GC time, not only total system CPU.
- Rover CPU is not expected to improve from browser-leg binary serialization in normal split mode.

### Feasibility gate result

| Item | Result |
|---|---|
| `socketioxide 0.12` binary attachment API | Feasible; already used by video |
| Browser receives metadata plus binary argument | Feasible; existing video handler proves shape |
| Web Audio timestamp scheduling | Supported by established browser API |
| Zero new port for Approach A | Confirmed |
| Under-150 ms with current five-frame + 100 ms startup | Not feasible |
| Claim TCP HoL as confirmed root cause | Not supported by current evidence |
| Claim binary transfer improves Rover CPU in split deployment | Incorrect |

## Evidence Required Before Any Implementation Plan

This is a diagnostic gate, not a plan:

- One controlled reproduction with no audio control interaction.
- Browser timestamps for receive, convert, schedule, actual horizon, underrun, and long tasks.
- Capture-origin audio sequence/timestamp propagated through all hops.
- Socket.IO emit error counters and selected transport.
- Audio-only versus audio+video inter-arrival and loss comparison.
- Network path recorded: localhost, LAN, Tailscale, reverse proxy, or tunnel.

## Unresolved Questions

- Is the failing browser local to Orchestra, on LAN, over Tailscale, or behind a tunnel/proxy?
- Were the repeated June 27 audio stop/start commands intentional operator actions?
- Does stutter reproduce with DevTools closed, where persistent per-frame console logging costs less?
- Do audio frame sequence gaps occur before `web_bridge`, at Socket.IO emit, or only in browser receipt?
- Is under 150 ms a hard capture-to-audible SLA or a browser-buffer target?
