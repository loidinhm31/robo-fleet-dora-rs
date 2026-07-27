# Audio playback and TTS fix report

## Problem

Regressions after source-aware playback work:

1. Walkie-talkie audio from browser to rover speaker is unclear/choppy. Rover microphone republishes speaker audio, creating echo.
2. Rover TTS plays only an initial fragment, often not a complete word, then ends.

Required outcome:

- live walkie remains low-latency and intelligible;
- rover microphone publishes no speaker loopback during playback plus suppression tail;
- TTS of any accepted length plays through once, in order, before completion;
- walkie preemption of TTS remains intentional and deterministic;
- all drops, overruns, sequence gaps, and terminal reasons are observable.

## Evidence and root-cause assessment

### Confirmed: Dora external-node input queues default to one

All relevant dataflow inputs omit `queue_size`. Dora node API 0.5 builds each external-node scheduler queue with `config.queue_size.unwrap_or(1)` and removes the oldest event when the limit is reached. The daemon has a separate default of 10; it does not remove the node-side size-one scheduler behavior.

Affected boundaries include:

- `rover-kiwi/rover-kiwi-dataflow.yml:69-73`: walkie PCM, TTS PCM, TTS terminal state;
- `rover-kiwi/rover-kiwi-dataflow.yml:9`: playback suppression state;
- `rover-kiwi/rover-kiwi-dataflow.yml:45-46`: walkie authority and playback result;
- matching inputs in `rover-kiwi/rover-kiwi-direct-dataflow.yml`;
- `orchestra/orchestra-dataflow.yml:60`: browser walkie PCM before Zenoh.

This is a data-loss policy, not backpressure. For audio, dropping an arbitrary oldest chunk creates waveform discontinuities. For lifecycle edges, dropping `Active` while retaining `Idle` disables echo suppression.

### Confirmed: TTS is burst-produced without a media clock

`edge_voice` creates 882-sample/20 ms chunks in `rover-kiwi/edge_voice/src/worker.rs:24,228-245`. Its runtime drains all ready worker events and sends all chunks to Dora immediately in `rover-kiwi/edge_voice/src/runtime.rs:405-415,513-527`. Offline synthesis can therefore publish much faster than real-time playback.

The bounded worker channel prevents unlimited worker memory, but the runtime drains it in a burst, so it does not currently provide end-to-end playback backpressure.

### Confirmed: five-second TTS capacity causes destructive truncation

`rover-kiwi/audio_playback/src/runtime.rs:17,33-36` allocates five output-seconds for all TTS. When one resampled chunk does not fit, `rover-kiwi/audio_playback/src/tts-arbiter.rs:178-195`:

1. partially enqueues the chunk;
2. clears all queued TTS;
3. marks the command failed and blocks later chunks.

The existing test `tts_overflow_is_explicit_failure_and_clears_audio` proves this is intended current behavior, but not acceptable playback behavior. Fast synthesis can fill the buffer before CPAL consumes much, leaving only the already-consumed initial fragment audible.

### Confirmed: playback edges can collapse and leak echo

The CPAL callback records `Active -> Idle` consumption transitions. On each 20 ms tick, `audio_playback` drains every transition and publishes each one (`runtime.rs:69-83`; `state.rs:54-97`). Dora's size-one `audio_capture/playback_state` queue can retain only the newest transition. During an underrun, that is commonly `Idle`, so `CaptureGate` never sees `Active` and continues publishing rover microphone audio.

Frame loss makes this worse: discontinuous walkie input creates repeated short active/idle playback periods.

### Confirmed contract gap: web bridge invents 16 kHz metadata

`common/web_bridge/src/main.rs:399-402` accepts only `audio_data`. `common/web_bridge/src/walkie-audio.rs:7,23-52` then declares every frame as 16 kHz mono, assigns a new timestamp, and creates server-side sequence IDs.

The browser's actual capture rate is unavailable in this repository snapshot. If it sends native AudioContext PCM at 44.1 or 48 kHz, declaring it as 16 kHz causes wrong resampling, distorted/slow audio, and roughly 2.8-3x excess buffered duration. Even if the current UI downsamples to 16 kHz, the contract cannot prove that fact.

### Attribution

- Commit `0169368` introduced the source-aware playback runtime, size-one-unspecified input set, five-second TTS buffer, destructive overflow, playback-state suppression path, and server-generated walkie metadata. It is the primary regression commit.
- Commit `33622d8` changed TTS contract types and web command gating, but did not implement the failing playback/dataflow path.
- Commit `e2c74be` is not present in this repository's reachable history, so its UI attribution cannot be independently verified here.
- Later commits through current `b8e3ffc` do not correct these mechanisms.

## Evaluated approaches

### A. Queue and buffer enlargement only

Changes: set larger Dora queues; increase TTS buffer beyond five seconds.

Pros:

- smallest patch;
- likely improves short utterances and light walkie load.

Cons:

- only moves TTS truncation to a longer utterance;
- no real backpressure;
- deep walkie queues increase latency and replay stale speech;
- cannot correct false 16 kHz metadata;
- does not guarantee `Active` survives an `Active -> Idle` burst.

Verdict: emergency mitigation only; reject as final fix.

### B. Paced media pipeline with bounded jitter queues

Changes:

- Pace TTS chunk publication against a monotonic media clock: one 20 ms chunk per 20 ms deadline. Use the existing bounded worker channel to block offline synthesis when downstream pacing falls behind.
- Retain at most one pending resampled TTS chunk in playback. Make frame enqueue all-or-none; retry when capacity becomes available. Never partially enqueue then clear on ordinary capacity pressure.
- Size TTS playback storage for scheduling jitter, not maximum utterance length. Keep intentional clears only for walkie preemption, cancellation, device failure, or explicit timeout.
- Add explicit per-input Dora queue sizes. Media queues stay small and duration-bounded; lifecycle/control queues preserve all nearby transitions.
- Coalesce playback transitions per tick: if any real sample was consumed, publish `Active`; publish `Idle` no earlier than the next fully idle tick. Never publish `Active` and `Idle` back-to-back in one scheduler turn.
- Replace browser `audio_data`-only input with a versioned PCM frame containing actual `stream_id`, `frame_id`, capture timestamp, sample rate, channels, format, and sample count. Validate stable format and monotonic IDs. Browser either sends native-rate mono PCM honestly or downsamples and declares the resulting rate.
- Bound the web bridge's current `Vec` walkie queue with `VecDeque`; drop oldest only when latency budget is exceeded and count every drop.

Pros:

- fixes source behavior instead of extending limits;
- bounded memory and bounded live latency;
- preserves TTS duration independent of text length;
- cancellation/preemption remains responsive at one chunk interval;
- moderate change using existing worker and playback architecture.

Cons:

- requires deterministic pacing tests;
- TTS synthesis worker remains occupied while paced output is emitted;
- browser contract requires a coordinated UI update not present in this checkout.

Verdict: recommended. Best YAGNI/KISS balance.

### C. Explicit credit/ACK flow control across Dora

Changes: playback publishes available sample credits; edge voice sends only against credits and receives per-frame acknowledgements.

Pros:

- strongest end-to-end backpressure;
- adapts to device stalls and varying output periods;
- precise accounting.

Cons:

- new protocol, states, recovery rules, and deadlock risks;
- more tests and operational complexity;
- unnecessary if media-clock pacing and bounded queues meet acceptance criteria.

Verdict: defer. Adopt only if option B still overruns under measured workstation load.

## Recommended design

Implement option B in this order.

### 1. Make queue policy explicit in all deployed dataflows

Use Dora's option form:

```yaml
walkie_audio:
  source: zenoh-bridge/audio_stream
  queue_size: 4
```

Initial policy, then tune by measured duration:

- `tick`: 1, latest tick sufficient;
- walkie PCM: 4 x standardized 20 ms frames, maximum 80 ms Dora jitter backlog;
- paced TTS PCM: 4 x 20 ms frames;
- terminal/control/state inputs: 8, preserving short transition bursts;
- orchestra `audio_stream_web`: same walkie media policy;
- apply identically to remote and direct rover dataflows.

Do not use a large walkie queue. Stale live audio is worse than an explicitly counted late-frame drop.

### 2. Add source pacing and bounded backpressure

- Extract a small TTS pacer driven by `Instant`/monotonic deadlines.
- Release chunks according to cumulative sample duration, not loop iterations.
- Delay `tts_synthesis_state=Completed` until all audio chunks have been handed to playback in order.
- Keep the bounded worker event channel. Do not drain audio events faster than the pacer can release them; continue handling cancellation and walkie state while waiting.
- On cancellation, discard only that command's unsent chunks and emit one terminal result.

### 3. Replace destructive capacity handling

- Add an all-or-none `try_enqueue_tts_frame` operation.
- If insufficient room, retain that frame and retry after CPAL consumption.
- Add a bounded stall deadline. A stopped device produces explicit `PlaybackFailed`; normal temporary fullness does not erase queued speech.
- Make playback jitter capacity configurable in milliseconds. Suggested starting range: 500-1,000 ms, validated against device callback timing.
- Keep `clear_tts` for deliberate walkie preemption and terminal failure only.

### 4. Harden echo suppression transitions

- Coalesce callback transitions so one tick cannot publish both `Active` and `Idle`.
- Sequence playback states and ignore stale states in capture.
- Preserve the existing 400 ms acoustic tail.
- Log state sequence, consumed source, command ID, and suppression duration.
- Consider walkie authority as an additional conservative suppression lease, but retain actual-consumption state for TTS. Do not replace consumption truth with browser intent.

### 5. Make browser PCM metadata authoritative

- Reuse `AudioFrameMetadata`/`PcmFramePacket` semantics instead of another parallel shape.
- Require protocol version, stream UUID, monotonic frame ID, capture timestamp, actual sample rate, channels, sample count, and format.
- Enforce mono and finite samples; validate payload length and stable stream format.
- Preserve metadata through web bridge, orchestra bridge, Zenoh, rover bridge, and playback.
- If browser resampling is chosen, test and declare the post-resample rate. Never label native 48 kHz samples as 16 kHz.

### 6. Add observability before hardware acceptance

Per stream/command counters:

- received, forwarded, enqueued, consumed, dropped, late, sequence-gap samples/frames;
- current/high-water queue duration in milliseconds;
- TTS generated samples, resampled samples, consumed samples;
- playback state sequence and capture-suppressed frames;
- terminal reason and time from command to final consumed sample.

Counters must emit periodically, not only at process shutdown.

## Validation and success criteria

Current fresh tests pass but do not prove audibility:

```text
audio_capture: 11 passed
audio_playback: 23 passed
edge_voice: 13 passed
```

Required new tests:

1. Dataflow parser test asserts explicit queue sizes in remote, direct, and orchestra YAML.
2. Fake-clock pacer test proves exact chunk order, no early burst, cancellation within one chunk, terminal after final chunk.
3. Long TTS render test for 10, 30, and 60 seconds: generated/resampled/consumed sample counts match expected resampler tolerance; zero ordinary-capacity drops; completion follows final consumption.
4. Dora end-to-end synthetic test sends numbered 20 ms chunks faster than real time, then with pacing. It must demonstrate the old gap and zero gaps with the new path.
5. Walkie waveform test at declared 16, 44.1, and 48 kHz checks duration, RMS continuity, frame sequence, and output sample count after resampling.
6. Transition test forces repeated buffer underruns and proves capture observes `Active` before `Idle`; zero microphone frames publish during active playback plus 400 ms tail.
7. Preemption test proves walkie interrupts TTS once, removes only interrupted TTS, and begins live audio within the latency budget.
8. Workstation Docker hardware test records speaker output and rover microphone publication counters. Acceptance: intelligible walkie, complete long TTS, no echo frames, no unexplained sequence gaps or TTS drops.

Target operational thresholds:

- nominal walkie/TTS media gaps: 0;
- ordinary TTS capacity drops: 0;
- walkie queued latency: <= 80 ms at Dora boundary, <= 250 ms total playback jitter buffer;
- echo leakage: 0 rover microphone frames published while playback is active and for 400 ms after idle;
- lifecycle: exactly one terminal result, only after final sample consumption or an explicit interruption/failure.

## Risks and dependencies

- UI source is missing from this checkout. Browser capture/resampling behavior and commit `e2c74be` need the UI repository/worktree before implementation.
- Pacing based on sleep alone can drift. Deadlines must derive from cumulative samples and a monotonic start time.
- Queue sizes are frame-count based. Standardize frame duration first; otherwise the latency bound is false.
- State queues alone do not solve echo if `Active` and `Idle` are generated in one turn. Transition coalescing is required.
- A larger TTS buffer without pacing is not an accepted substitute.

## Next steps

1. Confirm browser walkie frame shape, native sample rate, frame duration, and whether it currently downsamples.
2. Create a detailed implementation plan for option B, split into queue/contract, TTS pacing, playback/state, and end-to-end validation phases.
3. Reserve option C only behind failed measured acceptance of option B.

## Unresolved questions

1. Where is the current web UI checkout containing commit `e2c74be`?
2. Does browser walkie currently send native AudioContext PCM or pre-resampled 16 kHz mono?
3. What maximum accepted walkie one-way latency should replace the proposed 250 ms ceiling?
