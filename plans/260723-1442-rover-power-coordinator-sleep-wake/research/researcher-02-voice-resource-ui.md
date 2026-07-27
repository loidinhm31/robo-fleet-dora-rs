# Research: voice wake, resource policy, and UI integration

## Findings

### Continuous KWS v1

- Sherpa's Rust KWS API creates a reusable `KeywordSpotter` stream, accepts
  waveform chunks, repeatedly calls `is_ready/decode/get_result`, and resets
  after a non-empty keyword result ([keyword_spotter.rs:28-55](/mnt/data/ws/sherpa-onnx/rust-api-examples/examples/keyword_spotter.rs:28)).
- KWS supports file-defined keywords plus per-stream extra keywords
  ([keyword_spotter.rs:57-91](/mnt/data/ws/sherpa-onnx/rust-api-examples/examples/keyword_spotter.rs:57)). For v1, load a fixed, signed keyword file at startup; avoid runtime keyword mutation.
- Model config exposes provider, thread count, and debug flags
  ([keyword_spotter.rs:93-111](/mnt/data/ws/sherpa-onnx/rust-api-examples/examples/keyword_spotter.rs:93)). Benchmark `num_threads=1` first on the target rover; only increase if p95 detection latency improves without violating CPU budget.
- Silero VAD is chunk-fed (example window 512), with tunable threshold, min speech/silence, max speech, and a 30.0 detector buffer parameter ([silero_vad_remove_silence.rs:37-62](/mnt/data/ws/sherpa-onnx/rust-api-examples/examples/silero_vad_remove_silence.rs:37), [silero_vad_remove_silence.rs:64-84](/mnt/data/ws/sherpa-onnx/rust-api-examples/examples/silero_vad_remove_silence.rs:64)). VAD detects speech, not the wake phrase; use it only as an optional CPU gate after target-hardware measurements.
- Recommendation: continuous low-cost KWS in `IdleListening`; transition to `NormalRover` only on a debounced keyword event. Do not run full ASR while dormant. Preserve 0.5–1.0 s pre-roll if VAD-gating is introduced, otherwise wake words can be clipped.

### Existing audio/lifecycle gates

- `CaptureGate` already separates user intent from lifecycle resume: quiesce closes capture, marks `fresh_start_required`, and resume does not reopen the microphone automatically ([capture_gate.rs:40-65](/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_capture/src/capture_gate.rs:40)). Power coordinator should call this gate explicitly for `Dormant`/`ScheduledCapture` transitions.
- Playback suppression is sequence-fenced by producer instance and sequence ID; active playback suppresses capture, idle/unavailable adds a 400 ms tail ([capture_gate.rs:67-112](/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_capture/src/capture_gate.rs:67)). A wake acknowledgment must use the same playback state path, and KWS must ignore audio during active playback/tail.
- Playback state reports source (`Tts`/`Walkie`), command ID, producer instance, and monotonic sequence ([state.rs:64-108](/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_playback/src/state.rs:64)). A prerecorded “I am on” clip should be a distinct `WakeAck` source/reason, not an unlabelled TTS command; this makes suppression and audit deterministic.
- `edge_voice` has lifecycle admission, worker stop timeout (25 s), voice status, metrics, and lifecycle component status outputs ([runtime.rs:31-37](/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/edge_voice/src/runtime.rs:31), [runtime.rs:40-72](/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/edge_voice/src/runtime.rs:127-145)). Coordinator should gate ASR/TTS workers by profile, while leaving KWS separate and always-on only in IdleListening.

### Auto demand and per-domain CPU policy

- Model state as explicit profiles: `Dormant` (KWS + lifecycle/status only), `IdleListening` (KWS + mic), `NormalRover` (normal command/audio stack), and `ScheduledCapture` (prewarm dependencies + capture). Avoid one global boolean.
- Coordinator should maintain a demand ledger: source, domain (`audio_in`, `audio_out`, `vision`, `network`, `recording`), expiry/lease, priority, and required-ready deadline. `Auto` computes the least-power profile satisfying active demands; manual `Awake`/`Sleep` overrides may be safety constrained.
- Deterministic state transitions require epoch/revision and readiness acknowledgements. On wake phrase: local coordinator increments epoch, enables normal profile, plays prerecorded ack, and starts an idle timeout. Orchestra sync later may supersede local authority with higher epoch.
- Resource monitor should publish per-domain CPU/RSS/power proxy, lifecycle readiness, and transition latency. Do not claim energy savings until target-hardware tests measure KWS-only vs full stack.

### Socket.IO policy/status/history UI

- Existing UI hooks use authenticated Socket.IO subscriptions, entity filtering, stale request IDs, reset-on-connect/disconnect, and readiness/degraded states ([use-recording-schedule-events.ts:12-66](/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/use-recording-schedule-events.ts:12)). Reuse this pattern for `power_policy_snapshot`, `power_transition`, `power_status`, and `power_history` events.
- Main control page already centralizes socket lifecycle, clears audio/voice/resource state on disconnect, and maintains bounded transcription history ([RoboRoverControl.tsx:274-329](/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/pages/RoboRoverControl.tsx:274)). Add a power card showing profile, policy (`Auto/Awake/Sleep`), pending demands, readiness, epoch, and last transition; render history as bounded append/prepend events with dedupe by transition ID.
- UI actions should emit authenticated `power_policy_set` with request ID/entity ID and display command result/error. Never let browser UI directly toggle hardware nodes; coordinator remains authority.

## Validation and benchmarks

1. Target rover benchmark matrix: KWS-only, VAD+KWS, and full voice stack; 16 kHz mono, realistic noise, 30–60 minute idle runs. Record CPU%, RSS, thermal/power proxy, false accepts/hour, false rejects, wake latency p50/p95, and audio-tail suppression failures.
2. Acceptance gates: keyword detection p95 (target to be agreed), wake acknowledgment p95, normal-profile readiness p95, no duplicate wake transitions per epoch, no capture during playback/tail, and no stale command replay after Orchestra reconnect.
3. Rust tests: KWS result debounce/reset, lifecycle admission, playback suppression, demand expiry, epoch fencing, and offline local wake.
4. Browser tests: Socket.IO connect/disconnect, auth failure, entity isolation, stale event rejection, policy command acknowledgement, transition history dedupe; use Playwright/E2E only after event contract stabilizes.

## Unresolved questions

- Continuous KWS versus VAD-gated KWS: choose only after rover benchmark; VAD may reduce CPU but adds buffering/latency and false negatives.
- Exact wake phrase, language(s), noise environment, and acceptable false-wake rate.
- Required p95 targets for acknowledgment and normal readiness; fixed prewarm lead time versus measured p95 scheduler latency.
- Whether `Awake/Sleep` manual policy may override safety-critical recording/telemetry demands, or only request a profile.
- Exact Socket.IO event schemas and retention limits for power history.
