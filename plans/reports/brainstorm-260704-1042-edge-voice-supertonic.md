# Edge Voice Supertonic Brainstorm

Date: 2026-07-04
Status: agreed architecture; comprehensive research complete; implementation not started
Repositories inspected:

- Backend/dataflows: `/mnt/data/ws/sharing/robo-fleet-dora-rs`
- Active web UI: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`
- Sherpa-ONNX source/examples: `/mnt/data/ws/sherpa-onnx`

## Executive Decision

Replace Orchestra Kokoro and rover Piper TTS with one rover-side `edge_voice` node using the official pinned `sherpa-onnx = 1.13.3` Rust API and the Sherpa Supertonic 3 INT8 model bundle.

Agreed behavior:

- Supertonic 3 is the only production TTS engine in this scope.
- English and Vietnamese use one resident model; language changes per request configuration without loading another model.
- Runtime TTS configuration is global across the active fleet.
- Runtime configuration is not persisted. Restart resets it to English defaults.
- Configuration applies live. An in-progress utterance finishes with its captured configuration; the next utterance uses the new configuration.
- Rover microphone frames are suppressed while rover speaker playback is active, plus a short tail, preventing rover speech from re-entering central STT.
- Browser-origin STT remains active because it is a separate audio source.
- `audio_playback` remains the only owner of physical rover speaker output.
- Remove `orchestra/kokoro_tts`; do not retain dual TTS engines as fallback debt.

## Problem Statement

Current manual browser TTS terminates at Orchestra Kokoro, so sound comes from the workstation. The rover already has a `sherpa_tts` crate and Zenoh TTS command path, but the node:

- is disabled in the default rover dataflow
- uses `sherpa-rs` VITS/Piper rather than the official `sherpa-onnx` crate
- supports one English model
- opens a second speaker path directly with Rodio
- has no authoritative runtime configuration or fleet status
- has no playback/STT conflict mitigation

Target workflow:

```text
browser
  -> Socket.IO tts_command
  -> common/web_bridge
  -> Dora tts_command
  -> orchestra/zenoh_bridge
  -> Zenoh rover/{entity_id}/cmd/tts
  -> rover/zenoh_bridge
  -> Dora edge_voice/tts_command
  -> Supertonic 3 synthesis
  -> typed 44.1 kHz PCM
  -> audio_playback
  -> rover speaker
```

Global runtime configuration follows a separate control path and fans out to every active rover.

## Current Repository Findings

### Command routing already exists

Most data transport is implemented:

- UI emits `tts_command` with `{ text }`.
- `common/web_bridge` validates text, converts it to shared `TtsCommand`, and emits Dora `tts_command`.
- Orchestra bridge maps `tts_command_web` to `rover/{selected_entity}/cmd/tts`.
- Rover bridge subscribes to `rover/{entity_id}/cmd/tts` and emits Dora `tts_command`.

The missing default-dataflow consumer is the main reason rover TTS is inactive.

### TTS currently has two consumers

`orchestra/orchestra-dataflow.yml` sends `web-bridge/tts_command` to both:

- `orchestra-bridge`, which routes it toward the selected rover
- `kokoro-tts`, which plays it at the workstation

Activating rover TTS without removing Kokoro would create duplicate playback at different machines.

### Rover speaker ownership is already split

- `audio_playback` owns a CPAL output stream for browser walkie-talkie audio.
- `sherpa_tts` opens a separate Rodio output stream.

Two independent processes opening the same embedded audio device is not a reliable contract. It may work through PipeWire/dmix on some hosts and fail or contend on direct ALSA deployments.

### Sample-rate mismatch is blocking

- Existing `audio_playback` assumes fixed 16 kHz mono input.
- Supertonic produces 44.1 kHz mono Float32 audio.

Passing Supertonic samples through the current 16 kHz path would alter duration and pitch. The playback contract must carry sample rate and resample/mix correctly.

### Official Sherpa runtime is already in the workspace

`orchestra/central_speech_recognizer` already pins:

```toml
sherpa-onnx = { version = "=1.13.3", default-features = false, features = ["static"] }
```

`edge_voice` should use the same exact crate version and linkage strategy. This removes `sherpa-rs` and avoids two wrapper/runtime families in one workspace.

## Supertonic 3 Evidence

Official sources:

- [Supertonic 3 model card](https://huggingface.co/Supertone/supertonic-3)
- [Supertonic GitHub project](https://github.com/supertone-inc/supertonic)
- [Sherpa-ONNX Supertonic documentation](https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html)
- [Sherpa Rust Supertonic example](https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/supertonic_tts.rs)
- [Sherpa TTS model release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)

Confirmed capabilities:

- 31 languages, including English `en` and Vietnamese `vi`
- one model supports all languages
- 10 bundled speaker styles selected with `sid` 0 through 9
- per-generation language, speaker, speed, denoising steps, random seed, chunk length, and inter-chunk silence
- 44.1 kHz output
- expression tags including `<laugh>`, `<breath>`, and `<sigh>`
- CPU inference without GPU requirement
- ONNX Runtime deployment on edge/Raspberry Pi class devices
- official Sherpa `OfflineTts` and `GenerationConfig` integration
- progress callback capable of cancellation and incremental sample delivery

Sherpa bundle selected:

```text
sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
archive size: 128,774,318 bytes (~122.8 MiB)
sha256: 82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427
```

Required extracted files: `duration_predictor.int8.onnx`, `text_encoder.int8.onnx`, `vector_estimator.int8.onnx`, `vocoder.int8.onnx`, `tts.json`, `unicode_indexer.bin`, and `voice.bin`.

The upstream voice pack is built from sorted style names, so expected SID mapping is:

Expected mapping: SIDs 0-4 are F1-F5; SIDs 5-9 are M1-M5.

Validate this mapping against the downloaded release before exposing names in UI. If not verifiable, expose neutral `Voice 1` through `Voice 10` labels initially.

### Performance evidence limitation

Official Supertonic material says Raspberry Pi/CPU/edge ready, but does not publish a reproducible Raspberry Pi 5 RTF, peak RSS, cold-start duration, or concurrent-vision benchmark. Vendor WER/CER and latency plots are useful directional evidence, not a substitute for this rover's workload measurement.

Therefore implementation approval depends on a Pi 5 benchmark gate. The model decision remains Supertonic; benchmark failure triggers tuning or explicit product reassessment, not an undocumented Piper fallback.

### License

- Supertonic sample code: MIT.
- Supertonic model: BigScience OpenRAIL-M.
- Sherpa-ONNX crate: Apache-2.0.

OpenRAIL-M includes redistribution requirements and use restrictions. Distribution must include the model license and required notices. Product/legal review is required before commercial redistribution. Generated robot speech must not violate restricted uses.

## Recommended Runtime Configuration

Use an enum and bounded values. Never accept model paths, provider names, or arbitrary JSON from a browser.

```rust
enum TtsLanguage {
    En,
    Vi,
}

struct TtsRuntimeConfig {
    revision: u64,
    language: TtsLanguage,
    speaker_id: u8,
    speed: f32,
    num_steps: u8,
    volume: f32,
}
```

Recommended defaults after every `web_bridge` or rover restart:

```text
language=en
speaker_id=5        # expected M1; verify release mapping
speed=1.0
num_steps=8
volume=0.8
revision=0
```

Server validation:

```text
language: en | vi
speaker_id: 0..9
speed: 0.7..2.0
num_steps: 5..12
volume: 0.0..1.0
```

These values are runtime state only:

- no MongoDB collection
- no localStorage authority
- no environment mutation
- no generated config file
- no survival across process restart

Environment variables configure immutable deployment concerns only: `TTS_MODEL_DIR`, `TTS_NUM_THREADS`, default language/speaker/speed/steps/volume, queue capacity, and microphone tail duration.

`TTS_DEFAULT_LANGUAGE` defaults to English. Other default overrides are deployment conveniences, not runtime persistence.

## Runtime State Semantics

### Authority

`common/web_bridge` is the session authority for desired global TTS configuration.

- Initialize defaults at startup.
- Send current state to each authenticated browser after connection.
- Accept authenticated, rate-limited updates.
- Increment a monotonic runtime `revision` after validation.
- Emit one global config command to the Orchestra bridge.
- Broadcast the authoritative desired state to all clients.

The Orchestra bridge retains the latest desired config as a runtime cache so it can apply it to a rover activated after the original update. It is not a second persistent authority.

### Fleet fan-out

Do not route global configuration through `selected_entity`.

```text
web_bridge/tts_config
  -> orchestra bridge
  -> for each active rover:
       rover/{entity_id}/cmd/voice/config
```

On rover activation, Orchestra bridge sends the cached latest config to that rover. An offline rover starts with English defaults and converges when it connects and becomes active.

### Live application

Supertonic model files do not reload for language, speaker, speed, steps, or volume changes.

- Validate config.
- Atomically replace current config.
- Emit applied status with revision.
- A synthesis worker snapshots config when dequeuing an utterance.
- Do not mutate an in-progress utterance.
- Do not cancel speech merely because config changed.

This gives deterministic utterances and immediate control response without expensive model reload.

### Desired versus applied state

Global delivery can be partial. UI must distinguish:

- desired global revision/config
- per-rover applied revision/config
- rover lifecycle: `loading`, `ready`, `speaking`, `error`

Do not claim fleet-wide success until all currently active rovers acknowledge the desired revision.

## Proposed Shared Contracts

Keep text commands separate from global configuration.

```rust
struct TtsCommand {
    text: String,
    timestamp: u64,
    priority: TtsPriority,
    command_id: String,
}

struct TtsConfigCommand {
    revision: u64,
    config: TtsRuntimeConfig,
}

enum VoiceState {
    Loading,
    Ready,
    Speaking,
    Error,
}

struct VoiceStatus {
    entity_id: String,
    state: VoiceState,
    applied_revision: u64,
    active_config: TtsRuntimeConfig,
    active_command_id: Option<String>,
    error: Option<String>,
    timestamp: u64,
}
```

Add `command_id`; timestamp alone is not a safe correlation key.

Suggested Zenoh topics:

```text
rover/{entity_id}/cmd/tts                 existing utterance command
rover/{entity_id}/cmd/voice/config        global-config fan-out
rover/{entity_id}/voice/status            applied status/telemetry
```

Suggested Socket.IO events:

```text
client -> server: tts_command
client -> server: tts_config_update
server -> client: tts_config_state
server -> client: voice_status
server -> client: tts_command_ack
```

An acknowledgement should report accepted/routed/rejected, not imply audible completion. Completion comes from `VoiceStatus` or a dedicated command result.

## `edge_voice` Node Design

Naming:

- directory/crate/binary: `rover-kiwi/edge_voice`, `edge_voice`
- Dora node ID: `edge-voice` to match existing dataflow kebab-case convention

Responsibilities:

- validate all required model files before constructing the engine
- construct one `OfflineTts` instance at startup
- hold atomic runtime config
- accept bounded TTS commands
- synthesize sequentially on a worker thread
- map config to Sherpa `GenerationConfig`
- emit typed PCM carrying sample rate/channels/command ID
- emit status and metrics
- remain responsive to config and stop events during synthesis

Non-responsibilities: physical speaker ownership, configuration persistence, rover targeting, language detection, acoustic echo cancellation, and runtime model download.

### Sherpa mapping

```text
language       -> GenerationConfig.extra["lang"] = "en" | "vi"
speaker_id     -> GenerationConfig.sid
speed          -> GenerationConfig.speed
num_steps      -> GenerationConfig.num_steps
```

Keep `seed=-1`, Sherpa chunk limits, and inter-chunk silence at upstream defaults initially. Add controls only after a measured need.

### Concurrency and queue policy

The current synchronous `sherpa_tts` event loop blocks Dora input handling throughout synthesis and playback. Replace it with:

- responsive Dora/control loop
- one synthesis worker
- bounded queue
- sequential output to prevent overlapping robot speech

Recommended queue behavior:

- capacity 8 initially
- reject newest normal/low-priority command when full
- emergency command may clear queued lower-priority speech
- never grow an unbounded `Vec`
- never run multiple Supertonic generations concurrently on a Pi 5 initially

Priority arbitration needs explicit tests. Existing `TtsPriority` ordering is defined but unused.

### Startup failure

Initialize the Dora node before expensive model loading when feasible, then report:

```text
loading -> ready
loading -> error
```

On missing/corrupt assets, remain control/status responsive in error state instead of crashing the entire rover dataflow. Reject TTS commands with a sanitized reason.

## Audio Playback Architecture

### Single speaker owner

`edge_voice` emits PCM; `audio_playback` owns the device and playback lifecycle.

```text
edge_voice/tts_audio (44.1 kHz mono F32 + metadata)
                                 \
                                  -> audio_playback -> speaker
zenoh_bridge/audio_stream (16 kHz mono F32) /
```

`audio_playback` must distinguish sources and normalize them to the hardware stream rate.

Recommended metadata:

```text
source=tts|walkie
stream_id/command_id
sample_rate
channels
sample_count
format=f32le
priority
```

### Resampling

Do not relabel 44.1 kHz data as 16 kHz.

Preferred design:

- open device at a supported/native rate
- resample every source to that rate
- keep one output mixer/ring buffer
- emit actual playback start/finish, not enqueue start/finish

A simple 44.1-to-16 kHz downsample would work for intelligibility but wastes Supertonic output quality and complicates mixing. One source-aware playback/resampling boundary is cleaner.

### Arbitration

TTS and walkie audio must not overlap accidentally.

Initial policy:

- emergency TTS preempts queued normal TTS
- active walkie playback and TTS serialize
- define TTS as higher priority than normal walkie chunks, or reject TTS while walkie is active
- no mixing until product explicitly requires it

The implementation plan should choose one deterministic policy and expose it in UI feedback.

## STT Self-Trigger Mitigation

Moving speech from workstation to rover creates this acoustic path:

```text
edge_voice -> rover speaker -> rover microphone -> audio_capture
  -> Zenoh -> central STT -> command_parser -> rover actuator command
```

This is a safety issue. A synthesized phrase containing command vocabulary can be recognized as operator speech.

### Recommended half-duplex gate

Use physical playback lifecycle, not synthesis lifecycle:

```text
audio_playback/playback_state
  -> audio_capture/playback_state
```

While any rover speaker source is active:

- keep microphone device open
- drain/discard captured frames locally
- do not publish them toward Orchestra
- flush buffered microphone samples before resuming
- continue suppression for configurable tail, initially 400 ms

Keeping the input stream open avoids slow/reliability-sensitive ALSA reopen cycles and does not overwrite manual `AudioAction::Start/Stop` state.

Track two independent conditions:

```text
capture_enabled_by_user
playback_suppressed

effective_publish = capture_enabled_by_user && !playback_suppressed
```

This also prevents walkie-talkie playback from feeding central STT. Browser STT remains independent and must not be suppressed.

### Deferred AEC/barge-in

Half-duplex means the rover cannot hear a local operator during its own speech. That is accepted for this phase. Full-duplex barge-in requires acoustic echo cancellation with playback reference timing and device calibration; it is out of scope.

## Web UI Design

The active UI is the adjacent `robo-control-app` monorepo, not a backend subdirectory.

Place controls in `VoiceControls`:

- language: English / Vietnamese
- voice style: verified named style or neutral Voice 1..10
- speed
- quality: map UI labels to bounded steps, e.g. Fast=5, Balanced=8, Quality=12
- volume
- global/fleet label to make scope explicit
- desired/applied state indicator

Behavior:

- UI initializes only from server `tts_config_state`.
- Do not store authoritative config in localStorage.
- Update is not optimistic; show pending revision until rover acknowledgements arrive.
- Show partial convergence, e.g. `2/3 rovers applied`.
- Reconnect fetches current in-memory state from `web_bridge`.
- Server restart returns English defaults and broadcasts revision 0/default state.
- Disable or debounce rapid controls to avoid config floods.
- Keep TTS text command targeted to the currently selected rover.
- Keep global configuration independent of selected rover.

Suggested TypeScript contracts mirror Rust enums exactly and use discriminated unions for state.

## Removal and Rename Scope

### Remove Orchestra Kokoro

- `orchestra/kokoro_tts/`
- workspace member in root `Cargo.toml`
- Orchestra Docker workspace/member/scaffold/build references
- `kokoro-tts` node from Orchestra dataflow
- Kokoro model download/check scripts and documentation if no other consumer remains
- regenerated lockfile dependencies, including `kokoro-tiny`, when no longer referenced

Keep the Orchestra bridge `tts_command_web` input; it is now the only manual TTS route.

### Rename rover TTS

- `rover-kiwi/sherpa_tts/` -> `rover-kiwi/edge_voice/`
- package/binary `sherpa_tts` -> `edge_voice`
- workspace and Docker member paths
- Docker scaffolding/build/copy/diagnostic labels
- performance-monitor process list
- dataflow comments and node ID
- README, architecture, setup, troubleshooting, and model docs

### Replace model provisioning

- retire `download_sherpa_vits_models.sh`
- download pinned Supertonic archive
- verify SHA-256 before extraction
- update `make models` and `make check-models`
- update Docker model checks and read-only mount expectations
- fail clearly when any required component is absent

Do not download models from inside the rover node.

## Dataflow Changes

Rover default dataflow:

```yaml
- id: edge-voice
  inputs:
    tts_command: zenoh-bridge/tts_command
    tts_config: zenoh-bridge/tts_config
  outputs:
    - tts_audio
    - voice_status

- id: audio-playback
  inputs:
    walkie_audio: zenoh-bridge/audio_stream
    tts_audio: edge-voice/tts_audio
  outputs:
    - playback_state

- id: audio-capture
  inputs:
    tick: dora/timer/millis/50
    audio_control: zenoh-bridge/audio_command
    playback_state: audio-playback/playback_state
```

Rover bridge gains:

- config subscriber/output
- voice-status input/publisher

Orchestra bridge gains:

- global config input fan-out
- latest-config runtime cache
- status subscriptions/output

Web bridge gains:

- runtime global config state
- config update queue/output
- voice status input and Socket.IO broadcast
- initial/reconnect state delivery

Direct/standalone rover dataflow needs equivalent local wiring. Global fleet fan-out does not apply in standalone mode, but the same Socket.IO contracts and defaults should.

## Security and Validation

- Require authenticated session for config and TTS commands.
- Apply existing command rate limiter plus a stricter config-update cooldown.
- Validate UTF-8 text, trim whitespace, and retain maximum length.
- Consider reducing current 1,000-character maximum for interactive rover speech or chunk explicitly.
- Reject unsupported expression/control markup if arbitrary tags are not intended.
- Enum-only language; never pass arbitrary `lang` strings.
- Clamp/reject speaker, speed, steps, and volume server-side and rover-side.
- Never expose filesystem model paths through Socket.IO.
- Sanitize model errors before sending to browsers.
- Include `command_id`, config revision, and rover identity in logs; do not log secrets.

## Observability

Per command:

- queue wait milliseconds
- synthesis milliseconds
- audio duration milliseconds
- real-time factor
- time to first generated samples
- playback start/end timestamps
- command result
- config revision, language, SID, steps

Node/runtime:

- queue depth and rejected commands
- CPU and RSS for `edge-voice`
- model load duration
- synthesis failures
- playback underruns/overruns
- resampler failures
- microphone-suppressed frames and duration
- applied config revision per rover

Update `performance_monitor` from `sherpa-tts` to `edge-voice` so existing fleet metrics include CPU/RSS.

## Validation Strategy

### Unit and contract tests

- runtime config validation boundaries
- config revision monotonicity and stale-update rejection
- `en` and `vi` generation config mapping
- SID and step bounds
- command ID propagation
- queue capacity and priority behavior
- config snapshot at dequeue
- PCM metadata and sample-count validation
- resampling duration/tone tests
- playback state transitions
- capture gate truth table and buffer flush
- bridge selected-rover TTS routing
- bridge all-active-rover config fan-out
- newly activated rover receives cached config
- Socket.IO authentication, validation, reconnect state

### Model loading tests

Verify every required file/checksum, successful `OfflineTts::create`, 44.1 kHz output, 10 speakers, non-empty English/Vietnamese synthesis, and controlled invalid-config failure.

### Raspberry Pi 5 benchmark gate

Run on the actual rover image and power/cooling profile, with vision pipeline active.

Corpus:

- short command acknowledgements in English and Vietnamese
- numbers, units, dates, abbreviations, punctuation
- Vietnamese diacritics and common robot vocabulary
- long text near chosen maximum
- all supported voice SIDs at balanced quality

Measure:

- cold model load time
- peak and steady RSS
- per-utterance synthesis latency
- RTF and time to first audio
- total CPU and per-core CPU
- thermal throttling over repeated synthesis
- impact on camera FPS, tracking latency, and servo loop
- playback underruns
- subjective intelligibility on rover speaker

Initial acceptance targets for planning:

- no rover process crash or OOM
- RTF < 1.0 for both languages at balanced steps under active vision load
- no sustained visual-servo deadline regression
- no audio underruns in 100 sequential short utterances
- config update visible on every active rover within 2 seconds on healthy LAN
- zero rover-microphone frames forwarded during speaker-active window and tail
- zero self-triggered parser commands during adversarial spoken command tests

Do not invent a latency target before the first Pi 5 baseline. Record the baseline, then set a product SLA.

### End-to-end scenarios

1. Start system; UI and all rovers report English defaults.
2. Send English text; only selected rover speaks.
3. Change global language to Vietnamese; all active rovers acknowledge same revision.
4. Send Vietnamese text; selected rover speaks without model reload.
5. Activate a rover after config change; it receives cached current config.
6. Restart web bridge; desired global state returns to English.
7. Disconnect a rover during config update; UI reports partial convergence.
8. Speak text containing `move forward`; rover microphone suppression prevents command execution.
9. Browser STT remains usable during rover playback.
10. Missing model on one rover reports error without taking down the rest of its dataflow.

## Risks and Mitigations

### Pi 5 performance unknown

Risk: official material lacks reproducible Pi 5 metrics under concurrent YOLO/ReID/tracking load.

Mitigation: mandatory hardware benchmark; tune threads and steps; serialize generation; monitor thermal and vision regression.

### Newer Sherpa Supertonic integration

Risk: Supertonic support is recent relative to VITS/Piper.

Mitigation: exact crate/model pin, checksum, model-loading integration tests, controlled error state.

### Audio device and rate handling

Risk: dual device ownership or incorrect sample rate causes silence, contention, or distorted speech.

Mitigation: one playback owner, explicit PCM metadata, resampling tests, real hardware validation.

### Partial fleet convergence

Risk: runtime-only Zenoh delivery means offline rovers miss updates.

Mitigation: Orchestra runtime cache, apply-on-activation, per-rover revision status. Restart resetting to English is agreed behavior.

### Self-triggered commands

Risk: rover hears itself and central STT executes spoken command text.

Mitigation: playback-derived local microphone publication gate plus tail; adversarial end-to-end test.

### Half-duplex operator experience

Risk: local operator speech is ignored while rover speaks.

Mitigation: accepted for this phase; keep utterances short; consider cancellation or AEC later.

### OpenRAIL-M obligations

Risk: model distribution/use requirements differ from permissive source licenses.

Mitigation: ship license/notices, document model origin/checksum, legal review before distribution.

### User-owned concurrent changes

The worktree currently contains unrelated modifications and untracked STT plan artifacts. Implementation must preserve them and avoid broad formatting or destructive Git operations.

## Success Criteria

- `orchestra/kokoro_tts` and all exclusive Kokoro dependencies/assets are removed.
- `edge_voice` uses official exact-version Sherpa Rust API and pinned Supertonic INT8 assets.
- Default rover dataflow activates edge TTS.
- Browser TTS reaches and plays only on selected rover.
- English and Vietnamese switch live through one resident model.
- Global runtime config applies to all active rovers and resets to English after restart.
- UI reflects authoritative desired and per-rover applied state.
- One rover process owns physical speaker output.
- 44.1 kHz Supertonic audio is handled at correct rate.
- Rover microphone audio is not forwarded during speaker playback/tail.
- Browser STT remains operational during rover playback.
- No self-triggered actuator command in adversarial tests.
- Pi 5 benchmark meets accepted RTF, stability, thermal, and vision-regression gates.
- Docker and native workflows provision/check the same pinned model.
- Documentation contains no active Kokoro/Piper architecture claims.

## Recommended Planning Sequence

1. Shared voice/config/status/PCM contracts.
2. Official Sherpa Supertonic `edge_voice` node with bounded worker and model tests.
3. Source-aware `audio_playback` and sample-rate handling.
4. Playback-state microphone suppression.
5. Rover dataflow and Zenoh config/status transport.
6. Web bridge runtime authority and global fleet fan-out semantics.
7. UI global controls and convergence status.
8. Kokoro/Piper retirement, model provisioning, Docker, and docs.
9. Pi 5 benchmark and full end-to-end safety gate.

## Unresolved Questions

- Confirm default speaker style after validating release SID mapping; recommended expected `M1`/SID 5.
- Choose deterministic TTS-versus-walkie arbitration: TTS priority or reject TTS while walkie playback is active.
- Confirm microphone suppression tail after hardware measurement; initial recommendation 400 ms.
- Confirm interactive maximum text length; current 1,000 characters may produce excessive blocking speech.
- Confirm whether expression tags are allowed from ordinary users or stripped to plain text.
- Legal review of OpenRAIL-M distribution obligations remains required.
