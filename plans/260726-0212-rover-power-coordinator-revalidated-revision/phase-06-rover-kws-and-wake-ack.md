# Phase 06 — Rover KWS and WakeAck

## Context links

- Parent: [plan.md](./plan.md); design: [local voice wake](../../docs/power-coordinator-architecture.md#ui-and-local-voice-wake).
- Evidence: [voice/UI research](./research/researcher-02-voice-ui-history-revision.md), [scheduler/voice scout](./scout/scout-02-scheduler-voice-ui.md).
- Dependencies: Phases 01–04.

## Overview

- Date: 2026-07-26; priority: P1; implementation accepted 2026-07-28; manual target acceptance is tracked separately in [Phase 09](./phase-09-manual-rover-kws-and-wakeack-acceptance.md).
- Add one low-cost local KWS worker for `Hey Kiwi` and one output-only bundled `I am on` acknowledgement after readiness.

## Key Insights

- `audio_capture` is the only microphone owner and already honors playback suppression plus 400 ms tail.
- KWS is intent only. It creates one bounded local demand while disconnected, never a transcript or actuator/media/recording command.

## Requirements

- KWS is active only in `IdleListening`, disabled in Dormant, uses one default thread, and consumes existing 16 kHz mono capture frames.
- Wake detection is debounced/reset and maps to one transition-correlated bounded `NormalRover` demand.
- WakeAck uses a pinned bundled PCM asset and a distinct `PlaybackSource::WakeAck`, triggered once only after playback/profile readiness.
- Playback suppression prevents self-trigger; full ASR/TTS stays quiesced until independently demanded.

## Architecture

`audio_capture -> voice-wake -> Rover coordinator demand -> lifecycle/profile Ready -> WakeAck trigger -> audio_playback`. No voice-wake edge connects to command parser, controller, tracking, recorder, or media command ports.

## Delivered implementation

- Added the `voice_wake` Dora node. It uses one Sherpa-ONNX Zipformer KWS
  worker/thread and consumes the `audio_capture` `kws_audio` branch; it never
  opens a second microphone or emits transcript/actuator/media/recording data.
- The compiled BPE keyword contract is `Hey Kiwi`; detections are reset after
  every decoded result and debounced for 10 seconds. Each accepted wake emits
  one deterministic UUID-v5 `PowerDemandSource::Kws` demand for
  `NormalRover`, with a five-minute TTL and a 60-second command TTL.
- `IdleListening` keeps the shared capture device alive while browser audio is
  stopped; playback suppression and the existing 400 ms tail gate both audio
  branches, preventing WakeAck self-triggering. `Dormant` quiesces the node.
- Added the output-only `PlaybackSource::WakeAck` path. The bundled
  `i-am-on.pcm` asset is f32 mono at 44.1 kHz, resampled by the playback
  arbiter, and emitted once only when the accepted demand reaches Active /
  `NormalRover` and playback reports Idle.
- Split and direct Rover dataflows, Docker packaging, and model setup now wire
  the node and checksum-pinned model bundle.

## Validation performed

`cargo test -p voice_wake` passes (7 tests), covering model-token contract,
silence rejection, debounce, bounded demand, one-shot readiness gating, and
PCM asset validity. Audio capture and playback unit tests also include KWS
branch, suppression/tail, WakeAck source, bounded-buffer, and resampler cases.

## Deferred manual acceptance

The user-operated physical-Rover checks are intentionally deferred to
[Phase 09](./phase-09-manual-rover-kws-and-wakeack-acceptance.md). The amd64
workstation/Docker path proves packaging and topology, not ARM acoustic or
performance acceptance.

## Related code files

- Implemented `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/voice-wake/{Cargo.toml,src/main.rs,src/config.rs,src/controller.rs,src/kws.rs,src/debounce.rs,src/wake_ack.rs,assets/i-am-on.pcm}`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_capture/src/{main.rs,capture_gate.rs}` and `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_playback/src/{protocol.rs,state.rs,arbiter.rs,runtime.rs}`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/models/scripts/{model-manifest.sh,setup-models.sh}`, `models/README.md`, Rover dataflows, Dockerfile, and workspace Cargo manifest.

## Implementation Steps

1. Select checksum-pinned KWS model/token assets that recognize `Hey Kiwi`; validate fixed deployment-owned paths.
2. Branch existing capture frames to one KWS stream; never open another device.
3. Implement decode/reset, phrase match, cooldown/debounce, metrics, and deterministic local demand ID.
4. Add WakeAck source/PCM and trigger only after aggregate playback readiness; reuse capture suppression and tail.
5. Wire profile gates in split/direct dataflows and test offline wake/reconnect reconciliation.
6. Prove phrase/no-phrase, duplicate, Dormant, playback self-hearing, restart, noise, and zero-controller-output behavior.

## Todo list

- [x] Pin model/checksum and phrase contract.
- [x] Add voice-wake node and bounded demand.
- [x] Add WakeAck playback path/suppression tests.
- [ ] Benchmark target hardware/noise.

## Success Criteria

- `Hey Kiwi` wakes locally without Orchestra/Zenoh, yet emits no actuator/media/recording action.
- `I am on` plays once after readiness, KWS cannot hear it, and p95 targets pass on physical Rover.
- Dormant has no voice wake and NormalRover readiness stays below 5 s p95.

## Risk Assessment

- Motor/noise false wakes and continuous CPU use require corpus benchmarks; VAD gating remains deferred unless needed.

## Security Considerations

- Assets are checksum-pinned. Raw wake audio is neither persisted nor sent to Mongo. Demand source/TTL are local and bounded.

## Next steps

Phase 08 approves model, thresholds, and acoustic limits from target evidence.
