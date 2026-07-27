# Central Sherpa STT Brainstorm

Date: 2026-07-02
Status: agreed architecture; implementation not started

## Problem

- Remove unused rover `edge_speech_recognizer` placeholder.
- Replace Whisper in `orchestra/central_speech_recognizer` with official `sherpa-onnx` Rust API.
- Reuse rover S16LE audio already transported through Zenoh.
- Support true online Zipformer and VAD-backed offline Zipformer.
- Keep command interpretation on stronger Orchestra machine.
- Defer rover TTS acoustic feedback/echo handling as explicit backlog debt.

## Agreed Workflow

```text
rover audio_capture (16 kHz mono F32)
  -> audio_converter (S16LE)
  -> rover zenoh_bridge (versioned PCMF packet)
  -> Zenoh rover/{entity_id}/audio
  -> orchestra zenoh_bridge (validated raw S16LE + Dora metadata)
  -> central_speech_recognizer (S16LE -> normalized F32)
  -> command_parser / future command agent
  -> typed, validated command targeted to source rover
  -> orchestra zenoh_bridge
  -> rover controller
```

Existing `orchestra-bridge/audio_frame` can fan out to both `web_bridge` and `central_speech_recognizer`. No second rover audio stream or Zenoh topic needed.

## Converted Audio Feasibility

Yes. Current transport is directly usable for STT.

- `audio_capture` emits normalized `Float32Array`, 16 kHz, mono, 800 samples/frame.
- `audio_converter` quantizes each sample to signed 16-bit PCM and emits a one-element `BinaryArray` with complete metadata.
- Rover bridge wraps bytes in `PcmFramePacket`; Orchestra bridge validates and unwraps it.
- Orchestra bridge emits raw S16LE payload as `BinaryArray`; metadata includes entity ID, stream ID, frame ID, capture timestamp, sample rate, channels, sample count, and format.
- Sherpa online and offline Rust APIs both accept normalized `&[f32]` waveform samples.

Central conversion:

```text
i16 = i16::from_le_bytes([lo, hi])
f32 = i16 as f32 / 32768.0
```

Requirements:

- Reject odd byte counts and non-S16LE input.
- Validate payload length against `sample_count * channels * 2`.
- Require mono input initially.
- Accept 16 kHz directly; resample only if future metadata differs.
- Preserve stream/frame identity and reset recognizer state on stream changes or unrecoverable gaps.

S16LE quantization is normal ASR input, not a compressed codec. It halves current audio bandwidth versus F32, from roughly 512 Kbit/s to 256 Kbit/s before envelope overhead. Expected recognition impact negligible compared with microphone/noise/model effects.

## Central Recognizer Design

One Dora node, selectable backend:

| Mode | Sherpa components | Intended use |
|---|---|---|
| `online` | `OnlineRecognizer` + online Zipformer | English, partial hypotheses, low latency |
| `vad_offline` | `VoiceActivityDetector` + `OfflineRecognizer` | final utterances, Vietnamese, offline-model accuracy |

Suggested configuration:

```text
STT_MODE=online|vad_offline
STT_LANGUAGE=en|vi
STT_MODEL_DIR=...
STT_VAD_MODEL=...
STT_NUM_THREADS=...
STT_PROVIDER=cpu
```

Default runtime profile: `en-online`.

Expose only valid named profiles to clients rather than allowing arbitrary mode/model combinations:

- `en-online`: online English Zipformer; default after restart.
- `en-vad-offline`: Silero VAD plus offline English Zipformer.
- `vi-vad-offline`: Silero VAD plus offline Vietnamese Zipformer.

On a profile change, keep the current backend active while the requested model loads, report `loading`, switch at an utterance boundary, reset affected stream state, then report `ready`. Report `error` and retain the previous backend if loading fails. Lazily load non-default models; cache them after first successful load if Orchestra memory permits.

Online and offline Zipformer model files are not interchangeable. Provision separate model bundles.

For multiple active rovers, maintain independent state keyed by `entity_id` and `stream_id`:

- Online mode: one `OnlineStream` per rover; recognizer model can be shared.
- VAD/offline mode: one VAD/buffer state per rover; offline model can be shared.
- Never append interleaved fleet audio into a global buffer.

Output final `SpeechTranscription` with source rover identity and utterance/correlation ID. Sherpa results do not expose a directly comparable confidence value; make confidence optional or remove confidence-based gating rather than fabricate it.

## Web UI Runtime Control

UI repository: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`

Control remains local to the Orchestra Dora graph; it does not traverse rover Zenoh:

```text
Web UI --Socket.IO stt_control--> web_bridge/stt_control
  -> central_speech_recognizer/control

central_speech_recognizer/status -> web_bridge/stt_status
  --Socket.IO stt_status--> Web UI
```

Dataflow additions:

```yaml
central-speech-recognizer:
  inputs:
    audio_rover: orchestra-bridge/audio_frame
    control: web-bridge/stt_control
  outputs:
    - transcription
    - status

web-bridge:
  inputs:
    stt_status: central-speech-recognizer/status
  outputs:
    - stt_control
```

Suggested Socket.IO contracts:

```ts
type SttProfile = "en-online" | "en-vad-offline" | "vi-vad-offline";
type SttBackendState = "loading" | "ready" | "error";

interface SttControl {
  profile: SttProfile;
  entity_id?: string;
  request_id: string;
}

interface SttStatus {
  requested_profile: SttProfile;
  active_profile: SttProfile;
  state: SttBackendState;
  entity_id?: string;
  request_id?: string;
  error?: string;
}
```

Backend requirements:

- Add authenticated, validated, rate-limited `stt_control` handler to `common/web_bridge`.
- Allow only server-defined profile enum values; reject arbitrary model paths/provider values.
- Broadcast authoritative `stt_status` after connect and every transition.
- Rate-limit/cool down model changes to avoid repeated expensive loads.
- Keep `STT_DEFAULT_PROFILE=en-online` as restart source of truth; no database persistence initially.

UI touchpoints:

- Add STT contracts to `packages/shared/src/types/voice.ts` and Socket.IO event maps in `packages/shared/src/types/socket.ts`.
- Put a compact profile selector and backend status badge in `packages/ui/src/components/features/VoiceControls.tsx`.
- Disable selector while state is `loading`; preserve prior active selection on `error`.
- If configuration is per rover, pass `fleetStatus.selected_entity` from `RoboRoverControl.tsx` into `VoiceControls` and include it as `entity_id`.
- Show active profile/source rover in `TranscriptionDisplay.tsx`.
- Change UI transcription confidence to nullable/optional. Current UI assumes a number and always renders a percentage; show no confidence badge or `--` when Sherpa has none.
- Existing browser `voice_commands` worklet creates 4096-sample messages but does not attach a message handler that emits `voice_command_audio`; fix or remove that browser-microphone mode separately from rover-microphone STT.

UI validation:

- Default status renders `en-online` after connection.
- Selecting a profile emits one `stt_control` with request/source identity.
- `loading -> ready` updates UI only from server status, not optimistic local state.
- `loading -> error` preserves prior active profile and displays actionable failure.
- Reconnect obtains authoritative active status.
- Invalid profile payloads are rejected server-side.
- Transcriptions without confidence render without `NaN%` or exceptions.

## Command Processing

Keep `command_parser` on Orchestra.

Recommended evolution:

```text
transcription
  -> deterministic parser for safety-critical/simple commands
  -> unhandled text to optional AI interpreter
  -> schema and policy validator
  -> typed command
```

AI output must never directly control actuators. Validate command type, parameters, limits, authorization, and target.

Current bridge routes parser outputs to UI-selected rover. Rover-origin speech must instead route back to the transcription's source `entity_id`; otherwise rover A speech can control selected rover B. Use an explicit targeted command envelope or equivalent preserved routing context.

## Edge Recognizer Removal Scope

Remove during implementation:

- `rover-kiwi/edge_speech_recognizer/`
- Workspace member and regenerated `Cargo.lock` entry
- `docker/Cargo.rover.toml` member
- Rover Dockerfile dependency scaffolding, build, and binary copy
- Commented nodes in both rover dataflow files
- Architecture/documentation references to future edge STT

## Whisper Retirement

Remove `whisper-rs`, GGML runtime configuration, model checks/download wiring, and Whisper documentation only after both Sherpa modes pass recorded-audio and live-stream validation. Do not keep both engines indefinitely without a concrete fallback requirement.

## Rover TTS Backlog

Technical debt: `rover-kiwi/sherpa_tts` plays through the rover speaker while rover microphone capture can remain active. There is no playback-state output, recognizer suppression, or acoustic echo cancellation.

Backlog enhancement:

- Migrate TTS to official pinned `sherpa-onnx` API/runtime family.
- Add `playback_state` output (`started`, `finished`, correlation ID).
- Feed state into whichever STT path is active.
- Reset/discard recognition audio during playback plus 300-500 ms tail.
- Add hardware loop test; later consider AEC only if barge-in required.
- Keep lightweight VITS default. Benchmark Kokoro on Raspberry Pi 5 before considering edge use.

This debt does not block centralized STT because the Orchestra recognizer still receives the rover microphone and can hear rover-speaker playback. It should remain disabled or its responses designed carefully until suppression exists.

## Risks

- Voice command execution now depends on rover-Orchestra network availability.
- Multi-rover audio interleaves on one Dora output unless central separates state by metadata.
- Zenoh gaps can damage an utterance; track sequence gaps and reset when necessary.
- Vietnamese is VAD-backed offline recognition, not true token-streaming recognition.
- Continuous fleet-wide STT can scale CPU roughly with active speech sources.
- Source entity context can be lost across parser/agent boundaries unless explicit.

## Validation Criteria

- Unit-test S16LE boundary conversion and payload validation.
- Replay the same corpus through original F32 and S16LE transport; command results equivalent.
- Verify 16 kHz mono 800-sample frames reach central without ordering regressions.
- Verify independent simultaneous rover streams never mix samples or transcripts.
- Online mode emits partials and one final result per endpoint.
- VAD/offline mode emits one final result per detected utterance.
- Benchmark RTF, endpoint-to-final latency, CPU, and memory on Orchestra.
- Test English command corpus; test Vietnamese corpus before enabling `vi` mode.
- Verify every parsed command and TTS reply targets transcription source rover.

## Next Steps

1. Create detailed implementation plan.
2. Add shared S16LE decoder/session contract tests.
3. Implement and validate online English backend.
4. Implement VAD/offline backend and Vietnamese model.
5. Add `stt_control`/`stt_status` to web bridge and the external control UI.
6. Remove Whisper after parity validation.
7. Record rover TTS echo work in project backlog; implement separately.

## Unresolved Questions

- Keep browser `voice_command_audio` as a second central input, or scope central STT to rover audio only?
- Apply the UI-selected profile globally or per selected rover? Per-rover is recommended for fleet correctness.
- Maximum simultaneously recognized rover streams?
- Is the AI interpreter part of this implementation or a later phase?
