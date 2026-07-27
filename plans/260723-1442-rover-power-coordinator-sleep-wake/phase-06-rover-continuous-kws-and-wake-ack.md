# Phase 06 — Rover Continuous KWS and WakeAck

## Context links

- Parent: [plan.md](./plan.md)
- Design: [UI and Local Voice Wake](../../docs/power-coordinator-architecture.md#ui-and-local-voice-wake)
- Input: [voice/resource research](./research/researcher-02-voice-resource-ui.md)
- Dependencies: Phases 01–04 local demands, profile readiness, authority.

## Overview

- Date: 2026-07-24
- Description: add continuous local keyword spotting and deterministic prerecorded wake acknowledgment without local command interpretation.
- Priority: P1
- Implementation status: Pending
- Review status: Pending

## Key Insights

- KWS identifies phrase; VAD alone does not. Continuous one-thread KWS is v1.
- Reuse the single microphone owner and playback suppression path; do not open a second device.
- Wake phrase creates intent only. Motion, arm, tracking, and recording remain stopped until fresh commands.

## Requirements

- KWS active only in `IdleListening`; `Dormant` has no voice wake.
- Consume local 16 kHz mono capture frames; full ASR/TTS workers remain quiesced.
- Debounce/reset one keyword result into one bounded local `NormalRover` demand/transition.
- On aggregate playback readiness, play bundled PCM `"I am on"` once per transition as source `WakeAck`.
- Suppress KWS during playback and existing 400 ms tail; never self-trigger.
- Target gates: first WakeAck sample <1.5 s p95; NormalRover Ready <5 s p95; benchmark noisy false accepts/rejects.

## Architecture

`audio_capture -> voice-wake(KWS) -> Rover coordinator local demand -> lifecycle wake -> coordinator wake_ack_trigger -> voice-wake bundled PCM -> audio_playback(WakeAck)`. Playback state closes capture/KWS during audio and tail. No KWS output connects to command parser/controllers.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/voice-wake/Cargo.toml`.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/voice-wake/src/{main.rs,config.rs,kws.rs,debounce.rs,wake-ack.rs}`.
- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/voice-wake/assets/i-am-on.pcm`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_capture/src/{main.rs,capture_gate.rs}` — local KWS branch/profile gate.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/rover-kiwi/audio_playback/src/{protocol.rs,state.rs,arbiter.rs,runtime.rs}` — `WakeAck` source/priority/result.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/models/scripts/{model-manifest.sh,setup-models.sh}` and `/mnt/data/ws/sharing/robo-fleet-dora-rs/models/README.md` — pinned checksum KWS assets.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/Cargo.toml`, Rover dataflows, and `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/Dockerfile.rover-kiwi`.

## Implementation Steps

1. Select and checksum-pin KWS model/tokens/keyword file; validate fixed startup paths and one thread by default.
2. Feed voice-wake from existing capture output in 16 kHz mono chunks; never open microphone directly.
3. Implement Sherpa stream loop (`accept_waveform`, ready/decode/result/reset), exact allowed keyword match, debounce/cooldown, and metrics.
4. Emit bounded local KWS demand with transition-correlated deterministic ID; duplicate detections renew/ignore, never create parallel wake.
5. Add `WakeAck` playback source and bundled PCM metadata; coordinator triggers only after audio playback target reports Ready.
6. Reuse playback state to suppress capture/KWS during active playback + 400 ms tail; require fresh post-wake commands for every actuator/media action.
7. Wire `IdleListening`/`Dormant` lifecycle gates in split and direct Rover dataflows; leave full edge voice and central STT asleep until demanded.
8. Test phrase/no-phrase, duplicate result, noisy clips, playback feedback, partitioned local wake, restart, and no controller output.

## Todo list

- [ ] Freeze KWS model/phrase/checksums.
- [ ] Add voice-wake node using existing capture.
- [ ] Add bounded local demand/debounce.
- [ ] Add distinct prerecorded WakeAck path.
- [ ] Add suppression, safety, and offline tests.

## Success Criteria

- Exact phrase wakes Rover with Orchestra/Zenoh absent; status later reconciles.
- KWS event produces zero movement/arm/tracking/recording commands.
- `"I am on"` plays once only after playback readiness and meets <1.5 s p95.
- Normal profile meets <5 s p95; continuous KWS CPU/RSS/noise metrics captured.
- KWS cannot hear its own WakeAck or replay it after restart.

## Risk Assessment

- Rover/motor/TV noise false wakes: target corpus, threshold tuning, cooldown.
- Continuous KWS exceeds idle budget: benchmark first; VAD gating is deferred optimization only if needed.
- Audio ownership conflict: enforce audio_capture as sole microphone owner.

## Security Considerations

- Fixed checksum-pinned keyword assets; no runtime/browser path or keyword injection.
- Do not persist raw wake audio; log bounded keyword ID/result only.
- Wake demand TTL/capacity and source are hard-coded/validated locally.

## Next steps

Phase 08 freezes target noise thresholds and decides whether continuous KWS passes. Exact phrase/language/model remain unresolved until stakeholder and hardware corpus approval.

