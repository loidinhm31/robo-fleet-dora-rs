# Phase 09 — Manual Rover KWS and WakeAck Acceptance

## Context links

- Parent: [plan.md](./plan.md); implementation: [Phase 06](./phase-06-rover-kws-and-wake-ack.md).
- Scope: user-operated physical Rover acceptance for the already implemented local `Hey Kiwi` KWS and `I am on` WakeAck path.

## Overview

- Priority: P1; owner: user/operator; status: Pending.
- This phase does not authorize source changes. It records manual test evidence and any follow-up defects.

## Preconditions

- Target Rover runs the Phase 06 build with the verified KWS model cache and WakeAck asset.
- Start Orchestra before Rover. Verify the Rover is in `IdleListening`, browser audio is stopped, and no TTS/walkie playback is active.
- Record target hardware, image/commit identifier, model checksum, environment, and test location/noise conditions.

## Manual checklist

- [ ] From `IdleListening`, say `Hey Kiwi` at normal distance/volume. Verify exactly one bounded KWS demand, transition to `NormalRover`, and one audible `I am on` WakeAck.
- [ ] Confirm no controller, tracking, recorder, camera, or media command is emitted by the wake path.
- [ ] Repeat with browser audio stopped. Verify the local microphone remains available to KWS.
- [ ] During WakeAck, verify capture suppression prevents self-triggering; repeat after the 400 ms tail expires.
- [ ] Exercise speaker/motor/background-noise conditions. Record false accepts, false rejects, and any wake repetitions.
- [ ] Repeat across reboot/reconnect and split/direct Rover dataflows.
- [ ] Collect at least 30 successful trials and record WakeAck latency plus `NormalRover` readiness latency (p50/p95/p99 where available).
- [ ] Record CPU/RSS/thermal observations during IdleListening and wake transitions.

## Acceptance targets

- WakeAck p95 is below 1.5 seconds.
- `NormalRover` Ready p95 is below 5 seconds.
- No WakeAck self-trigger or controller-path output occurs.
- False-accept/false-reject behavior is acceptable for the target deployment noise envelope.

## Evidence record

For each run, record date/time, Rover hardware, image/commit, KWS model checksum,
dataflow mode, audio route, conditions, trial count, latency percentiles, failure
counts, and logs/screenshots. Link any defects or follow-up implementation plan.

## Completion

Mark this phase complete only after the operator accepts the recorded evidence.
Failures remain evidence, not a reason to alter Phase 06 implementation status.
