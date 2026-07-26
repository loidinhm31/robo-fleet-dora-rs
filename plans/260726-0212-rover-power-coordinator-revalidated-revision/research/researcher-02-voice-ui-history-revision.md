# Research Report: Voice wake, power UI, routing, and durable history (revalidated)

> Historical pre-implementation evidence. Commits `ff6624e`, `a9ba1c4`, and
> `a1cbc38` added power contracts, coordinators, journal/projector code, and
> dataflow entries. Revalidate every code claim against `HEAD` and use the
> cutoff audit as the current baseline.

Timestamp: 2026-07-26 (Asia/Ho_Chi_Minh)

## Executive summary

As of the pre-`ff6624e` baseline, the prior sleep/wake plan was directionally
correct but entirely pending. At that historical point, code had reusable
lifecycle, auth/rate-limit, entity-target, Socket.IO reconnect, and
playback-suppression primitives, but no power coordinator, `voice-wake` crate,
WakeAck source, power history/current-state projection, or power UI.
Revalidation confirmed the revised product decisions: input phrase is exactly
**“Hey Kiwi”**; **“I am on”** is output-only prerecorded WakeAck; all
authenticated users may set persistent Awake/Sleep, subject to exact entity
validation, rate limits, and server-derived audit actor; UI activity demand is
2 minutes; offline Rover KWS may wake the coordinator but must not issue
actuator/media actions.

## Evidence / exact touchpoints

- `rover-kiwi/zenoh_bridge/src/main.rs:291-320,757-774` currently routes only lifecycle command, wake-lease, and query topics (`rover/{entity}/cmd/lifecycle*`); no dedicated power topics, snapshots, events, or local coordinator reconciliation. `orchestra/zenoh_bridge/src/main.rs:157-174,540-576,1043-1061` mirrors lifecycle status/result/capabilities and validates active entity, but no power channel. Keep power routing separate from movement/arm/media and preserve direct mode parity.
- `robo_rover_lib/src/types/lifecycle_types/*` provides epoch/revision/target validation and stale/duplicate fencing. Reuse contract style, but add power policy/effective profile/demand/current-state/event contracts rather than overloading `LifecycleDesiredState`.
- `common/web_bridge/src/lifecycle-socket.rs:20-118` caches status/capabilities, bounds pending commands (128), checks authenticated session, command limiter, expiry, active target, advertised capability, and derives audit actor (`session_registry.audit_actor`). This is the template for `power_policy_set`/`power_wake`; persistent policy authorization should be “any authenticated user” (not role-gated), while server derives actor and exact target.
- `common/web_bridge/src/security.rs:1-110` has auth/IP/command rate limiters and Mongo user/index helpers. Add a distinct power-policy limiter and audit records; do not trust browser actor/entity/epoch/source fields.
- `rover-kiwi/audio_capture/src/capture_gate.rs:8-120` is the sole mic gate, sequence-fences playback state, suppresses capture while active and for a fixed 400 ms tail. Feed KWS from this owner; extend gate/profile semantics so KWS is enabled in `IdleListening`, disabled in `Dormant`, and suppressed during WakeAck playback/tail.
- `rover-kiwi/audio_playback/src/state.rs:1-110` reports only `PlaybackSource::Tts`/`Walkie` and requires TTS command IDs. Add distinct `WakeAck` source/reason and bundled PCM path; trigger only after aggregate NormalRover/playback readiness. Never route WakeAck through general TTS/NLU.
- `rover-kiwi/edge_voice/src/runtime.rs` (per prior plan evidence) already has lifecycle admission, worker stop timeout, status/metrics. Gate full ASR/TTS by profile; KWS must be a separate low-cost worker and must not emit parser/controller commands.
- UI is external sibling repo `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`. `packages/ui/src/components/pages/RoboRoverControl.tsx` centralizes Socket.IO lifecycle and bounded transcription history; `packages/ui/src/hooks/use-recording-schedule-events.ts` demonstrates authenticated subscription, exact entity filtering, stale request IDs, reset on disconnect, and readiness degradation. Reuse for power snapshot/status/transition/history events. No power types/hooks/components currently exist.

## Required plan deltas

1. Phase 06: freeze exact keyword contract to `Hey Kiwi`; treat `I am on` strictly as output-only asset/source. Add offline tests proving keyword event produces only bounded local wake demand, zero movement/arm/tracking/recording/media commands, and no wake after `Dormant`. Define KWS wake behavior while Zenoh is disconnected and reconciliation on reconnect.
2. Phase 04: add dedicated topics `rover/{entity_id}/power/v1/{command,status,snapshot,event}` plus direct-mode ports. Rover-local wake may advance a local authority epoch/demand while disconnected; Orchestra must consume a fresh snapshot before issuing a strictly newer takeover epoch.
3. Phase 07: state explicitly that every authenticated session may set persistent Awake/Sleep; exact active entity is server-pinned/validated. Add policy command rate limit, duplicate-changed rejection, audit actor, and audit of unauthorized/cross-entity/expired/rate-limited attempts. UI wake/activity creates a bounded 2-minute demand; reconnect/disconnect cleanup and server sweep required.
4. Phase 03/07: implement durable append-only power event journal plus Mongo `power_current_state` (no TTL) and 90-day `power_lifecycle_events` projection. Live coordinator `(epoch,sequence)` outranks cold history; old projection cannot regress current state. Cursor pagination and event-ID dedupe required in UI.
5. UI contract: add `power_policy_snapshot`, `power_status`, `power_transition`, `power_history_query/result` (or equivalent consistently named events), display policy separately from effective profile/readiness, and label cold data Historical/Stale. Disable policy mutation/Wake on disconnected or stale authority; no optimistic Ready.

## Risks / acceptance gates

- False accepts, CPU/RSS, thermal/power, and p50/p95 wake latency need target-rover corpus tests; continuous KWS is not accepted on workstation-only evidence.
- Playback feedback can self-trigger unless capture suppression includes WakeAck source and 400 ms tail, with producer/sequence fencing.
- Local wake + reconnect can split-brain unless snapshot-first reconciliation and monotonic epoch/sequence rules are mandatory.
- Activity-demand leaks can keep rovers awake; enforce 120-second TTL, disconnect cleanup, bounded queue, and sweep.
- History replay/out-of-order events must be idempotent and non-regressing; never persist raw wake audio, model paths, tokens, or native errors.

## Unresolved questions

- Sherpa model/token/checksum, language, threshold, and false-accept/false-reject limits for `Hey Kiwi` remain unapproved.
- Exact policy command rate values and whether safety/maintenance actors need a separate override class are not frozen.
- Mongo deployment/availability and local journal capacity/high-water behavior need operational values.
- Definition of “stale” authority and Orchestra takeover timeout must come from Phase 08 fault measurements.
