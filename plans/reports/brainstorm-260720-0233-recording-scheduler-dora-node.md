# Brainstorm: Orchestra recording scheduler Dora node

## Problem and agreed requirements

Today scheduled media control does not exist. The linked `robo-control-app` can manually start/stop sessions, while `common/web_bridge` aggregates browser and manual-recording demand and `orchestra/media_recorder` owns FFmpeg/MP4 output. A new Orchestra Dora-RS node is requested, plus a scheduler tab in the linked app.

Agreed v1 behavior:

- Schedule continuous MP4 recording windows, not media-only power windows.
- At window start, acquire rover camera/JPEG/audio demand and start one owned `media-recorder` session.
- At window end, stop only the session owned by that occurrence; never interrupt an unrelated manual/browser demand.
- Support one-time, daily, and weekly recurrence.
- Calculate recurrence in an IANA timezone; keep wire/persisted instants as Unix milliseconds, matching existing recording timestamps.
- Retry while the window remains active if the rover, recorder, or bridge is temporarily unavailable.
- Coalesce overlapping schedules for one rover so one active recording covers the union of windows.
- Manual app commands override a schedule immediately; the schedule resumes at its next transition unless explicitly disabled.
- Persist schedules and occurrence state so restart/reconnect does not duplicate or lose a recording.

Out of scope for v1: event-triggered/pre-roll recording, monthly/holiday calendar rules, a second FFmpeg implementation, and a new rover-side encoder.

## Current implementation surface

- `orchestra/media_recorder`: Dora node, FFmpeg/session owner, existing `recording_session_*` command/status protocol.
- `common/web_bridge`: authenticated Socket.IO, in-memory `MediaDemandRegistry`, browser/manual demand aggregation, and recording command queues.
- `orchestra/zenoh_bridge`: converts aggregate targeted media transitions to rover camera/stream/audio commands.
- `rover-kiwi/kornia_capture` and `rover-kiwi/audio_capture`: consume camera/stream/audio controls and produce source frames.
- `orchestra/orchestra-dataflow.yml` and workspace `Cargo.toml`: no scheduler node/member currently.
- Linked UI: `RoboRoverControl.tsx` has CONTROL and RECORDINGS tabs. `packages/shared/src/types/recording.ts`, `packages/shared/src/types/socket.ts`, `use-scheduler-device-overrides.ts`, and `scheduler-media-commands.ts` contain an unused device-override draft, but no schedule CRUD/editor.
- Historical commits `92c2fb9`, `0855a01`, `caa47c7`, and `992afac` contain useful Mongo/timezone/lease ideas, but their device-lease and recording contracts do not match current HEAD and should not be cherry-picked wholesale.

## Options considered

### 1. Narrow scheduler through web-bridge — recommended

Add `orchestra/recording_scheduler` as a control-plane-only Dora node. It owns schedule persistence, recurrence evaluation, occurrence IDs, restart reconciliation, and status. Authenticated schedule commands enter through `web-bridge`; scheduler emits an internal scheduled-session intent. A web-bridge recording coordinator translates that intent into the existing demand registry plus `recording_session_command` path. Recorder status/results return to the scheduler and UI.

Pros: one media-demand authority; reuses proven FFmpeg/session code; scheduled OFF cannot stop browser/manual demand; clean restart/idempotency boundaries; no duplicate encoder or direct rover authority.

Cons: adds scheduler/web-bridge protocol and origin-aware result routing; requires Mongo schema/indexes and fake-clock tests.

### 2. Scheduler dispatches recorder and rover controls directly

The scheduler sends `recording_session_command` to `media-recorder` and separate camera/audio controls to the bridge/Zenoh.

Pros: fewer web-bridge changes initially.

Cons: two authorities can race; partial start/stop failures are hard to reconcile; scheduled OFF can cut another consumer; restart behavior is ambiguous. Prototype-only, not suitable for production.

### 3. Put timers inside web-bridge or media-recorder

Keep scheduling in an existing process and do not add a standalone Dora node.

Pros: least wiring.

Cons: violates the requested node boundary; mixes authentication, wall-clock persistence, demand arbitration, and encoding; weaker failure isolation and restart testing.

## Recommended contract and flow

```text
Scheduler tab
   ↕ authenticated schedule CRUD/status
web-bridge ── Dora command/status ── recording-scheduler
   ↕ scheduled owned-session intent
MediaDemandRegistry ── targeted_media_control ── orchestra-bridge ── rover capture/audio
   ↕ recording_session_command
media-recorder (FFmpeg/MP4) ── status/result ── web-bridge ── scheduler + UI
```

Schedule document should contain: UUID, revision/CAS token, entity ID, title, enabled flag, recurrence kind, local start, duration, IANA timezone, validated relative output-directory template, audit actor/timestamps, and deterministic occurrence ID. Occurrence state should include planned/start/end epoch ms, active/started/stopped/missed/failed status, retry metadata, owned recording ID, and last error.

Use one protocol family for schedule CRUD/status (for example `recording_schedule_*`) and keep it distinct from existing manual `recording_session_*` events. Every command/result needs request ID, schedule ID, occurrence ID, entity ID, revision, and an authoritative status snapshot. Reject stale revisions with conflict responses. On restart, reconcile persisted active occurrences against recorder status before issuing a start; duplicate starts must be idempotent.

Precedence and ownership rules:

- Manual start/stop is an immediate override.
- A scheduled stop releases only that occurrence's demand/session ownership.
- The aggregate demand remains on while browser or another recording consumer still needs it.
- Overlapping occurrences for one rover share one active session; stop after the last owning window ends.
- Retry boundedly while the window is open; surface offline, storage, and encoder failures to the UI.

## Touchpoints

- Add shared Rust schedule/occurrence/status/command types in `robo_rover_lib` and matching TypeScript types.
- Add `orchestra/recording_scheduler` crate, workspace member, tests, and dataflow node/ports.
- Extend `common/web_bridge` with authenticated schedule CRUD/status, scheduler Dora queues, a recording coordinator that marks demand origin, and restart reconciliation hooks.
- Preserve existing `media_recorder` session protocol; add ownership metadata only where needed for idempotent scheduled stop.
- Keep rover/Zenoh control paths behind the existing aggregate demand unless later work introduces lease acknowledgements.
- Add a Scheduler tab/page and schedule editor/list/status UI in `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`; reuse existing override/status types only after aligning them with the new schedule contract.
- Add Mongo collections/indexes for schedules and occurrences, role checks, audit fields, and migration/retention guidance.

## Risks and mitigations

- Dual media authorities: route all scheduled demand through `MediaDemandRegistry`; never emit direct scheduled OFF commands.
- DST ambiguity/gaps: resolve with explicit deterministic policy and fake-clock tests; display the resolved next run.
- Crash between start and acknowledgement: deterministic occurrence IDs, revisioned state, and recorder-status reconciliation.
- Rover offline/full disk/FFmpeg failure: bounded retry, explicit occurrence state, and visible error detail.
- Multi-rover leakage: entity-scoped keys and tests that interleave two rovers.
- Existing draft UI contracts may encode obsolete lease semantics: treat them as compatibility clues, not as the source of truth.

## Acceptance and validation

- CRUD is authenticated, role-checked, revision-safe, and persists across scheduler/web-bridge restart.
- A one-time and recurring schedule starts/stops exactly once at expected Unix-ms instants; DST gap/fold tests are deterministic.
- Start acquires rover media and creates a non-empty MP4; end stops only the owned occurrence.
- Manual/browser demand remains active after scheduled stop; overlapping schedules do not stop early.
- Offline rover and recorder failures retry while active and become visible without busy-looping.
- Two rovers remain isolated; recorder concurrency and queue limits are respected.
- UI shows enabled/disabled state, next run, active occurrence, authoritative status, missed/retry/error state, and manual override outcome.
- Rust unit/integration tests cover recurrence, overlap/ownership, stale revisions, idempotent restart, and demand aggregation. Linked app unit and Playwright tests cover create/edit/disable/delete, tab reconnect, and live status transitions.

## Next steps

1. Convert this decision into an implementation plan with phase gates (shared contract → scheduler core → bridge/recorder integration → UI → end-to-end verification).
2. Decide the exact Mongo deployment/retention policy and schedule-editor roles.
3. Define the explicit DST resolution rule (recommended: earliest instant on fall-back; shift spring-gap times forward) and retry/backoff limits.
4. Confirm whether manual stop should suppress only the current occurrence or disable the schedule (recommended: suppress current occurrence; leave schedule enabled).

