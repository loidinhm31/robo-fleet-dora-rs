# Rover Workload Sleep/Wake Power Coordinator

## Problem

Robo-Fleet needs coordinated workload sleep/wake across Orchestra and Rover-Kiwi:

- save CPU, memory, device, and model resources during inactivity;
- preserve safety, lifecycle authority, networking, scheduling, and observability;
- wake from UI, future recording occurrence, or local voice phrase;
- let Rover wake locally without Orchestra, then reconcile when Zenoh returns;
- remain workload-level v1; no OS suspend, container stop, or host power management.

## Confirmed Scope

In:

- new power-policy coordinator above existing lifecycle managers;
- `Awake`, `Auto`, and `Sleep` policy;
- dependency-aware workload profiles;
- semantic wake demands plus resource confirmation;
- scheduler wake reservations and measured prewarm;
- Rover-local continuous KWS;
- authenticated web UI wake;
- prerecorded local wake acknowledgment;
- Orchestra authority reconciliation;
- resource and transition telemetry.

Out:

- OS suspend/Wake-on-LAN;
- GPIO wake;
- general offline Rover STT/NLU;
- KWS-triggered actuator commands;
- generic process/container kill;
- ML-based power-policy scoring.

## Existing Reusable Architecture

### Recording scheduler

Reuse:

- deterministic occurrence/group IDs;
- durable outbox and replay;
- generation fencing;
- owner/refcount model;
- feedback dedupe;
- restart reconciliation;
- recurrence horizon.

Do not make scheduler the power authority. It remains source of truth for schedules and occurrence state.

Gap: scheduler currently emits recording intents only. It does not connect schedule timing to lifecycle wake.

### Lifecycle manager

Reuse:

- exact target validation;
- manager epoch and revision CAS;
- idempotent request handling;
- admission result separate from applied status;
- transition timeout;
- stale/late status rejection;
- Orchestra-to-Rover Zenoh relay;
- node-owned safe teardown/resume.

Keep lifecycle manager as per-target executor. Do not embed whole-system policy or dependency graph into it.

Current wake leases are prototype-quality, not safe to wire directly:

- accepted lease does not reliably emit workload resume;
- user desired state conflicts with temporary effective wake;
- timeout can be refreshed during reconciliation;
- no group barrier/dependency ordering;
- incomplete capability, priority, source, and revocation semantics.

Coordinator should issue normal fenced lifecycle commands. Repair/replace wake-lease internals later as execution detail.

### Sherpa voice primitives

- Silero VAD detects speech, not wake phrase.
- KWS identifies configured wake phrase.
- VAD-gated KWS may reduce idle inference but adds pre-roll, tuning, and false-reject risk.
- v1 decision: continuous KWS first. Benchmark on target hardware; add VAD gating only if idle cost justifies it.
- full central STT remains asleep while idle.

## Evaluated Approaches

### Extend lifecycle manager into global policy engine

Pros:

- fewer nodes;
- reuses existing contracts.

Cons:

- mixes policy with exact-target execution;
- dependency, demand, profile, and scheduling logic make manager tangled;
- harder local/remote authority reasoning.

Rejected.

### Put power policy in recording scheduler

Pros:

- schedule timing already durable;
- smallest path for recording wake.

Cons:

- recording-centric authority cannot correctly arbitrate UI, voice, safety, media, and maintenance;
- duplicates lifecycle target/dependency knowledge;
- violates single responsibility.

Rejected.

### New hierarchical power coordinator

Pros:

- one power-policy source of truth;
- scheduler/UI/voice become demand producers;
- lifecycle managers stay small target executors;
- supports profiles, dependency barriers, priorities, hysteresis, and aggregate status;
- deployable on both Orchestra and Rover for local autonomy.

Cons:

- new contracts/node/state machine;
- authority reconciliation required;
- more integration and failure testing.

Selected.

## Final Architecture

### Policy versus effective state

Policy:

- `Awake`: hold normal workload profile active.
- `Auto`: derive minimal profile from active demands; autosleep after idle rules.
- `Sleep`: quiesce normal workloads; UI or accepted scheduler reservation may cause bounded wake.

Effective coordinator state:

- `Active`
- `IdlePending`
- `Quiescing`
- `IdleListening`
- `Dormant`
- `Prewarming`
- `Waking`
- `Degraded`
- `Failed`

Auto transitions never rewrite policy to Sleep. Policy and effective state remain separate.

### Always-on control spine

- web bridge;
- Orchestra/Rover Zenoh bridges;
- power coordinator;
- lifecycle managers;
- recording scheduler;
- resource monitors;
- rover/arm controllers;
- watchdog and emergency-stop path.

`IdleListening` additionally keeps:

- Rover microphone capture;
- local continuous KWS;
- minimum audio path required for wake detection.

`Dormant` disables voice wake. Current v1 has authenticated remote UI wake only.

### Workload profiles

- `NormalRover`: normal camera/audio/voice workloads; ML stays lazy unless tracking requested.
- `IdleListening`: control spine + microphone + KWS.
- `ScheduledCapture`: exact camera/audio/encoder/bridge/recorder dependencies only.
- `Dormant`: control spine only.
- `OrchestraSpeech`: central STT/command path when Orchestra voice control is required.

Profile expansion uses static, reviewed dependency phases. No dynamic graph learning.

### Demand contract

Conceptual:

```text
WakeDemand {
  demand_id,
  source,
  entity_id,
  profile,
  priority,
  issued_at_ms,
  not_before_ms,
  expires_at_ms,
  authority_epoch
}
```

Sources:

- authenticated UI;
- recording scheduler;
- local KWS;
- manual recording/media demand;
- safety/maintenance.

Deterministic IDs and idempotent replay required. Transient demands never survive restart.

### Automatic idle algorithm

Auto may enter low power only when:

```text
no active demand
AND no protected operation/session
AND targets report idle/quiesce-capable
AND fresh per-domain CPU remains below configured threshold
AND five-minute idle grace expires
```

Rules:

- semantic demand/activity is authority;
- CPU is confirmation only;
- memory is outcome telemetry, never idleness input;
- stale resource data blocks an automatic transition;
- any new demand cancels `IdlePending`;
- one transition per domain at a time;
- thresholds derived from target-hardware baselines, not hard-coded globally.

### UI wake

External UI: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`.

`Wake Rover`:

- authenticated exact-Rover command;
- changes policy from Sleep to Auto;
- creates immediate bounded UI demand;
- live UI/media/control activity renews demand;
- after demand expiry, five-minute idle and low-CPU rules may return Rover to low power.

Explicit `Awake` remains available for maintenance/long sessions.

Button disabled when Rover coordinator status is disconnected or stale. No optimistic success.

### Scheduled wake

Scheduler registers a future wake reservation with deterministic occurrence/group generation ID.

Coordinator:

1. validates reservation;
2. keeps only control spine active until prewarm;
3. wakes `ScheduledCapture` at `planned_start - measured p95 profile-ready latency - safety margin`;
4. reports `accepted`, `prewarming`, `ready`, `blocked`, or `failed`;
5. starts no recording itself.

Scheduler starts occurrence only after `ready`. Terminal recorder feedback releases demand. If policy returns to Sleep, coordinator quiesces the scheduled profile after release.

Wake time is derived from measured p95, not fixed 30 seconds.

### Local voice wake

Rover-local continuous KWS:

1. microphone/KWS active in `IdleListening`;
2. exact wake phrase emits wake intent only;
3. local coordinator activates `NormalRover`;
4. movement and arm remain stopped;
5. after playback readiness, bundled prerecorded PCM says “I am on”;
6. general commands require direct UI or Orchestra STT in v1.

No local general STT/NLU in v1.

### Authority and reconnect

- Rover may wake locally while Orchestra/Zenoh unavailable.
- Rover reports policy epoch, effective profile, active bounded demands, and lifecycle status when link returns.
- Orchestra is final authority after reconnect.
- Reconciliation consumes Rover status before issuing newer authority.
- stale pre-disconnect commands never replay;
- Orchestra override uses a newer policy authority epoch;
- safety always wins.

### Restart

Fresh restart semantics:

- forget previous Awake/Auto/Sleep selection;
- discard transient demands/reservations after durable scheduler reconciliation;
- boot normal workloads into fresh `Awake`;
- preserve safety startup ordering;
- scheduler rebuilds future reservations from durable schedule/occurrence state.

### Durable event history and current-state projection

Every policy, demand, transition, phase, target, reconciliation, and terminal
state change emits a versioned event.

Durability order:

```text
append intent to local durable journal/outbox
→ apply local transition
→ emit authoritative live status
→ replicate event idempotently to MongoDB
→ update MongoDB current-state projection
```

Central MongoDB is not on the Rover safety/wake critical path. Rover can wake,
sleep, and record ordered events while Orchestra/network is unavailable. On
reconnect, the outbox replicates by stable event ID and authority sequence.

Event envelope includes:

- schema version and event ID;
- entity/role;
- coordinator authority epoch and monotonic sequence;
- transition ID and phase;
- event kind;
- policy and requested/effective profile;
- demand ID/source when applicable;
- exact target and lifecycle revision when applicable;
- actor/audit identity for authenticated actions;
- occurred/recorded timestamps;
- bounded reason code and sanitized detail.

MongoDB stores:

- append-only `power_lifecycle_events` history with 90-day TTL;
- materialized `power_current_state` per deployment/entity;
- indexes for entity/time, transition, demand, event kind, and reason.

UI source:

- live Socket.IO coordinator status is authoritative for current state;
- Mongo-backed API supplies timeline, filters, reconnect history, and failure
  investigation;
- current-state projection provides cold-start fallback until live status
  arrives and is visibly marked historical/stale.

This is event logging plus a materialized projection, not full event sourcing.
Coordinator does not rebuild safety authority solely from Mongo history.
Schema remains suitable for future Grafana/export pipelines without coupling v1
to Grafana.

## State Transition Ordering

Sleep:

1. close new workload admission;
2. cancel/finalize active operations;
3. safe-stop autonomous motion;
4. quiesce dependents;
5. close devices/workers;
6. drop model/session ownership;
7. acknowledge profile low-power only after every required target reports terminal state.

Wake:

1. validate policy/demand/authority;
2. resume prerequisites;
3. await authoritative `Running`/ready;
4. resume dependents;
5. open fresh admission;
6. emit aggregate ready status.

Failure yields `Degraded` or `Failed`; never claim ready/dormant from intent.

## Safety and Security

- KWS wake never executes movement, arm, recording, or tracking command.
- Fresh post-wake command required for actuators.
- Web wake requires valid session, rate limit, exact target, audit, and authority fencing.
- Always-on controllers/emergency path never pausable.
- No pre-sleep command replay after wake.
- Bounded demand TTL and capacity.
- Failure details sanitized.
- Orchestra reconnect cannot silently overwrite newer local state with stale command.

## Acceptance Criteria

Functional:

- UI can select Awake/Auto/Sleep and distinguish policy from effective profile.
- UI Wake moves Sleep to Auto and waits for authoritative ready.
- local KWS wakes Rover without Orchestra and emits status after reconnect.
- prerecorded “I am on” plays only after audio playback readiness.
- schedule reservation prewarms correct minimal profile and starts only after ready.
- final demand release returns to current policy.
- stale/duplicate/reordered commands and demands cannot regress state.
- restart boots fresh Awake and scheduler rebuilds future reservations.

Performance:

- prerecorded wake acknowledgment: under 1.5 s p95;
- normal Rover profile ready: under 5 s p95;
- schedule prewarm: measured p95 plus safety margin;
- Auto idle grace: five minutes;
- per-domain CPU below benchmark-derived threshold for consecutive fresh samples;
- low-power profile proves deterministic device/model/worker release;
- CPU reduction measured against active baseline;
- RSS reported as evidence, not hard release guarantee.

Voice benchmark:

- continuous KWS CPU/RSS on target Rover;
- detection latency;
- false accepts per noisy operating hour;
- false rejects across distance/noise dataset;
- motor/fan/TV/conversation test set;
- VAD-gated KWS considered only if continuous KWS misses idle budget.

Reliability:

- disconnected Rover local wake works;
- reconnect converges without stale replay;
- failed/timeout target prevents false aggregate ready/dormant;
- scheduler/web/coordinator restart recovery remains idempotent;
- resource-status staleness blocks automatic sleep.
- every applied transition has a preceding local journal intent;
- offline events replicate exactly once logically by stable ID after reconnect;
- live UI state is not regressed by an older Mongo projection;
- history UI can filter 90 days by entity, transition, source, result, and time.

## Likely Touchpoints

- new `common/power-coordinator/`;
- `robo_rover_lib/src/types/` power policy/demand/profile/status contracts;
- `common/lifecycle_manager/` execution fixes and coordinator interface;
- `orchestra/recording_scheduler/` future wake reservation/outcome;
- `common/web_bridge/` authenticated policy/wake Socket.IO;
- local durable power-event journal/outbox and MongoDB projector/repository;
- both Zenoh bridges and lifecycle topics;
- Orchestra/Rover/direct dataflow YAML;
- Rover audio capture and new local KWS node;
- audio playback fixed acknowledgment path;
- resource monitor/domain aggregation;
- external React shared types, state hook, Fleet Resources/power controls;
- architecture documentation and sequence diagrams;
- contract, state-machine, journal/replay, Mongo projection/TTL, reconnect,
  fault, resource, and browser tests.

## Risks

- two-authority split brain during reconnect;
- CPU thresholds incorrectly infer idle;
- microphone/KWS false wake under rover noise;
- dependency profile misses hidden device owner;
- repeated wake/sleep reload churn;
- scheduler prewarm too late;
- UI demand leak keeps Rover awake;
- current wake-lease defects accidentally reused;
- status fanout queue loses aggregate target state;
- external UI/backend releases drift.
- local journal fills during prolonged Mongo/network outage;
- duplicate/out-of-order replication regresses Mongo current-state projection;
- event detail leaks sensitive actor, path, or audio information.

Mitigations:

- authority epochs and status-first reconciliation;
- semantic demand first, CPU confirmation only;
- five-minute grace and minimum awake hold;
- static reviewed profiles;
- deterministic IDs and bounded TTL;
- bounded journal capacity, backpressure policy, idempotent upsert, and
  sequence-guarded projection;
- p95-derived prewarm;
- cross-language fixtures;
- fault injection and target-hardware benchmark.

## Next Steps

1. Create detailed implementation plan.
2. Benchmark continuous KWS and active/idle domain CPU/RSS on target Rover.
3. Freeze power contracts and authority model.
4. Freeze event envelope, local journal, Mongo projection, indexes, and TTL.
5. Repair lifecycle transition/deadline behavior before coordinator integration.
6. Implement coordinator state machine and profiles.
7. Connect scheduler reservations.
8. Add local KWS and prerecorded acknowledgment.
9. Add UI live state/history and end-to-end fault tests.

## Unresolved Questions

- Exact KWS model and wake phrase language.
- Target-hardware per-domain CPU thresholds.
- Consecutive low-CPU sample count and minimum awake hold duration.
- Scheduler behavior if prewarm succeeds but occurrence becomes invalid before start.
- Exact Orchestra-over-Rover authority takeover timeout on reconnect.
- Local journal capacity and behavior after sustained replication outage.
- Grafana export mechanism and long-term aggregate retention beyond raw 90-day
  events.
- Whether future v2 adds GPIO wake, local bounded STT/NLU, or OS suspend.
