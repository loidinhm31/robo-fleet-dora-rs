# Research journal: power authority and sleep/wake contracts

Date: 2026-07-24  |  Scope: lifecycle manager, recording scheduler, web/Zenoh routing, resource monitor, Mongo durability

## Executive finding

The repository already has useful primitives—fenced lifecycle revisions, local stale-status rejection, scheduler deterministic IDs/outbox, media demand reference counting—but no cross-workload power authority. Add a coordinator above these components. Keep `recording_scheduler` authoritative for schedules/occurrences only; make it a demand producer. Do not expose current wake leases as the public contract until their authority and recovery semantics are redesigned.

## Exact current gaps

- `LifecycleManager` stores wake leases only in memory (`manager.rs:35-36`); restart loses demands and there is no durable replay/journal.
- Lease acquire validates only non-empty ID/expiry (`manager.rs:294-315`). It has no owner, generation, priority, profile, reason, or renew contract; an arbitrary producer can hold a lease until expiry.
- User quiesce revokes leases (`manager.rs:442-464`), but release removes the revocation tombstone (`manager.rs:341-347`). A delayed duplicate acquire can therefore revive a user-paused target if it reuses the ID.
- Effective state is computed as binary “any lease => Running” (`manager.rs:467-489`), so no hierarchical profiles, dependency ordering, graceful prewarm, or safety veto exists.
- Transition timeout marks a target failed and fences late status (`manager.rs:373-393`), but there is no retry/backoff or coordinator-level degraded policy.
- Scheduler loop emits recording intents and reconciliation messages (`recording_scheduler/src/node_loop.rs:135-159`); no lifecycle wake reservation or “ready before start” handshake is present.
- Scheduler Mongo outbox is keyed by unique `intent_id` (`mongo_repository.rs:93-96`) and has recovery tests, but no power-demand collection, lease ownership, or idempotent projection of coordinator events.
- `MediaDemandRegistry` is an in-memory reference-counted set (`media-demand-registry.rs:38-72`); it is a good local pattern, but disconnect cleanup is caller-driven and it cannot represent deadlines or profile dependencies.
- Resource monitor reports sampled host/cgroup usage (`resource_sampler.rs:25-66`) but does not publish readiness/capability evidence or enforce power transitions.

## Recommended contracts/state machine

Coordinator owns per entity: `policy = Awake|Auto|Sleep`, `authority_epoch`, monotonic `revision`, effective profile, active demands, transition deadline, and last acknowledged component states. Demand fields: stable `demand_id`, producer, reason, required profile, `not_before`, `expires_at`, renew sequence, and priority. Use a set keyed by `(entity,demand_id)`; duplicate acquire is idempotent only when payload matches.

Profiles should be ordered and dependency-aware, e.g. `MinimalWake < Network < NormalRover < Recording < Tracking`; transition planner expands a profile into topologically ordered component commands and rolls back/marks degraded on failure. `Sleep` is an explicit user veto for normal demands; safety/maintenance demands may be separately classified and must be visible.

State flow: `Dormant -> Waking -> Ready|Degraded|Failed`; `Ready -> Quiescing -> Dormant`; any state can enter `Reconcile` after reconnect. Every command/status carries `(authority_epoch, revision)`; on Orchestra restart increment epoch, publish a snapshot, and ignore old Rover status until a relayed command establishes the new epoch. Local Rover wake may create an ephemeral local demand and epoch; reconciliation must compare epochs and never replay stale motion/control commands.

Scheduler integration: calculate prewarm from measured p95 transition latency plus safety margin; acquire a demand before `not_before`, wait for coordinator `Ready(profile, epoch, revision)`, then emit recording intent. Release only after terminal recorder feedback. If readiness misses deadline, mark occurrence `Missed/Failed` deterministically; never start capture merely because a wake command was accepted.

Durability: use a small coordinator outbox/journal (append intent before publishing; status acknowledged separately). Persist idempotency key and epoch/revision. Mongo projections should use unique `(entity_id,event_id)` or `(entity_id,revision)` and conditional updates; retries must be safe. TTL indexes are suitable only for terminal events/expired demands, never authority snapshots or pending commands. Mongo TTL deletion is asynchronous, so application filtering remains required ([MongoDB TTL indexes](https://www.mongodb.com/docs/manual/core/index-ttl/)).

## Failure/reconnect handling

1. Producer crash: demand expires; coordinator sweeps and recomputes profile.
2. Coordinator crash: recover journal, increment epoch, publish authoritative snapshot, re-drive only non-terminal intents.
3. Zenoh partition: Rover keeps safety-critical local control and local wake detector; normal remote commands expire. On reconnect, exchange epoch/snapshot, then reconcile desired profile before accepting recording/voice commands.
4. Component timeout: retain fenced revision, mark `Degraded`, retry with bounded backoff; scheduler receives explicit not-ready reason.
5. Duplicate/reordered events: reject lower epoch/revision; accept exact duplicate as idempotent acknowledgement.

Zenoh routing should add explicit coordinator topics (per-entity demand, command, status, snapshot), authenticated at the existing bridge boundary. Do not overload `rover/{id}/cmd/*` with power events that could be mistaken for motion commands.

## Test gates

- Property tests: demand add/release/expiry commutativity; duplicate payload idempotency; stale epoch/revision rejection; user `Sleep` veto; profile dependency ordering.
- Lifecycle integration: restart/replay journal; delayed old status; timeout then retry; local Rover wake followed by Orchestra authority takeover.
- Scheduler integration: prewarm reservation, missed-ready deadline, release after terminal feedback, Mongo outage/outbox replay, duplicate projection.
- Zenoh/Dora smoke: partition/reconnect with no stale motion replay; authenticated producer identity; bounded queue/backpressure.
- Metrics: wake latency p50/p95/p99, false wake count, demand age, readiness misses, transition timeout rate, stale-event count, and power-profile dwell time.

## Authoritative patterns consulted

- MongoDB TTL semantics and asynchronous cleanup: <https://www.mongodb.com/docs/manual/core/index-ttl/>
- Kubernetes-style explicit lifecycle/readiness separation is a useful analogy (readiness is not process existence): <https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/>.

## Unresolved questions

- Choose policy semantics: recommended `Awake|Auto|Sleep` with demand ledger and explicit safety override.
- Choose prewarm target: recommended measured p95 + margin; fixed 30s only as temporary fallback.
- Define whether local voice wake is enabled in `Dormant`; it requires an always-on audio detector and power budget.
- Determine exact component dependency graph and per-profile latency/energy budgets on target Rover hardware.
