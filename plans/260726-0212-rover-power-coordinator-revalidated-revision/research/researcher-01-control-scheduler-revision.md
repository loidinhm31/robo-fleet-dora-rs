# Research Report: Power coordinator control/scheduler revalidation

> Historical pre-implementation evidence. Commits `ff6624e`, `a9ba1c4`, and
> `a1cbc38` changed this inventory. Revalidate every code claim against `HEAD`
> and use the cutoff audit as the current baseline.

Timestamp: 2026-07-26 (Asia/Ho_Chi_Minh)

## Executive summary

The prior sleep/wake plan is directionally sound, but revision must make lifecycle
authority and recorder startup explicitly snapshot-gated. Confirmed policy: Auto
rests at `IdleListening` (KWS + control spine); explicit Sleep targets `Dormant`.
Orchestra must wait for a Rover authoritative snapshot/reconciliation and must
never force-takeover a live Rover state. Scheduler reservation is a future demand,
not readiness; all delete/edit/supersession and terminal paths release ownership.
Only transient recorder/storage faults retry, and retries must remain inside the
occurrence's bounded window.

## Evidence (current code/design)

- `docs/power-coordinator-architecture.md:26-41,100-158` assigns policy/profile
  authority to coordinator, exact-target fencing to lifecycle manager, and says
  Orchestra consumes Rover status before a newer epoch on reconnect. It defines
  Auto/Sleep policy, `IdleListening` vs `Dormant`, and reviewed dependency order.
- `docs/power-coordinator-architecture.md:133-135` requires fresh restart to
  clear transient policy/demands and boot `Awake`; scheduler rebuilds future
  reservations from durable occurrence state.
- `docs/power-coordinator-architecture.md:195-214` requires no demand/protected
  operation, fresh per-domain samples, five-minute grace, and cancellation on
  stale/high CPU; CPU alone is not idleness.
- `docs/power-coordinator-architecture.md:216-253` makes reservation accepted
  distinct from Ready, uses measured p95+margin prewarm, rechecks occurrence
  window/revision, and releases after terminal feedback.
- `common/lifecycle_manager/src/manager.rs:96-127` provides duplicate-ID
  idempotency/cache; `:157-215` relayed commands reject expiry/unsupported/stale
  epoch and mirror Orchestra revision; `:218-247` accepts only matching
  authority/revision component status. `:249-291` marks unknown Rover reports
  stale rather than assuming Running. `:294-348` wake leases are temporary and
  independent of user desired state; `:373-395` timeout becomes Failed and blocks
  late success. These are the fencing primitives the coordinator must consume,
  not bypass.
- `orchestra/recording_scheduler/src/runtime.rs:40-145` already reconstructs
  persisted occurrences/groups and recovers outbox intents before normal work;
  `:209-248` refuses due processing until reconciliation completes.
- `orchestra/recording_scheduler/src/runtime.rs:250-373` handles manual
  suppression atomically across overlapping owners and records terminal/partial
  recorder failures. `:374-420` validates exact intent/action/generation before
  applying feedback. `state_machine.rs:29-39` bounds retries (1,2,4,8,16, then
  30s) and stops retrying once the planned end is reached.
- `common/web_bridge/src/scheduled-recording-coordinator.rs:96-138` treats the
  recorder reconciliation snapshot as startup barrier; duplicates become an
  invariant violation and desired groups are retained while waiting. Main bridge
  status handling at `common/web_bridge/src/main.rs:3957-3975` propagates scheduler
  readiness. Media demand acquire/release and rollback are at `:4779-4849`.

## Required plan deltas

1. Add a coordinator `RoverSnapshotPending`/`AuthorityUnknown` gate. On connect,
   query and wait for matching Rover epoch/revision + applied status; mark stale
   and expose `Degraded/Unknown` meanwhile. No automatic profile command or
   epoch takeover. Only after snapshot may Orchestra issue a strictly newer epoch.
2. Encode profile mapping unambiguously: `Auto` demand-free transition ends in
   `IdleListening` when KWS is enabled; explicit `Sleep` ends in `Dormant` and
   disables KWS. A local KWS wake is valid only from IdleListening and never an
   actuator command.
3. Keep lifecycle manager as exact-target executor. Coordinator barriers must
   require authoritative terminal component statuses, preserve timeout fencing,
   and journal intent before every applied phase. Reconnect reconciliation must
   not replay stale actuator/media commands.
4. Extend scheduler reservation state/outbox to release on schedule delete,
   edit/revision mismatch, supersession, manual suppression, terminal recorder
   feedback, storage/recorder terminal failure, and missed window. Release must be
   idempotent and group/generation scoped; final-owner release reconciles current
   policy (`Awake|Auto|Sleep`).
5. Readiness failures: retry only classified transient recorder/storage/unavailable
   faults; use existing bounded exponential delays and stop before
   `planned_end_ms`. Invalid revision, missing rover, authority conflict,
   unsupported capability, or safety failure are terminal/non-retryable. Surface
   distinct reason codes; never retry arbitrary coordinator/lifecycle errors.
6. Keep snapshot barrier on recorder startup and add power snapshot barrier to
   restart tests. During Mongo/network outage local journal and Rover safety
   continue; durable scheduler outbox rebuilds without duplicate acquire.

## Exact touchpoints for revision

- New/modified `common/power-coordinator/src/{state-machine,readiness,transition-planner}.rs`:
  snapshot gate, IdleListening/Dormant reducer, authority epoch/revision checks,
  journal-before-apply, bounded retry classification.
- `common/lifecycle_manager/src/manager.rs` integration adapter (do not weaken
  existing checks); add tests around `mark_remote_status_stale`, relayed epoch,
  lease release, timeout/late status.
- `orchestra/recording_scheduler/src/{domain,state_machine,runtime,runtime_groups,
  node_intents,node_loop,service_actions,node_persistence,mongo_documents,
  mongo_repository}.rs`: reservation lifecycle, revision supersession/delete,
  outbox replay/release, transient-only retry.
- `common/web_bridge/src/{scheduled-recording-coordinator.rs,main.rs,
  recording-schedule-feedback-spool.rs}`: Ready gate, recorder/storage feedback,
  release propagation and reconnect snapshot handling.
- Add fake-clock/recovery tests in `orchestra/recording_scheduler/tests/` for
  delete/edit/supersession, overlap final-owner release, late Ready, bounded
  transient retry, non-retryable failure, restart/outbox replay, and Rover
  snapshot-before-authority takeover.

## Risks

- Treating a missing Rover snapshot as Running can overwrite real Rover state;
  treating it as Dormant can issue an unsafe wake/stop. Explicit Unknown is safer.
- KWS accidentally enabled in Sleep violates explicit Dormant semantics and can
  create hidden wake demand; profile validation must enforce this.
- Release leaks on schedule edits/deletes leave permanent media/power demand;
  require generation-scoped idempotency and reconciliation sweeps.
- Retrying non-transient authority or revision errors can churn transitions and
  cross occurrence windows; classify errors before scheduling retry.
- Sparse p95 samples underestimate prewarm; retain conservative bootstrap and
  expose estimate/sample count/misses.

## Unresolved questions

- Exact Rover snapshot freshness/timeout and whether UI labels pre-snapshot state
  `Unknown`, `Stale`, or `Degraded`.
- Which recorder/storage reason codes are guaranteed transient across media
  recorder and Mongo paths; freeze an allowlist before implementation.
- Product wording for an occurrence invalidated after Ready (cancelled vs
  power-suppressed); release behavior is non-negotiable either way.
