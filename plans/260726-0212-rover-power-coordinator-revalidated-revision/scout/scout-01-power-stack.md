# Scout report: current lifecycle/resource/power stack

> Historical pre-implementation inventory. It predates commits `ff6624e`,
> `a9ba1c4`, and `a1cbc38`; statements that power crates/types/dataflow entries
> do not exist are no longer current. Re-scout `HEAD` before implementation.

## Lifecycle manager

- `common/lifecycle_manager/src/manager.rs`: `LifecycleManager` state machine (`new`, `apply`, `apply_relayed`, `tick`, status/capability handling), 30s transition timeout, request cache, wake-lease maps, remote stale handling. Unit tests cover duplicate/conflicting commands, lease expiry/reconcile, stale epochs/status, timeout/late status, remote authority and cache exhaustion (lines ~584-932).
- `common/lifecycle_manager/src/main.rs`: Dora node; accepts `lifecycle_command`, `lifecycle_command_relay`, component-status inputs, wake lease/query inputs; emits result/status/capabilities/authorized command and wake lease outputs. Epoch is startup wall-clock; remote targets from `LIFECYCLE_REMOTE_ROVER_ENTITIES`.
- `common/lifecycle_manager/src/lib.rs`: `REMOTE_ROVER_SAFE_NODE_IDS` (gst-camera, audio-capture, edge-voice, audio-playback), `MAX_REMOTE_ROVERS=15`, capability generation and packaging/dataflow queue tests.

## Resource monitor and shared contracts

- `common/resource_monitor/src/main.rs`: samples on Dora `tick` every configured flow interval (snapshot hard-coded `sample_interval_ms: 5000`), validates and emits JSON `resource_snapshot`.
- `common/resource_monitor/src/config.rs`: required `RESOURCE_MONITOR_ROLE`, `ENTITY_ID`, `RESOURCE_MONITOR_NODES`; validates bounded/unique node manifests.
- `common/resource_monitor/src/resource_sampler.rs`, `cgroup.rs`, `process_resolver.rs`: procfs/cgroup-v2 sampling and exact executable/process aggregation; tests distinguish missing vs zero and ambiguous processes.
- `robo_rover_lib/src/types/resource_types.rs`: v1 `ResourceSnapshot`, host/container scope/source, CPU/memory/node usage and validation; `robo_rover_lib/tests/fixtures/resource-snapshot-v1.json`, `resource_contract_tests.rs`.
- Lifecycle contracts: `robo_rover_lib/src/types/lifecycle_types/{command,gate,lease,status,validation}.rs`; `robo_rover_lib/src/types/lifecycle_contract_tests.rs`, `common/web_bridge/src/stt_bridge/tests/lifecycle.rs`, and lifecycle golden fixtures.

## Zenoh bridges

- `orchestra/zenoh_bridge/src/main.rs`: subscribes per active rover to `rover/{entity}/resources/v1`, lifecycle status/result/capabilities; forwards to Dora outputs; publishes authorized lifecycle commands to `rover/{entity}/cmd/lifecycle/v1`, wake leases to `.../cmd/lifecycle-wake-lease/v1`, and query to `.../cmd/lifecycle-query/v1`. Includes active-rover capacity/routing tests (~1666+).
- `rover-kiwi/zenoh_bridge/src/main.rs`: publishes resource/lifecycle status/result/capabilities; subscribes lifecycle command, wake lease, query topics and validates payloads before Dora relay. Tests malformed/invalid/valid command forwarding (~860+).
- Config files: `orchestra/zenoh_bridge/zenoh_config*.json5`, `rover-kiwi/zenoh_bridge/zenoh_config*.json5`.

## Dataflow wiring likely to change

- `orchestra/orchestra-dataflow.yml`: existing `resource-monitor` (~243), `lifecycle-manager` (~258), orchestra-bridge inputs/outputs (~60-96, 311-320), web-bridge lifecycle command/status wiring (~100+). No power-coordinator node or power demand/profile inputs.
- `rover-kiwi/rover-kiwi-dataflow.yml`: resource-monitor (~121), lifecycle-manager (~135-151), zenoh bridge relay wiring (~303-331), safe-node lifecycle inputs; no power-coordinator node.
- Also verify `rover-kiwi/rover-kiwi-direct-dataflow.yml` for lifecycle queue contract (covered by lifecycle_manager tests).

## Historical pre-`ff6624e` mismatch with the target architecture

The architecture document describes a new `common/power-coordinator` running on Orchestra and Rover, policy profiles (`Awake`/`Auto`/`Sleep`), demand aggregation, scheduler reservations, local KWS wake, journal/projection, and a “Zenoh power v1” protocol. No `common/power-coordinator` source crate, shared power types, power Zenoh topics, scheduler/KWS coordinator node, Mongo projection, or power dataflow entries currently exist. Current implementation only provides lifecycle control, wake leases, and resource snapshots; resource monitor is explicitly measurement-only and has no freshness/policy authority. Plan must define these missing contracts and integrate them without confusing existing lifecycle command/wake-lease topics.

## Relevant verification targets

`cargo test -p lifecycle_manager`; `cargo test -p resource_monitor`; `cargo test -p robo_rover_lib`; bridge unit tests via `cargo test -p orchestra_zenoh_bridge` and `cargo test -p rover_kiwi_zenoh_bridge` (package names from each Cargo manifest). Existing lifecycle/resource contract tests and dataflow packaging tests should be extended for power coordinator topics, queue sizes, authority/failover, stale resource blocking, and restart-to-Awake behavior.
