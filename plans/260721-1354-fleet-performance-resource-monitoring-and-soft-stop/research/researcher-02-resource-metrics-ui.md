# Resource-only fleet metrics and UI research

## Conclusion

Replace the current `SystemMetrics`/`NodeMetrics` payload atomically with a versioned resource snapshot. Keep only CPU, memory, optional battery, process presence, scope, sampling, and freshness fields. Do not encode unavailable values as zero. The current FPS/latency/queue/drop fields are synthetic or permanently empty and should leave Rust, Socket.IO, shared TypeScript, `FloatingMetrics`, and `FleetSelector` together.

## Current path and correctness

`rover-kiwi/performance_monitor/src/main.rs` samples every 5 s from the rover dataflows, serializes `robo_rover_lib/src/types/performance_types.rs`, then:

`performance-monitor/metrics` -> `rover-kiwi/zenoh_bridge/src/main.rs` -> Zenoh `rover/{entity}/metrics` -> `orchestra/zenoh_bridge/src/main.rs` -> `common/web_bridge/src/main.rs` -> Socket.IO `performance_metrics` -> `RoboRoverControl.tsx` -> `FloatingMetrics.tsx`/`FleetSelector.tsx`.

Problems:

- Node FPS is inferred from CPU (`cpu / 5`), processing time is inferred from that FPS, end-to-end latency sums inferred values, and dataflow FPS references obsolete in-process detector/tracker “nodes”. These values are not measurements.
- Queue size and dropped frames never receive observations; max processing time is never populated.
- Missing processes become CPU `0`, memory `0`, indistinguishable from a healthy idle process.
- Matching `p.name().contains(node_id.replace('-', '_'))` is fragile: Dora ID != executable (`gst-camera` -> `kornia_capture`, `zenoh-bridge` -> `rover_zenoh_bridge`); long Linux `comm` names can truncate; first-match loses duplicate processes.
- `sysinfo` system CPU/memory describe the visible OS view. In a default container, the PID namespace may restrict visible processes while `/proc/meminfo` and CPU totals commonly remain host-level, not cgroup quota/usage. Current labels do not state scope.
- Process CPU can exceed 100% for multithreaded work (per-core basis), unlike global CPU. UI thresholds assume one 0-100 scale.
- Battery comes from the first matching `/sys/class/power_supply` device. It is valid only when that host sysfs is visible and the device is the intended rover battery; it may be absent or represent a workstation battery.
- UI retains the last sample indefinitely. Pausing forwarding freezes a seemingly healthy snapshot.

## Active node inventory

Rover Zenoh dataflow (`rover-kiwi/rover-kiwi-dataflow.yml`): `audio-capture`, `audio-converter`, `edge-voice`, `audio-playback`, `performance-monitor`, `gst-camera` (`kornia_capture`), `video-encoder`, `visual-servo-controller`, `arm-controller`, `rover-controller`, `zenoh-bridge`. `object-detector`, `reid-extractor`, and `object-tracker` are libraries inside `gst-camera`, not processes. `sim-interface` is commented out.

Rover direct dataflow substitutes `web-bridge` for `zenoh-bridge`; otherwise monitor the enabled YAML nodes, not a hard-coded historical list.

Orchestra (`orchestra/orchestra-dataflow.yml`): `orchestra-bridge`, `central-speech-recognizer`, `command-parser`, `recording-scheduler`, `media-recorder`, `web-bridge`. There is currently no Orchestra `performance-monitor`, so resource parity requires adding/configuring a collector there or one host collector able to classify both runtimes.

## Recommended wire schema

Use new names to make the breaking change explicit, for example `ResourceSnapshot` and `NodeResourceUsage`:

```text
ResourceSnapshot {
  schema_version: 1,
  entity_id: string, role: "rover" | "orchestra",
  scope: "host" | "container", source: "procfs" | "cgroup-v2",
  sequence: u64, sampled_at_ms: i64, sample_interval_ms: u64,
  cpu_usage_percent: number?, cpu_capacity_cores: number?,
  memory_used_bytes: u64?, memory_available_bytes: u64?, memory_limit_bytes: u64?,
  battery: { percent: number?, voltage_volts: number?, source: string }?,
  nodes: Record<node_id, {
    state: "running" | "not_found" | "ambiguous" | "paused",
    cpu_usage_percent: number?, memory_rss_bytes: u64?, process_count: u32,
    sampled_at_ms: i64
  }>
}
```

Rules:

- `null`/absent means unavailable; zero means measured zero.
- Define system CPU as normalized 0-100 across allocated capacity. Expose capacity cores. For node CPU, either normalize the same way or label it `cpu_cores_percent` and permit values above 100; do not mix scales.
- Prefer cgroup v2 `cpu.stat`, `memory.current`, and `memory.max` in containers. Use sysinfo/procfs for native host scope. Emit the chosen scope/source.
- Memory is bytes on wire; format MiB/GiB in UI. RSS per process/node; system/container working usage and limit must share scope.
- Battery should require an explicit power-supply name/path. Omit it when unconfigured or inaccessible.
- Keep a monotonic sequence and wall timestamp. UI marks stale after `max(3 * sample_interval, 15 s)` and removes/archives after a bounded timeout.

## Node identification

Preferred: derive a monitor manifest from the selected dataflow YAML at launch: `{node_id, executable basename, optional child policy}`. Resolve candidates by exact `/proc/<pid>/exe` basename or full command path; aggregate all exact matches and report `process_count`. Avoid substring matching.

More robust later: Dora/launcher passes each node PID or a node-specific cgroup to the collector. Cgroup membership handles subprocesses and duplicate executables correctly. Self-reporting resource usage from every node is unnecessary duplication.

Do not show disabled/commented nodes. A configured node with no process is `not_found`; an intentionally soft-stopped node is `paused`. This distinction is required for operations.

## UI and pause behavior

Change `FLEET PERFORMANCE` to `FLEET RESOURCES`. Collapsed view: entity status, normalized CPU, scoped memory percent, optional battery, and stale/paused badge. Expanded view: CPU and memory tabs only; rows show node state and `—` for unavailable values. Show `Host` or `Container` scope beside totals.

Remove FPS/latency code from `/packages/ui/src/components/features/FloatingMetrics.tsx`, and remove the FleetSelector FPS badge in `/packages/ui/src/components/organisms/FleetSelector.tsx`. Do not remove camera-local FPS/drop diagnostics from `CameraViewer.tsx`; those are separate operator stream diagnostics unless product scope explicitly says all UI FPS.

Current `performance_control` in `common/web_bridge/src/main.rs` flips one global boolean and only suppresses broadcast. Collection and Zenoh traffic continue, one client affects all clients, there is no acknowledgement, and UI optimistically changes local state.

For display subscription pause, make it per Socket.IO connection and acknowledge authoritative state. For actual resource saving, use a separate authenticated control command with explicit `{request_id, target_entity_id, target_role, action: pause|resume}` and propagate UI -> Orchestra web bridge -> Orchestra control plane -> selected rover. Return accepted/applied/failed status; never call a frozen sample live. Monitoring itself should remain lightweight and running so paused state/resources remain observable.

## Migration and tests

Files requiring coordinated update:

- `robo_rover_lib/src/types/performance_types.rs`
- `rover-kiwi/performance_monitor/src/main.rs` and both rover dataflow YAMLs
- `rover-kiwi/zenoh_bridge/src/main.rs`, `orchestra/zenoh_bridge/src/main.rs` only if event/topic names change
- `common/web_bridge/src/main.rs` deserialization, logs, control/auth/ack behavior
- `robo-control-app/packages/shared/src/types/performance.ts`
- `RoboRoverControl.tsx`, `FloatingMetrics.tsx`, `FleetSelector.tsx`, telemetry adapter interfaces/mocks

Use a new event (`resource_metrics`) or require `schema_version`; an old UI will call `.toFixed()` on removed fields and can fail. During rolling deployment, web bridge can accept old/new payloads but should emit only the contract negotiated/supported by the UI.

Add Rust tests for JSON round-trip/unknown optionals, two-refresh CPU sampling, exact process aggregation, missing/ambiguous state, cgroup limits, battery selection, and monotonic sequence. Add web-bridge contract/auth/ack and per-socket subscription tests. Add UI tests for CPU/memory-only rendering, unavailable values, scope labels, staleness, disconnect cleanup, pause acknowledgement/failure, and removal of FPS references. No dedicated performance metrics tests were found in the searched Rust/UI test files.

## Unresolved questions

1. Does “remove FPS from UI” include `CameraViewer` stream diagnostics, or only Fleet Resources/FleetSelector? Recommendation: only fleet resource surfaces.
2. Should system totals mean physical host usage or each service container’s cgroup usage? Recommendation: container scope in Docker, host scope native, always labeled.
3. Which physical power-supply device represents rover traction battery? Laptop `BAT0` should not be assumed.
4. Is resource-saving pause per node, predefined node group, whole role, or whole dataflow? The control schema should support target groups only after safe pause dependencies are defined.
