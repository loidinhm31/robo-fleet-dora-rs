# Phase 01 — Resource-Only Contracts and Collectors

## Context links

- [Parent plan](./plan.md)
- [Resource research](./research/researcher-02-resource-metrics-ui.md)
- [Architecture](../../ARCHITECTURE.md#planned-resource-monitoring-and-soft-stop-lifecycle)
- Current: `robo_rover_lib/src/types/performance_types.rs`, `rover-kiwi/performance_monitor/src/main.rs`

## Overview

- Date: 2026-07-21
- Priority: P1
- Implementation status: Done (2026-07-21T22:35:04+07:00)
- Review status: Approved (2026-07-21T22:35:04+07:00)
- Description: replace synthetic performance telemetry with resource-only snapshots on both roles.

## Key Insights

- Current FPS and latency derive from CPU guesses; queue/drop/max values are never observed.
- Dora IDs do not reliably match executable names. Missing process must differ from measured zero.
- Native host and container cgroup values are different scopes and must be labeled.

## Requirements

- Remove FPS, processing time, latency, queue, and drop fields from Rust and Fleet UI contracts.
- Measure system/container CPU and memory plus per-configured-process CPU and RSS only.
- Required metadata: schema version, role/entity, scope/source, sequence, sample time, interval.
- Use null/absent for unavailable. Zero means measured zero.
- Remove battery from this resource contract; retain battery only in hardware telemetry if needed.
- Add Orchestra collector parity; keep collector running during workload pause.

## Architecture

Create `ResourceSnapshot` and `NodeResourceUsage`. Prefer cgroup v2 totals inside containers and procfs/sysinfo host totals natively. Configure `{node_id, exact executable}` per dataflow; aggregate exact matches and report `running`, `not_found`, or `ambiguous`. Publish Rover snapshots on `rover/{entity_id}/resources/v1`; Orchestra snapshots stay local. Web event: `resource_snapshot`.

## Related code files

- Move/modify: `rover-kiwi/performance_monitor/` → `common/resource_monitor/`.
- Modify: `Cargo.toml`, `robo_rover_lib/src/types/mod.rs`, replace `performance_types.rs` with `resource_types.rs`.
- Modify: both Rover dataflows and `orchestra/orchestra-dataflow.yml`.
- Modify: both Zenoh bridges and `common/web_bridge/src/main.rs`.
- Modify: UI shared `types/performance.ts`, `types/socket.ts`, exports/adapters; rename to resource terminology.
- Delete after migration: old `SystemMetrics`, `NodeMetrics`, `performance_metrics`, `performance_control` paths.

## Implementation Steps

1. Define validated versioned snapshot types and JSON fixtures in Rust and TypeScript.
2. Extract sampler/resolver modules below 200 LOC; inject clocks/process source for tests.
3. Detect native vs cgroup-v2 scope; normalize system CPU to allocated 0–100 and expose capacity.
4. Replace substring matching with explicit executable manifest; aggregate exact PIDs.
5. Add collector to Orchestra, Rover Zenoh mode, and Rover direct mode with 5 s interval.
6. Route new Dora ports, Zenoh topic, and Socket event; reject invalid/range-breaking samples.
7. Remove old fields/types/events atomically and update logs/mocks/fixtures.

## Todo list

- [ ] Shared Rust and TS contracts
- [ ] Native/cgroup samplers
- [ ] Exact process resolver
- [ ] Orchestra/Rover/direct wiring
- [ ] Old performance contract removed
- [ ] Unit and contract tests

## Success Criteria

- No fleet resource code references FPS, latency, queue size, or dropped frames.
- Orchestra and Rover emit scoped CPU/memory snapshots with monotonic sequence.
- Missing process renders as state, not `0% / 0 MiB`.
- CameraViewer diagnostic FPS remains unchanged.

## Risk Assessment

- Rolling old/new UI mismatch: use coordinated deployment or brief backend dual-read only; final emit is new contract.
- Container scope ambiguity: fail clearly to `unknown` rather than label host data as container.
- Duplicate executable: aggregate and expose process count; manifest validation fails closed.

## Security Considerations

- Do not emit command lines, paths, PIDs, environment, or host secrets to browsers.
- Bound node counts and strings; validate finite percentages and memory limits.

## Next steps

- Phase 02 adds lifecycle state/control; collector stays independent and always on.

## Unresolved Questions

- Exact production container cgroup layout must be confirmed during Phase 05.
