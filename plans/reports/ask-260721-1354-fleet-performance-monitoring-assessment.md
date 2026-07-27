# Fleet Performance Monitoring Assessment

## Scope

Static trace of `FLEET PERFORMANCE` in the embedded UI and its Rover/Orchestra producers. No code changed. No runtime deployment was available to validate physical-device values.

## Architecture and collection path

```text
performance-monitor (Rover; every 5 s)
  -> rover Zenoh bridge: rover/{entity_id}/metrics
  -> Orchestra Zenoh bridge (injects authoritative entity_id)
  -> web-bridge Dora input
  -> Socket.IO performance_metrics broadcast
  -> RoboRoverControl Map<entity_id, SystemMetrics>
  -> FloatingMetrics (FLEET PERFORMANCE)
```

The live Rover dataflow wires `performance-monitor/metrics` to `zenoh-bridge/performance_metrics`; Orchestra wires the bridge output to `web-bridge/performance_metrics`. The UI preserves one latest sample per `entity_id`.

## What is monitored today

- **System**: `sysinfo` global CPU, used/available/total memory; Linux `/sys/class/power_supply` capacity/voltage when present.
- **Named processes**: `gst-camera`, `object-detector`, `object-tracker`, `visual-servo-controller`, `audio-capture`, `audio-playback`, `edge-voice`, `arm-controller`, `rover-controller`, `sim-interface`, `zenoh-bridge`.
- **UI**: overall dataflow FPS, system CPU/memory, claimed end-to-end latency, optional battery, and each node's FPS/CPU/memory/average latency.

## Correctness assessment

### Critical — displayed pipeline FPS is deterministically invalid

The monitor estimates FPS from process CPU (`min(cpu / 5, 30)`) rather than frame counts. It calculates dataflow FPS as the minimum for `gst-camera`, `object-detector`, `object-tracker`, and `visual-servo-controller`.

`object-detector` and `object-tracker` are no longer Dora processes; they are libraries inside `kornia_capture`. Their samples are therefore zero. `gst-camera` launches the `kornia_capture` executable, which also does not contain `gst_camera` in its process name. The minimum is consequently zero in the current dataflow, even while the camera works.

### Critical — latency is not end-to-end and not measured

Per-node processing vectors are never populated, max processing time is never updated, queue depth is hard-coded to zero, and dropped frames are never incremented. Average time falls back to `1000 / estimated_fps`. The claimed camera-to-web latency sums the estimated values for old stage names and omits video encoding, Zenoh transport, Orchestra bridge, web bridge, Socket.IO, browser decode/render, and current in-process detector/tracker work. It must not be used for operating or tuning decisions.

### High — the node inventory is stale and process matching is fragile

The monitor tracks retired standalone detector/tracker processes and optional/commented `sim-interface`; it omits active `audio-converter` and `video-encoder`. The alternative direct dataflow uses `web-bridge` rather than `zenoh-bridge`, also yielding a missing sample. Matching the first process whose executable name contains a transformed Dora id is not a reliable association or lifecycle check.

### High — a paused panel globally hides metrics without stopping collection

The UI's Pause button sends `performance_control`. The web bridge stores one process-wide boolean, so any client pauses/resumes forwarding for every browser and rover. The Rover monitor continues sampling and publishing. There is no acknowledgement, persisted/replayed state, or per-client subscription. The handler itself has no session check or rate limit; this is a global observability denial-of-service surface if an unauthenticated socket can reach it.

### High — stale data looks live

The UI keeps the last sample forever, has no cadence/sequence/freshness rule, and does not remove metrics when a rover is inactive. A disconnected rover can retain a healthy-looking card indefinitely. Its selector defaults to the first insertion-order entry, not the fleet's selected rover.

### Medium — transport failures become silent gaps

Both bridge paths discard publish/send errors for this telemetry. The message is typed/deserialized and Orchestra overwrites `entity_id` from the subscribed topic correctly, but there is no sequence number, delivery/drop counter, or cached snapshot for reconnecting clients.

### Medium — resource and battery semantics need explicit scope

System CPU/memory are useful host health signals, but their meaning changes under containers/cgroups. Battery is absent when `/sys/class/power_supply` does not expose a laptop-style BAT device; the UI cannot distinguish unavailable hardware from an unavailable collector. Node CPU can exceed 100% while the visual scale begins at 100%, which is acceptable but poorly communicated.

## Existing measured data worth reusing

`kornia_capture` already records real five-second windows for capture, view emission, worker submission/result/stale drops/errors, and YOLO/ReID/CMC/tracker/serialization timing. Today these are structured logs only. Promote those snapshots into the performance contract instead of inferring them from CPU.

`CameraViewer` independently measures browser-received FPS/bitrate, frame-id gaps, decode/render duration, and capture-to-render age. Those are also local UI state/console logs rather than fleet telemetry. They are the right source for browser-view performance, subject to Rover/browser clock synchronization for capture-to-render age.

## Recommendations

1. **Stop presenting estimated FPS/latency as measurements immediately.** Until replacement, mark them `unavailable` (not zero) and retain only genuine system CPU/memory/battery.
2. **Use explicit producer reports, not process-name discovery.** Each active node should publish `{node_id, interval_ms, frames_completed, frames_dropped, queue_depth, cpu_percent, rss_bytes, processing_ms p50/p95/max, sampled_at, sequence, state}`. Compute FPS as `frames_completed / interval`; omit metrics that do not apply.
3. **Make `kornia_capture` the vision producer.** Export its existing real `MetricWindow` snapshots under logical stages such as `capture`, `vision-worker`, `yolo`, `reid`, `tracker`, and `view-emit`. Do not invent separate detector/tracker processes.
4. **Define latency precisely.** Report per-stage latency now. For true camera-to-render latency, carry the existing capture timestamp/frame id to the browser, record render completion, and only calculate cross-machine latency after verifying clock synchronization/offset. Otherwise label it approximate.
5. **Add freshness and availability.** Every rover sample needs `sequence`, `sampled_at`, and collection interval. Mark data stale after three missed intervals, remove it on deactivation, and show `Unknown` rather than red zero for disabled/not-applicable stages.
6. **Replace the global Pause gate.** Prefer a per-socket metrics subscription (or simply client-side rendering pause) and an authenticated/rate-limited server handler. Collection should be controlled separately only if it materially saves Rover resources.
7. **Add minimal fleet value next.** Per-rover health state, heartbeat age, recent drop/error rate, stage p95 latency, and threshold alerts provide more operational value than a bar chart of guessed values. Keep a short in-memory rolling window first; durable metrics storage is not justified until alerting/history requirements are real.

## Implementation order

1. Correct schema semantics and UI unknown/stale states; remove the global control side effect.
2. Export actual `kornia_capture` snapshots and self-reported metrics from the remaining active nodes; aggregate by explicit `node_id`.
3. Add frame-id/capture-time correlation, transport loss counters, threshold alerts, and targeted tests covering stale, disabled, multi-rover, and reconnect behavior.

## Unresolved questions

- Is `performance_control` deliberately exposed before in-band authentication, or is Socket.IO authentication enforced before handlers run?
- Is the deployed Rover native, containerized, or both? This determines the correct CPU/memory scope (host vs cgroup).
- Are rover and browser clocks synchronized sufficiently for a cross-machine latency SLA?
