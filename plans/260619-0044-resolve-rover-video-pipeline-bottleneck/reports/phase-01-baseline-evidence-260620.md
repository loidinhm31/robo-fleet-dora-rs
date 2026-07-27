# Phase 01 Baseline Evidence

## Status

- Implementation: complete
- Automated validation: pass
- Code review: approved 2026-06-21

## Environment

- Host: Fedora x86_64, Podman Docker compatibility
- Camera: PC-LM1E UVC, 640x480 at 30 FPS
- Corpus: 300 RGB frames; checksums in `corpus-sha256.txt`
- Constrained rover: 3 CPUs, 4 GiB, no OOM

## Native Direct Baselines

Observed after warmup on 2026-06-20:

| Case | Duration | Key result |
|---|---:|---|
| Camera only | >10 min | ~30 FPS capture; capture errors 0; JPEG p95 ~8.5 ms |
| Detection only | >10 min | ~16-17 FPS; YOLO p50 ~60 ms; errors 0 |
| Typical tracking | >10 min | ~14 FPS; YOLO p50 ~64 ms; ReID ~3.5-4 ms/object; CMC ~3 ms; errors 0 |

## Browser Correlation

- Canvas: 640x480
- Viewer: 12.2 FPS during full tracking
- Stable browser window: capture-to-render p50 47 ms, p95 63 ms, p99 67 ms
- Receive-to-render p50 5.9 ms, p95 18.8 ms, p99 19.7 ms
- Browser frame IDs 55368, 55428, and 55489 align with backend emit windows 55411 and 55472; IDs remain monotonic.

## Constrained Container

- Image: `localhost/docker_rover-kiwi:latest`
- Limits: 3,000,000,000 NanoCPUs; 4,294,967,296 bytes
- State: running; OOM killed: false
- Camera-only RSS sample: 60.68 MB; CPU sample: 20.31%
- Zenoh config: `/app/config/zenoh_config.json5`, found and loaded
- Raw Zenoh publish p95: ~1.7 ms; publish age p95: 2 ms; errors 0

### Persisted Typical Tracking Window

- Time: 13:46:43Z-13:58:20Z (11m37s)
- Windows: 139
- Stage errors: 0
- Median detections: 2; brief maximum: 3
- YOLO median p50/p95/p99: 88.1/93.8/96.4 ms
- Vision total median p50/p95/p99: 98.2/104.1/113.1 ms
- Web receive age median p50/p95/p99: 36/47/52 ms
- No OOM; resource snapshot: `20260620T135800Z-tracking-typical-constrained-container.log`
- Resource contract baseline: failed. The snapshot averaged 290.09% CPU, above the
  270% target. This is a measured optimization input, not a Phase 01 measurement
  failure.

### Mixed-Scene Crowded Scaling

- Contract revision: a hand-held phone image could not remain stable for 10 minutes. The replacement requires a 10-minute mixed run with at least 30 high-object-count windows and a direct three-object payload sample.
- Direct payload sample: 144 frames, median 3 objects, maximum 3; latest classes were `person`, `person`, and `tv`.
- Time: 14:09:21Z-14:19:47Z (10m26s)
- Windows: 125; high-object-count timing windows: 35
- Stage errors: 0; OOM killed: false
- YOLO median p50/p95/p99: 86.3/92.7/95.8 ms
- Vision total median p50/p95/p99: 97.3/104.7/119.3 ms
- Web receive age median p50/p95/p99: 36/46/51 ms
- Resource snapshot: `20260620T141933Z-tracking-crowded-constrained-container.log`

### Final Reviewed-Artifact Resource Monitor

- Image: `localhost/docker_rover-kiwi:latest` (`7953717fe919...`)
- Time: 19:08:34Z-19:18:34Z (10 minutes); 119 five-second samples
- Limits: 3 CPUs and 4 GiB; OOM killed: false
- Cgroup CPU usage delta: approximately 3.0 CPU equivalents
- Cgroup throttling: 6,013/6,013 periods; throttled time delta 472.1 seconds
- Peak cgroup memory: 181,989,376 bytes (173.6 MiB); memory OOM/high/max events: 0
- Resource contract baseline: failed because full tracking saturated the CPU quota.
- Evidence: `20260620T190834Z-phase01-final-tracking-monitor.log`
- Plan consequence: Phase 02 isolates transport CPU and preserves full-tracking
  performance; the absolute <=2.7 CPU full-tracking gate remains in Phase 04 and
  final validation, after ML isolation and thread controls exist.

## Validation

- Rust affected packages: 25 tests passed; 1 doctest passed
- UI web/native TypeScript checks: passed
- Shell syntax, Compose config, and `git diff --check`: passed
- Existing compiler warnings only

## Review Remediation

- Regressed frame IDs no longer lower the sequence high-water mark.
- Browser render windows now report bytes, drops, errors, both latency
  distributions, and maximums.
- Benchmark collection now records cgroup CPU throttling, current/peak memory,
  memory events, effective limits, and OOM state; a 10-minute monitor command was
  added for sustained samples.

## Unresolved Questions

- None.
