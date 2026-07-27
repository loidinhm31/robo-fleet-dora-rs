# Scheduler Recording and R2 Retention Brainstorm

Date: 2026-07-13  
Status: agreed architecture; implementation not started

## Problem

`rover-kiwi/kornia_capture` opens the webcam at node startup and keeps capturing even when no browser consumes video. Stream demand only gates frame publication. Need scheduled camera operation, smart/manual overrides, event A/V recording, central quality control, Cloudflare R2 upload, fleet-wide storage control, and two-week retention behavior.

## Locked requirements

- Hybrid control: recurring schedule normally forces webcam and microphone ON; operator/viewer/tracking demand may turn them ON outside the schedule.
- Manual hard OFF wins over schedule, recording, and tracking. Manual hard ON bypasses schedule but expires by configurable TTL; proposed default 2 hours.
- Dedicated Orchestra scheduler renews device/stream leases. If Orchestra or Zenoh disappears, rover capture nodes close devices after configurable grace; proposed default 30 seconds.
- Record event clips, not continuous 24/7 footage.
- Orchestra creates clips from existing rover JPEG and timestamped PCM streams. No rover recording node, video spool, or R2 credential.
- Output: H.264 video + AAC mono audio in MP4. Preserve current source resolution; do not upscale. Current path provides 640x480 at up to 15 fps and 16 kHz mono audio.
- Keep current microphone echo suppression. Insert timestamp-correct silence while rover speaker/TTS playback suppresses microphone frames; do not mix playback audio in v1.
- Private R2 bucket. One global fleet ceiling of 8 decimal GB, leaving headroom below R2 Standard's 10 GB-month free allowance.
- Retention is a 14-day deletion target, not a strict provider guarantee. Request deletion at age 14 days, send notification, escalate critical if still present at 15 days. Storage pressure may evict older clips sooner.
- Orchestra owns R2 credentials, uploads, byte ledger, retention, and reconciliation.
- Backend and UI included. Active UI scope is the separate checkout `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`, not the deleted in-repo path.

## Evidence from current code

- `rover-kiwi/kornia_capture/src/main.rs`: camera starts during node initialization; `CameraControl::Start/Stop` can already open and physically release it. `ViewOutputGate` only controls outgoing view frames.
- `rover-kiwi/video_encoder`: raw RGB becomes TurboJPEG 4:2:2 JPEG. No MP4/H.264 recording exists.
- Rover and Orchestra Zenoh bridges already transport timestamped JPEG frames on `rover/{entity_id}/video/jpeg/v1`.
- Rover audio capture publishes 50 ms, 16 kHz mono chunks with stream ID, frame ID, sample count, and Unix capture timestamp. Audio converter preserves metadata.
- Microphone publication intentionally pauses during rover playback plus a 400 ms tail.
- `common/web_bridge` aggregates browser view demand, but has no recording demand, camera state acknowledgement, schedule persistence, clip metadata, or R2 integration.
- Current Orchestra video sequence tracking is global while recording must be per rover. Browser video payload also omits `entity_id`.
- Orchestra image currently lacks FFmpeg/GStreamer runtime and has no persistent recording spool volume.
- MongoDB use is currently authentication-only.

## Evaluated approaches

### 1. Rover records and uploads directly

Pros: survives Orchestra outage; avoids sending all recording frames over LAN.  
Cons: new rover node/storage, cloud secrets or presign protocol, global quota races, extra rover CPU.  
Decision: rejected by requested scope.

### 2. Rover transfers finalized clips to Orchestra

Pros: better rover-side compression; central R2 credentials and quota.  
Cons: still requires rover recorder/spool and new large-file transfer protocol.  
Decision: rejected by requested scope.

### 3. Orchestra records existing Zenoh media

Pros: no rover recording node; one quality and quota authority; secrets remain on workstation; reuses existing JPEG/PCM topics.  
Cons: no recording during Orchestra/Zenoh outage; JPEG-to-H.264 encoding costs workstation CPU; network frame loss becomes clip loss; demands must be decoupled from browser viewers.  
Decision: selected.

## Recommended architecture

Add one Orchestra Dora node, tentatively `recording-scheduler`, implemented as focused modules inside one crate:

- schedule store and evaluator
- per-rover device-demand/lease arbiter
- event trigger and clip state machine
- bounded per-rover pre-roll buffers
- timestamp normalizer and A/V gap handler
- FFmpeg worker supervisor
- atomic local spool and manifest
- R2 uploader, quota ledger, and reconciler
- retention sweeper and alert publisher

Keep encoding, disk, Mongo, and network work off the Dora event loop through bounded queues and supervised background workers.

```text
UI -> web_bridge -> recording-scheduler
                         | schedule/manual leases
                         v
                 orchestra Zenoh bridge -> rover camera/audio/stream controls

rover JPEG + PCM + detections -> orchestra Zenoh bridge -> recording-scheduler
                                                          | H.264/AAC MP4
                                                          v
                                                   bounded local spool
                                                          |
                                                          v
                                                    private R2 bucket
```

### Device-demand rules

Use one per-rover arbiter; consumers never send conflicting direct stop commands.

Precedence:

1. manual hard OFF
2. valid manual hard ON TTL
3. active recurring schedule
4. browser/view demand
5. tracking/mission demand
6. otherwise OFF

Effective camera, microphone, JPEG publication, ML, and recording demand remain separate. A schedule acquires camera + microphone + JPEG publication. Tracking may acquire camera + ML without recording. Last-demand release starts an idle/grace timer before closing hardware to prevent flapping.

Commands from Orchestra are renewable leases containing `entity_id`, revision, reason, and expiry. Rover capture nodes auto-close when the lease expires. Explicit status/ack returns actual device and stream state so the UI does not infer state locally.

### Event clip behavior

MVP triggers:

- manual record action
- confirmed tracked detection during a scheduled/manual-active period
- configured rover mission activity during a scheduled/manual-active period

Default clip policy:

- 10-second pre-roll
- 30-second post-roll
- merge related events during post-roll
- finalize/split at 5 minutes
- per-rover timestamps and sequence tracking
- variable video PTS based on capture timestamp
- audio duration from sample count/rate
- insert silence for missing or deliberately suppressed mic chunks
- atomic `.partial` to finalized MP4 rename

A powered-off webcam cannot wake from visual motion. PIR/GPIO wake and periodic visual probing are out of v1.

### Central quality policy

Orchestra encodes JPEG + PCM directly; this is not a second transcode. Use one deterministic storage-pressure policy, configurable after benchmarking:

| Fleet R2 usage | Video target | FPS target | Audio target |
|---|---:|---:|---:|
| below 60% | about 1,000 kbps | up to 15 | 64 kbps mono AAC |
| 60-80% | about 700 kbps | up to 12 | 48 kbps mono AAC |
| above 80% | about 450 kbps | up to 10 | 48 kbps mono AAC |

Never upscale the 640x480 source. Quality pressure reduces new clip size; it does not replace quota eviction. At roughly 1 Mbps plus audio, 8 GB holds about 16.7 total recorded hours across the fleet. At roughly 500 kbps total, it holds about 35.6 hours. Therefore 14 days cannot mean every event survives for 14 days under heavy activity.

### R2 quota and retention

Use R2 Standard only. Cloudflare's free storage is 10 GB-month per month, not a provider-enforced 10 GB bucket cap. An 8 GB application ceiling provides billing and lifecycle-delay headroom.

Single Orchestra uploader simplifies correctness:

1. Persist finalized clip size, checksum, rover, timestamps, event type, quality profile, and state in MongoDB.
2. Before upload, reconcile `uploaded_bytes + active_upload_bytes` against 8,000,000,000 bytes.
3. Delete age-expired objects first, then oldest unprotected objects until the projected upload fits.
4. If it still does not fit, keep within bounded local spool or evict oldest local clip according to policy; never cross the cloud ceiling.
5. Upload idempotently under a server-generated key; verify object size/checksum before marking uploaded.
6. Reconcile Mongo ledger against R2 listing on startup and daily.
7. Run application age sweep at least hourly. Configure a 14-day R2 lifecycle rule as defense in depth.
8. Emit `deletion_requested` notification at 14 days; emit critical retention alert at 15 days if reconciliation still finds the object.

R2 lifecycle deletion is typically completed within 24 hours, so lifecycle alone cannot enforce a precise age or byte ceiling. App deletion remains primary.

### Local spool

Spool exists on Orchestra only. It must be a persistent Docker volume with both byte cap and minimum-free-disk guard. On full spool, delete oldest finalized/unprotected clip first; never block rover control, Zenoh receive, or live UI. Partial files and manifests must recover after restart.

### Security and privacy

- R2 credentials only in Orchestra secret environment/runtime store; never in UI, rover, logs, or repository.
- Private bucket; authenticated role-checked playback through short-lived signed GET URLs.
- Server-generated object keys; validate MIME, size, checksum, rover identity, and clip metadata.
- Rate-limit and audit schedule, manual override, view, delete, and playback actions.
- UI must show camera/microphone/recording state, override expiry, retention warning, R2 usage, gaps, and deletion status.
- Document consent/privacy implications of automatic video and audio capture.

## Main touchpoints

- `robo_rover_lib/src/types/video_types.rs` and audio/shared types: schedule, lease, actual-state, recording status, clip metadata contracts.
- `rover-kiwi/kornia_capture/src/main.rs`: expiring camera/JPEG-publication lease and actual-state output; no recording code.
- `rover-kiwi/audio_capture`: expiring microphone lease and actual-state output; retain playback suppression.
- `rover-kiwi/zenoh_bridge` and `orchestra/zenoh_bridge`: targeted per-rover leases, acknowledgements, schedule/recording status, media routing.
- `orchestra/orchestra-dataflow.yml`: new recording-scheduler node and inputs/outputs.
- New Orchestra recording-scheduler crate and modular internals.
- `common/web_bridge`: authenticated Socket.IO commands/status, no media encoding in hot loop.
- Mongo collections: schedules, clip ledger, deletion/audit events.
- `docker/Dockerfile.orchestra` and Compose: encoder runtime, writable bounded spool, R2 secrets/config, healthcheck.
- Separate UI checkout shared socket types, camera control rail, schedule editor, recording browser/status, quota and retention alerts.

## Explicitly out of scope

- Rover-side recording, clip spool, R2 upload, or long-lived R2 credentials.
- Continuous 24/7 recording.
- Raw/JPEG frame archival in R2.
- Speaker audio mixing in recorded clips.
- PIR/GPIO wake, visual probing while nominally off, facial recognition, cloud video analytics.
- Multiple codecs/resolutions or user-defined encoding profiles in v1.
- Protected clips beyond the global 8 GB ceiling.

## Risks and mitigations

- Orchestra/network outage loses recordings: report explicit recording gap; expire rover leases and close devices safely.
- Encoding stalls: bounded per-rover queues, worker concurrency limit, frame-drop policy, watchdog, metrics.
- A/V drift or clock jumps: short segments, timestamp discontinuity guards, per-rover timelines, silence insertion.
- Browser demand races schedule: one aggregate demand arbiter and state acknowledgements.
- Multi-rover corruption: per-entity frame/audio sequence trackers; never reuse current global video tracker.
- Disk/R2 full: hard spool guards, 8 GB pre-upload gate, oldest-first eviction, reconciliation.
- R2 lifecycle delay: application sweeper and 14/15-day alerts.
- Privacy exposure: private bucket, least-privilege auth, audit, short playback URLs.

## Acceptance criteria

- Webcam and microphone remain closed after startup when no valid demand exists.
- Active schedule opens required devices and JPEG flow for the targeted rover; schedule works across restart and configured IANA timezone/DST behavior.
- Manual OFF wins immediately. Manual ON works outside schedule and expires automatically.
- Missed Orchestra lease heartbeat closes rover camera/mic/stream within configured grace and records an observable gap/status event.
- Browser view stop cannot interrupt scheduled recording; recording stop cannot interrupt other active consumers.
- Confirmed event produces playable H.264/AAC MP4 with configured pre/post-roll and bounded A/V skew; playback-suppressed mic periods become silence.
- Slow encoding, full disk, R2 outage, and frame loss never block rover command/tracking or the Dora event loop.
- Restart recovers partial/finalized spool state without duplicate cloud objects.
- Concurrent multi-rover workload never takes R2 ledger/projected usage above 8,000,000,000 bytes.
- Objects receive deletion request at age 14 days; operator notification fires; any object still present at 15 days creates critical alert.
- R2 lifecycle backup is configured and reconciliation repairs ledger drift.
- Unauthorized users cannot change schedules/overrides, browse clips, sign playback, or delete clips.
- UI reconnect displays backend-confirmed camera, mic, recording, schedule, override TTL, quota, upload, gap, and retention states.
- Benchmarks prove selected Orchestra hardware can encode the supported concurrent rover count without unacceptable media/control degradation.

## Validation plan outline

- Unit: schedule/timezone, precedence, lease expiry, clip state, quality thresholds, quota eviction, retention alerts, idempotency.
- Integration: timestamped synthetic JPEG + PCM -> playable MP4; missing audio/video; clock discontinuity; FFmpeg crash; Mongo/R2 reconciliation.
- Multi-rover: interleaved entity frames, simultaneous events, global quota race, per-rover state isolation.
- Failure: Zenoh disconnect, Orchestra restart, R2 unavailable, disk full, corrupt partial, lifecycle delay.
- Security: role matrix, signed playback expiry, secret/log inspection, forged object keys/metadata.
- Docker/live: real webcam/mic, device lease auto-off, CPU/memory/disk/network metrics, UI E2E in separate checkout.

## External constraints verified

- [Cloudflare R2 pricing](https://developers.cloudflare.com/r2/pricing/): Standard free tier includes 10 GB-month storage, 1 million Class A operations, 10 million Class B operations, and free egress; delete operations are free.
- [R2 object lifecycle behavior](https://developers.cloudflare.com/r2/buckets/object-lifecycles/): expired objects are typically removed within 24 hours; rule changes and large migrations can take longer.
- [R2 upload guidance](https://developers.cloudflare.com/r2/objects/upload-objects/): single PUT is suitable for small/medium objects; multipart is intended for large or resumable uploads.
- [R2 platform limits](https://developers.cloudflare.com/r2/platform/limits/): single-part upload limit is about 5 GiB; event clips should remain far below it.

## Next steps

1. Create a hard implementation plan covering contracts, scheduler/lease state, media pipeline, storage, UI, security, and staged validation.
2. Benchmark FFmpeg H.264/AAC presets on the target Orchestra workstation before locking concurrent-rover count and exact bitrate tiers.
3. Confirm production ownership/path of the separate UI checkout before implementation begins.

## Unresolved tuning questions

- Maximum concurrent recording rovers supported by target Orchestra hardware; must come from benchmark.
- Production schedule timezone(s), recurrence editor details, manual ON TTL, lease interval, and disconnect grace; proposed defaults above need product confirmation during planning.
- Orchestra local spool byte cap and minimum-free-disk threshold; derive from host capacity and expected outage duration.
- Exact event confirmation thresholds and mission-state triggers; reuse existing tracking telemetry first.
