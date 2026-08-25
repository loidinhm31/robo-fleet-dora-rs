# Phase 06 Verification Report

Date: 2026-07-21 (local; live-window timestamps below are UTC)

## Scope and platform boundary

All live evidence below was run on the current **rootless Podman/Docker-compatibility workstation** with `linux/amd64`, `WORKSTATION_AUDIO_DEVICE=sysdefault:CARD=Camera`, and recordings mounted at `/home/loidinh/robo-fleet-phase06-recordings`. It is workstation-amd64 evidence only; it is **not** ARM64 or Raspberry Pi physical acceptance.

The healthy stack was checked with:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
docker top robo-orchestra -eo pid,comm,args
```

MongoDB, Orchestra, and Rover-Kiwi were healthy. Orchestra contained `media_recorder`, `recording_scheduler`, and `web_bridge` processes.

Actual status output immediately before the live rerun:

```text
robo-mongodb     Up 52 minutes (healthy)
robo-rover-kiwi  Up 52 minutes (healthy)
robo-orchestra   Up 20 seconds (healthy)
70 media_recorder /app/bin/media_recorder
72 recording_scheduler /app/bin/recording_scheduler
73 web_bridge /app/bin/web_bridge
```

## Automated gates

- Previously completed: `cargo test -p recording_scheduler` (26), real-Mongo scheduler integration (3), `cargo test -p web_bridge` (108), `cargo test -p media_recorder --test recording-workflow` (11), UI type/lint/build/Vitest/Playwright, Docker smoke, amd64 image, and container health.
- Re-run after the Phase 6 repairs:

  ```bash
  cargo test -p recording_scheduler node_persistence::tests::reconciliation_adopts_only_live_overlap_members
  cargo test -p recording_scheduler --test recurrence
  cargo test -p web_bridge final_owner_successor_release_stops_the_active_session
  git diff --check
  ```

  Results: scheduler reconciliation regression 1/1, recurrence 6/6, coordinator regression 1/1, and whitespace validation all passed. The recurrence test asserts exact UTC starts for 2028-02-28/29 and 03-01 plus the 2026-12-30, 2027-01-04, and 2027-01-06 weekly year crossing.

## Repair verified during live validation

The initial one-time stop at `2026-07-20T17:42:46Z` exposed a real conflict: scheduler final-owner `Release` advances generation before emitting the intent, while the coordinator still had the previous generation. The occurrence was correctly recorded as `failed` with a retained partial clip; this was a real live failure, not an audio or Podman limitation.

The repair:

- accepts a final group release at the successor generation and releases media using the active generation;
- logs release-generation/owner diagnostics;
- limits reconciliation adoption to occurrences whose planned window contains `now`, preventing future overlap owners from becoming active early.

The latter has a deterministic scheduler regression test. Live Mongo state then showed later overlap owners remained `planned` until their own boundary, and one shared recorder session continued through non-final stops.

## Live scheduled recording evidence

### One-time schedule

The successful one-time rerun used planned start `2026-07-20T17:51:47Z`, schedule `4ce6d6bc-a7ac-46c4-a398-1bb88eff2298`, occurrence `85ebe6bc-d92d-574f-a063-b80b5a21a93b`, and recording `a1924555-600f-4041-82b2-f8af928ab2ee`. Its state sequence was `schedule_accepted → start_pending → active → stop_pending → completed`.

```bash
ffprobe -v error -show_entries format=duration,size \
  -show_entries stream=codec_name -of compact=p=0:nk=1 \
  /home/loidinh/robo-fleet-phase06-recordings/phase-06/live-one-time-rerun/a1924555-600f-4041-82b2-f8af928ab2ee.mp4
```

Result: playable H.264/AAC MP4, 60.159 seconds, 3,674,981 bytes, with its 376-byte manifest in the same directory.

### Overlap and last-owner stop

Two sequentially admitted schedules were created through authenticated Socket.IO (test credential omitted):

- `3fa706b4-cfa0-4c4c-b8bb-0c0a9fea8f39`, `2026-07-20T18:17:00Z`–`18:19:00Z`
- `7624989d-836b-4d0b-af2e-4b54786f8607`, `2026-07-20T18:18:00Z`–`18:20:00Z`

At 18:19, the first occurrence was `completed` while the second was `active` and FFmpeg remained running. At 18:20, both were `completed` with the same recording ID `8e1ff925-360c-4570-bdce-d242c9547400`.

```bash
ffprobe -v error -show_entries format=duration,size \
  -show_entries stream=codec_name -of compact=p=0:nk=1 \
  /home/loidinh/robo-fleet-phase06-recordings/phase-06/live-overlap-repair/8e1ff925-360c-4570-bdce-d242c9547400.mp4
```

Result: playable H.264/AAC MP4, 179.074 seconds, 8,038,147 bytes, plus manifest. The manifest reports `duration_ms: 178438`, zero dropped video frames, and the shared relative path. A preceding three-owner union-window run also completed all three occurrences on one recording ID, with no premature final release.

#### Sanitized command/result transcript

The authenticated Socket.IO harness (credential deliberately omitted) waited for `auth_token`, then sent these two commands one second apart to avoid command-rate limiting:

```javascript
socket.emit("recording_schedule_command", {
  protocol_version: 1, request_id: "<generated UUID>", action: "create",
  schedule: {
    entity_id: "rover-kiwi", title: "phase-06 overlap sequential first",
    enabled: true,
    recurrence: { kind: "one_time", local_start: {
      date: "2026-07-20", time: "18:17", timezone: "UTC"
    }}, duration_ms: 120000,
    relative_directory_template: "phase-06/live-overlap-repair"
  }
});
// Same command: title "phase-06 overlap sequential second", time "18:18".
```

Returned command results were `accepted: true` with schedule IDs `3fa706b4-cfa0-4c4c-b8bb-0c0a9fea8f39` and `7624989d-836b-4d0b-af2e-4b54786f8607`.

The durable-state command and its concise returned state were:

```bash
docker exec robo-mongodb mongosh --quiet robo_fleet_edge_voice_e2e --eval \
'const ids=["3fa706b4-cfa0-4c4c-b8bb-0c0a9fea8f39","7624989d-836b-4d0b-af2e-4b54786f8607"]; db.recording_occurrences.find({schedule_id:{$in:ids}},{schedule_id:1,state:1,attempts:1,_id:0}).toArray().forEach(x=>printjson(x))'
```

At 18:19: first `completed`, second `active`; both referenced `8e1ff925-360c-4570-bdce-d242c9547400`. At 18:20: both `completed`, still with that same recording ID. `docker top robo-orchestra -eo pid,comm,args` showed FFmpeg after the first boundary and no FFmpeg after finalization.

### Feature-disable rollback and manual recording

Orchestra was force-recreated with:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export HOST_RECORDING_PATH=/home/loidinh/robo-fleet-phase06-recordings
export WORKSTATION_RECORDING_SCHEDULER_ENABLED=false
export ROVER_PLATFORM=linux/amd64
export WORKSTATION_AUDIO_DEVICE=sysdefault:CARD=Camera
docker compose -f docker/docker-compose.yml -f docker/docker-compose.workstation.yml \
  --profile orchestra up -d --no-deps --no-build --force-recreate orchestra
```

Podman printed shared-pod cleanup warnings while replacing the host-networked Orchestra container, then created a healthy replacement. A Socket.IO schedule create returned `accepted: false`, `reason: unavailable`, detail `recording scheduler is disabled`. A manual `recording_session_command` start/stop remained accepted and completed recording `8e992e12-c5a2-4dbc-bbf2-867f88ccb250`.

```bash
ffprobe -v error -show_entries format=duration,size \
  -show_entries stream=codec_name -of compact=p=0:nk=1 \
  /home/loidinh/robo-fleet-phase06-recordings/phase-06/manual-rollback/8e992e12-c5a2-4dbc-bbf2-867f88ccb250.mp4
```

Result: playable H.264/AAC MP4, 7.721 seconds, 194,526 bytes, with matching manifest. This verifies the scheduler-disabled/degraded feature path preserves manual recording.

## Residual risk

- The final workstation container is intentionally left with scheduler admission disabled after rollback validation; manual recording remains healthy.
- Rootless Podman emitted shared-pod cleanup warnings on `--force-recreate`, despite each replacement container becoming healthy. Treat this as environment noise to investigate separately, not ARM validation.
- No ARM64 image execution or Raspberry Pi camera/audio hardware acceptance was performed.
- No dedicated axe package is configured; semantic/keyboard coverage remains in the prior Vitest/Playwright gates.

## Unresolved questions

1. Canary duration/count before general scheduler enablement?
2. Is ARM/Raspberry Pi physical acceptance required before release?
