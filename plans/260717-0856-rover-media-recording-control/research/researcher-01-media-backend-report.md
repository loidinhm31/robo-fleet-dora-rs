# Media recording backend research

## Scope and recommendation

Add one Orchestra Dora node, `media-recorder`, downstream of `orchestra-bridge`. It should consume the bridge's already-validated JPEG and rover-microphone S16LE streams, select one `entity_id`, and stream both into one playable container. Never write individual JPEGs. For MVP, supervise an `ffmpeg` child and write `*.partial.mp4`, then flush, wait, and atomically rename to `*.mp4` after a successful stop.

Keep the filesystem boundary server-controlled: configure `RECORDING_ROOT`; let the UI choose only a normalized relative folder beneath it. Do not accept an arbitrary absolute host path from Socket.IO. In Docker, explicitly bind-mount the desired host folder at `/recordings`.

## Existing contracts found

- Rover camera path: `gst-camera/frame` -> `video-encoder/encoded_frame` -> rover `zenoh-bridge` -> Zenoh `rover/{entity_id}/video/jpeg/v1`.
- Rover view branch is nominally 15 FPS (`VIEW_STREAM_FPS=15`), JPEG quality 80, dimensions 640x480.
- `JpegFramePacket` is versioned and validated in `robo_rover_lib/src/types/video_types.rs`. Orchestra strips its envelope and sends a Dora `BinaryArray` containing one JPEG.
- Video Dora metadata: `entity_id`, `width`, `height`, `encoding=jpeg`, `codec=jpeg`, `compressed_size`, `frame_id`, `capture_timestamp_ms`.
- Rover microphone path: `audio-capture/audio` (16 kHz mono F32, 800 samples/50 ms) -> `audio-converter/audio_output` (S16LE) -> rover `zenoh-bridge` -> Zenoh `rover/{entity_id}/audio/raw`.
- Orchestra validates `PcmFramePacket`, rejects duplicate/regressed frames, and emits its payload as one Dora `BinaryArray`.
- Audio Dora metadata: `entity_id`, `stream_id`, `frame_id`, `capture_timestamp_ms`, `sample_rate`, `channels`, `sample_count`, `format`, `size`. Current normal format is S16LE, 16 kHz, mono.
- `playback_audio_frame` is a separate rover-speaker monitor. It is explicitly described as browser-only and should not silently replace microphone audio.
- `orchestra/orchestra-dataflow.yml` already fans bridge video/audio into `web-bridge`; Dora can fan the same outputs into the recorder.
- Fleet streams are multiplexed by `entity_id`; a recording command must pin a rover for the session and discard frames from all others even if UI selection later changes.
- Workspace web server is `common/web_bridge`, despite stale repository prose referring to `orchestra/web_bridge`.
- Orchestra runtime image currently has no FFmpeg/GStreamer encoder runtime and no writable recordings volume.

## Proposed Dora and web contracts

Add bridge fan-out without changing Zenoh topics or rover code:

```yaml
- id: media-recorder
  inputs:
    video_frame: { source: orchestra-bridge/video_frame, queue_size: 16 }
    audio_frame: { source: orchestra-bridge/audio_frame, queue_size: 32 }
    recording_command: { source: web-bridge/recording_command, queue_size: 8 }
  outputs: [recording_status]
  env:
    RECORDING_ROOT: "${RECORDING_ROOT:-./recordings}"
    RECORDING_MAX_DURATION_SECONDS: "${RECORDING_MAX_DURATION_SECONDS:-3600}"
```

Use JSON shared types in `robo_rover_lib`, not loosely shaped handler-only JSON:

- `RecordingCommand::Start { request_id, entity_id, relative_directory }`
- `RecordingCommand::Stop { request_id }`
- `RecordingStatus { request_id, state, entity_id, recording_id, relative_path, started_at_ms, duration_ms, bytes_written, error }`
- States: `starting | recording | stopping | completed | failed`.
- One concurrent session is sufficient for MVP; reject a second start explicitly.
- Generate the filename server-side (`{entity}-{UTC timestamp}-{short id}.mp4`), use create-new semantics, and never overwrite.

Suggested browser boundary:

- Socket.IO inbound: `recording_start`, `recording_stop`, `recordings_list`.
- Socket.IO outbound: `recording_status`, `recordings_list_result`.
- HTTP: authenticated/signed `GET /api/recordings/{recording_id}` with byte-range support for `<video>` playback and seeking.
- Do not return the host absolute path. Return an opaque ID, display-relative path, and playback URL.
- Apply the same signed-in-session validation and rate limiting used by existing control handlers. Do not add role checks or RBAC. Re-check current web auth plumbing before selecting cookie versus short-lived signed playback URL; an HTML `<video>` element cannot add a normal bearer header.

## Muxing approaches

### 1. FFmpeg child process (recommended MVP)

Feed JPEG bytes and PCM bytes through two inherited pipes; no shell and no raw frame files. Typical inputs are `-f mjpeg` and `-f s16le -ar 16000 -ac 1`; output H.264/AAC MP4. Construct arguments as an array. Never interpolate the user path into a shell command.

Pros: least application code, mature codec/mux support, easy `ffprobe` verification, broadly playable output. Cons: add `ffmpeg` to the Orchestra runtime, supervise child/pipes carefully, and define timestamp/drop policy. Real-time pipe arrival plus wall-clock timestamps is adequate for MVP; use audio resampling/async correction and video VFR/CFR policy explicitly. Close inputs on stop, wait with timeout, and only rename on exit success.

### 2. Rust GStreamer `appsrc` pipeline

Push JPEG and S16LE buffers with PTS derived from `capture_timestamp_ms`, then decode/encode and mux. This gives the cleanest A/V timestamp control and bus/error handling. It adds Rust bindings plus system/plugin packages to the Orchestra build/runtime and codec availability differs by image. Prefer it if precise sync, long recordings, or process-free operation is a hard requirement.

### 3. Native Rust codec/mux crates

Not recommended for this scope. Combining JPEG decode, H.264/VPx encode, AAC/Opus, timestamping, and MP4/WebM finalization creates substantially more code and codec-specific failure modes.

Container-only MJPEG+PCM Matroska avoids re-encoding but is not reliably browser-playable. The requested page makes H.264/AAC MP4 the safer default. WebM VP8/Opus is a viable fallback if deployment codec policy rejects H.264/AAC.

## Exact implementation surface

- Add `orchestra/media_recorder/Cargo.toml` and focused `src/` modules for node loop, recording session/process, config, and safe path resolution (keep modules under repository's 200-line preference).
- Add the package to root `Cargo.toml` and `docker/Cargo.orchestra.toml`.
- Add command/status types to `robo_rover_lib/src/types/recording_types.rs` and export from `robo_rover_lib/src/types/mod.rs`.
- Update `orchestra/orchestra-dataflow.yml` with the node, bridge media fan-out, web command output, and recorder status input.
- Update `common/web_bridge/Cargo.toml` only for actually needed HTTP/range support; extend handlers under `common/web_bridge/src/` and its tests with the events above.
- Update `docker/Dockerfile.orchestra`: copy recorder manifest, build/copy binary, install FFmpeg runtime, create/chown `/recordings`.
- Update `docker/docker-compose.yml` and workstation override as needed: bind `${HOST_RECORDING_PATH:-...}:/recordings`, set `RECORDING_ROOT=/recordings`, and document host ownership/SELinux `:Z` behavior for Fedora/Podman.
- Update the external UI shared types and controls in `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`; it is outside this repository and needs coordinated changes.

## State, failure, and security rules

- Start only after validating authenticated command, active rover, canonical destination, free-space threshold, and encoder availability.
- Bound video/audio queues. Drop oldest video under pressure; preserve audio continuity where possible and report counters.
- Establish session zero from capture timestamps; reject timestamp regression and detect stream resets. Decide whether missing frames duplicate prior video, make VFR gaps, or shorten output; do not let audio drift silently.
- If microphone audio is unavailable, use a documented policy: fail start, record video-only, or synthesize silence. Avoid an indefinitely blocked FFmpeg second input.
- On Dora stop/SIGTERM: stop accepting frames, close pipes, finalize with timeout, kill only the owned child if necessary, and emit final status.
- Restrict extensions and path components; reject absolute paths, `..`, symlink escape, NUL/control characters, and excessively long names. Canonicalize the existing parent under `RECORDING_ROOT`.
- Cap duration and optionally bytes; check disk space; never expose directory traversal through listing/playback APIs.
- A `.partial.mp4` is an encoded container, not raw JPEG persistence. Clean stale partials by age on startup or report them; do not delete arbitrary user files.

## Verification plan

- Unit: path containment/symlink escape, command state machine, entity pinning, metadata parsing, timestamp normalization, collision-free filenames, size/duration caps, child argument construction.
- Process integration: synthetic valid JPEG sequence + 16 kHz mono S16LE; start/stop; assert no `.jpg`; use `ffprobe` to assert one video and one audio stream, duration tolerance, resolution, H.264/AAC codecs.
- Fault integration: no audio, dropped/regressed frames, FFmpeg missing/crash/hang, unwritable/full destination, second start, Dora stop mid-recording.
- Web integration: unauthorized controls/playback denied, malicious paths rejected, status correlated by `request_id`, list confined to root, byte-range `206` and seeking work.
- Container smoke: build Orchestra image, verify `ffmpeg -version`, writable mounted `/recordings`, complete a recording, replay from UI, and verify host file ownership under Docker-compatible Podman.

## Unresolved questions

1. Does “sound” mean rover microphone only, rover speaker monitor only, or a mix of both?
2. Must UI users enter an arbitrary absolute host path, or is a deployment-configured root plus relative folder acceptable (recommended)?
3. Is one active recording globally sufficient, or are simultaneous per-rover recordings required?
4. If audio is unavailable at start, should recording fail, continue video-only, or add silence?
5. Required browser/platform support: is H.264/AAC MP4 acceptable, including its deployment/licensing implications?
