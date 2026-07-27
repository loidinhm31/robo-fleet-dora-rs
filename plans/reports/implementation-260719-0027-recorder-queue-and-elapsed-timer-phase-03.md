# Phase 03 verification: recorder queue and elapsed timer repair

Date: 2026-07-19

## Automated evidence

- `cargo fmt -p media_recorder -- --check`: passed.
- `cargo test -p media_recorder --lib`: 12/12 passed.
- `cargo test -p media_recorder --test recording-workflow`: 6/6 passed,
  including sustained timestamped A/V coverage, manifest counter checks, and
  final duration >= 1.9 seconds.
- UI focused recording-session-control tests: 5/5 passed.
- Full `@robo-fleet/ui` tests: 23 files, 141/141 passed.
- `pnpm check-types`: passed.
- `pnpm lint`: passed.
- `docker info` and `docker run --rm hello-world`: passed through rootless
  Podman compatibility.

## Live smoke evidence

The documented amd64 workstation compose stack built and reached healthy state
for `robo-orchestra`, `robo-rover-kiwi`, and `robo-mongodb`. An authenticated
Socket.IO client performed one start, held the session for six seconds, and
performed one explicit stop. Sanitized final media metadata:

```text
streams: h264 video 640x480 30fps; aac audio 16000Hz mono
duration_ms: 340 (default queue), 407 (queue 64), 416 (source/view 15 FPS)
bytes_written: 20191 (default queue)
dropped_video: 82 (default queue)
audio_gaps: 0
silence_samples: 32 (default queue)
timestamp_regressions: 0
```

The smoke did not meet the planned >5-second media-duration criterion. The
default `sysdefault:CARD=Camera` value did not match the cpal selector; setting
`WORKSTATION_AUDIO_DEVICE=Camera` selected `hw:CARD=Camera,DEV=0`, downmixed
stereo to mono, and produced sustained audio with zero capture, conversion, or
Zenoh-publish drops. Rover video capture and orchestra receipt were active, but
the recorder finalized only 4--6 video frames. Raising
`RECORDING_QUEUE_CAPACITY` from 8 to 64 and matching source/view cadence at
15 FPS did not materially improve the duration. The remaining live blocker is
the recorder's encoder path, not the missing microphone and not the in-scope
bounded recorder queue repair.

A direct Rust diagnostic with 640x480 frames at real-time cadence reproduced
the short timeline without audio: a six-second feed finalized at about 1.6
seconds, and queue capacity 64 did not improve it. A second direct
`FfmpegSession` diagnostic repeatedly wrote live-sized MJPEG and 16 kHz PCM
through the production two-pipe API from one worker thread and stalled. In
contrast, independent FFmpeg lavfi and MJPEG-pipe benchmarks completed 180
frames at roughly 16--23x real time. Together these results identify the
remaining blocker as dual-input pipe scheduling in
`orchestra/media_recorder/src/ffmpeg-session.rs`: the single recorder worker
can wait for one full nonblocking pipe while FFmpeg needs progress on the
other, so it stops draining the bounded input queue. This is separate from
audio capture and from the in-scope queue-admission repair.

## Review disposition

- Code review: 7.5/10, no critical issues.
- Warnings addressed: stronger manifest/counter assertions, defensive UI fake
  timer cleanup, and this sanitized evidence report.
- Phase remains pending until a workstation smoke with sustained video delivery
  can satisfy the live-duration criterion. Resolving the identified dual-pipe
  scheduling defect requires a separate recorder/FFmpeg architecture change and
  is not included in this plan.

## Unresolved questions

- Whether the recorder should use independent audio/video pipe writers (or a
  coordinated interleaved pump) so one full FFmpeg input cannot starve the
  other. A fix needs its own design, regression test, and review.
