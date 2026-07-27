# Phase 03: Verify recording regressions

## Context links

- [Parent plan](./plan.md)
- [Queue research](./research/researcher-01-recorder-queue.md)
- [Elapsed-time research](./research/researcher-02-elapsed-time.md)

## Overview

- Date: 2026-07-18
- Description: Prove queue admission, final media duration, and timer lifecycle.
- Priority: P1
- Implementation status: Blocked — live sustained A/V duration criterion unmet.
- Review status: Deferred until the live encoder-path failure is resolved.

## Key Insights

- Prior focused recorder tests do not saturate simultaneous A/V input.
- UI test execution currently needs the external checkout dependencies installed.
- Existing recent manifests and ffprobe evidence show approximately 1.6-second
  clips and high video-drop counts despite healthy upstream video.

## Requirements

- Validate Rust unit/integration coverage after Phase 01.
- Validate UI timer coverage, type checking, and lint after Phase 02.
- Run a controlled live recording longer than five seconds and inspect media and
  manifest outcomes without exposing host paths, credentials, or tickets.

## Architecture

Validation follows the live path: rover JPEG/PCM → orchestra bridge → recorder
→ finalized MP4/manifest → ticketed UI playback. It must distinguish a true
admission drop from an explicit user stop or dataflow shutdown.

## Related code files

- Verify: `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/tests/recording-workflow.rs`.
- Verify: `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/src/session-manager.rs`.
- Verify: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/recording-session-control.test.tsx`.

## Implementation Steps

1. Run focused recorder tests and relevant workspace formatting/type checks.
2. In the UI checkout, install declared dependencies if absent, then run focused
   Vitest, `pnpm check-types`, and lint.
3. Start the documented native or Podman-compatible workflow with
   `XDG_RUNTIME_DIR` exported; record sustained active rover A/V for at least
   five seconds before issuing one explicit stop.
4. Inspect final MP4 duration/streams with ffprobe and matching manifest fields:
   duration, dropped video, gaps, and bytes. Report only sanitized metadata.
5. Confirm UI elapsed advances during recording and freezes at finalization;
   repeat after reconnect if practical.
6. Run a code review before handoff; document any remaining encoder-throughput
   issue separately from this queue-admission repair.

## Todo list

- [x] Run backend regression suite: `cargo test -p media_recorder` 20/20.
- [x] Run UI tests, type checks, and lint: focused 5/5, full 141/141.
- [ ] Complete sustained live A/V smoke. The final deployed scheduling attempt
  still misses the five-second media-duration criterion.
- [x] Inspect sanitized MP4/manifest results.
- [x] Independently recheck the final recorder diff, format, and regression suite.
- [ ] Complete handoff code review after the live criterion passes.

## Verification result

Automated coverage is passing. The final implementation gives FFmpeg separate
bounded blocking audio/video writer pumps, cancels stalled pumps safely during
timeout finalization, and disables unnecessary input analysis for the two fixed
pipe formats. Its final independent check passed `cargo test -p media_recorder`
(20/20), scoped Rust formatting, and diff checks. The existing UI results also
remain green: focused timer tests 5/5, full UI tests 141/141, type check, and
lint.

The Podman-backed amd64 stack was rebuilt for the scheduling implementation and
reached healthy state. An authenticated Socket.IO client started a recording,
held it active for 6.5 seconds, issued one explicit stop, and received a
`completed` terminal status. The final MP4 contains H.264 video and AAC mono
audio, but sanitized artifact metadata is still insufficient:

```text
duration_ms: 628
bytes_written: 14463
dropped_video: 81
audio_gaps: 0
silence_samples: 10048
timestamp_regressions: 0
```

The duration and drop count show that the remaining fault is still on the live
encoder/input-consumption path. It is no longer appropriate to mark the phase
complete or hand the change off as a successful recorder repair.

See [Phase 03 verification report](../reports/implementation-260719-0027-recorder-queue-and-elapsed-timer-phase-03.md)
for sanitized metadata and command results.

## Success Criteria

- All focused automated tests pass.
- The controlled clip duration is materially longer than one second and matches
  the capture interval through explicit stop.
- No video loss is attributable to an audio-only full recorder queue.
- UI elapsed visibly advances, then freezes to final recorder duration.

## Risk Assessment

- Missing UI dependencies block validation; install them only inside the UI
  checkout, not this Rust workspace.
- Physical capture and FFmpeg throughput can introduce unrelated drops; preserve
  counters and isolate them in the report.

## Security Considerations

- Do not log auth tokens, playback tickets, absolute recording paths, or media.
- Preserve authenticated admission and ticketed playback checks.

## Next steps

Start a focused encoder-path investigation. Keep the current queue-admission and
UI timer fixes intact; identify why the live FFmpeg process stops consuming
video after approximately 0.6 seconds despite healthy inputs and separate pipe
writers.

## Unresolved questions

- Is the remaining stall caused by FFmpeg's input demux/thread behavior, Dora
  delivery, or a recorder-worker scheduling interaction that is not observable
  in the current manifest counters?
