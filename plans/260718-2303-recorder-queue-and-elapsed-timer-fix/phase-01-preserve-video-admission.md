# Phase 01: Preserve video admission in bounded recorder queue

## Context links

- [Parent plan](./plan.md)
- [Queue research](./research/researcher-01-recorder-queue.md)
- `ARCHITECTURE.md` — manual recording bounded-memory invariant

## Overview

- Date: 2026-07-18
- Description: Prevent a full audio backlog from rejecting all subsequent video.
- Priority: P1
- Implementation status: Done
- Review status: Done
- Completion timestamp: 2026-07-19 00:03:22 +07

## Key Insights

- Current queue capacity is shared across audio and video.
- A full audio-only queue rejects video, freezing recorded-video PTS.
- Recording duration follows the last admitted video timestamp, not wall time.

## Requirements

- Keep one total bounded queue and existing FIFO consumption.
- Admit a new video when the full queue has only audio by evicting its oldest
  audio item.
- Preserve the existing replacement of the oldest queued video by a newer one.
- Do not change `RECORDING_QUEUE_CAPACITY`, dataflow queue sizes, manifests, or
  external contracts.

## Architecture

`orchestra-bridge` ingress stays independently bounded. Inside each recorder
session, video becomes preferred only at full capacity: the queue sacrifices an
old audio item rather than losing the frame needed for video timeline progress.
The worker remains the single FIFO consumer and FFmpeg ownership remains unchanged.

## Related code files

- Modify: `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/src/session-manager.rs` — full-queue video admission and unit tests.
- Modify: `/mnt/data/ws/sharing/robo-fleet-dora-rs/orchestra/media_recorder/tests/recording-workflow.rs` — sustained A/V regression coverage if deterministic at the session boundary.

## Implementation Steps

1. Read the current `BoundedInputs::video` and `audio` admission paths; preserve
   their lock scope, capacity check, and no-backpressure behavior.
2. At full capacity, select an oldest queued video for replacement; if none
   exists, select the oldest queued audio. Enqueue the new video in both cases.
3. Increment `dropped_video` only when an existing video is displaced or a video
   cannot be admitted for a real invariant violation; an audio displacement is
   not a video drop.
4. Add queue-level tests for audio-only saturation, mixed saturation, capacity,
   newest-video retention, and existing FIFO behavior.
5. Add or extend deterministic session coverage that feeds sustained concurrent
   A/V data and asserts final duration follows the video capture interval.

## Todo list

- [x] Implement video-preferred full-queue admission.
- [x] Add saturation regression tests.
- [x] Confirm no queue capacity or wire-format change.

## Success Criteria

- A full audio-only queue retains an incoming video frame.
- Newest video still replaces the oldest queued video under video pressure.
- Sustained A/V fixture produces a clip whose duration exceeds one second and
  follows its final video PTS.

## Risk Assessment

- Audio may be dropped under overload. This is intentional and already possible;
  it prevents the more visible complete video-timeline starvation.
- A slow FFmpeg process can still drop media; this change fixes admission only.

## Security Considerations

- No input-validation, storage, process-spawn, or authorization surface changes.

## Next steps

Implement Phase 02 independently; both changes meet in Phase 03 validation.

## Unresolved questions

- None blocking.
