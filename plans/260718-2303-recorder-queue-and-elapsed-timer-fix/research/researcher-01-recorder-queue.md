# Recorder queue research

## Finding

`BoundedInputs` is one FIFO with a total capacity of eight. A full queue of
audio rejects all new video, so the video timeline stops advancing while input
media is still healthy.

## Recommended design

Keep one bounded queue and its current newest-video replacement policy. When a
video arrives to a full queue with no queued video, evict the oldest audio and
admit the video. This preserves the total memory bound and avoids introducing
per-stream capacities or a second scheduler.

## Evidence

- `orchestra/media_recorder/src/session-manager.rs:81-115` admits/rejects
  queue inputs.
- `orchestra/media_recorder/src/session-manager.rs:342-548` has one FIFO
  consumer.
- `orchestra/media_recorder/src/session-manager.rs:584-586` derives duration
  from the last admitted video timestamp.
- `orchestra/orchestra-dataflow.yml:152-158` independently bounds Dora ingress.

## Required tests

- Full audio-only queue then video: video remains queued; total capacity holds.
- Full mixed queue: oldest video is replaced by the latest video.
- Existing newest-video behavior remains unchanged.
- Sustained A/V recording: completed duration tracks the capture interval and
  manifests expose no video loss caused solely by audio backlog.

## Risks

The policy intentionally loses audio under overload to retain video continuity.
Do not increase queue limits first; that only hides the admission defect.

## Unresolved questions

- Optional later observability can distinguish displaced audio from rejected
  input; it is not required to correct admission.
