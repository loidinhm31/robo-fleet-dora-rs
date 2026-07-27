# Phase 01 — Recorder delete contract and saved timestamp

## Context links

- [Architecture recording invariants](../../ARCHITECTURE.md#manual-fleet-media-recording-and-playback)
- `orchestra/media_recorder/src/{clip-catalog.rs,session-manager.rs,ffmpeg-session.rs}`
- `common/web_bridge/src/{main.rs,recording-playback.rs}`
- `robo_rover_lib` recording command/status types and orchestra dataflow YAML

## Overview

Priority P1; complete. Added a versioned delete request/result through the existing authenticated request-correlation path. Filesystem mutation remains inside `media_recorder`; web bridge validates auth/rate limits and revokes tickets.

## Requirements and architecture

1. Add opaque recording-id delete request/result types and Socket.IO/Dora names. Reject missing, malformed, active, partial, or stale requests; return bounded reason codes.
2. In the catalog, resolve only allowlisted relative paths under `RECORDING_ROOT`, use no-follow/regular-file checks, verify manifest ↔ MP4 identity, and delete the finalized pair. Hide the pair from listing before unlink, fsync the parent directory, and return an idempotent not-found result on repeats.
3. Revoke all playback tickets for the recording ID before reporting success. Never accept a browser path or absolute filesystem path.
4. Add timestamp config (enabled by default) to the existing FFmpeg spec. Burn UTC `YYYY-MM-DD HH:MM:SS UTC` in saved video only, tied to capture/session time; validate `drawtext`/font support at startup and fail a start clearly if enabled but unavailable. Preserve JPEG bytes and live demand flow.
5. Add tests for contract JSON, auth/correlation/rate limit, active and traversal rejection, symlink/identity checks, paired deletion/restart absence, ticket invalidation, FFmpeg args, and extracted-frame text.

## Related files

Modify: recorder config/spec/session/catalog/tests; shared Rust recording types; web bridge handlers/state/ticket registry; orchestra dataflow if a Dora port is added. Do not modify rover capture ownership.

## Risks/security

Deletion is destructive: require authenticated admission, bounded rate, explicit UI confirmation, and backend revalidation immediately before unlink. Handle partial pair failures with a clear terminal error and an observable orphan cleanup path. Avoid logging absolute paths or ticket secrets. Burn-in adds CPU and must remain behind bounded recorder queues.

## Success and next steps

All backend/media tests pass; ffprobe accepts output and a decoded frame visibly contains UTC text; viewer demand regression proves recording stop cannot stop an active viewer. Then hand the stable event contract to Phase 02.
