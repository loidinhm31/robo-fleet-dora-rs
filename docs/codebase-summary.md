# Codebase Summary

Snapshot date: 2026-06-24

## Scope

- Distributed rover control stack with Dora and Zenoh.
- Workstation/orchestra side for UI, speech, TTS, and bridge fan-out.
- Rover side for capture, ML inference, control, and rover-side JPEG view output.
- Shared Rust types in `robo_rover_lib`.
- Web UI in `robo-control-app`.

## Current Architecture

- Rover keeps ML and servo processing on capture cadence.
- View/video output is throttled separately with `SOURCE_FPS` and `VIEW_STREAM_FPS`.
- Published video topic is `rover/{entity_id}/video/jpeg/v1`.
- Orchestra decodes and forwards JPEG to the web bridge.

## Documentation Notes

- `ARCHITECTURE.md` is the main system reference.
- `README.md` is the quick-start and feature entry point.
- `SETUP_ENVIRONMENT.md` is the local dependency checklist.
- Phase plans under `plans/` capture benchmark evidence and rollout constraints.

## Recent Evidence

- Phase 2 rover JPEG / Zenoh cutover completed.
- Native split benchmark: 600s, 8986 encoded frames, 7.3ms average encode cost, 0 errors, 14.98 FPS.
- Final hybrid cadence not rerun under constrained 3 CPU / 4 GiB container profile; native split passed and was approved.
