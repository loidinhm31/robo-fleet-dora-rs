## Code Review Summary

### Scope
- Files reviewed: `Makefile`, `docker/Cargo.orchestra.toml`, `docker/Dockerfile.orchestra`, `docker/Dockerfile.rover-kiwi`, `docker/docker-compose.yml`, `docker/docker-compose.workstation.yml`, `docker/scripts/entrypoint-rover.sh`
- Lines of code analyzed: ~800
- Review focus: uncommitted Phase 07 Docker cleanup and local MongoDB changes
- Updated plans: `plans/260704-1318-edge-voice-supertonic-x86/phase-07-docker-cleanup-and-local-mongodb.md`

### Overall Assessment
Build-time cleanup looks directionally correct. The remaining problems are runtime/lifecycle issues: one likely breaks rover vision model resolution in container runs, one makes the new workstation stack nondeterministic at startup, and one leaves stale workstation images behind after `make clean`.

### Critical Issues
- None.

### High Priority Findings
1. `docker/scripts/entrypoint-rover.sh:226` and `docker/scripts/entrypoint-rover.sh:227` still rewrite the old model-path literals (`${HOME}/.cache/...`), but current rover dataflows use `MODEL_PATH: "${ROVER_YOLO_MODEL_PATH:-models/.cache/yolo/yolo12n.onnx}"` and `REID_MODEL_PATH: "${ROVER_REID_MODEL_PATH:-models/.cache/reid/osnet_x0_25.onnx}"` in `rover-kiwi/rover-kiwi-dataflow.yml:119,128` and `rover-kiwi/rover-kiwi-direct-dataflow.yml:95,103`. Result: the sed replacements no-op, the node env falls back to relative `models/.cache/...` paths, and `kornia_capture` will try to load models from the wrong location when detection/tracking is enabled inside the container.

### Medium Priority Improvements
1. `docker/docker-compose.workstation.yml:1-26` adds the amd64 workstation override, but neither it nor `docker/docker-compose.yml:26-76` declares a MongoDB readiness dependency for orchestra. `common/web_bridge/src/main.rs:1460-1475` fails startup immediately if MongoDB is unavailable. `make up-workstation` therefore races MongoDB vs orchestra startup; on a cold start the orchestra container can crash-loop until MongoDB becomes healthy. This is a real reliability gap in the new local MongoDB workflow.
2. `Makefile:213-215` switched `clean` to `$(WORKSTATION_COMPOSE) ... down --rmi local -v`, while `docker/docker-compose.workstation.yml:3` and `docker/docker-compose.workstation.yml:16` now assign explicit `image:` tags. With Compose, `--rmi local` removes only untagged/local auto-generated images, not custom-tagged images. `make clean` will therefore leave `localhost/robo-orchestra:latest` and `localhost/robo-rover-kiwi:latest` behind, which can cause stale workstation images to be reused unexpectedly.

### Low Priority Suggestions
1. `Makefile:167-170` stops MongoDB via Compose but removes it with raw `docker rm -f`. That bypasses the compose wrapper the rest of the file uses and slightly weakens the claimed Docker-compatible/Podman-friendly story. `docker compose rm -f mongodb` would be more consistent.

### Positive Observations
- `docker/Dockerfile.rover-kiwi` correctly stops copying `object_detector`, `reid_extractor`, and `object_tracker` runtime binaries and fixes the dummy prebuild sources for library crates.
- `docker/Dockerfile.orchestra` and `docker/Cargo.orchestra.toml` cleanly remove stale Kokoro references without expanding scope.
- MongoDB loopback binding and named volume usage align with the phase requirements.

### Recommended Actions
1. Fix the rover entrypoint substitutions so they match the current dataflow placeholders, or set `ROVER_YOLO_MODEL_PATH` / `ROVER_REID_MODEL_PATH` explicitly in Compose and stop relying on brittle sed rewrites.
2. Add MongoDB readiness gating for `up-workstation`, preferably in `docker/docker-compose.workstation.yml`, so local MongoDB is only coupled to the workstation override path.
3. Change `make clean` to remove tagged workstation images explicitly or use `--rmi all` if that is acceptable for this repo workflow.
4. Optionally replace the raw `docker rm` in `down-mongodb` with a Compose-native removal command.

### Metrics
- Type Coverage: not measured
- Test Coverage: not measured
- Linting Issues: not run in this review

### Unresolved Questions
- None.
