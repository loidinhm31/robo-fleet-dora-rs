# Phase 02 — Model Cache Reset and Bootstrap

## Context Links

- [Parent plan](./plan.md)
- [Supertonic research](./research/01-supertonic-rust-production-comparison.md)
- [Model documentation](../../models/README.md)
- Depends on: Phase 01 contract names/default paths

## Overview

| Field | Value |
|---|---|
| Date | 2026-07-04 |
| Description | Create one first-clone model workflow and an atomic full repo-local cache reset. |
| Priority | P1 |
| Implementation status | Completed |
| Review status | Approved 2026-07-04 |
| Recommended model | GPT-5.4; GPT-5.4-mini only for download/check logs |
| Estimated effort | 6h |

## Key Insights

- Current model setup is split between Makefile, Docker scripts, and stale individual downloaders.
- Cache contains 216 MiB retired GGML and 78 MiB retired Piper data.
- Reset must not destroy the last valid cache if a large download fails.
- Native vision needs ONNX Runtime 1.16.x; `edge_voice` links Sherpa statically.

## Requirements

### Functional

- `make models`: idempotently ensure every current asset.
- `make models-reset`: build a complete replacement and remove all retired repo-local assets.
- `make check-models`: no downloads; exact missing/corrupt report; nonzero on failure.
- First-clone instructions must cover native x86 and Docker paths.

### Non-functional

- Pin URLs, archive names, required paths, and checksums in one sourced manifest.
- Download to `.part`; extract into temporary directory; validate before move.
- Keep cache ignored; track manifest, scripts, license/notices, and documentation.
- Never delete outside repository `models/.cache`.

## Architecture

```text
model-manifest.sh
      ├── setup-models.sh ensure -> models/.cache
      ├── setup-models.sh reset  -> models/.cache-reset-*/ -> validate -> swap
      ├── setup-models.sh check  -> read-only validation
      └── Docker entrypoint checks -> same required-file definitions
```

Target cache:

```text
models/.cache/
├── yolo/yolo12n.onnx
├── reid/osnet_x0_25.onnx
└── sherpa-onnx/
    ├── asr/...
    └── tts/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/...
```

Store repo-local ONNX Runtime under `models/.runtime/onnxruntime-linux-x64-1.16.3`; do not mix it into model reset.

## Related Code Files

| Action | Absolute path | Purpose |
|---|---|---|
| Create | `/mnt/data/ws/sharing/robo-fleet-dora-rs/models/scripts/model-manifest.sh` | Single model/runtime catalog |
| Create | `/mnt/data/ws/sharing/robo-fleet-dora-rs/models/scripts/setup-models.sh` | Ensure/reset/check workflow |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/Makefile` | Canonical targets |
| Modify | `/mnt/data/ws/sharing/robo-fleet-dora-rs/models/README.md` | First-clone instructions |
| Delete | `/mnt/data/ws/sharing/robo-fleet-dora-rs/models/scripts/download_sherpa_vits_models.sh` | Retired Piper downloader |
| Delete | `/mnt/data/ws/sharing/robo-fleet-dora-rs/models/scripts/download_kokoro_models.sh` | Retired Kokoro downloader |
| Replace | `/mnt/data/ws/sharing/robo-fleet-dora-rs/docker/scripts/download-models.sh` | Thin compatibility wrapper |

## Implementation Steps

1. Define bundle constants, URL, SHA-256, and required relative files in the manifest.
2. Move existing ASR profile constants behind the shared manifest without duplicating names.
3. Implement `ensure`: skip only bundles whose complete required-file set validates.
4. Implement Supertonic download with exact SHA-256 before extraction.
5. Implement x86_64 ONNX Runtime 1.16.3 repo-local setup and print the required `ROVER_ORT_DYLIB_PATH`.
6. Implement `check`: concise per-bundle results and aggregate exit status.
7. Implement `reset`: create sibling staging cache, populate all current assets, run check against staging, rename old cache to backup, install staging, then remove backup.
8. Add signal/error cleanup that preserves the original cache until successful validation.
9. Retire GGML/Piper/Kokoro setup paths and stale model checker.
10. Execute full reset once; use GPT-5.4-mini to summarize bounded download/check logs if needed.

## Todo List

- [x] Shared manifest created
- [x] Ensure mode implemented
- [x] Check mode implemented
- [x] Atomic reset implemented
- [x] Supertonic checksum verified
- [x] Repo-local ORT installed/verified
- [x] Retired scripts removed
- [x] Full reset completed
- [x] Documentation rewritten

## Success Criteria

- Fresh clone reaches a complete cache through one documented command.
- Interrupted reset leaves the original valid cache usable.
- Final cache has no `ggml`, Piper, or Kokoro directories.
- `make check-models` validates ASR, YOLO, ReID, Supertonic, and native ORT.
- Re-running `make models` performs no unnecessary downloads.

## Risk Assessment

- Risk: reset temporarily needs double disk space. Mitigation: check free space before starting and report required estimate.
- Risk: Python export dependency downloads are large. Mitigation: reuse one isolated venv during a run and report its path/size.
- Risk: archive layout changes. Mitigation: exact name/checksum/required-file validation.

## Security Considerations

- HTTPS only; fail on checksum mismatch.
- Reject archive traversal by inspecting paths or extracting with safe layout assumptions plus post-validation.
- Do not execute downloaded model content.
- Do not print credential-bearing proxy URLs.

## Review Notes

- 2026-07-04 final review: `bash -n` passed for the Phase 02 shell scripts, local `make check-models` passed, and the latest reported `make models-reset` passed after the reset-backup, Silero staging, repo-local validator, OSNet export, and README scope fixes.
- Historical ignored `models/.cache-reset-*` directories from earlier failed runs may still exist on disk. Latest successful runs did not leave new reset/backup directories behind.

## Next Steps

- Proceed to [Phase 03](./phase-03-edge-voice-engine.md).
- Optional maintenance: prune pre-fix historical `models/.cache-reset-*` directories if disk pressure matters.
