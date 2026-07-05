# Models Directory

Use the repository root commands:

```bash
make models
make check-models
make models-reset
```

`make models` is the canonical first-clone bootstrap for native x86 and Docker
workflows. It ensures the pinned repo-local model cache under `models/.cache`
and the repo-local ONNX Runtime under `models/.runtime`, validates checksums,
and prints the required `ROVER_ORT_DYLIB_PATH`.

## Managed Layout

```text
models/
├── .cache/
│   ├── yolo/yolo12n.onnx
│   ├── reid/osnet_x0_25.onnx
│   └── sherpa-onnx/
│       ├── asr/
│       │   ├── silero/silero_vad.onnx
│       │   ├── icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/...
│       │   └── sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/...
│       └── tts/
│           └── sherpa-onnx-supertonic-3-tts-int8-2026-05-11/...
├── .runtime/
│   └── onnxruntime-linux-x64-1.16.3/
└── SUPERTONIC-OPENRAIL-M-NOTICE.txt
```

Retired repo-local caches are not part of the target state. After a successful
`make models-reset`, the cache must not contain `ggml`, Piper, or Kokoro model
directories.

## Workflow

### `make models`

- Ensures every pinned current asset.
- Uses temporary `.part` files for downloads.
- Extracts archives into temporary directories, validates required file hashes,
  then installs them into the repo-local cache.
- Provisions `models/scripts/venv` for exported ONNX validation plus YOLO and
  OSNet export tooling.

### `make check-models`

- Read-only validation.
- Reuses the repo-local validator environment created by `make models` or
  `make models-reset`.
- Reports `OK`, `MISSING`, or `CORRUPT` per required file.
- Exits nonzero on any failure.

### `make models-reset`

- Checks available disk space up front.
- Builds a full sibling staging cache.
- Validates the staging cache before swapping it into `models/.cache`.
- Preserves the original cache until the staging cache has passed validation.
- Leaves `models/.runtime` separate from the cache reset.

## Native x86 First Clone

```bash
make models
export ROVER_ORT_DYLIB_PATH="$PWD/models/.runtime/onnxruntime-linux-x64-1.16.3/lib/libonnxruntime.so"
make check-models
```

For rover-native runs, keep `ROVER_ORT_DYLIB_PATH` pointed at the repo-local
runtime above. `pip install onnxruntime` is not sufficient for the Rust vision
crates in this workspace.

## Docker First Clone

```bash
make models
make build-orchestra
make build-rover
```

The Docker images mount `models/.cache/*` as read-only model inputs. The
repo-local ONNX Runtime is for native x86 runs; container runtime validation is
handled by the image entrypoints. The current validated workstation Docker flow
uses:

```bash
docker compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.workstation.yml \
  --profile mongodb --profile orchestra --profile rover-kiwi \
  up -d --build
```

## STT Profiles

Select exactly one startup profile:

| Profile | Language | Bundle |
|---|---|---|
| `en-vad-offline` | English | `icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04` |
| `vi-vad-offline` | Vietnamese | `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09` |

Configuration remains server-owned. Clients cannot submit model paths or switch
profiles at runtime.

## Supertonic Notice

The pinned rover TTS bundle is:

```text
sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
sha256: 82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427
```

See [SUPERTONIC-OPENRAIL-M-NOTICE.txt](./SUPERTONIC-OPENRAIL-M-NOTICE.txt) for
the tracked notice file. Legal approval for redistribution remains outside this
repository's engineering scope.
