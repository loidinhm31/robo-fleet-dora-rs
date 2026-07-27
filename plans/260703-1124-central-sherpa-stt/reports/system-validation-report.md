# Central Sherpa STT System Validation Report

## Decision

**Phase 04 is approved with accepted follow-up backlog.** On 2026-07-04 the user approved
the phase after manual testing, noting that STT quality differs between `Browser Voice Commands`
and `Fleet Speech Transcription`, with better quality on rover, and that overall detection
still needs improvement. Core contracts, model loading, format equivalence, source isolation,
privacy, and target routing passed. The soak loss, missing bilingual accuracy corpus, and
missing profile latency/resource measurements remain open backlog items rather than release
blocking gates for this phase.

## Environment

| Item | Value |
|---|---|
| Date | 2026-07-03 |
| Host | x86_64 Linux |
| CPU | AMD Ryzen 7 8840U, 8 cores / 16 threads |
| Memory | 23 GiB RAM, 8 GiB swap |
| English model | `icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04` |
| Vietnamese model | `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09` |
| VAD model | Silero VAD |
| Runtime profile | Two decode threads, decode queue capacity 8 |

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| Rust units/contracts | PASS | 142 passed, 0 failed; 3 model tests ignored by the deterministic suite and run explicitly for both profiles (6 passes) |
| UI tests/checks/builds | PASS | 68 tests; lint clean; 2 uncached type-check tasks and 2 production builds passed |
| Release linkage | PASS | 41 MiB release binary; only standard C/C++ runtime libraries are dynamically required |
| English model load/decode | PASS (functional only) | Bundled 6.625 s fixture decoded with RTF 0.014 |
| Vietnamese model load/decode | PASS (functional only) | Bundled 3.740 s fixture decoded with RTF 0.015 |
| Labeled acoustic accuracy corpus | BACKLOG | No equivalent labeled Vietnamese corpus or silence/noise/clipping/short-speech command set was available |
| F32/S16LE equivalence | PASS | Normalized transcript matched exactly for each bundled profile fixture |
| Concurrent source isolation | PASS | Two rover streams and one browser stream retained distinct stream/source/target identities |
| Browser privacy | PASS | Owner received 85/85 browser finals; second authenticated client received none; rover finals were broadcast to both |
| Target snapshot | PASS | Browser selection changed during speech but final retained the start-time `rover-kiwi` target |
| Rover command routing | PASS | Rover A speech routed only to rover A while rover B was selected; no selected-rover fallback occurred |
| Missing model/reconnect | PASS | Graph remained available, emitted a sanitized missing-model status, and replayed authoritative ready status after reconnect |
| Malformed/sequence fault | PASS | Malformed and sequence-fault paths did not emit a transcript; status remained available |
| 10-minute concurrent soak | BACKLOG | 85/85 browser finals, rover stream finals 66/66 and 53/66; 13 rover utterances did not finalize |
| RTF below 1.0 | PASS (fixture decode) | English 0.014; Vietnamese 0.015 |
| Endpoint-to-final P95 <= 2 s/profile | BACKLOG | Not measured with a repeated labeled corpus for both profiles |
| Bounded resources/state | BACKLOG | Runtime stayed healthy for 734.180 s with no central queue drops, but a clean final-run CPU/RSS series was not captured |

## Commands

```bash
cargo test -p robo_rover_lib -p central_speech_recognizer -p web_bridge \
  -p command_parser -p orchestra_zenoh_bridge

STT_PROFILE=en-vad-offline \
STT_MODEL_ROOT="$PWD/models/.cache/sherpa-onnx/asr" \
cargo test -p central_speech_recognizer --test model_loading -- --ignored --nocapture

STT_PROFILE=vi-vad-offline \
STT_MODEL_ROOT="$PWD/models/.cache/sherpa-onnx/asr" \
cargo test -p central_speech_recognizer --test model_loading -- --ignored --nocapture

cargo check -p orchestra_zenoh_bridge --example replay-pcm-fixture
cargo build --release -p central_speech_recognizer
ldd target/release/central_speech_recognizer
readelf -d target/release/central_speech_recognizer
```

Frontend checks were run from
`/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`:

```bash
pnpm --filter @robo-fleet/ui test
pnpm lint
pnpm check-types --force
pnpm build --force
```

The soak used `orchestra_zenoh_bridge --example replay-pcm-fixture` to publish a mono
16 kHz PCM16 fixture in real-time 50 ms frames. Each rover used one persistent stream ID
and monotonic frame sequence for 66 repetitions. Every repetition appended two seconds of
silence. Two authenticated Socket.IO clients ran concurrently with 85 browser captures.

## Model Evidence

English fixture transcript:

```text
AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
```

Vietnamese fixture transcript:

```text
RỒI CŨNG HỖ TRỢ CHO LÂU LÂU CŨNG CHO GẠO CHO NÀY KIA
```

These runs prove native model loading, VAD segmentation, decoding, and transport-format
equivalence. They do not establish command-corpus accuracy because the Vietnamese bundle
does not include a ground-truth transcript for the selected fixture and no representative
acoustic corpus was supplied. The user also reported a manual quality gap between browser
and rover transcription that should be treated as a separate tuning task.

## Live and Soak Evidence

Final soak dataflow UUID: `019f28ce-c415-74d5-b5e2-c7d698b8ac4d`.

Central shutdown counters:

```text
uptime_ms=734180 frames=34021 validation_errors=0 sequence_resets=115
speech_segments=205 queue_drops=0
```

Web bridge shutdown counters:

```text
browser queue_drops=0 terminated_streams=85 expired_streams=85 late_transcriptions=0
audio deliveries_emitted=45652 sequence_drops=10 client_drops=10
routing_errors=0 emit_errors=0 errors=0
```

Both rover publishers completed 66 repetitions. One stream emitted all 66 finals; the other
emitted 53. The central process and web bridge shut down cleanly, but the asymmetric missing
finals and sequence resets remain unexplained and are tracked as follow-up work.

During earlier validation, browser audio could arrive before its Dora start control because
the two inputs are independent edges. The central session manager now keeps a bounded
eight-frame pre-start buffer, validates identity/sample rate/sequence, drains it on start,
and clears it on stop or shutdown. Unit coverage verifies the ordering, capacity, metadata,
and cleanup behavior.

## Backlog Follow-up

1. Improve STT quality consistency between `Browser Voice Commands` and `Fleet Speech Transcription`, with focus on the weaker browser path and overall detection quality.
2. Diagnose the 13 missing rover finalizations and 115 sequence resets under concurrent replay.
3. Run labeled English and Vietnamese command corpora including silence, noise, clipping, and short speech.
4. Capture endpoint-to-final P50/P95/P99, CPU, and RSS time series for both profiles on target Orchestra hardware.
5. Repeat the 10-minute soak after the quality and replay issues are addressed.
