# Phase 03 Binary Audio Smoke Evidence

Date: 2026-06-30
Status: Automated validation passed; live stack smoke unavailable

## Wire Contract

- Event: `audio_frame`
- Argument 1: metadata only, including protocol version and capture identity
- Argument 2: exactly one S16LE binary attachment
- Standard frame: 16 kHz, mono, 800 scalar samples, 50 ms, 1,600 bytes
- Rust packet test confirms one attachment placeholder, exact attachment bytes, and no JSON `data`
- Browser decoder accepts ArrayBuffer, bounded typed-array views, Blob, transitional number arrays, and legacy JSON frames
- Browser decoder rejects missing, malformed, oversized, mismatched, duplicate JSON/binary, and unsupported-version payloads

## Payload Measurement

Controlled input used deterministic byte values `0..255` repeated across one standard 1,600-byte frame. Counts exclude common Socket.IO and Engine.IO framing.

| Shape | Bytes |
|---|---:|
| Phase 02 legacy metadata plus JSON byte array | 5,911 |
| Phase 03 binary metadata | 257 |
| Phase 03 binary attachment | 1,600 |
| Phase 03 combined | 1,857 |

Reduction: **68.58%**, meeting the phase gate of at least 65%.

## Browser Processing Proxy

Headless Chromium 148 processed 20,000 identical frames after warmup. Legacy work parsed the full JSON byte array and allocated `Uint8Array`; binary work parsed metadata and created a zero-copy view over the attachment.

| Metric | Legacy JSON | Binary |
|---|---:|---:|
| Processing time | 162.20 ms | 11.30 ms |
| Used heap before forced GC | 1,434,916 bytes | 1,394,044 bytes |
| Forced GC time | 3.06 ms | 1.95 ms |

Processing speedup: **14.35x**. Forced-GC time improvement: **1.57x**.

These are controlled shape-cost measurements, not a live stream profile. They isolate JSON array parsing/allocation from the binary path and must not be interpreted as end-to-end latency.

## Validation

- `cargo test -p web_bridge`: 25/25 passed
- `pnpm --filter @robo-fleet/ui test`: 36/36 passed across 4 files
- `pnpm check-types`: 2/2 tasks passed
- `pnpm build`: 2/2 tasks passed for web and native
- Repository `pnpm lint`: exit 0 but executed zero tasks
- Targeted ESLint with the existing missing-browser-globals rule disabled: zero errors, two pre-existing `CameraViewer.tsx` warnings
- `git diff --check`: passed in backend and frontend repositories

## Compatibility

- Frontend-first rollback compatibility is covered by legacy JSON decoder tests.
- Binary backend shape is covered by the Socket.IO packet test.
- Blob normalization is serialized in event order with a four-frame pre-decode cap and explicit drop metric before the unchanged Phase 02 scheduler.
- No separate port, namespace, manager, socket, or feature flag was added.

## Limitations

- No Dora, web bridge, or Robo Control frontend process was listening during this validation.
- Live old-backend/new-frontend and new-backend/new-frontend playback were not exercised.
- Hardware-audible latency remains outside this phase and requires loopback measurement.
- Root ESLint configuration imports a nonexistent `@repo/eslint-config` package and Turbo lint executes zero tasks; lint infrastructure repair is outside this phase.

## Unresolved Questions

- Should live binary playback be required before Phase 3 approval, or handled by the Phase 5 end-to-end rollout gate?
