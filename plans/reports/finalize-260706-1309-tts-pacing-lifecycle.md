# Phase 03 Finalize — TTS Pacing and Lifecycle

## Summary

- Implemented `edge_voice` source pacing with a single-slot `TtsPacer`.
- Runtime now emits due chunks before worker/input polling and bounds polling to the next media deadline or 20 ms.
- Completion now waits for final accepted `tts_audio` send before `tts_synthesis_state=Completed`.
- Walkie/playback cancellation clears unsent chunks, discards stale worker audio, and reports one terminal result.
- Metrics now include generated/emitted frames and samples, pending depth, pacing lag, worker backpressure, cancellation count, and terminal reason.

## Validation

- `cargo test -p edge_voice`: 24 passed.
- `cargo clippy -p edge_voice --all-targets --no-deps -- -D warnings`: passed.
- Tester agent validation: `cargo check -p edge_voice` passed and focused pacer tests passed.
- Code review: approved by user after review-cycle fixes; no critical issues remained.

## Onboarding Check

- No new API keys, credentials, services, model downloads, or environment variables are required.
- No UI deployment change is required for this phase.
- Runtime behavior change is internal to rover `edge_voice`; deploy with the normal rover service image/binary refresh.

## Next Steps

- Proceed to Phase 04: playback buffer retry, suppression, and observability.

## Unresolved Questions

- None.
