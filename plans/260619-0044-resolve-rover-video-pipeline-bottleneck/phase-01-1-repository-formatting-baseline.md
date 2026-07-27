# Phase 01.1: Repository Formatting Baseline

## Context Links

- [Parent plan](./plan.md)
- [Phase 01](./phase-01-measurement-contract-and-baseline.md)
- Depends on: Phase 01 completed and committed

## Overview

- Priority: P1
- Implementation status: Complete
- Review status: Approved
- Purpose: make formatting deterministic in one isolated, non-functional change before Phase 2.

## Requirements

- Pin the Rust toolchain/rustfmt version used by local development and CI.
- Add repository rustfmt configuration only where current defaults are insufficient.
- Reformat the Rust workspace in one dedicated commit with no semantic edits.
- Add `cargo fmt --all -- --check` to CI and documented pre-commit validation.
- Document that `cargo fmt -- <paths>` does not limit formatting to those paths.

## Implementation Steps

1. Start from a clean tree after Phase 01 is committed.
2. Record current Rust/rustfmt versions and decide the pinned version.
3. Add toolchain/config files, then run full workspace formatting once.
4. Verify the diff is formatting-only through compile, test, and focused review.
5. Add the formatting check to CI and contributor commands.
6. Commit Phase 01.1 separately before starting Phase 2.

## Todo List

- [x] Toolchain and rustfmt component pinned to Rust 1.88.0.
- [x] Full formatting diff isolated from functional work.
- [x] Formatting check and non-broken workspace tests pass after formatting.
- [x] CI formatting guard passes locally.
- [x] Contributor guidance documents safe targeted/full formatting commands.

## Validation Results

- `cargo +1.88.0 fmt --all -- --check`: pass and idempotent.
- `make format-check`: pass.
- Workspace excluding three pre-existing broken packages: 25 unit tests and one
  doctest passed.
- `kokoro_tts`: pre-existing host linker failure; `sonic` and `pcaudio` development
  libraries are unavailable.
- `speech_recognizer`: pre-existing `whisper-rs` generated binding mismatch.
- `visual_servo_controller`: pre-existing exact floating-point equality assertion
  compares `0.30000000000000004` with `0.3`.
- All 39 changed Rust files match the exact Rust 1.88.0 rustfmt output generated
  directly from their pre-format contents; no manual Rust edits are included.

## Success Criteria

- Re-running full formatting produces no diff.
- CI rejects unformatted Rust without modifying files.
- Review confirms zero semantic changes.
- Phase 2 begins from the committed formatting baseline.

## Risks

- Large diff obscures behavior changes: prohibit functional edits and use a dedicated commit.
- Toolchain drift recreates churn: pin the formatter version used by CI.

## Unresolved Questions

- None. Rust 1.88.0 is pinned to match the production Docker builder.
