# Brainstorm Reconciliation Report

Date: 2026-07-03
Source: `plans/reports/brainstorm-260702-1750-central-sherpa-stt.md`

## Finding

The brainstorm is technically useful but no longer authoritative for all scope. Repository history shows a later design decision and three completed implementation phases.

## Requirement Mapping

| Brainstorm item | Current decision/state | Plan treatment |
|---|---|---|
| Reuse rover S16LE transport | Implemented centrally | Preserve; validate end to end |
| Independent fleet stream state | Implemented in central runtime | Validate concurrent sources |
| Official Sherpa Rust API | Implemented, pinned `1.13.3` | Keep |
| Online English backend | Superseded | Out of scope |
| VAD/offline English + Vietnamese | Implemented runtime/model base | Complete transport and validate |
| Runtime/per-rover profile control | Superseded by global startup profile | Out of scope |
| Partial hypotheses | Superseded by final-only contract | Out of scope |
| Browser microphone central input | Selected in later plan | Complete bounded private path |
| Source-aware command target | Not implemented | Required safety phase |
| AI command interpreter | Deferred | Out of scope |
| Remove edge placeholder | Pending | Only after validation gate |
| Remove Whisper/GGML wiring | Partly complete | Finish only after validation gate |
| Rover TTS suppression/AEC | Deferred debt | Document; do not implement here |

## Completed Work To Reuse

- Contract/architecture baseline: commit `3d8c57c`.
- Sherpa runtime/models: commit `bbbe251`.
- Central VAD/offline recognizer: commit `96b54e6`.
- Existing detailed Phase 04–08 drafts remain useful input but are untracked and must not be silently overwritten.

## Plan Boundary

Create a residual five-phase plan. Do not repeat completed contracts, model provisioning, or central runtime implementation. Do not restore removed scope without a new decision.

## Unresolved Questions

None.
