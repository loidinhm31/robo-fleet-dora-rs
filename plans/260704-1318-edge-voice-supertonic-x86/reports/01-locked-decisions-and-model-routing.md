# Locked Decisions and Model Routing

Date: 2026-07-04
Status: approved for planning

## Product and Deployment Decisions

- Keep `edge_voice` in the rover-role dataflow.
- Run Orchestra and Rover as separate Dora dataflows on the same x86_64 host.
- Native validation commands remain `dora up`, Orchestra `dora start ... --attach`, then Rover `dora start ... --attach`.
- Supply MongoDB from a loopback-only local container.
- Run UI live E2E from `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`.
- Run a separate amd64 Docker stack verification after native E2E.
- No Raspberry Pi, ARM build, thermal, or Pi performance acceptance.

## Runtime Decisions

- Supertonic 3 only; no fallback engine.
- English and Vietnamese share one resident model.
- Global runtime config; no persistence; restart returns defaults.
- Defaults: English, M1/SID 5, speed 1.0, steps 8, volume 0.8.
- Queue capacity 8; one synthesis worker.
- Keep current 1,000-character limit; strip markup/expression tags in this scope.
- Walkie audio preempts/cancels TTS. Reject new TTS while walkie is active.
- Show a visible accessible UI alert for TTS rejection/interruption.
- Suppress rover microphone publication during speaker playback plus 400 ms.
- `audio_playback` remains the only physical speaker owner.

## Model Cache Decision

- Provide normal idempotent `make models`.
- Provide full repo-local cache replacement through `make models-reset`.
- Build replacement cache in staging; validate before swap.
- Final repo cache contains current ASR, YOLO, ReID, Supertonic only.
- Do not delete `$HOME/.cache` or other external paths.

## GPT Routing Policy

| Work | Primary | Optional subagent |
|---|---|---|
| Architecture/contracts | GPT-5.5 | None |
| Supertonic FFI/concurrency | GPT-5.5 | GPT-5.4 for tests |
| Playback/resampling | GPT-5.5 | GPT-5.4 for fixtures |
| Bridges, UI, Docker | GPT-5.4 | GPT-5.4-mini for inventory/logs |
| Native/Docker/E2E logs | GPT-5.4 | GPT-5.4-mini |
| Cross-subsystem failure | GPT-5.5 after evidence | GPT-5.4-mini for bounded collection |

### Mini-agent output contract

Each GPT-5.4-mini log task must be bounded by command and line/byte limit. Required response:

```text
command:
exit_code:
first_causal_error:
evidence:
affected_subsystem:
likely_root_cause:
recommended_next_check:
```

Do not let the mini agent declare completion, edit broadly, or infer success from absence of obvious errors.

## Dirty Worktree Constraint

Backend has user changes in `.gitignore`, command parser, Orchestra bridge, and plan artifacts. UI has user changes in `.gitignore` and `apps/web/package.json`. Every phase must inspect and merge overlapping diffs. No reset, checkout, broad formatter, or unrelated cleanup.

## Unresolved Questions

- None affecting implementation sequence.
