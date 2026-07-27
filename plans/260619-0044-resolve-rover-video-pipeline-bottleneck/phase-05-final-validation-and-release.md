# Phase 05: Final Validation and Release

## Context Links

- [Parent plan](./plan.md)
- [Phase 04](./phase-04-latest-frame-ml-isolation.md)
- Depends on: all prior headless milestones passed

## Overview

- Date: 2026-06-19
- Priority: P1
- Implementation status: Complete headless closure 2026-06-25
- Review status: Code review findings resolved 2026-06-25
- Purpose: prove the deployable headless contracts only, document unavailable field-only contracts, and establish rollback evidence.

## Key Insights

- Component tests cannot prove cross-process queue, resource-limit, or browser behavior.
- Headless Linux cannot provide browser-render proof or stable camera/soak evidence; those contracts are field certification, not release-closure blockers.
- Final acceptance must distinguish verified contracts from deferred field-only contracts.
- The UI repository is heavily dirty; validation must distinguish task changes from existing user changes.

## Requirements

- All affected Rust packages compile and test.
- Both Dora dataflows validate and start in documented order.
- UI web/native type checks, lint, and builds pass.
- Headless workstation validation includes preflight, compile/test, graph validation, Dora status, Docker smoke, packet/security/unit coverage, and previously captured direct/split smoke evidence.
- Browser render, constrained resources, long live-camera soak, and physical failure matrix are deferred to field certification.
- Architecture and deployment documentation match actual code after implementation.

## Architecture

Final active path:

```text
camera -> local latest-frame ML -> servo
      \-> demand-gated 15 FPS JPEG -> Zenoh -> binary Socket.IO -> browser
```

No active raw-video cross-machine path or orchestra JPEG encoder remains.

## Related Code Files

- Modify root `ARCHITECTURE.md` and applicable deployment documentation.
- Modify UI `docs/architecture.md`, codebase summary, and shared event documentation.
- Update plan status/checklists and store benchmark report under this plan.
- No unrelated source refactors.

## Implementation Steps

1. Run targeted unit tests for packet parsing, rate limiting, binary serialization, demand transitions, latest-frame replacement, stale rejection, and shutdown.
2. Run `cargo check` and `cargo test` for affected packages, then workspace tests where hardware-independent.
3. Validate normal rover, direct rover, and orchestra Dora graphs.
4. In UI repo, run `pnpm check-types`, `pnpm lint`, `pnpm build`, and native checks available without destructive packaging.
5. Run preflight for camera identity, model/runtime presence, Dora, Docker info, and Docker smoke.
6. Reuse bounded direct/split smoke evidence where the environment cannot keep a browser session or long camera setup stable.
7. Compare headless results with Phase 01 baseline and every phase gate that can run without browser/field hardware.
8. Record deferred field gates: browser render, constrained 3 CPU / 4 GiB soak, 10-minute scenarios, 30-minute full-tracking soak, and physical failure matrix.
9. Review code against final architecture. Update docs for intended drift; fix code for unintended drift.
10. Conduct focused code review for concurrency, payload validation, lock scope, resource cleanup, environment portability, and dirty-tree preservation.
11. Mark plan complete only after fresh headless verification evidence; document exact rollback revision/config.

## Todo List

- [x] Rust checks/tests pass.
- [x] UI checks/builds pass.
- [x] Dora graphs and startup pass.
- [x] Headless preflight passes.
- [x] Direct and split smoke evidence recorded.
- [x] Field-only browser/soak/failure matrix deferred.
- [x] Architecture/docs match headless release state.
- [x] Code review findings resolved.
- [x] Rollback documented.

## Success Criteria

- Raw RGB topic absent; JPEG video average <=15 Mbps.
- Viewer average >=14.5 FPS; capture-to-display p95 <=500 ms. Deferred to field/browser certification.
- Servo >=10 Hz; input age p95 <=150 ms. Deferred to live constrained tracking certification.
- Rover container limit is exactly 3 CPU/4 GiB; average CPU <=2.7 equivalents, peak RSS <=3.5 GiB, no OOM, and no unbounded growth. Deferred to constrained field certification.
- Zero view encoding/network traffic without demand.
- Protocol-level binary payload tests pass; browser runtime confirmation is deferred to field/browser certification.
- Pending ML depth <=1; all intentional drops observable.
- All automated headless checks pass with fresh evidence.

## Risk Assessment

- Container runtime or device mapping prevents representative execution: fail preflight rather than silently switching to an unconstrained process.
- Long soak reveals resource saturation: stop and revise rather than tune around it.
- Dirty UI tree obscures regressions: capture scoped diffs before edits and review only intentional overlap.

## Security Considerations

- Re-run auth/rate-limit tests for new stream command.
- Confirm malformed packet handling cannot allocate beyond bounds or crash bridges.
- Ensure logs/reports contain no JWTs, credentials, environment secrets, or image payloads.

## Next Steps

- Headless release gate is closed for this revision.
- Browser-render, constrained resource envelope, long soak, physical failure matrix, Raspberry Pi 5, and other edge-device execution remain separate future certification tasks.
- Do not infer browser/display or constrained-field performance from this headless closure.

## Unresolved Questions

- None.
