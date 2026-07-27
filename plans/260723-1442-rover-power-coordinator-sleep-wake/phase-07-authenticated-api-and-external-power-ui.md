# Phase 07 — Authenticated API and External Power UI

## Context links

- Parent: [plan.md](./plan.md)
- Design: [Policy and Effective State](../../docs/power-coordinator-architecture.md#policy-and-effective-state)
- Input: [voice/resource/UI research](./research/researcher-02-voice-resource-ui.md)
- External UI: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`
- Dependencies: Phases 01–05 live/status/history and scheduler integration.

## Overview

- Date: 2026-07-24
- Description: expose authenticated exact-Rover power controls, authoritative live state, and filtered 90-day history in shared web/Tauri UI.
- Priority: P1
- Implementation status: Pending
- Review status: Pending

## Key Insights

- UI Wake is not Awake: when sleeping it atomically sets policy to Auto and creates bounded UI demand.
- Live Socket.IO status outranks Mongo projection/history.
- Existing lifecycle/recording hooks already model auth, reconnect reset, entity pinning, stale rejection, and non-optimistic commands.

## Requirements

- Socket.IO: `power_policy_set`, `power_wake`, result events, `power_status`, `power_transition`, `power_history_query/result`.
- Authenticate, rate-limit, pin exact active entity, server-derive actor, validate epoch/request/expiry, and bound pending/history queues.
- Disable Wake when coordinator status is disconnected/stale; no optimistic Ready.
- Show policy, requested/effective profile, transition state, epoch, active demand summaries, resource freshness, last reason/time.
- Cold Mongo projection is visibly Historical/Stale and must disappear when newer live status arrives.
- History filters: 90 days by entity, time, transition, source, result/reason; stable cursor and dedupe by event ID.

## Architecture

`UI -> web_bridge power socket -> local Orchestra/direct Rover coordinator`. Live coordinator status updates normalized selected-entity store. Web bridge queries projector for history/cold projection; reducer compares live authority `(epoch,sequence)` and never lets older persisted data regress it.

## Related code files

- Create `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/{power-socket.rs,power-history-gateway.rs,power-queues.rs}`.
- Modify `/mnt/data/ws/sharing/robo-fleet-dora-rs/common/web_bridge/src/{main.rs,security.rs,lifecycle-socket.rs}` — shared auth/rate/exact-target patterns and Dora ports.
- Modify Orchestra/Rover direct dataflows for command/status/history ports.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/{power.ts,power-fixtures.ts,power-v1.json}` and modify `types/index.ts`.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/lib/{power-state.ts,power-state.test.ts}`.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/hooks/{use-power-state.ts,use-power-state.test.tsx}`.
- Create `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/{power-control-card.tsx,power-history-panel.tsx}` plus tests.
- Modify `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/pages/{RoboRoverControl.tsx,RoboRoverControl.test.tsx}`.

## Implementation Steps

1. Mirror Rust v1 contracts in TypeScript and verify shared golden fixture.
2. Add bounded web-bridge pending queues/caches and authenticated handlers; derive actor/target server-side.
3. Implement Wake transaction: if policy Sleep, request Auto plus immediate bounded UI demand under one request/transition ID; activity renews within max TTL.
4. Cache/broadcast only validated monotonic live status; clear on auth loss/disconnect/selection change and request snapshot on reconnect.
5. Add history query gateway with server-enforced 90-day/time/entity filters, cursor bounds, rate limit, and sanitized results.
6. Implement normalized hook/reducer: live > cold projection, entity/epoch/sequence guards, event-ID dedupe, bounded history.
7. Add power card to Fleet Resources/control view: policy buttons, Wake Rover, requested/effective state, transition progress/failure, resource freshness, demands.
8. Add history panel/filter/pagination and responsive/accessibility tests for web/Tauri shared UI.

## Todo list

- [ ] Add backend Socket.IO/history contracts.
- [ ] Add TypeScript golden contracts.
- [ ] Add monotonic live/cold/history store.
- [ ] Add power controls and history UI.
- [ ] Add auth, stale, reconnect, entity, accessibility tests.

## Success Criteria

- Wake from Sleep changes policy to Auto, waits for authoritative Ready, then Auto can sleep after demand expiry + five-minute gate.
- Awake holds normal profile; Auto/Sleep display effective state separately.
- Disconnected/stale status disables Wake and policy mutation; rejection is visible.
- Older Mongo projection/history cannot overwrite live state.
- Unauthorized, cross-entity, expired, duplicate-changed, or rate-limited actions fail closed and are audited.

## Risk Assessment

- UI/back-end release drift: one golden fixture and coordinated version gate.
- Activity renew leak keeps Rover awake: bounded TTL, disconnect cleanup, server sweep.
- History volume hurts UI: server pagination and bounded client store.

## Security Considerations

- Session claims supply actor; browser never supplies authority epoch/source privilege.
- Exact active rover validation at web bridge and coordinator.
- History exposes sanitized bounded metadata only; no wake audio, tokens, paths, or native errors.

## Next steps

Phase 08 runs browser and distributed release gates. Product must approve activity-renew TTL and which signed-in roles may set persistent Awake/Sleep before production enablement.
