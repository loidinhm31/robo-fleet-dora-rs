# Code Review Summary

### Scope
- Files reviewed: `packages/ui/src/components/features/VoiceControls.tsx`, `voice-config-card.tsx`, `voice-alert-region.tsx`, `voice-controls-helpers.ts`, `packages/ui/src/components/pages/RoboRoverControl.tsx`, `VoiceControls.test.tsx`, `RoboRoverControl.test.tsx`
- Review focus: Phase 06 Web UI Controls and Alerts, scoped diff only
- Validation referenced: `pnpm --filter @robo-fleet/ui test`, `pnpm check-types`, `pnpm lint`
- Updated plans: `phase-06-web-ui-controls-and-alerts.md`

### Overall Assessment
Solid direction. Socket ownership stays at page level, config storage is not persisted locally, and the UI now exposes the right TTS controls. Main gaps are concurrency/auth-state edge cases: one stale-revision bug in the debounced config path, stale alerts surviving disconnect, and one accessibility mismatch on interruption alerts.

### Critical Issues
1. Debounced config updates can emit a stale `base_revision`, violating the authoritative-state requirement and causing avoidable `stale_revision` rejects during concurrent updates.
   - Evidence: [packages/ui/src/components/features/VoiceControls.tsx](/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/VoiceControls.tsx:223) computes `base_revision` from render-captured `ttsConfigState` / `pendingRevision`, and [packages/ui/src/components/features/VoiceControls.tsx](/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/VoiceControls.tsx:257) schedules that callback for later. If authoritative state changes before the timeout fires, the timer still sends the old base revision.

### High Priority Findings
1. Voice alerts are not cleared on disconnect/session loss, so reconnect can show stale rover-TTS failures from the previous session.
   - Evidence: `RoboRoverControl` clears page-level voice props in [packages/ui/src/components/pages/RoboRoverControl.tsx](/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/pages/RoboRoverControl.tsx:236), but `VoiceControls` keeps its own `alerts` state in [packages/ui/src/components/features/VoiceControls.tsx](/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/VoiceControls.tsx:162) and never resets it when connection/auth state is cleared. The ack/result effects at [packages/ui/src/components/features/VoiceControls.tsx](/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/VoiceControls.tsx:530) only append alerts; `null` props do not remove existing ones.

### Medium Priority Improvements
1. `interrupted_by_walkie` is rendered as polite `status`, not `alert`, which misses the explicit accessibility contract in the phase doc.
   - Evidence: [packages/ui/src/components/features/voice-controls-helpers.ts](/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/voice-controls-helpers.ts:127) sets `liveMode: "polite"` for the walkie interruption case, and [packages/ui/src/components/features/voice-alert-region.tsx](/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/ui/src/components/features/voice-alert-region.tsx:27) maps polite alerts to `role="status"`.

### Low Priority Suggestions
1. Add one regression test where a debounced slider change is pending, then `tts_config_state` advances before the timeout fires. Current tests only cover the single-client happy path.
2. Add one reconnect/disconnect test that asserts old alert cards disappear, not just that config placeholder text returns.
3. Consider moving config revision bookkeeping into refs or a small reducer. Current split between state, refs, and timers in `VoiceControls` is workable but easy to desync under concurrency.

### Positive Observations
- Page-level socket ownership in `RoboRoverControl` matches the repo standards and avoids local config persistence.
- Listener registration/cleanup moved to stable callbacks; normal reconnect path is much cleaner than inline `socket.on(...)` wiring.
- `VoiceConfigCard` / `VoiceAlertRegion` extraction improved separation versus keeping everything in `VoiceControls`.

### Recommended Actions
1. Fix debounced dispatch to read current authoritative revision from refs at fire time, not from the render that created the timeout.
2. Reset `VoiceControls` alert state on disconnect/auth loss, likely keyed off connection/auth or a dedicated reset prop.
3. Change walkie interruption alerts to assertive `alert` semantics and keep them visible until dismissed/replaced.

### Metrics
- Type Coverage: not measured
- Test Coverage: not measured
- Linting Issues: 0 reported in provided validation

### Unresolved Questions
- What server-side guarantee replays `tts_config_state` / `voice_status` after reconnect? The UI currently relies on passive delivery; no explicit hydration request exists in the client event map.
