# Phase 02 — Shared recording UI and CameraViewer controls

## Context links

- `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/packages/shared/src/types/recording.ts`
- `packages/shared/src/types/socket.ts`
- `packages/ui/src/hooks/use-recording-store*.ts`
- `packages/ui/src/components/{pages/RoboRoverControl.tsx,media-recording-page.tsx,features/CameraViewer.tsx,recording-clip-browser.tsx}`

## Overview

Priority P1; complete. One `useRecordingStore` instance is hoisted above the Control/Recordings switch and supplies the Camera Feed, Recordings page, and tab badge.

## Requirements and implementation steps

1. Add typed delete request/result, store action, pending/error/result handling, timeout, disconnect cleanup, and stale-result guards.
2. Restructure clip rows so select and delete are sibling controls (no nested buttons). Require confirmation, disable while pending, show bounded error, clear selected playback only after success, and refresh the current rover filter.
3. Add CameraViewer start/stop using the shared store and render an accessible red REC badge with elapsed duration; disable conflicting buttons during `STARTING`/`FINALIZING`.
4. Add a compact active-count badge to the Recordings tab. Preserve state/listeners when switching views and keep per-rover target IDs authoritative.
5. Extend unit tests, fake socket harness, and recording Playwright flow for delete confirmation/cancel, selected playback clearing, navigation persistence, and keyboard-visible controls. Preserve the existing dirty `recording-session-control.test.tsx` cleanup change.

## UI/security/performance

Use existing Tailwind tokens and accessible button labels; do not display host paths. Surface server reason codes without raw filesystem errors. Keep socket listeners stable, clean up on disconnect/auth loss, and avoid duplicate list requests on tab switches.

## Success and next steps

Typecheck/build/lint pass; Vitest covers happy/error/timeout/navigation cases; browser test can start/stop from CameraViewer, switch tabs, delete a selected finalized clip, and observe the list/playback state update. Then run Phase 03 gates.
