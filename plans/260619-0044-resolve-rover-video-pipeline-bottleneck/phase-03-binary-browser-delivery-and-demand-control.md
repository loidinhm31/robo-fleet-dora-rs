# Phase 03: Binary Browser Delivery and Demand Control

## Context Links

- [Parent plan](./plan.md)
- [Phase 02](./phase-02-rover-jpeg-and-zenoh-cutover.md)
- UI repository: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`
- UI architecture: `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app/docs/architecture.md`
- Depends on: Phase 02 milestone passed

## Overview

- Date: 2026-06-19
- Priority: P1
- Implementation status: Complete 2026-06-24
- Review status: Complete 2026-06-24
- Purpose: eliminate JSON JPEG arrays and stop view work when no client requests it.

## Key Insights

- `serde_json::Value` encodes `Vec<u8>` as a number array, not a Socket.IO binary attachment.
- UI currently declares `data: number[]` and reconstructs a `Uint8Array` every frame.
- UI stream toggle currently changes local state only; encoder/network continue running.

## Requirements

- Preserve `video_frame` object shape and event name.
- Set `timestamp` to original rover capture time; preserve original `frame_id`.
- Encode `data` as a real Socket.IO binary attachment using a typed serializable structure and `serde_bytes`/supported binary type.
- Browser accepts `ArrayBuffer | Uint8Array` and does not support legacy JSON arrays after coordinated cutover.
- Validate both Vite web and Tauri application type/build paths in the adjacent UI repository.
- Aggregate viewer demand; zero subscribers stops only the view branch, not camera/ML needed for tracking.

## Architecture

```text
UI stream toggle/disconnect
  -> web bridge aggregate demand
  -> orchestra bridge -> Zenoh stream command -> rover bridge
  -> capture view-output enabled/disabled

JPEG BinaryArray -> typed web event with binary attachment -> browser Blob -> canvas
```

## Related Code Files

- Modify `common/web_bridge/src/main.rs`: typed payload, lock-free emission boundary, stream state/command queue.
- Modify shared Rust `StreamControl`, both bridges, and dataflows: view demand routing.
- Modify UI shared socket/media types and `CameraViewer.tsx`: binary normalization, demand event, render-age metrics.
- Modify UI and root architecture documents after behavior is verified.

## Implementation Steps

1. Define a typed borrowed browser video payload. Use binary serialization for the JPEG slice; prohibit construction through `serde_json::json!`.
2. Add a serializer/integration test proving JPEG bytes become a Socket.IO binary attachment rather than JSON integers.
3. Under the client-state lock, determine eligible sockets and update counters; release the lock before serialization/emission.
4. Keep per-client FPS limiting defensive, but set its default to the upstream 15 FPS contract.
5. Add authenticated `stream_control` start/stop handling and rate limiting to web bridge.
6. Track aggregate video demand. Emit upstream transitions only on `0 -> 1` and `1 -> 0`; handle disconnect, token expiry, and idle sweep.
7. Route `StreamControl` through Dora and a versioned Zenoh command topic to the selected rover. Gate only view-frame publication at capture.
8. Update UI shared `VideoFrame.data` to `ArrayBuffer | Uint8Array`; reconcile the currently inconsistent shared `string` and local `number[]` definitions without retaining legacy JSON-array handling.
9. Make UI stream start/stop emit demand transitions. Normalize incoming binary without a JSON-array conversion.
10. Preserve/decode one frame at a time; always revoke object URLs on load, error, replacement, and unmount.
11. Measure capture-to-render latency after canvas draw and expose it in existing stream statistics.

## Todo List

- [x] Backend emits actual binary attachment.
- [x] UI shared type is binary.
- [x] Frame ID/timestamp remain capture values.
- [x] Client locks released before emit.
- [x] Demand transitions routed end-to-end.
- [x] Disconnect/expiry stops demand.
- [x] Resume and repeated-toggle tests pass.

## Validation Summary

- `cargo test -p web_bridge`: passed, 19 passed.
- `cargo test -p kornia_capture`: passed, 10 passed.
- `cargo test -p rover_zenoh_bridge --no-run`: passed.
- `cargo test -p orchestra_zenoh_bridge --no-run`: passed.
- UI `pnpm check-types`: passed outside sandbox; direct web/native package type checks passed.
- UI `pnpm lint`: passed, no tasks configured.
- UI `pnpm build`: passed for web/native.
- Code review: 9/10, no critical issues.

## Success Criteria

- Browser receives binary bytes and renders JPEG plus overlays correctly.
- Wire capture contains binary attachment, not a JSON number array.
- No view frame, JPEG encode, or video Zenoh payload occurs within two seconds of last viewer stop/disconnect.
- Local detection/tracking continues when view demand is zero.
- First resumed frame displays within 500 ms.
- Memory remains bounded through a 10-minute start/stop/disconnect loop.
- `pnpm check-types`, `pnpm lint`, and `pnpm build` pass from `/mnt/data/ws/sharing/glean-oak/embed-app/robo-control-app`.

## Risk Assessment

- Socket parser version handles binary differently: test against locked Socketioxide 0.12 and socket.io-client 4.8.
- Lost stop transition wastes resources: recompute aggregate demand on every disconnect/expiry sweep.
- Object URL leak grows browser memory: centralize cleanup and test unmount/error paths.
- Adjacent UI repository changes are obscured by unrelated work: capture its scoped status/diff before edits and preserve all existing changes.

## Security Considerations

- Authenticate and rate-limit stream control like other commands.
- Reject commands for inactive/unauthorized sessions.
- Keep Socket.IO payload and buffer limits above expected JPEG size but below the packet maximum.

## Next Steps

- Proceed to Phase 4 only after binary and demand milestones pass.
- Any parser incompatibility or demand-state race stops the phase for diagnosis.

## Unresolved Questions

- None.
