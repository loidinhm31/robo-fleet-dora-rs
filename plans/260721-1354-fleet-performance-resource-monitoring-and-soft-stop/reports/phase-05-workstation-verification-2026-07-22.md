# Phase 05 workstation verification — 2026-07-22

## Scope

Acceptance used the current x86_64 workstation only.  The Docker-compatible
runtime is Podman via the `docker` CLI, which passed `docker info`, an actual
image build, compose startup, health checks, and the Socket.IO smoke below.
No Raspberry Pi or ARM physical-device requirement was added.

## Socket.IO and lifecycle contract

The original Docker smoke timeout was caused by lifecycle-status fanout loss,
not MongoDB connectivity or Socket.IO port exposure.  The manager emits a
snapshot of safe-node statuses in one tick; one-item Dora queues retained only
the tail (`gst-camera`). The manager-to-web/bridge edges now use 64 entries,
which preserves a complete tick for up to 15 Rovers (four safe-node statuses
per Rover) plus four local Orchestra statuses.

Compose runtime inspection confirmed:

- `MONGODB_URI=mongodb://127.0.0.1:27017` inside the Orchestra container (host
  network mode), local MongoDB healthy, and database
  `robo_fleet_phase05_lifecycle`.
- `SOCKET_IO_PORT=3030`, host-network Socket.IO handshake reachable at
  `127.0.0.1:3030`, and `ALLOW_DEFAULT_CREDENTIALS=true` for the isolated
  smoke only.
- The client used the web app's installed `socket.io-client`, from
  `robo-control-app/apps/web` (not an ad-hoc dependency install).

The final rebuilt `linux/amd64` Docker smoke authenticated and observed all
eight capabilities and statuses. An expired command was rejected with
`expired`; remote `edge-voice` paused to `quiesced` and resumed to `running`
at revisions 1 and 2. On a fresh Orchestra start, untouched Rover workloads
correctly remained `superseded` with `stale_epoch` rather than being reported
as invented `running` states. `audio-playback` remained present in the Rover
container after the foreign `edge-voice` command and did not emit the former
target-mismatch crash.

Native Orchestra/Rover verification used the same local MongoDB isolation.
Bridge audit logs showed the exact request was authorized by Orchestra and
received by Rover.  `edge-voice` reached `quiesced`, then `running`.  A live
expired command returned `expired`; a stale revision returned `conflict` with
`expected revision is stale`.

The reconnect case was also exercised with Rover started before Orchestra.
When the fresh Orchestra manager received the Rover's mismatched prior epoch,
it published `superseded` with component state `degraded` and reason
`stale_epoch` instead of its default `running` state.  An explicit browser
resume established the shared authority and reached `running` at revision 1.
Lifecycle-manager audit logs then recorded the local admission and Rover
relay outcome with origin, request ID, target, acceptance, and reason code.

Direct Rover verification repeated the same contract without Zenoh: five
capabilities/statuses, expired rejection, `edge-voice` pause/resume at
revisions 1/2, and a still-running `audio-playback` sibling.

An offline-Rover startup verification recorded the expected initial authority
result: `superseded` with reason `stale_epoch` at revision 0. This confirms that
Orchestra does not synthesize a `running` state when no matching Rover epoch is
available.

## Safety and recovery matrix

- A live Rover-disconnect fault was injected by stopping the Rover dataflow
  before a remote pause.  Orchestra moved the target to `failed` at revision 5
  after its transition timeout; it never reported `quiesced`.
- Restarting the Rover produced no relayed lifecycle-command log, proving the
  failed command was not replayed.  An explicit resume reconciled it to
  `running` at revision 6.
- A shared broadcast command previously caused `audio-playback` (and could
  have caused `edge-voice`) to exit on a command addressed to a sibling.  Both
  now ignore foreign targets before lifecycle admission.  The native, direct,
  and Docker pause tests prove that the non-target playback node remains live.
- Lifecycle manager tests cover duplicate/conflicting admission, stale epoch,
  expiry, timeout with late status, stale component status, relay epoch
  preservation, request-cache expiry, and final wake-lease reconciliation.
- Focused follow-up tests cover four-or-more Rover capacity, reject initial
  and runtime activation beyond the 15-Rover queue bound, preserve both
  sibling nodes' lifecycle gate state, and reject malformed or
  validation-invalid Zenoh lifecycle commands before manager relay.
- Web-bridge scheduler/coordinator tests cover duplicate/reordered intents,
  stale generation after restart, queued-start invalidation, final-owner
  release, failed/inactive Rover handling, reconciliation snapshots, and
  expired readiness leases.  These validate the scheduled wake/release and
  recording safety paths without introducing a physical-device dependency.
- Existing direct acceptance evidence remains valid: a 12-node flow used both
  cameras and USB audio, an active TTS operation ended with
  `interrupted_by_lifecycle`, resume was ready in 401 ms, and all three paused
  CPU samples met the >=50% reduction gate.

## Container scope

Fresh images started healthy and contained the expected lifecycle/resource
processes.  Final inspect values were:

| Container | CPU limit | Memory limit | Network |
| --- | ---: | ---: | --- |
| `robo-orchestra` | 2 CPUs | 3 GiB | host |
| `robo-rover-kiwi` | 3 CPUs | 4 GiB | host |

## Automated checks

- `cargo fmt --check`
- `cargo test -p robo_rover_lib -p audio_playback -p edge_voice`
- `cargo test -p lifecycle_manager -p orchestra_zenoh_bridge -p rover_zenoh_bridge -p web_bridge`
- `cargo test -p lifecycle_manager -p robo_rover_lib -p orchestra_zenoh_bridge -p rover_zenoh_bridge`
- `pnpm check-types` and `pnpm lint` in the web monorepo
- Docker/Podman compose rebuild plus the Socket.IO smoke above
- Final follow-up code review: all High and Medium reconnect/reordering
  findings resolved; no remaining findings in the focused review.

## Observation retained for follow-up

Stopping native Dora dataflows and stopping the compose containers sometimes
exceeded their 10-second graceful-stop window and Dora/Podman resorted to a
signal/`SIGKILL` for camera/audio or container teardown.  This did not affect
the lifecycle quiesce/resume contract tested above, but it is recorded for a
separate shutdown-latency hardening follow-up.  No rollout or commit occurred.

The lifecycle-status fanout follow-up is bounded at 15 active Rovers to match
the 64-entry queue exactly. This does not change the accepted x86_64-only
scope and does not authorize rollout or a commit.
