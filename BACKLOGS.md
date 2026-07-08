# Robo-Fleet Dora-RS — Architecture Backlog

> Generated: 2026-02-20
> Source: Automated security + performance review across all subsystems
> Status: Pending review

Items are grouped by domain and ordered by severity within each group. Each item includes file paths, line numbers, estimated effort, and concrete remediation steps.

---

## Table of Contents

1. [Security — Critical](#security--critical)
2. [Security — High](#security--high)
3. [Security — Medium](#security--medium)
4. [Performance — Critical](#performance--critical)
5. [Performance — High](#performance--high)
6. [Performance — Medium](#performance--medium)
7. [Infrastructure & DevOps](#infrastructure--devops)
8. [Frontend](#frontend)

---

## Security — Critical

---

### ~~SEC-001 · Plaintext Password Comparison (No Hashing)~~ ✅ RESOLVED

**Severity:** Critical → Resolved
**Component:** `common/web_bridge`
**Resolved:** 2026-03-26 — Phase 01 (web-bridge-auth-improvements)

Replaced plaintext comparison with bcrypt verify via `verify_password_blocking` (spawn_blocking).
Credentials now stored as bcrypt hashes in MongoDB (`db.robo_control_user`).

**Effort:** Small (< 1 day)

---

### ~~SEC-002 · Hardcoded Default Credentials in Source Control~~ ✅ RESOLVED

**Severity:** Critical → Resolved
**Component:** `common/web_bridge`
**Resolved:** 2026-03-26 — Phase 01 (web-bridge-auth-improvements)

Credentials moved to MongoDB. `AUTH_USERNAME`/`AUTH_PASSWORD` env vars removed from `orchestra-dataflow.yml`.
Startup guard: if admin account still uses default password and `ALLOW_DEFAULT_CREDENTIALS != "true"` → `process::exit(1)`.

**Effort:** Small (< 1 day)

---

### SEC-003 · Zenoh Network is Completely Unauthenticated

**Severity:** Critical
**Component:** Zenoh bridges (orchestra + rover)

**Description:**
Both Zenoh bridges use default configs with no authentication, no TLS, and multicast peer discovery enabled. Any device on the same LAN can:
- Discover the Zenoh mesh automatically via multicast
- Subscribe to all rover telemetry (video, sensor data, ML results)
- **Publish arbitrary rover commands** (`rover/{entity_id}/cmd/*`) without any credentials

This is the highest-risk finding: a malicious actor on the WiFi network can take full control of all rovers.

**Affected files:**
- `orchestra/zenoh_bridge/zenoh_config.json5:9` — listens on `tcp/0.0.0.0:7447`
- `orchestra/zenoh_bridge/zenoh_config.json5:24-28` — multicast enabled
- `rover-kiwi/zenoh_bridge/zenoh_config.json5:23-29` — same multicast config
- `rover-kiwi/zenoh_bridge/src/main.rs:118-158` — subscribes to command topics without any source verification
- `rover-kiwi/zenoh_bridge/src/main.rs:283-295` — trusts `metadata.source` from untrusted network payload

**Remediation (short-term):**
1. Restrict Zenoh orchestra listener to specific interface: `tcp/192.168.x.x:7447`
2. Disable multicast; use explicit peer addresses:
   ```json5
   "scouting": {
     "multicast": { "enabled": false },
     "gossip": { "enabled": false }
   },
   "connect": {
     "endpoints": ["tcp/${ORCHESTRA_IP}:7447"]
   }
   ```
3. Add firewall rules (ufw/iptables) to limit who can reach port 7447.

**Remediation (long-term):**
1. Enable Zenoh TLS transport with mutual certificate auth (Zenoh 1.0 supports this).
2. Add application-layer command signing: each bridge signs outgoing commands with an HMAC key; rover verifies before executing.
3. Use Zenoh ACLs (Access Control Lists) to restrict which topics each peer can publish to.

**Effort:** Medium (2-3 days for short-term, 1-2 weeks for full mTLS)

---

### SEC-004 · Unsafe Pointer Casts in Zenoh Bridge (No Length Validation)

**Severity:** Critical
**Component:** `rover-kiwi/zenoh_bridge`

**Description:**
Two `unsafe` blocks cast raw byte slices to/from `f32` arrays without validating that the byte length is a multiple of `size_of::<f32>()`. If a malformed Zenoh message arrives (e.g., from a malicious peer on the network), this causes **undefined behavior** — potentially a crash or memory corruption.

**Affected files:**
- `rover-kiwi/zenoh_bridge/src/main.rs:228-232` — bytes-to-f32 cast
- `rover-kiwi/zenoh_bridge/src/main.rs:345-349` — f32-to-bytes cast

**Current code:**
```rust
let bytes: &[u8] = unsafe {
    std::slice::from_raw_parts(
        float_slice.as_ptr() as *const u8,
        float_slice.len() * std::mem::size_of::<f32>()
    )
};
```

**Remediation:**
```rust
// Replace unsafe cast with safe bytemuck or manual validation
fn bytes_to_f32_slice(bytes: &[u8]) -> Result<Vec<f32>, &'static str> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err("Invalid byte length for f32 slice");
    }
    // Use bytemuck crate (already sound) or manual copy
    Ok(bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
```
Add `bytemuck = "1"` to Cargo.toml for safe, audited casting.

**Effort:** Small (< 1 day)

---

## Security — High

---

### SEC-005 · No Session Management (JWT/Token) — 🔄 PARTIAL (server done, client pending)

**Severity:** High → In Progress
**Component:** `common/web_bridge`, frontend

**Server-side (Phase 01, 2026-03-26):** JWT issued on auth (`auth_token` event), 1h TTL, validated on every command event. `SessionRegistry` with background sweep. `auth_refresh` event for proactive renewal.

**Client-side (Phase 03, pending):** React client needs to store JWT in memory/sessionStorage, handle `auth_token`/`auth_error`, proactive refresh at ~55 min.

**Effort remaining:** Phase 03 (~1h)

---

### SEC-006 · Docker Container Runs as Root with `privileged: true`

**Severity:** High
**Component:** Docker infrastructure

**Description:**
The rover container runs fully privileged, granting it every Linux capability and the ability to escape container isolation. The Dockerfile comments acknowledge this but leave it unresolved.

**Affected files:**
- `docker/docker-compose.yml:108` — `privileged: true`
- `docker/Dockerfile.rover-kiwi:185-186` — `USER dora` commented out
- `docker/Dockerfile.orchestra:88-89` — `USER dora` commented out

**Remediation:**
Replace `privileged: true` with explicit grants (already suggested in the compose file comments):
```yaml
# docker-compose.yml - rover-kiwi service
devices:
  - /dev/video0:/dev/video0
  - /dev/snd:/dev/snd
cap_add:
  - SYS_RAWIO
group_add:
  - video
  - audio
security_opt:
  - no-new-privileges:true
user: "1000:1000"
```
For the orchestra container, `USER dora` can be enabled without hardware access concerns.

**Effort:** Small (1 day)

---

### SEC-007 · Auth Credentials Stored in Browser localStorage

**Severity:** High
**Component:** Web frontend

**Description:**
The server URL (and implicitly auth context) is persisted to `localStorage`. Credentials passed to Socket.IO are visible in browser developer tools. If any XSS vulnerability is ever introduced, `localStorage` is the first thing attackers extract.

**Affected files:**
- `robo-control-app/packages/ui/src/components/pages/RoboRoverControl.tsx:86-88` — `localStorage.setItem(STORAGE_KEY, url)`
- `robo-control-app/apps/web/src/App.tsx:179-182` — credentials passed in Socket.IO `auth` object

**Remediation:**
1. Store server URL in `sessionStorage` (cleared on tab close) instead of `localStorage`.
2. After implementing SEC-005 (JWT), store only the JWT token, not raw credentials.
3. Never log or store the password string after the initial auth exchange.

**Effort:** Small (< 1 day)

---

### SEC-008 · Missing Content Security Policy

**Severity:** High
**Component:** Web frontend

**Description:**
The web application has no Content Security Policy. This is the primary browser defense against XSS — without it, any injected script runs with full page privileges.

**Affected files:**
- `robo-control-app/apps/web/index.html` (no CSP meta tag)
- `orchestra/web_bridge/src/main.rs` (no CSP response header)

**Remediation:**
Add CSP via HTTP header in web_bridge (preferred) or meta tag:
```
Content-Security-Policy:
  default-src 'self';
  connect-src 'self' ws: wss:;
  img-src 'self' blob: data:;
  media-src 'self' blob:;
  script-src 'self';
  style-src 'self' 'unsafe-inline';
  frame-ancestors 'none';
```
`'unsafe-inline'` for styles is acceptable; avoid it for scripts.

**Effort:** Small (< 1 day)

---

## Security — Medium

---

### ~~SEC-009 · Five Socket.IO Event Handlers Lack Rate Limiting~~ ✅ RESOLVED

**Severity:** Medium → Resolved
**Component:** `common/web_bridge`
**Resolved:** 2026-03-26 — Phase 02 (web-bridge-auth-improvements)

All previously unguarded handlers now have session validation + `CommandRateLimiter` guards:
`camera_control`, `audio_control`, `voice_command_audio`, `tracking_command`, `fleet_subscription`.
Also: `auth_refresh` rate-limited via `AuthRateLimiter`.
Per-IP rate limiting added (`IpRateLimiter`, env `RATE_LIMIT_AUTH_PER_MINUTE_IP=20`).

**Effort:** Small (< 1 day)

---

### SEC-010 · Rate Limiter HashMap Grows Unbounded

**Severity:** Medium
**Component:** `orchestra/web_bridge/src/security.rs`

**Description:**
`AuthRateLimiter` and `CommandRateLimiter` store per-socket-ID entries in a `HashMap` that is cleaned every 5 minutes. Between cleanups, every unique socket ID that ever connected occupies memory. An attacker can exhaust memory by opening many short-lived connections with unique IDs.

**Affected files:**
- `orchestra/web_bridge/src/security.rs:8-85`

**Remediation:**
1. Set a maximum HashMap size (e.g., 10,000 entries) and evict oldest on overflow.
2. Use a time-aware LRU cache (`lru` crate) keyed by socket ID.
3. Reduce cleanup interval from 5 minutes to 60 seconds.

**Effort:** Small (< 1 day)

---

### SEC-011 · Detection Index Not Validated Before Rover Dispatch

**Severity:** Medium
**Component:** `orchestra/web_bridge`

**Description:**
`SelectTarget` commands include a `detection_index: usize` that is validated in `security.rs` for `Option::Some` presence, but not checked against the actual number of live detections before being sent to the rover. An out-of-range index can cause a panic or incorrect target selection on the rover.

**Affected files:**
- `orchestra/web_bridge/src/main.rs:1407-1427`
- `orchestra/web_bridge/src/security.rs:153-158`

**Remediation:**
Before dispatching, check against the current detection frame:
```rust
if let Some(idx) = web_cmd.detection_index {
    let detections = current_detections.lock().unwrap();
    if idx >= detections.len() {
        tracing::warn!("Rejected SelectTarget: index {} out of bounds ({})", idx, detections.len());
        return;
    }
}
```

**Effort:** Small (< 1 day)

---

### SEC-012 · Hardcoded Developer Filesystem Path in Dataflow YAML

**Severity:** Medium
**Component:** Orchestra dataflow config

**Description:**
A developer's local absolute path is committed to the repository. This breaks deployment on any other machine and reveals filesystem structure in version control.

**Affected files:**
- `orchestra/orchestra-dataflow.yml:77` — `ZENOH_CONFIG: "/home/loidinh/ws/robo-fleet-dora-rs/..."`
- `rover-kiwi/rover-kiwi-dataflow.yml:209` — `ZENOH_CONFIG: "${HOME}/ws/robo-fleet-dora-rs/..."`
- `rover-kiwi/rover-kiwi-dataflow.yml:29` — `LD_LIBRARY_PATH: "${HOME}/ws/..."`

**Remediation:**
Use environment variable substitution with a documented default:
```yaml
ZENOH_CONFIG: ${ZENOH_CONFIG:-/app/config/zenoh_config.json5}
```
Document required env vars in a `.env.example` file.

**Effort:** Trivial (< 2 hours)

---

### SEC-013 · No Runtime Schema Validation on Socket.IO Events (Frontend)

**Severity:** Medium
**Component:** Web frontend

**Description:**
Socket.IO event handlers cast incoming data directly to TypeScript interface types without runtime validation. TypeScript types are erased at runtime, so a malformed or malicious server message (e.g., from a compromised backend or MITM) can crash the UI or cause unexpected behavior.

**Affected files:**
- `robo-control-app/packages/ui/src/components/pages/RoboRoverControl.tsx:217-266`
- `robo-control-app/packages/ui/src/components/organisms/CameraViewer.tsx:247, 595`

**Remediation:**
Add runtime validation using Zod:
```typescript
import { z } from 'zod';

const JPEGVideoFrameSchema = z.object({
  data: z.array(z.number()),
  width: z.number().positive(),
  height: z.number().positive(),
});

socket.on("video_frame", (raw) => {
  const result = JPEGVideoFrameSchema.safeParse(raw);
  if (!result.success) return;
  handleVideoFrame(result.data);
});
```

**Effort:** Small (1-2 days to cover all events)

---

### SEC-014 · Server URL Input Not Validated

**Severity:** Medium
**Component:** Web frontend

**Description:**
The server URL input in settings accepts any string without URL validation. A `javascript:` or `data:` URI could be entered and later used in a context that executes it.

**Affected files:**
- `robo-control-app/packages/ui/src/components/organisms/ServerSettings.tsx:18-22`

**Remediation:**
```typescript
const validateUrl = (input: string): string | null => {
  try {
    const url = new URL(input.trim());
    if (!['ws:', 'wss:', 'http:', 'https:'].includes(url.protocol)) return null;
    return url.toString();
  } catch {
    return null;
  }
};
```

**Effort:** Trivial (< 2 hours)

---

### SEC-015 · Production Console Logging (Information Disclosure)

**Severity:** Low
**Component:** Web frontend

**Description:**
26 `console.log/warn/error` calls across the UI package ship to production builds, including detailed frame timing, detection data, and audio state. This reveals internal system state in browser developer tools.

**Remediation:**
Gate all debug logging behind a flag:
```typescript
const DEBUG = import.meta.env.DEV;
const log = DEBUG ? console.log : () => {};
```
Or use a logging library (`debug`, `loglevel`) with production silencing.

**Effort:** Trivial (< 2 hours)

---

## Performance — Critical

---

### PERF-001 · ReID Extractor Copies Full Frame 3+ Times Per Batch

**Severity:** Critical
**Component:** `rover-kiwi/reid_extractor`

**Description:**
For every frame processed, the ReID extractor creates at least 3 full copies of the raw image buffer (~921KB for 640×480 RGB). At 30 FPS this is ~82 MB/s of unnecessary allocation and copying, putting pressure on the Raspberry Pi 5's memory bandwidth and garbage collector.

**Copy sites:**
- `src/main.rs:139` — `frame_data.to_vec()` inside `ImageBuffer::from_raw()`
- `src/main.rs:161` — `.clone()` when reading from `current_frame: Option<(Vec<u8>, u32, u32)>`
- `src/main.rs:273` — `array.values().as_ref().to_vec()` when reading Dora input

**Remediation:**
Share frame ownership using `Arc`:
```rust
// Store frame as Arc<Vec<u8>> — zero-copy sharing
struct ReIDExtractor {
    current_frame: Option<Arc<Vec<u8>>>,
    // ...
}

// On frame input
let frame = Arc::new(frame_data); // single allocation
self.current_frame = Some(Arc::clone(&frame));

// Pass to crop extraction — no copy
extract_crops(&frame, &detections);
```
Also replace manual pixel-by-pixel crop loop (`src/main.rs:139-152`) with `image::imageops::crop()`.

**Effort:** Medium (1-2 days)

---

### PERF-002 · Web Bridge Uses Polling Loops Instead of Event-Driven Channels

**Severity:** Critical
**Component:** `orchestra/web_bridge`

**Description:**
All 10 command types are processed by separate async tasks that `sleep(10ms)` and then lock a `Mutex<Vec<Command>>` to check for work. This pattern:
1. Wakes up 1000×/sec across all queues with no work to do (busy waiting)
2. Uses `std::sync::Mutex` in async context — blocks the Tokio thread during lock wait
3. Uses `Vec::remove(0)` — O(n) operation for queue removal

**Affected files:**
- `orchestra/web_bridge/src/main.rs:697-977` — all 10 command processor tasks

**Remediation:**
Replace with `tokio::sync::mpsc` channels:
```rust
// Sender side (Socket.IO handler)
let (arm_tx, mut arm_rx) = tokio::sync::mpsc::channel::<WebArmCommand>(32);

// Receiver side (command processor)
tokio::spawn(async move {
    while let Some(cmd) = arm_rx.recv().await {
        // Woken only when work arrives — no polling, no sleep
        node.lock().await.send_output(...)?;
    }
});
```
Also replace `Vec` queues with proper channel backpressure (bounded channels drop oldest on overflow).

**Effort:** Medium (2-3 days)

---

### PERF-003 · CMC Full-Resolution Pixel Operations + Fixed-Iteration RANSAC

**Severity:** Critical
**Component:** `rover-kiwi/object_tracker/src/cmc.rs`

**Description:**
Camera Motion Compensation processes the full frame at full resolution using manual pixel loops:
1. `src/cmc.rs:302-314` — manual RGB→Gray conversion iterating 307,200 pixels
2. `src/cmc.rs:93-155` — O(corners_prev × corners_curr) patch matching (~40M comparisons/frame for 200 corners each)
3. `src/cmc.rs:194-237` — fixed 100 RANSAC iterations with new RNG allocation per iteration and matrix inversion each time

When enabled, this adds an estimated 50-100ms to every frame's processing budget.

**Remediation:**
1. Use vectorized RGB→Gray: `image::DynamicImage::to_luma8()` (SIMD-accelerated internally)
2. Process at ½ or ¼ resolution: downscale before corner detection, scale result back up
3. Replace fixed RANSAC with adaptive termination:
   ```rust
   let max_iter = (1.0_f32 / (1.0 - 0.95_f32.powf(2.0 / corners.len() as f32)).ln()).ceil() as usize;
   let max_iter = max_iter.clamp(10, 50); // adaptive, not fixed 100
   ```
4. Cache RNG: create `thread_rng()` once per CMC call, not per iteration

**Effort:** Medium (2-3 days)

---

### PERF-004 · Per-Client Video Emission Instead of Room Broadcast

**Severity:** Critical
**Component:** `orchestra/web_bridge`

**Description:**
Video frames (~80KB JPEG) are sent to each connected client in a sequential loop with a lock held. For 10 clients at 30 FPS: 10 × 80KB × 30 = 24 MB/s of repeated serialization + Socket.IO framing overhead.

**Affected files:**
- `orchestra/web_bridge/src/main.rs:1125-1155` — per-client iteration
- `orchestra/web_bridge/src/main.rs:1068, 1129` — nested mutex locks

**Remediation:**
socketioxide supports room-based broadcasting. Place all video clients in a `"video"` room at connection, then emit once:
```rust
// At connection
socket.join("video");

// At frame arrival — single emit to all clients
io.to("video").emit("video_frame", &frame_payload)?;
```
Replace the outer `Mutex<HashMap>` with a `RwLock` (read-heavy workload).

**Effort:** Small (1 day)

---

## Performance — High

---

### PERF-005 · Object Detector: Unnecessary Frame Copy Before ImageBuffer

**Severity:** High
**Component:** `rover-kiwi/object_detector`

**Description:**
`to_vec()` at `src/main.rs:219` creates a full copy of the incoming frame before constructing `ImageBuffer`. `ImageBuffer::from_raw()` accepts a borrowed slice and can work zero-copy.

**Remediation:**
```rust
// Before
let img = ImageBuffer::from_raw(width, height, frame_data.to_vec()).unwrap();

// After — borrow directly, no copy
let img = ImageBuffer::from_raw(width, height, frame_data).unwrap();
// or if Vec ownership is needed downstream:
let img = ImageBuffer::from_raw(width, height, &frame_data[..]).unwrap();
```

**Effort:** Trivial (< 1 hour)

---

### PERF-006 · ReID Extractor Uses CPU-Only ONNX Execution

**Severity:** High
**Component:** `rover-kiwi/reid_extractor`

**Description:**
The ONNX Runtime session is initialized with `ExecutionProvider::CPU` only. The Raspberry Pi 5 has GPU compute capability (VideoCore VII) and the ORT library may support it. YOLO already uses GPU — ReID feature extraction (512D for each detected person) is also compute-bound.

**Affected files:**
- `rover-kiwi/reid_extractor/src/main.rs:34`

**Remediation:**
```rust
let session = Session::builder()?
    .with_execution_providers([
        CUDAExecutionProvider::default().build(),  // try GPU first
        CPUExecutionProvider::default().build(),   // fallback
    ])?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .commit_from_file(&model_path)?;
```
Test whether the Pi 5's Vulkan/OpenCL backend in ORT helps ReID latency.

**Effort:** Small (< 1 day, mostly testing)

---

### PERF-007 · Object Tracker: Entire Detection Vector Cloned Per Frame

**Severity:** High
**Component:** `rover-kiwi/object_tracker`

**Description:**
At `src/main.rs:713`, `tracker.update(detection_frame.detections.clone())` clones the full detection vector. Each `DetectionResult` contains a 512×f32 `reid_features` field (~2KB). For 50 detections: 100KB cloned every frame = 3 MB/s on the critical path.

**Remediation:**
Change `Tracker::update` signature to accept a reference or take ownership of the already-consumed input:
```rust
// Before
pub fn update(&mut self, detections: Vec<DetectionResult>) { ... }
tracker.update(detection_frame.detections.clone());

// After — consume directly, no clone needed
pub fn update(&mut self, detections: Vec<DetectionResult>) { ... }
tracker.update(detection_frame.detections); // moved, not cloned
```

**Effort:** Small (< 1 day)

---

### PERF-008 · Web Bridge: `std::sync::Mutex` Used in Async Context

**Severity:** High
**Component:** `orchestra/web_bridge`

**Description:**
54 `.lock()` calls on `std::sync::Mutex` exist throughout the async web_bridge. Locking a blocking mutex in an async task blocks the Tokio thread for the duration of the lock hold, potentially stalling other tasks on the same thread.

**Remediation:**
Replace with `tokio::sync::Mutex` for guards held across `.await` points, or restructure to hold locks only briefly with synchronous code:
```rust
// If lock is held briefly (no await inside):
// std::sync::Mutex is fine — acquire, copy data, release immediately
let data = { state.some_mutex.lock().unwrap().clone() }; // release before await

// If lock must be held across await:
use tokio::sync::Mutex;
let mut guard = state.some_mutex.lock().await;
do_async_thing(&mut guard).await;
```

**Effort:** Medium (1-2 days)

---

### PERF-009 · ndarray Pixel Normalization Uses Non-Contiguous Indexing

**Severity:** High
**Component:** `rover-kiwi/object_detector`

**Description:**
YOLO preprocessing at `src/main.rs:100-104` normalizes pixel values using `array[[0, c, y, x]]` indexing in nested loops. This accesses memory in column-major order against ndarray's row-major layout, causing cache misses on every pixel.

**Remediation:**
Iterate in row-major order (y outer, x inner, c innermost) and use a flat slice for writes:
```rust
// Process channel-by-channel in y-major order for cache locality
for c in 0..3 {
    let channel_slice = array.slice_mut(s![0, c, .., ..]);
    for (idx, pixel_val) in channel_slice.iter_mut().enumerate() {
        let y = idx / width;
        let x = idx % width;
        *pixel_val = img_rgb.get_pixel(x as u32, y as u32)[c] as f32 / 255.0;
    }
}
```
Or use `ndarray`'s `Zip` iterator which handles layout automatically.

**Effort:** Small (< 1 day)

---

## Performance — Medium

---

### PERF-010 · Visual Servo: PID Time-Delta Not Capped

**Severity:** Medium
**Component:** `rover-kiwi/visual_servo_controller`

**Description:**
`dt` is computed from elapsed time since the last command (`src/main.rs:335-343`) with no upper bound. If tracking telemetry is delayed (e.g., due to network jitter or a missed frame), a large `dt` causes the PID integrator to wind up and output a large velocity spike, potentially causing abrupt rover motion.

**Remediation:**
```rust
let dt = elapsed.min(Duration::from_millis(500)).as_secs_f64();
```

**Effort:** Trivial (< 1 hour)

---

### PERF-011 · Unbounded Flume Channel in Zenoh Bridge

**Severity:** Medium
**Component:** `orchestra/zenoh_bridge`

**Description:**
`flume::unbounded()` at `src/main.rs:292` creates an unbounded channel between the Zenoh async receiver and the Dora synchronous sender. If Dora processing is slower than the Zenoh message rate, memory grows without limit.

**Remediation:**
```rust
// Use bounded channel — drop oldest on overflow
let (tx, rx) = flume::bounded::<Message>(64);

// On send, use try_send and drop if full
if tx.try_send(msg).is_err() {
    tracing::warn!("Zenoh→Dora channel full, dropping message");
}
```

**Effort:** Trivial (< 1 hour)

---

### PERF-012 · Command Queues Use `Vec::remove(0)` — O(n) Dequeue

**Severity:** Medium
**Component:** `orchestra/web_bridge`

**Description:**
All 10 command queue types are `Arc<Mutex<Vec<Cmd>>>` using `remove(0)` to dequeue. `Vec::remove(0)` shifts all remaining elements left — O(n) per dequeue. Under load this wastes CPU cycles.

**Affected files:**
- `orchestra/web_bridge/src/main.rs:165-174` — queue type definitions
- Multiple command processor tasks using `remove(0)`

**Remediation (interim, before PERF-002 channel refactor):**
```rust
use std::collections::VecDeque;

// Before
Arc<Mutex<Vec<WebArmCommand>>>

// After — O(1) pop_front
Arc<Mutex<VecDeque<WebArmCommand>>>
queue.pop_front() // instead of remove(0)
```

**Effort:** Trivial (< 2 hours)

---

### PERF-013 · React Atom/Molecule Components Not Memoized

**Severity:** Medium
**Component:** Web frontend

**Description:**
Rover state updates (velocity, telemetry, detections) at ~10 FPS trigger re-renders of the entire `RoboRoverControl` component tree. Atom and molecule components (`StatusBadge`, `ValueDisplay`, `SliderControl`, etc.) have no `React.memo` wrapping, causing unnecessary re-renders of unchanged UI.

**Affected files:**
- `robo-control-app/packages/ui/src/components/atoms/*.tsx`
- `robo-control-app/packages/ui/src/components/molecules/*.tsx`

**Remediation:**
```typescript
// Wrap pure display components
export const StatusBadge = React.memo(({ status, label }: Props) => {
  return <div className={...}>{label}</div>;
});

// Use useCallback for handlers passed to children
const handleJointChange = useCallback((joint: number, value: number) => {
  // ...
}, [/* deps */]);
```

**Effort:** Small (1 day)

---

### PERF-014 · Frontend: No Frame Dropping Under Network Congestion

**Severity:** Medium
**Component:** Web frontend

**Description:**
The frontend renders every video frame it receives, regardless of how far behind the render loop is. Under slow network or high CPU, frames queue up causing increasing latency (the live view shows older and older video).

**Affected files:**
- `robo-control-app/packages/ui/src/components/organisms/CameraViewer.tsx:247-253`

**Remediation:**
Track frame receive time and skip render if behind threshold:
```typescript
const handleVideoFrame = useCallback((frame: JPEGVideoFrame) => {
  const now = Date.now();
  if (lastFrameRenderTime.current && now - lastFrameRenderTime.current < 16) {
    // Already rendering at >60 FPS equivalent, skip
    frameCountRef.current++;
    return;
  }
  lastFrameRenderTime.current = now;
  // ... render frame
}, []);
```

**Effort:** Small (< 1 day)

---

### PERF-015 · JPEG Blob Created and Object URL Allocated Per Frame

**Severity:** Medium
**Component:** Web frontend

**Description:**
Every video frame creates a new `Blob`, allocates an `ObjectURL`, and relies on garbage collection to free the previous one. At 30 FPS this is 30 GC-managed allocations per second, creating memory pressure.

**Affected files:**
- `robo-control-app/packages/ui/src/components/organisms/CameraViewer.tsx:247-253`

**Remediation:**
Use `OffscreenCanvas` or decode JPEG directly to `ImageData` using the browser's native `createImageBitmap`:
```typescript
const handleVideoFrame = async (frame: JPEGVideoFrame) => {
  const blob = new Blob([new Uint8Array(frame.data)], { type: 'image/jpeg' });
  const bitmap = await createImageBitmap(blob); // hardware-decoded
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close(); // explicit cleanup, no GC dependency
};
```

**Effort:** Small (< 1 day)

---

## Infrastructure & DevOps

---

### INFRA-001 · Docker Base Images Not Pinned to Content Digest

**Severity:** Medium
**Description:** Tags like `rust:1.76-bookworm` can be updated by Docker Hub. Pin to digest for reproducible builds.

**Files:** `docker/Dockerfile.orchestra:4`, `docker/Dockerfile.rover-kiwi:4`

**Remediation:**
```dockerfile
FROM rust:1.76-bookworm@sha256:<digest> AS builder
FROM debian:bookworm-slim@sha256:<digest> AS runtime
```
Use `docker manifest inspect rust:1.76-bookworm` to get current digest.

**Effort:** Trivial (< 1 hour)

---

### INFRA-002 · Model Downloads Have No Checksum Verification

**Severity:** Medium
**Description:** `download-models.sh` fetches ML models from HuggingFace and GitHub with no SHA256 verification. A compromised CDN or MITM attack could substitute a malicious model.

**Files:** `docker/scripts/download-models.sh:30, 43`

**Remediation:**
```bash
EXPECTED_SHA="<sha256>"
wget -O model.onnx "https://..."
echo "${EXPECTED_SHA}  model.onnx" | sha256sum -c - || { echo "Checksum mismatch!"; exit 1; }
```

**Effort:** Trivial (< 1 hour per model)

---

### INFRA-003 · No Automated Dependency Vulnerability Scanning

**Severity:** Medium
**Description:** Neither `cargo audit` nor `npm audit` are integrated into CI/CD. Known CVEs in transitive dependencies go undetected.

**Remediation:**
Add to CI pipeline:
```bash
# Rust
cargo install cargo-audit
cargo audit --deny warnings

# Node.js
pnpm audit --audit-level=moderate
```
Consider `cargo deny` for more granular license + advisory policy enforcement.

**Effort:** Small (< 1 day for CI integration)

---

### INFRA-004 · Entrypoint Scripts Use Fragile `sed` for YAML Patching

**Severity:** Low
**Description:** `entrypoint-orchestra.sh` and `entrypoint-rover.sh` patch the dataflow YAML using `sed` substitutions with no error checking. Special characters in paths or env vars can break the substitution silently.

**Files:** `docker/scripts/entrypoint-orchestra.sh:31-34`, `docker/scripts/entrypoint-rover.sh`

**Remediation:**
Use `yq` (YAML processor) for structured YAML editing, or use a Python/jinja template to generate the dataflow YAML at startup with proper escaping.

**Effort:** Small (< 1 day)

---

### INFRA-005 · `docker-compose.yml` Uses Host Network Mode

**Severity:** Low
**Description:** `network_mode: host` bypasses Docker network isolation, exposing all host ports and allowing containers to sniff host traffic. Required for Zenoh multicast but should be documented and mitigated.

**Files:** `docker/docker-compose.yml:16, 105`

**Remediation (if multicast is mandatory):** Document explicitly with a comment explaining the requirement. Add host-level firewall rules (ufw) to limit which IPs can reach Zenoh port 7447.

**Remediation (if unicast is acceptable):** Configure Zenoh with explicit peer endpoints and use a `bridge` network instead.

**Effort:** Small (research + config, 1 day)

---

## Frontend

---

### FE-001 · `window as any` Cast for WebKit AudioContext

**Severity:** Low
**Description:** Safari compatibility requires accessing `window.webkitAudioContext`, currently done by casting `window` to `any`.

**File:** `robo-control-app/packages/ui/src/components/organisms/CameraViewer.tsx:348`

**Remediation:**
Extend the Window interface properly:
```typescript
// global.d.ts
interface Window {
  webkitAudioContext?: typeof AudioContext;
}

// usage
const AudioCtx = window.AudioContext || window.webkitAudioContext;
```

**Effort:** Trivial (< 1 hour)

---

### FE-002 · Socket.IO Reconnection Uses Fixed Delay (No Exponential Backoff)

**Severity:** Low
**Description:** Reconnection is configured with `reconnectionDelay: 1000` and `reconnectionAttempts: 5`. Under server overload, clients collectively hammering the same fixed interval can prevent recovery (thundering herd).

**File:** `robo-control-app/packages/ui/src/components/pages/RoboRoverControl.tsx:174-178`

**Remediation:**
```typescript
reconnectionDelay: 1000,
reconnectionDelayMax: 30000, // caps exponential growth
randomizationFactor: 0.5,    // adds jitter to spread reconnects
```
socketioxide's client supports these options natively.

**Effort:** Trivial (< 1 hour)

---

*End of backlog. Total items: 30 (4 Critical Security, 4 High Security, 7 Medium Security, 4 Critical Performance, 5 High Performance, 6 Medium Performance, 5 Infrastructure/Frontend)*
