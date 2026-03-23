# Phase 4: Cleanup & Verification

## 4.1 — Workspace Cargo.toml

Check workspace `members` list. The 3 crates remain as workspace members (still crates, just lib not bin). No change expected unless workspace has explicit `[[bin]]` overrides.

Verify:
```bash
grep -A 50 '\[workspace\]' Cargo.toml | grep -E 'object_detector|reid_extractor|object_tracker'
```

If workspace explicitly lists binaries, update to reflect lib-only status.

---

## 4.2 — Build verification

```bash
# Individual lib crates
cargo build -p object_detector   # lib only, no binary produced
cargo build -p reid_extractor    # lib only
cargo build -p object_tracker    # lib only

# Consolidated binary
cargo build --release -p kornia_capture  # pulls in all 3 libs + ort

# Full workspace (ensure nothing broken)
cargo build --release
```

Check no stale binary references:
```bash
ls -la target/release/object_detector 2>/dev/null   # should NOT exist
ls -la target/release/reid_extractor 2>/dev/null    # should NOT exist
ls -la target/release/object_tracker 2>/dev/null    # should NOT exist
ls -la target/release/kornia_capture 2>/dev/null    # SHOULD exist
```

---

## 4.3 — Functional test on Pi

### Test 1: Camera only (default state)

1. Start rover dataflow
2. Open web UI → start camera
3. Verify: video streaming works in web UI
4. Verify: `htop` shows kornia_capture at low CPU (~5-10%, camera grab only)
5. Verify: no YOLO/ReID model loading in logs

### Test 2: Enable tracking

1. Click "Enable Tracking" in web UI
2. Verify: logs show "Loading ML models (first enable)..."
3. Verify: ~2-3s delay, then tracked_detections appear in web UI
4. Verify: `htop` shows increased CPU (YOLO inference active)
5. Verify: tracking_telemetry reaches visual-servo-controller

### Test 3: Disable tracking

1. Click "Disable Tracking" in web UI
2. Verify: ML processing stops immediately
3. Verify: camera continues streaming (video_frame still flowing)
4. Verify: CPU drops back to ~5-10%

### Test 4: Re-enable (no cold start)

1. Click "Enable Tracking" again
2. Verify: instant start (no "Loading ML models..." log)
3. Verify: tracked_detections resume immediately

### Test 5: Camera stop/start cycle

1. Stop camera via web UI
2. Verify: all outputs stop
3. Start camera again
4. Verify: video resumes, pipeline state preserved (still enabled/disabled per last command)

### Test 6: Error recovery

1. (If testable) Corrupt model path in env vars
2. Enable tracking
3. Verify: error logged, pipeline auto-disables
4. Verify: camera continues streaming

---

## 4.4 — Update CLAUDE.md

### Changes to Architecture section

Update the "Critical Design Decision: ML on Rover" section:

**Before:**
```
Vision pipeline (all on rover): gst-camera → object-detector (YOLOv12n) → reid-extractor (OSNet x0.25) → object-tracker (BoTSORT+CMC) → visual-servo-controller (PID)
```

**After:**
```
Vision pipeline (all on rover, single process):
gst-camera/kornia_capture → [internal: YOLO → ReID → BoTSORT+CMC] → visual-servo-controller (PID)

object_detector, reid_extractor, object_tracker are library crates consumed by kornia_capture.
ML pipeline is lazy-loaded and gated by TrackingCommand::Enable/Disable.
Default state: camera only (zero ML overhead).
```

### Update env vars section

Note that detector/reid/tracker env vars are now on `gst-camera` node, not separate nodes.

### Update "Additional Nodes" section

Remove object-detector, reid-extractor, object-tracker from the nodes list. Note they are now lib crates, not Dora nodes.

---

## 4.5 — Cleanup stale files

Verify no leftover files:
- `rover-kiwi/object_detector/src/main.rs` — should be deleted
- `rover-kiwi/reid_extractor/src/main.rs` — should be deleted
- `rover-kiwi/object_tracker/src/main.rs` — should be deleted

Check for any references to old node names in:
- Docker compose files
- Makefile
- CI/CD configs
- README or other docs
