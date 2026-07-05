#!/usr/bin/env bash
# Benchmark helper tests.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HELPER="$ROOT/scripts/benchmark-audio-video-stream.sh"
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT
ROVER_LOG="$TEMP_DIR/rover.log"
ORCHESTRA_LOG="$TEMP_DIR/orchestra.log"
BROWSER_LOG="$TEMP_DIR/browser.log"
SUMMARY="$TEMP_DIR/summary.md"

write_rover_logs() {
  cat > "$ROVER_LOG" <<'EOF'
metric="audio_pipeline" stage="rover_convert" errors=0
metric="audio_pipeline" stage="orchestra_zenoh_receive" errors=0
metric="audio_pipeline" stage="rover_publish" format="s16le" errors=0
EOF
  cat > "$ORCHESTRA_LOG" <<'EOF'
metric="audio_pipeline" stage="orchestra_zenoh_receive" errors=0
EOF
}

analyze() {
  "$HELPER" analyze \
    --scenario "$1" \
    --browser-log "$BROWSER_LOG" \
    --rover-log "$ROVER_LOG" \
    --orchestra-log "$ORCHESTRA_LOG" \
    --network-path LAN \
    --browser-host browser-a \
    --rover-host rover-a \
    --orchestra-host orchestra-a \
    --devtools closed \
    --clock-offset-ms 2 \
    --output "$SUMMARY" \
    --json-baseline-bytes 5911 \
    --binary-payload-bytes 1857
}

TAB=$'\t'
PASS_HORIZON="^PASS${TAB}scheduled_horizon${TAB}140 ms <= 150 ms$"
PASS_UNDERRUNS="^PASS${TAB}underruns${TAB}0$"
PASS_DROPS="^PASS${TAB}drops${TAB}0$"
PASS_REDUCTION="^PASS${TAB}binary_reduction${TAB}68\\.58%"
PASS_FORMAT="^PASS${TAB}rover_audio_format${TAB}s16le$"
PASS_WARMUP="^PASS${TAB}warmup_gating${TAB}warmup complete at 1500 ms$"
PASS_CADENCE="^PASS${TAB}cadence${TAB}20\\.000 fps$"
PASS_CLOCK="^PASS${TAB}clock_offset${TAB}2 ms; age metric valid$"
FAIL_HORIZON="^FAIL${TAB}scheduled_horizon${TAB}180 ms > 150 ms$"
FAIL_UNDERRUNS="^FAIL${TAB}underruns${TAB}3$"
FAIL_DROPS="^FAIL${TAB}drops${TAB}7$"
FAIL_REDUCTION="^FAIL${TAB}binary_reduction${TAB}32\\.33%"
FAIL_CONTROL="^FAIL${TAB}control_events${TAB}audio stop/start command found$"
WARN_FORMAT="^WARN${TAB}rover_audio_format${TAB}no rover_publish format marker found$"

assert_grep() {
  local file=$1 pattern=$2 label=$3
  if ! grep -Eq "$pattern" "$file"; then
    printf 'expected %s to contain: %s\n' "$label" "$pattern" >&2
    cat "$file" >&2
    exit 1
  fi
}

# --- 1. audio-only happy path -------------------------------------------------
write_rover_logs
cat > "$BROWSER_LOG" <<'EOF'
audio_stream_metrics {"capturedAtMs":1000,"framesReceived":0,"invalidFrames":0,"sequenceGaps":0,"duplicates":0,"regressions":0,"interArrivalMs":{"max":50},"scheduledHorizonMs":{"max":100},"transport":"websocket","underruns":0,"drops":0,"frameAgeMs":{"p95":80},"warmupCompleteMs":1500,"binaryReductionPercent":68.5}
audio_stream_metrics {"capturedAtMs":592000,"framesReceived":11820,"invalidFrames":0,"sequenceGaps":0,"duplicates":0,"regressions":0,"interArrivalMs":{"max":55},"scheduledHorizonMs":{"max":140},"transport":"websocket","underruns":0,"drops":0,"frameAgeMs":{"p95":110},"warmupCompleteMs":1500,"binaryReductionPercent":68.5}
EOF
analyze audio-only >"$TEMP_DIR/audio-only.out" 2>&1
assert_grep "$TEMP_DIR/audio-only.out" "$PASS_HORIZON" 'audio-only scheduled_horizon'
assert_grep "$TEMP_DIR/audio-only.out" "$PASS_UNDERRUNS" 'audio-only underruns'
assert_grep "$TEMP_DIR/audio-only.out" "$PASS_DROPS" 'audio-only drops'
assert_grep "$TEMP_DIR/audio-only.out" "$PASS_REDUCTION" 'audio-only binary_reduction'
assert_grep "$TEMP_DIR/audio-only.out" "$PASS_FORMAT" 'audio-only rover_audio_format'
assert_grep "$TEMP_DIR/audio-only.out" "$PASS_WARMUP" 'audio-only warmup_gating'
assert_grep "$SUMMARY" '# audio-only Summary' 'audio-only summary heading'

# --- 2. audio-video happy path ------------------------------------------------
analyze audio-video >"$TEMP_DIR/audio-video.out" 2>&1
assert_grep "$TEMP_DIR/audio-video.out" "$PASS_HORIZON" 'audio-video scheduled_horizon'
assert_grep "$TEMP_DIR/audio-video.out" "$PASS_CADENCE" 'audio-video cadence'
assert_grep "$TEMP_DIR/audio-video.out" "$PASS_CLOCK" 'audio-video clock_offset'
assert_grep "$SUMMARY" '# audio-video Summary' 'audio-video summary heading'

# --- 3. scheduled horizon > 150 ms should fail --------------------------------
cat > "$BROWSER_LOG" <<'EOF'
audio_stream_metrics {"capturedAtMs":1000,"framesReceived":0,"invalidFrames":0,"sequenceGaps":0,"duplicates":0,"regressions":0,"interArrivalMs":{"max":50},"scheduledHorizonMs":{"max":100},"transport":"websocket","underruns":0,"drops":0}
audio_stream_metrics {"capturedAtMs":592000,"framesReceived":11820,"invalidFrames":0,"sequenceGaps":0,"duplicates":0,"regressions":0,"interArrivalMs":{"max":55},"scheduledHorizonMs":{"max":180},"transport":"websocket","underruns":0,"drops":0}
EOF
analyze audio-only >"$TEMP_DIR/bad-horizon.out" 2>&1 || true
assert_grep "$TEMP_DIR/bad-horizon.out" "$FAIL_HORIZON" 'bad-horizon scheduled_horizon'

# --- 4. underrun > 0 should fail ----------------------------------------------
cat > "$BROWSER_LOG" <<'EOF'
audio_stream_metrics {"capturedAtMs":1000,"framesReceived":0,"invalidFrames":0,"sequenceGaps":0,"duplicates":0,"regressions":0,"interArrivalMs":{"max":50},"scheduledHorizonMs":{"max":100},"transport":"websocket","underruns":0,"drops":0}
audio_stream_metrics {"capturedAtMs":592000,"framesReceived":11820,"invalidFrames":0,"sequenceGaps":0,"duplicates":0,"regressions":0,"interArrivalMs":{"max":55},"scheduledHorizonMs":{"max":120},"transport":"websocket","underruns":3,"drops":0}
EOF
analyze audio-only >"$TEMP_DIR/bad-underrun.out" 2>&1 || true
assert_grep "$TEMP_DIR/bad-underrun.out" "$FAIL_UNDERRUNS" 'bad-underrun underruns'

# --- 5. drop > 0 should fail --------------------------------------------------
cat > "$BROWSER_LOG" <<'EOF'
audio_stream_metrics {"capturedAtMs":1000,"framesReceived":0,"invalidFrames":0,"sequenceGaps":0,"duplicates":0,"regressions":0,"interArrivalMs":{"max":50},"scheduledHorizonMs":{"max":100},"transport":"websocket","underruns":0,"drops":0}
audio_stream_metrics {"capturedAtMs":592000,"framesReceived":11820,"invalidFrames":0,"sequenceGaps":0,"duplicates":0,"regressions":0,"interArrivalMs":{"max":55},"scheduledHorizonMs":{"max":120},"transport":"websocket","underruns":0,"drops":7}
EOF
analyze audio-only >"$TEMP_DIR/bad-drop.out" 2>&1 || true
assert_grep "$TEMP_DIR/bad-drop.out" "$FAIL_DROPS" 'bad-drop drops'

# --- 6. binary reduction < 65 % should fail -----------------------------------
cat > "$BROWSER_LOG" <<'EOF'
audio_stream_metrics {"capturedAtMs":1000,"framesReceived":0,"invalidFrames":0,"sequenceGaps":0,"duplicates":0,"regressions":0,"interArrivalMs":{"max":50},"scheduledHorizonMs":{"max":100},"transport":"websocket","underruns":0,"drops":0}
audio_stream_metrics {"capturedAtMs":592000,"framesReceived":11820,"invalidFrames":0,"sequenceGaps":0,"duplicates":0,"regressions":0,"interArrivalMs":{"max":55},"scheduledHorizonMs":{"max":140},"transport":"websocket","underruns":0,"drops":0}
EOF
"$HELPER" analyze \
  --scenario audio-only \
  --browser-log "$BROWSER_LOG" \
  --rover-log "$ROVER_LOG" \
  --orchestra-log "$ORCHESTRA_LOG" \
  --network-path LAN --browser-host b --rover-host r --orchestra-host o \
  --devtools closed --clock-offset-ms 2 \
  --output "$SUMMARY" \
  --json-baseline-bytes 5911 --binary-payload-bytes 4000 \
  >"$TEMP_DIR/bad-reduction.out" 2>&1 || true
assert_grep "$TEMP_DIR/bad-reduction.out" "$FAIL_REDUCTION" 'bad-reduction binary_reduction'

# --- 7. missing rover_publish s16le marker should WARN, not FAIL --------------
cat > "$ROVER_LOG" <<'EOF'
metric="audio_pipeline" stage="rover_convert" errors=0
metric="audio_pipeline" stage="orchestra_zenoh_receive" errors=0
EOF
analyze audio-only >"$TEMP_DIR/no-s16le.out" 2>&1 || true
assert_grep "$TEMP_DIR/no-s16le.out" "$WARN_FORMAT" 'missing s16le marker warns'

# --- 8. audio stop/start control event still fails ---------------------------
write_rover_logs
printf '\n%s\n' 'Audio control received: Stop' 'Stopping audio stream' >> "$ROVER_LOG"
analyze audio-only >"$TEMP_DIR/control-event.out" 2>&1 || true
assert_grep "$TEMP_DIR/control-event.out" "$FAIL_CONTROL" 'control_events'

printf 'benchmark helper tests passed\n'
