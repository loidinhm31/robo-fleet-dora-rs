#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HELPER="$ROOT/scripts/benchmark-audio-video-stream.sh"
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

write_valid_logs() {
  printf '%s\n' \
    'audio_stream_metrics {"capturedAtMs":1000,"framesReceived":0,"invalidFrames":0,"sequenceGaps":0,"duplicates":0,"regressions":0,"interArrivalMs":{"max":50},"scheduledHorizonMs":{"max":100},"transport":"websocket"}' \
    'audio_stream_metrics {"capturedAtMs":592000,"framesReceived":11820,"invalidFrames":0,"sequenceGaps":0,"duplicates":0,"regressions":0,"interArrivalMs":{"max":55},"scheduledHorizonMs":{"max":120},"transport":"websocket"}' \
    > "$TEMP_DIR/browser.log"
  printf '%s\n' 'metric="audio_pipeline" stage="rover_convert" errors=0' > "$TEMP_DIR/rover.log"
  printf '%s\n' 'metric="audio_pipeline" stage="orchestra_zenoh_receive" errors=0' > "$TEMP_DIR/orchestra.log"
}

analyze() {
  "$HELPER" analyze \
    --scenario audio-only \
    --browser-log "$TEMP_DIR/browser.log" \
    --rover-log "$TEMP_DIR/rover.log" \
    --orchestra-log "$TEMP_DIR/orchestra.log" \
    --network-path LAN \
    --browser-host browser-a \
    --rover-host rover-a \
    --orchestra-host orchestra-a \
    --devtools closed \
    --clock-offset-ms 2 \
    --output "$TEMP_DIR/summary.md"
}

write_valid_logs
analyze >/dev/null

printf '%s\n' 'Audio control received: Stop' 'Stopping audio stream' >> "$TEMP_DIR/rover.log"
if analyze >"$TEMP_DIR/control-event-result.log" 2>&1; then
  printf 'expected control-event analysis to fail\n' >&2
  exit 1
fi
grep -q $'FAIL\tcontrol_events\taudio stop/start command found' "$TEMP_DIR/control-event-result.log"

if "$HELPER" analyze --scenario invalid >"$TEMP_DIR/invalid-result.log" 2>&1; then
  printf 'expected invalid-input analysis to fail\n' >&2
  exit 1
fi
grep -q $'FAIL\tscenario\tinvalid' "$TEMP_DIR/invalid-result.log"

printf 'benchmark helper tests passed\n'
