#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PLAN_DIR="$ROOT/plans/260628-1619-audio-video-stream-performance-approach-a"
SCENARIO=""
BROWSER_LOG=""
ROVER_LOG=""
ORCHESTRA_LOG=""
OUTPUT=""
NETWORK_PATH=""
BROWSER_HOST=""
ROVER_HOST=""
ORCHESTRA_HOST=""
DEVTOOLS_STATE=""
CLOCK_OFFSET_MS=""
EXPECTED_DURATION_SECONDS=600
EXPECTED_FPS=20
FPS_TOLERANCE=1
FAILURES=0

usage() {
  cat <<'USAGE'
usage: benchmark-audio-video-stream.sh analyze [options]

Required:
  --scenario audio-only|audio-video
  --browser-log FILE --rover-log FILE --orchestra-log FILE
  --network-path TEXT
  --browser-host TEXT --rover-host TEXT --orchestra-host TEXT
  --devtools closed|open --clock-offset-ms NUMBER

Optional:
  --output FILE                    Default: phase report directory
  --duration-seconds NUMBER        Default: 600
  --expected-fps NUMBER            Default: 20
  --fps-tolerance NUMBER           Default: 1

Browser logs must contain lines emitted with ?audioDebug=1:
  audio_stream_metrics {JSON snapshot}
USAGE
}

result() {
  local status=$1 name=$2 detail=$3
  printf '%s\t%s\t%s\n' "$status" "$name" "$detail"
  if [[ $status == FAIL ]]; then FAILURES=$((FAILURES + 1)); fi
}

require_value() {
  [[ -n ${2:-} ]] || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
}

parse_args() {
  while (($#)); do
    case $1 in
      --scenario) require_value "$1" "${2:-}"; SCENARIO=$2; shift 2 ;;
      --browser-log) require_value "$1" "${2:-}"; BROWSER_LOG=$2; shift 2 ;;
      --rover-log) require_value "$1" "${2:-}"; ROVER_LOG=$2; shift 2 ;;
      --orchestra-log) require_value "$1" "${2:-}"; ORCHESTRA_LOG=$2; shift 2 ;;
      --network-path) require_value "$1" "${2:-}"; NETWORK_PATH=$2; shift 2 ;;
      --browser-host) require_value "$1" "${2:-}"; BROWSER_HOST=$2; shift 2 ;;
      --rover-host) require_value "$1" "${2:-}"; ROVER_HOST=$2; shift 2 ;;
      --orchestra-host) require_value "$1" "${2:-}"; ORCHESTRA_HOST=$2; shift 2 ;;
      --devtools) require_value "$1" "${2:-}"; DEVTOOLS_STATE=$2; shift 2 ;;
      --clock-offset-ms) require_value "$1" "${2:-}"; CLOCK_OFFSET_MS=$2; shift 2 ;;
      --output) require_value "$1" "${2:-}"; OUTPUT=$2; shift 2 ;;
      --duration-seconds) require_value "$1" "${2:-}"; EXPECTED_DURATION_SECONDS=$2; shift 2 ;;
      --expected-fps) require_value "$1" "${2:-}"; EXPECTED_FPS=$2; shift 2 ;;
      --fps-tolerance) require_value "$1" "${2:-}"; FPS_TOLERANCE=$2; shift 2 ;;
      *) printf 'unknown argument: %s\n' "$1" >&2; usage; exit 2 ;;
    esac
  done
}

validate_inputs() {
  [[ $SCENARIO == audio-only || $SCENARIO == audio-video ]] || result FAIL scenario "$SCENARIO"
  [[ $DEVTOOLS_STATE == closed || $DEVTOOLS_STATE == open ]] || result FAIL devtools "$DEVTOOLS_STATE"
  for value in NETWORK_PATH BROWSER_HOST ROVER_HOST ORCHESTRA_HOST CLOCK_OFFSET_MS; do
    [[ -n ${!value} ]] || result FAIL "${value,,}" "missing"
  done
  for item in "$BROWSER_LOG" "$ROVER_LOG" "$ORCHESTRA_LOG"; do
    [[ -r $item ]] || result FAIL input_log "$item is not readable"
  done
  command -v jq >/dev/null || result FAIL jq "jq is required"
  ((FAILURES == 0))
}

analyze() {
  validate_inputs || return 1
  local snapshots summary elapsed frames fps min_fps max_fps transport
  snapshots=$(mktemp)
  trap "rm -f '$snapshots'" EXIT
  sed -n 's/^.*audio_stream_metrics[[:space:]]\+//p' "$BROWSER_LOG" > "$snapshots"
  if ! jq -e -s 'length >= 2' "$snapshots" >/dev/null 2>&1; then
    result FAIL browser_metrics "fewer than two valid snapshots"
    return 1
  fi

  summary=$(jq -s '{
    first: .[0], last: .[-1], transports: ([.[].transport] | unique),
    maxInterArrivalMs: ([.[].interArrivalMs.max // 0] | max),
    maxHorizonMs: ([.[].scheduledHorizonMs.max // 0] | max)
  }' "$snapshots")
  elapsed=$(jq -r '(.last.capturedAtMs - .first.capturedAtMs) / 1000' <<<"$summary")
  frames=$(jq -r '.last.framesReceived - .first.framesReceived' <<<"$summary")
  fps=$(awk -v frames="$frames" -v elapsed="$elapsed" 'BEGIN { if (elapsed > 0) printf "%.3f", frames / elapsed; else print 0 }')
  min_fps=$(awk -v fps="$EXPECTED_FPS" -v tolerance="$FPS_TOLERANCE" 'BEGIN { print fps - tolerance }')
  max_fps=$(awk -v fps="$EXPECTED_FPS" -v tolerance="$FPS_TOLERANCE" 'BEGIN { print fps + tolerance }')
  transport=$(jq -r '.transports | join(",")' <<<"$summary")

  awk -v actual="$elapsed" -v expected="$EXPECTED_DURATION_SECONDS" 'BEGIN { exit !(actual >= expected - 10) }' &&
    result PASS duration "${elapsed}s" || result FAIL duration "${elapsed}s; expected at least $((EXPECTED_DURATION_SECONDS - 10))s"
  awk -v actual="$fps" -v low="$min_fps" -v high="$max_fps" 'BEGIN { exit !(actual >= low && actual <= high) }' &&
    result PASS cadence "${fps} fps" || result FAIL cadence "${fps} fps outside ${min_fps}-${max_fps}"
  [[ -n $transport && $transport != unknown ]] && result PASS transport "$transport" || result FAIL transport "$transport"
  awk -v offset="$CLOCK_OFFSET_MS" 'BEGIN { if (offset < 0) offset = -offset; exit !(offset <= 5) }' &&
    result PASS clock_offset "${CLOCK_OFFSET_MS} ms; age metric valid" ||
    result WARN clock_offset "${CLOCK_OFFSET_MS} ms; capture-age acceptance invalid"

  local counter value
  for counter in invalidFrames sequenceGaps duplicates regressions; do
    value=$(jq -r ".last.$counter" <<<"$summary")
    [[ $value == 0 ]] && result PASS "$counter" "$value" || result FAIL "$counter" "$value"
  done

  grep -Eq 'metric[ ="]+audio_pipeline.*errors=[1-9][0-9]*' "$ROVER_LOG" "$ORCHESTRA_LOG" &&
    result FAIL backend_errors "non-zero audio pipeline errors" || result PASS backend_errors "none"
  grep -Eq 'stage[ ="]+rover_convert' "$ROVER_LOG" &&
    result PASS rover_converter "conversion metrics present" || result FAIL rover_converter "conversion metrics absent"
  grep -Eqi 'audio[ _]control received:[[:space:]]*(start|stop)|start(ing)? audio stream|stop(ping)? audio stream' "$ROVER_LOG" "$ORCHESTRA_LOG" &&
    result FAIL control_events "audio stop/start command found" || result PASS control_events "none"

  OUTPUT=${OUTPUT:-"$PLAN_DIR/reports/phase-02-${SCENARIO}-summary.md"}
  mkdir -p "$(dirname "$OUTPUT")"
  {
    printf '# Phase 02 %s Summary\n\n' "$SCENARIO"
    printf -- '- Network path: %s\n- Browser host: %s\n- Orchestra host: %s\n- Rover host: %s\n' \
      "$NETWORK_PATH" "$BROWSER_HOST" "$ORCHESTRA_HOST" "$ROVER_HOST"
    printf -- '- DevTools: %s\n- Clock offset: %s ms\n- Duration: %s seconds\n- Cadence: %s fps\n- Engine.IO transport: %s\n' \
      "$DEVTOOLS_STATE" "$CLOCK_OFFSET_MS" "$elapsed" "$fps" "$transport"
    printf '\n## Browser snapshot summary\n\n```json\n%s\n```\n' "$summary"
    printf '\n## Gate\n\n- Failures: %s\n' "$FAILURES"
  } > "$OUTPUT"
  result PASS summary "$OUTPUT"
  ((FAILURES == 0))
}

[[ ${1:-} == analyze ]] || { usage; exit 2; }
shift
parse_args "$@"
analyze
