#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PLAN_DIR="$ROOT/plans/260619-0044-resolve-rover-video-pipeline-bottleneck"
OUTPUT_DIR=${BENCHMARK_OUTPUT_DIR:-"$PLAN_DIR/reports"}
CORPUS_DIR=${BENCHMARK_CORPUS_DIR:-"$ROOT/out/video-benchmark-corpus"}
CAMERA=${ROVER_CAMERA_URI:-}
YOLO_MODEL=${ROVER_YOLO_MODEL_PATH:-"$ROOT/models/.cache/yolo/yolo12n.onnx"}
REID_MODEL=${ROVER_REID_MODEL_PATH:-"$ROOT/models/.cache/reid/osnet_x0_25.onnx"}
ORT_LIBRARY=${ROVER_ORT_DYLIB_PATH:-/usr/local/lib/libonnxruntime.so}
FAILURES=0

mkdir -p "$OUTPUT_DIR"

result() {
  local status=$1 name=$2 detail=$3
  printf '%s\t%s\t%s\n' "$status" "$name" "$detail"
  if [[ $status == FAIL ]]; then FAILURES=$((FAILURES + 1)); fi
}

resolve_camera() {
  if [[ -n $CAMERA ]]; then return; fi
  CAMERA=$(find /dev/v4l/by-id -maxdepth 1 -type l 2>/dev/null | sort | head -1)
  CAMERA=${CAMERA:-/dev/video0}
}

find_ort() {
  if [[ -f $ORT_LIBRARY ]]; then return; fi
  ORT_LIBRARY=$(find "$HOME/.cache/sherpa-rs" -name 'libonnxruntime.so' -type f 2>/dev/null | head -1)
}

preflight() {
  resolve_camera
  find_ort
  export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
  [[ -e $CAMERA ]] && result PASS camera "$CAMERA" || result FAIL camera "$CAMERA missing"
  if command -v v4l2-ctl >/dev/null && [[ -e $CAMERA ]]; then
    local formats
    formats=$(v4l2-ctl --device "$CAMERA" --list-formats-ext 2>&1)
    if grep -q '640x480' <<<"$formats" && grep -Eq '30\.000 fps|\(30\.000 fps\)' <<<"$formats"; then
      result PASS camera_format '640x480@30 advertised'
    else
      result FAIL camera_format '640x480@30 not advertised'
    fi
  else
    result FAIL v4l2_ctl 'command or camera unavailable'
  fi
  [[ -f $YOLO_MODEL ]] && result PASS yolo_model "$YOLO_MODEL" || result FAIL yolo_model "$YOLO_MODEL missing"
  [[ -f $REID_MODEL ]] && result PASS reid_model "$REID_MODEL" || result FAIL reid_model "$REID_MODEL missing"
  [[ -f $ORT_LIBRARY ]] && result PASS onnxruntime "$ORT_LIBRARY" || result FAIL onnxruntime "$ORT_LIBRARY missing"
  command -v dora >/dev/null && result PASS dora "$(dora --version 2>&1 | head -1)" || result FAIL dora 'not installed'
  if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
    result PASS docker_info 'runtime responsive'
    if timeout 30 docker run --rm hello-world >/dev/null 2>&1; then
      result PASS docker_smoke 'hello-world completed'
    else
      result FAIL docker_smoke 'hello-world failed'
    fi
  else
    result FAIL docker_info 'runtime unavailable'
  fi
  printf 'ROVER_CAMERA_URI=%q\nROVER_YOLO_MODEL_PATH=%q\nROVER_REID_MODEL_PATH=%q\nROVER_ORT_DYLIB_PATH=%q\n' \
    "$CAMERA" "$YOLO_MODEL" "$REID_MODEL" "$ORT_LIBRARY"
  ((FAILURES == 0))
}

capture_corpus() {
  resolve_camera
  [[ -e $CAMERA ]] || { result FAIL camera "$CAMERA missing"; return 1; }
  mkdir -p "$CORPUS_DIR"
  local frames=${CORPUS_FRAMES:-300}
  if command -v ffmpeg >/dev/null; then
    ffmpeg -hide_banner -loglevel error -f v4l2 -video_size 640x480 -framerate 30 \
      -i "$CAMERA" -frames:v "$frames" "$CORPUS_DIR/frame-%06d.ppm"
  elif command -v gst-launch-1.0 >/dev/null; then
    gst-launch-1.0 -q v4l2src device="$CAMERA" num-buffers="$frames" \
      ! video/x-raw,width=640,height=480,framerate=30/1 \
      ! videoconvert ! video/x-raw,format=RGB \
      ! multifilesink location="$CORPUS_DIR/frame-%06d.rgb"
  else
    result FAIL capture_tool 'ffmpeg and gst-launch-1.0 unavailable'
    return 1
  fi
  find "$CORPUS_DIR" -type f \( -name 'frame-*.ppm' -o -name 'frame-*.rgb' \) \
    -print0 | sort -z | xargs -0 sha256sum \
    > "$OUTPUT_DIR/corpus-sha256.txt"
  printf 'camera=%s\nwidth=640\nheight=480\nfps=30\nframes=%s\n' \
    "$CAMERA" "$frames" > "$OUTPUT_DIR/corpus-metadata.txt"
  result PASS corpus "$CORPUS_DIR ($(wc -l < "$OUTPUT_DIR/corpus-sha256.txt") frames)"
}

collect() {
  local scenario=${SCENARIO:-unspecified}
  local stamp
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  local report="$OUTPUT_DIR/${stamp}-${scenario}.log"
  {
    printf 'scenario=%s\ntimestamp_utc=%s\nhost=%s\nkernel=%s\n' \
      "$scenario" "$stamp" "$(hostname)" "$(uname -srmo)"
    printf 'cpu_count=%s\nmemory_kib=%s\n' "$(nproc)" "$(awk '/MemTotal/{print $2}' /proc/meminfo)"
    ps -eo pid,comm,%cpu,rss --sort=-%cpu | head -30
    docker stats --no-stream 2>&1 || true
    docker inspect robo-rover-kiwi --format \
      'nano_cpus={{.HostConfig.NanoCpus}} memory={{.HostConfig.Memory}} oom_killed={{.State.OOMKilled}}' 2>&1 || true
    docker exec robo-rover-kiwi sh -c '
      for path in cpu.stat memory.current memory.peak memory.events; do
        file=/sys/fs/cgroup/$path
        if [ -r "$file" ]; then
          printf "cgroup_%s_begin\n" "$path"
          cat "$file"
          printf "cgroup_%s_end\n" "$path"
        else
          printf "cgroup_%s=unavailable\n" "$path"
        fi
      done
    ' 2>&1 || true
    ss -tinp 2>&1 || true
  } > "$report"
  result PASS collection "$report"
}

monitor() {
  local scenario=${SCENARIO:-unspecified}
  local duration=${BENCHMARK_DURATION_SECONDS:-600}
  local interval=${BENCHMARK_SAMPLE_SECONDS:-5}
  local stamp report deadline
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  report="$OUTPUT_DIR/${stamp}-${scenario}-monitor.log"
  deadline=$((SECONDS + duration))
  {
    printf 'scenario=%s\ntimestamp_utc=%s\nduration_seconds=%s\nsample_seconds=%s\n' \
      "$scenario" "$stamp" "$duration" "$interval"
    docker inspect robo-rover-kiwi --format \
      'nano_cpus={{.HostConfig.NanoCpus}} memory={{.HostConfig.Memory}} oom_killed={{.State.OOMKilled}}' 2>&1 || true
    printf 'cgroup_start\n'
    docker exec robo-rover-kiwi sh -c \
      'cat /sys/fs/cgroup/cpu.stat; cat /sys/fs/cgroup/memory.current; cat /sys/fs/cgroup/memory.peak; cat /sys/fs/cgroup/memory.events' 2>&1 || true
    printf 'samples_begin\n'
    while ((SECONDS < deadline)); do
      printf 'sample_utc=%s\t' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      docker stats --no-stream --format \
        'cpu={{.CPUPerc}}\tmemory={{.MemUsage}}' robo-rover-kiwi 2>&1 || true
      sleep "$interval"
    done
    printf 'samples_end\ncgroup_end\n'
    docker exec robo-rover-kiwi sh -c \
      'cat /sys/fs/cgroup/cpu.stat; cat /sys/fs/cgroup/memory.current; cat /sys/fs/cgroup/memory.peak; cat /sys/fs/cgroup/memory.events' 2>&1 || true
    docker inspect robo-rover-kiwi --format \
      'nano_cpus={{.HostConfig.NanoCpus}} memory={{.HostConfig.Memory}} oom_killed={{.State.OOMKilled}}' 2>&1 || true
  } > "$report"
  result PASS monitor "$report"
}

case ${1:-preflight} in
  preflight) preflight ;;
  capture-corpus) capture_corpus ;;
  collect) collect ;;
  monitor) monitor ;;
  *) printf 'usage: %s {preflight|capture-corpus|collect|monitor}\n' "$0" >&2; exit 2 ;;
esac
