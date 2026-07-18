#!/bin/bash
set -e

. /app/scripts/sherpa-stt-profile-files.sh

RECORDING_ROOT="${RECORDING_ROOT:-/recordings}"
RECORDING_MIN_FREE_BYTES="${RECORDING_MIN_FREE_BYTES:-1073741824}"
HOST_RECORDING_PATH="${HOST_RECORDING_PATH:-}"
export RECORDING_ROOT RECORDING_MIN_FREE_BYTES

fail_recording_readiness() {
    echo "ERROR: orchestra recording readiness failed: $1" >&2
    exit 1
}

case "$RECORDING_MIN_FREE_BYTES" in
    ''|*[!0-9]*) fail_recording_readiness "RECORDING_MIN_FREE_BYTES must be an integer" ;;
esac

[ "$RECORDING_ROOT" = "/recordings" ] || fail_recording_readiness "RECORDING_ROOT must be /recordings in the container"
case "$HOST_RECORDING_PATH" in
    /home/*) ;;
    *) fail_recording_readiness "HOST_RECORDING_PATH must be an absolute directory below /home" ;;
esac
[ "$HOST_RECORDING_PATH" != "/home" ] || fail_recording_readiness "HOST_RECORDING_PATH must not be /home"
case "$HOST_RECORDING_PATH" in
    *..*) fail_recording_readiness "HOST_RECORDING_PATH must not contain '..'" ;;
esac
[ ! -e /recordings/.image-layer-recording-root-marker ] || fail_recording_readiness "HOST_RECORDING_PATH is not mounted"
[ -d "$RECORDING_ROOT" ] || fail_recording_readiness "recording directory is missing"
[ -w "$RECORDING_ROOT" ] || fail_recording_readiness "recording directory is not writable"

available_kb="$(df -Pk "$RECORDING_ROOT" | awk 'NR==2 {print $4}')"
[ -n "$available_kb" ] && [ "$available_kb" -ge "$((RECORDING_MIN_FREE_BYTES / 1024))" ] || \
    fail_recording_readiness "recording directory has insufficient free space"

ffmpeg -hide_banner -encoders 2>&1 | grep -Eq '[[:space:]]libx264[[:space:]]' || \
    fail_recording_readiness "FFmpeg libx264 encoder is unavailable"
ffmpeg -hide_banner -encoders 2>&1 | grep -Eq '[[:space:]]aac[[:space:]]' || \
    fail_recording_readiness "FFmpeg AAC encoder is unavailable"
ffmpeg -hide_banner -muxers 2>&1 | grep -Eq '[[:space:]]E[[:space:]]mp4[[:space:]]' || \
    fail_recording_readiness "FFmpeg MP4 muxer is unavailable"
ffprobe -version >/dev/null 2>&1 || fail_recording_readiness "ffprobe is unavailable"

recording_probe="$RECORDING_ROOT/.orchestra-readiness.$$"
recording_probe_renamed="$recording_probe.renamed"
trap 'rm -f "$recording_probe" "$recording_probe_renamed"' EXIT
printf 'ready' > "$recording_probe" || fail_recording_readiness "recording root create failed"
sync -f "$recording_probe" || fail_recording_readiness "recording root fsync failed"
mv "$recording_probe" "$recording_probe_renamed" || fail_recording_readiness "recording root rename failed"
[ "$(cat "$recording_probe_renamed")" = "ready" ] || fail_recording_readiness "recording root read failed"
rm -f "$recording_probe_renamed"
trap - EXIT

echo "==================================================================="
echo "  Robo-Fleet Orchestra Container Starting"
echo "==================================================================="

STT_PROFILE="${STT_PROFILE:-en-vad-offline}"
STT_MODEL_ROOT="${STT_MODEL_ROOT:-/models/sherpa-onnx/asr}"
export ORCHESTRA_ZENOH_CONFIG="${ORCHESTRA_ZENOH_CONFIG:-/app/config/zenoh_config.json5}"
ORCHESTRA_ZENOH_LISTEN_ENDPOINT="${ORCHESTRA_ZENOH_LISTEN_ENDPOINT:-}"
required_files="$(mktemp /tmp/orchestra-stt-required-files.XXXXXX)"
trap 'rm -f "$required_files" /tmp/orchestra-zenoh_config.json5' EXIT

echo "Checking required Sherpa STT models..."
if ! required_stt_files "$STT_PROFILE" >"$required_files"; then
    echo "ERROR: invalid STT_PROFILE '$STT_PROFILE'"
    echo "Valid values: en-vad-offline, vi-vad-offline"
    exit 1
fi
while IFS= read -r relative_path; do
    [ -n "$relative_path" ] || continue
    if [ ! -f "$STT_MODEL_ROOT/$relative_path" ]; then
        echo "ERROR: required STT model file missing: $relative_path"
        echo "Expected under: $STT_MODEL_ROOT"
        echo ""
        echo "Download models with:"
        echo "  make models"
        echo "  or"
        echo "  ./docker/scripts/download-models.sh"
        exit 1
    fi
done < "$required_files"

echo "✓ Sherpa STT profile '$STT_PROFILE' is available"

if [ -n "$ORCHESTRA_ZENOH_LISTEN_ENDPOINT" ]; then
    tmp_zenoh_config="/tmp/orchestra-zenoh_config.json5"
    cp "$ORCHESTRA_ZENOH_CONFIG" "$tmp_zenoh_config"
    escaped_orchestra_zenoh_listen_endpoint="$(printf '%s\n' "$ORCHESTRA_ZENOH_LISTEN_ENDPOINT" | sed 's/[&\\]/\\&/g')"
    sed -i -E "s|\"tcp/[^\"]+\"|\"$escaped_orchestra_zenoh_listen_endpoint\"|" "$tmp_zenoh_config"
    export ORCHESTRA_ZENOH_CONFIG="$tmp_zenoh_config"
    echo "✓ Orchestra Zenoh listen endpoint override: $ORCHESTRA_ZENOH_LISTEN_ENDPOINT"
fi

# Create a modified dataflow YAML with updated paths
echo "Updating dataflow YAML paths for container environment..."
cp /app/dataflow/orchestra-dataflow.yml /tmp/orchestra-dataflow.yml

# Update binary paths from ../target/release/ to /app/bin/
sed -i 's|path: ../target/release/|path: /app/bin/|g' /tmp/orchestra-dataflow.yml

echo "✓ Dataflow YAML updated"

# Display configuration
echo ""
echo "Configuration:"
echo "  - Entity ID: ${ENTITY_ID:-orchestra}"
echo "  - Zenoh Mode: ${ZENOH_MODE:-peer}"
echo "  - Zenoh Config: ${ORCHESTRA_ZENOH_CONFIG}"
echo "  - Socket.IO Port: ${SOCKET_IO_PORT:-3030}"
echo "  - Active Rovers: ${ACTIVE_ROVERS:-rover-kiwi}"
echo "  - STT Profile: ${STT_PROFILE}"
echo "  - STT Model Root: ${STT_MODEL_ROOT}"
echo ""

echo ""
echo "Starting Orchestra dataflow..."
echo "==================================================================="

# Start the dataflow locally as dora-rs best practice in docker
exec dora run /tmp/orchestra-dataflow.yml
