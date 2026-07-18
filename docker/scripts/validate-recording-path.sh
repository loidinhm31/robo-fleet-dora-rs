#!/usr/bin/env bash
set -euo pipefail

path="${HOST_RECORDING_PATH:-}"
if [[ -z "$path" ]]; then
    echo "ERROR: HOST_RECORDING_PATH is required" >&2
    exit 1
fi
case "$path" in
    /home/*) ;;
    *) echo "ERROR: HOST_RECORDING_PATH must be below /home" >&2; exit 1 ;;
esac
if [[ "$path" == *..* || -L "$path" || ! -d "$path" ]]; then
    echo "ERROR: HOST_RECORDING_PATH must be an existing non-symlink directory" >&2
    exit 1
fi

resolved_path="$(realpath -- "$path")"
case "$resolved_path" in
    /home/*) ;;
    *) echo "ERROR: HOST_RECORDING_PATH resolves outside /home" >&2; exit 1 ;;
esac
[[ "$resolved_path" != "/home" ]] || {
    echo "ERROR: HOST_RECORDING_PATH must not be /home" >&2
    exit 1
}

echo "✓ Recording host path is a dedicated directory below /home"
