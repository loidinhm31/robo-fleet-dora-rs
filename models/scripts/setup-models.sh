#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=./model-manifest.sh
. "$SCRIPT_DIR/model-manifest.sh"

CACHE_DIR="$(model_cache_dir)"
RUNTIME_DIR="$(model_runtime_dir)"
WORK_ROOT="$PROJECT_ROOT/models"
VENV_DIR="$SCRIPT_DIR/venv"
declare -a CLEANUP_PATHS=()
SWAP_BACKUP_DIR=""

cleanup() {
    local idx path
    if [ -n "$SWAP_BACKUP_DIR" ] && [ -d "$SWAP_BACKUP_DIR" ]; then
        if [ -d "$CACHE_DIR" ]; then
            rm -rf -- "$SWAP_BACKUP_DIR"
        else
            mv -- "$SWAP_BACKUP_DIR" "$CACHE_DIR" || printf 'WARN: failed to restore cache backup from %s\n' "$SWAP_BACKUP_DIR" >&2
        fi
        SWAP_BACKUP_DIR=""
    fi
    for ((idx=${#CLEANUP_PATHS[@]}-1; idx>=0; idx--)); do
        path="${CLEANUP_PATHS[$idx]}"
        [ -e "$path" ] || continue
        rm -rf -- "$path"
    done
}
trap cleanup EXIT INT TERM

log() {
    printf '%s\n' "$*"
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

register_cleanup_path() {
    local path="$1"
    local existing
    [ -n "$path" ] || return 0
    for existing in "${CLEANUP_PATHS[@]}"; do
        [ "$existing" = "$path" ] && return 0
    done
    CLEANUP_PATHS+=("$path")
}

unregister_cleanup_path() {
    local path="$1"
    local existing
    local -a remaining=()
    for existing in "${CLEANUP_PATHS[@]}"; do
        [ "$existing" = "$path" ] || remaining+=("$existing")
    done
    CLEANUP_PATHS=("${remaining[@]}")
}

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

ensure_parent_dir() {
    mkdir -p "$(dirname "$1")"
}

warn_stale_cache_dirs() {
    local path
    local -a stale_paths=()
    while IFS= read -r path; do
        [ -n "$path" ] || continue
        stale_paths+=("$path")
    done < <(find "$WORK_ROOT" -maxdepth 1 -type d \( -name '.cache-reset-*' -o -name '.cache-backup-*' \) | sort)

    [ "${#stale_paths[@]}" -gt 0 ] || return 0

    printf 'WARN: stale cache staging directories detected:\n' >&2
    for path in "${stale_paths[@]}"; do
        printf '  %s\n' "${path#$PROJECT_ROOT/}" >&2
    done
    printf "  Remove them manually after confirming they are obsolete.\n" >&2
}

download_file() {
    local url="$1"
    local destination="$2"
    local temporary="${destination}.part"
    ensure_parent_dir "$destination"
    rm -f "$temporary"
    register_cleanup_path "$temporary"
    curl -L --fail --silent --show-error -o "$temporary" "$url"
    mv "$temporary" "$destination"
    unregister_cleanup_path "$temporary"
}

safe_extract_tarball() {
    local archive_path="$1"
    local destination_dir="$2"
    local compression_flag="$3"
    local entry metadata

    while IFS= read -r metadata; do
        [ -n "$metadata" ] || continue
        case "${metadata:0:1}" in
            -|d) ;;
            *)
                fail "unsafe archive member type in $(basename "$archive_path"): $metadata"
                ;;
        esac
    done < <(tar --list --verbose "$compression_flag" --file "$archive_path")

    while IFS= read -r entry; do
        [ -n "$entry" ] || continue
        entry="${entry#./}"
        case "$entry" in
            /*|../*|*/../*|..)
                fail "unsafe archive entry '$entry' in $(basename "$archive_path")"
                ;;
        esac
    done < <(tar --list "$compression_flag" --file "$archive_path")

    mkdir -p "$destination_dir"
    tar --extract "$compression_flag" --file "$archive_path" -C "$destination_dir"
}

validate_specs() {
    local base_dir="$1"
    local specs="$2"
    local status=0
    local expected_hash relative_path actual_hash

    while IFS= read -r spec; do
        [ -n "$spec" ] || continue
        expected_hash="${spec%%  *}"
        relative_path="${spec#*  }"
        if [ ! -f "$base_dir/$relative_path" ]; then
            status=1
            continue
        fi
        actual_hash="$(sha256_file "$base_dir/$relative_path")"
        [ "$actual_hash" = "$expected_hash" ] || status=1
    done <<< "$specs"

    return "$status"
}

report_specs() {
    local label="$1"
    local base_dir="$2"
    local specs="$3"
    local expected_hash relative_path actual_hash

    while IFS= read -r spec; do
        [ -n "$spec" ] || continue
        expected_hash="${spec%%  *}"
        relative_path="${spec#*  }"
        if [ ! -f "$base_dir/$relative_path" ]; then
            printf '  %s MISSING %s\n' "$label" "$relative_path"
            continue
        fi
        actual_hash="$(sha256_file "$base_dir/$relative_path")"
        if [ "$actual_hash" = "$expected_hash" ]; then
            printf '  %s OK %s\n' "$label" "$relative_path"
        else
            printf '  %s CORRUPT %s\n' "$label" "$relative_path"
        fi
    done <<< "$specs"
}

repo_validator_ready() {
    [ -x "$VENV_DIR/bin/python" ] || return 1
    "$VENV_DIR/bin/python" -c 'import onnx' >/dev/null 2>&1
}

ensure_validator_tools() {
    ensure_python_venv
    if ! repo_validator_ready; then
        "$VENV_DIR/bin/python" -m pip install --quiet onnx==1.22.0
    fi
}

validate_yolo_export() {
    local model_path="$1/$YOLO12N_ONNX_PATH"
    [ -f "$model_path" ] || return 1
    repo_validator_ready || return 1
    MODEL_PATH="$model_path" "$VENV_DIR/bin/python" - <<'PYTHON' >/dev/null
import os

import onnx

model = onnx.load(os.environ["MODEL_PATH"])
opsets = {entry.domain or "ai.onnx": entry.version for entry in model.opset_import}
input_dims = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
output_dims = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]

if model.ir_version > 9:
    raise SystemExit(1)
if opsets.get("ai.onnx") != 14:
    raise SystemExit(1)
if input_dims != [1, 3, 640, 640]:
    raise SystemExit(1)
if output_dims != [1, 84, 8400]:
    raise SystemExit(1)
PYTHON
}

validate_osnet_export() {
    local model_path="$1/$OSNET_ONNX_PATH"
    [ -f "$model_path" ] || return 1
    repo_validator_ready || return 1
    MODEL_PATH="$model_path" "$VENV_DIR/bin/python" - <<'PYTHON' >/dev/null
import os

import onnx

model = onnx.load(os.environ["MODEL_PATH"])
opsets = {entry.domain or "ai.onnx": entry.version for entry in model.opset_import}
input_dims = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
output_dims = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]

if model.ir_version > 9:
    raise SystemExit(1)
if opsets.get("ai.onnx") != 12:
    raise SystemExit(1)
if input_dims[1:] != [3, 256, 128]:
    raise SystemExit(1)
if len(output_dims) != 2 or output_dims[1] != 512:
    raise SystemExit(1)
PYTHON
}

report_yolo_export() {
    local base_dir="$1"
    if validate_yolo_export "$base_dir"; then
        printf '  CACHE OK %s\n' "$YOLO12N_ONNX_PATH"
    elif [ -f "$base_dir/$YOLO12N_ONNX_PATH" ]; then
        printf '  CACHE CORRUPT %s\n' "$YOLO12N_ONNX_PATH"
    else
        printf '  CACHE MISSING %s\n' "$YOLO12N_ONNX_PATH"
    fi
}

report_osnet_export() {
    local base_dir="$1"
    if validate_osnet_export "$base_dir"; then
        printf '  CACHE OK %s\n' "$OSNET_ONNX_PATH"
    elif [ -f "$base_dir/$OSNET_ONNX_PATH" ]; then
        printf '  CACHE CORRUPT %s\n' "$OSNET_ONNX_PATH"
    else
        printf '  CACHE MISSING %s\n' "$OSNET_ONNX_PATH"
    fi
}

validate_cache_asset() {
    local asset_id="$1"
    local base_dir="$2"
    case "$asset_id" in
        yolo12n-onnx)
            validate_yolo_export "$base_dir"
            ;;
        osnet-x0_25-onnx)
            validate_osnet_export "$base_dir"
            ;;
        *)
            validate_specs "$base_dir" "$(cache_asset_specs "$asset_id")"
            ;;
    esac
}

report_cache_asset() {
    local asset_id="$1"
    local base_dir="$2"
    case "$asset_id" in
        yolo12n-onnx)
            report_yolo_export "$base_dir"
            ;;
        osnet-x0_25-onnx)
            report_osnet_export "$base_dir"
            ;;
        *)
            report_specs "CACHE" "$base_dir" "$(cache_asset_specs "$asset_id")"
            ;;
    esac
}

ensure_python_venv() {
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
    "$VENV_DIR/bin/python" -m ensurepip --upgrade >/dev/null
}

ensure_yolo_asset() {
    local target_root="$1"
    if validate_yolo_export "$target_root"; then
        log "  OK YOLO ONNX"
        return
    fi

    local temp_root="$target_root/.yolo-build.$$"
    local temp_pt="$temp_root/yolo12n.pt"
    mkdir -p "$temp_root"
    register_cleanup_path "$temp_root"
    ensure_python_venv
    "$VENV_DIR/bin/python" -m pip install --quiet ultralytics==8.4.0 onnx==1.22.0 onnxslim==0.1.94 onnxruntime==1.27.0
    download_file "$YOLO12N_PT_URL" "$temp_pt"
    [ "$(sha256_file "$temp_pt")" = "$YOLO12N_PT_SHA256" ] || fail "YOLO source checksum mismatch"
    (
        cd "$temp_root"
        MODELS_DIR="$temp_root" \
        ULTRALYTICS_HOME="$temp_root/.ultralytics" \
        XDG_CACHE_HOME="$temp_root/.xdg-cache" \
        "$VENV_DIR/bin/python" "$SCRIPT_DIR/export_yolo_to_onnx.py"
    )
    validate_yolo_export "$temp_root" || fail "YOLO export produced unexpected artifact"
    ensure_parent_dir "$target_root/$YOLO12N_ONNX_PATH"
    mv "$temp_root/$YOLO12N_ONNX_PATH" "$target_root/$YOLO12N_ONNX_PATH"
    rm -rf "$temp_root"
    unregister_cleanup_path "$temp_root"
    log "  OK YOLO ONNX"
}

ensure_osnet_asset() {
    local target_root="$1"
    local weights_path
    if validate_osnet_export "$target_root"; then
        log "  OK OSNet ONNX"
        return
    fi

    local temp_root="$target_root/.osnet-build.$$"
    mkdir -p "$temp_root"
    register_cleanup_path "$temp_root"
    ensure_python_venv
    "$VENV_DIR/bin/python" -m pip install --quiet torch==2.12.1 torchvision==0.27.1 torchreid==0.2.5 onnx==1.22.0 gdown==6.1.0 tensorboard==2.21.0 onnxscript==0.7.1
    weights_path="$temp_root/weights/osnet_x0_25_imagenet.pth"
    mkdir -p "$(dirname "$weights_path")"
    "$VENV_DIR/bin/gdown" --output "$weights_path" "$OSNET_SOURCE_URL"
    [ "$(sha256_file "$weights_path")" = "$OSNET_WEIGHTS_SHA256" ] || fail "OSNet pretrained weight checksum mismatch"
    MODELS_DIR="$temp_root" \
    OSNET_WEIGHTS_PATH="$weights_path" \
    "$VENV_DIR/bin/python" - <<'PYTHON'
import os
from pathlib import Path

import onnx
import torch
import torchreid
from torchreid.reid.utils import load_pretrained_weights

output_dir = Path(os.environ["MODELS_DIR"]) / "reid"
output_dir.mkdir(parents=True, exist_ok=True)
onnx_path = output_dir / "osnet_x0_25.onnx"
weights_path = Path(os.environ["OSNET_WEIGHTS_PATH"])

model = torchreid.models.build_model(
    name="osnet_x0_25",
    num_classes=1000,
    pretrained=False,
    loss="softmax",
)
if not weights_path.is_file():
    raise SystemExit("missing OSNet pretrained weights")
load_pretrained_weights(model, str(weights_path))
model.eval()

dummy_input = torch.randn(1, 3, 256, 128)
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )

onnx.checker.check_model(onnx.load(str(onnx_path)))
PYTHON
    validate_osnet_export "$temp_root" || fail "OSNet export produced unexpected artifact"
    ensure_parent_dir "$target_root/$OSNET_ONNX_PATH"
    mv "$temp_root/$OSNET_ONNX_PATH" "$target_root/$OSNET_ONNX_PATH"
    rm -rf "$temp_root"
    unregister_cleanup_path "$temp_root"
    log "  OK OSNet ONNX"
}

install_cache_archive_asset() {
    local asset_id="$1"
    local target_root="$2"
    local url archive_name archive_sha target_dir specs archive_path extract_root stage_root extracted_dir

    case "$asset_id" in
        sherpa-asr-en)
            url="$EN_ASR_BUNDLE_URL"
            archive_name="${EN_ASR_BUNDLE}.tar.bz2"
            archive_sha="$EN_ASR_BUNDLE_SHA256"
            target_dir="$target_root/sherpa-onnx/asr/${EN_ASR_BUNDLE}"
            ;;
        sherpa-asr-vi)
            url="$VI_ASR_BUNDLE_URL"
            archive_name="${VI_ASR_BUNDLE}.tar.bz2"
            archive_sha="$VI_ASR_BUNDLE_SHA256"
            target_dir="$target_root/sherpa-onnx/asr/${VI_ASR_BUNDLE}"
            ;;
        supertonic-tts)
            url="$SUPERTONIC_TTS_URL"
            archive_name="$SUPERTONIC_TTS_ARCHIVE"
            archive_sha="$SUPERTONIC_TTS_ARCHIVE_SHA256"
            target_dir="$target_root/${SUPERTONIC_TTS_PATH}"
            ;;
        kws-hey-kiwi)
            url="$KWS_URL"
            archive_name="$KWS_ARCHIVE"
            archive_sha="$KWS_ARCHIVE_SHA256"
            target_dir="$target_root/${KWS_PATH}"
            ;;
        *)
            fail "unsupported archive asset '$asset_id'"
            ;;
    esac

    specs="$(cache_asset_specs "$asset_id")"
    if validate_specs "$target_root" "$specs"; then
        log "  OK $asset_id"
        return
    fi

    extract_root="$target_root/.extract-${asset_id}.$$"
    archive_path="$target_root/.${archive_name}"
    stage_root="$extract_root/cache-root"
    mkdir -p "$extract_root"
    register_cleanup_path "$extract_root"
    register_cleanup_path "$archive_path"
    download_file "$url" "$archive_path"
    if [ -n "${archive_sha:-}" ] && [ "$(sha256_file "$archive_path")" != "$archive_sha" ]; then
        fail "$asset_id archive checksum mismatch"
    fi
    safe_extract_tarball "$archive_path" "$extract_root" --bzip2
    extracted_dir="${archive_name%.tar.bz2}"
    mkdir -p "$(dirname "$stage_root/${target_dir#"$target_root"/}")"
    mv "$extract_root/$extracted_dir" "$stage_root/${target_dir#"$target_root"/}"
    validate_specs "$stage_root" "$specs" || fail "$asset_id extracted files failed validation"
    ensure_parent_dir "$target_dir"
    rm -rf "$target_dir"
    mv "$stage_root/${target_dir#"$target_root"/}" "$target_dir"
    rm -f "$archive_path"
    rm -rf "$extract_root"
    unregister_cleanup_path "$archive_path"
    unregister_cleanup_path "$extract_root"
    log "  OK $asset_id"
}

ensure_silero_asset() {
    local target_root="$1"
    local specs
    specs="$(cache_asset_specs silero-vad)"
    if validate_specs "$target_root" "$specs"; then
        log "  OK silero-vad"
        return
    fi

    local temp_root="$target_root/.silero.$$"
    mkdir -p "$temp_root"
    register_cleanup_path "$temp_root"
    download_file "$SILERO_VAD_URL" "$temp_root/$SILERO_VAD_PATH"
    validate_specs "$temp_root" "$specs" || fail "silero-vad failed checksum validation"
    ensure_parent_dir "$target_root/$SILERO_VAD_PATH"
    mv "$temp_root/$SILERO_VAD_PATH" "$target_root/$SILERO_VAD_PATH"
    rm -rf "$temp_root"
    unregister_cleanup_path "$temp_root"
    log "  OK silero-vad"
}

ensure_cache_assets() {
    local target_root="$1"
    mkdir -p "$target_root"
    ensure_validator_tools
    ensure_silero_asset "$target_root"
    install_cache_archive_asset sherpa-asr-en "$target_root"
    install_cache_archive_asset sherpa-asr-vi "$target_root"
    install_cache_archive_asset supertonic-tts "$target_root"
    install_cache_archive_asset kws-hey-kiwi "$target_root"
    ensure_yolo_asset "$target_root"
    ensure_osnet_asset "$target_root"
}

ensure_runtime_asset() {
    local specs archive_path extract_root
    specs="$(runtime_asset_specs)"
    if validate_specs "$RUNTIME_DIR" "$specs"; then
        log "  OK onnxruntime-linux-x64-${ORT_VERSION}"
        return
    fi

    mkdir -p "$RUNTIME_DIR"
    archive_path="$RUNTIME_DIR/${ORT_ARCHIVE}"
    extract_root="$RUNTIME_DIR/.extract-ort.$$"
    register_cleanup_path "$extract_root"
    register_cleanup_path "$archive_path"
    download_file "$ORT_URL" "$archive_path"
    [ "$(sha256_file "$archive_path")" = "$ORT_ARCHIVE_SHA256" ] || fail "ONNX Runtime archive checksum mismatch"
    safe_extract_tarball "$archive_path" "$extract_root" --gzip
    validate_specs "$extract_root" "$specs" || fail "ONNX Runtime extraction failed validation"
    rm -rf "$RUNTIME_DIR/$ORT_DIRNAME"
    mv "$extract_root/$ORT_DIRNAME" "$RUNTIME_DIR/$ORT_DIRNAME"
    rm -f "$archive_path"
    rm -rf "$extract_root"
    unregister_cleanup_path "$archive_path"
    unregister_cleanup_path "$extract_root"
    log "  OK onnxruntime-linux-x64-${ORT_VERSION}"
}

check_assets() {
    local status=0
    local asset_id specs

    repo_validator_ready || fail "missing repo-local ONNX validator tools; run 'make models'"

    log "Cache root: $CACHE_DIR"
    while IFS= read -r asset_id; do
        [ -n "$asset_id" ] || continue
        report_cache_asset "$asset_id" "$CACHE_DIR"
        validate_cache_asset "$asset_id" "$CACHE_DIR" || status=1
    done < <(cache_asset_ids)

    log "Runtime root: $RUNTIME_DIR"
    specs="$(runtime_asset_specs)"
    report_specs "RUNTIME" "$RUNTIME_DIR" "$specs"
    validate_specs "$RUNTIME_DIR" "$specs" || status=1

    if [ "$status" -eq 0 ]; then
        printf 'ROVER_ORT_DYLIB_PATH=%s/%s\n' "$RUNTIME_DIR" "$ORT_LIB_PATH"
    fi
    return "$status"
}

ensure_reset_space() {
    local current_kib free_kib required_kib
    current_kib=0
    [ -d "$CACHE_DIR" ] && current_kib="$(du -sk "$CACHE_DIR" | awk '{print $1}')"
    free_kib="$(df -Pk "$WORK_ROOT" | awk 'NR==2 {print $4}')"
    required_kib=$(( current_kib > 0 ? (current_kib * 2) + 262144 : 786432 ))
    log "Reset disk estimate: need ~${required_kib} KiB free, have ${free_kib} KiB"
    [ "$free_kib" -ge "$required_kib" ] || fail "not enough free disk space for atomic reset"
}

reset_cache() {
    local parent_dir stage_dir backup_dir
    ensure_reset_space
    parent_dir="$(dirname "$CACHE_DIR")"
    mkdir -p "$parent_dir"
    stage_dir="$parent_dir/.cache-reset-$(date +%Y%m%d%H%M%S)-$$"
    register_cleanup_path "$stage_dir"
    ensure_cache_assets "$stage_dir"
    if ! validate_cache_asset silero-vad "$stage_dir"; then
        fail "staging cache failed validation"
    fi
    while IFS= read -r asset_id; do
        validate_cache_asset "$asset_id" "$stage_dir" || fail "staging validation failed for $asset_id"
    done < <(cache_asset_ids)

    backup_dir="$parent_dir/.cache-backup-$(date +%Y%m%d%H%M%S)-$$"
    if [ -d "$CACHE_DIR" ]; then
        mv "$CACHE_DIR" "$backup_dir"
        SWAP_BACKUP_DIR="$backup_dir"
    fi
    mv "$stage_dir" "$CACHE_DIR"
    unregister_cleanup_path "$stage_dir"
    if [ -n "$SWAP_BACKUP_DIR" ] && [ -d "$SWAP_BACKUP_DIR" ]; then
        rm -rf "$SWAP_BACKUP_DIR"
        SWAP_BACKUP_DIR=""
    fi
    ensure_runtime_asset
}

usage() {
    cat <<EOF
Usage: $0 <ensure|check|reset|print-ort-path>
EOF
}

command="${1:-ensure}"
case "$command" in
    ensure)
        warn_stale_cache_dirs
        ensure_cache_assets "$CACHE_DIR"
        ensure_runtime_asset
        printf 'ROVER_ORT_DYLIB_PATH=%s/%s\n' "$RUNTIME_DIR" "$ORT_LIB_PATH"
        ;;
    check)
        warn_stale_cache_dirs
        check_assets
        ;;
    reset)
        warn_stale_cache_dirs
        reset_cache
        printf 'ROVER_ORT_DYLIB_PATH=%s/%s\n' "$RUNTIME_DIR" "$ORT_LIB_PATH"
        ;;
    print-ort-path)
        printf '%s/%s\n' "$RUNTIME_DIR" "$ORT_LIB_PATH"
        ;;
    *)
        usage
        exit 2
        ;;
esac
