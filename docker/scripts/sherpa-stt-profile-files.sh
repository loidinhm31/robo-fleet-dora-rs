#!/bin/bash

SILERO_VAD_PATH="silero/silero_vad.onnx"
EN_ASR_BUNDLE="icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04"
VI_ASR_BUNDLE="sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"

required_stt_files() {
    local profile="$1"
    case "$profile" in
        en-vad-offline)
            cat <<EOF
${SILERO_VAD_PATH}
${EN_ASR_BUNDLE}/exp/encoder-epoch-30-avg-4.int8.onnx
${EN_ASR_BUNDLE}/exp/decoder-epoch-30-avg-4.onnx
${EN_ASR_BUNDLE}/exp/joiner-epoch-30-avg-4.int8.onnx
${EN_ASR_BUNDLE}/data/lang_bpe_500/tokens.txt
EOF
            ;;
        vi-vad-offline)
            cat <<EOF
${SILERO_VAD_PATH}
${VI_ASR_BUNDLE}/encoder.int8.onnx
${VI_ASR_BUNDLE}/decoder.onnx
${VI_ASR_BUNDLE}/joiner.int8.onnx
${VI_ASR_BUNDLE}/tokens.txt
EOF
            ;;
        *)
            return 1
            ;;
    esac
}
