#!/usr/bin/env bash

MODEL_MANIFEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_MANIFEST_ROOT="$(cd "$MODEL_MANIFEST_DIR/../.." && pwd)"

MODEL_CACHE_DIR_DEFAULT="$MODEL_MANIFEST_ROOT/models/.cache"
MODEL_RUNTIME_DIR_DEFAULT="$MODEL_MANIFEST_ROOT/models/.runtime"

SILERO_VAD_PATH="sherpa-onnx/asr/silero/silero_vad.onnx"
SILERO_VAD_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
SILERO_VAD_SHA256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"

EN_ASR_BUNDLE="icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04"
EN_ASR_BUNDLE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${EN_ASR_BUNDLE}.tar.bz2"
EN_ASR_BUNDLE_SHA256="4a51079b4d09387f1502e37074f7c322d846adbefca83a7efae123286a01010e"

VI_ASR_BUNDLE="sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
VI_ASR_BUNDLE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${VI_ASR_BUNDLE}.tar.bz2"
VI_ASR_BUNDLE_SHA256="da8b637947091829d7ee9eda23da2a4ec7caa399233a3f4e34eb719fb2ea6b9b"

SUPERTONIC_TTS_BUNDLE="sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
SUPERTONIC_TTS_ARCHIVE="${SUPERTONIC_TTS_BUNDLE}.tar.bz2"
SUPERTONIC_TTS_PATH="sherpa-onnx/tts/${SUPERTONIC_TTS_BUNDLE}"
SUPERTONIC_TTS_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/${SUPERTONIC_TTS_ARCHIVE}"
SUPERTONIC_TTS_ARCHIVE_SHA256="82fa96f91c4ef8abaae3a14a3f4153facf88bed821d1f7331cec2700f432c427"

YOLO12N_PT_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12n.pt"
YOLO12N_PT_SHA256="419ff3dca37d69bacc93a50fa0c186a1c6f9fe62fae0f108b0872829689e9ca6"
YOLO12N_ONNX_PATH="yolo/yolo12n.onnx"

OSNET_SOURCE_URL="https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs"
OSNET_WEIGHTS_SHA256="f54941a66bad4ddd07f2907f498c810ce639ce7a1abeaf2a151f8da118d84693"
OSNET_ONNX_PATH="reid/osnet_x0_25.onnx"

ORT_VERSION="1.16.3"
ORT_ARCHIVE="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}"
ORT_ARCHIVE_SHA256="b072f989d6315ac0e22dcb4771b083c5156d974a3496ac3504c77f4062eb248e"
ORT_DIRNAME="onnxruntime-linux-x64-${ORT_VERSION}"
ORT_LIB_PATH="${ORT_DIRNAME}/lib/libonnxruntime.so"

model_cache_dir() {
    printf '%s\n' "$MODEL_CACHE_DIR_DEFAULT"
}

model_runtime_dir() {
    printf '%s\n' "$MODEL_RUNTIME_DIR_DEFAULT"
}

cache_asset_ids() {
    cat <<'EOF'
silero-vad
sherpa-asr-en
sherpa-asr-vi
supertonic-tts
yolo12n-onnx
osnet-x0_25-onnx
EOF
}

required_stt_files() {
    local profile="$1"
    case "$profile" in
        en-vad-offline)
            cat <<EOF
silero/silero_vad.onnx
${EN_ASR_BUNDLE}/exp/encoder-epoch-30-avg-4.int8.onnx
${EN_ASR_BUNDLE}/exp/decoder-epoch-30-avg-4.onnx
${EN_ASR_BUNDLE}/exp/joiner-epoch-30-avg-4.int8.onnx
${EN_ASR_BUNDLE}/data/lang_bpe_500/tokens.txt
EOF
            ;;
        vi-vad-offline)
            cat <<EOF
silero/silero_vad.onnx
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

cache_asset_specs() {
    local asset_id="$1"
    case "$asset_id" in
        silero-vad)
            printf '%s  %s\n' "$SILERO_VAD_SHA256" "$SILERO_VAD_PATH"
            ;;
        sherpa-asr-en)
            cat <<EOF
f5f9e62459e8055d61e1becbc9f181f94cd39310423e647d46386ac4715aeaf0  sherpa-onnx/asr/${EN_ASR_BUNDLE}/exp/encoder-epoch-30-avg-4.int8.onnx
c4f2931113a93facb97da03ea58ef9fecf622b4d653b41b91646544d961f6f2d  sherpa-onnx/asr/${EN_ASR_BUNDLE}/exp/decoder-epoch-30-avg-4.onnx
b9a5c90b7ceaa7a3f87d64e82782f6946b266c37f433083414c1bc60e23d06ca  sherpa-onnx/asr/${EN_ASR_BUNDLE}/exp/joiner-epoch-30-avg-4.int8.onnx
49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb  sherpa-onnx/asr/${EN_ASR_BUNDLE}/data/lang_bpe_500/tokens.txt
EOF
            ;;
        sherpa-asr-vi)
            cat <<EOF
8ef5286dd427eb108055c2ddc1982aa31e544706072d5ea228729292dacade68  sherpa-onnx/asr/${VI_ASR_BUNDLE}/encoder.int8.onnx
cf2aa385b82c9d5d40cd29c3188af52d0249b3b78f0d4b7eb84ad502d50c7e7f  sherpa-onnx/asr/${VI_ASR_BUNDLE}/decoder.onnx
7311d2e17b810ecea515d79c71cc4668af8759256a06fa01d27047772320c821  sherpa-onnx/asr/${VI_ASR_BUNDLE}/joiner.int8.onnx
ca8171f8bbd516c050b627582f2125c8f5f1f6ed967ab41b0fa9aae2cf61b492  sherpa-onnx/asr/${VI_ASR_BUNDLE}/tokens.txt
EOF
            ;;
        supertonic-tts)
            cat <<EOF
c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db  ${SUPERTONIC_TTS_PATH}/duration_predictor.int8.onnx
c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff  ${SUPERTONIC_TTS_PATH}/text_encoder.int8.onnx
20cd86fa5c6effedfda0e7cffe5b0569ca401c440a0c3a1d72bf39286c0db3fd  ${SUPERTONIC_TTS_PATH}/vector_estimator.int8.onnx
e923d60f53f95eb1ce235f1dc33ec56d9c057823c96fa6f8acf98f32b0da6152  ${SUPERTONIC_TTS_PATH}/vocoder.int8.onnx
42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09  ${SUPERTONIC_TTS_PATH}/tts.json
8402ca48e5189a8950138580b0fff64db6f072f24ac07cd54ba8b2fbb9883b30  ${SUPERTONIC_TTS_PATH}/unicode_indexer.bin
67d5209b0ee8ce6c74105ffbe12fe6a7628aea3b4ba2fcb308a4a67938a93ce8  ${SUPERTONIC_TTS_PATH}/voice.bin
EOF
            ;;
        *)
            return 1
            ;;
    esac
}

runtime_asset_specs() {
    cat <<EOF
ce3752ba35018ee6d8127ff4cba955b68b9c8b8b0fed8798a8f2e5c4c5a35fa5  ${ORT_DIRNAME}/lib/libonnxruntime.so
2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c  ${ORT_DIRNAME}/LICENSE
1787e003e8344c29dc21331a8c4fcde6484c1a5ef43e6e7cffd8ce1107b8b8a5  ${ORT_DIRNAME}/VERSION_NUMBER
EOF
}
