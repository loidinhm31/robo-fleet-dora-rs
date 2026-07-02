use robo_rover_lib::SttProfile;
use std::path::{Path, PathBuf};

pub const DEFAULT_PROFILE: &str = "en-vad-offline";
pub const DEFAULT_MODEL_ROOT: &str = "models/.cache/sherpa-onnx/asr";

const EN_BUNDLE: &str = "icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04";
const VI_BUNDLE: &str = "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPaths {
    pub profile: SttProfile,
    pub language: &'static str,
    pub bundle_name: &'static str,
    pub vad: PathBuf,
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub joiner: PathBuf,
    pub tokens: PathBuf,
}

impl ModelPaths {
    pub fn required_files(&self) -> [(&'static str, &Path); 5] {
        [
            ("Silero VAD", &self.vad),
            ("encoder", &self.encoder),
            ("decoder", &self.decoder),
            ("joiner", &self.joiner),
            ("tokens", &self.tokens),
        ]
    }
}

pub fn resolve(profile: SttProfile, root: &Path) -> ModelPaths {
    let vad = root.join("silero/silero_vad.onnx");
    match profile {
        SttProfile::EnVadOffline => {
            let bundle = root.join(EN_BUNDLE);
            ModelPaths {
                profile,
                language: "en",
                bundle_name: EN_BUNDLE,
                vad,
                encoder: bundle.join("exp/encoder-epoch-30-avg-4.int8.onnx"),
                decoder: bundle.join("exp/decoder-epoch-30-avg-4.onnx"),
                joiner: bundle.join("exp/joiner-epoch-30-avg-4.int8.onnx"),
                tokens: bundle.join("data/lang_bpe_500/tokens.txt"),
            }
        }
        SttProfile::ViVadOffline => {
            let bundle = root.join(VI_BUNDLE);
            ModelPaths {
                profile,
                language: "vi",
                bundle_name: VI_BUNDLE,
                vad,
                encoder: bundle.join("encoder.int8.onnx"),
                decoder: bundle.join("decoder.onnx"),
                joiner: bundle.join("joiner.int8.onnx"),
                tokens: bundle.join("tokens.txt"),
            }
        }
    }
}
