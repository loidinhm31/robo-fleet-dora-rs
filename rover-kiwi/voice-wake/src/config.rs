use eyre::{eyre, Result};
use std::path::PathBuf;

pub const DEFAULT_MODEL_RELATIVE_DIR: &str =
    "models/.cache/sherpa-onnx/kws/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01";
// BPE keyword lists contain model tokens. The annotation preserves a stable,
// space-free result label from the native KWS decoder.
pub const KEYWORD_TOKENS: &str = "▁HE Y ▁K I W I :2.5 #0.35 @Hey_Kiwi\n";
pub const SAMPLE_RATE: i32 = 16_000;

#[derive(Debug, Clone)]
pub struct KwsConfig {
    pub model_dir: PathBuf,
}

impl KwsConfig {
    pub fn from_env() -> Result<Self> {
        let model_dir = std::env::var("KWS_MODEL_DIR")
            .map(PathBuf::from)
            // Cargo executes package tests from the package directory, while Dora
            // dataflows run from the repository root. Anchor the built-in default
            // at this crate so both contexts resolve the same verified cache.
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../")
                    .join(DEFAULT_MODEL_RELATIVE_DIR)
            });
        for file in required_files() {
            let path = model_dir.join(file);
            if !path.is_file() {
                return Err(eyre!("missing required KWS model file: {}", path.display()));
            }
        }
        Ok(Self { model_dir })
    }

    pub fn path(&self, filename: &str) -> String {
        self.model_dir.join(filename).to_string_lossy().into_owned()
    }
}

pub fn required_files() -> &'static [&'static str] {
    &[
        "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        "tokens.txt",
        "bpe.model",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phrase_contract_is_model_tokenized_and_annotated() {
        assert_eq!(KEYWORD_TOKENS, "▁HE Y ▁K I W I :2.5 #0.35 @Hey_Kiwi\n");
    }
}
