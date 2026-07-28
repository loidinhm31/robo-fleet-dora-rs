use eyre::{eyre, Result};
use sherpa_onnx::{KeywordSpotter, KeywordSpotterConfig, OnlineStream};

use crate::config::{KwsConfig, KEYWORD_TOKENS, SAMPLE_RATE};

pub struct KwsEngine {
    spotter: KeywordSpotter,
    stream: OnlineStream,
}

impl KwsEngine {
    pub fn create(config: &KwsConfig) -> Result<Self> {
        let mut settings = KeywordSpotterConfig::default();
        settings.model_config.transducer.encoder =
            Some(config.path("encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"));
        settings.model_config.transducer.decoder =
            Some(config.path("decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"));
        settings.model_config.transducer.joiner =
            Some(config.path("joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"));
        settings.model_config.tokens = Some(config.path("tokens.txt"));
        settings.model_config.bpe_vocab = Some(config.path("bpe.model"));
        settings.model_config.modeling_unit = Some("bpe".into());
        settings.model_config.num_threads = 1;
        settings.keywords_buf = Some(KEYWORD_TOKENS.into());
        let spotter = KeywordSpotter::create(&settings)
            .ok_or_else(|| eyre!("failed to create Sherpa keyword spotter"))?;
        let stream = spotter.create_stream();
        Ok(Self { spotter, stream })
    }

    pub fn observe(&mut self, samples: &[f32]) -> bool {
        self.stream.accept_waveform(SAMPLE_RATE, samples);
        while self.spotter.is_ready(&self.stream) {
            self.spotter.decode(&self.stream);
        }
        let Some(result) = self.spotter.get_result(&self.stream) else {
            return false;
        };
        let detected = canonical_keyword(&result.keyword).eq_ignore_ascii_case("Hey Kiwi");
        // A decoded result remains attached to an online stream until it is
        // reset. Clear both accepted and rejected phrases so a false match
        // cannot starve the next 16 kHz capture frame.
        self.spotter.reset(&self.stream);
        detected
    }

    pub fn reset(&self) {
        self.spotter.reset(&self.stream);
    }
}

fn canonical_keyword(keyword: &str) -> String {
    keyword.replace('_', " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_engine_rejects_silence() {
        let config = KwsConfig::from_env().expect("pinned KWS model installed");
        let mut engine = KwsEngine::create(&config).expect("pinned KWS model opens");
        assert!(!engine.observe(&vec![0.0; SAMPLE_RATE as usize]));
    }

    #[test]
    fn only_the_annotated_hey_kiwi_result_is_accepted() {
        assert_eq!(canonical_keyword("Hey_Kiwi"), "Hey Kiwi");
        assert_ne!(canonical_keyword("Hey_Siri"), "Hey Kiwi");
    }
}
