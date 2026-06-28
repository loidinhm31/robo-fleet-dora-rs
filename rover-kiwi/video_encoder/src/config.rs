use std::env;

const DEFAULT_JPEG_QUALITY: u8 = 80;
const DEFAULT_WIDTH: u32 = 640;
const DEFAULT_HEIGHT: u32 = 480;

#[derive(Debug, Clone, Copy)]
pub(crate) struct EncoderConfig {
    pub(crate) jpeg_quality: u8,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            jpeg_quality: DEFAULT_JPEG_QUALITY,
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
        }
    }
}

impl EncoderConfig {
    pub(crate) fn from_env() -> Self {
        Self::from_values(
            env::var("JPEG_QUALITY").ok().as_deref(),
            env::var("IMAGE_WIDTH").ok().as_deref(),
            env::var("IMAGE_HEIGHT").ok().as_deref(),
        )
    }

    fn from_values(quality: Option<&str>, width: Option<&str>, height: Option<&str>) -> Self {
        Self {
            jpeg_quality: quality
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_JPEG_QUALITY)
                .clamp(1, 100),
            width: width
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_WIDTH),
            height: height
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_HEIGHT),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_is_clamped_and_invalid_values_use_default() {
        assert_eq!(
            EncoderConfig::from_values(Some("0"), None, None).jpeg_quality,
            1
        );
        assert_eq!(
            EncoderConfig::from_values(Some("101"), None, None).jpeg_quality,
            100
        );
        assert_eq!(
            EncoderConfig::from_values(Some("invalid"), None, None).jpeg_quality,
            DEFAULT_JPEG_QUALITY
        );
    }

    #[test]
    fn dimensions_keep_existing_defaults_and_overrides() {
        let defaults = EncoderConfig::from_values(None, None, None);
        assert_eq!((defaults.width, defaults.height), (640, 480));

        let configured = EncoderConfig::from_values(None, Some("1280"), Some("720"));
        assert_eq!((configured.width, configured.height), (1280, 720));
    }
}
