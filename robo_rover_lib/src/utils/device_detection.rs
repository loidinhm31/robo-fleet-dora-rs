use cpal::{BufferSize, SampleFormat, SampleRate, StreamConfig, SupportedStreamConfigRange};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputCapturePlan {
    pub capture_channels: u16,
    pub output_channels: u16,
    pub capture_sample_rate: u32,
    pub output_sample_rate: u32,
    pub sample_format: SampleFormat,
    pub downmix_to_mono: bool,
    pub resample_to_output_rate: bool,
    pub selection_reason: &'static str,
}

impl InputCapturePlan {
    pub fn stream_config(&self) -> StreamConfig {
        StreamConfig {
            channels: self.capture_channels,
            sample_rate: SampleRate(self.capture_sample_rate),
            buffer_size: BufferSize::Default,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SupportedInputConfigDescriptor {
    pub channels: u16,
    pub min_sample_rate: u32,
    pub max_sample_rate: u32,
    pub sample_format: SampleFormat,
}

impl From<SupportedStreamConfigRange> for SupportedInputConfigDescriptor {
    fn from(value: SupportedStreamConfigRange) -> Self {
        Self {
            channels: value.channels(),
            min_sample_rate: value.min_sample_rate().0,
            max_sample_rate: value.max_sample_rate().0,
            sample_format: value.sample_format(),
        }
    }
}

pub fn matches_device_override(name: &str, target: &str) -> bool {
    name.to_ascii_lowercase()
        .contains(&target.trim().to_ascii_lowercase())
}

pub fn device_preference_score(name: &str) -> u8 {
    let normalized = name.to_ascii_lowercase();
    let has_usb = normalized.contains("usb");
    let has_camera = normalized.contains("camera") || normalized.contains("pc-lm1e");
    match (has_usb, has_camera) {
        (true, true) => 3,
        (true, false) => 2,
        _ => 0,
    }
}

pub fn describe_device_preference(name: &str, default_name: Option<&str>) -> &'static str {
    let normalized = name.to_ascii_lowercase();
    let is_default = default_name.is_some_and(|default| default == name);
    let has_usb = normalized.contains("usb");
    let has_camera = normalized.contains("camera") || normalized.contains("pc-lm1e");
    match (has_usb, has_camera, is_default) {
        (true, true, _) => "preferred-usb-camera",
        (true, false, _) => "preferred-usb-audio",
        (false, false, true) => "fallback-default-input",
        _ => "fallback-unmatched-input",
    }
}

pub fn select_preferred_device_name(
    device_names: &[String],
    default_name: Option<&str>,
) -> Option<String> {
    device_names
        .iter()
        .max_by_key(|name| {
            (
                device_preference_score(name),
                u8::from(default_name.is_some_and(|default| default == name.as_str())),
                std::cmp::Reverse(name.len()),
            )
        })
        .and_then(|name| {
            let score = device_preference_score(name);
            (score > 0)
                .then(|| name.clone())
                .or_else(|| default_name.map(str::to_string))
                .or_else(|| device_names.first().cloned())
        })
}

pub fn select_input_capture_plan(
    configs: &[SupportedInputConfigDescriptor],
    requested_sample_rate: u32,
    requested_channels: u16,
) -> Option<InputCapturePlan> {
    configs
        .iter()
        .filter_map(|config| {
            sample_format_rank(config.sample_format).map(|format_rank| {
                let exact_channels = config.channels == requested_channels;
                let mono_downmix = requested_channels == 1 && config.channels > 1;
                let supports_rate = config.min_sample_rate <= requested_sample_rate
                    && requested_sample_rate <= config.max_sample_rate;
                let channel_rank = if exact_channels {
                    2
                } else if mono_downmix {
                    1
                } else {
                    0
                };
                let capture_sample_rate = if supports_rate {
                    requested_sample_rate
                } else {
                    preferred_native_sample_rate(config, requested_sample_rate)
                };
                let rate_rank = u8::from(supports_rate);
                let selection_reason = match (exact_channels, mono_downmix, supports_rate) {
                    (true, false, true) => "exact-channels-exact-rate",
                    (false, true, true) => "downmix-channels-exact-rate",
                    (true, false, false) => "exact-channels-native-rate-resample",
                    (false, true, false) => "downmix-channels-native-rate-resample",
                    _ => "unsupported",
                };
                (
                    (
                        channel_rank,
                        rate_rank,
                        format_rank,
                        capture_sample_rate,
                        std::cmp::Reverse(config.channels),
                    ),
                    InputCapturePlan {
                        capture_channels: if exact_channels {
                            requested_channels
                        } else {
                            config.channels
                        },
                        output_channels: requested_channels,
                        capture_sample_rate,
                        output_sample_rate: requested_sample_rate,
                        sample_format: config.sample_format,
                        downmix_to_mono: mono_downmix,
                        resample_to_output_rate: !supports_rate,
                        selection_reason,
                    },
                )
            })
        })
        .filter(|((channel_rank, _, _, _, _), _)| *channel_rank > 0)
        .max_by_key(|(rank, _)| *rank)
        .map(|(_, plan)| plan)
}

fn preferred_native_sample_rate(
    config: &SupportedInputConfigDescriptor,
    requested_sample_rate: u32,
) -> u32 {
    if requested_sample_rate < config.min_sample_rate {
        config.min_sample_rate
    } else if requested_sample_rate > config.max_sample_rate {
        config.max_sample_rate
    } else {
        requested_sample_rate
    }
}

fn sample_format_rank(format: SampleFormat) -> Option<u8> {
    match format {
        SampleFormat::F32 => Some(3),
        SampleFormat::I16 => Some(2),
        SampleFormat::U16 => Some(1),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cpal::SupportedBufferSize;

    #[test]
    fn auto_detect_prefers_usb_camera_names() {
        let devices = vec![
            "HD-Audio Generic Analog".to_string(),
            "USB Audio Device".to_string(),
            "PC-LM1E Camera USB Audio".to_string(),
        ];

        assert_eq!(
            select_preferred_device_name(&devices, Some("HD-Audio Generic Analog")),
            Some("PC-LM1E Camera USB Audio".to_string())
        );
    }

    #[test]
    fn auto_detect_falls_back_to_default_when_no_usb_device_exists() {
        let devices = vec![
            "HD-Audio Generic Analog".to_string(),
            "Monitor of Built-in Audio".to_string(),
        ];

        assert_eq!(
            select_preferred_device_name(&devices, Some("HD-Audio Generic Analog")),
            Some("HD-Audio Generic Analog".to_string())
        );
    }

    #[test]
    fn auto_detect_prefers_default_when_all_candidates_have_zero_score() {
        let devices = vec![
            "Monitor of Built-in Audio".to_string(),
            "HD-Audio Generic Analog".to_string(),
        ];

        assert_eq!(
            select_preferred_device_name(&devices, Some("HD-Audio Generic Analog")),
            Some("HD-Audio Generic Analog".to_string())
        );
    }

    #[test]
    fn capture_plan_uses_exact_channel_match_when_available() {
        let configs = vec![
            SupportedStreamConfigRange::new(
                1,
                SampleRate(8_000),
                SampleRate(48_000),
                SupportedBufferSize::Unknown,
                SampleFormat::I16,
            )
            .into(),
            SupportedStreamConfigRange::new(
                2,
                SampleRate(8_000),
                SampleRate(48_000),
                SupportedBufferSize::Unknown,
                SampleFormat::F32,
            )
            .into(),
        ];

        assert_eq!(
            select_input_capture_plan(&configs, 16_000, 1),
            Some(InputCapturePlan {
                capture_channels: 1,
                output_channels: 1,
                capture_sample_rate: 16_000,
                output_sample_rate: 16_000,
                sample_format: SampleFormat::I16,
                downmix_to_mono: false,
                resample_to_output_rate: false,
                selection_reason: "exact-channels-exact-rate",
            })
        );
    }

    #[test]
    fn capture_plan_downmixes_multichannel_input_for_mono_requests() {
        let configs = vec![SupportedStreamConfigRange::new(
            2,
            SampleRate(8_000),
            SampleRate(48_000),
            SupportedBufferSize::Unknown,
            SampleFormat::F32,
        )
        .into()];

        assert_eq!(
            select_input_capture_plan(&configs, 16_000, 1),
            Some(InputCapturePlan {
                capture_channels: 2,
                output_channels: 1,
                capture_sample_rate: 16_000,
                output_sample_rate: 16_000,
                sample_format: SampleFormat::F32,
                downmix_to_mono: true,
                resample_to_output_rate: false,
                selection_reason: "downmix-channels-exact-rate",
            })
        );
    }

    #[test]
    fn capture_plan_falls_back_to_native_rate_when_resample_is_required() {
        let configs = vec![SupportedStreamConfigRange::new(
            2,
            SampleRate(48_000),
            SampleRate(48_000),
            SupportedBufferSize::Unknown,
            SampleFormat::F32,
        )
        .into()];

        assert_eq!(
            select_input_capture_plan(&configs, 16_000, 1),
            Some(InputCapturePlan {
                capture_channels: 2,
                output_channels: 1,
                capture_sample_rate: 48_000,
                output_sample_rate: 16_000,
                sample_format: SampleFormat::F32,
                downmix_to_mono: true,
                resample_to_output_rate: true,
                selection_reason: "downmix-channels-native-rate-resample",
            })
        );
    }
}
