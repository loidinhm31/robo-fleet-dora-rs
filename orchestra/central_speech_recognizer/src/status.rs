use crate::decoder::SharedNode;
use dora_node_api::{arrow::array::BinaryArray, dora_core::config::DataId};
use eyre::{eyre, Result};
use robo_rover_lib::{SttProfile, SttState, SttStatus};

pub fn build_status(state: SttState, profile: SttProfile, error: Option<String>) -> SttStatus {
    SttStatus {
        state,
        profile,
        language: profile.language_code().into(),
        timestamp: current_timestamp_ms(),
        error,
    }
}

pub fn emit_status(node: &SharedNode, status: &SttStatus) -> Result<()> {
    let json = serde_json::to_vec(status)?;
    let array = BinaryArray::from_vec(vec![json.as_slice()]);
    node.lock()
        .map_err(|_| eyre!("Dora node lock poisoned"))?
        .send_output(
            DataId::from("stt_status".to_owned()),
            Default::default(),
            array,
        )
}

pub fn startup_profile() -> SttProfile {
    match std::env::var("STT_PROFILE").as_deref() {
        Ok("vi-vad-offline") => SttProfile::ViVadOffline,
        _ => SttProfile::EnVadOffline,
    }
}

pub fn sanitize_startup_error(error: &eyre::Report) -> String {
    let message = error.to_string();
    if message.starts_with("invalid ")
        || message.starts_with("required model file missing:")
        || message.starts_with("failed to initialize ")
    {
        message
    } else {
        "speech recognizer initialization failed".into()
    }
}

fn current_timestamp_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_preserves_profile_and_sanitizes_unknown_errors() {
        let status = build_status(
            SttState::Error,
            SttProfile::ViVadOffline,
            Some("bad".into()),
        );
        assert_eq!(status.language, "vi");
        assert_eq!(
            sanitize_startup_error(&eyre!("/private/model/path failed")),
            "speech recognizer initialization failed"
        );
        assert_eq!(
            sanitize_startup_error(&eyre!("invalid STT_PROFILE")),
            "invalid STT_PROFILE"
        );
    }

    #[test]
    fn loading_status_is_request_safe() {
        let status = build_status(SttState::Loading, SttProfile::EnVadOffline, None);
        assert_eq!(status.state, SttState::Loading);
        assert_eq!(status.language, "en");
        assert_eq!(status.error, None);
    }
}
