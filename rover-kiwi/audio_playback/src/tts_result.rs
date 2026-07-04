use dora_node_api::arrow::array::{Array, BinaryArray};
use eyre::{eyre, Result};
use robo_rover_lib::TtsCommandResult;

pub fn parse_tts_result(data: &dyn Array) -> Result<TtsCommandResult> {
    let binary = data
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| eyre!("TTS result must be BinaryArray"))?;
    if binary.len() != 1 {
        return Err(eyre!("TTS result must contain one payload"));
    }
    let result = serde_json::from_slice::<TtsCommandResult>(binary.value(0))?;
    result.validate().map_err(eyre::Report::msg)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dora_node_api::arrow::array::Float32Array;
    use robo_rover_lib::TtsResultState;
    use uuid::Uuid;

    #[test]
    fn parses_valid_terminal_result_and_rejects_wrong_payload_type() {
        let result = TtsCommandResult {
            command_id: Uuid::new_v4().to_string(),
            entity_id: "rover-kiwi".into(),
            state: TtsResultState::Completed,
            timestamp: 1,
            reason_code: None,
            detail: None,
        };
        let bytes = serde_json::to_vec(&result).unwrap();
        let binary = BinaryArray::from_vec(vec![bytes.as_slice()]);

        assert_eq!(parse_tts_result(&binary).unwrap(), result);
        assert!(parse_tts_result(&Float32Array::from(vec![0.1])).is_err());
    }
}
