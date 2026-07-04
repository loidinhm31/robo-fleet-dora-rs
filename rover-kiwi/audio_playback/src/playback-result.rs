use dora_node_api::{arrow::array::BinaryArray, DoraNode, MetadataParameters};
use eyre::Result;
use robo_rover_lib::{TtsCommandResult, TtsResultState, VoiceReasonCode};

use crate::state::{current_time_ms, PlaybackOutputs};

pub fn report_tts_result(
    node: &mut DoraNode,
    outputs: &PlaybackOutputs,
    entity_id: &str,
    command_id: String,
    completed: bool,
) -> Result<()> {
    let result = TtsCommandResult {
        command_id,
        entity_id: entity_id.to_owned(),
        state: if completed {
            TtsResultState::Completed
        } else {
            TtsResultState::Failed
        },
        timestamp: current_time_ms(),
        reason_code: (!completed).then_some(VoiceReasonCode::PlaybackFailed),
        detail: None,
    };
    result.validate().map_err(eyre::Report::msg)?;
    let bytes = serde_json::to_vec(&result)?;
    node.send_output(
        outputs.playback_result.clone(),
        MetadataParameters::default(),
        BinaryArray::from_vec(vec![bytes.as_slice()]),
    )?;
    Ok(())
}
