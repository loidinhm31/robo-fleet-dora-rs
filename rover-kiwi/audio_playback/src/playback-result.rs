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
    report_tts_terminal(
        node,
        outputs,
        entity_id,
        command_id,
        if completed {
            TtsResultState::Completed
        } else {
            TtsResultState::Failed
        },
        (!completed).then_some(VoiceReasonCode::PlaybackFailed),
    )
}

pub fn report_tts_interrupted_by_lifecycle(
    node: &mut DoraNode,
    outputs: &PlaybackOutputs,
    entity_id: &str,
    command_id: String,
) -> Result<()> {
    report_tts_terminal(
        node,
        outputs,
        entity_id,
        command_id,
        TtsResultState::Interrupted,
        Some(VoiceReasonCode::InterruptedByLifecycle),
    )
}

fn report_tts_terminal(
    node: &mut DoraNode,
    outputs: &PlaybackOutputs,
    entity_id: &str,
    command_id: String,
    state: TtsResultState,
    reason_code: Option<VoiceReasonCode>,
) -> Result<()> {
    let result = TtsCommandResult {
        command_id,
        entity_id: entity_id.to_owned(),
        state,
        timestamp: current_time_ms(),
        reason_code,
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
