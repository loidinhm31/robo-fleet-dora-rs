use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use dora_node_api::{
    arrow::array::BinaryArray, dora_core::config::DataId, DoraNode, MetadataParameters,
};
use eyre::{eyre, Result};
use robo_rover_lib::{PlaybackSource, PlaybackState, PlaybackStateKind, VoiceReasonCode};

use crate::buffers::{SOURCE_IDLE, SOURCE_TTS, SOURCE_WALKIE};

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReportedState {
    kind: PlaybackStateKind,
    source: Option<PlaybackSource>,
    command_id: Option<String>,
}

pub struct PlaybackOutputs {
    playback_state: DataId,
    walkie_state: DataId,
    pub(crate) playback_result: DataId,
}

impl PlaybackOutputs {
    pub fn new() -> Self {
        Self {
            playback_state: DataId::from("playback_state".to_owned()),
            walkie_state: DataId::from("walkie_state".to_owned()),
            playback_result: DataId::from("playback_result".to_owned()),
        }
    }
}

pub struct StateReporter {
    entity_id: String,
    last_playback: Option<ReportedState>,
    walkie_control_active: bool,
}

impl StateReporter {
    pub fn new(entity_id: String) -> Self {
        Self {
            entity_id,
            last_playback: None,
            walkie_control_active: false,
        }
    }

    pub fn entity_id(&self) -> &str {
        &self.entity_id
    }

    pub fn report_consumption(
        &mut self,
        node: &mut DoraNode,
        outputs: &PlaybackOutputs,
        source: u8,
        token: u64,
        command_ids: &BTreeMap<u64, String>,
    ) -> Result<()> {
        let next = match source {
            SOURCE_IDLE => ReportedState {
                kind: PlaybackStateKind::Idle,
                source: None,
                command_id: None,
            },
            SOURCE_TTS => ReportedState {
                kind: PlaybackStateKind::Active,
                source: Some(PlaybackSource::Tts),
                command_id: command_ids.get(&token).cloned(),
            },
            SOURCE_WALKIE => ReportedState {
                kind: PlaybackStateKind::Active,
                source: Some(PlaybackSource::Walkie),
                command_id: None,
            },
            _ => return Err(eyre!("invalid playback callback source")),
        };
        if source == SOURCE_TTS && next.command_id.is_none() {
            return Err(eyre!("missing command ID for consumed TTS samples"));
        }
        if self.last_playback.as_ref() == Some(&next) {
            return Ok(());
        }
        let state = PlaybackState {
            entity_id: self.entity_id.clone(),
            state: next.kind,
            source: next.source,
            command_id: next.command_id.clone(),
            timestamp: current_time_ms(),
            reason_code: None,
            detail: None,
        };
        send_state(node, outputs.playback_state.clone(), &state)?;
        self.last_playback = Some(next);
        Ok(())
    }

    pub fn report_walkie_active(
        &mut self,
        node: &mut DoraNode,
        outputs: &PlaybackOutputs,
        interrupted_command_id: Option<String>,
    ) -> Result<()> {
        if self.walkie_control_active {
            return Ok(());
        }
        let interrupted = interrupted_command_id.is_some();
        let state = PlaybackState {
            entity_id: self.entity_id.clone(),
            state: PlaybackStateKind::Active,
            source: Some(PlaybackSource::Walkie),
            command_id: interrupted_command_id,
            timestamp: current_time_ms(),
            reason_code: interrupted.then_some(VoiceReasonCode::InterruptedByWalkie),
            detail: None,
        };
        send_state(node, outputs.walkie_state.clone(), &state)?;
        self.walkie_control_active = true;
        Ok(())
    }

    pub fn report_walkie_idle(
        &mut self,
        node: &mut DoraNode,
        outputs: &PlaybackOutputs,
    ) -> Result<()> {
        if !self.walkie_control_active {
            return Ok(());
        }
        send_state(node, outputs.walkie_state.clone(), &self.idle_state())?;
        self.walkie_control_active = false;
        Ok(())
    }

    pub fn report_unavailable(
        &mut self,
        node: &mut DoraNode,
        outputs: &PlaybackOutputs,
    ) -> Result<()> {
        let state = PlaybackState {
            entity_id: self.entity_id.clone(),
            state: PlaybackStateKind::Unavailable,
            source: None,
            command_id: None,
            timestamp: current_time_ms(),
            reason_code: Some(VoiceReasonCode::PlaybackUnavailable),
            detail: Some("audio output unavailable".to_owned()),
        };
        send_state(node, outputs.playback_state.clone(), &state)?;
        send_state(node, outputs.walkie_state.clone(), &state)?;
        Ok(())
    }

    fn idle_state(&self) -> PlaybackState {
        PlaybackState {
            entity_id: self.entity_id.clone(),
            state: PlaybackStateKind::Idle,
            source: None,
            command_id: None,
            timestamp: current_time_ms(),
            reason_code: None,
            detail: None,
        }
    }
}

fn send_state(node: &mut DoraNode, output: DataId, state: &PlaybackState) -> Result<()> {
    state.validate().map_err(eyre::Report::msg)?;
    let bytes = serde_json::to_vec(state)?;
    node.send_output(
        output,
        MetadataParameters::default(),
        BinaryArray::from_vec(vec![bytes.as_slice()]),
    )?;
    Ok(())
}

pub(crate) fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
