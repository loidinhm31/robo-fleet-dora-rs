use dora_node_api::{
    arrow::array::{Array, BinaryArray},
    dora_core::config::DataId,
    DoraNode, Event, MetadataParameters, Parameter,
};
use eyre::Result;
use robo_rover_lib::{
    init_tracing, AudioFrameMetadata, LifecycleCommand, LifecycleComponentState, LifecycleGate,
    LifecycleReasonCode, LifecycleRole, LifecycleTarget, LifecycleTransition, PcmSampleFormat,
    RecordingClipQuery, RecordingClipQueryResult, RecordingDeleteRequest, RecordingDeleteResult,
    RecordingPlaybackTicketRequest, RecordingReasonCode, RecordingReconciliationRequest,
    RecordingReconciliationSnapshot, RecordingSessionAction, RecordingSessionCommand,
    RecordingSessionCommandResult, VideoFrameMetadata, RECORDING_PROTOCOL_VERSION,
};
use std::convert::TryFrom;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use media_recorder::config::RecorderConfig;
use media_recorder::frame_timeline::{AudioFrame, VideoFrame};
use media_recorder::session_manager::{SessionManager, StartRequest};

fn main() -> Result<()> {
    let _guard = init_tracing();
    let config = RecorderConfig::from_env().map_err(eyre::Report::msg)?;
    let mut manager = SessionManager::new(config).map_err(eyre::Report::msg)?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let result_output = DataId::from("recording_session_command_result".to_owned());
    let status_output = DataId::from("recording_session_status".to_owned());
    let clip_list_output = DataId::from("recording_clip_list_result".to_owned());
    let playback_lookup_output = DataId::from("recording_playback_clip_result".to_owned());
    let delete_output = DataId::from("recording_delete_result".to_owned());
    let reconciliation_output = DataId::from("recording_reconciliation_snapshot".to_owned());
    let lifecycle_status_output = DataId::from("lifecycle_component_status".to_owned());
    let mut lifecycle_gate = LifecycleGate::new(LifecycleTarget {
        role: LifecycleRole::Orchestra,
        entity_id: "orchestra".into(),
        node_id: "media-recorder".into(),
    });
    tracing::info!("media recorder ready");
    while let Some(event) = events.recv_timeout(Duration::from_millis(100)) {
        for status in manager.reap() {
            send_json(&mut node, &status_output, &status.wire())?;
        }
        for status in manager.statuses() {
            send_json(&mut node, &status_output, &status.wire())?;
        }
        match event {
            Event::Input { id, data, metadata } => match id.as_str() {
                "recording_session_command" => {
                    if lifecycle_gate.admission_open() {
                        handle_command(
                            &mut node,
                            &result_output,
                            &status_output,
                            &mut manager,
                            &*data,
                        )?;
                    } else {
                        tracing::warn!("recording command rejected while lifecycle-quiesced");
                    }
                }
                "recording_clip_query" => {
                    handle_clip_query(&mut node, &clip_list_output, &manager, &*data)?
                }
                "recording_playback_ticket" => {
                    handle_playback_lookup(&mut node, &playback_lookup_output, &manager, &*data)?
                }
                "recording_delete" => handle_delete(&mut node, &delete_output, &manager, &*data)?,
                "recording_reconciliation_request" => handle_reconciliation_request(
                    &mut node,
                    &reconciliation_output,
                    &mut manager,
                    &*data,
                )?,
                "video_frame" => {
                    if lifecycle_gate.admission_open() {
                        if let Ok((entity, frame)) = parse_video(&metadata.parameters, &*data) {
                            let _ = manager.push_video(&entity, frame);
                        }
                    }
                }
                "audio_frame" => {
                    if lifecycle_gate.admission_open() {
                        if let Ok((entity, frame)) = parse_audio(&metadata.parameters, &*data) {
                            let _ = manager.push_audio(&entity, frame);
                        }
                    }
                }
                "lifecycle_command" => handle_lifecycle_command(
                    &mut node,
                    &lifecycle_status_output,
                    &status_output,
                    &mut manager,
                    &mut lifecycle_gate,
                    &*data,
                )?,
                _ => tracing::debug!(input = %id, "ignored recorder input"),
            },
            Event::Stop(_) => break,
            _ => {}
        }
    }
    for status in manager.shutdown().statuses {
        send_json(&mut node, &status_output, &status.wire())?;
    }
    Ok(())
}

fn handle_lifecycle_command(
    node: &mut DoraNode,
    lifecycle_output: &DataId,
    recording_output: &DataId,
    manager: &mut SessionManager,
    gate: &mut LifecycleGate,
    data: &dyn Array,
) -> Result<()> {
    let Ok(bytes) = single_binary(data) else {
        return Ok(());
    };
    let Ok(command) = serde_json::from_slice::<LifecycleCommand>(bytes) else {
        tracing::warn!("invalid lifecycle command for media recorder");
        return Ok(());
    };
    let transition = match gate.begin(&command) {
        Ok(transition) => transition,
        Err(error) => {
            tracing::warn!(%error, "rejected lifecycle command for media recorder");
            return Ok(());
        }
    };
    let Some(transition) = transition else {
        return Ok(());
    };
    send_lifecycle_status(
        node,
        lifecycle_output,
        gate,
        match transition {
            LifecycleTransition::Quiesce => LifecycleComponentState::Cancelling,
            LifecycleTransition::Resume => LifecycleComponentState::Resuming,
        },
        None,
    )?;
    if transition == LifecycleTransition::Quiesce {
        let shutdown = manager.shutdown();
        for status in &shutdown.statuses {
            send_json(node, recording_output, &status.wire())?;
        }
        if !shutdown.all_sessions_finalized {
            // Keep admission closed. A pause that cannot prove every accepted
            // recording finalized successfully is not a successful quiesce.
            return send_lifecycle_status(
                node,
                lifecycle_output,
                gate,
                LifecycleComponentState::Failed,
                Some(LifecycleReasonCode::Internal),
            );
        }
    } else if manager.has_draining_workers() {
        return send_lifecycle_status(
            node,
            lifecycle_output,
            gate,
            LifecycleComponentState::Failed,
            Some(LifecycleReasonCode::Timeout),
        );
    }
    gate.complete(transition);
    send_lifecycle_status(
        node,
        lifecycle_output,
        gate,
        match transition {
            LifecycleTransition::Quiesce => LifecycleComponentState::Quiesced,
            LifecycleTransition::Resume => LifecycleComponentState::Running,
        },
        None,
    )
}

fn send_lifecycle_status(
    node: &mut DoraNode,
    output: &DataId,
    gate: &LifecycleGate,
    state: LifecycleComponentState,
    reason: Option<LifecycleReasonCode>,
) -> Result<()> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
    send_json(node, output, &gate.status(state, reason, timestamp))
}

fn handle_reconciliation_request(
    node: &mut DoraNode,
    output: &DataId,
    manager: &mut SessionManager,
    data: &dyn Array,
) -> Result<()> {
    let bytes = match single_binary(data) {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(%error, "invalid reconciliation request payload");
            return Ok(());
        }
    };
    let request: RecordingReconciliationRequest = match serde_json::from_slice(bytes) {
        Ok(request) => request,
        Err(error) => {
            tracing::warn!(%error, "invalid reconciliation request");
            return Ok(());
        }
    };
    if let Err(error) = request.validate() {
        tracing::warn!(%error, "rejected reconciliation request");
        return Ok(());
    }
    let sessions = manager
        .statuses()
        .into_iter()
        .filter(|status| {
            request
                .entity_id
                .as_deref()
                .is_none_or(|entity| entity == status.entity_id)
        })
        .map(|status| status.reconciliation_session())
        .collect();
    send_json(
        node,
        output,
        &RecordingReconciliationSnapshot {
            request_id: request.request_id,
            sessions,
        },
    )
}

fn handle_delete(
    node: &mut DoraNode,
    output: &DataId,
    manager: &SessionManager,
    data: &dyn Array,
) -> Result<()> {
    let bytes = match single_binary(data) {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(%error, "invalid recording delete payload");
            return Ok(());
        }
    };
    let request: RecordingDeleteRequest = match serde_json::from_slice(bytes) {
        Ok(request) => request,
        Err(error) => {
            tracing::warn!(%error, "invalid recording delete request");
            return Ok(());
        }
    };
    let result = match request
        .validate()
        .and_then(|_| manager.delete(&request.recording_id))
    {
        Ok(()) => RecordingDeleteResult {
            protocol_version: RECORDING_PROTOCOL_VERSION,
            request_id: request.request_id,
            accepted: true,
            recording_id: Some(request.recording_id),
            reason_code: None,
            detail: None,
        },
        Err(error) => RecordingDeleteResult {
            protocol_version: RECORDING_PROTOCOL_VERSION,
            request_id: request.request_id,
            accepted: false,
            recording_id: None,
            reason_code: Some(if error.contains("active") {
                RecordingReasonCode::ActiveRecording
            } else if error.contains("partial") || error.contains("incomplete") {
                RecordingReasonCode::PartialRecording
            } else if error.contains("not found") {
                RecordingReasonCode::NotFound
            } else {
                RecordingReasonCode::DeleteFailed
            }),
            detail: Some(error.chars().take(256).collect()),
        },
    };
    send_json(node, output, &result)?;
    Ok(())
}

fn handle_clip_query(
    node: &mut DoraNode,
    output: &DataId,
    manager: &SessionManager,
    data: &dyn Array,
) -> Result<()> {
    let bytes = match single_binary(data) {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(%error, "invalid recording clip query payload");
            return Ok(());
        }
    };
    let query: RecordingClipQuery = match serde_json::from_slice(bytes) {
        Ok(query) => query,
        Err(error) => {
            tracing::warn!(%error, "invalid recording clip query");
            return Ok(());
        }
    };
    let result = match query.validate().and_then(|_| {
        manager.catalog().list(
            query.entity_id.as_deref(),
            query.relative_directory.as_deref(),
        )
    }) {
        Ok(clips) => RecordingClipQueryResult {
            protocol_version: RECORDING_PROTOCOL_VERSION,
            request_id: query.request_id,
            accepted: true,
            clips,
            reason_code: None,
            detail: None,
        },
        Err(error) => query_failure(&query.request_id, error),
    };
    send_json(node, output, &result)?;
    Ok(())
}

fn handle_playback_lookup(
    node: &mut DoraNode,
    output: &DataId,
    manager: &SessionManager,
    data: &dyn Array,
) -> Result<()> {
    let bytes = match single_binary(data) {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(%error, "invalid recording playback lookup payload");
            return Ok(());
        }
    };
    let request: RecordingPlaybackTicketRequest = match serde_json::from_slice(bytes) {
        Ok(request) => request,
        Err(error) => {
            tracing::warn!(%error, "invalid recording playback lookup");
            return Ok(());
        }
    };
    let result = match request
        .validate()
        .and_then(|_| manager.catalog().lookup(&request.recording_id))
    {
        Ok(clip) => RecordingClipQueryResult {
            protocol_version: RECORDING_PROTOCOL_VERSION,
            request_id: request.request_id,
            accepted: true,
            clips: vec![clip],
            reason_code: None,
            detail: None,
        },
        Err(error) => query_failure(&request.request_id, error),
    };
    send_json(node, output, &result)?;
    Ok(())
}

fn query_failure(request_id: &str, detail: String) -> RecordingClipQueryResult {
    RecordingClipQueryResult {
        protocol_version: RECORDING_PROTOCOL_VERSION,
        request_id: request_id.into(),
        accepted: false,
        clips: Vec::new(),
        reason_code: Some(if detail.contains("not found") {
            RecordingReasonCode::NotFound
        } else {
            RecordingReasonCode::InvalidRequest
        }),
        detail: Some(detail.chars().take(256).collect()),
    }
}

fn handle_command(
    node: &mut DoraNode,
    result_output: &DataId,
    status_output: &DataId,
    manager: &mut SessionManager,
    data: &dyn Array,
) -> Result<()> {
    let bytes = single_binary(data).map_err(eyre::Report::msg)?;
    let command: RecordingSessionCommand = match serde_json::from_slice(bytes) {
        Ok(command) => command,
        Err(error) => {
            tracing::warn!(%error, "invalid recording command");
            return Ok(());
        }
    };
    if let Err(error) = command.validate() {
        tracing::warn!(%error, "rejected recording command");
        let result = rejected(
            &command.request_id,
            RecordingReasonCode::InvalidRequest,
            error,
        );
        send_json(node, result_output, &result)?;
        return Ok(());
    }
    let request_id = command.request_id.clone();
    let result = match command.action {
        RecordingSessionAction::Start {
            entity_id,
            relative_directory,
        } => {
            match manager.start(StartRequest {
                request_id: request_id.clone(),
                entity_id,
                relative_directory,
            }) {
                Ok(status) => {
                    send_json(
                        node,
                        result_output,
                        &RecordingSessionCommandResult {
                            protocol_version: RECORDING_PROTOCOL_VERSION,
                            request_id: request_id.clone(),
                            accepted: true,
                            recording_id: Some(status.recording_id.clone()),
                            reason_code: None,
                            detail: None,
                        },
                    )?;
                    send_json(node, status_output, &status.wire())?;
                    return Ok(());
                }
                Err(error) => rejected(&request_id, reason(&error), error),
            }
        }
        RecordingSessionAction::Stop { recording_id } => match manager.stop(&recording_id) {
            Ok(()) => {
                send_json(
                    node,
                    result_output,
                    &RecordingSessionCommandResult {
                        protocol_version: RECORDING_PROTOCOL_VERSION,
                        request_id,
                        accepted: true,
                        recording_id: Some(recording_id),
                        reason_code: None,
                        detail: None,
                    },
                )?;
                for status in manager.statuses() {
                    send_json(node, status_output, &status.wire())?;
                }
                return Ok(());
            }
            Err(error) => rejected(&request_id, reason(&error), error),
        },
    };
    send_json(node, result_output, &result)?;
    Ok(())
}

fn rejected(
    request_id: &str,
    reason_code: RecordingReasonCode,
    detail: String,
) -> RecordingSessionCommandResult {
    RecordingSessionCommandResult {
        protocol_version: RECORDING_PROTOCOL_VERSION,
        request_id: request_id.into(),
        accepted: false,
        recording_id: None,
        reason_code: Some(reason_code),
        detail: Some(detail.chars().take(256).collect()),
    }
}

fn reason(error: &str) -> RecordingReasonCode {
    if error.contains("already") {
        RecordingReasonCode::AlreadyRecording
    } else if error.contains("free-space") || error.contains("limit") {
        RecordingReasonCode::ResourceLimit
    } else if error.contains("directory") || error.contains("path") {
        RecordingReasonCode::InvalidDirectory
    } else {
        RecordingReasonCode::Internal
    }
}

fn parse_video(
    metadata: &MetadataParameters,
    data: &dyn Array,
) -> Result<(String, VideoFrame), String> {
    let payload = single_binary(data)?;
    let entity_id = string(metadata, "entity_id")?;
    let frame_metadata = VideoFrameMetadata {
        frame_id: integer(metadata, "frame_id")?,
        capture_timestamp_ms: integer(metadata, "capture_timestamp_ms")?,
        width: integer(metadata, "width")?
            .try_into()
            .map_err(|_| "invalid width")?,
        height: integer(metadata, "height")?
            .try_into()
            .map_err(|_| "invalid height")?,
    };
    if payload.len() < 4
        || payload[..2] != [0xff, 0xd8]
        || payload[payload.len() - 2..] != [0xff, 0xd9]
    {
        return Err("invalid JPEG payload".into());
    }
    Ok((
        entity_id,
        VideoFrame {
            metadata: frame_metadata,
            payload: payload.to_vec(),
        },
    ))
}

fn parse_audio(
    metadata: &MetadataParameters,
    data: &dyn Array,
) -> Result<(String, AudioFrame), String> {
    let payload = single_binary(data)?;
    let format_name = string(metadata, "format")?;
    let format = PcmSampleFormat::from_metadata_name(&format_name)?;
    let stream_id = string(metadata, "stream_id")?;
    let frame_metadata = AudioFrameMetadata {
        stream_id: Uuid::parse_str(&stream_id).map_err(|_| "invalid stream_id")?,
        frame_id: integer(metadata, "frame_id")?,
        capture_timestamp_ms: integer(metadata, "capture_timestamp_ms")?,
        sample_rate: integer(metadata, "sample_rate")?
            .try_into()
            .map_err(|_| "invalid sample rate")?,
        channels: integer(metadata, "channels")?
            .try_into()
            .map_err(|_| "invalid channels")?,
        sample_count: integer(metadata, "sample_count")?
            .try_into()
            .map_err(|_| "invalid sample count")?,
        format,
    };
    frame_metadata.validate_payload_len(payload.len())?;
    if format != PcmSampleFormat::S16Le {
        return Err("recorder accepts only s16le".into());
    }
    Ok((
        string(metadata, "entity_id")?,
        AudioFrame {
            metadata: frame_metadata,
            payload: payload.to_vec(),
        },
    ))
}

fn single_binary(data: &dyn Array) -> Result<&[u8], String> {
    let array = data
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or("media input must be BinaryArray")?;
    if array.len() != 1 || array.is_null(0) {
        return Err("media input must have one non-null payload".into());
    }
    Ok(array.value(0))
}

fn integer(metadata: &MetadataParameters, key: &str) -> Result<u64, String> {
    match metadata.get(key) {
        Some(Parameter::Integer(value)) => {
            u64::try_from(*value).map_err(|_| format!("invalid {key}"))
        }
        _ => Err(format!("missing {key}")),
    }
}

fn string<'a>(metadata: &'a MetadataParameters, key: &str) -> Result<String, String> {
    match metadata.get(key) {
        Some(Parameter::String(value)) if !value.is_empty() => Ok(value.clone()),
        _ => Err(format!("missing {key}")),
    }
}

fn send_json<T: serde::Serialize>(node: &mut DoraNode, output: &DataId, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    node.send_output(
        output.clone(),
        dora_node_api::MetadataParameters::default(),
        BinaryArray::from_vec(vec![bytes.as_slice()]),
    )?;
    Ok(())
}
