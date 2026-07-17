use dora_node_api::{
    arrow::array::{Array, BinaryArray},
    dora_core::config::DataId,
    DoraNode, Event, MetadataParameters, Parameter,
};
use eyre::Result;
use robo_rover_lib::{
    init_tracing, AudioFrameMetadata, PcmSampleFormat, RecordingClipQuery,
    RecordingClipQueryResult, RecordingPlaybackTicketRequest, RecordingReasonCode,
    RecordingSessionAction, RecordingSessionCommand, RecordingSessionCommandResult,
    VideoFrameMetadata, RECORDING_PROTOCOL_VERSION,
};
use std::convert::TryFrom;
use std::time::Duration;
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
                "recording_session_command" => handle_command(
                    &mut node,
                    &result_output,
                    &status_output,
                    &mut manager,
                    &*data,
                )?,
                "recording_clip_query" => {
                    handle_clip_query(&mut node, &clip_list_output, &manager, &*data)?
                }
                "recording_playback_ticket" => {
                    handle_playback_lookup(&mut node, &playback_lookup_output, &manager, &*data)?
                }
                "video_frame" => {
                    if let Ok((entity, frame)) = parse_video(&metadata.parameters, &*data) {
                        let _ = manager.push_video(&entity, frame);
                    }
                }
                "audio_frame" => {
                    if let Ok((entity, frame)) = parse_audio(&metadata.parameters, &*data) {
                        let _ = manager.push_audio(&entity, frame);
                    }
                }
                _ => tracing::debug!(input = %id, "ignored recorder input"),
            },
            Event::Stop(_) => break,
            _ => {}
        }
    }
    for status in manager.shutdown() {
        send_json(&mut node, &status_output, &status.wire())?;
    }
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
