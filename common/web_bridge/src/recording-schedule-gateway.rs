use std::time::{SystemTime, UNIX_EPOCH};

use robo_rover_lib::{
    AuthenticatedRecordingScheduleCommand, RecordingScheduleCommand, RecordingScheduleQuery,
    RecordingScheduleReasonCode, RECORDING_SCHEDULE_PROTOCOL_VERSION,
};
use serde_json::Value;
use socketioxide::extract::{Data, SocketRef};

use crate::{
    recording_schedule_queues::ScheduleRequestKind, security::log_rate_limit_exceeded,
    security::log_validation_error, SharedState,
};

pub fn register(socket: &SocketRef, socket_id: &str, state: SharedState) {
    register_command(socket, socket_id, state.clone());
    register_query(socket, socket_id, state);
}

fn register_command(socket: &SocketRef, socket_id: &str, state: SharedState) {
    let socket_id = socket_id.to_owned();
    socket.on(
        "recording_schedule_command",
        move |socket: SocketRef, Data::<Value>(data)| {
            if let Some(detail) = state.schedule.unavailable_reason() {
                reject(
                    &socket,
                    request_id_from_value(&data),
                    RecordingScheduleReasonCode::Unavailable,
                    detail,
                );
                return;
            }
            let Some(actor) = state.session_registry.audit_actor(&socket_id) else {
                socket
                    .emit("auth_error", serde_json::json!({"reason": "token_expired"}))
                    .ok();
                socket.disconnect().ok();
                return;
            };
            if !state.command_rate_limiter.check_command(&socket_id) {
                log_rate_limit_exceeded(&socket_id, "recording_schedule_command");
                return;
            }
            let command = match serde_json::from_value::<RecordingScheduleCommand>(data) {
                Ok(command) => command,
                Err(error) => {
                    log_validation_error(
                        &socket_id,
                        &format!("recording schedule command: {error}"),
                    );
                    return;
                }
            };
            let request_id = command.request_id.clone();
            if let Err(error) = command.validate_at(now_ms(), Default::default()) {
                log_validation_error(&socket_id, &format!("recording schedule command: {error}"));
                reject(
                    &socket,
                    &request_id,
                    RecordingScheduleReasonCode::InvalidRequest,
                    &error,
                );
                return;
            }
            if let Err(error) =
                state
                    .schedule
                    .admit(&request_id, &socket_id, ScheduleRequestKind::Command)
            {
                reject(
                    &socket,
                    &request_id,
                    RecordingScheduleReasonCode::Unavailable,
                    error,
                );
                return;
            }
            let wrapped = AuthenticatedRecordingScheduleCommand {
                command,
                audit_actor: actor,
            };
            if state
                .schedule
                .commands
                .lock()
                .map(|mut queue| queue.push_back(wrapped))
                .is_err()
            {
                state.schedule.take(&request_id);
                reject(
                    &socket,
                    &request_id,
                    RecordingScheduleReasonCode::Internal,
                    "schedule queue unavailable",
                );
            } else {
                tracing::info!(
                    event = "recording_schedule_queue_depth",
                    operation = "command",
                    queue_depth = state.schedule.queue_depth(),
                    "recording schedule command admitted"
                );
            }
        },
    );
}

fn register_query(socket: &SocketRef, socket_id: &str, state: SharedState) {
    let socket_id = socket_id.to_owned();
    socket.on(
        "recording_schedule_query",
        move |socket: SocketRef, Data::<Value>(data)| {
            if let Some(detail) = state.schedule.unavailable_reason() {
                reject(
                    &socket,
                    request_id_from_value(&data),
                    RecordingScheduleReasonCode::Unavailable,
                    detail,
                );
                return;
            }
            if state.session_registry.audit_actor(&socket_id).is_none() {
                socket
                    .emit("auth_error", serde_json::json!({"reason": "token_expired"}))
                    .ok();
                socket.disconnect().ok();
                return;
            }
            if !state.command_rate_limiter.check_command(&socket_id) {
                log_rate_limit_exceeded(&socket_id, "recording_schedule_query");
                return;
            }
            let query = match serde_json::from_value::<RecordingScheduleQuery>(data) {
                Ok(query) => query,
                Err(error) => {
                    log_validation_error(&socket_id, &format!("recording schedule query: {error}"));
                    return;
                }
            };
            let request_id = query.request_id.clone();
            if let Err(error) = query.validate() {
                reject(
                    &socket,
                    &request_id,
                    RecordingScheduleReasonCode::InvalidRequest,
                    &error,
                );
                return;
            }
            if let Err(error) =
                state
                    .schedule
                    .admit(&request_id, &socket_id, ScheduleRequestKind::Query)
            {
                reject(
                    &socket,
                    &request_id,
                    RecordingScheduleReasonCode::Unavailable,
                    error,
                );
                return;
            }
            if state
                .schedule
                .queries
                .lock()
                .map(|mut queue| queue.push_back(query))
                .is_err()
            {
                state.schedule.take(&request_id);
                reject(
                    &socket,
                    &request_id,
                    RecordingScheduleReasonCode::Internal,
                    "schedule queue unavailable",
                );
            } else {
                tracing::info!(
                    event = "recording_schedule_queue_depth",
                    operation = "query",
                    queue_depth = state.schedule.queue_depth(),
                    "recording schedule query admitted"
                );
            }
        },
    );
}

fn request_id_from_value(value: &Value) -> &str {
    value
        .get("request_id")
        .and_then(Value::as_str)
        .filter(|request_id| request_id.len() <= 128)
        .unwrap_or("")
}

fn reject(
    socket: &SocketRef,
    request_id: &str,
    reason_code: RecordingScheduleReasonCode,
    detail: &str,
) {
    socket
        .emit(
            "recording_schedule_command_result",
            serde_json::json!({
                "protocol_version": RECORDING_SCHEDULE_PROTOCOL_VERSION, "request_id": request_id,
                "accepted": false, "reason_code": reason_code,
                "detail": detail.chars().take(256).collect::<String>(),
            }),
        )
        .ok();
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
