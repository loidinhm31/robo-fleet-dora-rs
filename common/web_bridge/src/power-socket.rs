use crate::{
    power_history_gateway::PowerHistoryQuery,
    power_queues::{PendingPowerKind, PowerSocketState},
    security::{log_rate_limit_exceeded, log_validation_error},
    selected_browser_target, unix_now_ms, SharedState,
};
use robo_rover_lib::{
    PowerAuthority, PowerCommandResult, PowerPolicy, PowerReasonCode, POWER_PROTOCOL_VERSION,
};
use serde::Deserialize;
use serde_json::Value;
use socketioxide::extract::{Data, SocketRef};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyRequest {
    protocol_version: u8,
    request_id: String,
    policy: PowerPolicy,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WakeRequest {
    protocol_version: u8,
    request_id: String,
}

pub fn register(socket: &SocketRef, socket_id: &str, state: SharedState) {
    register_policy(socket, socket_id, state.clone());
    register_wake(socket, socket_id, state.clone());
    register_history(socket, socket_id, state);
}

pub fn handle_result(
    state: &PowerSocketState,
    result: &PowerCommandResult,
    now_ms: u64,
) -> Option<(String, &'static str, Value)> {
    let pending = state.take_pending(&result.command_id)?;
    tracing::info!(security_event = "power_command_result", socket_id = %pending.socket_id, entity_id = %pending.entity_id, request_id = %pending.request_id, command_id = %result.command_id, accepted = result.accepted, "power command result received");
    if matches!(
        &pending.kind,
        PendingPowerKind::Policy | PendingPowerKind::Release
    ) {
        return Some((
            pending.socket_id,
            "power_command_result",
            command_result(&pending.request_id, result)?,
        ));
    }
    match pending.kind {
        PendingPowerKind::Policy | PendingPowerKind::Release => Some((
            pending.socket_id,
            "power_command_result",
            command_result(&pending.request_id, result)?,
        )),
        PendingPowerKind::WakePolicy { .. } if result.accepted => {
            state.accept_wake_policy(pending, result.authority, now_ms);
            None
        }
        PendingPowerKind::WakePolicy { .. } => Some((
            pending.socket_id,
            "power_wake_result",
            wake_result(
                &pending.request_id,
                false,
                result.reason_code,
                result.detail.clone(),
            ),
        )),
        PendingPowerKind::WakeDemand { .. } if result.accepted => {
            state.complete_wake_demand(&pending, result.authority, now_ms);
            Some((
                pending.socket_id,
                "power_wake_result",
                wake_result(&pending.request_id, true, None, None),
            ))
        }
        PendingPowerKind::WakeDemand { .. } => Some((
            pending.socket_id,
            "power_wake_result",
            wake_result(
                &pending.request_id,
                false,
                result.reason_code,
                result.detail.clone(),
            ),
        )),
    }
}

fn command_result(request_id: &str, result: &PowerCommandResult) -> Option<Value> {
    let mut value = serde_json::to_value(result).ok()?;
    value
        .as_object_mut()?
        .insert("request_id".into(), Value::String(request_id.into()));
    Some(value)
}

fn register_policy(socket: &SocketRef, socket_id: &str, state: SharedState) {
    let socket_id = socket_id.to_owned();
    socket.on("power_policy", move |socket: SocketRef, Data::<Value>(value)| {
        let request = match serde_json::from_value::<PolicyRequest>(value) { Ok(request) => request, Err(error) => { log_validation_error(&socket_id, &format!("power policy: {error}")); return; } };
        if request.protocol_version != POWER_PROTOCOL_VERSION || !canonical_uuid(&request.request_id) { reject_command(&socket, &request.request_id, PowerReasonCode::InvalidRequest, "invalid power policy request"); return; }
        let Some(entity_id) = authenticated_target(&socket, &socket_id, &state, "power_policy") else { reject_command(&socket, &request.request_id, PowerReasonCode::InvalidTarget, "selected rover is not active"); return; };
        let now_ms = unix_now_ms().max(0) as u64;
        match state.power.queue_policy(socket_id.clone(), request.request_id.clone(), entity_id.clone(), request.policy, now_ms) {
            Ok(()) => tracing::info!(security_event = "power_policy_admission", entity_id, request_id = %request.request_id, "power policy request queued"),
            Err(detail) => reject_command(&socket, &request.request_id, rejection_reason(&detail), &detail),
        }
    });
}

fn register_wake(socket: &SocketRef, socket_id: &str, state: SharedState) {
    let socket_id = socket_id.to_owned();
    socket.on("power_wake", move |socket: SocketRef, Data::<Value>(value)| {
        let request = match serde_json::from_value::<WakeRequest>(value) { Ok(request) => request, Err(_) => { socket.emit("power_wake_result", wake_result("", false, Some(PowerReasonCode::InvalidRequest), Some("invalid wake request".into()))).ok(); return; } };
        if request.protocol_version != POWER_PROTOCOL_VERSION || !canonical_uuid(&request.request_id) { socket.emit("power_wake_result", wake_result(&request.request_id, false, Some(PowerReasonCode::InvalidRequest), Some("invalid wake request".into()))).ok(); return; }
        let Some(entity_id) = authenticated_target(&socket, &socket_id, &state, "power_wake") else { socket.emit("power_wake_result", wake_result(&request.request_id, false, Some(PowerReasonCode::InvalidTarget), Some("selected rover is not active".into()))).ok(); return; };
        let now_ms = unix_now_ms().max(0) as u64;
        match state.power.queue_wake(socket_id.clone(), request.request_id.clone(), entity_id.clone(), now_ms) {
            Ok(()) => tracing::info!(security_event = "power_wake_admission", entity_id, request_id = %request.request_id, "power wake request queued"),
            Err(detail) => { socket.emit("power_wake_result", wake_result(&request.request_id, false, Some(rejection_reason(&detail)), Some(detail))).ok(); }
        }
    });
}

fn register_history(socket: &SocketRef, socket_id: &str, state: SharedState) {
    let socket_id = socket_id.to_owned();
    socket.on(
        "power_history",
        move |socket: SocketRef, Data::<Value>(value)| {
            let query = match serde_json::from_value::<PowerHistoryQuery>(value) {
                Ok(query) => query,
                Err(_) => return,
            };
            let Some(entity_id) =
                authenticated_target(&socket, &socket_id, &state, "power_history")
            else {
                return;
            };
            if let Some(status) = state.power.status(&entity_id) {
                socket.emit("power_status", status).ok();
            }
            let gateway = state.power_history.clone();
            tokio::spawn(async move {
                socket
                    .emit(
                        "power_history_result",
                        gateway.query(&entity_id, query, unix_now_ms()).await,
                    )
                    .ok();
            });
        },
    );
}

fn authenticated_target(
    socket: &SocketRef,
    socket_id: &str,
    state: &SharedState,
    event: &str,
) -> Option<String> {
    if !state.session_registry.is_valid(socket_id) {
        socket
            .emit("auth_error", serde_json::json!({"reason":"token_expired"}))
            .ok();
        socket.clone().disconnect().ok();
        return None;
    }
    if !state.power_rate_limiter.check_command(socket_id) {
        log_rate_limit_exceeded(socket_id, event);
        return None;
    }
    selected_browser_target(state)
}

fn canonical_uuid(value: &str) -> bool {
    uuid::Uuid::parse_str(value)
        .map(|id| id.hyphenated().to_string() == value)
        .unwrap_or(false)
}

fn rejection_reason(detail: &str) -> PowerReasonCode {
    if detail.starts_with("duplicate") || detail.starts_with("wake request") {
        PowerReasonCode::Conflict
    } else {
        PowerReasonCode::SnapshotStale
    }
}

fn reject_command(
    socket: &SocketRef,
    command_id: &str,
    reason_code: PowerReasonCode,
    detail: &str,
) {
    socket
        .emit(
            "power_command_result",
            PowerCommandResult {
                protocol_version: POWER_PROTOCOL_VERSION,
                command_id: command_id.into(),
                accepted: false,
                authority: PowerAuthority {
                    epoch: 1,
                    sequence: 1,
                },
                reason_code: Some(reason_code),
                detail: Some(detail.chars().take(256).collect()),
            },
        )
        .ok();
}
fn wake_result(
    request_id: &str,
    accepted: bool,
    reason_code: Option<PowerReasonCode>,
    detail: Option<String>,
) -> Value {
    serde_json::json!({ "protocol_version": POWER_PROTOCOL_VERSION, "request_id": request_id, "accepted": accepted, "reason_code": reason_code, "detail": detail })
}
