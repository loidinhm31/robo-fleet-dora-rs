use super::*;
use std::collections::{BTreeMap, HashMap, VecDeque};

const MAX_PENDING_COMMANDS: usize = 128;

#[derive(Clone)]
struct PendingLifecycleRequest {
    socket_id: String,
    expires_at_ms: u64,
}

#[derive(Clone, Default)]
pub struct LifecycleSocketState {
    commands: Arc<Mutex<VecDeque<LifecycleCommand>>>,
    pending: Arc<Mutex<HashMap<String, PendingLifecycleRequest>>>,
    statuses: Arc<Mutex<BTreeMap<robo_rover_lib::LifecycleTarget, LifecycleStatus>>>,
    capabilities: Arc<Mutex<Vec<LifecycleCapability>>>,
    status_query_pending: Arc<Mutex<bool>>,
}

impl LifecycleSocketState {
    pub fn register(&self, socket: &SocketRef, socket_id: &str, shared: &SharedState) {
        for capability in self
            .capabilities
            .lock()
            .ok()
            .map(|value| value.clone())
            .unwrap_or_default()
        {
            socket.emit("lifecycle_capability", capability).ok();
        }
        for status in self
            .statuses
            .lock()
            .ok()
            .map(|value| value.values().cloned().collect::<Vec<_>>())
            .unwrap_or_default()
        {
            socket.emit("lifecycle_status", status).ok();
        }
        if self
            .statuses
            .lock()
            .ok()
            .is_some_and(|statuses| statuses.is_empty())
        {
            self.request_status_query();
        }
        let state = self.clone();
        let shared = shared.clone();
        let socket_id = socket_id.to_owned();
        socket.on("node_lifecycle_command", move |socket: SocketRef, Data::<Value>(data)| {
            if !shared.session_registry.is_valid(&socket_id) { socket.emit("auth_error", serde_json::json!({"reason":"token_expired"})).ok(); socket.disconnect().ok(); return; }
            if !shared.command_rate_limiter.check_command(&socket_id) { log_rate_limit_exceeded(&socket_id, "node_lifecycle_command"); return; }
            let command = match serde_json::from_value::<LifecycleCommand>(data) { Ok(value) => value, Err(error) => { Self::reject(&socket, "00000000-0000-0000-0000-000000000000", LifecycleReasonCode::InvalidRequest, error.to_string()); return; } };
            if let Err(error) = command.validate() { Self::reject(&socket, &command.request_id, LifecycleReasonCode::InvalidRequest, error); return; }
            if command_is_expired(&command, unix_now_ms().max(0) as u64) {
                Self::reject(&socket, &command.request_id, LifecycleReasonCode::Expired, "lifecycle command expired");
                return;
            }
            let allowed = match command.target.role { LifecycleRole::Orchestra => command.target.entity_id == "orchestra", LifecycleRole::Rover => is_target_active(&shared, &command.target.entity_id) };
            if !allowed { Self::reject(&socket, &command.request_id, LifecycleReasonCode::InvalidTarget, "target is not active"); return; }
            let capability = state.capabilities.lock().ok().and_then(|capabilities| capabilities.iter().find(|capability| capability.target == command.target).cloned());
            match capability {
                Some(capability) if capability.supported && !capability.always_on => {}
                Some(_) => { Self::reject(&socket, &command.request_id, LifecycleReasonCode::Unsupported, "target is not lifecycle-controllable"); return; }
                None => { Self::reject(&socket, &command.request_id, LifecycleReasonCode::InvalidTarget, "target is not server-advertised"); state.request_status_query(); return; }
            }
            if !state.register_pending(&command.request_id, &socket_id, command.expires_at_ms) {
                Self::reject(&socket, &command.request_id, LifecycleReasonCode::Internal, "lifecycle pending request capacity reached");
                return;
            }
            let actor = shared.session_registry.audit_actor(&socket_id).unwrap_or_else(|| "unknown".into());
            tracing::info!(security_event = "lifecycle_admission", actor, request_id = %command.request_id, target = ?command.target, "lifecycle command queued");
            match state.commands.lock() {
                Ok(mut queue) if queue.len() < MAX_PENDING_COMMANDS => queue.push_back(command),
                Ok(_) => { state.remove_pending(&command.request_id, &socket_id); Self::reject(&socket, &command.request_id, LifecycleReasonCode::Internal, "lifecycle command queue is full"); }
                Err(_) => { state.remove_pending(&command.request_id, &socket_id); Self::reject(&socket, &command.request_id, LifecycleReasonCode::Internal, "lifecycle command queue unavailable"); }
            }
        });
    }
    pub fn next_command(&self) -> Option<LifecycleCommand> {
        self.commands.lock().ok()?.pop_front()
    }
    pub fn request_status_query(&self) {
        if let Ok(mut pending) = self.status_query_pending.lock() {
            *pending = true;
        }
    }
    pub fn take_status_query(&self) -> bool {
        self.status_query_pending
            .lock()
            .map(|mut pending| std::mem::take(&mut *pending))
            .unwrap_or(false)
    }
    pub fn cache_status(&self, status: LifecycleStatus) {
        self.statuses
            .lock()
            .ok()
            .map(|mut cache| cache.insert(status.target.clone(), status));
    }
    pub fn cache_capabilities(&self, capabilities: Vec<LifecycleCapability>) {
        if let Ok(mut cache) = self.capabilities.lock() {
            for capability in capabilities {
                cache.retain(|current| current.target != capability.target);
                cache.push(capability);
            }
        }
    }
    pub fn sweep_pending(&self, now_ms: u64) {
        if let Ok(mut pending) = self.pending.lock() {
            pending.retain(|_, request| request.expires_at_ms > now_ms);
        }
    }
    pub fn take_pending(&self, request_id: &str) -> Option<String> {
        self.pending
            .lock()
            .ok()?
            .remove(request_id)
            .map(|request| request.socket_id)
    }
    fn register_pending(&self, request_id: &str, socket_id: &str, expires_at_ms: u64) -> bool {
        let Ok(mut pending) = self.pending.lock() else {
            return false;
        };
        if let Some(request) = pending.get(request_id) {
            return request.socket_id == socket_id;
        }
        if pending.len() >= MAX_PENDING_COMMANDS {
            return false;
        }
        pending.insert(
            request_id.to_owned(),
            PendingLifecycleRequest {
                socket_id: socket_id.to_owned(),
                expires_at_ms,
            },
        );
        true
    }
    fn remove_pending(&self, request_id: &str, socket_id: &str) {
        if let Ok(mut pending) = self.pending.lock() {
            if pending
                .get(request_id)
                .is_some_and(|request| request.socket_id == socket_id)
            {
                pending.remove(request_id);
            }
        }
    }
    fn reject(
        socket: &SocketRef,
        request_id: &str,
        reason_code: LifecycleReasonCode,
        detail: impl Into<String>,
    ) {
        socket
            .emit(
                "node_lifecycle_command_result",
                LifecycleCommandResult {
                    protocol_version: 1,
                    request_id: request_id.into(),
                    accepted: false,
                    manager_epoch: 0,
                    revision: 0,
                    reason_code: Some(reason_code),
                    detail: Some(detail.into()),
                },
            )
            .ok();
    }
}

fn command_is_expired(command: &LifecycleCommand, now_ms: u64) -> bool {
    command.expires_at_ms <= now_ms
}

#[cfg(test)]
mod tests {
    use super::*;
    use robo_rover_lib::{LifecycleDesiredState, LifecycleRole, LifecycleTarget};

    #[test]
    fn expired_commands_are_rejected_before_pending_registration() {
        let command = LifecycleCommand {
            protocol_version: 1,
            request_id: "request".into(),
            manager_epoch: 1,
            target: LifecycleTarget {
                role: LifecycleRole::Rover,
                entity_id: "rover-kiwi".into(),
                node_id: "edge-voice".into(),
            },
            desired_state: LifecycleDesiredState::Quiesced,
            expected_revision: 0,
            issued_at_ms: 1,
            expires_at_ms: 2,
            origin: Default::default(),
            transition_id: None,
        };

        assert!(command_is_expired(&command, 2));
        assert!(!command_is_expired(&command, 1));
    }
}
