use robo_rover_lib::{AuthenticatedRecordingScheduleCommand, RecordingScheduleQuery};
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const DEFAULT_CAPACITY: usize = 64;
const MAX_CAPACITY: usize = 1_024;
const DEFAULT_TTL: Duration = Duration::from_secs(15);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleRequestKind {
    Command,
    Query,
}

#[derive(Clone, Debug)]
pub struct PendingScheduleRequest {
    pub socket_id: String,
    pub kind: ScheduleRequestKind,
    pub deadline: Instant,
}

#[derive(Clone)]
pub struct RecordingScheduleState {
    pub commands: Arc<Mutex<VecDeque<AuthenticatedRecordingScheduleCommand>>>,
    pub queries: Arc<Mutex<VecDeque<RecordingScheduleQuery>>>,
    pending: Arc<Mutex<HashMap<String, PendingScheduleRequest>>>,
    capacity: usize,
    ttl: Duration,
}

impl RecordingScheduleState {
    pub fn from_env() -> Self {
        let capacity = std::env::var("RECORDING_SCHEDULE_QUEUE_CAPACITY")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_CAPACITY)
            .clamp(8, MAX_CAPACITY);
        let ttl = std::env::var("RECORDING_SCHEDULE_REQUEST_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map(|seconds| Duration::from_secs(seconds.clamp(1, 60)))
            .unwrap_or(DEFAULT_TTL);
        Self {
            commands: Arc::new(Mutex::new(VecDeque::new())),
            queries: Arc::new(Mutex::new(VecDeque::new())),
            pending: Arc::new(Mutex::new(HashMap::new())),
            capacity,
            ttl,
        }
    }

    pub fn admit(
        &self,
        request_id: &str,
        socket_id: &str,
        kind: ScheduleRequestKind,
    ) -> Result<(), &'static str> {
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| "schedule state unavailable")?;
        if pending.len() >= self.capacity {
            return Err("recording schedule queue is full");
        }
        if pending.contains_key(request_id) {
            return Err("request_id is already pending");
        }
        pending.insert(
            request_id.to_owned(),
            PendingScheduleRequest {
                socket_id: socket_id.to_owned(),
                kind,
                deadline: Instant::now() + self.ttl,
            },
        );
        Ok(())
    }

    pub fn take(&self, request_id: &str) -> Option<PendingScheduleRequest> {
        self.pending.lock().ok()?.remove(request_id)
    }

    pub fn is_live(&self, request_id: &str) -> bool {
        self.pending
            .lock()
            .ok()
            .and_then(|pending| pending.get(request_id).cloned())
            .is_some_and(|request| request.deadline > Instant::now())
    }

    pub fn next_command(&self) -> Option<AuthenticatedRecordingScheduleCommand> {
        let mut queue = self.commands.lock().ok()?;
        while let Some(command) = queue.pop_front() {
            if self.is_live(&command.command.request_id) {
                return Some(command);
            }
        }
        None
    }

    pub fn next_query(&self) -> Option<RecordingScheduleQuery> {
        let mut queue = self.queries.lock().ok()?;
        while let Some(query) = queue.pop_front() {
            if self.is_live(&query.request_id) {
                return Some(query);
            }
        }
        None
    }

    pub fn expire(&self) -> Vec<(String, PendingScheduleRequest)> {
        let now = Instant::now();
        let Ok(mut pending) = self.pending.lock() else {
            return Vec::new();
        };
        let expired = pending
            .iter()
            .filter(|(_, request)| request.deadline <= now)
            .map(|(request_id, _)| request_id.clone())
            .collect::<Vec<_>>();
        expired
            .into_iter()
            .filter_map(|request_id| {
                pending
                    .remove(&request_id)
                    .map(|request| (request_id, request))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_is_unique_across_schedule_operations() {
        let state = RecordingScheduleState::from_env();
        state
            .admit("id", "socket-a", ScheduleRequestKind::Command)
            .unwrap();
        assert!(state
            .admit("id", "socket-b", ScheduleRequestKind::Query)
            .is_err());
        assert_eq!(state.take("id").unwrap().socket_id, "socket-a");
    }

    #[test]
    fn expired_queued_requests_never_execute() {
        let state = RecordingScheduleState {
            commands: Arc::new(Mutex::new(VecDeque::new())),
            queries: Arc::new(Mutex::new(VecDeque::new())),
            pending: Arc::new(Mutex::new(HashMap::new())),
            capacity: 1,
            ttl: Duration::ZERO,
        };
        state
            .admit("request", "socket", ScheduleRequestKind::Query)
            .unwrap();
        state
            .queries
            .lock()
            .unwrap()
            .push_back(RecordingScheduleQuery {
                protocol_version: 1,
                request_id: "request".into(),
                entity_id: "rover-a".into(),
            });
        assert!(state.next_query().is_none());
    }
}
