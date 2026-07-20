use robo_rover_lib::{AuthenticatedRecordingScheduleCommand, RecordingScheduleQuery};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

const DEFAULT_CAPACITY: usize = 64;
const MAX_CAPACITY: usize = 1_024;
const DEFAULT_TTL: Duration = Duration::from_secs(15);
const DEFAULT_READY_LEASE: Duration = Duration::from_secs(90);

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
    ready_lease: Duration,
    enabled: bool,
    ready: Arc<AtomicBool>,
    ready_until: Arc<Mutex<Option<Instant>>>,
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
        let ready_lease = std::env::var("RECORDING_SCHEDULER_READY_LEASE_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map(|seconds| Duration::from_secs(seconds.clamp(10, 300)))
            .unwrap_or(DEFAULT_READY_LEASE);
        Self {
            commands: Arc::new(Mutex::new(VecDeque::new())),
            queries: Arc::new(Mutex::new(VecDeque::new())),
            pending: Arc::new(Mutex::new(HashMap::new())),
            capacity,
            ttl,
            ready_lease,
            enabled: feature_enabled(),
            ready: Arc::new(AtomicBool::new(false)),
            ready_until: Arc::new(Mutex::new(None)),
        }
    }

    pub fn unavailable_reason(&self) -> Option<&'static str> {
        if !self.enabled {
            Some("recording scheduler is disabled")
        } else if !self.scheduler_ready() {
            Some("recording scheduler is unavailable")
        } else {
            None
        }
    }

    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::Release);
        if let Ok(mut ready_until) = self.ready_until.lock() {
            *ready_until = ready.then(|| Instant::now() + self.ready_lease);
        }
    }

    fn scheduler_ready(&self) -> bool {
        if !self.ready.load(Ordering::Acquire) {
            return false;
        }
        let ready = self
            .ready_until
            .lock()
            .map(|ready_until| ready_until.is_some_and(|until| until > Instant::now()))
            .unwrap_or(false);
        if !ready {
            self.ready.store(false, Ordering::Release);
        }
        ready
    }

    pub fn queue_depth(&self) -> usize {
        let commands = self.commands.lock().map(|queue| queue.len()).unwrap_or(0);
        let queries = self.queries.lock().map(|queue| queue.len()).unwrap_or(0);
        commands.saturating_add(queries)
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

fn feature_enabled() -> bool {
    feature_enabled_value(std::env::var("RECORDING_SCHEDULER_ENABLED").ok().as_deref())
}

fn feature_enabled_value(value: Option<&str>) -> bool {
    matches!(value, Some("true"))
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
            ready_lease: DEFAULT_READY_LEASE,
            enabled: true,
            ready: Arc::new(AtomicBool::new(true)),
            ready_until: Arc::new(Mutex::new(Some(Instant::now() + DEFAULT_READY_LEASE))),
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

    #[test]
    fn feature_flag_only_accepts_the_compose_enable_value() {
        assert!(feature_enabled_value(Some("true")));
        assert!(!feature_enabled_value(Some("1")));
        assert!(!feature_enabled_value(Some("yes")));
        assert!(!feature_enabled_value(None));
    }

    #[test]
    fn enabled_scheduler_rejects_requests_until_ready() {
        let state = RecordingScheduleState {
            commands: Arc::new(Mutex::new(VecDeque::new())),
            queries: Arc::new(Mutex::new(VecDeque::new())),
            pending: Arc::new(Mutex::new(HashMap::new())),
            capacity: 8,
            ttl: DEFAULT_TTL,
            ready_lease: DEFAULT_READY_LEASE,
            enabled: true,
            ready: Arc::new(AtomicBool::new(false)),
            ready_until: Arc::new(Mutex::new(None)),
        };
        assert_eq!(
            state.unavailable_reason(),
            Some("recording scheduler is unavailable")
        );
        state.set_ready(true);
        assert_eq!(state.unavailable_reason(), None);
    }

    #[test]
    fn expired_readiness_lease_fails_closed() {
        let state = RecordingScheduleState {
            commands: Arc::new(Mutex::new(VecDeque::new())),
            queries: Arc::new(Mutex::new(VecDeque::new())),
            pending: Arc::new(Mutex::new(HashMap::new())),
            capacity: 8,
            ttl: DEFAULT_TTL,
            ready_lease: DEFAULT_READY_LEASE,
            enabled: true,
            ready: Arc::new(AtomicBool::new(true)),
            ready_until: Arc::new(Mutex::new(Some(Instant::now() - Duration::from_secs(1)))),
        };
        assert_eq!(
            state.unavailable_reason(),
            Some("recording scheduler is unavailable")
        );
    }
}
