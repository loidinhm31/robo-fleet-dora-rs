use robo_rover_lib::{
    RecordingClipQuery, RecordingDeleteRequest, RecordingPlaybackTicketRequest,
    RecordingSessionCommand, RecordingSessionStatus,
};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DEFAULT_QUEUE_CAPACITY: usize = 64;
const MAX_QUEUE_CAPACITY: usize = 1_024;
const DEFAULT_REQUEST_TTL: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    Command,
    ClipList,
    PlaybackTicket,
    Delete,
}

#[derive(Debug, Clone)]
pub struct PendingRequest {
    pub socket_id: String,
    pub kind: RequestKind,
    pub deadline: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledCommandGuard {
    pub group_id: String,
    pub generation: u64,
    pub consumer_id: String,
    pub planned_end_ms: i64,
}

impl ScheduledCommandGuard {
    pub fn is_expired_at(&self, now_ms: i64) -> bool {
        now_ms >= self.planned_end_ms
    }
}

#[derive(Debug, Clone)]
pub struct QueuedRecordingCommand {
    pub command: RecordingSessionCommand,
    pub scheduled_guard: Option<ScheduledCommandGuard>,
}

#[derive(Clone)]
pub struct RecordingState {
    pub commands: Arc<Mutex<VecDeque<QueuedRecordingCommand>>>,
    pub clip_queries: Arc<Mutex<VecDeque<RecordingClipQuery>>>,
    pub playback_queries: Arc<Mutex<VecDeque<RecordingPlaybackTicketRequest>>>,
    pub delete_queries: Arc<Mutex<VecDeque<RecordingDeleteRequest>>>,
    pub pending: Arc<Mutex<HashMap<String, PendingRequest>>>,
    pub statuses: Arc<Mutex<HashMap<String, RecordingSessionStatus>>>,
    pub active_entities: Arc<Mutex<HashMap<String, String>>>,
    capacity: usize,
    request_ttl: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_request_ids_are_unique_and_bounded_by_admission() {
        let state = RecordingState::from_env();
        state
            .admit("request-a", "socket-a", RequestKind::ClipList)
            .unwrap();
        assert!(state
            .admit("request-a", "socket-b", RequestKind::PlaybackTicket)
            .is_err());
        let pending = state.take("request-a").unwrap();
        assert_eq!(pending.socket_id, "socket-a");
        assert_eq!(pending.kind, RequestKind::ClipList);
    }

    #[test]
    fn scheduled_command_retains_generation_guard_until_dispatch() {
        let state = RecordingState::from_env();
        let command = RecordingSessionCommand {
            protocol_version: 1,
            request_id: "start-request".into(),
            action: robo_rover_lib::RecordingSessionAction::Start {
                entity_id: "rover-a".into(),
                relative_directory: "scheduled/a".into(),
            },
        };
        state
            .enqueue_scheduled_command(
                command.clone(),
                ScheduledCommandGuard {
                    group_id: "group-a".into(),
                    generation: 7,
                    consumer_id: "scheduled:group-a:7".into(),
                    planned_end_ms: 100,
                },
            )
            .unwrap();
        let queued = state.next_command().unwrap();
        assert_eq!(queued.command, command);
        let guard = queued.scheduled_guard.unwrap();
        assert_eq!(guard.generation, 7);
        assert!(!guard.is_expired_at(99));
        assert!(guard.is_expired_at(100));
    }
}

impl RecordingState {
    pub fn from_env() -> Self {
        let capacity = std::env::var("RECORDING_CONTROL_QUEUE_CAPACITY")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_QUEUE_CAPACITY)
            .clamp(8, MAX_QUEUE_CAPACITY);
        let request_ttl = std::env::var("RECORDING_REQUEST_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map(|seconds| Duration::from_secs(seconds.clamp(1, 60)))
            .unwrap_or(DEFAULT_REQUEST_TTL);
        Self {
            commands: Arc::new(Mutex::new(VecDeque::new())),
            clip_queries: Arc::new(Mutex::new(VecDeque::new())),
            playback_queries: Arc::new(Mutex::new(VecDeque::new())),
            delete_queries: Arc::new(Mutex::new(VecDeque::new())),
            pending: Arc::new(Mutex::new(HashMap::new())),
            statuses: Arc::new(Mutex::new(HashMap::new())),
            active_entities: Arc::new(Mutex::new(HashMap::new())),
            capacity,
            request_ttl,
        }
    }

    pub fn admit(
        &self,
        request_id: &str,
        socket_id: &str,
        kind: RequestKind,
    ) -> Result<(), &'static str> {
        let mut pending = self.pending.lock().map_err(|_| "state unavailable")?;
        if pending.len() >= self.capacity {
            return Err("recording request queue is full");
        }
        if pending.contains_key(request_id) {
            return Err("request_id is already pending");
        }
        pending.insert(
            request_id.into(),
            PendingRequest {
                socket_id: socket_id.into(),
                kind,
                deadline: Instant::now() + self.request_ttl,
            },
        );
        Ok(())
    }

    pub fn take(&self, request_id: &str) -> Option<PendingRequest> {
        self.pending.lock().ok()?.remove(request_id)
    }

    /// Commands from both the browser and the scheduler use one bounded recorder
    /// queue so scheduled demand can be rolled back before an unbounded backlog forms.
    pub fn enqueue_command(&self, command: RecordingSessionCommand) -> Result<(), &'static str> {
        let mut queue = self
            .commands
            .lock()
            .map_err(|_| "recording command queue unavailable")?;
        if queue.len() >= self.capacity {
            return Err("recording command queue is full");
        }
        queue.push_back(QueuedRecordingCommand {
            command,
            scheduled_guard: None,
        });
        Ok(())
    }

    pub fn enqueue_scheduled_command(
        &self,
        command: RecordingSessionCommand,
        scheduled_guard: ScheduledCommandGuard,
    ) -> Result<(), &'static str> {
        let mut queue = self
            .commands
            .lock()
            .map_err(|_| "recording command queue unavailable")?;
        if queue.len() >= self.capacity {
            return Err("recording command queue is full");
        }
        queue.push_back(QueuedRecordingCommand {
            command,
            scheduled_guard: Some(scheduled_guard),
        });
        Ok(())
    }

    /// A non-mutating final recorder admission probe used immediately before a
    /// scheduled acquire is enqueued. The later enqueue repeats this check
    /// under the same mutex, so a concurrent producer cannot bypass it.
    pub fn scheduled_admission_available(&self) -> Result<(), &'static str> {
        let queue = self
            .commands
            .lock()
            .map_err(|_| "recording command queue unavailable")?;
        if queue.len() >= self.capacity {
            return Err("recording command queue is full");
        }
        Ok(())
    }

    pub fn next_command(&self) -> Option<QueuedRecordingCommand> {
        self.commands.lock().ok()?.pop_front()
    }

    pub fn expire(&self) -> Vec<(String, PendingRequest)> {
        let now = Instant::now();
        let Ok(mut pending) = self.pending.lock() else {
            return Vec::new();
        };
        let expired_ids: Vec<String> = pending
            .iter()
            .filter(|(_, request)| request.deadline <= now)
            .map(|(request_id, _)| request_id.clone())
            .collect();
        expired_ids
            .into_iter()
            .filter_map(|request_id| {
                pending
                    .remove(&request_id)
                    .map(|request| (request_id, request))
            })
            .collect()
    }

    pub fn active_entity(&self, recording_id: &str) -> Option<String> {
        self.active_entities.lock().ok()?.get(recording_id).cloned()
    }

    pub fn remember_active(&self, recording_id: &str, entity_id: &str) {
        if let Ok(mut active) = self.active_entities.lock() {
            active.insert(recording_id.into(), entity_id.into());
        }
    }

    pub fn forget_active(&self, recording_id: &str) {
        if let Ok(mut active) = self.active_entities.lock() {
            active.remove(recording_id);
        }
    }

    pub fn cache_status(&self, status: RecordingSessionStatus) -> bool {
        if let Ok(mut statuses) = self.statuses.lock() {
            let changed = statuses
                .get(&status.recording_id)
                .is_none_or(|previous| previous != &status);
            statuses.insert(status.recording_id.clone(), status);
            changed
        } else {
            false
        }
    }

    pub fn status_snapshot(&self) -> Vec<RecordingSessionStatus> {
        self.statuses
            .lock()
            .map(|statuses| statuses.values().cloned().collect())
            .unwrap_or_default()
    }
}
