use robo_rover_lib::{
    LifecycleRole, PowerAuthority, PowerAuthoritySnapshot, PowerCommand, PowerCommandAction,
    PowerDemand, PowerDemandAction, PowerDemandPriority, PowerDemandSource, PowerPolicy,
    PowerProfile, PowerStatus, POWER_PROTOCOL_VERSION,
};
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

const POWER_COMMAND_TTL_MS: u64 = 30_000;
const UI_DEMAND_TTL_MS: u64 = 120_000;
const MAX_PENDING: usize = 128;

#[derive(Clone, PartialEq, Eq)]
enum RequestFingerprint {
    Policy(PowerPolicy),
    Wake,
}

#[derive(Clone)]
struct RequestAdmission {
    fingerprint: RequestFingerprint,
    expires_at_ms: u64,
}

#[derive(Clone)]
pub enum PendingPowerKind {
    Policy,
    WakePolicy {
        demand_id: String,
        action: PowerDemandAction,
        renew_sequence: u64,
    },
    WakeDemand {
        demand_id: String,
        renew_sequence: u64,
    },
    Release,
}

#[derive(Clone)]
pub struct PendingPowerRequest {
    pub socket_id: String,
    pub request_id: String,
    pub entity_id: String,
    pub kind: PendingPowerKind,
    pub expires_at_ms: u64,
}

#[derive(Clone)]
struct WaitingWake {
    socket_id: String,
    request_id: String,
    demand_id: String,
    action: PowerDemandAction,
    renew_sequence: u64,
    entity_id: String,
    after: PowerAuthority,
    expires_at_ms: u64,
}

#[derive(Clone)]
struct UiDemandLease {
    socket_id: String,
    entity_id: String,
    demand_id: String,
    authority: PowerAuthority,
    renew_sequence: u64,
    expires_at_ms: u64,
}

#[derive(Clone, Default)]
pub struct PowerSocketState {
    commands: Arc<Mutex<VecDeque<PowerCommand>>>,
    pending: Arc<Mutex<BTreeMap<String, PendingPowerRequest>>>,
    statuses: Arc<Mutex<BTreeMap<String, PowerStatus>>>,
    snapshots: Arc<Mutex<BTreeMap<String, PowerAuthoritySnapshot>>>,
    waiting_wakes: Arc<Mutex<Vec<WaitingWake>>>,
    ui_demands: Arc<Mutex<Vec<UiDemandLease>>>,
    request_admissions: Arc<Mutex<BTreeMap<String, RequestAdmission>>>,
}

impl PowerSocketState {
    pub fn next_command(&self) -> Option<PowerCommand> {
        self.commands.lock().ok()?.pop_front()
    }

    pub fn status(&self, entity_id: &str) -> Option<PowerStatus> {
        self.statuses.lock().ok()?.get(entity_id).cloned()
    }

    pub fn cache_status(&self, status: PowerStatus) -> bool {
        if status.validate().is_err() {
            return false;
        }
        let Ok(mut values) = self.statuses.lock() else {
            return false;
        };
        let replace = values.get(&status.entity_id).map_or(true, |current| {
            status.authority > current.authority
                || (status.authority == current.authority
                    && status.updated_at_ms >= current.updated_at_ms)
        });
        if replace {
            values.insert(status.entity_id.clone(), status);
        }
        replace
    }

    pub fn observe_snapshot(&self, snapshot: PowerAuthoritySnapshot, now_ms: u64) {
        if snapshot.validate().is_err()
            || snapshot.expires_at_ms <= now_ms
            || snapshot.captured_at_ms > now_ms
            || snapshot.role != LifecycleRole::Rover
        {
            return;
        }
        if let Ok(mut values) = self.snapshots.lock() {
            if values
                .get(&snapshot.entity_id)
                .map_or(true, |old| snapshot.authority >= old.authority)
            {
                values.insert(snapshot.entity_id.clone(), snapshot.clone());
            }
        }
        self.advance_waiting_wakes(&snapshot, now_ms);
        self.release_expired(now_ms, &snapshot);
    }

    pub fn queue_policy(
        &self,
        socket_id: String,
        request_id: String,
        entity_id: String,
        policy: PowerPolicy,
        now_ms: u64,
    ) -> Result<(), String> {
        let command = self.command_for_snapshot(
            &entity_id,
            now_ms,
            PowerCommandAction::SetPolicy { policy },
        )?;
        self.register(
            command,
            PendingPowerRequest {
                socket_id,
                request_id,
                entity_id,
                kind: PendingPowerKind::Policy,
                expires_at_ms: now_ms + POWER_COMMAND_TTL_MS,
            },
            Some(RequestFingerprint::Policy(policy)),
        )
    }

    pub fn queue_wake(
        &self,
        socket_id: String,
        request_id: String,
        entity_id: String,
        now_ms: u64,
    ) -> Result<(), String> {
        if self.wake_is_in_flight(&socket_id, &entity_id) {
            return Err("wake request already pending".into());
        }
        let (demand_id, action, renew_sequence) = self
            .ui_demands
            .lock()
            .ok()
            .and_then(|values| {
                values
                    .iter()
                    .find(|value| value.socket_id == socket_id && value.entity_id == entity_id)
                    .map(|value| {
                        (
                            value.demand_id.clone(),
                            PowerDemandAction::Renew,
                            value.renew_sequence.saturating_add(1),
                        )
                    })
            })
            .unwrap_or_else(|| {
                (
                    Uuid::new_v4().hyphenated().to_string(),
                    PowerDemandAction::Acquire,
                    1,
                )
            });
        let command = self.command_for_snapshot(
            &entity_id,
            now_ms,
            PowerCommandAction::SetPolicy {
                policy: PowerPolicy::Auto,
            },
        )?;
        self.register(
            command,
            PendingPowerRequest {
                socket_id,
                request_id,
                entity_id,
                kind: PendingPowerKind::WakePolicy {
                    demand_id,
                    action,
                    renew_sequence,
                },
                expires_at_ms: now_ms + POWER_COMMAND_TTL_MS,
            },
            Some(RequestFingerprint::Wake),
        )
    }

    pub fn take_pending(&self, command_id: &str) -> Option<PendingPowerRequest> {
        self.pending.lock().ok()?.remove(command_id)
    }

    pub fn accept_wake_policy(
        &self,
        pending: PendingPowerRequest,
        authority: PowerAuthority,
        now_ms: u64,
    ) {
        let PendingPowerKind::WakePolicy {
            demand_id,
            action,
            renew_sequence,
        } = pending.kind
        else {
            return;
        };
        if let Ok(mut values) = self.waiting_wakes.lock() {
            values.push(WaitingWake {
                socket_id: pending.socket_id,
                request_id: pending.request_id,
                demand_id,
                action,
                renew_sequence,
                entity_id: pending.entity_id,
                after: authority,
                expires_at_ms: now_ms + POWER_COMMAND_TTL_MS,
            });
        }
    }

    pub fn complete_wake_demand(
        &self,
        pending: &PendingPowerRequest,
        authority: PowerAuthority,
        now_ms: u64,
    ) {
        let PendingPowerKind::WakeDemand {
            demand_id,
            renew_sequence,
        } = &pending.kind
        else {
            return;
        };
        if let Ok(mut values) = self.ui_demands.lock() {
            values.retain(|value| {
                value.socket_id != pending.socket_id || value.entity_id != pending.entity_id
            });
            values.push(UiDemandLease {
                socket_id: pending.socket_id.clone(),
                entity_id: pending.entity_id.clone(),
                demand_id: demand_id.clone(),
                authority,
                renew_sequence: *renew_sequence,
                expires_at_ms: now_ms + UI_DEMAND_TTL_MS,
            });
        }
    }

    pub fn release_socket(&self, socket_id: &str, now_ms: u64) {
        if let Ok(mut pending) = self.pending.lock() {
            pending.retain(|_, value| value.socket_id != socket_id);
        }
        if let Ok(mut admissions) = self.request_admissions.lock() {
            admissions.retain(|key, _| !key.starts_with(&format!("{socket_id}:")));
        }
        if let Ok(mut values) = self.waiting_wakes.lock() {
            values.retain(|value| value.socket_id != socket_id);
        }
        if let Ok(mut leases) = self.ui_demands.lock() {
            for lease in leases
                .iter_mut()
                .filter(|lease| lease.socket_id == socket_id)
            {
                lease.expires_at_ms = now_ms;
            }
        }
        let snapshots = self
            .snapshots
            .lock()
            .map(|values| values.values().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        for snapshot in snapshots {
            self.release_expired(now_ms, &snapshot);
        }
    }

    pub fn sweep(&self, now_ms: u64) {
        if let Ok(mut pending) = self.pending.lock() {
            pending.retain(|_, value| value.expires_at_ms > now_ms);
        }
        if let Ok(mut values) = self.waiting_wakes.lock() {
            values.retain(|value| value.expires_at_ms > now_ms);
        }
        if let Ok(mut admissions) = self.request_admissions.lock() {
            admissions.retain(|_, value| value.expires_at_ms > now_ms);
        }
        let snapshots = self
            .snapshots
            .lock()
            .map(|values| values.values().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        for snapshot in snapshots {
            self.release_expired(now_ms, &snapshot);
        }
    }

    fn advance_waiting_wakes(&self, snapshot: &PowerAuthoritySnapshot, now_ms: u64) {
        let waiting = self
            .waiting_wakes
            .lock()
            .map(|mut values| {
                let mut ready = Vec::new();
                values.retain(|value| {
                    let matches = value.entity_id == snapshot.entity_id
                        && snapshot.authority > value.after
                        && value.expires_at_ms > now_ms;
                    if matches {
                        ready.push(value.clone());
                    }
                    !matches
                });
                ready
            })
            .unwrap_or_default();
        for wake in waiting {
            let command = command(
                snapshot,
                now_ms,
                PowerCommandAction::RegisterDemand {
                    demand: PowerDemand {
                        protocol_version: POWER_PROTOCOL_VERSION,
                        demand_id: wake.demand_id.clone(),
                        action: wake.action,
                        source: PowerDemandSource::Ui,
                        priority: PowerDemandPriority::Normal,
                        role: snapshot.role,
                        entity_id: snapshot.entity_id.clone(),
                        required_profile: PowerProfile::NormalRover,
                        authority: next_authority(snapshot.authority),
                        issued_at_ms: now_ms,
                        not_before_ms: now_ms,
                        expires_at_ms: now_ms + UI_DEMAND_TTL_MS,
                        renew_sequence: wake.renew_sequence,
                    },
                },
            );
            let _ = self.register(
                command,
                PendingPowerRequest {
                    socket_id: wake.socket_id,
                    request_id: wake.request_id,
                    entity_id: wake.entity_id,
                    kind: PendingPowerKind::WakeDemand {
                        demand_id: wake.demand_id,
                        renew_sequence: wake.renew_sequence,
                    },
                    expires_at_ms: now_ms + POWER_COMMAND_TTL_MS,
                },
                None,
            );
        }
    }

    fn release_expired(&self, now_ms: u64, snapshot: &PowerAuthoritySnapshot) {
        if snapshot.expires_at_ms <= now_ms {
            return;
        }
        let ready = self
            .ui_demands
            .lock()
            .map(|values| {
                values
                    .iter()
                    .filter(|lease| {
                        lease.entity_id == snapshot.entity_id
                            && lease.expires_at_ms <= now_ms
                            && snapshot.authority >= lease.authority
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for lease in ready {
            let demand_id = lease.demand_id.clone();
            let socket_id = lease.socket_id.clone();
            let command = command(
                snapshot,
                now_ms,
                PowerCommandAction::ReleaseDemand {
                    demand_id: demand_id.clone(),
                },
            );
            if self
                .register(
                    command,
                    PendingPowerRequest {
                        socket_id: socket_id.clone(),
                        request_id: Uuid::new_v4().hyphenated().to_string(),
                        entity_id: lease.entity_id,
                        kind: PendingPowerKind::Release,
                        expires_at_ms: now_ms + POWER_COMMAND_TTL_MS,
                    },
                    None,
                )
                .is_ok()
            {
                if let Ok(mut values) = self.ui_demands.lock() {
                    values.retain(|value| {
                        value.demand_id != demand_id || value.socket_id != socket_id
                    });
                }
            }
        }
    }

    fn command_for_snapshot(
        &self,
        entity_id: &str,
        now_ms: u64,
        action: PowerCommandAction,
    ) -> Result<PowerCommand, String> {
        let snapshot = self
            .snapshots
            .lock()
            .ok()
            .and_then(|values| values.get(entity_id).cloned())
            .ok_or("authority snapshot is unavailable")?;
        if snapshot.expires_at_ms <= now_ms
            || snapshot.captured_at_ms > now_ms
            || snapshot.role != LifecycleRole::Rover
        {
            return Err("authority snapshot is stale".into());
        }
        Ok(command(&snapshot, now_ms, action))
    }

    fn wake_is_in_flight(&self, socket_id: &str, entity_id: &str) -> bool {
        self.pending
            .lock()
            .map(|values| {
                values.values().any(|value| {
                    value.socket_id == socket_id
                        && value.entity_id == entity_id
                        && matches!(
                            value.kind,
                            PendingPowerKind::WakePolicy { .. }
                                | PendingPowerKind::WakeDemand { .. }
                        )
                })
            })
            .unwrap_or(true)
            || self
                .waiting_wakes
                .lock()
                .map(|values| {
                    values
                        .iter()
                        .any(|value| value.socket_id == socket_id && value.entity_id == entity_id)
                })
                .unwrap_or(true)
    }

    fn register(
        &self,
        command: PowerCommand,
        pending: PendingPowerRequest,
        fingerprint: Option<RequestFingerprint>,
    ) -> Result<(), String> {
        let admission_key = fingerprint
            .as_ref()
            .map(|_| format!("{}:{}", pending.socket_id, pending.request_id));
        if let (Some(key), Some(fingerprint)) = (&admission_key, &fingerprint) {
            let Ok(mut admissions) = self.request_admissions.lock() else {
                return Err("power admission unavailable".into());
            };
            admissions.retain(|_, value| {
                value.expires_at_ms > pending.expires_at_ms.saturating_sub(POWER_COMMAND_TTL_MS)
            });
            if let Some(existing) = admissions.get(key) {
                return Err(if existing.fingerprint == *fingerprint {
                    "duplicate power request"
                } else {
                    "duplicate request payload mismatch"
                }
                .into());
            }
            admissions.insert(
                key.clone(),
                RequestAdmission {
                    fingerprint: fingerprint.clone(),
                    expires_at_ms: pending.expires_at_ms,
                },
            );
        }
        let Ok(mut values) = self.pending.lock() else {
            if let Some(key) = admission_key {
                self.request_admissions
                    .lock()
                    .ok()
                    .map(|mut admissions| admissions.remove(&key));
            }
            return Err("power admission unavailable".into());
        };
        if values.len() >= MAX_PENDING {
            if let Some(key) = admission_key {
                self.request_admissions
                    .lock()
                    .ok()
                    .map(|mut admissions| admissions.remove(&key));
            }
            return Err("power pending capacity reached".into());
        }
        let mut commands = self
            .commands
            .lock()
            .map_err(|_| "power command queue unavailable".to_string())
            .map_err(|error| {
                if let Some(key) = &admission_key {
                    self.request_admissions
                        .lock()
                        .ok()
                        .map(|mut admissions| admissions.remove(key));
                }
                error
            })?;
        values.insert(command.command_id.clone(), pending);
        commands.push_back(command);
        Ok(())
    }
}

fn command(
    snapshot: &PowerAuthoritySnapshot,
    now_ms: u64,
    action: PowerCommandAction,
) -> PowerCommand {
    PowerCommand {
        protocol_version: POWER_PROTOCOL_VERSION,
        command_id: Uuid::new_v4().hyphenated().to_string(),
        role: snapshot.role,
        entity_id: snapshot.entity_id.clone(),
        authority: next_authority(snapshot.authority),
        action,
        issued_at_ms: now_ms,
        not_before_ms: now_ms,
        expires_at_ms: now_ms + POWER_COMMAND_TTL_MS,
        detail: None,
    }
}

fn next_authority(authority: PowerAuthority) -> PowerAuthority {
    PowerAuthority {
        epoch: authority.epoch.saturating_add(1),
        sequence: 1,
    }
}
