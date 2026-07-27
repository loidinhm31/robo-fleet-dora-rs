use robo_rover_lib::PowerCommand;
use serde::{Deserialize, Serialize};

/// Durable scheduler-owned intent to deliver one exact reservation command.
/// A result is correlated only by this command id; aggregate PowerStatus never
/// acknowledges a row in this outbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReservationCommandAction {
    Register,
    Release,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReservationCommandOutboxRecord {
    pub group_id: String,
    pub reservation_id: String,
    pub action: ReservationCommandAction,
    pub command: PowerCommand,
    pub created_at_ms: i64,
}

impl ReservationCommandOutboxRecord {
    pub fn command_id(&self) -> &str {
        &self.command.command_id
    }

    pub fn is_expired(&self, now_ms: i64) -> bool {
        u64::try_from(now_ms).map_or(true, |now_ms| now_ms >= self.command.expires_at_ms)
    }
}
