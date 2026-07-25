use crate::{JournalIntent, JournalRecord};
use robo_rover_lib::{
    PowerCommand, PowerCommandResult, PowerEvent, PowerEventType, PowerReasonCode, PowerStatus,
    POWER_PROTOCOL_VERSION,
};
use uuid::Uuid;

pub fn record(
    intent: JournalIntent,
    event_type: PowerEventType,
    transition_id: Option<String>,
    status: PowerStatus,
    now_ms: u64,
) -> JournalRecord {
    JournalRecord {
        format_version: 1,
        sequence: 0,
        intent,
        event: PowerEvent {
            protocol_version: POWER_PROTOCOL_VERSION,
            event_id: Uuid::new_v4().hyphenated().to_string(),
            role: status.role,
            entity_id: status.entity_id.clone(),
            authority: status.authority,
            transition_id,
            event_type,
            reason_code: status.reason_code,
            detail: None,
            occurred_at_ms: now_ms,
        },
        status: Some(status),
    }
}

pub fn rejected(command: PowerCommand, status: PowerStatus, detail: String) -> PowerCommandResult {
    PowerCommandResult {
        protocol_version: POWER_PROTOCOL_VERSION,
        command_id: command.command_id,
        accepted: false,
        authority: status.authority,
        reason_code: Some(if detail.contains("CapacityExceeded") {
            PowerReasonCode::CapacityExceeded
        } else {
            PowerReasonCode::InvalidRequest
        }),
        detail: Some(detail),
    }
}
