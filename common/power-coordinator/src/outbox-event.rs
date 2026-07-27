use crate::{JournalIntent, JournalRecord};
use robo_rover_lib::{
    LifecycleCommand, PowerCommand, PowerCommandAction, PowerCommandActionKind, PowerCommandResult,
    PowerEvent, PowerEventContext, PowerEventType, PowerLifecycleTargetContext, PowerReasonCode,
    PowerStatus, POWER_PROTOCOL_VERSION,
};
use uuid::Uuid;

pub fn record(
    intent: JournalIntent,
    event_type: PowerEventType,
    transition_id: Option<String>,
    status: PowerStatus,
    now_ms: u64,
    context: PowerEventContext,
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
            context,
            occurred_at_ms: now_ms,
        },
        status: Some(status),
    }
}

pub fn command_context(command: &PowerCommand) -> PowerEventContext {
    let mut context = PowerEventContext {
        command_id: Some(command.command_id.clone()),
        command_action: Some(match &command.action {
            PowerCommandAction::SetPolicy { .. } => PowerCommandActionKind::SetPolicy,
            PowerCommandAction::RegisterDemand { .. } => PowerCommandActionKind::RegisterDemand,
            PowerCommandAction::ReleaseDemand { .. } => PowerCommandActionKind::ReleaseDemand,
            PowerCommandAction::RegisterReservation { .. } => {
                PowerCommandActionKind::RegisterReservation
            }
            PowerCommandAction::ReleaseReservation { .. } => {
                PowerCommandActionKind::ReleaseReservation
            }
        }),
        ..Default::default()
    };
    match &command.action {
        PowerCommandAction::SetPolicy { policy } => context.policy = Some(*policy),
        PowerCommandAction::RegisterDemand { demand } => {
            context.demand_id = Some(demand.demand_id.clone());
            context.demand_source = Some(demand.source);
            context.required_profile = Some(demand.required_profile);
            context.demand_priority = Some(demand.priority);
            context.not_before_ms = Some(demand.not_before_ms);
            context.expires_at_ms = Some(demand.expires_at_ms);
        }
        PowerCommandAction::ReleaseDemand { demand_id } => {
            context.demand_id = Some(demand_id.clone())
        }
        PowerCommandAction::RegisterReservation { reservation } => {
            context.reservation_id = Some(reservation.reservation_id.clone());
            context.demand_source = Some(robo_rover_lib::PowerDemandSource::Scheduler);
            context.required_profile = Some(reservation.required_profile);
            context.not_before_ms = Some(reservation.not_before_ms);
            context.expires_at_ms = Some(reservation.expires_at_ms);
        }
        PowerCommandAction::ReleaseReservation { reservation_id } => {
            context.reservation_id = Some(reservation_id.clone());
        }
    }
    context
}

pub fn lifecycle_context(commands: &[LifecycleCommand]) -> PowerEventContext {
    PowerEventContext {
        lifecycle_targets: commands
            .iter()
            .map(|command| PowerLifecycleTargetContext {
                node_id: command.target.node_id.clone(),
                manager_epoch: command.manager_epoch,
                expected_revision: command.expected_revision,
            })
            .collect(),
        ..Default::default()
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
