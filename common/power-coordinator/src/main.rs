use dora_node_api::{
    arrow::array::{Array, BinaryArray},
    dora_core::config::DataId,
    DoraNode, Event,
};
use eyre::Result;
use power_coordinator::{
    CoordinatorConfig, CoordinatorTime, DurablePowerCoordinator, JournalAcknowledgement,
};
use robo_rover_lib::{
    init_tracing, LifecycleCommandResult, LifecycleStatus, PowerCommand, RecordingOccurrence,
    ResourceSnapshot,
};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn main() -> Result<()> {
    let _guard = init_tracing();
    let config = CoordinatorConfig::from_env().map_err(eyre::Report::msg)?;
    let started = Instant::now();
    let mut coordinator =
        DurablePowerCoordinator::open(config, wall_ms()).map_err(eyre::Report::msg)?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    while let Some(event) = events.recv() {
        let poll_lifecycle = matches!(&event, Event::Input { id, .. } if id.as_str() == "tick");
        let now = CoordinatorTime {
            wall_ms: wall_ms(),
            monotonic_ms: started.elapsed().as_millis() as u64,
        };
        match event {
            Event::Input { id, data, .. } if id.as_str() == "resource_snapshot" => {
                if let Some(bytes) = binary(&data) {
                    if let Ok(snapshot) = serde_json::from_slice::<ResourceSnapshot>(bytes) {
                        coordinator.observe_resources(snapshot);
                    }
                }
            }
            Event::Input { id, data, .. } if id.as_str() == "lifecycle_status" => {
                if let Some(bytes) = binary(&data) {
                    if let Ok(status) = serde_json::from_slice::<LifecycleStatus>(bytes) {
                        coordinator.observe_lifecycle(status);
                    }
                }
            }
            Event::Input { id, data, .. } if id.as_str() == "lifecycle_command_result" => {
                if let Some(bytes) = binary(&data) {
                    if let Ok(result) = serde_json::from_slice::<LifecycleCommandResult>(bytes) {
                        coordinator.observe_lifecycle_result(result);
                    }
                }
            }
            Event::Input { id, data, .. } if id.as_str() == "power_command" => {
                if let Some(bytes) = binary(&data) {
                    if let Ok(command) = serde_json::from_slice::<PowerCommand>(bytes) {
                        send(
                            &mut node,
                            "power_command_result",
                            &coordinator.apply_command(command, now),
                        )?;
                    }
                }
            }
            Event::Input { id, data, .. } if id.as_str() == "recording_occurrence_status" => {
                if let Some(bytes) = binary(&data) {
                    if let Ok(occurrence) = serde_json::from_slice::<RecordingOccurrence>(bytes) {
                        coordinator.observe_protected_occurrence(occurrence);
                    }
                }
            }
            Event::Input { id, data, .. } if id.as_str() == "protected_work_snapshot" => {
                if let Some(bytes) = binary(&data) {
                    if let Ok(snapshot) = serde_json::from_slice(bytes) {
                        coordinator.observe_protected_work_snapshot(snapshot);
                    }
                }
            }
            Event::Input { id, data, .. } if id.as_str() == "power_event_ack" => {
                if let Some(bytes) = binary(&data) {
                    if let Ok(ack) = serde_json::from_slice::<JournalAcknowledgement>(bytes) {
                        coordinator
                            .acknowledge(&ack.event_id)
                            .map_err(eyre::Report::msg)?;
                        coordinator.compact().map_err(eyre::Report::msg)?;
                    }
                }
            }
            Event::Stop { .. } => break,
            _ => {}
        }
        let effects = coordinator.tick(now).map_err(eyre::Report::msg)?;
        send(&mut node, "power_status", &effects.status)?;
        if poll_lifecycle {
            send(&mut node, "lifecycle_status_query", &serde_json::json!({}))?;
        }
        if let Some(transition) = effects.transition {
            send(&mut node, "power_transition", &transition)?;
        }
        for record in coordinator.pending_records() {
            send(&mut node, "power_journal_record", &record)?;
        }
        send(
            &mut node,
            "power_journal_health",
            &coordinator.journal_health(),
        )?;
        for command in effects.lifecycle_commands {
            send(&mut node, "lifecycle_command", &command)?;
        }
    }
    Ok(())
}

fn wall_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
fn binary(data: &dora_node_api::arrow::array::ArrayRef) -> Option<&[u8]> {
    data.as_any()
        .downcast_ref::<BinaryArray>()
        .and_then(|array| (!array.is_empty()).then(|| array.value(0)))
}
fn send<T: serde::Serialize>(node: &mut DoraNode, id: &str, value: &T) -> Result<()> {
    let payload = serde_json::to_vec(value)?;
    node.send_output(
        DataId::from(id.to_owned()),
        Default::default(),
        BinaryArray::from_vec(vec![payload.as_slice()]),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use robo_rover_lib::{occurrence_requires_protection, RecordingOccurrenceState};

    #[test]
    fn only_live_recording_states_hold_the_protected_work_gate() {
        assert!(!occurrence_requires_protection(
            RecordingOccurrenceState::Planned
        ));
        assert!(occurrence_requires_protection(
            RecordingOccurrenceState::StartPending
        ));
        assert!(occurrence_requires_protection(
            RecordingOccurrenceState::Active
        ));
        assert!(occurrence_requires_protection(
            RecordingOccurrenceState::StopPending
        ));
        assert!(!occurrence_requires_protection(
            RecordingOccurrenceState::Completed
        ));
    }
}
