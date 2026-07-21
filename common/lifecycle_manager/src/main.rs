use dora_node_api::{
    arrow::array::{Array, BinaryArray},
    dora_core::config::DataId,
    DoraNode, Event,
};
use eyre::Result;
use lifecycle_manager::LifecycleManager;
use robo_rover_lib::{
    init_tracing, LifecycleCapability, LifecycleCommand, LifecycleCommandResult, LifecycleRole,
    LifecycleTarget, LifecycleWakeLease,
};
use std::time::{SystemTime, UNIX_EPOCH};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn main() -> Result<()> {
    let _guard = init_tracing();
    let epoch = now_ms().max(1);
    let mut capabilities = std::env::var("LIFECYCLE_CAPABILITIES")
        .ok()
        .map(|value| serde_json::from_str::<Vec<LifecycleCapability>>(&value))
        .transpose()?
        .unwrap_or_default();
    if let Ok(entities) = std::env::var("LIFECYCLE_REMOTE_ROVER_ENTITIES") {
        for entity_id in entities
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            for node_id in ["gst-camera", "audio-capture"] {
                let capability = LifecycleCapability {
                    target: LifecycleTarget {
                        role: LifecycleRole::Rover,
                        entity_id: entity_id.to_owned(),
                        node_id: node_id.into(),
                    },
                    // Phase 3 enables these only after adapter-backed release.
                    supported: false,
                    always_on: false,
                };
                if !capabilities
                    .iter()
                    .any(|current| current.target == capability.target)
                {
                    capabilities.push(capability);
                }
            }
        }
    }
    let mut manager =
        LifecycleManager::new(epoch, capabilities).map_err(|error| eyre::eyre!(error))?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let result_output = DataId::from("lifecycle_command_result".to_owned());
    let status_output = DataId::from("lifecycle_status".to_owned());
    let capabilities_output = DataId::from("lifecycle_capabilities".to_owned());
    let authorized_command_output = DataId::from("lifecycle_authorized_command".to_owned());
    let authorized_wake_lease_output = DataId::from("lifecycle_authorized_wake_lease".to_owned());
    send(&mut node, &capabilities_output, &manager.capabilities())?;
    while let Some(event) = events.recv() {
        let now = now_ms();
        manager.tick(now);
        match event {
            Event::Input { id, data, .. } if id.as_str() == "lifecycle_command" => {
                if let Some(bytes) = binary(&data) {
                    let parsed = serde_json::from_slice::<LifecycleCommand>(bytes);
                    let target = parsed.as_ref().ok().map(|command| command.target.clone());
                    let result = parsed
                        .map(|command| manager.apply(command, now))
                        .unwrap_or_else(|error| invalid_result(epoch, error.to_string()));
                    send(&mut node, &result_output, &result)?;
                    if let Some(status) = target.and_then(|target| manager.status(&target, now)) {
                        send(&mut node, &status_output, &status)?;
                    }
                    if result.accepted {
                        if let Ok(command) = serde_json::from_slice::<LifecycleCommand>(bytes) {
                            send(&mut node, &authorized_command_output, &command)?;
                        }
                    }
                }
            }
            Event::Input { id, data, .. } if id.as_str() == "lifecycle_command_relay" => {
                if let Some(bytes) = binary(&data) {
                    let parsed = serde_json::from_slice::<LifecycleCommand>(bytes);
                    let target = parsed.as_ref().ok().map(|command| command.target.clone());
                    let result = parsed
                        .map(|command| manager.apply_relayed(command, now))
                        .unwrap_or_else(|error| invalid_result(epoch, error.to_string()));
                    send(&mut node, &result_output, &result)?;
                    if let Some(status) = target.and_then(|target| manager.status(&target, now)) {
                        send(&mut node, &status_output, &status)?;
                    }
                }
            }
            Event::Input { id, data, .. } if id.as_str() == "lifecycle_component_status" => {
                if let Some(bytes) = binary(&data) {
                    if let Ok(status) =
                        serde_json::from_slice::<robo_rover_lib::LifecycleStatus>(bytes)
                    {
                        if manager.apply_component_status(&status) {
                            if let Some(updated) = manager.status(&status.target, now) {
                                send(&mut node, &status_output, &updated)?;
                            }
                        }
                    }
                }
            }
            Event::Input { id, data, .. }
                if id.as_str() == "lifecycle_wake_lease"
                    || id.as_str() == "lifecycle_wake_lease_relay" =>
            {
                if let Some(bytes) = binary(&data) {
                    if let Ok(lease) = serde_json::from_slice::<LifecycleWakeLease>(bytes) {
                        let target = lease.target.clone();
                        if manager.apply_wake_lease(lease.clone(), now).is_ok() {
                            if id.as_str() == "lifecycle_wake_lease" {
                                send(&mut node, &authorized_wake_lease_output, &lease)?;
                            }
                            if let Some(status) = manager.status(&target, now) {
                                send(&mut node, &status_output, &status)?;
                            }
                        }
                    }
                }
            }
            Event::Input { id, .. } if id.as_str() == "lifecycle_status_query" => {
                for status in manager.statuses(now) {
                    send(&mut node, &status_output, &status)?;
                }
                send(&mut node, &capabilities_output, &manager.capabilities())?;
            }
            Event::Input { id, .. } if id.as_str() == "tick" => {
                for status in manager.statuses(now) {
                    send(&mut node, &status_output, &status)?;
                }
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}

fn binary(data: &dora_node_api::ArrowData) -> Option<&[u8]> {
    data.0
        .as_any()
        .downcast_ref::<BinaryArray>()
        .filter(|array| array.len() == 1)
        .map(|array| array.value(0))
}
fn send<T: serde::Serialize>(node: &mut DoraNode, output: &DataId, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    node.send_output(
        output.clone(),
        Default::default(),
        BinaryArray::from_vec(vec![bytes.as_slice()]),
    )?;
    Ok(())
}
fn invalid_result(epoch: u64, detail: String) -> LifecycleCommandResult {
    LifecycleCommandResult {
        protocol_version: 1,
        request_id: "00000000-0000-0000-0000-000000000000".into(),
        accepted: false,
        manager_epoch: epoch,
        revision: 0,
        reason_code: Some(robo_rover_lib::LifecycleReasonCode::InvalidRequest),
        detail: Some(detail),
    }
}
