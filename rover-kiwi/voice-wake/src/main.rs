mod config;
mod controller;
mod debounce;
mod kws;
mod wake_ack;

use dora_node_api::{
    arrow::array::{Array, BinaryArray, Float32Array},
    dora_core::config::DataId,
    DoraNode, Event, MetadataParameters, Parameter,
};
use eyre::Result;
use robo_rover_lib::{
    init_tracing, LifecycleCommand, LifecycleComponentState, LifecycleGate, LifecycleReasonCode,
    LifecycleRole, LifecycleTarget, LifecycleTransition, PlaybackState, PowerCommandResult,
    PowerStatus,
};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::{config::KwsConfig, controller::WakeController, kws::KwsEngine, wake_ack::*};

fn main() -> Result<()> {
    let _guard = init_tracing();
    let config = KwsConfig::from_env()?;
    let mut engine = KwsEngine::create(&config)?;
    let entity_id = std::env::var("ENTITY_ID").unwrap_or_else(|_| "rover-kiwi".into());
    let mut controller = WakeController::new(entity_id.clone());
    let mut gate = LifecycleGate::new(LifecycleTarget {
        role: LifecycleRole::Rover,
        entity_id,
        node_id: "voice-wake".into(),
    });
    let (mut node, mut events) = DoraNode::init_from_env()?;
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, .. } if id.as_str() == "audio_frame" => {
                if gate.admission_open() && controller.listens() {
                    if let Some(samples) = data.as_any().downcast_ref::<Float32Array>() {
                        if engine.observe(samples.values().as_ref()) {
                            if let Some(command) = controller.wake_command(now_ms()) {
                                send_json(&mut node, "power_command", &command)?;
                            }
                        }
                    }
                }
            }
            Event::Input { id, data, .. } if id.as_str() == "power_status" => {
                if let Some(status) = parse_json::<PowerStatus>(data.as_ref()) {
                    controller.observe_status(status);
                    send_wake_ack_if_ready(&mut node, &mut controller)?;
                }
            }
            Event::Input { id, data, .. } if id.as_str() == "power_command_result" => {
                if let Some(result) = parse_json::<PowerCommandResult>(data.as_ref()) {
                    controller.observe_result(result);
                    send_wake_ack_if_ready(&mut node, &mut controller)?;
                }
            }
            Event::Input { id, data, .. } if id.as_str() == "playback_state" => {
                if let Some(state) = parse_json::<PlaybackState>(data.as_ref()) {
                    controller.observe_playback(state);
                    send_wake_ack_if_ready(&mut node, &mut controller)?;
                }
            }
            Event::Input { id, data, .. } if id.as_str() == "lifecycle_command" => {
                handle_lifecycle(
                    &mut node,
                    &mut gate,
                    &mut engine,
                    &mut controller,
                    data.as_ref(),
                )?;
            }
            Event::Stop(_) => break,
            _ => {}
        }
    }
    Ok(())
}

fn send_wake_ack_if_ready(node: &mut DoraNode, controller: &mut WakeController) -> Result<()> {
    let Some(demand_id) = controller.ready_wake_ack() else {
        return Ok(());
    };
    let samples = samples();
    let parameters = MetadataParameters::from([
        ("source_kind".into(), Parameter::String("wake_ack".into())),
        ("command_id".into(), Parameter::String(demand_id)),
        (
            "stream_id".into(),
            Parameter::String(Uuid::new_v4().to_string()),
        ),
        ("frame_id".into(), Parameter::Integer(0)),
        (
            "capture_timestamp_ms".into(),
            Parameter::Integer(now_ms().try_into()?),
        ),
        (
            "sample_rate".into(),
            Parameter::Integer(i64::from(WAKE_ACK_SAMPLE_RATE)),
        ),
        ("channels".into(), Parameter::Integer(1)),
        (
            "sample_count".into(),
            Parameter::Integer(samples.len().try_into()?),
        ),
        ("format".into(), Parameter::String("f32le".into())),
        ("priority".into(), Parameter::String("high".into())),
    ]);
    node.send_output(
        DataId::from("wake_ack_audio".to_owned()),
        parameters,
        Float32Array::from(samples),
    )?;
    Ok(())
}

fn handle_lifecycle(
    node: &mut DoraNode,
    gate: &mut LifecycleGate,
    engine: &mut KwsEngine,
    controller: &mut WakeController,
    data: &dyn Array,
) -> Result<()> {
    let Some(command) = parse_json::<LifecycleCommand>(data) else {
        return Ok(());
    };
    let Some(transition) = gate.begin(&command).map_err(eyre::Report::msg)? else {
        return Ok(());
    };
    send_lifecycle(
        node,
        gate,
        match transition {
            LifecycleTransition::Quiesce => LifecycleComponentState::Quiescing,
            LifecycleTransition::Resume => LifecycleComponentState::Resuming,
        },
        None,
    )?;
    if transition == LifecycleTransition::Quiesce {
        engine.reset();
        controller.reset();
    }
    gate.complete(transition);
    send_lifecycle(node, gate, LifecycleComponentState::Running, None)
}

fn send_lifecycle(
    node: &mut DoraNode,
    gate: &LifecycleGate,
    state: LifecycleComponentState,
    reason: Option<LifecycleReasonCode>,
) -> Result<()> {
    send_json(
        node,
        "lifecycle_component_status",
        &gate.status(state, reason, now_ms()),
    )
}

fn parse_json<T: serde::de::DeserializeOwned>(data: &dyn Array) -> Option<T> {
    data.as_any()
        .downcast_ref::<BinaryArray>()
        .filter(|array| array.len() == 1)
        .and_then(|array| serde_json::from_slice(array.value(0)).ok())
}

fn send_json<T: serde::Serialize>(node: &mut DoraNode, id: &str, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    node.send_output(
        DataId::from(id.to_owned()),
        MetadataParameters::default(),
        BinaryArray::from_vec(vec![bytes.as_slice()]),
    )?;
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
