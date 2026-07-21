use crate::config::SttConfig;
use crate::decoder::SharedNode;
use crate::native::{self, NativeModels};
use crate::runtime::{lifecycle_transition, send_lifecycle_status};
use crate::status::emit_status;
use dora_node_api::{dora_core::config::DataId, Event, EventStream, TryRecvError};
use eyre::{eyre, Result};
use robo_rover_lib::{
    LifecycleComponentState, LifecycleGate, LifecycleReasonCode, LifecycleTransition, SttStatus,
};
use std::sync::mpsc;
use std::time::Duration;

pub(crate) type Initialization = Result<(SttConfig, NativeModels)>;

pub(crate) fn wait_for_initialization(
    node: &SharedNode,
    events: &mut EventStream,
    loading: &SttStatus,
    lifecycle_gate: &mut LifecycleGate,
    lifecycle_status_output: &DataId,
) -> Result<Option<Initialization>> {
    let (sender, receiver) = mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("sherpa-model-loader".into())
        .spawn(move || {
            let result = SttConfig::from_env()
                .map_err(eyre::Report::from)
                .and_then(|config| native::load_models(&config).map(|models| (config, models)));
            let _ = sender.send(result);
        })?;
    let mut rejected_audio = 0u64;
    loop {
        match receiver.try_recv() {
            Ok(result) => return Ok(Some(result)),
            Err(mpsc::TryRecvError::Disconnected) => {
                return Ok(Some(Err(eyre!("speech model loader stopped unexpectedly"))));
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
        match events.try_recv() {
            Ok(Event::Input { id, .. }) if id.as_str() == "stt_status_request" => {
                emit_status(node, loading)?;
            }
            Ok(Event::Input { id, data, .. }) if id.as_str() == "lifecycle_command" => {
                match lifecycle_transition(data.as_ref(), lifecycle_gate) {
                    Ok(Some(transition)) => {
                        // Preserve lifecycle authority while native models load. The loader may
                        // not be safely interrupted, but its result is dropped rather than
                        // reported ready when a newer quiesce is pending.
                        let state = match transition {
                            LifecycleTransition::Quiesce => LifecycleComponentState::Quiescing,
                            LifecycleTransition::Resume => LifecycleComponentState::Resuming,
                        };
                        send_lifecycle_status(
                            node,
                            lifecycle_status_output,
                            lifecycle_gate,
                            state,
                            (transition == LifecycleTransition::Quiesce)
                                .then_some(LifecycleReasonCode::InterruptedByLifecycle),
                        )?;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        tracing::warn!(%error, "rejected central STT lifecycle command while loading")
                    }
                }
            }
            Ok(Event::Input { id, .. }) => {
                rejected_audio += 1;
                tracing::warn!(input = %id, rejected_audio, "speech input rejected while STT is loading");
            }
            Ok(Event::Stop(_)) | Err(TryRecvError::Closed) => return Ok(None),
            Err(TryRecvError::Empty) => std::thread::sleep(Duration::from_millis(10)),
            Ok(_) => {}
        }
    }
}
