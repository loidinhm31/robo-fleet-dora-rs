use dora_node_api::{
    arrow::array::{Array, BinaryArray},
    dora_core::config::DataId,
    DoraNode, Event,
};
use eyre::Result;
use power_coordinator::{JournalAcknowledgement, JournalRecord};
use power_event_projector::{config::ProjectorConfig, projector::PowerEventProjector};

fn main() -> Result<()> {
    let config = ProjectorConfig::from_env().map_err(eyre::Report::msg)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let mut projector = runtime.block_on(open_projector(&config));
    if let Err(health) = &projector {
        send(&mut node, "power_projector_health", health)?;
    }
    while let Some(event) = events.recv() {
        if let Event::Input { id, data, .. } = event {
            if matches!(
                id.as_str(),
                "power_journal_record" | "remote_power_journal_record"
            ) {
                if let Some(bytes) = binary(&data) {
                    if let Ok(record) = serde_json::from_slice::<JournalRecord>(bytes) {
                        if projector.is_err() {
                            projector = runtime.block_on(open_projector(&config));
                        }
                        let Ok(active_projector) = projector.as_ref() else {
                            send(
                                &mut node,
                                "power_projector_health",
                                projector.as_ref().err().expect("projector retry error"),
                            )?;
                            continue;
                        };
                        let health = runtime.block_on(active_projector.project_with_retry(
                            &record,
                            config.retry_attempts,
                            config.retry_backoff_ms,
                        ));
                        send(&mut node, "power_projector_health", &health)?;
                        if health.healthy {
                            send(
                                &mut node,
                                if id.as_str() == "power_journal_record" {
                                    "power_event_ack"
                                } else {
                                    "remote_power_event_ack"
                                },
                                &JournalAcknowledgement {
                                    protocol_version: robo_rover_lib::POWER_PROTOCOL_VERSION,
                                    event_id: record.event.event_id,
                                    deployment_id: config.deployment_id.clone(),
                                },
                            )?;
                        } else {
                            projector = Err(health);
                        }
                    }
                }
            }
        } else {
            break;
        }
    }
    Ok(())
}

async fn open_projector(
    config: &ProjectorConfig,
) -> Result<PowerEventProjector, power_event_projector::projector::ProjectionHealth> {
    PowerEventProjector::open_with_retry(
        config.deployment_id.clone(),
        &config.mongodb_uri,
        &config.mongodb_database,
        config.retry_attempts,
        config.retry_backoff_ms,
    )
    .await
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
