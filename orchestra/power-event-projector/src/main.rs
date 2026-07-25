use dora_node_api::{
    arrow::array::{Array, BinaryArray},
    dora_core::config::DataId,
    DoraNode, Event,
};
use eyre::Result;
use power_coordinator::{JournalAcknowledgement, JournalRecord};
use power_event_projector::{
    config::ProjectorConfig, mongo_repository::MongoRepository, projector::PowerEventProjector,
};

fn main() -> Result<()> {
    let config = ProjectorConfig::from_env().map_err(eyre::Report::msg)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let repository = runtime.block_on(MongoRepository::connect(
        &config.mongodb_uri,
        &config.mongodb_database,
    ))?;
    let projector = PowerEventProjector::new(config.deployment_id, repository);
    runtime
        .block_on(projector.initialize())
        .map_err(eyre::Report::msg)?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    while let Some(event) = events.recv() {
        if let Event::Input { id, data, .. } = event {
            if id.as_str() == "power_journal_record" {
                if let Some(bytes) = binary(&data) {
                    if let Ok(record) = serde_json::from_slice::<JournalRecord>(bytes) {
                        runtime
                            .block_on(projector.project(&record))
                            .map_err(eyre::Report::msg)?;
                        send(
                            &mut node,
                            "power_event_ack",
                            &JournalAcknowledgement {
                                event_id: record.event.event_id,
                            },
                        )?;
                    }
                }
            }
        } else {
            break;
        }
    }
    Ok(())
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
