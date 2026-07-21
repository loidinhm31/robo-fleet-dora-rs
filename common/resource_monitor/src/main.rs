mod cgroup;
mod config;
mod process_resolver;
mod resource_sampler;

use dora_node_api::{arrow::array::BinaryArray, dora_core::config::DataId, DoraNode, Event};
use eyre::Result;
use robo_rover_lib::{init_tracing, ResourceSnapshot, RESOURCE_SCHEMA_VERSION};

use config::MonitorConfig;
use process_resolver::resolve_nodes;
use resource_sampler::ResourceSampler;

fn main() -> Result<()> {
    let _guard = init_tracing();
    let config = MonitorConfig::from_env().map_err(eyre::Report::msg)?;
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let mut sampler = ResourceSampler::new();
    let mut sequence = 0_u64;

    tracing::info!(role = ?config.role, entity_id = %config.entity_id, nodes = config.nodes.len(), "resource monitor started");
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, .. } if id.as_str() == "tick" => {
                sequence = sequence.saturating_add(1);
                let sampled_at_ms = chrono::Utc::now().timestamp_millis();
                let usage = sampler.sample();
                let snapshot = ResourceSnapshot {
                    schema_version: RESOURCE_SCHEMA_VERSION,
                    role: config.role,
                    entity_id: config.entity_id.clone(),
                    scope: usage.scope,
                    source: usage.source,
                    sequence,
                    sampled_at_ms,
                    sample_interval_ms: 5_000,
                    cpu_usage_percent: usage.cpu_usage_percent,
                    cpu_capacity_cores: usage.cpu_capacity_cores,
                    memory_used_bytes: usage.memory_used_bytes,
                    memory_available_bytes: usage.memory_available_bytes,
                    memory_limit_bytes: usage.memory_limit_bytes,
                    nodes: resolve_nodes(
                        &config.nodes,
                        &sampler.processes(),
                        usage.cpu_capacity_cores,
                        sampled_at_ms,
                    ),
                };
                if let Err(error) = snapshot.validate() {
                    tracing::error!(%error, "resource monitor rejected its invalid snapshot");
                    continue;
                }
                let payload = serde_json::to_vec(&snapshot)?;
                node.send_output(
                    DataId::from("resource_snapshot".to_owned()),
                    Default::default(),
                    BinaryArray::from_vec(vec![payload.as_slice()]),
                )?;
            }
            Event::Stop { .. } => break,
            _ => {}
        }
    }
    tracing::info!("resource monitor stopped");
    Ok(())
}
