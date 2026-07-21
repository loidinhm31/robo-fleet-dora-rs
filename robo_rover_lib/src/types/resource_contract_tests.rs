use super::{
    NodeResourceState, NodeResourceUsage, ResourceRole, ResourceScope, ResourceSnapshot,
    ResourceSource, RESOURCE_SCHEMA_VERSION,
};
use std::collections::BTreeMap;

fn snapshot() -> ResourceSnapshot {
    ResourceSnapshot {
        schema_version: RESOURCE_SCHEMA_VERSION,
        role: ResourceRole::Rover,
        entity_id: "rover-kiwi".into(),
        scope: ResourceScope::Host,
        source: ResourceSource::Procfs,
        sequence: 1,
        sampled_at_ms: 1_784_599_200_000,
        sample_interval_ms: 5_000,
        cpu_usage_percent: Some(0.0),
        cpu_capacity_cores: Some(4.0),
        memory_used_bytes: Some(1024),
        memory_available_bytes: Some(2048),
        memory_limit_bytes: None,
        nodes: BTreeMap::from([(
            "camera".into(),
            NodeResourceUsage {
                state: NodeResourceState::Running,
                cpu_usage_percent: Some(0.0),
                memory_rss_bytes: Some(0),
                process_count: 1,
                sampled_at_ms: 1_784_599_200_000,
            },
        )]),
    }
}

#[test]
fn resource_snapshot_round_trips_and_keeps_measured_zero() {
    let snapshot = snapshot();
    snapshot.validate().unwrap();
    let decoded: ResourceSnapshot =
        serde_json::from_str(&serde_json::to_string(&snapshot).unwrap()).unwrap();
    assert_eq!(decoded, snapshot);
}

#[test]
fn resource_snapshot_rejects_invalid_percent_and_missing_process() {
    let mut snapshot = snapshot();
    snapshot.cpu_usage_percent = Some(f32::NAN);
    assert!(snapshot.validate().is_err());
    snapshot.cpu_usage_percent = Some(1.0);
    snapshot.nodes.get_mut("camera").unwrap().process_count = 0;
    assert!(snapshot.validate().is_err());
}

#[test]
fn unavailable_node_cannot_claim_measured_resources() {
    let mut snapshot = snapshot();
    {
        let usage = snapshot.nodes.get_mut("camera").unwrap();
        usage.state = NodeResourceState::NotFound;
        usage.process_count = 0;
    }
    assert!(snapshot.validate().is_err());
    let usage = snapshot.nodes.get_mut("camera").unwrap();
    usage.cpu_usage_percent = None;
    usage.memory_rss_bytes = None;
    snapshot.validate().unwrap();
}

#[test]
fn resource_snapshot_fixture_is_a_valid_version_one_contract() {
    let snapshot: ResourceSnapshot = serde_json::from_str(include_str!(
        "../../tests/fixtures/resource-snapshot-v1.json"
    ))
    .unwrap();
    snapshot.validate().unwrap();
    assert_eq!(snapshot.sequence, 42);
}
