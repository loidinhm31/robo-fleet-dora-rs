mod manager;

pub use manager::*;

use robo_rover_lib::{LifecycleCapability, LifecycleRole, LifecycleTarget};
use std::collections::BTreeSet;

/// Safe Rover workloads whose desired lifecycle state is owned by Orchestra in
/// Zenoh mode. The Rover manager mirrors the Orchestra authority when it
/// applies a relayed command.
pub const REMOTE_ROVER_SAFE_NODE_IDS: [&str; 4] = [
    "gst-camera",
    "audio-capture",
    "edge-voice",
    "audio-playback",
];
/// The 64-entry lifecycle-status queues hold 60 remote status reports and
/// four local Orchestra reports in a complete manager tick.
pub const MAX_REMOTE_ROVERS: usize = 15;

pub fn remote_rover_capabilities(entities: &str) -> Result<Vec<LifecycleCapability>, String> {
    let entity_ids = entities
        .split(',')
        .map(str::trim)
        .filter(|entity_id| !entity_id.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if entity_ids.len() > MAX_REMOTE_ROVERS {
        return Err(format!(
            "configured remote Rovers ({}) exceed lifecycle status queue limit ({MAX_REMOTE_ROVERS})",
            entity_ids.len()
        ));
    }
    Ok(entity_ids
        .into_iter()
        .flat_map(|entity_id| {
            REMOTE_ROVER_SAFE_NODE_IDS
                .iter()
                .map(move |node_id| LifecycleCapability {
                    target: LifecycleTarget {
                        role: LifecycleRole::Rover,
                        entity_id: entity_id.to_owned(),
                        node_id: (*node_id).into(),
                    },
                    supported: true,
                    always_on: false,
                })
        })
        .collect())
}

#[cfg(test)]
mod container_packaging_tests {
    use std::{fs, path::Path};

    fn read_repo_file(root: &Path, path: &str) -> String {
        fs::read_to_string(root.join(path)).unwrap_or_else(|error| panic!("{path}: {error}"))
    }

    #[test]
    fn container_images_ship_the_lifecycle_manager() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for manifest in ["docker/Cargo.orchestra.toml", "docker/Cargo.rover.toml"] {
            assert!(read_repo_file(&root, manifest).contains("\"common/lifecycle_manager\""));
        }
        for dockerfile in [
            "docker/Dockerfile.orchestra",
            "docker/Dockerfile.rover-kiwi",
        ] {
            let content = read_repo_file(&root, dockerfile);
            assert!(content.contains("common/lifecycle_manager/Cargo.toml"));
            assert!(content.contains("-p lifecycle_manager"));
            assert!(content.contains("/build/target/release/lifecycle_manager"));
        }
        let compose = read_repo_file(&root, "docker/docker-compose.yml");
        assert!(compose.contains("/app/bin/resource_monitor"));
        assert!(compose.contains("/app/bin/lifecycle_manager"));
    }

    #[test]
    fn orchestra_browser_lifecycle_uses_manager_authority() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dataflow = read_repo_file(&root, "orchestra/orchestra-dataflow.yml");
        let web_bridge = dataflow
            .split("  - id: web-bridge")
            .nth(1)
            .expect("web bridge dataflow node");

        assert!(web_bridge.contains(
            "lifecycle_status:\n        source: lifecycle-manager/lifecycle_status\n        queue_size: 64"
        ));
        assert!(
            web_bridge.contains("lifecycle_capabilities: lifecycle-manager/lifecycle_capabilities")
        );
        assert!(!web_bridge.contains("rover_lifecycle_status:"));
        assert!(!web_bridge.contains("rover_lifecycle_capabilities:"));
        assert!(!web_bridge.contains("rover_lifecycle_command_result:"));
    }

    #[test]
    fn lifecycle_status_fanout_has_room_for_every_safe_node() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let status_input =
            "lifecycle_status:\n        source: lifecycle-manager/lifecycle_status\n";
        for dataflow in [
            "orchestra/orchestra-dataflow.yml",
            "rover-kiwi/rover-kiwi-dataflow.yml",
            "rover-kiwi/rover-kiwi-direct-dataflow.yml",
        ] {
            let content = read_repo_file(&root, dataflow);
            let status_start = content
                .find(status_input)
                .unwrap_or_else(|| panic!("{dataflow} lifecycle status input"));
            let status_section = &content[status_start..];
            assert!(
                status_section.starts_with(&format!("{status_input}        queue_size: 64")),
                "{dataflow} must retain a complete lifecycle status tick"
            );
        }
    }

    #[test]
    fn orchestra_manager_retains_the_complete_remote_status_snapshot() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let orchestra = read_repo_file(&root, "orchestra/orchestra-dataflow.yml");

        assert!(orchestra.contains(
            "lifecycle_component_status:\n        source: orchestra-bridge/lifecycle_status\n        queue_size: 64"
        ));
    }
}

#[cfg(test)]
mod remote_rover_tests {
    use super::*;

    #[test]
    fn remote_rover_safe_nodes_are_server_controlled() {
        let capabilities = remote_rover_capabilities("rover-kiwi, rover-b").unwrap();
        assert_eq!(capabilities.len(), REMOTE_ROVER_SAFE_NODE_IDS.len() * 2);
        assert!(capabilities
            .iter()
            .all(|capability| capability.supported && !capability.always_on));
        assert!(capabilities.iter().any(|capability| {
            capability.target.entity_id == "rover-kiwi" && capability.target.node_id == "edge-voice"
        }));
    }

    #[test]
    fn remote_rover_capabilities_deduplicate_trimmed_entity_ids() {
        let capabilities =
            remote_rover_capabilities("rover-kiwi, rover-kiwi, rover-b, rover-b").unwrap();
        assert_eq!(capabilities.len(), REMOTE_ROVER_SAFE_NODE_IDS.len() * 2);
    }

    #[test]
    fn lifecycle_status_queue_covers_four_configured_rovers() {
        const STATUS_QUEUE_CAPACITY: usize = 64;
        const ORCHESTRA_LOCAL_STATUS_COUNT: usize = 4;
        let capabilities = remote_rover_capabilities("rover-a, rover-b, rover-c, rover-d").unwrap();

        assert_eq!(capabilities.len(), REMOTE_ROVER_SAFE_NODE_IDS.len() * 4);
        assert!(
            STATUS_QUEUE_CAPACITY >= capabilities.len() + ORCHESTRA_LOCAL_STATUS_COUNT,
            "the status queue must retain every safe-node status in a four-Rover tick"
        );
        assert_eq!(
            REMOTE_ROVER_SAFE_NODE_IDS.len() * MAX_REMOTE_ROVERS + ORCHESTRA_LOCAL_STATUS_COUNT,
            STATUS_QUEUE_CAPACITY
        );
    }

    #[test]
    fn remote_rover_configuration_rejects_queue_overflow() {
        let entities = (0..=MAX_REMOTE_ROVERS)
            .map(|index| format!("rover-{index}"))
            .collect::<Vec<_>>()
            .join(",");

        assert!(remote_rover_capabilities(&entities).is_err());
    }
}
