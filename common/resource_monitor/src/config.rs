use robo_rover_lib::ResourceRole;
use serde::Deserialize;

use crate::process_resolver::NodeManifest;

#[derive(Debug, Deserialize)]
pub struct MonitorConfig {
    pub role: ResourceRole,
    pub entity_id: String,
    pub nodes: Vec<NodeManifest>,
}

impl MonitorConfig {
    pub fn from_env() -> Result<Self, String> {
        let role = match required("RESOURCE_MONITOR_ROLE")?.as_str() {
            "rover" => ResourceRole::Rover,
            "orchestra" => ResourceRole::Orchestra,
            _ => return Err("RESOURCE_MONITOR_ROLE must be rover or orchestra".into()),
        };
        let entity_id = required("ENTITY_ID")?;
        let nodes = serde_json::from_str(&required("RESOURCE_MONITOR_NODES")?)
            .map_err(|error| format!("invalid RESOURCE_MONITOR_NODES: {error}"))?;
        let config = Self {
            role,
            entity_id,
            nodes,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if self.entity_id.is_empty() || self.entity_id.len() > 128 {
            return Err("ENTITY_ID must be a non-empty bounded string".into());
        }
        if self.nodes.len() > 128 {
            return Err("RESOURCE_MONITOR_NODES has too many nodes".into());
        }
        for node in &self.nodes {
            node.validate()?;
        }
        for (index, node) in self.nodes.iter().enumerate() {
            if self.nodes[..index]
                .iter()
                .any(|previous| previous.node_id == node.node_id)
            {
                return Err("RESOURCE_MONITOR_NODES has duplicate node IDs".into());
            }
        }
        Ok(())
    }
}

fn required(key: &str) -> Result<String, String> {
    std::env::var(key).map_err(|_| format!("missing required {key}"))
}
