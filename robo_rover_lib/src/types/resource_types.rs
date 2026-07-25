use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const RESOURCE_SCHEMA_VERSION: u16 = 1;
const MAX_RESOURCE_NODES: usize = 128;
const MAX_RESOURCE_ID_LENGTH: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceRole {
    Orchestra,
    Rover,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceScope {
    Host,
    Container,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceSource {
    Procfs,
    CgroupV2,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeResourceState {
    Running,
    NotFound,
    Ambiguous,
    Paused,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeResourceUsage {
    pub state: NodeResourceState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_usage_percent: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_rss_bytes: Option<u64>,
    pub process_count: u32,
    pub sampled_at_ms: i64,
}

/// Measured resource evidence for one configured workload domain. Domains are
/// built only from the process identities in the monitor manifest; an absent
/// CPU value is therefore evidence that Auto must not infer a zero load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DomainResourceUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_usage_percent: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_rss_bytes: Option<u64>,
    pub process_count: u32,
    pub configured_node_count: u32,
    pub sampled_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub schema_version: u16,
    pub role: ResourceRole,
    pub entity_id: String,
    pub scope: ResourceScope,
    pub source: ResourceSource,
    pub sequence: u64,
    pub sampled_at_ms: i64,
    pub sample_interval_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_usage_percent: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_capacity_cores: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_used_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_available_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_limit_bytes: Option<u64>,
    pub nodes: BTreeMap<String, NodeResourceUsage>,
    #[serde(default)]
    pub domains: BTreeMap<String, DomainResourceUsage>,
}

impl ResourceSnapshot {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != RESOURCE_SCHEMA_VERSION {
            return Err("unsupported resource schema version".into());
        }
        if !valid_id(&self.entity_id) {
            return Err("invalid resource entity_id".into());
        }
        if self.sampled_at_ms <= 0 || self.sample_interval_ms == 0 {
            return Err("resource sample time and interval must be positive".into());
        }
        if !matches!(
            (self.scope, self.source),
            (ResourceScope::Host, ResourceSource::Procfs)
                | (ResourceScope::Container, ResourceSource::CgroupV2)
                | (ResourceScope::Unknown, ResourceSource::Unknown)
        ) {
            return Err("resource scope and source disagree".into());
        }
        if self.nodes.len() > MAX_RESOURCE_NODES {
            return Err("too many resource nodes".into());
        }
        valid_percent(self.cpu_usage_percent, "resource cpu usage")?;
        if let Some(capacity) = self.cpu_capacity_cores {
            if !capacity.is_finite() || capacity <= 0.0 {
                return Err("resource cpu capacity must be finite and positive".into());
            }
        }
        for (node_id, usage) in &self.nodes {
            if !valid_id(node_id) {
                return Err("invalid resource node id".into());
            }
            valid_percent(usage.cpu_usage_percent, "node cpu usage")?;
            if usage.sampled_at_ms != self.sampled_at_ms {
                return Err("node resource sample time must match its snapshot".into());
            }
            match usage.state {
                NodeResourceState::Running if usage.process_count == 0 => {
                    return Err("running resource node has no process".into())
                }
                NodeResourceState::NotFound | NodeResourceState::Paused
                    if usage.process_count != 0
                        || usage.cpu_usage_percent.is_some()
                        || usage.memory_rss_bytes.is_some() =>
                {
                    return Err("unavailable resource node has measurements or processes".into())
                }
                NodeResourceState::Ambiguous if usage.process_count < 2 => {
                    return Err("ambiguous resource node needs multiple processes".into())
                }
                _ => {}
            }
        }
        if self.domains.len() > MAX_RESOURCE_NODES {
            return Err("too many resource domains".into());
        }
        for (domain_id, usage) in &self.domains {
            if !valid_id(domain_id) || usage.configured_node_count == 0 {
                return Err("invalid resource domain".into());
            }
            valid_percent(usage.cpu_usage_percent, "domain cpu usage")?;
            if usage.sampled_at_ms != self.sampled_at_ms {
                return Err("domain resource sample time must match its snapshot".into());
            }
            if usage.cpu_usage_percent.is_none() && usage.memory_rss_bytes.is_some() {
                return Err("incomplete resource domain has memory evidence".into());
            }
        }
        Ok(())
    }
}

fn valid_id(value: &str) -> bool {
    !value.is_empty() && value.len() <= MAX_RESOURCE_ID_LENGTH
}

fn valid_percent(value: Option<f32>, label: &str) -> Result<(), String> {
    if let Some(value) = value {
        if !value.is_finite() || !(0.0..=100.0).contains(&value) {
            return Err(format!("{label} must be finite and within 0..=100"));
        }
    }
    Ok(())
}
