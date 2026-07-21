use robo_rover_lib::{NodeResourceState, NodeResourceUsage};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Deserialize)]
pub struct NodeManifest {
    pub node_id: String,
    pub executable: String,
}

impl NodeManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.node_id.is_empty()
            || self.executable.is_empty()
            || self.node_id.len() > 128
            || self.executable.len() > 128
            || self.executable.contains('/')
        {
            return Err("resource node manifest must use bounded basename executables".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ProcessRecord {
    pub executable: String,
    pub cpu_percent_per_core: f32,
    pub memory_rss_bytes: u64,
}

pub fn resolve_nodes(
    manifests: &[NodeManifest],
    processes: &[ProcessRecord],
    capacity_cores: Option<f32>,
    sampled_at_ms: i64,
) -> BTreeMap<String, NodeResourceUsage> {
    manifests
        .iter()
        .map(|manifest| {
            let matches: Vec<_> = processes
                .iter()
                .filter(|process| process.executable == manifest.executable)
                .collect();
            let state = match matches.len() {
                0 => NodeResourceState::NotFound,
                1 => NodeResourceState::Running,
                _ => NodeResourceState::Ambiguous,
            };
            let cpu_usage_percent = (!matches.is_empty())
                .then_some(())
                .and_then(|_| capacity_cores)
                .and_then(|capacity| {
                    (capacity.is_finite() && capacity > 0.0).then(|| {
                        (matches
                            .iter()
                            .map(|process| process.cpu_percent_per_core)
                            .sum::<f32>()
                            / capacity)
                            .clamp(0.0, 100.0)
                    })
                });
            let memory_rss_bytes = (!matches.is_empty())
                .then(|| matches.iter().map(|process| process.memory_rss_bytes).sum());
            (
                manifest.node_id.clone(),
                NodeResourceUsage {
                    state,
                    cpu_usage_percent,
                    memory_rss_bytes,
                    process_count: matches.len() as u32,
                    sampled_at_ms,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_executable_matches_aggregate_and_report_ambiguity() {
        let nodes = resolve_nodes(
            &[NodeManifest {
                node_id: "camera".into(),
                executable: "kornia_capture".into(),
            }],
            &[
                ProcessRecord {
                    executable: "kornia_capture".into(),
                    cpu_percent_per_core: 20.0,
                    memory_rss_bytes: 10,
                },
                ProcessRecord {
                    executable: "kornia_capture".into(),
                    cpu_percent_per_core: 40.0,
                    memory_rss_bytes: 15,
                },
                ProcessRecord {
                    executable: "not-a-camera".into(),
                    cpu_percent_per_core: 100.0,
                    memory_rss_bytes: 100,
                },
            ],
            Some(2.0),
            10,
        );
        let camera = &nodes["camera"];
        assert_eq!(camera.state, NodeResourceState::Ambiguous);
        assert_eq!(camera.process_count, 2);
        assert_eq!(camera.cpu_usage_percent, Some(30.0));
        assert_eq!(camera.memory_rss_bytes, Some(25));
    }

    #[test]
    fn missing_process_is_not_a_measured_zero() {
        let nodes = resolve_nodes(
            &[NodeManifest {
                node_id: "camera".into(),
                executable: "kornia_capture".into(),
            }],
            &[],
            Some(4.0),
            10,
        );
        let camera = &nodes["camera"];
        assert_eq!(camera.state, NodeResourceState::NotFound);
        assert_eq!(camera.cpu_usage_percent, None);
        assert_eq!(camera.memory_rss_bytes, None);
    }
}
