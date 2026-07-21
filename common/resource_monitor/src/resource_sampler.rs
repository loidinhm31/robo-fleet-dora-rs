use robo_rover_lib::{ResourceScope, ResourceSource};
use std::path::Path;
use sysinfo::System;

use crate::cgroup::CgroupSampler;
use crate::process_resolver::ProcessRecord;

pub struct ResourceSampler {
    system: System,
    backend: Backend,
}

enum Backend {
    Native,
    CgroupV2(CgroupSampler),
    Unknown,
}

pub struct SystemUsage {
    pub scope: ResourceScope,
    pub source: ResourceSource,
    pub cpu_usage_percent: Option<f32>,
    pub cpu_capacity_cores: Option<f32>,
    pub memory_used_bytes: Option<u64>,
    pub memory_available_bytes: Option<u64>,
    pub memory_limit_bytes: Option<u64>,
}

impl ResourceSampler {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        let backend = match std::env::var("RESOURCE_MONITOR_SCOPE").ok().as_deref() {
            Some("host") => Backend::Native,
            Some("container") if Path::new("/sys/fs/cgroup/cpu.stat").exists() => {
                Backend::CgroupV2(CgroupSampler::new("/sys/fs/cgroup"))
            }
            Some("container") => Backend::Unknown,
            _ if is_container() && Path::new("/sys/fs/cgroup/cpu.stat").exists() => {
                Backend::CgroupV2(CgroupSampler::new("/sys/fs/cgroup"))
            }
            _ => Backend::Native,
        };
        Self { system, backend }
    }

    pub fn sample(&mut self) -> SystemUsage {
        self.system.refresh_all();
        match &mut self.backend {
            Backend::Native => native_usage(&self.system),
            Backend::CgroupV2(sampler) => sampler.sample(),
            Backend::Unknown => unknown_usage(),
        }
    }

    pub fn processes(&self) -> Vec<ProcessRecord> {
        self.system
            .processes()
            .values()
            .filter_map(|process| {
                let executable = process.exe()?.file_name()?.to_str()?.to_owned();
                Some(ProcessRecord {
                    executable,
                    cpu_percent_per_core: process.cpu_usage(),
                    memory_rss_bytes: process.memory(),
                })
            })
            .collect()
    }
}

fn native_usage(system: &System) -> SystemUsage {
    let capacity = system.cpus().len().max(1) as f32;
    SystemUsage {
        scope: ResourceScope::Host,
        source: ResourceSource::Procfs,
        cpu_usage_percent: finite_percent(system.global_cpu_usage()),
        cpu_capacity_cores: Some(capacity),
        memory_used_bytes: Some(system.used_memory()),
        memory_available_bytes: Some(system.available_memory()),
        memory_limit_bytes: Some(system.total_memory()),
    }
}

fn unknown_usage() -> SystemUsage {
    SystemUsage {
        scope: ResourceScope::Unknown,
        source: ResourceSource::Unknown,
        cpu_usage_percent: None,
        cpu_capacity_cores: None,
        memory_used_bytes: None,
        memory_available_bytes: None,
        memory_limit_bytes: None,
    }
}

fn is_container() -> bool {
    Path::new("/.dockerenv").exists() || Path::new("/run/.containerenv").exists()
}

fn finite_percent(value: f32) -> Option<f32> {
    value.is_finite().then(|| value.clamp(0.0, 100.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_scope_never_labels_host_values_as_container() {
        let usage = unknown_usage();
        assert_eq!(usage.scope, ResourceScope::Unknown);
        assert_eq!(usage.memory_used_bytes, None);
    }
}
