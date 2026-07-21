use std::fs;
use std::time::Instant;

use robo_rover_lib::{ResourceScope, ResourceSource};

use crate::resource_sampler::SystemUsage;

pub struct CgroupSampler {
    root: String,
    previous_usage_usec: Option<u64>,
    previous_at: Option<Instant>,
}

impl CgroupSampler {
    pub fn new(root: impl Into<String>) -> Self {
        Self {
            root: root.into(),
            previous_usage_usec: None,
            previous_at: None,
        }
    }

    pub fn sample(&mut self) -> SystemUsage {
        self.sample_at(Instant::now())
    }

    fn sample_at(&mut self, now: Instant) -> SystemUsage {
        let usage = read_cpu_usage_usec(&format!("{}/cpu.stat", self.root));
        let capacity = read_cpu_capacity(&format!("{}/cpu.max", self.root));
        let cpu_usage_percent = usage
            .zip(self.previous_usage_usec)
            .zip(self.previous_at)
            .and_then(|((current, prior), at)| {
                let elapsed = now.duration_since(at).as_micros() as f32;
                let capacity = capacity?;
                (elapsed > 0.0 && capacity > 0.0).then(|| {
                    (((current.saturating_sub(prior)) as f32 / elapsed) / capacity * 100.0)
                        .clamp(0.0, 100.0)
                })
            });
        self.previous_usage_usec = usage;
        self.previous_at = Some(now);
        let memory_used_bytes = read_u64(&format!("{}/memory.current", self.root));
        let memory_limit_bytes = read_memory_max(&format!("{}/memory.max", self.root));
        SystemUsage {
            scope: ResourceScope::Container,
            source: ResourceSource::CgroupV2,
            cpu_usage_percent,
            cpu_capacity_cores: capacity,
            memory_used_bytes,
            memory_available_bytes: memory_limit_bytes
                .zip(memory_used_bytes)
                .map(|(limit, used)| limit.saturating_sub(used)),
            memory_limit_bytes,
        }
    }
}

fn read_cpu_usage_usec(path: &str) -> Option<u64> {
    fs::read_to_string(path)
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("usage_usec ")?.parse().ok())
}

fn read_cpu_capacity(path: &str) -> Option<f32> {
    parse_cpu_capacity(&fs::read_to_string(path).ok()?)
}

fn parse_cpu_capacity(value: &str) -> Option<f32> {
    let mut fields = value.split_whitespace();
    let quota = fields.next()?;
    let period = fields.next()?.parse::<f32>().ok()?;
    if quota == "max" {
        Some(std::thread::available_parallelism().ok()?.get() as f32)
    } else {
        Some(quota.parse::<f32>().ok()? / period).filter(|value| *value > 0.0)
    }
}

fn read_memory_max(path: &str) -> Option<u64> {
    parse_memory_max(&fs::read_to_string(path).ok()?)
}

fn parse_memory_max(value: &str) -> Option<u64> {
    (value.trim() != "max")
        .then(|| value.trim().parse().ok())
        .flatten()
}

fn read_u64(path: &str) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn cgroup_limits_parse_finite_and_unlimited_values() {
        assert_eq!(parse_memory_max("1048576\n"), Some(1_048_576));
        assert_eq!(parse_memory_max("max\n"), None);
        assert_eq!(parse_cpu_capacity("200000 100000\n"), Some(2.0));
    }

    #[test]
    fn cgroup_samples_normalize_cpu_and_memory_with_an_injected_clock() {
        let root =
            std::env::temp_dir().join(format!("resource-monitor-cgroup-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("cpu.max"), "200000 100000\n").unwrap();
        fs::write(root.join("memory.max"), "1000\n").unwrap();
        fs::write(root.join("memory.current"), "400\n").unwrap();
        fs::write(root.join("cpu.stat"), "usage_usec 100000\n").unwrap();

        let mut sampler = CgroupSampler::new(root.to_string_lossy());
        let start = Instant::now();
        assert_eq!(sampler.sample_at(start).cpu_usage_percent, None);
        fs::write(root.join("cpu.stat"), "usage_usec 200000\n").unwrap();
        let usage = sampler.sample_at(start + Duration::from_secs(1));
        assert_eq!(usage.cpu_usage_percent, Some(5.0));
        assert_eq!(usage.memory_available_bytes, Some(600));
        fs::remove_dir_all(root).unwrap();
    }
}
