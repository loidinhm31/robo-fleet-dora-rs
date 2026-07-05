use dora_node_api::{arrow::array::BinaryArray, dora_core::config::DataId, DoraNode, Event};
use eyre::Result;
use robo_rover_lib::init_tracing;
use robo_rover_lib::types::{NodeMetrics, SystemMetrics};
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use sysinfo::System;

/// Performance tracker for a single node
struct NodePerformanceTracker {
    node_id: String,
    dropped_frames: u64,
    processing_times: Vec<f32>,
    last_fps_calculation: Instant,
    max_processing_time: f32,
}

impl NodePerformanceTracker {
    fn new(node_id: String) -> Self {
        Self {
            node_id,
            dropped_frames: 0,
            processing_times: Vec::with_capacity(100),
            last_fps_calculation: Instant::now(),
            max_processing_time: 0.0,
        }
    }

    fn calculate_metrics(&mut self, cpu_percent: f32, memory_mb: f32) -> NodeMetrics {
        // Estimate FPS based on CPU activity
        // Nodes with higher CPU usage are likely processing more frames
        // This is a heuristic since we don't have direct frame counting
        let fps = if cpu_percent > 1.0 {
            // Assume ~30 FPS target for active nodes
            // Scale based on CPU usage (higher CPU = more work = likely higher FPS)
            (cpu_percent / 5.0).min(30.0)
        } else {
            0.0
        };

        let avg_processing_time = if !self.processing_times.is_empty() {
            self.processing_times.iter().sum::<f32>() / self.processing_times.len() as f32
        } else {
            // Estimate based on FPS: if running at 30fps, each frame takes ~33ms
            if fps > 0.0 {
                1000.0 / fps
            } else {
                0.0
            }
        };

        let metrics = NodeMetrics {
            node_id: self.node_id.clone(),
            fps,
            avg_processing_time_ms: avg_processing_time,
            max_processing_time_ms: self.max_processing_time,
            cpu_usage_percent: cpu_percent,
            memory_usage_mb: memory_mb,
            queue_size: 0, // Would need dataflow introspection
            dropped_frames: self.dropped_frames,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        // Reset for next interval
        self.last_fps_calculation = Instant::now();
        self.max_processing_time = 0.0;

        metrics
    }
}

/// Read battery information from Linux /sys/class/power_supply/
/// Returns (battery_level_percent, battery_voltage_volts) if available
fn read_battery_info() -> (Option<f32>, Option<f32>) {
    let power_supply_path = Path::new("/sys/class/power_supply");

    if !power_supply_path.exists() {
        return (None, None);
    }

    // Try to find a battery device (BAT0, BAT1, etc.)
    if let Ok(entries) = fs::read_dir(power_supply_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();

            // Look for battery devices (skip AC adapters)
            if name.starts_with("BAT") || name.to_lowercase().contains("battery") {
                let mut battery_level = None;
                let mut battery_voltage = None;

                // Read battery capacity (percentage)
                if let Ok(capacity) = fs::read_to_string(path.join("capacity")) {
                    if let Ok(level) = capacity.trim().parse::<f32>() {
                        battery_level = Some(level);
                    }
                }

                // Read battery voltage (microvolts -> volts)
                if let Ok(voltage_now) = fs::read_to_string(path.join("voltage_now")) {
                    if let Ok(voltage_uv) = voltage_now.trim().parse::<f32>() {
                        battery_voltage = Some(voltage_uv / 1_000_000.0);
                    }
                }

                // If we found battery info, return it
                if battery_level.is_some() || battery_voltage.is_some() {
                    return (battery_level, battery_voltage);
                }
            }
        }
    }

    (None, None)
}

fn main() -> Result<()> {
    let _guard = init_tracing();

    tracing::info!("Starting performance_monitor node");

    // Get entity ID from environment
    let entity_id = std::env::var("ENTITY_ID").ok();
    if let Some(ref id) = entity_id {
        tracing::info!("Monitoring rover: {}", id);
    }

    // Initialize system info
    let mut sys = System::new_all();

    // Wait for system to gather initial data
    std::thread::sleep(Duration::from_millis(500));
    sys.refresh_all();

    let (mut node, mut events) = DoraNode::init_from_env()?;
    let mut trackers: HashMap<String, NodePerformanceTracker> = HashMap::new();

    // Nodes to monitor - specific nodes
    let monitored_nodes = vec![
        "gst-camera",
        "object-detector",
        "object-tracker",
        "visual-servo-controller",
        "audio-capture",
        "audio-playback",
        "edge-voice",
        "arm-controller",
        "rover-controller",
        "sim-interface",
        "zenoh-bridge",
    ];

    // Initialize trackers
    for node_name in monitored_nodes.iter() {
        trackers.insert(
            node_name.to_string(),
            NodePerformanceTracker::new(node_name.to_string()),
        );
    }

    tracing::info!(
        "Monitoring {} nodes, triggered by timer ticks",
        trackers.len()
    );

    // Main event loop - collect metrics on each tick
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, .. } => {
                if id.as_str() == "tick" {
                    // Refresh system info
                    sys.refresh_all();

                    let mut system_metrics = SystemMetrics::new();

                    // Set entity ID for fleet tracking
                    system_metrics.entity_id = entity_id.clone();

                    // Collect system-wide metrics
                    system_metrics.total_cpu_percent = sys.global_cpu_usage();
                    system_metrics.total_memory_mb = (sys.used_memory() as f32) / 1024.0 / 1024.0;
                    system_metrics.available_memory_mb =
                        (sys.available_memory() as f32) / 1024.0 / 1024.0;
                    system_metrics.total_system_memory_mb =
                        (sys.total_memory() as f32) / 1024.0 / 1024.0;

                    // Collect battery information if available
                    let (battery_level, battery_voltage) = read_battery_info();
                    system_metrics.battery_level = battery_level;
                    system_metrics.battery_voltage = battery_voltage;

                    // Collect per-node metrics
                    for (node_name, tracker) in trackers.iter_mut() {
                        // Find process by name (approximate matching)
                        let (cpu_percent, memory_mb) = sys
                            .processes()
                            .iter()
                            .find(|(_, p)| {
                                let name_str = p.name().to_string_lossy().to_lowercase();
                                name_str.contains(&node_name.replace("-", "_"))
                            })
                            .map(|(_, p)| (p.cpu_usage(), (p.memory() as f32) / 1024.0 / 1024.0))
                            .unwrap_or((0.0, 0.0));

                        let metrics = tracker.calculate_metrics(cpu_percent, memory_mb);
                        system_metrics.update_node_metrics(metrics);
                    }

                    // Calculate overall dataflow FPS
                    system_metrics.calculate_dataflow_fps();

                    // Estimate end-to-end latency (sum of avg processing times in vision pipeline)
                    // For rover-kiwi: camera -> detector -> tracker -> zenoh-bridge
                    let vision_pipeline = [
                        "gst-camera",
                        "object-detector",
                        "object-tracker",
                        "zenoh-bridge",
                    ];
                    system_metrics.end_to_end_latency_ms = vision_pipeline
                        .iter()
                        .filter_map(|node| system_metrics.node_metrics.get(*node))
                        .map(|m| m.avg_processing_time_ms)
                        .sum();

                    let edge_voice_metrics = system_metrics.node_metrics.get("edge-voice");
                    let edge_voice_cpu_percent = edge_voice_metrics
                        .map(|metrics| metrics.cpu_usage_percent)
                        .unwrap_or(0.0);
                    let edge_voice_memory_mb = edge_voice_metrics
                        .map(|metrics| metrics.memory_usage_mb)
                        .unwrap_or(0.0);
                    let edge_voice_fps =
                        edge_voice_metrics.map(|metrics| metrics.fps).unwrap_or(0.0);
                    let edge_voice_found = edge_voice_metrics.is_some();

                    // Send metrics
                    let metrics_json = serde_json::to_vec(&system_metrics)?;
                    let arrow_data = BinaryArray::from_vec(vec![metrics_json.as_slice()]);
                    node.send_output(
                        DataId::from("metrics".to_owned()),
                        Default::default(),
                        arrow_data,
                    )?;

                    // Log metrics at info level so the benchmark can see edge_voice RSS/CPU.
                    let battery_info =
                        match (system_metrics.battery_level, system_metrics.battery_voltage) {
                            (Some(level), Some(voltage)) => {
                                format!(", Battery: {:.1}% ({:.2}V)", level, voltage)
                            }
                            (Some(level), None) => format!(", Battery: {:.1}%", level),
                            (None, Some(voltage)) => format!(", Battery: {:.2}V", voltage),
                            (None, None) => String::new(),
                        };

                    tracing::info!(
                        total_cpu_percent = system_metrics.total_cpu_percent,
                        total_memory_mb = system_metrics.total_memory_mb,
                        available_memory_mb = system_metrics.available_memory_mb,
                        dataflow_fps = system_metrics.dataflow_fps,
                        end_to_end_latency_ms = system_metrics.end_to_end_latency_ms,
                        edge_voice_found,
                        edge_voice_cpu_percent,
                        edge_voice_memory_mb,
                        edge_voice_fps,
                        "System metrics{}",
                        battery_info
                    );
                }
            }
            Event::Stop { .. } => {
                tracing::info!("Received stop event");
                break;
            }
            _ => {}
        }
    }

    tracing::info!("Performance monitor shutting down");
    Ok(())
}
