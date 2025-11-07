use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metrics for a single node in the dataflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetrics {
    /// Node ID (name)
    pub node_id: String,
    /// Frames per second (if applicable)
    pub fps: f32,
    /// Average processing time in milliseconds
    pub avg_processing_time_ms: f32,
    /// Maximum processing time in milliseconds
    pub max_processing_time_ms: f32,
    /// CPU usage percentage (0-100 per core, can exceed 100 for multi-threaded)
    pub cpu_usage_percent: f32,
    /// Memory usage in megabytes
    pub memory_usage_mb: f32,
    /// Queue size (number of pending messages)
    pub queue_size: usize,
    /// Total number of dropped frames
    pub dropped_frames: u64,
    /// Timestamp when metrics were collected (Unix milliseconds)
    pub timestamp: i64,
}

/// System-wide performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,
    /// Overall system CPU usage percentage (0-100)
    pub total_cpu_percent: f32,
    /// Total memory usage in megabytes
    pub total_memory_mb: f32,
    /// Total available memory in megabytes
    pub available_memory_mb: f32,
    /// Total system memory in megabytes
    pub total_system_memory_mb: f32,
    /// Battery level percentage (0.0-100.0, None if not available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub battery_level: Option<f32>,
    /// Battery voltage (None if not available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub battery_voltage: Option<f32>,
    /// Overall dataflow FPS (minimum FPS across all vision nodes)
    pub dataflow_fps: f32,
    /// End-to-end latency in milliseconds (camera → web UI)
    pub end_to_end_latency_ms: f32,
    /// Per-node metrics
    pub node_metrics: HashMap<String, NodeMetrics>,
    /// Timestamp when metrics were collected (Unix milliseconds)
    pub timestamp: i64,
}

impl SystemMetrics {
    /// Create a new SystemMetrics instance
    pub fn new() -> Self {
        Self {
            entity_id: None,
            total_cpu_percent: 0.0,
            total_memory_mb: 0.0,
            available_memory_mb: 0.0,
            total_system_memory_mb: 0.0,
            battery_level: None,
            battery_voltage: None,
            dataflow_fps: 0.0,
            end_to_end_latency_ms: 0.0,
            node_metrics: HashMap::new(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Add or update metrics for a specific node
    pub fn update_node_metrics(&mut self, metrics: NodeMetrics) {
        self.node_metrics
            .insert(metrics.node_id.clone(), metrics);
    }

    /// Calculate overall dataflow FPS (minimum FPS across vision nodes)
    pub fn calculate_dataflow_fps(&mut self) {
        let vision_nodes = [
            "gst-camera",
            "object-detector",
            "object-tracker",
            "visual-servo-controller",
        ];

        self.dataflow_fps = vision_nodes
            .iter()
            .filter_map(|node| self.node_metrics.get(*node))
            .map(|metrics| metrics.fps)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Process information for resource monitoring
#[derive(Debug, Clone)]
pub struct ProcessInfo {
    /// Process ID
    pub pid: u32,
    /// Process name
    pub name: String,
    /// CPU usage percentage
    pub cpu_percent: f32,
    /// Memory usage in bytes
    pub memory_bytes: u64,
}

/// Configuration for performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Metrics collection interval in milliseconds
    pub collection_interval_ms: u64,
    /// Enable CPU monitoring
    pub monitor_cpu: bool,
    /// Enable memory monitoring
    pub monitor_memory: bool,
    /// Enable queue size monitoring
    pub monitor_queues: bool,
    /// List of nodes to monitor (empty = all nodes)
    pub monitored_nodes: Vec<String>,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            collection_interval_ms: 1000, // 1 second
            monitor_cpu: true,
            monitor_memory: true,
            monitor_queues: true,
            monitored_nodes: vec![],
        }
    }
}
