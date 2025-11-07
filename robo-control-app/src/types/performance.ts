// Performance monitoring types

export interface NodeMetrics {
  node_id: string;
  fps: number;
  avg_processing_time_ms: number;
  max_processing_time_ms: number;
  cpu_usage_percent: number;
  memory_usage_mb: number;
  queue_size: number;
  dropped_frames: number;
  timestamp: number;
}

export interface SystemMetrics {
  total_cpu_percent: number;
  total_memory_mb: number;
  available_memory_mb: number;
  total_system_memory_mb: number;
  dataflow_fps: number;
  end_to_end_latency_ms: number;
  node_metrics: Record<string, NodeMetrics>;
  timestamp: number;
}
