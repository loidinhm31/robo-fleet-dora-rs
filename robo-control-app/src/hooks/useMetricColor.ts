import { useMemo } from "react";

export type MetricType = "fps" | "cpu" | "memory" | "latency" | "confidence";

export interface MetricThresholds {
  good: number;
  warning: number;
}

const defaultThresholds: Record<MetricType, MetricThresholds> = {
  fps: { good: 20, warning: 10 },
  cpu: { good: 50, warning: 80 },
  memory: { good: 50, warning: 80 },
  latency: { good: 100, warning: 300 },
  confidence: { good: 0.8, warning: 0.6 },
};

export interface UseMetricColorOptions {
  type: MetricType;
  value: number;
  inverted?: boolean;
  thresholds?: MetricThresholds;
}

export interface UseMetricColorReturn {
  color: string;
  status: "good" | "warning" | "critical";
}

export const useMetricColor = ({
  type,
  value,
  inverted = false,
  thresholds,
}: UseMetricColorOptions): UseMetricColorReturn => {
  const { color, status } = useMemo(() => {
    const threshold = thresholds || defaultThresholds[type];

    if (!inverted) {
      // Higher is better (e.g., FPS, confidence)
      if (value >= threshold.good) {
        return { color: "text-emerald-400", status: "good" as const };
      }
      if (value >= threshold.warning) {
        return { color: "text-amber-400", status: "warning" as const };
      }
      return { color: "text-rose-400", status: "critical" as const };
    } else {
      // Lower is better (e.g., CPU, memory, latency)
      if (value <= threshold.good) {
        return { color: "text-emerald-400", status: "good" as const };
      }
      if (value <= threshold.warning) {
        return { color: "text-amber-400", status: "warning" as const };
      }
      return { color: "text-rose-400", status: "critical" as const };
    }
  }, [type, value, inverted, thresholds]);

  return { color, status };
};

export const getMetricColor = (
  type: MetricType,
  value: number,
  inverted = false
): string => {
  const threshold = defaultThresholds[type];

  if (!inverted) {
    if (value >= threshold.good) return "text-emerald-400";
    if (value >= threshold.warning) return "text-amber-400";
    return "text-rose-400";
  } else {
    if (value <= threshold.good) return "text-emerald-400";
    if (value <= threshold.warning) return "text-amber-400";
    return "text-rose-400";
  }
};

export const getMetricStatus = (
  type: MetricType,
  value: number,
  inverted = false
): "good" | "warning" | "critical" => {
  const threshold = defaultThresholds[type];

  if (!inverted) {
    if (value >= threshold.good) return "good";
    if (value >= threshold.warning) return "warning";
    return "critical";
  } else {
    if (value <= threshold.good) return "good";
    if (value <= threshold.warning) return "warning";
    return "critical";
  }
};
