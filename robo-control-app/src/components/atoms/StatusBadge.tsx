import React from "react";
import { LucideIcon, Radio, WifiOff, AlertTriangle, CheckCircle } from "lucide-react";

export type StatusVariant = "online" | "offline" | "warning" | "success" | "tracking" | "disabled";

export interface StatusBadgeProps {
  variant: StatusVariant;
  label?: string;
  animated?: boolean;
  icon?: LucideIcon;
  className?: string;
}

const variantConfig: Record<StatusVariant, {
  color: string;
  defaultIcon: LucideIcon;
  defaultLabel: string;
}> = {
  online: {
    color: "text-green-400",
    defaultIcon: Radio,
    defaultLabel: "ONLINE",
  },
  offline: {
    color: "text-gray-400",
    defaultIcon: WifiOff,
    defaultLabel: "OFFLINE",
  },
  warning: {
    color: "text-amber-400",
    defaultIcon: AlertTriangle,
    defaultLabel: "WARNING",
  },
  success: {
    color: "text-emerald-400",
    defaultIcon: CheckCircle,
    defaultLabel: "SUCCESS",
  },
  tracking: {
    color: "text-cyan-400",
    defaultIcon: Radio,
    defaultLabel: "TRACKING",
  },
  disabled: {
    color: "text-gray-500",
    defaultIcon: WifiOff,
    defaultLabel: "DISABLED",
  },
};

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  variant,
  label,
  animated = false,
  icon,
  className = "",
}) => {
  const config = variantConfig[variant];
  const Icon = icon || config.defaultIcon;
  const displayLabel = label || config.defaultLabel;

  return (
    <div className={`glass-card-light rounded-lg px-2 py-1 flex items-center gap-1.5 ${className}`}>
      <Icon className={`w-3 h-3 ${config.color} ${animated ? "animate-pulse" : ""}`} />
      <span className={`text-xs font-semibold ${config.color}`}>{displayLabel}</span>
    </div>
  );
};
