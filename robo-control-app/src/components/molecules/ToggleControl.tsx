import React from "react";
import { LucideIcon } from "lucide-react";
import { StatusBadge, StatusVariant } from "../atoms";

export interface ToggleControlProps {
  label: string;
  description?: string;
  isEnabled: boolean;
  onToggle: () => void;
  icon: LucideIcon;
  statusVariant?: StatusVariant;
  disabled?: boolean;
  className?: string;
}

export const ToggleControl: React.FC<ToggleControlProps> = ({
  label,
  description,
  isEnabled,
  onToggle,
  icon: Icon,
  statusVariant,
  disabled = false,
  className = "",
}) => {
  const status = statusVariant || (isEnabled ? "online" : "offline");

  return (
    <div className={`glass-card-light rounded-xl p-4 space-y-3 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className="w-5 h-5 text-cyan-400" />
          <span className="font-semibold text-white">{label}</span>
        </div>
        <StatusBadge variant={status} animated={isEnabled} />
      </div>
      {description && (
        <p className="text-xs text-white/60">{description}</p>
      )}
      <button
        onClick={onToggle}
        disabled={disabled}
        className={`
          w-full py-2 px-4 rounded-lg font-semibold
          transition-all duration-300
          ${isEnabled
            ? "bg-red-500/20 text-red-400 hover:bg-red-500/30"
            : "bg-green-500/20 text-green-400 hover:bg-green-500/30"
          }
          disabled:opacity-50 disabled:cursor-not-allowed
        `}
      >
        {isEnabled ? "Stop" : "Start"}
      </button>
    </div>
  );
};
