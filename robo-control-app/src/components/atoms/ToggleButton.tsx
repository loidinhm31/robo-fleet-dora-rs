import React from "react";
import { LucideIcon } from "lucide-react";

export interface ToggleButtonProps {
  isEnabled: boolean;
  onText: string;
  offText: string;
  onIcon: LucideIcon;
  offIcon: LucideIcon;
  onClick: () => void;
  disabled?: boolean;
  className?: string;
  variant?: "cyan" | "orange" | "green" | "red";
}

const variantConfig = {
  cyan: {
    on: "bg-gradient-to-r from-cyan-400 via-cyan-500 to-blue-500",
    off: "bg-gradient-to-r from-cyan-400 via-cyan-500 to-blue-500",
    iconOn: "text-white",
    iconOff: "text-white",
  },
  orange: {
    on: "bg-gradient-to-br from-orange-500 via-orange-600 to-amber-500",
    off: "bg-gradient-to-br from-orange-500 via-orange-600 to-amber-500",
    iconOn: "text-white",
    iconOff: "text-white",
  },
  green: {
    on: "bg-gradient-to-br from-green-500 via-green-600 to-emerald-500",
    off: "bg-gradient-to-br from-green-500 via-green-600 to-emerald-500",
    iconOn: "text-white",
    iconOff: "text-white",
  },
  red: {
    on: "bg-gradient-to-r from-red-500 via-red-600 to-rose-500",
    off: "bg-gradient-to-r from-gray-400 via-gray-500 to-gray-600",
    iconOn: "text-white",
    iconOff: "text-white",
  },
};

export const ToggleButton: React.FC<ToggleButtonProps> = ({
  isEnabled,
  onText,
  offText,
  onIcon: OnIcon,
  offIcon: OffIcon,
  onClick,
  disabled = false,
  className = "",
  variant = "cyan",
}) => {
  const config = variantConfig[variant];
  const gradient = isEnabled ? config.on : config.off;
  const iconColor = isEnabled ? config.iconOn : config.iconOff;

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`
        ${gradient}
        text-white font-bold
        shadow-lg hover:shadow-xl
        transition-all duration-300 hover:scale-105
        rounded-xl px-4 py-2 flex items-center gap-2
        disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100
        ${className}
      `}
    >
      {isEnabled ? (
        <>
          <OnIcon className={`w-5 h-5 ${iconColor}`} />
          <span>{onText}</span>
        </>
      ) : (
        <>
          <OffIcon className={`w-5 h-5 ${iconColor}`} />
          <span>{offText}</span>
        </>
      )}
    </button>
  );
};
