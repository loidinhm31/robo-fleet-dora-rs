import React from "react";

export interface StatCardProps {
  label: string;
  value: number | string;
  unit?: string;
  color?: string;
  size?: "sm" | "md" | "lg";
  monospace?: boolean;
  decimals?: number;
  className?: string;
}

const sizeConfig = {
  sm: {
    labelSize: "text-xs",
    valueSize: "text-base",
  },
  md: {
    labelSize: "text-sm",
    valueSize: "text-lg",
  },
  lg: {
    labelSize: "text-base",
    valueSize: "text-2xl",
  },
};

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  unit = "",
  color = "text-cyan-300",
  size = "md",
  monospace = true,
  decimals = 1,
  className = "",
}) => {
  const sizes = sizeConfig[size];

  const formattedValue = typeof value === "number"
    ? value.toFixed(decimals)
    : value;

  return (
    <div className={`space-y-1 ${className}`}>
      <div className={`${sizes.labelSize} text-cyan-200`}>
        {label}
      </div>
      <div className={`${sizes.valueSize} ${monospace ? "font-mono" : ""} font-bold ${color}`}>
        {formattedValue}{unit && <span className="text-xs ml-1">{unit}</span>}
      </div>
    </div>
  );
};
