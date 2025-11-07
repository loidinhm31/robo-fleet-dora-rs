import React from "react";

export interface ValueDisplayProps {
  label: string;
  value: number | string;
  unit?: string;
  color?: string;
  labelColor?: string;
  monospace?: boolean;
  decimals?: number;
  className?: string;
}

export const ValueDisplay: React.FC<ValueDisplayProps> = ({
  label,
  value,
  unit = "",
  color = "text-purple-300",
  labelColor = "text-white/70",
  monospace = true,
  decimals = 2,
  className = "",
}) => {
  const formattedValue = typeof value === "number"
    ? value.toFixed(decimals)
    : value;

  return (
    <div className={`flex justify-between items-center ${className}`}>
      <span className={`text-xs md:text-sm ${labelColor} capitalize`}>
        {label}
      </span>
      <span className={`text-xs md:text-sm ${monospace ? "font-mono" : ""} font-bold ${color}`}>
        {formattedValue} {unit}
      </span>
    </div>
  );
};
