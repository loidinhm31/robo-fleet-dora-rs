import React from "react";
import { ValueDisplay } from "../atoms";

export interface SliderControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onChange: (value: number) => void;
  color?: string;
  disabled?: boolean;
  decimals?: number;
  showLimits?: boolean;
  className?: string;
}

export const SliderControl: React.FC<SliderControlProps> = ({
  label,
  value,
  min,
  max,
  step = 0.01,
  unit = "rad",
  onChange,
  color = "purple",
  disabled = false,
  decimals = 2,
  showLimits = true,
  className = "",
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(parseFloat(e.target.value));
  };

  return (
    <div className={`glass-card-light rounded-2xl p-3 md:p-4 space-y-2 ${className}`}>
      <ValueDisplay
        label={label}
        value={value}
        unit={unit}
        color={`text-${color}-300`}
        decimals={decimals}
      />
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleChange}
        disabled={disabled}
        className="glass-slider w-full"
      />
      {showLimits && (
        <div className="flex justify-between text-xs text-white/50 font-mono">
          <span>{min.toFixed(decimals)}</span>
          <span>0.00</span>
          <span>{max.toFixed(decimals)}</span>
        </div>
      )}
    </div>
  );
};
