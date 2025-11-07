import React from "react";
import { LucideIcon } from "lucide-react";

export interface IconBadgeProps {
  icon: LucideIcon;
  label?: string;
  color?: string;
  size?: "sm" | "md" | "lg";
  animated?: boolean;
  className?: string;
}

const sizeConfig = {
  sm: {
    iconSize: "w-3 h-3",
    textSize: "text-xs",
    padding: "px-2 py-1",
  },
  md: {
    iconSize: "w-4 h-4",
    textSize: "text-sm",
    padding: "px-3 py-1.5",
  },
  lg: {
    iconSize: "w-5 h-5",
    textSize: "text-base",
    padding: "px-4 py-2",
  },
};

export const IconBadge: React.FC<IconBadgeProps> = ({
  icon: Icon,
  label,
  color = "text-cyan-400",
  size = "md",
  animated = false,
  className = "",
}) => {
  const sizes = sizeConfig[size];

  return (
    <div className={`glass-card-light rounded-lg ${sizes.padding} flex items-center gap-2 ${className}`}>
      <Icon className={`${sizes.iconSize} ${color} ${animated ? "animate-pulse" : ""}`} />
      {label && <span className={`${sizes.textSize} font-semibold ${color}`}>{label}</span>}
    </div>
  );
};
