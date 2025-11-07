import React from "react";
import { StatCard, StatCardProps } from "../atoms";

export interface StatPanelProps {
  stats: StatCardProps[];
  columns?: 2 | 3 | 4;
  className?: string;
}

const gridCols = {
  2: "grid-cols-2",
  3: "grid-cols-2 md:grid-cols-3",
  4: "grid-cols-2 md:grid-cols-4",
};

export const StatPanel: React.FC<StatPanelProps> = ({
  stats,
  columns = 3,
  className = "",
}) => {
  return (
    <div className={`grid ${gridCols[columns]} gap-3 md:gap-4 ${className}`}>
      {stats.map((stat, index) => (
        <StatCard key={`${stat.label}-${index}`} {...stat} />
      ))}
    </div>
  );
};
