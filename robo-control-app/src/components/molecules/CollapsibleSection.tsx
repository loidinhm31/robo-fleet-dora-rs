import React, { ReactNode } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

export interface CollapsibleSectionProps {
  title: string;
  isExpanded: boolean;
  onToggle: () => void;
  children: ReactNode;
  headerRight?: ReactNode;
  className?: string;
  contentClassName?: string;
}

export const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  isExpanded,
  onToggle,
  children,
  headerRight,
  className = "",
  contentClassName = "",
}) => {
  return (
    <div className={`glass-card rounded-3xl p-4 md:p-6 space-y-4 ${className}`}>
      <div className="flex items-center justify-between">
        <button
          onClick={onToggle}
          className="flex items-center gap-2 text-lg md:text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 hover:from-purple-300 hover:to-pink-300 transition-all"
        >
          {title}
          {isExpanded ? (
            <ChevronUp className="w-5 h-5 text-purple-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-purple-400" />
          )}
        </button>
        {headerRight}
      </div>
      {isExpanded && (
        <div className={contentClassName}>
          {children}
        </div>
      )}
    </div>
  );
};
