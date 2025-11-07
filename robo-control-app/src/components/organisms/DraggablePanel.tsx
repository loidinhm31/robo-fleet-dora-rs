import React, { ReactNode } from "react";
import { useDraggableWindow } from "../../hooks";
import { GripHorizontal, X, Maximize2, Minimize2 } from "lucide-react";

export interface DraggablePanelProps {
  title: string;
  children: ReactNode;
  isVisible: boolean;
  onToggleVisible: () => void;
  onClose?: () => void;
  collapsedContent?: ReactNode;
  initialPosition?: { x: number; y: number };
  className?: string;
  contentClassName?: string;
  showControls?: boolean;
}

export const DraggablePanel: React.FC<DraggablePanelProps> = ({
  title,
  children,
  isVisible,
  onToggleVisible,
  onClose,
  collapsedContent,
  initialPosition,
  className = "",
  contentClassName = "",
  showControls = true,
}) => {
  const { position, isDragged, handleMouseDown } = useDraggableWindow({
    initialPosition,
  });

  const centerX = position.x === 0 ? 'left-1/2 -translate-x-1/2' : '';
  const positionStyle = position.x !== 0 ? {
    left: `${position.x}px`,
    top: `${position.y}px`,
    transform: 'none',
  } : {
    top: `${position.y}px`,
  };

  if (!isVisible && collapsedContent) {
    return (
      <div
        className={`fixed z-40 ${centerX}`}
        style={positionStyle}
        onMouseDown={handleMouseDown}
      >
        <div
          onClick={() => {
            if (!isDragged) {
              onToggleVisible();
            }
          }}
        >
          {collapsedContent}
        </div>
      </div>
    );
  }

  if (!isVisible) {
    return null;
  }

  return (
    <div
      className={`fixed z-40 ${centerX}`}
      style={positionStyle}
      onMouseDown={handleMouseDown}
    >
      <div className={`bg-slate-900/95 backdrop-blur-md border border-slate-700/50 rounded-2xl shadow-2xl max-w-sm md:max-w-md lg:max-w-lg ${className}`}>
        {/* Header */}
        <div className="drag-handle cursor-move flex items-center justify-between px-4 py-3 border-b border-slate-700/50">
          <div className="flex items-center gap-2">
            <GripHorizontal className="w-4 h-4 text-slate-400" />
            <h3 className="font-bold text-white">{title}</h3>
          </div>
          {showControls && (
            <div className="flex items-center gap-2">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleVisible();
                }}
                className="p-1 hover:bg-slate-700/50 rounded-lg transition-colors"
                title={isVisible ? "Minimize" : "Maximize"}
              >
                {isVisible ? (
                  <Minimize2 className="w-4 h-4 text-slate-400" />
                ) : (
                  <Maximize2 className="w-4 h-4 text-slate-400" />
                )}
              </button>
              {onClose && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onClose();
                  }}
                  className="p-1 hover:bg-red-500/20 rounded-lg transition-colors"
                  title="Close"
                >
                  <X className="w-4 h-4 text-red-400" />
                </button>
              )}
            </div>
          )}
        </div>

        {/* Content */}
        <div className={`p-4 ${contentClassName}`}>
          {children}
        </div>
      </div>
    </div>
  );
};
