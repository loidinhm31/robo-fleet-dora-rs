import { useCallback, useEffect, useState } from "react";

export interface Position {
  x: number;
  y: number;
}

export interface UseDraggableWindowOptions {
  initialPosition?: Position;
  dragThreshold?: number;
}

export interface UseDraggableWindowReturn {
  position: Position;
  isDragging: boolean;
  isDragged: boolean;
  handleMouseDown: (e: React.MouseEvent) => void;
  resetPosition: () => void;
}

export const useDraggableWindow = (
  options: UseDraggableWindowOptions = {}
): UseDraggableWindowReturn => {
  const {
    initialPosition = { x: 0, y: 80 },
    dragThreshold = 5,
  } = options;

  const [position, setPosition] = useState<Position>(initialPosition);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState<Position>({ x: 0, y: 0 });
  const [dragStart, setDragStart] = useState<Position>({ x: 0, y: 0 });
  const [hasMoved, setHasMoved] = useState(false);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    if (target.closest('.drag-handle')) {
      setIsDragging(true);
      setHasMoved(false);
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      setDragOffset({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      });
      setDragStart({
        x: e.clientX,
        y: e.clientY,
      });
    }
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        const dx = Math.abs(e.clientX - dragStart.x);
        const dy = Math.abs(e.clientY - dragStart.y);
        if (dx > dragThreshold || dy > dragThreshold) {
          setHasMoved(true);
        }
        setPosition({
          x: e.clientX - dragOffset.x,
          y: e.clientY - dragOffset.y,
        });
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, dragOffset, dragStart, dragThreshold]);

  const resetPosition = useCallback(() => {
    setPosition(initialPosition);
    setHasMoved(false);
  }, [initialPosition]);

  return {
    position,
    isDragging,
    isDragged: hasMoved,
    handleMouseDown,
    resetPosition,
  };
};
