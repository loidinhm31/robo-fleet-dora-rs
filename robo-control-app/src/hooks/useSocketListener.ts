import { useEffect, useCallback } from "react";
import { Socket } from "socket.io-client";

export interface UseSocketListenerOptions<T> {
  socket: Socket | null;
  event: string;
  onData: (data: T) => void;
  enabled?: boolean;
}

export const useSocketListener = <T>({
  socket,
  event,
  onData,
  enabled = true,
}: UseSocketListenerOptions<T>) => {
  const handler = useCallback(
    (data: T) => {
      if (enabled) {
        onData(data);
      }
    },
    [onData, enabled]
  );

  useEffect(() => {
    if (!socket || !enabled) return;

    socket.on(event, handler);

    return () => {
      socket.off(event, handler);
    };
  }, [socket, event, handler, enabled]);
};

export interface UseMultiSocketListenersOptions {
  socket: Socket | null;
  listeners: Array<{
    event: string;
    handler: (data: unknown) => void;
    enabled?: boolean;
  }>;
}

export const useMultiSocketListeners = ({
  socket,
  listeners,
}: UseMultiSocketListenersOptions) => {
  useEffect(() => {
    if (!socket) return;

    const activeListeners = listeners.filter((l) => l.enabled !== false);

    activeListeners.forEach(({ event, handler }) => {
      socket.on(event, handler);
    });

    return () => {
      activeListeners.forEach(({ event, handler }) => {
        socket.off(event, handler);
      });
    };
  }, [socket, listeners]);
};
