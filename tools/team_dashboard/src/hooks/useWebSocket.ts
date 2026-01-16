import { useEffect, useRef, useState, useCallback } from 'react';
import type { DashboardEvent, WebSocketMessage } from '../types';

interface UseWebSocketOptions {
  onEvent?: (event: DashboardEvent) => void;
  onError?: (error: Event) => void;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface WebSocketState {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  reconnectCount: number;
}

/**
 * WebSocket hook for real-time team execution updates
 *
 * Provides automatic reconnection, event handling, and connection state management.
 *
 * @param url - WebSocket URL
 * @param options - Configuration options
 * @returns WebSocket state and send function
 */
export function useWebSocket(
  url: string,
  options: UseWebSocketOptions = {}
): {
  state: WebSocketState;
  send: (message: WebSocketMessage) => void;
  connect: () => void;
  disconnect: () => void;
} {
  const {
    onEvent,
    onError,
    onOpen,
    onClose,
    reconnectInterval = 3000,
    maxReconnectAttempts = 10,
  } = options;

  const [socketState, setSocketState] = useState<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    reconnectCount: 0,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const manuallyClosedRef = useRef(false);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setSocketState((prev) => ({ ...prev, isConnecting: true, error: null }));

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = (event) => {
        setSocketState({
          isConnected: true,
          isConnecting: false,
          error: null,
          reconnectCount: 0,
        });
        onOpen?.(event);

        // Send ping to keep connection alive
        const pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: 'ping' }));
          } else {
            clearInterval(pingInterval);
          }
        }, 30000);
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);

          // Handle pong
          if (message.action === 'pong') {
            return;
          }

          // Handle dashboard events
          if (message.event_type && message.timestamp) {
            const dashboardEvent: DashboardEvent = {
              event_type: message.event_type as any,
              execution_id: message.execution_id || '',
              timestamp: message.timestamp,
              data: message.data || {},
            };
            onEvent?.(dashboardEvent);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onerror = (event) => {
        setSocketState((prev) => ({
          ...prev,
          error: 'WebSocket error occurred',
          isConnecting: false,
        }));
        onError?.(event);
      };

      ws.onclose = (event) => {
        setSocketState((prev) => ({
          ...prev,
          isConnected: false,
          isConnecting: false,
        }));

        wsRef.current = null;
        onClose?.(event);

        // Auto-reconnect if not manually closed
        if (!manuallyClosedRef.current) {
          setSocketState((prev) => {
            const newReconnectCount = prev.reconnectCount + 1;
            if (newReconnectCount <= maxReconnectAttempts) {
              reconnectTimeoutRef.current = setTimeout(() => {
                connect();
              }, reconnectInterval);
              return {
                ...prev,
                reconnectCount: newReconnectCount,
                error: `Reconnecting... (${newReconnectCount}/${maxReconnectAttempts})`,
              };
            } else {
              return {
                ...prev,
                error: 'Max reconnection attempts reached',
              };
            }
          });
        }
      };
    } catch (error) {
      setSocketState((prev) => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to connect',
        isConnecting: false,
      }));
    }
  }, [url, onEvent, onError, onOpen, onClose, reconnectInterval, maxReconnectAttempts]);

  const disconnect = useCallback(() => {
    manuallyClosedRef.current = true;

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setSocketState({
      isConnected: false,
      isConnecting: false,
      error: null,
      reconnectCount: 0,
    });
  }, []);

  const send = useCallback(
    (message: WebSocketMessage) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(message));
      } else {
        console.warn('Cannot send message: WebSocket is not connected');
      }
    },
    []
  );

  // Auto-connect on mount
  useEffect(() => {
    manuallyClosedRef.current = false;
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    state: socketState,
    send,
    connect,
    disconnect,
  };
}
