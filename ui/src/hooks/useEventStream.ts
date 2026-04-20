/**
 * WebSocket hook for real-time event streaming
 *
 * Connects to the event WebSocket and provides:
 * - Automatic reconnection with exponential backoff
 * - Event filtering and state management
 * - Connection state tracking
 */

import { useState, useEffect, useRef, useCallback } from 'react';

type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error';

interface UseEventStreamOptions {
  wsUrl?: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface BridgeEvent {
  id: string;
  event_type: string;
  timestamp: string;
  session_id: string;
  data: Record<string, unknown>;
  tool_name?: string;
  severity?: string;
}

interface UseEventStreamResult {
  events: BridgeEvent[];
  connectionState: ConnectionState;
  reconnect: () => void;
  disconnect: () => void;
}

const MAX_EVENTS = 100; // Keep only the most recent events in memory

export function useEventStream({
  wsUrl = '/ws/events',
  reconnectInterval = 3000,
  maxReconnectAttempts = 10,
}: UseEventStreamOptions = {}): UseEventStreamResult {
  const [events, setEvents] = useState<BridgeEvent[]>([]);
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    setConnectionState('connecting');

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionState('connected');
        reconnectAttemptsRef.current = 0;

        // Subscribe to all events
        ws.send(
          JSON.stringify({
            type: 'subscribe',
            categories: ['*'], // Subscribe to all event categories
          })
        );
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as BridgeEvent;
          setEvents((prevEvents) => {
            const newEvents = [payload, ...prevEvents];
            // Keep only the most recent events
            return newEvents.slice(0, MAX_EVENTS);
          });
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionState('error');
      };

      ws.onclose = () => {
        setConnectionState('disconnected');
        wsRef.current = null;

        // Attempt reconnection with exponential backoff
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = reconnectInterval * Math.pow(2, reconnectAttemptsRef.current);
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++;
            connect();
          }, delay);
        }
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setConnectionState('error');
    }
  }, [wsUrl, reconnectInterval, maxReconnectAttempts]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionState('disconnected');
  }, []);

  const reconnect = useCallback(() => {
    disconnect();
    reconnectAttemptsRef.current = 0;
    connect();
  }, [connect, disconnect]);

  // Connect on mount
  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    events,
    connectionState,
    reconnect,
    disconnect,
  };
}
