/**
 * Live Dashboard - Real-time event stream and metrics
 *
 * Displays:
 * - Real-time event stream via WebSocket
 * - Live metrics cards
 * - Active sessions list
 * - Recent alerts
 */

import { useState, useEffect, useCallback } from 'react';
import { useEventStream } from '../../hooks/useEventStream';

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  trend?: number;
  icon: string;
}

interface LiveDashboardProps {
  wsUrl?: string;
}

export function LiveDashboard({ wsUrl = '/ws/events' }: LiveDashboardProps) {
  const { events, connectionState } = useEventStream({ wsUrl });
  const [metrics, setMetrics] = useState({
    toolCallsTotal: 0,
    toolCallsSuccess: 0,
    toolCallsError: 0,
    activeSessions: 0,
    errorRate: 0,
  });

  // Fetch metrics on mount and update periodically
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch('/obs/metrics/summary');
        if (response.ok) {
          const data = await response.json();
          setMetrics(data.metrics);
        }
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="live-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <h2>Live Dashboard</h2>
        <div className="connection-status">
          <span className={`status-indicator ${connectionState}`}>
            {connectionState === 'connected' ? '●' : '○'}
          </span>
          <span className="status-text">
            {connectionState === 'connected' ? 'Connected' : 'Connecting...'}
          </span>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="metrics-grid">
        <MetricCard
          title="Tool Calls"
          value={metrics.toolCallsTotal}
          icon="🔧"
        />
        <MetricCard
          title="Success Rate"
          value={metrics.toolCallsTotal > 0
            ? ((metrics.toolCallsSuccess / metrics.toolCallsTotal) * 100).toFixed(1)
            : '0'}
          unit="%"
          icon="✅"
        />
        <MetricCard
          title="Errors"
          value={metrics.toolCallsError}
          icon="❌"
        />
        <MetricCard
          title="Active Sessions"
          value={metrics.activeSessions}
          icon="👥"
        />
      </div>

      {/* Event Stream */}
      <div className="event-stream-section">
        <h3>Recent Events</h3>
        <EventStream events={events.slice(0, 50)} />
      </div>
    </div>
  );
}

function MetricCard({ title, value, unit, icon }: MetricCardProps) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <div className="metric-content">
        <div className="metric-title">{title}</div>
        <div className="metric-value">
          {value} {unit && <span className="metric-unit">{unit}</span>}
        </div>
      </div>
    </div>
  );
}

interface EventStreamProps {
  events: Array<{
    id: string;
    event_type: string;
    timestamp: string;
    data: Record<string, unknown>;
  }>;
}

function EventStream({ events }: EventStreamProps) {
  if (events.length === 0) {
    return (
      <div className="event-stream empty">
        <div className="empty-state">Waiting for events...</div>
      </div>
    );
  }

  return (
    <div className="event-stream">
      {events.map((event) => (
        <div key={event.id} className="event-item">
          <div className="event-header">
            <span className="event-type">{event.event_type}</span>
            <span className="event-timestamp">
              {new Date(event.timestamp).toLocaleTimeString()}
            </span>
          </div>
          <div className="event-data">
            {formatEventData(event)}
          </div>
        </div>
      ))}
    </div>
  );
}

function formatEventData(event: { event_type: string; data: Record<string, unknown> }) {
  const { event_type, data } = event;

  switch (event_type) {
    case 'TOOL_CALL':
      return (
        <div>
          <span className="data-label">Tool:</span>{' '}
          <span className="data-value">{String(data.tool_name || 'unknown')}</span>
          {data.arguments && (
            <pre className="data-arguments">
              {JSON.stringify(data.arguments, null, 2)}
            </pre>
          )}
        </div>
      );

    case 'CONTENT':
      return (
        <div>
          <span className="data-label">Role:</span>{' '}
          <span className="data-value">{String(data.role || 'unknown')}</span>
          {data.content && (
            <div className="data-content">
              {String(data.content).slice(0, 200)}
              {String(data.content).length > 200 && '...'}
            </div>
          )}
        </div>
      );

    case 'ERROR':
      return (
        <div className="event-error">
          <span className="data-label">Error:</span>{' '}
          <span className="data-value">{String(data.message || 'Unknown error')}</span>
        </div>
      );

    default:
      return (
        <pre className="data-raw">
          {JSON.stringify(data, null, 2).slice(0, 200)}
        </pre>
      );
  }
}
