import React, { useEffect, useState } from 'react';
import { AlertCircle, RefreshCw, Wifi, WifiOff } from 'lucide-react';
import { useDashboardStore } from '../store/dashboardStore';
import { useWebSocket } from '../hooks/useWebSocket';
import { MemberStatusCard } from './MemberStatusCard';
import { CommunicationFlow } from './CommunicationFlow';
import { SharedContextTable } from './SharedContextTable';
import { NegotiationPanel } from './NegotiationPanel';
import { MetricsPanel } from './MetricsPanel';
import type { DashboardEvent } from '../types';

interface TeamExecutionViewProps {
  executionId: string;
  wsUrl?: string;
  className?: string;
}

/**
 * Connection status indicator
 */
function ConnectionStatus({ isConnected }: { isConnected: boolean }) {
  return (
    <div className="flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium">
      {isConnected ? (
        <>
          <Wifi className="w-4 h-4 text-green-500" />
          <span className="text-green-700">Connected</span>
        </>
      ) : (
        <>
          <WifiOff className="w-4 h-4 text-red-500" />
          <span className="text-red-700">Disconnected</span>
        </>
      )}
    </div>
  );
}

/**
 * Main team execution view component
 *
 * Displays real-time team execution with:
 * - Member status cards
 * - Communication flow diagram
 * - Shared context table
 * - Negotiation panel
 * - Metrics summary
 * - WebSocket connection status
 */
export function TeamExecutionView({
  executionId,
  wsUrl = `ws://localhost:8000/ws/team/${executionId}`,
  className = '',
}: TeamExecutionViewProps) {
  const [error, setError] = useState<string | null>(null);

  // Get data from store
  const execution = useDashboardStore((state) => state.executions[executionId]);
  const metrics = useDashboardStore((state) => state.metrics);
  const setErrorStore = useDashboardStore((state) => state.setError);
  const setLoading = useDashboardStore((state) => state.setLoading);
  const handleDashboardEvent = useDashboardStore((state) => state.handleDashboardEvent);
  const setMetrics = useDashboardStore((state) => state.setMetrics);
  const updateExecution = useDashboardStore((state) => state.updateExecution);

  // WebSocket connection
  const { state: wsState, send } = useWebSocket(wsUrl, {
    onEvent: (event: DashboardEvent) => {
      console.log('Received event:', event);
      handleDashboardEvent(event);
    },
    onError: (errorEvent) => {
      console.error('WebSocket error:', errorEvent);
      setError('WebSocket connection error');
      setErrorStore('Connection error');
    },
    onOpen: () => {
      console.log('WebSocket connected');
      setError(null);
      setErrorStore(null);
      // Request initial state
      send({ action: 'query_state', execution_id: executionId });
    },
    onClose: () => {
      console.log('WebSocket disconnected');
      setError('Disconnected from server');
    },
  });

  // Fetch initial data
  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        // Fetch execution details
        const execResponse = await fetch(`/api/v1/executions/${executionId}`);
        if (!execResponse.ok) {
          throw new Error('Execution not found');
        }
        const execData = await execResponse.json();
        updateExecution(executionId, execData);

        // Fetch metrics
        const metricsResponse = await fetch('/api/v1/metrics/summary');
        if (metricsResponse.ok) {
          const metricsData = await metricsResponse.json();
          setMetrics(metricsData);
        }

        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
        setErrorStore(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, [executionId]);

  // Refresh data
  const handleRefresh = () => {
    send({ action: 'query_state', execution_id: executionId });
  };

  if (!execution && !error) {
    return (
      <div className={`flex items-center justify-center p-12 ${className}`}>
        <div className="text-center">
          <RefreshCw className="w-8 h-8 text-blue-500 animate-spin mx-auto mb-2" />
          <p className="text-gray-600">Loading execution data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-red-50 border border-red-200 rounded-lg p-6 ${className}`}>
        <div className="flex items-start gap-3">
          <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-semibold text-red-900 mb-1">Error</h3>
            <p className="text-sm text-red-700">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!execution) {
    return null;
  }

  const members = Object.values(execution.member_states);

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="bg-white rounded-lg shadow p-4 border border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{execution.team_id}</h1>
            <p className="text-sm text-gray-600 mt-1">
              Formation: <span className="font-medium capitalize">{execution.formation}</span>
              {' • '}
              Depth: <span className="font-medium">{execution.recursion_depth}</span>
            </p>
          </div>
          <div className="flex items-center gap-4">
            <ConnectionStatus isConnected={wsState.isConnected} />
            <button
              onClick={handleRefresh}
              className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg"
              title="Refresh"
            >
              <RefreshCw className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Error message */}
      {wsState.error && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
          <p className="text-sm text-yellow-800">{wsState.error}</p>
        </div>
      )}

      {/* Metrics summary */}
      {metrics && <MetricsPanel metrics={metrics} />}

      {/* Member status cards */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-3">Team Members</h2>
        {members.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-6 border border-gray-200 text-center text-gray-500">
            No team members
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {members.map((member) => (
              <MemberStatusCard key={member.member_id} member={member} />
            ))}
          </div>
        )}
      </div>

      {/* Two-column layout for larger components */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Communication flow */}
        <CommunicationFlow
          logs={execution.communication_logs}
          members={execution.member_states}
        />

        {/* Negotiation panel */}
        <NegotiationPanel status={execution.negotiation_status} />
      </div>

      {/* Shared context */}
      <SharedContextTable context={execution.shared_context} />
    </div>
  );
}
