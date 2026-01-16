import React from 'react';
import { Activity, Clock, Users, CheckCircle, XCircle, TrendingUp } from 'lucide-react';
import type { MetricsSummary } from '../types';

interface MetricsPanelProps {
  metrics: MetricsSummary | null;
  className?: string;
}

/**
 * Metric card component
 */
function MetricCard({
  icon: Icon,
  label,
  value,
  unit,
  color = 'blue',
}: {
  icon: React.ElementType;
  label: string;
  value: number | string;
  unit?: string;
  color?: 'blue' | 'green' | 'red' | 'yellow';
}) {
  const colors = {
    blue: 'bg-blue-50 text-blue-600',
    green: 'bg-green-50 text-green-600',
    red: 'bg-red-50 text-red-600',
    yellow: 'bg-yellow-50 text-yellow-600',
  };

  return (
    <div className="bg-white p-4 rounded-lg border border-gray-200">
      <div className="flex items-center gap-2 mb-2">
        <div className={`p-2 rounded-lg ${colors[color]}`}>
          <Icon size={20} />
        </div>
        <p className="text-sm text-gray-600">{label}</p>
      </div>
      <p className="text-2xl font-bold text-gray-900">
        {value}
        {unit && <span className="text-sm font-normal text-gray-500 ml-1">{unit}</span>}
      </p>
    </div>
  );
}

/**
 * Formation distribution bar
 */
function FormationDistribution({
  distribution,
}: {
  distribution: Record<string, number>;
}) {
  const total = Object.values(distribution).reduce((sum, count) => sum + count, 0);
  const entries = Object.entries(distribution).sort(([, a], [, b]) => b - a);

  if (total === 0) {
    return <p className="text-sm text-gray-500">No formation data</p>;
  }

  const colors = [
    'bg-blue-500',
    'bg-green-500',
    'bg-purple-500',
    'bg-yellow-500',
    'bg-red-500',
    'bg-pink-500',
  ];

  return (
    <div className="space-y-2">
      {entries.map(([formation, count], index) => {
        const percentage = total > 0 ? (count / total) * 100 : 0;
        const color = colors[index % colors.length];

        return (
          <div key={formation}>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium text-gray-700 capitalize">{formation}</span>
              <span className="text-gray-500">
                {count} ({percentage.toFixed(1)}%)
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`${color} h-2 rounded-full transition-all duration-500`}
                style={{ width: `${percentage}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Metrics panel component
 *
 * Displays aggregated team execution metrics including:
 * - Total teams executed
 * - Success/failure rates
 * - Average duration and member count
 * - Formation distribution
 * - Tool usage statistics
 */
export function MetricsPanel({ metrics, className = '' }: MetricsPanelProps) {
  if (!metrics) {
    return (
      <div
        className={`bg-white rounded-lg shadow p-6 border border-gray-200 ${className}`}
      >
        <div className="text-center text-gray-500">
          <Activity className="w-12 h-12 mx-auto mb-2 text-gray-300" />
          <p className="text-sm">No metrics data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-gray-600" />
          <h3 className="text-lg font-semibold text-gray-900">Team Metrics</h3>
        </div>
        <p className="text-sm text-gray-600 mt-1">
          Aggregated statistics across all team executions
        </p>
      </div>

      {/* Metrics grid */}
      <div className="p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <MetricCard
            icon={Activity}
            label="Total Teams"
            value={metrics.total_teams_executed}
          />
          <MetricCard
            icon={CheckCircle}
            label="Successful"
            value={metrics.successful_teams}
            color="green"
          />
          <MetricCard
            icon={XCircle}
            label="Failed"
            value={metrics.failed_teams}
            color="red"
          />
          <MetricCard
            icon={Users}
            label="Active Teams"
            value={metrics.active_teams}
            color="yellow"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <MetricCard
            icon={CheckCircle}
            label="Success Rate"
            value={(metrics.success_rate * 100).toFixed(1)}
            unit="%"
            color="green"
          />
          <MetricCard
            icon={Clock}
            label="Avg Duration"
            value={metrics.average_duration_seconds.toFixed(2)}
            unit="s"
          />
          <MetricCard
            icon={Users}
            label="Avg Members"
            value={metrics.average_member_count.toFixed(1)}
          />
        </div>

        {/* Formation distribution */}
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-900 mb-3">
            Formation Distribution
          </h4>
          <FormationDistribution distribution={metrics.formation_distribution} />
        </div>

        {/* Tool usage */}
        <div>
          <h4 className="text-sm font-semibold text-gray-900 mb-3">Tool Usage</h4>
          <div className="p-3 bg-gray-50 rounded-lg">
            <p className="text-2xl font-bold text-gray-900">
              {metrics.total_tool_calls.toLocaleString()}
            </p>
            <p className="text-sm text-gray-600">Total tool calls</p>
          </div>
        </div>
      </div>
    </div>
  );
}
