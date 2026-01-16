import React from 'react';
import { Clock, CheckCircle, XCircle, Loader, AlertCircle } from 'lucide-react';
import type { MemberState, MemberStatus } from '../types';
import { formatDistanceToNow } from 'date-fns';

interface MemberStatusCardProps {
  member: MemberState;
  className?: string;
}

/**
 * Status icon component
 */
function StatusIcon({ status }: { status: MemberStatus }) {
  switch (status) {
    case 'running':
      return <Loader className="w-5 h-5 text-blue-500 animate-spin" />;
    case 'completed':
      return <CheckCircle className="w-5 h-5 text-green-500" />;
    case 'failed':
      return <XCircle className="w-5 h-5 text-red-500" />;
    case 'waiting':
      return <AlertCircle className="w-5 h-5 text-yellow-500" />;
    default:
      return <Clock className="w-5 h-5 text-gray-400" />;
  }
}

/**
 * Status badge component
 */
function StatusBadge({ status }: { status: MemberStatus }) {
  const colors: Record<MemberStatus, string> = {
    idle: 'bg-gray-100 text-gray-800',
    running: 'bg-blue-100 text-blue-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
    waiting: 'bg-yellow-100 text-yellow-800',
  };

  return (
    <span className={`px-2 py-1 text-xs font-medium rounded-full ${colors[status]}`}>
      {status}
    </span>
  );
}

/**
 * Member status card component
 *
 * Displays real-time status of a team member including:
 * - Current status with visual indicator
 * - Execution duration
 * - Tool usage statistics
 * - Error messages (if any)
 */
export function MemberStatusCard({ member, className = '' }: MemberStatusCardProps) {
  const startTime = member.start_time ? new Date(member.start_time) : null;
  const endTime = member.end_time ? new Date(member.end_time) : null;

  return (
    <div className={`bg-white rounded-lg shadow p-4 border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <StatusIcon status={member.status} />
          <h3 className="font-semibold text-gray-900">{member.member_id}</h3>
        </div>
        <StatusBadge status={member.status} />
      </div>

      {/* Role */}
      {member.role && (
        <p className="text-sm text-gray-600 mb-3">
          Role: <span className="font-medium">{member.role}</span>
        </p>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 mb-3">
        {/* Duration */}
        <div>
          <p className="text-xs text-gray-500 mb-1">Duration</p>
          <p className="text-sm font-medium text-gray-900">
            {member.duration_seconds > 0
              ? `${member.duration_seconds.toFixed(2)}s`
              : startTime
              ? formatDistanceToNow(startTime, { addSuffix: true })
              : '-'}
          </p>
        </div>

        {/* Tool calls */}
        <div>
          <p className="text-xs text-gray-500 mb-1">Tool Calls</p>
          <p className="text-sm font-medium text-gray-900">{member.tool_calls_used}</p>
        </div>
      </div>

      {/* Tools used */}
      {member.tools_used.length > 0 && (
        <div className="mb-3">
          <p className="text-xs text-gray-500 mb-1">Tools Used</p>
          <div className="flex flex-wrap gap-1">
            {member.tools_used.slice(0, 5).map((tool) => (
              <span
                key={tool}
                className="px-2 py-0.5 text-xs bg-gray-100 text-gray-700 rounded"
              >
                {tool}
              </span>
            ))}
            {member.tools_used.length > 5 && (
              <span className="px-2 py-0.5 text-xs bg-gray-100 text-gray-700 rounded">
                +{member.tools_used.length - 5}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Error message */}
      {member.error_message && (
        <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded">
          <p className="text-xs text-red-700">{member.error_message}</p>
        </div>
      )}

      {/* Timestamps */}
      <div className="mt-3 pt-3 border-t border-gray-200">
        {startTime && (
          <p className="text-xs text-gray-500">
            Started: {startTime.toLocaleString()}
          </p>
        )}
        {endTime && (
          <p className="text-xs text-gray-500">
            Ended: {endTime.toLocaleString()}
          </p>
        )}
      </div>
    </div>
  );
}
