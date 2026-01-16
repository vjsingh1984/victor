/**
 * Custom Team Node component with visual team representation
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Users } from 'lucide-react';

const TeamNode: React.FC<NodeProps> = ({ data, selected }) => {
  const memberCount = data.members?.length || 0;
  const formation = data.formation || 'parallel';

  return (
    <div
      className={`
        node-team
        px-4 py-3 rounded-lg border-2 min-w-[220px]
        ${selected ? 'ring-2 ring-purple-500 ring-offset-2' : ''}
      `}
      style={{
        backgroundColor: '#f3e5f5',
        borderColor: '#9c27b0',
      }}
    >
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-purple-600"
      />

      {/* Node content */}
      <div className="flex items-center gap-2 mb-2">
        <Users className="w-5 h-5 text-purple-600" />
        <span className="font-semibold text-sm text-slate-800">
          {data.label || 'Team'}
        </span>
        <span className="ml-auto text-xs bg-purple-200 text-purple-800 px-2 py-0.5 rounded-full">
          {memberCount} {memberCount === 1 ? 'member' : 'members'}
        </span>
      </div>

      {/* Formation badge */}
      <div className="mb-2">
        <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded font-medium">
          {formation}
        </span>
      </div>

      {/* Node details */}
      <div className="text-xs text-slate-600 space-y-1">
        {data.goal && (
          <div className="line-clamp-2">
            <span className="font-medium">Goal:</span>
            <span className="ml-1">{data.goal}</span>
          </div>
        )}
        {data.max_iterations !== undefined && (
          <div className="flex items-center gap-1">
            <span className="font-medium">Max iterations:</span>
            <span>{data.max_iterations}</span>
          </div>
        )}
        {data.timeout_seconds && (
          <div className="flex items-center gap-1">
            <span className="font-medium">Timeout:</span>
            <span>{data.timeout_seconds}s</span>
          </div>
        )}
      </div>

      {/* Team member avatars */}
      {memberCount > 0 && (
        <div className="mt-2 flex -space-x-2">
          {Array.from({ length: Math.min(memberCount, 4) }).map((_, i) => (
            <div
              key={i}
              className="w-6 h-6 rounded-full bg-purple-500 border-2 border-white flex items-center justify-center text-white text-xs font-medium"
            >
              {data.members?.[i]?.role?.[0]?.toUpperCase() || '?'}
            </div>
          ))}
          {memberCount > 4 && (
            <div className="w-6 h-6 rounded-full bg-purple-200 border-2 border-white flex items-center justify-center text-purple-700 text-xs font-medium">
              +{memberCount - 4}
            </div>
          )}
        </div>
      )}

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-purple-600"
      />
    </div>
  );
};

export default TeamNode;
