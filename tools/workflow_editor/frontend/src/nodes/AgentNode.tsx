/**
 * Custom Agent Node component
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Bot } from 'lucide-react';

const AgentNode: React.FC<NodeProps> = ({ data, selected }) => {
  return (
    <div
      className={`
        node-agent
        px-4 py-3 rounded-lg border-2 min-w-[200px]
        ${selected ? 'ring-2 ring-blue-500 ring-offset-2' : ''}
      `}
      style={{
        backgroundColor: '#e3f2fd',
        borderColor: '#2196f3',
      }}
    >
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-blue-600"
      />

      {/* Node content */}
      <div className="flex items-center gap-2 mb-2">
        <Bot className="w-5 h-5 text-blue-600" />
        <span className="font-semibold text-sm text-slate-800">
          {data.label || 'Agent'}
        </span>
      </div>

      {/* Node details */}
      <div className="text-xs text-slate-600 space-y-1">
        {data.role && (
          <div className="flex items-center gap-1">
            <span className="font-medium">Role:</span>
            <span>{data.role}</span>
          </div>
        )}
        {data.goal && (
          <div className="line-clamp-2">
            <span className="font-medium">Goal:</span>
            <span className="ml-1">{data.goal}</span>
          </div>
        )}
        {data.tool_budget !== undefined && (
          <div className="flex items-center gap-1">
            <span className="font-medium">Budget:</span>
            <span>{data.tool_budget}</span>
          </div>
        )}
      </div>

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-blue-600"
      />
    </div>
  );
};

export default AgentNode;
