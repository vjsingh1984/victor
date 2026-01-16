/**
 * Custom Condition Node component
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { GitBranch } from 'lucide-react';

const ConditionNode: React.FC<NodeProps> = ({ data, selected }) => {
  return (
    <div
      className={`
        node-condition
        px-4 py-3 rounded-lg border-2 min-w-[200px]
        ${selected ? 'ring-2 ring-orange-500 ring-offset-2' : ''}
      `}
      style={{
        backgroundColor: '#fff3e0',
        borderColor: '#ff9800',
      }}
    >
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-orange-600"
      />

      {/* Node content */}
      <div className="flex items-center gap-2 mb-2">
        <GitBranch className="w-5 h-5 text-orange-600" />
        <span className="font-semibold text-sm text-slate-800">
          {data.label || 'Condition'}
        </span>
      </div>

      {/* Node details */}
      <div className="text-xs text-slate-600 space-y-1">
        {data.condition && (
          <div className="line-clamp-2">
            <span className="font-medium">Condition:</span>
            <span className="ml-1 font-mono">{data.condition}</span>
          </div>
        )}
        {data.branches && (
          <div>
            <span className="font-medium">Branches:</span>
            <span className="ml-1">
              {Object.keys(data.branches).join(', ')}
            </span>
          </div>
        )}
      </div>

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-orange-600"
      />
    </div>
  );
};

export default ConditionNode;
