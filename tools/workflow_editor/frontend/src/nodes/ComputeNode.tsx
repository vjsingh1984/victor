/**
 * Custom Compute Node component
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Cpu } from 'lucide-react';

const ComputeNode: React.FC<NodeProps> = ({ data, selected }) => {
  return (
    <div
      className={`
        node-compute
        px-4 py-3 rounded-lg border-2 min-w-[200px]
        ${selected ? 'ring-2 ring-green-500 ring-offset-2' : ''}
      `}
      style={{
        backgroundColor: '#e8f5e9',
        borderColor: '#4caf50',
      }}
    >
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-green-600"
      />

      {/* Node content */}
      <div className="flex items-center gap-2 mb-2">
        <Cpu className="w-5 h-5 text-green-600" />
        <span className="font-semibold text-sm text-slate-800">
          {data.label || 'Compute'}
        </span>
      </div>

      {/* Node details */}
      <div className="text-xs text-slate-600 space-y-1">
        {data.handler && (
          <div className="flex items-center gap-1">
            <span className="font-medium">Handler:</span>
            <span className="font-mono">{data.handler}</span>
          </div>
        )}
        {data.tools && data.tools.length > 0 && (
          <div>
            <span className="font-medium">Tools:</span>
            <span className="ml-1">{data.tools.join(', ')}</span>
          </div>
        )}
        {data.timeout !== undefined && (
          <div className="flex items-center gap-1">
            <span className="font-medium">Timeout:</span>
            <span>{data.timeout}s</span>
          </div>
        )}
      </div>

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-green-600"
      />
    </div>
  );
};

export default ComputeNode;
