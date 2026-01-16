/**
 * Custom Transform Node component
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { ArrowRight } from 'lucide-react';

const TransformNode: React.FC<NodeProps> = ({ data, selected }) => {
  return (
    <div
      className={`
        node-transform
        px-4 py-3 rounded-lg border-2 min-w-[200px]
        ${selected ? 'ring-2 ring-slate-500 ring-offset-2' : ''}
      `}
      style={{
        backgroundColor: '#eceff1',
        borderColor: '#607d8b',
      }}
    >
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-slate-600"
      />

      {/* Node content */}
      <div className="flex items-center gap-2 mb-2">
        <ArrowRight className="w-5 h-5 text-slate-600" />
        <span className="font-semibold text-sm text-slate-800">
          {data.label || 'Transform'}
        </span>
      </div>

      {/* Node details */}
      <div className="text-xs text-slate-600">
        {data.transform && (
          <div className="line-clamp-3 font-mono bg-slate-100 p-2 rounded">
            {data.transform}
          </div>
        )}
      </div>

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-slate-600"
      />
    </div>
  );
};

export default TransformNode;
