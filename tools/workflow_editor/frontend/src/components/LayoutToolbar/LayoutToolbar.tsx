/**
 * Layout toolbar with auto-layout options
 */

import React from 'react';
import { useWorkflowStore } from '../../store/useWorkflowStore';
import { LayoutGrid, Network, TreeStructure } from 'lucide-react';

export const LayoutToolbar: React.FC = () => {
  const { applyLayout, nodes } = useWorkflowStore();

  const handleLayout = (type: 'hierarchical' | 'force-directed' | 'grid') => {
    if (nodes.length === 0) {
      alert('Add some nodes to the canvas first');
      return;
    }

    if (
      confirm(
        `Apply ${type} layout? This will reposition all nodes on the canvas.`
      )
    ) {
      applyLayout(type);
    }
  };

  return (
    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-white rounded-lg shadow-lg border border-slate-200 p-2 flex items-center gap-2 z-10">
      <span className="text-sm text-slate-600 px-2">Auto-Layout:</span>

      <button
        onClick={() => handleLayout('hierarchical')}
        className="flex items-center gap-2 px-3 py-2 hover:bg-slate-100 rounded-lg transition-colors"
        title="Hierarchical layout - Top to bottom flow"
      >
        <TreeStructure className="w-4 h-4" />
        <span className="text-sm">Hierarchical</span>
      </button>

      <button
        onClick={() => handleLayout('force-directed')}
        className="flex items-center gap-2 px-3 py-2 hover:bg-slate-100 rounded-lg transition-colors"
        title="Force-directed layout - Organic spacing"
      >
        <Network className="w-4 h-4" />
        <span className="text-sm">Force-Directed</span>
      </button>

      <button
        onClick={() => handleLayout('grid')}
        className="flex items-center gap-2 px-3 py-2 hover:bg-slate-100 rounded-lg transition-colors"
        title="Grid layout - Organized grid"
      >
        <LayoutGrid className="w-4 h-4" />
        <span className="text-sm">Grid</span>
      </button>
    </div>
  );
};
