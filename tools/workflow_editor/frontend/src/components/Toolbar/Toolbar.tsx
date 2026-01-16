/**
 * Toolbar with zoom controls and other actions
 */

import React, { useState } from 'react';
import { useWorkflowStore } from '../../store/useWorkflowStore';
import {
  ZoomIn,
  ZoomOut,
  Maximize,
  Undo,
  Redo,
  Save,
  Download,
  Upload,
  Trash2,
  Keyboard,
} from 'lucide-react';
import { KeyboardShortcutsHelp } from '../KeyboardShortcuts/KeyboardShortcuts';

export const Toolbar: React.FC = () => {
  const {
    zoomIn,
    zoomOut,
    resetZoom,
    fitToScreen,
    undo,
    redo,
    canUndo,
    canRedo,
    clearWorkflow,
    viewport,
    nodes,
    edges,
  } = useWorkflowStore();

  const [showHelp, setShowHelp] = useState(false);

  const handleExport = () => {
    const workflow = {
      nodes: nodes.map((node) => ({
        id: node.id,
        type: node.type,
        name: node.data.label || '',
        config: node.data,
        position: node.position,
      })),
      edges: edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        label: edge.label,
      })),
      metadata: {
        version: '1.0',
        exported_at: new Date().toISOString(),
      },
    };

    const blob = new Blob([JSON.stringify(workflow, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'workflow.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json,.yaml,.yml';
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      const text = await file.text();
      try {
        const workflow = JSON.parse(text);
        useWorkflowStore.getState().loadWorkflow(workflow);
      } catch (error) {
        console.error('Failed to import workflow:', error);
        alert('Failed to import workflow. Please check the file format.');
      }
    };
    input.click();
  };

  const handleSave = () => {
    // Save to localStorage
    const workflow = {
      nodes,
      edges,
      metadata: {
        version: '1.0',
        saved_at: new Date().toISOString(),
      },
    };
    localStorage.setItem('workflow_autosave', JSON.stringify(workflow));
    alert('Workflow saved successfully!');
  };

  const handleClear = () => {
    if (confirm('Are you sure you want to clear the workflow?')) {
      clearWorkflow();
    }
  };

  return (
    <>
      <div className="bg-white border-b border-slate-200 px-4 py-2 flex items-center justify-between">
        {/* Left section - Zoom controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={zoomOut}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            title="Zoom Out (Ctrl+-)"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-sm text-slate-600 min-w-[60px] text-center">
            {Math.round(viewport.zoom * 100)}%
          </span>
          <button
            onClick={zoomIn}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            title="Zoom In (Ctrl++)"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={resetZoom}
            className="px-3 py-1 text-sm hover:bg-slate-100 rounded-lg transition-colors"
            title="Reset Zoom (Ctrl+0)"
          >
            Reset
          </button>
          <button
            onClick={fitToScreen}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            title="Fit to Screen (Ctrl+9)"
          >
            <Maximize className="w-4 h-4" />
          </button>
        </div>

        {/* Center section - Undo/Redo and Stats */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <button
              onClick={undo}
              disabled={!canUndo()}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              title="Undo (Ctrl+Z)"
            >
              <Undo className="w-4 h-4" />
            </button>
            <button
              onClick={redo}
              disabled={!canRedo()}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              title="Redo (Ctrl+Y)"
            >
              <Redo className="w-4 h-4" />
            </button>
          </div>
          <div className="text-sm text-slate-600">
            {nodes.length} nodes, {edges.length} connections
          </div>
        </div>

        {/* Right section - Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleSave}
            className="flex items-center gap-2 px-3 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition-colors"
            title="Save workflow (Ctrl+S)"
          >
            <Save className="w-4 h-4" />
            Save
          </button>
          <button
            onClick={handleImport}
            className="flex items-center gap-2 px-3 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors"
            title="Import workflow"
          >
            <Upload className="w-4 h-4" />
            Import
          </button>
          <button
            onClick={handleExport}
            className="flex items-center gap-2 px-3 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors"
            title="Export workflow"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
          <button
            onClick={handleClear}
            className="p-2 hover:bg-red-100 text-red-600 rounded-lg transition-colors"
            title="Clear workflow"
          >
            <Trash2 className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowHelp(true)}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            title="Keyboard shortcuts (Ctrl+/)"
          >
            <Keyboard className="w-4 h-4" />
          </button>
        </div>
      </div>

      {showHelp && <KeyboardShortcutsHelp onClose={() => setShowHelp(false)} />}
    </>
  );
};
