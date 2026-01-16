/**
 * Main App component with advanced features
 */

import React, { useCallback, useEffect, useState } from 'react';
import ReactFlow, {
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap as ReactFlowMiniMap,
  useReactFlow,
} from 'reactflow';
import { useWorkflowStore } from './store/useWorkflowStore';
import WorkflowCanvas from './components/Canvas/WorkflowCanvas';
import NodePalette from './components/NodePalette/NodePalette';
import PropertyPanel from './components/PropertyPanel/PropertyPanel';
import YAMLPreview from './components/YAMLPreview/YAMLPreview';
import TeamNodeEditor from './components/TeamNodeEditor/TeamNodeEditor';
import Toolbar from './components/Toolbar/Toolbar';
import SearchBar from './components/SearchBar/SearchBar';
import { KeyboardShortcuts, KeyboardShortcutsHelp } from './components/KeyboardShortcuts/KeyboardShortcuts';
import { TemplateLibrary, TemplateLibraryButton } from './components/TemplateLibrary/TemplateLibrary';
import LayoutToolbar from './components/LayoutToolbar/LayoutToolbar';
import { NodeGroupingControls } from './components/NodeGrouping/NodeGrouping';
import MiniMap from './components/MiniMap/MiniMap';
import { Download, Upload, Play, FileText, Folder } from 'lucide-react';
import './index.css';

const App: React.FC = () => {
  const {
    nodes,
    edges,
    selectedNode,
    addNode,
    setNodes,
    setEdges,
    selectNode,
    validationErrors,
    validationWarnings,
    clearWorkflow,
    loadWorkflow,
  } = useWorkflowStore();

  const [showTeamEditor, setShowTeamEditor] = useState(false);
  const [showYAML, setShowYAML] = useState(false);
  const [showTemplates, setShowTemplates] = useState(false);
  const [showHelp, setShowHelp] = useState(false);

  // Handle drop from palette
  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/reactflow');
      if (!type) return;

      // Calculate position accounting for zoom and pan
      const reactFlowBounds = (event.target as Element)
        .getBoundingClientRect()
        .toJSON();
      const position = {
        x: event.clientX - reactFlowBounds.left - 75, // Approximate node width/2
        y: event.clientY - reactFlowBounds.top - 20, // Approximate node height/2
      };

      const newNode = {
        id: `${type}_${Date.now()}`,
        type,
        position,
        data: {
          label: type.charAt(0).toUpperCase() + type.slice(1),
        },
      };

      addNode(newNode);
    },
    [addNode]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Open team editor when team node is selected
  useEffect(() => {
    if (selectedNode?.type === 'team') {
      setShowTeamEditor(true);
    }
  }, [selectedNode]);

  // Handle export
  useEffect(() => {
    const handleExportEvent = () => {
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

    window.addEventListener('workflow-export', handleExportEvent);
    return () => window.removeEventListener('workflow-export', handleExportEvent);
  }, [nodes, edges]);

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
        loadWorkflow(workflow);
      } catch (error) {
        console.error('Failed to import workflow:', error);
        alert('Failed to import workflow. Please check the file format.');
      }
    };
    input.click();
  };

  const handleClear = () => {
    if (confirm('Are you sure you want to clear the workflow?')) {
      clearWorkflow();
    }
  };

  const handleInsertTemplate = (template: any) => {
    // Offset positions to avoid overlap with existing nodes
    const offsetX = Math.max(...nodes.map((n) => n.position.x)) + 300 || 100;
    const offsetY = Math.max(...nodes.map((n) => n.position.y)) + 200 || 100;

    const newNodes = template.nodes.map((node: any) => ({
      ...node,
      id: `${node.id}_${Date.now()}`,
      position: {
        x: node.position.x + offsetX,
        y: node.position.y + offsetY,
      },
    }));

    const newEdges = template.edges.map((edge: any) => ({
      ...edge,
      id: `${edge.id}_${Date.now()}`,
      source: `${edge.source}_${Date.now()}`,
      target: `${edge.target}_${Date.now()}`,
    }));

    setNodes([...nodes, ...newNodes]);
    setEdges([...edges, ...newEdges]);
  };

  return (
    <ReactFlowProvider>
      <div className="app">
        {/* Header */}
        <header className="app-header">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-xl">V</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-800">
                Victor Workflow Editor
              </h1>
              <p className="text-sm text-slate-600">
                Visual workflow builder with team support
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <TemplateLibraryButton onClick={() => setShowTemplates(!showTemplates)} />
            <button
              onClick={() => setShowYAML(!showYAML)}
              className="flex items-center gap-2 px-3 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors"
              title="Toggle YAML preview"
            >
              <FileText className="w-4 h-4" />
              YAML
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
              onClick={handleClear}
              className="px-3 py-2 bg-red-100 hover:bg-red-200 text-red-700 rounded-lg transition-colors"
            >
              Clear
            </button>
          </div>
        </header>

        {/* Toolbar */}
        <Toolbar />

        {/* Search Bar */}
        <SearchBar />

        {/* Main content */}
        <div className="app-content">
          {/* Left sidebar - Node Palette */}
          <NodePalette />

          {/* Center - Canvas */}
          <div className="canvas-container">
            {validationErrors.length > 0 && (
              <div className="validation-banner validation-errors">
                <strong>Errors:</strong> {validationErrors.join(', ')}
              </div>
            )}
            {validationWarnings.length > 0 && (
              <div className="validation-banner validation-warnings">
                <strong>Warnings:</strong> {validationWarnings.join(', ')}
              </div>
            )}

            <div
              className="canvas-drop-zone"
              onDrop={onDrop}
              onDragOver={onDragOver}
            >
              <WorkflowCanvas />
              <LayoutToolbar />
              <NodeGroupingControls />
            </div>
          </div>

          {/* Right sidebar - Property Panel */}
          <PropertyPanel />
        </div>

        {/* MiniMap */}
        <MiniMap />

        {/* YAML Preview Panel */}
        {showYAML && <YAMLPreview onClose={() => setShowYAML(false)} />}

        {/* Team Node Editor Modal */}
        {showTeamEditor && (
          <TeamNodeEditor onClose={() => setShowTeamEditor(false)} />
        )}

        {/* Template Library Sidebar */}
        {showTemplates && (
          <TemplateLibrary
            onClose={() => setShowTemplates(false)}
            onInsertTemplate={handleInsertTemplate}
          />
        )}

        {/* Keyboard Shortcuts Help */}
        {showHelp && <KeyboardShortcutsHelp onClose={() => setShowHelp(false)} />}

        {/* Keyboard Shortcuts Handler */}
        <KeyboardShortcuts onHelp={() => setShowHelp(true)} />
      </div>
    </ReactFlowProvider>
  );
};

export default App;
