/**
 * Property Panel - Edit selected node properties
 */

import React, { useState, useEffect } from 'react';
import { useWorkflowStore } from '../../store/useWorkflowStore';
import { Settings } from 'lucide-react';
import './PropertyPanel.css';

const PropertyPanel: React.FC = () => {
  const { selectedNode, updateNode, deleteNode } = useWorkflowStore();
  const [properties, setProperties] = useState<Record<string, unknown>>({});

  useEffect(() => {
    if (selectedNode?.data) {
      setProperties(selectedNode.data);
    }
  }, [selectedNode]);

  if (!selectedNode) {
    return (
      <div className="property-panel property-panel-empty">
        <div className="property-panel-empty-state">
          <Settings className="w-12 h-12 text-slate-300" />
          <p className="text-slate-500 mt-2">Select a node to edit its properties</p>
        </div>
      </div>
    );
  }

  const handlePropertyChange = (key: string, value: unknown) => {
    const updated = { ...properties, [key]: value };
    setProperties(updated);
    updateNode(selectedNode.id, updated);
  };

  const handleDelete = () => {
    if (confirm('Delete this node?')) {
      deleteNode(selectedNode.id);
    }
  };

  return (
    <div className="property-panel">
      <div className="property-panel-header">
        <h2 className="property-panel-title">Properties</h2>
        <button
          onClick={handleDelete}
          className="property-panel-delete"
          title="Delete node"
        >
          Delete
        </button>
      </div>

      <div className="property-panel-content">
        {/* Node ID */}
        <div className="property-group">
          <label className="property-label">Node ID</label>
          <input
            type="text"
            value={selectedNode.id}
            disabled
            className="property-input property-input-disabled"
          />
        </div>

        {/* Node Type */}
        <div className="property-group">
          <label className="property-label">Node Type</label>
          <input
            type="text"
            value={selectedNode.type}
            disabled
            className="property-input property-input-disabled"
          />
        </div>

        {/* Node Label */}
        <div className="property-group">
          <label className="property-label">Label</label>
          <input
            type="text"
            value={(properties.label as string) || ''}
            onChange={(e) => handlePropertyChange('label', e.target.value)}
            className="property-input"
            placeholder="Node label"
          />
        </div>

        {/* Custom properties based on node type */}
        {selectedNode.type === 'agent' && (
          <>
            <div className="property-group">
              <label className="property-label">Role</label>
              <select
                value={(properties.role as string) || 'assistant'}
                onChange={(e) => handlePropertyChange('role', e.target.value)}
                className="property-input"
              >
                <option value="assistant">Assistant</option>
                <option value="researcher">Researcher</option>
                <option value="planner">Planner</option>
                <option value="executor">Executor</option>
                <option value="reviewer">Reviewer</option>
                <option value="writer">Writer</option>
              </select>
            </div>

            <div className="property-group">
              <label className="property-label">Goal</label>
              <textarea
                value={(properties.goal as string) || ''}
                onChange={(e) => handlePropertyChange('goal', e.target.value)}
                className="property-textarea"
                rows={3}
                placeholder="What should this agent accomplish?"
              />
            </div>

            <div className="property-group">
              <label className="property-label">Tool Budget</label>
              <input
                type="number"
                value={(properties.tool_budget as number) || 25}
                onChange={(e) =>
                  handlePropertyChange('tool_budget', parseInt(e.target.value))
                }
                className="property-input"
                min={1}
                max={100}
              />
            </div>
          </>
        )}

        {selectedNode.type === 'compute' && (
          <>
            <div className="property-group">
              <label className="property-label">Handler</label>
              <input
                type="text"
                value={(properties.handler as string) || ''}
                onChange={(e) =>
                  handlePropertyChange('handler', e.target.value)
                }
                className="property-input"
                placeholder="Handler function name"
              />
            </div>

            <div className="property-group">
              <label className="property-label">Timeout (seconds)</label>
              <input
                type="number"
                value={(properties.timeout as number) || 60}
                onChange={(e) =>
                  handlePropertyChange('timeout', parseInt(e.target.value))
                }
                className="property-input"
                min={1}
              />
            </div>
          </>
        )}

        {selectedNode.type === 'condition' && (
          <>
            <div className="property-group">
              <label className="property-label">Condition</label>
              <input
                type="text"
                value={(properties.condition as string) || ''}
                onChange={(e) =>
                  handlePropertyChange('condition', e.target.value)
                }
                className="property-input"
                placeholder="e.g., status == 'success'"
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default PropertyPanel;
