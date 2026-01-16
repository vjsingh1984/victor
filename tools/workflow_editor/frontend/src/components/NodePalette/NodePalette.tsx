/**
 * Node Palette - Drag-and-drop node templates
 */

import React from 'react';
import { useWorkflowStore } from '../../store/useWorkflowStore';
import { NodeType } from '../../types';
import { Bot, Cpu, Users, GitBranch, ArrowRight, MessageSquare } from 'lucide-react';
import './NodePalette.css';

interface NodeTypeTemplate {
  type: NodeType;
  label: string;
  description: string;
  icon: React.ReactNode;
  color: string;
}

const NODE_TYPES: NodeTypeTemplate[] = [
  {
    type: 'agent',
    label: 'Agent',
    description: 'LLM-powered agent with role and goal',
    icon: <Bot className="w-5 h-5" />,
    color: '#e3f2fd',
  },
  {
    type: 'compute',
    label: 'Compute',
    description: 'Execute tools without LLM',
    icon: <Cpu className="w-5 h-5" />,
    color: '#e8f5e9',
  },
  {
    type: 'team',
    label: 'Team',
    description: 'Multi-agent team',
    icon: <Users className="w-5 h-5" />,
    color: '#f3e5f5',
  },
  {
    type: 'condition',
    label: 'Condition',
    description: 'Branching logic',
    icon: <GitBranch className="w-5 h-5" />,
    color: '#fff3e0',
  },
  {
    type: 'transform',
    label: 'Transform',
    description: 'State transformation',
    icon: <ArrowRight className="w-5 h-5" />,
    color: '#eceff1',
  },
  {
    type: 'hitl',
    label: 'Human-in-the-Loop',
    description: 'Human interaction',
    icon: <MessageSquare className="w-5 h-5" />,
    color: '#ffebee',
  },
];

const NodePalette: React.FC = () => {
  const { addNode } = useWorkflowStore();

  const handleDragStart = (
    event: React.DragEvent,
    nodeType: NodeType
  ) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="node-palette">
      <h3 className="node-palette-title">Node Palette</h3>
      <div className="node-palette-list">
        {NODE_TYPES.map((nodeType) => (
          <div
            key={nodeType.type}
            className="node-palette-item"
            draggable
            onDragStart={(e) => handleDragStart(e, nodeType.type)}
            style={{ backgroundColor: nodeType.color }}
          >
            <div className="node-palette-item-icon">{nodeType.icon}</div>
            <div className="node-palette-item-info">
              <div className="node-palette-item-label">
                {nodeType.label}
              </div>
              <div className="node-palette-item-description">
                {nodeType.description}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default NodePalette;
