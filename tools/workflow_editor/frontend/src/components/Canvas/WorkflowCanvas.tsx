/**
 * Main workflow canvas component using React Flow
 */

import React, { useCallback, useEffect } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  NodeTypes,
  EdgeTypes,
  useReactFlow,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { useWorkflowStore } from '../../store/useWorkflowStore';
import AgentNode from '../../nodes/AgentNode';
import ComputeNode from '../../nodes/ComputeNode';
import TeamNode from '../../nodes/TeamNode';
import ConditionNode from '../../nodes/ConditionNode';
import TransformNode from '../../nodes/TransformNode';
import './WorkflowCanvas.css';

const nodeTypes: NodeTypes = {
  agent: AgentNode,
  compute: ComputeNode,
  team: TeamNode,
  condition: ConditionNode,
  transform: TransformNode,
  parallel: TransformNode, // Reuse transform node for now
  hitl: TransformNode, // Reuse transform node for now
};

const WorkflowCanvas: React.FC = () => {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    selectedNode,
    selectNode,
    deleteNode,
  } = useWorkflowStore();

  const { deleteElements } = useReactFlow();

  // Handle node selection
  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      selectNode(node);
    },
    [selectNode]
  );

  // Handle background click to deselect
  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  // Handle keyboard shortcuts
  const onKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (event.key === 'Delete' || event.key === 'Backspace') {
        if (selectedNode && !event.shiftKey) {
          // Don't delete if editing text
          const activeElement = document.activeElement;
          if (
            activeElement &&
            (activeElement.tagName === 'INPUT' ||
              activeElement.tagName === 'TEXTAREA' ||
              activeElement.getAttribute('contenteditable') === 'true')
          ) {
            return;
          }
          deleteNode(selectedNode.id);
        }
      }
    },
    [selectedNode, deleteNode]
  );

  useEffect(() => {
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onKeyDown]);

  return (
    <div className="workflow-canvas">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        className="bg-slate-50"
      >
        <Background color="#cbd5e1" gap={16} />
        <Controls />
        <MiniMap
          nodeColor={(node) => {
            switch (node.type) {
              case 'agent':
                return '#e3f2fd';
              case 'compute':
                return '#e8f5e9';
              case 'team':
                return '#f3e5f5';
              case 'condition':
                return '#fff3e0';
              default:
                return '#eceff1';
            }
          }}
          maskColor="rgba(0, 0, 0, 0.1)"
        />
      </ReactFlow>
    </div>
  );
};

export default WorkflowCanvas;
