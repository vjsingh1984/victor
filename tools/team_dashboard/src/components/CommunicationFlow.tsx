import React, { useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';
import type { CommunicationLog, MemberState } from '../types';
import { formatDistanceToNow } from 'date-fns';

interface CommunicationFlowProps {
  logs: CommunicationLog[];
  members: Record<string, MemberState>;
  className?: string;
}

/**
 * Convert communication logs to ReactFlow nodes and edges
 */
function buildFlowGraph(
  logs: CommunicationLog[],
  members: Record<string, MemberState>
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const memberIds = new Set<string>();

  // Add all members as nodes
  Object.entries(members).forEach(([memberId, member], index) => {
    memberIds.add(memberId);

    nodes.push({
      id: memberId,
      type: 'default',
      position: {
        x: (index % 3) * 300,
        y: Math.floor(index / 3) * 200,
      },
      data: {
        label: (
          <div className="p-2">
            <div className="font-semibold">{memberId}</div>
            <div className="text-xs text-gray-500">{member.role}</div>
            <div
              className={`text-xs mt-1 ${
                member.status === 'running'
                  ? 'text-blue-500'
                  : member.status === 'completed'
                  ? 'text-green-500'
                  : member.status === 'failed'
                  ? 'text-red-500'
                  : 'text-gray-500'
              }`}
            >
              {member.status}
            </div>
          </div>
        ),
      },
      style: {
        background: member.status === 'running' ? '#e0f2fe' : '#f9fafb',
        border: '2px solid',
        borderColor:
          member.status === 'running'
            ? '#3b82f6'
            : member.status === 'completed'
            ? '#22c55e'
            : member.status === 'failed'
            ? '#ef4444'
            : '#9ca3af',
        borderRadius: '8px',
        width: 180,
      },
    });
  });

  // Add edges for communications
  logs.slice(-50).forEach((log, index) => {
    const { sender_id, recipient_id, timestamp, content, message_type } = log;

    if (sender_id && recipient_id) {
      edges.push({
        id: `edge-${index}`,
        source: sender_id,
        target: recipient_id,
        label: message_type,
        labelStyle: { fontSize: 10, fontWeight: 500 },
        labelShowBg: true,
        labelBgStyle: { fill: '#f3f4f6', fillOpacity: 0.8 },
        style: { stroke: '#6366f1', strokeWidth: 2 },
        animated: true,
        data: {
          timestamp,
          content,
        },
      });
    } else if (sender_id && !recipient_id) {
      // Broadcast - add edges to all members
      memberIds.forEach((memberId) => {
        if (memberId !== sender_id) {
          edges.push({
            id: `broadcast-${index}-${memberId}`,
            source: sender_id,
            target: memberId,
            label: 'broadcast',
            labelStyle: { fontSize: 10 },
            labelShowBg: true,
            labelBgStyle: { fill: '#fef3c7', fillOpacity: 0.8 },
            style: { stroke: '#f59e0b', strokeWidth: 2, strokeDasharray: '5,5' },
            animated: true,
          });
        }
      });
    }
  });

  return { nodes, edges };
}

/**
 * Communication flow diagram component
 *
 * Visualizes message flow between team members using ReactFlow.
 * Shows member nodes and communication edges with animation.
 */
export function CommunicationFlow({
  logs,
  members,
  className = '',
}: CommunicationFlowProps) {
  const { nodes, edges } = useMemo(
    () => buildFlowGraph(logs, members),
    [logs, members]
  );

  if (nodes.length === 0) {
    return (
      <div
        className={`bg-white rounded-lg shadow p-6 border border-gray-200 ${className}`}
      >
        <div className="text-center text-gray-500">
          <p className="text-sm">No communication data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow border border-gray-200 ${className}`}>
      <div className="p-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">Communication Flow</h3>
        <p className="text-sm text-gray-600 mt-1">
          Real-time message flow between team members ({logs.length} messages)
        </p>
      </div>

      <div style={{ height: '500px', width: '100%' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          attributionPosition="bottom-left"
        >
          <Background color="#aaa" gap={16} />
          <Controls />
          <MiniMap
            nodeColor={(node) => {
              const member = members[node.id];
              if (member?.status === 'running') return '#3b82f6';
              if (member?.status === 'completed') return '#22c55e';
              if (member?.status === 'failed') return '#ef4444';
              return '#9ca3af';
            }}
            maskColor="rgba(0, 0, 0, 0.1)"
          />
        </ReactFlow>
      </div>

      {/* Legend */}
      <div className="p-4 border-t border-gray-200">
        <div className="flex flex-wrap gap-4 text-xs text-gray-600">
          <div className="flex items-center gap-1">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: '#6366f1' }}
            />
            <span>Direct message</span>
          </div>
          <div className="flex items-center gap-1">
            <div
              className="w-3 h-3 rounded-full border-2 border-dashed"
              style={{ borderColor: '#f59e0b' }}
            />
            <span>Broadcast</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span>Running</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span>Completed</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span>Failed</span>
          </div>
        </div>
      </div>
    </div>
  );
}
