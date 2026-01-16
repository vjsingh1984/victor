/**
 * Mini-map component for navigation in large workflows
 */

import React, { useRef, useEffect, useState } from 'react';
import { useWorkflowStore } from '../../store/useWorkflowStore';
import { Node } from 'reactflow';

interface MiniMapProps {
  width?: number;
  height?: number;
}

export const MiniMap: React.FC<MiniMapProps> = ({ width = 200, height = 150 }) => {
  const { nodes, viewport } = useWorkflowStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [bounds, setBounds] = useState({ minX: 0, maxX: 0, minY: 0, maxY: 0 });

  // Calculate bounds of all nodes
  useEffect(() => {
    if (nodes.length === 0) return;

    const padding = 50;
    const minX = Math.min(...nodes.map((n) => n.position.x)) - padding;
    const maxX = Math.max(...nodes.map((n) => n.position.x + 200)) + padding;
    const minY = Math.min(...nodes.map((n) => n.position.y)) - padding;
    const maxY = Math.max(...nodes.map((n) => n.position.y + 100)) + padding;

    setBounds({ minX, maxX, minY, maxY });
  }, [nodes]);

  // Draw mini-map
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (nodes.length === 0) {
      // Draw empty state
      ctx.fillStyle = '#94a3b8';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No nodes', width / 2, height / 2);
      return;
    }

    const graphWidth = bounds.maxX - bounds.minX;
    const graphHeight = bounds.maxY - bounds.minY;
    const scaleX = width / graphWidth;
    const scaleY = height / graphHeight;
    const scale = Math.min(scaleX, scaleY) * 0.9; // 90% to leave some padding

    // Center the graph in the mini-map
    const offsetX = (width - graphWidth * scale) / 2 - bounds.minX * scale;
    const offsetY = (height - graphHeight * scale) / 2 - bounds.minY * scale;

    // Draw edges first
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    useWorkflowStore.getState().edges.forEach((edge) => {
      const sourceNode = nodes.find((n) => n.id === edge.source);
      const targetNode = nodes.find((n) => n.id === edge.target);

      if (sourceNode && targetNode) {
        const sourceX = sourceNode.position.x * scale + offsetX + 100 * scale;
        const sourceY = sourceNode.position.y * scale + offsetY + 50 * scale;
        const targetX = targetNode.position.x * scale + offsetX + 100 * scale;
        const targetY = targetNode.position.y * scale + offsetY + 50 * scale;

        ctx.beginPath();
        ctx.moveTo(sourceX, sourceY);
        ctx.lineTo(targetX, targetY);
        ctx.stroke();
      }
    });

    // Draw nodes
    nodes.forEach((node) => {
      const x = node.position.x * scale + offsetX;
      const y = node.position.y * scale + offsetY;
      const nodeWidth = 200 * scale;
      const nodeHeight = 100 * scale;

      // Node background based on type
      const colors: Record<string, string> = {
        agent: '#3b82f6',
        team: '#8b5cf6',
        compute: '#10b981',
        condition: '#f59e0b',
        parallel: '#ec4899',
        transform: '#6366f1',
        hitl: '#ef4444',
      };

      ctx.fillStyle = colors[node.type || 'agent'] || '#64748b';
      ctx.fillRect(x, y, nodeWidth, nodeHeight);

      // Node border
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, nodeWidth, nodeHeight);
    });

    // Draw viewport rectangle
    const viewportWidth = (window.innerWidth - 400) / viewport.zoom; // Approximate canvas width
    const viewportHeight = (window.innerHeight - 200) / viewport.zoom; // Approximate canvas height

    const viewportX = -viewport.x * scale + offsetX;
    const viewportY = -viewport.y * scale + offsetY;

    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.strokeRect(viewportX, viewportY, viewportWidth * scale, viewportHeight * scale);

    // Semi-transparent fill for viewport
    ctx.fillStyle = 'rgba(59, 130, 246, 0.1)';
    ctx.fillRect(viewportX, viewportY, viewportWidth * scale, viewportHeight * scale);
  }, [nodes, bounds, viewport, width, height]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const graphWidth = bounds.maxX - bounds.minX;
    const graphHeight = bounds.maxY - bounds.minY;
    const scaleX = width / graphWidth;
    const scaleY = height / graphHeight;
    const scale = Math.min(scaleX, scaleY) * 0.9;

    const offsetX = (width - graphWidth * scale) / 2 - bounds.minX * scale;
    const offsetY = (height - graphHeight * scale) / 2 - bounds.minY * scale;

    // Convert click position to graph coordinates
    const graphX = (x - offsetX) / scale;
    const graphY = (y - offsetY) / scale;

    // Center viewport on clicked position
    const viewportWidth = window.innerWidth - 400;
    const viewportHeight = window.innerHeight - 200;

    useWorkflowStore.getState().setViewport({
      x: -graphX + viewportWidth / 2 / viewport.zoom,
      y: -graphY + viewportHeight / 2 / viewport.zoom,
      zoom: viewport.zoom,
    });
  };

  return (
    <div className="fixed bottom-4 right-4 bg-white rounded-lg shadow-lg border border-slate-200 p-2">
      <div className="text-xs text-slate-600 mb-1 font-medium">Mini Map</div>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onClick={handleClick}
        className="cursor-pointer border border-slate-200 rounded"
        style={{ display: 'block' }}
      />
      <div className="text-xs text-slate-500 mt-1 text-center">
        {nodes.length} nodes
      </div>
    </div>
  );
};
