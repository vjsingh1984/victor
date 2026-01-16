/**
 * Zustand store for workflow state management with undo/redo support
 */

import { create } from 'zustand';
import { Connection, Edge, Node, addEdge, applyNodeChanges, applyEdgeChanges } from 'reactflow';
import { WorkflowGraph, WorkflowNode, TeamConfig } from '../types';

interface HistoryState {
  nodes: Node[];
  edges: Edge[];
  timestamp: number;
}

interface WorkflowState {
  nodes: Node[];
  edges: Edge[];
  selectedNode: Node | null;
  selectedNodes: string[]; // For multi-select
  validationErrors: string[];
  validationWarnings: string[];
  yamlPreview: string;

  // History for undo/redo
  history: HistoryState[];
  historyIndex: number;
  maxHistorySize: number;

  // Viewport state for zoom/pan
  viewport: {
    x: number;
    y: number;
    zoom: number;
  };

  // Search state
  searchQuery: string;
  searchMatches: string[];
  currentMatchIndex: number;

  // Grouping state
  nodeGroups: Record<string, { nodes: string[]; color: string; label: string; collapsed: boolean }>;

  // Actions
  onNodesChange: (changes: unknown[]) => void;
  onEdgesChange: (changes: unknown[]) => void;
  onConnect: (connection: Connection) => void;
  setNodes: (nodes: Node[]) => void;
  setEdges: (edges: Edge[]) => void;
  addNode: (node: Node) => void;
  updateNode: (id: string, data: Partial<WorkflowNode>) => void;
  deleteNode: (id: string) => void;
  deleteNodes: (ids: string[]) => void;
  selectNode: (node: Node | null) => void;
  selectNodes: (ids: string[]) => void;
  duplicateNode: (id: string) => void;
  copyNodes: (ids: string[]) => void;
  pasteNodes: () => void;
  setValidationErrors: (errors: string[]) => void;
  setValidationWarnings: (warnings: string[]) => void;
  setYamlPreview: (yaml: string) => void;
  clearWorkflow: () => void;
  loadWorkflow: (graph: WorkflowGraph) => void;

  // Undo/Redo actions
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;
  saveToHistory: () => void;

  // Viewport actions
  setViewport: (viewport: { x: number; y: number; zoom: number }) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  resetZoom: () => void;
  fitToScreen: () => void;

  // Search actions
  setSearchQuery: (query: string) => void;
  nextMatch: () => void;
  previousMatch: () => void;
  clearSearch: () => void;

  // Grouping actions
  createGroup: (nodeIds: string[], label: string, color: string) => void;
  deleteGroup: (groupId: string) => void;
  toggleGroupCollapse: (groupId: string) => void;
  addToGroup: (groupId: string, nodeIds: string[]) => void;
  removeFromGroup: (groupId: string, nodeIds: string[]) => void;

  // Layout actions
  applyLayout: (type: 'hierarchical' | 'force-directed' | 'grid') => void;
}

const MAX_HISTORY_SIZE = 100;

export const useWorkflowStore = create<WorkflowState>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNode: null,
  selectedNodes: [],
  validationErrors: [],
  validationWarnings: [],
  yamlPreview: '',

  history: [],
  historyIndex: -1,
  maxHistorySize: MAX_HISTORY_SIZE,

  viewport: {
    x: 0,
    y: 0,
    zoom: 1,
  },

  searchQuery: '',
  searchMatches: [],
  currentMatchIndex: -1,

  nodeGroups: {},

  onNodesChange: (changes) => {
    const newNodes = applyNodeChanges(changes, get().nodes);
    set({ nodes: newNodes });
    get().saveToHistory();
  },

  onEdgesChange: (changes) => {
    const newEdges = applyEdgeChanges(changes, get().edges);
    set({ edges: newEdges });
    get().saveToHistory();
  },

  onConnect: (connection) => {
    const newEdges = addEdge(connection, get().edges);
    set({ edges: newEdges });
    get().saveToHistory();
  },

  setNodes: (nodes) => {
    set({ nodes });
    get().saveToHistory();
  },

  setEdges: (edges) => {
    set({ edges });
    get().saveToHistory();
  },

  addNode: (node) => {
    set({
      nodes: [...get().nodes, node],
    });
    get().saveToHistory();
  },

  updateNode: (id, data) => {
    set({
      nodes: get().nodes.map((node) =>
        node.id === id
          ? { ...node, data: { ...node.data, ...data } }
          : node
      ),
    });
    get().saveToHistory();
  },

  deleteNode: (id) => {
    set({
      nodes: get().nodes.filter((node) => node.id !== id),
      edges: get().edges.filter(
        (edge) => edge.source !== id && edge.target !== id
      ),
      selectedNode: get().selectedNode?.id === id ? null : get().selectedNode,
    });
    get().saveToHistory();
  },

  deleteNodes: (ids) => {
    set({
      nodes: get().nodes.filter((node) => !ids.includes(node.id)),
      edges: get().edges.filter(
        (edge) => !ids.includes(edge.source) && !ids.includes(edge.target)
      ),
      selectedNode: ids.includes(get().selectedNode?.id || '') ? null : get().selectedNode,
      selectedNodes: get().selectedNodes.filter((id) => !ids.includes(id)),
    });
    get().saveToHistory();
  },

  selectNode: (node) => set({ selectedNode: node }),

  selectNodes: (ids) => set({ selectedNodes: ids }),

  duplicateNode: (id) => {
    const node = get().nodes.find((n) => n.id === id);
    if (!node) return;

    const newId = `${node.type}_${Date.now()}`;
    const newNode = {
      ...node,
      id: newId,
      position: {
        x: node.position.x + 50,
        y: node.position.y + 50,
      },
    };

    set({
      nodes: [...get().nodes, newNode],
    });
    get().saveToHistory();
  },

  copyNodes: (ids) => {
    const nodes = get().nodes.filter((n) => ids.includes(n.id));
    // Store in clipboard (using localStorage for persistence)
    localStorage.setItem('workflow_clipboard', JSON.stringify(nodes));
  },

  pasteNodes: () => {
    const clipboard = localStorage.getItem('workflow_clipboard');
    if (!clipboard) return;

    const nodes: Node[] = JSON.parse(clipboard);
    const offset = { x: 100, y: 100 };

    const newNodes = nodes.map((node) => ({
      ...node,
      id: `${node.type}_${Date.now()}_${Math.random()}`,
      position: {
        x: node.position.x + offset.x,
        y: node.position.y + offset.y,
      },
    }));

    set({
      nodes: [...get().nodes, ...newNodes],
    });
    get().saveToHistory();
  },

  setValidationErrors: (errors) => set({ validationErrors: errors }),

  setValidationWarnings: (warnings) => set({ validationWarnings: warnings }),

  setYamlPreview: (yaml) => set({ yamlPreview: yaml }),

  clearWorkflow: () =>
    set({
      nodes: [],
      edges: [],
      selectedNode: null,
      selectedNodes: [],
      validationErrors: [],
      validationWarnings: [],
      yamlPreview: '',
      history: [],
      historyIndex: -1,
      nodeGroups: {},
    }),

  loadWorkflow: (graph) => {
    const nodes: Node[] = graph.nodes.map((node) => ({
      id: node.id,
      type: node.type,
      position: node.position,
      data: {
        label: node.name,
        ...node.config,
      },
    }));

    const edges: Edge[] = graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label,
    }));

    set({ nodes, edges });
    get().saveToHistory();
  },

  // Undo/Redo implementation
  saveToHistory: () => {
    const state = get();
    const newHistoryState: HistoryState = {
      nodes: JSON.parse(JSON.stringify(state.nodes)),
      edges: JSON.parse(JSON.stringify(state.edges)),
      timestamp: Date.now(),
    };

    // Remove any future history if we're not at the end
    const newHistory = state.history.slice(0, state.historyIndex + 1);

    // Add new state
    newHistory.push(newHistoryState);

    // Limit history size
    if (newHistory.length > state.maxHistorySize) {
      newHistory.shift();
    }

    set({
      history: newHistory,
      historyIndex: newHistory.length - 1,
    });
  },

  undo: () => {
    const state = get();
    if (state.historyIndex <= 0) return;

    const prevState = state.history[state.historyIndex - 1];
    set({
      nodes: JSON.parse(JSON.stringify(prevState.nodes)),
      edges: JSON.parse(JSON.stringify(prevState.edges)),
      historyIndex: state.historyIndex - 1,
    });
  },

  redo: () => {
    const state = get();
    if (state.historyIndex >= state.history.length - 1) return;

    const nextState = state.history[state.historyIndex + 1];
    set({
      nodes: JSON.parse(JSON.stringify(nextState.nodes)),
      edges: JSON.parse(JSON.stringify(nextState.edges)),
      historyIndex: state.historyIndex + 1,
    });
  },

  canUndo: () => {
    const state = get();
    return state.historyIndex > 0;
  },

  canRedo: () => {
    const state = get();
    return state.historyIndex < state.history.length - 1;
  },

  // Viewport actions
  setViewport: (viewport) => set({ viewport }),

  zoomIn: () => {
    const state = get();
    const newZoom = Math.min(state.viewport.zoom + 0.1, 2);
    set({ viewport: { ...state.viewport, zoom: newZoom } });
  },

  zoomOut: () => {
    const state = get();
    const newZoom = Math.max(state.viewport.zoom - 0.1, 0.1);
    set({ viewport: { ...state.viewport, zoom: newZoom } });
  },

  resetZoom: () => {
    const state = get();
    set({ viewport: { ...state.viewport, zoom: 1 } });
  },

  fitToScreen: () => {
    // Calculate bounds to fit all nodes
    const state = get();
    if (state.nodes.length === 0) return;

    const bounds = {
      minX: Math.min(...state.nodes.map((n) => n.position.x)),
      maxX: Math.max(...state.nodes.map((n) => n.position.x + 200)),
      minY: Math.min(...state.nodes.map((n) => n.position.y)),
      maxY: Math.max(...state.nodes.map((n) => n.position.y + 100)),
    };

    const width = bounds.maxX - bounds.minX;
    const height = bounds.maxY - bounds.minY;

    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;

    set({
      viewport: {
        x: -centerX + width / 2,
        y: -centerY + height / 2,
        zoom: 0.8,
      },
    });
  },

  // Search actions
  setSearchQuery: (query) => {
    const state = get();
    const q = query.toLowerCase();

    if (!q) {
      set({ searchQuery: '', searchMatches: [], currentMatchIndex: -1 });
      return;
    }

    const matches = state.nodes
      .filter(
        (node) =>
          node.id.toLowerCase().includes(q) ||
          node.data.label?.toLowerCase().includes(q) ||
          node.type.toLowerCase().includes(q)
      )
      .map((node) => node.id);

    set({
      searchQuery: query,
      searchMatches: matches,
      currentMatchIndex: matches.length > 0 ? 0 : -1,
    });
  },

  nextMatch: () => {
    const state = get();
    if (state.searchMatches.length === 0) return;

    const nextIndex = (state.currentMatchIndex + 1) % state.searchMatches.length;
    set({ currentMatchIndex: nextIndex });
  },

  previousMatch: () => {
    const state = get();
    if (state.searchMatches.length === 0) return;

    const prevIndex =
      (state.currentMatchIndex - 1 + state.searchMatches.length) %
      state.searchMatches.length;
    set({ currentMatchIndex: prevIndex });
  },

  clearSearch: () => {
    set({ searchQuery: '', searchMatches: [], currentMatchIndex: -1 });
  },

  // Grouping actions
  createGroup: (nodeIds, label, color) => {
    const state = get();
    const groupId = `group_${Date.now()}`;

    set({
      nodeGroups: {
        ...state.nodeGroups,
        [groupId]: {
          nodes: nodeIds,
          color,
          label,
          collapsed: false,
        },
      },
    });
  },

  deleteGroup: (groupId) => {
    const state = get();
    const newGroups = { ...state.nodeGroups };
    delete newGroups[groupId];
    set({ nodeGroups: newGroups });
  },

  toggleGroupCollapse: (groupId) => {
    const state = get();
    const group = state.nodeGroups[groupId];
    if (!group) return;

    set({
      nodeGroups: {
        ...state.nodeGroups,
        [groupId]: {
          ...group,
          collapsed: !group.collapsed,
        },
      },
    });
  },

  addToGroup: (groupId, nodeIds) => {
    const state = get();
    const group = state.nodeGroups[groupId];
    if (!group) return;

    set({
      nodeGroups: {
        ...state.nodeGroups,
        [groupId]: {
          ...group,
          nodes: [...new Set([...group.nodes, ...nodeIds])],
        },
      },
    });
  },

  removeFromGroup: (groupId, nodeIds) => {
    const state = get();
    const group = state.nodeGroups[groupId];
    if (!group) return;

    set({
      nodeGroups: {
        ...state.nodeGroups,
        [groupId]: {
          ...group,
          nodes: group.nodes.filter((id) => !nodeIds.includes(id)),
        },
      },
    });
  },

  // Layout algorithms
  applyLayout: (type) => {
    const state = get();
    if (state.nodes.length === 0) return;

    let positionedNodes: Node[] = [];

    switch (type) {
      case 'hierarchical':
        positionedNodes = applyHierarchicalLayout(state.nodes, state.edges);
        break;
      case 'force-directed':
        positionedNodes = applyForceDirectedLayout(state.nodes, state.edges);
        break;
      case 'grid':
        positionedNodes = applyGridLayout(state.nodes);
        break;
    }

    set({ nodes: positionedNodes });
    get().saveToHistory();
  },
}));

// Layout algorithms
function applyHierarchicalLayout(nodes: Node[], edges: Edge[]): Node[] {
  // Simple topological sort with layers
  const levels: Record<string, number> = {};
  const visited = new Set<string>();

  // Calculate levels for each node
  const calculateLevel = (nodeId: string, currentLevel: number = 0): void => {
    if (visited.has(nodeId)) return;
    visited.add(nodeId);

    const incomingEdges = edges.filter((e) => e.target === nodeId);
    let maxParentLevel = 0;

    for (const edge of incomingEdges) {
      calculateLevel(edge.source, currentLevel + 1);
      maxParentLevel = Math.max(maxParentLevel, levels[edge.source] || 0);
    }

    levels[nodeId] = maxParentLevel + 1;
  };

  nodes.forEach((node) => calculateLevel(node.id));

  // Group nodes by level
  const nodesByLevel: Record<number, Node[]> = {};
  Object.entries(levels).forEach(([nodeId, level]) => {
    if (!nodesByLevel[level]) nodesByLevel[level] = [];
    const node = nodes.find((n) => n.id === nodeId);
    if (node) nodesByLevel[level].push(node);
  });

  // Position nodes
  const positioned: Node[] = [];
  const levelHeight = 200;
  const nodeWidth = 250;

  Object.entries(nodesByLevel).forEach(([level, levelNodes]) => {
    const levelNum = parseInt(level);
    const totalWidth = levelNodes.length * nodeWidth;
    const startX = -totalWidth / 2;

    levelNodes.forEach((node, index) => {
      positioned.push({
        ...node,
        position: {
          x: startX + index * nodeWidth,
          y: levelNum * levelHeight,
        },
      });
    });
  });

  return positioned;
}

function applyForceDirectedLayout(nodes: Node[], edges: Edge[]): Node[] {
  // Simple force-directed layout
  const positioned = [...nodes];
  const iterations = 50;
  const repulsion = 5000;
  const attraction = 0.01;
  const damping = 0.9;

  // Initialize positions randomly around center
  positioned.forEach((node) => {
    node.position = {
      x: (Math.random() - 0.5) * 800,
      y: (Math.random() - 0.5) * 600,
    };
  });

  for (let i = 0; i < iterations; i++) {
    const forces: Record<string, { x: number; y: number }> = {};

    // Initialize forces
    positioned.forEach((node) => {
      forces[node.id] = { x: 0, y: 0 };
    });

    // Repulsion between all nodes
    for (let a = 0; a < positioned.length; a++) {
      for (let b = a + 1; b < positioned.length; b++) {
        const nodeA = positioned[a];
        const nodeB = positioned[b];

        const dx = nodeA.position.x - nodeB.position.x;
        const dy = nodeA.position.y - nodeB.position.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;

        const force = repulsion / (dist * dist);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;

        forces[nodeA.id].x += fx;
        forces[nodeA.id].y += fy;
        forces[nodeB.id].x -= fx;
        forces[nodeB.id].y -= fy;
      }
    }

    // Attraction along edges
    edges.forEach((edge) => {
      const nodeA = positioned.find((n) => n.id === edge.source);
      const nodeB = positioned.find((n) => n.id === edge.target);

      if (!nodeA || !nodeB) return;

      const dx = nodeB.position.x - nodeA.position.x;
      const dy = nodeB.position.y - nodeA.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;

      const force = attraction * dist;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;

      forces[nodeA.id].x += fx;
      forces[nodeA.id].y += fy;
      forces[nodeB.id].x -= fx;
      forces[nodeB.id].y -= fy;
    });

    // Apply forces with damping
    positioned.forEach((node) => {
      node.position.x += forces[node.id].x * damping;
      node.position.y += forces[node.id].y * damping;
    });
  }

  return positioned;
}

function applyGridLayout(nodes: Node[]): Node[] {
  const positioned = [...nodes];
  const cols = Math.ceil(Math.sqrt(nodes.length));
  const nodeWidth = 250;
  const nodeHeight = 150;

  positioned.forEach((node, index) => {
    const row = Math.floor(index / cols);
    const col = index % cols;

    node.position = {
      x: col * nodeWidth,
      y: row * nodeHeight,
    };
  });

  return positioned;
}
