/**
 * Type definitions for workflow editor
 */

export interface WorkflowNode {
  id: string;
  type: NodeType;
  name: string;
  config: Record<string, unknown>;
  position: { x: number; y: number };
  data?: Record<string, unknown>;
}

export type NodeType =
  | 'agent'
  | 'compute'
  | 'team'
  | 'condition'
  | 'parallel'
  | 'transform'
  | 'hitl';

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
}

export interface WorkflowGraph {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  metadata?: Record<string, unknown>;
}

export interface NodeTypeConfig {
  name: string;
  description: string;
  color: string;
  config_schema: Record<string, unknown>;
}

export interface TeamMember {
  id: string;
  role: string;
  goal: string;
  tool_budget: number;
  tools?: string[];
  backstory?: string;
  expertise?: string[];
  personality?: string;
}

export interface TeamConfig {
  formation: TeamFormation;
  goal: string;
  max_iterations: number;
  timeout_seconds?: number;
  members: TeamMember[];
}

export type TeamFormation =
  | 'parallel'
  | 'sequential'
  | 'pipeline'
  | 'hierarchical'
  | 'consensus';

export interface FormationInfo {
  name: string;
  description: string;
  icon: string;
  best_for: string[];
  communication_style: string;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface NodePosition {
  x: number;
  y: number;
}
