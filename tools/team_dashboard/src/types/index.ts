/**
 * Type definitions for Victor Team Dashboard
 */

// =============================================================================
// Member Types
// =============================================================================

export type MemberStatus = 'idle' | 'running' | 'completed' | 'failed' | 'waiting';

export interface MemberState {
  member_id: string;
  role: string;
  status: MemberStatus;
  start_time: string | null;
  end_time: string | null;
  duration_seconds: number;
  tool_calls_used: number;
  tools_used: string[];
  error_message: string | null;
  last_activity: number;
}

// =============================================================================
// Team Execution Types
// =============================================================================

export type TeamFormation = 'sequential' | 'parallel' | 'pipeline' | 'hierarchical' | 'consensus';
export type ExecutionStatus = 'running' | 'completed' | 'failed';

export interface TeamExecutionState {
  execution_id: string;
  team_id: string;
  formation: TeamFormation;
  member_states: Record<string, MemberState>;
  shared_context: Record<string, unknown>;
  communication_logs: CommunicationLog[];
  negotiation_status: NegotiationStatus | null;
  start_time: string | null;
  end_time: string | null;
  duration_seconds: number;
  success: boolean | null;
  recursion_depth: number;
  consensus_achieved: boolean | null;
}

export interface TeamExecutionSummary {
  execution_id: string;
  team_id: string;
  formation: TeamFormation;
  status: ExecutionStatus;
  start_time: string;
  end_time: string | null;
  duration_seconds: number;
  member_count: number;
  recursion_depth: number;
  success: boolean | null;
  consensus_achieved: boolean | null;
}

// =============================================================================
// Communication Types
// =============================================================================

export type CommunicationType = 'request_response' | 'broadcast' | 'multicast' | 'pubsub';

export interface CommunicationLog {
  timestamp: number;
  message_type: string;
  sender_id: string;
  recipient_id: string | null;
  content: string;
  communication_type: CommunicationType;
  metadata: Record<string, unknown>;
  response_id: string | null;
  duration_ms: number | null;
}

// =============================================================================
// Negotiation Types
// =============================================================================

export interface NegotiationStatus {
  success: boolean;
  rounds: number;
  consensus_achieved: boolean;
  agreed_proposal: Proposal | null;
  votes: Record<string, unknown>;
}

export interface Proposal {
  id: string;
  content: string;
  proposer_id: string;
  timestamp: number;
  metadata: Record<string, unknown>;
  votes: Record<string, unknown>;
  rank: number | null;
}

// =============================================================================
// WebSocket Event Types
// =============================================================================

export type DashboardEventType =
  | 'team.started'
  | 'team.completed'
  | 'member.started'
  | 'member.updated'
  | 'member.completed'
  | 'member.failed'
  | 'message.sent'
  | 'message.received'
  | 'context.updated'
  | 'context.merged'
  | 'negotiation.started'
  | 'negotiation.vote'
  | 'negotiation.completed'
  | 'progress.update'
  | 'metrics.update';

export interface DashboardEvent {
  event_type: DashboardEventType;
  execution_id: string;
  timestamp: number;
  data: Record<string, unknown>;
}

// =============================================================================
// API Response Types
// =============================================================================

export interface MetricsSummary {
  total_teams_executed: number;
  successful_teams: number;
  failed_teams: number;
  active_teams: number;
  success_rate: number;
  average_duration_seconds: number;
  average_member_count: number;
  total_tool_calls: number;
  formation_distribution: Record<string, number>;
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  tracked_executions: number;
  active_executions: number;
}

// =============================================================================
// WebSocket Message Types
// =============================================================================

export interface WebSocketMessage {
  action?: string;
  event_type?: DashboardEventType;
  execution_id?: string;
  timestamp?: number;
  data?: Record<string, unknown>;
  state?: TeamExecutionState;
}

export interface ClientMessage {
  action: 'subscribe' | 'query_state' | 'ping';
  event_types?: DashboardEventType[];
}
