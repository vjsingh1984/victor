import { create } from 'zustand';
import type {
  TeamExecutionState,
  TeamExecutionSummary,
  DashboardEvent,
  MemberState,
  CommunicationLog,
  MetricsSummary,
} from '../types';

interface DashboardState {
  // Execution data
  executions: Record<string, TeamExecutionState>;
  executionSummaries: TeamExecutionSummary[];
  selectedExecutionId: string | null;

  // Metrics
  metrics: MetricsSummary | null;

  // UI state
  isLoading: boolean;
  error: string | null;
  filter: {
    status?: string;
    formation?: string;
    search?: string;
  };
}

interface DashboardActions {
  // Execution actions
  setSelectedExecution: (executionId: string | null) => void;
  updateExecution: (executionId: string, state: Partial<TeamExecutionState>) => void;
  addExecution: (state: TeamExecutionState) => void;
  setExecutionSummaries: (summaries: TeamExecutionSummary[]) => void;

  // Member actions
  updateMember: (executionId: string, memberId: string, member: Partial<MemberState>) => void;

  // Communication actions
  addCommunicationLog: (executionId: string, log: CommunicationLog) => void;

  // Metrics actions
  setMetrics: (metrics: MetricsSummary) => void;

  // UI actions
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setFilter: (filter: Partial<DashboardState['filter']>) => void;

  // Event handler
  handleDashboardEvent: (event: DashboardEvent) => void;
}

type DashboardStore = DashboardState & DashboardActions;

/**
 * Zustand store for dashboard state management
 *
 * Manages team execution data, metrics, and UI state with Zustand for
 * efficient reactivity and performance.
 */
export const useDashboardStore = create<DashboardStore>((set, get) => ({
  // Initial state
  executions: {},
  executionSummaries: [],
  selectedExecutionId: null,
  metrics: null,
  isLoading: false,
  error: null,
  filter: {},

  // =========================================================================
  // Execution actions
  // =========================================================================

  setSelectedExecution: (executionId) => {
    set({ selectedExecutionId: executionId });
  },

  updateExecution: (executionId, stateUpdate) => {
    set((state) => ({
      executions: {
        ...state.executions,
        [executionId]: {
          ...state.executions[executionId],
          ...stateUpdate,
        } as TeamExecutionState,
      },
    }));
  },

  addExecution: (executionState) => {
    set((state) => ({
      executions: {
        ...state.executions,
        [executionState.execution_id]: executionState,
      },
    }));
  },

  setExecutionSummaries: (summaries) => {
    set({ executionSummaries: summaries });
  },

  // =========================================================================
  // Member actions
  // =========================================================================

  updateMember: (executionId, memberId, memberUpdate) => {
    set((state) => {
      const execution = state.executions[executionId];
      if (!execution) return state;

      return {
        executions: {
          ...state.executions,
          [executionId]: {
            ...execution,
            member_states: {
              ...execution.member_states,
              [memberId]: {
                ...execution.member_states[memberId],
                ...memberUpdate,
              } as MemberState,
            },
          },
        },
      };
    });
  },

  // =========================================================================
  // Communication actions
  // =========================================================================

  addCommunicationLog: (executionId, log) => {
    set((state) => {
      const execution = state.executions[executionId];
      if (!execution) return state;

      return {
        executions: {
          ...state.executions,
          [executionId]: {
            ...execution,
            communication_logs: [...execution.communication_logs, log],
          },
        },
      };
    });
  },

  // =========================================================================
  // Metrics actions
  // =========================================================================

  setMetrics: (metrics) => {
    set({ metrics });
  },

  // =========================================================================
  // UI actions
  // =========================================================================

  setLoading: (isLoading) => {
    set({ isLoading });
  },

  setError: (error) => {
    set({ error });
  },

  setFilter: (filterUpdate) => {
    set((state) => ({
      filter: { ...state.filter, ...filterUpdate },
    }));
  },

  // =========================================================================
  // Event handler
  // =========================================================================

  handleDashboardEvent: (event) => {
    const { executions, selectedExecutionId } = get();
    const { execution_id, event_type, data } = event;

    switch (event_type) {
      case 'team.started':
        // Add new execution
        get().addExecution(data as unknown as TeamExecutionState);
        break;

      case 'team.completed':
        // Update execution with completion data
        get().updateExecution(execution_id, {
          end_time: new Date().toISOString(),
          success: data.success as boolean,
          duration_seconds: data.duration_seconds as number,
          consensus_achieved: data.consensus_achieved as boolean | null,
        });
        break;

      case 'member.started':
        // Add or update member
        get().updateMember(execution_id, data.member_id as string, {
          member_id: data.member_id as string,
          role: data.role as string,
          status: 'running',
          start_time: data.start_time as string,
          duration_seconds: 0,
          tool_calls_used: 0,
          tools_used: [],
          error_message: null,
          last_activity: Date.now() / 1000,
        });
        break;

      case 'member.updated':
        // Update member progress
        get().updateMember(execution_id, data.member_id as string, {
          tool_calls_used: data.tool_calls_used as number,
          tools_used: data.tools_used as string[],
          last_activity: Date.now() / 1000,
        });
        break;

      case 'member.completed':
      case 'member.failed':
        // Update member completion
        get().updateMember(execution_id, data.member_id as string, {
          status: event_type === 'member.completed' ? 'completed' : 'failed',
          end_time: data.end_time as string,
          duration_seconds: data.duration_seconds as number,
          error_message: data.error_message as string | null,
          last_activity: Date.now() / 1000,
        });
        break;

      case 'message.sent':
        // Add communication log
        get().addCommunicationLog(execution_id, data.log as CommunicationLog);
        break;

      case 'context.updated':
        // Update shared context
        const key = data.key as string;
        const value = data.value;
        get().updateExecution(execution_id, {
          shared_context: {
            ...executions[execution_id]?.shared_context,
            [key]: value,
          },
        });
        break;

      case 'negotiation.completed':
      case 'negotiation.started':
        // Update negotiation status
        get().updateExecution(execution_id, {
          negotiation_status: data as unknown as any,
        });
        break;

      default:
        console.log('Unhandled event type:', event_type);
    }
  },
}));

// Selectors for optimized re-renders
export const selectExecutionById = (executionId: string | null) => (state: DashboardStore) =>
  executionId ? state.executions[executionId] : null;

export const selectMemberById = (
  executionId: string | null,
  memberId: string
) => (state: DashboardStore) => {
  if (!executionId) return null;
  const execution = state.executions[executionId];
  return execution?.member_states[memberId] || null;
};

export const selectFilteredExecutions = () => (state: DashboardStore) => {
  const { executionSummaries, filter } = state;

  return executionSummaries.filter((summary) => {
    if (filter.status && summary.status !== filter.status) return false;
    if (filter.formation && summary.formation !== filter.formation) return false;
    if (filter.search) {
      const searchLower = filter.search.toLowerCase();
      return (
        summary.team_id.toLowerCase().includes(searchLower) ||
        summary.execution_id.toLowerCase().includes(searchLower)
      );
    }
    return true;
  });
};
