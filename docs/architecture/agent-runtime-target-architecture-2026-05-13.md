# Agent Runtime Target Architecture

Date: 2026-05-13

Status: proposed target state

Companion tracker: [Agent Runtime Competitive Gap Analysis](agent-runtime-competitive-gap-analysis-2026-05-13.md)

## Design Goal

Victor should compete as a production agent runtime, not just a coding assistant implementation. The target shape is:

```text
Clients
  CLI, TUI, API, MCP, VS Code
    |
    v
Framework Runtime
  StateGraph + RuntimeExecutionContext + Observability
    |
    v
Canonical Services
  ChatService, ToolService, SessionService, ContextService,
  ProviderService, RecoveryService, RuntimeIntelligenceService
    |
    v
Execution Patterns
  Single agent loop, autonomous planner, team formations,
  workflows, handoffs, guarded tools
    |
    v
Project Intelligence
  project.db graph, symbols, conversations, checkpoints,
  embeddings, module metrics
```

The framework should expose a small number of durable primitives and let higher-level experiences compose them.

## Target Principles

1. `StateGraph` is the execution engine.

   Long-running agent work, autonomous planning, team coordination, workflows, handoffs, and human approval points should be graph nodes and edges. Control flow should be inspectable, checkpointable, replayable, and observable.

2. Services own effects.

   `ChatService`, `ToolService`, `SessionService`, `ContextService`, `ProviderService`, and `RecoveryService` own runtime behavior. Facades and orchestrators only compose and forward.

3. Teams are formations.

   A team is not a separate graph type. A team is a `UnifiedTeamCoordinator` node using a formation strategy: sequential, parallel, hierarchical, pipeline, or consensus.

4. Context has one source of truth.

   `ContextService` owns message selection, token accounting, compaction policy, and context lifecycle. `ConversationStore` owns persistence. Other components read through those APIs.

5. Prompts are pipeline products.

   `UnifiedPromptPipeline` owns stable system prompts and per-turn dynamic overlays. Planner, skill, research, and recovery guidance should be scoped overlays instead of mutable global system prompts.

6. Subagents are isolated by spec.

   Tool access, context scope, memory scope, model choice, max turns, permissions, and worktree isolation should be declared in a reusable subagent or team-member spec.

7. Checkpoints are holistic.

   A checkpoint should bind graph state, session/conversation state, tool intent, approval status, and filesystem snapshot metadata.

8. Project graph data is canonical and rebuildable.

   The graph belongs in `.victor/project.db`; it should use stable repo-relative file identity, exclude generated artifacts, and degrade gracefully when optional vector or native dependencies are absent.

## Canonical Runtime Contracts

### RuntimeExecutionContext

`RuntimeExecutionContext` should be the explicit dependency carrier for runtime graph execution.

It should provide:

- Settings and profile metadata
- Service accessors
- Session identity
- Provider/model metadata
- Tool permission mode
- Observability trace context
- Checkpoint context
- Project root and graph index metadata

Avoid:

- Pulling global containers from business logic
- Passing raw orchestrator objects into graph nodes
- Mutating orchestrator fields from planner or subagent code

### ChatService

Responsibilities:

- Run a chat turn through the selected execution graph
- Enter and exit turn scope
- Delegate tool execution to `ToolService`
- Delegate context selection to `ContextService`
- Delegate provider calls to `ProviderService`
- Emit trace spans and task lifecycle events

Non-responsibilities:

- Direct graph indexing
- Direct compaction policy implementation
- Direct provider registry mutation
- Direct prompt mutation

### ContextService

Responsibilities:

- Maintain the active working context
- Estimate and reconcile token counts
- Select messages for provider calls
- Apply compaction policy
- Persist summaries and compaction metadata
- Serve context metrics and trend information

Non-responsibilities:

- Acting as a separate persistent store
- Owning provider calls
- Owning autonomous planner state

### ToolService

Responsibilities:

- Tool selection
- Tool validation and guardrails
- Tool execution
- Tool budget accounting
- Tool result normalization
- Tool output compaction hooks through `ContextService`

Non-responsibilities:

- Planning entire user goals
- Mutating conversation history directly
- Managing provider-specific prompt instructions outside prompt pipeline contracts

### UnifiedTeamCoordinator

Responsibilities:

- Execute team members through formation strategies
- Run directly as a `StateGraph` node
- Preserve per-run execution state without mutating coordinator defaults
- Enforce member specs: allowed tools, budgets, context scope, worktree isolation

Non-responsibilities:

- Creating a parallel graph abstraction
- Reimplementing autonomous planner DAG execution
- Duplicating legacy team coordinator behavior

## Target Graphs

### Single Agent Loop

```text
prompt -> perceive -> plan -> act -> evaluate -> decide
                                      ^          |
                                      |----------|
```

State should include:

- User query
- Working context
- Available tools
- Plan or current intent
- Tool call ledger
- Provider response
- Evaluation result
- Stop reason
- Checkpoint metadata

### Autonomous Planner

```text
classify_goal -> generate_plan -> approve_plan -> select_ready_step
       |                                |              |
       v                                v              v
  ask_clarifying_question           stop/hold       execute_step
                                                        |
                                                        v
                                                  evaluate_step
                                                        |
                                                        v
                                               replan_or_continue
```

Why this matters:

- Plan execution becomes resumable.
- Human approval is a graph interrupt, not ad hoc callback logic.
- Step failure can branch to recovery or replanning.
- Subagents are graph nodes with isolated specs.

### Team Formation Node

```text
prepare_team_context -> UnifiedTeamCoordinator -> merge_team_result
```

The coordinator chooses or receives a formation:

- Sequential for ordered dependency chains
- Parallel for independent research/test tasks
- Hierarchical for manager-worker decomposition
- Pipeline for staged generation/review/test flows
- Consensus for high-confidence decisions

### File-Changing Workflow

```text
prepare -> checkpoint -> execute_tools -> validate -> merge_or_restore -> summarize
```

Checkpoint should include:

- Graph checkpoint id
- Conversation/session checkpoint id
- Tool call that caused the checkpoint
- Filesystem/git snapshot id
- Approval decision state
- Recovery instructions

## Design Changes To Make

### P0: Make StateGraph Default

Current concern:

The framework has graph execution, but legacy loops and planning paths still run outside it.

Target:

- `AgenticLoop` uses the StateGraph-backed executor by default.
- `AutonomousPlanner` emits and executes graph state.
- `UnifiedTeamCoordinator` is invoked as a graph node for teams.
- Legacy APIs adapt to graph execution results.

Acceptance criteria:

- A chat turn can be traced as graph node transitions.
- A planner run can pause and resume from a checkpoint.
- A team run appears as one or more graph nodes with formation metadata.

### P0: Consolidate Context Lifecycle

Current concern:

Message history, conversation controller, context service, context compactor, and conversation store each own part of context state.

Target:

- `ContextService` becomes the canonical context policy owner.
- `ConversationStore` becomes the canonical persistence owner.
- `MessageHistory` becomes a projection for provider-compatible message lists.
- `ConversationController` narrows to stage/state-machine compatibility or is retired.

Acceptance criteria:

- One API adds a message.
- One API selects provider context.
- One API applies compaction.
- All compaction summaries persist through the same path.

Progress:

- `ContextServiceRegistry` now scopes in-memory context by `session_id` and `agent_id`, so team members or subagents that share a session boundary do not accidentally share working histories.

### P0: Fix Project Graph Identity

Current concern:

Graph tables use mostly absolute `file` paths while `files` uses relative paths. Generated artifacts are present in the index.

Target:

- Store repo-relative paths in graph and file tables.
- Store project root in project metadata.
- Rebuild graph after exclusion fixes.
- Keep path migration fallback for old rows during transition.

Acceptance criteria:

- `graph_node.file` joins cleanly to `files.path`.
- `site/`, `docs/_build/`, minified assets, caches, and `.victor/` are absent from graph/symbol tables.
- `graph_module_metric` is populated for source modules.

### P1: Retire Legacy Team Coordinator Path

Current concern:

`victor/agent/teams/coordinator.py` duplicates formation execution that should live in `UnifiedTeamCoordinator`.

Target:

- Legacy imports forward to `UnifiedTeamCoordinator` or a thin adapter.
- Formation strategy code exists in one place.
- Planning team execution uses the unified coordinator.

Acceptance criteria:

- All production team runs use `victor.teams.UnifiedTeamCoordinator`.
- Tests assert legacy wrappers delegate to the unified path.

Progress:

- `victor.agent.teams.TeamCoordinator.execute_team()` now delegates formation execution to `UnifiedTeamCoordinator.execute_team_config()` while preserving the legacy compatibility surface.

### P1: Replace Prompt Mutation With Overlays

Current concern:

Planner and research execution can set and restore system prompts around calls.

Target:

- Prompt changes are scoped `TurnContext` overlays.
- Planning, research, skill, and recovery guidance flow through `UnifiedPromptPipeline.compose_turn_prefix()`.
- Stable system prompts remain stable until explicit invalidation.

Acceptance criteria:

- Planner does not call `orchestrator.set_system_prompt()`.
- Prompt overlays are visible in trace metadata.
- Provider cache-tier behavior remains deterministic.

### P1: Unified Checkpoint Contract

Current concern:

Graph checkpoints, git checkpoints, conversation persistence, and tool call state are separate.

Target:

Introduce an `ExecutionCheckpoint` contract:

```text
ExecutionCheckpoint
  id
  graph_checkpoint_id
  session_id
  conversation_checkpoint_id
  filesystem_checkpoint_id
  triggering_tool_call
  approval_state
  created_at
  metadata
```

Acceptance criteria:

- Before a file-changing tool runs, the runtime can create an execution checkpoint.
- Restore can recover conversation state and filesystem state together.
- Checkpoints are traceable from graph execution history.

## Competitive Positioning

Victor should be positioned as:

- LangGraph-compatible in spirit for durable stateful orchestration
- DSPy-aligned for prompt/program optimization
- AutoGen/CrewAI-aligned for team coordination patterns
- OpenAI Agents SDK-aligned for clear agents, tools, handoffs, guardrails, and tracing
- Claude Code-aligned for subagent isolation and tool scoping
- Gemini CLI-aligned for restoreable coding workflows

The differentiator is combining these into one provider-agnostic, SDK-first, project-graph-aware coding and agent framework.

## Migration Sequence

1. Document canonical owners and mark compatibility owners.
2. Add adapter tests proving legacy paths delegate to canonical services.
3. Move planner execution to `StateGraph`.
4. Move planner prompt guidance to `UnifiedPromptPipeline` overlays.
5. Make team execution use `UnifiedTeamCoordinator` only.
6. Normalize graph path identity and rebuild the project graph.
7. Add unified execution checkpoint contract.
8. Populate module metrics and wire them into graph/search ranking.
9. Remove or archive dead compatibility surfaces.

## Implementation Progress

### 2026-05-13: Planner Prompt Isolation

Status: started P1 prompt-overlay migration.

Changes:

- `AutonomousPlanner` no longer mutates `orchestrator.set_system_prompt()` for plan generation or research steps.
- Planner and research guidance now flow through scoped `runtime_context_overrides`.
- `ChatService.chat()` forwards scoped runtime overrides to the bound turn executor when supported.
- `TurnExecutor.execute_agentic_loop()` carries those overrides into the existing `AgenticLoop` turn context so `execute_turn()` can apply and restore them through the canonical runtime override path.
- `AgenticLoop` now treats `runtime_context_overrides` as the generic state key and mirrors `topology_overrides` only for compatibility with existing topology callers.
- `UnifiedPromptPipeline` now accepts named `PromptOverlay` objects on `TurnContext`, and planner/research guidance uses named `prompt_overlays` instead of raw system-prompt replacement.
- Session prompt metadata now preserves trace-safe prompt overlay names/placement and emits a `prompt_overlays.active` usage event without persisting overlay prompt text.

Follow-up:

- Replace remaining non-planner `system_prompt` runtime override call sites with named overlays where they are dynamic per-turn guidance rather than true session prompt replacement.

## Anti-Goals

- Do not create a new multi-agent graph type.
- Do not put new production behavior into `victor.verticals` compatibility paths.
- Do not add another context manager or compaction owner.
- Do not bypass services from graph nodes.
- Do not make graph/vector dependencies mandatory for basic Python operation.
- Do not hand-edit generated documentation output.

## Open Decisions

1. Should graph path migration rewrite existing rows in-place, or should Victor rebuild `.victor/project.db` after the identity fix?
2. Should `ExecutionCheckpoint` live in `victor/framework` as a general primitive or in `victor/agent` as a coding-agent runtime primitive?
3. Should planner state use the existing `ExecutionPlan` dataclasses or a new Pydantic state model aligned to `StateGraph`?
4. Should `ConversationController` remain as a state-machine compatibility adapter or be fully absorbed by `ContextService` and `SessionService`?
5. Which trace schema should be canonical for LLM generation, tool calls, handoffs, guardrails, graph transitions, and checkpoints?
