# Agent Runtime Competitive Gap Analysis

Date: 2026-05-13

Status: active architecture tracker

Scope: `victor/agent`, `victor/framework`, `victor/teams`, autonomous planning, runtime context, message history, compaction, lifecycle management, and project graph data in `.victor/project.db`.

## Executive Summary

Victor has the right strategic direction: service-first runtime ownership, `StateGraph` as the execution engine, teams as formations, SDK-first plugin boundaries, and prompt optimization through `UnifiedPromptPipeline`. These choices align with the strongest patterns in LangGraph, AutoGen, CrewAI, DSPy, OpenAI Agents SDK, Claude Code, and Gemini CLI.

The main architectural risk is incomplete consolidation. Canonical services, facades, runtime helpers, coordinators, managers, and legacy compatibility modules still coexist for the same responsibilities. This creates isolation and duplication around conversation state, compaction, planning, team execution, and lifecycle management.

The most important design move is to make the durable graph runtime the common substrate for chat, autonomous planning, team formations, and long-running work. Services should own effectful behavior. Facades should only group or forward. Compatibility paths should shrink over time.

## What Is Strong

1. Service-first target shape is clear.

   Canonical services exist for chat, tools, sessions, context, provider management, and recovery:

   - `victor/agent/services/chat_service.py`
   - `victor/agent/services/tool_service.py`
   - `victor/agent/services/session_service.py`
   - `victor/agent/services/context_service.py`
   - `victor/agent/services/provider_service.py`
   - `victor/agent/services/recovery_service.py`

   This aligns with modern production agent designs where orchestration is a thin composition root and effectful owners are explicit.

2. `StateGraph` is a strong execution primitive.

   `victor/framework/graph.py` supports typed state, conditional edges, subgraphs, checkpointing, replay, and interrupts. `victor/framework/graph_execution.py` adds iteration control, copy-on-write state, timeout handling, checkpoint save/load, and graph events. This aligns closely with LangGraph's durable, stateful orchestration model.

3. Teams are correctly modeled as formations, not separate graph engines.

   `victor/teams/unified_coordinator.py` treats sequential, parallel, hierarchical, pipeline, and consensus as formation strategies and supports direct `StateGraph` node invocation through `__call__`. This is the correct direction: teams should be logical coordination patterns inside the normal execution graph.

4. Prompt architecture is competitive.

   `victor/agent/prompt_pipeline.py` separates stable system prompt construction from per-turn dynamic guidance. It also accounts for provider cache tiers and can inject GEPA, MiPRO-style few-shots, failure hints, credit guidance, skills, and context reminders. This is one of Victor's strongest design assets and is directionally aligned with DSPy-style optimization.

5. Project-scoped graph intelligence is valuable.

   `.victor/project.db` contains a rich code graph with symbol nodes, CCG/CFG/CDG/DDG edges, sessions, messages, summaries, and FTS indexes. The architecture has the right idea: code intelligence belongs in the project database, while user-wide learning belongs in the global database.

## Primary Gaps

### 1. Runtime Ownership Is Still Split

`AgentOrchestrator` is still a large composition root and compatibility surface. It initializes provider runtime, memory runtime, metrics runtime, workflow runtime, coordination runtime, resilience runtime, interaction runtime, credit runtime, and service registration. It is moving toward a thin facade, but it still knows too much.

Risk:

- New behavior can accidentally land in the orchestrator instead of canonical services.
- Service bootstrap order becomes hard to reason about.
- Runtime state can drift between orchestrator attributes and service state.

Design direction:

- Treat `AgentOrchestrator` as a temporary compatibility shell.
- Move new production behavior to canonical services or graph nodes.
- Keep orchestrator methods as adapters over `RuntimeExecutionContext` and service calls.

### 2. Conversation, Context, and Compaction Have Too Many Owners

Current overlapping surfaces include:

- `victor/agent/message_history.py`
- `victor/agent/conversation/controller.py`
- `victor/agent/conversation/store.py`
- `victor/agent/context_compactor.py`
- `victor/agent/services/context_service.py`
- `victor/agent/compaction_router.py`
- `victor/agent/compaction_hierarchy.py`

Risk:

- Message type conversion, token accounting, system prompt insertion, persistence, compaction summaries, and context metrics can diverge.
- A bug fixed in one context path may remain in another.
- Compaction may preserve or drop different information depending on which owner is active.
- Context isolation can leak when in-memory context scope is keyed too broadly, such as sharing a session id across multiple agent identities.

Design direction:

- Make `ContextService` the policy and lifecycle owner.
- Make `ConversationStore` the persistence owner.
- Make `MessageHistory` a thin in-memory projection, not an independent source of truth.
- Make `ConversationController` a compatibility adapter or narrow state-machine owner.
- Route all compaction through a single `ContextCompactionPolicy` plus strategy implementations.
- Scope runtime context by agent identity as well as session identity.

Current progress:

- Root turn and streaming pre-checks now ask the wired context service for compaction recommendations before falling back to direct legacy compactor calls.
- Planning runtime now uses the same context-service compaction decision path before falling back to the legacy direct compactor.
- The legacy async `ContextManager` compaction path can now use a wired context service before the direct compactor fallback.

### 3. Autonomous Planning Is Not Graph-Native Enough

`victor/agent/planning/autonomous.py` represents plans as a DAG, but execution is still mostly sequential control flow around `orchestrator.chat()`. Planner, framework decorator, and agentic graph prompt guidance now use scoped overlays, and subagent role prompts are constructor-scoped child runtime configuration.

Risk:

- Plans are harder to pause, resume, replay, inspect, or branch.
- Step-level approvals and failures are not first-class graph checkpoints.
- Remaining direct system-prompt mutation weakens prompt cache stability and isolation.

Design direction:

- Express autonomous planning as a `StateGraph`: plan, approve, execute step, evaluate, replan, finish.
- Persist plan state using graph checkpoints.
- Use scoped prompt overlays through `UnifiedPromptPipeline` instead of setting orchestrator system prompts directly.

### 4. Team Execution Has Legacy Duplication

`victor/teams/unified_coordinator.py` is the correct production direction. `victor/agent/teams/coordinator.py` still implements a separate coordinator with its own formation dispatch and execution state.

Risk:

- Behavior diverges between team APIs.
- Tests can pass against the old path while product behavior uses the new path.
- Formation improvements need multiple edits.

Design direction:

- Make `UnifiedTeamCoordinator` the only production team executor.
- Keep `victor/agent/teams/coordinator.py` as a compatibility wrapper or retire it.
- Ensure planning team execution uses `UnifiedTeamCoordinator` directly as a `StateGraph` node.

Current progress:

- Legacy `TeamCoordinator.execute_team()` delegates formation execution to `UnifiedTeamCoordinator.execute_team_config()`, reducing duplicate formation dispatch while keeping old imports stable.
- Legacy `TeamCoordinator.execute_task()` now keeps registered protocol members and caller shared context intact when delegating through the unified coordinator path.

### 4a. Checkpoint Identity Is Fragmented

Victor has graph, conversation, git/filesystem, HITL, policy-learning, and time-budget checkpoints with separate IDs and owners.

Risk:

- Restore flows cannot reliably tell which graph state, conversation snapshot, filesystem snapshot, tool intent, and approval decision belong together.
- Trace events can mention a checkpoint ID without enough context to resume or audit the whole execution point.

Design direction:

- Keep existing checkpoint managers as persistence owners.
- Use `victor.framework.ExecutionCheckpoint` as the framework-level envelope that binds their IDs for file-changing and long-running execution.
- Emit `ExecutionCheckpoint.to_trace_metadata()` on graph events, tool spans, and approval transitions.

Current progress:

- Added the `ExecutionCheckpoint` and `ApprovalState` framework contract with serialization and trace-focused metadata.
- `ToolExecutionRuntime` now creates one `ExecutionCheckpoint` envelope before write-category tool batches, using existing session and filesystem checkpoint owners when present and recording the checkpoint ID in runtime/stream metadata.

### 5. Graph Index Data Needs Hygiene and Canonical Identity

Observed `.victor/project.db` counts:

- `files`: 3,670
- `symbols`: 75,350
- `imports`: 28,777
- `graph_node`: 909,010
- `graph_edge`: 10,271,553
- `sessions`: 393
- `messages`: 8,472
- `context_summaries`: 163
- `compaction_history`: 0

Observed graph shape:

- `statement`: 756,028 nodes
- `function`: 123,289 nodes
- `class`: 22,710 nodes
- `CDG`: 7,437,764 edges
- `CDG_LOOP`: 1,582,320 edges
- `CFG_SUCCESSOR`: 700,724 edges
- `REFERENCES`: 162,535 edges
- `CALLS`: 154,073 edges

Issues:

- `graph_node.file` is mostly absolute paths, while `files.path` is relative. This breaks simple joins and causes `nodes_no_file_record` style mismatches.
- Generated artifacts are indexed, including `site/`, `docs/_build/`, and minified JavaScript.
- `graph_module_metric` is empty, so module ranking, coupling, hotspot, and TDD-priority signals are unavailable.
- CCG/CDG edge volume dominates the graph and can drown higher-level relationship queries unless views and materialized summaries are used.

Current progress:

- New SQLite graph writes now canonicalize project-local node, edge, and mtime file values to repo-relative paths while keeping absolute-path reads and deletes compatible.
- New SQLite graph store initialization records `project_root` and `graph_file_path_identity=repo_relative` in project metadata.
- Relative graph file reads, stale checks, and cleanup now include legacy absolute path variants for transition compatibility.
- Incremental graph indexing now compares discovered, stale, and indexed files with repo-relative graph keys so relative storage does not make unchanged files look deleted.
- Universal graph-index exclusions now cover `htmlcov/` and explicit `docs/_build/` outputs.
- Graph indexing now refreshes module metrics through the existing `ModuleAnalyzer` after graph-changing runs.

Latest local `.victor/project.db` spot check before rebuild:

- `graph_node`: 909,010 rows
- `graph_node.file` absolute paths: 836,513 rows
- generated-artifact graph rows matching `site/`, `.victor/`, `htmlcov/`, or `docs/_build/`: 946 rows
- `files.path` absolute paths: 0 rows
- `graph_module_metric`: 0 rows

Design direction:

- Canonicalize all project graph file identities as repo-relative paths plus project root metadata.
- Enforce generated artifact exclusions before indexing.
- Populate and refresh module metrics incrementally.
- Add precomputed graph views for common queries: module dependencies, runtime calls, important symbols, hotspots, and affected files.

## Best-of-Breed Alignment

### LangGraph and LangSmith

Relevant pattern: low-level durable graph runtime, streaming, human-in-the-loop, memory, and traceability.

Victor alignment:

- `StateGraph`, checkpoints, interrupts, and graph events already exist.

Gap:

- StateGraph is still not the single default runtime for agentic loops, planner execution, and teams.

Reference:

- https://docs.langchain.com/oss/python/langgraph/overview

### DSPy

Relevant pattern: declare AI behavior as modules/signatures, then optimize prompts or weights using metrics.

Victor alignment:

- `UnifiedPromptPipeline`, runtime intelligence, GEPA/MiPRO hooks, credit guidance, and evaluation feedback are strong foundations.

Gap:

- Prompt behavior is still partly string/mutation driven. Planning and task policies should become typed, evaluable prompt modules with metrics.

Reference:

- https://dspy.ai/

### AutoGen

Relevant pattern: high-level AgentChat/Teams over lower-level event-driven Core.

Victor alignment:

- Victor has both high-level teams and low-level graph/runtime primitives.

Gap:

- The separation is not yet clean enough. Some high-level compatibility coordinators still duplicate lower-level orchestration.

Reference:

- https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html

### CrewAI

Relevant pattern: distinguish agents/crews from flows, and use stateful orchestration for durable processes.

Victor alignment:

- Team formations and workflow nodes exist.

Gap:

- Autonomous planner and team execution should be graph/flow-native, not separate control loops.

Reference:

- https://docs.crewai.com/

### OpenAI Agents SDK

Relevant pattern: few primitives, clear handoffs, guardrails, and built-in tracing.

Victor alignment:

- Victor has agents, tools, handoff-like delegation, guardrails/safety concepts, tracing/events, and workflow composition.

Gap:

- Public and internal primitive count is too high. The architecture should expose fewer stable concepts and demote compatibility layers.

References:

- https://openai.github.io/openai-agents-python/
- https://openai.github.io/openai-agents-python/tracing/
- https://openai.github.io/openai-agents-python/guardrails/

### Claude Code

Relevant pattern: specialized subagents with separate context windows, scoped tools, persistent memory, and optional worktree isolation.

Victor alignment:

- Subagents, teams, tool budgets, allowed tools, and worktree isolation exist.

Gap:

- Isolation policy is not declarative enough across all paths. Tool permissions, memory scope, context scope, and worktree scope should be part of a reusable subagent/team spec.

Reference:

- https://code.claude.com/docs/en/sub-agents

### Gemini CLI

Relevant pattern: checkpoint filesystem state, conversation history, and tool call state before file modification.

Victor alignment:

- Git checkpointing and graph workflow checkpointing exist.

Gap:

- Checkpoints are not unified across graph state, conversation state, tool intent, and filesystem state.

Reference:

- https://google-gemini.github.io/gemini-cli/docs/checkpointing.html

## Priority Tracker

| Priority | Area | Recommendation | Target Outcome |
| --- | --- | --- | --- |
| P0 | Runtime substrate | Make `StateGraph` the default execution substrate for agent loop, planning, and teams | Durable, inspectable, resumable execution |
| P0 | Context ownership | Collapse context, history, compaction, and persistence ownership into `ContextService` + `ConversationStore` | One source of truth for message lifecycle |
| P0 | Graph hygiene | Normalize graph file paths and rebuild project graph without generated artifacts | Reliable graph queries and faster retrieval |
| P1 | Team consolidation | Retire legacy team coordinator behavior behind `UnifiedTeamCoordinator` | One formation implementation |
| P1 | Prompt overlays | Replace prompt mutation with scoped `UnifiedPromptPipeline` overlays | Stable cache-friendly prompts and safer planning |
| P1 | Unified checkpoints | Bind graph, conversation, tool call, and git/filesystem snapshots | Safe restore for long-running coding work |
| P2 | Module metrics | Populate `graph_module_metric` incrementally | Better refactor, review, and planning prioritization |
| P2 | Primitive simplification | Collapse duplicated manager/coordinator/facade/runtime classes | Smaller API surface and lower maintenance cost |

## Definition of Done

The architecture should be considered competitively aligned when:

1. Agent loop, autonomous planner, and team execution all run on `StateGraph` by default.
2. Context compaction and message persistence have one canonical policy path.
3. Teams are formations inside the graph runtime, not a separate execution abstraction.
4. Prompt changes are scoped overlays or pipeline inputs, not raw orchestrator mutations.
5. Graph index identity is consistent and excludes generated artifacts.
6. Every long-running or file-changing workflow can checkpoint and restore graph state, conversation state, tool call state, and filesystem state.
7. Observability emits trace spans for LLM generations, tool calls, handoffs/delegation, guardrails, graph transitions, and checkpoints.
