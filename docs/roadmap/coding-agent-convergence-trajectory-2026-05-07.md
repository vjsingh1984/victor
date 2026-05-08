# Coding-Agent Convergence Trajectory

Date: 2026-05-07
Owner: `victor/agent` canonical runtime owners

## Target Shape

Victor should feel like one coding agent with one obvious runtime path:

- `ChatService` owns canonical task execution entry and reporting hooks.
- `ToolService`, `ContextService`, `ProviderService`, `SessionService`, and `RecoveryService` own effectful runtime behavior.
- `AgentOrchestrator` becomes a composition root and compatibility facade, not a business-logic sink.
- Prompt and instruction import behavior is explicit, deterministic, and cache-invalidated by signature rather than guesswork.
- Token efficiency is measured per task, not inferred from logs after the fact.

## Sequencing

The work should land in four phases because later phases depend on earlier visibility and canonical ownership.

### Phase 0: Foundations

Scope:
- Reuse one shared instruction-discovery path for `.victor`, `AGENTS.md`, `CLAUDE.md`, and compatibility files.
- Add canonical per-task reports on the `ChatService` path.
- Expose task-level token, cache, tool-schema, and compaction metrics through `MetricsCoordinator`.

Exit criteria:
- Runtime and framework prompt builders discover the same instruction files in the same order.
- `ProjectContext` reports explicit import behavior and supports signature-based invalidation.
- Every canonical task can produce a structured report with:
  - API prompt/completion/total tokens
  - cache-hit rate
  - tool-schema tokens
  - compaction savings
  - tokens per successful task

Initial implementation in this change:
- Shared instruction discovery now drives `ProjectContext`, `ProjectContextDiscovery`, and init synthesis enrichment.
- `ChatService` now starts and finishes task reports on the canonical runtime path.
- `MetricsCoordinator` now owns structured task reports and task history.

### Phase 1: Runtime Convergence And Compaction Continuity

Scope:
- Keep shrinking `AgentOrchestrator` by moving effectful runtime work behind service owners.
- Remove new production logic from deprecated coordinator-named seams.
- Make compaction continuity explicit and model-aware.

Implementation track:
- Move remaining live chat-flow branches out of `AgentOrchestrator` into `ChatService` and `StreamingChatExecutor`.
- Introduce a canonical continuation ledger containing:
  - task intent
  - active plan
  - open tool loops
  - completion criteria
- Compact before large tool-output injection, not after context is already polluted.
- Add model-specific compaction policies keyed by provider/model capabilities.

Exit criteria:
- `AgentOrchestrator` no longer owns task execution branches that duplicate `ChatService` or `StreamingChatExecutor`.
- Post-compaction continuation prompts always include explicit prior intent/plan state.
- Compaction policy selection is observable in task reports and streaming metrics.

### Phase 2: Code Intelligence And UX Narrowing

Scope:
- Productize existing LSP and code-navigation capabilities.
- Ship an opinionated coding-agent mode instead of exposing framework breadth by default.

Implementation track:
- Promote symbol lookup, diagnostics, references, and workspace navigation into first-class coding workflows and prompt guidance.
- Teach tool selection and planning to prefer symbol/diagnostic navigation before blind file reads.
- Add a narrow default mode surface:
  - `plan`
  - `build`
  - `review`
  - `delegate`
- Keep advanced framework and workflow controls behind explicit opt-in.

Exit criteria:
- The default coding workflow can complete common navigation tasks through symbols/diagnostics without broad file reads.
- The default UX exposes a small, documented set of coding modes with clear semantics.

### Phase 3: Parallel Work Productization

Scope:
- Turn formations and teams into a practical coding workflow, not only an architecture primitive.

Implementation track:
- Add worktree isolation for delegated coding tasks.
- Add concise worker return contracts:
  - task summary
  - changed files
  - validation run
  - merge risks
- Add orchestrated review/merge loops on top of existing formations.

Exit criteria:
- Multi-agent execution can run isolated code changes with deterministic merge/review contracts.
- Parallel work has measurable throughput and regression metrics.

### Phase 4: Competitive Benchmarking

Scope:
- Replace one-off README success-rate claims with durable, publishable coding-agent benchmarks.

Benchmark set:
- issue-fix success rate
- review bug-catch rate
- tokens to merge
- time to first edit
- cost per accepted patch

Exit criteria:
- Benchmark scripts are reproducible and versioned.
- Public reporting distinguishes:
  - model effects
  - runtime-policy effects
  - tool-selection effects

## Ownership Map

- `ChatService`: canonical task lifecycle, task reporting hooks, non-deprecated chat path
- `StreamingChatExecutor`: streaming loop, continuation continuity, compaction-aware flow
- `MetricsCoordinator`: task reports, token/cost/cache accounting, benchmark-ready exports
- `ProjectContext` and prompt builders: explicit instruction import behavior and invalidation
- `ContextService`: compaction policy, savings metrics, overflow prevention
- `ToolService` plus LSP tools: code-intelligence-first navigation

## Guardrails

- Do not create a second multi-agent graph abstraction.
- Do not add new business logic to deprecated coordinator surfaces.
- Do not add new instruction-loading one-offs outside the shared discovery module.
- Do not treat token-efficiency claims as complete without task-level measurements.

## Validation

Minimum validation for each milestone:

- `python -m pytest tests/unit/context/test_project_context.py`
- `python -m pytest tests/unit/agent/services/test_chat_service.py`
- `python -m pytest tests/unit/agent/services/test_metrics_service.py`
- `python -m pytest tests/unit/agent/services/test_prompt_builder_runtime.py`

Broader milestones should also add integration coverage for:

- compaction continuity
- LSP-first navigation paths
- delegated multi-agent worktree execution
- benchmark report generation
