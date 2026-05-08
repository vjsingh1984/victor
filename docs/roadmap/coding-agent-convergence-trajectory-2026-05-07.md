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

## Current Status And Criticality

Status snapshot after the current convergence stream:

| Priority | Phase | Item | Status | Current state / next action |
| --- | --- | --- | --- | --- |
| Critical | 4 | Reproducible published benchmark suites (`issue-fix`, `review bug-catch`, merge cost/time) | In progress | Task reports, merge-efficiency metrics, reusable report bundles, multi-artifact compare, runtime-variant summaries, fixture manifests, and bundled local result copies are live; the remaining gap is durable benchmark corpora plus checked-in/public fixture sets built from stable runs. |
| Critical | 3 | Practical default worktree-isolated delegation flow | In progress | Worker contracts, merge-review contracts, auto-materialized delegate worktrees, safe auto-merge for execution-eligible plans, preserved follow-up worktrees, and executable delegate re-entry contracts are live; the remaining gap is tighter approval UX and richer delegated review/fix retry ergonomics on top of that resume path. |
| Critical | 2 | Code-intelligence-first live navigation | In progress | The runtime now blocks or rewrites broad code reads toward `lsp`, `symbol`, `refs`, and `project_overview`, including symbol-aware empty-result fallback; the remaining gap is deeper follow-through after refs/project-overview and tighter default UX documentation. |
| High | 1 | Remove remaining deprecated runtime wrapper surfaces | In progress | Canonical chat/runtime helpers are already preferred, wrapper methods are deprecated, and the package-level service protocol surface no longer re-exports `ChatCompatRuntimeProtocol`; the remaining gap is deleting the wrapper methods and module-level compat protocol entirely. |
| High | 4 | Distinguish runtime-policy and tool-selection effects in reports | In progress | Planning-policy deltas, code-intelligence tool-selection deltas, prompt/runtime variant metadata, saved fixture manifests, and bundled local artifacts are now preserved in comparison bundles; the remaining gap is richer public breakdowns on stable corpora. |
| High | 2 | Narrow default coding UX | In progress | `plan`, `build`, `review`, and `delegate` modes exist and runtime steering is live; advanced framework controls still need stronger opt-in boundaries and docs. |
| Medium | 1 | Continuation ledger and model-aware compaction continuity | Done | The canonical streaming path preserves intent/plan state and reports compaction strategy metadata. |
| Medium | 0 | Unified instruction discovery and per-task efficiency reporting | Done | Shared instruction discovery, signature invalidation, and canonical task reports are in place and should now be treated as guardrails. |

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
- Add orchestrated review/merge loops and deterministic re-entry contracts on top of existing formations.

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
