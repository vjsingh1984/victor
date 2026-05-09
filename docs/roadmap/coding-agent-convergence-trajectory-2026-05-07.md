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

The work should land in five phases (0-4) because later phases depend on earlier
visibility, canonical ownership, and measurable runtime behavior.

## Current Status And Criticality

Verified snapshot: 2026-05-08

| Priority | Phase | Item | Status | Current state / next action |
| --- | --- | --- | --- | --- |
| Critical | 3 | Workspace-isolated delegation as the canonical parallel-work seam | In progress | `WorkspaceIsolationService` owns planning, materialization, changed-file collection, merge orchestration, cleanup policy, and structured diagnostics; `WorkspaceIsolationPolicy` centralizes mode/materialization/dry-run/merge/cleanup flags; `UnifiedTeamCoordinator` surfaces workspace diagnostics in shared state, result payloads, and delegate follow-up/approval contracts; `MetricsCoordinator`, `BenchmarkAgent`, and team-feedback aggregation now preserve workspace policy/diagnostic fields for task and benchmark reports while public `worktree_*` payloads are preserved. Chat workflow mode, `/delegate-follow-up`, and `/delegate-follow-up list` now consume delegate follow-up contracts through the same `TeamStep` state path as `victor workflow run`. Next: carry saved-contract suggestions directly in runtime artifacts/tool output where useful. |
| Critical | 3 | Deterministic delegate follow-up and resume contracts | In progress | Worker contracts, merge-review contracts, preserved follow-up worktrees, direct follow-up request/contract execution, `IDelegateFollowUpCoordinator`, workflow `TeamStep` routing, `victor workflow run --delegate-follow-up-contract ... --delegate-next-step-id ...`, `victor chat --workflow ... --delegate-follow-up-contract ... --delegate-next-step-id ...`, `/delegate-follow-up <workflow> <contract> [step_id]`, and selectable saved-contract summaries via `/delegate-follow-up list <workflow> <contract>` are live. Next: surface those suggestions automatically from delegate result artifacts without adding local delegate runtimes. |
| Critical | 4 | Reproducible published benchmark suites (`issue-fix`, `review bug-catch`, merge cost/time) | In progress | Task reports, merge-efficiency metrics, reusable report bundles, multi-artifact compare, fixture manifests, portable publication bundles, publication-catalog consumption, stable publisher summaries, and full catalog-coverage reporting are live across checked-in fixture-set examples. Next: publish durable benchmark corpora depth and stable-run outputs for public comparison. |
| Critical | 2 | Code-intelligence-first live navigation | In progress | The runtime blocks or rewrites broad code reads toward `lsp`, `symbol`, `refs`, and `project_overview`, including symbol-aware empty-result fallback and follow-up narrowing. Next: extend multi-step follow-through and document the default code-intelligence workflow. |
| High | 1 | Runtime convergence and deprecated seam removal | In progress | Deprecated chat runtime wrapper methods and module-level `ChatCompatRuntimeProtocol` are removed, and canonical helper getters are the supported planning/context-limit seam. Next: keep auditing `AgentOrchestrator` so new production behavior lands in services, not compatibility facades. |
| High | 4 | Distinguish runtime-policy, workspace-policy, and tool-selection effects in reports | In progress | Planning-policy deltas, code-intelligence tool-selection deltas, prompt/runtime variant metadata, fixture manifests, task-report workspace policy/diagnostic fields, benchmark team-feedback workspace rollups, and bundled artifacts are preserved. Next: surface workspace-policy effects in comparison tables across saved runs. |
| High | 2 | Narrow default coding UX | In progress | `plan`, `build`, `review`, and `delegate` modes exist and runtime steering is live. Next: make advanced framework/workflow controls opt-in and document the small default coding-agent mode surface. |
| Medium | 1 | Continuation ledger and model-aware compaction continuity | Done | The canonical streaming path preserves intent/plan state, performs pre-tool-output compaction for large outputs, emits post-compaction continuation prompts, and reports compaction strategy metadata. Keep regression coverage as a guardrail. |
| Medium | 0 | Unified instruction discovery and per-task efficiency reporting | Done | Shared instruction discovery, `AGENTS.md`/`CLAUDE.md` compatibility, signature invalidation, and canonical task reports are in place. Keep task-level token/cache/tool-schema/compaction reporting as a release gate. |

## Verified Implementation Anchors

- Phase 0: `victor/context/instruction_discovery.py`, `ProjectContext`, and `MetricsCoordinator` provide shared instruction discovery and task reports with API tokens, cache-hit rate, tool-schema tokens, compaction savings, and tokens per successful task.
- Phase 1: `StreamingChatExecutor`, `ToolExecutionRuntime`, and the session ledger preserve continuation state across compaction and compact before large tool-output injection.
- Phase 2: `tool_pipeline.py` steers broad code reads toward `lsp`, `symbol`, `refs`, and `project_overview`; default modes are present but need tighter docs and opt-in boundaries.
- Phase 3: `WorkspaceIsolationService`, `WorkspaceIsolationPolicy`, `GitWorktreeRuntime`, `UnifiedTeamCoordinator.execute_follow_up_contract`, `TeamStep`, and workflow CLI follow-up injection form the canonical delegated-work path.
- Phase 4: benchmark fixture catalogs, publication bundles, runtime-variant summaries, merge-efficiency metrics, and workspace-policy/diagnostic rollups exist; public stable-run depth remains the gap.

## Refined Next Execution Order

| Order | Priority | Phase | Work Item | Correct Seam | Done When |
| --- | --- | --- | --- | --- | --- |
| 1 | Critical | 4 | Publish stable benchmark corpora and run outputs | Benchmark fixture/catalog/report pipeline | Public reports include issue-fix success, review bug-catch, tokens-to-merge, time-to-first-edit, and cost per accepted patch across stable corpora. |
| 2 | Critical | 2 | Complete code-intelligence-first default navigation | `ToolService`/tool pipeline/LSP tools | Default coding workflow completes common navigation through symbols/diagnostics/refs/project overview before broad reads, with regression tests. |
| 3 | High | 4 | Report workspace-policy effects in comparisons | Benchmark report exporters | Comparison reports distinguish model, runtime policy, workspace policy, and tool-selection effects across runs. |
| 4 | High | 2 | Narrow and document default UX | mode controller, slash/CLI docs, coding-agent docs | `plan`, `build`, `review`, and `delegate` are the documented default surface; advanced framework controls are explicit opt-in. |
| 5 | High | 3 | Auto-surface delegate follow-up suggestions from runtime artifacts | Result artifact formatting only; execution stays workflow/`TeamStep` | Delegate result artifacts expose concise follow-up suggestions automatically when contract file/workflow context is known. |
| 6 | High | 1 | Continue `AgentOrchestrator` shrinkage | service owners only | New production behavior is owned by `ChatService`, `ToolService`, `ContextService`, `ProviderService`, `SessionService`, `RecoveryService`, or team workspace services. |
| 7 | Medium | 0-1 | Preserve completed guardrails | focused regression suites | Instruction discovery, task reports, compaction continuity, and model-aware continuation stay green while later phases move. |

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
