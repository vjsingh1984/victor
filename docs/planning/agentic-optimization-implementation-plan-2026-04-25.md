# Agentic Optimization Implementation Plan

Date: 2026-04-25

Scope:
- `victor/framework/rl/`
- `victor/agent/`
- `victor/framework/`
- `victor-research` / `victor-rag` integration points when needed

Goal:
- Reuse the strong existing agentic stack and enhance it incrementally with TDD.
- Prefer high-ROI improvements that close existing loops before building net-new systems.

## Working Rules

- Start with enhancements that build on existing mechanisms.
- Use tests first for each implementation slice.
- Keep each slice independently shippable.
- Favor feature additions that degrade gracefully when data is absent.
- Track each slice with `Not started`, `In progress`, `Done`, or `Deferred`.

## Phase 1: Prompt Evolution Governance

Objective:
- Add benchmark-gated candidate lifecycle management on top of the existing GEPA / MiPRO / CoT prompt optimizer stack.

Rationale:
- Prompt evolution already exists.
- Candidate promotion and rollback are not benchmark-disciplined.
- This is high leverage and bounded enough for TDD.

Planned slices:

| Slice | Status | TDD plan | Notes |
|---|---|---|---|
| 1.1 Candidate benchmark metadata persistence | Done | Added failing unit tests for load/save of benchmark fields first | Reused existing `agent_prompt_candidate` table with additive columns |
| 1.2 Benchmark result recording API | Done | Added failing unit tests for score/runs/pass state updates first | Running-average implementation landed |
| 1.3 Promotion / active-candidate switching | Done | Added failing unit tests for promotion semantics first | Reused existing `is_active` column |
| 1.4 Rollback to prior approved candidate | Done | Added failing unit tests for fallback behavior first | Deterministic approved-candidate fallback landed |
| 1.5 Recommendation gating preference | Done | Added failing unit tests for active/approved candidates being preferred first | Backward-compatible fallback preserved when no benchmark data exists |

Definition of done:
- Unit coverage for each slice
- No regression to current recommendation behavior when benchmark data is absent
- Candidate state persists through DB reload

## Phase 2: Search-First and Retrieval Repair

Objective:
- Strengthen `search-first` behavior and richer failure-aware retrieval repair.

Planned slices:

| Slice | Status | TDD plan | Notes |
|---|---|---|---|
| 2.1 Search-first coding fallback hints | Done | Added prompt-pipeline tests first | Prompt completeness guard now injects search-first guidance for symbol-scoped requests without file paths |
| 2.2 Richer retrieval diagnosis classes | Done | Added workflow decision tests first | `victor-rag` now classifies repair gaps explicitly and routes repair vs revise vs clarify through a stable vocabulary |
| 2.3 Utility-aware retrieval ranking | Done | Added retrieval-ranking tests first | Replaced the inline utility script with a named `victor-rag` transform that scores authority, diversity, and redundancy and reranks results boundedly |

## Phase 3: Context Budget and Prompt Compression

Objective:
- Make context and prompt compression benchmark-aware, not heuristic-only.

Planned slices:

| Slice | Status | TDD plan | Notes |
|---|---|---|---|
| 3.1 Prompt-section measurement hooks | Done | Added allocator metrics tests first | `PromptSectionBudgetAllocator` now records observed token costs and can budget against measured section sizes when available |
| 3.2 Dictionary compression for repeated tool/prompt boilerplate | Done | Added round-trip compression tests first | Added a lossless prompt dictionary compressor and wired it into `UnifiedPromptPipeline` for repeated per-turn guidance blocks |
| 3.3 Safe default-on preview pruning for read-only tools | Done | Added tool-result processing and streaming renderer tests first | Read-only tool results now surface a pruned preview to users by default while preserving full formatted output for expansion/debug and full LLM context for execution |

## Phase 4: Memory Evolution

Objective:
- Move from passive memory federation toward proactive and trace-aware memory improvement.

Planned slices:

| Slice | Status | TDD plan | Notes |
|---|---|---|---|
| 4.1 Dual-trace memory encoding | Done | Added store, controller, and adapter tests first | Conversation memory now persists semantic vs execution traces separately and can retrieve both buckets through the store/controller/adapter path |
| 4.2 Memory transfer hooks across verticals | Done | Added policy / filter tests first | Unified memory transfer now accepts project / vertical / transfer-group context, blocks scoped cross-project reuse by default when project scope is known, and allows bounded opt-in reuse across matching transfer groups or verticals |
| 4.3 Proactive memory hints | Done | Added next-turn hint generation tests first | Unified memory now derives bounded proactive hints from successful traces, tracks latest next-turn hints, and keeps them scoped through the same transfer policy used for cross-session reuse |

## Phase 5: Benchmark Harnesses

Objective:
- Add realistic benchmark discipline for deep research, browser, and GUI workflows.

Planned slices:

| Slice | Status | TDD plan | Notes |
|---|---|---|---|
| 5.1 Deep-research benchmark adapter | Done | Added catalog + harness tests first | Added a dedicated DR3-style deep-research runner under `victor/evaluation/benchmarks`, with manifest loading plus claim/citation/unsupported-claim scoring integrated into the shared benchmark catalog |
| 5.2 Browser / web-task benchmark adapter | Done | Added benchmark catalog + browser-runner tests first | Added a dedicated browser-task runner for `clawbench` / `guide` / `vlaa-gui`, updated the shared catalog and CLI routing, and evaluate action traces plus final-answer coverage through local manifests |
| 5.3 Hierarchical failure taxonomy | Done | Added evaluator diagnosis + persistence tests first | Added a structured failure diagnosis layer on `TaskResult`, aggregate stage/path metrics in `EvaluationResult`, and persisted taxonomy output in the benchmark harness while preserving the existing flat failure-category contract |

## Phase 6: Calibration and Confidence

Objective:
- Add truth-aligned confidence signals on the canonical benchmark evaluation path.

Planned slices:

| Slice | Status | TDD plan | Notes |
|---|---|---|---|
| 6.1 Truth-aligned confidence assessment | Done | Added task/evaluation/harness confidence tests first | `TaskResult` now derives confidence + uncertainty from objective evidence and failure taxonomy, and benchmark summaries persist confidence buckets plus truth-alignment rates |
| 6.2 Clarification-aware runtime gating | Done | Added perception, streaming-pipeline, and loop-evaluation tests first | `PerceptionIntegration` now flags underspecified action requests, the streaming pipeline exits early with a targeted clarification prompt before provider/tool execution, and `AgenticLoop` fails fast with clarification metadata instead of blind retry |
| 6.3 Low-confidence retry budget | Done | Added loop-evaluation tests first | `AgenticLoop` now bounds repeated low-confidence retries on both enhanced and legacy evaluation paths, resets the budget when progress resumes, and emits structured exhaustion metadata when the retry budget is spent |

## Immediate Execution Order

1. Finish Phase 1 end-to-end.
2. If tests are stable, start Phase 2 slice 2.1 or 2.2 depending on coupling.
3. Leave benchmark harness work until the governance and runtime loops are stronger.

## Tracking Log

| Date | Update |
|---|---|
| 2026-04-25 | Plan created. Phase 1 selected as first implementation target. |
| 2026-04-25 | Phase 1 completed via TDD: benchmark state persistence, benchmark result recording, promotion, rollback, and recommendation gating preference implemented in `PromptOptimizerLearner`. |
| 2026-04-25 | Phase 2.1 completed via TDD: prompt completeness guard now emits search-first guidance for symbol-scoped requests without explicit file paths. |
| 2026-04-25 | Phase 2.2 completed via TDD in `victor-rag`: retrieval gaps are now explicitly classified and repair policy routes between retrieval repair, answer revision, and clarification accordingly. |
| 2026-04-25 | Phase 2.3 completed via TDD in `victor-rag`: retrieval utility scoring moved to a named escape hatch that emits richer metrics and applies bounded authority/diversity-aware reranking. |
| 2026-04-25 | Phase 3.1 completed via TDD in `codingagent`: prompt section budgeting now records rolling token-cost measurements and uses them to improve future section selection under budget constraints. |
| 2026-04-25 | Phase 3.2 completed via TDD in `codingagent`: repeated long guidance blocks can now be dictionary-compressed losslessly, and `UnifiedPromptPipeline` uses the compressor for repeated per-turn reminder content when it produces real savings. |
| 2026-04-25 | Prompt architecture refactor inserted before Phase 3.3. Roadmap resume point remains Phase 3.3; details tracked in `docs/planning/prompt-canonical-architecture-refactor-2026-04-25.md`. |
| 2026-04-25 | Phase 3.3 completed via TDD in `codingagent`: read-only tool results now default to pruned user previews while keeping full model-visible output intact and preserving full output for expansion/debug in streaming renderers. |
| 2026-04-25 | Phase 5.3 completed via TDD in `codingagent`: benchmark results now derive and persist a hierarchical failure diagnosis with stage + subtype paths, and aggregate metrics report both flat categories and taxonomy breakdowns across DR3/browser/external benchmark adapters. |
| 2026-04-25 | Phase 6.1 completed via TDD in `codingagent`: benchmark results now derive truth-aligned confidence/uncertainty from evidence plus failure taxonomy, and persisted reports include confidence buckets and alignment rates for calibration-aware triage. |
| 2026-04-25 | Phase 6.2 completed via TDD in `codingagent`: underspecified low-confidence action requests now trigger targeted clarification from `PerceptionIntegration`, the streaming runtime returns that clarification before provider/tool execution, and `AgenticLoop` records clarification-required failure metadata instead of retrying blindly. |
| 2026-04-25 | Phase 6.3 completed via TDD in `codingagent`: repeated low-confidence retries are now budgeted in `AgenticLoop`, including results returned by the enhanced completion evaluator, and the loop resets that budget on renewed progress while surfacing structured exhaustion metadata on failure. |
| 2026-04-25 | Runtime-intelligence consolidation inserted before the next feature slice. The architecture and migration plan is tracked in `docs/planning/runtime-intelligence-consolidation-2026-04-25.md`, and the first implementation slice migrated `UnifiedPromptPipeline`, `StreamingChatPipeline`, `ServiceStreamingRuntime`, `AgenticLoop`, and orchestrator/factory wiring onto a shared `RuntimeIntelligenceService` boundary. |
| 2026-04-25 | Runtime-intelligence consolidation Phase 3.1 completed via TDD in `codingagent`: task completion, thinking-loop detection, continuation strategy, intent-classification handoff, and workflow factory construction now depend on `RuntimeIntelligenceService` instead of wiring low-level decision services directly on the active runtime path. |
