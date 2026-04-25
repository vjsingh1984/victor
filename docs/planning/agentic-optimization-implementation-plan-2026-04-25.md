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
| 4.3 Proactive memory hints | Not started | Add next-turn hint generation tests | PRiME-inspired, bounded first version |

## Phase 5: Benchmark Harnesses

Objective:
- Add realistic benchmark discipline for deep research, browser, and GUI workflows.

Planned slices:

| Slice | Status | TDD plan | Notes |
|---|---|---|---|
| 5.1 Deep-research benchmark adapter | Not started | Add harness tests | DR3-Eval-aligned first |
| 5.2 Browser / web-task benchmark adapter | Not started | Add benchmark catalog tests | ClawBench-aligned |
| 5.3 Hierarchical failure taxonomy | Not started | Add evaluator diagnosis tests | GUIDE-aligned |

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
