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
| 2.2 Richer retrieval diagnosis classes | Not started | Add workflow decision tests | Build on `victor-rag` repair flow |
| 2.3 Utility-aware retrieval ranking | Not started | Add retrieval-ranking tests | Use existing retrieval utility stage as insertion point |

## Phase 3: Context Budget and Prompt Compression

Objective:
- Make context and prompt compression benchmark-aware, not heuristic-only.

Planned slices:

| Slice | Status | TDD plan | Notes |
|---|---|---|---|
| 3.1 Prompt-section measurement hooks | Not started | Add allocator metrics tests | Build on `PromptSectionBudgetAllocator` |
| 3.2 Dictionary compression for repeated tool/prompt boilerplate | Not started | Add round-trip compression tests | Must be lossless |
| 3.3 Safe default-on preview pruning for read-only tools | Not started | Add tool-executor pruning tests | Preserve full LLM context until explicitly changed |

## Phase 4: Memory Evolution

Objective:
- Move from passive memory federation toward proactive and trace-aware memory improvement.

Planned slices:

| Slice | Status | TDD plan | Notes |
|---|---|---|---|
| 4.1 Dual-trace memory encoding | Not started | Add memory retrieval tests | Separate semantic and execution traces |
| 4.2 Memory transfer hooks across verticals | Not started | Add policy / filter tests | Avoid naive cross-project leakage |
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
