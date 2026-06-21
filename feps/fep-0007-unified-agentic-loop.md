---
fep: 0007
title: "Unified Agentic Loop (single loop, two I/O modes)"
type: Standards Track
status: Draft
created: 2026-06-21
modified: 2026-06-21
authors:
  - name: Vijaykumar Singh
    email: singhvjd@gmail.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/vjsingh1984/victor/discussions/0007
---

# FEP-0007: Unified Agentic Loop

## Summary

Victor runs **two** agentic iteration loops:

- `AgenticLoop.run()` (`victor/framework/agentic_loop.py`) — the headless/buffered loop for
  `victor chat`. A formal phased loop: PERCEIVE → PLAN → ACT → EVALUATE → DECIDE (named
  `LoopStage`s), returning a `LoopResult`.
- `StreamingChatExecutor.run()` (`victor/agent/services/chat_stream_executor.py`) — the live
  streaming loop for `victor ui` / `Agent.stream()`, created by
  `ChatStreamRuntime.get_executor()` (`chat_stream_runtime.py:159`). A procedural loop that
  yields `StreamChunk`s token-by-token. It has **no PLAN/DECIDE phase**, receives perception
  injected from upstream rather than running a PERCEIVE phase, and reimplements the per-turn
  EVALUATE guards inline.

The **only fundamental difference is I/O timing** — buffered `LoopResult` vs streamed
`AsyncIterator[StreamChunk]`. Everything else (perceive, the per-turn continue/nudge/stop
decision) is conceptually the same and was, until recently, duplicated. This FEP proposes
collapsing the two into **one loop with two I/O modes**, so PERCEIVE / PLAN / EVALUATE /
DECIDE exist once and only ACT differs (buffered turn vs streamed turn).

## Motivation

- **Maintenance surface.** Every loop-quality fix (loop guards, convergence, recovery) has had
  to be written twice and has **drifted** — e.g. content-repetition detection used a cruder
  content-length heuristic in the headless loop and a better hash/overlap detector in the
  streaming loop. The plateau formula and nudge activation also diverged.
- **Capability gap.** The streaming loop lacks PLAN/DECIDE and the structured fulfillment /
  convergence machinery the headless loop has, so the UI path is "thinner" and behaves
  differently from the CLI path for the same task.
- **Target architecture.** CLAUDE.md states the runtime is service-first and "AgenticLoop is
  the canonical execution authority for chat," and explicitly notes "the streaming path
  (StreamingChatPipeline) has its own iteration loop and is not yet integrated with
  AgenticLoop." This FEP closes that gap.

## Prerequisite (landed)

PR #170 extracted the per-turn EVALUATE guards (content-repetition, plateau, spin, nudge) into
a shared `TurnEvaluationController` in `victor/agent/turn_policy.py` that **both** loops call.
This FEP builds on that controller as the shared decision seam.

## Implementation Status & Decision (2026-06-21)

Phases 1–2 have landed, and a key delivery decision was made that **supersedes the flag-gated
Phase 3/4 rollout originally drafted below**:

> **Decision: no feature flag, no legacy fallback.** The unified per-turn primitive is the
> *single canonical* streaming loop body, on by default. A `USE_UNIFIED_STREAMING_LOOP` flag
> with the old loop retained as a fallback would be exactly the branch-proliferation / dual-path
> tech debt the runtime guidance pushes against (cf. the W3 service-layer flag removal). We
> guard the refactor with a byte-stable characterization battery instead of a runtime A/B.

Shipped:

- **Phase 1 (EVALUATE consolidation)** — the streaming loop's per-turn decision (content-repetition,
  plateau, spin, nudge, search-novelty, fulfillment) routes through the shared
  `TurnEvaluationController` / `turn_policy` seam.
- **Phase 2 (`stream_turn()` extraction + assembly)** — the per-turn body of
  `StreamingChatExecutor.run()` was decomposed into named ACT/EVALUATE/emit/recovery/continuation
  helpers (PRs #176–#190) and then **assembled into a single `_stream_turn()` async-generator
  primitive** (PR #192); `run()` is now a thin while-driver over it. A deterministic, offline
  streaming **characterization battery** (`tests/integration/streaming/`, PRs #186/#188) pins the
  loop's observable behavior on the QA scenarios (S1/S2/M1/M2/C1/W1–W4/U1) and gated every step
  byte-stable. The flag-gated comparison scaffolding was removed once the no-flag decision was made
  (PR #193).

Remaining (optional): if deeper unification is desired, have the headless `AgenticLoop` drive the
same `_stream_turn()` primitive as its streaming ACT — again as the single implementation, with no
flag and no fallback.

## Proposed Change

Make `AgenticLoop` own the iteration for both modes:

```python
class AgenticLoop:
    async def run(self, query, ...) -> LoopResult:            # buffered (today)
    async def run_streaming(self, query, ...) -> AsyncIterator[StreamChunk]:  # NEW
```

Both drive the same phase sequence and the same `TurnEvaluationController`. They differ only in
the ACT primitive:

- buffered ACT → `_act()` → `ExecutionCoordinator.execute_turn_with_tools()` (returns a
  `TurnResult`).
- streaming ACT → a new `stream_turn()` primitive extracted from `StreamingChatExecutor.run()`
  that yields `StreamChunk`s **and** produces the same per-turn observation
  (`TurnObservation`) the EVALUATE phase needs.

`StreamingChatExecutor` is reduced to: (a) the `stream_turn()` per-turn primitive (provider SSE
+ tool execution + recovery for ONE turn) and (b) a thin governance/adapter shell.
`ChatStreamRuntime.get_executor()` is repointed to drive `AgenticLoop.run_streaming()`.

## Implementation Plan

Each phase is independently landable and revertible.

- **Phase 1 — finish EVALUATE consolidation (no structural risk).** Route the streaming loop's
  remaining inline guards (plateau, and later fulfillment) through `TurnEvaluationController`
  so the *entire* per-turn decision is shared, not just content-repetition. Unify the
  perception injection so both loops perceive via the same path. *No FEP gate — extends PR
  #170.*
- **Phase 2 — extract and assemble `stream_turn()`. ✅ Done.** Pull the per-turn streaming (one
  provider stream + tool execution + recovery, yielding chunks and signalling the per-turn outcome)
  out of `StreamingChatExecutor.run()`, first as named helpers and then as a single `_stream_turn()`
  primitive. Behavior-neutral: `run()` is a thin while-driver that calls it. Locked byte-stable by
  the streaming characterization battery.
- **Phase 3 — drive the single per-turn primitive (no flag, no fallback).** The unified
  `_stream_turn()` is the one canonical streaming loop body, on by default — there is no
  `USE_UNIFIED_STREAMING_LOOP` gate and no retained legacy loop. (Optional deeper unification: have
  `AgenticLoop` drive `_stream_turn()` as its streaming ACT so PERCEIVE/EVALUATE/DECIDE live once
  across both modes — still as the single implementation, no flag, no fallback.)
- **Phase 4 — n/a (folded into the no-flag approach).** There is no separate "flip the flag /
  delete the old loop" step: the old iteration loop was replaced in place by the `_stream_turn()`
  driver rather than kept behind a flag, so there is no dual path to cut over from.

## Benefits

- One place for loop control, perceive, evaluate, decide, convergence, and recovery.
- The streaming UI path gains the headless loop's PLAN/DECIDE + fulfillment/convergence.
- No more drift between CLI and UI behavior for the same task.

## Drawbacks and Alternatives

- **Risk.** The refactor touches the live UI streaming path; a regression breaks `victor ui`.
  Mitigated **not** by a feature flag + fallback, but by a deterministic, offline characterization
  battery that drives the real loop and pins every QA scenario byte-stable, plus live-provider
  verification, on each incremental change.
- **Alternative (rejected): leave two loops, share only guards.** PR #170 already did the
  cheap part; leaving the structural split keeps two iteration controls, the missing
  PLAN/DECIDE in streaming, and ongoing drift risk.
- **Alternative (rejected): make `StreamingChatExecutor` canonical.** It lacks the phase
  structure and is service-layer, not framework; CLAUDE.md designates `AgenticLoop` canonical.

## Unresolved Questions

- **Chunk granularity. ✅ Resolved in Phase 2.** Token/segment-granularity emission stayed inside
  the provider/tool helpers (`_stream_provider_turn`, `_execute_tools_turn`); `_stream_turn()` is one
  loop iteration that re-yields those chunks. Loop control crosses the generator boundary via a
  small mutable `_TurnOutcome` holder (`should_stop` + the inter-turn separator state) rather than a
  return value.
- **Recovery placement. ✅ Resolved in Phase 2.** Per-turn recovery (`_apply_turn_recovery`) and the
  empty-response ladder (`_emit_assistant_turn`) live inside the per-turn primitive; loop-level
  control (continue/stop) is signalled out via the helper holders. The cross-task cleanup hardened
  in PR #155 stays loop-level and was kept byte-stable by the characterization battery.
- **StateGraph mode interaction.** `USE_STATEGRAPH_AGENTIC_LOOP` swaps the inner *headless* executor
  and is independent of the streaming primitive. Only relevant if optional deeper unification routes
  the headless loop through `_stream_turn()`; can be deferred.

## Migration Path

The change is behavior-neutral and in-place (no flag, no parallel path), so there is no breaking
step for callers:

1. Phases 1–2 landed with no behavior change (EVALUATE consolidation + the `_stream_turn()`
   extraction and assembly); the streaming characterization battery is the regression gate.
2. The unified `_stream_turn()` primitive replaced `StreamingChatExecutor.run()`'s iteration body
   **in place** — `run()` became a thin driver over it. There is no dual path and nothing to opt
   into: the single canonical streaming loop is always active.
3. Each incremental change was held byte-stable by the QA battery (S1/S2/M1/M2/C1/W1–W4/U1) plus
   live-provider verification (same answers, no streaming tracebacks).
4. `StreamChunk` (the public streaming type) and the `Agent.stream()` async-iterator contract are
   unchanged throughout, so UI, API, and SDK consumers need no migration.

## Compatibility

`AgenticLoop.run()` and the buffered path are unchanged. The streaming refactor is internal to
`StreamingChatExecutor` and behavior-neutral, so existing callers keep the current streaming
behavior. `StreamChunk` (the public streaming type) and the `Agent.stream()` async-iterator contract
are unchanged throughout, so UI, API, and SDK consumers require no code changes.

## Acceptance Criteria

- The unified `_stream_turn()` primitive yields `StreamChunk`s byte-identical to the pre-refactor
  `StreamingChatExecutor.run()` on the QA battery (S1/S2/M1/M2/C1/W1–W4/U1) — same answers, same
  tool sequence, no streaming tracebacks. *(Met: the characterization battery passes byte-stable.)*
- The streaming per-turn decision shares one `TurnEvaluationController` / `turn_policy` seam.
- Single canonical streaming loop — no feature flag, no retained legacy path.

## References

- PR #170 — shared `TurnEvaluationController` (prerequisite).
- PRs #176–#190 — Phase 2 per-turn helper extractions (ACT / EVALUATE / emit / recovery /
  continuation), plus #184 (streaming fulfillment fix).
- PR #192 — Phase 2 assembly: the single `_stream_turn()` primitive + thin `run()` driver.
- PRs #186/#188 — streaming characterization battery (`tests/integration/streaming/`);
  PR #193 — removal of the flag-gated comparison scaffolding after the no-flag decision.
- CLAUDE.md — Agentic Loop (Phase 10), Agent Runtime Target State.
