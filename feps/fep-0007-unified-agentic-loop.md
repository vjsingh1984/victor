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
- **Phase 2 — extract `stream_turn()`.** Pull the per-turn streaming (one provider stream +
  tool execution + recovery, yielding chunks and returning a `TurnObservation`) out of
  `StreamingChatExecutor.run()` as a method. Behavior-neutral: `run()` calls it. Locked by the
  existing streaming tests.
- **Phase 3 — `AgenticLoop.run_streaming()` behind a flag.** New method drives
  PERCEIVE → ACT(`stream_turn`) → EVALUATE(controller) → DECIDE, yielding chunks. Gated by
  `USE_UNIFIED_STREAMING_LOOP` (default OFF). `ChatStreamRuntime` chooses old `executor.run()`
  vs `AgenticLoop.run_streaming()` on the flag, with the old path as fallback.
- **Phase 4 — cut over and remove the old loop.** After the QA battery confirms parity
  (streaming UI + `Agent.stream()`), flip the flag default ON, then delete
  `StreamingChatExecutor.run()`'s iteration loop, leaving only `stream_turn()` + the adapter.

## Benefits

- One place for loop control, perceive, evaluate, decide, convergence, and recovery.
- The streaming UI path gains the headless loop's PLAN/DECIDE + fulfillment/convergence.
- No more drift between CLI and UI behavior for the same task.

## Drawbacks and Alternatives

- **Risk.** Phases 3–4 touch the live UI streaming path; a regression breaks `victor ui`.
  Mitigated by the feature flag + fallback + QA-battery parity gate before cutover.
- **Alternative (rejected): leave two loops, share only guards.** PR #170 already did the
  cheap part; leaving the structural split keeps two iteration controls, the missing
  PLAN/DECIDE in streaming, and ongoing drift risk.
- **Alternative (rejected): make `StreamingChatExecutor` canonical.** It lacks the phase
  structure and is service-layer, not framework; CLAUDE.md designates `AgenticLoop` canonical.

## Unresolved Questions

- **Chunk granularity.** `AgenticLoop.run_streaming()` must yield at token/segment granularity
  (matching today's `StreamingChatExecutor`), not at loop-iteration granularity like the
  existing `AgenticLoop.stream()` (which yields `LoopIteration`). The exact boundary between
  the `stream_turn()` primitive and the loop's per-iteration yielding needs to be settled in
  Phase 2 against the current chunk-emission points.
- **Recovery placement.** The streaming loop's recovery/continuation handling (empty-response
  fallback, stall-timeout, `on_chat_end` cancel drain) is entangled with its iteration today.
  Phase 2 must decide which parts belong to the per-turn `stream_turn()` primitive and which
  remain loop-level, without regressing the cross-task cleanup hardened in PR #155.
- **StateGraph mode interaction.** `USE_STATEGRAPH_AGENTIC_LOOP` swaps the inner executor;
  whether `run_streaming()` supports the StateGraph executor in Phase 3 or only the in-class
  while-loop is open and can be deferred.

## Migration Path

The change is additive and flag-gated, so there is no breaking step for callers:

1. Phases 1–2 land with no behavior change (consolidation + a behavior-neutral `stream_turn()`
   extraction); existing streaming tests are the regression gate.
2. Phase 3 introduces `AgenticLoop.run_streaming()` behind `USE_UNIFIED_STREAMING_LOOP`
   (default OFF). `ChatStreamRuntime.get_executor()` selects the old `executor.run()` by
   default; opting in routes through the unified loop. The old path remains the fallback.
3. The QA battery (S1/S2/M1/M2/C1/W1–W4/U1) plus `Agent.stream()` integration tests must show
   parity (same answers, no streaming tracebacks) before the flag default flips to ON.
4. Phase 4 removes `StreamingChatExecutor.run()`'s iteration loop only after the flag has
   defaulted ON across a release with no regressions reported; `StreamChunk` and the public
   `Agent.stream()` signature never change, so downstream consumers need no migration.

## Compatibility

`AgenticLoop.run()` and the buffered path are unchanged. `run_streaming()` is additive and the
cutover is flag-gated, so existing callers keep the current streaming path until the flag flips.
`StreamChunk` (the public streaming type) and the `Agent.stream()` async-iterator contract are
unchanged throughout, so UI, API, and SDK consumers require no code changes at any phase.

## Acceptance Criteria

- `AgenticLoop.run_streaming()` yields `StreamChunk`s equivalent to `StreamingChatExecutor.run()`
  on the QA battery (S1/S2/M1/M2/C1/W1–W4/U1) — same answers, no streaming tracebacks.
- Both modes share one `TurnEvaluationController` and one perceive path.
- Behind-flag rollout with the old path intact until parity is proven; old loop removed only in
  Phase 4.

## References

- PR #170 — shared `TurnEvaluationController` (prerequisite).
- CLAUDE.md — Agentic Loop (Phase 10), Agent Runtime Target State.
