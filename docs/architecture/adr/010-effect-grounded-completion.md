# ADR-010: Effect-Grounded Completion

## Metadata

- **Status**: Proposed
- **Date**: 2026-06-21
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 009 (rubric completion), 011 (judge reliability), 012 (regression-gated harness)
- **Related**: [FEP-0008](../../../feps/fep-0008-evaluation-centric-completion.md)

## Context

Victor's completion decision is asserted from **text** — the model says it is done. HarnessFix
(`2606.06324`) names the dominant failure "completion-without-effect": an agent calls a completion
path without a verifiable workspace state delta. Harness-Bench (`2605.27922`) frames the same as an
"execution-alignment failure" — plausible reasoning decoupled from tool feedback / workspace state —
and makes an **integrity gate** (no credit without a verifiable check) a benchmark requirement.
Victor has `victor/tools/verification/` (claim verifier, cross-reference, false-positive detector)
and `FulfillmentDetector`, but they are a *separate* tool surface never consulted by the loop's
completion decision.

## Decision

Make COMPLETE conditional on a **verifiable effect**. Before `_evaluate` returns COMPLETE, require
one of: (a) a workspace artifact/state delta observed in the turn's `tool_results`, or (b) for
no-mutation/Q&A tasks, a grounded-claim check via `victor/tools/verification/`. A COMPLETE lacking a
verifiable effect is downgraded to RETRY with a `completion-without-effect` reason. The effect class
(file delta vs grounded claim) is chosen from the existing task-type / `is_qa_response` signals.

## Rationale

- Attacks the highest-frequency, highest-severity completion failure named in the 2026 harness
  literature, and connects two subsystems Victor already has (verification + completion) that are
  currently disconnected.
- Composes with ADR-009: the rubric's "Tool-Grounding"/"Completeness" dimensions and this gate are
  complementary — the gate is a hard precondition, the rubric is graded quality.
- **Pros**: stops confident-but-empty completions; reuses existing verifiers. **Cons**: risk of
  *blocking* legitimate completion if the effect detector is too strict (mitigated: Q&A uses a
  grounded-claim check, not a file delta; gate is downgrade-to-RETRY, not FAIL).

## Consequences

- **Positive**: completion means something verifiable happened; verifiers become loop-integrated, not
  side-by-side; measurable via the trajectory harness.
- **Negative**: added per-completion verification cost; possible false-blocks on tasks with subtle
  effects (tune the effect detector; keep the Q&A path lenient).
- **Neutral**: tools, providers, and the StateGraph engine are unaffected.

## Implementation

1. Define an `effect-present?` check over `tool_results` + a `verification/` grounded-claim path.
2. Insert as a precondition in `_evaluate` (applies to all completion strategies), reason
   `completion-without-effect` on downgrade.
3. Battery-gate; add trajectory-harness cases that assert no completion-without-effect.
