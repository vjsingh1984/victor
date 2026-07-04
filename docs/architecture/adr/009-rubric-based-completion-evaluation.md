# ADR-009: Rubric-Based Completion Evaluation

## Metadata

- **Status**: Accepted (2026-07-02 — shipped opt-in via `completion_strategy=rubric` (#203/#216/#217/#223); default remains `enhanced` until the ADR-011 reliability gate passes; was Proposed)
- **Date**: 2026-06-21
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: 010 (effect-grounded completion), 011 (judge reliability), 012 (regression-gated harness)
- **Related**: [FEP-0008](../../../feps/fep-0008-evaluation-centric-completion.md), [Vision](../vision-evaluation-centric-runtime.md)

## Context

`AgenticLoop._evaluate` decides completion via `EnhancedCompletionEvaluator`, an algorithmic
multi-signal fusion that returns a single scalar. FEP-0007 live verification showed it **under-scores
finished answers** (~0.26–0.41, "insufficient progress"), so the loop burns low-confidence retries
and the model restates its answer to the iteration cap. We patched it twice (HIGH-confidence marker
signal; `_is_terminal_answer` override). A single scalar cannot express "correct answer, poor
recovery," and one strong axis can mask a failed one. AdaRubric (`2603.21362`) shows task-adaptive
rubric scoring reaches r=0.79 vs human (+0.15 over static) and +4.9pp SWE-bench, and that
*adaptivity matters more than judge-model strength*.

## Decision

Introduce `RubricCompletionEvaluator`: generate N≈4–6 orthogonal, task-conditioned dimensions
(cached per task family), score each with a confidence weight, and gate COMPLETE on a
**DimensionAwareFilter** (every dimension must clear its threshold). Select via
`settings.evaluation.completion_strategy` (`rubric` | `enhanced` | `legacy`). `EnhancedCompletionEvaluator`
remains the fallback and the baseline the rubric path must match-or-beat before becoming default.

## Rationale

- Directly fixes the under-scoring/restatement defect with an *interpretable, per-dimension* signal.
- The DimensionAwareFilter both prevents premature stop (one strong axis) and prevents the inverse
  (a finished answer under-scored overall) — the exact two failure modes seen.
- Per-task-family rubric caching makes it cheap (AdaRubric reports >95% cost reduction).
- A strategy setting (not a feature flag) matches Victor's prompt-optimization configuration pattern
  and avoids dual-path tech debt.
- **Pros**: explainable, calibrated, low infra (prompt + cache, no training). **Cons**: judge cost
  per turn (mitigated by cache); requires Phase-D judge validation before trust.

## Consequences

- **Positive**: completion is explainable and dimension-gated; restatement reduced; reusable rubric
  signal feeds trajectory eval (FEP-0008 Phase E) and credit assignment.
- **Negative**: a new LLM call path (gated/cached); behavior change to the shared loop (buffered +
  streaming) — must be battery-gated.
- **Neutral**: the StateGraph engine, tool layer, and provider layer are unaffected.

## Implementation

1. `RubricCompletionEvaluator` + per-task-family cache; offline unit tests with scripted dimensions.
2. Wire into `_evaluate` behind `completion_strategy="rubric"` (default stays `enhanced`).
3. Gate the default flip on: parity battery 14/14, characterization byte-stable-or-justified,
   judge agreement α ≥ threshold (ADR-011), and reduced restatement on the live multi-step task.
