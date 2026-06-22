---
fep: 0008
title: "Evaluation-Centric Completion (calibrated, multi-dimensional, effect-grounded, judge-validated)"
type: Standards Track
status: Draft
created: 2026-06-21
modified: 2026-06-21
authors:
  - name: Vijaykumar Singh
    email: singhvjd@gmail.com
    github: vjsingh1984
reviewers: []
discussion: https://github.com/vjsingh1984/victor/discussions/0008
---

# FEP-0008: Evaluation-Centric Completion

## Summary

Replace Victor's heuristic completion/EVALUATE decision with an **evaluation-centric** one:
task-adaptive **rubric** dimensions scored with confidence and gated by a **DimensionAwareFilter**
(AdaRubric `2603.21362`); a **verifiable-effect** precondition before any COMPLETE (HarnessFix
`2606.06324` "completion-without-effect"); an **online per-turn auditor** that flags the decisive
error before it cascades (AgentForesight `2605.08715`); and a **judge-reliability harness** that
gates whether any LLM-as-judge is trusted at all (AgentProp-Bench `2604.16706`). A
**trajectory-level evaluation harness** (TRBench `2604.08178`) extends `victor/evaluation/` so these
changes are measured, not assumed.

This supersedes the heuristic stack the FEP-0007 cutover left in `AgenticLoop._evaluate`
(`EnhancedCompletionEvaluator` scalar + the `forced_task_completion` marker signal + the
`_is_terminal_answer` override) with one calibrated, explainable decision — keeping those as the
behavioral baseline the new path must match-or-beat on the acceptance batteries.

## Motivation

The FEP-0007 live verification exposed the core defect: `EnhancedCompletionEvaluator` scores a
*complete* answer at ~0.26–0.41 ("insufficient progress"), so the loop burns low-confidence retries
and the model restates its answer until the iteration cap. We patched it twice (a HIGH-confidence
active-marker signal; a `_is_terminal_answer` heuristic). Both work, but they are heuristics on a
heuristic, and **nothing measures whether completion judgments are actually correct.**

Specific gaps (grounded against the code):

- **Scalar, static completion.** A single fused score can't express "the answer is correct but the
  recovery was poor"; one strong axis masks a failed one.
- **No effect grounding.** Completion is asserted from text; an agent can `complete` without a
  verifiable workspace delta (HarnessFix's "completion-without-effect"; Harness-Bench's
  "execution-alignment failure").
- **Post-hoc failure handling.** Spin/repetition/plateau are detected *after* the loop is already
  stuck; there is no per-turn "is this prefix heading for failure?" signal.
- **Unvalidated judges.** Substring/keyword completion checks agree with humans at κ≈0.05 (chance,
  AgentProp-Bench). Victor has *no* judge-reliability measurement anywhere.
- **No trajectory-level eval.** `AgenticExecutionTrace` exists offline but the loop's online control
  uses heuristic scores; there is no harness scoring *full trajectories* on planning / tool-grounding
  / recovery / refusal with confidence intervals.

## Design principles

- **One decision point.** All signals resolve inside `AgenticLoop._evaluate` → `EvaluationResult`.
  No new control-flow path; the streaming loop inherits this via the shared `_evaluate` (FEP-0007).
- **Cache to keep it cheap.** Rubrics are generated once per *task family* and cached (>95% cost cut
  per AdaRubric); the online auditor is a single small-model call on the prefix.
- **Gated by the acceptance oracle.** Every change ships behind the parity + characterization
  batteries (FEP-0007) plus the new judge-agreement metric; see Acceptance Criteria.
- **Fail safe toward not-stopping-early.** The DimensionAwareFilter + effect gate make premature
  completion *harder*, not easier; the auditor only raises alarms, never force-completes.

## Proposed change

### Phase A — Rubric-based completion evaluator (AdaRubric)

Introduce `RubricCompletionEvaluator` in `victor/framework/` implementing the existing completion
seam consumed by `_evaluate`:

1. **Generate** a rubric `R(task)` = N≈4–6 orthogonal dimensions (e.g. Correctness, Tool-Grounding,
   Error-Recovery, Completeness), each with a weight and calibrated 5-point criteria. Validate
   (pairwise cosine-distance > 0.3; weights sum to 1). **Cache per task family** (keyed by the
   existing `TaskAnalyzer` task type/signature).
2. **Score** the turn/trajectory per dimension with a confidence `c∈[0,1]` (low when the turn doesn't
   engage that dimension). Aggregate via confidence-weighted mean (default), geometric mean
   (balanced), or min (safety).
3. **DimensionAwareFilter** — require `s̄ⱼ ≥ θⱼ` for *every* dimension before COMPLETE; otherwise
   CONTINUE/RETRY with the failing dimension as the reason. This is the structural fix for "one
   strong signal triggers premature stop" *and* "a finished answer is under-scored overall."

`EnhancedCompletionEvaluator` remains as a fallback and as the baseline the rubric path must
match-or-beat on the batteries; selection via `settings.evaluation.completion_strategy`
(`rubric` | `enhanced` | `legacy`) — a strategy setting, not a feature flag, consistent with the
prompt-optimization settings pattern.

### Phase B — Effect-grounded completion gate (HarnessFix / Harness-Bench)

Before returning COMPLETE, require a **verifiable effect**: a workspace artifact/state delta or a
passed check, sourced from `victor/tools/verification/` (claim verifier / cross-reference) and the
turn's `tool_results`. A COMPLETE with no observable effect is downgraded to RETRY with a
"completion-without-effect" reason. For pure Q&A tasks (no expected mutation), the "effect" is a
grounded-claim check, not a file delta. Wired as a precondition inside `_evaluate`, reusing the
existing `is_qa_response` / task-type signals to choose the effect class.

### Phase C — Online per-turn auditor (AgentForesight)

Add `TurnAuditor`: a cheap, prefix-only `f(trajectory[:k]) → {CONTINUE | ALARM(step, reason)}` run
each turn inside the loop (and the streaming `_stream_turn` ACT/EVALUATE band). On ALARM, the loop
opens an intervention window (replan / nudge / escalate via the existing `NudgePolicy` /
`RecoveryService`) instead of waiting for spin detection to trip post-hoc. Backed by the edge model
(`use_edge_model`); falls back to the existing spin/repetition guards when unavailable.

### Phase D — Judge-reliability harness (AgentProp-Bench)

Before any LLM-as-judge (rubric scoring, trajectory grading) is *trusted in production*, it must pass
a reliability gate: ship a small human-labeled trajectory set under `tests/` /
`victor/evaluation/judge_calibration/`, compute **Krippendorff α / Cohen κ** vs the labels, and
require α ≥ 0.7 (configurable) to enable the judge; otherwise fall back to algorithmic scoring. Use a
small **order-swap ensemble** (≥2 reorderings) to defuse position/verbosity bias. Track **rejection**
(catch bad input) and **recovery** (fix after accepting) as *separate* metrics — they are
statistically independent.

### Phase E — Trajectory-level evaluation harness (TRBench)

Extend `victor/evaluation/` with a `trajectory_eval/` harness that scores *full trajectories*
(reusing `AgenticExecutionTrace`) on Planning, Tool-Grounding, Recovery, and Refusal, with
length-balanced cases and **confidence intervals** on reported scores (`2605.10448`). This is the
measurement substrate for Phases A–D and feeds the acceptance oracle (P5 / ADR-012).

## Implementation plan

Each phase is independently landable, behavior-gated, and reversible.

- **Phase A1** — `RubricCompletionEvaluator` + per-task-family rubric cache; offline unit tests with
  scripted dimensions. *No wiring.*
- **Phase A2** — wire into `_evaluate` behind `completion_strategy="rubric"`; default stays
  `enhanced`. Run batteries + judge-agreement on the labeled set; flip default only if match-or-beat.
- **Phase B** — effect-grounded gate (independent of A; benefits all strategies).
- **Phase C** — `TurnAuditor` (edge-model), additive; nudge/replan on ALARM.
- **Phase D** — judge-reliability harness + α/κ gate (prerequisite to trusting A/E judges).
- **Phase E** — trajectory eval harness (can land first as pure measurement; recommended early).

Recommended order for *measurement-first* discipline: **E → D → A → B → C.**

## Acceptance criteria

- **Parity battery 14/14 and characterization battery byte-stable-or-justified** (FEP-0007 gate) for
  every phase that touches `_evaluate` / `_stream_turn`.
- **Judge agreement reported** (Krippendorff α / κ) on the human-labeled set; the rubric judge is
  enabled only at α ≥ threshold.
- **Restatement metric down** vs the FEP-0007 baseline on the live multi-step task (the observable
  the cutover fix targeted) — measured, not assumed.
- **No premature completion regression**: the multi-step parity scenarios keep their full tool
  sequences; the DimensionAwareFilter + effect gate must not stop before the task's effect exists.
- Eval results reported with confidence intervals at *(model, completion_strategy)* granularity.

## Drawbacks / alternatives

- **Cost/latency of a judge per turn.** Mitigated by per-task-family rubric caching, a small edge
  auditor, and the α-gate (don't run an untrusted judge). Alternative — keep the algorithmic
  evaluator — is the status quo whose under-scoring this FEP exists to fix.
- **Premature completion risk** from a new gate. Mitigated structurally: the filter + effect gate
  make stopping *harder*; the auditor never force-completes.
- **Behavior change to the shared loop** (buffered + streaming). Mitigated by the strategy setting
  (default unchanged until proven) and the acceptance batteries.

## Related

- ADR-009 (rubric-based completion), ADR-010 (effect-grounded completion), ADR-011 (LLM-judge
  reliability gating), ADR-012 (regression-gated harness acceptance).
- [Vision: Evaluation-Centric Agent Runtime](../docs/architecture/vision-evaluation-centric-runtime.md).
- FEP-0007 (unified agentic loop) — provides the single `_evaluate` seam this builds on.

## References

arXiv: `2603.21362` (AdaRubric), `2605.08715` (AgentForesight), `2604.16706` (AgentProp-Bench),
`2606.06324` (HarnessFix), `2605.27922` (Harness-Bench), `2604.08178` (TRBench), `2605.10448`
(benchmark confidence bounds), `2603.18683` (HISR), `2605.06200` (A²TGPO), `2604.00445`
(truth-aligned uncertainty).
