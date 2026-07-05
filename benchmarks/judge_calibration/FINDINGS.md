# Judge-Calibration Findings — EVR-2 / ADR-011 first live measurements

**Date**: 2026-07-02/05 · **Corpus**: `default_calibration_corpus(variants=8)` — 48 tasks,
6 families · **Executor**: scripted (`alternating_scripted_executor(period=5)`, ~38 solved /
~10 completion-without-effect fakes) · **Gate**: Krippendorff α ≥ 0.7 (binary completion
verdicts vs programmatic verifier gold) · **Judges measured**: local Ollama (offline,
runs 1–5) and cloud DeepSeek (run 6, `--judge-delay` paced, integrity-verified)

## Run series

| # | Judge | View / code state | Overall α | Verdict | Diagnosis |
|---|-------|-------------------|-----------|---------|-----------|
| 0a | credulous (claim-trusting baseline) | — | **−0.188** | NOT TRUSTED | Reproduces the AgentProp-Bench result locally: trusting agent claims agrees with ground truth at worse than chance. |
| 0b | evidence (tool-activity baseline) | — | **1.000** | TRUSTED | On scripted trajectories, tool-activity presence is a perfect completion signal (will not hold for real agents that work and still fail). |
| 0c | rubric-heuristic (ADR-009 no-LLM fallback) | names-only view | **0.852** | TRUSTED overall | **At chance (α=0.000) on code-fix** — the family where solved/unsolved workspaces differ only semantically. Overall-α gating would wrongly bless it; per-family gating catches it. |
| 1 | rubric-llm, qwen2.5-coder:14b | names-only view | **0.450** | NOT TRUSTED | 8/10 errors were missed completions, 6 on code-fix: the judge could not assess correctness because the view showed file *names* without *contents*. |
| 2 | rubric-llm, qwen2.5-coder:14b | + workspace file contents | **0.457** | NOT TRUSTED | code-fix fixed (−0.406 → 0.615) — confirming run 1's diagnosis — but docs-link regressed (5 missed completions: judge believed the "right" fix was creating the missing doc, not repointing the link). Failure moved, α unchanged. |
| 3 | rubric-llm, qwen2.5-coder:32b | same view as run 2 | **−0.237** | NOT TRUSTED | Zero false completions, 30 missed: mass refusal. Probing the raw output exposed the scripted executor's claim echoing a mid-word-truncated prompt — the stronger model *correctly* penalized the garbled message the weaker model glossed over. Label poisoning, not judge failure. |
| 4 | rubric-llm, qwen2.5-coder:32b | clean claim message | **−0.412** | NOT TRUSTED | Still mass refusal (34 missed, 0 false). Raw output on a perfect task: three dimensions 1.0@1.0, then `recovery: score=0.0 confidence=0.3` — the un-engaged axis scored just above the DimensionAwareFilter engagement floor (0.25), gating completion. **Framework bug (ADR-009), fixed in `82e20be9`.** |
| 5 | rubric-llm, qwen2.5-coder:32b | + engagement-convention fix | **0.173** | NOT TRUSTED | Large improvement (−0.412 → 0.173): file-create and qa now α=1.000, zero false completions. Remaining 19 misses concentrate on code-fix/docs-link/dead-code. Probing shows the convention fix took (recovery now 0.0@0.0) and the judge *sees* the correct workspace (`tool_grounding: 0.8 — "workspace state shows the correct implementation"`) — but penalizes correctness/completeness because the terse scripted claim ("Done — I completed the requested task.") **doesn't narrate the fix**. Near-identical variants flip between perfect and penalized grades at temperature 0 (variant 0 graded 1.0/1.0/1.0; variant 1 graded 0.5/0.8/0.6), so the residual is part scripted-transcript artifact, part judge instability on narration-free claims. |
| 6 | rubric-llm, **deepseek-chat (cloud)** | run-5 code state; `--judge-delay 5`; first clean cloud run (integrity: calls=48 retries=0 failures=0) | **0.279** | NOT TRUSTED | **New best**, and the first cross-provider data point. Per-family fingerprint nearly identical to run 5: file-create 1.000, qa 1.000, refactor 0.372, code-fix −0.190, docs −0.500 — and again **zero false completions**; all 15 errors are missed completions on solved docs-link (6/7), code-fix (5/7), and dead-code (4/7) tasks. Two unrelated judges (local qwen-32b, cloud DeepSeek) producing the same family-level failure signature confirms the bottleneck is the **narration-free scripted transcripts**, not judge capability — the judges refuse to certify work nobody described, which is defensible behavior against a corpus artifact. Preceding runs on this machine also validated the new guardrails live: a fully rate-limited GLM-5.2 attempt was correctly VOIDed (α would have equaled the heuristic's to 3 decimals), and a DeepSeek attempt exposed the per-call-event-loop bug (PR #394). |

## What the series established

1. **The gate works.** Every NOT TRUSTED verdict above was correct — each traced to a real,
   specific defect (in the judge, the view, or the harness itself), not measurement noise.
2. **Per-family α is mandatory.** Run 0c passes the overall gate while blind on code-fix;
   runs 1–2 show failures migrating between families as the view changes. Gate per family,
   with n ≥ 30 per family (`--variants 16`+) before treating per-family α as evidence.
3. **Stronger judges surface weaker harnesses.** Runs 3–4 both looked like "the 32b model is
   worse." Both times the model was right and the harness was wrong (truncated claim echo;
   engagement-floor mismatch). Diagnose mass-refusal patterns (0 false completions, many
   missed) by probing raw judge output before blaming the model.
4. **Fixes found in framework code, not just calibration code** (commit `82e20be9`):
   - Grading prompt now states the numeric not-applicable convention
     (`score=0.0 confidence=0.0`) instead of "use a LOW confidence" — judges' idea of low
     (0.3) sat above the filter's engagement floor (0.25).
   - Unparseable judge rows default to confidence 0.2 (below the floor) so an ungradable
     line cannot gate completion.
   - Scripted-executor claims no longer echo truncated prompts.

## Reproduction

```bash
python benchmarks/judge_calibration/run_offline_calibration.py \
    --variants 8 \
    --llm-judge-provider ollama --llm-judge-model qwen2.5-coder:32b \
    --llm-judge-base-url http://<ollama-host>:11434   # e.g. WSL → Windows gateway
```

Reports (per-family α, gate decision, every sample) land in
`benchmarks/judge_calibration/reports/*.json`.

## Verdict and open items

**`completion_strategy=rubric` stays opt-in.** Best measured configuration
(deepseek-chat, contents view, aligned convention, run 6): **α=0.279** overall, well below
the 0.7 gate — but with a consistently safe error profile across both measured judges (zero
false completions in runs 5 and 6; every error burns retries rather than shipping unverified
completions).

Next measurements, in order of expected leverage:

1. **Real agent trajectories** (`make_agent_executor` + `VictorAgentAdapter.from_profile`) —
   now decisively the top lever: runs 5 and 6 show two unrelated judges failing on the SAME
   families with zero false completions, i.e. the corpus's narration-free scripted claims —
   not judge capability — cap α. Both judges' residual misses are on claims that real agents
   don't produce; this is now the biggest known artifact. (Scripted transcripts also make
   the evidence baseline artificially perfect.)
2. **Prompt guidance that workspace state is authoritative** — the judge already extracts
   the correct evidence into `tool_grounding` but doesn't let it carry `correctness`;
   one sentence in the grading prompt may transfer it. Any change re-measures here first.
3. **Order-swap / repeated-sample ensemble (ADR-011)** — the variant 0-vs-1 grade flips at
   temperature 0 are exactly the instability ensembling averages out.
4. Raise `--variants` to 16+ so per-family α is gating-grade (n ≥ 30/family).
