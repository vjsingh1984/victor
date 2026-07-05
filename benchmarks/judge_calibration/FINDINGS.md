# Judge-Calibration Findings — EVR-2 / ADR-011 first live measurements

**Date**: 2026-07-02/05 (9 runs) · **Corpus**: `default_calibration_corpus(variants=8)` — 48 tasks,
6 families · **Executor**: scripted (`alternating_scripted_executor(period=5)`, ~38 solved /
~10 completion-without-effect fakes) · **Gate**: Krippendorff α ≥ 0.7 (binary completion
verdicts vs programmatic verifier gold) · **Judges measured**: local Ollama (offline,
runs 1–5, 7) and cloud DeepSeek (run 6); all cloud/paced runs integrity-verified

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
| 7 | rubric-llm, **gemma4:31b (LOCAL, Ollama on Apple Silicon)** | run-5 code state; default 2 s pacing (~84 s/call inference); integrity: calls=48 retries=0 failures=0 | **0.929** | **TRUSTED — first gate pass** | 47/48 verdicts correct. Per-family: code-fix **1.000** (the family that broke every prior judge), docs 1.000, file-create 1.000, qa 1.000, refactor 0.823 — clears the gate overall AND per-family. The single error is the series' **first false completion** (refactor-rename-02: unsolved rename judged complete) — a changed error polarity worth watching at larger n. This result **overturns run 5/6's corpus-ceiling diagnosis**: gemma4 passed on the same narration-free scripted claims that capped qwen-32b and DeepSeek, proving the workspace-state evidence in the view was sufficient all along and the earlier judges' refusals were model disposition, not a corpus artifact. The gate-passing judge is fully local — no API cost, no rate limits, no data egress. |
| 8 | rubric-llm, qwen3.5:27b-q4 (local) | run-7 code state; integrity: calls=48 retries=0 failures=0 — but see diagnosis | **−0.092** | NOT TRUSTED (should be VOID) | **Third integrity blindspot found**: every verdict was 1.0 — 10 false completions, 0 misses — matching rubric-heuristic's α sample-for-sample, the signature of a judge contributing zero signal. Transport succeeded, but no `score=` line ever parsed: every dimension fell back to 0.5@≤0.2 (below the engagement floor), so the DimensionAwareFilter defaulted every task to COMPLETE. Suspected cause: qwen3.5 is a thinking-family model that exhausted the 512-token grading budget on reasoning preamble (~72 s/call supports this). Fixed: `JudgeCallStats.ungradable` now counts all-fallback results and VOIDs the run; `--judge-max-tokens` (default raised to 1024) gives thinking judges headroom. Retest qwen3.5 with `--judge-max-tokens 2048` to distinguish format-mismatch from truncation. |
| 9 | rubric-llm, **llama3.3:70b (local, Ollama on the Windows-host GPU)** | run-8 code state (`--judge-max-tokens 1024`, ungradable guard active); integrity: calls=48 retries=0 failures=0 ungradable=0 | **1.000** | **TRUSTED — perfect score** | 48/48 verdicts correct, zero errors of either polarity, every family at α=1.000. Second gate-passer (of six judge models measured), confirming gate-passing judges are not rare among strong models — and the first ceiling hit: a perfect score at n=48 means this corpus can no longer discriminate among top judges. Ranking so far: llama3.3:70b 1.000 > gemma4:31b 0.929 > deepseek-chat 0.279 > qwen2.5-coder:32b 0.173 ≫ qwen3.5:27b (ungraded). Run also validated the run-8 ungradable guard end-to-end in a live measurement (`ungradable=0` reported). |

## What the series established

1. **The gate works.** Every NOT TRUSTED verdict above was correct — each traced to a real,
   specific defect (in the judge, the view, or the harness itself), not measurement noise.
   And it passes when a judge deserves it (run 7).
2. **Per-family α is mandatory.** Run 0c passes the overall gate while blind on code-fix;
   runs 1–2 show failures migrating between families as the view changes. Gate per family,
   with n ≥ 30 per family (`--variants 16`+) before treating per-family α as evidence.
3. **Stronger judges surface weaker harnesses.** Runs 3–4 both looked like "the 32b model is
   worse." Both times the model was right and the harness was wrong (truncated claim echo;
   engagement-floor mismatch). Diagnose mass-refusal patterns (0 false completions, many
   missed) by probing raw judge output before blaming the model.
4. **Judge choice dominates.** Runs 5–7 ran identical code, views, and corpus; α spanned
   0.173 → 0.279 → 0.929 purely by model. A shared failure pattern across two judges
   (runs 5–6) looked like a corpus ceiling until a third judge broke through it — beware
   concluding "artifact" from N=2 judges.
5. **Fixes found in framework code, not just calibration code** (commit `82e20be9`):
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

**Two gate-passers: llama3.3:70b at α=1.000 (run 9, perfect) and gemma4:31b at α=0.929
(run 7)** — both local Ollama. `completion_strategy=rubric` stays opt-in pending the
confirmation steps below, but graduation now has two candidate judge configurations:
llama3.3:70b (most accurate, ~42 GB) and gemma4:31b (efficient, ~19 GB, one false
completion at n=48). Trust is **judge-specific**: this evidence graduates these models as
rubric judges, not LLM-judging in the abstract (identical code scored 0.173–0.279 with
other models, and one thinking model produced zero usable grades — see points 4 and the
run-8 guard).

Run 9's perfect score also marks the corpus's **ceiling**: at `variants=8` it can no
longer discriminate among top judges, which makes the confirmation steps below the only
way to separate the two candidates.

Graduation checklist for `completion_strategy=rubric` + a pinned candidate judge (TD-17
evidence):

1. **Confirmation at gating-grade n** — re-run `--variants 16`+ (n ≥ 16/family; ~2–4 h at
   local inference speed) so per-family α and the new false-completion polarity (run 7's
   single error blessed unsolved work — the series' first) are measured, not anecdotal.
2. **Real agent trajectories** (`make_agent_executor` + `VictorAgentAdapter.from_profile`) —
   the production distribution: real agents narrate, fail partially, and recover;
   scripted transcripts do none of that (and make the evidence baseline artificially
   perfect). Gate must hold there too.
3. **Pin the judge identity in the flag criteria** — default-on only with a calibrated
   judge model; falling back to the heuristic (α=−0.092) or an uncalibrated model must
   revert to `enhanced`, per the ADR-011 fallback contract.

Deprioritized after run 7 (were top levers when the corpus looked like the ceiling):
prompt evidence-transfer guidance and the order-swap ensemble — both remain available if
the confirmation runs regress, and the ensemble may still help judges below the gate.
