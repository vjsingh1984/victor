# Flag Graduation Policy (TD-17)

**Status**: Proposed (criteria below are proposals; the owner ratifies per-flag) ·
**Date**: 2026-07-05 · **Tracks**: TD-17 in the
[tech-debt register](../tech-stack.md#technical-debt-register)

## Problem

Victor's quality/safety loop is almost entirely opt-in: `USE_POLICY_ENGINE`,
`sandbox_enabled`, `USE_SMART_ROUTING`, `completion_strategy=rubric`, and L1
reference-aware pruning all default OFF, with no written criteria for when a flag
graduates to default-on — or gets deleted. Opt-in flags that never graduate are dead
weight; flags that graduate without evidence are risk.

## The policy

Every graduation-track flag must carry four things. A flag that cannot state them is not
a graduation candidate and should be scheduled for removal.

1. **A measurable claim** — what the flag improves, stated falsifiably.
2. **A gate** — the metric, threshold, and measurement procedure that decides
   graduation. The template is ADR-011's judge-reliability gate: an offline, repeatable
   measurement against trusted ground truth, with the evidence artifact (report JSON)
   linked from the graduation PR.
3. **A fallback contract** — what the system does when the flag's premise fails at
   runtime (provider down, judge uncalibrated, platform unsupported). Fail toward the
   pre-flag behavior, never toward silent degradation.
4. **A kill criterion** — the condition under which the flag is deleted instead of
   graduated (e.g. gate unmet after N attempts, superseded design).

Graduation PRs flip the default, link the evidence, and keep the flag for one release as
an opt-out before removal of the old path.

## Per-flag status and proposed gates

| Flag | Claim | Proposed gate | Fallback contract | Status |
|------|-------|---------------|-------------------|--------|
| `completion_strategy=rubric` | Rubric+LLM completion verdicts agree with ground truth better than `EnhancedCompletionEvaluator` | ADR-011: Krippendorff α ≥ 0.7 overall **and per family** vs programmatic verifier gold, n ≥ 16/family, integrity-clean, **judge identity pinned**; must hold on scripted AND real-agent trajectories | Judge unavailable/uncalibrated → revert to `enhanced`; heuristic fallback (measured α=−0.092) must never gate alone | **Candidate found**: gemma4:31b (local Ollama) passed at α=0.929, run 7 of [FINDINGS](../../benchmarks/judge_calibration/FINDINGS.md); pending variants-16 confirmation + real-trajectory validation |
| `USE_POLICY_ENGINE` | ALLOW/DENY/ASK verdicts enforce governance without blocking legitimate tool use | Zero false-DENY on a recorded corpus of accepted-tool-call traces; 100% DENY on the builtin policy violation suite; latency overhead < 5 ms/call | Engine error → fail per `governance.enabled` posture (deny-closed when governance on) | No gate corpus yet — build from HTIR traces (ADR-012 machinery) |
| `sandbox_enabled` | Subprocess/code tools run isolated with no capability loss for allowed operations | Full tool test suite green under bwrap (Linux CI) and seatbelt (macOS CI); documented escape-hatch list reviewed | Sandbox init failure → currently fail-open by design; graduation requires making fail-open an explicit, logged decision | MVP hardening gaps (seccomp/egress) noted in code; not gateable until CI runs the suite sandboxed |
| `USE_SMART_ROUTING` | Cost/latency-aware routing cuts spend without quality loss | C0 cost-trace shows ≥ 20% cost or latency reduction on the benchmark suite with no drop in task success rate (same harness, A/B) | Router error → static profile routing | No measurement yet; benchmark harness exists (`victor/evaluation/`) |
| L1 reference-aware pruning (`enable_reference_aware_pruning`) | Prunes tool results without losing referenced context | No answer-quality regression on the evaluation suite with ≥ 30% context reduction on long-trajectory tasks | Pruning error → unpruned context (fail-open, safe) | No measurement yet |

## Precedent

The `completion_strategy=rubric` row is the template working end-to-end: the flag shipped
opt-in (ADR-009), the gate was defined before any measurement existed (ADR-011), the
measurement infrastructure was built offline-first
(`benchmarks/judge_calibration/`), seven runs produced a gate-passing candidate with the
evidence artifact recorded, and graduation is now a checklist rather than a debate. Every
other flag on this table should follow the same arc or be removed.
