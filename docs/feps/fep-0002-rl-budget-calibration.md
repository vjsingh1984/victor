---
fep: "0002"
title: "RL-Driven Tool-Budget Calibration"
type: Standards Track
status: Draft
created: 2026-07-08
modified: 2026-07-08
authors:
  - name: "Vijaykumar Singh"
reviewers: []
discussion: null
implementation: null
---

# FEP-0002: RL-Driven Tool-Budget Calibration

## Abstract

Introduce an opt-in, settings-gated pipeline that calibrates the session tool
budget (`tool_call_budget`) from aggregated reinforcement-learning signals
(`rl_tool_q` Q-values and `decision_outcome` aggregates) already stored in the
global Victor database. The pipeline is pure and three-phase — a signal reader,
a calibration policy, and an application seam — and applies an *immutable*
`ToolSettings` overlay only when (a) the user opts in and (b) the calibrator's
confidence exceeds a configurable threshold. The default-off path returns the
baseline unchanged, yielding zero behavior change until explicitly enabled. This
proposal covers the runtime wiring of the already-implemented (but unwired)
calibration modules, which is gated behind this FEP's review window per the
framework's public-API / runtime-behavior change policy.

## Motivation

The static `tool_call_budget` does not adapt to observed execution reality.
Aggregate analysis of the global RL database showed two principled signals the
static budget ignores:

- **Per-tool learned value** (`rl_tool_q`): high-selection, high-Q tools deserve
  more headroom; low-Q tools are budget sinks.
- **Outcome history** (`decision_outcome`): a high observed failure rate means
  budget is being spent on fruitless calls and should tighten; consistent
  success justifies relaxing.

Today the budget over-allocates when execution is failing and under-allocates
when high-value tools consistently succeed. Calibration closes that gap
principled-ly, reducing wasted tool calls / cost while preserving headroom for
productive sessions.

## Specification

### Pipeline (pure, three-phase)

1. **`BudgetSignalReader`** — loads aggregated `rl_tool_q` Q-values and
   `decision_outcome` success/failure/reward from the **existing** global DB
   (`~/.victor/victor.db`) via the existing `db.cursor()` / `Tables.*` pattern.
   No new tables, no new DB path.
2. **`BudgetCalibrator`** — pure policy producing a `BudgetRecommendation`
   (`recommended_tool_call_budget`, `confidence` in [0,1], `rationale`).
3. **`apply_budget_calibration()` seam** — the single composition entry point,
   gated by:
   - `tool_budget_calibration_enabled: bool = False`
   - `tool_budget_calibration_min_confidence: float = 0.5` (0.0-1.0)

### The identity-return contract

The seam returns `settings.tools` **by identity** when: the toggle is off,
confidence < threshold, confidence <= 0.0 (cold-start / degraded reads), or a
pipeline exception occurs. Consequences:

- **Zero behavior change by default** — default-off + identity-return means the
  feature is inert until explicit opt-in *and* sufficient confidence.
- **One wiring site, zero consumption-site changes** — the overlay is a new
  immutable `ToolSettings` with identical schema, so all ~30
  `settings.tools.tool_call_budget` readers consume the calibrated value with no
  edit.

### Wiring (this FEP's subject)

A single call inside `AgentFactory.create()`, after
`SessionConfig.apply_to_settings()`:

    from victor.framework.rl.budget_calibration_seam import apply_budget_calibration
    settings.tools = apply_budget_calibration(settings)

Because the seam is identity-returning when disabled, this line is a no-op unless
a user opts in.

### UX and observability (non-runtime, not FEP-blocked)

- Structured logging at the seam on every apply decision, with confidence,
  baseline, recommended budget, and rationale.
- `/system` provenance: when enabled, the tool-budget line shows the calibrated
  value with provenance (confidence, baseline). Provenance is *not* threaded
  through `ToolSettings` (config stays pure); it is surfaced via the log/observation path.

## Rationale and Alternatives

- **Settings-gated (chosen)** — consistent with the prompt-optimization precedent
  ("controlled entirely via settings, not feature flags").
- **Mutable `ToolSettings` (rejected)** — breaks immutability and the
  "SessionConfig is the single mutation point" mandate.
- **`calibration_status` provenance field on `ToolSettings` (rejected)** — couples
  a config type to runtime observability; read in ~30 sites. Provenance is logged
  instead.
- **Feature-flag gating (rejected)** — inconsistent with the settings-driven
  precedent for runtime-evolution behavior.
- **Per-call dynamic admission control (rejected, out of scope)** — this FEP is
  *session* budget calibration only.

## Backwards Compatibility

100%. Default-off + identity-return means no existing session changes behavior.
The two new settings are optional with safe defaults. No schema migrations
(global DB schema unchanged). No consumption-site edits.

## Security and Privacy

No new data is collected or persisted. The reader queries *existing* RL tables
already governed by the global-database access pattern. Calibration is a pure
read->compute->immutable-overlay; it writes nothing.

## Implementation Status

**Completed and committed (34 tests GREEN across 3 modules):**

| Phase | Commit | Module | Tests |
|-------|--------|--------|-------|
| 1 - Pure policy | `0f6f8db45` | `budget_calibration.py` | 20 |
| 2 - Persistence adapter | `b33ee3c87` | `budget_signal_reader.py` | 8 |
| 3 - Application seam | `b47861ef4` | `budget_calibration_seam.py` + 2 settings | 6 |

Docs parity: `docs/reference/configuration-options.md` documents both settings
(commit `6cca24360`). Decision record: ADR-017.

**Deferred (explicitly blocked on this FEP's review window):**

- The single `AgentFactory.create()` wiring call above.
- An integration test asserting the orchestrator receives the overlay.
- This is the only *runtime behavior change* in the proposal; everything else is
  inert / additive.

## Open Questions

1. Is the default `min_confidence = 0.5` the right gate, or should it be higher
   (e.g. 0.7) for a first opt-in release?
2. Should the seam apply calibration *before* or *after* session-config budget
   overrides take precedence? Current design: calibration runs after
   `apply_to_settings`, so it can re-tighten — needs reviewer decision on
   precedence.

## References

- ADR-017: RL-Driven Tool-Budget Calibration
- ADR-004: Tool System Architecture
- CLAUDE.md - "Prompt Optimization Settings" (settings-vs-flag precedent);
  "When framework changes need more than code" (FEP triggers)
- `victor/framework/rl/budget_calibration.py`,
  `victor/framework/rl/budget_signal_reader.py`,
  `victor/framework/rl/budget_calibration_seam.py`
