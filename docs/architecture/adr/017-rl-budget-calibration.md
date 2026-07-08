# ADR-017: RL-Driven Tool-Budget Calibration

## Metadata

- **Status**: Proposed
- **Date**: 2026-07-08
- **Decision Makers**: Vijaykumar Singh
- **Related ADRs**: ADR-004 (Tool System Architecture)

## Context

The session tool budget (`tool_call_budget`) is a static, globally configured value
that bounds how many tool calls an agent may make per session. It does not adapt
to observed execution reality. Analysis of the global RL database
(`~/.victor/victor.db`) revealed two principled signals that the static budget
ignores:

- **Per-tool learned value** (`rl_tool_q`): tools with high selection counts and
  high Q-values are "worth" more budget headroom than the baseline implies; tools
  with low Q-values are budget sinks.
- **Aggregate outcome history** (`decision_outcome`): a high observed failure rate
  indicates the agent is spending budget on fruitless tool calls and should be
  tightened.

The current static budget therefore over-allocates budget when execution is
failing and under-allocates when high-value tools are consistently succeeding.
The question is *how* to let the budget respond to these signals without
destabilizing the runtime, coupling config types to observability data, or
introducing a parallel data path.

## Decision

Introduce a **three-phase, pure, settings-gated, default-off** calibration
pipeline. The budget is calibrated by composing three pure modules that map RL
signals onto an immutable `ToolSettings` overlay:

1. **`BudgetSignalReader`** (`victor/framework/rl/budget_signal_reader.py`) —
   loads aggregated `rl_tool_q` Q-values and `decision_outcome`
   success/failure/reward aggregates from the **existing** global database via the
   existing `db.cursor()` / `Tables.*` access pattern. No new tables, no new DB
   path.
2. **`BudgetCalibrator`** (`victor/framework/rl/budget_calibration.py`) — pure
   policy that maps the signals into a `BudgetRecommendation` carrying a calibrated
   `tool_call_budget`, a `confidence` score, and a `rationale` string.
3. **`apply_budget_calibration()` seam**
   (`victor/framework/rl/budget_calibration_seam.py`) — the single composition
   entry point that runs reader → calibrator → overlay, gated by two settings:

   - `tool_budget_calibration_enabled: bool = False` (opt-in, mirroring the
     prompt-optimization precedent of "controlled entirely via settings").
   - `tool_budget_calibration_min_confidence: float = 0.5` (0.0–1.0).

### The identity-return contract (the load-bearing invariant)

The seam returns `settings.tools` **by identity** whenever it does not apply an
overlay — i.e. when the toggle is off, when confidence is below threshold, or
when confidence is `<= 0.0` (cold-start / degraded DB reads). Pipeline
exceptions are caught and also retain the baseline. This has two consequences:

- **Zero behavior change by default.** Default-off + identity-return means the
  feature is inert until a user explicitly opts in *and* the calibrator is
  sufficiently confident. No FEP-mandated behavior change occurs at
  implementation time; the behavior change occurs only at opt-in.
- **One wiring site, zero consumption-site changes.** Because the overlay is a
  new immutable `ToolSettings` with the same schema, all ~30 existing
  `settings.tools.tool_call_budget` readers (streaming/tool execution,
  coordinators, tool builders, slash commands) consume the calibrated value
  with no code change.

## Rationale

- **Why settings-gated, not feature-flag-gated?** CLAUDE.md establishes the
  precedent that runtime evolution behavior (prompt optimization, GEPA) is
  "controlled entirely via settings, not feature flags." Budget calibration is
  the same class of concern; settings-gating keeps it consistent and avoids
  expanding the `FeatureFlag` enum.
- **Why immutable overlay + identity-return?** It makes the feature provably
  reversible (re-toggle off → baseline restored, unchanged object) and makes
  the default-off path free of any allocation or branching beyond the toggle
  check.
- **Why pure policy?** `BudgetCalibrator.recommend()` is a pure function of its
  inputs. This is maximally testable (no I/O, no global state) and lets the same
  pipeline serve both an "inspect" (dry-run) mode and an "apply" mode without a
  parallel harness.
- **Pro:** Adaptivity to observed execution; principled signal derivation;
  safe-by-default; minimal blast radius.
- **Con:** Adds two optional settings to a surface already carrying ~26 config
  groups; opt-in means the benefit is realized only by users who enable it.

## Consequences

- **Positive**: The budget can tighten on observed failure and relax on
  consistent high-value tool success, reducing wasted tool calls and cost while
  preserving headroom for productive sessions. The pipeline is reusable for A/B
  evaluation without runtime commitment.
- **Negative**: Introduces a bootstrap-time DB read when enabled (amortized once
  per session). Opt-in means most users see no change; the feature requires
  accumulated RL signal to produce non-baseline recommendations.
- **Neutral**: `ToolSettings` adds two new optional fields with safe defaults
  and bounds (`tool_budget_calibration_enabled=False`,
  `tool_budget_calibration_min_confidence` in [0,1]); no existing field is removed
  or renamed, so no migration is needed and all 4 canonical budget consumers
  require no edits. The global DB schema is unchanged.

## Implementation

**Completed (committed, 34 tests GREEN across 3 modules):**

| Phase | Commit | Module | Tests |
|-------|--------|--------|-------|
| 1 — Pure policy | `0f6f8db45` | `budget_calibration.py` | 20 |
| 2 — Persistence adapter | `b33ee3c87` | `budget_signal_reader.py` | 8 |
| 3 — Application seam | `b47861ef4` | `budget_calibration_seam.py` + 2 settings | 6 |

Docs parity: `docs/reference/configuration-options.md` documents both settings
(commit `6cca24360`).

**Deferred (blocked on FEP-0002 review — see "References"):**

- The single wiring call inside `Agent.create()` (`victor/framework/agent.py`,
  after the existing `session_config.apply_to_settings(settings)` call), assigning
  `settings.tools = apply_budget_calibration(settings)`. (Review correction: the
  apply point is `Agent.create()`, not `AgentFactory.create()`.)
- A prerequisite fix: `SessionConfig.apply_to_settings()` must actually apply
  `self.tool_budget` (currently validated but dropped), so the
  `explicit_override` precedence path is exercisable.
- An integration test asserting the orchestrator receives the overlay only when
  no explicit budget override wins.
- Runtime behavior change is intentionally gated behind the FEP's review window.

## Alternatives Considered

- **Mutate `ToolSettings` in place.** Rejected: breaks immutability of the
  overlay and the "single mutation point is `SessionConfig`" mandate.
- **Add a `calibration_status` provenance field to `ToolSettings`.** Rejected:
  couples a config type to runtime observability; read in ~30 sites. Provenance
  is surfaced via structured logs and a slash command instead, keeping the config
  object pure.
- **Feature-flag gating.** Rejected: inconsistent with the settings-driven
  precedent for runtime-evolution behavior.
- **Per-tool dynamic budget allocation at call time.** Rejected as
  out-of-scope: this ADR addresses *session* budget calibration, not per-call
  admission control.

## References

- FEP-0002: RL-Driven Tool-Budget Calibration (`docs/feps/fep-0002-rl-budget-calibration.md`)
- ADR-004: Tool System Architecture
- `docs/reference/configuration-options.md` — "Tool Execution & Budgets"
- CLAUDE.md — "Prompt Optimization Settings" (settings-vs-flag precedent)

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-07-08 | 1.0 | Initial ADR | Vijaykumar Singh |
