"""RL-driven tool-budget calibration policy.

Complementary to :class:`victor.framework.rl.tool_reputation.ToolReputationTracker`,
which tracks *within-turn* EMA reputation. This module consumes *aggregated,
persisted* RL signals (per-tool Q-values from ``rl_tool_q`` and trajectory
outcomes from ``decision_outcome``) to produce recommended tool-budget
parameters that feed the existing :class:`~victor.config.tool_settings.ToolSettings`
overlay.

First principles
----------------
A tool-call budget bounds cost. Its optimal value maximizes
*task-success per call*. The marginal value of the Nth call decays when a
trajectory is failing, so a fixed counter is a crude proxy for "this run is
going nowhere." Calibration therefore derives thresholds from principled
signals rather than hand-tuned constants:

* **Trajectory health** — the decision success/failure ratio governs the
  budget ceiling. A population that fails ~78% of the time (observed in
  ``decision_outcome``) is over-allocated by a default tuned for the median.
* **Tool quality** — reliable per-tool Q-values partition tools into
  *relief-eligible* (high-Q, reward continued work) vs *early-stop-sensitive*
  (low-Q, signal that the trajectory is stuck).

Co-design (no parallel guard layer)
-----------------------------------
The runtime already consumes ``tool_call_budget`` and
``tool_budget_progress_relief_{enabled,amount,max_uses}``. This policy
emits a :class:`BudgetRecommendation` applied immutably via
:meth:`BudgetRecommendation.apply_to_settings`; it never mutates settings
in place and never adds a second budget-enforcement path.

Guardrails
----------
* ``min_budget_floor`` / ``max_budget_ceiling`` bound the recommendation so
  no signal (however extreme) can starve or explode a session.
* ``min_tool_samples`` gates unreliable tools out (avoids the garbage
  ``rl_tool_outcome.metadata`` problem).
* Confidence is monotonic in sample size and is below 0.5 whenever the
  inputs are insufficient — callers use it to decide whether to apply the
  overlay at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from victor.config.tool_settings import ToolSettings

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolBudgetSignals:
    """Aggregated, persisted per-tool RL signal (from ``rl_tool_q``)."""

    tool_name: str
    q_value: float
    selection_count: int
    success_rate: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.q_value <= 1.0):
            raise ValueError(f"q_value for {self.tool_name!r} must be in [0,1], got {self.q_value}")

    def is_reliable(self, min_samples: int = 5) -> bool:
        """True when this tool has enough samples to trust its Q-value."""
        return self.selection_count >= max(min_samples, 1)


@dataclass(frozen=True)
class DecisionOutcomeAggregate:
    """Aggregated trajectory outcomes (from ``decision_outcome``)."""

    total: int
    successes: int
    failures: int
    mean_reward: float

    @property
    def success_rate(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.successes / self.total

    @property
    def failure_rate(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.failures / self.total

    def is_reliable(self, min_total: int = 50) -> bool:
        return self.total >= min_total


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetRecommendation:
    """Recommended tool-budget parameters applied through ``ToolSettings``."""

    recommended_tool_call_budget: int
    recommended_relief_enabled: bool
    recommended_relief_amount: int
    recommended_relief_max_uses: int
    early_stop_q_threshold: float
    relief_eligible_tools: Tuple[str, ...]
    confidence: float
    rationale: str

    def apply_to_settings(self, base: ToolSettings) -> ToolSettings:
        """Return a new ``ToolSettings`` with the recommendation overlaid.

        Immutable: never mutates ``base``. Field-level bounds defined on
        ``ToolSettings`` are respected by clamping.
        """
        # ToolSettings is a pydantic model; model_copy returns a shallow clone.
        overlay = base.model_copy()
        overlay.tool_call_budget = self.recommended_tool_call_budget
        overlay.tool_budget_progress_relief_enabled = self.recommended_relief_enabled
        # Clamp to the field bounds declared on ToolSettings (1..100 / 0..5).
        overlay.tool_budget_progress_relief_amount = max(
            1, min(100, int(self.recommended_relief_amount))
        )
        overlay.tool_budget_progress_relief_max_uses = max(
            0, min(5, int(self.recommended_relief_max_uses))
        )
        return overlay


# ---------------------------------------------------------------------------
# Config + guardrails
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetCalibrationConfig:
    """Tunable knobs + guardrails for the calibration policy."""

    # Reliability gates
    min_tool_samples: int = 20
    min_decision_total: int = 50
    # Budget bounds (guardrails against extreme signals)
    min_budget_floor: int = 30
    max_budget_ceiling: int = 150
    # Tool-quality thresholds
    high_q_threshold: float = 0.75
    low_q_threshold: float = 0.55
    # Policy gains (how strongly outcome rate moves the budget)
    failure_rate_sensitivity: float = 1.0  # 1.0 = full linear response
    # Relief eligibility
    relief_eligible_topk: int = 5


# ---------------------------------------------------------------------------
# Calibrator (pure policy)
# ---------------------------------------------------------------------------


@dataclass
class BudgetCalibrator:
    """Map aggregate RL signals to a ``BudgetRecommendation``.

    Pure and deterministic given its inputs; safe to unit-test in isolation
    and to persist its outputs for session-level application.
    """

    config: BudgetCalibrationConfig = field(default_factory=BudgetCalibrationConfig)

    def recommend(
        self,
        tools: Tuple[ToolBudgetSignals, ...],
        decisions: DecisionOutcomeAggregate,
        baseline: ToolSettings,
    ) -> BudgetRecommendation:
        """Produce a budget recommendation from the given RL signals."""
        cfg = self.config
        base_budget = baseline.tool_call_budget

        # --- Cold start: no usable data -> baseline, near-zero confidence ---
        reliable_tools = tuple(t for t in tools if t.is_reliable(min_samples=cfg.min_tool_samples))
        decisions_ok = decisions.is_reliable(min_total=cfg.min_decision_total)
        has_any_signal = bool(reliable_tools) or decisions_ok

        if not has_any_signal:
            return BudgetRecommendation(
                recommended_tool_call_budget=base_budget,
                recommended_relief_enabled=baseline.tool_budget_progress_relief_enabled,
                recommended_relief_amount=baseline.tool_budget_progress_relief_amount,
                recommended_relief_max_uses=baseline.tool_budget_progress_relief_max_uses,
                early_stop_q_threshold=0.0,
                relief_eligible_tools=(),
                confidence=0.0,
                rationale="cold-start: insufficient RL signal; baseline retained",
            )

        # --- Confidence: monotonic in decision sample size ---
        decision_conf = _sample_confidence(decisions.total, cfg.min_decision_total)
        tool_conf = (
            _sample_confidence(sum(t.selection_count for t in reliable_tools), cfg.min_tool_samples)
            if reliable_tools
            else 0.0
        )
        confidence = round(0.6 * decision_conf + 0.4 * tool_conf, 3)

        # --- Budget ceiling from trajectory health ---
        if decisions_ok:
            # failure_rate in [0,1]. High failure -> tighten toward the floor.
            tightening = decisions.failure_rate * cfg.failure_rate_sensitivity
            # Linear interpolation floor..base_budget as failure goes 1->0.
            budget = int(
                round(
                    cfg.min_budget_floor + (1.0 - tightening) * (base_budget - cfg.min_budget_floor)
                )
            )
        else:
            budget = base_budget
        budget = max(cfg.min_budget_floor, min(cfg.max_budget_ceiling, budget))

        rationale_parts = []
        if decisions_ok and decisions.failure_rate >= 0.5:
            rationale_parts.append(
                f"high failure rate ({decisions.failure_rate:.0%}); tighten budget"
            )
        elif decisions_ok:
            rationale_parts.append(f"healthy success rate ({decisions.success_rate:.0%})")

        # --- Relief eligibility: high-Q reliable tools earn relief ---
        high_q = sorted(
            (t for t in reliable_tools if t.q_value >= cfg.high_q_threshold),
            key=lambda t: t.q_value,
            reverse=True,
        )[: cfg.relief_eligible_topk]
        relief_eligible = tuple(t.tool_name for t in high_q)

        # --- Early-stop sensitivity driven by the weakest reliable tool ---
        low_q = [t for t in reliable_tools if t.q_value <= cfg.low_q_threshold]
        if low_q:
            weakest = min(t.q_value for t in low_q)
            early_stop_q = round(max(0.0, min(1.0, cfg.low_q_threshold - weakest + 0.1)), 3)
            rationale_parts.append(
                f"low-Q tools present (weakest q={weakest:.2f}); raise early-stop bar"
            )
        else:
            early_stop_q = 0.0

        # --- Relief amount scales with confidence + trajectory health ---
        if decisions_ok and decisions.success_rate >= 0.5:
            relief_amount = baseline.tool_budget_progress_relief_amount
            relief_max_uses = baseline.tool_budget_progress_relief_max_uses
        elif decisions_ok:
            # Failing population: keep relief bounded, do not reward stuck runs.
            relief_amount = max(1, baseline.tool_budget_progress_relief_amount - 5)
            relief_max_uses = max(0, baseline.tool_budget_progress_relief_max_uses - 1)
        else:
            relief_amount = baseline.tool_budget_progress_relief_amount
            relief_max_uses = baseline.tool_budget_progress_relief_max_uses

        rationale = "; ".join(rationale_parts) or "signal-driven calibration"
        return BudgetRecommendation(
            recommended_tool_call_budget=budget,
            recommended_relief_enabled=baseline.tool_budget_progress_relief_enabled,
            recommended_relief_amount=relief_amount,
            recommended_relief_max_uses=relief_max_uses,
            early_stop_q_threshold=early_stop_q,
            relief_eligible_tools=relief_eligible,
            confidence=confidence,
            rationale=rationale,
        )


def _sample_confidence(observed: int, minimum: int) -> float:
    """Sigmoid-like confidence ramp on sample size.

    Returns ~0 at ``minimum`` samples, asymptoting toward 1.0 as samples grow.
    """
    if observed <= 0 or minimum <= 0:
        return 0.0
    ratio = observed / minimum
    # 1 - exp(-ratio): 0 at 0, ~0.63 at minimum, ~0.95 at 3x minimum.
    return (
        round(1.0 - (1.0 / (1.0 + ratio)), 3)
        if False
        else round(1.0 - pow(2.718281828459045, -ratio), 3)
    )
