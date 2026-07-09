"""TDD tests for RL-driven tool-budget calibration.

Covers the pure calibration policy: mapping aggregate RL signals
(per-tool Q-values + trajectory decision outcomes) into recommended
tool-budget parameters that feed the existing settings overlay.

Verified RL data (from ``~/.victor/victor.db``) is used as fixtures so the
tests pin behavior against the real distribution:

    rl_tool_q (reliable):
        read        6189 sel / 5855 ok  -> q=0.918
        ls          1801 sel / 1771 ok  -> q=0.939
        code_search 1086 sel / 1056 ok  -> q=0.931
        shell        271 sel /  120 ok  -> q=0.360
        read_file    144 sel /   72 ok  -> q=0.488

    decision_outcome (reliable): 1303 total, 288 success, 1015 failure,
        mean_reward_success=1.0, mean_reward_failure=0.075  (~78% failure).

Co-design: the calibrator is a pure policy complementary to
``ToolReputationTracker`` (which tracks within-turn EMA reputation). It
produces a ``BudgetRecommendation`` applied through the existing
``ToolSettings`` overlay rather than adding a parallel guard layer.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from victor.config.tool_settings import ToolSettings
from victor.framework.rl.budget_calibration import (
    BudgetCalibrationConfig,
    BudgetCalibrator,
    BudgetRecommendation,
    DecisionOutcomeAggregate,
    ToolBudgetSignals,
)

# ---------------------------------------------------------------------------
# Fixtures grounded in verified RL data
# ---------------------------------------------------------------------------

HIGH_Q_TOOLS = (
    ToolBudgetSignals(tool_name="read", q_value=0.918, selection_count=6189, success_rate=0.946),
    ToolBudgetSignals(tool_name="ls", q_value=0.939, selection_count=1801, success_rate=0.983),
    ToolBudgetSignals(
        tool_name="code_search", q_value=0.931, selection_count=1086, success_rate=0.972
    ),
)
LOW_Q_TOOLS = (
    ToolBudgetSignals(tool_name="shell", q_value=0.360, selection_count=271, success_rate=0.442),
    ToolBudgetSignals(
        tool_name="read_file", q_value=0.488, selection_count=144, success_rate=0.500
    ),
)
ALL_TOOLS = HIGH_Q_TOOLS + LOW_Q_TOOLS

# 78% failure rate observed in decision_outcome.
MOSTLY_FAILING = DecisionOutcomeAggregate(
    total=1303, successes=288, failures=1015, mean_reward=0.21
)
MOSTLY_SUCCEEDING = DecisionOutcomeAggregate(
    total=1000, successes=800, failures=200, mean_reward=0.85
)


# ---------------------------------------------------------------------------
# ToolBudgetSignals invariants
# ---------------------------------------------------------------------------


class TestToolBudgetSignals:
    def test_filters_below_min_samples(self):
        sig = ToolBudgetSignals(tool_name="rare", q_value=0.9, selection_count=4, success_rate=0.9)
        assert sig.is_reliable(min_samples=5) is False

    def test_passes_min_samples(self):
        sig = ToolBudgetSignals(
            tool_name="common", q_value=0.9, selection_count=100, success_rate=0.9
        )
        assert sig.is_reliable(min_samples=5) is True

    def test_rejects_invalid_q_value(self):
        with pytest.raises(ValueError):
            ToolBudgetSignals(tool_name="bad", q_value=1.5, selection_count=10, success_rate=0.5)


# ---------------------------------------------------------------------------
# DecisionOutcomeAggregate invariants
# ---------------------------------------------------------------------------


class TestDecisionOutcomeAggregate:
    def test_failure_rate(self):
        assert MOSTLY_FAILING.failure_rate == pytest.approx(1015 / 1303, abs=0.001)

    def test_success_rate(self):
        assert MOSTLY_SUCCEEDING.success_rate == pytest.approx(0.8, abs=0.001)

    def test_is_reliable_requires_min_total(self):
        sparse = DecisionOutcomeAggregate(total=5, successes=3, failures=2, mean_reward=0.5)
        assert sparse.is_reliable(min_total=20) is False


# ---------------------------------------------------------------------------
# BudgetRecommendation: co-design with ToolSettings overlay
# ---------------------------------------------------------------------------


class TestBudgetRecommendationOverlay:
    def test_apply_overlay_returns_new_settings_instance(self):
        rec = BudgetRecommendation(
            recommended_tool_call_budget=120,
            recommended_relief_enabled=True,
            recommended_relief_amount=12,
            recommended_relief_max_uses=2,
            early_stop_q_threshold=0.4,
            relief_eligible_tools=("read", "ls"),
            confidence=0.7,
            rationale="test",
        )
        base = ToolSettings()
        overlay = rec.apply_to_settings(base)
        assert overlay is not base  # immutable overlay, not mutation
        assert overlay.tool_call_budget == 120
        assert overlay.tool_budget_progress_relief_amount == 12
        assert overlay.tool_budget_progress_relief_max_uses == 2

    def test_apply_overlay_clamps_to_field_bounds(self):
        # relief_amount bounded [1, 100]; max_uses bounded [0, 5]
        rec = BudgetRecommendation(
            recommended_tool_call_budget=50,
            recommended_relief_enabled=True,
            recommended_relief_amount=9999,  # out of range
            recommended_relief_max_uses=-3,  # out of range
            early_stop_q_threshold=0.4,
            relief_eligible_tools=(),
            confidence=0.7,
            rationale="clamp",
        )
        overlay = rec.apply_to_settings(ToolSettings())
        assert 1 <= overlay.tool_budget_progress_relief_amount <= 100
        assert 0 <= overlay.tool_budget_progress_relief_max_uses <= 5

    def test_disabled_relief_overlay_keeps_relief_off(self):
        rec = BudgetRecommendation(
            recommended_tool_call_budget=50,
            recommended_relief_enabled=False,
            recommended_relief_amount=10,
            recommended_relief_max_uses=1,
            early_stop_q_threshold=0.4,
            relief_eligible_tools=(),
            confidence=0.7,
            rationale="off",
        )
        assert rec.apply_to_settings(ToolSettings()).tool_budget_progress_relief_enabled is False


# ---------------------------------------------------------------------------
# BudgetCalibrator policy
# ---------------------------------------------------------------------------


class TestBudgetCalibratorPolicy:
    def test_low_confidence_when_insufficient_data(self):
        cal = BudgetCalibrator()
        empty = DecisionOutcomeAggregate(total=0, successes=0, failures=0, mean_reward=0.0)
        rec = cal.recommend(tools=(), decisions=empty, baseline=ToolSettings())
        # No tool signal AND no reliable decisions -> low confidence, baseline retained.
        assert rec.confidence < 0.5
        assert rec.recommended_relief_enabled is True  # safe default unchanged

    def test_low_confidence_when_decisions_unreliable(self):
        cal = BudgetCalibrator()
        sparse = DecisionOutcomeAggregate(total=3, successes=1, failures=2, mean_reward=0.3)
        rec = cal.recommend(tools=ALL_TOOLS, decisions=sparse, baseline=ToolSettings())
        assert rec.confidence < 0.5

    def test_high_failure_rate_tightens_budget(self):
        """~78% failure -> the current default over-allocates to failing runs."""
        cal = BudgetCalibrator()
        baseline = ToolSettings()
        rec = cal.recommend(tools=ALL_TOOLS, decisions=MOSTLY_FAILING, baseline=baseline)
        assert rec.recommended_tool_call_budget <= baseline.tool_call_budget
        assert rec.early_stop_q_threshold >= 0.0
        assert "failure" in rec.rationale.lower() or "tighten" in rec.rationale.lower()

    def test_high_success_rate_relaxes_relief_cap(self):
        """Healthy trajectories earn more relief headroom."""
        cal = BudgetCalibrator()
        baseline = ToolSettings()
        rec_fail = cal.recommend(tools=ALL_TOOLS, decisions=MOSTLY_FAILING, baseline=baseline)
        rec_ok = cal.recommend(tools=ALL_TOOLS, decisions=MOSTLY_SUCCEEDING, baseline=baseline)
        # Succeeding runs should not be punished with tighter caps than failing ones.
        assert rec_ok.recommended_tool_call_budget >= rec_fail.recommended_tool_call_budget

    def test_high_q_tools_become_relief_eligible(self):
        cal = BudgetCalibrator()
        rec = cal.recommend(tools=ALL_TOOLS, decisions=MOSTLY_SUCCEEDING, baseline=ToolSettings())
        assert "read" in rec.relief_eligible_tools
        assert "ls" in rec.relief_eligible_tools
        # Low-Q tools never become relief-eligible.
        assert "shell" not in rec.relief_eligible_tools

    def test_low_q_tools_drives_early_stop_threshold(self):
        """Presence of low-Q tools (shell=0.36) raises the early-stop bar."""
        cal = BudgetCalibrator()
        rec = cal.recommend(tools=ALL_TOOLS, decisions=MOSTLY_FAILING, baseline=ToolSettings())
        # Early stop should be sensitive to the weakest reliable tool.
        assert rec.early_stop_q_threshold > 0.0

    def test_cold_start_returns_safe_baseline(self):
        """No data at all -> baseline values, confidence ~0, no risky changes."""
        cal = BudgetCalibrator()
        rec = cal.recommend(
            tools=(),
            decisions=DecisionOutcomeAggregate(total=0, successes=0, failures=0, mean_reward=0.0),
            baseline=ToolSettings(),
        )
        assert rec.confidence < 0.2
        base = ToolSettings()
        assert rec.recommended_tool_call_budget == base.tool_call_budget

    def test_confidence_scales_with_sample_size(self):
        cal = BudgetCalibrator()
        base = ToolSettings()
        small = replace(MOSTLY_FAILING, total=60, successes=13, failures=47)
        large = replace(MOSTLY_FAILING, total=1303, successes=288, failures=1015)
        rec_small = cal.recommend(tools=ALL_TOOLS, decisions=small, baseline=base)
        rec_large = cal.recommend(tools=ALL_TOOLS, decisions=large, baseline=base)
        assert rec_large.confidence >= rec_small.confidence


# ---------------------------------------------------------------------------
# Config + guardrails
# ---------------------------------------------------------------------------


class TestCalibrationConfig:
    def test_min_samples_gates_tool_signal(self):
        cfg = BudgetCalibrationConfig(min_tool_samples=200)
        cal = BudgetCalibrator(cfg)
        rec = cal.recommend(tools=ALL_TOOLS, decisions=MOSTLY_FAILING, baseline=ToolSettings())
        # Only read(6189), ls(1801), code_search(1086) pass; shell(271) passes too,
        # read_file(144) is filtered out.
        assert "read_file" not in rec.relief_eligible_tools

    def test_max_budget_reduction_is_bounded(self):
        """Guardrail: never cut budget below a sane floor regardless of signal."""
        cfg = BudgetCalibrationConfig(min_budget_floor=30)
        cal = BudgetCalibrator(cfg)
        # Extreme failure signal.
        bad = DecisionOutcomeAggregate(total=10000, successes=1, failures=9999, mean_reward=0.0)
        rec = cal.recommend(tools=ALL_TOOLS, decisions=bad, baseline=ToolSettings())
        assert rec.recommended_tool_call_budget >= cfg.min_budget_floor

    def test_max_budget_increase_is_bounded(self):
        """Guardrail: never inflate budget beyond a sane ceiling."""
        cfg = BudgetCalibrationConfig(max_budget_ceiling=150)
        cal = BudgetCalibrator(cfg)
        good = DecisionOutcomeAggregate(total=10000, successes=9999, failures=1, mean_reward=1.0)
        rec = cal.recommend(tools=ALL_TOOLS, decisions=good, baseline=ToolSettings())
        assert rec.recommended_tool_call_budget <= cfg.max_budget_ceiling
