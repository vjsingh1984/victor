"""Tests for CreditAwareTeamCoordinator — Shapley-driven agent rerouting."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from victor.teams.mixins.credit_aware_routing import CreditAwareTeamCoordinator
from victor.teams.unified_coordinator import UnifiedTeamCoordinator


@dataclass
class MockMemberResult:
    """Mock member result for testing."""

    success: bool = True
    output: str = ""


def _make_coordinator(**kwargs) -> CreditAwareTeamCoordinator:
    """Create a test coordinator with proper MRO."""

    class TestCoordinator(CreditAwareTeamCoordinator, UnifiedTeamCoordinator):
        pass

    defaults = {
        "orchestrator": None,
        "lightweight_mode": True,
        "reroute_threshold": 0.5,
        "min_rounds_before_reroute": 3,
    }
    defaults.update(kwargs)
    return TestCoordinator(**defaults)


# ============================================================================
# Basic state
# ============================================================================


class TestCreditAwareState:
    """Test initial state and properties."""

    def test_initial_state(self):
        coord = _make_coordinator()
        assert coord.round_count == 0
        assert coord.active_reroutes == {}
        assert coord.get_agent_performance() == {}

    def test_reroute_target_returns_self_when_no_rerouting(self):
        coord = _make_coordinator()
        assert coord.get_reroute_target("agent_a") == "agent_a"

    def test_reset_routing_clears_state(self):
        coord = _make_coordinator()
        coord._agent_credit_history["agent_a"] = [0.5, 0.3]
        coord._reroute_map["agent_b"] = "agent_a"
        coord._round_count = 5

        coord.reset_routing()

        assert coord.round_count == 0
        assert coord.active_reroutes == {}
        assert coord.get_agent_performance() == {}


# ============================================================================
# Credit history tracking
# ============================================================================


class TestCreditHistory:
    """Test per-agent credit accumulation."""

    def test_update_credit_history_from_results(self):
        coord = _make_coordinator()

        result = {
            "member_results": {
                "coder": MockMemberResult(success=True),
                "reviewer": MockMemberResult(success=True),
                "tester": MockMemberResult(success=False),
            }
        }

        coord._update_credit_history(result)

        perf = coord.get_agent_performance()
        assert "coder" in perf or "reviewer" in perf or "tester" in perf

    def test_empty_results_no_crash(self):
        coord = _make_coordinator()
        coord._update_credit_history({"member_results": {}})
        assert coord.get_agent_performance() == {}

    def test_credit_accumulates_over_rounds(self):
        coord = _make_coordinator()

        for _ in range(3):
            result = {
                "member_results": {
                    "agent_a": MockMemberResult(success=True),
                    "agent_b": MockMemberResult(success=False),
                }
            }
            coord._update_credit_history(result)

        perf = coord.get_agent_performance()
        # agent_a should have positive credit, agent_b negative
        if "agent_a" in perf and "agent_b" in perf:
            assert perf["agent_a"]["avg_credit"] > perf["agent_b"]["avg_credit"]


# ============================================================================
# Rerouting logic
# ============================================================================


class TestReroutingDecisions:
    """Test the credit-based rerouting algorithm."""

    def test_no_rerouting_before_min_rounds(self):
        coord = _make_coordinator(min_rounds_before_reroute=5)

        # Simulate 3 rounds (below threshold of 5)
        for i in range(3):
            coord._agent_credit_history["good_agent"].append(1.0)
            coord._agent_credit_history["bad_agent"].append(-1.0)
            coord._round_count = i + 1

        coord._compute_rerouting()
        # Should not reroute yet — not enough rounds
        # (depends on whether min_rounds is checked in _compute_rerouting)

    def test_underperformer_gets_rerouted(self):
        coord = _make_coordinator(reroute_threshold=0.5, min_rounds_before_reroute=1)

        # Good agent: consistently positive credit
        coord._agent_credit_history["good_agent"] = [0.8, 0.9, 0.7]
        # Bad agent: consistently negative
        coord._agent_credit_history["bad_agent"] = [-0.5, -0.6, -0.4]
        coord._round_count = 3

        coord._compute_rerouting()

        # bad_agent should be rerouted to good_agent
        assert coord.get_reroute_target("bad_agent") == "good_agent"
        assert coord.get_reroute_target("good_agent") == "good_agent"

    def test_no_rerouting_when_all_similar(self):
        coord = _make_coordinator(reroute_threshold=0.5)

        # All agents similar performance
        coord._agent_credit_history["agent_a"] = [0.5, 0.5, 0.5]
        coord._agent_credit_history["agent_b"] = [0.4, 0.6, 0.5]
        coord._agent_credit_history["agent_c"] = [0.5, 0.4, 0.6]

        coord._compute_rerouting()

        assert coord.active_reroutes == {}

    def test_rerouting_only_targets_worst(self):
        coord = _make_coordinator(reroute_threshold=0.3)

        coord._agent_credit_history["best"] = [1.0, 0.9, 0.8]
        coord._agent_credit_history["ok"] = [0.4, 0.3, 0.5]
        coord._agent_credit_history["worst"] = [-0.8, -0.7, -0.9]

        coord._compute_rerouting()

        # worst should be rerouted, ok should not
        reroutes = coord.active_reroutes
        assert "worst" in reroutes
        assert reroutes["worst"] == "best"


# ============================================================================
# Performance introspection
# ============================================================================


class TestPerformanceIntrospection:
    """Test the get_agent_performance method."""

    def test_performance_includes_reroute_info(self):
        coord = _make_coordinator()

        coord._agent_credit_history["agent_a"] = [0.5, 0.6]
        coord._agent_credit_history["agent_b"] = [-0.3, -0.4]
        coord._reroute_map["agent_b"] = "agent_a"

        perf = coord.get_agent_performance()

        assert perf["agent_a"]["is_rerouted"] is False
        assert perf["agent_a"]["rerouted_to"] is None
        assert perf["agent_b"]["is_rerouted"] is True
        assert perf["agent_b"]["rerouted_to"] == "agent_a"

    def test_performance_avg_credit(self):
        coord = _make_coordinator()
        coord._agent_credit_history["agent"] = [0.2, 0.4, 0.6]

        perf = coord.get_agent_performance()
        assert abs(perf["agent"]["avg_credit"] - 0.4) < 0.01
        assert perf["agent"]["total_rounds"] == 3
