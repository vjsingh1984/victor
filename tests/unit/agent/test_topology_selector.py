# Copyright 2025 Vijaykumar Singh <singhv@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for heuristic topology selector."""

from victor.agent.topology_contract import TopologyAction, TopologyDecisionInput, TopologyKind
from victor.agent.topology_selector import TopologySelector


class TestTopologySelector:
    """Test heuristic topology choices."""

    def test_selects_single_agent_for_shallow_observable_task(self):
        """Low-complexity observable work should stay in a single-agent loop."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="rename this variable",
                task_type="edit",
                task_complexity="low",
                confidence_hint=0.7,
                expected_depth="low",
                expected_breadth="low",
                observability_level="high",
                provider_candidates=["ollama", "openai"],
            )
        )

        assert decision.action == TopologyAction.SINGLE_AGENT
        assert decision.topology == TopologyKind.SINGLE_AGENT
        assert decision.provider == "ollama"
        assert "single-agent loop" in decision.rationale

    def test_selects_parallel_exploration_for_breadth_heavy_task(self):
        """Breadth-heavy search should favor parallel exploration."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="search for all auth-related failures",
                task_type="analysis",
                task_complexity="high",
                confidence_hint=0.5,
                tool_budget=8,
                expected_depth="medium",
                expected_breadth="high",
                observability_level="low",
                latency_sensitivity="low",
                prior_failures=1,
                provider_candidates=["ollama", "openai"],
            )
        )

        assert decision.action == TopologyAction.PARALLEL_EXPLORATION
        assert decision.topology == TopologyKind.PARALLEL_EXPLORATION
        assert decision.grounding_requirements.max_workers >= 2
        assert "parallel exploration" in decision.rationale

    def test_selects_team_plan_for_deep_complex_task(self):
        """Deep architectural work should favor team planning when formations exist."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="redesign the orchestration architecture",
                task_type="design",
                task_complexity="high",
                confidence_hint=0.6,
                tool_budget=12,
                iteration_budget=3,
                expected_depth="high",
                expected_breadth="medium",
                observability_level="medium",
                latency_sensitivity="low",
                available_team_formations=["parallel", "hierarchical", "adaptive"],
                provider_candidates=["openai", "anthropic"],
            )
        )

        assert decision.action == TopologyAction.TEAM_PLAN
        assert decision.topology == TopologyKind.TEAM
        assert decision.formation == "hierarchical"
        assert decision.grounding_requirements.formation == "hierarchical"
        assert "team plan" in decision.rationale

    def test_selects_escalate_model_for_ambiguous_low_observability_task(self):
        """Low-confidence, low-observability work should escalate model strength."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="determine the root cause from sparse logs",
                task_type="debug",
                task_complexity="high",
                confidence_hint=0.2,
                tool_budget=6,
                expected_depth="medium",
                expected_breadth="low",
                observability_level="low",
                prior_failures=1,
                provider_candidates=["openai", "anthropic"],
            )
        )

        assert decision.action == TopologyAction.ESCALATE_MODEL
        assert decision.topology == TopologyKind.ESCALATED_SINGLE_AGENT
        assert decision.provider == "openai"
        assert decision.grounding_requirements.escalation_target == "openai"
        assert "stronger model path" in decision.rationale

    def test_selects_safe_stop_for_high_privacy_remote_only_case(self):
        """High-privacy remote-only cases should stop safely instead of escalating."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="inspect confidential customer incident data",
                task_type="analysis",
                task_complexity="high",
                confidence_hint=0.2,
                tool_budget=6,
                expected_depth="high",
                expected_breadth="medium",
                privacy_sensitivity="high",
                observability_level="low",
                prior_failures=2,
                provider_candidates=["openai", "anthropic"],
            )
        )

        assert decision.action == TopologyAction.SAFE_STOP
        assert decision.topology == TopologyKind.SAFE_STOP
        assert decision.provider is None
        assert decision.grounding_requirements.safety_mode == "privacy_block"
        assert "safe stop" in decision.rationale

    def test_sets_fallback_action_when_decision_is_close(self):
        """Close or weak decisions should preserve a fallback action."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="investigate an intermittent issue",
                task_type="analysis",
                task_complexity="medium",
                confidence_hint=0.55,
                tool_budget=6,
                expected_depth="medium",
                expected_breadth="medium",
                observability_level="medium",
                latency_sensitivity="medium",
                prior_failures=1,
                provider_candidates=["ollama", "openai"],
            )
        )

        assert decision.fallback_action is not None
        assert decision.fallback_action != decision.action
        assert decision.score_breakdown
        assert "single_agent" in decision.score_breakdown
        assert "escalate_model" in decision.score_breakdown
