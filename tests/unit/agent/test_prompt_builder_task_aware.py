"""Tests for task-aware system prompts — Layer 2 of agentic execution quality."""

import pytest

from victor.agent.prompt_builder import SystemPromptBuilder
from victor.agent.query_classifier import QueryClassification, QueryType
from victor.framework.task.protocols import TaskComplexity


def _make_classification(query_type: QueryType) -> QueryClassification:
    return QueryClassification(
        query_type=query_type,
        complexity=TaskComplexity.MEDIUM,
        should_plan=False,
        should_use_subagents=False,
        continuation_budget_hint=4,
        confidence=0.8,
    )


def _make_builder(query_type=None, available_tools=None):
    classification = _make_classification(query_type) if query_type else None
    return SystemPromptBuilder(
        provider_name="anthropic",
        model="claude-sonnet-4-20250514",
        available_tools=available_tools,
        query_classification=classification,
    )


class TestTaskGuidance:
    def test_exploration_prompt_includes_systematic_guidance(self):
        builder = _make_builder(QueryType.EXPLORATION)
        prompt = builder.build()
        assert "systematically" in prompt.lower() or "map structure" in prompt.lower()

    def test_implementation_prompt_includes_plan_guidance(self):
        builder = _make_builder(QueryType.IMPLEMENTATION)
        prompt = builder.build()
        assert "plan before" in prompt.lower() or "break into" in prompt.lower()

    def test_debugging_prompt_includes_error_focus(self):
        builder = _make_builder(QueryType.DEBUGGING)
        prompt = builder.build()
        assert "error messages" in prompt.lower() or "stack traces" in prompt.lower()

    def test_quick_question_prompt_is_concise(self):
        builder = _make_builder(QueryType.QUICK_QUESTION)
        prompt = builder.build()
        assert "directly" in prompt.lower() or "concisely" in prompt.lower()

    def test_no_classification_omits_task_guidance(self):
        builder = _make_builder(None)
        prompt = builder.build()
        # Without classification, no task-specific guidance injected
        assert "TASK GUIDANCE:" not in prompt

    def test_tool_constraint_lists_available_tools(self):
        builder = _make_builder(available_tools=["read_file", "write_file", "shell"])
        prompt = builder.build()
        assert "read_file" in prompt
        assert "write_file" in prompt
        assert "shell" in prompt

    def test_prompt_content_varies_by_type(self):
        quick = _make_builder(QueryType.QUICK_QUESTION).build()
        explore = _make_builder(QueryType.EXPLORATION).build()
        # Each type includes its own task guidance
        assert "Answer directly" in quick or "concisely" in quick.lower()
        assert "systematically" in explore.lower() or "map structure" in explore.lower()
        # They should differ (different task guidance injected)
        assert quick != explore


class TestGEPAPromptIntegration:
    """Tests for GEPA-evolved prompt section integration in the builder."""

    def test_default_prompt_includes_static_grounding(self):
        """System prompt always includes static GROUNDING_RULES."""
        from victor.agent.prompt_builder import GROUNDING_RULES

        builder = _make_builder()
        prompt = builder.build()
        assert GROUNDING_RULES in prompt

    def test_default_prompt_includes_static_completion(self):
        """System prompt always includes static COMPLETION_GUIDANCE."""
        from victor.agent.prompt_builder import COMPLETION_GUIDANCE

        builder = _make_builder()
        prompt = builder.build()
        assert COMPLETION_GUIDANCE in prompt

    def test_optimized_grounding_via_optimization_injector(self):
        """GEPA-evolved GROUNDING_RULES is served via OptimizationInjector."""
        from unittest.mock import patch, MagicMock
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )
        from victor.framework.rl.base import RLRecommendation
        from victor.agent.optimization_injector import OptimizationInjector

        evolved_text = "EVOLVED GROUNDING: Always verify tool output."

        mock_rec = RLRecommendation(
            value=evolved_text,
            confidence=0.9,
            reason="GEPA gen-2",
            sample_size=10,
            is_baseline=False,
        )
        mock_learner = MagicMock()
        mock_learner.get_recommendation.return_value = mock_rec
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = PromptOptimizationSettings(enabled=True)

        with (
            patch(
                "victor.config.settings.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "victor.framework.rl.coordinator.get_rl_coordinator",
                return_value=mock_coordinator,
            ),
        ):
            injector = OptimizationInjector()
            sections = injector.get_evolved_sections("deepseek", "deepseek-chat", "edit")

        assert any(evolved_text in s for s in sections)

    def test_optimized_completion_replaces_static(self):
        """GEPA-evolved COMPLETION_GUIDANCE is served via OptimizationInjector."""
        from unittest.mock import patch, MagicMock
        from victor.agent.prompt_builder import COMPLETION_GUIDANCE
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )
        from victor.framework.rl.base import RLRecommendation
        from victor.agent.optimization_injector import OptimizationInjector

        evolved_text = "EVOLVED COMPLETION: Signal with **FINISHED**: marker."

        mock_rec = RLRecommendation(
            value=evolved_text,
            confidence=0.9,
            reason="GEPA gen-3",
            sample_size=10,
            is_baseline=False,
        )
        mock_learner = MagicMock()
        mock_learner.get_recommendation.return_value = mock_rec
        mock_coordinator = MagicMock()
        mock_coordinator.get_learner.return_value = mock_learner

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = PromptOptimizationSettings(enabled=True)

        with (
            patch(
                "victor.config.settings.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "victor.framework.rl.coordinator.get_rl_coordinator",
                return_value=mock_coordinator,
            ),
        ):
            injector = OptimizationInjector()
            sections = injector.get_evolved_sections("deepseek", "deepseek-chat", "edit")

            # System prompt retains static COMPLETION_GUIDANCE
            builder = _make_builder()
            prompt = builder.build()

        # Evolved version goes to user prefix via injector
        assert any(evolved_text in s for s in sections)
        # Static baseline remains in system prompt
        assert COMPLETION_GUIDANCE in prompt

    def test_prompt_optimization_disabled_uses_static_prompts(self):
        """When prompt_optimization.enabled=False, no GEPA candidates used."""
        from unittest.mock import patch, MagicMock
        from victor.agent.prompt_builder import GROUNDING_RULES
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        mock_settings = MagicMock()
        mock_settings.prompt_optimization = PromptOptimizationSettings(enabled=False)

        with patch(
            "victor.config.settings.get_settings",
            return_value=mock_settings,
        ):
            builder = _make_builder()
            prompt = builder.build()

        # Static GROUNDING_RULES should remain (no optimization applied)
        assert GROUNDING_RULES in prompt


class TestPromptOptimizationSettings:
    """Tests for the strategy-agnostic prompt optimization config."""

    def test_defaults(self):
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        s = PromptOptimizationSettings()
        assert s.enabled  # Enabled by default for GEPA/MIPROv2/CoT
        assert s.default_strategies == ["gepa"]
        assert s.section_strategies == {}

    def test_get_strategies_returns_default(self):
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        s = PromptOptimizationSettings(enabled=True)
        assert s.get_strategies_for_section("GROUNDING_RULES") == ["gepa"]

    def test_get_strategies_uses_override(self):
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        s = PromptOptimizationSettings(
            enabled=True,
            section_strategies={"COMPLETION_GUIDANCE": []},
        )
        assert s.get_strategies_for_section("COMPLETION_GUIDANCE") == []
        assert s.get_strategies_for_section("GROUNDING_RULES") == ["gepa"]

    def test_get_strategies_empty_when_disabled(self):
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        s = PromptOptimizationSettings(enabled=False)
        assert s.get_strategies_for_section("GROUNDING_RULES") == []

    def test_is_strategy_active(self):
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        s = PromptOptimizationSettings(enabled=True)
        assert s.is_strategy_active("gepa")
        assert not s.is_strategy_active("nonexistent")

    def test_nested_gepa_config(self):
        from victor.config.prompt_optimization_settings import (
            PromptOptimizationSettings,
        )

        s = PromptOptimizationSettings(enabled=True)
        assert s.gepa.max_prompt_chars == 1500
        assert s.gepa.default_tier == "balanced"
