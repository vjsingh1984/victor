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

    def test_no_classification_unchanged_output(self):
        builder_with = _make_builder(None)
        builder_without = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
        )
        # Both should produce the same output when no classification
        assert builder_with.build() == builder_without.build()

    def test_tool_constraint_lists_available_tools(self):
        builder = _make_builder(available_tools=["read_file", "write_file", "shell"])
        prompt = builder.build()
        assert "read_file" in prompt
        assert "write_file" in prompt
        assert "shell" in prompt

    def test_prompt_length_varies_by_type(self):
        quick = _make_builder(QueryType.QUICK_QUESTION).build()
        explore = _make_builder(QueryType.EXPLORATION).build()
        # Exploration guidance is longer than quick question guidance
        assert len(explore) > len(quick)
