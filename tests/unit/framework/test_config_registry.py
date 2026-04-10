# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Test VerticalConfigRegistry."""

import pytest

from victor.core.verticals.config_registry import VerticalConfigRegistry


class TestVerticalConfigRegistry:
    """Test configuration registry functionality."""

    def test_get_coding_provider_hints(self):
        """Test getting coding provider hints."""
        hints = VerticalConfigRegistry.get_provider_hints("coding")
        assert "preferred_providers" in hints
        assert hints["min_context_window"] == 100000
        assert hints["requires_tool_calling"] is True
        assert "claude-sonnet-4-20250514" in hints["preferred_models"]

    def test_get_research_provider_hints(self):
        """Test getting research provider hints."""
        hints = VerticalConfigRegistry.get_provider_hints("research")
        assert "preferred_providers" in hints
        assert hints["min_context_window"] == 100000
        assert "web_search" in hints["features"]

    def test_get_devops_provider_hints(self):
        """Test getting devops provider hints."""
        hints = VerticalConfigRegistry.get_provider_hints("devops")
        assert "preferred_providers" in hints
        assert hints["min_context_window"] == 100000
        assert hints["requires_tool_calling"] is True
        assert "large_context" in hints["features"]

    def test_get_data_analysis_provider_hints(self):
        """Test getting data_analysis provider hints."""
        hints = VerticalConfigRegistry.get_provider_hints("data_analysis")
        assert "preferred_providers" in hints
        assert hints["min_context_window"] == 128000
        assert "code_execution" in hints["features"]

    def test_get_rag_provider_hints(self):
        """Test getting rag provider hints."""
        hints = VerticalConfigRegistry.get_provider_hints("rag")
        assert "preferred_providers" in hints
        assert hints["min_context_window"] == 8000
        assert hints.get("temperature") == 0.3

    def test_get_coding_evaluation_criteria(self):
        """Test getting coding evaluation criteria."""
        criteria = VerticalConfigRegistry.get_evaluation_criteria("coding")
        assert "Code correctness and functionality" in criteria
        assert "Test coverage and validation" in criteria

    def test_get_research_evaluation_criteria(self):
        """Test getting research evaluation criteria."""
        criteria = VerticalConfigRegistry.get_evaluation_criteria("research")
        assert "accuracy" in criteria
        assert "source_quality" in criteria

    def test_get_devops_evaluation_criteria(self):
        """Test getting devops evaluation criteria."""
        criteria = VerticalConfigRegistry.get_evaluation_criteria("devops")
        assert "configuration_correctness" in criteria
        assert "security_best_practices" in criteria

    def test_get_data_analysis_evaluation_criteria(self):
        """Test getting data_analysis evaluation criteria."""
        criteria = VerticalConfigRegistry.get_evaluation_criteria("data_analysis")
        assert "statistical_correctness" in criteria
        assert "visualization_quality" in criteria

    def test_get_rag_evaluation_criteria(self):
        """Test getting rag evaluation criteria."""
        criteria = VerticalConfigRegistry.get_evaluation_criteria("rag")
        assert "Answer is grounded in retrieved documents" in criteria
        assert "Sources are properly cited" in criteria

    def test_fallback_to_default_provider_hints(self):
        """Test fallback to default for unknown verticals."""
        hints = VerticalConfigRegistry.get_provider_hints("unknown_vertical")
        assert hints["min_context_window"] >= 8000
        assert "preferred_providers" in hints
        assert hints["requires_tool_calling"] is True

    def test_fallback_to_default_evaluation_criteria(self):
        """Test fallback to default for unknown verticals."""
        criteria = VerticalConfigRegistry.get_evaluation_criteria("unknown_vertical")
        assert "Task completion accuracy" in criteria

    def test_register_custom_provider_hints(self):
        """Test registering custom provider hints."""
        VerticalConfigRegistry.register_provider_hints(
            "custom", {"custom_key": "custom_value", "test": True}
        )
        hints = VerticalConfigRegistry.get_provider_hints("custom")
        assert hints["custom_key"] == "custom_value"
        assert hints["test"] is True

    def test_register_custom_evaluation_criteria(self):
        """Test registering custom evaluation criteria."""
        custom_criteria = ["Custom criterion 1", "Custom criterion 2"]
        VerticalConfigRegistry.register_evaluation_criteria("custom_eval", custom_criteria)
        criteria = VerticalConfigRegistry.get_evaluation_criteria("custom_eval")
        assert criteria == custom_criteria

    def test_returns_copy_to_prevent_mutation(self):
        """Test that registry returns copies to prevent accidental mutation."""
        hints1 = VerticalConfigRegistry.get_provider_hints("coding")
        hints2 = VerticalConfigRegistry.get_provider_hints("coding")
        hints1["custom_key"] = "custom_value"
        assert "custom_key" not in hints2
        assert "custom_key" not in VerticalConfigRegistry._provider_hints["coding"]

    def test_evaluation_criteria_returns_copy(self):
        """Test that evaluation criteria returns copies to prevent accidental mutation."""
        criteria1 = VerticalConfigRegistry.get_evaluation_criteria("coding")
        criteria2 = VerticalConfigRegistry.get_evaluation_criteria("coding")
        criteria1.append("Custom criterion")
        assert "Custom criterion" not in criteria2
        assert "Custom criterion" not in VerticalConfigRegistry._evaluation_criteria["coding"]
