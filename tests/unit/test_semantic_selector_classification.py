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

"""Tests for SemanticToolSelector + UnifiedTaskClassifier integration.

Tests cover:
- Task type to tool category mapping
- Negation-aware tool exclusion
- Confidence-based threshold adjustment
- Classification-aware tool selection
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np

from victor.tools.semantic_selector import SemanticToolSelector


class TestTaskTypeCategoryMapping:
    """Tests for task type to category mapping."""

    @pytest.fixture
    def selector(self):
        """Create selector without initializing embeddings."""
        return SemanticToolSelector(cache_embeddings=False)

    def test_analysis_task_categories(self, selector):
        """Test that analysis tasks map to analysis categories."""
        tools = selector._get_tools_for_task_type("analysis")
        # Should include code intelligence and analysis tools
        assert len(tools) > 0

    def test_action_task_categories(self, selector):
        """Test that action tasks map to execution categories."""
        tools = selector._get_tools_for_task_type("action")
        # Should include execution-related tools
        assert len(tools) > 0

    def test_edit_task_categories(self, selector):
        """Test that edit tasks map to file and refactoring categories."""
        tools = selector._get_tools_for_task_type("edit")
        assert len(tools) > 0

    def test_search_task_categories(self, selector):
        """Test that search tasks map to code intel categories."""
        tools = selector._get_tools_for_task_type("search")
        assert len(tools) > 0

    def test_default_task_categories(self, selector):
        """Test that default tasks get basic tools."""
        tools = selector._get_tools_for_task_type("default")
        assert len(tools) > 0

    def test_unknown_task_type_fallback(self, selector):
        """Test that unknown task types get fallback categories."""
        tools = selector._get_tools_for_task_type("unknown_type")
        # Should fall back to file_ops + execution
        assert len(tools) > 0


class TestNegationAwareExclusion:
    """Tests for negation-aware tool exclusion."""

    @pytest.fixture
    def selector(self):
        return SemanticToolSelector(cache_embeddings=False)

    def test_analyze_negation_excludes_analysis_tools(self, selector):
        """Test that negated 'analyze' excludes analysis tools."""
        mock_match = MagicMock()
        mock_match.keyword = "analyze"

        excluded = selector._get_excluded_tools_from_negations([mock_match])

        assert "metrics" in excluded or "review" in excluded or "docs_coverage" in excluded

    def test_test_negation_excludes_test_tools(self, selector):
        """Test that negated 'test' excludes test tools."""
        mock_match = MagicMock()
        mock_match.keyword = "test"

        excluded = selector._get_excluded_tools_from_negations([mock_match])

        assert "test" in excluded

    def test_search_negation_excludes_search_tools(self, selector):
        """Test that negated 'search' excludes search tools."""
        mock_match = MagicMock()
        mock_match.keyword = "search"

        excluded = selector._get_excluded_tools_from_negations([mock_match])

        assert "search" in excluded or "grep" in excluded

    def test_multiple_negations(self, selector):
        """Test multiple negated keywords."""
        matches = [MagicMock(keyword="analyze"), MagicMock(keyword="test")]

        excluded = selector._get_excluded_tools_from_negations(matches)

        # Should exclude tools for both keywords
        assert len(excluded) >= 2

    def test_empty_negations_no_exclusions(self, selector):
        """Test that empty negations produce no exclusions."""
        excluded = selector._get_excluded_tools_from_negations([])
        assert len(excluded) == 0

    def test_unknown_keyword_no_exclusion(self, selector):
        """Test that unknown keywords don't cause exclusions."""
        mock_match = MagicMock()
        mock_match.keyword = "unknown_keyword_xyz"

        excluded = selector._get_excluded_tools_from_negations([mock_match])

        assert len(excluded) == 0


class TestConfidenceThresholdAdjustment:
    """Tests for confidence-based threshold adjustment."""

    @pytest.fixture
    def selector(self):
        return SemanticToolSelector(cache_embeddings=False)

    def test_high_confidence_raises_threshold(self, selector):
        """Test that high confidence raises the threshold."""
        base = 0.15
        adjusted = selector._adjust_threshold_by_confidence(base, 0.9)

        assert adjusted > base

    def test_low_confidence_lowers_threshold(self, selector):
        """Test that low confidence lowers the threshold."""
        base = 0.15
        adjusted = selector._adjust_threshold_by_confidence(base, 0.2)

        assert adjusted < base

    def test_medium_confidence_minimal_change(self, selector):
        """Test that medium confidence has minimal change."""
        base = 0.15
        adjusted = selector._adjust_threshold_by_confidence(base, 0.5)

        # Should be very close to base
        assert abs(adjusted - base) < 0.02

    def test_threshold_bounded_min(self, selector):
        """Test that threshold doesn't go below minimum."""
        adjusted = selector._adjust_threshold_by_confidence(0.05, 0.1)

        assert adjusted >= 0.1  # Minimum bound

    def test_threshold_bounded_max(self, selector):
        """Test that threshold doesn't go above maximum."""
        adjusted = selector._adjust_threshold_by_confidence(0.25, 1.0)

        assert adjusted <= 0.3  # Maximum bound


class TestClassificationAwareSelection:
    """Tests for the full classification-aware tool selection."""

    @pytest.fixture
    def selector(self):
        selector = SemanticToolSelector(cache_embeddings=False)
        # Pre-fill embedding cache with dummy embeddings (using new short names)
        selector._tool_embedding_cache = {
            "read": np.random.randn(384).astype(np.float32),
            "write": np.random.randn(384).astype(np.float32),
            "shell": np.random.randn(384).astype(np.float32),
            "search": np.random.randn(384).astype(np.float32),
            "docs_coverage": np.random.randn(384).astype(np.float32),
            "test": np.random.randn(384).astype(np.float32),
        }
        return selector

    @pytest.fixture
    def mock_tools(self):
        """Create mock tool registry."""
        tools = MagicMock()

        mock_tool_list = []
        for name in [
            "read",
            "write",
            "shell",
            "search",
            "docs_coverage",
            "test",
        ]:
            tool = MagicMock()
            tool.name = name
            tool.description = f"Mock {name} tool"
            tool.parameters = {"properties": {}}
            mock_tool_list.append(tool)

        tools.list_tools.return_value = mock_tool_list
        tools.get.side_effect = lambda n: next((t for t in mock_tool_list if t.name == n), None)
        tools.is_tool_enabled.return_value = True
        tools.get_tool_cost.return_value = None

        return tools

    @pytest.fixture
    def mock_classification_result(self):
        """Create mock classification result."""
        from victor.agent.unified_classifier import TaskType, ClassificationResult

        return ClassificationResult(
            task_type=TaskType.ANALYSIS,
            confidence=0.8,
            is_analysis_task=True,
            is_action_task=False,
            negated_keywords=[],
            matched_keywords=[],
        )

    @pytest.mark.asyncio
    async def test_analysis_task_selects_analysis_tools(
        self, selector, mock_tools, mock_classification_result
    ):
        """Test that analysis classification selects analysis-related tools."""
        with patch.object(selector, "_get_embedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = np.random.randn(384).astype(np.float32)

            tools = await selector.select_tools_with_classification(
                "Analyze the codebase",
                mock_tools,
                mock_classification_result,
            )

            assert len(tools) > 0

    @pytest.mark.asyncio
    async def test_negation_excludes_tools(self, selector, mock_tools):
        """Test that negated keywords exclude related tools."""
        from victor.agent.unified_classifier import TaskType, ClassificationResult, KeywordMatch

        # Create classification with negated "analyze"
        result = ClassificationResult(
            task_type=TaskType.ACTION,
            confidence=0.7,
            is_action_task=True,
            negated_keywords=[KeywordMatch("analyze", "analysis", 0, True)],
            matched_keywords=[],
        )

        with patch.object(selector, "_get_embedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = np.random.randn(384).astype(np.float32)

            tools = await selector.select_tools_with_classification(
                "Don't analyze, just run the tests",
                mock_tools,
                result,
            )

            tool_names = [t.name for t in tools]
            # docs_coverage (analysis tool) should be excluded
            assert "docs_coverage" not in tool_names

    @pytest.mark.asyncio
    async def test_high_confidence_stricter_selection(self, selector, mock_tools):
        """Test that high confidence leads to stricter selection."""
        from victor.agent.unified_classifier import TaskType, ClassificationResult

        result = ClassificationResult(
            task_type=TaskType.ANALYSIS,
            confidence=0.95,
            is_analysis_task=True,
            negated_keywords=[],
            matched_keywords=[],
        )

        with patch.object(selector, "_get_embedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = np.random.randn(384).astype(np.float32)

            tools = await selector.select_tools_with_classification(
                "Analyze",
                mock_tools,
                result,
            )

            # Should still return tools (with stricter threshold)
            assert len(tools) >= 2  # Fallback ensures minimum

    @pytest.mark.asyncio
    async def test_low_confidence_broader_selection(self, selector, mock_tools):
        """Test that low confidence leads to broader selection."""
        from victor.agent.unified_classifier import TaskType, ClassificationResult

        result = ClassificationResult(
            task_type=TaskType.DEFAULT,
            confidence=0.2,
            negated_keywords=[],
            matched_keywords=[],
        )

        with patch.object(selector, "_get_embedding", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = np.random.randn(384).astype(np.float32)

            tools = await selector.select_tools_with_classification(
                "Something vague",
                mock_tools,
                result,
            )

            # Should return tools with looser threshold
            assert len(tools) >= 2


class TestClassificationToolStats:
    """Tests for classification tool statistics."""

    def test_get_stats_returns_expected_keys(self):
        """Test that get_classification_tool_stats returns expected keys."""
        selector = SemanticToolSelector(cache_embeddings=False)

        stats = selector.get_classification_tool_stats()

        assert "task_type_categories" in stats
        assert "keyword_tool_mappings" in stats
        assert "usage_cache_size" in stats
        assert "embedding_cache_size" in stats

    def test_stats_values_are_numeric(self):
        """Test that stats values are numeric."""
        selector = SemanticToolSelector(cache_embeddings=False)

        stats = selector.get_classification_tool_stats()

        assert isinstance(stats["task_type_categories"], int)
        assert isinstance(stats["keyword_tool_mappings"], int)
        assert stats["task_type_categories"] > 0
        assert stats["keyword_tool_mappings"] > 0
