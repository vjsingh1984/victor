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

"""Integration tests for prompt enrichment E2E.

Tests the full enrichment pipeline including:
- Enrichment service with vertical strategies
- Prompt builder integration
- RL tracking for enrichment outcomes
- Caching behavior
"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.prompt_enrichment import (
    PromptEnrichmentService,
    EnrichmentContext,
    ContextEnrichment,
    EnrichedPrompt,
    EnrichmentOutcome,
    EnrichmentType,
    EnrichmentPriority,
)
from victor.agent.prompt_builder import SystemPromptBuilder
from victor.agent.rl.learners.prompt_template import PromptTemplateLearner


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create temporary database for RL learner."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def rl_learner(temp_db):
    """Create RL learner with temporary database."""
    return PromptTemplateLearner(
        name="prompt_template",
        db_connection=temp_db,
        exploration_rate=0.0,  # Disable exploration for predictable tests
    )


@pytest.fixture
def enrichment_service():
    """Create enrichment service."""
    return PromptEnrichmentService(
        max_tokens=2000,
        timeout_ms=500,
        cache_enabled=True,
    )


@pytest.fixture
def mock_coding_strategy():
    """Mock coding enrichment strategy."""

    class MockCodingStrategy:
        async def get_enrichments(
            self, prompt: str, context: EnrichmentContext
        ) -> List[ContextEnrichment]:
            return [
                ContextEnrichment(
                    type=EnrichmentType.KNOWLEDGE_GRAPH,
                    content=f"Symbol: calculate_total found in utils.py:42",
                    priority=EnrichmentPriority.HIGH,
                    token_estimate=50,
                ),
                ContextEnrichment(
                    type=EnrichmentType.CODE_SNIPPET,
                    content="def calculate_total(items): return sum(items)",
                    priority=EnrichmentPriority.NORMAL,
                    token_estimate=30,
                ),
            ]

        def get_priority(self) -> int:
            return 50

        def get_token_allocation(self) -> float:
            return 0.4

    return MockCodingStrategy()


@pytest.fixture
def mock_research_strategy():
    """Mock research enrichment strategy."""

    class MockResearchStrategy:
        async def get_enrichments(
            self, prompt: str, context: EnrichmentContext
        ) -> List[ContextEnrichment]:
            return [
                ContextEnrichment(
                    type=EnrichmentType.WEB_SEARCH,
                    content="Search results: Python best practices 2025",
                    priority=EnrichmentPriority.HIGH,
                    token_estimate=100,
                ),
            ]

        def get_priority(self) -> int:
            return 50

        def get_token_allocation(self) -> float:
            return 0.35

    return MockResearchStrategy()


# =============================================================================
# E2E Tests: Full Pipeline
# =============================================================================


class TestEnrichmentPipeline:
    """Tests for the full enrichment pipeline."""

    @pytest.mark.asyncio
    async def test_enrichment_service_with_strategy(
        self, enrichment_service, mock_coding_strategy
    ):
        """Enrichment service applies strategy enrichments."""
        enrichment_service.register_strategy("coding", mock_coding_strategy)

        context = EnrichmentContext(
            task_type="edit",
            file_mentions=["src/utils.py"],
            symbol_mentions=["calculate_total"],
        )

        result = await enrichment_service.enrich(
            prompt="Fix the calculate_total function",
            vertical="coding",
            context=context,
        )

        assert result.enrichment_count == 2
        assert result.total_tokens_added == 80
        assert "calculate_total" in result.enriched_prompt
        assert "utils.py:42" in result.enriched_prompt

    @pytest.mark.asyncio
    async def test_multiple_verticals(
        self, enrichment_service, mock_coding_strategy, mock_research_strategy
    ):
        """Different verticals get different enrichments."""
        enrichment_service.register_strategy("coding", mock_coding_strategy)
        enrichment_service.register_strategy("research", mock_research_strategy)

        coding_context = EnrichmentContext(task_type="edit")
        research_context = EnrichmentContext(task_type="fact_check")

        # Coding enrichment
        coding_result = await enrichment_service.enrich(
            prompt="Fix the bug",
            vertical="coding",
            context=coding_context,
        )

        # Research enrichment
        research_result = await enrichment_service.enrich(
            prompt="What are Python best practices?",
            vertical="research",
            context=research_context,
        )

        # Different enrichment types
        assert coding_result.enrichment_count == 2
        assert research_result.enrichment_count == 1
        assert EnrichmentType.KNOWLEDGE_GRAPH.value in coding_result.enrichment_types
        assert EnrichmentType.WEB_SEARCH.value in research_result.enrichment_types


class TestPromptBuilderIntegration:
    """Tests for prompt builder integration with enrichment."""

    @pytest.mark.asyncio
    async def test_prompt_builder_with_enrichment(
        self, enrichment_service, mock_coding_strategy
    ):
        """Prompt builder can enrich user prompts."""
        enrichment_service.register_strategy("coding", mock_coding_strategy)

        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-3-opus",
            enrichment_service=enrichment_service,
            vertical="coding",
            task_type="edit",
        )

        context = EnrichmentContext(
            task_type="edit",
            symbol_mentions=["calculate_total"],
        )

        result = await builder.enrich_prompt(
            prompt="Update the calculate_total function",
            context=context,
        )

        assert result.enrichment_count == 2
        assert "calculate_total" in result.enriched_prompt

    @pytest.mark.asyncio
    async def test_prompt_builder_without_enrichment(self):
        """Prompt builder works without enrichment service."""
        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-3-opus",
            vertical="coding",
        )

        result = await builder.enrich_prompt(prompt="Do something")

        assert result.enriched_prompt == "Do something"
        assert result.enrichment_count == 0


class TestRLIntegration:
    """Tests for RL learner integration with enrichment."""

    def test_rl_learner_records_enrichment_outcome(self, rl_learner):
        """RL learner records enrichment outcomes."""
        # Record some outcomes
        rl_learner.record_enrichment_outcome(
            vertical="coding",
            enrichment_type="knowledge_graph",
            enrichment_count=2,
            task_success=True,
            quality_improvement=0.2,
            task_type="edit",
        )

        rl_learner.record_enrichment_outcome(
            vertical="coding",
            enrichment_type="knowledge_graph",
            enrichment_count=1,
            task_success=False,
            quality_improvement=-0.1,
            task_type="edit",
        )

        # Check probabilities were updated
        probs = rl_learner.get_enrichment_probabilities("coding")
        assert "knowledge_graph" in probs
        # After 1 success and 1 failure, probability should be around 0.5
        assert 0.3 < probs["knowledge_graph"] < 0.7

    def test_rl_learner_enrichment_callback(self, rl_learner, enrichment_service):
        """RL learner callback integrates with enrichment service."""
        callback = rl_learner.create_enrichment_callback()
        enrichment_service.on_outcome(callback)

        # Record outcome through service
        outcome = EnrichmentOutcome(
            enrichment_type="web_search",
            enrichment_count=3,
            task_success=True,
            quality_improvement=0.3,
            vertical="research",
            task_type="fact_check",
        )

        enrichment_service.record_outcome(outcome)

        # Verify RL learner received it
        probs = rl_learner.get_enrichment_probabilities("research")
        assert "web_search" in probs
        assert probs["web_search"] > 0.5  # Success should increase probability

    def test_rl_learner_enrichment_recommendation(self, rl_learner):
        """RL learner provides enrichment recommendations."""
        # Train with some positive outcomes
        for _ in range(5):
            rl_learner.record_enrichment_outcome(
                vertical="coding",
                enrichment_type="knowledge_graph",
                enrichment_count=2,
                task_success=True,
                quality_improvement=0.2,
            )

        # Get recommendation
        recommendations = rl_learner.get_enrichment_recommendation("coding")

        assert isinstance(recommendations, dict)
        assert "knowledge_graph" in recommendations
        # After positive training, should recommend using it
        assert recommendations["knowledge_graph"] is True

    def test_rl_metrics_include_enrichment(self, rl_learner):
        """RL metrics export includes enrichment stats."""
        rl_learner.record_enrichment_outcome(
            vertical="devops",
            enrichment_type="project_context",
            enrichment_count=1,
            task_success=True,
            quality_improvement=0.1,
        )

        metrics = rl_learner.export_metrics()

        assert "enrichment_posteriors_count" in metrics
        assert "enrichment_samples_total" in metrics
        assert "enrichment_stats" in metrics
        assert metrics["enrichment_samples_total"] == 1


class TestCachingBehavior:
    """Tests for enrichment caching behavior."""

    @pytest.mark.asyncio
    async def test_cache_hit_on_same_request(
        self, enrichment_service, mock_coding_strategy
    ):
        """Same request uses cached result."""
        call_count = 0

        class CountingStrategy:
            async def get_enrichments(self, prompt, context):
                nonlocal call_count
                call_count += 1
                return [
                    ContextEnrichment(
                        type=EnrichmentType.KNOWLEDGE_GRAPH,
                        content="Test content",
                        token_estimate=50,
                    )
                ]

            def get_priority(self):
                return 50

            def get_token_allocation(self):
                return 0.4

        enrichment_service.register_strategy("coding", CountingStrategy())

        context = EnrichmentContext(task_type="edit")

        # First request
        result1 = await enrichment_service.enrich(
            prompt="Test prompt",
            vertical="coding",
            context=context,
        )

        # Second request (should use cache)
        result2 = await enrichment_service.enrich(
            prompt="Test prompt",
            vertical="coding",
            context=context,
        )

        # Strategy should only be called once
        assert call_count == 1
        assert result1.from_cache is False
        assert result2.from_cache is True

    @pytest.mark.asyncio
    async def test_cache_miss_on_different_context(
        self, enrichment_service, mock_coding_strategy
    ):
        """Different context causes cache miss."""
        enrichment_service.register_strategy("coding", mock_coding_strategy)

        context1 = EnrichmentContext(task_type="edit", file_mentions=["file1.py"])
        context2 = EnrichmentContext(task_type="edit", file_mentions=["file2.py"])

        result1 = await enrichment_service.enrich(
            prompt="Test",
            vertical="coding",
            context=context1,
        )

        result2 = await enrichment_service.enrich(
            prompt="Test",
            vertical="coding",
            context=context2,
        )

        # Both should be fresh (different contexts)
        assert result1.from_cache is False
        assert result2.from_cache is False


class TestErrorHandling:
    """Tests for error handling in the enrichment pipeline."""

    @pytest.mark.asyncio
    async def test_strategy_error_returns_original(self, enrichment_service):
        """Strategy error returns original prompt gracefully."""

        class FailingStrategy:
            async def get_enrichments(self, prompt, context):
                raise RuntimeError("Strategy failed")

            def get_priority(self):
                return 50

            def get_token_allocation(self):
                return 0.4

        enrichment_service.register_strategy("coding", FailingStrategy())

        context = EnrichmentContext(task_type="edit")

        result = await enrichment_service.enrich(
            prompt="Test prompt",
            vertical="coding",
            context=context,
        )

        # Should return original prompt on error
        assert result.enriched_prompt == "Test prompt"
        assert result.enrichment_count == 0

    @pytest.mark.asyncio
    async def test_missing_vertical_returns_original(self, enrichment_service):
        """Missing vertical returns original prompt."""
        context = EnrichmentContext(task_type="edit")

        result = await enrichment_service.enrich(
            prompt="Test prompt",
            vertical="unknown_vertical",
            context=context,
        )

        assert result.enriched_prompt == "Test prompt"
        assert result.enrichment_count == 0


class TestTokenBudget:
    """Tests for token budget enforcement."""

    @pytest.mark.asyncio
    async def test_exceeding_budget_drops_enrichments(self):
        """Enrichments exceeding budget are dropped."""
        service = PromptEnrichmentService(max_tokens=60)

        class LargeEnrichmentStrategy:
            async def get_enrichments(self, prompt, context):
                return [
                    ContextEnrichment(
                        type=EnrichmentType.KNOWLEDGE_GRAPH,
                        content="A" * 200,  # ~50 tokens
                        priority=EnrichmentPriority.HIGH,
                        token_estimate=50,
                    ),
                    ContextEnrichment(
                        type=EnrichmentType.CODE_SNIPPET,
                        content="B" * 200,  # ~50 tokens - should be dropped
                        priority=EnrichmentPriority.NORMAL,
                        token_estimate=50,
                    ),
                ]

            def get_priority(self):
                return 50

            def get_token_allocation(self):
                return 0.4

        service.register_strategy("coding", LargeEnrichmentStrategy())

        context = EnrichmentContext(task_type="edit")

        result = await service.enrich(
            prompt="Test",
            vertical="coding",
            context=context,
        )

        # Only first enrichment should fit within budget
        assert result.enrichment_count == 1
        assert result.total_tokens_added == 50


# =============================================================================
# Vertical Strategy Tests
# =============================================================================


class TestVerticalStrategies:
    """Tests for vertical-specific enrichment strategies."""

    @pytest.mark.asyncio
    async def test_coding_strategy_integration(self):
        """Coding strategy integrates with enrichment service."""
        from victor.verticals.coding.enrichment import CodingEnrichmentStrategy

        strategy = CodingEnrichmentStrategy(graph_store=None)
        service = PromptEnrichmentService()
        service.register_strategy("coding", strategy)

        context = EnrichmentContext(
            task_type="edit",
            file_mentions=["src/main.py"],
        )

        # Without graph store, should return empty enrichments
        result = await service.enrich(
            prompt="Fix the bug in main.py",
            vertical="coding",
            context=context,
        )

        # No enrichments without graph store, but no errors
        assert result.enriched_prompt is not None

    @pytest.mark.asyncio
    async def test_devops_strategy_integration(self):
        """DevOps strategy integrates with enrichment service."""
        from victor.verticals.devops.enrichment import DevOpsEnrichmentStrategy

        strategy = DevOpsEnrichmentStrategy()
        service = PromptEnrichmentService()
        service.register_strategy("devops", strategy)

        context = EnrichmentContext(
            task_type="infrastructure",
            file_mentions=["Dockerfile"],
        )

        result = await service.enrich(
            prompt="Optimize the Docker build",
            vertical="devops",
            context=context,
        )

        # Should get Docker best practices
        assert result.enrichment_count >= 1
        assert "Docker" in result.enriched_prompt

    @pytest.mark.asyncio
    async def test_data_analysis_strategy_integration(self):
        """Data Analysis strategy integrates with enrichment service."""
        from victor.verticals.data_analysis.enrichment import (
            DataAnalysisEnrichmentStrategy,
        )

        strategy = DataAnalysisEnrichmentStrategy()
        service = PromptEnrichmentService()
        service.register_strategy("data_analysis", strategy)

        context = EnrichmentContext(task_type="data_profiling")

        result = await service.enrich(
            prompt="Analyze the correlation between price and sales",
            vertical="data_analysis",
            context=context,
        )

        # Should get correlation analysis guidance
        assert result.enrichment_count >= 1
        assert "correlation" in result.enriched_prompt.lower()

    @pytest.mark.asyncio
    async def test_research_strategy_integration(self):
        """Research strategy integrates with enrichment service."""
        from victor.verticals.research.enrichment import ResearchEnrichmentStrategy

        strategy = ResearchEnrichmentStrategy(web_search_fn=None)
        service = PromptEnrichmentService()
        service.register_strategy("research", strategy)

        context = EnrichmentContext(task_type="fact_check")

        # Without web search, should return empty enrichments
        result = await service.enrich(
            prompt='Is "Python 4.0" released?',
            vertical="research",
            context=context,
        )

        # No enrichments without web search, but no errors
        assert result.enriched_prompt is not None
