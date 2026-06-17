"""Tests for resource-aware exploration budget.

Covers:
- calculate_exploration_budget returns correct values per provider
- Local providers (ollama) get max_parallel_agents=1
- Cloud providers get CPU-scaled parallelism
- Complexity scaling (simple=0, complex=full)
- ExplorationCoordinator uses ResourceBudget for parallelism
- GPU/memory detection in budget calculation
"""

from __future__ import annotations

from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path

import pytest


class TestCalculateExplorationBudget:
    """calculate_exploration_budget returns provider-aware budgets."""

    def test_ollama_gets_one_agent(self):
        from victor.agent.budget.resource_calculator import calculate_exploration_budget

        budget = calculate_exploration_budget(complexity="complex", provider="ollama")
        assert budget.max_parallel_agents == 1

    def test_cloud_provider_gets_multiple_agents(self):
        from victor.agent.budget.resource_calculator import calculate_exploration_budget

        budget = calculate_exploration_budget(complexity="complex", provider="anthropic")
        assert budget.max_parallel_agents >= 1
        assert budget.max_parallel_agents <= 3

    def test_simple_complexity_gets_zero_agents(self):
        from victor.agent.budget.resource_calculator import calculate_exploration_budget

        budget = calculate_exploration_budget(complexity="simple", provider="anthropic")
        assert budget.max_parallel_agents == 0
        assert budget.tool_budget_per_agent == 0

    def test_timeout_scales_with_provider_speed(self):
        from victor.agent.budget.resource_calculator import calculate_exploration_budget

        cloud_budget = calculate_exploration_budget(complexity="complex", provider="anthropic")
        local_budget = calculate_exploration_budget(complexity="complex", provider="ollama")
        # Ollama should have longer timeout (2x speed multiplier)
        assert local_budget.exploration_timeout > cloud_budget.exploration_timeout

    def test_analysis_gets_highest_budget(self):
        from victor.agent.budget.resource_calculator import calculate_exploration_budget

        analysis = calculate_exploration_budget(complexity="analysis", provider="anthropic")
        action = calculate_exploration_budget(complexity="action", provider="anthropic")
        assert analysis.exploration_timeout >= action.exploration_timeout


class TestExplorationCoordinatorUsesBudget:
    """ExplorationCoordinator.explore_parallel uses ResourceBudget."""

    @pytest.mark.asyncio
    async def test_respects_max_parallel_agents(self):
        from victor.agent.services.exploration_runtime import (
            ExplorationCoordinator,
        )

        coordinator = ExplorationCoordinator()

        # Mock budget with 1 agent (local provider)
        mock_budget = MagicMock()
        mock_budget.max_parallel_agents = 1
        mock_budget.exploration_timeout = 30
        mock_budget.tool_budget_per_agent = 5

        with patch(
            "victor.agent.budget.resource_calculator.calculate_exploration_budget",
            return_value=mock_budget,
        ):
            with patch.object(
                coordinator, "_search_codebase", new_callable=AsyncMock
            ) as mock_search:
                mock_search.return_value = {"files": [], "summary": ""}
                with patch.object(
                    coordinator, "_list_project_structure", new_callable=AsyncMock
                ) as mock_ls:
                    mock_ls.return_value = {"structure": ""}

                    result = await coordinator.explore_parallel(
                        "fix the auth bug",
                        Path("/tmp/test"),
                        provider="ollama",
                    )

                    # With budget.max_parallel_agents=1, should only run 1 search + 1 ls
                    assert mock_search.call_count <= 1


class TestResourceBudgetHardwareDetection:
    """Resource budget detects hardware capabilities."""

    def test_cpu_count_affects_cloud_parallelism(self):
        from victor.agent.budget.resource_calculator import calculate_exploration_budget

        with patch("os.cpu_count", return_value=8):
            budget = calculate_exploration_budget(complexity="complex", provider="anthropic")
            assert budget.max_parallel_agents >= 2

        with patch("os.cpu_count", return_value=2):
            budget = calculate_exploration_budget(complexity="complex", provider="anthropic")
            assert budget.max_parallel_agents == 1
