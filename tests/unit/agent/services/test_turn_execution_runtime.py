"""Focused tests for TurnExecutor runtime behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.services.turn_execution_runtime import TurnExecutor
from victor.framework.task.protocols import TaskComplexity


def _make_executor(exploration_coordinator=None) -> TurnExecutor:
    chat_context = MagicMock()
    chat_context.settings = MagicMock()
    chat_context.add_message = MagicMock()
    chat_context.conversation = MagicMock()

    tool_context = MagicMock()
    provider_context = MagicMock()
    provider_context.provider_name = "ollama"
    provider_context.model = "test-model"
    execution_provider = MagicMock()

    return TurnExecutor(
        chat_context=chat_context,
        tool_context=tool_context,
        provider_context=provider_context,
        execution_provider=execution_provider,
        exploration_coordinator=exploration_coordinator,
    )


@pytest.mark.asyncio
async def test_parallel_exploration_uses_injected_coordinator():
    explorer = MagicMock()
    explorer.explore_parallel = AsyncMock(
        return_value=SimpleNamespace(
            summary="found relevant files",
            file_paths=["victor/agent/services/turn_execution_runtime.py"],
            tool_calls=2,
            duration_seconds=0.4,
        )
    )
    executor = _make_executor(exploration_coordinator=explorer)
    task_classification = SimpleNamespace(complexity=TaskComplexity.COMPLEX)

    with (
        patch(
            "victor.config.settings.load_settings",
            return_value=SimpleNamespace(
                pipeline=SimpleNamespace(parallel_exploration=True)
            ),
        ),
        patch(
            "victor.config.settings.get_project_paths",
            return_value=SimpleNamespace(project_root="/tmp/project"),
        ),
        patch(
            "victor.agent.budget.resource_calculator.calculate_exploration_budget",
            return_value=SimpleNamespace(
                max_parallel_agents=2,
                tool_budget_per_agent=3,
                exploration_timeout=5,
            ),
        ),
    ):
        await executor._run_parallel_exploration(
            "inspect the failing runtime path",
            task_classification,
        )

    explorer.explore_parallel.assert_awaited_once()
    executor._chat_context.add_message.assert_called_once()
    assert executor._exploration_done is True


@pytest.mark.asyncio
async def test_parallel_exploration_lazily_materializes_shared_coordinator():
    explorer = MagicMock()
    explorer.explore_parallel = AsyncMock(
        return_value=SimpleNamespace(
            summary="shared helper exploration",
            file_paths=["victor/agent/coordinators/factory_support.py"],
            tool_calls=1,
            duration_seconds=0.2,
        )
    )
    executor = _make_executor()
    task_classification = SimpleNamespace(complexity=TaskComplexity.COMPLEX)

    with (
        patch(
            "victor.config.settings.load_settings",
            return_value=SimpleNamespace(
                pipeline=SimpleNamespace(parallel_exploration=True)
            ),
        ),
        patch(
            "victor.config.settings.get_project_paths",
            return_value=SimpleNamespace(project_root="/tmp/project"),
        ),
        patch(
            "victor.agent.budget.resource_calculator.calculate_exploration_budget",
            return_value=SimpleNamespace(
                max_parallel_agents=1,
                tool_budget_per_agent=2,
                exploration_timeout=5,
            ),
        ),
        patch(
            "victor.agent.coordinators.factory_support.create_exploration_coordinator",
            return_value=explorer,
        ) as create_explorer,
    ):
        await executor._run_parallel_exploration(
            "trace the lazy helper path",
            task_classification,
        )

    create_explorer.assert_called_once_with()
    explorer.explore_parallel.assert_awaited_once()
    assert executor._exploration_coordinator is explorer
