from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.services.tool_budget_runtime import BudgetManager, ToolBudgetRuntime


def test_budget_runtime_uses_local_budget_without_pipeline():
    manager = BudgetManager(max_budget=5)
    runtime = ToolBudgetRuntime(manager, lambda: None)

    runtime.consume(2)

    assert runtime.get_limit() == 5
    assert runtime.get_used() == 2
    assert runtime.get_remaining() == 3
    assert runtime.get_info() == {"max": 5, "used": 2, "remaining": 3}


def test_budget_runtime_prefers_pipeline_budget_and_usage():
    manager = BudgetManager(max_budget=5)
    pipeline = SimpleNamespace(tool_budget=9, calls_used=4)
    runtime = ToolBudgetRuntime(manager, lambda: pipeline)

    runtime.sync_from_runtime()

    assert runtime.get_limit() == 9
    assert runtime.get_used() == 4
    assert manager.max_budget == 9
    assert manager.calls_made == 4


def test_budget_runtime_sets_pipeline_budget_with_method():
    manager = BudgetManager(max_budget=5)
    pipeline = SimpleNamespace(set_tool_budget=MagicMock())
    runtime = ToolBudgetRuntime(manager, lambda: pipeline)

    runtime.set_limit(12)

    assert manager.max_budget == 12
    pipeline.set_tool_budget.assert_called_once_with(12)


def test_budget_runtime_sets_pipeline_config_budget_without_method():
    manager = BudgetManager(max_budget=5)
    pipeline = SimpleNamespace(config=SimpleNamespace(tool_budget=5))
    runtime = ToolBudgetRuntime(manager, lambda: pipeline)

    runtime.set_limit(12)

    assert manager.max_budget == 12
    assert pipeline.config.tool_budget == 12


def test_budget_runtime_consumes_pipeline_budget_with_method():
    manager = BudgetManager(max_budget=10)
    pipeline = SimpleNamespace(calls_used=3, consume_budget=MagicMock())
    runtime = ToolBudgetRuntime(manager, lambda: pipeline)

    runtime.consume(2)

    pipeline.consume_budget.assert_called_once_with(2)
    assert manager.calls_made == 2


def test_budget_runtime_consumes_pipeline_budget_without_method():
    manager = BudgetManager(max_budget=10)
    pipeline = SimpleNamespace(calls_used=3)
    runtime = ToolBudgetRuntime(manager, lambda: pipeline)

    runtime.consume(2)

    assert pipeline._calls_used == 5
    assert manager.calls_made == 2


def test_budget_runtime_start_new_turn_resets_pipeline_and_syncs():
    manager = BudgetManager(max_budget=10)
    manager.record_usage(7)
    pipeline = SimpleNamespace(tool_budget=10, calls_used=7, start_new_turn=MagicMock())
    pipeline.start_new_turn.side_effect = lambda: setattr(pipeline, "calls_used", 0)
    runtime = ToolBudgetRuntime(manager, lambda: pipeline)

    runtime.start_new_turn()

    pipeline.start_new_turn.assert_called_once()
    assert manager.max_budget == 10
    assert manager.calls_made == 0
    assert runtime.is_exhausted() is False


def test_budget_runtime_rejects_negative_values():
    runtime = ToolBudgetRuntime(BudgetManager(), lambda: None)

    with pytest.raises(ValueError, match="non-negative"):
        runtime.set_limit(-1)
    with pytest.raises(ValueError, match="negative budget"):
        runtime.consume(-1)
