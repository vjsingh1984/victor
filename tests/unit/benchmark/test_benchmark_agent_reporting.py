from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.benchmark.agent import BenchmarkAgent, BenchmarkAgentConfig
from victor.evaluation.protocol import BenchmarkTask, BenchmarkType
from victor.framework.task import TaskResult


class _FakeFrameworkAgent:
    def __init__(self, result: TaskResult, orchestrator: MagicMock):
        self.run = AsyncMock(return_value=result)
        self.subscribe_to_events = MagicMock(return_value=lambda: None)
        self.get_orchestrator = MagicMock(return_value=orchestrator)


@pytest.mark.asyncio
async def test_execute_task_captures_task_report_metrics() -> None:
    orchestrator = MagicMock()
    orchestrator.get_last_task_report.return_value = {
        "task_id": "bench-1",
        "api_prompt_tokens": 40,
        "api_completion_tokens": 18,
        "api_total_tokens": 58,
        "request_count": 3,
        "cache_read_tokens": 12,
        "cache_hit_rate": 0.25,
        "tool_schema_tokens": 144,
        "compaction_saved_tokens": 33,
        "compaction_messages_removed": 2,
        "total_cost_usd": 0.0021,
        "metadata": {
            "continuation_ledger": "Intent: fix regression; Plan: inspect tests; patch parser",
        },
    }
    framework_result = TaskResult(
        content="patched",
        success=True,
        tool_calls=[{"name": "read"}],
        metadata={
            "tokens_input": 40,
            "tokens_output": 18,
            "turns": 3,
            "cached_tokens": 12,
            "reasoning_tokens": 5,
            "cost_usd_micros": 2100,
            "cache_hit_rate": 0.25,
            "tool_schema_tokens": 144,
            "compaction_saved_tokens": 33,
            "compaction_messages_removed": 2,
            "task_report": orchestrator.get_last_task_report.return_value,
        },
    )
    agent = BenchmarkAgent(
        _FakeFrameworkAgent(framework_result, orchestrator),
        BenchmarkAgentConfig(),
    )
    task = BenchmarkTask(
        task_id="bench-1",
        benchmark=BenchmarkType.CUSTOM,
        description="Fix the regression",
        prompt="fix the parser regression",
    )

    trace = await agent.execute_task(task)

    assert trace.tokens_input == 40
    assert trace.tokens_output == 18
    assert trace.cached_tokens == 12
    assert trace.reasoning_tokens == 5
    assert trace.cost_usd_micros == 2100
    assert trace.cache_hit_rate == pytest.approx(0.25)
    assert trace.tool_schema_tokens == 144
    assert trace.compaction_saved_tokens == 33
    assert trace.compaction_messages_removed == 2
    assert trace.task_report["metadata"]["continuation_ledger"].startswith("Intent:")
    assert trace.build_result_metadata()["task_report"]["task_id"] == "bench-1"
