from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.services.tool_batch_executor import execute_tool_call_batch


class FakeBatchHost:
    def __init__(
        self,
        *,
        enable_parallel_execution: bool = True,
        remaining_budget: int = 100,
        invalid_calls: list[dict] | None = None,
        fail_tools: set[str] | None = None,
        delay: float = 0.0,
    ) -> None:
        self._config = SimpleNamespace(enable_parallel_execution=enable_parallel_execution)
        self._logger = MagicMock()
        self.remaining_budget = remaining_budget
        self.invalid_calls = invalid_calls or []
        self.fail_tools = fail_tools or set()
        self.delay = delay
        self.executed: list[str] = []
        self.started: list[str] = []

    def validate_tool_calls(self, tool_calls):
        invalid_ids = {id(call) for call in self.invalid_calls}
        valid = [call for call in tool_calls if id(call) not in invalid_ids]
        return valid, self.invalid_calls

    def get_remaining_budget(self):
        return self.remaining_budget

    async def execute_tool_call(self, tool_call, validate=True, check_budget=True):
        self.started.append(tool_call.get("name", "unknown"))
        start_time = asyncio.get_running_loop().time()
        if self.delay:
            await asyncio.sleep(self.delay)
        name = tool_call.get("name", "unknown")
        self.executed.append(name)
        end_time = asyncio.get_running_loop().time()
        if not hasattr(self, "_timing"):
            self._timing: list[dict[str, float]] = []
        self._timing.append({"name": name, "start": start_time, "end": end_time})
        if name in self.fail_tools:
            raise RuntimeError(f"{name} failed")
        return {
            "tool": name,
            "success": True,
            "result": tool_call.get("arguments", {}),
            "validate": validate,
            "check_budget": check_budget,
        }


@pytest.mark.asyncio
async def test_execute_tool_call_batch_validates_and_truncates_to_budget():
    invalid = {"name": "bad", "_validation_error": "bad payload"}
    calls = [
        {"name": "read", "arguments": {"path": "a.py"}},
        invalid,
        {"name": "search", "arguments": {"query": "needle"}},
    ]
    host = FakeBatchHost(remaining_budget=1, invalid_calls=[invalid])

    results = await execute_tool_call_batch(host, calls)

    assert results == [
        {"tool": "bad", "error": "bad payload"},
        {
            "tool": "read",
            "success": True,
            "result": {"path": "a.py"},
            "validate": False,
            "check_budget": False,
        },
    ]
    assert host.executed == ["read"]
    host._logger.warning.assert_called_once_with(
        "Insufficient budget: %s calls, %s remaining",
        2,
        1,
    )


@pytest.mark.asyncio
async def test_execute_tool_call_batch_converts_parallel_exceptions_to_results():
    calls = [{"name": "ok"}, {"name": "boom"}]
    host = FakeBatchHost(fail_tools={"boom"})

    results = await execute_tool_call_batch(host, calls, validate=False)

    assert results[0]["tool"] == "ok"
    assert results[0]["success"] is True
    assert results[1] == {
        "tool": "boom",
        "success": False,
        "result": None,
        "error": "boom failed",
    }


@pytest.mark.asyncio
async def test_execute_tool_call_batch_respects_sequential_dispatch():
    calls = [{"name": "first"}, {"name": "second"}]
    host = FakeBatchHost(enable_parallel_execution=False)

    results = await execute_tool_call_batch(host, calls, validate=False, parallel=True)

    assert [result["tool"] for result in results] == ["first", "second"]
    assert host.executed == ["first", "second"]


@pytest.mark.asyncio
async def test_execute_tool_call_batch_parallel_execution_overlaps_work():
    calls = [{"name": "first"}, {"name": "second"}]
    host = FakeBatchHost(delay=0.02)

    await execute_tool_call_batch(host, calls, validate=False, parallel=True)
    timings = sorted(host._timing, key=lambda item: item["start"])

    assert len(timings) == 2
    assert timings[1]["start"] < timings[0]["start"] + host.delay
    assert set(host.executed) == {"first", "second"}
    assert set(host.started) == {"first", "second"}
