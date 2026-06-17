from __future__ import annotations

from victor.agent.services.tool_usage_stats import ToolUsageStats


def test_tool_usage_stats_records_successes_and_failures():
    stats = ToolUsageStats()

    stats.record("read", success=True)
    stats.record("read", success=True)
    stats.record("grep", success=False)

    assert stats.get_tool_call_count("read") == 2
    assert stats.get_tool_error_count("grep") == 1
    assert stats.snapshot(budget_remaining=7, budget_used=3) == {
        "total_calls": 3,
        "successful_calls": 2,
        "failed_calls": 1,
        "success_rate": 2 / 3,
        "by_tool": {"read": 2, "error:grep": 1},
        "budget_remaining": 7,
        "budget_used": 3,
    }


def test_tool_usage_stats_empty_snapshot_has_perfect_success_rate():
    stats = ToolUsageStats()

    assert stats.snapshot(budget_remaining=10, budget_used=0) == {
        "total_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "success_rate": 1.0,
        "by_tool": {},
        "budget_remaining": 10,
        "budget_used": 0,
    }


def test_tool_usage_stats_clear_removes_counts():
    stats = ToolUsageStats()
    stats.record("read", success=True)

    stats.clear()

    assert stats.counts == {}


def test_tool_usage_stats_counts_property_preserves_compatibility():
    stats = ToolUsageStats()

    stats.counts = {"read": 2}
    stats.counts["write"] = 1

    assert stats.snapshot(budget_remaining=1, budget_used=2)["by_tool"] == {
        "read": 2,
        "write": 1,
    }
