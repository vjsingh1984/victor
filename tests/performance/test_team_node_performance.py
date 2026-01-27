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

"""Performance benchmarks for team node execution.

This module provides comprehensive benchmarks to measure the performance
characteristics of team nodes with different formations, team sizes,
and execution scenarios.

Benchmark Categories:
    1. Formation Comparison: sequential vs parallel vs pipeline vs hierarchical vs consensus
    2. Team Size Scaling: 2, 5, 10 members
    3. Recursion Depth Overhead: Nested team execution
    4. Tool Budget Impact: Performance with different budgets
    5. Timeout Handling: Graceful degradation
    6. Memory Profiling: Memory usage patterns

Key Metrics:
    - Execution latency (avg, p95, p99)
    - Throughput (teams/second)
    - Memory overhead per member
    - Recursion depth impact
    - Formation-specific overhead

Usage:
    pytest tests/performance/test_team_node_performance.py -v
    pytest tests/performance/test_team_node_performance.py -k formation -v
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Configure logging to reduce noise during benchmarks
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


@pytest.fixture
def run_async():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop.run_until_complete
    finally:
        loop.close()
        asyncio.set_event_loop(None)


# =============================================================================
# Mock Team Members for Benchmarking
# =============================================================================


class MockTeamMember:
    """Lightweight mock team member for benchmarking.

    Attributes:
        id: Member identifier
        role: Member role
        execution_delay: Simulated execution delay in seconds
        tool_calls: Number of tool calls to simulate
        fail_rate: Failure rate (0.0 - 1.0)
    """

    def __init__(
        self,
        member_id: str,
        role: str = "assistant",
        execution_delay: float = 0.01,
        tool_calls: int = 5,
        fail_rate: float = 0.0,
    ):
        self.id = member_id
        self.role = role
        self._execution_delay = execution_delay
        self._tool_calls = tool_calls
        self._fail_rate = fail_rate
        self._call_count = 0

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        """Execute a task with simulated delay."""
        self._call_count += 1

        # Simulate tool call overhead
        await asyncio.sleep(self._execution_delay * self._tool_calls * 0.1)

        # Simulate execution delay
        await asyncio.sleep(self._execution_delay)

        # Simulate failures
        import random

        if random.random() < self._fail_rate:
            raise Exception(f"Member {self.id} failed")

        return f"Task completed by {self.id}: {task[:50]}..."

    @property
    def tool_calls_used(self) -> int:
        """Return simulated tool call count."""
        return self._tool_calls


# =============================================================================
# Benchmark Fixtures
# =============================================================================


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator for testing."""
    orchestrator = MagicMock()
    orchestrator.settings = MagicMock()
    orchestrator.settings.provider = "anthropic"
    orchestrator.settings.model = "claude-sonnet-4-5"
    return orchestrator


@pytest.fixture
def team_coordinator(mock_orchestrator):
    """Create team coordinator for benchmarking."""
    from victor.teams import UnifiedTeamCoordinator

    return UnifiedTeamCoordinator(
        orchestrator=mock_orchestrator,
        enable_observability=False,
        enable_rl=False,
        lightweight_mode=True,
    )


@pytest.fixture
def small_team():
    """Create small team (2 members)."""
    return [
        MockTeamMember("member_1", "researcher", execution_delay=0.01),
        MockTeamMember("member_2", "executor", execution_delay=0.01),
    ]


@pytest.fixture
def medium_team():
    """Create medium team (5 members)."""
    return [
        MockTeamMember(f"member_{i}", ["researcher", "executor"][i % 2], execution_delay=0.01)
        for i in range(5)
    ]


@pytest.fixture
def large_team():
    """Create large team (10 members)."""
    return [
        MockTeamMember(
            f"member_{i}", ["researcher", "executor", "reviewer"][i % 3], execution_delay=0.01
        )
        for i in range(10)
    ]


@pytest.fixture
def recursion_context():
    """Create recursion context for testing."""
    from victor.workflows.recursion import RecursionContext

    return RecursionContext(max_depth=5)


# =============================================================================
# Formation Comparison Benchmarks
# =============================================================================


@pytest.mark.benchmark(group="team_formations")
@pytest.mark.parametrize(
    "formation", ["sequential", "parallel", "pipeline", "hierarchical", "consensus"]
)
def test_formation_performance(benchmark, team_coordinator, small_team, formation, run_async):
    """Benchmark performance of different team formations.

    Measures:
        - Sequential: Members execute one after another
        - Parallel: All members execute simultaneously
        - Pipeline: Output flows through stages
        - Hierarchical: Manager delegates to workers
        - Consensus: Members must agree (multiple rounds)
    """
    from victor.teams.types import TeamFormation

    # Clear any previous members
    team_coordinator.clear()

    # Add members
    for member in small_team:
        team_coordinator.add_member(member)

    # Set formation
    team_coordinator.set_formation(TeamFormation(formation))

    # Benchmark execution
    async def run_team():
        return await team_coordinator.execute_task(
            task="Analyze codebase for performance bottlenecks",
            context={"team_name": f"benchmark_{formation}"},
        )

    result = benchmark(lambda: run_async(run_team()))
    assert result["success"]


@pytest.mark.benchmark(group="team_size")
@pytest.mark.parametrize("team_size", [2, 5, 10])
def test_team_size_scaling(benchmark, team_coordinator, team_size, run_async):
    """Benchmark performance scaling with team size.

    Expected behavior:
        - Sequential: O(n) - linear growth
        - Parallel: O(1) - constant time (limited by slowest member)
        - Pipeline: O(n) - linear but faster than sequential
        - Hierarchical: O(n) - depends on delegation pattern
        - Consensus: O(n * rounds) - depends on convergence
    """
    from victor.teams.types import TeamFormation

    # Create team of specified size
    team = [
        MockTeamMember(f"member_{i}", "assistant", execution_delay=0.01) for i in range(team_size)
    ]

    team_coordinator.clear()
    for member in team:
        team_coordinator.add_member(member)

    # Use parallel formation (should show best scaling)
    team_coordinator.set_formation(TeamFormation.PARALLEL)

    async def run_team():
        return await team_coordinator.execute_task(
            task="Process large dataset", context={"team_name": f"size_{team_size}"}
        )

    result = benchmark(lambda: run_async(run_team()))
    assert result["success"]


@pytest.mark.benchmark(group="tool_budget")
@pytest.mark.parametrize("budget", [5, 25, 50, 100])
def test_tool_budget_impact(benchmark, team_coordinator, budget, run_async):
    """Benchmark impact of tool budget on performance.

    Tool budget affects:
        - Maximum iterations allowed
        - Tool call limits per member
        - Timeout calculations

    Expected behavior:
        - Higher budgets = more iterations = longer execution
        - Linear relationship between budget and time
    """
    from victor.teams.types import TeamFormation

    # Create team with varying tool budgets
    team = [
        MockTeamMember("member_1", "researcher", execution_delay=0.01, tool_calls=budget),
        MockTeamMember("member_2", "executor", execution_delay=0.01, tool_calls=budget),
    ]

    team_coordinator.clear()
    for member in team:
        team_coordinator.add_member(member)

    team_coordinator.set_formation(TeamFormation.SEQUENTIAL)

    async def run_team():
        return await team_coordinator.execute_task(
            task="Implement feature with full testing",
            context={
                "team_name": f"budget_{budget}",
                "tool_budget": budget,
            },
        )

    result = benchmark(lambda: run_async(run_team()))
    assert result["success"]


# =============================================================================
# Recursion Depth Benchmarks
# =============================================================================


@pytest.mark.benchmark(group="recursion")
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_recursion_depth_overhead(benchmark, mock_orchestrator, depth, run_async):
    """Benchmark overhead of recursion depth tracking.

    Measures the performance impact of RecursionGuard when teams
    spawn nested teams or workflows.

    Expected behavior:
        - Minimal overhead (< 1ms per depth level)
        - Linear growth with depth
    """
    from victor.teams import UnifiedTeamCoordinator
    from victor.teams.types import TeamFormation
    from victor.workflows.recursion import RecursionContext, RecursionGuard

    recursion_ctx = RecursionContext(max_depth=10)

    async def run_nested_team():
        # Simulate nested team execution
        for i in range(depth):
            with RecursionGuard(recursion_ctx, "team", f"team_level_{i}"):
                coordinator = UnifiedTeamCoordinator(
                    orchestrator=mock_orchestrator,
                    lightweight_mode=True,
                )
                member = MockTeamMember(f"member_{i}", "assistant", execution_delay=0.001)
                coordinator.add_member(member)
                coordinator.set_formation(TeamFormation.SEQUENTIAL)
                await coordinator.execute_task(
                    task=f"Level {i} task", context={"team_name": f"nested_{i}"}
                )

    benchmark(lambda: run_async(run_nested_team()))


# =============================================================================
# Timeout Handling Benchmarks
# =============================================================================


@pytest.mark.benchmark(group="timeout")
@pytest.mark.parametrize("timeout", [1, 5, 10])
def test_timeout_performance(benchmark, team_coordinator, timeout, run_async):
    """Benchmark timeout handling performance.

    Measures overhead of timeout enforcement and graceful degradation.

    Expected behavior:
        - Minimal overhead for normal completion
        - Fast termination on timeout (< 100ms overhead)
    """
    from victor.teams.types import TeamFormation

    team = [
        MockTeamMember("member_1", "assistant", execution_delay=0.01),
        MockTeamMember("member_2", "assistant", execution_delay=0.01),
    ]

    team_coordinator.clear()
    for member in team:
        team_coordinator.add_member(member)

    team_coordinator.set_formation(TeamFormation.PARALLEL)

    async def run_with_timeout():
        try:
            return await asyncio.wait_for(
                team_coordinator.execute_task(
                    task="Quick task", context={"team_name": "timeout_test"}
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return {"success": False, "error": "timeout"}

    result = benchmark(lambda: run_async(run_with_timeout()))
    assert result is not None


# =============================================================================
# Memory Profiling Benchmarks
# =============================================================================


@pytest.mark.benchmark(group="memory")
@pytest.mark.parametrize("team_size", [2, 5, 10])
def test_memory_per_member(benchmark, team_coordinator, team_size, run_async):
    """Benchmark memory usage per team member.

    Measures:
        - Base coordinator overhead
        - Per-member memory footprint
        - Context storage overhead
        - Message history growth

    Expected behavior:
        - Linear growth with team size
        - < 10KB overhead per member
    """
    from victor.teams.types import TeamFormation

    tracemalloc.start()

    team = [
        MockTeamMember(f"member_{i}", "assistant", execution_delay=0.01) for i in range(team_size)
    ]

    team_coordinator.clear()
    for member in team:
        team_coordinator.add_member(member)

    team_coordinator.set_formation(TeamFormation.PARALLEL)

    async def run_team():
        result = await team_coordinator.execute_task(
            task="Memory test task", context={"team_name": f"memory_{team_size}"}
        )
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {"result": result, "memory": peak}

    output = benchmark(lambda: run_async(run_team()))
    assert output["result"]["success"]
    # Memory should be reasonable (< 1MB for 10 members)
    assert output["memory"] < 1_000_000


# =============================================================================
# Complex Scenario Benchmarks
# =============================================================================


@pytest.mark.benchmark(group="scenarios")
def test_simple_task_scenario(benchmark, team_coordinator, run_async):
    """Benchmark simple single-tool task scenario.

    Scenario: Quick task with minimal coordination overhead.
    Expected: Fast execution (< 100ms for 2 members).
    """
    from victor.teams.types import TeamFormation

    team = [
        MockTeamMember("quick_researcher", "researcher", execution_delay=0.005, tool_calls=2),
        MockTeamMember("quick_executor", "executor", execution_delay=0.005, tool_calls=2),
    ]

    team_coordinator.clear()
    for member in team:
        team_coordinator.add_member(member)

    team_coordinator.set_formation(TeamFormation.SEQUENTIAL)

    async def run_simple():
        return await team_coordinator.execute_task(
            task="Read and summarize file",
            context={"team_name": "simple_task"},
        )

    result = benchmark(lambda: run_async(run_simple()))
    assert result["success"]


@pytest.mark.benchmark(group="scenarios")
def test_complex_task_scenario(benchmark, team_coordinator, run_async):
    """Benchmark complex multi-tool task scenario.

    Scenario: Complex task requiring multiple tools and coordination.
    Expected: Longer execution but acceptable (< 500ms for 4 members).
    """
    from victor.teams.types import TeamFormation

    team = [
        MockTeamMember("architect", "planner", execution_delay=0.02, tool_calls=10),
        MockTeamMember("researcher", "researcher", execution_delay=0.03, tool_calls=15),
        MockTeamMember("implementer", "executor", execution_delay=0.05, tool_calls=20),
        MockTeamMember("reviewer", "reviewer", execution_delay=0.02, tool_calls=10),
    ]

    team_coordinator.clear()
    for member in team:
        team_coordinator.add_member(member)

    team_coordinator.set_formation(TeamFormation.PIPELINE)

    async def run_complex():
        return await team_coordinator.execute_task(
            task="Design and implement authentication system",
            context={
                "team_name": "complex_task",
                "requirements": ["JWT", "OAuth2", "session management"],
            },
        )

    result = benchmark(lambda: run_async(run_complex()))
    assert result["success"]


@pytest.mark.benchmark(group="scenarios")
def test_large_context_scenario(benchmark, team_coordinator, run_async):
    """Benchmark with large context (10+ KB state).

    Scenario: Team coordination with large shared context.
    Expected: Minimal context overhead (< 50ms for 10KB).
    """
    from victor.teams.types import TeamFormation

    # Create large context (10+ KB)
    large_context = {
        "team_name": "large_context_test",
        "codebase": {"files": [f"file_{i}.py: {'x' * 1000}" for i in range(10)]},
        "history": [{"step": i, "data": "x" * 500} for i in range(20)],
    }

    team = [
        MockTeamMember("member_1", "assistant", execution_delay=0.01),
        MockTeamMember("member_2", "assistant", execution_delay=0.01),
    ]

    team_coordinator.clear()
    for member in team:
        team_coordinator.add_member(member)

    team_coordinator.set_formation(TeamFormation.PARALLEL)

    async def run_with_large_context():
        return await team_coordinator.execute_task(
            task="Process large context", context=large_context
        )

    result = benchmark(lambda: run_async(run_with_large_context()))
    assert result["success"]


# =============================================================================
# Consensus Formation Benchmarks
# =============================================================================


@pytest.mark.benchmark(group="consensus")
@pytest.mark.parametrize("members,rounds", [(3, 2), (5, 3), (7, 4)])
def test_consensus_formation_performance(benchmark, team_coordinator, members, rounds, run_async):
    """Benchmark consensus formation with varying team sizes and rounds.

    Consensus formation requires multiple rounds until all members agree.
    Performance depends on:
        - Number of members (more members = harder to agree)
        - Rounds needed (more rounds = more communication)

    Expected behavior:
        - O(members * rounds) complexity
        - Each round adds member_count messages
    """
    from victor.teams.types import TeamFormation

    team = [
        MockTeamMember(f"member_{i}", "assistant", execution_delay=0.01) for i in range(members)
    ]

    team_coordinator.clear()
    for member in team:
        team_coordinator.add_member(member)

    team_coordinator.set_formation(TeamFormation.CONSENSUS)

    async def run_consensus():
        return await team_coordinator.execute_task(
            task="Reach consensus on decision",
            context={
                "team_name": f"consensus_{members}",
                "max_rounds": rounds,
            },
        )

    result = benchmark(lambda: run_async(run_consensus()))
    assert result["success"]

# =============================================================================
# Performance Summary Tests
# =============================================================================

# Note: test_team_node_performance_summary moved to test_team_node_performance_benchmark.py
# which provides more comprehensive coverage including nested execution and recursion overhead.
# The benchmark version includes: single_level, nested, formations, memory, recursion_overhead.
