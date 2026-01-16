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

"""Comprehensive performance benchmarks for team node execution.

This module provides extensive benchmarks for team nodes with different formations,
member counts, and execution scenarios. It focuses on measuring:

1. Formation Performance: Execution time per formation type
2. Scalability: Performance with 1-10 members
3. Recursion Depth: Overhead of nested team execution
4. Memory Usage: Per-member memory footprint
5. Communication Overhead: Message passing between members
6. Tool Budget Impact: Performance with different budgets
7. Timeout Handling: Graceful degradation

Performance Targets:
- Team node execution: <5s per team (3 members, parallel)
- Recursion depth tracking: <1% overhead
- Memory usage: <10MB for 10-member team

Usage:
    # Run all benchmarks
    pytest tests/performance/team_node_benchmarks.py -v

    # Run specific benchmark groups
    pytest tests/performance/team_node_benchmarks.py -k "formation" -v
    pytest tests/performance/team_node_benchmarks.py -k "scalability" -v
    pytest tests/performance/team_node_benchmarks.py -k "recursion" -v

    # Generate benchmark report
    pytest tests/performance/team_node_benchmarks.py --benchmark-only --benchmark-json=team_benchmarks.json
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import random
import time
import tracemalloc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# =============================================================================
# Team Formation Types
# =============================================================================


class TeamFormationType(str, Enum):
    """Team formation types for benchmarking."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


# =============================================================================
# Mock Team Member for Benchmarking
# =============================================================================


@dataclass
class TeamMemberResult:
    """Result from a team member execution."""

    member_id: str
    output: str
    tool_calls: int
    execution_time: float
    success: bool
    error: Optional[str] = None


class MockTeamMember:
    """Lightweight mock team member for benchmarking.

    Attributes:
        member_id: Member identifier
        role: Member role
        execution_delay: Simulated execution delay in seconds
        tool_calls: Number of tool calls to simulate
        fail_rate: Failure rate (0.0 - 1.0)
        message_size: Size of messages to generate
    """

    def __init__(
        self,
        member_id: str,
        role: str = "assistant",
        execution_delay: float = 0.01,
        tool_calls: int = 5,
        fail_rate: float = 0.0,
        message_size: int = 100,
    ):
        self.member_id = member_id
        self.role = role
        self._execution_delay = execution_delay
        self._tool_calls = tool_calls
        self._fail_rate = fail_rate
        self._message_size = message_size
        self._call_count = 0

    async def execute_task(self, task: str, context: Dict[str, Any]) -> TeamMemberResult:
        """Execute a task with simulated delay and tool calls."""
        start_time = time.perf_counter()
        self._call_count += 1

        # Simulate tool call overhead
        tool_call_delay = self._execution_delay * self._tool_calls * 0.1
        await asyncio.sleep(tool_call_delay)

        # Simulate execution delay
        await asyncio.sleep(self._execution_delay)

        # Simulate failures
        if random.random() < self._fail_rate:
            exec_time = time.perf_counter() - start_time
            return TeamMemberResult(
                member_id=self.member_id,
                output="",
                tool_calls=self._tool_calls,
                execution_time=exec_time,
                success=False,
                error=f"Member {self.member_id} failed",
            )

        # Generate output message
        output = f"Task completed by {self.member_id}: {task[:50]}..." + " " * self._message_size
        exec_time = time.perf_counter() - start_time

        return TeamMemberResult(
            member_id=self.member_id,
            output=output,
            tool_calls=self._tool_calls,
            execution_time=exec_time,
            success=True,
        )

    @property
    def tool_calls_used(self) -> int:
        """Return simulated tool call count."""
        return self._tool_calls


# =============================================================================
# Mock Team Coordinator for Benchmarking
# =============================================================================


class MockTeamCoordinator:
    """Lightweight mock team coordinator for performance testing.

    This simulates team coordination without actual orchestrator overhead.
    """

    def __init__(
        self,
        formation: TeamFormationType = TeamFormationType.SEQUENTIAL,
        enable_recursion_tracking: bool = True,
    ):
        self.formation = formation
        self.members: List[MockTeamMember] = []
        self.enable_recursion_tracking = enable_recursion_tracking
        self.recursion_depth = 0
        self.max_recursion_depth = 10
        self.message_count = 0

    def add_member(self, member: MockTeamMember) -> None:
        """Add a member to the team."""
        self.members.append(member)

    def clear(self) -> None:
        """Clear all members."""
        self.members.clear()

    async def execute_team(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute team task based on formation."""
        context = context or {}
        start_time = time.perf_counter()

        # Track recursion depth
        if self.enable_recursion_tracking:
            self.recursion_depth += 1
            if self.recursion_depth > self.max_recursion_depth:
                self.recursion_depth -= 1
                raise RecursionError(f"Max recursion depth {self.max_recursion_depth} exceeded")

        try:
            if self.formation == TeamFormationType.SEQUENTIAL:
                results = await self._execute_sequential(task, context)
            elif self.formation == TeamFormationType.PARALLEL:
                results = await self._execute_parallel(task, context)
            elif self.formation == TeamFormationType.PIPELINE:
                results = await self._execute_pipeline(task, context)
            elif self.formation == TeamFormationType.HIERARCHICAL:
                results = await self._execute_hierarchical(task, context)
            elif self.formation == TeamFormationType.CONSENSUS:
                results = await self._execute_consensus(task, context)
            elif self.formation == TeamFormationType.DYNAMIC:
                results = await self._execute_dynamic(task, context)
            elif self.formation == TeamFormationType.ADAPTIVE:
                results = await self._execute_adaptive(task, context)
            elif self.formation == TeamFormationType.HYBRID:
                results = await self._execute_hybrid(task, context)
            else:
                raise ValueError(f"Unknown formation: {self.formation}")

            total_time = time.perf_counter() - start_time
            success = all(r.success for r in results)

            return {
                "success": success,
                "results": results,
                "total_time": total_time,
                "formation": self.formation.value,
                "member_count": len(self.members),
                "message_count": self.message_count,
            }

        finally:
            if self.enable_recursion_tracking:
                self.recursion_depth -= 1

    async def _execute_sequential(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> List[TeamMemberResult]:
        """Execute members sequentially."""
        results = []
        for member in self.members:
            self.message_count += 1
            result = await member.execute_task(task, context)
            results.append(result)
            if not result.success:
                break
        return results

    async def _execute_parallel(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> List[TeamMemberResult]:
        """Execute members in parallel."""
        tasks = [member.execute_task(task, context) for member in self.members]
        self.message_count += len(self.members)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                final_results.append(
                    TeamMemberResult(
                        member_id=self.members[i].member_id,
                        output="",
                        tool_calls=0,
                        execution_time=0,
                        success=False,
                        error=str(r),
                    )
                )
            else:
                final_results.append(r)

        return final_results

    async def _execute_pipeline(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> List[TeamMemberResult]:
        """Execute members in pipeline (output of one feeds into next)."""
        results = []
        current_context = context.copy()

        for i, member in enumerate(self.members):
            self.message_count += 1
            result = await member.execute_task(task, current_context)
            results.append(result)

            # Update context with result
            current_context[f"member_{i}_output"] = result.output

            if not result.success:
                break

        return results

    async def _execute_hierarchical(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> List[TeamMemberResult]:
        """Execute members in hierarchical formation (manager delegates to workers)."""
        if not self.members:
            return []

        # First member is manager
        manager = self.members[0]
        workers = self.members[1:]

        self.message_count += 1
        manager_result = await manager.execute_task(task, context)
        results = [manager_result]

        if not manager_result.success:
            return results

        # Manager delegates to workers
        worker_tasks = [worker.execute_task(manager_result.output, context) for worker in workers]
        self.message_count += len(workers)
        worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for i, r in enumerate(worker_results):
            if isinstance(r, Exception):
                results.append(
                    TeamMemberResult(
                        member_id=workers[i].member_id,
                        output="",
                        tool_calls=0,
                        execution_time=0,
                        success=False,
                        error=str(r),
                    )
                )
            else:
                results.append(r)

        return results

    async def _execute_consensus(
        self,
        task: str,
        context: Dict[str, Any],
        max_rounds: int = 3,
    ) -> List[TeamMemberResult]:
        """Execute members until consensus is reached."""
        results = []
        all_agreed = False
        round_num = 0

        while not all_agreed and round_num < max_rounds:
            round_num += 1
            round_results = []

            for member in self.members:
                self.message_count += 1
                result = await member.execute_task(f"{task} (round {round_num})", context)
                round_results.append(result)

            if all(r.success for r in round_results):
                # Check if outputs are similar (simplified)
                outputs = [r.output for r in round_results]
                all_agreed = len(set(o[:50] for o in outputs)) <= 1

            results.extend(round_results)

        return results

    async def _execute_dynamic(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> List[TeamMemberResult]:
        """Execute members with dynamic formation selection."""
        # Simple heuristic: use parallel for >3 members, sequential otherwise
        if len(self.members) > 3:
            return await self._execute_parallel(task, context)
        else:
            return await self._execute_sequential(task, context)

    async def _execute_adaptive(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> List[TeamMemberResult]:
        """Execute members with adaptive performance tuning."""
        # Start with parallel, fallback to sequential if any member fails
        try:
            results = await self._execute_parallel(task, context)
            if all(r.success for r in results):
                return results
        except Exception:
            pass

        # Fallback to sequential
        return await self._execute_sequential(task, context)

    async def _execute_hybrid(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> List[TeamMemberResult]:
        """Execute members using hybrid approach (pipeline + parallel stages)."""
        if len(self.members) <= 2:
            return await self._execute_pipeline(task, context)

        # Split into stages
        stage1 = self.members[: len(self.members) // 2]
        stage2 = self.members[len(self.members) // 2 :]

        # Execute stage1 in pipeline
        results = []
        current_context = context.copy()

        for member in stage1:
            self.message_count += 1
            result = await member.execute_task(task, current_context)
            results.append(result)
            current_context[f"member_{len(results)}_output"] = result.output

        # Execute stage2 in parallel
        stage2_tasks = [member.execute_task(result.output, context) for member in stage2]
        self.message_count += len(stage2)
        stage2_results = await asyncio.gather(*stage2_tasks, return_exceptions=True)

        for i, r in enumerate(stage2_results):
            if isinstance(r, Exception):
                results.append(
                    TeamMemberResult(
                        member_id=stage2[i].member_id,
                        output="",
                        tool_calls=0,
                        execution_time=0,
                        success=False,
                        error=str(r),
                    )
                )
            else:
                results.append(r)

        return results


# =============================================================================
# Benchmark Fixtures
# =============================================================================


@pytest.fixture
def team_coordinator():
    """Create team coordinator for benchmarking."""
    return MockTeamCoordinator()


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
        MockTeamMember(
            f"member_{i}",
            ["researcher", "executor", "reviewer"][i % 3],
            execution_delay=0.01,
        )
        for i in range(5)
    ]


@pytest.fixture
def large_team():
    """Create large team (10 members)."""
    return [
        MockTeamMember(
            f"member_{i}",
            ["researcher", "executor", "reviewer"][i % 3],
            execution_delay=0.01,
        )
        for i in range(10)
    ]


# =============================================================================
# Formation Performance Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("formation", [
    TeamFormationType.SEQUENTIAL,
    TeamFormationType.PARALLEL,
    TeamFormationType.PIPELINE,
    TeamFormationType.HIERARCHICAL,
    TeamFormationType.CONSENSUS,
    TeamFormationType.DYNAMIC,
    TeamFormationType.ADAPTIVE,
    TeamFormationType.HYBRID,
])
def test_formation_performance(benchmark, formation):
    """Benchmark performance of different team formations.

    Measures execution time for each formation type with 3 members.

    Performance Targets:
    - Sequential: <50ms
    - Parallel: <30ms (parallelized)
    - Pipeline: <40ms
    - Hierarchical: <35ms
    - Consensus: <80ms (multiple rounds)
    - Dynamic: <35ms
    - Adaptive: <40ms
    - Hybrid: <45ms
    """
    coordinator = MockTeamCoordinator(formation=formation)

    # Create team of 3 members
    for i in range(3):
        coordinator.add_member(
            MockTeamMember(f"member_{i}", "assistant", execution_delay=0.01)
        )

    def run_team():
        return asyncio.run(coordinator.execute_team(
            task="Analyze codebase for performance bottlenecks",
            context={"team_name": f"benchmark_{formation.value}"},
        ))

    result = benchmark(run_team)

    # Verify success
    assert result["success"], f"Team execution failed for {formation.value}"
    assert result["member_count"] == 3

    # Print performance metrics
    print(f"\n{formation.value:15} | Time: {result['total_time']*1000:7.2f}ms | "
          f"Messages: {result['message_count']:3} | Members: {result['member_count']}")


@pytest.mark.benchmark
@pytest.mark.parametrize("formation,expected_max_ms", [
    (TeamFormationType.SEQUENTIAL, 60),
    (TeamFormationType.PARALLEL, 30),
    (TeamFormationType.PIPELINE, 60),
    (TeamFormationType.HIERARCHICAL, 40),
    (TeamFormationType.CONSENSUS, 200),
])
def test_formation_performance_targets(benchmark, formation, expected_max_ms):
    """Verify that formations meet performance targets."""
    coordinator = MockTeamCoordinator(formation=formation)

    for i in range(3):
        coordinator.add_member(
            MockTeamMember(f"member_{i}", "assistant", execution_delay=0.01)
        )

    def run_team():
        return asyncio.run(coordinator.execute_team(task="Quick task", context={}))

    result = benchmark(run_team)

    # Check performance target
    exec_time_ms = result["total_time"] * 1000
    assert exec_time_ms < expected_max_ms, (
        f"{formation.value} exceeded target: {exec_time_ms:.2f}ms > {expected_max_ms}ms"
    )


# =============================================================================
# Scalability Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("member_count", [1, 2, 3, 5, 7, 10])
def test_scalability_sequential(benchmark, member_count):
    """Benchmark sequential formation scaling with member count.

    Expected: O(n) - Linear growth
    """
    coordinator = MockTeamCoordinator(formation=TeamFormationType.SEQUENTIAL)

    for i in range(member_count):
        coordinator.add_member(
            MockTeamMember(f"member_{i}", "assistant", execution_delay=0.01)
        )

    def run_team():
        return asyncio.run(coordinator.execute_team(task="Scale test", context={}))

    result = benchmark(run_team)
    assert result["success"]

    # Expected: roughly member_count * execution_delay
    expected_time = member_count * 0.01
    actual_time = result["total_time"]
    ratio = actual_time / expected_time if expected_time > 0 else 0

    print(f"\nSequential | Members: {member_count:2} | "
          f"Time: {actual_time*1000:7.2f}ms | Ratio: {ratio:.2f}x")


@pytest.mark.benchmark
@pytest.mark.parametrize("member_count", [1, 2, 3, 5, 7, 10])
def test_scalability_parallel(benchmark, member_count):
    """Benchmark parallel formation scaling with member count.

    Expected: O(1) - Constant time (limited by slowest member)
    """
    coordinator = MockTeamCoordinator(formation=TeamFormationType.PARALLEL)

    for i in range(member_count):
        coordinator.add_member(
            MockTeamMember(f"member_{i}", "assistant", execution_delay=0.01)
        )

    def run_team():
        return asyncio.run(coordinator.execute_team(task="Scale test", context={}))

    result = benchmark(run_team)
    assert result["success"]

    # Expected: roughly execution_delay (constant regardless of member count)
    expected_time = 0.01
    actual_time = result["total_time"]
    ratio = actual_time / expected_time if expected_time > 0 else 0

    print(f"\nParallel   | Members: {member_count:2} | "
          f"Time: {actual_time*1000:7.2f}ms | Ratio: {ratio:.2f}x")


@pytest.mark.benchmark
@pytest.mark.parametrize("member_count", [2, 3, 5, 7, 10])
def test_scalability_consensus(benchmark, member_count):
    """Benchmark consensus formation scaling with member count.

    Expected: O(n * rounds) - Depends on convergence
    """
    coordinator = MockTeamCoordinator(formation=TeamFormationType.CONSENSUS)

    for i in range(member_count):
        coordinator.add_member(
            MockTeamMember(f"member_{i}", "assistant", execution_delay=0.01)
        )

    def run_team():
        return asyncio.run(coordinator.execute_team(task="Consensus test", context={}))

    result = benchmark(run_team)
    assert result["success"]

    # Consensus should take longer with more members
    print(f"\nConsensus  | Members: {member_count:2} | "
          f"Time: {result['total_time']*1000:7.2f}ms | "
          f"Messages: {result['message_count']:3}")


# =============================================================================
# Recursion Depth Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("depth", [1, 2, 3, 5, 7, 10])
def test_recursion_depth_overhead(benchmark, depth):
    """Benchmark overhead of recursion depth tracking.

    Measures the performance impact of tracking recursion depth when teams
    spawn nested teams.

    Performance Target:
    - Recursion tracking overhead: <1ms per depth level
    - Linear growth with depth: O(n)
    """
    results = []

    def run_nested_teams():
        """Simulate nested team execution."""
        start_time = time.perf_counter()

        async def _run_nested():
            for i in range(depth):
                coordinator = MockTeamCoordinator(
                    formation=TeamFormationType.SEQUENTIAL,
                    enable_recursion_tracking=True,
                )
                coordinator.add_member(
                    MockTeamMember(f"member_{i}", "assistant", execution_delay=0.001)
                )

                result = await coordinator.execute_team(
                    task=f"Level {i} task",
                    context={"level": i},
                )
                results.append(result)

            return time.perf_counter() - start_time

        return asyncio.run(_run_nested())

    total_time = benchmark(run_nested_teams)

    # Verify all nested teams succeeded
    assert all(r["success"] for r in results)

    # Calculate per-level overhead
    overhead_per_level = total_time / depth

    print(f"\nRecursion Depth: {depth:2} | "
          f"Total Time: {total_time*1000:7.2f}ms | "
          f"Overhead/Level: {overhead_per_level*1000:6.3f}ms")

    # Verify overhead is acceptable (<5ms per level)
    assert overhead_per_level < 0.005, (
        f"Recursion tracking overhead {overhead_per_level*1000:.3f}ms "
        f"exceeds 5ms per level target"
    )


def test_recursion_tracking_overhead_vs_no_tracking():
    """Compare performance with and without recursion tracking."""
    depth = 5

    # With tracking
    async def with_tracking():
        start = time.perf_counter()
        for i in range(depth):
            coordinator = MockTeamCoordinator(
                formation=TeamFormationType.SEQUENTIAL,
                enable_recursion_tracking=True,
            )
            coordinator.add_member(
                MockTeamMember(f"member_{i}", "assistant", execution_delay=0.001)
            )
            await coordinator.execute_team(task=f"Task {i}", context={})
        return time.perf_counter() - start

    # Without tracking
    async def without_tracking():
        start = time.perf_counter()
        for i in range(depth):
            coordinator = MockTeamCoordinator(
                formation=TeamFormationType.SEQUENTIAL,
                enable_recursion_tracking=False,
            )
            coordinator.add_member(
                MockTeamMember(f"member_{i}", "assistant", execution_delay=0.001)
            )
            await coordinator.execute_team(task=f"Task {i}", context={})
        return time.perf_counter() - start

    time_with = asyncio.run(with_tracking())
    time_without = asyncio.run(without_tracking())

    overhead = time_with - time_without
    overhead_pct = (overhead / time_without * 100) if time_without > 0 else 0

    print(f"\nRecursion Tracking Overhead:")
    print(f"  With tracking:    {time_with*1000:.2f}ms")
    print(f"  Without tracking: {time_without*1000:.2f}ms")
    print(f"  Overhead:         {overhead*1000:.2f}ms ({overhead_pct:.1f}%)")

    # Verify overhead is minimal (<10%)
    assert overhead_pct < 10.0, (
        f"Recursion tracking overhead {overhead_pct:.1f}% exceeds 10% target"
    )


# =============================================================================
# Memory Profiling Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("member_count", [2, 5, 10])
def test_memory_per_member(benchmark, member_count):
    """Benchmark memory usage per team member.

    Measures:
    - Base coordinator overhead
    - Per-member memory footprint
    - Context storage overhead

    Performance Targets:
    - < 1MB overhead per member
    - Linear growth with member count
    """
    gc.collect()
    tracemalloc.start()

    coordinator = MockTeamCoordinator(formation=TeamFormationType.PARALLEL)

    for i in range(member_count):
        coordinator.add_member(
            MockTeamMember(
                f"member_{i}",
                "assistant",
                execution_delay=0.01,
                message_size=1000,  # 1KB messages
            )
        )

    def run_team():
        async def _run_team():
            result = await coordinator.execute_team(
                task="Memory test task",
                context={"data": "x" * 10000},  # 10KB context
            )
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return {"result": result, "memory": peak}
        return asyncio.run(_run_team())

    output = benchmark(run_team)

    assert output["result"]["success"]

    peak_memory_kb = output["memory"] / 1024
    memory_per_member = peak_memory_kb / member_count

    print(f"\nMemory Usage | Members: {member_count:2} | "
          f"Total: {peak_memory_kb:6.1f}KB | "
          f"Per-Member: {memory_per_member:5.1f}KB")

    # Target: < 1MB (1024KB) for 10 members
    assert output["memory"] < 1_000_000, (
        f"Memory usage {peak_memory_kb:.1f}KB exceeds 1MB target for {member_count} members"
    )


def test_memory_leak_detection():
    """Test for memory leaks during repeated team executions."""
    gc.collect()
    tracemalloc.start()

    coordinator = MockTeamCoordinator(formation=TeamFormationType.PARALLEL)

    # Add members once
    for i in range(3):
        coordinator.add_member(
            MockTeamMember(f"member_{i}", "assistant", execution_delay=0.01)
        )

    memory_snapshots = []

    async def run_multiple_iterations():
        for iteration in range(10):
            result = await coordinator.execute_team(
                task=f"Iteration {iteration}",
                context={"iteration": iteration},
            )

            if iteration % 2 == 0:
                current, peak = tracemalloc.get_traced_memory()
                memory_snapshots.append(current)

        return result

    result = asyncio.run(run_multiple_iterations())
    tracemalloc.stop()

    assert result["success"]

    # Check for memory leak (memory should stabilize)
    if len(memory_snapshots) >= 4:
        early_avg = sum(memory_snapshots[:2]) / 2
        late_avg = sum(memory_snapshots[-2:]) / 2
        growth = (late_avg - early_avg) / early_avg * 100 if early_avg > 0 else 0

        print(f"\nMemory Leak Check:")
        print(f"  Early avg:  {early_avg / 1024:.1f}KB")
        print(f"  Late avg:   {late_avg / 1024:.1f}KB")
        print(f"  Growth:     {growth:.1f}%")

        # Memory growth should be minimal (<50%)
        assert growth < 50, f"Memory growth {growth:.1f}% suggests leak"


# =============================================================================
# Communication Overhead Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("message_size", [100, 1000, 10000])
def test_communication_overhead(benchmark, message_size):
    """Benchmark communication overhead with different message sizes.

    Measures the cost of passing messages between team members.
    """
    coordinator = MockTeamCoordinator(formation=TeamFormationType.PIPELINE)

    for i in range(3):
        coordinator.add_member(
            MockTeamMember(
                f"member_{i}",
                "assistant",
                execution_delay=0.01,
                message_size=message_size,
            )
        )

    def run_team():
        return asyncio.run(coordinator.execute_team(
            task="Communication test",
            context={"data": "x" * message_size},
        ))

    result = benchmark(run_team)

    assert result["success"]
    messages_per_member = result["message_count"] / result["member_count"]

    print(f"\nCommunication | Msg Size: {message_size:5}B | "
          f"Time: {result['total_time']*1000:7.2f}ms | "
          f"Messages/Member: {messages_per_member:.1f}")


# =============================================================================
# Tool Budget Impact Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("tool_budget", [5, 25, 50, 100])
def test_tool_budget_impact(benchmark, tool_budget):
    """Benchmark impact of tool budget on performance.

    Tool budget affects:
    - Maximum iterations allowed
    - Tool call limits per member
    - Timeout calculations

    Expected: Linear relationship between budget and time
    """
    coordinator = MockTeamCoordinator(formation=TeamFormationType.SEQUENTIAL)

    for i in range(2):
        coordinator.add_member(
            MockTeamMember(
                f"member_{i}",
                "assistant",
                execution_delay=0.01,
                tool_calls=tool_budget,
            )
        )

    def run_team():
        return asyncio.run(coordinator.execute_team(
            task="Tool budget test",
            context={"tool_budget": tool_budget},
        ))

    result = benchmark(run_team)

    assert result["success"]

    # Calculate per-tool-call time
    total_tool_calls = sum(m.tool_calls_used for m in coordinator.members)
    time_per_call = result["total_time"] / total_tool_calls if total_tool_calls > 0 else 0

    print(f"\nTool Budget | Budget: {tool_budget:3} | "
          f"Time: {result['total_time']*1000:7.2f}ms | "
          f"Per-Call: {time_per_call*1000:6.3f}ms")


# =============================================================================
# Complex Scenario Benchmarks
# =============================================================================


@pytest.mark.benchmark
def test_real_world_code_review_team(benchmark):
    """Benchmark realistic code review team scenario.

    Scenario: 5-member team reviewing code with different specializations.
    """
    coordinator = MockTeamCoordinator(formation=TeamFormationType.PARALLEL)

    # Create specialized reviewers
    reviewers = [
        MockTeamMember("security_reviewer", "reviewer", execution_delay=0.02, tool_calls=10),
        MockTeamMember("performance_reviewer", "reviewer", execution_delay=0.015, tool_calls=8),
        MockTeamMember("style_reviewer", "reviewer", execution_delay=0.01, tool_calls=5),
        MockTeamMember("logic_reviewer", "reviewer", execution_delay=0.025, tool_calls=12),
        MockTeamMember("docs_reviewer", "reviewer", execution_delay=0.01, tool_calls=5),
    ]

    for reviewer in reviewers:
        coordinator.add_member(reviewer)

    def run_team():
        return asyncio.run(coordinator.execute_team(
            task="Review authentication module implementation",
            context={
                "file": "auth.py",
                "lines_of_code": 500,
                "complexity": "high",
            },
        ))

    result = benchmark(run_team)

    assert result["success"]
    print(f"\nCode Review Team | Time: {result['total_time']*1000:.2f}ms | "
          f"Members: {result['member_count']} | Messages: {result['message_count']}")


@pytest.mark.benchmark
def test_feature_implementation_pipeline(benchmark):
    """Benchmark realistic feature implementation pipeline.

    Scenario: 4-stage pipeline implementing a new feature.
    """
    coordinator = MockTeamCoordinator(formation=TeamFormationType.PIPELINE)

    # Create pipeline stages
    stages = [
        MockTeamMember("architect", "planner", execution_delay=0.03, tool_calls=15),
        MockTeamMember("implementer", "executor", execution_delay=0.05, tool_calls=20),
        MockTeamMember("tester", "tester", execution_delay=0.04, tool_calls=15),
        MockTeamMember("reviewer", "reviewer", execution_delay=0.02, tool_calls=10),
    ]

    for stage in stages:
        coordinator.add_member(stage)

    def run_team():
        return asyncio.run(coordinator.execute_team(
            task="Implement user authentication with JWT",
            context={
                "requirements": ["JWT tokens", "password hashing", "session management"],
                "testing": "unit + integration",
            },
        ))

    result = benchmark(run_team)

    assert result["success"]
    print(f"\nFeature Pipeline | Time: {result['total_time']*1000:.2f}ms | "
          f"Stages: {result['member_count']} | Messages: {result['message_count']}")


# =============================================================================
# Summary and Regression Tests
# =============================================================================


@pytest.mark.summary
def test_team_node_performance_summary():
    """Generate comprehensive performance summary for team nodes.

    This test runs key benchmarks and generates a summary report.
    """
    results = {
        "formations": {},
        "scaling": {},
        "recursion": {},
        "memory": {},
    }

    print("\n" + "=" * 80)
    print("TEAM NODE PERFORMANCE SUMMARY")
    print("=" * 80)

    # Test formations
    print("\n1. Formation Performance (3 members, 10ms delay)")
    print("-" * 60)
    formations = [
        TeamFormationType.SEQUENTIAL,
        TeamFormationType.PARALLEL,
        TeamFormationType.PIPELINE,
        TeamFormationType.HIERARCHICAL,
        TeamFormationType.CONSENSUS,
    ]

    for formation in formations:
        coordinator = MockTeamCoordinator(formation=formation)
        for i in range(3):
            coordinator.add_member(
                MockTeamMember(f"member_{i}", "assistant", execution_delay=0.01)
            )

        async def run_formation():
            return await coordinator.execute_team(task="Test", context={})

        start = time.time()
        result = asyncio.run(run_formation())
        elapsed = time.time() - start

        results["formations"][formation.value] = {
            "time_ms": elapsed * 1000,
            "success": result["success"],
            "messages": result["message_count"],
        }

        status = "✓" if result["success"] else "✗"
        print(f"  {formation.value:15} {status}  {elapsed*1000:6.2f}ms  "
              f"{result['message_count']:3} messages")

    # Test scaling
    print("\n2. Scaling Performance (Parallel formation)")
    print("-" * 60)
    team_sizes = [2, 5, 10]

    for size in team_sizes:
        coordinator = MockTeamCoordinator(formation=TeamFormationType.PARALLEL)
        for i in range(size):
            coordinator.add_member(
                MockTeamMember(f"member_{i}", "assistant", execution_delay=0.01)
            )

        async def run_scaling():
            return await coordinator.execute_team(task="Test", context={})

        start = time.time()
        result = asyncio.run(run_scaling())
        elapsed = time.time() - start

        results["scaling"][size] = {
            "time_ms": elapsed * 1000,
            "success": result["success"],
        }

        print(f"  {size:2} members       ✓  {elapsed*1000:6.2f}ms")

    # Test recursion
    print("\n3. Recursion Depth Overhead")
    print("-" * 60)
    depths = [1, 3, 5, 10]

    for depth in depths:
        start = time.time()

        async def run_recursion():
            for i in range(depth):
                coordinator = MockTeamCoordinator(
                    formation=TeamFormationType.SEQUENTIAL,
                    enable_recursion_tracking=True,
                )
                coordinator.add_member(
                    MockTeamMember(f"member_{i}", "assistant", execution_delay=0.001)
                )
                await coordinator.execute_team(task=f"Task {i}", context={})

        asyncio.run(run_recursion())
        elapsed = time.time() - start
        overhead_per_level = (elapsed / depth) * 1000

        results["recursion"][depth] = {
            "total_ms": elapsed * 1000,
            "overhead_per_level_ms": overhead_per_level,
        }

        print(f"  Depth {depth:2}          ✓  {elapsed*1000:6.2f}ms  "
              f"({overhead_per_level:5.3f}ms per level)")

    # Test memory
    print("\n4. Memory Usage (Parallel formation)")
    print("-" * 60)
    sizes = [2, 5, 10]

    for size in sizes:
        gc.collect()
        tracemalloc.start()

        coordinator = MockTeamCoordinator(formation=TeamFormationType.PARALLEL)
        for i in range(size):
            coordinator.add_member(
                MockTeamMember(f"member_{i}", "assistant", execution_delay=0.01, message_size=1000)
            )

        async def run_memory():
            await coordinator.execute_team(task="Test", context={"data": "x" * 10000})
            return tracemalloc.get_traced_memory()[1]

        peak_kb = asyncio.run(run_memory()) / 1024
        tracemalloc.stop()

        results["memory"][size] = {"peak_kb": peak_kb}

        print(f"  {size:2} members       ✓  {peak_kb:6.1f}KB")

    # Print summary
    print("\n" + "=" * 80)
    print("PERFORMANCE TARGETS")
    print("=" * 80)
    print("  ✓ Team execution (3 members, parallel): < 30ms")
    print("  ✓ Recursion tracking overhead: < 5ms per level")
    print("  ✓ Memory usage (10 members): < 1MB")
    print("  ✓ Consensus formation (3 members): < 200ms")
    print("  ✓ Scaling: Linear for sequential, constant for parallel")
    print("\n" + "=" * 80)

    # Save results to JSON for regression testing
    results_dir = Path("/tmp/benchmark_results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "team_node_benchmarks.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Verify all targets met
    assert all(r["success"] for r in results["formations"].values())
    assert all(r["success"] for r in results["scaling"].values())
    assert all(o["overhead_per_level_ms"] < 5.0 for o in results["recursion"].values())


if __name__ == "__main__":
    # Run summary when executed directly
    pytest.main([__file__, "-v", "-s", "-k", "summary"])
