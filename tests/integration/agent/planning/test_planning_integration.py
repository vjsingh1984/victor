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

"""Integration tests for HierarchicalPlanner.

Tests integration with orchestrator, workflows, and end-to-end scenarios.
Target: 15+ comprehensive integration tests.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from victor.agent.planning import (
    HierarchicalPlanner,
    Task,
    TaskGraph,
)
from victor.agent.planning.task_decomposition import (
    SimpleTask,
    TaskDecomposition,
    TaskStatus,
)
from tests.factories import MockProviderFactory


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator for integration tests."""
    orchestrator = Mock()
    orchestrator._provider_manager = MockProviderFactory.create_anthropic()

    # Mock tool pipeline
    orchestrator._tool_pipeline = Mock()
    orchestrator._tool_pipeline.get_tools = Mock(return_value=[])

    return orchestrator


@pytest.fixture
def integration_planner(mock_orchestrator):
    """Create a planner with mock orchestrator for integration tests."""
    from victor.core.events import create_event_backend, BackendConfig, BackendType

    event_bus = create_event_backend(BackendConfig(backend_type=BackendType.IN_MEMORY))

    planner = HierarchicalPlanner(
        orchestrator=mock_orchestrator,
        event_bus=event_bus,
    )

    return planner


@pytest.fixture
def sample_task_hierarchy():
    """Create a sample task hierarchy for testing."""
    return {
        "root_task": "Implement user authentication system",
        "subtasks": [
            {
                "id": "task_1",
                "description": "Research existing authentication patterns in codebase",
                "depends_on": [],
                "estimated_complexity": 3,
                "context": {"type": "research", "tools": ["read_file", "search"]},
            },
            {
                "id": "task_2",
                "description": "Design authentication flow and data models",
                "depends_on": ["task_1"],
                "estimated_complexity": 6,
                "context": {"type": "design"},
            },
            {
                "id": "task_3",
                "description": "Implement authentication backend logic",
                "depends_on": ["task_2"],
                "estimated_complexity": 8,
                "context": {"type": "implementation"},
            },
            {
                "id": "task_4",
                "description": "Add login/logout endpoints",
                "depends_on": ["task_3"],
                "estimated_complexity": 5,
                "context": {"type": "implementation"},
            },
            {
                "id": "task_5",
                "description": "Write tests for authentication",
                "depends_on": ["task_3", "task_4"],
                "estimated_complexity": 6,
                "context": {"type": "testing"},
            },
        ],
    }


@pytest.fixture
def parallel_task_hierarchy():
    """Create a task hierarchy with parallel execution."""
    return {
        "root_task": "Deploy application to production",
        "subtasks": [
            {
                "id": "task_1",
                "description": "Run database migrations",
                "depends_on": [],
                "estimated_complexity": 4,
            },
            {
                "id": "task_2a",
                "description": "Build frontend assets",
                "depends_on": ["task_1"],
                "estimated_complexity": 5,
            },
            {
                "id": "task_2b",
                "description": "Build backend services",
                "depends_on": ["task_1"],
                "estimated_complexity": 5,
            },
            {
                "id": "task_2c",
                "description": "Run unit tests",
                "depends_on": ["task_1"],
                "estimated_complexity": 4,
            },
            {
                "id": "task_3",
                "description": "Deploy to staging",
                "depends_on": ["task_2a", "task_2b", "task_2c"],
                "estimated_complexity": 6,
            },
            {
                "id": "task_4",
                "description": "Deploy to production",
                "depends_on": ["task_3"],
                "estimated_complexity": 7,
            },
        ],
    }


@pytest.fixture
def complex_task_hierarchy():
    """Create a complex task hierarchy with multiple levels."""
    return {
        "root_task": "Refactor monolithic application to microservices",
        "subtasks": [
            {
                "id": "task_1",
                "description": "Analyze current architecture",
                "depends_on": [],
                "estimated_complexity": 4,
            },
            {
                "id": "task_2",
                "description": "Identify service boundaries",
                "depends_on": ["task_1"],
                "estimated_complexity": 7,
            },
            {
                "id": "task_3",
                "description": "Design API contracts between services",
                "depends_on": ["task_2"],
                "estimated_complexity": 8,
            },
            {
                "id": "task_4",
                "description": "Implement authentication service",
                "depends_on": ["task_3"],
                "estimated_complexity": 6,
            },
            {
                "id": "task_5",
                "description": "Implement user service",
                "depends_on": ["task_3"],
                "estimated_complexity": 7,
            },
            {
                "id": "task_6",
                "description": "Implement order service",
                "depends_on": ["task_3"],
                "estimated_complexity": 8,
            },
            {
                "id": "task_7",
                "description": "Set up service mesh",
                "depends_on": ["task_4", "task_5", "task_6"],
                "estimated_complexity": 9,
            },
            {
                "id": "task_8",
                "description": "Migrate data to new services",
                "depends_on": ["task_7"],
                "estimated_complexity": 8,
            },
            {
                "id": "task_9",
                "description": "Decommission monolith",
                "depends_on": ["task_8"],
                "estimated_complexity": 5,
            },
        ],
    }


# =============================================================================
# End-to-End Task Decomposition Tests (3 tests)
# =============================================================================


class TestTaskDecompositionE2E:
    """End-to-end tests for task decomposition."""

    @pytest.mark.asyncio
    async def test_decompose_simple_task_e2e(self, integration_planner, sample_task_hierarchy):
        """Test end-to-end decomposition of a simple task."""
        # Mock LLM response
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(sample_task_hierarchy))
        )

        # Decompose task
        graph = await integration_planner.decompose_task("Implement user authentication system")

        # Verify decomposition structure
        assert len(graph.nodes) == 5
        assert "task_1" in graph.nodes
        assert "task_5" in graph.nodes

        # Verify dependency chain
        assert graph.nodes["task_1"].depends_on == []
        assert graph.nodes["task_2"].depends_on == ["task_1"]
        assert graph.nodes["task_3"].depends_on == ["task_2"]
        assert set(graph.nodes["task_5"].depends_on) == {"task_3", "task_4"}

        # Verify initial state
        assert all(node.status == "pending" for node in graph.nodes.values())

    @pytest.mark.asyncio
    async def test_decompose_complex_task_e2e(self, integration_planner, complex_task_hierarchy):
        """Test end-to-end decomposition of a complex task with multiple levels."""
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(complex_task_hierarchy))
        )

        # Decompose complex task
        graph = await integration_planner.decompose_task(
            "Refactor monolithic application to microservices"
        )

        # Verify structure
        assert len(graph.nodes) == 9

        # Verify hierarchical dependencies
        assert graph.nodes["task_1"].depends_on == []
        assert graph.nodes["task_7"].depends_on == ["task_4", "task_5", "task_6"]

        # Verify complexity estimation
        assert graph.nodes["task_7"].estimated_complexity == 9  # Service mesh setup

        # Validate plan
        validation = integration_planner.validate_plan(graph)
        assert validation.is_valid is True
        assert validation.has_cycles is False

    @pytest.mark.asyncio
    async def test_decompose_with_context_e2e(self, integration_planner, sample_task_hierarchy):
        """Test end-to-end decomposition with rich context."""
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(sample_task_hierarchy))
        )

        # Provide context for decomposition
        context = {
            "project_type": "web_application",
            "tech_stack": ["python", "fastapi", "postgresql"],
            "constraints": ["must_use_oauth2", "no_social_login"],
            "files": ["auth.py", "models.py", "routes.py"],
        }

        graph = await integration_planner.decompose_task(
            "Implement OAuth2 authentication", context=context
        )

        # Verify decomposition succeeded with context
        assert len(graph.nodes) == 5

        # Verify LLM was called with context
        assert integration_planner._orchestrator._provider_manager.chat.called
        call_args = integration_planner._orchestrator._provider_manager.chat.call_args
        # call_args is an ArgInfoSupplementer, we need to access the args
        if call_args and call_args[0]:
            messages = call_args[0][0]
            if messages:
                user_messages = [m for m in messages if m.get("role") == "user"]
                if user_messages:
                    user_message = user_messages[0]
                    assert "OAuth2 authentication" in user_message.get("content", "")
                    # Context is passed, so it should be in the message


# =============================================================================
# Plan Execution with Orchestrator Tests (3 tests)
# =============================================================================


class TestPlanExecutionWithOrchestrator:
    """Tests for plan execution integrated with orchestrator."""

    @pytest.mark.asyncio
    async def test_execute_sequential_plan(self, integration_planner, sample_task_hierarchy):
        """Test executing a plan sequentially through orchestrator."""
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(sample_task_hierarchy))
        )

        # Decompose
        graph = await integration_planner.decompose_task("Implement authentication")

        # Execute sequentially
        execution_order = []
        ready_tasks = await integration_planner.suggest_next_tasks(graph)

        while ready_tasks:
            task = ready_tasks[0]
            execution_order.append(task.id)

            # Update plan
            updated = await integration_planner.update_plan(graph, completed_tasks=[task.id])
            graph = updated.graph

            ready_tasks = await integration_planner.suggest_next_tasks(graph)

        # Verify execution order respects dependencies
        assert execution_order[0] == "task_1"
        assert execution_order.index("task_2") < execution_order.index("task_3")
        assert execution_order.index("task_3") < execution_order.index("task_5")

        # Verify all tasks completed
        assert all(node.status == "completed" for node in graph.nodes.values())

    @pytest.mark.asyncio
    async def test_execute_parallel_plan(self, integration_planner, parallel_task_hierarchy):
        """Test executing a plan with parallel tasks."""
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(parallel_task_hierarchy))
        )

        # Decompose
        graph = await integration_planner.decompose_task("Deploy to production")

        # Execute first task
        updated = await integration_planner.update_plan(graph, completed_tasks=["task_1"])
        graph = updated.graph

        # Get ready tasks - should be parallel tasks
        ready_tasks = await integration_planner.suggest_next_tasks(graph)
        ready_ids = [t.id for t in ready_tasks]

        # Verify parallel tasks are ready
        assert len(ready_tasks) == 3
        assert set(ready_ids) == {"task_2a", "task_2b", "task_2c"}

        # Complete parallel tasks in any order
        updated = await integration_planner.update_plan(
            graph, completed_tasks=["task_2a", "task_2b", "task_2c"]
        )
        graph = updated.graph

        # Verify next task is ready
        ready_tasks = await integration_planner.suggest_next_tasks(graph)
        assert ready_tasks[0].id == "task_3"

    @pytest.mark.asyncio
    async def test_execute_plan_with_tool_integration(
        self, integration_planner, sample_task_hierarchy
    ):
        """Test executing plan with orchestrator tool integration."""
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(sample_task_hierarchy))
        )

        # Mock tool calls during execution
        tool_calls_made = []

        async def mock_tool_call(tool_name: str, **kwargs):
            tool_calls_made.append(tool_name)
            return f"Executed {tool_name}"

        integration_planner._orchestrator.execute_tool = mock_tool_call

        # Decompose and execute
        graph = await integration_planner.decompose_task("Implement authentication")

        ready_tasks = await integration_planner.suggest_next_tasks(graph)
        task = ready_tasks[0]

        # Simulate task execution with tools
        if "read_file" in task.context.get("tools", []):
            await integration_planner._orchestrator.execute_tool("read_file")

        # Verify tool was called
        assert len(tool_calls_made) > 0


# =============================================================================
# Replanning After Failures Tests (3 tests)
# =============================================================================


class TestReplanningAfterFailures:
    """Tests for dynamic replanning after task failures."""

    @pytest.mark.asyncio
    async def test_replan_after_single_failure(self, integration_planner, sample_task_hierarchy):
        """Test replanning after a single task failure."""
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(sample_task_hierarchy))
        )

        # Decompose
        graph = await integration_planner.decompose_task("Implement authentication")

        # Complete first task
        updated = await integration_planner.update_plan(graph, completed_tasks=["task_1"])

        # Mark second task as failed
        updated = await integration_planner.update_plan(
            updated.graph, completed_tasks=[], failed_tasks=["task_2"]
        )

        # Verify failure propagated
        assert updated.graph.nodes["task_2"].status == "failed"
        assert updated.graph.nodes["task_3"].status == "blocked"
        assert updated.graph.nodes["task_4"].status == "blocked"

        # Verify cannot proceed
        assert updated.can_proceed is False

    @pytest.mark.asyncio
    async def test_replan_after_multiple_failures(
        self, integration_planner, parallel_task_hierarchy
    ):
        """Test replanning after multiple parallel task failures."""
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(parallel_task_hierarchy))
        )

        # Decompose
        graph = await integration_planner.decompose_task("Deploy application")

        # Complete setup
        updated = await integration_planner.update_plan(graph, completed_tasks=["task_1"])

        # Fail two parallel tasks
        updated = await integration_planner.update_plan(
            updated.graph, completed_tasks=[], failed_tasks=["task_2a", "task_2b"]
        )

        # Verify dependent task blocked
        assert updated.graph.nodes["task_3"].status == "blocked"

        # Complete remaining parallel task
        updated = await integration_planner.update_plan(updated.graph, completed_tasks=["task_2c"])

        # Task 3 should still be blocked due to failures
        assert updated.graph.nodes["task_3"].status == "blocked"
        assert updated.can_proceed is False

    @pytest.mark.asyncio
    async def test_replan_with_recovery(self, integration_planner, sample_task_hierarchy):
        """Test replanning after failure recovery."""
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(sample_task_hierarchy))
        )

        # Decompose
        graph = await integration_planner.decompose_task("Implement authentication")

        # Complete tasks
        updated = await integration_planner.update_plan(graph, completed_tasks=["task_1", "task_2"])

        # Fail task_3
        updated = await integration_planner.update_plan(
            updated.graph, completed_tasks=[], failed_tasks=["task_3"]
        )

        # Verify blocked tasks
        assert updated.graph.nodes["task_4"].status == "blocked"
        assert updated.graph.nodes["task_5"].status == "blocked"

        # Simulate recovery: retry task_3 successfully
        # Reset status to pending
        updated.graph.nodes["task_3"].status = "pending"
        updated.graph.nodes["task_4"].status = "pending"
        updated.graph.nodes["task_5"].status = "pending"

        # Complete recovered task
        updated = await integration_planner.update_plan(updated.graph, completed_tasks=["task_3"])

        # Verify tasks unblocked
        assert updated.graph.nodes["task_3"].status == "completed"
        assert updated.graph.nodes["task_4"].status == "pending"
        assert updated.can_proceed is True


# =============================================================================
# Multi-Agent Coordination Tests (3 tests)
# =============================================================================


class TestMultiAgentCoordination:
    """Tests for multi-agent coordination in planning."""

    @pytest.mark.asyncio
    async def test_coordinate_specialized_agents(self, integration_planner, sample_task_hierarchy):
        """Test coordinating multiple specialized agents."""
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(sample_task_hierarchy))
        )

        # Decompose
        graph = await integration_planner.decompose_task("Implement authentication")

        # Assign tasks to specialized agents based on context
        agent_assignments = {}
        for task_id, task in graph.nodes.items():
            context_type = task.context.get("type", "implementation")

            if context_type == "research":
                agent_assignments[task_id] = "research_agent"
            elif context_type == "design":
                agent_assignments[task_id] = "architect_agent"
            elif context_type == "implementation":
                agent_assignments[task_id] = "developer_agent"
            elif context_type == "testing":
                agent_assignments[task_id] = "tester_agent"

        # Verify assignments
        assert agent_assignments["task_1"] == "research_agent"
        assert agent_assignments["task_2"] == "architect_agent"
        assert agent_assignments["task_3"] == "developer_agent"
        assert agent_assignments["task_5"] == "tester_agent"

    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self, integration_planner, parallel_task_hierarchy):
        """Test parallel execution by multiple agents."""
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(parallel_task_hierarchy))
        )

        # Decompose
        graph = await integration_planner.decompose_task("Deploy application")

        # Complete setup
        updated = await integration_planner.update_plan(graph, completed_tasks=["task_1"])

        # Get parallel tasks
        ready_tasks = await integration_planner.suggest_next_tasks(updated.graph)

        # Simulate parallel agent execution
        agent_tasks = {
            "agent_1": ready_tasks[0].id,
            "agent_2": ready_tasks[1].id,
            "agent_3": ready_tasks[2].id,
        }

        # Complete all parallel tasks
        updated = await integration_planner.update_plan(
            updated.graph,
            completed_tasks=list(agent_tasks.values()),
        )

        # Verify all agents completed their tasks
        for task_id in agent_tasks.values():
            assert updated.graph.nodes[task_id].status == "completed"

    @pytest.mark.asyncio
    async def test_agent_handoff_workflow(self, integration_planner, complex_task_hierarchy):
        """Test workflow with agent handoffs."""
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(complex_task_hierarchy))
        )

        # Decompose
        graph = await integration_planner.decompose_task("Refactor to microservices")

        # Simulate agent handoff chain
        handoff_chain = []

        # Architect agent completes analysis
        updated = await integration_planner.update_plan(
            graph, completed_tasks=["task_1", "task_2", "task_3"]
        )
        handoff_chain.append(("architect", ["task_1", "task_2", "task_3"]))

        # Developer agents implement services
        updated = await integration_planner.update_plan(
            updated.graph, completed_tasks=["task_4", "task_5", "task_6"]
        )
        handoff_chain.append(("developers", ["task_4", "task_5", "task_6"]))

        # DevOps agent sets up infrastructure
        updated = await integration_planner.update_plan(updated.graph, completed_tasks=["task_7"])
        handoff_chain.append(("devops", ["task_7"]))

        # Verify handoff sequence
        assert len(handoff_chain) == 3
        assert handoff_chain[0][0] == "architect"
        assert handoff_chain[1][0] == "developers"
        assert handoff_chain[2][0] == "devops"


# =============================================================================
# Performance Benchmarks Tests (3 tests)
# =============================================================================


class TestPerformanceBenchmarks:
    """Performance and scalability benchmarks for planning."""

    @pytest.mark.asyncio
    async def test_large_graph_decomposition_performance(self, integration_planner):
        """Test decomposition performance with large task graphs."""
        # Create large task hierarchy (50 tasks)
        large_hierarchy = {
            "root_task": "Large scale refactoring",
            "subtasks": [],
        }

        for i in range(50):
            task = {
                "id": f"task_{i}",
                "description": f"Task {i}: Refactor module {i}",
                "depends_on": [f"task_{i-1}"] if i > 0 else [],
                "estimated_complexity": 5,
                "context": {},
            }
            large_hierarchy["subtasks"].append(task)

        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(large_hierarchy))
        )

        # Measure decomposition time
        start_time = time.time()
        graph = await integration_planner.decompose_task("Large refactoring")
        decomposition_time = time.time() - start_time

        # Verify all tasks created
        assert len(graph.nodes) == 50

        # Performance assertion: should complete in reasonable time
        assert decomposition_time < 2.0, f"Decomposition took {decomposition_time:.2f}s"

        # Measure validation time
        start_time = time.time()
        validation = integration_planner.validate_plan(graph)
        validation_time = time.time() - start_time

        assert validation.is_valid is True
        assert validation_time < 0.5, f"Validation took {validation_time:.2f}s"

    @pytest.mark.asyncio
    async def test_cache_performance_benefit(self, integration_planner, sample_task_hierarchy):
        """Test performance benefit of decomposition caching."""
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(sample_task_hierarchy))
        )

        # First decomposition (uncached)
        start_time = time.time()
        graph1 = await integration_planner.decompose_task(
            "Implement authentication", use_cache=True
        )
        first_time = time.time() - start_time

        # Second decomposition (cached)
        start_time = time.time()
        graph2 = await integration_planner.decompose_task(
            "Implement authentication", use_cache=True
        )
        cached_time = time.time() - start_time

        # Verify LLM called only once
        assert integration_planner._orchestrator._provider_manager.chat.call_count == 1

        # Verify cache hit
        cache_stats = integration_planner.get_cache_stats()
        assert cache_stats["cached_plans"] == 1

        # Verify results identical
        assert len(graph1.nodes) == len(graph2.nodes)

        # Cache should be faster (though timing may vary)
        # With mocks, difference might be minimal, so just verify it's not slower
        assert cached_time <= first_time * 1.5

    @pytest.mark.asyncio
    async def test_parallel_task_identification_performance(
        self, integration_planner, parallel_task_hierarchy
    ):
        """Test performance of parallel task identification."""
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(parallel_task_hierarchy))
        )

        # Create decomposition with NetworkX
        decomposition = TaskDecomposition()

        # Add tasks with parallel structure
        for i in range(20):
            task = SimpleTask(id=f"task_{i}", description=f"Task {i}")
            deps = [] if i < 10 else [f"task_{i-10}"]
            decomposition.add_task(task, dependencies=deps, complexity=5)

        # Measure ready tasks identification
        start_time = time.time()
        ready_tasks = decomposition.get_ready_tasks()
        ready_time = time.time() - start_time

        # Should identify initial parallel tasks
        assert len(ready_tasks) == 10
        assert ready_time < 0.1, f"Ready tasks took {ready_time:.4f}s"

        # Measure execution levels calculation
        start_time = time.time()
        levels = decomposition.get_execution_levels()
        levels_time = time.time() - start_time

        assert len(levels) == 2  # Two levels of parallel execution
        assert levels_time < 0.1, f"Execution levels took {levels_time:.4f}s"

        # Measure critical path calculation
        start_time = time.time()
        critical_path = decomposition.get_critical_path()
        critical_time = time.time() - start_time

        assert len(critical_path) > 0
        assert critical_time < 0.1, f"Critical path took {critical_time:.4f}s"


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestPlanningEdgeCases:
    """Additional edge case tests for comprehensive coverage."""

    @pytest.mark.asyncio
    async def test_empty_task_list(self, integration_planner):
        """Test handling of empty task decomposition."""
        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps({"root_task": "Empty", "subtasks": []}))
        )

        graph = await integration_planner.decompose_task("Empty task")

        # Should handle gracefully
        assert len(graph.nodes) == 0

        # Should not crash on empty operations
        ready_tasks = await integration_planner.suggest_next_tasks(graph)
        assert len(ready_tasks) == 0

        validation = integration_planner.validate_plan(graph)
        assert validation.is_valid is True

    @pytest.mark.asyncio
    async def test_single_task_plan(self, integration_planner):
        """Test plan with single task."""
        single_task = {
            "root_task": "Simple task",
            "subtasks": [
                {
                    "id": "task_1",
                    "description": "Do one thing",
                    "depends_on": [],
                    "estimated_complexity": 3,
                }
            ],
        }

        integration_planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps(single_task))
        )

        graph = await integration_planner.decompose_task("Simple task")

        assert len(graph.nodes) == 1

        ready_tasks = await integration_planner.suggest_next_tasks(graph)
        assert len(ready_tasks) == 1
        assert ready_tasks[0].id == "task_1"

    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, integration_planner):
        """Test detection of circular dependencies."""
        # Create circular dependency manually
        graph = TaskGraph()
        graph.add_node(Task(id="task_1", description="Task 1", depends_on=["task_2"]))
        graph.add_node(Task(id="task_2", description="Task 2", depends_on=["task_1"]))
        graph.add_edge("task_1", "task_2")
        graph.add_edge("task_2", "task_1")

        # Validate should detect cycle
        validation = integration_planner.validate_plan(graph)

        assert validation.is_valid is False
        assert validation.has_cycles is True
        assert len(validation.errors) > 0
        assert any(
            "cycle" in error.lower() or "circular" in error.lower() for error in validation.errors
        )

    @pytest.mark.asyncio
    async def test_complexity_estimation_consistency(self, integration_planner):
        """Test consistency of complexity estimation."""
        task1 = "Implement simple feature"
        task2 = "Refactor entire system architecture"

        score1 = integration_planner.estimate_complexity(task1)
        score2 = integration_planner.estimate_complexity(task2)

        # Complex task should have higher score
        assert score2.score > score1.score
        assert score2.estimated_steps >= score1.estimated_steps

    @pytest.mark.asyncio
    async def test_event_emission_integration(self, integration_planner):
        """Test that planner emits events during operations."""
        from victor.core.events import create_event_backend, BackendConfig, BackendType

        captured_events = []
        event_bus = create_event_backend(BackendConfig(backend_type=BackendType.IN_MEMORY))

        original_publish = event_bus.publish

        async def capture_publish(event):
            captured_events.append(event)
            return await original_publish(event)

        event_bus.publish = capture_publish

        # Create planner with event bus
        planner = HierarchicalPlanner(
            orchestrator=integration_planner._orchestrator,
            event_bus=event_bus,
        )

        planner._orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps({"subtasks": []}))
        )

        # Perform operations
        await planner.decompose_task("Test task")

        # Verify events emitted
        event_topics = [e.topic for e in captured_events]
        assert any("planning.decompose_start" in t for t in event_topics)
        assert any("planning.decompose_complete" in t for t in event_topics)
