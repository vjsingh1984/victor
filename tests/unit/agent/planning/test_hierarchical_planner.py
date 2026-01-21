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

"""Unit tests for HierarchicalPlanner.

Tests task decomposition, planning, and complexity estimation.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from victor.agent.planning import (
    ComplexityScore,
    HierarchicalPlanner,
    Task,
    TaskGraph,
    UpdatedPlan,
    ValidationResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    event_bus = AsyncMock()
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def mock_provider_manager():
    """Create a mock provider manager."""
    provider = Mock()
    provider.name = "anthropic"
    provider.model = "claude-sonnet-4-5"

    response = Mock()
    response.content = "Test response"
    response.role = "assistant"
    response.tool_calls = None

    provider_manager = AsyncMock()
    provider_manager.chat = AsyncMock(return_value=response)
    provider_manager.model = "claude-sonnet-4-5"

    return provider_manager


@pytest.fixture
def mock_orchestrator(mock_provider_manager):
    """Create a mock orchestrator."""
    orchestrator = Mock()
    orchestrator._provider_manager = mock_provider_manager
    return orchestrator


@pytest.fixture
def planner(mock_orchestrator, mock_event_bus):
    """Create a HierarchicalPlanner with mock dependencies."""
    return HierarchicalPlanner(
        orchestrator=mock_orchestrator,
        event_bus=mock_event_bus,
    )


@pytest.fixture
def sample_tasks():
    """Create sample tasks for testing."""
    return [
        Task(
            id="task_1",
            description="Research existing code",
            depends_on=[],
            estimated_complexity=3,
        ),
        Task(
            id="task_2",
            description="Design new module",
            depends_on=["task_1"],
            estimated_complexity=6,
        ),
        Task(
            id="task_3",
            description="Implement module",
            depends_on=["task_2"],
            estimated_complexity=8,
        ),
    ]


@pytest.fixture
def sample_graph(sample_tasks):
    """Create a sample task graph."""
    graph = TaskGraph()
    for task in sample_tasks:
        graph.add_node(task)
    for task in sample_tasks:
        for dep_id in task.depends_on:
            graph.add_edge(task.id, dep_id)
    return graph


# =============================================================================
# Task Decomposition Tests
# =============================================================================


class TestTaskDecomposition:
    """Tests for task decomposition functionality."""

    @pytest.mark.asyncio
    async def test_decompose_simple_task(self, planner, mock_provider_manager):
        """Test decomposing a simple task."""
        # Mock LLM response
        decomposition_response = json.dumps(
            {
                "root_task": "Implement feature",
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": "First step",
                        "depends_on": [],
                        "estimated_complexity": 3,
                    }
                ],
            }
        )

        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=decomposition_response)
        )

        # Decompose task
        graph = await planner.decompose_task("Implement a simple feature")

        # Verify
        assert len(graph.nodes) == 1
        assert "task_1" in graph.nodes
        assert graph.nodes["task_1"].description == "First step"

    @pytest.mark.asyncio
    async def test_decompose_complex_task(self, planner, mock_provider_manager):
        """Test decomposing a complex task with dependencies."""
        decomposition_response = json.dumps(
            {
                "root_task": "Implement authentication",
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": "Research existing auth",
                        "depends_on": [],
                        "estimated_complexity": 3,
                    },
                    {
                        "id": "task_2",
                        "description": "Design auth flow",
                        "depends_on": ["task_1"],
                        "estimated_complexity": 6,
                    },
                    {
                        "id": "task_3",
                        "description": "Implement auth",
                        "depends_on": ["task_2"],
                        "estimated_complexity": 8,
                    },
                ],
            }
        )

        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=decomposition_response)
        )

        # Decompose task
        graph = await planner.decompose_task("Implement user authentication")

        # Verify structure
        assert len(graph.nodes) == 3
        assert graph.nodes["task_1"].depends_on == []
        assert graph.nodes["task_2"].depends_on == ["task_1"]
        assert graph.nodes["task_3"].depends_on == ["task_2"]

    @pytest.mark.asyncio
    async def test_decompose_with_cache(self, planner, mock_provider_manager):
        """Test that decomposition results are cached."""
        decomposition_response = json.dumps(
            {
                "root_task": "Test task",
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": "First step",
                        "depends_on": [],
                        "estimated_complexity": 5,
                    }
                ],
            }
        )

        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=decomposition_response)
        )

        # First call - should hit LLM
        graph1 = await planner.decompose_task("Test task", use_cache=True)
        assert mock_provider_manager.chat.call_count == 1

        # Second call - should use cache
        graph2 = await planner.decompose_task("Test task", use_cache=True)
        assert mock_provider_manager.chat.call_count == 1  # No additional call

        # Verify same graph
        assert graph1.nodes.keys() == graph2.nodes.keys()

    @pytest.mark.asyncio
    async def test_decompose_cache_bypass(self, planner, mock_provider_manager):
        """Test bypassing cache."""
        decomposition_response = json.dumps(
            {
                "root_task": "Test task",
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": "First step",
                        "depends_on": [],
                        "estimated_complexity": 5,
                    }
                ],
            }
        )

        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=decomposition_response)
        )

        # First call
        await planner.decompose_task("Test task", use_cache=True)
        assert mock_provider_manager.chat.call_count == 1

        # Second call with cache disabled
        await planner.decompose_task("Test task", use_cache=False)
        assert mock_provider_manager.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_decompose_with_context(self, planner, mock_provider_manager):
        """Test decomposition with additional context."""
        decomposition_response = json.dumps(
            {
                "root_task": "Test task",
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": "Context-aware task",
                        "depends_on": [],
                        "estimated_complexity": 5,
                        "context": {"files": ["test.py"]},
                    }
                ],
            }
        )

        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=decomposition_response)
        )

        # Decompose with context
        context = {"file_count": 5, "domains": ["coding"]}
        graph = await planner.decompose_task("Test task", context=context)

        # Verify LLM was called
        assert mock_provider_manager.chat.called

        # Verify context in task
        assert graph.nodes["task_1"].context == {"files": ["test.py"]}

    @pytest.mark.asyncio
    async def test_decompose_invalid_json_response(self, planner, mock_provider_manager):
        """Test handling of invalid JSON in LLM response."""
        # Mock invalid JSON response
        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content="This is not valid JSON")
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="No JSON found"):
            await planner.decompose_task("Test task")

    @pytest.mark.asyncio
    async def test_decompose_missing_subtasks_field(self, planner, mock_provider_manager):
        """Test handling of response missing subtasks field."""
        # Mock response without subtasks
        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps({"root_task": "test"}))
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="Missing 'subtasks'"):
            await planner.decompose_task("Test task")

    @pytest.mark.asyncio
    async def test_decompose_json_extraction_from_text(self, planner, mock_provider_manager):
        """Test extracting JSON from text response."""
        # Mock response with text around JSON
        decomposition_response = f"""
        Here's the task decomposition:

        {json.dumps({
            "root_task": "Test task",
            "subtasks": [
                {
                    "id": "task_1",
                    "description": "First step",
                    "depends_on": [],
                    "estimated_complexity": 5,
                }
            ]
        })}

        Let me know if you need adjustments!
        """

        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=decomposition_response)
        )

        # Should successfully extract and parse
        graph = await planner.decompose_task("Test task")
        assert len(graph.nodes) == 1


# =============================================================================
# Plan Update Tests
# =============================================================================


class TestPlanUpdate:
    """Tests for plan update functionality."""

    @pytest.mark.asyncio
    async def test_update_plan_after_completion(self, planner, sample_graph):
        """Test updating plan after tasks complete."""
        # Update with completed tasks
        result = await planner.update_plan(sample_graph, completed_tasks=["task_1"])

        # Verify
        assert sample_graph.nodes["task_1"].status == "completed"
        assert "task_1" in result.completed_tasks
        assert len(result.new_ready_tasks) == 1
        assert result.new_ready_tasks[0].id == "task_2"

    @pytest.mark.asyncio
    async def test_update_plan_with_failures(self, planner, sample_graph):
        """Test updating plan with failed tasks."""
        # Update with failed task
        result = await planner.update_plan(
            sample_graph, completed_tasks=[], failed_tasks=["task_1"]
        )

        # Verify
        assert sample_graph.nodes["task_1"].status == "failed"
        assert sample_graph.nodes["task_2"].status == "blocked"
        assert "task_1" in result.failed_tasks

    @pytest.mark.asyncio
    async def test_update_plan_multiple_completed(self, planner, sample_graph):
        """Test updating plan with multiple completed tasks."""
        # Mark first task as completed
        sample_graph.nodes["task_1"].status = "completed"

        # Update with second task completed
        result = await planner.update_plan(sample_graph, completed_tasks=["task_2"])

        # Verify
        assert sample_graph.nodes["task_2"].status == "completed"
        assert len(result.new_ready_tasks) == 1
        assert result.new_ready_tasks[0].id == "task_3"

    @pytest.mark.asyncio
    async def test_update_plan_cannot_proceed(self, planner, sample_graph):
        """Test plan update when execution cannot proceed."""
        # Create graph with single task
        graph = TaskGraph()
        graph.add_node(Task(id="task_1", description="Only task"))

        # Mark as failed
        result = await planner.update_plan(
            graph, completed_tasks=[], failed_tasks=["task_1"]
        )

        # Verify
        assert result.can_proceed is False

    @pytest.mark.asyncio
    async def test_update_plan_invalid_task_id(self, planner, sample_graph):
        """Test updating plan with non-existent task ID."""
        # Should raise ValueError
        with pytest.raises(ValueError, match="not found in graph"):
            await planner.update_plan(sample_graph, completed_tasks=["nonexistent"])

    @pytest.mark.asyncio
    async def test_update_plan_event_emission(self, planner, sample_graph, mock_event_bus):
        """Test that plan update emits events."""
        await planner.update_plan(sample_graph, completed_tasks=["task_1"])

        # Verify event was emitted
        assert mock_event_bus.publish.called
        topic = mock_event_bus.publish.call_args[0][0].topic
        assert "planning.update_plan" in topic


# =============================================================================
# Task Suggestion Tests
# =============================================================================


class TestTaskSuggestion:
    """Tests for task suggestion functionality."""

    @pytest.mark.asyncio
    async def test_suggest_next_tasks_ready(self, planner, sample_graph):
        """Test suggesting next ready tasks."""
        # All tasks pending initially
        tasks = await planner.suggest_next_tasks(sample_graph)

        # Only task_1 should be ready (no dependencies)
        assert len(tasks) == 1
        assert tasks[0].id == "task_1"

    @pytest.mark.asyncio
    async def test_suggest_next_tasks_after_completion(self, planner, sample_graph):
        """Test suggesting tasks after some complete."""
        # Mark task_1 as completed
        sample_graph.nodes["task_1"].status = "completed"

        # Get suggestions
        tasks = await planner.suggest_next_tasks(sample_graph)

        # task_2 should now be ready
        assert len(tasks) == 1
        assert tasks[0].id == "task_2"

    @pytest.mark.asyncio
    async def test_suggest_next_tasks_sorted_by_complexity(self, planner):
        """Test that tasks are sorted by complexity."""
        # Create graph with parallel tasks
        graph = TaskGraph()
        graph.add_node(Task(id="task_1", description="Low complexity", estimated_complexity=2))
        graph.add_node(Task(id="task_2", description="High complexity", estimated_complexity=8))
        graph.add_node(Task(id="task_3", description="Medium complexity", estimated_complexity=5))

        # Get suggestions
        tasks = await planner.suggest_next_tasks(graph)

        # Should be sorted by complexity (descending)
        assert tasks[0].id == "task_2"  # Highest complexity
        assert tasks[1].id == "task_3"
        assert tasks[2].id == "task_1"  # Lowest complexity

    @pytest.mark.asyncio
    async def test_suggest_next_tasks_no_ready_tasks(self, planner, sample_graph):
        """Test suggesting when no tasks are ready."""
        # Mark all as blocked/in-progress
        for task in sample_graph.nodes.values():
            task.status = "in_progress"

        # Get suggestions
        tasks = await planner.suggest_next_tasks(sample_graph)

        # Should be empty
        assert len(tasks) == 0


# =============================================================================
# Complexity Estimation Tests
# =============================================================================


class TestComplexityEstimation:
    """Tests for complexity estimation functionality."""

    def test_estimate_complexity_simple_task(self, planner):
        """Test estimating complexity of a simple task."""
        score = planner.estimate_complexity("List all files in the directory")

        # Should be low complexity
        assert score.score < 5
        assert len(score.factors) > 0
        assert score.confidence > 0.5

    def test_estimate_complexity_complex_task(self, planner):
        """Test estimating complexity of a complex task."""
        score = planner.estimate_complexity(
            "Refactor the authentication system to use OAuth2"
        )

        # Should be high complexity
        assert score.score >= 6
        assert "Complex keywords" in str(score.factors)
        assert score.estimated_steps > 5

    def test_estimate_complexity_with_context(self, planner):
        """Test complexity estimation with context."""
        context = {
            "file_count": 20,
            "lines_of_code": 5000,
            "domains": ["coding", "devops"],
        }

        score = planner.estimate_complexity("Implement feature", context=context)

        # Context should increase complexity
        assert score.score > 5
        assert score.estimated_steps > 5

    def test_estimate_complexity_clamps_to_range(self, planner):
        """Test that complexity scores are clamped to 1-10 range."""
        # Very simple
        score1 = planner.estimate_complexity("list files")
        assert 1 <= score1.score <= 10

        # Very complex
        score2 = planner.estimate_complexity(
            "refactor architecture and migrate system while implementing design"
        )
        assert 1 <= score2.score <= 10

    def test_estimate_complexity_confidence_calculation(self, planner):
        """Test confidence calculation based on factors."""
        score = planner.estimate_complexity("Implement a feature")

        # More factors = higher confidence
        expected_confidence = 0.5 + (len(score.factors) * 0.1)
        assert score.confidence == min(0.9, expected_confidence)

    def test_estimate_complexity_estimated_steps(self, planner):
        """Test estimated steps calculation."""
        score = planner.estimate_complexity("Test task")

        # Steps should be based on complexity
        expected_steps = int(score.score * 1.5) + 2
        assert score.estimated_steps == expected_steps


# =============================================================================
# Plan Validation Tests
# =============================================================================


class TestPlanValidation:
    """Tests for plan validation functionality."""

    def test_validate_plan_valid_graph(self, planner, sample_graph):
        """Test validating a valid graph."""
        result = planner.validate_plan(sample_graph)

        # Should be valid
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.has_cycles is False

    def test_validate_plan_with_cycles(self, planner):
        """Test detecting cycles in graph."""
        # Create graph with cycle: task_1 -> task_2 -> task_1
        graph = TaskGraph()
        graph.add_node(Task(id="task_1", description="Task 1", depends_on=["task_2"]))
        graph.add_node(Task(id="task_2", description="Task 2", depends_on=["task_1"]))
        graph.add_edge("task_1", "task_2")
        graph.add_edge("task_2", "task_1")

        result = planner.validate_plan(graph)

        # Should detect cycle
        assert result.is_valid is False
        assert result.has_cycles is True
        assert len(result.errors) > 0

    def test_validate_plan_missing_dependency(self, planner):
        """Test detecting missing dependencies."""
        # Create task with non-existent dependency
        graph = TaskGraph()
        graph.add_node(
            Task(id="task_1", description="Task 1", depends_on=["nonexistent"])
        )

        result = planner.validate_plan(graph)

        # Should detect missing dependency
        assert result.is_valid is False
        assert any("non-existent" in error for error in result.errors)

    def test_validate_plan_warnings_orphaned_tasks(self, planner):
        """Test warning for orphaned tasks."""
        # Create graph with disconnected tasks
        graph = TaskGraph()
        graph.add_node(Task(id="root", description="Root"))
        graph.add_node(Task(id="orphan", description="Orphan"))
        graph.root_task_id = "root"

        result = planner.validate_plan(graph)

        # Should warn about orphan
        assert result.is_valid is True  # Still valid
        assert len(result.warnings) > 0
        assert any("no path from root" in warning for warning in result.warnings)

    def test_validate_plan_no_dependencies_warning(self, planner):
        """Test warning for tasks with no dependencies."""
        # Create graph where non-root task has no deps
        graph = TaskGraph()
        graph.add_node(Task(id="root", description="Root"))
        graph.add_node(Task(id="parallel", description="Parallel"))
        graph.root_task_id = "root"

        result = planner.validate_plan(graph)

        # Should warn about parallel tasks
        assert len(result.warnings) > 0


# =============================================================================
# Cache Management Tests
# =============================================================================


class TestCacheManagement:
    """Tests for cache management."""

    @pytest.mark.asyncio
    async def test_clear_cache(self, planner, mock_provider_manager):
        """Test clearing the decomposition cache."""
        # Add something to cache
        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps({"subtasks": []}))
        )
        await planner.decompose_task("Test task")

        # Verify cache has entry
        stats = planner.get_cache_stats()
        assert stats["cached_plans"] == 1

        # Clear cache
        planner.clear_cache()

        # Verify empty
        stats = planner.get_cache_stats()
        assert stats["cached_plans"] == 0

    def test_get_cache_stats(self, planner):
        """Test getting cache statistics."""
        stats = planner.get_cache_stats()

        # Should have expected fields
        assert "cached_plans" in stats
        assert "cache_keys" in stats
        assert isinstance(stats["cached_plans"], int)
        assert isinstance(stats["cache_keys"], list)


# =============================================================================
# LLM Integration Tests
# =============================================================================


class TestLLMIntegration:
    """Tests for LLM integration."""

    @pytest.mark.asyncio
    async def test_uses_orchestrator_when_available(self, planner, mock_orchestrator):
        """Test that planner uses orchestrator for LLM calls."""
        # Mock response
        mock_orchestrator._provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps({"subtasks": []}))
        )

        # Decompose
        await planner.decompose_task("Test task")

        # Verify orchestrator was called
        assert mock_orchestrator._provider_manager.chat.called

    @pytest.mark.asyncio
    async def test_falls_back_to_provider_manager(self, planner, mock_provider_manager):
        """Test fallback to provider manager when orchestrator fails."""
        # Make orchestrator fail
        planner._orchestrator = None

        # Ensure provider_manager is set
        planner._provider_manager = mock_provider_manager
        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps({"subtasks": []}))
        )

        await planner.decompose_task("Test task")

        # Verify provider manager was called
        assert mock_provider_manager.chat.called

    @pytest.mark.asyncio
    async def test_no_llm_provider_raises_error(self, planner):
        """Test that missing LLM provider raises error."""
        # Remove both
        planner._orchestrator = None
        planner._provider_manager = None

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="No LLM provider available"):
            await planner.decompose_task("Test task")


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_task_list(self, planner, mock_provider_manager):
        """Test decomposing into empty task list."""
        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps({"root_task": "test", "subtasks": []}))
        )

        graph = await planner.decompose_task("Empty task")

        # Should create empty graph
        assert len(graph.nodes) == 0

    @pytest.mark.asyncio
    async def test_task_id_auto_generation(self, planner, mock_provider_manager):
        """Test auto-generation of task IDs when not provided."""
        # Response without IDs
        response = json.dumps(
            {
                "root_task": "test",
                "subtasks": [
                    {"description": "Task 1", "depends_on": [], "estimated_complexity": 5},
                    {"description": "Task 2", "depends_on": [], "estimated_complexity": 5},
                ],
            }
        )

        mock_provider_manager.chat = AsyncMock(return_value=Mock(content=response))

        graph = await planner.decompose_task("Test task")

        # Should auto-generate IDs
        assert "task_1" in graph.nodes
        assert "task_2" in graph.nodes

    def test_estimate_complexity_empty_task(self, planner):
        """Test complexity estimation for empty task."""
        score = planner.estimate_complexity("")

        # Should still return valid score
        assert 1 <= score.score <= 10
        assert score.confidence > 0

    def test_validate_empty_graph(self, planner):
        """Test validating empty graph."""
        graph = TaskGraph()
        result = planner.validate_plan(graph)

        # Empty graph is valid
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_multi_level_decomposition(self, planner, mock_provider_manager):
        """Test decomposition with multiple levels of dependencies."""
        response = json.dumps(
            {
                "root_task": "Build system",
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": "Foundation",
                        "depends_on": [],
                        "estimated_complexity": 2,
                    },
                    {
                        "id": "task_2",
                        "description": "Mid level",
                        "depends_on": ["task_1"],
                        "estimated_complexity": 5,
                    },
                    {
                        "id": "task_3",
                        "description": "High level",
                        "depends_on": ["task_2"],
                        "estimated_complexity": 8,
                    },
                    {
                        "id": "task_4",
                        "description": "Another high level",
                        "depends_on": ["task_2"],
                        "estimated_complexity": 7,
                    },
                ],
            }
        )

        mock_provider_manager.chat = AsyncMock(return_value=Mock(content=response))

        graph = await planner.decompose_task("Multi-level task")

        # Verify structure
        assert len(graph.nodes) == 4
        assert graph.nodes["task_3"].depends_on == ["task_2"]
        assert graph.nodes["task_4"].depends_on == ["task_2"]
        # task_3 and task_4 are parallel after task_2

    @pytest.mark.asyncio
    async def test_decomposition_with_multiple_dependencies(self, planner, mock_provider_manager):
        """Test task with multiple dependencies."""
        response = json.dumps(
            {
                "root_task": "Integration task",
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": "First component",
                        "depends_on": [],
                        "estimated_complexity": 4,
                    },
                    {
                        "id": "task_2",
                        "description": "Second component",
                        "depends_on": [],
                        "estimated_complexity": 4,
                    },
                    {
                        "id": "task_3",
                        "description": "Integration",
                        "depends_on": ["task_1", "task_2"],
                        "estimated_complexity": 6,
                    },
                ],
            }
        )

        mock_provider_manager.chat = AsyncMock(return_value=Mock(content=response))

        graph = await planner.decompose_task("Integration task")

        # Verify dependencies
        assert set(graph.nodes["task_3"].depends_on) == {"task_1", "task_2"}

    @pytest.mark.asyncio
    async def test_update_plan_blocking_propagation(self, planner):
        """Test that task failure blocks dependent tasks."""
        graph = TaskGraph()
        graph.add_node(Task(id="task_1", description="Base"))
        graph.add_node(Task(id="task_2", description="Depends on 1", depends_on=["task_1"]))
        graph.add_node(Task(id="task_3", description="Depends on 2", depends_on=["task_2"]))
        graph.add_edge("task_2", "task_1")
        graph.add_edge("task_3", "task_2")

        # Fail task_1
        result = await planner.update_plan(graph, completed_tasks=[], failed_tasks=["task_1"])

        # Verify blocking propagation
        assert graph.nodes["task_1"].status == "failed"
        assert graph.nodes["task_2"].status == "blocked"
        assert graph.nodes["task_3"].status == "blocked"
        assert result.can_proceed is False

    @pytest.mark.asyncio
    async def test_update_plan_partial_failure(self, planner):
        """Test updating plan with both successful and failed tasks."""
        graph = TaskGraph()
        graph.add_node(Task(id="task_1", description="Task 1"))
        graph.add_node(Task(id="task_2", description="Task 2"))
        graph.add_node(Task(id="task_3", description="Task 3", depends_on=["task_1", "task_2"]))
        graph.add_edge("task_3", "task_1")
        graph.add_edge("task_3", "task_2")

        # task_1 succeeds, task_2 fails
        result = await planner.update_plan(
            graph, completed_tasks=["task_1"], failed_tasks=["task_2"]
        )

        # Verify states
        assert graph.nodes["task_1"].status == "completed"
        assert graph.nodes["task_2"].status == "failed"
        assert graph.nodes["task_3"].status == "blocked"  # Blocked by task_2
        assert result.can_proceed is False

    @pytest.mark.asyncio
    async def test_suggest_next_tasks_priority_ordering(self, planner):
        """Test that higher complexity tasks are suggested first."""
        graph = TaskGraph()
        graph.add_node(
            Task(id="simple", description="Simple", estimated_complexity=2)
        )
        graph.add_node(
            Task(id="complex", description="Complex", estimated_complexity=9)
        )
        graph.add_node(
            Task(id="medium", description="Medium", estimated_complexity=5)
        )

        tasks = await planner.suggest_next_tasks(graph)

        # Should be sorted by complexity descending
        assert tasks[0].id == "complex"
        assert tasks[1].id == "medium"
        assert tasks[2].id == "simple"

    @pytest.mark.asyncio
    async def test_suggest_parallel_tasks(self, planner):
        """Test detecting parallel execution opportunities."""
        graph = TaskGraph()
        graph.add_node(Task(id="task_1", description="Independent 1"))
        graph.add_node(Task(id="task_2", description="Independent 2"))
        graph.add_node(Task(id="task_3", description="Dependent", depends_on=["task_1", "task_2"]))
        graph.add_edge("task_3", "task_1")
        graph.add_edge("task_3", "task_2")

        tasks = await planner.suggest_next_tasks(graph)

        # task_1 and task_2 should both be ready
        assert len(tasks) == 2
        task_ids = {t.id for t in tasks}
        assert task_ids == {"task_1", "task_2"}

    @pytest.mark.asyncio
    async def test_suggest_respects_dependencies(self, planner):
        """Test that suggestions respect task dependencies."""
        graph = TaskGraph()
        graph.add_node(Task(id="task_1", description="Base"))
        graph.add_node(Task(id="task_2", description="Depends on 1", depends_on=["task_1"]))
        graph.add_edge("task_2", "task_1")

        # Initially only task_1 should be ready
        tasks = await planner.suggest_next_tasks(graph)
        assert len(tasks) == 1
        assert tasks[0].id == "task_1"

        # After completing task_1
        graph.nodes["task_1"].status = "completed"
        tasks = await planner.suggest_next_tasks(graph)
        assert len(tasks) == 1
        assert tasks[0].id == "task_2"

    def test_estimate_complexity_unknown_task(self, planner):
        """Test complexity estimation for unknown task type."""
        score = planner.estimate_complexity("Do something completely unknown")

        # Should return moderate default
        assert score.score == 5.0  # Default base score
        assert score.confidence == 0.5  # Base confidence with no factors
        assert len(score.factors) == 0

    def test_estimate_complexity_keyword_combinations(self, planner):
        """Test complexity with multiple keyword types."""
        score = planner.estimate_complexity(
            "Refactor system to show what needs design"
        )

        # Should have both positive and negative adjustments
        factors_str = " ".join(score.factors)
        assert "Complex keywords" in factors_str
        assert "Simple keywords" in factors_str
        # Net result should still be reasonable
        assert 1 <= score.score <= 10

    def test_estimate_complexity_large_file_count(self, planner):
        """Test complexity adjustment for many files."""
        score = planner.estimate_complexity(
            "Implement feature",
            context={"file_count": 100}
        )

        # Should cap file count adjustment
        assert any("files" in factor.lower() for factor in score.factors)
        assert score.score > 5  # Should be higher than base

    def test_estimate_complexity_large_codebase(self, planner):
        """Test complexity adjustment for large codebase."""
        score = planner.estimate_complexity(
            "Implement feature",
            context={"lines_of_code": 50000}
        )

        # Should cap LOC adjustment
        assert any("loc" in factor.lower() or "codebase" in factor.lower()
                   for factor in score.factors)
        assert score.score > 5

    def test_estimate_complexity_multiple_domains(self, planner):
        """Test complexity adjustment for multiple domains."""
        score = planner.estimate_complexity(
            "Implement feature",
            context={"domains": ["coding", "devops", "testing"]}
        )

        # Multiple domains should increase complexity
        assert any("domains" in factor.lower() for factor in score.factors)
        assert score.score > 5

    def test_validate_plan_with_complex_dependencies(self, planner):
        """Test validation of complex dependency graph."""
        graph = TaskGraph()
        graph.add_node(Task(id="a", description="A"))
        graph.add_node(Task(id="b", description="B", depends_on=["a"]))
        graph.add_node(Task(id="c", description="C", depends_on=["a"]))
        graph.add_node(Task(id="d", description="D", depends_on=["b", "c"]))
        graph.add_edge("b", "a")
        graph.add_edge("c", "a")
        graph.add_edge("d", "b")
        graph.add_edge("d", "c")

        result = planner.validate_plan(graph)

        # Should be valid diamond dependency
        assert result.is_valid is True
        assert result.has_cycles is False

    def test_validate_plan_self_cycle(self, planner):
        """Test detecting task that depends on itself."""
        graph = TaskGraph()
        graph.add_node(Task(id="task_1", description="Task 1", depends_on=["task_1"]))
        graph.add_edge("task_1", "task_1")

        result = planner.validate_plan(graph)

        # Should detect self-cycle
        assert result.is_valid is False
        assert result.has_cycles is True

    def test_validate_plan_multiple_missing_deps(self, planner):
        """Test detecting multiple missing dependencies."""
        graph = TaskGraph()
        graph.add_node(
            Task(id="task_1", description="Task 1", depends_on=["missing1", "missing2"])
        )

        result = planner.validate_plan(graph)

        # Should detect both missing
        assert result.is_valid is False
        assert len(result.errors) >= 2

    @pytest.mark.asyncio
    async def test_decompose_event_emission(self, planner, mock_provider_manager, mock_event_bus):
        """Test that decomposition emits events."""
        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps({"subtasks": []}))
        )

        await planner.decompose_task("Test task")

        # Should emit start and complete events
        assert mock_event_bus.publish.call_count >= 2

    @pytest.mark.asyncio
    async def test_decompose_failure_event(self, planner, mock_provider_manager, mock_event_bus):
        """Test that decomposition failure emits error event."""
        mock_provider_manager.chat = AsyncMock(side_effect=Exception("LLM failed"))

        try:
            await planner.decompose_task("Test task")
        except RuntimeError:
            pass

        # Should emit failure event
        failure_calls = [
            call for call in mock_event_bus.publish.call_args_list
            if "failed" in str(call).lower()
        ]
        assert len(failure_calls) > 0

    @pytest.mark.asyncio
    async def test_cache_hit_emits_event(self, planner, mock_provider_manager, mock_event_bus):
        """Test that cache hit emits specific event."""
        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps({"subtasks": []}))
        )

        # First call
        await planner.decompose_task("Test task", use_cache=True)

        # Second call - cache hit
        await planner.decompose_task("Test task", use_cache=True)

        # Check for cache hit event
        cache_hit_calls = [
            call for call in mock_event_bus.publish.call_args_list
            if "cache_hit" in str(call).lower()
        ]
        assert len(cache_hit_calls) == 1

    @pytest.mark.asyncio
    async def test_custom_decomposition_prompt(self, planner, mock_provider_manager):
        """Test using custom decomposition prompt."""
        custom_prompt = "Custom prompt for testing"

        planner._decomposition_prompt = custom_prompt
        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps({"subtasks": []}))
        )

        await planner.decompose_task("Test task")

        # Verify custom prompt was used
        call_args = mock_provider_manager.chat.call_args
        messages = call_args[1]["messages"]
        system_message = [m for m in messages if m["role"] == "system"][0]
        assert custom_prompt in system_message["content"]

    @pytest.mark.asyncio
    async def test_decompose_without_orchestrator(self, planner, mock_provider_manager):
        """Test decomposition without orchestrator (provider manager only)."""
        planner._orchestrator = None
        planner._provider_manager = mock_provider_manager

        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps({"subtasks": []}))
        )

        await planner.decompose_task("Test task")

        # Should use provider manager
        assert mock_provider_manager.chat.called

    @pytest.mark.asyncio
    async def test_update_plan_with_in_progress_tasks(self, planner):
        """Test updating plan when some tasks are in progress."""
        graph = TaskGraph()
        graph.add_node(Task(id="task_1", description="Task 1"))
        graph.add_node(Task(id="task_2", description="Task 2", depends_on=["task_1"]))
        graph.add_edge("task_2", "task_1")

        # Mark task as in progress
        graph.nodes["task_1"].status = "in_progress"

        result = await planner.update_plan(graph, completed_tasks=[])

        # Should indicate can proceed
        assert result.can_proceed is True
        assert len(result.new_ready_tasks) == 0

    @pytest.mark.asyncio
    async def test_update_plan_blocked_task_not_marked_ready(self, planner):
        """Test that blocked tasks don't become ready."""
        graph = TaskGraph()
        graph.add_node(Task(id="task_1", description="Task 1"))
        graph.add_node(Task(id="task_2", description="Task 2", depends_on=["task_1"]))
        graph.add_edge("task_2", "task_1")

        # Mark task_2 as blocked
        graph.nodes["task_2"].status = "blocked"

        result = await planner.update_plan(graph, completed_tasks=["task_1"])

        # task_2 should not be in ready tasks
        assert "task_2" not in [t.id for t in result.new_ready_tasks]

    def test_get_cache_stats_with_entries(self, planner, mock_provider_manager):
        """Test cache stats with cached entries."""
        import asyncio

        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content=json.dumps({"subtasks": []}))
        )

        # Add to cache
        asyncio.run(planner.decompose_task("Test task"))

        stats = planner.get_cache_stats()

        assert stats["cached_plans"] == 1
        assert len(stats["cache_keys"]) == 1

    def test_estimate_complexity_all_keywords(self, planner):
        """Test complexity with all complex keywords."""
        task = " ".join([
            "refactor", "migrate", "restructure", "architecture",
            "implement", "design", "system"
        ])

        score = planner.estimate_complexity(task)

        # Should be high complexity
        assert score.score > 7
        assert len([f for f in score.factors if "Complex" in f]) > 0

    def test_estimate_complexity_min_score(self, planner):
        """Test that complexity score minimum is 1.0."""
        # All simple keywords
        task = "list show display what where check"

        score = planner.estimate_complexity(task)

        # Should not go below 1.0
        assert score.score >= 1.0

    def test_estimate_complexity_max_score(self, planner):
        """Test that complexity score maximum is 10.0."""
        # Many complex keywords with context
        task = " ".join(["refactor"] * 20)
        context = {
            "file_count": 1000,
            "lines_of_code": 1000000,
            "domains": ["a", "b", "c", "d", "e"]
        }

        score = planner.estimate_complexity(task, context)

        # Should not exceed 10.0
        assert score.score <= 10.0

    @pytest.mark.asyncio
    async def test_decompose_malformed_json_fails_gracefully(self, planner, mock_provider_manager):
        """Test handling of malformed JSON."""
        mock_provider_manager.chat = AsyncMock(
            return_value=Mock(content='{"subtasks": [{"id": "1"}]')  # Missing closing
        )

        # Should raise error
        with pytest.raises(ValueError):
            await planner.decompose_task("Test task")

    @pytest.mark.asyncio
    async def test_suggest_next_tasks_returns_copy(self, planner):
        """Test that suggestion doesn't modify graph."""
        graph = TaskGraph()
        original_status = "pending"
        task = Task(id="task_1", description="Task", status=original_status)
        graph.add_node(task)

        await planner.suggest_next_tasks(graph)

        # Graph should be unchanged
        assert graph.nodes["task_1"].status == original_status
