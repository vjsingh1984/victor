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

"""Unit tests for autonomous planning module (P2.2).

Tests cover:
- StepStatus and StepType enums
- PlanStep dataclass and is_ready() method
- StepResult dataclass and serialization
- ExecutionPlan with dependency handling
- PlanResult aggregation
- AutonomousPlanner plan generation (mocked)
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.planning import (
    AutonomousPlanner,
    ExecutionPlan,
    PlanResult,
    PlanStep,
    StepResult,
    StepStatus,
    StepType,
)

# =============================================================================
# StepStatus Tests
# =============================================================================


class TestStepStatus:
    """Tests for StepStatus enum."""

    def test_all_statuses_defined(self):
        """Verify all expected statuses exist."""
        expected = {"PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "SKIPPED", "BLOCKED"}
        actual = {s.name for s in StepStatus}
        assert actual == expected

    def test_status_values(self):
        """Verify status string values."""
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"


# =============================================================================
# StepType Tests
# =============================================================================


class TestStepType:
    """Tests for StepType enum."""

    def test_all_types_defined(self):
        """Verify all expected types exist."""
        expected = {"RESEARCH", "PLANNING", "IMPLEMENTATION", "TESTING", "REVIEW", "DEPLOYMENT"}
        actual = {t.name for t in StepType}
        assert actual == expected

    def test_type_values(self):
        """Verify type string values."""
        assert StepType.RESEARCH.value == "research"
        assert StepType.IMPLEMENTATION.value == "implementation"


# =============================================================================
# PlanStep Tests
# =============================================================================


class TestPlanStep:
    """Tests for PlanStep dataclass."""

    def test_minimal_step(self):
        """Test creating step with minimal fields."""
        step = PlanStep(
            id="1",
            description="Test step",
        )
        assert step.id == "1"
        assert step.description == "Test step"
        assert step.step_type == StepType.IMPLEMENTATION  # default
        assert step.depends_on == []
        assert step.status == StepStatus.PENDING

    def test_full_step(self):
        """Test creating step with all fields."""
        step = PlanStep(
            id="2",
            description="Full test step",
            step_type=StepType.RESEARCH,
            depends_on=["1"],
            estimated_tool_calls=15,
            requires_approval=True,
            sub_agent_role="researcher",
            context={"key": "value"},
        )
        assert step.step_type == StepType.RESEARCH
        assert step.depends_on == ["1"]
        assert step.estimated_tool_calls == 15
        assert step.requires_approval is True
        assert step.sub_agent_role == "researcher"

    def test_is_ready_no_dependencies(self):
        """Test is_ready with no dependencies."""
        step = PlanStep(id="1", description="Test")
        assert step.is_ready(set()) is True

    def test_is_ready_with_satisfied_dependencies(self):
        """Test is_ready with satisfied dependencies."""
        step = PlanStep(id="2", description="Test", depends_on=["1"])
        assert step.is_ready({"1"}) is True

    def test_is_ready_with_unsatisfied_dependencies(self):
        """Test is_ready with unsatisfied dependencies."""
        step = PlanStep(id="2", description="Test", depends_on=["1"])
        assert step.is_ready(set()) is False

    def test_is_ready_wrong_status(self):
        """Test is_ready returns False for non-pending steps."""
        step = PlanStep(id="1", description="Test", status=StepStatus.COMPLETED)
        assert step.is_ready(set()) is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        step = PlanStep(
            id="1",
            description="Test step",
            step_type=StepType.TESTING,
            depends_on=["0"],
            estimated_tool_calls=5,
        )
        d = step.to_dict()
        assert d["id"] == "1"
        assert d["description"] == "Test step"
        assert d["step_type"] == "testing"
        assert d["depends_on"] == ["0"]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "1",
            "description": "Test step",
            "step_type": "research",
            "depends_on": ["0"],
            "estimated_tool_calls": 10,
            "requires_approval": True,
        }
        step = PlanStep.from_dict(data)
        assert step.id == "1"
        assert step.step_type == StepType.RESEARCH
        assert step.requires_approval is True


# =============================================================================
# StepResult Tests
# =============================================================================


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_success_result(self):
        """Test creating successful result."""
        result = StepResult(
            success=True,
            output="Task completed",
            tool_calls_used=5,
            duration_seconds=2.5,
        )
        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        """Test creating failed result."""
        result = StepResult(
            success=False,
            output="",
            error="Timeout error",
        )
        assert result.success is False
        assert result.error == "Timeout error"

    def test_to_dict(self):
        """Test serialization."""
        result = StepResult(
            success=True,
            output="Done",
            tool_calls_used=3,
            duration_seconds=1.5,
            artifacts=["file.py"],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == "Done"
        assert d["artifacts"] == ["file.py"]

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "success": True,
            "output": "Output text",
            "tool_calls_used": 10,
            "duration_seconds": 5.0,
        }
        result = StepResult.from_dict(data)
        assert result.success is True
        assert result.tool_calls_used == 10


# =============================================================================
# ExecutionPlan Tests
# =============================================================================


class TestExecutionPlan:
    """Tests for ExecutionPlan dataclass."""

    @pytest.fixture
    def sample_plan(self):
        """Create a sample plan for testing."""
        return ExecutionPlan(
            id="plan_123",
            goal="Implement feature X",
            steps=[
                PlanStep(id="1", description="Research", step_type=StepType.RESEARCH),
                PlanStep(id="2", description="Implement", depends_on=["1"]),
                PlanStep(id="3", description="Test", depends_on=["2"]),
            ],
        )

    def test_basic_creation(self):
        """Test basic plan creation."""
        plan = ExecutionPlan(
            id="test_plan",
            goal="Test goal",
        )
        assert plan.id == "test_plan"
        assert plan.goal == "Test goal"
        assert plan.steps == []
        assert plan.approved is False

    def test_get_step(self, sample_plan):
        """Test getting step by ID."""
        step = sample_plan.get_step("2")
        assert step is not None
        assert step.description == "Implement"

    def test_get_step_not_found(self, sample_plan):
        """Test getting non-existent step."""
        step = sample_plan.get_step("nonexistent")
        assert step is None

    def test_get_ready_steps_initial(self, sample_plan):
        """Test ready steps at start (only step without deps)."""
        ready = sample_plan.get_ready_steps()
        assert len(ready) == 1
        assert ready[0].id == "1"

    def test_get_ready_steps_after_completion(self, sample_plan):
        """Test ready steps after completing first step."""
        sample_plan.steps[0].status = StepStatus.COMPLETED
        ready = sample_plan.get_ready_steps()
        assert len(ready) == 1
        assert ready[0].id == "2"

    def test_is_complete(self, sample_plan):
        """Test is_complete check."""
        assert sample_plan.is_complete() is False

        for step in sample_plan.steps:
            step.status = StepStatus.COMPLETED
        assert sample_plan.is_complete() is True

    def test_is_failed(self, sample_plan):
        """Test is_failed check."""
        assert sample_plan.is_failed() is False

        sample_plan.steps[1].status = StepStatus.FAILED
        assert sample_plan.is_failed() is True

    def test_total_estimated_tool_calls(self, sample_plan):
        """Test total tool call estimation."""
        # Default is 10 per step
        assert sample_plan.total_estimated_tool_calls() == 30

    def test_progress_percentage(self, sample_plan):
        """Test progress calculation."""
        assert sample_plan.progress_percentage() == 0.0

        sample_plan.steps[0].status = StepStatus.COMPLETED
        assert sample_plan.progress_percentage() == pytest.approx(33.33, rel=0.01)

        sample_plan.steps[1].status = StepStatus.COMPLETED
        sample_plan.steps[2].status = StepStatus.COMPLETED
        assert sample_plan.progress_percentage() == 100.0

    def test_to_markdown(self, sample_plan):
        """Test markdown generation."""
        md = sample_plan.to_markdown()
        assert "# Execution Plan:" in md
        assert "Research" in md
        assert "Implement" in md
        assert "Test" in md

    def test_to_dict_and_from_dict(self, sample_plan):
        """Test round-trip serialization."""
        d = sample_plan.to_dict()
        restored = ExecutionPlan.from_dict(d)

        assert restored.id == sample_plan.id
        assert restored.goal == sample_plan.goal
        assert len(restored.steps) == len(sample_plan.steps)


# =============================================================================
# PlanResult Tests
# =============================================================================


class TestPlanResult:
    """Tests for PlanResult dataclass."""

    def test_successful_result(self):
        """Test creating successful result."""
        result = PlanResult(
            plan_id="plan_123",
            success=True,
            steps_completed=3,
            steps_failed=0,
            total_tool_calls=25,
            total_duration=120.5,
            final_output="All steps completed",
        )
        assert result.success is True
        assert result.steps_completed == 3
        assert result.steps_failed == 0

    def test_failed_result(self):
        """Test creating failed result."""
        result = PlanResult(
            plan_id="plan_456",
            success=False,
            steps_completed=1,
            steps_failed=1,
        )
        assert result.success is False

    def test_to_dict(self):
        """Test serialization."""
        result = PlanResult(
            plan_id="test",
            success=True,
            steps_completed=2,
            total_tool_calls=10,
        )
        d = result.to_dict()
        assert d["plan_id"] == "test"
        assert d["success"] is True


# =============================================================================
# AutonomousPlanner Tests
# =============================================================================


class TestAutonomousPlanner:
    """Tests for AutonomousPlanner class."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        mock = MagicMock()
        mock.set_system_prompt = MagicMock()
        mock.chat = AsyncMock()
        return mock

    def test_initialization(self, mock_orchestrator):
        """Test planner initialization."""
        planner = AutonomousPlanner(mock_orchestrator)
        assert planner.orchestrator == mock_orchestrator
        assert planner.active_plan is None

    def test_default_approval_returns_false(self, mock_orchestrator):
        """Test default approval is safe (returns False)."""
        planner = AutonomousPlanner(mock_orchestrator)
        result = planner._default_approval("Test message")
        assert result is False

    def test_custom_approval_callback(self, mock_orchestrator):
        """Test custom approval callback."""
        approvals = []

        def custom_approval(msg):
            approvals.append(msg)
            return True

        planner = AutonomousPlanner(mock_orchestrator, approval_callback=custom_approval)
        result = planner.approval_callback("Approve this?")
        assert result is True
        assert "Approve this?" in approvals

    def test_build_planning_prompt(self, mock_orchestrator):
        """Test planning prompt construction."""
        planner = AutonomousPlanner(mock_orchestrator)
        prompt = planner._build_planning_prompt(
            goal="Implement auth",
            context="Using FastAPI",
            max_steps=5,
        )
        assert "Implement auth" in prompt
        assert "FastAPI" in prompt
        assert "5 steps" in prompt

    def test_parse_plan_json_valid(self, mock_orchestrator):
        """Test parsing valid plan JSON."""
        planner = AutonomousPlanner(mock_orchestrator)

        json_str = json.dumps(
            [
                {
                    "id": "1",
                    "description": "Step 1",
                    "step_type": "research",
                    "depends_on": [],
                    "estimated_tool_calls": 5,
                },
                {
                    "id": "2",
                    "description": "Step 2",
                    "step_type": "implementation",
                    "depends_on": ["1"],
                    "estimated_tool_calls": 15,
                },
            ]
        )

        plan = planner._parse_plan_json("Test goal", json_str)
        assert plan.goal == "Test goal"
        assert len(plan.steps) == 2
        assert plan.steps[0].step_type == StepType.RESEARCH
        assert plan.steps[1].depends_on == ["1"]

    def test_parse_plan_json_with_markdown(self, mock_orchestrator):
        """Test parsing JSON wrapped in markdown code blocks."""
        planner = AutonomousPlanner(mock_orchestrator)

        json_str = """Here's the plan:

```json
[
    {"id": "1", "description": "Only step"}
]
```

Let me know if you need changes."""

        plan = planner._parse_plan_json("Goal", json_str)
        assert len(plan.steps) == 1

    def test_parse_plan_json_fallback(self, mock_orchestrator):
        """Test fallback for invalid JSON."""
        planner = AutonomousPlanner(mock_orchestrator)

        plan = planner._parse_plan_json("Complex goal", "not valid json")
        # Should create single-step fallback plan
        assert len(plan.steps) == 1
        assert plan.steps[0].description == "Complex goal"

    def test_map_role_string(self, mock_orchestrator):
        """Test role string to enum mapping."""
        planner = AutonomousPlanner(mock_orchestrator)

        from victor.agent.subagents import SubAgentRole

        assert planner._map_role_string("researcher") == SubAgentRole.RESEARCHER
        assert planner._map_role_string("executor") == SubAgentRole.EXECUTOR
        assert planner._map_role_string("unknown") == SubAgentRole.EXECUTOR  # default

    def test_get_active_plan(self, mock_orchestrator):
        """Test getting active plan."""
        planner = AutonomousPlanner(mock_orchestrator)
        assert planner.get_active_plan() is None

        plan = ExecutionPlan(id="test", goal="Test")
        planner.active_plan = plan
        assert planner.get_active_plan() == plan

    def test_get_plan_history(self, mock_orchestrator):
        """Test getting plan history."""
        planner = AutonomousPlanner(mock_orchestrator)
        assert planner.get_plan_history() == []

        plan = ExecutionPlan(id="test", goal="Test")
        planner._plan_history.append(plan)
        history = planner.get_plan_history()
        assert len(history) == 1
        assert history[0].id == "test"


# =============================================================================
# Integration Tests
# =============================================================================


class TestPlanningIntegration:
    """Integration tests for planning module."""

    def test_module_exports(self):
        """Test all expected exports are available."""
        from victor.agent.planning import (
            AutonomousPlanner,
            ExecutionPlan,
            PlanResult,
            PlanStep,
            StepResult,
            StepStatus,
            StepType,
        )

        assert all(
            [
                AutonomousPlanner,
                ExecutionPlan,
                PlanResult,
                PlanStep,
                StepResult,
                StepStatus,
                StepType,
            ]
        )

    def test_step_workflow(self):
        """Test realistic step workflow."""
        # Create steps with dependencies
        plan = ExecutionPlan(
            id="workflow_test",
            goal="Build feature",
            steps=[
                PlanStep(id="a", description="Research", step_type=StepType.RESEARCH),
                PlanStep(id="b", description="Plan", depends_on=["a"], step_type=StepType.PLANNING),
                PlanStep(id="c", description="Build", depends_on=["b"]),
                PlanStep(id="d", description="Test", depends_on=["c"], step_type=StepType.TESTING),
            ],
        )

        # Initially only 'a' is ready
        assert [s.id for s in plan.get_ready_steps()] == ["a"]

        # Complete 'a', now 'b' is ready
        plan.steps[0].status = StepStatus.COMPLETED
        assert [s.id for s in plan.get_ready_steps()] == ["b"]

        # Complete 'b', now 'c' is ready
        plan.steps[1].status = StepStatus.COMPLETED
        assert [s.id for s in plan.get_ready_steps()] == ["c"]

        # If 'c' fails, plan should show failure
        plan.steps[2].status = StepStatus.FAILED
        assert plan.is_failed() is True
        assert plan.is_complete() is False

    def test_parallel_ready_steps(self):
        """Test parallel steps become ready together."""
        plan = ExecutionPlan(
            id="parallel_test",
            goal="Parallel tasks",
            steps=[
                PlanStep(id="setup", description="Setup"),
                PlanStep(id="task_a", description="Task A", depends_on=["setup"]),
                PlanStep(id="task_b", description="Task B", depends_on=["setup"]),
                PlanStep(id="merge", description="Merge", depends_on=["task_a", "task_b"]),
            ],
        )

        # Complete setup
        plan.steps[0].status = StepStatus.COMPLETED

        # Both task_a and task_b should be ready
        ready_ids = {s.id for s in plan.get_ready_steps()}
        assert ready_ids == {"task_a", "task_b"}

        # Complete both tasks
        plan.steps[1].status = StepStatus.COMPLETED
        plan.steps[2].status = StepStatus.COMPLETED

        # Now merge is ready
        assert [s.id for s in plan.get_ready_steps()] == ["merge"]
