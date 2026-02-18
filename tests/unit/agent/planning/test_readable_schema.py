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

"""Unit tests for readable task planning schema."""

import json
import pytest

from victor.agent.planning.readable_schema import (
    ReadableTaskPlan,
    TaskComplexity,
    TaskPlannerContext,
    generate_task_plan,
    plan_to_session_context,
    plan_to_workflow_yaml,
)
from victor.agent.planning.base import (
    ExecutionPlan,
    PlanStep,
    StepType,
)


class TestReadableTaskPlan:
    """Tests for ReadableTaskPlan model."""

    def test_simple_plan_validation(self):
        """Test validation of simple readable plan."""
        json_str = '{"name":"Fix bug","complexity":"simple","desc":"Fix login bug","steps":[[1,"analyze","Find bug","grep"],[2,"feature","Fix","write"]]}'

        plan = ReadableTaskPlan.model_validate_json(json_str)

        assert plan.name == "Fix bug"
        assert plan.complexity == TaskComplexity.SIMPLE
        assert plan.desc == "Fix login bug"
        assert len(plan.steps) == 2
        assert plan.approval is False

    def test_moderate_plan_validation(self):
        """Test validation of moderate plan with dependencies."""
        json_str = """{"name":"Add auth","complexity":"moderate","desc":"OAuth2 login",
        "steps":[[1,"research","Research patterns","overview"],[2,"feature","Create module","write,test"],
        [3,"test","Verify","pytest",[2]]]}"""

        plan = ReadableTaskPlan.model_validate_json(json_str)

        assert plan.complexity == TaskComplexity.MODERATE
        assert len(plan.steps) == 3

        # Convert to execution plan
        exec_plan = plan.to_execution_plan()
        assert exec_plan.goal == "OAuth2 login"
        assert len(exec_plan.steps) == 3

    def test_complex_plan_with_approval(self):
        """Test complex plan requiring approval."""
        json_str = """{"name":"Deploy API","complexity":"complex","desc":"Deploy to production",
        "steps":[[1,"test","Run tests","pytest"],[2,"deploy","Deploy","kubectl",[1]]],
        "approval":true}"""

        plan = ReadableTaskPlan.model_validate_json(json_str)

        assert plan.approval is True
        assert plan.complexity == TaskComplexity.COMPLEX

        exec_plan = plan.to_execution_plan()
        # Deployment step should require approval
        deploy_step = [s for s in exec_plan.steps if s.id == "2"][0]
        assert deploy_step.requires_approval is True

    def test_to_execution_plan(self):
        """Test conversion to full ExecutionPlan."""
        plan = ReadableTaskPlan(
            name="Feature X",
            complexity=TaskComplexity.MODERATE,
            desc="Implement feature X",
            steps=[[1, "feature", "Create module", "write"], [2, "test", "Test", "pytest", [1]]],
        )

        exec_plan = plan.to_execution_plan()

        assert exec_plan.goal == "Implement feature X"
        assert len(exec_plan.steps) == 2
        assert exec_plan.metadata["task_name"] == "Feature X"
        assert exec_plan.metadata["complexity"] == "moderate"

    def test_from_execution_plan(self):
        """Test creation of readable plan from ExecutionPlan."""
        exec_plan = ExecutionPlan(
            id="test-plan",
            goal="Implement auth",
            steps=[
                PlanStep(
                    id="1",
                    description="Create auth module",
                    step_type=StepType.IMPLEMENTATION,
                    depends_on=[],
                    estimated_tool_calls=10,
                ),
            ],
            metadata={"task_name": "auth", "complexity": "simple"},
        )

        readable = ReadableTaskPlan.from_execution_plan(exec_plan)

        assert readable.name == "auth"
        assert readable.complexity == TaskComplexity.SIMPLE
        assert len(readable.steps) == 1

    def test_to_yaml(self):
        """Test conversion to YAML format."""
        plan = ReadableTaskPlan(
            name="Task X",
            complexity=TaskComplexity.SIMPLE,
            desc="Test task",
            steps=[[1, "test", "Run tests", "pytest"]],
            duration="15min",
        )

        yaml_str = plan.to_yaml()

        assert "workflows:" in yaml_str
        assert "Task X:" in yaml_str
        assert "Test task" in yaml_str

    def test_to_markdown(self):
        """Test conversion to markdown display format."""
        plan = ReadableTaskPlan(
            name="Fix Login Bug",
            complexity=TaskComplexity.SIMPLE,
            desc="Fix authentication bug",
            steps=[[1, "analyze", "Find bug", "grep"], [2, "feature", "Fix", "write"]],
            duration="30min",
        )

        markdown = plan.to_markdown()

        assert "# Fix Login Bug" in markdown
        assert "**Description**: Fix authentication bug" in markdown
        assert "**Complexity**: simple" in markdown
        assert "## Steps" in markdown

    def test_get_llm_prompt(self):
        """Test LLM prompt generation."""
        prompt = ReadableTaskPlan.get_llm_prompt()

        assert "JSON" in prompt
        assert "name" in prompt
        assert "complexity" in prompt
        assert "steps" in prompt
        assert "research, planning, feature" in prompt

    def test_step_type_mappings(self):
        """Test all step type string mappings to StepType enum."""
        type_mappings = [
            ("research", StepType.RESEARCH),
            ("planning", StepType.PLANNING),
            ("feature", StepType.IMPLEMENTATION),
            ("implementation", StepType.IMPLEMENTATION),
            ("bugfix", StepType.IMPLEMENTATION),
            ("refactor", StepType.IMPLEMENTATION),
            ("test", StepType.TESTING),
            ("testing", StepType.TESTING),
            ("review", StepType.REVIEW),
            ("deploy", StepType.DEPLOYMENT),
            ("deployment", StepType.DEPLOYMENT),
            ("analyze", StepType.RESEARCH),
            ("analysis", StepType.RESEARCH),
            ("doc", StepType.RESEARCH),
            ("documentation", StepType.RESEARCH),
        ]

        for type_str, expected_type in type_mappings:
            # Create a minimal plan with this step type
            plan = ReadableTaskPlan(
                name="Test",
                complexity=TaskComplexity.SIMPLE,
                desc="Test",
                steps=[[1, type_str, "Test step"]],
            )
            exec_plan = plan.to_execution_plan()
            assert exec_plan.steps[0].step_type == expected_type, f"Failed for {type_str}"

    def test_token_efficiency(self):
        """Test that readable schema uses fewer tokens than verbose JSON."""
        # Verbose JSON (what NOT to use)
        verbose = {
            "task_name": "Add authentication",
            "complexity": "moderate",
            "description": "Implement OAuth2 login",
            "steps": [
                {
                    "id": "1",
                    "type": "feature",
                    "description": "Create auth module",
                    "tools": ["write", "test"],
                    "dependencies": [],
                }
            ],
        }

        # Readable JSON (what TO use)
        readable = {
            "name": "Add auth",
            "complexity": "moderate",
            "desc": "OAuth2 login",
            "steps": [[1, "feature", "Create module", "write,test"]],
        }

        verbose_tokens = len(json.dumps(verbose))
        readable_tokens = len(json.dumps(readable))

        # Readable should be significantly smaller
        assert readable_tokens < verbose_tokens
        # Rough estimate: readable should be ~60-70% of verbose
        assert readable_tokens < verbose_tokens * 0.8


class TestTaskPlannerContext:
    """Tests for TaskPlannerContext session management."""

    def test_context_initialization(self):
        """Test context initialization."""
        ctx = TaskPlannerContext()

        assert ctx.current_plan is None
        assert len(ctx.plans_history) == 0
        assert len(ctx.approved_plans) == 0

    def test_set_and_approve_plan(self):
        """Test setting and approving a plan."""
        ctx = TaskPlannerContext()

        plan = ExecutionPlan(
            id="test",
            goal="Test goal",
            steps=[],
        )

        ctx.set_plan(plan)
        assert ctx.current_plan == plan

        ctx.approve_plan()
        assert len(ctx.approved_plans) == 1

    def test_archive_plan(self):
        """Test archiving current plan."""
        ctx = TaskPlannerContext()

        plan = ExecutionPlan(
            id="test",
            goal="Test goal",
            steps=[],
        )

        ctx.set_plan(plan)
        ctx.archive_plan()

        assert ctx.current_plan is None
        assert len(ctx.plans_history) == 1
        assert ctx.plans_history[0] == plan

    def test_get_plan_summary(self):
        """Test getting plan summary."""
        ctx = TaskPlannerContext()

        plan = ExecutionPlan(
            id="test",
            goal="Test goal",
            steps=[],
            metadata={"task_name": "test", "complexity": "simple"},
        )

        ctx.set_plan(plan)
        summary = ctx.get_plan_summary()

        assert summary["current_plan"] == "Test goal"
        assert summary["total_plans"] == 1
        assert summary["approved_plans"] == 0

    def test_to_context_dict(self):
        """Test exporting context to dictionary."""
        ctx = TaskPlannerContext()

        plan = ExecutionPlan(
            id="test",
            goal="Test goal",
            steps=[
                PlanStep(
                    id="1",
                    description="Step 1",
                    step_type=StepType.RESEARCH,
                    depends_on=[],
                )
            ],
            metadata={"task_name": "test"},
        )

        ctx.set_plan(plan)
        context_dict = ctx.to_context_dict()

        assert "task_planner" in context_dict
        assert context_dict["task_planner"]["active"] is True
        assert "current_plan" in context_dict["task_planner"]
        assert context_dict["task_planner"]["current_plan"]["goal"] == "Test goal"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_plan_to_workflow_yaml(self):
        """Test conversion of plan to workflow YAML."""
        plan = ReadableTaskPlan(
            name="Feature X",
            complexity=TaskComplexity.MODERATE,
            desc="Implement feature X",
            steps=[[1, "feature", "Create module", "write"], [2, "test", "Test", "pytest", [1]]],
        )

        yaml_str = plan_to_workflow_yaml(plan)

        assert "workflows:" in yaml_str
        assert "Feature X:" in yaml_str
        assert "Implement feature X" in yaml_str

    def test_plan_to_session_context(self):
        """Test adding plan to session context."""
        plan = ReadableTaskPlan(
            name="Task X",
            complexity=TaskComplexity.SIMPLE,
            desc="Test task",
            steps=[[1, "test", "Run tests", "pytest"]],
        )

        context = plan_to_session_context(plan, "session-123")

        assert context["session_id"] == "session-123"
        assert "task_plan" in context
        assert context["task_plan"]["name"] == "Task X"
        assert context["task_plan"]["complexity"] == "simple"

    def test_roundtrip_conversion(self):
        """Test roundtrip: Readable -> Execution -> Readable."""
        original_plan = ReadableTaskPlan(
            name="Roundtrip Test",
            complexity=TaskComplexity.MODERATE,
            desc="Test roundtrip conversion",
            steps=[
                [1, "research", "Research", "overview"],
                [2, "feature", "Implement", "write", [1]],
                [3, "test", "Test", "pytest", [2]],
            ],
            duration="1hr",
            approval=True,
        )

        # Convert to execution plan
        exec_plan = original_plan.to_execution_plan()

        # Convert back to readable
        restored_plan = ReadableTaskPlan.from_execution_plan(exec_plan)

        # Verify key fields match
        assert restored_plan.name == original_plan.name
        assert restored_plan.complexity == original_plan.complexity
        assert restored_plan.desc == original_plan.desc
        # Note: approval flag might not be preserved in roundtrip

    def test_invalid_json_validation(self):
        """Test that invalid JSON is rejected."""
        # Missing required field 'name'
        invalid_json = '{"complexity":"simple","desc":"test","steps":[]}'

        with pytest.raises(Exception):
            ReadableTaskPlan.model_validate_json(invalid_json)

    def test_invalid_step_data(self):
        """Test that invalid step data is rejected."""
        # Step data too short (needs at least 3 elements: id, type, desc)
        invalid_json = '{"name":"test","complexity":"simple","desc":"test","steps":[[1]]}'

        with pytest.raises(ValueError):
            ReadableTaskPlan.model_validate_json(invalid_json)


class TestTokenEfficiency:
    """Tests for token efficiency metrics."""

    def test_simple_plan_token_count(self):
        """Test token count for simple plan."""
        # A typical simple task plan with readable keywords
        readable_json = '{"name":"Fix bug","complexity":"simple","desc":"Fix login bug","steps":[[1,"analyze","Find bug","grep"],[2,"feature","Fix","write"]]}'

        # Count characters (rough proxy for tokens)
        char_count = len(readable_json)
        # Still very compact even with readable keywords
        assert char_count < 140

    def test_moderate_plan_token_count(self):
        """Test token count for moderate plan."""
        moderate_json = """{"name":"Add auth","complexity":"moderate","desc":"OAuth2 login",
        "steps":[[1,"research","Research","overview"],[2,"feature","Create module","write,test"],
        [3,"test","Verify","pytest",[2]]]}"""

        char_count = len(moderate_json)
        assert char_count < 220  # Still very compact

    def test_complex_plan_token_count(self):
        """Test token count for complex plan."""
        complex_json = """{"name":"Deploy API","complexity":"complex","desc":"Production deployment",
        "steps":[[1,"test","Test","pytest"],[2,"analyze","Plan","overview"],[3,"feature","Create config","write"],
        [4,"doc","Document","write"],[5,"test","Integration test","pytest",[4]],[6,"deploy","Deploy","kubectl",[5]]]}"""

        char_count = len(complex_json)
        assert char_count < 350  # Complex but still compact

    def test_token_savings_vs_verbose_json(self):
        """Test token savings compared to verbose JSON."""
        # Verbose version (what NOT to use)
        verbose = {
            "task_name": "Add authentication",
            "complexity": "moderate",
            "description": "Implement OAuth2 login with JWT tokens",
            "steps": [
                {
                    "step_id": 1,
                    "type": "feature",
                    "description": "Create auth module structure",
                    "tools": ["write", "test"],
                    "depends_on": [],
                },
                {
                    "step_id": 2,
                    "type": "feature",
                    "description": "Implement JWT validation",
                    "tools": ["write", "test"],
                    "depends_on": [1],
                },
            ],
        }

        # Readable version
        readable = {
            "name": "Add auth",
            "complexity": "moderate",
            "desc": "OAuth2 login with JWT",
            "steps": [
                [1, "feature", "Create auth structure", "write,test"],
                [2, "feature", "Implement JWT", "write,test", [1]],
            ],
        }

        verbose_size = len(json.dumps(verbose))
        readable_size = len(json.dumps(readable))

        # Readable should be 40-60% smaller
        savings_percent = (1 - readable_size / verbose_size) * 100
        assert savings_percent > 30, f"Expected >30% savings, got {savings_percent}%"

        print(f"\nToken efficiency:")
        print(f"Verbose JSON: {verbose_size} chars")
        print(f"Readable JSON: {readable_size} chars")
        print(f"Savings: {savings_percent:.1f}%")
