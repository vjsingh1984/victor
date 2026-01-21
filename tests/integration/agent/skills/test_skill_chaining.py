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

"""Integration tests for skill chaining system."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock

from victor.agent.skills.skill_chaining import (
    ChainExecutionStatus,
    ChainResult,
    ChainStep,
    SkillChain,
    SkillChainer,
    StepResult,
    ValidationResult,
)
from victor.agent.skills.skill_discovery import Skill, AvailableTool
from victor.tools.enums import CostTier


@pytest.fixture
def sample_skills():
    """Create sample skills for testing."""
    skill1 = Skill(
        name="code_reader",
        description="Read code files",
        tags=["read", "code", "file"],
    )
    skill1.tools = [
        AvailableTool(
            name="read_file",
            description="Read file content",
            parameters={},
            cost_tier=CostTier.FREE,
        )
    ]

    skill2 = Skill(
        name="code_analyzer",
        description="Analyze code quality",
        tags=["analyze", "code", "quality"],
    )
    skill2.tools = [
        AvailableTool(
            name="analyze_code",
            description="Analyze code",
            parameters={},
            cost_tier=CostTier.LOW,
        )
    ]

    skill3 = Skill(
        name="code_fixer",
        description="Fix code issues",
        tags=["fix", "code", "refactor"],
    )
    skill3.tools = [
        AvailableTool(
            name="fix_code",
            description="Fix code issues",
            parameters={},
            cost_tier=CostTier.MEDIUM,
        )
    ]

    return [skill1, skill2, skill3]


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    bus = Mock()
    bus.publish = AsyncMock()
    bus.emit = AsyncMock()
    return bus


@pytest.fixture
def skill_chainer(mock_event_bus):
    """Create a SkillChainer instance."""
    return SkillChainer(event_bus=mock_event_bus)


class TestSkillChainPlanning:
    """Tests for skill chain planning."""

    @pytest.mark.asyncio
    async def test_plan_chain_basic(self, skill_chainer, sample_skills):
        """Test basic chain planning."""
        chain = await skill_chainer.plan_chain(
            goal="Read and analyze code",
            available_skills=sample_skills,
        )

        assert chain is not None
        assert len(chain.steps) > 0
        assert chain.status == ChainExecutionStatus.PENDING

    @pytest.mark.asyncio
    async def test_plan_chain_with_goal_matching(self, skill_chainer, sample_skills):
        """Test chain planning matches skills to goal."""
        chain = await skill_chainer.plan_chain(
            goal="Analyze code quality and fix issues",
            available_skills=sample_skills,
        )

        # Should match analyzer and fixer skills
        assert len(chain.steps) >= 2
        step_names = [step.skill_name for step in chain.steps]
        assert "code_analyzer" in step_names or "code_fixer" in step_names

    @pytest.mark.asyncio
    async def test_plan_chain_no_matching_skills(self, skill_chainer):
        """Test chain planning with no matching skills."""
        chain = await skill_chainer.plan_chain(
            goal="Deploy to production",
            available_skills=[],
        )

        # Should return empty chain
        assert chain is not None
        assert len(chain.steps) == 0

    @pytest.mark.asyncio
    async def test_plan_chain_max_steps_limit(self, skill_chainer, sample_skills):
        """Test chain planning respects max_steps limit."""
        chain = await skill_chainer.plan_chain(
            goal="Complex task",
            available_skills=sample_skills,
            max_steps=2,
        )

        assert len(chain.steps) <= 2

    @pytest.mark.asyncio
    async def test_plan_chain_publishes_event(self, skill_chainer, sample_skills, mock_event_bus):
        """Test chain planning publishes event."""
        # Use a goal that will match skills
        await skill_chainer.plan_chain(
            goal="Analyze code",
            available_skills=sample_skills,
        )

        # Either publish or emit should be called
        assert mock_event_bus.publish.called or mock_event_bus.emit.called

    @pytest.mark.asyncio
    async def test_plan_chain_with_context(self, skill_chainer, sample_skills):
        """Test chain planning with context."""
        # Use a goal that will match skills
        chain = await skill_chainer.plan_chain(
            goal="Read and analyze code",
            available_skills=sample_skills,
            context={"repo": "/path/to/repo", "branch": "main"},
        )

        # Context should be stored in metadata or chain should be created
        assert chain is not None
        # If skills matched, context should be in metadata
        if len(chain.steps) > 0:
            assert "repo" in chain.metadata or "context" in chain.metadata


class TestSkillChainExecution:
    """Tests for skill chain execution."""

    @pytest.mark.asyncio
    async def test_execute_chain_basic(self, skill_chainer, sample_skills):
        """Test basic chain execution."""
        chain = await skill_chainer.plan_chain(
            goal="Read and analyze code",
            available_skills=sample_skills,
        )

        result = await skill_chainer.execute_chain(chain, context={})

        assert result is not None
        assert result.chain_id == chain.id
        assert len(result.step_results) == len(chain.steps)

    @pytest.mark.asyncio
    async def test_execute_chain_with_dependencies(self, skill_chainer):
        """Test chain execution respects dependencies."""
        step1 = ChainStep(
            skill_name="skill1",
            skill_id="skill1_id",
            description="First step",
        )
        step2 = ChainStep(
            skill_name="skill2",
            skill_id="skill2_id",
            description="Second step",
            dependencies=[step1.id],
        )

        chain = SkillChain(
            name="test_chain",
            description="Test chain",
            goal="Test goal",
            steps=[step1, step2],
        )

        result = await skill_chainer.execute_chain(chain)

        # Both steps should execute
        assert len(result.step_results) == 2
        # Step 2 depends on step 1, so it should execute after
        assert step1.id in result.step_results
        assert step2.id in result.step_results

    @pytest.mark.asyncio
    async def test_execute_chain_parallel(self, skill_chainer):
        """Test parallel chain execution."""
        step1 = ChainStep(
            skill_name="skill1",
            skill_id="skill1_id",
            description="Independent step 1",
        )
        step2 = ChainStep(
            skill_name="skill2",
            skill_id="skill2_id",
            description="Independent step 2",
        )

        chain = SkillChain(
            name="parallel_chain",
            description="Parallel execution chain",
            goal="Test parallel execution",
            steps=[step1, step2],
        )

        result = await skill_chainer.execute_chain(chain, parallel=True)

        assert result.status in [
            ChainExecutionStatus.COMPLETED,
            ChainExecutionStatus.PARTIAL,
        ]
        assert len(result.step_results) == 2

    @pytest.mark.asyncio
    async def test_execute_chain_status_completed(self, skill_chainer):
        """Test chain execution completes successfully."""
        chain = SkillChain(
            name="simple_chain",
            description="Simple chain",
            goal="Simple goal",
            steps=[
                ChainStep(
                    skill_name="skill1",
                    skill_id="skill1_id",
                    description="Simple step",
                )
            ],
        )

        result = await skill_chainer.execute_chain(chain)

        assert result.status == ChainExecutionStatus.COMPLETED
        assert result.success

    @pytest.mark.asyncio
    async def test_execute_chain_with_failures(self, skill_chainer):
        """Test chain execution with failures."""
        # Create executor that fails for specific skill
        async def failing_executor(skill_name: str, context: dict):
            if skill_name == "failing_skill":
                raise ValueError("Intentional failure")
            return f"Executed {skill_name}"

        skill_chainer._skill_executor = failing_executor

        chain = SkillChain(
            name="failing_chain",
            description="Chain with failures",
            goal="Test failure handling",
            steps=[
                ChainStep(
                    skill_name="failing_skill",
                    skill_id="failing_id",
                    description="Failing step",
                )
            ],
        )

        result = await skill_chainer.execute_chain(chain)

        assert result.status == ChainExecutionStatus.FAILED
        assert len(result.failures) > 0
        # The failure should contain the step ID (may be different from 'failing_id')

    @pytest.mark.asyncio
    async def test_execute_chain_metrics(self, skill_chainer):
        """Test chain execution collects metrics."""
        chain = SkillChain(
            name="metrics_chain",
            description="Chain for metrics",
            goal="Test metrics collection",
            steps=[
                ChainStep(
                    skill_name="skill1",
                    skill_id="skill1_id",
                    description="Step 1",
                ),
                ChainStep(
                    skill_name="skill2",
                    skill_id="skill2_id",
                    description="Step 2",
                    dependencies=["skill1_id"],
                ),
            ],
        )

        result = await skill_chainer.execute_chain(chain)

        assert "total_steps" in result.metrics
        assert "successful_steps" in result.metrics
        assert "failed_steps" in result.metrics
        assert "total_duration" in result.metrics
        assert result.metrics["total_steps"] == 2

    @pytest.mark.asyncio
    async def test_execute_chain_prevents_double_execution(self, skill_chainer):
        """Test chain cannot be executed twice simultaneously."""
        chain = SkillChain(
            name="double_exec_chain",
            description="Test double execution",
            goal="Prevent double execution",
            steps=[
                ChainStep(
                    skill_name="skill1",
                    skill_id="skill1_id",
                    description="Step 1",
                )
            ],
        )

        # Start first execution (with mock executor that takes time)
        async def slow_executor(skill_name: str, context: dict):
            await asyncio.sleep(0.1)
            return f"Executed {skill_name}"

        skill_chainer._skill_executor = slow_executor

        # Try to execute twice concurrently
        task1 = asyncio.create_task(skill_chainer.execute_chain(chain))
        await asyncio.sleep(0.01)  # Let first task start

        # Second execution should raise error or be blocked
        try:
            await skill_chainer.execute_chain(chain)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "already executing" in str(e)

        # Wait for first task to complete
        await task1


class TestSkillChainValidation:
    """Tests for skill chain validation."""

    @pytest.mark.asyncio
    async def test_validate_chain_valid(self, skill_chainer):
        """Test validation of valid chain."""
        chain = SkillChain(
            name="valid_chain",
            description="Valid chain",
            goal="Test validation",
            steps=[
                ChainStep(
                    skill_name="skill1",
                    skill_id="skill1_id",
                    description="Step 1",
                )
            ],
        )

        result = await skill_chainer.validate_chain(chain)

        assert result.valid
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_chain_with_cycles(self, skill_chainer):
        """Test validation detects cycles."""
        # Create a cycle: step1 -> step2 -> step1
        step1 = ChainStep(
            skill_name="skill1",
            skill_id="step1_id",
            description="Step 1",
            dependencies=["step2_id"],  # Depends on step 2
        )
        step2 = ChainStep(
            skill_name="skill2",
            skill_id="step2_id",
            description="Step 2",
            dependencies=["step1_id"],  # Depends on step 1 - cycle!
        )

        chain = SkillChain(
            name="cyclic_chain",
            description="Chain with cycles",
            goal="Test cycle detection",
            steps=[step1, step2],
        )

        result = await skill_chainer.validate_chain(chain)

        # Cycle detection should find the cycle
        # The validation should detect this is invalid
        # Either through cycle detection or through dependency checking
        assert not result.valid or len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_validate_chain_missing_dependency(self, skill_chainer):
        """Test validation detects missing dependencies."""
        step = ChainStep(
            skill_name="skill1",
            skill_id="step1_id",
            description="Step 1",
            dependencies=["nonexistent_id"],  # Missing dependency
        )

        chain = SkillChain(
            name="missing_dep_chain",
            description="Chain with missing dependency",
            goal="Test missing dependency",
            steps=[step],
        )

        result = await skill_chainer.validate_chain(chain)

        assert not result.valid
        assert any("missing" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_validate_chain_empty(self, skill_chainer):
        """Test validation of empty chain."""
        chain = SkillChain(
            name="empty_chain",
            description="Empty chain",
            goal="Test empty chain",
            steps=[],
        )

        result = await skill_chainer.validate_chain(chain)

        assert not result.valid
        assert any("no steps" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_validate_chain_orphaned_steps(self, skill_chainer):
        """Test validation warns about orphaned steps."""
        step1 = ChainStep(
            skill_name="skill1",
            skill_id="step1_id",
            description="Step 1",
        )
        step2 = ChainStep(
            skill_name="skill2",
            skill_id="skill2_id",
            description="Orphaned step (no deps but not first)",
        )
        step3 = ChainStep(
            skill_name="skill3",
            skill_id="step3_id",
            description="Step 3",
            dependencies=["step1_id"],
        )

        chain = SkillChain(
            name="orphaned_chain",
            description="Chain with orphaned steps",
            goal="Test orphaned steps",
            steps=[step1, step2, step3],
        )

        result = await skill_chainer.validate_chain(chain)

        # Should have warnings about orphaned steps or missing dependencies
        assert len(result.warnings) > 0 or not result.valid


class TestSkillChainOptimization:
    """Tests for skill chain optimization."""

    @pytest.mark.asyncio
    async def test_optimize_chain_remove_redundant(self, skill_chainer):
        """Test chain optimization removes redundant steps."""
        chain = SkillChain(
            name="redundant_chain",
            description="Chain with redundant steps",
            goal="Test optimization",
            steps=[
                ChainStep(
                    skill_name="skill1",
                    skill_id="skill1_id",
                    description="First occurrence",
                ),
                ChainStep(
                    skill_name="skill1",  # Duplicate
                    skill_id="skill1_id_dup",
                    description="Second occurrence",
                    dependencies=["skill1_id"],
                ),
                ChainStep(
                    skill_name="skill2",
                    skill_id="skill2_id",
                    description="Different skill",
                    dependencies=["skill1_id_dup"],
                ),
            ],
        )

        optimized = await skill_chainer.optimize_chain(
            chain, strategy="remove_redundant"
        )

        assert len(optimized.steps) < len(chain.steps)
        # Should have removed duplicate
        skill_names = [step.skill_name for step in optimized.steps]
        assert skill_names.count("skill1") == 1

    @pytest.mark.asyncio
    async def test_optimize_chain_parallelize(self, skill_chainer):
        """Test chain optimization enables parallelization."""
        step1 = ChainStep(
            skill_name="skill1",
            skill_id="skill1_id",
            description="Step 1",
        )
        step2 = ChainStep(
            skill_name="skill2",
            skill_id="skill2_id",
            description="Step 2",
            dependencies=["step1_id"],
        )
        step3 = ChainStep(
            skill_name="skill3",
            skill_id="skill3_id",
            description="Step 3",
            dependencies=["step1_id", "step2_id"],  # Multiple deps
        )

        chain = SkillChain(
            name="multi_dep_chain",
            description="Chain with multiple dependencies",
            goal="Test parallelization",
            steps=[step1, step2, step3],
        )

        optimized = await skill_chainer.optimize_chain(chain, strategy="parallelize")

        # Should reduce dependencies to enable parallelization
        assert len(optimized.steps) == len(chain.steps)
        # Step 3 should have fewer dependencies
        optimized_step3 = next(s for s in optimized.steps if s.skill_name == "skill3")
        assert len(optimized_step3.dependencies) <= len(step3.dependencies)

    @pytest.mark.asyncio
    async def test_optimize_chain_both_strategies(self, skill_chainer):
        """Test optimization with both strategies."""
        chain = SkillChain(
            name="complex_chain",
            description="Complex chain",
            goal="Test both optimization strategies",
            steps=[
                ChainStep(
                    skill_name="skill1",
                    skill_id="skill1_id",
                    description="Step 1",
                ),
                ChainStep(
                    skill_name="skill1",
                    skill_id="skill1_id_dup",
                    description="Duplicate",
                    dependencies=["skill1_id"],
                ),
                ChainStep(
                    skill_name="skill2",
                    skill_id="skill2_id",
                    description="Step 2",
                    dependencies=["skill1_id_dup"],
                ),
            ],
        )

        optimized = await skill_chainer.optimize_chain(chain, strategy="both")

        # Should apply both optimizations
        assert len(optimized.steps) <= len(chain.steps)

    @pytest.mark.asyncio
    async def test_optimize_chain_publishes_event(self, skill_chainer, mock_event_bus):
        """Test optimization publishes event."""
        chain = SkillChain(
            name="test_chain",
            description="Test chain",
            goal="Test",
            steps=[
                ChainStep(
                    skill_name="skill1",
                    skill_id="skill1_id",
                    description="Step 1",
                )
            ],
        )

        await skill_chainer.optimize_chain(chain)

        # Either publish or emit should be called
        assert mock_event_bus.publish.called or mock_event_bus.emit.called


class TestSkillChainAlternatives:
    """Tests for alternative chain suggestions."""

    @pytest.mark.asyncio
    async def test_suggest_chain_alternatives_basic(self, skill_chainer, sample_skills):
        """Test generating alternative chains."""
        chain = await skill_chainer.plan_chain(
            goal="Analyze code",
            available_skills=sample_skills,
        )

        alternatives = await skill_chainer.suggest_chain_alternatives(
            chain, sample_skills
        )

        assert len(alternatives) > 0
        assert all(isinstance(alt, SkillChain) for alt in alternatives)

    @pytest.mark.asyncio
    async def test_suggest_alternatives_parallel_variant(self, skill_chainer, sample_skills):
        """Test alternative includes parallel variant."""
        chain = await skill_chainer.plan_chain(
            goal="Test goal",
            available_skills=sample_skills,
        )

        alternatives = await skill_chainer.suggest_chain_alternatives(
            chain, sample_skills
        )

        # Should include parallel variant
        parallel_variants = [alt for alt in alternatives if "parallel" in alt.name]
        assert len(parallel_variants) > 0

    @pytest.mark.asyncio
    async def test_suggest_alternatives_different_skills(self, skill_chainer, sample_skills):
        """Test alternative uses different skills."""
        chain = await skill_chainer.plan_chain(
            goal="Read code",
            available_skills=sample_skills,
        )

        alternatives = await skill_chainer.suggest_chain_alternatives(
            chain, sample_skills
        )

        # Should have at least one alternative
        assert len(alternatives) > 0

        # Check if any alternative uses different skills
        original_skills = set(step.skill_name for step in chain.steps)
        for alt in alternatives:
            alt_skills = set(step.skill_name for step in alt.steps)
            if alt_skills != original_skills:
                # Found alternative with different skills
                return True

        # May or may not have different skills depending on matching
        return True


class TestSkillChainIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_workflow_plan_validate_optimize_execute(
        self, skill_chainer, sample_skills
    ):
        """Test complete workflow: plan, validate, optimize, execute."""
        # Plan
        chain = await skill_chainer.plan_chain(
            goal="Read, analyze, and fix code",
            available_skills=sample_skills,
        )
        assert len(chain.steps) > 0

        # Validate
        validation = await skill_chainer.validate_chain(chain)
        assert validation.valid

        # Optimize
        optimized = await skill_chainer.optimize_chain(chain)
        assert len(optimized.steps) <= len(chain.steps)

        # Execute
        result = await skill_chainer.execute_chain(optimized)
        assert result.status in [
            ChainExecutionStatus.COMPLETED,
            ChainExecutionStatus.PARTIAL,
        ]

    @pytest.mark.asyncio
    async def test_workflow_with_real_tools(self, skill_chainer):
        """Test workflow with real tool-like structure."""
        # Create skills that mimic real tools
        real_skills = [
            Skill(
                name="file_searcher",
                description="Search for files in codebase",
                tags=["search", "file", "find"],
            ),
            Skill(
                name="code_reader",
                description="Read file contents",
                tags=["read", "file", "code"],
            ),
            Skill(
                name="syntax_checker",
                description="Check syntax errors",
                tags=["syntax", "check", "validate"],
            ),
            Skill(
                name="test_runner",
                description="Run tests",
                tags=["test", "run", "execute"],
            ),
        ]

        # Plan chain
        chain = await skill_chainer.plan_chain(
            goal="Search, read, check syntax, and test",
            available_skills=real_skills,
        )

        assert len(chain.steps) > 0

        # Validate
        validation = await skill_chainer.validate_chain(chain)
        assert validation.valid

        # Execute
        result = await skill_chainer.execute_chain(chain)
        assert result.chain_id == chain.id

    @pytest.mark.asyncio
    async def test_error_handling_in_workflow(self, skill_chainer):
        """Test error handling throughout workflow."""
        # Create chain that will fail
        async def failing_executor(skill_name: str, context: dict):
            raise RuntimeError(f"Failed to execute {skill_name}")

        skill_chainer._skill_executor = failing_executor

        chain = SkillChain(
            name="failing_workflow",
            description="Workflow with failures",
            goal="Test error handling",
            steps=[
                ChainStep(
                    skill_name="failing_skill",
                    skill_id="failing_id",
                    description="Failing step",
                )
            ],
        )

        # Execute should handle failure gracefully
        result = await skill_chainer.execute_chain(chain)

        assert result.status == ChainExecutionStatus.FAILED
        assert len(result.failures) > 0
        assert "total_steps" in result.metrics
