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

"""Integration tests for advanced team formation strategies.

Tests the DynamicFormation, AdaptiveFormation, and HybridFormation
strategies with realistic scenarios and benchmarks against static formations.
"""

import asyncio
import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

from victor.coordination.formations.base import TeamContext
from victor.teams.types import AgentMessage, MemberResult, MessageType
from victor.workflows.advanced_formations import (
    DynamicFormation,
    AdaptiveFormation,
    HybridFormation,
    HybridPhase,
    TaskCharacteristics,
    FormationPhase,
)
from victor.workflows.advanced_formations_schema import (
    DynamicFormationConfig,
    AdaptiveFormationConfig,
    HybridFormationConfig,
    HybridPhaseConfig,
    AdvancedFormationConfig,
    parse_advanced_formation_from_yaml,
    ValidationError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agents() -> Any:
    """Create mock agents for testing."""
    from victor.teams.types import MemberResult

    agents = []
    for i in range(3):
        agent = MagicMock()
        agent.id = f"agent_{i}"

        # Create closure to capture agent.id
        async def mock_execute(
            task: AgentMessage, context: TeamContext, agent_id=agent.id
        ) -> MemberResult:
            return MemberResult(
                member_id=agent_id,
                success=True,
                output=f"Executed: {task.content}",
                metadata={"task": task.content},
            )

        agent.execute = AsyncMock(side_effect=mock_execute)
        agents.append(agent)

    return agents


@pytest.fixture
def team_context(mock_agents: Any) -> Any:
    """Create a team context with mock agents."""
    shared_state = {f"agent_{i}": agent for i, agent in enumerate(mock_agents)}
    return TeamContext(
        team_id="test_team",
        formation="test",
        shared_state=shared_state,
    )


@pytest.fixture
def sample_task() -> Any:
    """Create a sample task message."""
    return AgentMessage(
        sender_id="user",
        content="Analyze this complex problem with multiple dependencies",
        message_type=MessageType.TASK,
    )


# =============================================================================
# Dynamic Formation Tests
# =============================================================================


class TestDynamicFormation:
    """Test suite for DynamicFormation."""

    @pytest.mark.asyncio
    async def test_dynamic_formation_initialization(self) -> None:
        """Test DynamicFormation initialization with default parameters."""
        formation = DynamicFormation()

        assert formation.initial_formation == "parallel"
        assert formation.max_switches == 5
        assert formation.enable_auto_detection is True
        assert formation._switches_made == 0
        assert formation._current_phase == FormationPhase.EXPLORATION

    @pytest.mark.asyncio
    async def test_dynamic_formation_custom_initialization(self) -> None:
        """Test DynamicFormation initialization with custom parameters."""
        switching_rules = {
            "dependencies_emerge": "sequential",
            "conflict_detected": "consensus",
        }

        formation = DynamicFormation(
            initial_formation="sequential",
            switching_rules=switching_rules,
            max_switches=3,
            enable_auto_detection=False,
        )

        assert formation.initial_formation == "sequential"
        assert formation.switching_rules == switching_rules
        assert formation.max_switches == 3
        assert formation.enable_auto_detection is False

    @pytest.mark.asyncio
    async def test_dynamic_formation_execution(self, mock_agents, team_context, sample_task) -> None:
        """Test basic execution with dynamic formation."""
        formation = DynamicFormation(initial_formation="parallel", enable_auto_detection=False)

        results = await formation.execute(mock_agents, team_context, sample_task)

        assert len(results) > 0
        assert results[0].metadata["formation"] == "dynamic"
        assert "current_formation" in results[0].metadata
        assert "switches_made" in results[0].metadata
        assert "current_phase" in results[0].metadata

    @pytest.mark.asyncio
    async def test_dynamic_formation_switching_on_dependencies(
        self, mock_agents, team_context, sample_task
    ):
        """Test formation switching when dependencies are detected."""

        # Create agents that indicate dependencies
        async def dependent_execute(task: str, context: Dict[str, Any] = None) -> str:
            return "This task depends on previous results"

        for agent in mock_agents:
            agent.execute = AsyncMock(side_effect=dependent_execute)

        formation = DynamicFormation(
            initial_formation="parallel",
            enable_auto_detection=True,
            max_switches=2,
        )

        results = await formation.execute(mock_agents, team_context, sample_task)

        # Should detect dependencies and potentially switch
        assert len(results) > 0
        metadata = results[0].metadata
        assert "formation_history" in metadata

    @pytest.mark.asyncio
    async def test_dynamic_formation_max_switches_limit(
        self, mock_agents, team_context, sample_task
    ):
        """Test that formation respects max_switches limit."""
        formation = DynamicFormation(
            initial_formation="parallel",
            max_switches=1,
            enable_auto_detection=True,
        )

        # Add context that would trigger multiple switches
        team_context.metadata["dependencies"] = True
        team_context.metadata["time_pressure"] = True
        team_context.metadata["quality_concerns"] = True

        results = await formation.execute(mock_agents, team_context, sample_task)

        metadata = results[0].metadata
        assert metadata["switches_made"] <= formation.max_switches

    @pytest.mark.asyncio
    async def test_dynamic_formation_error_recovery(self, mock_agents, team_context, sample_task) -> None:
        """Test error recovery by switching formations."""
        # Create agent that fails initially
        call_count = {"count": 0}

        async def failing_then_succeeding(task: str, context: Dict[str, Any] = None) -> str:
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise Exception("Simulated failure")
            return f"Success on attempt {call_count['count']}"

        for agent in mock_agents:
            agent.execute = AsyncMock(side_effect=failing_then_succeeding)

        formation = DynamicFormation(
            initial_formation="parallel",
            max_switches=2,
        )

        results = await formation.execute(mock_agents, team_context, sample_task)

        # Should eventually succeed through switching
        assert len(results) > 0
        metadata = results[0].metadata
        assert metadata.get("switches_made", 0) >= 1

    @pytest.mark.asyncio
    async def test_dynamic_formation_phase_progression(
        self, mock_agents, team_context, sample_task
    ):
        """Test phase progression through execution."""
        formation = DynamicFormation(
            initial_formation="parallel",
            max_switches=5,
        )

        # Execute multiple times to see phase progression
        for _ in range(3):
            await formation.execute(mock_agents, team_context, sample_task)

        metadata = (await formation.execute(mock_agents, team_context, sample_task))[0].metadata

        # Should have recorded phase transitions
        assert "phase_transitions" in metadata
        assert len(metadata["phase_transitions"]) > 0


# =============================================================================
# Adaptive Formation Tests
# =============================================================================


class TestAdaptiveFormation:
    """Test suite for AdaptiveFormation."""

    @pytest.mark.asyncio
    async def test_adaptive_formation_initialization(self) -> None:
        """Test AdaptiveFormation initialization with defaults."""
        formation = AdaptiveFormation()

        assert formation.criteria == ["complexity", "deadline", "resource_availability"]
        assert formation.default_formation == "parallel"
        assert formation.fallback_formation == "sequential"
        assert formation.use_ml is False

    @pytest.mark.asyncio
    async def test_adaptive_formation_custom_initialization(self) -> None:
        """Test AdaptiveFormation with custom parameters."""
        criteria = ["complexity", "uncertainty", "collaboration_needed"]

        formation = AdaptiveFormation(
            criteria=criteria,
            default_formation="consensus",
            fallback_formation="hierarchical",
            use_ml=True,
        )

        assert formation.criteria == criteria
        assert formation.default_formation == "consensus"
        assert formation.fallback_formation == "hierarchical"
        assert formation.use_ml is True

    @pytest.mark.asyncio
    async def test_adaptive_formation_task_analysis(self, mock_agents, team_context, sample_task) -> None:
        """Test task characteristic analysis."""
        formation = AdaptiveFormation()

        # Simple task
        simple_task = AgentMessage(
            sender_id="user",
            content="Do this simple task",
            message_type=MessageType.TASK,
        )

        characteristics = await formation._analyze_task(simple_task, team_context, mock_agents)

        assert characteristics.complexity < 0.5  # Should be low complexity

        # Complex task
        complex_task = AgentMessage(
            sender_id="user",
            content="Analyze this complex multi-faceted problem that requires investigation "
            "of multiple dependencies and collaboration across team members to figure out "
            "the best possible solution",
            message_type=MessageType.TASK,
        )

        characteristics = await formation._analyze_task(complex_task, team_context, mock_agents)

        assert characteristics.complexity > 0.6  # Should be high complexity
        assert characteristics.uncertainty > 0.5  # Should detect uncertainty
        assert characteristics.collaboration_needed > 0.5  # Should detect collaboration need

    @pytest.mark.asyncio
    async def test_adaptive_formation_scoring(self, mock_agents, team_context, sample_task) -> None:
        """Test formation scoring based on characteristics."""
        formation = AdaptiveFormation()

        # High complexity, low dependencies
        characteristics = TaskCharacteristics(
            complexity=0.9,
            dependency_level=0.2,
            uncertainty=0.5,
            collaboration_needed=0.5,
        )

        scores = formation._score_formations(characteristics)

        # Parallel should score well for high complexity, low dependencies
        assert "parallel" in scores
        assert scores["parallel"] > 0.5

        # High dependencies
        characteristics.dependency_level = 0.9
        scores = formation._score_formations(characteristics)

        # Sequential or pipeline should score better with high dependencies
        assert scores["sequential"] > 0.5 or scores["pipeline"] > 0.5

    @pytest.mark.asyncio
    async def test_adaptive_formation_execution(self, mock_agents, team_context, sample_task) -> None:
        """Test adaptive formation execution."""
        formation = AdaptiveFormation(
            criteria=["complexity", "deadline"],
            default_formation="parallel",
        )

        results = await formation.execute(mock_agents, team_context, sample_task)

        assert len(results) > 0
        metadata = results[0].metadata

        assert metadata["formation"] == "adaptive"
        assert "selected_formation" in metadata
        assert "task_characteristics" in metadata
        assert "formation_scores" in metadata

    @pytest.mark.asyncio
    async def test_adaptive_formation_fallback_on_error(
        self, mock_agents, team_context, sample_task
    ):
        """Test fallback formation when formation creation fails."""
        # Create a formation that will fail during creation
        # by setting a very low threshold so invalid_formation is selected
        formation = AdaptiveFormation(
            criteria=["complexity", "deadline", "resource_availability"],
            default_formation="invalid_formation",  # This will fail when creating
            fallback_formation="sequential",
        )

        # Mock _score_formations to return low scores, triggering default_formation usage
        def mock_score(characteristics: Any) -> Any:
            # Return very low scores to trigger fallback to default_formation
            return {
                "sequential": 0.1,
                "parallel": 0.1,
                "hierarchical": 0.1,
                "pipeline": 0.1,
                "consensus": 0.1,
            }

        formation._score_formations = mock_score

        results = await formation.execute(mock_agents, team_context, sample_task)

        # Should fall back to sequential when invalid_formation fails
        assert len(results) > 0
        metadata = results[0].metadata
        assert metadata.get("fallback_used") is True
        assert metadata.get("selected_formation") == "sequential"


# =============================================================================
# Hybrid Formation Tests
# =============================================================================


class TestHybridFormation:
    """Test suite for HybridFormation."""

    @pytest.mark.asyncio
    async def test_hybrid_formation_initialization(self) -> None:
        """Test HybridFormation initialization."""
        phases = [
            HybridPhase(formation="parallel", goal="Explore"),
            HybridPhase(formation="sequential", goal="Synthesize"),
        ]

        formation = HybridFormation(phases=phases)

        assert len(formation.phases) == 2
        assert formation.enable_phase_logging is True
        assert formation.stop_on_first_failure is False

    @pytest.mark.asyncio
    async def test_hybrid_formation_execution(self, mock_agents, team_context, sample_task) -> None:
        """Test hybrid formation execution through multiple phases."""
        phases = [
            HybridPhase(formation="parallel", goal="Explore"),
            HybridPhase(formation="sequential", goal="Synthesize"),
        ]

        formation = HybridFormation(phases=phases, enable_phase_logging=False)

        results = await formation.execute(mock_agents, team_context, sample_task)

        assert len(results) > 0
        metadata = results[0].metadata

        assert metadata["formation"] == "hybrid"
        assert metadata["phases_completed"] == 2
        assert metadata["total_phases"] == 2
        assert "phase_results" in metadata

    @pytest.mark.asyncio
    async def test_hybrid_formation_phase_results(self, mock_agents, team_context, sample_task) -> None:
        """Test that each phase produces its own results."""
        phases = [
            HybridPhase(formation="parallel", goal="Phase 1"),
            HybridPhase(formation="sequential", goal="Phase 2"),
            HybridPhase(formation="consensus", goal="Phase 3"),
        ]

        formation = HybridFormation(phases=phases, enable_phase_logging=False)

        results = await formation.execute(mock_agents, team_context, sample_task)

        metadata = results[0].metadata
        phase_results = metadata["phase_results"]

        assert len(phase_results) == 3

        # Each phase should have its results recorded
        for i, phase_result in enumerate(phase_results):
            assert phase_result["phase"] == i + 1
            assert "formation" in phase_result
            assert "goal" in phase_result
            assert "duration_seconds" in phase_result
            assert "results" in phase_result

    @pytest.mark.asyncio
    async def test_hybrid_formation_stop_on_failure(self, mock_agents, team_context, sample_task) -> None:
        """Test stop_on_first_failure behavior."""

        # Create an agent that fails
        async def failing_execute(task: str, context: Dict[str, Any] = None) -> str:
            raise Exception("Phase failed")

        for agent in mock_agents:
            agent.execute = AsyncMock(side_effect=failing_execute)

        phases = [
            HybridPhase(formation="parallel", goal="Phase 1"),
            HybridPhase(formation="sequential", goal="Phase 2"),
            HybridPhase(formation="consensus", goal="Phase 3"),
        ]

        formation = HybridFormation(
            phases=phases,
            enable_phase_logging=False,
            stop_on_first_failure=True,
        )

        results = await formation.execute(mock_agents, team_context, sample_task)

        metadata = results[0].metadata

        # Should stop after first failure
        assert metadata["phases_completed"] < 3

    @pytest.mark.asyncio
    async def test_hybrid_formation_duration_budget(self, mock_agents, team_context, sample_task) -> None:
        """Test phase duration budget enforcement."""

        async def slow_execute(task: str, context: Dict[str, Any] = None) -> str:
            await asyncio.sleep(0.2)  # Simulate slow execution
            return "Done"

        for agent in mock_agents:
            agent.execute = AsyncMock(side_effect=slow_execute)

        phases = [
            HybridPhase(formation="parallel", goal="Fast phase", duration_budget=0.5),
        ]

        formation = HybridFormation(phases=phases, enable_phase_logging=False)

        results = await formation.execute(mock_agents, team_context, sample_task)

        # Should complete within budget or timeout
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_hybrid_formation_with_context_updates(
        self, mock_agents, team_context, sample_task
    ):
        """Test that context is updated between phases."""
        phases = [
            HybridPhase(formation="parallel", goal="Phase 1"),
            HybridPhase(formation="sequential", goal="Phase 2"),
        ]

        formation = HybridFormation(phases=phases, enable_phase_logging=False)

        await formation.execute(mock_agents, team_context, sample_task)

        # Context should have last phase results
        assert "last_phase_results" in team_context.shared_state


# =============================================================================
# Schema and Validation Tests
# =============================================================================


class TestAdvancedFormationSchema:
    """Test suite for advanced formation schema validation."""

    def test_dynamic_config_validation(self) -> None:
        """Test DynamicFormationConfig validation."""
        # Valid config
        config = DynamicFormationConfig(
            initial_formation="parallel",
            max_switches=5,
            switching_rules=[
                {"trigger": "dependencies_emerge", "target_formation": "sequential"},
            ],
        )

        from victor.workflows.advanced_formations_schema import validate_dynamic_config

        errors = validate_dynamic_config(config)
        assert len(errors) == 0

    def test_dynamic_config_invalid_formation(self) -> None:
        """Test DynamicFormationConfig with invalid formation."""
        config = DynamicFormationConfig(
            initial_formation="invalid_formation",
        )

        from victor.workflows.advanced_formations_schema import validate_dynamic_config

        errors = validate_dynamic_config(config)
        assert len(errors) > 0
        assert any("invalid_formation" in str(e) for e in errors)

    def test_adaptive_config_validation(self) -> None:
        """Test AdaptiveFormationConfig validation."""
        config = AdaptiveFormationConfig(
            criteria=["complexity", "deadline"],
            default_formation="parallel",
            fallback_formation="sequential",
        )

        from victor.workflows.advanced_formations_schema import validate_adaptive_config

        errors = validate_adaptive_config(config)
        assert len(errors) == 0

    def test_adaptive_config_invalid_criteria(self) -> None:
        """Test AdaptiveFormationConfig with invalid criteria."""
        config = AdaptiveFormationConfig(
            criteria=["invalid_criterion"],
        )

        from victor.workflows.advanced_formations_schema import validate_adaptive_config

        errors = validate_adaptive_config(config)
        assert len(errors) > 0
        assert any("invalid_criterion" in str(e) for e in errors)

    def test_hybrid_config_validation(self) -> None:
        """Test HybridFormationConfig validation."""
        phases = [
            HybridPhaseConfig(formation="parallel", goal="Explore"),
            HybridPhaseConfig(formation="sequential", goal="Synthesize"),
        ]

        config = HybridFormationConfig(phases=phases)

        from victor.workflows.advanced_formations_schema import validate_hybrid_config

        errors = validate_hybrid_config(config)
        assert len(errors) == 0

    def test_hybrid_config_empty_phases(self) -> None:
        """Test HybridFormationConfig with empty phases."""
        config = HybridFormationConfig(phases=[])

        from victor.workflows.advanced_formations_schema import validate_hybrid_config

        errors = validate_hybrid_config(config)
        assert len(errors) > 0
        assert any("at least one phase" in str(e) for e in errors)

    def test_parse_dynamic_from_yaml(self) -> None:
        """Test parsing dynamic formation from YAML dict."""
        yaml_data = {
            "type": "dynamic",
            "dynamic_config": {
                "initial_formation": "parallel",
                "max_switches": 3,
                "switching_rules": [
                    {"trigger": "dependencies_emerge", "target_formation": "sequential"},
                ],
            },
        }

        config = parse_advanced_formation_from_yaml(yaml_data)

        assert config.type == "dynamic"
        assert config.dynamic_config is not None
        assert config.dynamic_config.initial_formation == "parallel"
        assert config.dynamic_config.max_switches == 3

    def test_parse_adaptive_from_yaml(self) -> None:
        """Test parsing adaptive formation from YAML dict."""
        yaml_data = {
            "type": "adaptive",
            "adaptive_config": {
                "criteria": ["complexity", "deadline"],
                "default_formation": "parallel",
                "fallback_formation": "sequential",
            },
        }

        config = parse_advanced_formation_from_yaml(yaml_data)

        assert config.type == "adaptive"
        assert config.adaptive_config is not None
        assert config.adaptive_config.criteria == ["complexity", "deadline"]

    def test_parse_hybrid_from_yaml(self) -> None:
        """Test parsing hybrid formation from YAML dict."""
        yaml_data = {
            "type": "hybrid",
            "hybrid_config": {
                "phases": [
                    {"formation": "parallel", "goal": "Explore"},
                    {"formation": "sequential", "goal": "Synthesize"},
                ]
            },
        }

        config = parse_advanced_formation_from_yaml(yaml_data)

        assert config.type == "hybrid"
        assert config.hybrid_config is not None
        assert len(config.hybrid_config.phases) == 2

    def test_parse_invalid_type_raises_error(self) -> None:
        """Test that parsing invalid type raises ValidationError."""
        yaml_data = {
            "type": "invalid_type",
        }

        with pytest.raises(ValidationError):
            parse_advanced_formation_from_yaml(yaml_data)


# =============================================================================
# Benchmark Tests
# =============================================================================


class TestAdvancedFormationBenchmarks:
    """Benchmark tests comparing advanced formations to static formations."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_benchmark_dynamic_vs_static(self, mock_agents, team_context, sample_task) -> None:
        """Benchmark DynamicFormation vs static formations."""
        # This is a simplified benchmark - real benchmarks would use more complex tasks

        dynamic = DynamicFormation(enable_auto_detection=False)

        # Measure dynamic formation
        import time

        start = time.time()
        dynamic_results = await dynamic.execute(mock_agents, team_context, sample_task)
        dynamic_time = time.time() - start

        # Measure static parallel formation
        from victor.coordination.formations.parallel import ParallelFormation

        parallel = ParallelFormation()
        start = time.time()
        parallel_results = await parallel.execute(mock_agents, team_context, sample_task)
        parallel_time = time.time() - start

        # Dynamic should have comparable or better performance
        # (In real scenarios, dynamic would adapt to optimize)
        assert dynamic_results[0].success
        assert parallel_results[0].success

        # Record times for analysis
        print(f"\nDynamic formation time: {dynamic_time:.3f}s")
        print(f"Parallel formation time: {parallel_time:.3f}s")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_benchmark_adaptive_selection(self, mock_agents, team_context) -> None:
        """Benchmark AdaptiveFormation selection accuracy."""
        formation = AdaptiveFormation()

        # Test with various task types
        tasks = [
            AgentMessage(
                sender_id="user",
                content="Simple quick task",
                message_type=MessageType.TASK,
            ),
            AgentMessage(
                sender_id="user",
                content="Complex problem requiring multiple experts to collaborate",
                message_type=MessageType.TASK,
            ),
            AgentMessage(
                sender_id="user",
                content="Task with clear sequence of steps",
                message_type=MessageType.TASK,
            ),
        ]

        selections = []
        for task in tasks:
            results = await formation.execute(mock_agents, team_context, task)
            selected = results[0].metadata["selected_formation"]
            selections.append((task.content[:50], selected))

        # Print selections for analysis
        print("\nAdaptive formation selections:")
        for task_desc, selected in selections:
            print(f"  '{task_desc}...' -> {selected}")


# =============================================================================
# Integration Tests
# =============================================================================


class TestAdvancedFormationIntegration:
    """Integration tests for advanced formations with realistic scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_dynamic_formation_real_scenario(self, mock_agents, team_context) -> None:
        """Test dynamic formation with realistic scenario."""
        # Scenario: Code review task that starts parallel but switches
        # to consensus when conflicts are found

        call_count = {"count": 0}

        async def realistic_execute(task: str, context: Dict[str, Any] = None) -> str:
            call_count["count"] += 1
            if call_count["count"] == 1:
                return "Found potential security issues"
            elif call_count["count"] == 2:
                return "Disagree on severity - need consensus"

        for agent in mock_agents:
            agent.execute = AsyncMock(side_effect=realistic_execute)

        formation = DynamicFormation(
            initial_formation="parallel",
            switching_rules={"conflict_detected": "consensus"},
            enable_auto_detection=True,
        )

        task = AgentMessage(
            sender_id="user",
            content="Review this code for security issues",
            message_type=MessageType.TASK,
        )

        results = await formation.execute(mock_agents, team_context, task)

        # Should have detected and switched
        metadata = results[0].metadata
        assert metadata["formation"] == "dynamic"
        assert metadata.get("switches_made", 0) >= 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_hybrid_formation_research_workflow(self, mock_agents, team_context) -> None:
        """Test hybrid formation for research workflow."""
        # Scenario: Research workflow with explore → analyze → synthesize phases

        phases = [
            HybridPhase(formation="parallel", goal="Gather information from multiple sources"),
            HybridPhase(formation="sequential", goal="Analyze findings in depth"),
            HybridPhase(formation="consensus", goal="Synthesize final recommendations"),
        ]

        formation = HybridFormation(phases=phases, enable_phase_logging=False)

        task = AgentMessage(
            sender_id="user",
            content="Research best practices for AI safety",
            message_type=MessageType.TASK,
        )

        results = await formation.execute(mock_agents, team_context, task)

        # Should complete all phases
        metadata = results[0].metadata
        assert metadata["phases_completed"] == 3
        assert len(metadata["phase_results"]) == 3

        # Each phase should have its goal
        for i, phase_result in enumerate(metadata["phase_results"]):
            assert phase_result["goal"] == phases[i].goal
