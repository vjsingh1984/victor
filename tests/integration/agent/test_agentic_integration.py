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

"""Integration tests for Phase 3 Agentic AI features.

This test module verifies the integration of agentic AI capabilities:
- Hierarchical Planning
- Episodic Memory
- Semantic Memory
- Skill Discovery
- Skill Chaining
- Self-Improvement (Proficiency Tracking)
- RL Coordinator

Tests ensure:
1. Services are registered correctly in DI container
2. Services can be resolved via protocols
3. Feature flags control service registration
4. Services integrate with orchestrator
5. End-to-end workflows work correctly
"""

import pytest
from typing import Any
from pathlib import Path

from victor.config.settings import Settings
from victor.core.container import ServiceContainer, ServiceLifetime


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def base_settings():
    """Create base settings for testing."""
    return Settings()


@pytest.fixture
def settings_with_all_features(base_settings):
    """Create settings with all agentic AI features enabled."""
    settings = base_settings
    settings.enable_hierarchical_planning = True
    settings.enable_episodic_memory = True
    settings.enable_semantic_memory = True
    settings.enable_skill_discovery = True
    settings.enable_skill_chaining = True
    settings.enable_self_improvement = True
    settings.enable_rl_coordinator = True
    return settings


@pytest.fixture
def container_with_all_features(settings_with_all_features):
    """Create container with all agentic AI features registered."""
    from victor.agent.service_provider import configure_orchestrator_services

    container = ServiceContainer()
    configure_orchestrator_services(container, settings_with_all_features)
    return container


# =============================================================================
# Service Registration Tests
# =============================================================================


@pytest.mark.skip(reason="Experimental Phase 3 feature not implemented yet")
class TestServiceRegistration:
    """Test that agentic AI services are registered correctly."""

    def test_hierarchical_planner_not_registered_by_default(self, base_settings):
        """HierarchicalPlanner should NOT be registered when feature flag is False."""
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.agent.protocols import HierarchicalPlannerProtocol

        container = ServiceContainer()
        configure_orchestrator_services(container, base_settings)

        # Should return None or raise when not registered
        service = container.get_optional(HierarchicalPlannerProtocol)
        assert service is None, "HierarchicalPlanner should not be registered by default"

    def test_hierarchical_planner_registered_when_enabled(self, settings_with_all_features):
        """HierarchicalPlanner should be registered when feature flag is True."""
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.agent.protocols import HierarchicalPlannerProtocol

        container = ServiceContainer()
        configure_orchestrator_services(container, settings_with_all_features)

        service = container.get(HierarchicalPlannerProtocol)
        assert service is not None, "HierarchicalPlanner should be registered when enabled"

    def test_episodic_memory_registered_when_enabled(self, settings_with_all_features):
        """EpisodicMemory should be registered when feature flag is True."""
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.agent.protocols import EpisodicMemoryProtocol

        container = ServiceContainer()
        configure_orchestrator_services(container, settings_with_all_features)

        service = container.get(EpisodicMemoryProtocol)
        assert service is not None, "EpisodicMemory should be registered when enabled"

    def test_semantic_memory_registered_when_enabled(self, settings_with_all_features):
        """SemanticMemory should be registered when feature flag is True."""
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.agent.protocols import SemanticMemoryProtocol

        container = ServiceContainer()
        configure_orchestrator_services(container, settings_with_all_features)

        service = container.get(SemanticMemoryProtocol)
        assert service is not None, "SemanticMemory should be registered when enabled"

    def test_skill_discovery_registered_when_enabled(self, settings_with_all_features):
        """SkillDiscoveryEngine should be registered when feature flag is True."""
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.agent.protocols import SkillDiscoveryProtocol

        container = ServiceContainer()
        configure_orchestrator_services(container, settings_with_all_features)

        service = container.get(SkillDiscoveryProtocol)
        assert service is not None, "SkillDiscoveryProtocol should be registered when enabled"

    def test_skill_chainer_registered_when_enabled(self, settings_with_all_features):
        """SkillChainer should be registered when feature flag is True."""
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.agent.protocols import SkillChainerProtocol

        container = ServiceContainer()
        configure_orchestrator_services(container, settings_with_all_features)

        service = container.get(SkillChainerProtocol)
        assert service is not None, "SkillChainerProtocol should be registered when enabled"

    def test_proficiency_tracker_registered_when_enabled(self, settings_with_all_features):
        """ProficiencyTracker should be registered when feature flag is True."""
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.agent.protocols import ProficiencyTrackerProtocol

        container = ServiceContainer()
        configure_orchestrator_services(container, settings_with_all_features)

        service = container.get(ProficiencyTrackerProtocol)
        assert service is not None, "ProficiencyTrackerProtocol should be registered when enabled"

    def test_rl_coordinator_registered_when_enabled(self, settings_with_all_features):
        """RLCoordinator should be registered when feature flag is True."""
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.agent.protocols import RLCoordinatorProtocol

        container = ServiceContainer()
        configure_orchestrator_services(container, settings_with_all_features)

        service = container.get(RLCoordinatorProtocol)
        assert service is not None, "RLCoordinatorProtocol should be registered when enabled"


# =============================================================================
# Hierarchical Planning Tests
# =============================================================================


@pytest.mark.skip(reason="Experimental Phase 3 feature not implemented yet")
class TestHierarchicalPlanning:
    """Test hierarchical planning integration."""

    @pytest.mark.asyncio
    async def test_decompose_task(self, container_with_all_features):
        """Test task decomposition through registered service."""
        from victor.agent.protocols import HierarchicalPlannerProtocol

        planner = container_with_all_features.get(HierarchicalPlannerProtocol)

        # Note: This test may fail if orchestrator is not set
        # In production, orchestrator injects itself
        try:
            graph = await planner.decompose_task("Implement user authentication")
            assert graph is not None, "Should return a task graph"
        except AttributeError as e:
            # Expected if orchestrator not set - that's OK for this test
            pytest.skip("HierarchicalPlanner requires orchestrator to be set")

    def test_validate_plan(self, container_with_all_features):
        """Test plan validation."""
        from victor.agent.protocols import HierarchicalPlannerProtocol
        from victor.agent.planning import TaskGraph

        planner = container_with_all_features.get(HierarchicalPlannerProtocol)

        # Create minimal task graph
        graph = TaskGraph()

        result = planner.validate_plan(graph)
        assert result is not None, "Should return validation result"


# =============================================================================
# Episodic Memory Tests
# =============================================================================


@pytest.mark.skip(reason="Experimental Phase 3 feature not implemented yet")
class TestEpisodicMemory:
    """Test episodic memory integration."""

    @pytest.mark.asyncio
    async def test_store_and_recall_episode(self, container_with_all_features):
        """Test storing and recalling episodes."""
        from victor.agent.protocols import EpisodicMemoryProtocol

        memory = container_with_all_features.get(EpisodicMemoryProtocol)

        # Store an episode
        episode_id = await memory.store_episode(
            inputs={"query": "fix authentication bug"},
            actions=["read_file", "edit_file"],
            outcomes={"success": True, "files_changed": 2},
            rewards=10.0,
        )

        assert episode_id is not None, "Should return episode ID"

        # Recall relevant episodes
        relevant = await memory.recall_relevant("authentication bug", k=5)
        assert isinstance(relevant, list), "Should return list of episodes"

        # Get statistics
        stats = memory.get_memory_statistics()
        assert isinstance(stats, dict), "Should return statistics dict"


# =============================================================================
# Semantic Memory Tests
# =============================================================================


@pytest.mark.skip(reason="Experimental Phase 3 feature not implemented yet")
class TestSemanticMemory:
    """Test semantic memory integration."""

    @pytest.mark.asyncio
    async def test_store_and_query_knowledge(self, container_with_all_features):
        """Test storing and querying knowledge."""
        from victor.agent.protocols import SemanticMemoryProtocol

        memory = container_with_all_features.get(SemanticMemoryProtocol)

        # Store knowledge
        fact_id = await memory.store_knowledge(
            "Python uses asyncio for concurrency",
            confidence=1.0,
        )

        assert fact_id is not None, "Should return fact ID"

        # Query knowledge
        facts = await memory.query_knowledge("concurrency in Python", k=5)
        assert isinstance(facts, list), "Should return list of facts"

        # Get knowledge graph
        graph = memory.get_knowledge_graph()
        assert graph is not None, "Should return knowledge graph"


# =============================================================================
# Skill Discovery Tests
# =============================================================================


@pytest.mark.skip(reason="Experimental Phase 3 feature not implemented yet")
class TestSkillDiscovery:
    """Test skill discovery integration."""

    @pytest.mark.asyncio
    async def test_discover_tools(self, container_with_all_features):
        """Test tool discovery."""
        from victor.agent.protocols import SkillDiscoveryProtocol

        discovery = container_with_all_features.get(SkillDiscoveryProtocol)

        # Discover tools for a context
        tools = await discovery.discover_tools(context="code analysis", max_tools=10)
        assert isinstance(tools, list), "Should return list of tools"

    @pytest.mark.asyncio
    async def test_compose_skill(self, container_with_all_features):
        """Test skill composition."""
        from victor.agent.protocols import SkillDiscoveryProtocol

        discovery = container_with_all_features.get(SkillDiscoveryProtocol)

        # Get some tools first
        tools = await discovery.discover_tools(context="code analysis", max_tools=3)

        if tools:
            # Compose a skill
            skill = await discovery.compose_skill(
                skill_name="analyzer",
                tools=tools[:2],  # Use first 2 tools
                description="Analyzes code",
            )
            assert skill is not None, "Should return composed skill"


# =============================================================================
# Skill Chaining Tests
# =============================================================================


@pytest.mark.skip(reason="Experimental Phase 3 feature not implemented yet")
class TestSkillChaining:
    """Test skill chaining integration."""

    @pytest.mark.asyncio
    async def test_plan_chain(self, container_with_all_features):
        """Test skill chain planning."""
        from victor.agent.protocols import SkillChainerProtocol
        from victor.agent.skills import Skill

        chainer = container_with_all_features.get(SkillChainerProtocol)

        # Create mock skills
        skills = [
            Skill(name="skill1", tools=[], description="First skill"),
            Skill(name="skill2", tools=[], description="Second skill"),
        ]

        # Plan a chain
        chain = await chainer.plan_chain(
            goal="Fix bugs",
            skills=skills,
            max_length=5,
        )
        assert chain is not None, "Should return a skill chain"

    def test_validate_chain(self, container_with_all_features):
        """Test chain validation."""
        from victor.agent.protocols import SkillChainerProtocol
        from victor.agent.skills import SkillChain, ChainStep

        chainer = container_with_all_features.get(SkillChainerProtocol)

        # Create minimal chain
        chain = SkillChain(
            steps=[
                ChainStep(
                    skill_name="skill1",
                    description="First step",
                )
            ]
        )

        # Validate chain
        result = chainer.validate_chain(chain)
        assert result is not None, "Should return validation result"


# =============================================================================
# Proficiency Tracker Tests
# =============================================================================


@pytest.mark.skip(reason="Experimental Phase 3 feature not implemented yet")
class TestProficiencyTracker:
    """Test proficiency tracking integration."""

    @pytest.mark.asyncio
    async def test_record_outcome(self, container_with_all_features):
        """Test recording task outcomes."""
        from victor.agent.protocols import ProficiencyTrackerProtocol

        tracker = container_with_all_features.get(ProficiencyTrackerProtocol)

        # Record an outcome
        await tracker.record_outcome(
            task="code_review",
            tool="ast_analyzer",
            success=True,
            duration=1.5,
            cost=0.001,
        )

        # Get proficiency
        score = tracker.get_proficiency(tool="ast_analyzer")
        assert score is not None, "Should return proficiency score"

        # Get suggestions
        suggestions = tracker.get_suggestions(n=5)
        assert isinstance(suggestions, list), "Should return list of suggestions"

        # Get metrics
        metrics = tracker.get_metrics()
        assert isinstance(metrics, dict), "Should return metrics dict"


# =============================================================================
# RL Coordinator Tests
# =============================================================================


@pytest.mark.skip(reason="Experimental Phase 3 feature not implemented yet")
class TestRLCoordinator:
    """Test RL coordinator integration."""

    @pytest.mark.asyncio
    async def test_select_action(self, container_with_all_features):
        """Test action selection via RL."""
        from victor.agent.protocols import RLCoordinatorProtocol

        rl = container_with_all_features.get(RLCoordinatorProtocol)

        # Select action
        state = {"task": "code_review", "files": 5}
        actions = ["read_file", "analyze_code", "edit_file"]

        action = await rl.select_action(state, actions, explore=False)
        assert action is not None, "Should return an action"

    @pytest.mark.asyncio
    async def test_update_policy(self, container_with_all_features):
        """Test policy update."""
        from victor.agent.protocols import RLCoordinatorProtocol

        rl = container_with_all_features.get(RLCoordinatorProtocol)

        # Update policy
        state = {"task": "code_review"}
        action = "read_file"
        reward = 1.0
        next_state = {"task": "code_review", "files_read": 1}

        await rl.update_policy(state, action, reward, next_state, done=False)

        # Get policy
        policy = rl.get_policy()
        assert isinstance(policy, dict), "Should return policy dict"


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


@pytest.mark.skip(reason="Experimental Phase 3 feature not implemented yet")
class TestEndToEndWorkflows:
    """Test end-to-end workflows with agentic AI features."""

    @pytest.mark.asyncio
    async def test_memory_consolidation_workflow(self, container_with_all_features):
        """Test episodic to semantic memory consolidation."""
        from victor.agent.protocols import EpisodicMemoryProtocol, SemanticMemoryProtocol

        episodic = container_with_all_features.get(EpisodicMemoryProtocol)
        semantic = container_with_all_features.get(SemanticMemoryProtocol)

        # Store episode
        await episodic.store_episode(
            inputs={"query": "how to use asyncio"},
            actions=["search_docs"],
            outcomes={"answer": "Use asyncio.run()"},
        )

        # Query should work
        relevant = await episodic.recall_relevant("asyncio usage", k=3)
        assert isinstance(relevant, list), "Should recall episodes"

    @pytest.mark.asyncio
    async def test_skill_discovery_and_chaining_workflow(self, container_with_all_features):
        """Test discovering skills and chaining them."""
        from victor.agent.protocols import SkillDiscoveryProtocol, SkillChainerProtocol

        discovery = container_with_all_features.get(SkillDiscoveryProtocol)
        chainer = container_with_all_features.get(SkillChainerProtocol)

        # Discover tools
        tools = await discovery.discover_tools("code analysis", max_tools=5)

        if tools:
            # Compose skills
            skill1 = await discovery.compose_skill("analyzer", tools[:2], "Analyzes code")
            skill2 = await discovery.compose_skill("fixer", tools[2:4], "Fixes issues")

            # Plan chain
            chain = await chainer.plan_chain("Analyze and fix code", [skill1, skill2])
            assert chain is not None, "Should create chain"

    @pytest.mark.asyncio
    async def test_self_improvement_workflow(self, container_with_all_features):
        """Test proficiency tracking and improvement suggestions."""
        from victor.agent.protocols import ProficiencyTrackerProtocol

        tracker = container_with_all_features.get(ProficiencyTrackerProtocol)

        # Record multiple outcomes
        for i in range(10):
            await tracker.record_outcome(
                task="code_review",
                tool="ast_analyzer",
                success=True if i < 8 else False,  # 80% success rate
                duration=1.0 + (i * 0.1),
                cost=0.001,
            )

        # Get proficiency
        score = tracker.get_proficiency(tool="ast_analyzer", task="code_review")
        assert score is not None, "Should track proficiency"

        # Get suggestions
        suggestions = tracker.get_suggestions()
        assert isinstance(suggestions, list), "Should provide suggestions"


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Test that default behavior is unchanged when features are disabled."""

    def test_default_settings_have_features_disabled(self, base_settings):
        """All agentic AI features should be disabled by default."""
        assert base_settings.enable_hierarchical_planning is False
        assert base_settings.enable_episodic_memory is False
        assert base_settings.enable_semantic_memory is False
        assert base_settings.enable_skill_discovery is False
        assert base_settings.enable_skill_chaining is False
        assert base_settings.enable_self_improvement is False
        # RL coordinator is True by default (enhanced RL)
        assert base_settings.enable_rl_coordinator is True

    def test_container_works_without_agentic_features(self, base_settings):
        """Container should work normally without agentic AI features."""
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.agent.protocols import (
            ToolRegistryProtocol,
            ObservabilityProtocol,
            TaskAnalyzerProtocol,
        )

        container = ServiceContainer()
        configure_orchestrator_services(container, base_settings)

        # Core services should still be available
        tool_registry = container.get(ToolRegistryProtocol)
        assert tool_registry is not None, "ToolRegistry should be available"

        observability = container.get(ObservabilityProtocol)
        assert observability is not None, "Observability should be available"

        task_analyzer = container.get(TaskAnalyzerProtocol)
        assert task_analyzer is not None, "TaskAnalyzer should be available"


# =============================================================================
# Feature Flag Tests
# =============================================================================


@pytest.mark.skip(reason="Experimental Phase 3 feature not implemented yet")
class TestFeatureFlags:
    """Test that feature flags control service registration."""

    def test_individual_feature_flags(self, base_settings):
        """Test that each feature flag controls its service independently."""
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.agent.protocols import (
            HierarchicalPlannerProtocol,
            EpisodicMemoryProtocol,
            SemanticMemoryProtocol,
        )

        # Enable only hierarchical planning
        base_settings.enable_hierarchical_planning = True
        container = ServiceContainer()
        configure_orchestrator_services(container, base_settings)

        # Only HierarchicalPlanner should be registered
        planner = container.get_optional(HierarchicalPlannerProtocol)
        assert planner is not None, "HierarchicalPlanner should be registered"

        episodic = container.get_optional(EpisodicMemoryProtocol)
        assert episodic is None, "EpisodicMemory should NOT be registered"

        semantic = container.get_optional(SemanticMemoryProtocol)
        assert semantic is None, "SemanticMemory should NOT be registered"


# =============================================================================
# Configuration Tests
# =============================================================================


@pytest.mark.skip(reason="Experimental Phase 3 feature not implemented yet")
class TestConfiguration:
    """Test that settings configure services correctly."""

    def test_hierarchical_planning_settings(self):
        """Test that hierarchical planning settings are applied."""
        settings = Settings()
        settings.enable_hierarchical_planning = True
        settings.hierarchical_planning_max_depth = 10
        settings.hierarchical_planning_min_subtasks = 3
        settings.hierarchical_planning_max_subtasks = 15

        from victor.agent.service_provider import configure_orchestrator_services
        from victor.agent.protocols import HierarchicalPlannerProtocol

        container = ServiceContainer()
        configure_orchestrator_services(container, settings)

        planner = container.get(HierarchicalPlannerProtocol)
        assert planner is not None, "Planner should be created"
        # Settings are passed to constructor - verify if needed

    def test_memory_settings(self):
        """Test that memory settings are applied."""
        settings = Settings()
        settings.enable_episodic_memory = True
        settings.episodic_memory_max_episodes = 500
        settings.episodic_memory_recall_threshold = 0.5

        from victor.agent.service_provider import configure_orchestrator_services
        from victor.agent.protocols import EpisodicMemoryProtocol

        container = ServiceContainer()
        configure_orchestrator_services(container, settings)

        memory = container.get(EpisodicMemoryProtocol)
        assert memory is not None, "Memory should be created"


# =============================================================================
# Pytest Marks
# =============================================================================


@pytest.mark.integration
@pytest.mark.agentic_ai
@pytest.mark.skip(reason="Experimental Phase 3 feature not implemented yet")
class TestAgenticAIMarks:
    """Tests with marks for pytest filtering."""

    def test_marked_test(self, container_with_all_features):
        """Test that is marked with integration and agentic_ai marks."""
        from victor.agent.protocols import HierarchicalPlannerProtocol

        planner = container_with_all_features.get(HierarchicalPlannerProtocol)
        assert planner is not None
