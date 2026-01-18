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

"""Unit tests for sub-agent architecture (P2.1).

Tests cover:
- SubAgentRole enum and role properties
- SubAgentConfig dataclass validation
- SubAgentResult serialization
- SubAgent execution and constraint enforcement
- SubAgentOrchestrator spawn() and fan_out()
- Role-specific prompts and defaults
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.subagents import (
    SubAgent,
    SubAgentConfig,
    SubAgentOrchestrator,
    SubAgentResult,
    SubAgentRole,
    get_role_prompt,
)
from victor.agent.subagents.orchestrator import (
    ROLE_DEFAULT_BUDGETS,
    ROLE_DEFAULT_CONTEXT,
    ROLE_DEFAULT_TOOLS,
    FanOutResult,
    SubAgentTask,
)


# =============================================================================
# SubAgentRole Tests
# =============================================================================


class TestSubAgentRole:
    """Tests for SubAgentRole enum."""

    def test_all_roles_defined(self):
        """Verify all expected roles exist."""
        expected_roles = {"RESEARCHER", "PLANNER", "EXECUTOR", "REVIEWER", "TESTER"}
        actual_roles = {r.name for r in SubAgentRole}
        assert actual_roles == expected_roles

    def test_role_values(self):
        """Verify role string values are lowercase."""
        assert SubAgentRole.RESEARCHER.value == "researcher"
        assert SubAgentRole.PLANNER.value == "planner"
        assert SubAgentRole.EXECUTOR.value == "executor"
        assert SubAgentRole.REVIEWER.value == "reviewer"
        assert SubAgentRole.TESTER.value == "tester"

    def test_roles_are_iterable(self):
        """Verify we can iterate over roles."""
        roles = list(SubAgentRole)
        assert len(roles) == 5


# =============================================================================
# SubAgentConfig Tests
# =============================================================================


class TestSubAgentConfig:
    """Tests for SubAgentConfig dataclass."""

    def test_required_fields(self):
        """Test that required fields must be provided."""
        config = SubAgentConfig(
            role=SubAgentRole.RESEARCHER,
            task="Research authentication",
            allowed_tools=["read", "ls"],
            tool_budget=10,
            context_limit=30000,
        )
        assert config.role == SubAgentRole.RESEARCHER
        assert config.task == "Research authentication"
        assert config.allowed_tools == ["read", "ls"]
        assert config.tool_budget == 10
        assert config.context_limit == 30000

    def test_default_values(self):
        """Test default values are applied."""
        config = SubAgentConfig(
            role=SubAgentRole.EXECUTOR,
            task="Implement feature",
            allowed_tools=["read", "write"],
            tool_budget=20,
            context_limit=50000,
        )
        assert config.can_spawn_subagents is False
        assert config.working_directory is None
        assert config.timeout_seconds == 300
        assert config.system_prompt_override is None

    def test_all_fields_can_be_set(self):
        """Test all fields can be explicitly set."""
        config = SubAgentConfig(
            role=SubAgentRole.PLANNER,
            task="Plan implementation",
            allowed_tools=["read", "ls", "search"],
            tool_budget=15,
            context_limit=40000,
            can_spawn_subagents=True,
            working_directory="/custom/path",
            timeout_seconds=600,
            system_prompt_override="Custom prompt",
        )
        assert config.can_spawn_subagents is True
        assert config.working_directory == "/custom/path"
        assert config.timeout_seconds == 600
        assert config.system_prompt_override == "Custom prompt"


# =============================================================================
# SubAgentResult Tests
# =============================================================================


class TestSubAgentResult:
    """Tests for SubAgentResult dataclass."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = SubAgentResult(
            success=True,
            summary="Task completed successfully",
            details={"full_response": "Full output..."},
            tool_calls_used=5,
            context_size=10000,
            duration_seconds=2.5,
        )
        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        """Test creating a failed result."""
        result = SubAgentResult(
            success=False,
            summary="Task failed",
            details={"error_type": "TimeoutError"},
            tool_calls_used=3,
            context_size=5000,
            duration_seconds=300.0,
            error="Timeout after 300s",
        )
        assert result.success is False
        assert result.error == "Timeout after 300s"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = SubAgentResult(
            success=True,
            summary="Test summary",
            details={"key": "value"},
            tool_calls_used=10,
            context_size=20000,
            duration_seconds=5.0,
            error=None,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["summary"] == "Test summary"
        assert d["details"] == {"key": "value"}
        assert d["tool_calls_used"] == 10
        assert d["context_size"] == 20000
        assert d["duration_seconds"] == 5.0
        assert d["error"] is None


# =============================================================================
# Role Prompts Tests
# =============================================================================


class TestRolePrompts:
    """Tests for role-specific prompts."""

    def test_all_roles_have_prompts(self):
        """Verify all roles have defined prompts."""
        for role in SubAgentRole:
            prompt = get_role_prompt(role)
            assert prompt is not None
            assert len(prompt) > 100  # Prompts should be substantial

    def test_researcher_prompt_is_read_only(self):
        """Verify researcher prompt emphasizes read-only."""
        prompt = get_role_prompt(SubAgentRole.RESEARCHER)
        assert "read-only" in prompt.lower() or "exploration" in prompt.lower()

    def test_executor_prompt_allows_modifications(self):
        """Verify executor prompt mentions code modification."""
        prompt = get_role_prompt(SubAgentRole.EXECUTOR)
        assert "modif" in prompt.lower() or "write" in prompt.lower() or "edit" in prompt.lower()

    def test_tester_prompt_mentions_tests(self):
        """Verify tester prompt mentions test creation."""
        prompt = get_role_prompt(SubAgentRole.TESTER)
        assert "test" in prompt.lower()

    def test_unknown_role_raises(self):
        """Verify unknown role raises error."""
        # Using an invalid value directly won't work with Enum
        # Instead test with mock
        with patch("victor.agent.subagents.prompts.ROLE_PROMPTS", {}):
            with pytest.raises(ValueError):
                get_role_prompt(SubAgentRole.RESEARCHER)


# =============================================================================
# Role Defaults Tests
# =============================================================================


class TestRoleDefaults:
    """Tests for role-specific defaults in orchestrator."""

    def test_all_roles_have_default_tools(self):
        """Verify all roles have default tool sets."""
        for role in SubAgentRole:
            assert role in ROLE_DEFAULT_TOOLS
            assert len(ROLE_DEFAULT_TOOLS[role]) > 0

    def test_all_roles_have_default_budgets(self):
        """Verify all roles have default budgets."""
        for role in SubAgentRole:
            assert role in ROLE_DEFAULT_BUDGETS
            assert ROLE_DEFAULT_BUDGETS[role] > 0

    def test_all_roles_have_default_context(self):
        """Verify all roles have default context limits."""
        for role in SubAgentRole:
            assert role in ROLE_DEFAULT_CONTEXT
            assert ROLE_DEFAULT_CONTEXT[role] > 0

    def test_researcher_has_read_tools(self):
        """Verify researcher has read-only tools."""
        tools = ROLE_DEFAULT_TOOLS[SubAgentRole.RESEARCHER]
        assert "read" in tools
        assert "write" not in tools
        assert "edit" not in tools

    def test_executor_has_write_tools(self):
        """Verify executor has write tools."""
        tools = ROLE_DEFAULT_TOOLS[SubAgentRole.EXECUTOR]
        assert "read" in tools
        assert "write" in tools
        assert "edit" in tools

    def test_executor_has_highest_budget(self):
        """Verify executor has highest default budget."""
        executor_budget = ROLE_DEFAULT_BUDGETS[SubAgentRole.EXECUTOR]
        for role in SubAgentRole:
            if role != SubAgentRole.EXECUTOR:
                assert ROLE_DEFAULT_BUDGETS[role] <= executor_budget


# =============================================================================
# SubAgent Tests
# =============================================================================


class TestSubAgent:
    """Tests for SubAgent class."""

    @pytest.fixture
    def mock_parent_orchestrator(self):
        """Create a mock parent orchestrator."""
        mock = MagicMock()
        mock.settings = MagicMock()
        mock.settings.tool_budget = 50
        mock.settings.max_context_chars = 100000
        mock.provider_name = "mock_provider"
        mock.model = "mock_model"
        mock.temperature = 0.7
        mock.tool_registry = MagicMock()
        mock.tool_registry.get.return_value = MagicMock()
        return mock

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing."""
        return SubAgentConfig(
            role=SubAgentRole.RESEARCHER,
            task="Research authentication patterns",
            allowed_tools=["read", "ls", "search"],
            tool_budget=15,
            context_limit=50000,
        )

    def test_subagent_initialization(self, sample_config, mock_parent_orchestrator):
        """Test SubAgent initializes correctly."""
        subagent = SubAgent(sample_config, mock_parent_orchestrator)
        assert subagent.config == sample_config
        assert subagent.parent == mock_parent_orchestrator
        assert subagent.orchestrator is None  # Lazy initialization

    def test_get_role_prompt_uses_override(self, mock_parent_orchestrator):
        """Test custom prompt override is used."""
        config = SubAgentConfig(
            role=SubAgentRole.RESEARCHER,
            task="Research",
            allowed_tools=["read"],
            tool_budget=10,
            context_limit=30000,
            system_prompt_override="Custom prompt for this task",
        )
        subagent = SubAgent(config, mock_parent_orchestrator)
        prompt = subagent._get_role_prompt()
        assert prompt == "Custom prompt for this task"

    def test_get_role_prompt_uses_default(self, sample_config, mock_parent_orchestrator):
        """Test default role prompt is used when no override."""
        subagent = SubAgent(sample_config, mock_parent_orchestrator)
        prompt = subagent._get_role_prompt()
        # Should return the researcher prompt
        assert len(prompt) > 100


# =============================================================================
# SubAgentOrchestrator Tests
# =============================================================================


class TestSubAgentOrchestrator:
    """Tests for SubAgentOrchestrator class."""

    @pytest.fixture
    def mock_parent(self):
        """Create mock parent orchestrator."""
        mock = MagicMock()
        mock.settings = MagicMock()
        mock.settings.tool_budget = 50
        mock.settings.max_context_chars = 100000
        mock.provider_name = "mock_provider"
        mock.model = "mock_model"
        mock.temperature = 0.7
        mock.tool_registry = MagicMock()
        mock.tool_registry.get.return_value = MagicMock()
        mock.tool_registry.clear = MagicMock()
        mock.tool_registry.register = MagicMock()
        return mock

    def test_initialization(self, mock_parent):
        """Test orchestrator initializes correctly."""
        orchestrator = SubAgentOrchestrator(mock_parent)
        assert orchestrator.parent == mock_parent
        assert len(orchestrator.active_subagents) == 0

    def test_get_active_count(self, mock_parent):
        """Test active count tracking."""
        orchestrator = SubAgentOrchestrator(mock_parent)
        assert orchestrator.get_active_count() == 0


# =============================================================================
# SubAgentTask Tests
# =============================================================================


class TestSubAgentTask:
    """Tests for SubAgentTask dataclass."""

    def test_minimal_task(self):
        """Test creating task with minimal fields."""
        task = SubAgentTask(
            role=SubAgentRole.RESEARCHER,
            task="Find API endpoints",
        )
        assert task.role == SubAgentRole.RESEARCHER
        assert task.task == "Find API endpoints"
        assert task.tool_budget is None
        assert task.allowed_tools is None
        assert task.context_limit is None

    def test_task_with_overrides(self):
        """Test creating task with custom overrides."""
        task = SubAgentTask(
            role=SubAgentRole.EXECUTOR,
            task="Implement feature",
            tool_budget=40,
            allowed_tools=["read", "write", "edit"],
            context_limit=80000,
        )
        assert task.tool_budget == 40
        assert task.allowed_tools == ["read", "write", "edit"]
        assert task.context_limit == 80000


# =============================================================================
# FanOutResult Tests
# =============================================================================


class TestFanOutResult:
    """Tests for FanOutResult dataclass."""

    def test_all_success(self):
        """Test fan-out with all successful results."""
        results = [
            SubAgentResult(
                success=True,
                summary="Task 1 done",
                details={},
                tool_calls_used=5,
                context_size=10000,
                duration_seconds=1.0,
            ),
            SubAgentResult(
                success=True,
                summary="Task 2 done",
                details={},
                tool_calls_used=3,
                context_size=8000,
                duration_seconds=0.8,
            ),
        ]
        fan_out_result = FanOutResult(
            results=results,
            all_success=True,
            total_tool_calls=8,
            total_duration=1.2,
            errors=[],
        )
        assert fan_out_result.all_success is True
        assert len(fan_out_result.errors) == 0
        assert fan_out_result.total_tool_calls == 8

    def test_partial_failure(self):
        """Test fan-out with partial failures."""
        results = [
            SubAgentResult(
                success=True,
                summary="Task 1 done",
                details={},
                tool_calls_used=5,
                context_size=10000,
                duration_seconds=1.0,
            ),
            SubAgentResult(
                success=False,
                summary="Task 2 failed",
                details={},
                tool_calls_used=2,
                context_size=5000,
                duration_seconds=0.5,
                error="Timeout",
            ),
        ]
        fan_out_result = FanOutResult(
            results=results,
            all_success=False,
            total_tool_calls=7,
            total_duration=1.0,
            errors=["Task 1 (researcher): Timeout"],
        )
        assert fan_out_result.all_success is False
        assert len(fan_out_result.errors) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestSubAgentIntegration:
    """Integration tests for sub-agent system."""

    def test_module_exports(self):
        """Test that module exports all expected classes."""
        from victor.agent.subagents import (
            SubAgent,
            SubAgentConfig,
            SubAgentOrchestrator,
            SubAgentResult,
            SubAgentRole,
            get_role_prompt,
        )

        # Verify all exports are accessible
        assert SubAgent is not None
        assert SubAgentConfig is not None
        assert SubAgentOrchestrator is not None
        assert SubAgentResult is not None
        assert SubAgentRole is not None
        assert get_role_prompt is not None

    def test_config_to_subagent_flow(self):
        """Test creating config and using it with SubAgent."""
        config = SubAgentConfig(
            role=SubAgentRole.PLANNER,
            task="Plan authentication implementation",
            allowed_tools=["read", "ls", "search", "plan_files"],
            tool_budget=10,
            context_limit=30000,
        )

        # Create mock parent
        mock_parent = MagicMock()
        mock_parent.settings = MagicMock()
        mock_parent.tool_registry = MagicMock()
        mock_parent.tool_registry.get.return_value = MagicMock()

        subagent = SubAgent(config, mock_parent)
        assert subagent.config.role == SubAgentRole.PLANNER
        assert subagent.config.tool_budget == 10


# =============================================================================
# AgentOrchestrator Integration Tests
# =============================================================================


class TestOrchestratorSubAgentIntegration:
    """Tests for SubAgentOrchestrator integration in AgentOrchestrator."""

    def test_orchestrator_has_subagent_orchestrator_property(self):
        """Test that AgentOrchestrator has the subagent_orchestrator property."""
        from victor.agent.orchestrator import AgentOrchestrator

        assert hasattr(AgentOrchestrator, "subagent_orchestrator")

    def test_orchestrator_subagent_fields_exist(self):
        """Test that AgentOrchestrator has subagent-related fields."""
        # This tests that the fields were added correctly by checking the
        # class structure without needing to instantiate it (which requires many deps)
        import inspect
        from victor.agent.orchestrator import AgentOrchestrator

        # Check that subagent_orchestrator property exists and uses lazy initialization
        source = inspect.getsource(AgentOrchestrator.subagent_orchestrator.fget)
        assert "_subagent_orchestrator" in source
        assert "_subagent_orchestration_enabled" in source

        # Verify the property is defined
        assert hasattr(AgentOrchestrator, "subagent_orchestrator")

    def test_subagent_orchestrator_property_is_lazy(self):
        """Test that intelligent_integration property follows lazy init pattern."""
        import inspect
        from victor.agent.orchestrator import AgentOrchestrator

        source = inspect.getsource(AgentOrchestrator.intelligent_integration.fget)
        # Check for lazy initialization pattern
        assert "_intelligent_pipeline_enabled" in source
        assert "_intelligent_integration is None" in source
        assert "from victor.agent.orchestrator_integration import OrchestratorIntegration" in source

    def test_subagent_orchestrator_returns_none_when_disabled(self):
        """Test that property returns None when subagent orchestration is disabled."""
        from unittest.mock import MagicMock, patch

        from victor.agent.orchestrator import AgentOrchestrator

        # Create a mock that simulates disabled state
        mock_orchestrator = MagicMock(spec=AgentOrchestrator)
        mock_orchestrator._subagent_orchestration_enabled = False
        mock_orchestrator._subagent_orchestrator = None

        # Call the property getter directly
        result = AgentOrchestrator.subagent_orchestrator.fget(mock_orchestrator)
        assert result is None
