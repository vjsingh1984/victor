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

"""Tests for WorkflowCoordinator.

This test file demonstrates comprehensive testing of the WorkflowCoordinator,
which handles workflow registration, discovery, and execution coordination.

Test Coverage Strategy:
1. Test all public methods of WorkflowCoordinator
2. Test integration with WorkflowRegistry
3. Test workflow discovery and registration
4. Test edge cases and error conditions
5. Test interaction with workflow discovery system
"""

import pytest
from unittest.mock import Mock, patch

from victor.agent.coordinators.workflow_coordinator import WorkflowCoordinator
from victor.workflows.definition import WorkflowDefinition, AgentNode
from victor.workflows.registry import WorkflowRegistry


def make_workflow(name: str, description: str = "Test workflow") -> WorkflowDefinition:
    """Create a minimal valid WorkflowDefinition for tests."""
    node = AgentNode(id="start", name="start", goal="test")
    return WorkflowDefinition(
        name=name,
        description=description,
        nodes={"start": node},
        start_node="start",
    )


class TestWorkflowCoordinator:
    """Test suite for WorkflowCoordinator.

    This coordinator handles workflow registration, discovery, and management,
    extracted from AgentOrchestrator as part of SOLID refactoring.
    """

    @pytest.fixture
    def mock_workflow_registry(self) -> Mock:
        """Create mock workflow registry."""
        registry = Mock(spec=WorkflowRegistry)
        registry._workflows = {}
        registry.list_workflows.side_effect = lambda: list(registry._workflows.keys())
        registry.get.side_effect = lambda name: registry._workflows.get(name)
        return registry

    @pytest.fixture
    def real_workflow_registry(self) -> WorkflowRegistry:
        """Create real workflow registry for integration tests."""
        return WorkflowRegistry()

    @pytest.fixture
    def coordinator(self, mock_workflow_registry: Mock) -> WorkflowCoordinator:
        """Create workflow coordinator with mock registry."""
        return WorkflowCoordinator(
            workflow_registry=mock_workflow_registry,
        )

    @pytest.fixture
    def coordinator_with_real_registry(
        self, real_workflow_registry: WorkflowRegistry
    ) -> WorkflowCoordinator:
        """Create workflow coordinator with real registry."""
        return WorkflowCoordinator(
            workflow_registry=real_workflow_registry,
        )

    # Test initialization

    def test_initialization_with_workflow_registry(self, mock_workflow_registry: Mock):
        """Test that coordinator initializes with workflow registry."""
        # Execute
        coordinator = WorkflowCoordinator(
            workflow_registry=mock_workflow_registry,
        )

        # Assert
        assert coordinator.workflow_registry == mock_workflow_registry

    def test_workflow_registry_property(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test that workflow_registry property returns the registry."""
        # Execute
        registry = coordinator.workflow_registry

        # Assert
        assert registry == mock_workflow_registry
        assert registry is mock_workflow_registry

    # Test register_default_workflows

    @patch("victor.workflows.mode_workflows.get_mode_workflow_provider")
    def test_register_default_workflows_calls_provider(
        self,
        mock_get_provider: Mock,
        coordinator: WorkflowCoordinator,
    ):
        """Test that register_default_workflows loads mode workflows."""
        provider = Mock()
        provider.get_workflow_definitions.return_value = {
            "explore": make_workflow("explore"),
            "plan": make_workflow("plan"),
        }
        mock_get_provider.return_value = provider

        # Execute
        count = coordinator.register_default_workflows()

        # Assert
        mock_get_provider.assert_called_once()
        provider.get_workflow_definitions.assert_called_once()
        assert count == 2

    @patch("victor.workflows.mode_workflows.get_mode_workflow_provider")
    def test_register_default_workflows_returns_count(
        self,
        mock_get_provider: Mock,
        coordinator: WorkflowCoordinator,
    ):
        """Test that register_default_workflows returns correct count."""
        provider = Mock()
        provider.get_workflow_definitions.return_value = {
            "explore": make_workflow("explore"),
        }
        mock_get_provider.return_value = provider

        # Execute
        count = coordinator.register_default_workflows()

        # Assert
        assert count == 1

    @patch("victor.workflows.mode_workflows.get_mode_workflow_provider")
    def test_register_default_workflows_with_zero_workflows(
        self,
        mock_get_provider: Mock,
        coordinator: WorkflowCoordinator,
    ):
        """Test register_default_workflows when no workflows found."""
        provider = Mock()
        provider.get_workflow_definitions.return_value = {}
        mock_get_provider.return_value = provider

        # Execute
        count = coordinator.register_default_workflows()

        # Assert
        assert count == 0
        mock_get_provider.assert_called_once()
        provider.get_workflow_definitions.assert_called_once()

    @patch("victor.workflows.mode_workflows.get_mode_workflow_provider")
    def test_register_default_workflows_with_provider_error(
        self,
        mock_get_provider: Mock,
        coordinator: WorkflowCoordinator,
    ):
        """Test register_default_workflows when provider raises error."""
        provider = Mock()
        provider.get_workflow_definitions.side_effect = RuntimeError("Provider failed")
        mock_get_provider.return_value = provider

        # Execute & Assert
        with pytest.raises(RuntimeError, match="Provider failed"):
            coordinator.register_default_workflows()

    def test_register_default_workflows_with_real_registry(
        self,
        coordinator_with_real_registry: WorkflowCoordinator,
        real_workflow_registry: WorkflowRegistry,
    ):
        """Test register_default_workflows with real registry."""
        # Execute
        count = coordinator_with_real_registry.register_default_workflows()

        # Assert - should register some workflows
        assert count >= 0
        assert isinstance(count, int)

    # Test list_workflows

    def test_list_workflows_returns_empty_list_when_no_workflows(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test that list_workflows returns empty list when no workflows registered."""
        # Setup
        mock_workflow_registry._workflows = {}

        # Execute
        workflows = coordinator.list_workflows()

        # Assert
        assert workflows == []
        assert isinstance(workflows, list)

    def test_list_workflows_returns_workflow_names(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test that list_workflows returns list of workflow names."""
        # Setup
        mock_workflow_registry._workflows = {
            "workflow1": Mock(),
            "workflow2": Mock(),
            "workflow3": Mock(),
        }

        # Execute
        workflows = coordinator.list_workflows()

        # Assert
        assert len(workflows) == 3
        assert "workflow1" in workflows
        assert "workflow2" in workflows
        assert "workflow3" in workflows
        assert isinstance(workflows, list)

    def test_list_workflows_with_single_workflow(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test list_workflows with single registered workflow."""
        # Setup
        mock_workflow_registry._workflows = {
            "single_workflow": Mock(),
        }

        # Execute
        workflows = coordinator.list_workflows()

        # Assert
        assert len(workflows) == 1
        assert workflows[0] == "single_workflow"

    def test_list_workflows_with_many_workflows(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test list_workflows with many registered workflows."""
        # Setup
        workflow_names = [f"workflow_{i}" for i in range(100)]
        mock_workflow_registry._workflows = {name: Mock() for name in workflow_names}

        # Execute
        workflows = coordinator.list_workflows()

        # Assert
        assert len(workflows) == 100
        assert set(workflows) == set(workflow_names)

    def test_list_workflows_returns_new_list_each_time(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test that list_workflows returns a new list each time."""
        # Setup
        mock_workflow_registry._workflows = {"workflow1": Mock()}

        # Execute
        workflows1 = coordinator.list_workflows()
        workflows2 = coordinator.list_workflows()

        # Assert - should be equal but different objects
        assert workflows1 == workflows2
        assert workflows1 is not workflows2

    # Test has_workflow

    def test_has_workflow_returns_true_when_workflow_exists(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test that has_workflow returns True for existing workflow."""
        # Setup
        mock_workflow_registry._workflows = {
            "existing_workflow": Mock(),
        }

        # Execute
        result = coordinator.has_workflow("existing_workflow")

        # Assert
        assert result is True

    def test_has_workflow_returns_false_when_workflow_not_exists(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test that has_workflow returns False for non-existing workflow."""
        # Setup
        mock_workflow_registry._workflows = {}

        # Execute
        result = coordinator.has_workflow("non_existing_workflow")

        # Assert
        assert result is False

    def test_has_workflow_with_empty_registry(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test has_workflow with empty workflow registry."""
        # Setup
        mock_workflow_registry._workflows = {}

        # Execute
        result = coordinator.has_workflow("any_workflow")

        # Assert
        assert result is False

    def test_has_workflow_with_multiple_workflows(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test has_workflow with multiple workflows registered."""
        # Setup
        mock_workflow_registry._workflows = {
            "workflow1": Mock(),
            "workflow2": Mock(),
            "workflow3": Mock(),
        }

        # Execute & Assert
        assert coordinator.has_workflow("workflow1") is True
        assert coordinator.has_workflow("workflow2") is True
        assert coordinator.has_workflow("workflow3") is True
        assert coordinator.has_workflow("workflow4") is False

    def test_has_workflow_case_sensitive(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test that has_workflow is case-sensitive."""
        # Setup
        mock_workflow_registry._workflows = {
            "MyWorkflow": Mock(),
        }

        # Execute & Assert
        assert coordinator.has_workflow("MyWorkflow") is True
        assert coordinator.has_workflow("myworkflow") is False
        assert coordinator.has_workflow("MYWORKFLOW") is False

    # Test get_workflow_count

    def test_get_workflow_count_returns_zero_when_empty(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test that get_workflow_count returns 0 for empty registry."""
        # Setup
        mock_workflow_registry._workflows = {}

        # Execute
        count = coordinator.get_workflow_count()

        # Assert
        assert count == 0

    def test_get_workflow_count_returns_correct_count(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test that get_workflow_count returns correct number of workflows."""
        # Setup
        mock_workflow_registry._workflows = {
            "workflow1": Mock(),
            "workflow2": Mock(),
            "workflow3": Mock(),
        }

        # Execute
        count = coordinator.get_workflow_count()

        # Assert
        assert count == 3

    def test_get_workflow_count_with_single_workflow(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test get_workflow_count with single workflow."""
        # Setup
        mock_workflow_registry._workflows = {
            "single": Mock(),
        }

        # Execute
        count = coordinator.get_workflow_count()

        # Assert
        assert count == 1

    def test_get_workflow_count_with_many_workflows(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test get_workflow_count with many workflows."""
        # Setup
        mock_workflow_registry._workflows = {f"workflow_{i}": Mock() for i in range(50)}

        # Execute
        count = coordinator.get_workflow_count()

        # Assert
        assert count == 50

    def test_get_workflow_count_changes_after_registration(
        self,
        coordinator_with_real_registry: WorkflowCoordinator,
        real_workflow_registry: WorkflowRegistry,
    ):
        """Test that get_workflow_count reflects workflow registration."""
        # Initial count
        count1 = coordinator_with_real_registry.get_workflow_count()
        assert count1 == 0

        # Register a workflow
        workflow = make_workflow("test_workflow")
        real_workflow_registry.register(workflow)

        # Count should increase
        count2 = coordinator_with_real_registry.get_workflow_count()
        assert count2 == 1

        # Register another workflow
        workflow2 = make_workflow("test_workflow_2")
        real_workflow_registry.register(workflow2)

        # Count should increase again
        count3 = coordinator_with_real_registry.get_workflow_count()
        assert count3 == 2


class TestWorkflowCoordinatorIntegration:
    """Integration tests for WorkflowCoordinator with real registry."""

    @pytest.fixture
    def workflow_registry(self) -> WorkflowRegistry:
        """Create real workflow registry."""
        return WorkflowRegistry()

    @pytest.fixture
    def coordinator(self, workflow_registry: WorkflowRegistry) -> WorkflowCoordinator:
        """Create workflow coordinator with real registry."""
        return WorkflowCoordinator(
            workflow_registry=workflow_registry,
        )

    def test_full_workflow_registration_lifecycle(
        self, coordinator: WorkflowCoordinator, workflow_registry: WorkflowRegistry
    ):
        """Test complete workflow registration lifecycle."""
        # Initial state
        assert coordinator.get_workflow_count() == 0
        assert coordinator.list_workflows() == []
        assert coordinator.has_workflow("test_workflow") is False

        # Register workflow
        workflow = make_workflow("test_workflow", "A test workflow")
        workflow_registry.register(workflow)

        # Verify registration
        assert coordinator.get_workflow_count() == 1
        assert "test_workflow" in coordinator.list_workflows()
        assert coordinator.has_workflow("test_workflow") is True

    def test_multiple_workflow_registration_and_listing(
        self, coordinator: WorkflowCoordinator, workflow_registry: WorkflowRegistry
    ):
        """Test registering and listing multiple workflows."""
        # Register multiple workflows
        workflows = [make_workflow(f"workflow_{i}", f"Description {i}") for i in range(5)]

        for workflow in workflows:
            workflow_registry.register(workflow)

        # Verify all are registered
        assert coordinator.get_workflow_count() == 5

        workflow_names = coordinator.list_workflows()
        for i in range(5):
            assert f"workflow_{i}" in workflow_names
            assert coordinator.has_workflow(f"workflow_{i}") is True

    def test_workflow_operations_with_real_registry(
        self, coordinator: WorkflowCoordinator, workflow_registry: WorkflowRegistry
    ):
        """Test various workflow operations with real registry."""
        # Register workflows
        workflow1 = make_workflow("workflow1", "First workflow")
        workflow2 = make_workflow("workflow2", "Second workflow")
        workflow_registry.register(workflow1)
        workflow_registry.register(workflow2)

        # Test has_workflow
        assert coordinator.has_workflow("workflow1") is True
        assert coordinator.has_workflow("workflow2") is True
        assert coordinator.has_workflow("workflow3") is False

        # Test get_workflow_count
        assert coordinator.get_workflow_count() == 2

        # Test list_workflows
        workflows = coordinator.list_workflows()
        assert len(workflows) == 2
        assert "workflow1" in workflows
        assert "workflow2" in workflows


class TestWorkflowCoordinatorEdgeCases:
    """Test edge cases and error conditions for WorkflowCoordinator."""

    @pytest.fixture
    def mock_workflow_registry(self) -> Mock:
        """Create mock workflow registry."""
        registry = Mock(spec=WorkflowRegistry)
        registry._workflows = {}
        registry.list_workflows.side_effect = lambda: list(registry._workflows.keys())
        registry.get.side_effect = lambda name: registry._workflows.get(name)
        return registry

    @pytest.fixture
    def coordinator(self, mock_workflow_registry: Mock) -> WorkflowCoordinator:
        """Create workflow coordinator."""
        return WorkflowCoordinator(
            workflow_registry=mock_workflow_registry,
        )

    def test_workflow_registry_with_special_characters_in_names(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test workflows with special characters in names."""
        # Setup
        special_names = [
            "workflow-with-dashes",
            "workflow_with_underscores",
            "workflow.with.dots",
            "workflow:with:colons",
            "workflow/with/slashes",
        ]
        mock_workflow_registry._workflows = {name: Mock() for name in special_names}

        # Execute & Assert
        for name in special_names:
            assert coordinator.has_workflow(name) is True

        workflows = coordinator.list_workflows()
        for name in special_names:
            assert name in workflows

        assert coordinator.get_workflow_count() == len(special_names)

    def test_workflow_names_with_unicode(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test workflow names with unicode characters."""
        # Setup
        unicode_names = [
            "workflow_世界",
            "workflow_🔧",
            "workflow_مرحبا",
            "workflow_🚀",
        ]
        mock_workflow_registry._workflows = {name: Mock() for name in unicode_names}

        # Execute & Assert
        for name in unicode_names:
            assert coordinator.has_workflow(name) is True

        workflows = coordinator.list_workflows()
        for name in unicode_names:
            assert name in workflows

    def test_list_workflows_returns_copy_not_reference(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test that list_workflows returns a copy, not reference to dict keys."""
        # Setup
        mock_workflow_registry._workflows = {"workflow1": Mock()}

        # Execute
        workflows1 = coordinator.list_workflows()

        # Modify returned list
        workflows1.append("workflow2")

        # Get list again
        workflows2 = coordinator.list_workflows()

        # Assert - modification should not affect registry
        assert len(workflows2) == 1
        assert "workflow1" in workflows2
        assert "workflow2" not in workflows2

    @patch("victor.workflows.mode_workflows.get_mode_workflow_provider")
    def test_register_default_workflows_multiple_calls(
        self,
        mock_get_provider: Mock,
        coordinator: WorkflowCoordinator,
    ):
        """Test calling register_default_workflows multiple times."""
        provider = Mock()
        provider.get_workflow_definitions.return_value = {
            "explore": make_workflow("explore"),
        }
        mock_get_provider.return_value = provider

        # Execute
        count1 = coordinator.register_default_workflows()
        count2 = coordinator.register_default_workflows()
        count3 = coordinator.register_default_workflows()

        # Assert - should call provider each time
        assert mock_get_provider.call_count == 3
        assert count1 == 1
        assert count2 == 1
        assert count3 == 1

    def test_has_workflow_with_empty_string(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test has_workflow with empty string."""
        # Setup
        mock_workflow_registry._workflows = {}

        # Execute
        result = coordinator.has_workflow("")

        # Assert
        assert result is False

    def test_list_workflows_with_empty_registry_keys(
        self, coordinator: WorkflowCoordinator, mock_workflow_registry: Mock
    ):
        """Test list_workflows when registry has empty keys dict."""
        # Setup
        mock_workflow_registry._workflows = {}

        # Execute
        workflows = coordinator.list_workflows()

        # Assert
        assert workflows == []
        assert isinstance(workflows, list)

    @patch("victor.workflows.mode_workflows.get_mode_workflow_provider")
    def test_register_default_workflows_with_exception_during_registration(
        self,
        mock_get_provider: Mock,
        coordinator: WorkflowCoordinator,
    ):
        """Test register_default_workflows when registration raises exception."""
        provider = Mock()
        provider.get_workflow_definitions.side_effect = ValueError("Registration error")
        mock_get_provider.return_value = provider

        # Execute & Assert
        with pytest.raises(ValueError, match="Registration error"):
            coordinator.register_default_workflows()

    def test_workflow_coordinator_with_mocked_registry_property_access(
        self,
    ):
        """Test WorkflowCoordinator with registry methods mocked."""
        registry = Mock(spec=WorkflowRegistry)
        registry.list_workflows.return_value = ["workflow1"]
        registry.get.side_effect = lambda name: object() if name == "workflow1" else None

        # Create coordinator
        coordinator = WorkflowCoordinator(workflow_registry=registry)

        # Test operations
        assert coordinator.has_workflow("workflow1") is True
        assert coordinator.has_workflow("workflow2") is False
        assert coordinator.get_workflow_count() == 1
        assert "workflow1" in coordinator.list_workflows()


class TestWorkflowCoordinatorWithRealWorkflows:
    """Test WorkflowCoordinator with real WorkflowDefinition implementations."""

    @pytest.fixture
    def workflow_registry(self) -> WorkflowRegistry:
        """Create real workflow registry."""
        return WorkflowRegistry()

    @pytest.fixture
    def coordinator(self, workflow_registry: WorkflowRegistry) -> WorkflowCoordinator:
        """Create workflow coordinator."""
        return WorkflowCoordinator(
            workflow_registry=workflow_registry,
        )

    def test_coordinator_with_real_workflow_registration(
        self, coordinator: WorkflowCoordinator, workflow_registry: WorkflowRegistry
    ):
        """Test coordinator with real workflow objects."""
        # Create and register real workflows
        workflow1 = make_workflow("workflow1", "First workflow")
        workflow2 = make_workflow("workflow2", "Second workflow")

        workflow_registry.register(workflow1)
        workflow_registry.register(workflow2)

        # Test all coordinator methods
        assert coordinator.get_workflow_count() == 2
        assert coordinator.has_workflow("workflow1") is True
        assert coordinator.has_workflow("workflow2") is True
        assert set(coordinator.list_workflows()) == {"workflow1", "workflow2"}

    def test_coordinator_workflow_access_via_registry(
        self, coordinator: WorkflowCoordinator, workflow_registry: WorkflowRegistry
    ):
        """Test that workflows can be accessed through coordinator's registry."""
        # Register workflow
        workflow = make_workflow("test", "Test workflow")
        workflow_registry.register(workflow)

        # Access through coordinator
        retrieved_workflow = coordinator.workflow_registry.get("test")

        # Assert
        assert retrieved_workflow is not None
        assert retrieved_workflow.name == "test"
        assert retrieved_workflow.description == "Test workflow"

    def test_duplicate_workflow_registration_prevented(
        self, coordinator: WorkflowCoordinator, workflow_registry: WorkflowRegistry
    ):
        """Test that duplicate workflow registration raises error."""
        # Register first workflow
        workflow1 = make_workflow("duplicate", "First")
        workflow_registry.register(workflow1)

        # Try to register duplicate
        workflow2 = make_workflow("duplicate", "Second")

        # Should raise ValueError
        with pytest.raises(ValueError, match="already registered"):
            workflow_registry.register(workflow2)

        # Coordinator should still only see one
        assert coordinator.get_workflow_count() == 1
        assert coordinator.has_workflow("duplicate") is True

    def test_coordinator_after_workflow_removal_simulation(
        self, coordinator: WorkflowCoordinator, workflow_registry: WorkflowRegistry
    ):
        """Test coordinator when workflows are removed from registry."""
        # Register workflows
        workflow1 = make_workflow("workflow1", "First")
        workflow2 = make_workflow("workflow2", "Second")
        workflow_registry.register(workflow1)
        workflow_registry.register(workflow2)

        # Verify initial state
        assert coordinator.get_workflow_count() == 2

        # Simulate removal by directly modifying registry
        # (In real scenario, registry would have a remove method)
        del workflow_registry._definitions["workflow1"]
        workflow_registry._metadata.pop("workflow1", None)

        # Coordinator should reflect the change
        assert coordinator.get_workflow_count() == 1
        assert coordinator.has_workflow("workflow1") is False
        assert coordinator.has_workflow("workflow2") is True
        assert coordinator.list_workflows() == ["workflow2"]
