"""Tests for CoordinatorFactory compatibility shims."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from victor.agent.coordinators.coordinator_factory import CoordinatorFactory
from victor.agent.coordinators.exploration_state_passed import (
    ExplorationStatePassedCoordinator,
)
from victor.agent.coordinators.safety_state_passed import SafetyStatePassedCoordinator
from victor.agent.coordinators.system_prompt_state_passed import (
    SystemPromptStatePassedCoordinator,
)
from victor.agent.coordinators.tool_coordinator import ToolCoordinator
from victor.agent.protocols import (
    IBudgetManager,
    IToolAccessController,
    ModeControllerProtocol,
    TaskAnalyzerProtocol,
    ToolCacheProtocol,
    ToolPipelineProtocol,
    ToolRegistryProtocol,
    ToolSelectorProtocol,
)
from victor.agent.services.protocols import ToolServiceProtocol
from victor.agent.services.exploration_runtime import ExplorationCoordinator


class TestCoordinatorFactoryToolShim:
    """CoordinatorFactory should build the current compatibility shim shape."""

    def test_create_tool_coordinator_builds_service_bound_shim(self):
        container = MagicMock()
        tool_pipeline = MagicMock()
        tool_registry = MagicMock()
        tool_selector = MagicMock()
        budget_manager = MagicMock()
        tool_cache = MagicMock()
        mode_controller = MagicMock()
        tool_access_controller = MagicMock()
        tool_service = MagicMock()

        mapping = {
            ToolPipelineProtocol: tool_pipeline,
            ToolRegistryProtocol: tool_registry,
            ToolSelectorProtocol: tool_selector,
            IBudgetManager: budget_manager,
            ToolCacheProtocol: tool_cache,
            ModeControllerProtocol: mode_controller,
            IToolAccessController: tool_access_controller,
            ToolServiceProtocol: tool_service,
        }
        container.get_optional.side_effect = lambda cls: mapping.get(cls)

        factory = CoordinatorFactory(container)

        with pytest.warns(
            DeprecationWarning,
            match="deprecated ToolCoordinator shim",
        ):
            coordinator = factory.create_tool_coordinator()

        assert isinstance(coordinator, ToolCoordinator)
        assert coordinator._mode_controller is mode_controller
        assert coordinator._tool_service is tool_service
        assert coordinator._cache is tool_cache
        assert coordinator._tool_access_controller is tool_access_controller

    def test_create_tool_coordinator_requires_pipeline_and_registry(self):
        container = MagicMock()
        container.get_optional.return_value = None
        factory = CoordinatorFactory(container)

        with (
            pytest.warns(
                DeprecationWarning,
                match="deprecated ToolCoordinator shim",
            ),
            pytest.raises(RuntimeError, match="ToolPipeline and ToolRegistry are required"),
        ):
            factory.create_tool_coordinator()


class TestCoordinatorFactoryCanonicalSurfaces:
    """CoordinatorFactory should expose canonical runtime/state-passed surfaces."""

    def test_create_exploration_coordinator(self):
        container = MagicMock()
        factory = CoordinatorFactory(container)

        coordinator = factory.create_exploration_coordinator()

        assert isinstance(coordinator, ExplorationCoordinator)

    def test_create_exploration_state_passed_coordinator_uses_project_root(self):
        container = MagicMock()
        container._settings = MagicMock(working_directory="/tmp/project-root")
        factory = CoordinatorFactory(container)

        coordinator = factory.create_exploration_state_passed_coordinator()

        assert isinstance(coordinator, ExplorationStatePassedCoordinator)
        assert coordinator._project_root == Path("/tmp/project-root")

    def test_create_system_prompt_coordinator_binds_task_analyzer(self):
        container = MagicMock()
        analyzer = MagicMock()
        container.get_optional.side_effect = lambda cls: (
            analyzer if cls is TaskAnalyzerProtocol else None
        )
        factory = CoordinatorFactory(container)

        coordinator = factory.create_system_prompt_coordinator()

        assert coordinator._task_analyzer is analyzer

    def test_create_system_prompt_state_passed_coordinator_binds_task_analyzer(self):
        container = MagicMock()
        analyzer = MagicMock()
        container.get_optional.side_effect = lambda cls: (
            analyzer if cls is TaskAnalyzerProtocol else None
        )
        factory = CoordinatorFactory(container)

        coordinator = factory.create_system_prompt_state_passed_coordinator()

        assert isinstance(coordinator, SystemPromptStatePassedCoordinator)
        assert coordinator._task_analyzer is analyzer

    def test_create_safety_state_passed_coordinator(self):
        container = MagicMock()
        factory = CoordinatorFactory(container)

        coordinator = factory.create_safety_state_passed_coordinator()

        assert isinstance(coordinator, SafetyStatePassedCoordinator)
