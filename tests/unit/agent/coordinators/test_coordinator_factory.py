"""Tests for CoordinatorFactory compatibility shims."""

from unittest.mock import MagicMock

import pytest

from victor.agent.coordinators.coordinator_factory import CoordinatorFactory
from victor.agent.coordinators.tool_coordinator import ToolCoordinator
from victor.agent.protocols import (
    IBudgetManager,
    IToolAccessController,
    ModeControllerProtocol,
    ToolCacheProtocol,
    ToolPipelineProtocol,
    ToolRegistryProtocol,
    ToolSelectorProtocol,
)
from victor.agent.services.protocols import ToolServiceProtocol


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
