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

"""Tests for WorkflowFacade domain facade."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.facades.workflow_facade import WorkflowFacade
from victor.agent.facades.protocols import WorkflowFacadeProtocol


class TestWorkflowFacadeInit:
    """Tests for WorkflowFacade initialization."""

    def test_init_with_all_components(self):
        """WorkflowFacade initializes with all components provided."""
        registry = MagicMock()
        runtime = MagicMock()

        facade = WorkflowFacade(
            workflow_registry=registry,
            workflow_runtime=runtime,
            workflow_optimization=MagicMock(),
            coordination_advisor=MagicMock(),
        )

        assert facade.workflow_registry is registry
        assert facade.workflow_runtime is runtime

    def test_init_with_minimal_components(self):
        """WorkflowFacade initializes with no required components (all optional)."""
        facade = WorkflowFacade()

        assert facade.workflow_registry is None
        assert facade.workflow_runtime is None
        assert facade.workflow_optimization is None
        assert facade.coordination_advisor is None
        with pytest.warns(
            DeprecationWarning,
            match="WorkflowFacade.mode_workflow_team_coordinator is deprecated",
        ):
            assert facade.mode_workflow_team_coordinator is None

    def test_init_with_legacy_mode_coordinator_warns_and_maps_to_coordination_advisor(self):
        """Legacy init alias should warn and map onto coordination_advisor."""
        legacy_coordinator = MagicMock(name="legacy_coordinator")

        with pytest.warns(
            DeprecationWarning,
            match="WorkflowFacade\\(mode_workflow_team_coordinator=\\.\\.\\.\\) is deprecated",
        ):
            facade = WorkflowFacade(mode_workflow_team_coordinator=legacy_coordinator)

        assert facade.coordination_advisor is legacy_coordinator


class TestWorkflowFacadeProperties:
    """Tests for WorkflowFacade property access."""

    @pytest.fixture
    def facade(self):
        """Create a WorkflowFacade with mock components."""
        return WorkflowFacade(
            workflow_registry=MagicMock(name="registry"),
            workflow_runtime=MagicMock(name="runtime"),
            workflow_optimization=MagicMock(name="optimization"),
            coordination_advisor=MagicMock(name="coordinator"),
        )

    def test_workflow_registry_property(self, facade):
        """WorkflowRegistry property returns the registry."""
        assert facade.workflow_registry._mock_name == "registry"

    def test_workflow_registry_setter(self, facade):
        """WorkflowRegistry setter updates the registry."""
        new_registry = MagicMock(name="new_registry")
        facade.workflow_registry = new_registry
        assert facade.workflow_registry is new_registry

    def test_workflow_runtime_property(self, facade):
        """WorkflowRuntime property returns the runtime."""
        assert facade.workflow_runtime._mock_name == "runtime"

    def test_workflow_optimization_property(self, facade):
        """WorkflowOptimization property returns the optimization components."""
        assert facade.workflow_optimization._mock_name == "optimization"

    def test_mode_coordinator_property(self, facade):
        """Compatibility alias returns the same advisor instance."""
        with pytest.warns(
            DeprecationWarning,
            match="WorkflowFacade.mode_workflow_team_coordinator is deprecated",
        ):
            assert facade.mode_workflow_team_coordinator._mock_name == "coordinator"

    def test_mode_coordinator_setter(self, facade):
        """Compatibility alias setter updates the advisor surface."""
        new_coordinator = MagicMock(name="new_coordinator")
        with pytest.warns(
            DeprecationWarning,
            match="WorkflowFacade.mode_workflow_team_coordinator is deprecated",
        ):
            facade.mode_workflow_team_coordinator = new_coordinator
        with pytest.warns(
            DeprecationWarning,
            match="WorkflowFacade.mode_workflow_team_coordinator is deprecated",
        ):
            assert facade.mode_workflow_team_coordinator is new_coordinator
        assert facade.coordination_advisor is new_coordinator

    def test_coordination_advisor_property(self, facade):
        """Framework-facing coordination advisor property returns the advisor."""
        assert facade.coordination_advisor._mock_name == "coordinator"

    def test_coordination_advisor_setter(self, facade):
        """Framework-facing coordination advisor setter updates the advisor."""
        new_advisor = MagicMock(name="new_advisor")
        facade.coordination_advisor = new_advisor
        assert facade.coordination_advisor is new_advisor
        with pytest.warns(
            DeprecationWarning,
            match="WorkflowFacade.mode_workflow_team_coordinator is deprecated",
        ):
            assert facade.mode_workflow_team_coordinator is new_advisor

    def test_runtime_state_host_keeps_workflow_runtime_state_live(self):
        """WorkflowFacade should reflect canonical mutable workflow runtime state."""
        runtime_state_host = SimpleNamespace(
            _workflow_registry=MagicMock(name="runtime_registry"),
            _coordination_advisor=MagicMock(name="runtime_advisor"),
        )
        facade = WorkflowFacade(
            workflow_registry=MagicMock(name="stale_registry"),
            workflow_runtime=MagicMock(name="runtime"),
            workflow_optimization=MagicMock(name="optimization"),
            coordination_advisor=MagicMock(name="stale_advisor"),
            runtime_state_host=runtime_state_host,
        )

        assert facade.workflow_registry is runtime_state_host._workflow_registry
        assert facade.coordination_advisor is runtime_state_host._coordination_advisor
        with pytest.warns(
            DeprecationWarning,
            match="WorkflowFacade.mode_workflow_team_coordinator is deprecated",
        ):
            assert facade.mode_workflow_team_coordinator is runtime_state_host._coordination_advisor

        runtime_state_host._workflow_registry = MagicMock(name="updated_registry")
        runtime_state_host._coordination_advisor = MagicMock(name="updated_advisor")

        assert facade.workflow_registry is runtime_state_host._workflow_registry
        assert facade.coordination_advisor is runtime_state_host._coordination_advisor

    def test_runtime_state_host_setters_update_canonical_workflow_state(self):
        """WorkflowFacade compatibility setters should write through to runtime state."""
        runtime_state_host = SimpleNamespace(
            _workflow_registry=MagicMock(name="runtime_registry"),
            _coordination_advisor=MagicMock(name="runtime_advisor"),
        )
        facade = WorkflowFacade(
            workflow_registry=MagicMock(name="stale_registry"),
            workflow_runtime=MagicMock(name="runtime"),
            workflow_optimization=MagicMock(name="optimization"),
            coordination_advisor=MagicMock(name="stale_advisor"),
            runtime_state_host=runtime_state_host,
        )
        new_registry = MagicMock(name="new_registry")
        new_advisor = MagicMock(name="new_advisor")

        facade.workflow_registry = new_registry
        facade.coordination_advisor = new_advisor

        assert runtime_state_host._workflow_registry is new_registry
        assert runtime_state_host._coordination_advisor is new_advisor

        legacy_advisor = MagicMock(name="legacy_advisor")
        with pytest.warns(
            DeprecationWarning,
            match="WorkflowFacade.mode_workflow_team_coordinator is deprecated",
        ):
            facade.mode_workflow_team_coordinator = legacy_advisor
        assert runtime_state_host._coordination_advisor is legacy_advisor


class TestWorkflowFacadeProtocolConformance:
    """Tests that WorkflowFacade satisfies WorkflowFacadeProtocol."""

    def test_satisfies_protocol(self):
        """WorkflowFacade structurally conforms to WorkflowFacadeProtocol."""
        facade = WorkflowFacade()
        assert isinstance(facade, WorkflowFacadeProtocol)

    def test_protocol_properties_present(self):
        """All protocol-required properties are present on WorkflowFacade."""
        required = [
            "workflow_registry",
            "workflow_runtime",
            "workflow_optimization",
            "coordination_advisor",
        ]
        facade = WorkflowFacade()
        for prop in required:
            assert hasattr(facade, prop), f"Missing protocol property: {prop}"
