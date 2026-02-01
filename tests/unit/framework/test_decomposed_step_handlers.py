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

"""Tests for decomposed step handlers (Phase 10.1).

Tests the single-responsibility step handlers that replace the monolithic
FrameworkStepHandler.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING
from unittest.mock import MagicMock, patch


if TYPE_CHECKING:
    pass


class MockVerticalContext:
    """Mock vertical context for testing."""

    def __init__(self):
        self.workflows: dict[str, Any] = {}
        self.rl_config: Any = None
        self.rl_hooks: Any = None
        self.team_specs: dict[str, Any] = {}
        self.chains: dict[str, Any] = {}
        self.personas: dict[str, Any] = {}

    def apply_workflows(self, workflows: dict[str, Any]) -> None:
        self.workflows.update(workflows)

    def apply_rl_config(self, config: Any) -> None:
        self.rl_config = config

    def apply_rl_hooks(self, hooks: Any) -> None:
        self.rl_hooks = hooks

    def apply_team_specs(self, specs: dict[str, Any]) -> None:
        self.team_specs.update(specs)


class MockIntegrationResult:
    """Mock integration result for testing."""

    def __init__(self):
        self.workflows_count: int = 0
        self.rl_learners_count: int = 0
        self.team_specs_count: int = 0
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.step_statuses: dict[str, Any] = {}
        self.step_details: dict[str, Any] = {}

    def add_info(self, msg: str) -> None:
        self.infos.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def record_step_status(
        self,
        step_name: str,
        status: str,
        details: Optional[dict] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Record step status (required by BaseStepHandler)."""
        self.step_statuses[step_name] = status
        if details:
            self.step_details[step_name] = details


class MockOrchestrator:
    """Mock orchestrator for testing."""

    def __init__(self):
        self._capability_loader: Any = None
        self._rl_hooks: Any = None

    def set_rl_hooks(self, hooks: Any) -> None:
        self._rl_hooks = hooks


class MockWorkflowProvider:
    """Mock workflow provider."""

    name = "test_vertical"

    @classmethod
    def get_workflows(cls) -> dict[str, Any]:
        return {"workflow1": {"steps": []}, "workflow2": {"steps": []}}


class MockRLConfigProvider:
    """Mock RL config provider."""

    name = "test_vertical"
    active_learners = ["tool_selector", "continuation_patience"]

    @classmethod
    def get_rl_config(cls):
        return cls


class MockTeamSpecProvider:
    """Mock team spec provider."""

    name = "test_vertical"

    @classmethod
    def get_team_specs(cls) -> dict[str, Any]:
        return {"team1": {"members": []}, "team2": {"members": []}}


class MockChainProvider:
    """Mock chain provider."""

    name = "test_vertical"

    @classmethod
    def get_chains(cls) -> dict[str, Any]:
        return {"chain1": {}, "chain2": {}}


class MockPersonaProvider:
    """Mock persona provider."""

    name = "test_vertical"

    @classmethod
    def get_personas(cls) -> dict[str, Any]:
        class MockPersona:
            def to_dict(self):
                return {"name": "test", "role": "tester", "expertise": []}

        return {"persona1": MockPersona()}


class MockHandlerProvider:
    """Mock handler provider."""

    name = "test_vertical"

    @classmethod
    def get_handlers(cls) -> dict[str, Any]:
        return {"handler1": lambda x: x}


class TestWorkflowStepHandler:
    """Tests for WorkflowStepHandler."""

    def test_applies_workflows_from_provider(self):
        """Test that workflows are applied from provider."""
        from victor.framework.decomposed_handlers import WorkflowStepHandler

        handler = WorkflowStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        handler.apply(orchestrator, MockWorkflowProvider, context, result)

        assert result.workflows_count == 2
        assert "workflow1" in context.workflows
        assert "workflow2" in context.workflows

    def test_skips_when_no_workflow_provider(self):
        """Test that handler skips when no workflow provider."""
        from victor.framework.decomposed_handlers import WorkflowStepHandler

        handler = WorkflowStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        class NoWorkflowVertical:
            name = "test"

        handler.apply(orchestrator, NoWorkflowVertical, context, result)

        assert result.workflows_count == 0
        assert len(context.workflows) == 0

    def test_handler_has_correct_order(self):
        """Test handler has correct order."""
        from victor.framework.decomposed_handlers import WorkflowStepHandler

        handler = WorkflowStepHandler()
        assert handler.order == 60

    def test_handler_has_correct_name(self):
        """Test handler has correct name."""
        from victor.framework.decomposed_handlers import WorkflowStepHandler

        handler = WorkflowStepHandler()
        assert handler.name == "workflow"


class TestRLConfigStepHandler:
    """Tests for RLConfigStepHandler."""

    def test_applies_rl_config_from_provider(self):
        """Test that RL config is applied from provider."""
        from victor.framework.decomposed_handlers import RLConfigStepHandler

        handler = RLConfigStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        handler.apply(orchestrator, MockRLConfigProvider, context, result)

        assert context.rl_config is not None
        assert result.rl_learners_count == 2

    def test_applies_rl_hooks(self):
        """Test that RL hooks are applied if available."""
        from victor.framework.decomposed_handlers import RLConfigStepHandler

        class VerticalWithHooks(MockRLConfigProvider):
            @classmethod
            def get_rl_hooks(cls):
                return MagicMock()

        handler = RLConfigStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        handler.apply(orchestrator, VerticalWithHooks, context, result)

        assert context.rl_hooks is not None

    def test_skips_when_no_rl_config(self):
        """Test handler skips when no RL config."""
        from victor.framework.decomposed_handlers import RLConfigStepHandler

        handler = RLConfigStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        class NoRLVertical:
            name = "test"

        handler.apply(orchestrator, NoRLVertical, context, result)

        assert context.rl_config is None
        assert result.rl_learners_count == 0

    def test_handler_has_correct_order(self):
        """Test handler has correct order."""
        from victor.framework.decomposed_handlers import RLConfigStepHandler

        handler = RLConfigStepHandler()
        assert handler.order == 61


class TestTeamSpecStepHandler:
    """Tests for TeamSpecStepHandler."""

    def test_applies_team_specs_from_provider(self):
        """Test that team specs are applied from provider."""
        from victor.framework.decomposed_handlers import TeamSpecStepHandler

        handler = TeamSpecStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        handler.apply(orchestrator, MockTeamSpecProvider, context, result)

        assert result.team_specs_count == 2
        assert "team1" in context.team_specs

    def test_skips_when_no_team_specs(self):
        """Test handler skips when no team specs."""
        from victor.framework.decomposed_handlers import TeamSpecStepHandler

        handler = TeamSpecStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        class NoTeamVertical:
            name = "test"

        handler.apply(orchestrator, NoTeamVertical, context, result)

        assert result.team_specs_count == 0

    def test_handler_has_correct_order(self):
        """Test handler has correct order."""
        from victor.framework.decomposed_handlers import TeamSpecStepHandler

        handler = TeamSpecStepHandler()
        assert handler.order == 62


class TestChainStepHandler:
    """Tests for ChainStepHandler."""

    def test_applies_chains_from_provider(self):
        """Test that chains are applied from provider."""
        from victor.framework.decomposed_handlers import ChainStepHandler

        handler = ChainStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        # Mock chain registry
        with patch("victor.framework.decomposed_handlers.get_chain_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_reg.return_value = mock_registry

            handler.apply(orchestrator, MockChainProvider, context, result)

            # Should have called register twice
            assert mock_registry.register.call_count == 2

    def test_skips_when_no_chains(self):
        """Test handler skips when no chains."""
        from victor.framework.decomposed_handlers import ChainStepHandler

        handler = ChainStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        class NoChainVertical:
            name = "test"

        handler.apply(orchestrator, NoChainVertical, context, result)

        assert len(result.infos) == 0

    def test_handler_has_correct_order(self):
        """Test handler has correct order."""
        from victor.framework.decomposed_handlers import ChainStepHandler

        handler = ChainStepHandler()
        assert handler.order == 63


class TestPersonaStepHandler:
    """Tests for PersonaStepHandler."""

    def test_applies_personas_from_provider(self):
        """Test that personas are applied from provider."""
        from victor.framework.decomposed_handlers import PersonaStepHandler

        handler = PersonaStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        # Mock persona registry
        with patch("victor.framework.decomposed_handlers.get_persona_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_reg.return_value = mock_registry

            handler.apply(orchestrator, MockPersonaProvider, context, result)

            # Should have called register
            assert mock_registry.register.call_count == 1

    def test_skips_when_no_personas(self):
        """Test handler skips when no personas."""
        from victor.framework.decomposed_handlers import PersonaStepHandler

        handler = PersonaStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        class NoPersonaVertical:
            name = "test"

        handler.apply(orchestrator, NoPersonaVertical, context, result)

        assert len(result.infos) == 0

    def test_handler_has_correct_order(self):
        """Test handler has correct order."""
        from victor.framework.decomposed_handlers import PersonaStepHandler

        handler = PersonaStepHandler()
        assert handler.order == 64


class TestCapabilityProviderStepHandler:
    """Tests for CapabilityProviderStepHandler."""

    def test_handler_has_correct_order(self):
        """Test handler has correct order."""
        from victor.framework.decomposed_handlers import CapabilityProviderStepHandler

        handler = CapabilityProviderStepHandler()
        assert handler.order == 65

    def test_skips_when_no_capability_provider(self):
        """Test handler skips when no capability provider."""
        from victor.framework.decomposed_handlers import CapabilityProviderStepHandler

        handler = CapabilityProviderStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        class NoCapabilityVertical:
            name = "test"

        handler.apply(orchestrator, NoCapabilityVertical, context, result)

        assert len(result.infos) == 0


class TestToolGraphStepHandler:
    """Tests for ToolGraphStepHandler."""

    def test_handler_has_correct_order(self):
        """Test handler has correct order."""
        from victor.framework.decomposed_handlers import ToolGraphStepHandler

        handler = ToolGraphStepHandler()
        assert handler.order == 66

    def test_skips_when_no_tool_graph(self):
        """Test handler skips when no tool graph."""
        from victor.framework.decomposed_handlers import ToolGraphStepHandler

        handler = ToolGraphStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        class NoToolGraphVertical:
            name = "test"

        handler.apply(orchestrator, NoToolGraphVertical, context, result)

        assert len(result.infos) == 0


class TestHandlerRegistrationStepHandler:
    """Tests for HandlerRegistrationStepHandler."""

    def test_applies_handlers_from_provider(self):
        """Test that handlers are registered from provider."""
        from victor.framework.decomposed_handlers import HandlerRegistrationStepHandler

        handler = HandlerRegistrationStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        # Mock handler registry
        with patch("victor.framework.decomposed_handlers.get_handler_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_reg.return_value = mock_registry

            handler.apply(orchestrator, MockHandlerProvider, context, result)

            # Should have called register
            assert mock_registry.register.call_count == 1

    def test_skips_when_no_handlers(self):
        """Test handler skips when no handlers."""
        from victor.framework.decomposed_handlers import HandlerRegistrationStepHandler

        handler = HandlerRegistrationStepHandler()
        orchestrator = MockOrchestrator()
        context = MockVerticalContext()
        result = MockIntegrationResult()

        class NoHandlerVertical:
            name = "test"

        handler.apply(orchestrator, NoHandlerVertical, context, result)

        assert len(result.infos) == 0

    def test_handler_has_correct_order(self):
        """Test handler has correct order."""
        from victor.framework.decomposed_handlers import HandlerRegistrationStepHandler

        handler = HandlerRegistrationStepHandler()
        assert handler.order == 67


class TestDecomposedHandlerRegistry:
    """Tests for the registry of decomposed handlers."""

    def test_all_handlers_registered(self):
        """Test that all decomposed handlers are available."""
        from victor.framework.decomposed_handlers import get_decomposed_handlers

        handlers = get_decomposed_handlers()

        assert len(handlers) == 8
        names = [h.name for h in handlers]
        assert "workflow" in names
        assert "rl_config" in names
        assert "team_spec" in names
        assert "chain" in names
        assert "persona" in names
        assert "capability_provider" in names
        assert "tool_graph" in names
        assert "handler_registration" in names

    def test_handlers_ordered_correctly(self):
        """Test that handlers are in correct order."""
        from victor.framework.decomposed_handlers import get_decomposed_handlers

        handlers = get_decomposed_handlers()

        orders = [h.order for h in handlers]
        # Should be in ascending order
        assert orders == sorted(orders)

    def test_handlers_have_unique_names(self):
        """Test that all handlers have unique names."""
        from victor.framework.decomposed_handlers import get_decomposed_handlers

        handlers = get_decomposed_handlers()

        names = [h.name for h in handlers]
        assert len(names) == len(set(names))


class TestBackwardCompatibility:
    """Tests for backward compatibility with FrameworkStepHandler."""

    def test_framework_step_handler_still_works(self):
        """Test that original FrameworkStepHandler still works."""
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()
        assert handler.name == "framework"
        assert handler.order == 60

    def test_decomposed_handlers_can_replace_framework_handler(self):
        """Test that decomposed handlers can replace framework handler."""
        from victor.framework.decomposed_handlers import get_decomposed_handlers

        handlers = get_decomposed_handlers()

        # Should cover orders 60-67
        orders = {h.order for h in handlers}
        assert 60 in orders  # workflow
        assert 61 in orders  # rl_config
        assert 62 in orders  # team_spec
        assert 63 in orders  # chain
        assert 64 in orders  # persona
        assert 65 in orders  # capability_provider
        assert 66 in orders  # tool_graph
        assert 67 in orders  # handler_registration
