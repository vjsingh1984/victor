"""Tests for orchestrator_properties module.

Verifies that install_properties() correctly installs property
definitions onto the AgentOrchestrator class.
"""

import pytest
from unittest.mock import MagicMock


class TestOrchestratorPropertyInstallation:
    """Verify that install_properties populated AgentOrchestrator with the expected properties."""

    def test_install_properties_adds_simple_accessors(self):
        """Simple properties should be installed as real descriptors."""
        from victor.agent.orchestrator import AgentOrchestrator

        # These properties should exist on the class (installed by install_properties)
        simple_props = [
            "conversation_controller",
            "tool_pipeline",
            "streaming_controller",
            "streaming_handler",
            "task_analyzer",
            "provider_manager",
            "context_compactor",
            "tool_output_formatter",
            "usage_analytics",
            "sequence_tracker",
            "recovery_coordinator",
            "chunk_generator",
            "tool_planner",
            "task_coordinator",
            "code_correction_middleware",
            "checkpoint_manager",
            "vertical_context",
        ]
        for prop_name in simple_props:
            assert hasattr(
                AgentOrchestrator, prop_name
            ), f"Property '{prop_name}' not found on AgentOrchestrator"
            assert isinstance(
                getattr(AgentOrchestrator, prop_name), property
            ), f"'{prop_name}' should be a property descriptor"

    def test_install_properties_adds_lazy_coordinators(self):
        """Lazy coordinator properties should also be installed."""
        from victor.agent.orchestrator import AgentOrchestrator

        lazy_props = [
            "protocol_adapter",
            "turn_executor",
            "sync_chat_coordinator",
            "streaming_chat_coordinator",
            "unified_chat_coordinator",
            "intelligent_integration",
            "subagent_orchestrator",
            "coordination",
        ]
        for prop_name in lazy_props:
            assert hasattr(
                AgentOrchestrator, prop_name
            ), f"Lazy property '{prop_name}' not found on AgentOrchestrator"
            assert isinstance(
                getattr(AgentOrchestrator, prop_name), property
            ), f"'{prop_name}' should be a property descriptor"

    def test_properties_module_has_install_function(self):
        """The module should export install_properties."""
        from victor.agent.orchestrator_properties import install_properties

        assert callable(install_properties)

    def test_properties_module_has_registry(self):
        """The module should have a property registry."""
        from victor.agent.orchestrator_properties import _PROPERTY_REGISTRY

        assert isinstance(_PROPERTY_REGISTRY, dict)
        assert len(_PROPERTY_REGISTRY) >= 20  # At least 20 properties

    def test_tool_coordinator_alias_warns_and_delegates_to_deprecated_shim(self):
        """The legacy _tool_coordinator alias should warn and proxy the shim."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        shim = MagicMock(name="tool_shim")
        orchestrator._deprecated_tool_coordinator = shim

        with pytest.warns(
            DeprecationWarning,
            match="AgentOrchestrator._tool_coordinator is deprecated compatibility surface",
        ):
            result = orchestrator._tool_coordinator

        assert result is shim

    def test_tool_coordinator_alias_setter_warns(self):
        """The legacy _tool_coordinator setter should warn and update the shim slot."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        shim = MagicMock(name="tool_shim")

        with pytest.warns(
            DeprecationWarning,
            match="AgentOrchestrator._tool_coordinator is deprecated compatibility surface",
        ):
            orchestrator._tool_coordinator = shim

        assert orchestrator._deprecated_tool_coordinator is shim
