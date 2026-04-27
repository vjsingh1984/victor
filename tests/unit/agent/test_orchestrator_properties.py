"""Tests for orchestrator_properties module.

Verifies that install_properties() correctly installs property
definitions onto the AgentOrchestrator class.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.agent.services.chat_compat_telemetry import (
    get_deprecated_chat_shim_telemetry,
    reset_deprecated_chat_shim_telemetry,
)


@pytest.fixture(autouse=True)
def _reset_chat_compat_telemetry():
    reset_deprecated_chat_shim_telemetry()
    yield
    reset_deprecated_chat_shim_telemetry()


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

    def test_chat_coordinator_alias_warns_and_delegates_to_deprecated_shim(self):
        """The legacy _chat_coordinator alias should warn and proxy the shim."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        shim = MagicMock(name="chat_shim")
        orchestrator._deprecated_chat_coordinator = shim

        with pytest.warns(
            DeprecationWarning,
            match="AgentOrchestrator._chat_coordinator is deprecated compatibility surface",
        ):
            result = orchestrator._chat_coordinator

        assert result is shim
        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["agent_orchestrator._chat_coordinator_get.compat_property"] == 1

    def test_chat_coordinator_alias_setter_warns(self):
        """The legacy _chat_coordinator setter should warn and update the shim slot."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        shim = MagicMock(name="chat_shim")

        with pytest.warns(
            DeprecationWarning,
            match="AgentOrchestrator._chat_coordinator is deprecated compatibility surface",
        ):
            orchestrator._chat_coordinator = shim

        assert orchestrator._deprecated_chat_coordinator is shim
        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["agent_orchestrator._chat_coordinator_set.compat_property"] == 1

    def test_session_coordinator_alias_warns_and_delegates_to_deprecated_shim(self):
        """The legacy _session_coordinator alias should warn and proxy the shim."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        shim = MagicMock(name="session_shim")
        orchestrator._deprecated_session_coordinator = shim

        with pytest.warns(
            DeprecationWarning,
            match="AgentOrchestrator._session_coordinator is deprecated compatibility surface",
        ):
            result = orchestrator._session_coordinator

        assert result is shim

    def test_session_coordinator_alias_setter_warns(self):
        """The legacy _session_coordinator setter should warn and update the shim slot."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        shim = MagicMock(name="session_shim")

        with pytest.warns(
            DeprecationWarning,
            match="AgentOrchestrator._session_coordinator is deprecated compatibility surface",
        ):
            orchestrator._session_coordinator = shim

        assert orchestrator._deprecated_session_coordinator is shim

    def test_provider_coordinator_alias_warns_and_derives_from_provider_runtime(self):
        """The legacy _provider_coordinator alias should warn and derive from provider_runtime."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        shim = MagicMock(name="provider_shim")
        orchestrator._provider_runtime = SimpleNamespace(provider_coordinator=shim)

        with pytest.warns(
            DeprecationWarning,
            match="AgentOrchestrator._provider_coordinator is deprecated compatibility surface",
        ):
            result = orchestrator._provider_coordinator

        assert result is shim

    def test_provider_coordinator_alias_setter_warns_and_overrides_runtime(self):
        """The legacy _provider_coordinator setter should warn and update the override slot."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        orchestrator._provider_runtime = SimpleNamespace(
            provider_coordinator=MagicMock(name="runtime_provider_shim")
        )
        shim = MagicMock(name="provider_shim")

        with pytest.warns(
            DeprecationWarning,
            match="AgentOrchestrator._provider_coordinator is deprecated compatibility surface",
        ):
            orchestrator._provider_coordinator = shim

        assert orchestrator._deprecated_provider_coordinator is shim
        with pytest.warns(
            DeprecationWarning,
            match="AgentOrchestrator._provider_coordinator is deprecated compatibility surface",
        ):
            assert orchestrator._provider_coordinator is shim

    def test_provider_switch_coordinator_alias_warns_and_derives_from_provider_runtime(self):
        """The legacy _provider_switch_coordinator alias should derive from provider_runtime."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        shim = MagicMock(name="provider_switch_shim")
        orchestrator._provider_runtime = SimpleNamespace(provider_switch_coordinator=shim)

        with pytest.warns(
            DeprecationWarning,
            match=(
                "AgentOrchestrator._provider_switch_coordinator is deprecated "
                "compatibility surface"
            ),
        ):
            result = orchestrator._provider_switch_coordinator

        assert result is shim

    def test_provider_switch_coordinator_alias_setter_warns_and_overrides_runtime(self):
        """The legacy _provider_switch_coordinator setter should warn and update the override slot."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        orchestrator._provider_runtime = SimpleNamespace(
            provider_switch_coordinator=MagicMock(name="runtime_provider_switch_shim")
        )
        shim = MagicMock(name="provider_switch_shim")

        with pytest.warns(
            DeprecationWarning,
            match=(
                "AgentOrchestrator._provider_switch_coordinator is deprecated "
                "compatibility surface"
            ),
        ):
            orchestrator._provider_switch_coordinator = shim

        assert orchestrator._deprecated_provider_switch_coordinator is shim
        with pytest.warns(
            DeprecationWarning,
            match=(
                "AgentOrchestrator._provider_switch_coordinator is deprecated "
                "compatibility surface"
            ),
        ):
            assert orchestrator._provider_switch_coordinator is shim

    def test_specialized_chat_coordinators_use_deprecated_backing_slots(self):
        """Lazy chat coordinator properties should store instances in deprecated slots."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        orchestrator._deprecated_sync_chat_coordinator = MagicMock(name="sync")
        orchestrator._deprecated_streaming_chat_coordinator = MagicMock(name="streaming")
        orchestrator._deprecated_unified_chat_coordinator = MagicMock(name="unified")

        with pytest.warns(
            DeprecationWarning,
            match="AgentOrchestrator.sync_chat_coordinator is deprecated compatibility surface",
        ):
            assert orchestrator.sync_chat_coordinator._mock_name == "sync"

        with pytest.warns(
            DeprecationWarning,
            match=(
                "AgentOrchestrator.streaming_chat_coordinator is deprecated "
                "compatibility surface"
            ),
        ):
            assert orchestrator.streaming_chat_coordinator._mock_name == "streaming"

        with pytest.warns(
            DeprecationWarning,
            match=(
                "AgentOrchestrator.unified_chat_coordinator is deprecated " "compatibility surface"
            ),
        ):
            assert orchestrator.unified_chat_coordinator._mock_name == "unified"

        telemetry = get_deprecated_chat_shim_telemetry()
        assert telemetry["agent_orchestrator.sync_chat_coordinator.compat_property"] == 1
        assert telemetry["agent_orchestrator.streaming_chat_coordinator.compat_property"] == 1
        assert telemetry["agent_orchestrator.unified_chat_coordinator.compat_property"] == 1

    def test_sync_chat_coordinator_helper_passes_protocol_adapter_runtime(self):
        """Sync chat shim creation should depend on the protocol adapter, not the concrete orchestrator."""
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.orchestrator_properties import _ensure_sync_chat_coordinator

        orchestrator = object.__new__(AgentOrchestrator)
        adapter = MagicMock(name="protocol_adapter")
        chat_service = MagicMock(name="chat_service")
        orchestrator._protocol_adapter = adapter
        orchestrator._deprecated_sync_chat_coordinator = None
        orchestrator._chat_service = chat_service

        with (
            patch("victor.agent.services.sync_chat_compat.SyncChatCoordinator") as coordinator_cls,
            patch("victor.agent.query_classifier.QueryClassifier") as query_classifier_cls,
        ):
            shim = MagicMock(name="sync_chat_shim")
            coordinator_cls.return_value = shim

            result = _ensure_sync_chat_coordinator(orchestrator)

        assert result is shim
        assert orchestrator._deprecated_sync_chat_coordinator is shim
        kwargs = coordinator_cls.call_args.kwargs
        assert kwargs["chat_context"] is adapter
        assert kwargs["tool_context"] is adapter
        assert kwargs["provider_context"] is adapter
        assert kwargs["orchestrator"] is adapter
        assert kwargs["chat_service"] is chat_service
        assert kwargs["query_classifier"] is query_classifier_cls.return_value

    @pytest.mark.parametrize(
        ("property_name", "backing_attr", "facade_attr", "facade_value_attr"),
        [
            (
                "conversation_controller",
                "_conversation_controller",
                "_chat_facade",
                "conversation_controller",
            ),
            ("tool_pipeline", "_tool_pipeline", "_tool_facade", "tool_pipeline"),
            (
                "streaming_handler",
                "_streaming_handler",
                "_orchestration_facade",
                "streaming_handler",
            ),
            ("provider_manager", "_provider_manager", "_provider_facade", "provider_manager"),
            ("vertical_context", "_vertical_context", "_orchestration_facade", "vertical_context"),
        ],
    )
    def test_simple_properties_prefer_direct_canonical_attributes(
        self,
        property_name,
        backing_attr,
        facade_attr,
        facade_value_attr,
    ):
        """Compatibility facades must not become the behavior owner for simple properties."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        direct_value = MagicMock(name=f"{property_name}_direct")
        facade_value = MagicMock(name=f"{property_name}_facade")
        setattr(orchestrator, backing_attr, direct_value)
        setattr(orchestrator, facade_attr, SimpleNamespace(**{facade_value_attr: facade_value}))

        assert getattr(orchestrator, property_name) is direct_value

    def test_session_ledger_property_prefers_direct_canonical_attribute(self):
        """session_ledger should read from the canonical backing attribute, not the facade copy."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        direct_value = MagicMock(name="ledger_direct")
        facade_value = MagicMock(name="ledger_facade")
        orchestrator._session_ledger = direct_value
        orchestrator._session_facade = SimpleNamespace(session_ledger=facade_value)

        assert orchestrator.session_ledger is direct_value

    def test_session_ledger_setter_updates_canonical_backing_attribute(self):
        """session_ledger setter should update the direct canonical slot."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        orchestrator._session_ledger = MagicMock(name="old_ledger")
        orchestrator._session_facade = SimpleNamespace(
            session_ledger=MagicMock(name="facade_ledger")
        )
        new_value = MagicMock(name="new_ledger")

        orchestrator.session_ledger = new_value

        assert orchestrator._session_ledger is new_value
