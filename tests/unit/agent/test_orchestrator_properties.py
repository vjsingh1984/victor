"""Tests for orchestrator_properties module.

Verifies that install_properties() correctly installs property
definitions onto the AgentOrchestrator class.
"""

import pytest
from types import SimpleNamespace
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
            "skill_matcher",
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
            "intelligent_integration",
            "subagent_orchestrator",
            "orchestration_facade",
            "coordination_advisor",
            "coordination",
        ]
        for prop_name in lazy_props:
            assert hasattr(
                AgentOrchestrator, prop_name
            ), f"Lazy property '{prop_name}' not found on AgentOrchestrator"
            assert isinstance(
                getattr(AgentOrchestrator, prop_name), property
            ), f"'{prop_name}' should be a property descriptor"

    def test_specialized_chat_coordinators_are_not_installed_on_orchestrator(self):
        """Deprecated specialized chat shims now live on the orchestration facade only."""
        from victor.agent.orchestrator import AgentOrchestrator

        removed_props = [
            "sync_chat_coordinator",
            "streaming_chat_coordinator",
            "unified_chat_coordinator",
        ]

        for prop_name in removed_props:
            assert hasattr(AgentOrchestrator, prop_name) is False

    def test_properties_module_has_install_function(self):
        """The module should export install_properties."""
        from victor.agent.orchestrator_properties import install_properties

        assert callable(install_properties)

    def test_properties_module_has_registry(self):
        """The module should have a property registry."""
        from victor.agent.orchestrator_properties import _PROPERTY_REGISTRY

        assert isinstance(_PROPERTY_REGISTRY, dict)
        assert len(_PROPERTY_REGISTRY) >= 20  # At least 20 properties

    def test_skill_matcher_property_returns_shared_matcher(self):
        """skill_matcher should expose the shared framework matcher surface."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        matcher = MagicMock(name="skill_matcher")
        orchestrator._skill_matcher = matcher

        assert orchestrator.skill_matcher is matcher

    def test_coordination_advisor_property_returns_lazy_framework_surface(self):
        """coordination_advisor should expose the normalized lazy coordination surface."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        advisor = MagicMock(name="coordination_advisor")
        orchestrator._coordination_advisor = advisor

        assert orchestrator.coordination_advisor is advisor
        assert orchestrator.coordination is advisor

    def test_orchestration_facade_property_resolves_lazy_surface(self):
        """orchestration_facade should resolve lazy runtime proxies to concrete facades."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        facade = MagicMock(name="orchestration_facade")
        orchestrator._orchestration_facade = SimpleNamespace(get_instance=lambda: facade)

        assert orchestrator.orchestration_facade is facade
        assert orchestrator._orchestration_facade is facade

    def test_mode_workflow_team_coordinator_alias_warns_and_maps_to_coordination_advisor(self):
        """The deprecated private coordinator alias should forward to _coordination_advisor."""
        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = object.__new__(AgentOrchestrator)
        advisor = MagicMock(name="coordination_advisor")

        with pytest.warns(
            DeprecationWarning,
            match=(
                "AgentOrchestrator._mode_workflow_team_coordinator is deprecated "
                "compatibility surface"
            ),
        ):
            orchestrator._mode_workflow_team_coordinator = advisor

        assert orchestrator._coordination_advisor is advisor

        with pytest.warns(
            DeprecationWarning,
            match=(
                "AgentOrchestrator._mode_workflow_team_coordinator is deprecated "
                "compatibility surface"
            ),
        ):
            assert orchestrator._mode_workflow_team_coordinator is advisor

    def test_removed_compatibility_aliases_are_not_installed_on_orchestrator(self):
        """Deprecated chat/session/tool coordinator aliases now live off-orchestrator."""
        from victor.agent.orchestrator import AgentOrchestrator

        removed_aliases = [
            "_tool_coordinator",
            "_chat_coordinator",
            "_session_coordinator",
        ]

        for prop_name in removed_aliases:
            assert hasattr(AgentOrchestrator, prop_name) is False

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
