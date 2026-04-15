"""Service layer integration tests.

Validates that the service layer is mandatory — all orchestrator delegation
methods call services directly without coordinator fallback.
"""

import inspect
import re
from unittest.mock import MagicMock


class TestServiceDelegationCompleteness:
    """Verify services are mandatory with no coordinator fallback."""

    def test_no_coordinator_fallback_remains(self):
        """No service None-guards should exist in the orchestrator.

        The orchestrator calls services directly — there should be zero
        `if self._*_service:` patterns.
        """
        source = _get_orchestrator_source()
        guards = re.findall(r"if self\._\w+_service:", source)
        assert len(guards) == 0, (
            f"Found {len(guards)} service None-guards. "
            f"Services are mandatory — remove coordinator fallbacks."
        )

    def test_service_methods_exist_on_orchestrator(self):
        """Key service delegation methods must exist on the orchestrator."""
        from victor.agent.orchestrator import AgentOrchestrator

        expected_methods = [
            "chat",
            "chat_with_planning",
            "stream_chat",
            "get_available_tools",
            "get_enabled_tools",
            "set_enabled_tools",
            "is_tool_enabled",
            "save_checkpoint",
            "restore_checkpoint",
            "get_recent_sessions",
            "get_session_stats",
            "get_context_metrics",
            "get_current_provider_info",
            "switch_provider",
        ]
        for method_name in expected_methods:
            assert hasattr(AgentOrchestrator, method_name), (
                f"AgentOrchestrator missing expected method: {method_name}"
            )


class TestBootstrapServiceCreation:
    """Verify all 6 services are created during bootstrap."""

    def test_bootstrap_creates_and_registers_all_services(self):
        """Full bootstrap should register Chat, Tool, Session, Context, Provider, Recovery."""
        from victor.core.bootstrap_services import bootstrap_new_services
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        bootstrap_new_services(
            container,
            conversation_controller=MagicMock(),
            streaming_coordinator=MagicMock(),
        )

        from victor.agent.services.protocols import (
            ChatServiceProtocol,
            ContextServiceProtocol,
            ProviderServiceProtocol,
            RecoveryServiceProtocol,
            SessionServiceProtocol,
            ToolServiceProtocol,
        )

        resolved = {}
        for name, proto in [
            ("chat", ChatServiceProtocol),
            ("tool", ToolServiceProtocol),
            ("session", SessionServiceProtocol),
            ("context", ContextServiceProtocol),
            ("provider", ProviderServiceProtocol),
            ("recovery", RecoveryServiceProtocol),
        ]:
            svc = container.get_optional(proto)
            resolved[name] = svc is not None

        assert all(resolved.values()), f"Not all services registered: {resolved}"

    def test_services_have_is_healthy_method(self):
        """Every resolved service must implement is_healthy() for monitoring."""
        from victor.core.bootstrap_services import bootstrap_new_services
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        bootstrap_new_services(
            container,
            conversation_controller=MagicMock(),
            streaming_coordinator=MagicMock(),
        )

        from victor.agent.services.protocols import (
            ChatServiceProtocol,
            ContextServiceProtocol,
            ProviderServiceProtocol,
            RecoveryServiceProtocol,
            SessionServiceProtocol,
            ToolServiceProtocol,
        )

        for proto in [
            ChatServiceProtocol,
            ToolServiceProtocol,
            SessionServiceProtocol,
            ContextServiceProtocol,
            ProviderServiceProtocol,
            RecoveryServiceProtocol,
        ]:
            svc = container.get_optional(proto)
            if svc is not None:
                assert hasattr(svc, "is_healthy"), f"{proto.__name__} service missing is_healthy()"


class TestExecutionContextServiceAccess:
    """Verify ExecutionContext provides access to all 6 services."""

    def test_execution_context_resolves_all_services(self):
        """ServiceAccessor should resolve all 6 services from bootstrapped container."""
        from victor.core.bootstrap_services import bootstrap_new_services
        from victor.core.container import ServiceContainer
        from victor.runtime.context import ExecutionContext

        container = ServiceContainer()
        bootstrap_new_services(
            container,
            conversation_controller=MagicMock(),
            streaming_coordinator=MagicMock(),
        )

        ctx = ExecutionContext.create(
            settings=MagicMock(),
            container=container,
            session_id="integration-test",
        )

        assert ctx.services.chat is not None, "chat service not resolved"
        assert ctx.services.tool is not None, "tool service not resolved"
        assert ctx.services.session is not None, "session service not resolved"
        assert ctx.services.context is not None, "context service not resolved"
        assert ctx.services.provider is not None, "provider service not resolved"
        assert ctx.services.recovery is not None, "recovery service not resolved"


def _get_orchestrator_source() -> str:
    from victor.agent.orchestrator import AgentOrchestrator

    return inspect.getsource(AgentOrchestrator)
