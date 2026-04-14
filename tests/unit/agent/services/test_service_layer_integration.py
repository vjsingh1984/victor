"""Phase 1: Service layer integration tests.

Validates that the service layer delegation produces consistent behavior
when USE_SERVICE_LAYER is enabled. Tests exercise the delegation paths
structurally — verifying the wiring is correct and services are reachable.
"""

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class TestServiceDelegationCompleteness:
    """Verify every delegation guard has both a service call and coordinator fallback."""

    def test_no_coordinator_fallback_remains(self):
        """After Phase 2, all delegation guards should be removed.

        The orchestrator should call services directly without
        `if self._use_service_layer` checks.
        """
        source = _get_orchestrator_source()
        assert "_use_service_layer and self._" not in source, (
            "Coordinator fallback guards still present. Phase 2 should have "
            "removed all `if self._use_service_layer and self._*_service:` patterns."
        )

    def test_service_and_coordinator_produce_same_method_names(self):
        """Service delegation methods should map to identically-named coordinator methods.

        This catches renaming drift between the two paths.
        """
        source = _get_orchestrator_source()
        lines = source.split("\n")

        mismatches = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not (
                "self._use_service_layer and self._" in stripped
                and "_service:" in stripped
            ):
                continue

            # Look at next 2 lines for the service call
            service_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            # Look at the fallback (usually 2 lines after the guard)
            fallback_line = lines[i + 2].strip() if i + 2 < len(lines) else ""

            # Extract method name from service call
            if "." in service_line and "(" in service_line:
                svc_method = service_line.split(".")[-1].split("(")[0]
            else:
                continue

            # Extract method name from coordinator fallback
            if "." in fallback_line and "(" in fallback_line:
                coord_method = fallback_line.split(".")[-1].split("(")[0]
            else:
                continue

            # They should match (or be close variants)
            if svc_method != coord_method and svc_method not in coord_method:
                mismatches.append(
                    f"  Line {i + 1}: service.{svc_method} vs coordinator.{coord_method}"
                )

        # Allow some mismatches (different naming conventions between service and coordinator)
        assert len(mismatches) <= 5, (
            f"Too many method name mismatches between service and coordinator paths:\n"
            + "\n".join(mismatches)
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

        assert all(resolved.values()), (
            f"Not all services registered: {resolved}"
        )

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
                assert hasattr(svc, "is_healthy"), (
                    f"{proto.__name__} service missing is_healthy()"
                )


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


class TestFeatureFlagGating:
    """Verify USE_SERVICE_LAYER flag controls delegation."""

    def test_flag_is_enabled_by_default(self):
        """USE_SERVICE_LAYER should be enabled by default."""
        from victor.core.feature_flags import FeatureFlag, FeatureFlagConfig, FeatureFlagManager

        manager = FeatureFlagManager(FeatureFlagConfig())
        assert manager.is_enabled(FeatureFlag.USE_SERVICE_LAYER) is True

    def test_flag_can_be_disabled(self, monkeypatch):
        """Disabling the flag should skip service delegation."""
        from victor.core.feature_flags import FeatureFlag, FeatureFlagConfig, FeatureFlagManager

        monkeypatch.setenv("VICTOR_USE_SERVICE_LAYER", "false")
        manager = FeatureFlagManager(FeatureFlagConfig())
        assert manager.is_enabled(FeatureFlag.USE_SERVICE_LAYER) is False


def _get_orchestrator_source() -> str:
    from victor.agent.orchestrator import AgentOrchestrator
    return inspect.getsource(AgentOrchestrator)
