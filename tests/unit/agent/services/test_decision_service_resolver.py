"""Tests for the shared decision-service resolver.

``get_decision_service`` is the single, None-safe way to fetch the LLM decision
service from a DI container. It centralizes the "may be disabled / unregistered"
contract that was previously duplicated (and inconsistent) across six call sites
— some of which used bare ``container.get()`` (which raises
``ServiceNotFoundError`` when the backend is ``HEURISTIC`` / unregistered) and
some of which gated on the now-legacy ``USE_LLM_DECISION_SERVICE`` flag.
"""

from unittest.mock import MagicMock

from victor.agent.services.protocols.decision_service import (
    LLMDecisionServiceProtocol,
    get_decision_service,
)
from victor.agent.services.runtime_intelligence import RuntimeIntelligenceService
from victor.core.container import ServiceContainer


def test_none_container_returns_none():
    """A None container (e.g. pre-bootstrap orchestrator) yields None, no raise."""
    assert get_decision_service(None) is None


def test_registered_service_is_returned():
    """A registered service is returned by identity."""
    container = ServiceContainer()
    svc = MagicMock(name="decision_service")
    container.register_instance(LLMDecisionServiceProtocol, svc)
    assert get_decision_service(container) is svc


def test_unregistered_returns_none_without_raising():
    """When no backend is registered (HEURISTIC / unhealthy), return None — not raise.

    This is the regression guard for the former bare ``container.get()`` sites
    (orchestrator._check_tool_necessity_via_edge, tool_selection decider) which
    would have raised ``ServiceNotFoundError`` in this state.
    """
    container = ServiceContainer()
    assert get_decision_service(container) is None


def test_get_optional_raising_is_swallowed():
    """Unexpected errors during resolution never crash a turn."""
    container = MagicMock()
    container.get_optional = MagicMock(side_effect=RuntimeError("boom"))
    assert get_decision_service(container) is None


def test_minimal_container_without_get_optional_falls_back_to_get():
    """Containers exposing only ``get()`` still resolve a registered service."""
    svc = MagicMock(name="decision_service")
    container = MagicMock()
    container.get_optional = None  # force the ``get()`` fallback path
    container.get = MagicMock(return_value=svc)
    assert get_decision_service(container) is svc


def test_minimal_container_get_raising_returns_none():
    """The ``get()`` fallback path is also None-safe on failure."""
    container = MagicMock()
    container.get_optional = None
    container.get = MagicMock(side_effect=RuntimeError("boom"))
    assert get_decision_service(container) is None


def test_runtime_intelligence_resolve_delegates_to_resolver():
    """RuntimeIntelligenceService._resolve_decision_service delegates to the helper."""
    container = ServiceContainer()
    # Unregistered -> None (no raise), matching the shared resolver contract.
    assert RuntimeIntelligenceService._resolve_decision_service(container) is None
    svc = MagicMock(name="decision_service")
    container.register_instance(LLMDecisionServiceProtocol, svc)
    assert RuntimeIntelligenceService._resolve_decision_service(container) is svc
