from unittest.mock import MagicMock

from victor.agent.runtime.interaction_runtime import create_interaction_runtime_components
from victor.runtime.context import ResolvedRuntimeServices


def _runtime_kwargs():
    return {
        "enabled_tools": ["shell"],
        "factory": MagicMock(),
        "tool_pipeline": MagicMock(),
        "tool_registry": MagicMock(),
        "tool_executor": MagicMock(),
        "tool_cache": MagicMock(),
        "tool_budget": 12,
        "tool_selector": MagicMock(),
        "tool_access_controller": MagicMock(),
        "mode_controller": MagicMock(),
        "argument_normalizer": MagicMock(),
        "session_state_manager": MagicMock(),
        "lifecycle_manager": MagicMock(),
        "memory_manager": MagicMock(),
        "memory_session_id": "session-1",
        "checkpoint_manager": MagicMock(),
        "cost_tracker": MagicMock(),
        "conversation_controller": MagicMock(),
        "streaming_coordinator": MagicMock(),
    }


def test_create_interaction_runtime_components_prefers_runtime_service_bundle():
    tool_service = MagicMock(name="tool_service")
    session_service = MagicMock(name="session_service")
    context_service = MagicMock(name="context_service")
    recovery_service = MagicMock(name="recovery_service")
    provider_service = MagicMock(name="provider_service")
    chat_service = MagicMock(name="chat_service")

    components = create_interaction_runtime_components(
        runtime_services=ResolvedRuntimeServices(
            chat=chat_service,
            tool=tool_service,
            session=session_service,
            context=context_service,
            provider=provider_service,
            recovery=recovery_service,
        ),
        **_runtime_kwargs(),
    )

    assert components.chat_service is chat_service
    assert components.tool_service is tool_service
    assert components.session_service is session_service
    assert components.context_service is context_service
    assert components.recovery_service is recovery_service
    tool_service.bind_runtime_components.assert_called_once()
    tool_service.set_enabled_tools.assert_called_once_with(["shell"])
    session_service.bind_runtime_components.assert_called_once()


def test_create_interaction_runtime_components_uses_context_adapter_fallback():
    from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

    components = create_interaction_runtime_components(
        runtime_services=ResolvedRuntimeServices(),
        **_runtime_kwargs(),
    )

    assert isinstance(components.context_service, ContextServiceAdapter)
