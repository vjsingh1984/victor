"""Tests for deprecated package-root coordinator exports."""

import pytest


def test_session_package_exports_warn():
    """Package-root session coordinator exports should be compatibility-only."""
    from victor.agent.coordinators.session_coordinator import (
        SessionCoordinator,
        create_session_coordinator,
    )

    with pytest.warns(
        DeprecationWarning,
        match="victor.agent.coordinators.SessionCoordinator is deprecated compatibility surface",
    ):
        from victor.agent.coordinators import SessionCoordinator as package_session_coordinator

    with pytest.warns(
        DeprecationWarning,
        match=(
            "victor.agent.coordinators.create_session_coordinator is deprecated "
            "compatibility surface"
        ),
    ):
        from victor.agent.coordinators import (
            create_session_coordinator as package_create_session_coordinator,
        )

    assert package_session_coordinator is SessionCoordinator
    assert package_create_session_coordinator is create_session_coordinator


def test_specialized_chat_package_exports_warn():
    """Package-root specialized chat exports should be compatibility-only."""
    from victor.agent.coordinators.sync_chat_coordinator import SyncChatCoordinator
    from victor.agent.coordinators.streaming_chat_coordinator import StreamingChatCoordinator
    from victor.agent.coordinators.unified_chat_coordinator import UnifiedChatCoordinator

    with pytest.warns(
        DeprecationWarning,
        match=(
            "victor.agent.coordinators.SyncChatCoordinator is deprecated "
            "compatibility surface"
        ),
    ):
        from victor.agent.coordinators import SyncChatCoordinator as package_sync_chat_coordinator

    with pytest.warns(
        DeprecationWarning,
        match=(
            "victor.agent.coordinators.StreamingChatCoordinator is deprecated "
            "compatibility surface"
        ),
    ):
        from victor.agent.coordinators import (
            StreamingChatCoordinator as package_streaming_chat_coordinator,
        )

    with pytest.warns(
        DeprecationWarning,
        match=(
            "victor.agent.coordinators.UnifiedChatCoordinator is deprecated "
            "compatibility surface"
        ),
    ):
        from victor.agent.coordinators import (
            UnifiedChatCoordinator as package_unified_chat_coordinator,
        )

    assert package_sync_chat_coordinator is SyncChatCoordinator
    assert package_streaming_chat_coordinator is StreamingChatCoordinator
    assert package_unified_chat_coordinator is UnifiedChatCoordinator


def test_chat_protocol_package_exports_warn():
    """Package-root chat protocol exports should be compatibility-only."""
    from victor.agent.services.protocols.chat_runtime import (
        ChatContextProtocol,
        ChatOrchestratorProtocol,
        ProviderContextProtocol,
        ToolContextProtocol,
    )

    with pytest.warns(
        DeprecationWarning,
        match=(
            "victor.agent.coordinators.ChatContextProtocol is deprecated "
            "compatibility surface"
        ),
    ):
        from victor.agent.coordinators import ChatContextProtocol as package_chat_context_protocol

    with pytest.warns(
        DeprecationWarning,
        match=(
            "victor.agent.coordinators.ChatOrchestratorProtocol is deprecated "
            "compatibility surface"
        ),
    ):
        from victor.agent.coordinators import (
            ChatOrchestratorProtocol as package_chat_orchestrator_protocol,
        )

    with pytest.warns(
        DeprecationWarning,
        match=(
            "victor.agent.coordinators.ProviderContextProtocol is deprecated "
            "compatibility surface"
        ),
    ):
        from victor.agent.coordinators import (
            ProviderContextProtocol as package_provider_context_protocol,
        )

    with pytest.warns(
        DeprecationWarning,
        match=(
            "victor.agent.coordinators.ToolContextProtocol is deprecated "
            "compatibility surface"
        ),
    ):
        from victor.agent.coordinators import ToolContextProtocol as package_tool_context_protocol

    assert package_chat_context_protocol is ChatContextProtocol
    assert package_chat_orchestrator_protocol is ChatOrchestratorProtocol
    assert package_provider_context_protocol is ProviderContextProtocol
    assert package_tool_context_protocol is ToolContextProtocol
