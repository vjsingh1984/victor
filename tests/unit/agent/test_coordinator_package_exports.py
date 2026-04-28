"""Tests for deprecated package-root coordinator exports."""

import pytest

from victor.agent.services.chat_compat_telemetry import (
    get_deprecated_chat_shim_telemetry,
    reset_deprecated_chat_shim_telemetry,
)


@pytest.fixture(autouse=True)
def _reset_chat_compat_telemetry():
    reset_deprecated_chat_shim_telemetry()
    yield
    reset_deprecated_chat_shim_telemetry()


def test_session_package_exports_warn():
    """Package-root session coordinator exports should be compatibility-only."""
    from victor.agent.services.session_compat import (
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
    from victor.agent.services.sync_chat_compat import SyncChatCoordinator
    from victor.agent.services.streaming_chat_compat import StreamingChatCoordinator
    from victor.agent.services.unified_chat_compat import UnifiedChatCoordinator

    with pytest.warns(
        DeprecationWarning,
        match=(
            "victor.agent.coordinators.SyncChatCoordinator is deprecated " "compatibility surface"
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
    telemetry = get_deprecated_chat_shim_telemetry()
    assert telemetry["coordinators_package.SyncChatCoordinator.package_export"] >= 1
    assert telemetry["coordinators_package.StreamingChatCoordinator.package_export"] >= 1
    assert telemetry["coordinators_package.UnifiedChatCoordinator.package_export"] >= 1


def test_chat_coordinator_package_export_warns_and_records_telemetry():
    from victor.agent.services.chat_compat import ChatCoordinator

    with pytest.warns(
        DeprecationWarning,
        match="victor.agent.coordinators.ChatCoordinator is deprecated compatibility surface",
    ):
        from victor.agent.coordinators import ChatCoordinator as package_chat_coordinator

    assert package_chat_coordinator is ChatCoordinator
    telemetry = get_deprecated_chat_shim_telemetry()
    assert telemetry["coordinators_package.ChatCoordinator.package_export"] >= 1


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
            "victor.agent.coordinators.ChatContextProtocol is deprecated " "compatibility surface"
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
            "victor.agent.coordinators.ToolContextProtocol is deprecated " "compatibility surface"
        ),
    ):
        from victor.agent.coordinators import ToolContextProtocol as package_tool_context_protocol

    assert package_chat_context_protocol is ChatContextProtocol
    assert package_chat_orchestrator_protocol is ChatOrchestratorProtocol
    assert package_provider_context_protocol is ProviderContextProtocol
    assert package_tool_context_protocol is ToolContextProtocol


def test_state_passed_package_exports_are_first_class():
    """State-passed exports should be available from the package root without warnings."""
    from victor.agent.coordinators.exploration_state_passed import (
        ExplorationStatePassedCoordinator,
    )
    from victor.agent.coordinators.safety_state_passed import SafetyStatePassedCoordinator
    from victor.agent.coordinators.system_prompt_state_passed import (
        SystemPromptStatePassedCoordinator,
    )
    from victor.agent.coordinators import (
        ExplorationStatePassedCoordinator as package_exploration_state_passed,
        SafetyStatePassedCoordinator as package_safety_state_passed,
        SystemPromptStatePassedCoordinator as package_system_prompt_state_passed,
    )

    assert package_exploration_state_passed is ExplorationStatePassedCoordinator
    assert package_system_prompt_state_passed is SystemPromptStatePassedCoordinator
    assert package_safety_state_passed is SafetyStatePassedCoordinator


def test_exploration_runtime_package_exports_are_first_class():
    """Canonical exploration runtime exports should be available from the package root."""
    from victor.agent.services.exploration_runtime import (
        ExplorationCoordinator,
        ExplorationResult,
    )
    from victor.agent.coordinators import (
        ExplorationCoordinator as package_exploration_coordinator,
        ExplorationResult as package_exploration_result,
    )

    assert package_exploration_coordinator is ExplorationCoordinator
    assert package_exploration_result is ExplorationResult
