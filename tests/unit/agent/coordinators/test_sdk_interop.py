"""Interop tests for SDK-owned coordinator contracts."""

from victor.agent.coordinators.conversation_coordinator import (
    ConversationCoordinator as CoreConversationCoordinator,
    TurnType as CoreTurnType,
)
from victor.agent.coordinators.safety_coordinator import (
    SafetyCoordinator as CoreSafetyCoordinator,
    SafetyRule as CoreSafetyRule,
)
from victor_sdk.conversation import ConversationCoordinator as SdkConversationCoordinator
from victor_sdk.conversation import TurnType as SdkTurnType
from victor_sdk.safety import SafetyCoordinator as SdkSafetyCoordinator
from victor_sdk.safety import SafetyRule as SdkSafetyRule


def test_safety_coordinator_contract_identity_is_shared() -> None:
    assert CoreSafetyCoordinator is SdkSafetyCoordinator
    assert CoreSafetyRule is SdkSafetyRule


def test_conversation_coordinator_contract_identity_is_shared() -> None:
    assert CoreConversationCoordinator is SdkConversationCoordinator
    assert CoreTurnType is SdkTurnType
