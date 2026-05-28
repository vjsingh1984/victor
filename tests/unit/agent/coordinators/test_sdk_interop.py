"""Interop tests for SDK-owned coordinator contracts."""

from __future__ import annotations

import importlib

import pytest

from victor_contracts.conversation import (
    ConversationCoordinator as SdkConversationCoordinator,
)
from victor_contracts.conversation import TurnType as SdkTurnType
from victor_contracts.safety import SafetyCoordinator as SdkSafetyCoordinator
from victor_contracts.safety import SafetyRule as SdkSafetyRule


def _reload_core_coordinators():
    module = importlib.import_module("victor.agent.coordinators")
    return importlib.reload(module)


def test_safety_coordinator_contract_identity_is_shared() -> None:
    coordinators = _reload_core_coordinators()

    with pytest.warns(
        DeprecationWarning,
        match=r"victor\.agent\.coordinators\.SafetyCoordinator is deprecated SDK compatibility surface",
    ):
        core_safety_coordinator = coordinators.SafetyCoordinator

    with pytest.warns(
        DeprecationWarning,
        match=r"victor\.agent\.coordinators\.SafetyRule is deprecated SDK compatibility surface",
    ):
        core_safety_rule = coordinators.SafetyRule

    assert core_safety_coordinator is SdkSafetyCoordinator
    assert core_safety_rule is SdkSafetyRule


def test_conversation_coordinator_contract_identity_is_shared() -> None:
    coordinators = _reload_core_coordinators()

    with pytest.warns(
        DeprecationWarning,
        match=(
            r"victor\.agent\.coordinators\.ConversationCoordinator is deprecated "
            r"SDK compatibility surface"
        ),
    ):
        core_conversation_coordinator = coordinators.ConversationCoordinator

    with pytest.warns(
        DeprecationWarning,
        match=r"victor\.agent\.coordinators\.TurnType is deprecated SDK compatibility surface",
    ):
        core_turn_type = coordinators.TurnType

    assert core_conversation_coordinator is SdkConversationCoordinator
    assert core_turn_type is SdkTurnType
