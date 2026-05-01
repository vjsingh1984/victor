"""Interop tests for SDK registry adapters."""

from victor.framework.multi_agent.persona_provider import FrameworkPersonaProvider
from victor.framework.team_registry import get_team_registry
from victor_sdk.registries import (
    get_default_persona_registry,
    get_default_team_registry,
)


def test_team_registry_sets_sdk_default() -> None:
    registry = get_team_registry()

    assert get_default_team_registry() is registry


def test_persona_provider_sets_sdk_default() -> None:
    provider = FrameworkPersonaProvider()

    assert get_default_persona_registry() is provider
