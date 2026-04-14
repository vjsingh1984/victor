"""Tests for SDK registry adapters."""

from victor_sdk.registries import (
    get_default_persona_registry,
    get_default_team_registry,
    set_default_persona_registry,
    set_default_team_registry,
)


def test_registry_setters_round_trip_defaults() -> None:
    team_registry = object()
    persona_registry = object()

    set_default_team_registry(team_registry)
    set_default_persona_registry(persona_registry)

    assert get_default_team_registry() is team_registry
    assert get_default_persona_registry() is persona_registry

    set_default_team_registry(None)
    set_default_persona_registry(None)
