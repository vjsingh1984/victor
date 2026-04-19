"""Interop tests for SDK-owned multi-agent models."""

from victor.framework.multi_agent.personas import (
    CommunicationStyle as CoreCommunicationStyle,
    PersonaTraits as CorePersonaTraits,
)
from victor.framework.multi_agent.teams import (
    TeamTemplate as CoreTeamTemplate,
    TeamTopology as CoreTeamTopology,
)
from victor_sdk.multi_agent import (
    CommunicationStyle as SdkCommunicationStyle,
    PersonaTraits as SdkPersonaTraits,
    TeamTemplate as SdkTeamTemplate,
    TeamTopology as SdkTeamTopology,
)


def test_persona_contract_identity_is_shared() -> None:
    assert CoreCommunicationStyle is SdkCommunicationStyle
    assert CorePersonaTraits is SdkPersonaTraits


def test_team_contract_identity_is_shared() -> None:
    assert CoreTeamTemplate is SdkTeamTemplate
    assert CoreTeamTopology is SdkTeamTopology
