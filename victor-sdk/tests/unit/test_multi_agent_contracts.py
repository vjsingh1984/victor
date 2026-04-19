"""Tests for SDK-owned multi-agent contracts."""

import pytest

from victor_sdk.multi_agent import (
    CommunicationStyle,
    PersonaTemplate,
    PersonaTraits,
    TaskAssignmentStrategy,
    TeamSpec,
    TeamTemplate,
    TeamTopology,
)


def test_persona_template_creates_overridden_persona() -> None:
    template = PersonaTemplate(
        base_traits=PersonaTraits(
            name="Base",
            role="reviewer",
            description="Base persona",
            communication_style=CommunicationStyle.TECHNICAL,
        ),
        overrides={"verbosity": 0.8},
    )

    created = template.create(name="Derived")

    assert created.name == "Derived"
    assert created.verbosity == 0.8


def test_team_template_validates_threshold() -> None:
    with pytest.raises(ValueError):
        TeamTemplate(
            name="Bad",
            description="Bad config",
            topology=TeamTopology.PIPELINE,
            assignment_strategy=TaskAssignmentStrategy.SKILL_MATCH,
            escalation_threshold=1.5,
        )


def test_team_spec_reports_missing_slots() -> None:
    spec = TeamSpec(
        template=TeamTemplate(
            name="Review",
            description="Review team",
            member_slots={"reviewer": 2},
        ),
        members=[],
    )

    assert spec.validate_slots() == ["Role 'reviewer' requires 2 members, got 0"]
