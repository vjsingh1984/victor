"""Unit tests for SDK team metadata declarations."""

import pytest

from victor_sdk import TeamMetadata, VerticalDefinition
from victor_sdk.core.exceptions import VerticalConfigurationError
from victor_sdk.verticals.protocols.base import VerticalBase


def test_vertical_definition_serializes_team_metadata_round_trip() -> None:
    """Team metadata should serialize, normalize, and round-trip cleanly."""

    definition = VerticalDefinition(
        name="security",
        description="Security vertical",
        tools=["read"],
        system_prompt="Inspect carefully.",
        team_metadata={
            "teams": {
                "security_review_team": {
                    "name": "Security Review Team",
                    "description": "Analyze then validate findings.",
                    "formation": "pipeline",
                    "members": [
                        {
                            "role": "researcher",
                            "goal": "Identify likely security risks.",
                            "name": "Security Analyst",
                            "tool_budget": 12,
                            "memory": True,
                        },
                        {
                            "role": "reviewer",
                            "goal": "Validate findings.",
                            "name": "Validation Reviewer",
                            "tool_budget": 8,
                        },
                    ],
                }
            },
            "default_team": "security_review_team",
        },
    )

    payload = definition.to_dict()
    config = definition.to_config()
    round_trip = VerticalDefinition.from_dict(payload)
    from_config = VerticalDefinition.from_config(config)

    assert definition.team_metadata.teams[0].team_id == "security_review_team"
    assert definition.team_metadata.default_team == "security_review_team"
    assert payload["team_metadata"]["teams"][0]["team_id"] == "security_review_team"
    assert config.extensions["team_metadata"]["default_team"] == "security_review_team"
    assert round_trip.team_metadata.teams[0].members[0].name == "Security Analyst"
    assert from_config.team_metadata.default_team == "security_review_team"


def test_team_metadata_default_team_must_reference_declared_team() -> None:
    """Definitions should reject dangling default-team references."""

    with pytest.raises(VerticalConfigurationError, match="default_team"):
        VerticalDefinition(
            name="invalid",
            description="Invalid team metadata",
            tools=["read"],
            system_prompt="Invalid",
            team_metadata=TeamMetadata(default_team="missing_team"),
        )


def test_sdk_vertical_base_exposes_team_metadata_hooks() -> None:
    """SDK base should derive team metadata from declarative team hooks."""

    class ExampleVertical(VerticalBase):
        @classmethod
        def get_name(cls) -> str:
            return "example"

        @classmethod
        def get_description(cls) -> str:
            return "Example"

        @classmethod
        def get_tools(cls) -> list[str]:
            return ["read"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "Example prompt"

        @classmethod
        def get_team_declarations(cls) -> dict[str, dict[str, object]]:
            return {
                "review_team": {
                    "name": "Review Team",
                    "formation": "sequential",
                    "members": [
                        {
                            "role": "researcher",
                            "goal": "Inspect the target carefully.",
                        }
                    ],
                }
            }

        @classmethod
        def get_default_team(cls) -> str:
            return "review_team"

    definition = ExampleVertical.get_definition()

    assert definition.team_metadata.default_team == "review_team"
    assert definition.team_metadata.teams[0].team_id == "review_team"
    assert definition.team_metadata.teams[0].members[0].role == "researcher"
