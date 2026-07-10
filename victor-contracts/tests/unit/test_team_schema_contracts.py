"""Tests for SDK-owned declarative team schema contracts."""

import pytest

from victor_contracts.team_schema import (
    RoleConfig,
    TeamAgentCategory,
    TeamFormation,
    TeamMemberSpec,
    TeamSpec,
    get_runtime_team_registry,
)


def test_role_config_canonicalizes_tool_names() -> None:
    config = RoleConfig(
        base_role="researcher",
        tools=["read_file", "code_search", "run_tests"],
        tool_budget=10,
    )

    assert config.tools == ["read", "code_search", "test"]


def test_team_spec_round_trips_from_dict() -> None:
    spec = TeamSpec(
        name="Review Team",
        description="Reviews code",
        vertical="coding",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="reviewer",
                goal="Review the changes",
                allowed_tools=["read_file", "git_diff"],
                memory=True,
            )
        ],
        tags=["review"],
        task_types=["code_review"],
    )

    restored = TeamSpec.from_dict(spec.to_dict())

    assert restored.name == "Review Team"
    assert restored.formation is TeamFormation.PIPELINE
    assert restored.members[0].allowed_tools == ["read", "git"]
    assert restored.task_types == ["code_review"]


def test_team_member_spec_uses_supervisor_category_as_manager_contract() -> None:
    member = TeamMemberSpec(
        role="planner",
        goal="Coordinate specialist agents",
        agent_category=TeamAgentCategory.SUPERVISOR,
    )

    assert member.is_supervisor is True
    assert member.is_manager is True
    assert member.can_delegate is True
    assert member.max_delegation_depth == 1


def test_team_member_spec_reads_legacy_agent_class_alias() -> None:
    restored = TeamSpec.from_dict(
        {
            "name": "Legacy Team",
            "description": "Uses the old field name",
            "vertical": "coding",
            "formation": "hierarchical",
            "members": [
                {
                    "role": "planner",
                    "goal": "Coordinate",
                    "agent_class": "supervisor",
                },
                {
                    "role": "executor",
                    "goal": "Execute",
                },
            ],
        }
    )

    assert restored.members[0].agent_category is TeamAgentCategory.SUPERVISOR
    assert restored.members[0].is_supervisor is True


def test_team_member_spec_bridges_to_runtime_member() -> None:
    pytest.importorskip("victor", reason="to_team_member() requires the victor-ai package")
    member = TeamMemberSpec(
        role="researcher",
        goal="Find the relevant code",
        allowed_tools=["read_file", "grep"],
        memory=True,
    ).to_team_member(index=2)

    assert member.role.value == "researcher"
    assert member.allowed_tools == ["read", "grep"]
    assert member.priority == 2
    assert member.memory is True


def test_team_schema_exposes_runtime_registry_adapter() -> None:
    assert callable(get_runtime_team_registry)
