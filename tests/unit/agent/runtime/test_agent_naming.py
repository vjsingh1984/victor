from unittest.mock import patch

from victor.agent.runtime.naming import (
    build_display_name,
    build_member_id,
    generate_agent_id,
)
from victor.agent.subagents import SubAgentRole


def test_build_display_name_uses_task_subject_and_role():
    assert (
        build_display_name(SubAgentRole.REVIEWER, task="Review Rust Arc usage and performance")
        == "Rust Arc Reviewer"
    )


def test_build_display_name_keeps_acronyms_readable():
    assert build_display_name("researcher", task="Find API endpoints") == "API Endpoints Researcher"


def test_build_display_name_can_include_ordinal_for_disambiguation():
    assert (
        build_display_name(SubAgentRole.RESEARCHER, task="Analyze schema migration", ordinal=2)
        == "Schema Migration Researcher 2"
    )


def test_build_member_id_uses_role_task_and_ordinal():
    assert (
        build_member_id(SubAgentRole.RESEARCHER, task="Find API endpoints", ordinal=1)
        == "api_endpoints_researcher_1"
    )


def test_generate_agent_id_uses_stable_prefix_and_role_slug():
    with patch("victor.agent.runtime.naming.uuid.uuid4") as uuid4:
        uuid4.return_value.hex = "abcdef1234567890"

        assert generate_agent_id(SubAgentRole.TESTER) == "agent_tester_abcdef123456"
