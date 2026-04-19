"""Tests for per-turn skill switching in multi-turn chat.

Covers:
- clear_active_skills() removes skill from system prompt
- inject_skill after clear replaces (not accumulates)
- update_system_prompt_for_query preserves active skill
- Multi-turn: different skills on different turns
- No skill on a turn clears previous
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pytest

from victor_sdk.skills import SkillDefinition


def _make_skill(name: str, **kwargs):
    return SkillDefinition(
        name=name,
        description=kwargs.get("description", f"Skill: {name}"),
        category=kwargs.get("category", "coding"),
        prompt_fragment=kwargs.get("prompt_fragment", f"Prompt for {name}."),
        required_tools=kwargs.get("required_tools", ["read_file"]),
        tags=kwargs.get("tags", frozenset()),
    )


def _make_orch_mock(kv_enabled=False):
    """Create a mock orchestrator with correct _kv_optimization_enabled behavior."""
    from victor.agent.orchestrator import AgentOrchestrator

    orch = MagicMock(spec=AgentOrchestrator)
    orch._system_prompt = "Base prompt."
    orch._base_system_prompt = "Base prompt."
    orch._active_skill_prompt = ""
    orch.conversation = None
    type(orch)._kv_optimization_enabled = PropertyMock(return_value=kv_enabled)
    return orch


class TestClearActiveSkills:
    """clear_active_skills() removes skill injection from system prompt."""

    def test_clear_removes_skill_prefix(self):
        from victor.agent.orchestrator import AgentOrchestrator

        orch = _make_orch_mock(kv_enabled=False)

        # Inject a skill (legacy path: mutates system prompt)
        AgentOrchestrator.inject_skill(orch, _make_skill("debug"))
        assert "ACTIVE SKILL: debug" in orch._system_prompt

        # Clear it
        AgentOrchestrator.clear_active_skills(orch)
        assert "ACTIVE SKILL" not in orch._system_prompt
        assert orch._system_prompt == "Base prompt."

    def test_clear_when_no_skill_is_noop(self):
        from victor.agent.orchestrator import AgentOrchestrator

        orch = _make_orch_mock(kv_enabled=False)

        AgentOrchestrator.clear_active_skills(orch)
        assert orch._system_prompt == "Base prompt."


class TestPerTurnSwitching:
    """Different skills on different turns without accumulation."""

    def test_inject_replaces_previous(self):
        from victor.agent.orchestrator import AgentOrchestrator

        orch = _make_orch_mock(kv_enabled=False)

        # Turn 1: debug skill
        AgentOrchestrator.inject_skill(orch, _make_skill("debug"))
        assert "debug" in orch._system_prompt

        # Turn 2: clear + refactor skill
        AgentOrchestrator.clear_active_skills(orch)
        AgentOrchestrator.inject_skill(orch, _make_skill("refactor"))

        assert "refactor" in orch._system_prompt
        assert "debug" not in orch._system_prompt
        # Only one ACTIVE SKILL header
        assert orch._system_prompt.count("ACTIVE SKILL") == 1

    def test_turn_with_no_skill_clears(self):
        from victor.agent.orchestrator import AgentOrchestrator

        orch = _make_orch_mock(kv_enabled=False)

        # Turn 1: skill active
        AgentOrchestrator.inject_skill(orch, _make_skill("debug"))
        assert "ACTIVE SKILL" in orch._system_prompt

        # Turn 2: no skill match → clear
        AgentOrchestrator.clear_active_skills(orch)
        assert "ACTIVE SKILL" not in orch._system_prompt
        assert orch._system_prompt == "Base prompt."

    def test_kv_enabled_stores_for_user_message(self):
        """When KV optimization is active, skill stored for user msg injection."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = _make_orch_mock(kv_enabled=True)

        AgentOrchestrator.inject_skill(orch, _make_skill("debug"))

        # System prompt NOT mutated (KV prefix preserved)
        assert orch._system_prompt == "Base prompt."
        # Skill stored for user message injection
        assert "debug" in orch._active_skill_prompt
