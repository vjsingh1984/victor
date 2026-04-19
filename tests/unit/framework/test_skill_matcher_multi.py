"""Tests for multi-skill matching and inject_skills.

Covers:
- match_multiple_sync: single dominant, multiple viable, phase ordering
- inject_skills (plural): compose 2, cap at 3, arrow notation
- Phase ordering: diagnostic before action before verification
- Backward compat: existing match_sync still works
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor_sdk.skills import SkillDefinition


def _make_skill(name: str, phase: str = "action", **kwargs):
    return SkillDefinition(
        name=name,
        description=kwargs.get("description", f"Skill: {name}"),
        category=kwargs.get("category", "coding"),
        prompt_fragment=kwargs.get("prompt_fragment", f"Prompt for {name}."),
        required_tools=kwargs.get("required_tools", ["read_file"]),
        tags=kwargs.get("tags", frozenset()),
        phase=phase,
    )


class _MockItem:
    def __init__(self, id: str):
        self.id = id
        self.text = ""
        self.metadata = {}


class TestMatchMultipleSync:
    """SkillMatcher.match_multiple_sync returns ordered skill list."""

    def test_single_dominant_returns_one(self):
        from victor.framework.skill_matcher import SkillMatcher

        skill = _make_skill("debug", phase="diagnostic")
        matcher = SkillMatcher(high_threshold=0.65, low_threshold=0.45)
        matcher._initialized = True
        matcher._skills = {"debug": skill}

        with patch.object(matcher, "_collection") as mock_coll:
            mock_coll.search_sync.return_value = [(_MockItem("debug"), 0.85)]
            result = matcher.match_multiple_sync("fix test")

        assert len(result) == 1
        assert result[0][0].name == "debug"

    def test_multiple_viable_returns_phase_ordered(self):
        from victor.framework.skill_matcher import SkillMatcher

        debug = _make_skill("debug", phase="diagnostic")
        refactor = _make_skill("refactor", phase="action")
        review = _make_skill("review", phase="verification")

        matcher = SkillMatcher(high_threshold=0.65, low_threshold=0.45)
        matcher._initialized = True
        matcher._skills = {
            "debug": debug,
            "refactor": refactor,
            "review": review,
        }

        with patch.object(matcher, "_collection") as mock_coll:
            # All above threshold, action scored highest but diagnostic should come first
            mock_coll.search_sync.return_value = [
                (_MockItem("refactor"), 0.70),
                (_MockItem("debug"), 0.68),
                (_MockItem("review"), 0.55),
            ]
            result = matcher.match_multiple_sync("debug and refactor")

        assert len(result) == 3
        # Phase order: diagnostic → action → verification
        assert result[0][0].name == "debug"
        assert result[1][0].name == "refactor"
        assert result[2][0].name == "review"

    def test_cap_at_max_skills(self):
        from victor.framework.skill_matcher import SkillMatcher

        skills = {f"s{i}": _make_skill(f"s{i}") for i in range(5)}
        matcher = SkillMatcher(high_threshold=0.65, low_threshold=0.45)
        matcher._initialized = True
        matcher._skills = skills

        with patch.object(matcher, "_collection") as mock_coll:
            mock_coll.search_sync.return_value = [(_MockItem(f"s{i}"), 0.60) for i in range(5)]
            result = matcher.match_multiple_sync("do everything", max_skills=3)

        assert len(result) <= 3

    def test_no_match_returns_empty(self):
        from victor.framework.skill_matcher import SkillMatcher

        matcher = SkillMatcher()
        matcher._initialized = True
        matcher._skills = {"debug": _make_skill("debug")}

        with patch.object(matcher, "_collection") as mock_coll:
            mock_coll.search_sync.return_value = []
            result = matcher.match_multiple_sync("hello world")

        assert result == []

    def test_uninitialized_returns_empty(self):
        from victor.framework.skill_matcher import SkillMatcher

        matcher = SkillMatcher()
        assert matcher.match_multiple_sync("anything") == []

    def test_backward_compat_match_sync_still_works(self):
        """Existing match_sync is unaffected."""
        from victor.framework.skill_matcher import SkillMatcher

        skill = _make_skill("debug")
        matcher = SkillMatcher()
        matcher._initialized = True
        matcher._skills = {"debug": skill}

        with patch.object(matcher, "_collection") as mock_coll:
            mock_coll.search_sync.return_value = [(_MockItem("debug"), 0.82)]
            result = matcher.match_sync("fix test")

        assert result is not None
        assert result[0].name == "debug"


class TestPhaseOrdering:
    """Skills ordered by phase: diagnostic → action → verification → documentation."""

    def test_phase_order_constant(self):
        from victor.framework.skill_matcher import PHASE_ORDER

        assert PHASE_ORDER["diagnostic"] < PHASE_ORDER["action"]
        assert PHASE_ORDER["action"] < PHASE_ORDER["verification"]
        assert PHASE_ORDER["verification"] < PHASE_ORDER["documentation"]


class TestInjectSkillsPlural:
    """Orchestrator.inject_skills() composes multiple skill prompts."""

    def test_inject_two_skills(self):
        from unittest.mock import PropertyMock
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch._system_prompt = "Base prompt."
        orch._cache_optimization_enabled = False
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=False)
        orch.conversation = None

        skill_a = _make_skill("debug", phase="diagnostic")
        skill_b = _make_skill("refactor", phase="action")

        AgentOrchestrator.inject_skills(orch, [(skill_a, 0.80), (skill_b, 0.70)])

        prompt = orch._system_prompt
        assert "ACTIVE SKILLS (2)" in prompt
        assert "debug" in prompt
        assert "refactor" in prompt
        assert "debug" in prompt[: prompt.index("refactor")]

    def test_inject_caps_at_three(self):
        from unittest.mock import PropertyMock
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch._system_prompt = "Base."
        orch._cache_optimization_enabled = False
        type(orch)._kv_optimization_enabled = PropertyMock(return_value=False)
        orch.conversation = None

        skills = [(_make_skill(f"s{i}"), 0.70) for i in range(5)]
        AgentOrchestrator.inject_skills(orch, skills)

        prompt = orch._system_prompt
        assert "ACTIVE SKILLS (3)" in prompt

    def test_inject_empty_is_noop(self):
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        orch._system_prompt = "Base."

        AgentOrchestrator.inject_skills(orch, [])
        assert orch._system_prompt == "Base."


class TestSkillDefinitionPhase:
    """SkillDefinition.phase field tests."""

    def test_default_phase_is_action(self):
        skill = SkillDefinition(
            name="x",
            description="x",
            category="x",
            prompt_fragment="x",
            required_tools=[],
        )
        assert skill.phase == "action"

    def test_phase_in_to_dict(self):
        skill = _make_skill("debug", phase="diagnostic")
        assert skill.to_dict()["phase"] == "diagnostic"

    def test_from_dict_without_phase_defaults(self):
        data = {
            "name": "x",
            "description": "x",
            "category": "x",
            "prompt_fragment": "x",
            "required_tools": [],
        }
        skill = SkillDefinition.from_dict(data)
        assert skill.phase == "action"

    def test_from_dict_with_phase(self):
        data = {
            "name": "x",
            "description": "x",
            "category": "x",
            "prompt_fragment": "x",
            "required_tools": [],
            "phase": "verification",
        }
        skill = SkillDefinition.from_dict(data)
        assert skill.phase == "verification"
