"""Tests for SkillMatcher — embedding-based skill auto-selection.

Covers:
- Initialization with skills
- High-confidence match (above high_threshold)
- Ambiguous match with edge LLM fallback
- No match below low_threshold
- Uninitialized returns None
- Sync wrapper
- Edge fallback disabled
- Edge fallback feature flag off
- Empty registry
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor_sdk.skills import SkillDefinition


def _make_skill(name: str = "debug", **kwargs):
    return SkillDefinition(
        name=name,
        description=kwargs.get("description", f"Skill: {name}"),
        category=kwargs.get("category", "coding"),
        prompt_fragment=kwargs.get("prompt_fragment", f"Prompt for {name}."),
        required_tools=kwargs.get("required_tools", ["read_file"]),
        tags=kwargs.get("tags", frozenset()),
    )


# A mock collection item matching StaticEmbeddingCollection's search return
class _MockItem:
    def __init__(self, id: str, text: str = ""):
        self.id = id
        self.text = text
        self.metadata = {}


class TestSkillMatcherInitialize:
    """SkillMatcher initializes embedding collection from registry."""

    @pytest.mark.asyncio
    async def test_initialize_embeds_skills(self):
        from victor.framework.skill_matcher import SkillMatcher

        registry = MagicMock()
        skills = [_make_skill("debug"), _make_skill("refactor")]
        registry.list_all.return_value = skills

        matcher = SkillMatcher()

        with patch.object(matcher, "_collection") as mock_coll:
            mock_coll.initialize = AsyncMock()
            await matcher.initialize(registry)

            mock_coll.initialize.assert_called_once()
            items = mock_coll.initialize.call_args[0][0]
            assert len(items) == 2
            assert items[0].id == "debug"
            assert items[1].id == "refactor"

        assert matcher._initialized is True
        assert "debug" in matcher._skills
        assert "refactor" in matcher._skills

    @pytest.mark.asyncio
    async def test_initialize_empty_registry(self):
        from victor.framework.skill_matcher import SkillMatcher

        registry = MagicMock()
        registry.list_all.return_value = []

        matcher = SkillMatcher()
        with patch.object(matcher, "_collection") as mock_coll:
            mock_coll.initialize = AsyncMock()
            await matcher.initialize(registry)

        assert matcher._initialized is True
        assert matcher._skills == {}


class TestSkillMatcherMatch:
    """SkillMatcher.match() returns best skill or None."""

    @pytest.mark.asyncio
    async def test_match_high_confidence(self):
        """Score > high_threshold returns skill directly, no edge LLM."""
        from victor.framework.skill_matcher import SkillMatcher

        skill = _make_skill("debug_test_failure", description="Debug a failing test")
        matcher = SkillMatcher(high_threshold=0.65, low_threshold=0.45)
        matcher._initialized = True
        matcher._skills = {"debug_test_failure": skill}

        mock_item = _MockItem("debug_test_failure")
        with patch.object(matcher, "_collection") as mock_coll:
            mock_coll.search = AsyncMock(return_value=[(mock_item, 0.82)])
            result = await matcher.match("fix the failing test in test_auth.py")

        assert result is not None
        matched_skill, score = result
        assert matched_skill.name == "debug_test_failure"
        assert score == 0.82

    @pytest.mark.asyncio
    async def test_match_no_match_below_threshold(self):
        """Score < low_threshold returns None."""
        from victor.framework.skill_matcher import SkillMatcher

        matcher = SkillMatcher(high_threshold=0.65, low_threshold=0.45)
        matcher._initialized = True
        matcher._skills = {"debug": _make_skill("debug")}

        with patch.object(matcher, "_collection") as mock_coll:
            mock_coll.search = AsyncMock(return_value=[])
            result = await matcher.match("hello world")

        assert result is None

    @pytest.mark.asyncio
    async def test_match_ambiguous_uses_edge_fallback(self):
        """Score in ambiguous zone triggers edge LLM when enabled."""
        from victor.framework.skill_matcher import SkillMatcher

        debug_skill = _make_skill("debug_test_failure")
        review_skill = _make_skill("code_review")

        matcher = SkillMatcher(high_threshold=0.65, low_threshold=0.45, use_edge_fallback=True)
        matcher._initialized = True
        matcher._skills = {
            "debug_test_failure": debug_skill,
            "code_review": review_skill,
        }

        mock_item_1 = _MockItem("debug_test_failure")
        mock_item_2 = _MockItem("code_review")

        with patch.object(matcher, "_collection") as mock_coll:
            # Score in ambiguous zone (0.45-0.65)
            mock_coll.search = AsyncMock(return_value=[(mock_item_1, 0.55), (mock_item_2, 0.50)])
            with patch.object(
                matcher, "_edge_llm_decide", return_value=(debug_skill, 0.80)
            ) as mock_edge:
                result = await matcher.match("something about tests")

                mock_edge.assert_called_once()

        assert result is not None
        assert result[0].name == "debug_test_failure"
        assert result[1] == 0.80  # Edge LLM confidence

    @pytest.mark.asyncio
    async def test_match_ambiguous_no_edge_fallback(self):
        """When edge fallback disabled, uses embedding top-1 for ambiguous."""
        from victor.framework.skill_matcher import SkillMatcher

        skill = _make_skill("debug")
        matcher = SkillMatcher(high_threshold=0.65, low_threshold=0.45, use_edge_fallback=False)
        matcher._initialized = True
        matcher._skills = {"debug": skill}

        mock_item = _MockItem("debug")
        with patch.object(matcher, "_collection") as mock_coll:
            mock_coll.search = AsyncMock(return_value=[(mock_item, 0.52)])
            result = await matcher.match("something vague")

        assert result is not None
        assert result[0].name == "debug"
        assert result[1] == 0.52  # Embedding score, not edge

    @pytest.mark.asyncio
    async def test_match_uninitialized_returns_none(self):
        from victor.framework.skill_matcher import SkillMatcher

        matcher = SkillMatcher()
        result = await matcher.match("anything")
        assert result is None


class TestSkillMatcherSync:
    """match_sync() wraps the async match."""

    def test_match_sync_returns_none_uninitialized(self):
        from victor.framework.skill_matcher import SkillMatcher

        matcher = SkillMatcher()
        result = matcher.match_sync("anything")
        assert result is None

    def test_match_sync_delegates(self):
        from victor.framework.skill_matcher import SkillMatcher

        skill = _make_skill("debug")
        matcher = SkillMatcher()
        matcher._initialized = True
        matcher._skills = {"debug": skill}

        mock_item = _MockItem("debug")
        with patch.object(matcher, "_collection") as mock_coll:
            mock_coll.search_sync.return_value = [(mock_item, 0.82)]
            result = matcher.match_sync("fix test")

        assert result is not None
        assert result[0].name == "debug"


class TestSkillMatcherEdgeLLM:
    """Edge LLM fallback tests."""

    def test_edge_decides_skill(self):
        from victor.framework.skill_matcher import SkillMatcher

        debug_skill = _make_skill("debug_test_failure")
        matcher = SkillMatcher(use_edge_fallback=True)
        matcher._skills = {"debug_test_failure": debug_skill}

        candidates = [(_MockItem("debug_test_failure"), 0.55)]

        with patch("victor.framework.skill_matcher.FeatureFlag") as mock_ff:
            mock_ff.USE_EDGE_MODEL.is_enabled.return_value = True
            with patch("victor.framework.skill_matcher.decide_sync") as mock_decide:
                mock_decide.return_value = "debug_test_failure"
                result = matcher._edge_llm_decide("fix the test", candidates)

        assert result is not None
        assert result[0].name == "debug_test_failure"

    def test_edge_returns_none_when_flag_off(self):
        from victor.framework.skill_matcher import SkillMatcher

        matcher = SkillMatcher(use_edge_fallback=True)
        matcher._skills = {"debug": _make_skill("debug")}

        with patch("victor.framework.skill_matcher.FeatureFlag") as mock_ff:
            mock_ff.USE_EDGE_MODEL.is_enabled.return_value = False
            result = matcher._edge_llm_decide("fix test", [(_MockItem("debug"), 0.55)])

        assert result is None

    def test_edge_returns_none_on_exception(self):
        from victor.framework.skill_matcher import SkillMatcher

        matcher = SkillMatcher(use_edge_fallback=True)
        matcher._skills = {"debug": _make_skill("debug")}

        with patch("victor.framework.skill_matcher.FeatureFlag") as mock_ff:
            mock_ff.USE_EDGE_MODEL.is_enabled.return_value = True
            with patch(
                "victor.framework.skill_matcher.decide_sync",
                side_effect=Exception("ollama down"),
            ):
                result = matcher._edge_llm_decide("fix test", [(_MockItem("debug"), 0.55)])

        assert result is None
