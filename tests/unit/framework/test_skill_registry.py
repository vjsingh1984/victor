"""Tests for SkillRegistry.

Covers:
- register / get / list_all
- search by query and category
- from_vertical class method
- from_entry_points with mocked entry points
- duplicate registration handling
- KeyError on missing skill
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from victor_sdk.skills import SkillDefinition


def _make_skill(name: str = "debug", category: str = "coding", **kwargs):
    """Helper to build a SkillDefinition with defaults."""
    return SkillDefinition(
        name=name,
        description=kwargs.get("description", f"Skill: {name}"),
        category=category,
        prompt_fragment=kwargs.get("prompt_fragment", f"Prompt for {name}."),
        required_tools=kwargs.get("required_tools", ["read_file"]),
        optional_tools=kwargs.get("optional_tools", []),
        tags=kwargs.get("tags", frozenset()),
    )


class TestSkillRegistryRegisterAndGet:
    """Register and retrieve skills by name."""

    def test_register_and_get(self):
        from victor.framework.skills import SkillRegistry

        registry = SkillRegistry()
        skill = _make_skill("debug")
        registry.register(skill)
        assert registry.get("debug") is skill

    def test_get_missing_raises_key_error(self):
        from victor.framework.skills import SkillRegistry

        registry = SkillRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            registry.get("nonexistent")

    def test_duplicate_register_overwrites(self):
        from victor.framework.skills import SkillRegistry

        registry = SkillRegistry()
        skill_v1 = _make_skill("debug", description="v1")
        skill_v2 = _make_skill("debug", description="v2")
        registry.register(skill_v1)
        registry.register(skill_v2)
        assert registry.get("debug").description == "v2"

    def test_get_optional_returns_none(self):
        from victor.framework.skills import SkillRegistry

        registry = SkillRegistry()
        assert registry.get_optional("missing") is None


class TestSkillRegistryListAll:
    """List all registered skills."""

    def test_list_all_empty(self):
        from victor.framework.skills import SkillRegistry

        registry = SkillRegistry()
        assert registry.list_all() == []

    def test_list_all(self):
        from victor.framework.skills import SkillRegistry

        registry = SkillRegistry()
        registry.register(_make_skill("debug"))
        registry.register(_make_skill("refactor"))
        skills = registry.list_all()
        names = {s.name for s in skills}
        assert names == {"debug", "refactor"}


class TestSkillRegistrySearch:
    """Search skills by query string and/or category."""

    def test_search_by_category(self):
        from victor.framework.skills import SkillRegistry

        registry = SkillRegistry()
        registry.register(_make_skill("debug", category="coding"))
        registry.register(_make_skill("deploy", category="devops"))
        results = registry.search(category="coding")
        assert len(results) == 1
        assert results[0].name == "debug"

    def test_search_by_query(self):
        from victor.framework.skills import SkillRegistry

        registry = SkillRegistry()
        registry.register(_make_skill("debug_test_failure", description="Debug a failing test"))
        registry.register(_make_skill("deploy_service", description="Deploy a service"))
        results = registry.search(query="debug")
        assert len(results) == 1
        assert results[0].name == "debug_test_failure"

    def test_search_by_query_and_category(self):
        from victor.framework.skills import SkillRegistry

        registry = SkillRegistry()
        registry.register(_make_skill("debug", category="coding", description="Debug code"))
        registry.register(_make_skill("debug_infra", category="devops", description="Debug infra"))
        results = registry.search(query="debug", category="coding")
        assert len(results) == 1
        assert results[0].name == "debug"

    def test_search_matches_tags(self):
        from victor.framework.skills import SkillRegistry

        registry = SkillRegistry()
        registry.register(_make_skill("refactor", tags=frozenset({"code-quality", "cleanup"})))
        registry.register(_make_skill("deploy"))
        results = registry.search(query="cleanup")
        assert len(results) == 1
        assert results[0].name == "refactor"

    def test_search_no_results(self):
        from victor.framework.skills import SkillRegistry

        registry = SkillRegistry()
        registry.register(_make_skill("debug"))
        assert registry.search(query="nonexistent") == []


class TestSkillRegistryFromVertical:
    """Load skills from a vertical class."""

    def test_from_vertical(self):
        from victor.framework.skills import SkillRegistry
        from victor_sdk.verticals.protocols.base import VerticalBase

        debug_skill = _make_skill("debug")
        refactor_skill = _make_skill("refactor")

        class CodingVertical(VerticalBase):
            name = "coding"
            description = "Coding assistant"

            @classmethod
            def get_name(cls) -> str:
                return "coding"

            @classmethod
            def get_description(cls) -> str:
                return "Coding assistant"

            @classmethod
            def get_tools(cls) -> list:
                return ["read_file", "edit_file"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "You are a coding assistant."

            @classmethod
            def get_skills(cls) -> list:
                return [debug_skill, refactor_skill]

        registry = SkillRegistry()
        registry.from_vertical(CodingVertical)
        assert registry.get("debug") is debug_skill
        assert registry.get("refactor") is refactor_skill

    def test_from_vertical_no_skills(self):
        from victor.framework.skills import SkillRegistry
        from victor_sdk.verticals.protocols.base import VerticalBase

        class EmptyVertical(VerticalBase):
            name = "empty"
            description = "Empty"

            @classmethod
            def get_name(cls) -> str:
                return "empty"

            @classmethod
            def get_description(cls) -> str:
                return "Empty"

            @classmethod
            def get_tools(cls) -> list:
                return []

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Empty."

        registry = SkillRegistry()
        registry.from_vertical(EmptyVertical)
        assert registry.list_all() == []


class TestSkillRegistryFromEntryPoints:
    """Load skills from victor.skills entry points."""

    def test_from_entry_points(self):
        from victor.framework.skills import SkillRegistry

        skill = _make_skill("external_debug")

        mock_ep = MagicMock()
        mock_ep.name = "external_debug"
        mock_ep.load.return_value = skill

        with patch(
            "victor.framework.skills.entry_points",
            return_value=[mock_ep],
        ):
            registry = SkillRegistry()
            registry.from_entry_points()
            assert registry.get("external_debug") is skill

    def test_from_entry_points_bad_load_skipped(self):
        from victor.framework.skills import SkillRegistry

        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = ImportError("broken")

        with patch(
            "victor.framework.skills.entry_points",
            return_value=[mock_ep],
        ):
            registry = SkillRegistry()
            registry.from_entry_points()
            assert registry.list_all() == []
