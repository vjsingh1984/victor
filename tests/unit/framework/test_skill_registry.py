"""Tests for SkillRegistry.

Covers:
- register / get / list_all
- search by query and category
- from_vertical class method
- from_entry_points with mocked entry points
- from_yaml_directory with temp YAML files
- duplicate registration handling
- KeyError on missing skill
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

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


class TestSkillRegistryFromYamlDirectory:
    """Load skills from YAML files in a directory."""

    def test_from_yaml_directory(self):
        from victor.framework.skills import SkillRegistry

        skill_yaml = {
            "name": "analyze_logs",
            "description": "Parse log files for errors",
            "category": "devops",
            "prompt_fragment": "Find and analyze log files.",
            "required_tools": ["read", "grep"],
            "optional_tools": ["shell"],
            "tags": ["logs", "debugging"],
            "max_tool_calls": 15,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "analyze_logs.yaml")
            with open(path, "w") as f:
                yaml.dump(skill_yaml, f)

            registry = SkillRegistry()
            registry.from_yaml_directory(tmpdir)
            skill = registry.get("analyze_logs")
            assert skill.name == "analyze_logs"
            assert skill.category == "devops"
            assert skill.required_tools == ["read", "grep"]
            assert "logs" in skill.tags

    def test_from_yaml_directory_multiple_files(self):
        from victor.framework.skills import SkillRegistry

        skills = [
            {
                "name": "skill_a",
                "description": "Skill A",
                "category": "general",
                "prompt_fragment": "Do A.",
                "required_tools": ["read"],
            },
            {
                "name": "skill_b",
                "description": "Skill B",
                "category": "general",
                "prompt_fragment": "Do B.",
                "required_tools": ["write"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for s in skills:
                path = os.path.join(tmpdir, f"{s['name']}.yaml")
                with open(path, "w") as f:
                    yaml.dump(s, f)

            registry = SkillRegistry()
            registry.from_yaml_directory(tmpdir)
            assert len(registry.list_all()) == 2
            assert registry.get("skill_a").description == "Skill A"
            assert registry.get("skill_b").description == "Skill B"

    def test_from_yaml_directory_skips_invalid(self):
        from victor.framework.skills import SkillRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write invalid YAML
            bad_path = os.path.join(tmpdir, "bad.yaml")
            with open(bad_path, "w") as f:
                f.write("not: valid: yaml: [[[")

            # Write non-YAML file
            txt_path = os.path.join(tmpdir, "readme.txt")
            with open(txt_path, "w") as f:
                f.write("ignore me")

            registry = SkillRegistry()
            registry.from_yaml_directory(tmpdir)
            assert registry.list_all() == []

    def test_from_yaml_directory_nonexistent(self):
        from victor.framework.skills import SkillRegistry

        registry = SkillRegistry()
        registry.from_yaml_directory("/nonexistent/path")
        assert registry.list_all() == []

    def test_from_yaml_yml_extension(self):
        from victor.framework.skills import SkillRegistry

        skill_yaml = {
            "name": "yml_skill",
            "description": "Skill from .yml file",
            "category": "general",
            "prompt_fragment": "Do something.",
            "required_tools": ["read"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "yml_skill.yml")
            with open(path, "w") as f:
                yaml.dump(skill_yaml, f)

            registry = SkillRegistry()
            registry.from_yaml_directory(tmpdir)
            assert registry.get("yml_skill").name == "yml_skill"
