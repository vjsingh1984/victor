"""Tests for SkillDefinition SDK contract.

Covers:
- Construction and frozen immutability
- Serialization (to_dict / from_dict)
- Validation (empty name, missing fields)
- Tool accessors
- SkillProvider protocol conformance
- VerticalBase.get_skills() default hook
"""

from __future__ import annotations

import pytest


class TestSkillDefinitionConstruction:
    """SkillDefinition can be created with required and optional fields."""

    def test_minimal_construction(self):
        from victor_sdk.skills import SkillDefinition

        skill = SkillDefinition(
            name="debug",
            description="Debug a failing test",
            category="coding",
            prompt_fragment="You are debugging a test failure.",
            required_tools=["read_file", "grep", "execute_bash"],
        )
        assert skill.name == "debug"
        assert skill.description == "Debug a failing test"
        assert skill.category == "coding"
        assert skill.required_tools == ["read_file", "grep", "execute_bash"]
        assert skill.optional_tools == []
        assert skill.tags == frozenset()
        assert skill.max_tool_calls == 20
        assert skill.version == "1.0.0"

    def test_full_construction(self):
        from victor_sdk.skills import SkillDefinition

        skill = SkillDefinition(
            name="refactor_extract",
            description="Extract method refactoring",
            category="coding",
            prompt_fragment="Extract the selected code into a new method.",
            required_tools=["read_file", "edit_file"],
            optional_tools=["grep", "code_search"],
            tags=frozenset({"refactoring", "code-quality"}),
            max_tool_calls=30,
            version="2.0.0",
        )
        assert skill.optional_tools == ["grep", "code_search"]
        assert "refactoring" in skill.tags
        assert skill.max_tool_calls == 30
        assert skill.version == "2.0.0"

    def test_frozen_immutability(self):
        from victor_sdk.skills import SkillDefinition

        skill = SkillDefinition(
            name="debug",
            description="Debug",
            category="coding",
            prompt_fragment="Debug.",
            required_tools=["read_file"],
        )
        with pytest.raises(AttributeError):
            skill.name = "changed"  # type: ignore[misc]


class TestSkillDefinitionSerialization:
    """SkillDefinition can round-trip through dict serialization."""

    def test_to_dict(self):
        from victor_sdk.skills import SkillDefinition

        skill = SkillDefinition(
            name="debug",
            description="Debug a test",
            category="coding",
            prompt_fragment="Debug prompt.",
            required_tools=["read_file"],
            optional_tools=["grep"],
            tags=frozenset({"testing"}),
            max_tool_calls=15,
        )
        d = skill.to_dict()
        assert d["name"] == "debug"
        assert d["description"] == "Debug a test"
        assert d["category"] == "coding"
        assert d["prompt_fragment"] == "Debug prompt."
        assert d["required_tools"] == ["read_file"]
        assert d["optional_tools"] == ["grep"]
        assert "testing" in d["tags"]
        assert d["max_tool_calls"] == 15

    def test_from_dict(self):
        from victor_sdk.skills import SkillDefinition

        d = {
            "name": "review",
            "description": "Code review",
            "category": "coding",
            "prompt_fragment": "Review the code.",
            "required_tools": ["read_file", "grep"],
            "optional_tools": ["code_search"],
            "tags": ["quality"],
            "max_tool_calls": 25,
            "version": "1.1.0",
        }
        skill = SkillDefinition.from_dict(d)
        assert skill.name == "review"
        assert skill.required_tools == ["read_file", "grep"]
        assert skill.optional_tools == ["code_search"]
        assert "quality" in skill.tags
        assert skill.max_tool_calls == 25
        assert skill.version == "1.1.0"

    def test_round_trip(self):
        from victor_sdk.skills import SkillDefinition

        original = SkillDefinition(
            name="debug",
            description="Debug",
            category="coding",
            prompt_fragment="Prompt.",
            required_tools=["read_file"],
            tags=frozenset({"testing"}),
        )
        restored = SkillDefinition.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.category == original.category
        assert restored.prompt_fragment == original.prompt_fragment
        assert restored.required_tools == original.required_tools
        assert restored.tags == original.tags

    def test_from_dict_minimal(self):
        """from_dict works with only required fields."""
        from victor_sdk.skills import SkillDefinition

        d = {
            "name": "minimal",
            "description": "Minimal skill",
            "category": "general",
            "prompt_fragment": "Do the thing.",
            "required_tools": [],
        }
        skill = SkillDefinition.from_dict(d)
        assert skill.name == "minimal"
        assert skill.optional_tools == []
        assert skill.tags == frozenset()


class TestSkillDefinitionToolAccessors:
    """SkillDefinition provides tool access helpers."""

    def test_all_tools(self):
        from victor_sdk.skills import SkillDefinition

        skill = SkillDefinition(
            name="debug",
            description="Debug",
            category="coding",
            prompt_fragment="Debug.",
            required_tools=["read_file", "grep"],
            optional_tools=["code_search", "web_search"],
        )
        all_tools = skill.all_tools
        assert all_tools == {"read_file", "grep", "code_search", "web_search"}

    def test_all_tools_no_optionals(self):
        from victor_sdk.skills import SkillDefinition

        skill = SkillDefinition(
            name="debug",
            description="Debug",
            category="coding",
            prompt_fragment="Debug.",
            required_tools=["read_file"],
        )
        assert skill.all_tools == {"read_file"}


class TestSkillProviderProtocol:
    """SkillProvider protocol is structurally typed."""

    def test_protocol_conformance(self):
        from victor_sdk.skills import SkillDefinition, SkillProvider

        class MyProvider:
            def get_skills(self) -> list:
                return [
                    SkillDefinition(
                        name="test",
                        description="Test",
                        category="testing",
                        prompt_fragment="Test.",
                        required_tools=[],
                    )
                ]

        provider = MyProvider()
        assert isinstance(provider, SkillProvider)
        skills = provider.get_skills()
        assert len(skills) == 1
        assert skills[0].name == "test"


class TestVerticalBaseGetSkills:
    """VerticalBase.get_skills() returns empty list by default."""

    def test_default_returns_empty(self):
        from victor_sdk.verticals.protocols.base import VerticalBase

        class TestVertical(VerticalBase):
            name = "test"
            description = "Test vertical"

            @classmethod
            def get_name(cls) -> str:
                return "test"

            @classmethod
            def get_description(cls) -> str:
                return "Test vertical"

            @classmethod
            def get_tools(cls) -> list:
                return ["read_file"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "You are a test assistant."

        assert TestVertical.get_skills() == []

    def test_override_provides_skills(self):
        from victor_sdk.skills import SkillDefinition
        from victor_sdk.verticals.protocols.base import VerticalBase

        debug_skill = SkillDefinition(
            name="debug",
            description="Debug",
            category="coding",
            prompt_fragment="Debug.",
            required_tools=["read_file"],
        )

        class TestVertical(VerticalBase):
            name = "test"
            description = "Test vertical"

            @classmethod
            def get_name(cls) -> str:
                return "test"

            @classmethod
            def get_description(cls) -> str:
                return "Test vertical"

            @classmethod
            def get_tools(cls) -> list:
                return ["read_file"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "You are a test assistant."

            @classmethod
            def get_skills(cls) -> list:
                return [debug_skill]

        skills = TestVertical.get_skills()
        assert len(skills) == 1
        assert skills[0].name == "debug"
