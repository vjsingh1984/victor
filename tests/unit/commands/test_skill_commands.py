"""Tests for victor skill CLI commands.

Covers:
- victor skill list
- victor skill list --category coding
- victor skill info <name>
- victor skill search <query>
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from victor_sdk.skills import SkillDefinition

runner = CliRunner()


def _make_skill(name: str = "debug", category: str = "coding", **kwargs):
    return SkillDefinition(
        name=name,
        description=kwargs.get("description", f"Skill: {name}"),
        category=category,
        prompt_fragment=kwargs.get("prompt_fragment", f"Prompt for {name}."),
        required_tools=kwargs.get("required_tools", ["read_file"]),
        optional_tools=kwargs.get("optional_tools", []),
        tags=kwargs.get("tags", frozenset()),
    )


def _get_app():
    """Import the skills CLI app."""
    from victor.ui.commands.skills import skills_app

    return skills_app


class TestSkillList:
    """victor skill list command."""

    def test_list_empty(self):
        app = _get_app()
        with patch(
            "victor.ui.commands.skills._build_registry",
            return_value=MagicMock(list_all=MagicMock(return_value=[])),
        ):
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "No skills" in result.output

    def test_list_shows_skills(self):
        app = _get_app()
        skills = [
            _make_skill("debug", category="coding", description="Debug a test"),
            _make_skill("deploy", category="devops", description="Deploy service"),
        ]
        mock_registry = MagicMock(list_all=MagicMock(return_value=skills))
        with patch(
            "victor.ui.commands.skills._build_registry",
            return_value=mock_registry,
        ):
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "debug" in result.output
            assert "deploy" in result.output

    def test_list_filter_by_category(self):
        app = _get_app()
        skills = [
            _make_skill("debug", category="coding"),
        ]
        mock_registry = MagicMock(
            search=MagicMock(return_value=skills),
        )
        with patch(
            "victor.ui.commands.skills._build_registry",
            return_value=mock_registry,
        ):
            result = runner.invoke(app, ["list", "--category", "coding"])
            assert result.exit_code == 0
            assert "debug" in result.output
            mock_registry.search.assert_called_once_with(category="coding")


class TestSkillInfo:
    """victor skill info <name> command."""

    def test_info_found(self):
        app = _get_app()
        skill = _make_skill(
            "debug",
            description="Debug a failing test",
            required_tools=["read_file", "grep"],
            optional_tools=["code_search"],
            tags=frozenset({"testing", "debugging"}),
        )
        mock_registry = MagicMock(get_optional=MagicMock(return_value=skill))
        with patch(
            "victor.ui.commands.skills._build_registry",
            return_value=mock_registry,
        ):
            result = runner.invoke(app, ["info", "debug"])
            assert result.exit_code == 0
            assert "debug" in result.output
            assert "Debug a failing test" in result.output
            assert "read_file" in result.output

    def test_info_not_found(self):
        app = _get_app()
        mock_registry = MagicMock(get_optional=MagicMock(return_value=None))
        with patch(
            "victor.ui.commands.skills._build_registry",
            return_value=mock_registry,
        ):
            result = runner.invoke(app, ["info", "nonexistent"])
            assert result.exit_code == 0
            assert "not found" in result.output.lower()


class TestSkillSearch:
    """victor skill search <query> command."""

    def test_search_finds_results(self):
        app = _get_app()
        skills = [_make_skill("debug_test", description="Debug a failing test")]
        mock_registry = MagicMock(search=MagicMock(return_value=skills))
        with patch(
            "victor.ui.commands.skills._build_registry",
            return_value=mock_registry,
        ):
            result = runner.invoke(app, ["search", "debug"])
            assert result.exit_code == 0
            assert "debug_test" in result.output
            mock_registry.search.assert_called_once_with(query="debug")

    def test_search_no_results(self):
        app = _get_app()
        mock_registry = MagicMock(search=MagicMock(return_value=[]))
        with patch(
            "victor.ui.commands.skills._build_registry",
            return_value=mock_registry,
        ):
            result = runner.invoke(app, ["search", "nonexistent"])
            assert result.exit_code == 0
            assert "No skills" in result.output
