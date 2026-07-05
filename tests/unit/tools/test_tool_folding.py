from __future__ import annotations

from typing import Any, Dict

import pytest

from victor.tools.base import BaseTool, ToolResult
from victor.tools.folding import (
    folded_tool_names_for_target,
    is_folded_tool,
    should_advertise_tool,
)
from victor.tools.registry import ToolRegistry


class _NamedTool(BaseTool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} tool"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output={"tool": self.name, "kwargs": kwargs})


def test_folded_tools_are_hidden_from_default_enabled_advertisements() -> None:
    registry = ToolRegistry()
    registry.register(_NamedTool("shell"))
    registry.register(_NamedTool("test"))

    assert [tool.name for tool in registry.list_tools()] == ["shell"]
    assert [tool.name for tool in registry.list_tools(include_folded=True)] == [
        "shell",
        "test",
    ]
    assert [tool.name for tool in registry.list_tools(only_enabled=False)] == [
        "shell",
        "test",
    ]


def test_folded_tool_schemas_are_hidden_by_default_but_available_on_request() -> None:
    registry = ToolRegistry()
    registry.register(_NamedTool("shell"))
    registry.register(_NamedTool("test"))

    assert [schema["function"]["name"] for schema in registry.get_tool_schemas()] == ["shell"]
    assert [
        schema["function"]["name"] for schema in registry.get_tool_schemas(include_folded=True)
    ] == ["shell", "test"]
    assert [
        schema["function"]["name"] for schema in registry.get_tool_schemas(only_enabled=False)
    ] == ["shell", "test"]


@pytest.mark.asyncio
async def test_folded_tools_remain_executable() -> None:
    registry = ToolRegistry()
    registry.register(_NamedTool("dependency"))

    result = await registry.execute("dependency", {})

    assert result.success is True
    assert result.output == {"tool": "dependency", "kwargs": {}}


def test_shell_description_does_not_include_folded_tool_guidance() -> None:
    from victor.tools.bash import shell

    description = shell.Tool.description

    # Folded-tool-guidance noise was removed from descriptions — it's internal
    # routing hints, not LLM-actionable.
    assert "Folded tool guidance:" not in description

    # Fold targets still exist for routing.
    assert folded_tool_names_for_target("shell") == [
        "dependency",
        "extract",
        "find",
        "inline",
        "ls",
        "organize_imports",
        "rename",
        "sandbox",
        "scaffold",
        "test",
    ]
    assert folded_tool_names_for_target("db") == ["database"]
    assert folded_tool_names_for_target("rag") == [
        "rag_delete",
        "rag_ingest",
        "rag_list",
        "rag_query",
        "rag_search",
        "rag_stats",
    ]


def test_granular_primitives_are_not_folded() -> None:
    """read/write/edit are first-class tools (no longer folded into fs — fs is removed).

    ls/find fold into shell. web_search/web_fetch fold into web. search folds into code.
    """
    # read/write/edit are EXPOSED (not folded).
    for name in ["read", "write", "edit"]:
        assert not is_folded_tool(name), f"{name} should NOT be folded (first-class tool)"
        assert should_advertise_tool(name), f"{name} should be advertised"

    # ls/find fold into shell.
    for name in ["ls", "find"]:
        assert is_folded_tool(name), f"{name} should be folded into shell"
        assert not should_advertise_tool(name), f"{name} should not be advertised"

    # web/search folds remain.
    assert folded_tool_names_for_target("web") == ["web_fetch", "web_search"]
    assert folded_tool_names_for_target("code") == ["search"]


def test_shell_git_web_code_domains_are_not_folded() -> None:
    """shell/git/web/code domains stay advertised (fs is removed)."""
    from victor.tools.folding import should_advertise_tool

    for name in ["shell", "git", "web", "code"]:
        assert should_advertise_tool(name), f"{name} should be advertised as a primary domain"


@pytest.mark.asyncio
async def test_folded_tool_remains_executable() -> None:
    """Folding hides a tool from advertisement but must not break execution."""
    registry = ToolRegistry()
    registry.register(_NamedTool("ls"))

    assert [tool.name for tool in registry.list_tools()] == []  # hidden from default
    assert [tool.name for tool in registry.list_tools(include_folded=True)] == ["ls"]

    result = await registry.execute("ls", {})
    assert result.success is True
    assert result.output == {"tool": "ls", "kwargs": {}}
