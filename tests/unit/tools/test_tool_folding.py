from __future__ import annotations

from typing import Any, Dict

import pytest

from victor.tools.base import BaseTool, ToolResult
from victor.tools.folding import folded_tool_names_for_target, is_folded_tool
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
    registry.register(_NamedTool("docker"))

    assert [tool.name for tool in registry.list_tools()] == ["shell"]
    assert [tool.name for tool in registry.list_tools(include_folded=True)] == [
        "shell",
        "docker",
    ]
    assert [tool.name for tool in registry.list_tools(only_enabled=False)] == [
        "shell",
        "docker",
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


def test_shell_description_includes_folded_tool_guidance() -> None:
    from victor.tools.bash import shell

    description = shell.Tool.description

    assert is_folded_tool("docker")
    assert folded_tool_names_for_target("shell") == [
        "database",
        "dependency",
        "docker",
        "extract",
        "inline",
        "organize_imports",
        "rag_delete",
        "rag_ingest",
        "rag_list",
        "rag_query",
        "rag_search",
        "rag_stats",
        "rename",
        "sandbox",
        "scaffold",
        "test",
    ]
    assert "Folded tool guidance:" in description
    assert "docker:" in description
    assert "pytest" in description
