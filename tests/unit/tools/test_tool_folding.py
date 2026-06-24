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


def test_shell_description_includes_folded_tool_guidance() -> None:
    from victor.tools.bash import shell

    description = shell.Tool.description

    # `docker` is now its own DevOps domain (advertised via vertical_tools.yaml),
    # and `database` folds into the `db` domain — so neither is in the shell set.
    assert not is_folded_tool("docker")
    assert folded_tool_names_for_target("shell") == [
        "dependency",
        "extract",
        "inline",
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
    assert "Folded tool guidance:" in description
    assert "pytest" in description


def test_unified_domain_folds_hide_granular_primitives() -> None:
    """Granular primitives subsumed by fs/web/code are folded (hidden-but-callable).

    This delivers the unified-primary surface: ``read``/``write``/``ls``/
    ``edit``/``find`` -> ``fs``; ``web_search``/``web_fetch`` -> ``web``;
    ``search`` -> ``code``. They remain registered and executable but are not
    advertised as separate default schemas.
    """
    assert folded_tool_names_for_target("fs") == ["edit", "find", "ls", "read", "write"]
    assert folded_tool_names_for_target("web") == ["web_fetch", "web_search"]
    assert folded_tool_names_for_target("code") == ["search"]

    for name in ["read", "write", "ls", "edit", "find", "web_search", "web_fetch", "search"]:
        assert is_folded_tool(name), f"{name} should be folded under unified-primary"
        assert not should_advertise_tool(name), f"{name} should not be advertised"


def test_unified_domain_tools_are_not_folded() -> None:
    """The fs/shell/git/web/code domains themselves stay advertised."""
    from victor.tools.folding import should_advertise_tool

    for name in ["fs", "shell", "git", "web", "code"]:
        assert should_advertise_tool(name), f"{name} should be advertised as a primary domain"


@pytest.mark.asyncio
async def test_folded_granular_primitive_remains_executable() -> None:
    """Folding hides a tool from advertisement but must not break execution."""
    registry = ToolRegistry()
    registry.register(_NamedTool("read"))

    assert [tool.name for tool in registry.list_tools()] == []  # hidden from default
    assert [tool.name for tool in registry.list_tools(include_folded=True)] == ["read"]

    result = await registry.execute("read", {})
    assert result.success is True
    assert result.output == {"tool": "read", "kwargs": {}}


def test_fs_description_includes_granular_fold_hints() -> None:
    """The fs domain advertises guidance for the primitives folded into it."""
    from victor.tools.unified.fs_tool import fs_tool

    description = fs_tool.Tool.description
    assert "Folded tool guidance:" in description
    assert "read:" in description
    assert "edit:" in description
    # web folds belong to web, not fs
    assert "web_fetch" not in description
