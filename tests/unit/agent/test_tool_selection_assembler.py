from victor.providers.base import ToolDefinition


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(name=name, description=f"{name} description", parameters={})


def test_assembler_blends_keywords_with_bounded_growth():
    from victor.agent.tool_selection_assembler import (
        SemanticToolSelectionAssembler,
        SemanticToolSelectionAssemblyContext,
    )

    assembler = SemanticToolSelectionAssembler()
    semantic_tools = [_tool(f"semantic_{index}") for index in range(5)]
    keyword_tools = [_tool(f"keyword_{index}") for index in range(6)]

    assembled = assembler.assemble(
        semantic_tools,
        keyword_tools=keyword_tools,
        all_tools=[],
        context=SemanticToolSelectionAssemblyContext(
            max_tools=6,
            include_web_tools=False,
            web_tool_names=set(),
        ),
    )

    assert [tool.name for tool in assembled] == [
        "semantic_0",
        "semantic_1",
        "semantic_2",
        "semantic_3",
        "semantic_4",
        "keyword_0",
        "keyword_1",
        "keyword_2",
    ]


def test_assembler_injects_explicit_web_tools_without_duplicates():
    from victor.agent.tool_selection_assembler import (
        SemanticToolSelectionAssembler,
        SemanticToolSelectionAssemblyContext,
    )

    assembler = SemanticToolSelectionAssembler()
    assembled = assembler.assemble(
        [_tool("read"), _tool("search")],
        keyword_tools=[_tool("search")],
        all_tools=[_tool("web_search"), _tool("browser_open"), _tool("search")],
        context=SemanticToolSelectionAssemblyContext(
            max_tools=5,
            include_web_tools=True,
            web_tool_names={"web_search", "browser_open"},
        ),
    )

    assert [tool.name for tool in assembled] == ["read", "search", "web_search", "browser_open"]


def test_assembler_dedupes_stably_by_tool_name():
    from victor.agent.tool_selection_assembler import (
        SemanticToolSelectionAssembler,
        SemanticToolSelectionAssemblyContext,
    )

    assembler = SemanticToolSelectionAssembler()
    assembled = assembler.assemble(
        [_tool("read"), _tool("search"), _tool("read")],
        keyword_tools=[_tool("search"), _tool("grep")],
        all_tools=[],
        context=SemanticToolSelectionAssemblyContext(
            max_tools=5,
            include_web_tools=False,
            web_tool_names=set(),
        ),
    )

    assert [tool.name for tool in assembled] == ["read", "search", "grep"]
