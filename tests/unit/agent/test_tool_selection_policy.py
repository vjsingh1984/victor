from victor.agent.conversation.state_machine import ConversationStage
from victor.providers.base import ToolDefinition


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(name=name, description=f"{name} description", parameters={})


def test_stage_policy_prioritizes_stage_core_web_and_adjacent_tools():
    from victor.agent.tool_selection_policy import (
        StageToolSelectionContext,
        ToolSelectionStagePolicy,
    )

    policy = ToolSelectionStagePolicy(fallback_max_tools=4)
    tools = [
        _tool("write"),
        _tool("read"),
        _tool("search"),
        _tool("web_search"),
        _tool("adjacent"),
    ]
    context = StageToolSelectionContext(
        current_stage=ConversationStage.ANALYSIS,
        stage_tools={"search"},
        core_tools={"read"},
        web_tools={"web_search"},
        mandatory_tools=set(),
        vertical_core_tools=set(),
    )

    selected = policy.prioritize_by_stage(
        tools,
        context=context,
        should_include_tool=lambda name: name == "adjacent",
        get_tool_priority_boost=lambda name: {
            "search": 0.9,
            "web_search": 0.7,
            "adjacent": 0.4,
            "read": 0.0,
        }.get(name, 0.0),
    )

    assert [tool.name for tool in selected] == ["search", "web_search", "adjacent", "read"]


def test_stage_policy_falls_back_to_core_and_vertical_tools():
    from victor.agent.tool_selection_policy import (
        StageToolSelectionContext,
        ToolSelectionStagePolicy,
    )

    policy = ToolSelectionStagePolicy(fallback_max_tools=4)
    tools = [_tool("custom"), _tool("docker"), _tool("read")]
    context = StageToolSelectionContext(
        current_stage=ConversationStage.READING,
        stage_tools={"grep"},
        core_tools={"read"},
        web_tools=set(),
        mandatory_tools=set(),
        vertical_core_tools={"docker"},
    )

    selected = policy.prioritize_by_stage(
        tools,
        context=context,
        should_include_tool=lambda _name: False,
        get_tool_priority_boost=lambda _name: 0.0,
    )

    assert [tool.name for tool in selected] == ["docker", "read"]


def test_stage_policy_uses_small_prefix_as_last_resort():
    from victor.agent.tool_selection_policy import (
        StageToolSelectionContext,
        ToolSelectionStagePolicy,
    )

    policy = ToolSelectionStagePolicy(fallback_max_tools=2)
    tools = [_tool("one"), _tool("two"), _tool("three")]
    context = StageToolSelectionContext(
        current_stage=ConversationStage.PLANNING,
        stage_tools={"search"},
        core_tools={"read"},
        web_tools=set(),
        mandatory_tools=set(),
        vertical_core_tools=set(),
    )

    selected = policy.prioritize_by_stage(
        tools,
        context=context,
        should_include_tool=lambda _name: False,
        get_tool_priority_boost=lambda _name: 0.0,
    )

    assert [tool.name for tool in selected] == ["one", "two"]


def test_stage_policy_builds_semantic_fallback_with_core_first_and_dedupes():
    from victor.agent.tool_selection_policy import ToolSelectionStagePolicy

    policy = ToolSelectionStagePolicy(fallback_max_tools=3)
    selected = policy.build_semantic_fallback_tools(
        all_tools=[_tool("read"), _tool("search"), _tool("grep")],
        core_tools={"read"},
        keyword_tools=[_tool("search"), _tool("read"), _tool("grep")],
    )

    assert [tool.name for tool in selected] == ["read", "search", "grep"]
