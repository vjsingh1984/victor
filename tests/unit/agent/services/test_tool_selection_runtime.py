from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.tool_selection_runtime import ToolSelectionRuntime


class _Message:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def model_dump(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@pytest.mark.asyncio
async def test_tool_selection_runtime_returns_none_when_tooling_not_allowed():
    provider = MagicMock()
    provider.supports_tools.return_value = False
    host = SimpleNamespace(
        provider=provider,
        _model_supports_tool_calls=MagicMock(return_value=True),
    )
    runtime = ToolSelectionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.select_tools_for_turn("hello", goals=None)

    assert result is None


@pytest.mark.asyncio
async def test_tool_selection_runtime_returns_none_when_tool_necessity_skips():
    provider = MagicMock()
    provider.supports_tools.return_value = True
    host = SimpleNamespace(
        provider=provider,
        _model_supports_tool_calls=MagicMock(return_value=True),
        _should_skip_tools_for_turn=MagicMock(return_value=True),
    )
    runtime = ToolSelectionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.select_tools_for_turn("what is python", goals=None)

    assert result is None


@pytest.mark.asyncio
async def test_tool_selection_runtime_selects_and_filters_tools_with_goals():
    provider = MagicMock()
    provider.supports_tools.return_value = True
    tool_selector = MagicMock()
    selected_tools = [SimpleNamespace(name="read_file"), SimpleNamespace(name="search")]
    prioritized_tools = [selected_tools[1], selected_tools[0]]
    filtered_tools = [selected_tools[1]]
    kv_tools = [selected_tools[1]]
    sorted_tools = [selected_tools[1]]
    tool_selector.select_tools = AsyncMock(return_value=selected_tools)
    tool_selector.prioritize_by_stage.return_value = prioritized_tools
    tool_planner = MagicMock()
    tool_planner.plan_tools.return_value = ["search"]
    tool_planner.filter_tools_by_intent.return_value = filtered_tools
    conversation = SimpleNamespace(message_count=MagicMock(return_value=3))
    host = SimpleNamespace(
        provider=provider,
        _model_supports_tool_calls=MagicMock(return_value=True),
        _should_skip_tools_for_turn=MagicMock(return_value=False),
        observed_files={"app.py"},
        _tool_planner=tool_planner,
        conversation=conversation,
        messages=[_Message("user", "inspect the file")],
        tool_selector=tool_selector,
        use_semantic_selection=True,
        _current_intent="read_only",
        _current_user_message="inspect app.py",
        _apply_kv_tool_strategy=MagicMock(return_value=kv_tools),
        _sort_tools_for_kv_stability=MagicMock(return_value=sorted_tools),
    )
    runtime = ToolSelectionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.select_tools_for_turn("inspect app.py", goals=["analyze"])

    assert result == sorted_tools
    tool_planner.plan_tools.assert_called_once_with(["analyze"], ["query", "file_contents"])
    tool_selector.select_tools.assert_awaited_once_with(
        "inspect app.py",
        use_semantic=True,
        conversation_history=[{"role": "user", "content": "inspect the file"}],
        conversation_depth=3,
        planned_tools=["search"],
    )
    tool_selector.prioritize_by_stage.assert_called_once_with("inspect app.py", selected_tools)
    tool_planner.filter_tools_by_intent.assert_called_once_with(
        prioritized_tools,
        "read_only",
        user_message="inspect app.py",
    )
    host._apply_kv_tool_strategy.assert_called_once_with(filtered_tools)
    host._sort_tools_for_kv_stability.assert_called_once_with(kv_tools)


@pytest.mark.asyncio
async def test_tool_selection_runtime_preserves_user_request_as_anchor():
    provider = MagicMock()
    provider.supports_tools.return_value = True
    tool_selector = MagicMock()
    selected_tools = [SimpleNamespace(name="shell"), SimpleNamespace(name="db")]
    tool_selector.select_tools = AsyncMock(return_value=selected_tools)
    tool_selector.prioritize_by_stage.return_value = selected_tools
    tool_planner = MagicMock()
    tool_planner.filter_tools_by_intent.return_value = selected_tools
    conversation = SimpleNamespace(message_count=MagicMock(return_value=2))
    host = SimpleNamespace(
        provider=provider,
        _model_supports_tool_calls=MagicMock(return_value=True),
        _should_skip_tools_for_turn=MagicMock(return_value=False),
        observed_files=set(),
        _tool_planner=tool_planner,
        conversation=conversation,
        messages=[_Message("user", "query the sqlite db directly")],
        tool_selector=tool_selector,
        use_semantic_selection=True,
        _current_intent="display_only",
        _current_user_message="query the sqllite db directly using shell or database tools",
        _apply_kv_tool_strategy=MagicMock(return_value=selected_tools),
        _sort_tools_for_kv_stability=MagicMock(return_value=selected_tools),
    )
    runtime = ToolSelectionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.select_tools_for_turn(
        "Now let me inspect the prompt optimizer implementation.",
        goals=None,
    )

    assert result == selected_tools
    tool_selector.select_tools.assert_awaited_once_with(
        "query the sqllite db directly using shell or database tools\n\n"
        "Current working step: Now let me inspect the prompt optimizer implementation.",
        use_semantic=True,
        conversation_history=[{"role": "user", "content": "query the sqlite db directly"}],
        conversation_depth=2,
        planned_tools=None,
    )
    tool_planner.filter_tools_by_intent.assert_called_once_with(
        selected_tools,
        "display_only",
        user_message="query the sqllite db directly using shell or database tools",
    )
