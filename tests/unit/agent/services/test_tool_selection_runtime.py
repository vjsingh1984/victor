from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.orchestrator_protocol_adapter import (
    OrchestratorProtocolAdapter,
)
from victor.agent.services.tool_selection_runtime import ToolSelectionRuntime
from victor.agent.action_authorizer import ActionIntent


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
async def test_tool_selection_runtime_restores_edit_write_for_write_allowed_turn():
    provider = MagicMock()
    provider.supports_tools.return_value = True
    read_tool = SimpleNamespace(name="read")
    code_search_tool = SimpleNamespace(name="code_search")
    edit_tool = SimpleNamespace(name="edit")
    write_tool = SimpleNamespace(name="write")
    shell_tool = SimpleNamespace(name="shell")
    selected_tools = [read_tool, code_search_tool]
    tool_selector = MagicMock()
    tool_selector.select_tools = AsyncMock(return_value=selected_tools)
    tool_selector.prioritize_by_stage.return_value = selected_tools
    tool_planner = MagicMock()
    tool_planner.filter_tools_by_intent.return_value = selected_tools
    conversation = SimpleNamespace(message_count=MagicMock(return_value=4))
    host = SimpleNamespace(
        provider=provider,
        _model_supports_tool_calls=MagicMock(return_value=True),
        _should_skip_tools_for_turn=MagicMock(return_value=False),
        observed_files={"rust/crates/python-bindings/src/similarity.rs"},
        _tool_planner=tool_planner,
        conversation=conversation,
        messages=[_Message("user", "Address these findings one by one and update code as needed.")],
        tool_selector=tool_selector,
        use_semantic_selection=True,
        _current_intent=ActionIntent.WRITE_ALLOWED,
        _current_user_message="Address these findings one by one and update code as needed.",
        tools=SimpleNamespace(
            list_tools=MagicMock(return_value=[read_tool, edit_tool, write_tool, shell_tool])
        ),
        _apply_kv_tool_strategy=MagicMock(side_effect=lambda tools: tools),
        _sort_tools_for_kv_stability=MagicMock(side_effect=lambda tools: tools),
    )
    runtime = ToolSelectionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.select_tools_for_turn(
        "Now let me read files before applying fixes.",
        goals=None,
    )

    assert [tool.name for tool in result] == [
        "read",
        "code_search",
        "edit",
        "write",
        "shell",
    ]
    host._apply_kv_tool_strategy.assert_called_once()
    applied_tools = host._apply_kv_tool_strategy.call_args.args[0]
    assert [tool.name for tool in applied_tools] == [
        "read",
        "code_search",
        "edit",
        "write",
        "shell",
    ]


@pytest.mark.asyncio
async def test_tool_selection_runtime_keeps_frozen_user_intent_despite_assistant_read_step():
    provider = MagicMock()
    provider.supports_tools.return_value = True
    read_tool = SimpleNamespace(name="read")
    edit_tool = SimpleNamespace(name="edit")
    write_tool = SimpleNamespace(name="write")
    shell_tool = SimpleNamespace(name="shell")
    selected_tools = [read_tool]
    tool_selector = MagicMock()
    tool_selector.select_tools = AsyncMock(return_value=selected_tools)
    tool_selector.prioritize_by_stage.return_value = selected_tools
    tool_planner = MagicMock()
    tool_planner.filter_tools_by_intent.return_value = selected_tools
    conversation = SimpleNamespace(message_count=MagicMock(return_value=5))
    host = SimpleNamespace(
        provider=provider,
        _model_supports_tool_calls=MagicMock(return_value=True),
        _should_skip_tools_for_turn=MagicMock(return_value=False),
        observed_files={"rust/crates/python-bindings/src/similarity.rs"},
        _tool_planner=tool_planner,
        conversation=conversation,
        messages=[
            _Message(
                "user",
                "Address these findings one by one and update code as needed.",
            )
        ],
        tool_selector=tool_selector,
        use_semantic_selection=True,
        _current_intent=ActionIntent.WRITE_ALLOWED,
        _current_user_message="Address these findings one by one and update code as needed.",
        tools=SimpleNamespace(
            list_tools=MagicMock(return_value=[read_tool, edit_tool, write_tool, shell_tool])
        ),
        _apply_kv_tool_strategy=MagicMock(side_effect=lambda tools: tools),
        _sort_tools_for_kv_stability=MagicMock(side_effect=lambda tools: tools),
    )
    runtime = ToolSelectionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.select_tools_for_turn(
        "Now let me read the source files before deciding what to change.",
        goals=None,
    )

    tool_selector.select_tools.assert_awaited_once_with(
        "Address these findings one by one and update code as needed.",
        use_semantic=True,
        conversation_history=[
            {
                "role": "user",
                "content": "Address these findings one by one and update code as needed.",
            }
        ],
        conversation_depth=5,
        planned_tools=None,
    )
    tool_selector.prioritize_by_stage.assert_called_once_with(
        "Address these findings one by one and update code as needed.",
        selected_tools,
    )
    tool_planner.filter_tools_by_intent.assert_called_once_with(
        selected_tools,
        ActionIntent.WRITE_ALLOWED,
        user_message="Address these findings one by one and update code as needed.",
    )
    assert [tool.name for tool in result] == ["read", "edit", "write", "shell"]


@pytest.mark.asyncio
async def test_tool_selection_runtime_does_not_restore_edit_write_for_display_only_turn():
    provider = MagicMock()
    provider.supports_tools.return_value = True
    read_tool = SimpleNamespace(name="read")
    edit_tool = SimpleNamespace(name="edit")
    write_tool = SimpleNamespace(name="write")
    shell_tool = SimpleNamespace(name="shell")
    selected_tools = [read_tool]
    tool_selector = MagicMock()
    tool_selector.select_tools = AsyncMock(return_value=selected_tools)
    tool_selector.prioritize_by_stage.return_value = selected_tools
    tool_planner = MagicMock()
    tool_planner.filter_tools_by_intent.return_value = selected_tools
    conversation = SimpleNamespace(message_count=MagicMock(return_value=1))
    host = SimpleNamespace(
        provider=provider,
        _model_supports_tool_calls=MagicMock(return_value=True),
        _should_skip_tools_for_turn=MagicMock(return_value=False),
        observed_files=set(),
        _tool_planner=tool_planner,
        conversation=conversation,
        messages=[_Message("user", "show me how this would look")],
        tool_selector=tool_selector,
        use_semantic_selection=True,
        _current_intent=ActionIntent.DISPLAY_ONLY,
        _current_user_message="show me how this would look",
        tools=SimpleNamespace(
            list_tools=MagicMock(return_value=[read_tool, edit_tool, write_tool, shell_tool])
        ),
        _apply_kv_tool_strategy=MagicMock(side_effect=lambda tools: tools),
        _sort_tools_for_kv_stability=MagicMock(side_effect=lambda tools: tools),
    )
    runtime = ToolSelectionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.select_tools_for_turn(
        "I can explain the pattern.",
        goals=None,
    )

    assert [tool.name for tool in result] == ["read"]


@pytest.mark.asyncio
async def test_tool_selection_runtime_reuses_precomputed_planned_tools():
    provider = MagicMock()
    provider.supports_tools.return_value = True
    tool_selector = MagicMock()
    selected_tools = [SimpleNamespace(name="read_file"), SimpleNamespace(name="search")]
    tool_selector.select_tools = AsyncMock(return_value=selected_tools)
    tool_selector.prioritize_by_stage.return_value = selected_tools
    tool_planner = MagicMock()
    tool_planner.filter_tools_by_intent.return_value = selected_tools
    conversation = SimpleNamespace(message_count=MagicMock(return_value=1))
    precomputed_plan = [SimpleNamespace(name="git_diff")]
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
        _current_intent=None,
        _current_user_message="inspect app.py",
        _apply_kv_tool_strategy=MagicMock(return_value=selected_tools),
        _sort_tools_for_kv_stability=MagicMock(return_value=selected_tools),
    )
    runtime = ToolSelectionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.select_tools_for_turn(
        "inspect app.py",
        goals=["analyze"],
        planned_tools=precomputed_plan,
    )

    assert result == selected_tools
    tool_planner.plan_tools.assert_not_called()
    tool_selector.select_tools.assert_awaited_once_with(
        "inspect app.py",
        use_semantic=True,
        conversation_history=[{"role": "user", "content": "inspect the file"}],
        conversation_depth=1,
        planned_tools=precomputed_plan,
    )


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
        "query the sqllite db directly using shell or database tools",
        use_semantic=True,
        conversation_history=[{"role": "user", "content": "query the sqlite db directly"}],
        conversation_depth=2,
        planned_tools=None,
    )
    tool_selector.prioritize_by_stage.assert_called_once_with(
        "query the sqllite db directly using shell or database tools",
        selected_tools,
    )
    tool_planner.filter_tools_by_intent.assert_called_once_with(
        selected_tools,
        "display_only",
        user_message="query the sqllite db directly using shell or database tools",
    )


@pytest.mark.asyncio
async def test_tool_selection_runtime_prioritizes_database_tools_for_explicit_request():
    provider = MagicMock()
    provider.supports_tools.return_value = True
    tool_selector = MagicMock()
    selected_tools = [
        SimpleNamespace(name="graph"),
        SimpleNamespace(name="read"),
        SimpleNamespace(name="shell"),
        SimpleNamespace(name="db"),
    ]
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
        messages=[_Message("user", "inspect the sqlite db directly")],
        tool_selector=tool_selector,
        use_semantic_selection=True,
        _current_intent="display_only",
        _current_user_message="inspect the sqllite db directly using shell or database tools",
        _apply_kv_tool_strategy=MagicMock(side_effect=lambda tools: tools),
        _sort_tools_for_kv_stability=MagicMock(side_effect=lambda tools: tools),
    )
    runtime = ToolSelectionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.select_tools_for_turn(
        "Now let me inspect the prompt optimizer implementation.",
        goals=None,
    )

    assert [tool.name for tool in result] == ["db", "shell", "graph", "read"]


@pytest.mark.asyncio
async def test_tool_selection_runtime_preserves_order_without_explicit_database_request():
    provider = MagicMock()
    provider.supports_tools.return_value = True
    tool_selector = MagicMock()
    selected_tools = [
        SimpleNamespace(name="graph"),
        SimpleNamespace(name="read"),
        SimpleNamespace(name="shell"),
        SimpleNamespace(name="db"),
    ]
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
        messages=[_Message("user", "inspect prompt evolution")],
        tool_selector=tool_selector,
        use_semantic_selection=True,
        _current_intent="display_only",
        _current_user_message="inspect prompt evolution in the repo",
        _apply_kv_tool_strategy=MagicMock(side_effect=lambda tools: tools),
        _sort_tools_for_kv_stability=MagicMock(side_effect=lambda tools: tools),
    )
    runtime = ToolSelectionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.select_tools_for_turn(
        "Now let me inspect the prompt optimizer implementation.",
        goals=None,
    )

    assert [tool.name for tool in result] == ["graph", "read", "shell", "db"]


# --- tool-supply P3: Q&A gate -> read-core (not None) ----------------------------


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"{name} tool"
        self.parameters = {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_qa_gate_skip_mode_returns_none():
    provider = MagicMock()
    provider.supports_tools.return_value = True
    host = SimpleNamespace(
        provider=provider,
        _model_supports_tool_calls=MagicMock(return_value=True),
        _tool_skip_mode=MagicMock(return_value="skip"),
        _should_skip_tools_for_turn=MagicMock(return_value=True),
        tool_selector=SimpleNamespace(tools=None),
    )
    runtime = ToolSelectionRuntime(OrchestratorProtocolAdapter(host))
    result = await runtime.select_tools_for_turn("hi", goals=None)
    assert result is None


@pytest.mark.asyncio
async def test_qa_gate_read_core_mode_returns_read_tools():
    provider = MagicMock()
    provider.supports_tools.return_value = True
    registry = MagicMock()
    registry.get.side_effect = lambda name: (
        _FakeTool(name)
        if name
        in {
            "read",
            "code_search",
            "ls",
        }
        else None
    )
    host = SimpleNamespace(
        provider=provider,
        _model_supports_tool_calls=MagicMock(return_value=True),
        _tool_skip_mode=MagicMock(return_value="read_core"),
        _should_skip_tools_for_turn=MagicMock(return_value=True),
        tool_selector=SimpleNamespace(tools=registry),
    )
    runtime = ToolSelectionRuntime(OrchestratorProtocolAdapter(host))
    result = await runtime.select_tools_for_turn("how does the auth flow work", goals=None)

    # Borderline Q&A keeps a minimal read-only core (not None), at STUB schema.
    assert result is not None
    names = [t.name for t in result]
    assert names == ["read", "code_search", "ls"]
    assert all(t.schema_level == "stub" for t in result)
