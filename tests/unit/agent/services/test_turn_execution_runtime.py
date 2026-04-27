"""Focused tests for TurnExecutor runtime behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.services.turn_execution_runtime import TurnExecutor
from victor.framework.task.protocols import TaskComplexity
from victor.providers.base import CompletionResponse, Message


def _make_executor(exploration_coordinator=None) -> TurnExecutor:
    chat_context = MagicMock()
    chat_context.settings = MagicMock()
    chat_context.add_message = MagicMock()
    chat_context.conversation = MagicMock()

    tool_context = MagicMock()
    provider_context = MagicMock()
    provider_context.provider_name = "ollama"
    provider_context.model = "test-model"
    execution_provider = MagicMock()

    return TurnExecutor(
        chat_context=chat_context,
        tool_context=tool_context,
        provider_context=provider_context,
        execution_provider=execution_provider,
        exploration_coordinator=exploration_coordinator,
    )


@pytest.mark.asyncio
async def test_parallel_exploration_uses_injected_coordinator():
    explorer = MagicMock()
    explorer.explore_parallel = AsyncMock(
        return_value=SimpleNamespace(
            summary="found relevant files",
            file_paths=["victor/agent/services/turn_execution_runtime.py"],
            tool_calls=2,
            duration_seconds=0.4,
        )
    )
    executor = _make_executor(exploration_coordinator=explorer)
    task_classification = SimpleNamespace(complexity=TaskComplexity.COMPLEX)

    with (
        patch(
            "victor.config.settings.load_settings",
            return_value=SimpleNamespace(pipeline=SimpleNamespace(parallel_exploration=True)),
        ),
        patch(
            "victor.config.settings.get_project_paths",
            return_value=SimpleNamespace(project_root="/tmp/project"),
        ),
        patch(
            "victor.agent.budget.resource_calculator.calculate_exploration_budget",
            return_value=SimpleNamespace(
                max_parallel_agents=2,
                tool_budget_per_agent=3,
                exploration_timeout=5,
            ),
        ),
    ):
        await executor._run_parallel_exploration(
            "inspect the failing runtime path",
            task_classification,
        )

    explorer.explore_parallel.assert_awaited_once()
    executor._chat_context.add_message.assert_called_once()
    assert executor._exploration_done is True


@pytest.mark.asyncio
async def test_parallel_exploration_lazily_materializes_shared_coordinator():
    explorer = MagicMock()
    explorer.explore_parallel = AsyncMock(
        return_value=SimpleNamespace(
            summary="shared helper exploration",
            file_paths=["victor/agent/coordinators/factory_support.py"],
            tool_calls=1,
            duration_seconds=0.2,
        )
    )
    executor = _make_executor()
    task_classification = SimpleNamespace(complexity=TaskComplexity.COMPLEX)

    with (
        patch(
            "victor.config.settings.load_settings",
            return_value=SimpleNamespace(pipeline=SimpleNamespace(parallel_exploration=True)),
        ),
        patch(
            "victor.config.settings.get_project_paths",
            return_value=SimpleNamespace(project_root="/tmp/project"),
        ),
        patch(
            "victor.agent.budget.resource_calculator.calculate_exploration_budget",
            return_value=SimpleNamespace(
                max_parallel_agents=1,
                tool_budget_per_agent=2,
                exploration_timeout=5,
            ),
        ),
        patch(
            "victor.agent.coordinators.factory_support.create_exploration_coordinator",
            return_value=explorer,
        ) as create_explorer,
    ):
        await executor._run_parallel_exploration(
            "trace the lazy helper path",
            task_classification,
        )

    create_explorer.assert_called_once_with()
    explorer.explore_parallel.assert_awaited_once()
    assert executor._exploration_coordinator is explorer


@pytest.mark.asyncio
async def test_execute_tool_calls_prefers_canonical_tool_context_method():
    executor = _make_executor()
    executor._tool_context.execute_tool_calls = AsyncMock(
        return_value=[{"name": "read", "success": True}]
    )
    executor._tool_context._handle_tool_calls = AsyncMock(
        side_effect=AssertionError("legacy _handle_tool_calls bridge should not be used")
    )

    result = await executor._execute_tool_calls([{"name": "read", "arguments": {}}])

    assert result == [{"name": "read", "success": True}]
    executor._tool_context.execute_tool_calls.assert_awaited_once_with(
        [{"name": "read", "arguments": {}}]
    )


@pytest.mark.asyncio
async def test_execute_tool_calls_requires_canonical_tool_context_method():
    chat_context = MagicMock()
    chat_context.settings = MagicMock()
    chat_context.add_message = MagicMock()
    chat_context.conversation = MagicMock()

    legacy_handle_tool_calls = AsyncMock(return_value=[{"name": "read", "success": True}])
    tool_context = SimpleNamespace(
        tool_calls_used=0,
        tool_budget=10,
        tool_selector=MagicMock(),
        use_semantic_selection=False,
        _handle_tool_calls=legacy_handle_tool_calls,
    )

    executor = TurnExecutor(
        chat_context=chat_context,
        tool_context=tool_context,
        provider_context=MagicMock(provider_name="ollama", model="test-model"),
        execution_provider=MagicMock(),
    )

    with pytest.raises(AttributeError, match="execute_tool_calls"):
        await executor._execute_tool_calls([{"name": "read", "arguments": {}}])

    legacy_handle_tool_calls.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_via_agentic_loop_passes_conversation_history_to_loop():
    messages = [
        Message(role="system", content="system prompt"),
        Message(role="user", content="previous"),
        Message(role="assistant", content="previous answer"),
        Message(role="user", content="hello"),
    ]
    conversation = SimpleNamespace(messages=messages)
    chat_context = SimpleNamespace(
        settings=SimpleNamespace(chat_max_iterations=5),
        conversation=conversation,
        messages=messages,
        add_message=MagicMock(),
        _cumulative_token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    tool_context = SimpleNamespace(tool_calls_used=0, tool_budget=10)
    provider_context = SimpleNamespace(
        provider_name="ollama",
        model="test-model",
        task_classifier=SimpleNamespace(
            classify=MagicMock(return_value=SimpleNamespace(tool_budget=2))
        ),
    )
    executor = TurnExecutor(
        chat_context=chat_context,
        tool_context=tool_context,
        provider_context=provider_context,
        execution_provider=MagicMock(),
    )
    executor._is_question_only = MagicMock(return_value=False)
    executor._run_parallel_exploration = AsyncMock()

    response = CompletionResponse(content="done", role="assistant")
    loop_instance = MagicMock()

    async def _loop_run(query: str, context=None, conversation_history=None):
        assert query == "hello"
        assert context == {
            "_task_classification": provider_context.task_classifier.classify.return_value,
            "_is_qa_task": False,
        }
        assert conversation_history == [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "previous"},
            {"role": "assistant", "content": "previous answer"},
        ]
        return SimpleNamespace(
            iterations=[SimpleNamespace(action_result=SimpleNamespace(response=response))]
        )

    loop_instance.run = AsyncMock(side_effect=_loop_run)

    with patch("victor.framework.agentic_loop.AgenticLoop", return_value=loop_instance):
        result = await executor._execute_via_agentic_loop("hello", max_iterations=5)

    assert result is response
    executor._run_parallel_exploration.assert_awaited_once_with(
        "hello",
        provider_context.task_classifier.classify.return_value,
    )


@pytest.mark.asyncio
async def test_execute_agentic_loop_restores_state_before_reraising_failure():
    messages = []

    class FakeConversation:
        def __init__(self):
            self.messages = messages
            self._system_added = False

        def ensure_system_prompt(self):
            if not self.messages or self.messages[0].role != "system":
                self.messages.insert(0, Message(role="system", content="system prompt"))
            self._system_added = True

    conversation = FakeConversation()

    def _add_message(role: str, content: str) -> None:
        messages.append(Message(role=role, content=content))

    chat_context = SimpleNamespace(
        settings=SimpleNamespace(chat_max_iterations=5),
        conversation=conversation,
        messages=messages,
        add_message=MagicMock(side_effect=_add_message),
        _cumulative_token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    tool_context = SimpleNamespace(tool_calls_used=7, tool_budget=10)
    provider_context = SimpleNamespace(
        provider=SimpleNamespace(supports_tools=MagicMock(return_value=False)),
        provider_name="ollama",
        model="test-model",
        thinking=False,
        task_classifier=SimpleNamespace(
            classify=MagicMock(return_value=SimpleNamespace(tool_budget=2))
        ),
    )
    executor = TurnExecutor(
        chat_context=chat_context,
        tool_context=tool_context,
        provider_context=provider_context,
        execution_provider=MagicMock(),
    )
    executor._is_question_only = MagicMock(return_value=True)
    executor._run_parallel_exploration = AsyncMock()
    executor._check_context_compaction = AsyncMock()

    loop_instance = MagicMock()

    async def _loop_run(query: str, context=None, conversation_history=None):
        assert query == "hello"
        chat_context.add_message("assistant", "partial delegated answer")
        tool_context.tool_calls_used = 4
        raise RuntimeError("delegated loop failed mid-turn")

    loop_instance.run = AsyncMock(side_effect=_loop_run)

    with (
        patch("victor.framework.agentic_loop.AgenticLoop", return_value=loop_instance),
        pytest.raises(RuntimeError, match="delegated loop failed mid-turn"),
    ):
        await executor.execute_agentic_loop("hello", max_iterations=5)

    assert tool_context.tool_calls_used == 7
    assert [message.role for message in messages] == ["system", "user"]
    assert [message.content for message in messages] == ["system prompt", "hello"]


@pytest.mark.asyncio
async def test_execute_turn_applies_runtime_overrides_and_restores_state():
    class FakeToolService:
        def __init__(self, budget: int):
            self.budget = budget
            self.history = []

        def get_tool_budget(self) -> int:
            return self.budget

        def set_tool_budget(self, budget: int) -> None:
            self.history.append(budget)
            self.budget = budget

    tool_service = FakeToolService(budget=8)
    messages = []
    observed_tool_contexts = []
    settings = SimpleNamespace(chat_max_iterations=9)
    fake_orchestrator = SimpleNamespace(
        _tool_context_cache=None,
        code_manager="code-manager",
        provider="provider",
        model="model",
        tools="tool-registry",
        workflow_registry="workflow-registry",
        settings=settings,
        tool_budget=8,
        _tool_service=tool_service,
        _tool_pipeline=SimpleNamespace(config=SimpleNamespace(tool_budget=8)),
    )
    chat_context = SimpleNamespace(
        settings=settings,
        add_message=MagicMock(),
        conversation=SimpleNamespace(message_count=lambda: 0),
        messages=messages,
        _cumulative_token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        _orchestrator=fake_orchestrator,
    )
    tool_context = SimpleNamespace(
        tool_calls_used=0,
        tool_budget=8,
        tool_selector=SimpleNamespace(
            select_tools=AsyncMock(return_value=[{"name": "read"}]),
            prioritize_by_stage=MagicMock(return_value=[{"name": "read"}]),
        ),
        use_semantic_selection=False,
        _tool_service=tool_service,
    )

    async def _execute_tool_calls(_tool_calls):
        observed_tool_contexts.append(AgentOrchestrator._get_tool_context(fake_orchestrator))
        return [{"tool_name": "read", "success": True}]

    tool_context.execute_tool_calls = AsyncMock(side_effect=_execute_tool_calls)
    provider_context = SimpleNamespace(
        provider=SimpleNamespace(supports_tools=MagicMock(return_value=True)),
        provider_name="ollama",
        model="test-model",
        temperature=0.2,
        max_tokens=1024,
        thinking=False,
    )
    execution_provider = MagicMock()
    execution_provider.execute_turn = AsyncMock(
        return_value=CompletionResponse(
            content="Need to inspect files",
            role="assistant",
            tool_calls=[{"name": "read", "arguments": {"path": "victor/framework/agentic_loop.py"}}],
        )
    )
    executor = TurnExecutor(
        chat_context=chat_context,
        tool_context=tool_context,
        provider_context=provider_context,
        execution_provider=execution_provider,
    )

    result = await executor.execute_turn(
        user_message="inspect the runtime path",
        runtime_context_overrides={
            "provider_hint": "smart-router",
            "execution_mode": "team_execution",
            "topology_action": "team_plan",
            "tool_budget": 2,
            "iteration_budget": 3,
            "formation_hint": "parallel",
            "max_workers": 2,
            "topology_metadata": {"source": "unit-test"},
        },
    )

    assert result.has_tool_calls is True
    provider_kwargs = execution_provider.execute_turn.await_args.kwargs
    assert provider_kwargs["provider_hint"] == "smart-router"
    assert provider_kwargs["execution_mode"] == "team_execution"
    assert provider_kwargs["topology_action"] == "team_plan"
    assert observed_tool_contexts[0]["formation_hint"] == "parallel"
    assert observed_tool_contexts[0]["max_workers"] == 2
    assert observed_tool_contexts[0]["tool_budget"] == 2
    assert fake_orchestrator.tool_budget == 8
    assert tool_service.budget == 8
    assert tool_service.history == [2, 8]
    assert fake_orchestrator._tool_pipeline.config.tool_budget == 8
    assert settings.chat_max_iterations == 9
    assert not hasattr(fake_orchestrator, "_runtime_tool_context_overrides")
    assert not hasattr(chat_context, "_runtime_context_overrides")
