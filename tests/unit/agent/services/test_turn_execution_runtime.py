"""Focused tests for TurnExecutor runtime behavior."""

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.services.turn_execution_runtime import TurnExecutor
from victor.agent.topology_contract import TopologyAction, TopologyKind
from victor.agent.topology_grounder import GroundedTopologyPlan
from victor.framework.task.protocols import TaskComplexity
from victor.framework.team_runtime import ResolvedTeamExecutionPlan
from victor.teams.types import TeamFormation
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


def test_resolve_orchestrator_prefers_explicit_runtime_owner():
    executor = _make_executor()
    direct_owner = SimpleNamespace(name="direct")
    fallback_owner = SimpleNamespace(name="fallback")
    executor._orchestrator = direct_owner
    executor._chat_context._orchestrator = fallback_owner

    assert executor._resolve_orchestrator() is direct_owner


def test_resolve_orchestrator_falls_back_to_chat_context_owner():
    executor = _make_executor()
    fallback_owner = SimpleNamespace(name="fallback")
    executor._chat_context._orchestrator = fallback_owner

    assert executor._resolve_orchestrator() is fallback_owner


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
async def test_select_tools_for_turn_delegates_intent_filtering_to_tool_planner():
    executor = _make_executor()
    executor._chat_context.messages = []
    executor._chat_context.conversation.message_count.return_value = 0
    executor._tool_context.use_semantic_selection = False
    executor._tool_context.tool_selector.select_tools = AsyncMock(
        return_value=[{"name": "shell"}, {"name": "write"}]
    )
    executor._tool_context.tool_selector.prioritize_by_stage.return_value = [
        {"name": "shell"},
        {"name": "write"},
    ]
    executor._tool_context._tool_planner = MagicMock()
    executor._tool_context._tool_planner.filter_tools_by_intent.return_value = [{"name": "shell"}]

    result = await executor._select_tools_for_turn(
        "use shell tool with sqlite commands to inspect the database",
        intent="read_only",
    )

    executor._tool_context._tool_planner.filter_tools_by_intent.assert_called_once_with(
        [{"name": "shell"}, {"name": "write"}],
        current_intent=ANY,
        user_message="use shell tool with sqlite commands to inspect the database",
    )
    assert result == [{"name": "shell"}]


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


@pytest.mark.asyncio
async def test_prepare_runtime_topology_delegates_parallel_exploration():
    executor = _make_executor()
    executor._run_parallel_exploration = AsyncMock(return_value=True)
    topology_plan = GroundedTopologyPlan(
        action=TopologyAction.PARALLEL_EXPLORATION,
        topology=TopologyKind.PARALLEL_EXPLORATION,
        execution_mode="parallel_exploration",
        tool_budget=4,
        iteration_budget=2,
    )
    task_classification = SimpleNamespace(complexity=TaskComplexity.COMPLEX)

    result = await executor.prepare_runtime_topology(
        topology_plan,
        user_message="inspect the runtime path",
        task_classification=task_classification,
    )

    executor._run_parallel_exploration.assert_awaited_once_with(
        "inspect the runtime path",
        task_classification,
        force=True,
        max_results_override=4,
    )
    assert result["action"] == "parallel_exploration"
    assert result["prepared"] is True
    assert result["execution_mode"] == "parallel_exploration"
    assert result["parallel_exploration"] == {"force": True, "max_results_override": 4}
    assert result["runtime_context_overrides"]["topology_action"] == "parallel_exploration"
    assert result["runtime_context_overrides"]["tool_budget"] == 4


@pytest.mark.asyncio
async def test_prepare_runtime_topology_resolves_framework_team_plan():
    executor = _make_executor()
    orchestrator = SimpleNamespace()
    executor._chat_context._orchestrator = orchestrator
    topology_plan = GroundedTopologyPlan(
        action=TopologyAction.TEAM_PLAN,
        topology=TopologyKind.TEAM,
        execution_mode="team_execution",
        formation="parallel",
        max_workers=2,
        tool_budget=6,
        iteration_budget=2,
    )
    task_classification = SimpleNamespace(task_type="feature", complexity=TaskComplexity.COMPLEX)

    with patch(
        "victor.framework.topology_runtime.resolve_configured_team",
        return_value=ResolvedTeamExecutionPlan(
            team_name="feature_team",
            display_name="Feature Team",
            formation=TeamFormation.PARALLEL,
            member_count=2,
            total_tool_budget=6,
            max_iterations=25,
            max_workers=2,
        ),
    ) as resolve_team:
        result = await executor.prepare_runtime_topology(
            topology_plan,
            user_message="implement the feature with a team",
            task_classification=task_classification,
        )

    resolve_team.assert_called_once_with(
        orchestrator,
        task_type="feature",
        complexity="complex",
        preferred_team=None,
        preferred_formation="parallel",
        max_workers=2,
        tool_budget=6,
    )
    assert result["action"] == "team_plan"
    assert result["prepared"] is True
    assert result["execution_mode"] == "team_execution"
    assert result["team_name"] == "feature_team"
    assert result["display_name"] == "Feature Team"
    assert result["formation"] == "parallel"
    assert result["member_count"] == 2
    assert result["runtime_context_overrides"]["team_name"] == "feature_team"
    assert result["runtime_context_overrides"]["team_display_name"] == "Feature Team"
    assert result["runtime_context_overrides"]["formation_hint"] == "parallel"
    assert result["runtime_context_overrides"]["max_workers"] == 2
    assert result["runtime_context_overrides"]["execution_mode"] == "team_execution"
    assert result["runtime_context_overrides"]["topology_action"] == "team_plan"


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
        def __init__(self, budget: int, used: int = 0):
            self.budget = budget
            self.used = used
            self.history = []

        def get_tool_budget(self) -> int:
            return max(0, self.budget - self.used)

        def get_remaining_budget(self) -> int:
            return self.get_tool_budget()

        def set_tool_budget(self, budget: int) -> None:
            self.history.append(budget)
            self.budget = budget

    tool_service = FakeToolService(budget=8, used=1)
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
            tool_calls=[
                {"name": "read", "arguments": {"path": "victor/framework/agentic_loop.py"}}
            ],
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
    assert tool_service.get_remaining_budget() == 7
    assert tool_service.history == [2, 8]
    assert fake_orchestrator._tool_pipeline.config.tool_budget == 8
    assert settings.chat_max_iterations == 9
    assert not hasattr(fake_orchestrator, "_runtime_tool_context_overrides")
    assert not hasattr(chat_context, "_runtime_context_overrides")


@pytest.mark.asyncio
async def test_execute_turn_injects_recovery_guidance_for_blocked_tool_batches():
    executor = _make_executor()
    executor._provider_context.provider = SimpleNamespace(
        supports_tools=MagicMock(return_value=True)
    )
    executor._provider_context.thinking = False
    executor._tool_context.tool_calls_used = 0
    executor._tool_context.tool_budget = 8
    executor._tool_context._tool_pipeline = SimpleNamespace(last_batch_effectively_blocked=True)
    executor._select_tools_for_turn = AsyncMock(return_value=[{"name": "read"}])
    executor._check_context_compaction = AsyncMock()
    executor._execute_model_turn = AsyncMock(
        return_value=CompletionResponse(
            content="",
            role="assistant",
            tool_calls=[{"name": "read", "arguments": {"path": "victor/core/container.py"}}],
        )
    )
    executor._execute_tool_calls = AsyncMock(
        return_value=[
            {
                "name": "read",
                "success": False,
                "error": "File already read",
                "skipped": True,
                "follow_up_suggestions": [
                    {
                        "tool": "overview",
                        "command": "overview(path='victor/core', max_depth=2)",
                        "arguments": {"path": "victor/core", "max_depth": 2},
                        "description": "Inspect the nearby module structure first.",
                        "reason": "Inspect the nearby module structure first.",
                    },
                    {
                        "tool": "graph",
                        "command": "graph(mode='overview', path='victor/core', top_k=5)",
                        "arguments": {"mode": "overview", "path": "victor/core", "top_k": 5},
                        "description": "Use the graph overview to find a different file to read.",
                        "reason": "Use the graph overview to find a different file to read.",
                    },
                ],
                "outcome_kind": "duplicate_read",
                "block_source": "session_read_dedup",
            }
        ]
    )

    result = await executor.execute_turn("inspect the container wiring")

    assert [suggestion["tool"] for suggestion in result.follow_up_suggestions] == [
        "overview",
        "graph",
    ]
    executor._chat_context.add_message.assert_called_once()
    role, content = executor._chat_context.add_message.call_args.args
    assert role == "user"
    assert "[Tool recovery guidance]" in content
    assert "overview(path='victor/core', max_depth=2)" in content
    assert "graph(mode='overview', path='victor/core', top_k=5)" in content


@pytest.mark.asyncio
async def test_execute_turn_applies_and_restores_system_prompt_override():
    executor = _make_executor()
    executor._provider_context.provider = SimpleNamespace(
        supports_tools=MagicMock(return_value=False)
    )
    executor._provider_context.thinking = False
    executor._tool_context.tool_calls_used = 0
    executor._tool_context.tool_budget = 8
    executor._chat_context.conversation.system_prompt = "Base prompt"
    executor._chat_context.conversation._system_added = True
    executor._chat_context.set_system_prompt = MagicMock()
    executor._check_context_compaction = AsyncMock()
    executor._execute_model_turn = AsyncMock(
        return_value=CompletionResponse(
            content="Done",
            role="assistant",
            tool_calls=[],
        )
    )

    await executor.execute_turn(
        user_message="respond with runtime prompt",
        runtime_context_overrides={"system_prompt": "Runtime prompt"},
    )

    assert executor._chat_context.set_system_prompt.call_args_list[0].args == ("Runtime prompt",)
    assert executor._chat_context.set_system_prompt.call_args_list[-1].args == ("Base prompt",)
    assert executor._chat_context.conversation._system_added is True


@pytest.mark.asyncio
async def test_execute_turn_deduplicates_repeated_recovery_guidance():
    executor = _make_executor()
    executor._provider_context.provider = SimpleNamespace(
        supports_tools=MagicMock(return_value=True)
    )
    executor._provider_context.thinking = False
    executor._tool_context.tool_calls_used = 0
    executor._tool_context.tool_budget = 8
    executor._tool_context._tool_pipeline = SimpleNamespace(last_batch_effectively_blocked=True)
    executor._select_tools_for_turn = AsyncMock(return_value=[{"name": "read"}])
    executor._check_context_compaction = AsyncMock()
    executor._execute_model_turn = AsyncMock(
        return_value=CompletionResponse(
            content="",
            role="assistant",
            tool_calls=[{"name": "read", "arguments": {"path": "victor/core/container.py"}}],
        )
    )
    executor._execute_tool_calls = AsyncMock(
        return_value=[
            {
                "name": "read",
                "success": False,
                "error": "File already read",
                "skipped": True,
                "follow_up_suggestions": [
                    {
                        "tool": "overview",
                        "command": "overview(path='victor/core', max_depth=2)",
                        "arguments": {"path": "victor/core", "max_depth": 2},
                        "description": "Inspect the nearby module structure first.",
                        "reason": "Inspect the nearby module structure first.",
                    }
                ],
                "outcome_kind": "duplicate_read",
                "block_source": "session_read_dedup",
            }
        ]
    )

    await executor.execute_turn("inspect the container wiring")
    await executor.execute_turn("inspect the container wiring")

    assert executor._chat_context.add_message.call_count == 1
