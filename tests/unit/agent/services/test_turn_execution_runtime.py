"""Focused tests for TurnExecutor runtime behavior."""

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from victor.agent.coordinators.state_context import CoordinatorResult, TransitionBatch
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.services.turn_execution_runtime import TurnExecutor, TurnResult
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
    chat_context._orchestrator = None

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
async def test_check_context_compaction_prefers_lifecycle_service():
    executor = _make_executor()
    lifecycle = SimpleNamespace(
        after_agent_turn=AsyncMock(
            return_value={
                "compacted": True,
                "messages_removed": 2,
                "tokens_freed": 80,
            }
        )
    )
    legacy_compactor = MagicMock()
    orchestrator = SimpleNamespace(
        _context_lifecycle_service=lifecycle,
        _context_compactor=legacy_compactor,
        active_session_id="session_root",
        agent_id="root_agent",
        display_name="Root Agent",
        get_messages=MagicMock(return_value=[{"role": "user", "content": "hello"}]),
    )
    executor._orchestrator = orchestrator
    executor._chat_context._context_compactor = legacy_compactor
    executor._tool_context.tool_calls_used = 0

    await executor._check_context_compaction(
        "hello",
        SimpleNamespace(complexity=TaskComplexity.COMPLEX),
    )

    lifecycle.after_agent_turn.assert_awaited_once()
    runtime_context = lifecycle.after_agent_turn.await_args.args[0]
    assert runtime_context.agent_id == "root_agent"
    assert runtime_context.session_id == "session_root"
    assert lifecycle.after_agent_turn.await_args.kwargs["messages"] == [
        {"role": "user", "content": "hello"}
    ]
    legacy_compactor.check_and_compact.assert_not_called()


@pytest.mark.asyncio
async def test_check_context_compaction_uses_context_service_before_legacy_compactor():
    executor = _make_executor()
    context_service = SimpleNamespace(
        get_compaction_recommendation=MagicMock(return_value={"should_compact": True}),
        compact_context=AsyncMock(return_value=3),
    )
    legacy_compactor = MagicMock()
    orchestrator = SimpleNamespace(
        _context_lifecycle_service=None,
        _context_service=context_service,
        _context_compactor=legacy_compactor,
        settings=SimpleNamespace(context_compaction_strategy="semantic"),
    )
    executor._orchestrator = orchestrator
    executor._chat_context._context_compactor = legacy_compactor
    executor._tool_context.tool_calls_used = 0

    await executor._check_context_compaction(
        "hello",
        SimpleNamespace(complexity=TaskComplexity.COMPLEX),
    )

    context_service.get_compaction_recommendation.assert_called_once()
    context_service.compact_context.assert_awaited_once_with(
        strategy="semantic",
        min_messages=6,
    )
    legacy_compactor.check_and_compact.assert_not_called()


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
async def test_parallel_exploration_lazily_materializes_service_runtime_fallback():
    explorer = MagicMock()
    explorer.explore_parallel = AsyncMock(
        return_value=SimpleNamespace(
            summary="shared helper exploration",
            file_paths=["victor/agent/services/exploration_runtime.py"],
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
            "victor.agent.services.exploration_runtime.ExplorationCoordinator",
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
async def test_parallel_exploration_prefers_state_passed_coordinator_from_orchestrator_facade():
    explorer = SimpleNamespace(
        explore=AsyncMock(
            return_value=CoordinatorResult(
                transitions=TransitionBatch()
                .update_state(
                    "explored_files",
                    ["victor/agent/coordinators/exploration_state_passed.py"],
                )
                .update_state(
                    "exploration_summary",
                    "state-passed exploration summary",
                )
                .update_state(
                    "exploration_metrics",
                    {
                        "duration_seconds": 0.6,
                        "tool_calls": 2,
                        "files_found": 1,
                    },
                ),
                metadata={
                    "file_paths": ["victor/agent/coordinators/exploration_state_passed.py"],
                    "summary": "state-passed exploration summary",
                    "tool_calls": 2,
                    "duration_seconds": 0.6,
                },
            )
        )
    )
    executor = _make_executor()
    executor._orchestrator = SimpleNamespace(
        _orchestration_facade=SimpleNamespace(exploration_state_passed=explorer),
        messages=[],
        session_id="turn-exec-session",
        conversation_stage="initial",
        settings=MagicMock(),
        model="test-model",
        provider_name="anthropic",
        max_tokens=4096,
        temperature=0.1,
        conversation_state={},
        session_state={},
        observed_files=[],
        _capabilities={"existing": True},
        add_message=MagicMock(),
    )
    task_classification = SimpleNamespace(complexity=TaskComplexity.ANALYSIS)

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
                tool_budget_per_agent=4,
                exploration_timeout=5,
            ),
        ),
        patch(
            "victor.agent.coordinators.factory_support.create_exploration_coordinator",
            side_effect=AssertionError("direct exploration helper should not be used"),
        ),
    ):
        await executor._run_parallel_exploration(
            "investigate the state-passed exploration path",
            task_classification,
        )

    explorer.explore.assert_awaited_once()
    snapshot = explorer.explore.await_args.args[0]
    assert snapshot.provider == "anthropic"
    assert snapshot.get_capability_value("task_complexity") == TaskComplexity.ANALYSIS.value
    assert str(explorer.explore.await_args.kwargs["project_root"]) == "/tmp/project"
    assert explorer.explore.await_args.kwargs["max_results"] == 4
    assert executor._orchestrator.conversation_state["explored_files"] == [
        "victor/agent/coordinators/exploration_state_passed.py"
    ]
    executor._chat_context.add_message.assert_called_once_with(
        "user",
        "[Parallel exploration results]\nstate-passed exploration summary",
        metadata={"source": "ag"},
    )
    assert executor._exploration_coordinator is explorer
    assert executor._exploration_done is True


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
async def test_execute_via_agentic_loop_marks_failed_loop_response_metadata():
    chat_context = SimpleNamespace(
        settings=SimpleNamespace(chat_max_iterations=5),
        conversation=SimpleNamespace(messages=[]),
        messages=[],
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

    response = CompletionResponse(content="partial", role="assistant")
    evaluation = SimpleNamespace(reason="Insufficient progress")
    loop_result = SimpleNamespace(
        success=False,
        iterations=[
            SimpleNamespace(
                action_result=SimpleNamespace(response=response),
                evaluation=evaluation,
            )
        ],
    )
    loop_instance = MagicMock()
    loop_instance.run = AsyncMock(return_value=loop_result)

    with patch("victor.framework.agentic_loop.AgenticLoop", return_value=loop_instance):
        result = await executor._execute_via_agentic_loop("hello", max_iterations=5)

    assert result is response
    assert result.metadata["agentic_loop_success"] is False
    assert result.metadata["agentic_loop_error"] == "Insufficient progress"


@pytest.mark.asyncio
async def test_execute_via_agentic_loop_synthesizes_after_tool_evidence_spin():
    chat_context = SimpleNamespace(
        settings=SimpleNamespace(chat_max_iterations=5),
        conversation=SimpleNamespace(messages=[]),
        messages=[Message(role="tool", content="Cargo workspace members: core, clients/rust")],
        add_message=MagicMock(),
        _cumulative_token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    tool_context = SimpleNamespace(tool_calls_used=0, tool_budget=10)
    response_completer = SimpleNamespace(
        ensure_response=AsyncMock(
            return_value=SimpleNamespace(
                content="Cargo.toml defines a Rust workspace with clients/rust included."
            )
        )
    )
    provider_context = SimpleNamespace(
        provider_name="ollama",
        model="test-model",
        temperature=0.2,
        max_tokens=1024,
        response_completer=response_completer,
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

    empty_response = CompletionResponse(content="", role="assistant")
    evaluation = SimpleNamespace(reason="Agent stuck: 3 turns without tool calls")
    loop_result = SimpleNamespace(
        success=False,
        iterations=[
            SimpleNamespace(
                action_result=TurnResult(
                    response=CompletionResponse(content="", role="assistant"),
                    tool_results=[{"tool_name": "read", "success": True}],
                    has_tool_calls=True,
                    tool_calls_count=1,
                ),
                evaluation=SimpleNamespace(reason="Successful tools produced execution evidence"),
            ),
            SimpleNamespace(
                action_result=SimpleNamespace(response=empty_response),
                evaluation=evaluation,
            ),
        ],
    )
    loop_instance = MagicMock()
    loop_instance.run = AsyncMock(return_value=loop_result)

    with patch("victor.framework.agentic_loop.AgenticLoop", return_value=loop_instance):
        result = await executor._execute_via_agentic_loop("hello", max_iterations=5)

    assert result.content == "Cargo.toml defines a Rust workspace with clients/rust included."
    assert result.metadata["agentic_loop_success"] is True
    assert result.metadata["agentic_loop_recovered"] is True
    assert result.metadata["agentic_loop_recovery_reason"] == (
        "Agent stuck: 3 turns without tool calls"
    )


@pytest.mark.asyncio
async def test_execute_turn_directly_runs_explicit_read_plan_step_without_model_call():
    executor = _make_executor()
    executor._tool_context.tool_calls_used = 0
    executor._tool_context.tool_budget = 10
    executor._check_context_compaction = AsyncMock()
    executor._execute_model_turn = AsyncMock(
        side_effect=AssertionError("model call should not be needed for explicit read step")
    )
    executor._execute_tool_calls = AsyncMock(
        return_value=[
            {
                "tool_name": "read",
                "success": True,
                "tool_call_id": "call_deterministic_test",
            }
        ]
    )

    result = await executor.execute_turn("Read root Cargo.toml to identify workspace members")

    executor._execute_model_turn.assert_not_awaited()
    executor._execute_tool_calls.assert_awaited_once()
    tool_call = executor._execute_tool_calls.await_args.args[0][0]
    assert tool_call["name"] == "read"
    assert tool_call["arguments"]["path"] == "Cargo.toml"
    assert result.has_tool_calls is True
    assert result.successful_tool_count == 1
    assistant_call = executor._chat_context.add_message.call_args_list[0]
    assert assistant_call.args == ("assistant", "")
    assert assistant_call.kwargs["tool_calls"][0]["name"] == "read"


@pytest.mark.asyncio
async def test_execute_turn_directly_runs_workspace_mapping_reads_without_model_call():
    executor = _make_executor()
    executor._tool_context.tool_calls_used = 0
    executor._tool_context.tool_budget = 10
    executor._check_context_compaction = AsyncMock()
    executor._execute_model_turn = AsyncMock(
        side_effect=AssertionError("model call should not be needed for workspace mapping reads")
    )
    executor._execute_tool_calls = AsyncMock(
        return_value=[
            {"tool_name": "read", "success": True},
            {"tool_name": "read", "success": True},
        ]
    )

    result = await executor.execute_turn(
        "Map Rust workspace structure: read root Cargo.toml and clients/rust/Cargo.toml "
        "to identify all workspace members, crate dependencies, feature flags, and shared "
        "configurations."
    )

    executor._execute_model_turn.assert_not_awaited()
    tool_calls = executor._execute_tool_calls.await_args.args[0]
    assert [call["arguments"]["path"] for call in tool_calls] == [
        "Cargo.toml",
        "clients/rust/Cargo.toml",
    ]
    assert result.has_tool_calls is True
    assert result.tool_calls_count == 2


@pytest.mark.asyncio
async def test_execute_turn_directly_runs_rust_inventory_without_model_call():
    executor = _make_executor()
    executor._tool_context.tool_calls_used = 0
    executor._tool_context.tool_budget = 10
    executor._check_context_compaction = AsyncMock()
    executor._execute_model_turn = AsyncMock(
        side_effect=AssertionError("model call should not be needed for Rust inventory")
    )
    executor._execute_tool_calls = AsyncMock(
        return_value=[
            {
                "tool_name": "shell",
                "name": "shell",
                "success": True,
                "content": "./src/lib.rs\n./tests/integration.rs",
            }
        ]
    )

    result = await executor.execute_turn(
        "Enumerate all Rust source directories and files across every workspace member "
        "using file inventory"
    )

    executor._execute_model_turn.assert_not_awaited()
    tool_calls = executor._execute_tool_calls.await_args.args[0]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "shell"
    assert tool_calls[0]["arguments"]["cmd"] == "find . -type f -name '*.rs' | sort"
    assert tool_calls[0]["arguments"]["readonly"] is True
    assert result.has_tool_calls is True
    assert result.tool_calls_count == 1
    assert "Deterministic read-only execution completed" in result.content
    assert "./src/lib.rs" in result.content


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

    def _add_message(role: str, content: str, **kwargs) -> None:
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
async def test_execute_agentic_loop_passes_runtime_overrides_to_loop_context():
    executor = _make_executor()
    executor._chat_context.settings.chat_max_iterations = 5
    executor._provider_context.task_classifier = SimpleNamespace(
        classify=MagicMock(return_value=SimpleNamespace(tool_budget=1))
    )
    executor._is_question_only = MagicMock(return_value=True)

    loop_instance = MagicMock()
    runtime_context_overrides = {
        "prompt_overlays": [
            {
                "name": "test.scoped_prompt",
                "content": "Scoped prompt",
                "placement": "turn_prefix",
            }
        ]
    }

    async def _loop_run(query: str, context=None, conversation_history=None):
        assert query == "hello"
        assert context["runtime_context_overrides"] == runtime_context_overrides
        return SimpleNamespace(
            success=True,
            iterations=[
                SimpleNamespace(
                    action_result=TurnResult(
                        response=CompletionResponse(
                            content="done",
                            role="assistant",
                            tool_calls=[],
                        ),
                        tool_results=[],
                        has_tool_calls=False,
                        tool_calls_count=0,
                    )
                )
            ],
        )

    loop_instance.run = AsyncMock(side_effect=_loop_run)

    with patch("victor.framework.agentic_loop.AgenticLoop", return_value=loop_instance):
        response = await executor.execute_agentic_loop(
            "hello",
            max_iterations=5,
            runtime_context_overrides=runtime_context_overrides,
        )

    assert response.content == "done"


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
    assert executor._chat_context.add_message.call_count == 2
    assistant_role, assistant_content = executor._chat_context.add_message.call_args_list[0].args
    assert assistant_role == "assistant"
    assert assistant_content == ""
    assert executor._chat_context.add_message.call_args_list[0].kwargs["tool_calls"] == [
        {"name": "read", "arguments": {"path": "victor/core/container.py"}}
    ]
    role, content = executor._chat_context.add_message.call_args_list[1].args
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

    assert executor._chat_context.add_message.call_count == 3
    assert [call.args[0] for call in executor._chat_context.add_message.call_args_list] == [
        "assistant",
        "user",
        "assistant",
    ]
