from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Import subagents module to ensure it's loaded before monkeypatching
from victor.agent.subagents import orchestrator as subagents_orchestrator_module

from victor.workflows.adapters import (
    AdapterWorkflowState,
    WorkflowState as AdapterWorkflowStateAlias,
)
from victor.workflows.adapters import WorkflowToGraphAdapter
from victor.core.async_utils import run_sync as shared_run_sync, run_sync_in_thread
from victor.core.container import ServiceContainer
from victor.workflows.definition import (
    AgentNode,
    ComputeNode,
    ConditionNode,
    ParallelNode,
    TeamNodeWorkflow,
    TransformNode,
)
from victor.workflows.executors.agent import AgentNodeExecutor
from victor.workflows.executors.compute import ComputeNodeExecutor
from victor.workflows.executors.condition import ConditionNodeExecutor
from victor.workflows.executors.factory import NodeExecutorFactory
from victor.workflows.executors.parallel import ParallelNodeExecutor
from victor.workflows.executors.transform import TransformNodeExecutor
from victor.workflows.executors.registry import (
    clear_registered_workflow_node_executors,
    register_workflow_node_executor,
)
from victor.workflows.runtime_types import GraphNodeResult


@pytest.mark.asyncio
async def test_transform_executor_records_runtime_graph_node_result() -> None:
    executor = TransformNodeExecutor(context=None)
    node = TransformNode(
        id="double",
        name="Double",
        transform=lambda state: {"value": state["value"] * 2},
    )

    result = await executor.execute(node, {"value": 4})

    assert result["value"] == 8
    node_result = result["_node_results"]["double"]
    assert isinstance(node_result, GraphNodeResult)
    assert node_result.success is True
    assert node_result.output == {"transformed_keys": ["value"]}


@pytest.mark.asyncio
async def test_compute_executor_uses_input_mapping_and_output_key() -> None:
    executor = ComputeNodeExecutor(context=None)
    node = ComputeNode(
        id="compute",
        name="Compute",
        input_mapping={"symbol": "$ctx.value"},
        output_key="computed",
    )

    result = await executor.execute(node, {"value": "AAPL"})

    assert result["computed"] == {
        "status": "no_tools_executed",
        "params": {"symbol": "AAPL"},
    }
    node_result = result["_node_results"]["compute"]
    assert isinstance(node_result, GraphNodeResult)
    assert node_result.success is True
    assert node_result.output == result["computed"]


@pytest.mark.asyncio
async def test_condition_executor_records_passthrough_output() -> None:
    executor = ConditionNodeExecutor(context=None)
    node = ConditionNode(
        id="decide",
        name="Decide",
        condition=lambda state: "yes",
        branches={"yes": "done"},
    )

    result = await executor.execute(node, {})

    node_result = result["_node_results"]["decide"]
    assert isinstance(node_result, GraphNodeResult)
    assert node_result.success is True
    assert node_result.output == {"passthrough": True, "branches": ["yes"]}


@pytest.mark.asyncio
async def test_parallel_executor_records_failure_with_runtime_graph_node_result() -> None:
    executor = ParallelNodeExecutor(context=None)
    node = ParallelNode(
        id="parallel",
        name="Parallel",
        parallel_nodes=["a", "b"],
        join_strategy="all",
    )

    result = await executor.execute(
        node,
        {
            "_parallel_results": {
                "a": {"success": True},
                "b": {"success": False},
            }
        },
    )

    assert result["_error"] == "Not all parallel nodes succeeded"
    node_result = result["_node_results"]["parallel"]
    assert isinstance(node_result, GraphNodeResult)
    assert node_result.success is False
    assert node_result.output == {
        "parallel_nodes": ["a", "b"],
        "join_strategy": "all",
        "results_count": 2,
    }


@pytest.mark.asyncio
async def test_agent_executor_uses_output_key_and_runtime_graph_node_result() -> None:
    from unittest.mock import patch, AsyncMock

    executor = AgentNodeExecutor(context=None)
    fake_result = SimpleNamespace(summary="done", success=True, tool_calls_used=2, error=None)

    # Create a mock orchestrator to be returned by _get_orchestrator
    mock_orchestrator = SimpleNamespace()

    # Patch SubAgentOrchestrator at the source module
    # The execute() method imports it from victor.agent.subagents.orchestrator
    class MockSubAgentOrchestrator:
        def __init__(self, orchestrator):
            self.orchestrator = orchestrator
            # Add _context attribute needed for provider-aware context sizing (added April 2026)
            self._context = None

        async def spawn(self, **kwargs):
            return fake_result

    # Use patch context manager for cleaner setup/teardown
    with patch(
        "victor.agent.subagents.orchestrator.SubAgentOrchestrator", MockSubAgentOrchestrator
    ):
        # Also patch _get_orchestrator to return the mock orchestrator
        with patch.object(executor, "_get_orchestrator", return_value=mock_orchestrator):
            node = AgentNode(
                id="analyze",
                name="Analyze",
                role="researcher",
                goal="Review {{task}}",
                input_mapping={"task": "task"},
                output_key="agent_output",
            )

            result = await executor.execute(node, {"task": "repo"})

            assert result["agent_output"] is fake_result
            node_result = result["_node_results"]["analyze"]
            assert isinstance(node_result, GraphNodeResult)
            assert node_result.success is True
            assert node_result.output is fake_result
            assert node_result.tool_calls_used == 2


@pytest.mark.asyncio
async def test_agent_executor_returns_placeholder_without_orchestrator() -> None:
    executor = AgentNodeExecutor(context=None)
    node = AgentNode(
        id="analyze",
        name="Analyze",
        role="researcher",
        goal="Review {{task}}",
        input_mapping={"task": "task"},
    )

    result = await executor.execute(node, {"task": "repo"})

    assert result["analyze"]["status"] == "placeholder"
    node_result = result["_node_results"]["analyze"]
    assert isinstance(node_result, GraphNodeResult)
    assert node_result.success is True
    assert node_result.output["input_context"] == {"task": "repo"}


@pytest.mark.asyncio
async def test_team_executor_records_runtime_graph_node_result(monkeypatch) -> None:
    from victor.workflows.executors.team import TeamNodeExecutor

    class FakeTeamNode:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def execute_async(self, orchestrator, state):
            assert orchestrator is None
            return {
                **state,
                "team_result": {
                    "success": True,
                    "final_output": "ship-it",
                    "member_count": 1,
                },
            }

    monkeypatch.setattr("victor.framework.workflows.nodes.TeamNode", FakeTeamNode)

    executor = TeamNodeExecutor(context=None)
    node = TeamNodeWorkflow(
        id="team",
        name="Team",
        goal="Review $ctx.subject",
        members=[{"id": "lead", "role": "researcher", "goal": "Inspect"}],
        output_key="team_result",
    )

    result = await executor.execute(node, {"subject": "repo"})

    assert result["team_result"]["final_output"] == "ship-it"
    node_result = result["_node_results"]["team"]
    assert isinstance(node_result, GraphNodeResult)
    assert node_result.success is True
    assert node_result.output == result["team_result"]


@pytest.mark.asyncio
async def test_hitl_executor_records_response_and_rejection() -> None:
    from victor.workflows.executors.hitl import HITLNodeExecutor
    from victor.workflows.hitl import HITLFallback, HITLNode, HITLNodeType

    executor = HITLNodeExecutor(context=None)
    node = HITLNode(
        id="approve",
        name="Approve",
        hitl_type=HITLNodeType.APPROVAL,
        prompt="Approve deploy?",
        context_keys=["proposal"],
        fallback=HITLFallback.ABORT,
    )

    result = await executor.execute(
        node,
        {
            "proposal": "release",
            "_hitl_pending": True,
            "_hitl_response": {
                "approved": False,
                "reason": "needs changes",
                "status": "rejected",
            },
        },
    )

    assert result["_hitl_pending"] is False
    assert result["_error"] == "HITL node 'approve' rejected: needs changes"
    node_result = result["_node_results"]["approve"]
    assert isinstance(node_result, GraphNodeResult)
    assert node_result.success is False
    assert node_result.output["response"]["status"] == "rejected"


def test_adapter_workflow_state_alias_remains_available() -> None:
    assert AdapterWorkflowStateAlias is AdapterWorkflowState


def test_adapter_execution_handler_uses_shared_sync_bridge_without_running_loop() -> None:
    adapter = WorkflowToGraphAdapter()
    executor = SimpleNamespace()

    async def execute_node(node, context):
        return {"status": "ok", "context": context}

    executor.execute_node = execute_node
    node = TransformNode(id="transform", name="Transform", transform=lambda state: {})
    handler = adapter._create_execution_handler(node, executor)

    with patch(
        "victor.workflows.adapters.run_sync",
        side_effect=lambda coro: shared_run_sync(coro),
    ) as mock_run_sync:
        result = handler({"context": {"value": 1}})

    assert result["current_node"] == "Transform"
    assert result["results"]["Transform"] == {"status": "ok", "context": {"value": 1}}
    assert result["visited_nodes"] == ["Transform"]
    mock_run_sync.assert_called_once()


@pytest.mark.asyncio
async def test_adapter_execution_handler_uses_thread_bridge_with_running_loop() -> None:
    adapter = WorkflowToGraphAdapter()
    executor = SimpleNamespace()

    async def execute_node(node, context):
        return {"status": "ok", "context": context}

    executor.execute_node = execute_node
    node = TransformNode(id="transform", name="Transform", transform=lambda state: {})
    handler = adapter._create_execution_handler(node, executor)

    with patch(
        "victor.workflows.adapters.run_sync_in_thread",
        side_effect=lambda coro: run_sync_in_thread(coro),
    ) as mock_run_sync_in_thread:
        result = handler({"context": {"value": 2}})

    assert result["current_node"] == "Transform"
    assert result["results"]["Transform"] == {"status": "ok", "context": {"value": 2}}
    assert result["visited_nodes"] == ["Transform"]
    mock_run_sync_in_thread.assert_called_once()


def test_node_executor_factory_prefers_registered_executor_classes() -> None:
    class StubTransformExecutor:
        def __init__(self, context=None):
            self.context = context

        async def execute(self, node, state):
            assert self.context is not None
            assert self.context.services is not None
            return {**state, "executed_by_stub": node.id}

    factory = NodeExecutorFactory(container=ServiceContainer())
    factory.register_executor_type("transform", StubTransformExecutor, replace=True)
    node = TransformNode(id="transform", name="Transform", transform=lambda state: {})

    result = asyncio.run(factory.create_executor(node)({}))

    assert result == {"executed_by_stub": "transform"}


def test_node_executor_factory_bootstraps_builtin_executor_types() -> None:
    factory = NodeExecutorFactory()
    node = TransformNode(
        id="double",
        name="Double",
        transform=lambda state: {"value": state["value"] * 2},
    )

    result = asyncio.run(factory.create_executor(node)({"value": 4}))

    assert result["value"] == 8


def test_node_executor_factory_raises_for_unregistered_node_types() -> None:
    factory = NodeExecutorFactory()
    unknown_node = SimpleNamespace(node_type=SimpleNamespace(value="custom_unknown"))

    with pytest.raises(ValueError, match="Unsupported workflow node type 'custom_unknown'"):
        factory.create_executor(unknown_node)


def test_node_executor_factory_loads_custom_registry_registrations() -> None:
    clear_registered_workflow_node_executors()

    try:

        class CustomExecutor:
            def __init__(self, context=None):
                self.context = context

            async def execute(self, node, state):
                return {**state, "custom_registered": node.id}

        register_workflow_node_executor("custom_plugin", CustomExecutor)

        factory = NodeExecutorFactory()
        node = SimpleNamespace(
            id="custom",
            name="Custom",
            node_type=SimpleNamespace(value="custom_plugin"),
        )

        result = asyncio.run(factory.create_executor(node)({"value": 1}))

        assert factory.supports_node_type("custom_plugin")
        assert result["custom_registered"] == "custom"
    finally:
        clear_registered_workflow_node_executors()
