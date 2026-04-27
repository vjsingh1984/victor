import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.chat_stream_runtime import ServiceStreamingRuntime
from victor.agent.topology_contract import (
    TopologyAction,
    TopologyDecision,
    TopologyDecisionInput,
    TopologyGroundingRequirements,
    TopologyKind,
)
from victor.agent.topology_grounder import GroundedTopologyPlan
from victor.framework.task import TaskComplexity
from victor.providers.base import StreamChunk


def _make_orchestrator_stub():
    orch = MagicMock()
    orch.has_capability.return_value = False
    orch.get_capability_value.return_value = None
    orch._cumulative_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    orch._conversation_controller = MagicMock()
    orch.messages = []
    return orch


def test_service_streaming_runtime_caches_pipeline(monkeypatch):
    orch = _make_orchestrator_stub()
    runtime = ServiceStreamingRuntime(orch)
    created = []

    class DummyPipeline:
        pass

    def fake_factory(owner, **kwargs):
        created.append((owner, kwargs))
        return DummyPipeline()

    streaming_module = importlib.import_module("victor.agent.streaming")

    monkeypatch.setattr(streaming_module, "create_streaming_chat_pipeline", fake_factory)

    first = runtime.get_pipeline()
    second = runtime.get_pipeline()

    assert first is second
    assert len(created) == 1
    owner, kwargs = created[0]
    assert owner is runtime
    assert kwargs["perception"] is None
    assert kwargs["fulfillment"] is None
    assert kwargs["runtime_intelligence"] is orch._runtime_intelligence


@pytest.mark.asyncio
async def test_service_streaming_runtime_supports_protocol_adapter_host(monkeypatch):
    orch = _make_orchestrator_stub()
    adapter = OrchestratorProtocolAdapter(orch)
    runtime = ServiceStreamingRuntime(adapter)
    chunk = StreamChunk(content="service", is_final=True)

    class DummyPipeline:
        async def run(self, user_message: str, **kwargs):
            assert user_message == "hello"
            assert kwargs == {"mode": "test"}
            yield chunk

    def fake_factory(owner, **kwargs):
        assert owner is runtime
        return DummyPipeline()

    streaming_module = importlib.import_module("victor.agent.streaming")
    monkeypatch.setattr(streaming_module, "create_streaming_chat_pipeline", fake_factory)

    ctx = SimpleNamespace(
        cumulative_usage={
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
        runtime_override_snapshot=None,
    )
    orch._current_stream_context = ctx

    chunks = [item async for item in runtime.stream_chat("hello", mode="test")]

    assert chunks == [chunk]
    assert orch._cumulative_token_usage["prompt_tokens"] == 2
    assert orch._cumulative_token_usage["completion_tokens"] == 3
    assert orch._cumulative_token_usage["total_tokens"] == 5
    assert orch._current_stream_context is None


@pytest.mark.asyncio
async def test_service_streaming_runtime_stream_chat_uses_pipeline(monkeypatch):
    orch = _make_orchestrator_stub()
    runtime = ServiceStreamingRuntime(orch)
    chunk = StreamChunk(content="service", is_final=True)

    class DummyPipeline:
        def __init__(self):
            self.calls = []

        async def run(self, user_message: str, **kwargs):
            self.calls.append((user_message, kwargs))
            yield chunk

    pipeline = DummyPipeline()

    def fake_factory(owner, **kwargs):
        return pipeline

    streaming_module = importlib.import_module("victor.agent.streaming")

    monkeypatch.setattr(streaming_module, "create_streaming_chat_pipeline", fake_factory)

    chunks = [item async for item in runtime.stream_chat("hello", mode="test")]

    assert chunks == [chunk]
    assert pipeline.calls == [("hello", {"mode": "test"})]


@pytest.mark.asyncio
async def test_service_streaming_runtime_create_stream_context_uses_blocked_threshold_setting():
    orch = _make_orchestrator_stub()
    orch.settings = SimpleNamespace(recovery_blocked_consecutive_threshold=7)
    orch._classify_task_keywords.return_value = {}
    orch._tool_planner = SimpleNamespace(infer_goals_from_message=lambda _: [])
    orch.tool_budget = 200
    orch.tool_calls_used = 0
    orch._task_completion_detector = None

    runtime = ServiceStreamingRuntime(orch)
    runtime._prepare_stream = AsyncMock(
        return_value=(
            SimpleNamespace(),
            0.0,
            0.0,
            {},
            30,
            10,
            0,
            False,
            SimpleNamespace(value="default"),
            None,
            None,
        )
    )

    ctx = await runtime._create_stream_context("hello")

    assert ctx.max_blocked_before_force == 7


@pytest.mark.asyncio
async def test_service_streaming_runtime_create_stream_context_applies_topology_overrides(
    monkeypatch,
):
    orch = _make_orchestrator_stub()
    orch.settings = SimpleNamespace(
        recovery_blocked_consecutive_threshold=7,
        enable_topology_routing=True,
    )
    orch._classify_task_keywords.return_value = {
        "coarse_task_type": "analysis",
        "is_analysis_task": True,
    }
    orch._tool_planner = SimpleNamespace(infer_goals_from_message=lambda _: ["inspect"])
    orch.tool_budget = 200
    orch.tool_calls_used = 0
    orch._task_completion_detector = None
    orch.messages = []
    orch.model = "test-model"
    orch.provider = SimpleNamespace()
    orch.task_coordinator = SimpleNamespace(tool_budget=200)
    orch._runtime_intelligence = SimpleNamespace(
        get_topology_routing_context=MagicMock(
            return_value={
                "learned_topology_action": "team_plan",
                "learned_provider_hint": "anthropic",
                "learned_formation_hint": "parallel",
                "learned_topology_support": 0.75,
            }
        )
    )
    orch._tool_service = SimpleNamespace(
        get_tool_budget=lambda: 200,
        set_tool_budget=MagicMock(),
    )
    orch._tool_pipeline = SimpleNamespace(config=SimpleNamespace(tool_budget=200))

    runtime = ServiceStreamingRuntime(orch)
    runtime._prepare_stream = AsyncMock(
        return_value=(
            SimpleNamespace(),
            0.0,
            0.0,
            {},
            30,
            10,
            0,
            False,
            SimpleNamespace(value="analysis"),
            SimpleNamespace(complexity=TaskComplexity.COMPLEX, task_type="design"),
            6,
        )
    )
    runtime._get_stream_topology_provider_hints = AsyncMock(
        return_value={"provider_hint": "smart-router", "fallback_chain": ["smart-router"]}
    )
    runtime._paradigm_router = SimpleNamespace(
        build_topology_input=MagicMock(
            return_value=TopologyDecisionInput(
                query="hello",
                task_type="design",
                task_complexity="high",
                tool_budget=6,
                iteration_budget=3,
                available_team_formations=["parallel", "hierarchical"],
                provider_candidates=["smart-router"],
            )
        )
    )
    runtime._topology_selector = SimpleNamespace(
        select=MagicMock(
            return_value=TopologyDecision(
                action=TopologyAction.TEAM_PLAN,
                topology=TopologyKind.TEAM,
                confidence=0.8,
                rationale="Deep task favors a team plan.",
                grounding_requirements=TopologyGroundingRequirements(
                    provider="smart-router",
                    formation="parallel",
                    max_workers=3,
                    tool_budget=4,
                    iteration_budget=2,
                ),
                provider="smart-router",
                formation="parallel",
            )
        )
    )
    runtime._topology_grounder = SimpleNamespace(
        ground=MagicMock(
            return_value=GroundedTopologyPlan(
                action=TopologyAction.TEAM_PLAN,
                topology=TopologyKind.TEAM,
                execution_mode="team_execution",
                provider="smart-router",
                formation="parallel",
                max_workers=3,
                tool_budget=4,
                iteration_budget=2,
            )
        )
    )

    helpers_module = importlib.import_module("victor.agent.services.chat_stream_helpers")
    emit_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(helpers_module, "emit_topology_telemetry_event", emit_mock)

    ctx = await runtime._create_stream_context("hello")

    assert ctx.topology_plan["execution_mode"] == "team_execution"
    assert ctx.provider_kwargs["provider_hint"] == "smart-router"
    assert ctx.runtime_context_overrides["formation_hint"] == "parallel"
    assert ctx.tool_budget == 4
    assert ctx.max_total_iterations == 2
    assert orch.tool_budget == 4
    assert orch._runtime_tool_context_overrides["max_workers"] == 3
    assert len(ctx.topology_events) == 1
    topology_context = runtime._paradigm_router.build_topology_input.call_args.kwargs["context"]
    assert topology_context["learned_topology_action"] == "team_plan"
    assert topology_context["learned_provider_hint"] == "anthropic"
    learned_scope_context = (
        orch._runtime_intelligence.get_topology_routing_context.call_args.kwargs["scope_context"]
    )
    assert orch._runtime_intelligence.get_topology_routing_context.call_args.kwargs["query"] == (
        "hello"
    )
    assert learned_scope_context["task_type"] == "design"
    assert learned_scope_context["provider_hint"] == "smart-router"
    emit_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_service_streaming_runtime_stream_chat_restores_runtime_overrides(monkeypatch):
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

    orch = _make_orchestrator_stub()
    orch.tool_budget = 9
    orch.task_coordinator = SimpleNamespace(tool_budget=9)
    orch._tool_service = FakeToolService(budget=9, used=3)
    orch._tool_pipeline = SimpleNamespace(config=SimpleNamespace(tool_budget=9))
    orch._conversation_controller.messages = [SimpleNamespace(content="abc")]
    orch._runtime_intelligence = SimpleNamespace(record_topology_outcome=MagicMock())
    orch.has_capability.side_effect = lambda name: name == "current_stream_context"

    runtime = ServiceStreamingRuntime(orch)
    ctx = SimpleNamespace(
        cumulative_usage={
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
        last_quality_score=0.81,
        topology_events=[
            {
                "action": "team_plan",
                "topology": "team",
                "execution_mode": "team_execution",
                "provider": "smart-router",
                "formation": "parallel",
                "confidence": 0.82,
                "outcome": {"status": "planned"},
            }
        ],
        total_iterations=2,
        tool_calls_used=4,
        force_completion=False,
        has_substantial_content=lambda: True,
        runtime_override_snapshot=None,
    )
    ctx.runtime_override_snapshot = runtime._apply_stream_runtime_overrides(
        {"tool_budget": 4, "provider_hint": "smart-router"}
    )
    orch.get_capability_value.side_effect = lambda name: ctx if name == "current_stream_context" else None
    orch._current_stream_context = ctx

    class DummyPipeline:
        async def run(self, user_message: str, **kwargs):
            yield StreamChunk(content="service", is_final=True)

    runtime._streaming_pipeline = DummyPipeline()

    chunks = [item async for item in runtime.stream_chat("hello")]

    assert chunks == [StreamChunk(content="service", is_final=True)]
    assert orch.tool_budget == 9
    assert orch.task_coordinator.tool_budget == 9
    assert orch._tool_service.budget == 9
    assert orch._tool_service.history == [4, 9]
    assert "_runtime_tool_context_overrides" not in orch.__dict__
    orch._runtime_intelligence.record_topology_outcome.assert_called_once()
    feedback_payload = orch._runtime_intelligence.record_topology_outcome.call_args.args[0]
    assert feedback_payload["status"] == "completed"
    assert feedback_payload["completion_score"] == pytest.approx(0.81)
    assert ctx.topology_events[-1]["outcome"]["runtime"] == "streaming"
    assert ctx.runtime_override_snapshot is None
    assert orch._current_stream_context is None


@pytest.mark.asyncio
async def test_service_streaming_runtime_stream_chat_normalizes_recovery_events():
    orch = _make_orchestrator_stub()
    orch._runtime_intelligence = SimpleNamespace(record_topology_outcome=MagicMock())
    orch.has_capability.side_effect = lambda name: name == "current_stream_context"

    runtime = ServiceStreamingRuntime(orch)
    ctx = SimpleNamespace(
        cumulative_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        last_quality_score=0.78,
        topology_events=[],
        degradation_events=[
            {
                "source": "streaming_confidence",
                "kind": "confidence_early_stop",
                "post_degraded": False,
                "recovered": False,
            }
        ],
        recovery_events=[
            {
                "action": "retry",
                "failure_type": "PROVIDER_ERROR",
                "strategy_name": "retry_with_hint",
                "reason": "empty response loop",
                "confidence": 0.61,
                "iteration": 2,
            }
        ],
        total_iterations=2,
        tool_calls_used=0,
        force_completion=False,
        has_substantial_content=lambda: True,
        runtime_override_snapshot=None,
    )
    orch.get_capability_value.side_effect = lambda name: ctx if name == "current_stream_context" else None
    orch._current_stream_context = ctx

    class DummyPipeline:
        async def run(self, user_message: str, **kwargs):
            yield StreamChunk(content="service", is_final=True)

    runtime._streaming_pipeline = DummyPipeline()

    chunks = [item async for item in runtime.stream_chat("hello")]

    assert chunks == [StreamChunk(content="service", is_final=True)]
    assert len(ctx.degradation_events) == 2
    recovery_event = ctx.degradation_events[-1]
    assert recovery_event["source"] == "streaming_recovery"
    assert recovery_event["failure_type"] == "PROVIDER_ERROR"
    assert recovery_event["recovered"] is True
    assert recovery_event["post_degraded"] is False
    assert recovery_event["degradation_reasons"] == [
        "retry",
        "retry_with_hint",
        "empty response loop",
    ]
