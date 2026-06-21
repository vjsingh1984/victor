# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the extracted per-turn guard factory (FEP-0007 Phase 2 step)."""

from types import SimpleNamespace

from victor.agent.services import chat_stream_executor
from victor.agent.services.chat_stream_executor import StreamingChatExecutor
from victor.agent.turn_policy import (
    NudgePolicy,
    PlateauDetector,
    SpinDetector,
    TurnEvaluationController,
)
from victor.framework.search_novelty import SearchNoveltyTracker


def test_create_stream_turn_guards_builds_expected_components():
    guards = StreamingChatExecutor._create_stream_turn_guards(SimpleNamespace(settings=None))
    assert isinstance(guards.spin, SpinDetector)
    assert isinstance(guards.nudge_policy, NudgePolicy)
    assert isinstance(guards.turn_eval, TurnEvaluationController)
    assert isinstance(guards.plateau, PlateauDetector)
    assert isinstance(guards.novelty, SearchNoveltyTracker)
    # The controller shares the same spin/nudge instances (per-loop wiring).
    assert guards.turn_eval.spin_detector is guards.spin
    # Plateau/budget/controller-novelty are disabled (this loop runs its own post-tool checks).
    assert guards.turn_eval._enable_plateau_nudge is False
    assert guards.turn_eval._enable_search_novelty is False
    assert guards.novelty_enabled is True  # default when no settings


def test_create_stream_turn_guards_honors_novelty_off_switch():
    orch = SimpleNamespace(
        settings=SimpleNamespace(exploration=SimpleNamespace(search_novelty_guard_enabled=False))
    )
    guards = StreamingChatExecutor._create_stream_turn_guards(orch)
    assert guards.novelty_enabled is False


def test_create_stream_turn_guards_applies_novelty_thresholds():
    orch = SimpleNamespace(
        settings=SimpleNamespace(
            exploration=SimpleNamespace(
                search_novelty_guard_enabled=True,
                novelty_consecutive_low_limit=2,
                novelty_min_search_turns=1,
            )
        )
    )
    guards = StreamingChatExecutor._create_stream_turn_guards(orch)
    assert guards.novelty._config.consecutive_low_novelty_limit == 2
    assert guards.novelty._config.min_search_turns == 1


async def test_extract_task_requirements_sets_orch_state():
    # A plain prompt yields no required files/outputs -> no event emit, no bus needed.
    orch = SimpleNamespace(
        _read_files_session={"stale.py"},
        _all_files_read_nudge_sent=True,
    )
    await StreamingChatExecutor._extract_task_requirements(orch, "hello there")
    assert isinstance(orch._required_files, list)
    assert isinstance(orch._required_outputs, list)
    assert orch._read_files_session == set()  # cleared
    assert orch._all_files_read_nudge_sent is False


def _guidance_orch(messages):
    return SimpleNamespace(
        _apply_intent_guard=lambda msg: messages.setdefault("intent", []).append(msg),
        _apply_task_guidance=lambda *a: messages.setdefault("task_guidance", []).append(a),
        add_message=lambda role, content, metadata=None: messages.setdefault("msg", []).append(
            content
        ),
    )


def test_apply_run_guidance_action_task_injects_guidance():
    messages = {}
    stream_ctx = SimpleNamespace(
        is_analysis_task=False,
        unified_task_type=SimpleNamespace(value="create"),
        coarse_task_type="action",
        is_action_task=True,
        needs_execution=True,
    )
    StreamingChatExecutor._apply_run_guidance(
        _guidance_orch(messages), stream_ctx, "build a script", 12
    )
    assert messages["intent"] == ["build a script"]
    assert len(messages["task_guidance"]) == 1
    assert any("ACTION-GUIDANCE" in c for c in messages.get("msg", []))


def test_apply_run_guidance_non_action_skips_message():
    messages = {}
    stream_ctx = SimpleNamespace(
        is_analysis_task=True,
        unified_task_type=SimpleNamespace(value="analyze"),
        coarse_task_type="analysis",
        is_action_task=False,
        needs_execution=False,
    )
    StreamingChatExecutor._apply_run_guidance(
        _guidance_orch(messages), stream_ctx, "explain the code", 50
    )
    assert messages.get("msg", []) == []  # no action-guidance for non-action tasks


def test_initialize_task_intent_returns_goals_and_seeds_ctx():
    seen = {}
    orch = SimpleNamespace(
        _tool_planner=SimpleNamespace(infer_goals_from_message=lambda m: ["goal1"]),
    )
    stream_ctx = SimpleNamespace(
        coarse_task_type="analysis",
        set_task_intent=lambda m: seen.__setitem__("intent", m),
        extend_plan_steps=lambda g: seen.__setitem__("steps", g),
        record_intent_event=lambda *a, **k: seen.__setitem__("event", True),
    )
    goals = StreamingChatExecutor._initialize_task_intent(orch, stream_ctx, "do the thing")
    assert goals == ["goal1"]  # returned for downstream tool planning
    assert seen["intent"] == "do the thing"
    assert seen["steps"] == ["goal1"]
    assert seen["event"] is True


def _provider_turn_executor():
    # Build an executor instance without running __init__ (we only exercise the
    # extracted ACT provider sub-step, which depends on no constructor state).
    return StreamingChatExecutor.__new__(StreamingChatExecutor)


async def test_stream_provider_turn_plans_tools_and_streams(monkeypatch):
    seen = {}

    class _Planner:
        def plan_tools(self, goals, available_inputs):
            seen["plan"] = (goals, available_inputs)
            return ["planned"]

    orch = SimpleNamespace(
        observed_files={"a.py"},
        _tool_planner=_Planner(),
        get_session_tools=lambda: None,
        thinking=False,
    )
    stream_ctx = SimpleNamespace(
        is_qa_task=False,
        context_msg="ctx",
        provider_kwargs={"execution_mode": "normal"},
    )

    async def _fake_stream(tools, provider_kwargs, stream_ctx):
        seen["stream"] = (tools, provider_kwargs)
        return ("hello", [{"name": "edit"}], 1.5, False)

    runtime_owner = SimpleNamespace(_stream_provider_response=_fake_stream)

    executor = _provider_turn_executor()

    async def _fake_get_tools_cached(self, o, ctx_msg, g, planned_tools=None):
        seen["tools_cached"] = (ctx_msg, g, planned_tools)
        return ["resolved_tool"]

    monkeypatch.setattr(
        StreamingChatExecutor, "_get_tools_cached", _fake_get_tools_cached, raising=True
    )

    tools, full_content, tool_calls, garbage = await executor._stream_provider_turn(
        orch, runtime_owner, stream_ctx, ["goal1"]
    )

    # planned tools seeded onto the context and used for tool resolution.
    assert stream_ctx.planned_tools == ["planned"]
    assert seen["plan"] == (["goal1"], ["query", "file_contents"])
    assert seen["tools_cached"] == ("ctx", ["goal1"], ["planned"])
    # provider response returned verbatim (the discarded float is the token estimate).
    assert tools == ["resolved_tool"]
    assert full_content == "hello"
    assert tool_calls == [{"name": "edit"}]
    assert garbage is False


async def test_stream_provider_turn_qa_task_skips_tools_and_enables_thinking():
    orch = SimpleNamespace(
        observed_files=set(),
        _tool_planner=SimpleNamespace(plan_tools=lambda *a: None),
        get_session_tools=lambda: None,
        thinking=True,
    )
    stream_ctx = SimpleNamespace(
        is_qa_task=True,
        context_msg="ctx",
        provider_kwargs={},
    )

    captured = {}

    async def _fake_stream(tools, provider_kwargs, stream_ctx):
        captured["tools"] = tools
        captured["provider_kwargs"] = provider_kwargs
        return ("", None, 0.0, True)

    runtime_owner = SimpleNamespace(_stream_provider_response=_fake_stream)
    executor = _provider_turn_executor()

    tools, full_content, tool_calls, garbage = await executor._stream_provider_turn(
        orch, runtime_owner, stream_ctx, None
    )

    # QA tasks pass tools=None; thinking=True injects the thinking provider kwarg.
    assert tools is None
    assert captured["tools"] is None
    assert captured["provider_kwargs"]["thinking"] == {
        "type": "enabled",
        "budget_tokens": 10000,
    }
    assert garbage is True


class _FakeToolExecResult:
    def __init__(self, *, tool_calls_executed=0, should_return=False, chunks=None):
        self.tool_calls_executed = tool_calls_executed
        self.should_return = should_return
        self.chunks = chunks or []
        self.tool_results = []


def _tools_turn_orch(seen):
    return SimpleNamespace(
        observed_files={"a.py"},
        tool_calls_used=3,
        tool_budget=50,
    )


async def test_execute_tools_turn_streams_and_records_outcome():
    seen = {}
    orch = _tools_turn_orch(seen)
    stream_ctx = SimpleNamespace(
        tool_calls_used=0,
        record_iteration_tool_count=lambda n: seen.__setitem__("recorded", n),
    )

    class _StreamingHandler:
        def update_observed_files(self, files):
            seen["observed"] = files

        async def execute_tools_streaming(self, **kwargs):
            seen["exec_kwargs"] = kwargs
            kwargs["result"].tool_calls_executed = 2
            kwargs["result"].should_return = False
            for c in ("chunk-a", "chunk-b"):
                yield c

    runtime_owner = SimpleNamespace(_tool_execution_handler=_StreamingHandler())
    executor = _provider_turn_executor()

    guard_calls = []
    executor._maybe_inject_write_action_guard = (
        lambda o, ctx, *, user_message, tool_calls_used: guard_calls.append(tool_calls_used)
    )

    holder = chat_stream_executor._ToolTurnOutcome()
    chunks = [
        c
        async for c in executor._execute_tools_turn(
            orch,
            runtime_owner,
            stream_ctx,
            user_message="do it",
            tool_calls=[{"name": "edit"}],
            full_content="content",
            result_holder=holder,
        )
    ]

    # Tool chunks streamed verbatim.
    assert chunks == ["chunk-a", "chunk-b"]
    # observed files snapshot passed to the handler.
    assert seen["observed"] == {"a.py"}
    # Post-exec bookkeeping: tool-call accounting + record + write-action guard.
    assert orch.tool_calls_used == 5  # 3 + 2 executed
    assert stream_ctx.tool_calls_used == 5
    assert seen["recorded"] == 2
    assert guard_calls == [5]
    # Result written to the holder for run() to read.
    assert holder.result.tool_calls_executed == 2
    assert holder.result.should_return is False


async def test_execute_tools_turn_buffered_handler_yields_result_chunks():
    seen = {}
    orch = _tools_turn_orch(seen)
    stream_ctx = SimpleNamespace(tool_calls_used=0)  # no record_iteration_tool_count

    buffered = _FakeToolExecResult(tool_calls_executed=1, should_return=True, chunks=["buf-1"])

    class _BufferedHandler:
        def update_observed_files(self, files):
            seen["observed"] = files

        async def execute_tools(self, **kwargs):
            return buffered

    # Buffered handler must NOT expose execute_tools_streaming.
    handler = _BufferedHandler()
    assert not hasattr(handler, "execute_tools_streaming")
    runtime_owner = SimpleNamespace(_tool_execution_handler=handler)

    executor = _provider_turn_executor()
    executor._maybe_inject_write_action_guard = lambda *a, **k: None

    holder = chat_stream_executor._ToolTurnOutcome()
    chunks = [
        c
        async for c in executor._execute_tools_turn(
            orch,
            runtime_owner,
            stream_ctx,
            user_message="do it",
            tool_calls=[{"name": "edit"}],
            full_content="content",
            result_holder=holder,
        )
    ]

    assert chunks == ["buf-1"]
    assert orch.tool_calls_used == 4  # 3 + 1
    # should_return propagated via the holder (loop control stays in run()).
    assert holder.result.should_return is True


def _spin(state_name, *, state_value="active"):
    record_calls = []
    spin = SimpleNamespace(
        state=SimpleNamespace(name=state_name, value=state_value),
        record_turn=lambda **kw: record_calls.append(kw),
    )
    spin.record_calls = record_calls
    return spin


def _chunk_gen_orch():
    return SimpleNamespace(
        _chunk_generator=SimpleNamespace(
            generate_content_chunk=lambda text, is_final=False: SimpleNamespace(
                content=text, is_final=is_final
            )
        )
    )


async def _collect_turn_stops(
    executor, *, orch, stream_ctx, full_content, tool_calls, spin, turn_eval
):
    outcome = chat_stream_executor._ProviderTurnEval()
    chunks = [
        c
        async for c in executor._evaluate_provider_turn_stops(
            orch,
            stream_ctx,
            full_content=full_content,
            tool_calls=tool_calls,
            spin=spin,
            turn_eval=turn_eval,
            outcome=outcome,
        )
    ]
    return chunks, outcome


async def test_evaluate_provider_turn_stops_normal_records_metrics_no_stop():
    executor = _provider_turn_executor()
    orch = _chunk_gen_orch()
    stream_ctx = SimpleNamespace()  # no skip_continuation preset
    spin = _spin("ACTIVE", state_value="active")
    turn_eval = SimpleNamespace(evaluate=lambda obs, record_spin: SimpleNamespace(stop=False))

    chunks, outcome = await _collect_turn_stops(
        executor,
        orch=orch,
        stream_ctx=stream_ctx,
        full_content="  hello world  ",
        tool_calls=[{"name": "edit"}],
        spin=spin,
        turn_eval=turn_eval,
    )

    # No stop -> no chunks, loop continues.
    assert chunks == []
    assert outcome.should_return is False
    # Derived metrics surfaced for the rest of the loop body.
    assert outcome.content_length == len("hello world")
    assert outcome.has_tools is True
    # Spin fed this turn's signatures.
    assert spin.record_calls[0]["has_tool_calls"] is True
    assert spin.record_calls[0]["tool_count"] == 1


async def test_evaluate_provider_turn_stops_terminated_spin_emits_final_and_returns():
    executor = _provider_turn_executor()
    orch = _chunk_gen_orch()
    stream_ctx = SimpleNamespace()
    spin = _spin("TERMINATED")
    turn_eval = SimpleNamespace(evaluate=lambda obs, record_spin: SimpleNamespace(stop=False))

    chunks, outcome = await _collect_turn_stops(
        executor,
        orch=orch,
        stream_ctx=stream_ctx,
        full_content="stuck",
        tool_calls=None,
        spin=spin,
        turn_eval=turn_eval,
    )

    assert outcome.should_return is True
    assert stream_ctx.force_completion is True
    assert stream_ctx.skip_continuation is True
    assert len(chunks) == 1 and chunks[0].is_final is True
    assert "response loop pattern" in chunks[0].content


async def test_evaluate_provider_turn_stops_content_repetition_emits_final_and_returns():
    executor = _provider_turn_executor()
    orch = _chunk_gen_orch()
    stream_ctx = SimpleNamespace()
    spin = _spin("ACTIVE")
    turn_eval = SimpleNamespace(
        evaluate=lambda obs, record_spin: SimpleNamespace(
            stop=True, stop_reason="repeat", stop_message="STOP-MSG"
        )
    )

    chunks, outcome = await _collect_turn_stops(
        executor,
        orch=orch,
        stream_ctx=stream_ctx,
        full_content="repeated content",
        tool_calls=None,
        spin=spin,
        turn_eval=turn_eval,
    )

    assert outcome.should_return is True
    assert stream_ctx.skip_continuation is True
    assert chunks[0].content == "STOP-MSG"
    assert chunks[0].is_final is True


async def test_evaluate_provider_turn_stops_preset_skip_continuation_returns_no_chunk():
    executor = _provider_turn_executor()
    orch = _chunk_gen_orch()
    stream_ctx = SimpleNamespace(skip_continuation=True)
    spin = _spin("ACTIVE")
    turn_eval = SimpleNamespace(evaluate=lambda obs, record_spin: SimpleNamespace(stop=False))

    chunks, outcome = await _collect_turn_stops(
        executor,
        orch=orch,
        stream_ctx=stream_ctx,
        full_content="some content",
        tool_calls=None,
        spin=spin,
        turn_eval=turn_eval,
    )

    # skip_continuation already set upstream -> return with no emitted chunk.
    assert outcome.should_return is True
    assert chunks == []
