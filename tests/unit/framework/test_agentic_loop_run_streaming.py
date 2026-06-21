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

"""Offline tests for AgenticLoop.run_streaming (FEP-0007 Addendum A, step 2).

These drive ``run_streaming`` with a fake streaming ACT port and stubbed phase methods, so they
assert the shared loop wiring — that streaming reuses the same PERCEIVE/PLAN/EVALUATE/DECIDE phase
sequence as run() and differs only in the ACT (yielding chunks + producing a TurnResult). The
port is unwired in production; these cover the framework loop in isolation.
"""

from types import SimpleNamespace

import pytest

from victor.framework.agentic_loop import AgenticLoop, StreamingTurnOutcome
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.providers.base import StreamChunk


def _fake_port(turns):
    """A streaming ACT port: ``turns[i]`` is turn i+1's list of chunk strings.

    Each turn yields its chunks and stamps a TurnResult-shaped object onto the outcome holder.
    """

    class _Port:
        async def stream_turn_act(self, *, query, state, perception, plan, turn_index, outcome):
            chunks = turns[turn_index - 1] if turn_index - 1 < len(turns) else []
            for text in chunks:
                yield StreamChunk(content=text)
            outcome.turn_result = SimpleNamespace(content="".join(chunks), tag=f"turn{turn_index}")

    return _Port()


def _loop(port, *, evaluations, turn_executor=None):
    """A bare AgenticLoop with the run_streaming collaborators stubbed."""
    loop = AgenticLoop.__new__(AgenticLoop)
    loop.streaming_act_port = port
    loop.max_iterations = 5
    loop._progress_scores = []
    loop._last_perception = None
    loop.turn_executor = turn_executor
    loop.spin_detector = object()
    loop.turn_evaluation_controller = SimpleNamespace(reset=lambda: None)
    loop.criteria_builder = SimpleNamespace(reset=lambda: None)
    loop.nudge_policy = SimpleNamespace(
        evaluate=lambda spin: SimpleNamespace(should_inject=False),
        budget_warning=lambda i, m: SimpleNamespace(should_inject=False),
    )

    async def fake_analyze(query, context, history):
        return SimpleNamespace(
            to_dict=lambda: {"query": query},
            task_analysis=SimpleNamespace(task_type="qa"),
        )

    async def fake_plan(perception, state):
        return {"plan": "noop"}

    evals = iter(evaluations)

    async def fake_evaluate(perception, action_result, state):
        # Record what ACT produced so tests can assert the EVALUATE input.
        state["_seen_action_result"] = action_result
        return next(evals)

    loop._analyze_turn = fake_analyze
    loop._plan = fake_plan
    loop._evaluate = fake_evaluate
    loop._apply_backslide_guard = lambda evaluation: evaluation
    return loop


async def test_run_streaming_yields_chunks_and_completes_on_first_turn():
    port = _fake_port([["hello ", "world"]])
    loop = _loop(
        port,
        evaluations=[
            EvaluationResult(decision=EvaluationDecision.COMPLETE, score=1.0, reason="done")
        ],
    )

    chunks = [c async for c in loop.run_streaming("q")]

    assert [c.content for c in chunks] == ["hello ", "world"]
    assert loop._progress_scores == [1.0]


async def test_run_streaming_iterates_until_complete():
    port = _fake_port([["a"], ["b"]])
    loop = _loop(
        port,
        evaluations=[
            EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.4, reason="more"),
            EvaluationResult(decision=EvaluationDecision.COMPLETE, score=0.9, reason="done"),
        ],
    )

    chunks = [c async for c in loop.run_streaming("q")]

    # Both turns' chunks stream through in order; progress recorded per turn.
    assert [c.content for c in chunks] == ["a", "b"]
    assert loop._progress_scores == [0.4, 0.9]


async def test_run_streaming_feeds_act_turn_result_to_evaluate():
    port = _fake_port([["payload"]])
    captured = {}

    loop = _loop(
        port,
        evaluations=[
            EvaluationResult(decision=EvaluationDecision.COMPLETE, score=1.0, reason="done")
        ],
    )

    original_evaluate = loop._evaluate

    async def spy_evaluate(perception, action_result, state):
        captured["action_result"] = action_result
        return await original_evaluate(perception, action_result, state)

    loop._evaluate = spy_evaluate

    _ = [c async for c in loop.run_streaming("q")]

    # The TurnResult the ACT port produced is exactly what EVALUATE consumed.
    assert captured["action_result"].tag == "turn1"
    assert captured["action_result"].content == "payload"


async def test_run_streaming_stops_on_fail():
    port = _fake_port([["x"]])
    loop = _loop(
        port,
        evaluations=[EvaluationResult(decision=EvaluationDecision.FAIL, score=0.0, reason="bad")],
    )

    chunks = [c async for c in loop.run_streaming("q")]

    assert [c.content for c in chunks] == ["x"]
    assert loop._progress_scores == [0.0]


async def test_run_streaming_requires_a_port():
    loop = _loop(None, evaluations=[])

    with pytest.raises(RuntimeError, match="streaming ACT port"):
        async for _ in loop.run_streaming("q"):
            pass


def _nudge_loop(chat_ctx, *, should_inject_nudge, should_inject_budget=False):
    loop = AgenticLoop.__new__(AgenticLoop)
    loop.turn_executor = SimpleNamespace(_chat_context=chat_ctx)
    loop.spin_detector = object()
    loop.nudge_policy = SimpleNamespace(
        evaluate=lambda spin: SimpleNamespace(
            should_inject=should_inject_nudge,
            role="system",
            message="use a tool",
            nudge_type=SimpleNamespace(value="USE_TOOLS"),
        ),
        budget_warning=lambda i, m: SimpleNamespace(
            should_inject=should_inject_budget,
            role="system",
            message="budget low",
        ),
    )
    return loop


def test_inject_decide_nudges_adds_message_on_continue():
    added = []
    chat_ctx = SimpleNamespace(
        add_message=lambda role, message, metadata=None: added.append((role, message))
    )
    loop = _nudge_loop(chat_ctx, should_inject_nudge=True)

    loop._inject_decide_nudges(
        EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.3, reason="x"), 1, 5
    )

    assert added == [("system", "use a tool")]


def test_inject_decide_nudges_noop_on_terminal_decision():
    added = []
    chat_ctx = SimpleNamespace(add_message=lambda *a, **k: added.append(a))
    loop = _nudge_loop(chat_ctx, should_inject_nudge=True)

    loop._inject_decide_nudges(
        EvaluationResult(decision=EvaluationDecision.COMPLETE, score=1.0, reason="done"), 1, 5
    )

    assert added == []


def test_inject_decide_nudges_noop_without_turn_executor():
    loop = AgenticLoop.__new__(AgenticLoop)
    loop.turn_executor = None
    # Should simply return without touching nudge_policy / chat context.
    loop._inject_decide_nudges(
        EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.2, reason="x"), 2, 5
    )
