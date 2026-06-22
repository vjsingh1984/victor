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

"""Run/stream behavioral-parity battery — the FEP-0007 cutover acceptance gate.

Addendum A (framing B) makes streaming a pure I/O mode of the one research-rooted loop, so the
acceptance bar is **behavioral parity**: ``Agent.run(p)`` (buffered ``AgenticLoop.run``) and
``Agent.stream(p)`` (``StreamingChatExecutor.run``) must produce the SAME executed tool sequence
and the SAME answer for each QA scenario. This battery drives BOTH real loops from one scripted
script and compares.

Status today (pre-cutover): the two loops are still separate and genuinely diverge on multi-step
tasks — the buffered loop's completion evaluation is more eager and stops after the first tool,
while the streaming loop runs the full scripted sequence. So this file is split:

* ``test_run_stream_parity`` asserts full parity on the scenarios that ALREADY agree — a real
  regression guard today.
* ``test_multistep_scenarios_still_diverge`` is the cutover **tripwire**: it pins the known
  divergence on the multi-step scenarios. When the step-3 cutover unifies the loops, those
  scenarios reach parity, this test fails, and the divergent ids move into the parity set above.

No ``xfail``: every test here is a positive, passing assertion about current behavior.
"""

import pytest

from .parity_harness import (
    SCENARIOS,
    ScriptedProvider,
    build_streaming_orchestrator,
    capture_buffered_transcript,
    capture_transcript,
)

pytestmark = [pytest.mark.integration, pytest.mark.agents]


# Scenarios whose buffered vs. streaming tool sequence diverges TODAY: the buffered AgenticLoop
# completes after the first tool (eager completion / fulfillment), where the streaming loop runs
# the whole scripted sequence. The cutover must reconcile completion so the unified loop runs the
# full multi-step task; when it does, the tripwire below fails and these move into the parity set.
_DIVERGES_UNTIL_CUTOVER = {
    "M1": "buffered completes after code_search; streaming also runs read",
    "M2": "buffered completes after first ls; streaming runs both",
    "C1": "buffered completes after code_search; streaming also runs read,read",
    "W1": "buffered completes after read; streaming also runs edit",
    "W3": "buffered completes after read; streaming also runs multi_edit",
    "W4": "buffered completes after read; streaming also runs patch",
}

_SCENARIOS_BY_ID = {s.id: s for s in SCENARIOS}
_PARITY_SCENARIOS = [s for s in SCENARIOS if s.id not in _DIVERGES_UNTIL_CUTOVER]


async def _run_both(scenario):
    """Drive the buffered and streaming loops from the same script (separate provider instances)."""
    buffered_orch = build_streaming_orchestrator(
        ScriptedProvider(list(scenario.turns)),
        scenario.tools,
        max_iterations=scenario.max_iterations,
        failing_tool_names=scenario.failing_tools,
    )
    streaming_orch = build_streaming_orchestrator(
        ScriptedProvider(list(scenario.turns)),
        scenario.tools,
        max_iterations=scenario.max_iterations,
        failing_tool_names=scenario.failing_tools,
    )
    buffered = await capture_buffered_transcript(buffered_orch, scenario.message)
    streaming = await capture_transcript(streaming_orch, scenario.message)
    return buffered, streaming


@pytest.mark.parametrize("scenario", _PARITY_SCENARIOS, ids=[s.id for s in _PARITY_SCENARIOS])
async def test_run_stream_parity(scenario):
    """Buffered and streaming loops agree on tool sequence + answer for the scenario."""
    buffered, streaming = await _run_both(scenario)

    assert not buffered.errored, f"{scenario.id} buffered: {buffered.error}"
    assert not streaming.errored, f"{scenario.id} streaming: {streaming.error}"

    # Primary parity signal: identical executed tool sequence (same actions, same order).
    assert buffered.tool_calls == streaming.tool_calls, (
        f"{scenario.id} ({scenario.label}): buffered tools {buffered.tool_calls} != "
        f"streaming tools {streaming.tool_calls}"
    )

    # Same answer present in both (streaming concatenates per-turn emits, so use containment,
    # not byte-equality, which the existing characterization battery also relies on).
    if scenario.expect_content_contains:
        needle = scenario.expect_content_contains.lower()
        assert needle in buffered.content.lower(), (
            f"{scenario.id}: expected {scenario.expect_content_contains!r} in buffered "
            f"{buffered.content!r}"
        )
        assert needle in streaming.content.lower(), (
            f"{scenario.id}: expected {scenario.expect_content_contains!r} in streaming "
            f"{streaming.content!r}"
        )


async def test_multistep_scenarios_still_diverge():
    """Cutover tripwire: the multi-step scenarios still diverge (buffered stops early).

    Pins the current behavior so the gap can't go unnoticed. When the step-3 cutover unifies the
    loops these reach parity and this test fails — the signal to move the now-matching ids into the
    parity set above and delete them here.
    """
    assert _DIVERGES_UNTIL_CUTOVER.keys() <= set(
        _SCENARIOS_BY_ID
    ), _DIVERGES_UNTIL_CUTOVER.keys() - set(_SCENARIOS_BY_ID)

    for scenario_id in _DIVERGES_UNTIL_CUTOVER:
        scenario = _SCENARIOS_BY_ID[scenario_id]
        buffered, streaming = await _run_both(scenario)
        assert not buffered.errored, f"{scenario_id} buffered: {buffered.error}"
        assert not streaming.errored, f"{scenario_id} streaming: {streaming.error}"
        # Buffered stops early — strictly fewer tools, and a prefix of the streaming sequence.
        assert (
            buffered.tool_calls != streaming.tool_calls
        ), f"{scenario_id} now reaches parity — move it into the parity set and remove it here"
        assert (
            streaming.tool_calls[: len(buffered.tool_calls)] == buffered.tool_calls
        ), f"{scenario_id}: buffered {buffered.tool_calls} is not a prefix of {streaming.tool_calls}"


async def test_parity_battery_has_matching_scenarios():
    """The parity set is non-empty, so test_run_stream_parity actually asserts something today."""
    assert _PARITY_SCENARIOS, "expected at least one scenario to already reach run/stream parity"
