"""P5 end-to-end: a degenerate looping generation terminates early via the
intra-turn repetition guard instead of streaming to exhaustion."""

from __future__ import annotations

import pytest

from .parity_harness import (
    Scenario,
    TurnScript,
    build_for_scenario,
    capture_transcript,
)

pytestmark = pytest.mark.integration

LOOP_SENTENCE = "Let me check the remote tracking state of the branch. "


async def test_degenerate_generation_terminates_early():
    scenario = Scenario(
        id="P5",
        label="intra-turn repetition loop",
        message="Check the branch state.",
        turns=[TurnScript(content=LOOP_SENTENCE * 200)],
        expect_tools=[],
    )
    orch = build_for_scenario(scenario)
    transcript = await capture_transcript(orch, scenario.message)

    assert not transcript.errored, transcript.error
    occurrences = transcript.content.count("remote tracking state")
    assert (
        1 <= occurrences < 20
    ), f"guard should truncate the loop (saw {occurrences} of 200 instances)"
