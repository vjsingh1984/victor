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

"""Streaming-loop parity battery — FEP-0007 Phase 3→4 acceptance gate.

The characterization tests pin the **legacy** ``StreamingChatExecutor.run()`` behavior on the
QA battery (the baseline). ``test_legacy_vs_unified_parity`` activates automatically once
``AgenticLoop.run_streaming`` lands (Phase 3), proving the unified loop is equivalent before
``USE_UNIFIED_STREAMING_LOOP`` flips ON (Phase 4).
"""

from __future__ import annotations

import pytest

from .parity_harness import (
    SCENARIOS,
    build_for_scenario,
    capture_transcript,
    unified_streaming_available,
)

pytestmark = [pytest.mark.integration, pytest.mark.agents]

_SCENARIO_IDS = [s.id for s in SCENARIOS]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_SCENARIO_IDS)
async def test_legacy_loop_characterization(scenario):
    """Each battery scenario runs through the real legacy loop without a streaming traceback,
    producing the expected answer + tool sequence. This is the baseline the unified loop must
    reproduce."""
    orch = build_for_scenario(scenario)
    transcript = await capture_transcript(orch, scenario.message)

    assert not transcript.errored, f"{scenario.id} ({scenario.label}): {transcript.error}"

    if scenario.expect_content_contains:
        assert scenario.expect_content_contains.lower() in transcript.content.lower(), (
            f"{scenario.id}: expected {scenario.expect_content_contains!r} in "
            f"{transcript.content!r}"
        )

    if scenario.expect_tools is not None:
        assert (
            transcript.tool_calls == scenario.expect_tools
        ), f"{scenario.id}: tool sequence {transcript.tool_calls} != {scenario.expect_tools}"


async def test_spin_scenario_terminates():
    """U1: an identical repeated tool call must be stopped by the loop's spin/repetition guard
    well before max_iterations — i.e. the loop converges instead of hanging."""
    scenario = next(s for s in SCENARIOS if s.id == "U1")
    orch = build_for_scenario(scenario)
    transcript = await capture_transcript(orch, scenario.message)

    assert not transcript.errored, transcript.error
    # The guard must terminate before exhausting the scripted turns / iteration ceiling.
    assert len(transcript.tool_calls) < scenario.max_iterations, (
        f"spin guard did not converge: {len(transcript.tool_calls)} tool calls "
        f"(max_iterations={scenario.max_iterations})"
    )


@pytest.mark.skipif(
    not unified_streaming_available(),
    reason="FEP-0007 Phase 3 (AgenticLoop.run_streaming) not landed yet",
)
@pytest.mark.parametrize("scenario", SCENARIOS, ids=_SCENARIO_IDS)
async def test_legacy_vs_unified_parity(scenario):
    """Once Phase 3 lands: the unified loop must yield an equivalent transcript to the legacy
    loop on every battery scenario before the flag flips ON.

    Wiring note for Phase 3: build the orchestrator under ``USE_UNIFIED_STREAMING_LOOP`` ON for
    the second capture and compare with :func:`parity_harness.assert_parity`. Left as the
    activation point so this file lands as the gate without depending on unbuilt code.
    """
    pytest.skip("Enable when run_streaming + USE_UNIFIED_STREAMING_LOOP land (Phase 3).")
