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

"""Tests for the shared per-turn evaluation components (content-repetition, plateau, controller).

These lock the behavior that the headless (AgenticLoop) and streaming (StreamingChatExecutor)
loops both delegate to, so the consolidation is verifiably faithful.
"""

from victor.agent.turn_policy import (
    ContentRepetitionDetector,
    PlateauDetector,
    TurnEvaluationController,
    TurnObservation,
    evaluate_overlap_repetition,
)

# --------------------------------------------------------------------------- #
# evaluate_overlap_repetition (the exact thresholds the streaming loop used)
# --------------------------------------------------------------------------- #


def test_overlap_thresholds():
    assert evaluate_overlap_repetition(0.9, 0) == (1, "near_duplicate")
    assert evaluate_overlap_repetition(0.6, 0) == (1, "accumulating")
    assert evaluate_overlap_repetition(0.6, 1) == (2, "high_overlap")
    assert evaluate_overlap_repetition(0.2, 3) == (0, "reset")
    assert evaluate_overlap_repetition(0.4, 2) == (2, "hold")  # ambiguous: hold, don't decay


# --------------------------------------------------------------------------- #
# ContentRepetitionDetector
# --------------------------------------------------------------------------- #


def test_identical_content_trips_near_duplicate_on_second_turn():
    # Matches the streaming loop: identical content gives overlap=1.0, so the second
    # turn is already a near-duplicate terminal (the exact-hash-3 path is a backstop).
    d = ContentRepetitionDetector()
    same = "I will now analyze the orchestrator and report what I find in detail."
    assert d.record(same) == "none"
    assert d.record(same) == "near_duplicate"


def test_near_duplicate_is_terminal_on_single_strong_repeat():
    d = ContentRepetitionDetector()
    a = "Let me drill into the orchestrator initialization phases and the service wiring details."
    b = "Let me drill into the orchestrator initialization phases and the service wiring details!!"
    assert d.record(a) == "none"
    assert d.record(b) == "near_duplicate"  # ~full overlap -> stop immediately


def test_distinct_content_does_not_trip():
    d = ContentRepetitionDetector()
    assert d.record("The orchestrator coordinates extracted components via a facade pattern.") in (
        "none",
        "reset",
    )
    assert d.record("Tool execution delegates to ToolPipeline with retry and caching logic.") in (
        "none",
        "reset",
        "hold",
    )


def test_short_content_ignored():
    d = ContentRepetitionDetector()
    assert d.record("ok") == "none"
    assert d.record("") == "none"
    assert d.record(None) == "none"


# --------------------------------------------------------------------------- #
# PlateauDetector (streaming productivity-weighted formula)
# --------------------------------------------------------------------------- #


def test_plateau_nudges_only_when_unproductive():
    d = PlateauDetector()
    # Three unproductive low-score turns -> plateau + nudge.
    d.record(0, 100)
    d.record(0, 100)
    r = d.record(0, 100)
    assert r.is_plateau and r.should_nudge

    # A productive turn flattening the score is a plateau but must NOT nudge.
    d2 = PlateauDetector()
    d2.record(2, 100)
    d2.record(2, 100)
    r2 = d2.record(2, 100)
    assert r2.is_plateau and not r2.should_nudge


def test_no_plateau_before_window():
    d = PlateauDetector()
    assert not d.record(0, 50).is_plateau
    assert not d.record(0, 50).is_plateau  # only 2 scores so far


# --------------------------------------------------------------------------- #
# TurnEvaluationController
# --------------------------------------------------------------------------- #


def test_controller_stops_on_content_repetition():
    c = TurnEvaluationController()
    same = "Reading the file in chunks and continuing the analysis as I go along here now."
    c.evaluate(TurnObservation(content=same, has_tool_calls=True, tool_count=1))
    c.evaluate(TurnObservation(content=same, has_tool_calls=True, tool_count=1))
    decision = c.evaluate(TurnObservation(content=same, has_tool_calls=True, tool_count=1))
    assert decision.stop is True
    assert decision.terminal_success is False
    assert decision.stop_reason == "content_repetition"
    assert "repetition" in (decision.stop_message or "").lower()


def test_controller_nudges_on_unproductive_plateau():
    c = TurnEvaluationController(enable_budget_warning=False)
    # Distinct short content each turn (no repetition stop), all unproductive -> plateau nudge.
    for n in range(3):
        decision = c.evaluate(
            TurnObservation(content=f"thinking step {n} about the task", productive_count=0)
        )
    assert decision.stop is False
    assert decision.nudge_message is not None
    assert decision.nudge_kind == "plateau"


def test_controller_write_intent_plateau_message():
    c = TurnEvaluationController(enable_budget_warning=False)
    for n in range(3):
        decision = c.evaluate(
            TurnObservation(
                content=f"looking at file {n}", productive_count=0, intent_is_write=True
            )
        )
    assert "edit(" in (decision.nudge_message or "")


def test_controller_clean_turn_continues():
    c = TurnEvaluationController(enable_budget_warning=False)
    decision = c.evaluate(
        TurnObservation(
            content="A clear, distinct, substantive answer.",
            productive_count=1,
            has_tool_calls=True,
            tool_count=1,
            iteration=1,
            max_iterations=8,
        )
    )
    assert decision.stop is False
    assert decision.nudge_message is None


def test_controller_reset_clears_state():
    c = TurnEvaluationController()
    same = "Repeating the same narration sentence over and over and over again now."
    c.evaluate(TurnObservation(content=same))
    c.evaluate(TurnObservation(content=same))
    c.reset()
    # After reset, the same content starts fresh (no accumulated repetition).
    assert c.evaluate(TurnObservation(content=same)).stop is False
