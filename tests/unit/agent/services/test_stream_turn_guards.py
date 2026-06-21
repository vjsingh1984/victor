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
