# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Property-based tests for StageTransitionEngine.

Phase 3.3: Property-based tests for stage transition logic.

Uses Hypothesis to test invariants across many iterations:
1. Transition graph properties (valid transitions)
2. Backward transition confidence requirements
3. Callback invocation guarantees
4. State consistency after transitions
"""

import pytest
from hypothesis import given, strategies as st, settings, Phase, assume
from typing import List, Callable

from victor.core.state import ConversationStage
from victor.agent.stage_transition_engine import (
    StageTransitionEngine,
    STAGE_ORDER,
    TRANSITION_GRAPH,
)


# Strategies
stage_strategy = st.sampled_from(list(ConversationStage))
confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
cooldown_strategy = st.floats(min_value=0.0, max_value=5.0, allow_nan=False)


class TestTransitionGraphProperties:
    """Property-based tests for transition graph invariants."""

    @given(initial_stage=stage_strategy)
    @settings(max_examples=50, phases=[Phase.generate])
    def test_same_stage_transition_always_succeeds(self, initial_stage: ConversationStage):
        """Transitioning to the same stage should always succeed."""
        engine = StageTransitionEngine(initial_stage=initial_stage, cooldown_seconds=0)

        result = engine.transition_to(initial_stage)

        assert result is True
        assert engine.current_stage == initial_stage

    @given(initial_stage=stage_strategy, target_stage=stage_strategy)
    @settings(max_examples=100, phases=[Phase.generate])
    def test_valid_forward_transitions_succeed(
        self, initial_stage: ConversationStage, target_stage: ConversationStage
    ):
        """Valid forward transitions (in graph) should succeed."""
        valid_targets = TRANSITION_GRAPH.get(initial_stage, set())

        # Skip if target is not a valid forward transition
        assume(target_stage in valid_targets)

        engine = StageTransitionEngine(initial_stage=initial_stage, cooldown_seconds=0)
        result = engine.transition_to(target_stage)

        assert result is True
        assert engine.current_stage == target_stage

    @given(initial_stage=stage_strategy, target_stage=stage_strategy)
    @settings(max_examples=100, phases=[Phase.generate])
    def test_invalid_forward_transitions_fail(
        self, initial_stage: ConversationStage, target_stage: ConversationStage
    ):
        """Invalid transitions (not in graph, not backward) should fail with low confidence."""
        valid_targets = TRANSITION_GRAPH.get(initial_stage, set())

        # Skip same stage
        assume(target_stage != initial_stage)
        # Skip valid forward transitions
        assume(target_stage not in valid_targets)
        # Skip backward transitions (tested separately)
        assume(STAGE_ORDER[target_stage] >= STAGE_ORDER[initial_stage])

        engine = StageTransitionEngine(initial_stage=initial_stage, cooldown_seconds=0)
        result = engine.transition_to(target_stage, confidence=0.5)

        assert result is False
        assert engine.current_stage == initial_stage


class TestBackwardTransitionProperties:
    """Property-based tests for backward transition confidence requirements."""

    @given(
        initial_stage=stage_strategy,
        target_stage=stage_strategy,
        confidence=st.floats(min_value=0.0, max_value=0.84, allow_nan=False),
    )
    @settings(max_examples=100, phases=[Phase.generate])
    def test_backward_transition_fails_with_low_confidence(
        self,
        initial_stage: ConversationStage,
        target_stage: ConversationStage,
        confidence: float,
    ):
        """Backward transitions should fail with confidence < 0.85."""
        # Only test backward transitions
        assume(STAGE_ORDER[target_stage] < STAGE_ORDER[initial_stage])

        engine = StageTransitionEngine(initial_stage=initial_stage, cooldown_seconds=0)
        result = engine.transition_to(target_stage, confidence=confidence)

        assert result is False
        assert engine.current_stage == initial_stage

    @given(
        initial_stage=stage_strategy,
        target_stage=stage_strategy,
        confidence=st.floats(min_value=0.85, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100, phases=[Phase.generate])
    def test_backward_transition_succeeds_with_high_confidence(
        self,
        initial_stage: ConversationStage,
        target_stage: ConversationStage,
        confidence: float,
    ):
        """Backward transitions should succeed with confidence >= 0.85."""
        # Only test backward transitions
        assume(STAGE_ORDER[target_stage] < STAGE_ORDER[initial_stage])

        engine = StageTransitionEngine(initial_stage=initial_stage, cooldown_seconds=0)
        result = engine.transition_to(target_stage, confidence=confidence)

        assert result is True
        assert engine.current_stage == target_stage


class TestToolPriorityMultiplierProperties:
    """Property-based tests for tool priority multiplier properties."""

    @given(stage=stage_strategy, tool_name=st.text(min_size=1, max_size=20))
    @settings(max_examples=100, phases=[Phase.generate])
    def test_multiplier_is_positive(self, stage: ConversationStage, tool_name: str):
        """Tool priority multiplier should always be positive."""
        engine = StageTransitionEngine(initial_stage=stage, cooldown_seconds=0)

        multiplier = engine.get_tool_priority_multiplier(tool_name)

        assert multiplier > 0
        assert multiplier >= 1.0 or multiplier == 1.0  # Default or boost


class TestCallbackProperties:
    """Property-based tests for callback invocation guarantees."""

    @given(
        initial_stage=stage_strategy,
        target_stages=st.lists(stage_strategy, min_size=1, max_size=5),
    )
    @settings(max_examples=50, phases=[Phase.generate])
    def test_callback_invoked_on_successful_transitions(
        self,
        initial_stage: ConversationStage,
        target_stages: List[ConversationStage],
    ):
        """Callbacks should be invoked on every successful transition."""
        callback_count = [0]

        def track_callback(old, new):
            callback_count[0] += 1

        engine = StageTransitionEngine(initial_stage=initial_stage, cooldown_seconds=0)
        engine.register_callback(track_callback)

        successful_transitions = 0
        current = initial_stage

        for target in target_stages:
            valid_targets = TRANSITION_GRAPH.get(current, set())
            if target in valid_targets or (
                STAGE_ORDER[target] < STAGE_ORDER[current] and True  # High confidence
            ):
                result = engine.transition_to(target, confidence=0.9)
                if result and target != current:
                    successful_transitions += 1
                    current = target

        assert callback_count[0] == successful_transitions


class TestHistoryProperties:
    """Property-based tests for transition history properties."""

    @given(
        initial_stage=stage_strategy,
        transitions=st.lists(
            st.tuples(stage_strategy, confidence_strategy),
            min_size=0,
            max_size=10,
        ),
    )
    @settings(max_examples=50, phases=[Phase.generate])
    def test_history_only_records_successful_transitions(
        self,
        initial_stage: ConversationStage,
        transitions: List[tuple],
    ):
        """Transition history should only record successful transitions."""
        engine = StageTransitionEngine(initial_stage=initial_stage, cooldown_seconds=0)

        successful_count = 0
        current = initial_stage

        for target, confidence in transitions:
            result = engine.transition_to(target, confidence=confidence)
            if result and target != current:
                successful_count += 1
                current = target

        assert len(engine.transition_history) == successful_count


class TestResetProperties:
    """Property-based tests for reset behavior."""

    @given(
        initial_stage=stage_strategy,
        transitions=st.lists(stage_strategy, min_size=1, max_size=5),
    )
    @settings(max_examples=50, phases=[Phase.generate])
    def test_reset_restores_initial_state(
        self,
        initial_stage: ConversationStage,
        transitions: List[ConversationStage],
    ):
        """Reset should restore engine to INITIAL stage with empty history."""
        engine = StageTransitionEngine(initial_stage=initial_stage, cooldown_seconds=0)

        # Make some transitions
        for target in transitions:
            engine.transition_to(target, confidence=0.9)

        # Reset
        engine.reset()

        assert engine.current_stage == ConversationStage.INITIAL
        assert len(engine.transition_history) == 0
