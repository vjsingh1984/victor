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

"""Stage transition coordination for Phase 1 optimization.

This module provides StageTransitionCoordinator, which batches tool
executions within a turn and applies Phase 1 optimizations (cooldown,
high confidence skip) before consulting the edge model.

Design Pattern: Strategy + Batch Processing
============================================
StageTransitionCoordinator implements batch processing of tool executions
and delegates transition detection to pluggable strategies:

- HeuristicOnlyTransitionStrategy: Fast, no edge model
- HybridTransitionStrategy: Heuristic + edge model fallback
- EdgeModelTransitionStrategy: Always use edge model

Usage:
    coordinator = StageTransitionCoordinator(
        state_machine=sm,
        strategy=HybridTransitionStrategy(),
    )

    # Start of turn
    coordinator.begin_turn()

    # During turn - batch tool executions
    for tool_call in tool_calls:
        execute_tool(tool_call)
        coordinator.record_tool(tool_name, args)

    # End of turn - process transitions
    new_stage = coordinator.end_turn()
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from victor.core.shared_types import ConversationStage

if TYPE_CHECKING:
    from victor.agent.coordinators.transition_strategies import TransitionStrategyProtocol

logger = logging.getLogger(__name__)


class TransitionDecision(str, Enum):
    """Decision result from transition evaluation."""

    NO_TRANSITION = "no_transition"  # Stay in current stage
    HEURISTIC_TRANSITION = "heuristic"  # Transition based on heuristic
    EDGE_MODEL_TRANSITION = "edge_model"  # Consult edge model
    COOLDOWN_SKIP = "cooldown_skip"  # Skipped due to cooldown
    HIGH_CONFIDENCE_SKIP = "high_confidence_skip"  # Skipped due to high confidence


@dataclass
class TurnContext:
    """Context for a single turn (batching window).

    Attributes:
        turn_id: Unique identifier for this turn
        start_time: Timestamp when turn started
        tools_executed: List of (tool_name, args) tuples
        current_stage: Stage at turn start
    """

    turn_id: str
    start_time: float
    tools_executed: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    current_stage: ConversationStage = ConversationStage.INITIAL

    def add_tool(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Add a tool execution to this turn."""
        self.tools_executed.append((tool_name, args))

    @property
    def unique_tools(self) -> set[str]:
        """Get unique tool names executed this turn."""
        return {tool for tool, _ in self.tools_executed}

    @property
    def tool_count(self) -> int:
        """Get total tool executions this turn."""
        return len(self.tools_executed)


@dataclass
class TransitionResult:
    """Result of a transition evaluation.

    Attributes:
        decision: Type of decision made
        new_stage: New stage if transition occurred
        confidence: Confidence in the transition (0.0-1.0)
        reason: Human-readable reason for the decision
        edge_model_called: Whether edge model was consulted
        calibration_applied: Whether Phase 2 calibration was applied
    """

    decision: TransitionDecision
    new_stage: Optional[ConversationStage]
    confidence: float
    reason: str
    edge_model_called: bool = False
    calibration_applied: bool = False


class StageTransitionCoordinator:
    """Coordinates stage transitions with Phase 1 optimizations.

    Implements batching of tool executions within a turn and applies
    Phase 1 optimizations (cooldown, high confidence skip) before
    consulting the edge model.

    Phase 1 Optimizations:
    1. Cooldown: Prevent transitions within 2 seconds of last transition
    2. High confidence skip: Skip edge model when tool overlap ≥ threshold
    3. Batch processing: Process all tools once per turn, not per tool

    Phase 2 Calibration:
    - Detects read-only loops (files read > 10, files modified = 0)
    - Calibrates overconfident EXECUTION predictions to ANALYSIS

    Usage:
        coordinator = StageTransitionCoordinator(
            state_machine=sm,
            strategy=HybridTransitionStrategy(),
        )

        # Start of turn
        coordinator.begin_turn()

        # During turn - batch tool executions
        for tool_call in tool_calls:
            execute_tool(tool_call)
            coordinator.record_tool(tool_name, args)

        # End of turn - process transitions
        new_stage = coordinator.end_turn()
    """

    def __init__(
        self,
        state_machine: Any,  # ConversationStateMachine
        strategy: "TransitionStrategyProtocol",
        cooldown_seconds: float = 2.0,
        min_tools_for_transition: int = 5,
    ):
        """Initialize the coordinator.

        Args:
            state_machine: ConversationStateMachine instance
            strategy: Transition strategy to use
            cooldown_seconds: Minimum seconds between transitions
            min_tools_for_transition: Min tools for high confidence skip
        """
        self._state_machine = state_machine
        self._strategy = strategy
        self._cooldown_seconds = cooldown_seconds
        self._min_tools_for_transition = min_tools_for_transition

        self._current_turn: Optional[TurnContext] = None
        self._last_transition_time: float = 0.0
        self._transition_count: int = 0

        logger.debug(
            f"StageTransitionCoordinator initialized: "
            f"strategy={type(strategy).__name__}, "
            f"cooldown={cooldown_seconds}s, "
            f"min_tools={min_tools_for_transition}"
        )

    def begin_turn(self) -> None:
        """Mark the start of a new turn.

        Creates a new batching window for tool executions.
        """
        self._current_turn = TurnContext(
            turn_id=str(uuid.uuid4())[:8],
            start_time=time.time(),
            current_stage=self._state_machine.get_stage(),
        )
        logger.debug(
            f"Turn {self._current_turn.turn_id} started, stage={self._current_turn.current_stage}"
        )

    def record_tool(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record a tool execution (batched, no immediate transition).

        Args:
            tool_name: Name of the tool executed
            args: Tool arguments
        """
        if self._current_turn is None:
            logger.warning("record_tool called before begin_turn, creating turn")
            self.begin_turn()

        self._current_turn.add_tool(tool_name, args)
        logger.debug(
            f"Recorded tool {tool_name} in turn {self._current_turn.turn_id}, "
            f"total tools={self._current_turn.tool_count}"
        )

    def end_turn(self) -> Optional[ConversationStage]:
        """Process batched tools and return new stage if transition occurred.

        Applies Phase 1 optimizations:
        1. Check cooldown (skip if in cooldown period)
        2. Check high confidence (skip if tool overlap ≥ threshold)
        3. Delegate to strategy for transition detection
        4. Apply calibration if needed

        Returns:
            New stage if transition occurred, None otherwise
        """
        if self._current_turn is None:
            logger.warning("end_turn called without active turn")
            return None

        turn = self._current_turn
        logger.debug(
            f"Turn {turn.turn_id} ending, tools={turn.tool_count}, "
            f"unique={len(turn.unique_tools)}, stage={turn.current_stage}"
        )

        # Check cooldown first (Phase 1 optimization #1)
        if self._is_in_cooldown():
            time_since_last = time.time() - self._last_transition_time
            logger.debug(
                f"In cooldown period ({time_since_last:.1f}s < {self._cooldown_seconds}s), "
                f"skipping transition evaluation"
            )
            # Clear turn context before returning
            self._current_turn = None
            return None

        # Get transition result from strategy
        result = self._evaluate_transition(turn)

        # Apply Phase 2 calibration if needed
        if self._should_calibrate(result):
            result = self._apply_calibration(result)

        # Execute transition if needed
        new_stage = None
        if result.decision in (
            TransitionDecision.HEURISTIC_TRANSITION,
            TransitionDecision.EDGE_MODEL_TRANSITION,
            TransitionDecision.HIGH_CONFIDENCE_SKIP,
        ):
            self._execute_transition(result)
            new_stage = result.new_stage

        # Clear the turn context after processing
        self._current_turn = None

        return new_stage

    def should_skip_edge_model(
        self, detected_stage: ConversationStage, current_stage: ConversationStage
    ) -> bool:
        """Check if edge model should be skipped based on Phase 1 optimizations.

        This is a convenience method for strategies to call.

        Args:
            detected_stage: Stage detected by heuristic
            current_stage: Current conversation stage

        Returns:
            True if edge model should be skipped
        """
        # Check cooldown
        if self._is_in_cooldown():
            return True

        # Check if no transition needed
        if detected_stage == current_stage:
            return True

        # Check high confidence skip
        if self._current_turn:
            overlap = self._calculate_stage_overlap(detected_stage)
            if overlap >= self._min_tools_for_transition:
                logger.debug(
                    f"High heuristic confidence (overlap={overlap} >= "
                    f"{self._min_tools_for_transition}), skipping edge model"
                )
                return True

        return False

    # Private methods

    def _is_in_cooldown(self) -> bool:
        """Check if currently in cooldown period."""
        time_since_last = time.time() - self._last_transition_time
        return time_since_last < self._cooldown_seconds

    def _calculate_stage_overlap(self, stage: ConversationStage) -> int:
        """Calculate tool overlap with a stage."""
        if not self._current_turn:
            return 0

        stage_tools = self._state_machine._get_tools_for_stage(stage)
        turn_tools = self._current_turn.unique_tools
        return len(turn_tools & stage_tools)

    def _evaluate_transition(self, turn: TurnContext) -> TransitionResult:
        """Evaluate if a transition should occur using the configured strategy."""
        try:
            return self._strategy.detect_transition(
                current_stage=turn.current_stage,
                tools_executed=turn.tools_executed,
                state_machine=self._state_machine,
                coordinator=self,
            )
        except Exception as e:
            logger.error(f"Error in transition strategy: {e}")
            return TransitionResult(
                decision=TransitionDecision.NO_TRANSITION,
                new_stage=None,
                confidence=0.0,
                reason=f"Strategy error: {e}",
                edge_model_called=False,
            )

    def _should_calibrate(self, result: TransitionResult) -> bool:
        """Check if Phase 2 calibration should be applied.

        Calibration corrects edge model bias when:
        - Edge model predicts EXECUTION with high confidence (≥ 0.95)
        - Agent has read many files (> 10) without editing (0 modifications)
        """
        if result.new_stage != ConversationStage.EXECUTION:
            return False

        if result.confidence < 0.95:
            return False

        files_read = len(self._state_machine.state.observed_files)
        files_modified = len(self._state_machine.state.modified_files)

        # Calibrate if reading many files without editing
        if files_read > 10 and files_modified == 0:
            return True

        return False

    def _apply_calibration(self, result: TransitionResult) -> TransitionResult:
        """Apply Phase 2 calibration to the result."""
        files_read = len(self._state_machine.state.observed_files)
        logger.warning(
            f"Edge model calibration: {result.new_stage} ({result.confidence:.2f}) → ANALYSIS. "
            f"Reason: Agent has read {files_read} files without any edits. "
            f"High confidence EXECUTION is likely biased/overconfident."
        )
        return TransitionResult(
            decision=result.decision,
            new_stage=ConversationStage.ANALYSIS,
            confidence=0.7,
            reason="Calibration applied: read-only exploration",
            edge_model_called=result.edge_model_called,
            calibration_applied=True,
        )

    def _execute_transition(self, result: TransitionResult) -> None:
        """Execute the stage transition."""
        if not result.new_stage:
            return

        if not self._current_turn:
            return

        old_stage = self._current_turn.current_stage

        if result.new_stage == old_stage:
            return

        logger.info(
            f"Stage transition: {old_stage} -> {result.new_stage} "
            f"(confidence: {result.confidence:.2f}, reason: {result.reason}, "
            f"edge_model={result.edge_model_called}, calibration={result.calibration_applied})"
        )

        self._state_machine._transition_to(result.new_stage, result.confidence)
        self._last_transition_time = time.time()
        self._transition_count += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics.

        Returns:
            Dictionary with coordinator stats
        """
        return {
            "transition_count": self._transition_count,
            "last_transition_time": self._last_transition_time,
            "current_turn_id": self._current_turn.turn_id if self._current_turn else None,
            "current_turn_tools": self._current_turn.tool_count if self._current_turn else 0,
            "strategy": type(self._strategy).__name__,
        }
