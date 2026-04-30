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

"""Transition strategies for stage detection.

This module provides pluggable strategies for detecting stage transitions:

- HeuristicOnlyTransitionStrategy: Use only heuristic, no edge model
- EdgeModelTransitionStrategy: Always use edge model
- HybridTransitionStrategy: Combine heuristic + edge model (default)

Design Pattern: Strategy
==========================
Strategies implement TransitionStrategyProtocol and can be swapped
without modifying the coordinator. This enables:

- Easy testing (mock strategies)
- Runtime configuration (choose strategy based on feature flags)
- Extensibility (add new strategies without changing coordinator)

Usage:
    strategy = HybridTransitionStrategy(edge_model_enabled=True)
    coordinator = StageTransitionCoordinator(
        state_machine=sm,
        strategy=strategy,
    )
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from victor.core.shared_types import ConversationStage

if TYPE_CHECKING:
    from victor.agent.coordinators.stage_transition_coordinator import (
        StageTransitionCoordinator,
        TransitionDecision,
        TransitionResult,
    )

logger = logging.getLogger(__name__)


class TransitionStrategyProtocol(ABC):
    """Protocol for stage transition strategies.

    All strategies must implement this protocol to be compatible
    with StageTransitionCoordinator.
    """

    @abstractmethod
    def detect_transition(
        self,
        current_stage: ConversationStage,
        tools_executed: List[Tuple[str, Dict[str, Any]]],
        state_machine: Any,
        coordinator: Any,  # StageTransitionCoordinator
    ) -> Any:  # TransitionResult
        """Detect if a stage transition should occur.

        Args:
            current_stage: Current conversation stage
            tools_executed: List of (tool_name, args) tuples executed this turn
            state_machine: ConversationStateMachine instance
            coordinator: StageTransitionCoordinator instance

        Returns:
            TransitionResult with decision, new_stage, confidence, etc.
        """

    @abstractmethod
    def requires_edge_model(self) -> bool:
        """Whether this strategy uses the edge model."""
        ...


class HeuristicOnlyTransitionStrategy:
    """Use only heuristic detection, no edge model.

    This is the fastest option but may be less accurate for ambiguous
    cases. Suitable for:
    - Simple tasks with clear tool patterns
    - When edge model is disabled
    - Performance-critical scenarios

    Advantages:
    - Fast (no LLM calls)
    - Deterministic
    - No external dependencies

    Disadvantages:
    - Less accurate for ambiguous cases
    - May miss subtle stage changes
    """

    def detect_transition(
        self,
        current_stage: ConversationStage,
        tools_executed: List[Tuple[str, Dict[str, Any]]],
        state_machine: Any,
        coordinator: Any,
    ) -> Any:
        """Detect transition using heuristic only."""
        # Use state machine's heuristic detection
        detected = state_machine._detect_stage_from_tools()

        if detected and detected != current_stage:
            # Calculate confidence based on tool overlap
            stage_tools = state_machine._get_tools_for_stage(detected)
            unique_tools = {tool for tool, _ in tools_executed}
            overlap = len(unique_tools & stage_tools)
            confidence = min(0.9, 0.6 + (overlap * 0.1))

            from victor.agent.coordinators.stage_transition_coordinator import (
                TransitionDecision,
                TransitionResult,
            )

            return TransitionResult(
                decision=TransitionDecision.HEURISTIC_TRANSITION,
                new_stage=detected,
                confidence=confidence,
                reason=f"Heuristic detection (overlap={overlap})",
                edge_model_called=False,
            )

        from victor.agent.coordinators.stage_transition_coordinator import (
            TransitionDecision,
            TransitionResult,
        )

        return TransitionResult(
            decision=TransitionDecision.NO_TRANSITION,
            new_stage=None,
            confidence=0.0,
            reason="No heuristic detection",
            edge_model_called=False,
        )

    def requires_edge_model(self) -> bool:
        return False


class EdgeModelTransitionStrategy:
    """Always use edge model for stage detection.

    This is the most accurate option but slower due to LLM calls.
    Suitable for:
    - Complex tasks with ambiguous tool patterns
    - When accuracy is critical
    - Analysis tasks requiring nuanced understanding

    Advantages:
    - Most accurate
    - Handles ambiguous cases
    - Context-aware

    Disadvantages:
    - Slower (LLM calls)
    - Higher cost
    - External dependency
    """

    def __init__(self, edge_model_enabled: bool = True):
        """Initialize the edge model strategy.

        Args:
            edge_model_enabled: Whether edge model is available
        """
        self._edge_model_enabled = edge_model_enabled

    def detect_transition(
        self,
        current_stage: ConversationStage,
        tools_executed: List[Tuple[str, Dict[str, Any]]],
        state_machine: Any,
        coordinator: Any,
    ) -> Any:
        """Detect transition using edge model."""
        if not self._edge_model_enabled:
            # Fallback to heuristic if edge model unavailable
            return HeuristicOnlyTransitionStrategy().detect_transition(
                current_stage=current_stage,
                tools_executed=tools_executed,
                state_machine=state_machine,
                coordinator=coordinator,
            )

        # Use edge model for detection
        edge_stage, edge_confidence = state_machine._try_edge_model_transition(
            heuristic_stage=current_stage,  # Not used, placeholder
            heuristic_confidence=0.5,  # Not used, placeholder
        )

        from victor.agent.coordinators.stage_transition_coordinator import (
            TransitionDecision,
            TransitionResult,
        )

        if edge_stage and edge_stage != current_stage and edge_confidence > 0.6:
            return TransitionResult(
                decision=TransitionDecision.EDGE_MODEL_TRANSITION,
                new_stage=edge_stage,
                confidence=edge_confidence,
                reason=f"Edge model detection (confidence={edge_confidence:.2f})",
                edge_model_called=True,
            )

        return TransitionResult(
            decision=TransitionDecision.NO_TRANSITION,
            new_stage=None,
            confidence=edge_confidence if edge_stage else 0.0,
            reason="Edge model: no transition detected",
            edge_model_called=True,
        )

    def requires_edge_model(self) -> bool:
        return True


class HybridTransitionStrategy:
    """Combine heuristic + edge model for optimal accuracy and performance.

    This is the recommended strategy for most use cases. It uses heuristic
    first (fast) and falls back to edge model only when:
    - Heuristic is uncertain (low tool overlap < threshold)
    - Cooldown period has expired
    - High confidence threshold not met

    Advantages:
    - Fast when confident (skips edge model)
    - Accurate when uncertain (uses edge model)
    - Best of both worlds

    Disadvantages:
    - More complex logic
    - Still requires edge model availability

    Phase 1 Optimizations Applied:
    1. Cooldown check (via coordinator.should_skip_edge_model)
    2. High confidence skip (tool overlap ≥ threshold)
    3. Heuristic first, edge model fallback
    """

    def __init__(self, edge_model_enabled: bool = True):
        """Initialize the hybrid strategy.

        Args:
            edge_model_enabled: Whether edge model is available
        """
        self._edge_model_enabled = edge_model_enabled

    def detect_transition(
        self,
        current_stage: ConversationStage,
        tools_executed: List[Tuple[str, Dict[str, Any]]],
        state_machine: Any,
        coordinator: Any,
    ) -> Any:
        """Detect transition using hybrid approach."""
        # Try heuristic first
        detected = state_machine._detect_stage_from_tools()

        if detected and detected != current_stage:
            # Calculate tool overlap
            stage_tools = state_machine._get_tools_for_stage(detected)
            unique_tools = {tool for tool, _ in tools_executed}
            overlap = len(unique_tools & stage_tools)

            # Get threshold from coordinator
            min_tools = coordinator._min_tools_for_transition

            # High confidence: skip edge model (Phase 1 optimization #2)
            if overlap >= min_tools:
                confidence = min(0.9, 0.6 + (overlap * 0.1))

                from victor.agent.coordinators.stage_transition_coordinator import (
                    TransitionDecision,
                    TransitionResult,
                )

                return TransitionResult(
                    decision=TransitionDecision.HIGH_CONFIDENCE_SKIP,
                    new_stage=detected,
                    confidence=confidence,
                    reason=f"High heuristic confidence (overlap={overlap} >= {min_tools})",
                    edge_model_called=False,
                )

            # Low confidence: consult edge model if available
            if self._edge_model_enabled:
                # Check if we should skip edge model due to cooldown
                if coordinator.should_skip_edge_model(detected, current_stage):
                    # Cooldown or other reason to skip
                    confidence = min(0.7, 0.6 + (overlap * 0.1))

                    from victor.agent.coordinators.stage_transition_coordinator import (
                        TransitionDecision,
                        TransitionResult,
                    )

                    return TransitionResult(
                        decision=TransitionDecision.COOLDOWN_SKIP,
                        new_stage=detected,
                        confidence=confidence,
                        reason="Edge model skipped (cooldown or high confidence)",
                        edge_model_called=False,
                    )

                # Consult edge model
                edge_stage, edge_confidence = state_machine._try_edge_model_transition(
                    heuristic_stage=detected,
                    heuristic_confidence=0.6 + (overlap * 0.1),
                )

                if edge_stage and edge_confidence > 0.6:
                    from victor.agent.coordinators.stage_transition_coordinator import (
                        TransitionDecision,
                        TransitionResult,
                    )

                    return TransitionResult(
                        decision=TransitionDecision.EDGE_MODEL_TRANSITION,
                        new_stage=edge_stage,
                        confidence=edge_confidence,
                        reason=f"Edge model override (confidence={edge_confidence:.2f})",
                        edge_model_called=True,
                    )

            # Fallback to heuristic if edge model unavailable or low confidence
            confidence = min(0.7, 0.6 + (overlap * 0.1))

            from victor.agent.coordinators.stage_transition_coordinator import (
                TransitionDecision,
                TransitionResult,
            )

            return TransitionResult(
                decision=TransitionDecision.HEURISTIC_TRANSITION,
                new_stage=detected,
                confidence=confidence,
                reason=f"Heuristic fallback (overlap={overlap})",
                edge_model_called=False,
            )

        from victor.agent.coordinators.stage_transition_coordinator import (
            TransitionDecision,
            TransitionResult,
        )

        return TransitionResult(
            decision=TransitionDecision.NO_TRANSITION,
            new_stage=None,
            confidence=0.0,
            reason="No transition detected",
            edge_model_called=False,
        )

    def requires_edge_model(self) -> bool:
        return True


def create_transition_strategy(
    strategy_type: str = "hybrid",
    edge_model_enabled: bool = True,
) -> TransitionStrategyProtocol:
    """Factory function to create transition strategies.

    Args:
        strategy_type: Type of strategy ("heuristic", "edge_model", "hybrid")
        edge_model_enabled: Whether edge model is available

    Returns:
        Configured transition strategy instance

    Raises:
        ValueError: If strategy_type is unknown
    """
    if strategy_type == "heuristic":
        return HeuristicOnlyTransitionStrategy()
    elif strategy_type == "edge_model":
        return EdgeModelTransitionStrategy(edge_model_enabled=edge_model_enabled)
    elif strategy_type == "hybrid":
        return HybridTransitionStrategy(edge_model_enabled=edge_model_enabled)
    else:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Valid types: heuristic, edge_model, hybrid"
        )
