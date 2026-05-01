# Phase 1 Streaming Integration Design
## Comprehensive Solution for Stage Transition Optimization

**Date**: 2026-04-30
**Status**: Design Document
**Priority**: P0 - Critical

---

## Executive Summary

**Problem**: The `StreamingChatPipeline` bypasses Phase 1 optimizations (cooldown, high confidence skip), causing:
- 5x more edge model calls (5 per iteration vs. 1 per session)
- Stage thrashing (EXECUTION ↔ ANALYSIS oscillation)
- Reactive calibration fixing after the fact instead of prevention

**Root Cause**: `conversation_state.record_tool_execution()` is called for EACH tool execution, triggering `_maybe_transition()` per tool instead of once per turn.

**Solution**: Implement a **Stage Transition Coordinator** that batches tool executions and applies Phase 1 optimizations consistently across both streaming and non-streaming paths.

---

## Current Architecture

### Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Non-Streaming Path                       │
│                     (AgenticLoop)                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Execute all tools in one turn                           │
│ 2. Call record_tool_execution() for each tool              │
│ 3. Call _maybe_transition() ONCE after all tools           │
│ 4. Result: 1 edge model call per session ✅                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Streaming Path                          │
│                 (StreamingChatPipeline)                     │
├─────────────────────────────────────────────────────────────┤
│ 1. Execute tools one-by-one as they stream in             │
│ 2. Call record_tool_execution() for EACH tool             │
│ 3. Call _maybe_transition() FOR EACH TOOL ❌               │
│ 4. Result: 5+ edge model calls per iteration ❌            │
└─────────────────────────────────────────────────────────────┘
```

### Current Call Site

**File**: `victor/agent/services/tool_service.py:265`
```python
# Called for EACH tool execution
if ctx.conversation_state:
    ctx.conversation_state.record_tool_execution(tool_name, normalized_args)
    # ↑ This triggers _maybe_transition() for EACH tool
```

---

## Design Principles

### SOLID Principles

1. **Single Responsibility Principle (SRP)**
   - Separate concerns: tool execution tracking vs. stage transition decisions
   - Coordinator handles ONLY transition logic, not tool execution

2. **Open/Closed Principle (OCP)**
   - Extensible transition strategies without modifying existing code
   - New optimization strategies can be added via configuration

3. **Liskov Substitution Principle (LSP)**
   - Streaming and non-streaming paths use same transition protocol
   - Either path can be swapped without breaking functionality

4. **Interface Segregation Principle (ISP)**
   - Focused protocols: only what's needed for stage transitions
   - No unnecessary dependencies

5. **Dependency Inversion Principle (DIP)**
   - Depend on abstractions (protocols), not concrete implementations
   - Easy to test and mock

### Additional Principles

- **Separation of Concerns**: Tool tracking separate from transition logic
- **Batch Processing**: Collect tool executions, process once
- **Strategy Pattern**: Pluggable transition strategies
- **Observer Pattern**: Decouple state changes from side effects
- **Immutability**: Transition state is immutable, new state on change

---

## Proposed Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                           │
│  ┌──────────────────┐        ┌──────────────────┐             │
│  │ AgenticLoop       │        │ StreamingChat    │             │
│  │ (Non-streaming)   │        │ Pipeline          │             │
│  └────────┬─────────┘        └────────┬─────────┘             │
└───────────┼──────────────────────────┼──────────────────────────┘
            │                          │
            └──────────┬───────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Coordination Layer                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │       StageTransitionCoordinator (NEW)                    │ │
│  │  - Batches tool executions                                │ │
│  │  - Applies Phase 1 optimizations                          │ │
│  │  - Delegates to strategies                                │ │
│  └────────┬───────────────────────────────────────────────────┘ │
└───────────┼──────────────────────────────────────────────────────┘
            │
            ├──────────────────┬──────────────────┐
            ▼                  ▼                  ▼
┌───────────────────┐  ┌──────────────┐  ┌──────────────────┐
│  TransitionStrategy│  │ CooldownMgr  │  │  ConfidenceCache │
│  Protocol (NEW)   │  │  (Refactored)│  │   (NEW)          │
└───────────────────┘  └──────────────┘  └──────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    State Layer                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         ConversationStateMachine (Enhanced)                │ │
│  │  - record_tool_execution() - NO transition logic          │ │
│  │  - _maybe_transition() - PRIVATE, called by coordinator  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. StageTransitionCoordinator (NEW)

**Purpose**: Batch tool executions and apply Phase 1 optimizations once per turn

**Responsibilities**:
- Collect tool executions during a turn
- Check cooldown before any edge model calls
- Apply high confidence skip logic
- Delegate to appropriate transition strategy
- Emit transition events

**Protocol**:
```python
@runtime_checkable
class StageTransitionCoordinatorProtocol(Protocol):
    """Protocol for stage transition coordination."""

    def begin_turn(self) -> None:
        """Mark the start of a new turn (batching window)."""
        ...

    def record_tool(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record a tool execution (batched, no immediate transition)."""
        ...

    def end_turn(self) -> Optional[ConversationStage]:
        """Process batched tools and return new stage if transition occurred."""
        ...

    def should_skip_edge_model(self, detected_stage: ConversationStage) -> bool:
        """Check if edge model should be skipped based on Phase 1 optimizations."""
        ...
```

#### 2. TransitionStrategy (NEW)

**Purpose**: Strategy pattern for different transition algorithms

**Strategies**:
- `HeuristicOnlyTransitionStrategy`: Use only heuristic, no edge model
- `EdgeModelTransitionStrategy`: Use edge model with cooldown
- `HybridTransitionStrategy`: Combine heuristic + edge model (default)

**Protocol**:
```python
@runtime_checkable
class TransitionStrategyProtocol(Protocol):
    """Protocol for stage transition strategies."""

    def detect_transition(
        self,
        current_stage: ConversationStage,
        tools_executed: List[Tuple[str, Dict[str, Any]]],
        confidence_threshold: float,
    ) -> Optional[Tuple[ConversationStage, float]]:
        """Detect if a stage transition should occur."""
        ...

    def requires_edge_model(self) -> bool:
        """Whether this strategy uses the edge model."""
        ...
```

#### 3. ConversationStateMachine (Enhanced)

**Changes**:
- `record_tool_execution()`: Remove `_maybe_transition()` call
- `_maybe_transition()`: Make PRIVATE, called only by coordinator
- Add `transition_coordinator`: Optional coordinator reference

**Before**:
```python
def record_tool_execution(self, tool_name: str, args: Dict[str, Any]) -> None:
    self.state.record_tool_execution(tool_name, args)
    if self._state_manager:
        self._sync_state_to_manager()
    self._maybe_transition()  # ← Called for EACH tool
```

**After**:
```python
def record_tool_execution(self, tool_name: str, args: Dict[str, Any]) -> None:
    self.state.record_tool_execution(tool_name, args)
    if self._state_manager:
        self._sync_state_to_manager()
    # No immediate transition - delegated to coordinator
    if self._transition_coordinator:
        self._transition_coordinator.record_tool(tool_name, args)

def _maybe_transition(self) -> None:  # PRIVATE
    """Internal transition logic, called by coordinator only."""
    ...
```

---

## Implementation Plan

### Phase 1: Foundation (1-2 days)

**Tasks**:
1. ✅ Create `StageTransitionCoordinatorProtocol`
2. ✅ Create `TransitionStrategyProtocol`
3. ✅ Implement `StageTransitionCoordinator`
4. ✅ Implement `HybridTransitionStrategy`
5. ✅ Add unit tests for coordinator
6. ✅ Add unit tests for strategies

**Files**:
- `victor/agent/coordinators/stage_transition_coordinator.py` (NEW)
- `victor/agent/coordinators/transition_strategies.py` (NEW)
- `tests/unit/agent/coordinators/test_stage_transition_coordinator.py` (NEW)
- `tests/unit/agent/coordinators/test_transition_strategies.py` (NEW)

### Phase 2: Integration (2-3 days)

**Tasks**:
1. ✅ Enhance `ConversationStateMachine` with coordinator support
2. ✅ Update `ToolService` to use coordinator
3. ✅ Add coordinator to `ComponentAssembler`
4. ✅ Update `AgenticLoop` integration
5. ✅ Update `StreamingChatPipeline` integration
6. ✅ Add integration tests

**Files**:
- `victor/agent/conversation/state_machine.py` (MODIFY)
- `victor/agent/services/tool_service.py` (MODIFY)
- `victor/agent/runtime/component_assembler.py` (MODIFY)
- `victor/framework/agentic_loop.py` (MODIFY)
- `victor/agent/streaming/pipeline.py` (MODIFY)
- `tests/integration/agent/test_stage_transition_integration.py` (NEW)

### Phase 3: Testing & Validation (1-2 days)

**Tasks**:
1. ✅ Run existing test suite (ensure no regressions)
2. ✅ Add regression tests for streaming path
3. ✅ Add performance benchmarks
4. ✅ Test with real workloads
5. ✅ Monitor edge model call frequency
6. ✅ Verify calibration still works

**Files**:
- `tests/regression/test_stage_transition_regression.py` (NEW)
- `tests/benchmark/test_stage_transition_performance.py` (NEW)

---

## Detailed Design

### StageTransitionCoordinator Implementation

```python
"""Stage transition coordination for Phase 1 optimization."""

import logging
import time
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
    """Context for a single turn (batching window)."""

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
    """Result of a transition evaluation."""

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
        strategy: Any,  # TransitionStrategyProtocol
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

    def begin_turn(self) -> None:
        """Mark the start of a new turn.

        Creates a new batching window for tool executions.
        """
        import uuid

        self._current_turn = TurnContext(
            turn_id=str(uuid.uuid4()),
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
        2. Check high confidence (skip if tool overlap >= threshold)
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
            f"unique={len(turn.unique_tools)}"
        )

        # Check cooldown
        if self._is_in_cooldown():
            logger.debug("In cooldown period, skipping transition evaluation")
            return None

        # Get transition result from strategy
        result = self._evaluate_transition(turn)

        # Apply calibration if needed
        if self._should_calibrate(result):
            result = self._apply_calibration(result)

        # Execute transition if needed
        if result.decision in (
            TransitionDecision.HEURISTIC_TRANSITION,
            TransitionDecision.EDGE_MODEL_TRANSITION,
        ):
            self._execute_transition(result)
            return result.new_stage

        return None

    def should_skip_edge_model(
        self, detected_stage: ConversationStage, current_stage: ConversationStage
    ) -> bool:
        """Check if edge model should be skipped based on Phase 1 optimizations.

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
        return self._strategy.detect_transition(
            current_stage=turn.current_stage,
            tools_executed=turn.tools_executed,
            state_machine=self._state_machine,
            min_tools_for_transition=self._min_tools_for_transition,
        )

    def _should_calibrate(self, result: TransitionResult) -> bool:
        """Check if calibration should be applied."""
        # Phase 2 calibration: files read > 10 and files modified = 0
        if result.new_stage == ConversationStage.EXECUTION and result.confidence >= 0.95:
            files_read = len(self._state_machine.state.observed_files)
            files_modified = len(self._state_machine.state.modified_files)
            if files_read > 10 and files_modified == 0:
                return True
        return False

    def _apply_calibration(self, result: TransitionResult) -> TransitionResult:
        """Apply Phase 2 calibration to the result."""
        logger.warning(
            f"Edge model calibration: {result.new_stage} ({result.confidence:.2f}) → ANALYSIS. "
            f"Reason: Agent has read {len(self._state_machine.state.observed_files)} files "
            f"without any edits. High confidence EXECUTION is likely biased/overconfident."
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
        if not result.new_stage or result.new_stage == self._current_turn.current_stage:
            return

        logger.info(
            f"Stage transition: {self._current_turn.current_stage} -> {result.new_stage} "
            f"(confidence: {result.confidence:.2f}, reason: {result.reason})"
        )

        self._state_machine._transition_to(result.new_stage, result.confidence)
        self._last_transition_time = time.time()
        self._transition_count += 1
```

### TransitionStrategy Implementation

```python
"""Transition strategies for stage detection."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from victor.core.shared_types import ConversationStage

logger = logging.getLogger(__name__)


class TransitionStrategyProtocol(ABC):
    """Protocol for stage transition strategies."""

    @abstractmethod
    def detect_transition(
        self,
        current_stage: ConversationStage,
        tools_executed: List[Tuple[str, Dict[str, Any]]],
        state_machine: Any,
        min_tools_for_transition: int,
    ) -> Any:  # TransitionResult
        """Detect if a stage transition should occur."""
        ...

    @abstractmethod
    def requires_edge_model(self) -> bool:
        """Whether this strategy uses the edge model."""
        ...


class HeuristicOnlyTransitionStrategy:
    """Use only heuristic detection, no edge model.

    Fastest option, suitable for simple tasks or when edge model is disabled.
    """

    def detect_transition(
        self,
        current_stage: ConversationStage,
        tools_executed: List[Tuple[str, Dict[str, Any]]],
        state_machine: Any,
        min_tools_for_transition: int,
    ) -> Any:
        """Detect transition using heuristic only."""
        # Use state machine's heuristic detection
        detected = state_machine._detect_stage_from_tools()

        if detected and detected != current_stage:
            # Calculate confidence based on tool overlap
            stage_tools = state_machine._get_tools_for_stage(detected)
            unique_tools = {tool for tool, _ in tools_executed}
            overlap = len(unique_tools & stage_tools)
            confidence = 0.6 + (overlap * 0.1)

            return TransitionResult(
                decision=TransitionDecision.HEURISTIC_TRANSITION,
                new_stage=detected,
                confidence=confidence,
                reason=f"Heuristic detection (overlap={overlap})",
                edge_model_called=False,
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


class HybridTransitionStrategy:
    """Combine heuristic + edge model for optimal accuracy.

    Uses heuristic first, falls back to edge model only when:
    - Heuristic is uncertain (low tool overlap)
    - Cooldown period has expired
    - High confidence threshold not met
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
        min_tools_for_transition: int,
    ) -> Any:
        """Detect transition using hybrid approach."""
        # Try heuristic first
        detected = state_machine._detect_stage_from_tools()

        if detected and detected != current_stage:
            # Calculate tool overlap
            stage_tools = state_machine._get_tools_for_stage(detected)
            unique_tools = {tool for tool, _ in tools_executed}
            overlap = len(unique_tools & stage_tools)

            # High confidence: skip edge model
            if overlap >= min_tools_for_transition:
                confidence = 0.6 + (overlap * 0.1)
                return TransitionResult(
                    decision=TransitionDecision.HIGH_CONFIDENCE_SKIP,
                    new_stage=detected,
                    confidence=confidence,
                    reason=f"High heuristic confidence (overlap={overlap})",
                    edge_model_called=False,
                )

            # Low confidence: consult edge model if available
            if self._edge_model_enabled:
                edge_stage, edge_confidence = state_machine._try_edge_model_transition(
                    heuristic_stage=detected,
                    heuristic_confidence=0.6 + (overlap * 0.1),
                )

                if edge_stage and edge_confidence > 0.6:
                    return TransitionResult(
                        decision=TransitionDecision.EDGE_MODEL_TRANSITION,
                        new_stage=edge_stage,
                        confidence=edge_confidence,
                        reason=f"Edge model override (confidence={edge_confidence:.2f})",
                        edge_model_called=True,
                    )

            # Fallback to heuristic if edge model unavailable or low confidence
            confidence = 0.6 + (overlap * 0.1)
            return TransitionResult(
                decision=TransitionDecision.HEURISTIC_TRANSITION,
                new_stage=detected,
                confidence=confidence,
                reason=f"Heuristic fallback (overlap={overlap})",
                edge_model_called=False,
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
```

---

## Integration Points

### 1. ToolService Integration

**File**: `victor/agent/services/tool_service.py`

**Before**:
```python
if ctx.conversation_state:
    ctx.conversation_state.record_tool_execution(tool_name, normalized_args)
    # ↑ Triggers _maybe_transition() for EACH tool
```

**After**:
```python
if ctx.conversation_state:
    ctx.conversation_state.record_tool_execution(tool_name, normalized_args)
    # ↑ Only records state, no transition

if ctx.transition_coordinator:
    # Batching handled by coordinator
    ctx.transition_coordinator.record_tool(tool_name, normalized_args)
```

### 2. AgenticLoop Integration

**File**: `victor/framework/agentic_loop.py`

**Add**:
```python
def run(self, query: str, **kwargs) -> Any:
    """Run the agentic loop with stage transition coordination."""
    # Begin turn
    if self._transition_coordinator:
        self._transition_coordinator.begin_turn()

    try:
        # Execute turn
        result = self._execute_turn(query, **kwargs)

        # End turn - process transitions
        if self._transition_coordinator:
            new_stage = self._transition_coordinator.end_turn()
            if new_stage:
                logger.info(f"Stage transition occurred: {new_stage}")

        return result
    finally:
        # Always end turn
        if self._transition_coordinator:
            self._transition_coordinator.end_turn()
```

### 3. StreamingChatPipeline Integration

**File**: `victor/agent/streaming/pipeline.py`

**Add**:
```python
async def _process_tool_calls(
    self,
    tool_calls: List[Dict[str, Any]],
    stream_ctx: Any,
) -> AsyncIterator[StreamChunk]:
    """Process tool calls with stage transition coordination."""
    # Begin turn
    if hasattr(self._runtime_owner, 'transition_coordinator'):
        self._runtime_owner.transition_coordinator.begin_turn()

    try:
        # Process each tool call
        for tool_call in tool_calls:
            result = await self._execute_tool(tool_call, stream_ctx)

            # Record tool execution (batched)
            if hasattr(self._runtime_owner, 'transition_coordinator'):
                self._runtime_owner.transition_coordinator.record_tool(
                    tool_name=tool_call["name"],
                    args=tool_call.get("arguments", {}),
                )

            yield result

        # End turn - process transitions
        if hasattr(self._runtime_owner, 'transition_coordinator'):
            new_stage = self._runtime_owner.transition_coordinator.end_turn()
            if new_stage:
                logger.info(f"Stage transition occurred: {new_stage}")

    finally:
        # Always end turn
        if hasattr(self._runtime_owner, 'transition_coordinator'):
            self._runtime_owner.transition_coordinator.end_turn()
```

---

## Testing Strategy

### Unit Tests

**Test Coverage**:
1. **Coordinator Tests**:
   - `test_begin_turn_creates_context`
   - `test_record_tool_adds_to_context`
   - `test_end_turn_processes_batch`
   - `test_cooldown_prevents_transition`
   - `test_high_confidence_skips_edge_model`
   - `test_strategy_delegation`
   - `test_calibration_applied`

2. **Strategy Tests**:
   - `test_heuristic_only_detection`
   - `test_hybrid_high_confidence_skip`
   - `test_hybrid_low_confidence_edge_model`
   - `test_hybrid_edge_model_unavailable`

3. **Integration Tests**:
   - `test_tool_service_batches_transitions`
   - `test_agentic_loop_uses_coordinator`
   - `test_streaming_pipeline_uses_coordinator`
   - `test_both_paths_consistent_behavior`

### Regression Tests

**Scenarios**:
1. **Session codingagent-9f788887 regression**:
   - Verify edge model calls reduced from 5 to 1 per iteration
   - Verify no stage thrashing
   - Verify calibration still works

2. **Session codingagent-54930a73 consistency**:
   - Verify non-streaming path still works
   - Verify 1 edge model call per session
   - Verify optimal stage transitions

### Performance Benchmarks

**Metrics**:
- Edge model call frequency (per session, per iteration)
- Stage transition frequency
- Cooldown hit rate
- High confidence skip rate
- Calibration accuracy

**Targets**:
- Edge model calls: ≤ 2 per iteration (down from 5+)
- Stage transitions: ≤ 1 per iteration (down from 2-3)
- Cooldown hit rate: ≥ 60%
- High confidence skip rate: ≥ 40%

---

## Migration Path

### Phase 1: Add Coordinator (Non-Breaking)

1. Create new coordinator components
2. Add optional coordinator to ConversationStateMachine
3. Add coordinator to ComponentAssembler (disabled by default)
4. Add feature flag: `USE_STAGE_TRANSITION_COORDINATOR` (default: False)

### Phase 2: Enable for Streaming Path (Non-Breaking)

1. Enable coordinator for streaming path when feature flag is True
2. Add telemetry to compare coordinator vs. legacy
3. Monitor edge model call frequency
4. Verify no regressions

### Phase 3: Enable for All Paths (Breaking Change)

1. Set feature flag to True by default
2. Update all integration points
3. Remove legacy `_maybe_transition()` calls from tool recording
4. Deprecate direct `_maybe_transition()` access
5. Update documentation

### Phase 4: Remove Legacy Code (Future Release)

1. Remove feature flag
2. Remove `_maybe_transition()` public method
3. Make coordinator mandatory
4. Clean up old code paths

---

## Rollback Plan

If issues occur:
1. Disable feature flag: `USE_STAGE_TRANSITION_COORDINATOR=false`
2. System reverts to legacy behavior
3. No breaking changes to existing functionality

---

## Success Criteria

### Functional Requirements

✅ **Phase 1 optimizations work in streaming path**:
- Cooldown prevents edge model calls within 2 seconds
- High confidence skip (≥ 5 tools) bypasses edge model
- Edge model called ≤ 2 per iteration (down from 5+)

✅ **No stage thrashing**:
- Stage transitions ≤ 1 per iteration
- No EXECUTION ↔ ANALYSIS oscillation
- Calibration still works correctly

✅ **Backward compatibility**:
- Non-streaming path still works
- Feature flag allows rollback
- No breaking changes to public APIs

### Non-Functional Requirements

✅ **Performance**:
- No measurable overhead from coordinator
- Batch processing is efficient
- Memory usage not increased significantly

✅ **Testability**:
- All components have unit tests
- Integration tests cover both paths
- Regression tests prevent future issues

✅ **Maintainability**:
- Clean separation of concerns
- Well-documented code
- Easy to understand and modify

---

## Risks and Mitigations

### Risk 1: Coordinator Adds Overhead

**Mitigation**:
- Benchmark before/after
- Use efficient data structures (lists, sets)
- Minimize allocations in hot path

### Risk 2: Breaking Changes

**Mitigation**:
- Feature flag for gradual rollout
- Extensive testing before enabling
- Clear rollback plan

### Risk 3: Integration Complexity

**Mitigation**:
- Protocol-based design (easy to mock)
- Clear integration points
- Comprehensive integration tests

### Risk 4: Calibration Interference

**Mitigation**:
- Keep calibration logic separate
- Test calibration with coordinator
- Monitor calibration accuracy

---

## Future Enhancements

### Adaptive Thresholds

- Dynamically adjust `min_tools_for_transition` based on task type
- Learn optimal thresholds from historical data
- Use RL to optimize transition decisions

### Predictive Transitions

- Use ML to predict next stage
- Pre-fetch stage-specific tools
- Reduce latency for stage changes

### Multi-Path Coordination

- Coordinator for multiple execution paths
- Unified transition strategy across all paths
- Shared state and metrics

---

## Conclusion

This design provides a **robust, scalable solution** to the Phase 1 bypassing issue in the streaming path by:

1. **Separating concerns**: Tool tracking separate from transition logic
2. **Batching processing**: Collect tools, process once per turn
3. **Strategy pattern**: Pluggable transition algorithms
4. **Protocol-based design**: Easy to test and extend
5. **Feature flags**: Gradual rollout, easy rollback

**Expected Impact**:
- Edge model calls: **5+ → ≤ 2 per iteration** (60%+ reduction)
- Stage transitions: **2-3 → ≤ 1 per iteration** (50%+ reduction)
- Stage thrashing: **Eliminated**
- Calibration: **Still works correctly**

**Timeline**: 5-7 days for full implementation and testing

**Next Step**: Begin Phase 1 implementation (Foundation)
