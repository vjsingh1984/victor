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

"""Agentic Loop - High-level task lifecycle orchestration.

This module provides a PERCEIVE -> PLAN -> ACT -> EVALUATE -> DECIDE loop
that sits ABOVE the AgentOrchestrator/ExecutionCoordinator layer:

    AgenticLoop (this module)     -- task lifecycle: what to do, is it done?
        └── AgentOrchestrator     -- single-turn execution: LLM call + tools

AgenticLoop owns:
  - Intent perception (ActionIntent + TaskAnalyzer)
  - Execution planning (PlanningCoordinator)
  - Quality evaluation (EvaluationNode, FulfillmentDetector)
  - Adaptive iteration (plateau detection, convergence, extension)
  - Progress tracking (SubSearch intermediate rewards)

AgentOrchestrator owns:
  - LLM provider calls and tool execution
  - Token/context management
  - Provider failover and retry

Design Principle: INTEGRATE existing components, don't build new ones.

Based on research from:
- arXiv:2604.07415 - SubSearch (intermediate rewards / progress tracking)
- arXiv:2604.01681 - Fast-Slow Architecture (adaptive iterations)
- arXiv:2604.00445 - Truth-Aligned Uncertainty (calibrated confidence)

Example:
    from victor.framework.agentic_loop import AgenticLoop

    loop = AgenticLoop(
        orchestrator=orchestrator,
        memory_coordinator=memory,
        max_iterations=5,
    )

    result = await loop.run(
        query="Fix the authentication bug",
        context={"project": "myapp"}
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from victor.framework.evaluation_nodes import (
    EvaluationDecision,
    EvaluationResult,
    create_agentic_loop_graph,
)
from victor.framework.fulfillment import FulfillmentDetector, TaskType
from victor.framework.perception_integration import Perception, PerceptionIntegration

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.agent.coordinators.execution_coordinator import (
        ExecutionCoordinator,
        TurnResult,
    )
    from victor.storage.memory.unified import UnifiedMemoryCoordinator
    from victor.agent.coordinators.planning_coordinator import PlanningCoordinator
    from victor.framework.agent import Agent


class LoopStage(Enum):
    """Stages in the agentic loop."""

    PERCEIVE = "perceive"
    PLAN = "plan"
    ACT = "act"
    EVALUATE = "evaluate"
    DECIDE = "decide"
    COMPLETE = "complete"


@dataclass
class LoopIteration:
    """Single iteration in the agentic loop.

    Attributes:
        iteration: Iteration number
        stage: Current stage
        perception: Perception result
        plan: Generated plan
        action_result: Result from action execution
        evaluation: Evaluation result
        fulfillment: Fulfillment check result
        timestamp: When iteration occurred
    """

    iteration: int
    stage: LoopStage
    perception: Optional[Perception] = None
    plan: Optional[Any] = None
    action_result: Optional[Any] = None
    evaluation: Optional[EvaluationResult] = None
    fulfillment: Optional[Any] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "stage": self.stage.value,
            "perception": self.perception.to_dict() if self.perception else None,
            "evaluation": (
                {
                    "decision": str(self.evaluation.decision),
                    "score": self.evaluation.score,
                    "reason": self.evaluation.reason,
                }
                if self.evaluation
                else None
            ),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LoopResult:
    """Final result from agentic loop execution.

    Attributes:
        success: Whether loop completed successfully
        iterations: All iterations
        final_state: Final state
        total_duration: Total execution time
        metadata: Additional metadata
    """

    success: bool
    iterations: List[LoopIteration]
    final_state: Dict[str, Any]
    total_duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "iterations_count": len(self.iterations),
            "total_duration": self.total_duration,
            "final_state": self.final_state,
            "metadata": self.metadata,
        }


class AgenticLoop:
    """Complete agentic loop integrating existing Victor components.

    Components:
    - PerceptionIntegration: Uses ActionIntent + TaskAnalyzer
    - PlanningCoordinator: Uses existing SkillPlanner + ToolPlanner
    - StateGraph: Extended with EvaluationNode
    - FulfillmentDetector: Task completion detection

    Loop Flow:
    1. PERCEIVE: Understand user intent
    2. PLAN: Generate execution plan
    3. ACT: Execute plan with tools/agents
    4. EVALUATE: Check progress and quality
    5. DECIDE: Continue, retry, or complete
    6. Repeat until complete or max iterations

    Example:
        loop = AgenticLoop(
            orchestrator=orchestrator,
            memory_coordinator=memory,
        )

        result = await loop.run(
            query="Add user authentication",
            context={"project": "myapp"}
        )

        print(f"Success: {result.success}")
        print(f"Iterations: {len(result.iterations)}")
    """

    def __init__(
        self,
        orchestrator: Optional[Any] = None,  # PlanningContextProtocol
        execution_coordinator: Optional["ExecutionCoordinator"] = None,
        memory_coordinator: Optional["UnifiedMemoryCoordinator"] = None,
        max_iterations: int = 10,
        enable_fulfillment_check: bool = True,
        enable_adaptive_iterations: bool = True,
        plateau_window: int = 3,
        plateau_tolerance: float = 0.02,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize agentic loop.

        Args:
            orchestrator: Victor orchestrator (used for planning + fallback)
            execution_coordinator: ExecutionCoordinator for single-turn execution.
                If provided, AgenticLoop calls execute_turn_with_tools() directly
                (one LLM call per iteration). If None, falls back to
                orchestrator.chat() (which runs its own multi-turn loop).
            memory_coordinator: Optional unified memory for context retrieval
            max_iterations: Maximum loop iterations (can be adapted at runtime)
            enable_fulfillment_check: Whether to check task fulfillment
            enable_adaptive_iterations: Allow early exit on plateau / extension
                on near-completion (Fast-Slow Architecture, arXiv:2604.01681)
            plateau_window: Iterations to check for progress plateau
            plateau_tolerance: Min score delta to not be a plateau
            config: Additional configuration
        """
        self.orchestrator = orchestrator
        self.execution_coordinator = execution_coordinator
        self.memory_coordinator = memory_coordinator
        self.max_iterations = max_iterations
        self.enable_fulfillment_check = enable_fulfillment_check
        self.enable_adaptive_iterations = enable_adaptive_iterations
        self.plateau_window = plateau_window
        self.plateau_tolerance = plateau_tolerance
        self.config = config or {}

        # Progress tracking (SubSearch arXiv:2604.07415 intermediate rewards)
        self._progress_scores: List[float] = []

        # Turn-level state (absorbed from ExecutionCoordinator's while-loop)
        self._consecutive_no_tool_turns: int = 0
        self._consecutive_all_blocked: int = 0
        self._total_tool_calls: int = 0
        self._MAX_NO_TOOL_TURNS: int = 3
        self._MAX_ALL_BLOCKED: int = 3

        # Initialize components
        self.perception = PerceptionIntegration(
            memory_coordinator=memory_coordinator,
        )

        self.fulfillment = FulfillmentDetector() if enable_fulfillment_check else None

        # Try to extract execution_coordinator from orchestrator if not provided
        if self.execution_coordinator is None and hasattr(orchestrator, "execution_coordinator"):
            self.execution_coordinator = orchestrator.execution_coordinator

        mode = "single-turn" if self.execution_coordinator else "orchestrator-fallback"
        logger.info(
            f"AgenticLoop initialized "
            f"(mode={mode}, max_iterations={max_iterations}, "
            f"adaptive={enable_adaptive_iterations}, "
            f"fulfillment={enable_fulfillment_check})"
        )

    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> LoopResult:
        """Run the complete agentic loop.

        Args:
            query: User's natural language query
            context: Additional execution context
            conversation_history: Prior turns for multi-turn context

        Returns:
            LoopResult with all iterations and final state
        """
        import time

        start_time = time.time()
        iterations: List[LoopIteration] = []
        state: Dict[str, Any] = {"query": query, **(context or {})}
        self._progress_scores = []
        self._consecutive_no_tool_turns = 0
        self._consecutive_all_blocked = 0
        self._total_tool_calls = 0
        effective_max = self.max_iterations

        try:
            for i in range(1, effective_max + 1):
                iteration = LoopIteration(iteration=i, stage=LoopStage.PERCEIVE)

                # PERCEIVE
                logger.info(f"[Iteration {i}/{effective_max}] PERCEIVE")
                perception = await self.perception.perceive(query, context, conversation_history)
                iteration.perception = perception
                state["perception"] = perception.to_dict()

                # PLAN (use existing PlanningCoordinator if available)
                logger.info(f"[Iteration {i}/{effective_max}] PLAN")
                plan = await self._plan(perception, state)
                iteration.plan = plan
                state["plan"] = plan

                # ACT
                logger.info(f"[Iteration {i}/{effective_max}] ACT")
                action_result = await self._act(plan, state)
                iteration.action_result = action_result
                state["action_result"] = action_result

                # EVALUATE
                logger.info(f"[Iteration {i}/{effective_max}] EVALUATE")
                evaluation = await self._evaluate(perception, action_result, state)
                iteration.evaluation = evaluation
                iteration.stage = LoopStage.EVALUATE

                # Track progress (SubSearch intermediate rewards)
                self._progress_scores.append(evaluation.score)

                # FULFILLMENT CHECK (optional)
                if self.enable_fulfillment_check:
                    logger.info(f"[Iteration {i}/{effective_max}] FULFILLMENT")
                    fulfillment = await self._check_fulfillment(perception, state)
                    iteration.fulfillment = fulfillment
                    state["fulfillment"] = fulfillment

                # ADAPTIVE ITERATION CHECK (Fast-Slow Architecture)
                if self.enable_adaptive_iterations:
                    adaptive_decision = self._check_adaptive_termination(i, evaluation)
                    if adaptive_decision == "plateau":
                        logger.info("Progress plateaued - exiting loop early")
                        evaluation = EvaluationResult(
                            decision=EvaluationDecision.FAIL,
                            score=evaluation.score,
                            reason=f"Progress plateaued after {i} iterations",
                        )
                        iteration.evaluation = evaluation
                    elif adaptive_decision == "extend" and i >= effective_max:
                        # Near completion — grant extra iterations (up to 50% more)
                        extension = min(3, self.max_iterations // 2)
                        effective_max = min(
                            self.max_iterations + extension, effective_max + extension
                        )
                        logger.info(f"Near completion - extending to {effective_max} iterations")

                # DECIDE
                logger.info(f"[Iteration {i}/{effective_max}] DECIDE: {evaluation.decision}")
                iteration.stage = LoopStage.DECIDE
                iterations.append(iteration)

                # Check termination conditions
                if evaluation.decision == EvaluationDecision.COMPLETE:
                    logger.info("Task complete - exiting loop")
                    break

                if evaluation.decision == EvaluationDecision.FAIL:
                    logger.warning("Task failed - exiting loop")
                    break

                # NUDGE INJECTION: when agent isn't using tools, inject
                # a nudge into the conversation so the LLM sees it on
                # the next turn. This replaces the nudge logic that was
                # previously inside ExecutionCoordinator's while-loop.
                if (
                    evaluation.decision == EvaluationDecision.RETRY
                    and self.execution_coordinator is not None
                    and self._consecutive_no_tool_turns >= 2
                ):
                    nudge = (
                        "You have not called any tools in the last "
                        f"{self._consecutive_no_tool_turns} turns. You MUST use "
                        "a tool now (read, edit, write, shell) to make progress "
                        "on the task. Do not respond with text only."
                    )
                    chat_ctx = getattr(self.execution_coordinator, "_chat_context", None)
                    if chat_ctx and hasattr(chat_ctx, "add_message"):
                        chat_ctx.add_message("user", nudge)
                        logger.info(
                            f"Nudge injected after {self._consecutive_no_tool_turns} "
                            "no-tool turns"
                        )

                    # Budget warning when past halfway
                    if i > effective_max // 2:
                        remaining = effective_max - i
                        if chat_ctx and hasattr(chat_ctx, "add_message"):
                            chat_ctx.add_message(
                                "system",
                                f"WARNING: {remaining} turns remaining out of "
                                f"{effective_max}. Make your edits NOW.",
                            )

                # Spin detection nudge: all tools blocked by dedup
                if self._consecutive_all_blocked >= 2 and self.execution_coordinator is not None:
                    chat_ctx = getattr(self.execution_coordinator, "_chat_context", None)
                    if chat_ctx and hasattr(chat_ctx, "add_message"):
                        chat_ctx.add_message(
                            "system",
                            "Your last tool calls were blocked because you "
                            "already called them with the same arguments. Try "
                            "a DIFFERENT tool or different arguments. If you've "
                            "made your fix, provide your final answer.",
                        )

            # Determine success
            success = self._determine_success(iterations)

            duration = time.time() - start_time

            return LoopResult(
                success=success,
                iterations=iterations,
                final_state=state,
                total_duration=duration,
                metadata={
                    "iterations_completed": len(iterations),
                    "max_iterations_reached": len(iterations) >= self.max_iterations,
                    "effective_max_iterations": effective_max,
                    "progress_scores": list(self._progress_scores),
                },
            )

        except Exception as e:
            logger.error(f"Agentic loop error: {e}", exc_info=True)
            duration = time.time() - start_time

            return LoopResult(
                success=False,
                iterations=iterations,
                final_state=state,
                total_duration=duration,
                metadata={
                    "error": str(e),
                    "progress_scores": list(self._progress_scores),
                },
            )

    async def stream(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[LoopIteration]:
        """Stream agentic loop iterations.

        Args:
            query: User's natural language query
            context: Additional execution context

        Yields:
            LoopIteration for each iteration
        """
        state: Dict[str, Any] = {"query": query, **(context or {})}

        for i in range(1, self.max_iterations + 1):
            iteration = LoopIteration(iteration=i, stage=LoopStage.PERCEIVE)

            # PERCEIVE
            perception = await self.perception.perceive(query, context)
            iteration.perception = perception
            state["perception"] = perception.to_dict()
            yield iteration

            # PLAN
            plan = await self._plan(perception, state)
            iteration.plan = plan
            state["plan"] = plan
            yield iteration

            # ACT
            action_result = await self._act(plan, state)
            iteration.action_result = action_result
            state["action_result"] = action_result
            yield iteration

            # EVALUATE
            evaluation = await self._evaluate(perception, action_result, state)
            iteration.evaluation = evaluation
            iteration.stage = LoopStage.EVALUATE
            yield iteration

            # Check termination
            if evaluation.decision in (
                EvaluationDecision.COMPLETE,
                EvaluationDecision.FAIL,
            ):
                break

    async def _plan(
        self,
        perception: Perception,
        state: Dict[str, Any],
    ) -> Any:
        """Generate execution plan.

        Uses existing PlanningCoordinator if available.
        """
        # Check if orchestrator has planning capabilities
        if hasattr(self.orchestrator, "planning_coordinator"):
            # Use existing PlanningCoordinator.chat_with_planning()
            planning_coordinator = self.orchestrator.planning_coordinator
            result = await planning_coordinator.chat_with_planning(
                state.get("query", ""),
            )
            return result

        # Fallback: Use orchestrator directly
        if hasattr(self.orchestrator, "plan"):
            return await self.orchestrator.plan(
                state.get("query", ""),
                context=state,
            )

        # Minimal fallback: return perception as plan
        return {"perception": perception.to_dict()}

    async def _act(
        self,
        plan: Any,
        state: Dict[str, Any],
    ) -> Any:
        """Execute a single turn: one LLM call + tool execution.

        When execution_coordinator is available, calls execute_turn_with_tools()
        for true single-turn execution (no nested loops). Falls back to
        orchestrator methods when coordinator is not available.

        Returns:
            TurnResult when using execution_coordinator, or Any from fallbacks.
        """
        query = state.get("query", "")

        # PRIMARY: Use execution_coordinator for single-turn (no nested loop)
        if self.execution_coordinator is not None:
            from victor.agent.coordinators.execution_coordinator import TurnResult

            task_classification = state.get("_task_classification")
            is_qa = state.get("_is_qa_task", False)

            turn_result = await self.execution_coordinator.execute_turn_with_tools(
                user_message=query,
                task_classification=task_classification,
                is_qa_task=is_qa,
            )

            # Update turn-level tracking (absorbed from ExecutionCoordinator loop)
            if turn_result.has_tool_calls:
                self._consecutive_no_tool_turns = 0
                self._total_tool_calls += turn_result.tool_calls_count
                if turn_result.all_tools_blocked:
                    self._consecutive_all_blocked += 1
                else:
                    self._consecutive_all_blocked = 0
            else:
                self._consecutive_no_tool_turns += 1

            return turn_result

        # FALLBACK: Use orchestrator methods (legacy path, nested loops)
        if hasattr(self.orchestrator, "chat"):
            return await self.orchestrator.chat(query)

        if hasattr(self.orchestrator, "execute"):
            return await self.orchestrator.execute(plan, state)

        if hasattr(self.orchestrator, "run"):
            return await self.orchestrator.run(query)

        return {"plan_executed": True, "plan": plan}

    async def _evaluate(
        self,
        perception: Perception,
        action_result: Any,
        state: Dict[str, Any],
    ) -> EvaluationResult:
        """Evaluate progress and decide next action.

        When action_result is a TurnResult (single-turn mode), uses
        turn-level signals (tool calls, spin detection, Q&A shortcut)
        in addition to fulfillment checks. This replaces the nudge/spin
        logic that was previously inside ExecutionCoordinator's while-loop.
        """
        # Check for TurnResult-specific signals (single-turn mode)
        from victor.agent.coordinators.execution_coordinator import TurnResult

        if isinstance(action_result, TurnResult):
            turn: TurnResult = action_result

            # Q&A shortcut: model answered without tools on a question task
            if turn.is_qa_response and turn.has_content:
                return EvaluationResult(
                    decision=EvaluationDecision.COMPLETE,
                    score=0.9,
                    reason="Q&A shortcut: accepted direct answer",
                )

            # Spin detection: all tool calls blocked by dedup
            if self._consecutive_all_blocked >= self._MAX_ALL_BLOCKED:
                return EvaluationResult(
                    decision=EvaluationDecision.FAIL,
                    score=0.1,
                    reason=(
                        f"Spin detected: {self._consecutive_all_blocked} "
                        "consecutive fully-blocked tool batches"
                    ),
                )

            # Agent stuck: no tool calls for too many turns
            if self._consecutive_no_tool_turns >= self._MAX_NO_TOOL_TURNS:
                return EvaluationResult(
                    decision=EvaluationDecision.FAIL,
                    score=0.2,
                    reason=(
                        f"Agent stuck: {self._consecutive_no_tool_turns} "
                        "turns without tool calls"
                    ),
                )

            # Model used tools and produced content = normal progress
            if turn.has_tool_calls and not turn.all_tools_blocked:
                # Check fulfillment if enabled
                if self.fulfillment:
                    try:
                        task_type = self._map_to_task_type(perception)
                        fulfillment = await self.fulfillment.check_fulfillment(
                            task_type=task_type,
                            criteria=state.get("criteria", {}),
                            context=state,
                        )
                        if fulfillment.is_fulfilled:
                            return EvaluationResult(
                                decision=EvaluationDecision.COMPLETE,
                                score=fulfillment.score,
                                reason=f"Fulfilled: {fulfillment.reason}",
                            )
                        return EvaluationResult(
                            decision=EvaluationDecision.CONTINUE,
                            score=fulfillment.score,
                            reason=f"Progress: {fulfillment.reason}",
                        )
                    except Exception as e:
                        logger.warning(f"Fulfillment check failed: {e}")

                # No fulfillment check — tools ran, so continue
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=0.5 + min(turn.successful_tool_count * 0.1, 0.3),
                    reason=f"Tools executed: {turn.successful_tool_count} ok, "
                    f"{turn.failed_tool_count} failed",
                )

            # No tools + has content + previously used tools = likely done
            if not turn.has_tool_calls and turn.has_content and self._total_tool_calls > 0:
                return EvaluationResult(
                    decision=EvaluationDecision.COMPLETE,
                    score=0.8,
                    reason="Model provided final answer after using tools",
                )

            # No tools, no previous tools = needs nudge (continue to retry)
            if not turn.has_tool_calls and self._consecutive_no_tool_turns < 2:
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=0.3,
                    reason="No tools used yet — giving another chance",
                )

            # Nudge territory: inject tool-use prompt
            if self._consecutive_no_tool_turns >= 2:
                return EvaluationResult(
                    decision=EvaluationDecision.RETRY,
                    score=0.2,
                    reason="Nudge: model not using tools, retrying with prompt",
                )

        # FALLBACK: Legacy path (orchestrator.chat() returned non-TurnResult)
        if self.fulfillment and hasattr(perception, "task_type"):
            try:
                task_type = self._map_to_task_type(perception)
                fulfillment = await self.fulfillment.check_fulfillment(
                    task_type=task_type,
                    criteria=state.get("criteria", {}),
                    context=state,
                )
                if fulfillment.is_fulfilled:
                    return EvaluationResult(
                        decision=EvaluationDecision.COMPLETE,
                        score=fulfillment.score,
                        reason=f"Fulfilled: {fulfillment.reason}",
                    )
                elif fulfillment.is_partial:
                    return EvaluationResult(
                        decision=EvaluationDecision.CONTINUE,
                        score=fulfillment.score,
                        reason=f"Partial: {fulfillment.reason}",
                    )
                else:
                    return EvaluationResult(
                        decision=EvaluationDecision.RETRY,
                        score=fulfillment.score,
                        reason=f"Not fulfilled: {fulfillment.reason}",
                    )
            except Exception as e:
                logger.warning(f"Fulfillment check failed: {e}")

        # Confidence-based fallback
        if perception.confidence >= 0.8:
            return EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=perception.confidence,
                reason="High confidence in perception",
            )
        elif perception.confidence >= 0.5:
            return EvaluationResult(
                decision=EvaluationDecision.CONTINUE,
                score=perception.confidence,
                reason="Medium confidence - continue",
            )
        else:
            return EvaluationResult(
                decision=EvaluationDecision.RETRY,
                score=perception.confidence,
                reason="Low confidence - retry",
            )

    async def _check_fulfillment(
        self,
        perception: Perception,
        state: Dict[str, Any],
    ) -> Any:
        """Check task fulfillment.

        Uses FulfillmentDetector.
        """
        if not self.fulfillment:
            return None

        task_type = self._map_to_task_type(perception)
        return await self.fulfillment.check_fulfillment(
            task_type=task_type,
            criteria=state.get("criteria", {}),
            context=state,
        )

    def _check_adaptive_termination(
        self,
        iteration: int,
        evaluation: EvaluationResult,
    ) -> Optional[str]:
        """Check if loop should adaptively terminate or extend.

        Inspired by Fast-Slow Architecture (arXiv:2604.01681):
        - "Fast" exit on detected plateau (no progress)
        - "Slow" extension when near completion

        Returns:
            "plateau" if should exit early,
            "extend" if should grant more iterations,
            None otherwise
        """
        scores = self._progress_scores

        # Need enough history for plateau detection
        if len(scores) >= self.plateau_window:
            recent = scores[-self.plateau_window :]
            improvement = max(recent) - min(recent)
            if improvement < self.plateau_tolerance:
                return "plateau"

        # Check near-completion: score > 0.7 and still improving
        if len(scores) >= 2 and scores[-1] >= 0.7:
            delta = scores[-1] - scores[-2]
            if delta > 0:
                return "extend"

        return None

    def _map_to_task_type(self, perception: Perception) -> TaskType:
        """Map perception to TaskType for fulfillment detection.

        Uses both intent and TaskAnalysis task_type for richer mapping.
        """
        from victor.agent.action_authorizer import ActionIntent

        # First try TaskAnalysis task_type (more specific)
        if perception.task_analysis:
            task_type_str = getattr(perception.task_analysis, "task_type", None)
            if task_type_str:
                # Direct mapping from unified classifier
                analysis_map = {
                    "code_generation": TaskType.CODE_GENERATION,
                    "code_modification": TaskType.CODE_MODIFICATION,
                    "debugging": TaskType.DEBUGGING,
                    "testing": TaskType.TESTING,
                    "analysis": TaskType.ANALYSIS,
                    "documentation": TaskType.DOCUMENTATION,
                    "search": TaskType.SEARCH,
                    "setup": TaskType.SETUP,
                    "deployment": TaskType.DEPLOYMENT,
                }
                mapped = analysis_map.get(task_type_str)
                if mapped:
                    return mapped

        # Fallback: map from intent
        intent_to_type = {
            ActionIntent.WRITE_ALLOWED: TaskType.CODE_GENERATION,
            ActionIntent.DISPLAY_ONLY: TaskType.SEARCH,
            ActionIntent.READ_ONLY: TaskType.ANALYSIS,
            ActionIntent.AMBIGUOUS: TaskType.UNKNOWN,
        }
        return intent_to_type.get(perception.intent, TaskType.UNKNOWN)

    def _determine_success(self, iterations: List[LoopIteration]) -> bool:
        """Determine if loop completed successfully."""
        if not iterations:
            return False

        last_iteration = iterations[-1]

        # Check last evaluation
        if last_iteration.evaluation:
            return last_iteration.evaluation.decision == EvaluationDecision.COMPLETE

        # Check fulfillment
        if last_iteration.fulfillment:
            return (
                hasattr(last_iteration.fulfillment, "is_fulfilled")
                and last_iteration.fulfillment.is_fulfilled
            )

        return False
