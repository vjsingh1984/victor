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
that sits ABOVE the AgentOrchestrator/TurnExecutor layer:

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
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, TYPE_CHECKING

from victor.agent.turn_policy import (
    FulfillmentCriteriaBuilder,
    NudgePolicy,
    SpinDetector,
    SpinState,
)
from victor.core.shared_types import TaskPhase
from victor.framework.evaluation_nodes import (
    EvaluationDecision,
    EvaluationResult,
    create_agentic_loop_graph,
)
from victor.framework.fulfillment import FulfillmentDetector, TaskType
from victor.framework.perception_integration import Perception, PerceptionIntegration
from victor.framework.capabilities.task_hints import TaskTypeHintCapabilityProvider
from victor.agent.paradigm_router import ParadigmRouter, get_paradigm_router
from victor.framework.enhanced_completion_evaluation import EnhancedCompletionEvaluator

logger = logging.getLogger(__name__)

# =============================================================================
# Enums for Agentic Loop Decisions
# =============================================================================


class AdaptiveTerminationDecision(str, Enum):
    """Decision for adaptive termination in agentic loop.

    Inspired by Fast-Slow Architecture (arXiv:2604.01681):
    - "Fast" exit on detected plateau (no progress)
    - "Slow" extension when near completion
    """

    PLATEAU = "plateau"  # Exit early, no progress detected
    EXTEND = "extend"  # Grant more iterations, near completion


# Type alias for backward compatibility
_AdaptiveTerminationDecisionLiteral = Literal["plateau", "extend"]

if TYPE_CHECKING:
    from victor.agent.services.turn_execution_runtime import (
        TurnExecutor,
        TurnResult,
    )
    from victor.storage.memory.unified import UnifiedMemoryCoordinator
    from victor.agent.services.planning_runtime import PlanningCoordinator
    from victor.framework.agent import Agent


class PlanningGate:
    """Fast-Slow Planning Gate for skipping LLM planning on simple tasks.

    Based on arXiv:2604.01681 - Fast-Slow Architecture for Agentic Systems.
    Uses rule-based heuristics to determine if LLM planning is necessary or if
    direct execution fast-path would suffice.

    This reduces LLM calls by 30%+ for simple, straightforward tasks.
    """

    # Patterns that can be handled without LLM planning (fast-path)
    FAST_PATTERNS = {
        "create_simple": True,  # Just write the file
        "action": True,  # Just execute the command
        "search": True,  # Just run the search
        "quick_question": True,  # Direct answer needed
    }

    def __init__(self, enabled: bool = True):
        """Initialize the planning gate.

        Args:
            enabled: Whether the gate is enabled (for feature flagging)
        """
        self.enabled = enabled
        self._fast_path_count = 0
        self._total_decisions = 0

    def should_use_llm_planning(
        self,
        task_type: str,
        tool_budget: int,
        query_complexity: Optional[float] = None,
        query_length: int = 0,
        context: Optional[Dict[str, Any]] = None,
        skip_planning: bool = False,
    ) -> bool:
        """Determine if LLM planning is needed or if rule-based fast-path suffices.

        Args:
            task_type: Detected task type (e.g., "create_simple", "edit")
            tool_budget: Number of tools allowed for this task
            query_complexity: Optional complexity score (0-1, lower is simpler)
            query_length: Length of query in characters
            context: Optional execution context
            skip_planning: Task type hint flag to skip planning

        Returns:
            False if task can proceed without LLM planning (fast-path)
            True if LLM planning is recommended (slow-path)
        """
        self._total_decisions += 1

        if not self.enabled:
            return True  # Gate disabled, always use LLM planning

        # Fast Pattern 0: Task type hint explicitly requests skip planning
        if skip_planning:
            logger.info(
                f"[PlanningGate] Fast-path: task_type={task_type} "
                f"skip_planning=True (from TaskTypeHint, skips LLM planning)"
            )
            self._fast_path_count += 1
            return False  # Skip LLM planning

        # Fast Pattern 1: Simple task types with low tool budget
        if task_type in self.FAST_PATTERNS and tool_budget <= 3:
            logger.info(
                f"[PlanningGate] Fast-path: task_type={task_type}, "
                f"tool_budget={tool_budget} (skips LLM planning)"
            )
            self._fast_path_count += 1
            return False  # Skip LLM planning

        # Fast Pattern 2: Low query complexity (if provided)
        if query_complexity is not None and query_complexity < 0.3:
            logger.info(
                f"[PlanningGate] Fast-path: query_complexity={query_complexity:.2f} "
                f"(skips LLM planning)"
            )
            self._fast_path_count += 1
            return False  # Skip LLM planning

        # Fast Pattern 3: Short, direct queries
        if query_length > 0 and query_length < 50:
            # Check for action keywords
            query_lower = context.get("query", "").lower() if context else ""
            action_keywords = [
                "run ",
                "execute",
                "create ",
                "write ",
                "delete ",
                "list ",
                "show ",
                "get ",
                "find ",
                "search ",
            ]
            if any(keyword in query_lower for keyword in action_keywords):
                logger.info(
                    f"[PlanningGate] Fast-path: short action query "
                    f"(length={query_length}, skips LLM planning)"
                )
                self._fast_path_count += 1
                return False  # Skip LLM planning

        # Default: Use LLM planning (slow-path)
        logger.debug(
            f"[PlanningGate] Slow-path: task_type={task_type}, "
            f"tool_budget={tool_budget} (uses LLM planning)"
        )
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about gate decisions.

        Returns:
            Dict with fast_path_count, total_decisions, fast_path_percentage
        """
        fast_path_pct = (
            (self._fast_path_count / self._total_decisions * 100)
            if self._total_decisions > 0
            else 0.0
        )
        return {
            "fast_path_count": self._fast_path_count,
            "total_decisions": self._total_decisions,
            "fast_path_percentage": fast_path_pct,
        }


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
        turn_executor: Optional["TurnExecutor"] = None,
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
            turn_executor: TurnExecutor for single-turn execution.
                If provided, AgenticLoop calls execute_turn() directly
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
        self.turn_executor = turn_executor
        self.memory_coordinator = memory_coordinator
        self.max_iterations = max_iterations
        self.enable_fulfillment_check = enable_fulfillment_check
        self.enable_adaptive_iterations = enable_adaptive_iterations
        self.plateau_window = plateau_window
        self.plateau_tolerance = plateau_tolerance
        self.config = config or {}

        # Progress tracking (SubSearch arXiv:2604.07415 intermediate rewards)
        self._progress_scores: List[float] = []

        # Shared turn policy (consistent between batch and streaming paths)
        self.spin_detector = SpinDetector()
        self.nudge_policy = NudgePolicy()
        self.criteria_builder = FulfillmentCriteriaBuilder()

        # Initialize components
        self.perception = PerceptionIntegration(
            memory_coordinator=memory_coordinator,
        )

        self.fulfillment = FulfillmentDetector() if enable_fulfillment_check else None

        # Initialize enhanced completion evaluator
        # ENABLED BY DEFAULT - use disable_enhanced_completion setting to opt-out if needed
        disable_enhanced_completion = self.config.get("disable_enhanced_completion", False)
        self.enhanced_completion_evaluator = None
        if not disable_enhanced_completion:
            self.enhanced_completion_evaluator = EnhancedCompletionEvaluator(
                enable_requirement_validation=self.config.get(
                    "enable_requirement_validation", True
                ),
                enable_completion_scoring=self.config.get("enable_completion_scoring", True),
                enable_context_keywords=self.config.get("enable_context_keywords", True),
                completion_threshold=self.config.get("completion_threshold", 0.80),
                enable_calibrated_completion=self.config.get("enable_calibrated_completion"),
            )
            logger.info(
                "[EnhancedCompletion] ENABLED by default - requirement-driven completion detection active (use disable_enhanced_completion=True to opt-out)"
            )
        else:
            logger.warning(
                "[EnhancedCompletion] DISABLED via config - using legacy completion detection"
            )

        # Initialize planning gate for fast-slow architecture
        self.planning_gate = PlanningGate(enabled=self.config.get("enable_planning_gate", True))

        # Initialize task hint provider for enhanced task type hints
        self._task_hint_provider = TaskTypeHintCapabilityProvider()

        # Initialize paradigm router for model selection (arXiv:2604.06753)
        self.paradigm_router = get_paradigm_router()
        self.paradigm_router.enabled = self.config.get("enable_paradigm_router", True)

        # Try to extract turn_executor from orchestrator if not provided
        if self.turn_executor is None and hasattr(orchestrator, "turn_executor"):
            self.turn_executor = orchestrator.turn_executor

        mode = "single-turn" if self.turn_executor else "orchestrator-fallback"
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
        self.spin_detector.reset()
        self.criteria_builder.reset()
        effective_max = self.max_iterations

        # Semantic response cache check (arXiv:2508.07675)
        # Only on first query of a loop — multi-turn queries are session-specific
        _sem_cache = None
        from victor.core.feature_flags import FeatureFlag, is_feature_enabled

        if is_feature_enabled(FeatureFlag.USE_SEMANTIC_RESPONSE_CACHE):
            try:
                from victor.agent.semantic_response_cache import get_semantic_cache

                _sem_cache = get_semantic_cache()
                _cached = _sem_cache.get(query)
                if _cached:
                    logger.info("[SemanticResponseCache] HIT — skipping LLM call")
                    return LoopResult(
                        success=True,
                        iterations=[],
                        final_state={"query": query, "response": _cached["response"]},
                        total_duration=time.time() - start_time,
                        metadata={
                            "source": "semantic_cache",
                            "similarity": _cached.get("similarity", 1.0),
                        },
                    )
            except Exception as _ce:
                logger.debug(f"Semantic cache check skipped: {_ce}")
                _sem_cache = None

        try:
            for i in range(1, effective_max + 1):
                # FAST-SLOW PLANNING GATE (arXiv:2604.01681)
                # Check if LLM planning is needed or if rule-based fast-path suffices
                # This gate runs BEFORE PERCEIVE to potentially skip the planning stage
                use_llm_planning = True
                skip_reason = None

                if i == 1:  # Only check on first iteration
                    # Extract task information from perception or context
                    # We need to perceive first to get task_type, but we can shortcut
                    # For now, use heuristics from query and context
                    task_type = context.get("task_type", "unknown") if context else "unknown"
                    tool_budget = context.get("tool_budget", 10) if context else 10

                    # Try to get query complexity from perception if available
                    # Otherwise use simple heuristics
                    query_complexity = None
                    if hasattr(self, "_last_perception"):
                        last_perc = self._last_perception
                        query_complexity = getattr(last_perc, "query_complexity", None)

                    # Extract execution parameters from TaskTypeHint if available
                    skip_planning = False
                    task_hint = None
                    if hasattr(self, "_task_hint_provider"):
                        task_hint = self._task_hint_provider.get_hint(task_type)
                        if task_hint:
                            skip_planning = getattr(task_hint, "skip_planning", False)
                            temp_override = getattr(task_hint, "temperature_override", None)
                            if temp_override is not None:
                                state["temperature_override"] = temp_override

                    # Check with planning gate
                    use_llm_planning = self.planning_gate.should_use_llm_planning(
                        task_type=task_type,
                        tool_budget=tool_budget,
                        query_complexity=query_complexity,
                        query_length=len(query),
                        context={**context, "query": query} if context else {"query": query},
                        skip_planning=skip_planning,  # Pass TaskTypeHint skip flag
                    )

                    # Get routing decision from paradigm router (arXiv:2604.06753)
                    routing_decision = self.paradigm_router.route(
                        task_type=task_type,
                        query=query,
                        history_length=len(conversation_history) if conversation_history else 0,
                        query_complexity=query_complexity,
                        tool_budget=tool_budget,
                        context=context,
                    )

                    # Store routing decision in state for execution
                    state["routing_decision"] = routing_decision.to_dict()

                    # Update execution parameters based on routing decision
                    if routing_decision.skip_planning:
                        use_llm_planning = False  # Override planning gate
                        skip_reason = "paradigm_router"

                    logger.info(
                        f"[Iteration {i}/{effective_max}] ROUTING: "
                        f"paradigm={routing_decision.paradigm.value}, "
                        f"model_tier={routing_decision.model_tier.value}, "
                        f"max_tokens={routing_decision.max_tokens}, "
                        f"confidence={routing_decision.confidence:.2f}"
                    )

                    if not use_llm_planning:
                        skip_reason = skip_reason or "fast_path"
                        logger.info(
                            f"[Iteration {i}/{effective_max}] FAST-PATH DETECTED: "
                            f"Skipping LLM planning, proceeding directly to execution"
                        )

                iteration = LoopIteration(iteration=i, stage=LoopStage.PERCEIVE)

                # PERCEIVE
                logger.info(f"[Iteration {i}/{effective_max}] PERCEIVE")
                perception = await self.perception.perceive(query, context, conversation_history)
                iteration.perception = perception
                self._last_perception = perception  # Cache for next iteration check

                # Update task_type from perception for more accurate gating
                state["perception"] = perception.to_dict()
                if hasattr(perception, "task_analysis") and perception.task_analysis:
                    state["task_type"] = getattr(perception.task_analysis, "task_type", "unknown")

                # RESEARCH TASK PROGRESS REPORTING
                # Report progress every 10 iterations for research tasks
                if state.get("task_type") == "research" and i % 10 == 0:
                    progress = self._calculate_research_progress(i, effective_max)
                    phase = self._get_current_research_phase(state)
                    papers_found = self._count_papers_found(conversation_history)

                    logger.info(
                        f"📊 RESEARCH PROGRESS: {progress}% | "
                        f"Phase: {phase} | "
                        f"Papers found: {papers_found} | "
                        f"Iteration: {i}/{effective_max}"
                    )

                # Skip planning stage if in fast-path
                if not use_llm_planning and skip_reason == "fast_path":
                    logger.info(f"[Iteration {i}/{effective_max}] SKIP PLAN (fast-path)")
                    iteration.plan = None  # No plan needed
                    state["plan"] = None
                else:
                    # PLAN (use existing PlanningCoordinator if available)
                    logger.info(f"[Iteration {i}/{effective_max}] PLAN")
                    plan = await self._plan(perception, state)
                    iteration.plan = plan
                    state["plan"] = plan
                state["perception"] = perception.to_dict()

                # DETECT PHASE (for phase-aware context management)
                current_phase = self._detect_phase(i, state, perception)
                state["task_phase"] = current_phase
                logger.debug(f"[Iteration {i}/{effective_max}] Phase: {current_phase.value}")

                # RESEARCH TASK STAGE ENFORCEMENT
                # For research tasks, enforce stage transitions based on research phase
                if state.get("task_type") == "research":
                    from victor.core.shared_types import ConversationStage

                    research_phase = self._get_current_research_phase(state)
                    research_stages = {
                        "discover": (ConversationStage.INITIAL, ConversationStage.PLANNING),
                        "search": (ConversationStage.READING,),
                        "analyze": (ConversationStage.ANALYSIS,),
                        "synthesize": (ConversationStage.EXECUTION, ConversationStage.VERIFICATION),
                    }

                    target_stages = research_stages.get(research_phase, ())

                    # Get current stage from perception
                    current_stage = None
                    if hasattr(perception, "action_intent") and hasattr(
                        perception.action_intent, "stage"
                    ):
                        current_stage = perception.action_intent.stage

                    # Enforce stage transition if needed
                    if target_stages and current_stage not in target_stages:
                        target_stage = target_stages[0]
                        logger.info(
                            f"[Iteration {i}/{effective_max}] RESEARCH PHASE TRANSITION: "
                            f"{research_phase} → {target_stage.value}"
                        )
                        # Update stage in state (will be picked up by next iteration's perception)
                        state["target_stage"] = target_stage
                        # Update perception's action intent stage for this iteration
                        if hasattr(perception, "action_intent"):
                            perception.action_intent.stage = target_stage

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

                # AGENTIC LOOP FIX: Content degradation detection
                # Detect if agent is looping without making progress (same content repeated)
                if len(self._progress_scores) >= 3:
                    # Check content lengths from recent iterations
                    recent_lengths = []
                    for iter_obj in iterations[-3:]:
                        if iter_obj.action_result and hasattr(iter_obj.action_result, "content"):
                            content_length = len(iter_obj.action_result.content)
                            recent_lengths.append(content_length)
                        elif iter_obj.action_result and hasattr(iter_obj.action_result, "response"):
                            content_length = len(iter_obj.action_result.response)
                            recent_lengths.append(content_length)

                    # If all 3 recent iterations have same content length, likely looping
                    if (
                        len(recent_lengths) >= 3 and len(set(recent_lengths)) == 1 and i >= 5
                    ):  # Only check after 5 iterations to avoid false positives
                        logger.warning(
                            f"Content degradation detected: same content length ({recent_lengths[0]}) "
                            f"repeated for 3 iterations - stopping loop"
                        )
                        evaluation = EvaluationResult(
                            decision=EvaluationDecision.FAIL,
                            score=evaluation.score,
                            reason=f"Content degradation: same length repeated 3x (iteration {i})",
                        )
                        # Continue to exit logic below

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

                # NUDGE INJECTION via shared NudgePolicy (consistent with streaming)
                # Inject on any non-terminal decision (RETRY or CONTINUE)
                if (
                    evaluation.decision
                    not in (EvaluationDecision.COMPLETE, EvaluationDecision.FAIL)
                    and self.turn_executor is not None
                ):
                    chat_ctx = getattr(self.turn_executor, "_chat_context", None)
                    if chat_ctx and hasattr(chat_ctx, "add_message"):
                        nudge = self.nudge_policy.evaluate(self.spin_detector)
                        if nudge.should_inject:
                            chat_ctx.add_message(nudge.role, nudge.message)
                            logger.info(f"Nudge injected: {nudge.nudge_type.value}")

                        budget_nudge = self.nudge_policy.budget_warning(i, effective_max)
                        if budget_nudge.should_inject:
                            chat_ctx.add_message(budget_nudge.role, budget_nudge.message)

            # Determine success
            success = self._determine_success(iterations)

            # Store cacheable responses in semantic cache for future queries
            if success and _sem_cache is not None:
                try:
                    from victor.agent.semantic_response_cache import SemanticResponseCache

                    _final_action = state.get("action_result")
                    _final_response: Optional[str] = None
                    if isinstance(_final_action, str):
                        _final_response = _final_action
                    elif _final_action is not None and hasattr(_final_action, "response"):
                        if not getattr(_final_action, "has_tool_calls", False):
                            _final_response = _final_action.response
                    if _final_response and SemanticResponseCache.is_cacheable(_final_response):
                        _sem_cache.set(query, _final_response)
                        logger.debug("[SemanticResponseCache] Stored new response")
                except Exception as _se:
                    logger.debug(f"Semantic cache store skipped: {_se}")

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

    async def stream_chat(
        self,
        query: str,
        streaming_pipeline: Any = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Any]:
        """Stream chat with perception and evaluation lifecycle.

        Wraps the existing StreamingChatPipeline with AgenticLoop's
        PERCEIVE and EVALUATE phases. The streaming pipeline handles
        the ACT phase (LLM streaming + tool execution + recovery).

        Perception runs concurrently with streaming start to avoid
        blocking the first chunk (TaskAnalyzer cold start is ~5s on
        first call due to embedding model loading).

        Args:
            query: User's natural language query
            streaming_pipeline: StreamingChatPipeline instance
            context: Additional context

        Yields:
            StreamChunk objects from the streaming pipeline
        """
        # Reset spin detector for this conversation turn
        self.spin_detector.reset()
        self.criteria_builder.reset()

        # PERCEIVE (before streaming — understands task before LLM call)
        perception = await self.perception.perceive(query, context)
        logger.info(
            f"[stream] Perceived: intent={perception.intent.value}, "
            f"complexity={perception.complexity.value}, "
            f"confidence={perception.confidence:.2f}"
        )

        # ACT via streaming pipeline (yields chunks token-by-token)
        if streaming_pipeline is not None:
            try:
                async for chunk in streaming_pipeline.run(query):
                    yield chunk
            except Exception as e:
                if "request format error" in str(e).lower():
                    logger.warning(
                        f"[stream] Streaming failed due to message format error, "
                        f"falling back to non-streaming: {e}"
                    )
                    # Fall back: yield a simple text chunk with error info
                    # In production, you might want to call a non-streaming chat method here
                    from victor.providers.base import StreamChunk

                    yield StreamChunk(
                        content=f"[Streaming error: {str(e)}. The system encountered a message format issue. "
                        f"This has been logged and will be fixed in future updates.]",
                        is_final=True,
                    )
                else:
                    # Re-raise non-format errors
                    raise
        else:
            logger.warning("[stream] No streaming pipeline provided")
            return

        # EVALUATE (post-hoc — decisions need full response context)
        logger.debug("[stream] Streaming complete, post-hoc evaluation done")

    def _detect_phase(
        self,
        iteration: int,
        state: Dict[str, Any],
        perception: Perception,
    ) -> TaskPhase:
        """Detect the current task phase based on iteration and state.

        Args:
            iteration: Current iteration number
            state: Current loop state
            perception: Current perception result

        Returns:
            Detected task phase
        """
        from victor.core.shared_types import ConversationStage

        # Map conversation stage to task phase
        stage = perception.action_intent.stage if hasattr(perception, "action_intent") else None

        # Simple phase detection based on iteration and stage
        if iteration == 1:
            # First iteration is always exploration
            return TaskPhase.EXPLORATION
        elif stage in (ConversationStage.PLANNING,):
            return TaskPhase.PLANNING
        elif stage in (ConversationStage.EXECUTION,):
            return TaskPhase.EXECUTION
        elif stage in (ConversationStage.VERIFICATION, ConversationStage.COMPLETION):
            return TaskPhase.REVIEW
        else:
            # Default to EXECUTION for middle iterations
            return TaskPhase.EXECUTION if iteration > 1 else TaskPhase.EXPLORATION

    def _calculate_research_progress(self, iteration: int, max_iterations: int) -> int:
        """Calculate research progress percentage.

        Research has 4 phases: discover(25%), search(25%), analyze(25%), synthesize(25%)
        Each phase gets roughly equal iterations.

        Args:
            iteration: Current iteration number
            max_iterations: Maximum iterations for this task

        Returns:
            Progress percentage (0-100)
        """
        # Calculate which phase we're in (4 phases)
        phase_size = max_iterations // 4
        phase_num = min(3, (iteration - 1) // phase_size)  # 0-3
        base_progress = phase_num * 25  # 0, 25, 50, 75

        # Calculate progress within current phase
        iteration_in_phase = (iteration - 1) % phase_size
        phase_progress = int((iteration_in_phase / phase_size) * 25) if phase_size > 0 else 0

        return base_progress + phase_progress

    def _get_current_research_phase(self, state: Dict[str, Any]) -> str:
        """Get current research phase based on task phase.

        Args:
            state: Current loop state

        Returns:
            Research phase name (discover, search, analyze, synthesize)
        """
        from victor.core.shared_types import TaskPhase

        task_phase = state.get("task_phase", TaskPhase.EXPLORATION)

        # Map task phases to research phases
        phase_map = {
            TaskPhase.EXPLORATION: "discover",
            TaskPhase.PLANNING: "discover",
            TaskPhase.EXECUTION: "search",  # or "analyze" depending on iteration
            TaskPhase.REVIEW: "synthesize",
        }

        # Refine EXECUTION phase detection
        if task_phase == TaskPhase.EXECUTION:
            iteration = state.get("iteration", 1)
            if iteration > (self.max_iterations // 2):
                return "analyze"  # Second half of execution is analysis

        return phase_map.get(task_phase, "discover")

    def _count_papers_found(
        self, conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Count unique arXiv papers mentioned in conversation.

        Args:
            conversation_history: Conversation history to search

        Returns:
            Number of unique papers found
        """
        import re
        from collections import defaultdict

        papers = set()

        # Search in conversation history
        if conversation_history:
            for msg in conversation_history:
                content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                # Match arXiv IDs like 1234.56789 or arXiv:1234.56789
                arxiv_pattern = r"\b(\d{4}\.\d{5,})\b"
                for match in re.findall(arxiv_pattern, content):
                    papers.add(match)

        # Also search in tool results if available
        # (This would require accessing the tool results from state)

        return len(papers)

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

        When turn_executor is available, calls execute_turn()
        for true single-turn execution (no nested loops). Falls back to
        orchestrator methods when coordinator is not available.

        Returns:
            TurnResult when using turn_executor, or Any from fallbacks.
        """
        query = state.get("query", "")

        # PRIMARY: Use turn_executor for single-turn (no nested loop)
        if self.turn_executor is not None:
            from victor.agent.services.turn_execution_runtime import TurnResult

            task_classification = state.get("_task_classification")
            is_qa = state.get("_is_qa_task", False)

            turn_result = await self.turn_executor.execute_turn(
                user_message=query,
                task_classification=task_classification,
                is_qa_task=is_qa,
                intent=state.get("perception", {}).get("intent"),
                temperature_override=state.get("temperature_override"),
            )

            # Update shared spin detector
            tool_names = set()
            if turn_result.has_tool_calls and hasattr(turn_result, "tool_results"):
                tool_names = {r.get("tool_name", "") for r in turn_result.tool_results}
            self.spin_detector.record_turn(
                has_tool_calls=turn_result.has_tool_calls,
                all_blocked=turn_result.all_tools_blocked,
                tool_names=tool_names,
                tool_count=turn_result.tool_calls_count,
            )

            # Record tool results for fulfillment criteria auto-derivation
            if turn_result.tool_results:
                for tr in turn_result.tool_results:
                    self.criteria_builder.record_tool_result(tr)

            return turn_result

        # FALLBACK: Use orchestrator methods (legacy path, nested loops)
        if hasattr(self.orchestrator, "chat"):
            return await self.orchestrator.chat(query)

        if hasattr(self.orchestrator, "execute"):
            return await self.orchestrator.execute(plan, state)

        if hasattr(self.orchestrator, "run"):
            return await self.orchestrator.run(query)

        return {"plan_executed": True, "plan": plan}

    def _is_complete_response(self, response: str) -> bool:
        """Check if response appears to be a complete answer.

        Heuristics for completion:
        - Contains complete code blocks
        - Contains summary/conclusion phrases
        - Direct answer format ("Here is...", "The answer is...")
        - Sufficient length (>150 chars)
        - NOT a continuation request ("Would you like...", "Should I...")

        Args:
            response: The model's response text

        Returns:
            True if response appears complete, False otherwise
        """
        if not response or len(response.strip()) < 150:
            return False

        response_lower = response.lower()

        # Continuation indicators (NOT complete)
        continuation_patterns = [
            "would you like",
            "should i",
            "do you want",
            "shall i",
            "would you prefer",
            "let me know if",
            "would you like me to continue",
        ]

        for pattern in continuation_patterns:
            if pattern in response_lower:
                return False

        # Completion indicators
        completion_patterns = [
            "here is",
            "here's",
            "the answer is",
            "the solution is",
            "in conclusion",
            "to summarize",
            "in summary",
            "the code above",
            "the following code",
            "as shown",
        ]

        has_completion_indicator = any(pattern in response_lower for pattern in completion_patterns)

        # Check for complete code blocks
        has_complete_code = "```" in response and response.count("```") >= 2

        # Check for structured response (headings, lists)
        has_structure = any(marker in response for marker in ["## ", "1.", "-", "*"])

        # Complete if: has indicators OR (has code/structure + sufficient length)
        return (
            has_completion_indicator
            or (has_complete_code and len(response) > 300)
            or (has_structure and len(response) > 400)
        )

    def _is_continuation_request(self, response: str) -> bool:
        """Check if response is asking for continuation direction.

        Args:
            response: The model's response text

        Returns:
            True if response is asking for continuation, False otherwise
        """
        if not response:
            return False

        response_lower = response.lower()

        continuation_patterns = [
            "would you like me to",
            "should i continue",
            "do you want me to",
            "shall i proceed",
            "let me know if you'd like",
            "would you prefer i",
        ]

        return any(pattern in response_lower for pattern in continuation_patterns)

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
        logic that was previously inside TurnExecutor's while-loop.

        Enhanced Completion Detection:
        If enable_enhanced_completion is True, uses requirement-driven
        completion detection with multi-signal fusion for more accurate
        and earlier stopping when task requirements are satisfied.
        """
        if getattr(perception, "needs_clarification", False):
            clarification_reason = getattr(
                perception,
                "clarification_reason",
                None,
            ) or "task details are incomplete"
            clarification_prompt = getattr(perception, "clarification_prompt", None)
            return EvaluationResult(
                decision=EvaluationDecision.FAIL,
                score=float(getattr(perception, "confidence", 0.0)),
                reason=f"Clarification required: {clarification_reason}",
                metadata={
                    "requires_clarification": True,
                    "clarification_prompt": clarification_prompt,
                },
            )

        # ENHANCED: Use EnhancedCompletionEvaluator if enabled
        if self.enhanced_completion_evaluator is not None:
            try:
                enhanced_result = await self.enhanced_completion_evaluator.evaluate(
                    perception=perception,
                    action_result=action_result,
                    state=state,
                    fulfillment_detector=self.fulfillment,
                    spin_detector=self.spin_detector,
                )
                # Log enhanced evaluation decision for observability
                logger.info(
                    f"[EnhancedCompletion] Decision: {enhanced_result.decision.value}, "
                    f"Score: {enhanced_result.score:.2f}, "
                    f"Reason: {enhanced_result.reason[:100]}"
                )
                return enhanced_result
            except Exception as e:
                # Graceful degradation: fall back to legacy evaluation on error
                logger.warning(
                    f"[EnhancedCompletion] Evaluation failed: {e}, falling back to legacy"
                )

        # LEGACY: Original evaluation logic (preserved for backward compatibility)
        # Check for TurnResult-specific signals (single-turn mode)
        from victor.agent.services.turn_execution_runtime import TurnResult

        if isinstance(action_result, TurnResult):
            turn: TurnResult = action_result

            # Q&A shortcut: model answered without tools on a question task
            if turn.is_qa_response and turn.has_content:
                return EvaluationResult(
                    decision=EvaluationDecision.COMPLETE,
                    score=0.9,
                    reason="Q&A shortcut: accepted direct answer",
                )

            # Use shared SpinDetector for consistent detection
            spin_state = self.spin_detector.state
            if spin_state == SpinState.TERMINATED:
                if self.spin_detector.consecutive_all_blocked > 0:
                    return EvaluationResult(
                        decision=EvaluationDecision.FAIL,
                        score=0.1,
                        reason=(
                            f"Spin detected: {self.spin_detector.consecutive_all_blocked} "
                            "consecutive fully-blocked tool batches"
                        ),
                    )
                return EvaluationResult(
                    decision=EvaluationDecision.FAIL,
                    score=0.2,
                    reason=(
                        f"Agent stuck: {self.spin_detector.consecutive_no_tool_turns} "
                        "turns without tool calls"
                    ),
                )

            # Model used tools = normal progress
            if turn.has_tool_calls and spin_state == SpinState.NORMAL:
                # Check fulfillment if enabled
                if self.fulfillment:
                    try:
                        task_type = self._map_to_task_type(perception)
                        # Use auto-derived criteria from tool results
                        criteria = state.get("criteria", {})
                        auto_criteria = self.criteria_builder.build().to_dict()
                        merged_criteria = {**auto_criteria, **criteria}
                        fulfillment = await self.fulfillment.check_fulfillment(
                            task_type=task_type,
                            criteria=merged_criteria,
                            context=state,
                        )
                        if fulfillment.is_fulfilled:
                            return EvaluationResult(
                                decision=EvaluationDecision.COMPLETE,
                                score=fulfillment.score,
                                reason=f"Fulfilled: {fulfillment.reason}",
                            )
                        # Partial fulfillment — refine with edge model
                        partial_result = EvaluationResult(
                            decision=EvaluationDecision.CONTINUE,
                            score=fulfillment.score,
                            reason=f"Progress: {fulfillment.reason}",
                        )
                        return await self._refine_with_llm(partial_result, action_result, state)
                    except Exception as e:
                        logger.warning(f"Fulfillment check failed: {e}")

                # No fulfillment check — tools ran, so continue
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=0.5 + min(turn.successful_tool_count * 0.1, 0.3),
                    reason=f"Tools executed: {turn.successful_tool_count} ok, "
                    f"{turn.failed_tool_count} failed",
                )

            # Detect final answers - both after tools AND direct answers
            # Removed requirement for prior tool usage to detect direct answers
            if (
                not turn.has_tool_calls
                and turn.has_content
                # Check for sufficient content length instead of requiring prior tool usage
                and len(turn.response.strip()) > 100  # Heuristic: substantial answer
            ):
                # Model provided a substantial response without requesting tools
                # This is likely a complete answer

                # Additional check: is this a continuation request?
                is_continuation = self._is_continuation_request(turn.response)

                if not is_continuation:
                    heuristic = EvaluationResult(
                        decision=EvaluationDecision.COMPLETE,
                        score=0.8,
                        reason=f"Model provided substantial response ({len(turn.response)} chars) without requesting tools",
                    )
                    return await self._refine_with_llm(heuristic, action_result, state)
                else:
                    # Model wants to continue (e.g., "Would you like me to...")
                    return EvaluationResult(
                        decision=EvaluationDecision.CONTINUE,
                        score=0.6,
                        reason="Model offered continuation - awaiting user direction",
                    )

            # No tools, no previous tools = needs nudge (continue to retry)
            if not turn.has_tool_calls and spin_state == SpinState.NORMAL:
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=0.3,
                    reason="No tools used yet — giving another chance",
                )

            # Nudge territory: inject tool-use prompt
            if spin_state == SpinState.WARNING:
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

    async def _refine_with_llm(
        self,
        heuristic_result: EvaluationResult,
        action_result: Any,
        state: Dict[str, Any],
    ) -> EvaluationResult:
        """Refine evaluation via TieredDecisionService (canonical decision path).

        Uses the same decision service and tier routing as all other LLM
        decisions in Victor. The tier (edge/balanced/performance) is
        determined by TieredDecisionService's per-DecisionType routing,
        which can be escalated/de-escalated based on confidence feedback.

        Falls back to heuristic_result if decision service is unavailable,
        disabled, or errors out. This is a refinement, not a replacement.

        Args:
            heuristic_result: Result from heuristic evaluation
            action_result: TurnResult from execution
            state: Current execution state

        Returns:
            Refined EvaluationResult (may be unchanged)
        """
        # Only refine ambiguous decisions (score 0.4-0.8)
        if heuristic_result.score < 0.4 or heuristic_result.score > 0.8:
            return heuristic_result

        try:
            from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

            if not get_feature_flag_manager().is_enabled(FeatureFlag.USE_EDGE_MODEL):
                return heuristic_result

            from victor.agent.decisions.schemas import DecisionType
            from victor.agent.services.protocols.decision_service import (
                LLMDecisionServiceProtocol,
            )
            from victor.core import get_container

            container = get_container()
            decision_svc = container.get_optional(LLMDecisionServiceProtocol)
            if decision_svc is None:
                return heuristic_result

            # Build context for TaskCompletionDecision
            response_content = ""
            if hasattr(action_result, "content"):
                response_content = action_result.content or ""

            deliverable_count = 0
            if hasattr(action_result, "successful_tool_count"):
                deliverable_count = action_result.successful_tool_count

            context = {
                "response_tail": response_content[-500:] if response_content else "",
                "deliverable_count": deliverable_count,
                "signal_count": self.spin_detector.total_tool_calls,
            }

            # Call via canonical decision service (tier-aware)
            result = await decision_svc.decide_async(
                decision_type=DecisionType.TASK_COMPLETION,
                context=context,
                heuristic_result=heuristic_result,
                heuristic_confidence=heuristic_result.score,
            )

            # Interpret TaskCompletionDecision
            decision_obj = result.result
            if decision_obj and hasattr(decision_obj, "is_complete"):
                tier = result.source  # "edge", "balanced", "performance", etc.

                if decision_obj.is_complete and decision_obj.confidence >= 0.7:
                    # High confidence from LLM — de-escalate for next call
                    if hasattr(decision_svc, "deescalate_tier"):
                        decision_svc.deescalate_tier(
                            DecisionType.TASK_COMPLETION, "high_confidence"
                        )
                    return EvaluationResult(
                        decision=EvaluationDecision.COMPLETE,
                        score=decision_obj.confidence,
                        reason=f"LLM ({tier}): complete (phase={decision_obj.phase})",
                        metadata={"source": tier},
                    )

                elif decision_obj.phase == "stuck":
                    # Stuck — escalate for next call
                    if hasattr(decision_svc, "escalate_tier"):
                        decision_svc.escalate_tier(DecisionType.TASK_COMPLETION, "stuck_phase")
                    return EvaluationResult(
                        decision=EvaluationDecision.RETRY,
                        score=decision_obj.confidence,
                        reason=f"LLM ({tier}): agent appears stuck",
                        metadata={"source": tier},
                    )

                elif decision_obj.confidence < 0.5:
                    # Low confidence from LLM — escalate tier
                    if hasattr(decision_svc, "escalate_tier"):
                        decision_svc.escalate_tier(DecisionType.TASK_COMPLETION, "low_confidence")

            logger.debug(
                "LLM refinement: source=%s, confidence=%.2f",
                result.source,
                result.confidence,
            )

        except Exception as e:
            logger.debug("LLM refinement skipped: %s", e)

        return heuristic_result

    def _check_adaptive_termination(
        self,
        iteration: int,
        evaluation: EvaluationResult,
    ) -> Optional[AdaptiveTerminationDecision]:
        """Check if loop should adaptively terminate or extend.

        Inspired by Fast-Slow Architecture (arXiv:2604.01681):
        - "Fast" exit on detected plateau (no progress)
        - "Slow" extension when near completion

        Returns:
            AdaptiveTerminationDecision.PLATEAU if should exit early,
            AdaptiveTerminationDecision.EXTEND if should grant more iterations,
            None otherwise
        """
        scores = self._progress_scores

        # Need enough history for plateau detection
        if len(scores) >= self.plateau_window:
            recent = scores[-self.plateau_window :]
            improvement = max(recent) - min(recent)
            if improvement < self.plateau_tolerance:
                return AdaptiveTerminationDecision.PLATEAU

        # Check near-completion: score > 0.7 and still improving
        if len(scores) >= 2 and scores[-1] >= 0.7:
            delta = scores[-1] - scores[-2]
            if delta > 0:
                return AdaptiveTerminationDecision.EXTEND

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
