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

"""Intelligent agent pipeline integrating all learning components.

This module provides a unified pipeline that integrates:
- IntelligentPromptBuilder: Embedding-based context selection, profile learning
- AdaptiveModeController: Q-learning for mode transitions
- ResponseQualityScorer: Multi-dimensional quality assessment
- GroundingVerifier: Hallucination detection
- ResilientExecutor: Circuit breaker, retry, rate limiting

Architecture (Facade + Chain of Responsibility patterns):

    ┌─────────────────────────────────────────────────────────┐
    │              IntelligentAgentPipeline                   │
    │  (Facade coordinating all intelligent components)       │
    └─────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
    ┌─────────┐        ┌─────────────┐      ┌──────────────┐
    │ Prompt  │        │    Mode     │      │  Response    │
    │ Builder │        │  Controller │      │  Validator   │
    └─────────┘        └─────────────┘      └──────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐        ┌─────────────┐      ┌──────────────┐
    │ Profile │        │  Q-Learning │      │   Quality    │
    │ Learner │        │    Store    │      │   Scorer     │
    └─────────┘        └─────────────┘      └──────────────┘

Usage:
    pipeline = await IntelligentAgentPipeline.create(
        provider_name="ollama",
        model="qwen2.5:32b",
        profile_name="local-qwen",
        project_root="/path/to/project",
    )

    # Pre-request: Build optimized prompt and get mode recommendation
    context = await pipeline.prepare_request(
        task="Analyze the auth module",
        task_type="analysis",
        current_mode="explore",
    )

    # Post-response: Validate, score, and learn from the result
    result = await pipeline.process_response(
        response="The auth module uses JWT...",
        tool_calls=5,
        success=True,
    )

    # Get pipeline stats
    stats = pipeline.get_stats()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from victor.providers.base import BaseProvider
    from victor.providers.types import Message
    from victor.agent.intelligent_prompt_builder import IntelligentPromptBuilder
    from victor.agent.adaptive_mode_controller import AdaptiveModeController
    from victor.agent.response_quality import ResponseQualityScorer
    from victor.agent.grounding_verifier import GroundingVerifier
    from victor.agent.resilience import ResilientExecutor

from victor.agent.output_deduplicator import OutputDeduplicator
from victor.protocols.provider_adapter import get_provider_adapter

logger = logging.getLogger(__name__)

# Providers known to have repetition issues requiring deduplication
PROVIDERS_WITH_REPETITION_ISSUES = {"xai", "grok", "x-ai"}


@dataclass
class RequestContext:
    """Context prepared for a request."""

    system_prompt: str
    recommended_tool_budget: int
    recommended_mode: str
    should_continue: bool
    continuation_context: Optional[str] = None
    mode_confidence: float = 0.5
    profile_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseResult:
    """Result of processing a response."""

    is_valid: bool
    quality_score: float
    grounding_score: float
    is_grounded: bool
    quality_details: Dict[str, float] = field(default_factory=dict)
    grounding_issues: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    learning_reward: float = 0.0
    # Grounding failure handling
    should_finalize: bool = False
    """True when grounding retries exceeded - finalize with best-effort response."""
    should_retry: bool = False
    """True when grounding failed but retries remain."""
    finalize_reason: str = ""
    """Reason for forced finalization (e.g., 'grounding failure limit exceeded')."""
    grounding_feedback: str = ""
    """Actionable feedback prompt for correcting grounding issues on retry."""


@dataclass
class PipelineStats:
    """Statistics for the intelligent pipeline."""

    total_requests: int = 0
    successful_requests: int = 0
    avg_quality_score: float = 0.0
    avg_grounding_score: float = 0.0
    total_learning_reward: float = 0.0
    mode_transitions: int = 0
    circuit_breaker_trips: int = 0
    retry_attempts: int = 0
    cache_state: str = "cold"
    profile_name: str = ""


class IntelligentAgentPipeline:
    """Unified pipeline integrating all intelligent agent components.

    This facade coordinates:
    1. Pre-request: Prompt building, mode selection, budget optimization
    2. Execution: Resilient provider calls with circuit breaker
    3. Post-response: Quality scoring, grounding verification, learning
    """

    def __init__(
        self,
        provider_name: str,
        model: str,
        profile_name: str,
        project_root: Optional[str] = None,
    ):
        """Initialize the intelligent pipeline.

        Args:
            provider_name: LLM provider name
            model: Model name/identifier
            profile_name: Profile name for learning tracking
            project_root: Project root for grounding verification
        """
        self.provider_name = provider_name
        self.model = model
        self.profile_name = profile_name
        self.project_root = project_root

        # Components (initialized lazily)
        self._prompt_builder: Optional["IntelligentPromptBuilder"] = None
        self._mode_controller: Optional["AdaptiveModeController"] = None
        self._quality_scorer: Optional["ResponseQualityScorer"] = None
        self._grounding_verifier: Optional["GroundingVerifier"] = None
        self._resilient_executor: Optional["ResilientExecutor"] = None
        self._output_deduplicator: Optional[OutputDeduplicator] = None

        # State tracking
        self._current_context: Optional[RequestContext] = None
        self._session_start = datetime.now()
        self._stats = PipelineStats(profile_name=profile_name)

        # Grounding failure handling
        self._grounding_failure_count: int = 0
        self._max_grounding_retries: int = 1  # Only 1 retry, then finalize

        # Provider adapter for capability-based behavior
        self._provider_adapter = get_provider_adapter(provider_name)

        # Provider-aware deduplication
        self._deduplication_enabled = self._provider_adapter.capabilities.output_deduplication

        # Observers for feedback
        self._observers: List[Callable[[ResponseResult], None]] = []

    @classmethod
    async def create(
        cls,
        provider_name: str,
        model: str,
        profile_name: Optional[str] = None,
        project_root: Optional[str] = None,
    ) -> "IntelligentAgentPipeline":
        """Factory method to create and initialize the pipeline.

        Args:
            provider_name: LLM provider name
            model: Model name
            profile_name: Optional profile name
            project_root: Optional project root

        Returns:
            Initialized IntelligentAgentPipeline
        """
        profile = profile_name or f"{provider_name}:{model}"
        pipeline = cls(
            provider_name=provider_name,
            model=model,
            profile_name=profile,
            project_root=project_root,
        )
        # Skip eager initialization - use lazy loading instead
        return pipeline

    async def _get_prompt_builder(self) -> Optional["IntelligentPromptBuilder"]:
        """Lazy initialize prompt builder."""
        if self._prompt_builder is None:
            try:
                import traceback

                logger.debug(
                    f"[IntelligentPipeline] Creating prompt builder with: "
                    f"provider_name={self.provider_name!r} (type={type(self.provider_name).__name__}), "
                    f"model={self.model!r} (type={type(self.model).__name__}), "
                    f"profile_name={self.profile_name!r} (type={type(self.profile_name).__name__})"
                )
                from victor.agent.intelligent_prompt_builder import IntelligentPromptBuilder

                self._prompt_builder = await IntelligentPromptBuilder.create(
                    self.provider_name, self.model, self.profile_name
                )
            except Exception as e:
                logger.warning(
                    f"[IntelligentPipeline] Prompt builder init failed: {e}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
        return self._prompt_builder

    def _get_mode_controller(self) -> Optional["AdaptiveModeController"]:
        """Lazy initialize mode controller (sync)."""
        if self._mode_controller is None:
            try:
                from victor.agent.adaptive_mode_controller import AdaptiveModeController

                # Get ModeTransitionLearner from RLCoordinator for unified RL
                mode_transition_learner = None
                try:
                    from victor.framework.rl.coordinator import get_rl_coordinator

                    coordinator = get_rl_coordinator()
                    mode_transition_learner = coordinator.get_learner("mode_transition")
                except Exception as e:
                    logger.debug(
                        f"[IntelligentPipeline] Could not get mode_transition learner: {e}"
                    )

                self._mode_controller = AdaptiveModeController(
                    profile_name=self.profile_name,
                    provider_name=self.provider_name,
                    model_name=self.model,
                    provider_adapter=self._provider_adapter,
                    mode_transition_learner=mode_transition_learner,
                )
            except Exception as e:
                logger.warning(f"[IntelligentPipeline] Mode controller init failed: {e}")
        return self._mode_controller

    def get_provider_quality_thresholds(self) -> Dict[str, Any]:
        """Get provider-specific quality thresholds.

        Returns:
            Dict with 'min_quality' and 'grounding_threshold' for current provider
        """
        # Use provider adapter's capabilities first
        if self._provider_adapter:
            caps = self._provider_adapter.capabilities
            return {
                "min_quality": caps.quality_threshold,
                "grounding_threshold": caps.grounding_strictness,
            }

        # Fall back to mode controller
        controller = self._get_mode_controller()
        if controller:
            return controller.get_quality_thresholds()

        # Default fallback
        return {"min_quality": 0.70, "grounding_threshold": 0.65}

    async def _get_quality_scorer(self) -> Optional["ResponseQualityScorer"]:
        """Lazy initialize quality scorer."""
        if self._quality_scorer is None:
            try:
                from victor.agent.response_quality import ResponseQualityScorer

                self._quality_scorer = ResponseQualityScorer()
            except Exception as e:
                logger.warning(f"[IntelligentPipeline] Quality scorer init failed: {e}")
        return self._quality_scorer

    async def _get_grounding_verifier(self) -> Optional["GroundingVerifier"]:
        """Lazy initialize grounding verifier."""
        if self._grounding_verifier is None and self.project_root:
            try:
                from victor.agent.grounding_verifier import GroundingVerifier

                # Get GroundingThresholdLearner from RLCoordinator for adaptive thresholds
                grounding_threshold_learner = None
                try:
                    from victor.framework.rl.coordinator import get_rl_coordinator

                    coordinator = get_rl_coordinator()
                    grounding_threshold_learner = coordinator.get_learner("grounding_threshold")
                except Exception as e:
                    logger.debug(
                        f"[IntelligentPipeline] Could not get grounding_threshold learner: {e}"
                    )

                self._grounding_verifier = GroundingVerifier(
                    project_root=self.project_root,
                    provider_adapter=self._provider_adapter,
                    grounding_threshold_learner=grounding_threshold_learner,
                )
            except Exception as e:
                logger.warning(f"[IntelligentPipeline] Grounding verifier init failed: {e}")
        return self._grounding_verifier

    def _emit_grounding_event(
        self,
        is_grounded: bool,
        grounding_score: float,
        task_type: str,
        grounding_result: Optional["GroundingVerificationResult"] = None,
    ) -> None:
        """Emit RL event for grounding verification with full context.

        This activates the grounding_threshold learner to learn optimal
        thresholds based on verification outcomes, and provides detailed
        context for observability and debugging.

        Args:
            is_grounded: Whether the response passed grounding verification
            grounding_score: Confidence score from grounding verifier
            task_type: Type of task being verified
            grounding_result: Full verification result with issues (for context)
        """
        try:
            from victor.framework.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return  # type: ignore[unreachable]

            # Build rich metadata with context about what failed
            metadata = {
                "is_grounded": is_grounded,
                "grounding_score": grounding_score,
            }

            # Add detailed issue information if available
            if grounding_result is not None and not is_grounded:
                metadata["issues"] = [
                    {
                        "issue_type": issue.issue_type.value,
                        "severity": issue.severity.value,
                        "description": issue.description,
                        "reference": issue.reference,
                        "suggestion": issue.suggestion,
                    }
                    for issue in grounding_result.issues
                ]
                # Add summary for quick filtering
                metadata["issue_summary"] = {
                    "total_issues": len(grounding_result.issues),
                    "issue_types": [issue.issue_type.value for issue in grounding_result.issues],
                    "has_path_issues": any(
                        issue.issue_type.value in ["path_invalid", "file_not_found", "ambiguous"]
                        for issue in grounding_result.issues
                    ),
                    "has_verification_issues": any(
                        issue.issue_type.value in ["unverifiable", "syntax_error"]
                        for issue in grounding_result.issues
                    ),
                }

            event = RLEvent(
                type=RLEventType.GROUNDING_CHECK,
                success=is_grounded,
                quality_score=grounding_score,
                provider=self.provider_name,
                model=self.model,
                task_type=task_type,
                threshold_value=grounding_score,
                metadata=metadata,
            )

            hooks.emit(event)

            # Log summary for debugging
            if not is_grounded and grounding_result is not None:
                logger.debug(
                    f"[IntelligentPipeline] Emitted grounding failure event: "
                    f"{len(grounding_result.issues)} issues, "
                    f"types: {[issue.issue_type.value for issue in grounding_result.issues]}"
                )

        except Exception as e:
            logger.debug(f"[IntelligentPipeline] Grounding event emission failed: {e}")

    async def _get_resilient_executor(self) -> Optional["ResilientExecutor"]:
        """Lazy initialize resilient executor."""
        if self._resilient_executor is None:
            try:
                from victor.agent.resilience import ResilientExecutor

                self._resilient_executor = ResilientExecutor()
            except Exception as e:
                logger.warning(f"[IntelligentPipeline] Resilient executor init failed: {e}")
        return self._resilient_executor

    def _get_output_deduplicator(self) -> OutputDeduplicator:
        """Lazy initialize output deduplicator."""
        if self._output_deduplicator is None:
            self._output_deduplicator = OutputDeduplicator(
                min_block_length=50,
                normalize_whitespace=True,
            )
        return self._output_deduplicator

    def _should_enable_deduplication(self) -> bool:
        """Check if deduplication should be enabled for current provider.

        Returns:
            True if the provider is known to have repetition issues, False otherwise
        """
        provider_lower = self.provider_name.lower()
        return provider_lower in PROVIDERS_WITH_REPETITION_ISSUES

    def deduplicate_response(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """Apply deduplication to response if enabled for provider.

        This method follows the Strategy Pattern - the deduplication strategy
        is selected based on the provider configuration.

        Args:
            response: Raw response from provider

        Returns:
            Tuple of (deduplicated_response, stats_dict)
        """
        if not self._deduplication_enabled or not response:
            return response, {"deduplication_applied": False}

        dedup = self._get_output_deduplicator()
        dedup.reset()  # Reset for fresh processing

        deduplicated = dedup.process(response)
        stats = dedup.get_stats()
        stats["deduplication_applied"] = True
        stats["provider"] = self.provider_name

        if stats.get("duplicates_removed", 0) > 0:
            logger.info(
                f"[IntelligentPipeline] Deduplication for {self.provider_name}: "
                f"removed {stats['duplicates_removed']} duplicates, "
                f"saved ~{stats['bytes_saved']} bytes"
            )

        return deduplicated, stats

    async def prepare_request(
        self,
        task: str,
        task_type: str = "general",
        current_mode: str = "explore",
        tool_calls_made: int = 0,
        tool_budget: int = 10,
        iteration_count: int = 0,
        iteration_budget: int = 20,
        quality_score: float = 0.5,
        session_id: Optional[str] = None,
        continuation_context: Optional[str] = None,
    ) -> RequestContext:
        """Prepare context for a request.

        Uses intelligent components to:
        1. Build an optimized system prompt
        2. Get mode transition recommendation
        3. Calculate optimal tool budget

        Args:
            task: Current task/query
            task_type: Detected task type
            current_mode: Current agent mode
            tool_calls_made: Tool calls already made
            tool_budget: Total tool budget
            iteration_count: Current iteration
            iteration_budget: Total iteration budget
            quality_score: Current quality score
            session_id: Session for context retrieval
            continuation_context: Context from continuation

        Returns:
            RequestContext with optimized settings
        """
        start_time = time.perf_counter()

        # Build intelligent prompt (lazy init)
        system_prompt = ""
        prompt_builder = await self._get_prompt_builder()
        if prompt_builder is not None:
            system_prompt = await prompt_builder.build(
                task=task,
                task_type=task_type,
                current_mode=current_mode,
                tool_budget=tool_budget,
                iteration_budget=iteration_budget,
                session_id=session_id,
                continuation_context=continuation_context,
            )

        # Get mode recommendation
        recommended_mode = current_mode
        mode_confidence = 0.5
        should_continue = True
        recommended_budget = tool_budget

        mode_controller = self._get_mode_controller()
        if mode_controller is not None:
            action = mode_controller.get_recommended_action(
                current_mode=current_mode,
                task_type=task_type,
                tool_calls_made=tool_calls_made,
                tool_budget=tool_budget,
                iteration_count=iteration_count,
                iteration_budget=iteration_budget,
                quality_score=quality_score,
            )
            recommended_mode = action.target_mode.value
            mode_confidence = action.confidence
            should_continue = action.should_continue
            recommended_budget = tool_budget + action.adjust_tool_budget

            # Get learned optimal budget for task type - used as GUIDANCE not hard cap
            # The RL system learns from outcomes:
            # - Success with efficiency → gradually decreases learned budget
            # - Failure/low quality → increases learned budget
            # - Budget exhaustion failures → strongly increases learned budget
            #
            # We use learned budget to potentially INCREASE the recommendation (if
            # history shows this task type needs more), but NEVER decrease below
            # the user's original tool_budget.
            learned_budget = mode_controller.get_optimal_tool_budget(task_type)

            # If learned budget suggests we need MORE than user specified, recommend
            # a slight increase (bounded). This helps tasks that historically need more.
            if learned_budget > tool_budget:
                # Recommend up to 20% more, but cap at learned budget
                suggested_increase = min(learned_budget, int(tool_budget * 1.2))
                recommended_budget = max(recommended_budget, suggested_increase)
                logger.debug(
                    f"[IntelligentPipeline] RL suggests higher budget for {task_type}: "
                    f"learned={learned_budget}, recommending={recommended_budget}"
                )

            # Floor: never go below user's original budget or minimum viable
            min_budget = max(15, tool_budget)
            recommended_budget = max(recommended_budget, min_budget)

        # Get profile stats
        profile_stats: Dict[str, Any] = {}
        prompt_builder = await self._get_prompt_builder()
        if prompt_builder is not None:
            profile_stats = prompt_builder.get_profile_stats()

        context = RequestContext(
            system_prompt=system_prompt,
            recommended_tool_budget=recommended_budget,
            recommended_mode=recommended_mode,
            should_continue=should_continue,
            continuation_context=continuation_context,
            mode_confidence=mode_confidence,
            profile_stats=profile_stats,
        )

        self._current_context = context
        self._stats.total_requests += 1

        elapsed = time.perf_counter() - start_time
        logger.debug(
            f"[IntelligentPipeline] Prepared request in {elapsed*1000:.1f}ms: "
            f"mode={recommended_mode}, budget={recommended_budget}"
        )

        return context

    async def process_response(
        self,
        response: str,
        query: str = "",
        tool_calls: int = 0,
        tool_budget: int = 10,
        success: bool = True,
        task_type: str = "general",
        context: Optional[Dict[str, Any]] = None,
    ) -> ResponseResult:
        """Process and validate a response.

        Uses intelligent components to:
        1. Score response quality
        2. Verify grounding
        3. Record feedback for learning

        Args:
            response: The model's response
            query: Original query
            tool_calls: Number of tool calls made
            tool_budget: Tool budget
            success: Whether the response was successful
            task_type: Task type
            context: Additional context for verification

        Returns:
            ResponseResult with quality and grounding info
        """
        start_time = time.perf_counter()

        # Step 0: Apply deduplication for providers with repetition issues
        # This follows the Decorator pattern - wrapping the response processing
        deduplicated_response, dedup_stats = self.deduplicate_response(response)
        if dedup_stats.get("deduplication_applied"):
            # Use deduplicated response for subsequent processing
            response = deduplicated_response

        # Score quality (lazy init)
        quality_score = 0.5
        quality_details: Dict[str, float] = {}
        improvement_suggestions: List[str] = []

        quality_scorer = await self._get_quality_scorer()
        if quality_scorer is not None and query:
            quality_result = await quality_scorer.score(
                query=query,
                response=response,
                context=context,
            )
            quality_score = quality_result.overall_score
            quality_details = {
                dim.dimension.value: dim.score for dim in quality_result.dimension_scores
            }
            improvement_suggestions = quality_result.improvement_suggestions

        # Verify grounding (lazy init)
        grounding_score = 1.0
        is_grounded = True
        grounding_issues: List[str] = []
        grounding_result = None  # Store full result for feedback generation

        grounding_verifier = await self._get_grounding_verifier()
        if grounding_verifier is not None:
            grounding_result = await grounding_verifier.verify(
                response=response,
                context=context,
            )
            grounding_score = grounding_result.confidence
            is_grounded = grounding_result.is_grounded
            grounding_issues = [
                f"{issue.issue_type.value}: {issue.description}"
                for issue in grounding_result.issues
            ]

            # Emit RL event for grounding verification with full context
            self._emit_grounding_event(
                is_grounded=is_grounded,
                grounding_score=grounding_score,
                task_type=task_type,
                grounding_result=grounding_result,  # Pass full result for context
            )

        # Track grounding failures for best-effort finalize logic
        should_finalize = False
        should_retry = False
        finalize_reason = ""
        grounding_feedback = ""

        if not is_grounded:
            self._grounding_failure_count += 1
            if self._grounding_failure_count > self._max_grounding_retries:
                # Max retries exceeded - force finalize with best-effort response
                should_finalize = True
                finalize_reason = "grounding failure limit exceeded"
                logger.warning(
                    f"[IntelligentPipeline] Grounding failure count ({self._grounding_failure_count}) "
                    f"exceeded max retries ({self._max_grounding_retries}). "
                    f"Forcing best-effort finalize."
                )
            else:
                # Can still retry - generate actionable feedback prompt
                should_retry = True
                # Generate feedback from grounding result if available
                if grounding_result is not None:
                    grounding_feedback = grounding_result.generate_feedback_prompt()
                    if grounding_feedback:
                        logger.info(
                            f"[IntelligentPipeline] Generated grounding feedback for retry: "
                            f"{len(grounding_feedback)} chars"
                        )
                logger.debug(
                    f"[IntelligentPipeline] Grounding failed, retry allowed "
                    f"({self._grounding_failure_count}/{self._max_grounding_retries})"
                )
        else:
            # Grounding succeeded - reset counter
            if self._grounding_failure_count > 0:
                logger.debug(
                    f"[IntelligentPipeline] Grounding succeeded, resetting failure count "
                    f"from {self._grounding_failure_count} to 0"
                )
            self._grounding_failure_count = 0

        # Record feedback for learning
        learning_reward = 0.0
        response_time_ms = (time.perf_counter() - start_time) * 1000

        prompt_builder = await self._get_prompt_builder()
        if prompt_builder is not None:
            prompt_builder.record_feedback(
                task_type=task_type,
                success=success,
                quality_score=quality_score,
                response_time_ms=response_time_ms,
                tool_calls=tool_calls,
                tool_budget=tool_budget,
                grounded=is_grounded,
            )

        mode_controller = self._get_mode_controller()
        if mode_controller is not None:
            learning_reward = mode_controller.record_outcome(
                success=success,
                quality_score=quality_score,
                user_satisfied=quality_score > 0.6,
                completed=success and quality_score > 0.7,
            )

        # Update stats efficiently
        if success:
            self._stats.successful_requests += 1

        # Use incremental average to avoid floating point drift
        # Guard against division by zero when process_response is called without prepare_request
        n = self._stats.total_requests
        if n == 0:
            # First response without prepare_request - initialize stats
            self._stats.total_requests = 1
            n = 1
        self._stats.avg_quality_score += (quality_score - self._stats.avg_quality_score) / n
        self._stats.avg_grounding_score += (grounding_score - self._stats.avg_grounding_score) / n
        self._stats.total_learning_reward += learning_reward

        result = ResponseResult(
            is_valid=success and is_grounded,
            quality_score=quality_score,
            grounding_score=grounding_score,
            is_grounded=is_grounded,
            quality_details=quality_details,
            grounding_issues=grounding_issues,
            improvement_suggestions=improvement_suggestions,
            learning_reward=learning_reward,
            should_finalize=should_finalize,
            should_retry=should_retry,
            finalize_reason=finalize_reason,
            grounding_feedback=grounding_feedback,
        )

        # Notify observers (skip if none)
        if self._observers:
            for observer in self._observers:
                try:
                    observer(result)
                except Exception as e:
                    logger.warning(f"[IntelligentPipeline] Observer error: {e}")

        elapsed = time.perf_counter() - start_time
        logger.debug(
            f"[IntelligentPipeline] Processed response in {elapsed*1000:.1f}ms: "
            f"quality={quality_score:.2f}, grounded={is_grounded}"
        )

        return result

    async def execute_with_resilience(
        self,
        provider: "BaseProvider",
        messages: List["Message"],
        circuit_name: Optional[str] = None,
        fallback: Optional[Callable[[], Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a provider call with resilience (circuit breaker, retry).

        Args:
            provider: The LLM provider
            messages: Messages to send
            circuit_name: Circuit breaker name (default: provider name)
            fallback: Optional fallback function
            **kwargs: Additional provider kwargs

        Returns:
            Provider response

        Raises:
            CircuitOpenError: If circuit is open and no fallback
            Exception: If retries exhausted and no fallback
        """
        resilient_executor = await self._get_resilient_executor()
        if resilient_executor is None:
            # No resilience - direct call
            return await provider.chat(messages=messages, **kwargs)

        circuit = circuit_name or f"{self.provider_name}:{self.model}"

        async def call_provider() -> Any:
            return await provider.chat(messages=messages, **kwargs)

        async def fallback_func() -> Any:
            if fallback:
                return await fallback()
            raise RuntimeError("No fallback provided")

        try:
            result = await resilient_executor.execute(
                circuit,  # name (positional)
                call_provider,  # func (positional)
                fallback=fallback_func if fallback else None,
            )
            return result
        except Exception:
            self._stats.circuit_breaker_trips += 1
            raise

    def should_continue(
        self,
        tool_calls_made: int,
        tool_budget: int,
        quality_score: float,
        iteration_count: int,
        iteration_budget: int,
    ) -> Tuple[bool, str]:
        """Determine if processing should continue.

        Args:
            tool_calls_made: Tools calls made
            tool_budget: Tool budget
            quality_score: Current quality score
            iteration_count: Iterations completed
            iteration_budget: Iteration budget

        Returns:
            Tuple of (should_continue, reason)
        """
        mode_controller = self._get_mode_controller()
        if mode_controller is not None:
            return mode_controller.should_continue(
                tool_calls_made=tool_calls_made,
                tool_budget=tool_budget,
                quality_score=quality_score,
                iteration_count=iteration_count,
                iteration_budget=iteration_budget,
            )

        # Fallback logic
        if tool_calls_made >= tool_budget:
            return False, "Tool budget exhausted"
        if iteration_count >= iteration_budget:
            return False, "Iteration budget exhausted"
        if quality_score > 0.85:
            return False, "High quality achieved"
        return True, "Continue processing"

    def add_observer(self, observer: Callable[[ResponseResult], None]) -> None:
        """Add observer for response results."""
        self._observers.append(observer)

    def remove_observer(self, observer: Callable[[ResponseResult], None]) -> None:
        """Remove observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def get_stats(self) -> PipelineStats:
        """Get pipeline statistics."""
        # Lazy update stats only when requested (sync access for existing components)
        if self._prompt_builder:
            self._stats.cache_state = self._prompt_builder.get_profile_stats().get(
                "cache_state", "unknown"
            )
        if self._mode_controller:
            self._stats.mode_transitions = self._mode_controller.get_session_stats().get(
                "mode_transitions", 0
            )
        return self._stats

    def reset_session(self) -> None:
        """Reset session tracking."""
        self._session_start = datetime.now()
        self._current_context = None
        self._stats = PipelineStats(profile_name=self.profile_name)

        if self._mode_controller:
            self._mode_controller.reset_session()

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learned behaviors."""
        summary = {
            "profile_name": self.profile_name,
            "total_requests": self._stats.total_requests,
            "success_rate": (self._stats.successful_requests / max(self._stats.total_requests, 1)),
        }

        if self._prompt_builder:
            summary["prompt_profile"] = self._prompt_builder.get_profile_stats()
        if self._mode_controller:
            summary["mode_session"] = self._mode_controller.get_session_stats()

        return summary


# Module-level convenience functions
_pipeline_cache: Dict[Tuple[str, str, str, Optional[str]], IntelligentAgentPipeline] = {}


async def get_pipeline(
    provider_name: str,
    model: str,
    profile_name: Optional[str] = None,
    project_root: Optional[str] = None,
) -> IntelligentAgentPipeline:
    """Get or create a pipeline instance.

    Args:
        provider_name: Provider name
        model: Model name
        profile_name: Optional profile name
        project_root: Optional project root

    Returns:
        IntelligentAgentPipeline instance
    """
    key = (provider_name, model, profile_name or "default", project_root)

    if key not in _pipeline_cache:
        _pipeline_cache[key] = await IntelligentAgentPipeline.create(
            provider_name=provider_name,
            model=model,
            profile_name=profile_name,
            project_root=project_root,
        )

    return _pipeline_cache[key]


def clear_pipeline_cache() -> None:
    """Clear the pipeline cache."""
    _pipeline_cache.clear()
