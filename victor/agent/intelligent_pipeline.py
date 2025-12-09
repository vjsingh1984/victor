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
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


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
        self._prompt_builder = None
        self._mode_controller = None
        self._quality_scorer = None
        self._grounding_verifier = None
        self._resilient_executor = None

        # State tracking
        self._current_context: Optional[RequestContext] = None
        self._session_start = datetime.now()
        self._stats = PipelineStats(profile_name=profile_name)

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
        await pipeline._initialize_components()
        return pipeline

    async def _initialize_components(self) -> None:
        """Initialize all intelligent components."""
        # Import here to avoid circular dependencies
        try:
            from victor.agent.intelligent_prompt_builder import IntelligentPromptBuilder

            self._prompt_builder = await IntelligentPromptBuilder.create(
                provider_name=self.provider_name,
                model=self.model,
                profile_name=self.profile_name,
            )
            logger.debug("[IntelligentPipeline] Prompt builder initialized")
        except Exception as e:
            logger.warning(f"[IntelligentPipeline] Prompt builder init failed: {e}")

        try:
            from victor.agent.adaptive_mode_controller import AdaptiveModeController

            self._mode_controller = AdaptiveModeController(
                profile_name=self.profile_name,
            )
            logger.debug("[IntelligentPipeline] Mode controller initialized")
        except Exception as e:
            logger.warning(f"[IntelligentPipeline] Mode controller init failed: {e}")

        try:
            from victor.agent.response_quality import ResponseQualityScorer

            self._quality_scorer = ResponseQualityScorer()
            logger.debug("[IntelligentPipeline] Quality scorer initialized")
        except Exception as e:
            logger.warning(f"[IntelligentPipeline] Quality scorer init failed: {e}")

        try:
            from victor.agent.grounding_verifier import GroundingVerifier

            if self.project_root:
                self._grounding_verifier = GroundingVerifier(
                    project_root=self.project_root,
                )
                logger.debug("[IntelligentPipeline] Grounding verifier initialized")
        except Exception as e:
            logger.warning(f"[IntelligentPipeline] Grounding verifier init failed: {e}")

        try:
            from victor.agent.resilience import ResilientExecutor

            self._resilient_executor = ResilientExecutor()
            logger.debug("[IntelligentPipeline] Resilient executor initialized")
        except Exception as e:
            logger.warning(f"[IntelligentPipeline] Resilient executor init failed: {e}")

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

        # Build intelligent prompt
        system_prompt = ""
        if self._prompt_builder:
            system_prompt = await self._prompt_builder.build(
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

        if self._mode_controller:
            action = self._mode_controller.get_recommended_action(
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

            # Get learned optimal budget for task type
            learned_budget = self._mode_controller.get_optimal_tool_budget(task_type)
            if learned_budget != 10:  # Not default
                recommended_budget = min(recommended_budget, learned_budget + 5)

        # Get profile stats
        profile_stats = {}
        if self._prompt_builder:
            profile_stats = self._prompt_builder.get_profile_stats()

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

        # Score quality
        quality_score = 0.5
        quality_details = {}
        improvement_suggestions = []

        if self._quality_scorer and query:
            quality_result = await self._quality_scorer.score(
                query=query,
                response=response,
                context=context,
            )
            quality_score = quality_result.overall_score
            quality_details = {
                dim.dimension.value: dim.score for dim in quality_result.dimension_scores
            }
            improvement_suggestions = quality_result.improvement_suggestions

        # Verify grounding
        grounding_score = 1.0
        is_grounded = True
        grounding_issues = []

        if self._grounding_verifier:
            grounding_result = await self._grounding_verifier.verify(
                response=response,
                context=context,
            )
            grounding_score = grounding_result.confidence
            is_grounded = grounding_result.is_grounded
            grounding_issues = [
                f"{issue.issue_type.value}: {issue.description}"
                for issue in grounding_result.issues
            ]

        # Record feedback for learning
        learning_reward = 0.0
        response_time_ms = (time.perf_counter() - start_time) * 1000

        if self._prompt_builder:
            self._prompt_builder.record_feedback(
                task_type=task_type,
                success=success,
                quality_score=quality_score,
                response_time_ms=response_time_ms,
                tool_calls=tool_calls,
                tool_budget=tool_budget,
                grounded=is_grounded,
            )

        if self._mode_controller:
            learning_reward = self._mode_controller.record_outcome(
                success=success,
                quality_score=quality_score,
                user_satisfied=quality_score > 0.6,
                completed=success and quality_score > 0.7,
            )

        # Update stats
        if success:
            self._stats.successful_requests += 1
        self._stats.avg_quality_score = 0.9 * self._stats.avg_quality_score + 0.1 * quality_score
        self._stats.avg_grounding_score = (
            0.9 * self._stats.avg_grounding_score + 0.1 * grounding_score
        )
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
        )

        # Notify observers
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
        messages: List[Dict[str, Any]],
        circuit_name: Optional[str] = None,
        fallback: Optional[Callable] = None,
        **kwargs,
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
        if not self._resilient_executor:
            # No resilience - direct call
            return await provider.chat(messages=messages, **kwargs)

        circuit = circuit_name or f"{self.provider_name}:{self.model}"

        async def call_provider():
            return await provider.chat(messages=messages, **kwargs)

        async def fallback_func():
            if fallback:
                return await fallback()
            raise RuntimeError("No fallback provided")

        try:
            result = await self._resilient_executor.execute(
                circuit_name=circuit,
                func=call_provider,
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
    ) -> tuple[bool, str]:
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
        if self._mode_controller:
            return self._mode_controller.should_continue(
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
        # Update cache state
        if self._prompt_builder:
            profile_stats = self._prompt_builder.get_profile_stats()
            self._stats.cache_state = profile_stats.get("cache_state", "unknown")

        # Update mode transitions
        if self._mode_controller:
            session_stats = self._mode_controller.get_session_stats()
            self._stats.mode_transitions = session_stats.get("mode_transitions", 0)

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
_pipeline_cache: Dict[str, IntelligentAgentPipeline] = {}


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
    key = f"{provider_name}:{model}:{profile_name or 'default'}"

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
