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

"""Orchestrator integration bridge for intelligent pipeline components.

This module provides a non-intrusive integration layer that connects the
IntelligentAgentPipeline to the AgentOrchestrator. It hooks into the
orchestrator's request/response flow to enable:

1. Resilient Provider Calls:
   - Circuit breaker for failing providers
   - Automatic retry with exponential backoff
   - Rate limiting for API protection

2. Response Quality Scoring:
   - Multi-dimensional quality assessment
   - Grounding verification (hallucination detection)
   - Learning from user feedback

3. Intelligent Mode Transitions:
   - Q-learning based mode optimization
   - Profile-specific learning
   - Optimal tool budget calculation

4. Prompt Optimization:
   - Embedding-based context selection
   - Task-type-specific prompt generation
   - Provider-aware prompt strategies

Architecture (Decorator Pattern):

    ┌─────────────────────────────────────────────────────────┐
    │                 OrchestratorIntegration                  │
    │     (Decorator/Wrapper around orchestrator methods)      │
    └─────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
    ┌───────────────────────┐  ┌────────────────────────────┐
    │  AgentOrchestrator    │  │  IntelligentAgentPipeline  │
    │  (existing behavior)  │  │  (learning components)     │
    └───────────────────────┘  └────────────────────────────┘

Usage:
    orchestrator = AgentOrchestrator(settings, provider, model)
    integration = await OrchestratorIntegration.create(orchestrator)

    # The integration hooks into orchestrator callbacks
    # and enhances request/response processing

    # Check stats
    stats = integration.get_pipeline_stats()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.intelligent_pipeline import (
        IntelligentAgentPipeline,
        RequestContext,
        ResponseResult,
    )

# Import protocols for runtime type hints (protocols.py has no heavy deps)
from victor.core.protocols import OrchestratorProtocol, IntelligentPipelineProtocol

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for orchestrator integration."""

    # Enable/disable individual features
    enable_resilient_calls: bool = True
    enable_quality_scoring: bool = True
    enable_mode_learning: bool = True
    enable_prompt_optimization: bool = True

    # Quality thresholds
    min_quality_threshold: float = 0.5
    grounding_confidence_threshold: float = 0.7

    # Logging
    log_learning_events: bool = True
    log_quality_scores: bool = True


@dataclass
class IntegrationMetrics:
    """Metrics tracked by the integration."""

    total_requests: int = 0
    enhanced_requests: int = 0
    quality_validated_responses: int = 0
    grounding_checked_responses: int = 0
    mode_transitions: int = 0
    resilient_calls: int = 0
    circuit_breaker_trips: int = 0
    avg_quality_score: float = 0.0
    avg_grounding_score: float = 0.0
    total_learning_reward: float = 0.0


class OrchestratorIntegration:
    """Integration bridge connecting IntelligentAgentPipeline to AgentOrchestrator.

    This class wraps around an AgentOrchestrator instance and enhances its
    functionality with intelligent pipeline features. It uses a non-intrusive
    approach - the orchestrator continues to work normally, but gains:

    1. Pre-request optimization (prompt building, mode selection)
    2. Post-response validation (quality scoring, grounding verification)
    3. Continuous learning (feedback recording, Q-learning updates)
    """

    def __init__(
        self,
        orchestrator: "AgentOrchestrator",
        pipeline: "IntelligentAgentPipeline",
        config: Optional[IntegrationConfig] = None,
    ):
        """Initialize the integration bridge.

        Args:
            orchestrator: The AgentOrchestrator to enhance
            pipeline: The IntelligentAgentPipeline to use
            config: Optional configuration settings
        """
        self._orchestrator = orchestrator
        self._pipeline = pipeline
        self._config = config or IntegrationConfig()
        self._metrics = IntegrationMetrics()

        # Track current request context
        self._current_context: Optional["RequestContext"] = None
        self._session_start = datetime.now()

        # Observers for quality/grounding events
        self._quality_observers: List[Callable[[float, Dict[str, float]], None]] = []
        self._grounding_observers: List[Callable[[bool, List[str]], None]] = []

    @classmethod
    async def create(
        cls,
        orchestrator: "AgentOrchestrator",
        config: Optional[IntegrationConfig] = None,
    ) -> "OrchestratorIntegration":
        """Factory method to create and initialize the integration.

        Args:
            orchestrator: The orchestrator to enhance
            config: Optional configuration

        Returns:
            Initialized OrchestratorIntegration
        """
        from victor.agent.intelligent_pipeline import IntelligentAgentPipeline

        # Create pipeline from orchestrator's current state
        pipeline = await IntelligentAgentPipeline.create(
            provider_name=orchestrator.provider_name,
            model=orchestrator.model,
            profile_name=f"{orchestrator.provider_name}:{orchestrator.model}",
            project_root=(
                str(orchestrator.project_context.context_file.parent)
                if orchestrator.project_context.context_file
                else None
            ),
        )

        return cls(orchestrator, pipeline, config)

    async def prepare_request(
        self,
        task: str,
        task_type: str = "general",
        current_mode: Optional[str] = None,
    ) -> "RequestContext":
        """Prepare an optimized request context.

        Uses the intelligent pipeline to:
        1. Build an optimized system prompt
        2. Get mode transition recommendation
        3. Calculate optimal tool budget

        Args:
            task: The current task/query
            task_type: Detected task type
            current_mode: Current agent mode

        Returns:
            RequestContext with optimized settings
        """
        self._metrics.total_requests += 1

        if not self._config.enable_prompt_optimization:
            # Return minimal context if optimization disabled
            from victor.agent.intelligent_pipeline import RequestContext

            return RequestContext(
                system_prompt="",
                recommended_tool_budget=self._orchestrator.tool_budget,
                recommended_mode=current_mode or "explore",
                should_continue=True,
            )

        # Get mode from conversation state if not provided
        if current_mode is None:
            stage = self._orchestrator.conversation_state.get_current_stage()
            current_mode = stage.value if stage else "explore"

        # Prepare context using pipeline
        context = await self._pipeline.prepare_request(
            task=task,
            task_type=task_type,
            current_mode=current_mode,
            tool_calls_made=self._orchestrator.tool_calls_used,
            tool_budget=self._orchestrator.tool_budget,
            iteration_count=self._orchestrator.unified_tracker.iteration_count,
            iteration_budget=20,  # Default iteration budget
            quality_score=0.5,  # Initial quality estimate
        )

        self._current_context = context
        self._metrics.enhanced_requests += 1

        if self._config.log_learning_events:
            logger.debug(
                f"[OrchestratorIntegration] Prepared request: "
                f"mode={context.recommended_mode}, budget={context.recommended_tool_budget}"
            )

        return context

    async def validate_response(
        self,
        response: str,
        query: str = "",
        tool_calls: int = 0,
        success: bool = True,
        task_type: str = "general",
    ) -> "ResponseResult":
        """Validate and score a response.

        Uses the intelligent pipeline to:
        1. Score response quality
        2. Verify grounding (hallucination detection)
        3. Record feedback for learning

        Args:
            response: The model's response
            query: Original query
            tool_calls: Number of tool calls made
            success: Whether the response was successful
            task_type: Task type

        Returns:
            ResponseResult with quality and grounding info
        """
        from victor.agent.intelligent_pipeline import ResponseResult

        if not self._config.enable_quality_scoring:
            return ResponseResult(
                is_valid=True,
                quality_score=0.5,
                grounding_score=1.0,
                is_grounded=True,
            )

        result = await self._pipeline.process_response(
            response=response,
            query=query,
            tool_calls=tool_calls,
            tool_budget=self._orchestrator.tool_budget,
            success=success,
            task_type=task_type,
        )

        # Batch metrics update
        self._metrics.quality_validated_responses += 1
        if result.grounding_score > 0:
            self._metrics.grounding_checked_responses += 1
        
        # Use faster exponential moving average
        alpha = 0.1
        self._metrics.avg_quality_score += alpha * (result.quality_score - self._metrics.avg_quality_score)
        self._metrics.avg_grounding_score += alpha * (result.grounding_score - self._metrics.avg_grounding_score)
        self._metrics.total_learning_reward += result.learning_reward

        # Single-pass observer notification
        if self._quality_observers:
            for observer in self._quality_observers:
                try:
                    observer(result.quality_score, result.quality_details)
                except Exception:
                    pass
        if self._grounding_observers:
            for observer in self._grounding_observers:
                try:
                    observer(result.is_grounded, result.grounding_issues)
                except Exception:
                    pass

        if self._config.log_quality_scores:
            logger.debug(
                f"[OrchestratorIntegration] Response validated: "
                f"quality={result.quality_score:.2f}, grounded={result.is_grounded}"
            )

        return result

    def should_continue(self) -> tuple[bool, str]:
        """Determine if processing should continue using learned behaviors.

        Returns:
            Tuple of (should_continue, reason)
        """
        if not self._config.enable_mode_learning:
            return True, "Mode learning disabled"

        return self._pipeline.should_continue(
            tool_calls_made=self._orchestrator.tool_calls_used,
            tool_budget=self._orchestrator.tool_budget,
            quality_score=self._metrics.avg_quality_score,
            iteration_count=self._orchestrator.unified_tracker.iteration_count,
            iteration_budget=20,
        )

    def get_optimal_tool_budget(self, task_type: str) -> int:
        """Get learned optimal tool budget for a task type.

        Args:
            task_type: The task type

        Returns:
            Optimal tool budget
        """
        if self._pipeline._mode_controller and self._config.enable_mode_learning:
            return self._pipeline._mode_controller.get_optimal_tool_budget(task_type)
        return self._orchestrator.tool_budget

    def add_quality_observer(self, observer: Callable[[float, Dict[str, float]], None]) -> None:
        """Add observer for quality score events.

        Args:
            observer: Callback receiving (quality_score, quality_details)
        """
        self._quality_observers.append(observer)

    def add_grounding_observer(self, observer: Callable[[bool, List[str]], None]) -> None:
        """Add observer for grounding verification events.

        Args:
            observer: Callback receiving (is_grounded, grounding_issues)
        """
        self._grounding_observers.append(observer)

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics.

        Returns:
            Dictionary with pipeline stats
        """
        pipeline_stats = self._pipeline.get_stats()
        
        return {
            "integration": {
                "total_requests": self._metrics.total_requests,
                "enhanced_requests": self._metrics.enhanced_requests,
                "quality_validated": self._metrics.quality_validated_responses,
                "grounding_checked": self._metrics.grounding_checked_responses,
                "avg_quality_score": round(self._metrics.avg_quality_score, 3),
                "avg_grounding_score": round(self._metrics.avg_grounding_score, 3),
                "total_learning_reward": round(self._metrics.total_learning_reward, 3),
            },
            "pipeline": {
                "total_requests": pipeline_stats.total_requests,
                "successful_requests": pipeline_stats.successful_requests,
                "circuit_breaker_trips": pipeline_stats.circuit_breaker_trips,
                "cache_state": pipeline_stats.cache_state,
            },
            "learning": self._pipeline.get_learning_summary(),
            "config": {
                "resilient_calls": self._config.enable_resilient_calls,
                "quality_scoring": self._config.enable_quality_scoring,
                "mode_learning": self._config.enable_mode_learning,
                "prompt_optimization": self._config.enable_prompt_optimization,
            },
        }

    def get_quality_threshold_status(self) -> Dict[str, Any]:
        """Check if quality metrics meet configured thresholds.

        Returns:
            Dictionary with threshold status
        """
        quality_score = round(self._metrics.avg_quality_score, 3)
        grounding_score = round(self._metrics.avg_grounding_score, 3)
        
        return {
            "quality_meets_threshold": quality_score >= self._config.min_quality_threshold,
            "quality_score": quality_score,
            "quality_threshold": self._config.min_quality_threshold,
            "grounding_meets_threshold": grounding_score >= self._config.grounding_confidence_threshold,
            "grounding_score": grounding_score,
            "grounding_threshold": self._config.grounding_confidence_threshold,
        }

    def reset_session(self) -> None:
        """Reset session tracking."""
        self._session_start = datetime.now()
        self._current_context = None
        self._metrics = IntegrationMetrics()
        self._pipeline.reset_session()

    @property
    def orchestrator(self) -> "AgentOrchestrator":
        """Get the wrapped orchestrator."""
        return self._orchestrator

    @property
    def pipeline(self) -> "IntelligentAgentPipeline":
        """Get the intelligent pipeline."""
        return self._pipeline


# Convenience function for quick integration
async def enhance_orchestrator(
    orchestrator: "AgentOrchestrator",
    config: Optional[IntegrationConfig] = None,
) -> OrchestratorIntegration:
    """Enhance an orchestrator with intelligent pipeline capabilities.

    This is the main entry point for adding intelligent features to
    an existing orchestrator.

    Args:
        orchestrator: The AgentOrchestrator to enhance
        config: Optional configuration

    Returns:
        OrchestratorIntegration instance

    Example:
        orchestrator = AgentOrchestrator(settings, provider, model)
        integration = await enhance_orchestrator(orchestrator)

        # Use the enhanced flow
        context = await integration.prepare_request(task, task_type)
        # ... orchestrator does its work ...
        result = await integration.validate_response(response)
    """
    return await OrchestratorIntegration.create(orchestrator, config)
