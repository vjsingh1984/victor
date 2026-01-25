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

"""Intelligent Pipeline Adapter - Bridges orchestrator and intelligent pipeline.

This adapter extracts the intelligent pipeline integration logic from the
orchestrator, providing a clean interface for pre-request validation and
post-response quality checks.

Responsibilities:
- Prepare intelligent request with task analysis
- Validate intelligent response with quality scoring
- Convert validation results to dict format (backward compatibility)
- Handle intelligent pipeline errors gracefully

Design Patterns:
- Adapter Pattern: Converts between orchestrator and intelligent pipeline interfaces
- Single Responsibility: Focuses only on intelligent pipeline integration
- Dependency Inversion: Depends on protocols, not concrete implementations
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from victor.agent.orchestrator_integration import OrchestratorIntegration
    from victor.coding.tracker.unified import UnifiedTracker
    from victor.agent.coordinators.validation_coordinator import (
        ValidationCoordinator,
        IntelligentValidationResult,
    )
    from victor.agent.conversation_state import ConversationState

from victor.agent.adapters.result_converters import ResultConverters

logger = logging.getLogger(__name__)


class IntelligentPipelineAdapter:
    """Adapter for intelligent pipeline integration.

    This adapter encapsulates the logic for integrating with the intelligent
    pipeline, which provides:
    - Pre-request analysis and recommendations
    - Post-response quality scoring
    - Grounding verification and hallucination detection
    - Finalization recommendations

    The adapter handles error cases gracefully and provides backward-compatible
    dict format results.

    Example:
        adapter = IntelligentPipelineAdapter(
            intelligent_integration=integration,
            validation_coordinator=validation_coordinator,
        )

        # Prepare request
        request_data = await adapter.prepare_intelligent_request(
            task="Analyze code",
            task_type="analysis",
            conversation_state=state,
            unified_tracker=tracker,
        )

        # Validate response
        validation_result = await adapter.validate_intelligent_response(
            response="The code shows...",
            query="Analyze code",
            tool_calls=3,
            task_type="analysis",
        )

        # Check if should finalize
        if validation_result and validation_result.get("should_finalize"):
            # Force finalization
            pass
    """

    def __init__(
        self,
        intelligent_integration: Optional[Any] = None,
        validation_coordinator: Optional["ValidationCoordinator"] = None,
    ):
        """Initialize the IntelligentPipelineAdapter.

        Args:
            intelligent_integration: Optional intelligent pipeline integration
            validation_coordinator: Optional validation coordinator
        """
        self._intelligent_integration = intelligent_integration
        self._validation_coordinator = validation_coordinator

        logger.debug(
            f"IntelligentPipelineAdapter initialized: "
            f"integration={'enabled' if intelligent_integration else 'disabled'}, "
            f"validation={'enabled' if validation_coordinator else 'disabled'}"
        )

    async def prepare_intelligent_request(
        self,
        task: str,
        task_type: str,
        conversation_state: Optional["ConversationState"] = None,
        unified_tracker: Optional["UnifiedTracker"] = None,
    ) -> Optional[Dict[str, Any]]:
        """Prepare intelligent request with task analysis.

        This method:
        1. Checks if intelligent integration is available
        2. Delegates to integration for request preparation
        3. Returns None if integration is disabled (backward compatibility)

        Args:
            task: The user's task/query
            task_type: Detected task type (analysis, edit, etc.)
            conversation_state: Current conversation state
            unified_tracker: Unified tracker for context

        Returns:
            Dictionary with recommendations, or None if pipeline disabled
        """
        if not self._intelligent_integration:
            return None

        try:
            result = await self._intelligent_integration.prepare_intelligent_request(
                task=task,
                task_type=task_type,
                conversation_state=conversation_state,
                unified_tracker=unified_tracker,
            )
            logger.debug(
                f"Intelligent request prepared for task_type={task_type}: "
                f"recommendations={bool(result)}"
            )
            if result is None:
                return None
            # Ensure we return a proper dict, not Any
            return cast("dict[str, Any] | None", result)
        except Exception as e:
            logger.warning(f"Intelligent request preparation failed: {e}")
            return None

    async def validate_intelligent_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Validate intelligent response with quality scoring.

        This method:
        1. Updates intelligent integration reference in validation coordinator
        2. Delegates to validation coordinator for validation
        3. Converts result to dict format for backward compatibility
        4. Returns None if validation was skipped or integration disabled

        Args:
            response: The model's response content
            query: Original user query
            tool_calls: Number of tool calls made so far
            task_type: Task type for context

        Returns:
            Dictionary with quality/grounding scores, or None if pipeline disabled
        """
        # Update intelligent integration reference in validation coordinator
        if self._intelligent_integration and self._validation_coordinator:
            self._validation_coordinator._intelligent_integration = self._intelligent_integration

        if not self._validation_coordinator:
            return None

        try:
            # Delegate to validation coordinator
            validation_result = (
                await self._validation_coordinator.validate_intelligent_response(
                    response=response,
                    query=query,
                    tool_calls=tool_calls,
                    task_type=task_type,
                )
            )

            # Return None if validation was skipped (backward compatibility)
            if validation_result is None:
                return None

            result: IntelligentValidationResult = validation_result

            # Convert result to dict format for backward compatibility
            result_dict = ResultConverters.intelligent_validation_to_dict(result)

            logger.debug(
                f"Intelligent validation completed: "
                f"quality={result.quality_score:.2f}, "
                f"grounding={result.grounding_score:.2f}, "
                f"is_grounded={result.is_grounded}"
            )

            return result_dict
        except Exception as e:
            logger.warning(f"Intelligent response validation failed: {e}")
            return None

    def should_continue_intelligent(self) -> tuple[bool, str]:
        """Check if processing should continue using learned behaviors.

        This method delegates to the intelligent integration if available
        to determine whether to continue processing based on learned
        patterns and outcomes.

        Returns:
            Tuple of (should_continue, reason)
        """
        if not self._intelligent_integration:
            return True, "No intelligent integration"

        try:
            if hasattr(self._intelligent_integration, "should_continue_intelligent"):
                result = self._intelligent_integration.should_continue_intelligent()
                # Ensure tuple[bool, str] return type
                if isinstance(result, tuple) and len(result) == 2:
                    should_continue, reason = result
                    return (bool(should_continue), str(reason))
                return (True, "Invalid return type")
        except Exception as e:
            logger.warning(f"Intelligent continuation check failed: {e}")

        return True, "Check failed"

    async def record_intelligent_outcome(
        self,
        success: bool,
        quality_score: float = 0.5,
        user_satisfied: bool = True,
        completed: bool = True,
    ) -> None:
        """Record outcome for Q-learning feedback.

        This method delegates to the intelligent integration to record
        the outcome of a task for reinforcement learning.

        Args:
            success: Whether the task was completed successfully
            quality_score: Final quality score (0.0-1.0)
            user_satisfied: Whether user seemed satisfied
            completed: Whether task reached completion
        """
        if not self._intelligent_integration:
            return

        try:
            if hasattr(self._intelligent_integration, "record_intelligent_outcome"):
                await self._intelligent_integration.record_intelligent_outcome(
                    success=success,
                    quality_score=quality_score,
                    user_satisfied=user_satisfied,
                    completed=completed,
                )
                logger.debug(
                    f"Intelligent outcome recorded: success={success}, "
                    f"quality={quality_score:.2f}"
                )
        except Exception as e:
            logger.warning(f"Intelligent outcome recording failed: {e}")

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def intelligent_integration(self) -> Optional[Any]:
        """Get the intelligent integration."""
        return self._intelligent_integration

    @property
    def validation_coordinator(self) -> Optional["ValidationCoordinator"]:
        """Get the validation coordinator."""
        return self._validation_coordinator

    @property
    def is_enabled(self) -> bool:
        """Check if intelligent pipeline is enabled."""
        return self._intelligent_integration is not None


__all__ = [
    "IntelligentPipelineAdapter",
]
