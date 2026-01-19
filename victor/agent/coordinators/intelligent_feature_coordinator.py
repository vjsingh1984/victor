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

"""Intelligent Feature Coordinator for intelligent feature management.

This coordinator manages:
- Intelligent request preparation
- Intelligent response validation
- Intelligent outcome recording
- Embedding preloading

Design Pattern: Coordinator Pattern
- Centralizes intelligent feature logic
- Integrates with Q-learning and RL systems
- Provides clean API for orchestrator

Phase 6 Refactoring: Extracted from AgentOrchestrator
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class IntelligentFeatureCoordinator:
    """Coordinator for intelligent feature management.

    Manages intelligent features including Q-learning request preparation,
    response validation, outcome recording, and embedding preloading.

    Attributes:
        _intelligent_enabled: Whether intelligent features are enabled
        _qlearning_enabled: Whether Q-learning is enabled
        _embeddings_preloaded: Whether embeddings have been preloaded

    Example:
        coordinator = IntelligentFeatureCoordinator(settings)
        await coordinator.preload_embeddings()
        request = await coordinator.prepare_intelligent_request("task", "coding")
        validation = await coordinator.validate_intelligent_response(...)
    """

    def __init__(
        self,
        settings: Any,
        qlearning_coordinator: Optional[Any] = None,
        evaluation_coordinator: Optional[Any] = None,
    ):
        """Initialize IntelligentFeatureCoordinator.

        Args:
            settings: Application settings
            qlearning_coordinator: Optional Q-learning coordinator
            evaluation_coordinator: Optional evaluation coordinator
        """
        self._settings = settings
        self._qlearning_coordinator = qlearning_coordinator
        self._evaluation_coordinator = evaluation_coordinator
        self._intelligent_enabled = getattr(settings, "enable_intelligent_features", False)
        self._qlearning_enabled = getattr(settings, "enable_qlearning", False)
        self._embeddings_preloaded = False

    # ========================================================================
    # Intelligent Request Preparation
    # ========================================================================

    async def prepare_intelligent_request(
        self,
        task: str,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Prepare an intelligent request using Q-learning optimization.

        Delegates to Q-learning coordinator if enabled.

        Args:
            task: The task description
            task_type: Type of task (coding, analysis, etc.)

        Returns:
            Prepared request dict with optimized parameters, or None if disabled
        """
        if not self._intelligent_enabled or not self._qlearning_coordinator:
            return None

        try:
            return await self._qlearning_coordinator.prepare_request(task, task_type)
        except Exception as e:
            logger.warning(f"Failed to prepare intelligent request: {e}")
            return None

    # ========================================================================
    # Intelligent Response Validation
    # ========================================================================

    async def validate_intelligent_response(
        self,
        response: Dict[str, Any],
        expected_outcomes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Validate an intelligent response using quality metrics.

        Delegates to evaluation coordinator if available.

        Args:
            response: The response to validate
            expected_outcomes: Optional list of expected outcome keywords

        Returns:
            Validation result dict with quality metrics
        """
        if not self._evaluation_coordinator:
            return {"validated": False, "reason": "Evaluation coordinator not available"}

        try:
            return await self._evaluation_coordinator.validate_response(
                response,
                expected_outcomes=expected_outcomes,
            )
        except Exception as e:
            logger.warning(f"Failed to validate intelligent response: {e}")
            return {"validated": False, "reason": str(e)}

    # ========================================================================
    # Intelligent Outcome Recording
    # ========================================================================

    async def record_intelligent_outcome(
        self,
        task: str,
        task_type: str,
        outcome: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record intelligent outcome for Q-learning.

        Delegates to evaluation coordinator if available.

        Args:
            task: The task that was performed
            task_type: Type of task
            outcome: Outcome (success, failure, partial)
            metadata: Optional metadata about the outcome

        Returns:
            True if recorded successfully, False otherwise
        """
        if not self._evaluation_coordinator:
            return False

        try:
            return await self._evaluation_coordinator.record_outcome(
                task=task,
                task_type=task_type,
                outcome=outcome,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.warning(f"Failed to record intelligent outcome: {e}")
            return False

    # ========================================================================
    # Embedding Preloading
    # ========================================================================

    async def preload_embeddings(
        self,
        project_path: Optional[Path] = None,
    ) -> bool:
        """Preload embeddings for faster semantic search.

        Preloads embeddings for common project files to speed up
        semantic search operations.

        Args:
            project_path: Optional project path. If None, uses current directory.

        Returns:
            True if preloaded successfully, False otherwise
        """
        if self._embeddings_preloaded:
            logger.debug("Embeddings already preloaded")
            return True

        try:
            from victor.storage.embeddings.service import EmbeddingService

            # Get embedding model from settings
            embedding_model = getattr(
                self._settings, "unified_embedding_model", "BAAI/bge-small-en-v1.5"
            )

            # Create embedding service
            embedding_service = EmbeddingService(model_name=embedding_model)

            # Preload common file patterns if project path provided
            if project_path and project_path.exists():
                common_files = list(project_path.glob("*.py"))[:10]  # Limit to 10 files
                for file_path in common_files:
                    try:
                        with open(file_path, "r") as f:
                            content = f.read()
                            # Trigger embedding by calling encode
                            _ = embedding_service.encode(content)
                    except Exception:
                        # Skip files that can't be read
                        continue

            self._embeddings_preloaded = True
            logger.info("Preloaded embeddings for faster semantic search")
            return True

        except Exception as e:
            logger.warning(f"Failed to preload embeddings: {e}")
            return False

    # ========================================================================
    # State Management
    # ========================================================================

    def get_state(self) -> Dict[str, Any]:
        """Get coordinator state for monitoring.

        Returns:
            Dictionary with coordinator state
        """
        return {
            "intelligent_enabled": self._intelligent_enabled,
            "qlearning_enabled": self._qlearning_enabled,
            "embeddings_preloaded": self._embeddings_preloaded,
            "has_qlearning_coordinator": self._qlearning_coordinator is not None,
            "has_evaluation_coordinator": self._evaluation_coordinator is not None,
        }

    def reset(self) -> None:
        """Reset coordinator state."""
        self._embeddings_preloaded = False

    # ========================================================================
    # Computed Properties
    # ========================================================================

    def is_intelligent_enabled(self) -> bool:
        """Check if intelligent features are enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self._intelligent_enabled

    def is_qlearning_enabled(self) -> bool:
        """Check if Q-learning is enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self._qlearning_enabled

    def are_embeddings_preloaded(self) -> bool:
        """Check if embeddings are preloaded.

        Returns:
            True if preloaded, False otherwise
        """
        return self._embeddings_preloaded


def create_intelligent_feature_coordinator(
    settings: Any,
    qlearning_coordinator: Optional[Any] = None,
    evaluation_coordinator: Optional[Any] = None,
) -> IntelligentFeatureCoordinator:
    """Factory function to create IntelligentFeatureCoordinator.

    Args:
        settings: Application settings
        qlearning_coordinator: Optional Q-learning coordinator
        evaluation_coordinator: Optional evaluation coordinator

    Returns:
        Configured IntelligentFeatureCoordinator instance
    """
    return IntelligentFeatureCoordinator(
        settings=settings,
        qlearning_coordinator=qlearning_coordinator,
        evaluation_coordinator=evaluation_coordinator,
    )


__all__ = [
    "IntelligentFeatureCoordinator",
    "create_intelligent_feature_coordinator",
]
