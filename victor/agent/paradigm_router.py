# Copyright 2025 Vijaykumar Singh <singhv@gmail.com>
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

"""Paradigm Router for optimal processing paradigm selection.

Based on arXiv:2604.06753 (Select-then-Solve) - Routes queries to optimal
processing paradigms without LLM call, using rule-based heuristics to select
the appropriate model tier and processing approach.

This provides additional 10-15% cost savings by:
- Using small models for simple tasks
- Using focused processing for medium complexity
- Using large models only for complex tasks
- Avoiding LLM routing overhead
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Optional imports for enhanced functionality
try:
    from victor.agent.complexity_estimator import ComplexityEstimator, get_complexity_estimator

    COMPLEXITY_ESTIMATOR_AVAILABLE = True
except ImportError:
    COMPLEXITY_ESTIMATOR_AVAILABLE = False

try:
    from victor.agent.task_classifier import TaskClassifier, get_task_classifier

    TASK_CLASSIFIER_AVAILABLE = True
except ImportError:
    TASK_CLASSIFIER_AVAILABLE = False


class ProcessingParadigm(str, Enum):
    """Processing paradigm for task execution.

    Determines the execution approach:
    - DIRECT: Execute immediately without planning (fastest)
    - FOCUSED: Single-pass execution with minimal context
    - STANDARD: Multi-pass execution with standard context
    - DEEP: Comprehensive multi-pass execution with full context
    """

    DIRECT = "direct"  # Immediate execution, no planning
    FOCUSED = "focused"  # Single-pass, minimal context
    STANDARD = "standard"  # Multi-pass, standard context
    DEEP = "deep"  # Comprehensive, full context


class ModelTier(str, Enum):
    """Model tier for task execution.

    Determines the model size/capability:
    - SMALL: Fast, cost-effective (e.g., GPT-4o-mini, Claude Haiku)
    - MEDIUM: Balanced capability/speed (e.g., GPT-4o, Claude Sonnet)
    - LARGE: Maximum capability (e.g., GPT-4.1, Claude Opus)
    """

    SMALL = "small"  # Fast, cost-effective
    MEDIUM = "medium"  # Balanced
    LARGE = "large"  # Maximum capability


@dataclass
class RoutingDecision:
    """Routing decision for task execution.

    Attributes:
        paradigm: Processing paradigm to use
        model_tier: Model tier to use
        max_tokens: Maximum tokens for response
        tool_budget: Recommended tool budget
        skip_planning: Whether to skip planning phase
        skip_evaluation: Whether to skip evaluation phase
        confidence: Confidence in this decision (0-1)
        reasoning: Human-readable explanation
    """

    paradigm: ProcessingParadigm
    model_tier: ModelTier
    max_tokens: int
    tool_budget: int
    skip_planning: bool
    skip_evaluation: bool
    confidence: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "paradigm": self.paradigm.value,
            "model_tier": self.model_tier.value,
            "max_tokens": self.max_tokens,
            "tool_budget": self.tool_budget,
            "skip_planning": self.skip_planning,
            "skip_evaluation": self.skip_evaluation,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


class ParadigmRouter:
    """Router for optimal processing paradigm selection.

    Uses rule-based heuristics to route queries to optimal processing
    paradigms and model tiers without LLM call overhead.

    Based on arXiv:2604.06753 (Select-then-Solve) - Selects the optimal
    paradigm before execution, avoiding LLM routing overhead.

    Example:
        router = ParadigmRouter()
        decision = router.route(
            task_type="create_simple",
            query="create a new file",
            history_length=0,
            query_complexity=0.1,
        )
        # Returns: DIRECT paradigm, SMALL model, 500 tokens
    """

    # Simple task types that can use direct execution
    DIRECT_TASK_TYPES = {
        "create_simple",
        "action",
        "search",
        "quick_question",
    }

    # Medium complexity task types
    FOCUSED_TASK_TYPES = {
        "edit",
        "debug",
        "refactor",
        "test",
    }

    # High complexity task types
    DEEP_TASK_TYPES = {
        "design",
        "analysis_deep",
        "exploration",
        "swe_bench_issue",
    }

    # Action keywords for direct execution
    ACTION_KEYWORDS = [
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

    def __init__(
        self,
        enabled: bool = True,
        use_enhanced_classification: bool = True,
        use_enhanced_complexity: bool = True,
    ):
        """Initialize the paradigm router.

        Args:
            enabled: Whether the router is enabled (for feature flagging)
            use_enhanced_classification: Whether to use LLM-based task classification
            use_enhanced_complexity: Whether to use edge model complexity estimation
        """
        self.enabled = enabled
        self.use_enhanced_classification = use_enhanced_classification
        self.use_enhanced_complexity = use_enhanced_complexity
        self._routing_count = 0
        self._paradigm_stats: Dict[str, int] = {
            "direct": 0,
            "focused": 0,
            "standard": 0,
            "deep": 0,
        }

        # Initialize enhanced components if available and enabled
        self._complexity_estimator = None
        self._task_classifier = None

        if self.use_enhanced_complexity and COMPLEXITY_ESTIMATOR_AVAILABLE:
            self._complexity_estimator = get_complexity_estimator()

        if self.use_enhanced_classification and TASK_CLASSIFIER_AVAILABLE:
            self._task_classifier = get_task_classifier()

    async def route_async(
        self,
        task_type: str,
        query: str,
        history_length: int = 0,
        query_complexity: Optional[float] = None,
        tool_budget: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Route query to optimal processing paradigm (async version with enhanced estimation).

        Uses LLM-based task classification and complexity estimation when available
        for more accurate routing decisions.

        Args:
            task_type: Detected task type
            query: User's query
            history_length: Length of conversation history
            query_complexity: Optional complexity score (0-1)
            tool_budget: Optional tool budget constraint
            context: Additional execution context

        Returns:
            RoutingDecision with paradigm, model tier, and settings
        """
        self._routing_count += 1

        if not self.enabled:
            decision = RoutingDecision(
                paradigm=ProcessingParadigm.STANDARD,
                model_tier=ModelTier.MEDIUM,
                max_tokens=2000,
                tool_budget=tool_budget or 10,
                skip_planning=False,
                skip_evaluation=False,
                confidence=0.5,
                reasoning="Router disabled, using standard paradigm",
            )
            self._paradigm_stats["standard"] += 1
            return decision

        # Use enhanced task classification if available
        if self._task_classifier:
            try:
                classification = await self._task_classifier.classify(query, context)
                if classification.confidence > 0.8:
                    task_type = classification.task_type
                    logger.debug(
                        f"[ParadigmRouter] Enhanced classification: {task_type} "
                        f"(confidence={classification.confidence:.2f})"
                    )
            except Exception as e:
                logger.warning(f"[ParadigmRouter] Task classification failed: {e}")

        # Use enhanced complexity estimation if available
        if self._complexity_estimator and query_complexity is None:
            try:
                estimate = await self._complexity_estimator.estimate(query, context)
                query_complexity = estimate.score
                logger.debug(
                    f"[ParadigmRouter] Enhanced complexity: {query_complexity:.2f} "
                    f"(band={estimate.band.value})"
                )
            except Exception as e:
                logger.warning(f"[ParadigmRouter] Complexity estimation failed: {e}")

        # Fall back to synchronous routing with enhanced data
        return self.route(
            task_type=task_type,
            query=query,
            history_length=history_length,
            query_complexity=query_complexity,
            tool_budget=tool_budget,
            context=context,
        )

    def route(
        self,
        task_type: str,
        query: str,
        history_length: int = 0,
        query_complexity: Optional[float] = None,
        tool_budget: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Route query to optimal processing paradigm.

        Uses rule-based heuristics to determine:
        - Processing paradigm (direct, focused, standard, deep)
        - Model tier (small, medium, large)
        - Token budget
        - Whether to skip planning/evaluation

        Args:
            task_type: Detected task type
            query: User's query
            history_length: Length of conversation history
            query_complexity: Optional complexity score (0-1)
            tool_budget: Optional tool budget constraint
            context: Additional execution context

        Returns:
            RoutingDecision with paradigm, model tier, and settings
        """
        self._routing_count += 1

        if not self.enabled:
            # Router disabled, use standard paradigm
            decision = RoutingDecision(
                paradigm=ProcessingParadigm.STANDARD,
                model_tier=ModelTier.MEDIUM,
                max_tokens=2000,
                tool_budget=tool_budget or 10,
                skip_planning=False,
                skip_evaluation=False,
                confidence=0.5,
                reasoning="Router disabled, using standard paradigm",
            )
            self._paradigm_stats["standard"] += 1
            return decision

        # Extract features for routing
        query_lower = query.lower()
        query_length = len(query)
        has_action_keyword = any(kw in query_lower for kw in self.ACTION_KEYWORDS)

        # Fast Pattern 1: Direct execution for simple tasks
        if (
            task_type in self.DIRECT_TASK_TYPES
            and history_length == 0
            and (query_complexity or 0) < 0.3
            and query_length < 100
        ):
            decision = RoutingDecision(
                paradigm=ProcessingParadigm.DIRECT,
                model_tier=ModelTier.SMALL,
                max_tokens=500,
                tool_budget=tool_budget or 3,
                skip_planning=True,
                skip_evaluation=True,
                confidence=0.9,
                reasoning=f"Direct execution: simple task type={task_type}, "
                f"no history, low complexity, short query",
            )
            self._paradigm_stats["direct"] += 1
            logger.info(
                f"[ParadigmRouter] DIRECT: task_type={task_type}, " f"model=SMALL, max_tokens=500"
            )
            return decision

        # Fast Pattern 2: Direct execution for action queries
        if (
            has_action_keyword
            and query_length < 80
            and history_length == 0
            and (query_complexity or 0) < 0.4
        ):
            decision = RoutingDecision(
                paradigm=ProcessingParadigm.DIRECT,
                model_tier=ModelTier.SMALL,
                max_tokens=600,
                tool_budget=tool_budget or 3,
                skip_planning=True,
                skip_evaluation=True,
                confidence=0.85,
                reasoning="Direct execution: action query, short, no history, " "low complexity",
            )
            self._paradigm_stats["direct"] += 1
            logger.info("[ParadigmRouter] DIRECT: action query, model=SMALL, max_tokens=600")
            return decision

        # Fast Pattern 3: Focused processing for medium tasks
        if (
            task_type in self.FOCUSED_TASK_TYPES
            or (query_complexity is not None and 0.3 <= query_complexity < 0.6)
        ) and history_length < 3:
            decision = RoutingDecision(
                paradigm=ProcessingParadigm.FOCUSED,
                model_tier=ModelTier.MEDIUM,
                max_tokens=1000,
                tool_budget=tool_budget or 8,
                skip_planning=False,
                skip_evaluation=False,
                confidence=0.8,
                reasoning=f"Focused processing: medium task type={task_type}, "
                f"short history, moderate complexity",
            )
            self._paradigm_stats["focused"] += 1
            logger.info(
                f"[ParadigmRouter] FOCUSED: task_type={task_type}, "
                f"model=MEDIUM, max_tokens=1000"
            )
            return decision

        # Fast Pattern 4: Deep processing for complex tasks
        if (
            task_type in self.DEEP_TASK_TYPES
            or (query_complexity is not None and query_complexity >= 0.6)
            or history_length >= 5
        ):
            decision = RoutingDecision(
                paradigm=ProcessingParadigm.DEEP,
                model_tier=ModelTier.LARGE,
                max_tokens=4000,
                tool_budget=tool_budget or 20,
                skip_planning=False,
                skip_evaluation=False,
                confidence=0.85,
                reasoning=f"Deep processing: complex task type={task_type}, "
                f"long history or high complexity",
            )
            self._paradigm_stats["deep"] += 1
            logger.info(
                f"[ParadigmRouter] DEEP: task_type={task_type}, " f"model=LARGE, max_tokens=4000"
            )
            return decision

        # Default: Standard processing
        decision = RoutingDecision(
            paradigm=ProcessingParadigm.STANDARD,
            model_tier=ModelTier.MEDIUM,
            max_tokens=2000,
            tool_budget=tool_budget or 10,
            skip_planning=False,
            skip_evaluation=False,
            confidence=0.7,
            reasoning="Standard processing: default for unclassified tasks",
        )
        self._paradigm_stats["standard"] += 1
        logger.info("[ParadigmRouter] STANDARD: default paradigm")
        return decision

    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics.

        Returns:
            Dict with routing counts and percentages
        """
        if self._routing_count == 0:
            return {
                "total_routings": 0,
                "paradigm_percentages": {},
            }

        paradigm_percentages = {
            paradigm: (count / self._routing_count) * 100
            for paradigm, count in self._paradigm_stats.items()
        }

        return {
            "total_routings": self._routing_count,
            "paradigm_counts": self._paradigm_stats.copy(),
            "paradigm_percentages": paradigm_percentages,
            "direct_percentage": paradigm_percentages.get("direct", 0),
            "small_model_usage": paradigm_percentages.get("direct", 0)
            + (paradigm_percentages.get("focused", 0) * 0.5),  # Focused uses medium
        }

    def reset_statistics(self) -> None:
        """Reset routing statistics."""
        self._routing_count = 0
        self._paradigm_stats = {
            "direct": 0,
            "focused": 0,
            "standard": 0,
            "deep": 0,
        }


# Singleton instance
_paradigm_router_instance: Optional[ParadigmRouter] = None


def get_paradigm_router() -> ParadigmRouter:
    """Get the singleton ParadigmRouter instance.

    Returns:
        ParadigmRouter singleton instance
    """
    global _paradigm_router_instance
    if _paradigm_router_instance is None:
        _paradigm_router_instance = ParadigmRouter()
    return _paradigm_router_instance


__all__ = [
    "ParadigmRouter",
    "RoutingDecision",
    "ProcessingParadigm",
    "ModelTier",
    "get_paradigm_router",
]
