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

"""Classification protocols for breaking circular dependencies.

This module provides protocol interfaces for classification results, enabling
components like semantic_selector to work with classification data without
direct imports of unified_classifier.

Key Protocols:
- IClassificationResult: Protocol for classification result data
- Task type and confidence extraction without circular dependencies

Example:
    from victor.protocols.classification import IClassificationResult

    def select_tools(
        user_message: str,
        classification: IClassificationResult,
    ) -> List[Tool]:
        # Access classification data without importing unified_classifier
        task_type = classification.task_type
        confidence = classification.confidence
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class IClassificationResult(Protocol):
    """Protocol for classification results.

    This protocol breaks circular dependencies between semantic_selector and
    unified_classifier by defining a minimal interface for classification data.

    Attributes:
        task_type: The classified task type (enum-like)
        confidence: Confidence score (0.0 to 1.0)
        is_action_task: Whether this is an action-oriented task
        is_analysis_task: Whether this is an analysis-oriented task
        is_generation_task: Whether this is a generation-oriented task
        needs_execution: Whether task needs execution
        source: Classification source (keyword, semantic, ensemble)
        recommended_tool_budget: Recommended tool budget for this task
        matched_keywords: List of matched keyword matches
        negated_keywords: List of negated keyword matches

    Example:
        # In semantic_selector - can use any object matching this protocol
        def select_tools(
            classification: IClassificationResult,
        ) -> List[Tool]:
            if classification.is_action_task:
                return get_action_tools()
            return []
    """

    task_type: Any  # Enum-like, should have .value attribute
    confidence: float
    is_action_task: bool
    is_analysis_task: bool
    is_generation_task: bool
    needs_execution: bool
    source: str
    recommended_tool_budget: int
    matched_keywords: List[Any]  # KeywordMatch objects
    negated_keywords: List[Any]  # KeywordMatch objects

    def get_task_type_value(self) -> str:
        """Get task type as string value.

        Returns:
            String representation of task type
        """
        ...


@runtime_checkable
class IKeywordMatch(Protocol):
    """Protocol for keyword match results.

    Attributes:
        keyword: The matched keyword string
        score: Confidence score for this match
        position: Character position in message
        negated: Whether this match was negated
        category: Category (action, analysis, generation, etc.)

    Example:
        match: IKeywordMatch = ...
        print(f"Matched '{match.keyword}' at position {match.position}")
    """

    keyword: str
    score: float
    position: int
    negated: bool
    category: str


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "IClassificationResult",
    "IKeywordMatch",
]
