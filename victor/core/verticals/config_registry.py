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

"""Centralized registry for vertical configuration templates.

Provides pre-built configurations for common vertical patterns,
eliminating duplication across vertical assistants.
"""

from typing import Any


class VerticalConfigRegistry:
    """Registry of pre-built vertical configuration templates.

    Implements Registry pattern for OCP compliance - new configurations
    can be registered without modifying existing code.

    Example:
        # Get pre-built coding provider hints
        hints = VerticalConfigRegistry.get_provider_hints("coding")

        # Register custom configuration
        VerticalConfigRegistry.register_provider_hints("my_vertical", {...})
    """

    # Provider hints templates
    _provider_hints: dict[str, dict[str, Any]] = {
        "coding": {
            "preferred_providers": ["anthropic", "openai"],
            "preferred_models": [
                "claude-sonnet-4-20250514",
                "gpt-4-turbo",
                "claude-3-5-sonnet-20241022",
            ],
            "min_context_window": 100000,
            "requires_tool_calling": True,
            "prefers_extended_thinking": True,
        },
        "research": {
            "preferred_providers": ["anthropic", "openai", "google"],
            "min_context_window": 100000,
            "features": ["web_search", "large_context"],
        },
        "devops": {
            "preferred_providers": ["anthropic", "openai"],
            "min_context_window": 100000,
            "features": ["tool_calling", "large_context"],
            "requires_tool_calling": True,
        },
        "data_analysis": {
            "preferred_providers": ["anthropic", "openai"],
            "min_context_window": 128000,
            "features": ["tool_calling", "large_context", "code_execution"],
        },
        "rag": {
            "preferred_providers": ["anthropic", "openai", "google"],
            "min_context_window": 8000,
            "features": ["tool_calling"],
            "temperature": 0.3,
        },
        "default": {
            "preferred_providers": ["anthropic", "openai"],
            "min_context_window": 100000,
            "requires_tool_calling": True,
        },
    }

    # Evaluation criteria templates
    _evaluation_criteria: dict[str, list[str]] = {
        "coding": [
            "Code correctness and functionality",
            "Test coverage and validation",
            "Code quality and maintainability",
            "Security best practices",
            "Performance considerations",
        ],
        "research": [
            "accuracy",
            "source_quality",
            "comprehensiveness",
            "clarity",
            "attribution",
            "objectivity",
            "timeliness",
        ],
        "devops": [
            "configuration_correctness",
            "security_best_practices",
            "idempotency",
            "documentation_completeness",
            "resource_efficiency",
            "disaster_recovery",
            "monitoring_coverage",
        ],
        "data_analysis": [
            "statistical_correctness",
            "visualization_quality",
            "insight_clarity",
            "reproducibility",
            "data_privacy",
            "methodology_transparency",
        ],
        "rag": [
            "Answer is grounded in retrieved documents",
            "Sources are properly cited",
            "No hallucination of facts not in documents",
            "Relevant documents were retrieved",
            "Answer is coherent and well-structured",
        ],
        "default": [
            "Task completion accuracy",
            "Tool usage efficiency",
            "Response relevance",
            "Error handling",
        ],
    }

    @classmethod
    def get_provider_hints(cls, vertical_name: str) -> dict[str, Any]:
        """Get provider hints for a vertical.

        Args:
            vertical_name: Name of vertical (e.g., "coding", "research")

        Returns:
            Provider hints dictionary (copy to prevent mutation)
        """
        if vertical_name not in cls._provider_hints:
            # Fallback to default
            return cls._provider_hints["default"].copy()
        return cls._provider_hints[vertical_name].copy()

    @classmethod
    def get_evaluation_criteria(cls, vertical_name: str) -> list[str]:
        """Get evaluation criteria for a vertical.

        Args:
            vertical_name: Name of vertical (e.g., "coding", "research")

        Returns:
            List of evaluation criteria (copy to prevent mutation)
        """
        if vertical_name not in cls._evaluation_criteria:
            # Fallback to default
            return cls._evaluation_criteria["default"].copy()
        return cls._evaluation_criteria[vertical_name].copy()

    @classmethod
    def register_provider_hints(cls, key: str, hints: dict[str, Any]) -> None:
        """Register custom provider hints (for extensibility).

        Args:
            key: Unique identifier for this configuration
            hints: Provider hints dictionary
        """
        cls._provider_hints[key] = hints

    @classmethod
    def register_evaluation_criteria(cls, key: str, criteria: list[str]) -> None:
        """Register custom evaluation criteria (for extensibility).

        Args:
            key: Unique identifier for this configuration
            criteria: List of evaluation criteria
        """
        cls._evaluation_criteria[key] = criteria
