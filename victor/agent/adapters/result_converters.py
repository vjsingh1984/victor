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

"""Result converters for adapter type transformations.

This module provides converter functions to transform between different
result types used throughout the codebase. These converters support
backward compatibility by converting protocol-based results to legacy
dict formats.

Design Pattern:
- Adapter Pattern: Converts between incompatible interfaces
- Single Responsibility: Each converter handles one type transformation
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ResultConverters:
    """Static utility class for result type conversions.

    This class provides static methods to convert between different
    result types used in the orchestrator and coordinators.

    Example:
        # Convert ValidationResult to dict
        result = ValidationResult(is_valid=True)
        result_dict = ResultConverters.validation_result_to_dict(result)

        # Convert IntelligentValidationResult to dict
        intelligent_result = IntelligentValidationResult(
            is_valid=True,
            quality_score=0.85,
            grounding_score=0.92
        )
        result_dict = ResultConverters.intelligent_validation_to_dict(intelligent_result)
    """

    @staticmethod
    def validation_result_to_dict(result: Any) -> dict[str, Any]:
        """Convert ValidationResult to dictionary format.

        Args:
            result: ValidationResult instance

        Returns:
            Dictionary with validation data
        """
        if result is None:
            return {}

        return {
            "is_valid": result.is_valid,
            "errors": list(result.errors) if hasattr(result, "errors") else [],
            "warnings": list(result.warnings) if hasattr(result, "warnings") else [],
        }

    @staticmethod
    def intelligent_validation_to_dict(
        result: Any,
    ) -> Optional[dict[str, Any]]:
        """Convert IntelligentValidationResult to dictionary format.

        This converter maintains backward compatibility with code expecting
        dict format validation results from intelligent pipeline.

        Args:
            result: IntelligentValidationResult instance or None

        Returns:
            Dictionary with validation scores and metadata, or None if input is None
        """
        if result is None:
            return None

        return {
            "quality_score": result.quality_score,
            "grounding_score": result.grounding_score,
            "is_grounded": result.is_grounded,
            "is_valid": result.is_valid,
            "grounding_issues": (
                list(result.grounding_issues) if hasattr(result, "grounding_issues") else []
            ),
            "should_finalize": (
                result.should_finalize if hasattr(result, "should_finalize") else False
            ),
            "should_retry": result.should_retry if hasattr(result, "should_retry") else False,
            "finalize_reason": result.finalize_reason if hasattr(result, "finalize_reason") else "",
            "grounding_feedback": (
                result.grounding_feedback if hasattr(result, "grounding_feedback") else ""
            ),
        }

    @staticmethod
    def token_usage_to_dict(usage: Any) -> dict[str, Any]:
        """Convert TokenUsage to dictionary format.

        Args:
            usage: TokenUsage instance or dict

        Returns:
            Dictionary with token usage data
        """
        if usage is None:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        if isinstance(usage, dict):
            return usage

        # Try to extract attributes
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    @staticmethod
    def tool_execution_to_dict(execution_result: Any) -> dict[str, Any]:
        """Convert tool execution result to dictionary format.

        Args:
            execution_result: Tool execution result (any type)

        Returns:
            Dictionary with standardized execution data
        """
        if execution_result is None:
            return {"success": False, "error": "No result"}

        if isinstance(execution_result, dict):
            return execution_result

        # Try to extract common attributes
        result = {}
        if hasattr(execution_result, "success"):
            result["success"] = execution_result.success
        if hasattr(execution_result, "output"):
            result["output"] = execution_result.output
        if hasattr(execution_result, "error"):
            result["error"] = execution_result.error
        if hasattr(execution_result, "duration"):
            result["duration"] = execution_result.duration

        return result

    @staticmethod
    def checkpoint_state_to_dict(state: Any) -> dict[str, Any]:
        """Convert checkpoint state to dictionary format.

        Args:
            state: Checkpoint state (any type)

        Returns:
            Dictionary with checkpoint data
        """
        if state is None:
            return {}

        if isinstance(state, dict):
            return state

        # Try to convert dataclass
        if hasattr(state, "__dataclass_fields__"):
            return asdict(state)

        # Try to extract attributes
        result = {}
        if hasattr(state, "stage"):
            result["stage"] = state.stage
        if hasattr(state, "tool_history"):
            result["tool_history"] = list(state.tool_history)
        if hasattr(state, "observed_files"):
            result["observed_files"] = list(state.observed_files)
        if hasattr(state, "modified_files"):
            result["modified_files"] = list(state.modified_files)

        return result


__all__ = [
    "ResultConverters",
]
