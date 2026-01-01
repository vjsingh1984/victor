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

"""DEPRECATED: Task complexity classification.

This module is deprecated and will be removed in a future version.
Please update your imports to use the new locations:

Classification and budgets:
    from victor.framework.task import (
        TaskComplexity,
        TaskClassification,
        TaskComplexityService,
        DEFAULT_BUDGETS,
        classify_task,
    )

Prompt hints:
    from victor.framework.enrichment.strategies import (
        ComplexityHintEnricher,
        COMPLEXITY_HINTS,
        get_complexity_hint,
    )

This shim provides backward compatibility during the migration period.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

# Re-export from new framework locations for backward compatibility
from victor.framework.task import (
    DEFAULT_BUDGETS,
    PATTERNS,
    TASK_TYPE_TO_COMPLEXITY,
    TaskClassification,
    TaskComplexity,
    TaskComplexityService,
    classify_task,
    get_budget_for_task,
    is_action_task,
    is_analysis_task,
)
from victor.framework.enrichment.strategies import (
    COMPLEXITY_HINTS as PROMPT_HINTS,
    EXTENDED_COMPLEXITY_HINTS as EXTENDED_PROMPT_HINTS,
    ComplexityHintEnricher,
    get_complexity_hint as _get_complexity_hint,
)

# Emit deprecation warning on import
warnings.warn(
    "victor.agent.complexity_classifier is deprecated. "
    "Use victor.framework.task for classification and "
    "victor.framework.enrichment.strategies for prompt hints.",
    DeprecationWarning,
    stacklevel=2,
)


# Backward-compatible alias
ComplexityClassifier = TaskComplexityService


def get_prompt_hint(
    complexity: TaskComplexity, extended: bool = False, provider: Optional[str] = None
) -> str:
    """DEPRECATED: Get prompt hint for a complexity level.

    Use victor.framework.enrichment.strategies.get_complexity_hint instead.
    """
    return _get_complexity_hint(complexity, extended, provider)


def get_task_prompt_hint(message: str) -> str:
    """DEPRECATED: Get prompt hint for a task message.

    Use TaskComplexityService + ComplexityHintEnricher instead.
    """
    result = classify_task(message)
    return _get_complexity_hint(result.complexity)


def should_force_answer(message: str, tool_calls: int) -> Tuple[bool, str]:
    """DEPRECATED: Check if task should complete based on tool calls.

    Use TaskClassification.should_force_completion_after() instead.
    """
    c = classify_task(message)
    if c.should_force_completion_after(tool_calls):
        return (
            True,
            f"Task classified as {c.complexity.value} (budget: {c.tool_budget}, calls: {tool_calls})",
        )
    return (False, "")


# Re-export all for backward compatibility
__all__ = [
    # Core types (from framework/task)
    "TaskComplexity",
    "TaskClassification",
    "DEFAULT_BUDGETS",
    "PATTERNS",
    "TASK_TYPE_TO_COMPLEXITY",
    # Service (alias)
    "ComplexityClassifier",
    "TaskComplexityService",
    # Prompt hints (from enrichment)
    "PROMPT_HINTS",
    "EXTENDED_PROMPT_HINTS",
    "ComplexityHintEnricher",
    # Functions
    "classify_task",
    "get_budget_for_task",
    "get_prompt_hint",
    "get_task_prompt_hint",
    "should_force_answer",
    "is_action_task",
    "is_analysis_task",
]
