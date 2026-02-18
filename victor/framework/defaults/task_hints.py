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

"""Standard stage-tool mappings and budget scaling for verticals.

Provides ready-made ``stage_tools`` dicts that verticals can use when
defining ``TaskTypeDefinition`` entries, plus a ``scale_budget`` helper
for adjusting tool budgets by complexity::

    from victor.framework.defaults import (
        MODIFICATION_STAGE_TOOLS,
        scale_budget,
    )
"""

from __future__ import annotations

import math
from typing import Dict, List

from victor.framework.task_types import TaskCategory

# ---------------------------------------------------------------------------
# Stage-tool mappings by task category
# ---------------------------------------------------------------------------

MODIFICATION_STAGE_TOOLS: Dict[str, List[str]] = {
    "initial": ["list_directory", "code_search"],
    "reading": ["read_file", "code_search"],
    "executing": ["edit_files", "write_file"],
    "verifying": ["read_file", "run_tests"],
}
"""Stage tools for tasks that change files (edit, create, refactor)."""

ANALYSIS_STAGE_TOOLS: Dict[str, List[str]] = {
    "initial": ["list_directory", "code_search", "read_file"],
    "reading": ["read_file", "code_search", "list_directory"],
    "executing": ["read_file"],
    "verifying": ["read_file"],
}
"""Stage tools for tasks that read/explore code (analyze, search, review)."""

EXECUTION_STAGE_TOOLS: Dict[str, List[str]] = {
    "initial": ["list_directory", "code_search"],
    "reading": ["read_file", "code_search", "execute_bash"],
    "executing": ["execute_bash", "edit_files"],
    "verifying": ["read_file", "run_tests"],
}
"""Stage tools for tasks that run commands/processes (test, build, deploy)."""

_CATEGORY_TO_STAGE_TOOLS: Dict[str, Dict[str, List[str]]] = {
    TaskCategory.MODIFICATION.value: MODIFICATION_STAGE_TOOLS,
    TaskCategory.ANALYSIS.value: ANALYSIS_STAGE_TOOLS,
    TaskCategory.EXECUTION.value: EXECUTION_STAGE_TOOLS,
}


def get_stage_tools_for_category(category: str) -> Dict[str, List[str]]:
    """Get standard stage tools for a task category.

    Args:
        category: Task category name (``MODIFICATION``, ``ANALYSIS``,
            ``EXECUTION``).  Accepts both upper- and lower-case as well
            as ``TaskCategory`` enum values.

    Returns:
        Dict mapping stage names to tool lists.  Returns a *copy* so
        callers can mutate freely.  Falls back to ``ANALYSIS_STAGE_TOOLS``
        for unknown categories.
    """
    key = category.lower()
    mapping = _CATEGORY_TO_STAGE_TOOLS.get(key, ANALYSIS_STAGE_TOOLS)
    return {stage: list(tools) for stage, tools in mapping.items()}


# ---------------------------------------------------------------------------
# Tool budget scaling
# ---------------------------------------------------------------------------

DEFAULT_BUDGET_SCALING: Dict[str, float] = {
    "trivial": 0.5,
    "simple": 0.75,
    "moderate": 1.0,
    "complex": 1.5,
    "highly_complex": 2.5,
}
"""Multiplier applied to a base tool budget by complexity level."""


def scale_budget(base_budget: int, complexity: str) -> int:
    """Scale a base tool budget by complexity level.

    Args:
        base_budget: Base number of tool calls allowed.
        complexity: Complexity level (``trivial``, ``simple``, ``moderate``,
            ``complex``, ``highly_complex``).

    Returns:
        Scaled budget rounded to the nearest integer.  Defaults to
        a 1.0 multiplier for unknown complexity levels.
    """
    multiplier = DEFAULT_BUDGET_SCALING.get(complexity, 1.0)
    return max(1, math.ceil(base_budget * multiplier))


__all__ = [
    "MODIFICATION_STAGE_TOOLS",
    "ANALYSIS_STAGE_TOOLS",
    "EXECUTION_STAGE_TOOLS",
    "get_stage_tools_for_category",
    "DEFAULT_BUDGET_SCALING",
    "scale_budget",
]
