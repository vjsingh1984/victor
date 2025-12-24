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

"""Centralized constants for orchestrator components.

This module provides dataclasses containing magic numbers and thresholds
used throughout the Victor framework. Centralizing these values makes the
codebase more maintainable and easier to tune.

**Design Rationale:**
Previously, threshold values were scattered across multiple files (context_compactor.py,
conversation_controller.py, etc.), making it difficult to understand and tune the system.
This module provides a single source of truth for all orchestrator-related constants.

**Usage:**
    from victor.config.orchestrator_constants import (
        CONTEXT_LIMITS,
        SEMANTIC_THRESHOLDS,
        BUDGET_LIMITS,
        TOOL_SELECTION_PRESETS,
    )

    # Check context overflow
    if utilization >= CONTEXT_LIMITS.overflow_threshold:
        compact_history()

    # Filter by semantic similarity
    if similarity >= SEMANTIC_THRESHOLDS.tool_selection:
        include_tool()

**Note:** These constants are defaults. Most can be overridden via Settings or ProfileConfig.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ContextLimits:
    """Context window utilization thresholds.

    These thresholds control when context compaction and warnings are triggered.

    Attributes:
        overflow_threshold: Utilization % to trigger overflow warnings (0.0-1.0)
        warning_threshold: Utilization % to show warnings in UI (0.0-1.0)
        proactive_compaction_threshold: Utilization % to trigger proactive compaction (0.0-1.0)
        critical_threshold: Utilization % indicating critical overflow risk (0.0-1.0)
        compaction_target: Target utilization after compaction (0.0-1.0)
        max_context_chars: Default maximum context size in characters
        chars_per_token_estimate: Approximate characters per token for estimation

    **Design Notes:**
    - overflow_threshold (0.8) provides headroom before hitting model limits
    - proactive_compaction_threshold (0.90) triggers compaction before overflow
    - compaction_target (0.5) reduces to 50% to allow for conversation growth
    """

    overflow_threshold: float = 0.8
    warning_threshold: float = 0.7
    proactive_compaction_threshold: float = 0.90
    critical_threshold: float = 0.95
    compaction_target: float = 0.5
    max_context_chars: int = 200000
    chars_per_token_estimate: int = 4


@dataclass(frozen=True)
class SemanticThresholds:
    """Semantic similarity thresholds for various operations.

    These thresholds control when content is considered "similar enough" for
    different semantic operations throughout the framework.

    Attributes:
        tool_selection: Minimum similarity to include a tool in the selection
        code_search: Minimum similarity for code search results
        history_retrieval: Minimum similarity for retrieving historical messages
        compaction_relevance: Minimum similarity for message retention during compaction

    **Design Notes:**
    - tool_selection (0.3) is lenient to avoid missing useful tools
    - code_search (0.5) is stricter to reduce false positives
    - history_retrieval (0.4) balances context preservation with relevance
    - compaction_relevance (0.3) matches conversation_controller.py
    """

    tool_selection: float = 0.3
    code_search: float = 0.5
    history_retrieval: float = 0.4
    compaction_relevance: float = 0.3


@dataclass(frozen=True)
class BudgetLimits:
    """Tool budget limits by task complexity.

    These budgets control how many tool calls are allowed based on the
    detected task type. Higher budgets allow more exploratory behavior.

    Attributes:
        simple_task: Budget for simple tasks (quick queries, single file edits)
        medium_task: Budget for medium tasks (multi-file analysis, refactoring)
        complex_task: Budget for complex tasks (architecture changes, migrations)
        action_task: Budget for action-heavy tasks (building features, large refactors)
        analysis_task: Budget for analysis-heavy tasks (codebase exploration, audits)
        max_session_budget: Maximum tool calls per session (hard limit)
        warning_threshold_pct: % of budget used to trigger warnings (0.0-1.0)

    **Design Notes:**
    - Budgets scale exponentially with complexity (2 → 6 → 15 → 50 → 60)
    - action_task (50) allows building non-trivial features
    - analysis_task (60) supports deep codebase exploration
    - max_session_budget (300) prevents runaway tool usage
    - warning_threshold_pct (0.83) gives early warning at 250/300 calls
    """

    simple_task: int = 2
    medium_task: int = 6
    complex_task: int = 15
    action_task: int = 50
    analysis_task: int = 60
    max_session_budget: int = 300
    warning_threshold_pct: float = 0.83  # 250/300


@dataclass(frozen=True)
class CompactionConfig:
    """Configuration for context compaction strategies.

    Attributes:
        min_messages_after_compact: Minimum messages to keep after compaction
        tool_result_retention_weight: Weight multiplier for tool results (higher = keep longer)
        recent_message_weight: Weight multiplier for recent messages (higher = keep longer)
        tool_result_max_chars: Maximum characters for tool results before truncation
        tool_result_max_lines: Maximum lines for tool results before truncation
        output_reserve_pct: % of context to reserve for output generation (0.0-1.0)
        parallel_read_target_files: Target number of parallel file reads to optimize for
        chars_per_token: Approximate characters per token for calculations

    **Design Notes:**
    - min_messages_after_compact (8) preserves enough context for coherent conversation
    - tool_result_retention_weight (1.5) prioritizes tool results over discussion
    - recent_message_weight (2.0) heavily weights recent messages
    - tool_result_max_chars (8192) allows ~10-12 parallel reads within 32K usable tokens
    - output_reserve_pct (0.5) reserves half the context for model output
    - parallel_read_target_files (10) optimizes for common read patterns
    """

    min_messages_after_compact: int = 8
    tool_result_retention_weight: float = 1.5
    recent_message_weight: float = 2.0
    tool_result_max_chars: int = 8192
    tool_result_max_lines: int = 230  # 8192 / 35 chars per line ≈ 234 lines
    output_reserve_pct: float = 0.5
    parallel_read_target_files: int = 10
    chars_per_token: float = 3.0


@dataclass(frozen=True)
class ToolSelectionPresets:
    """Presets for adaptive tool selection by model size.

    These presets configure semantic tool selection thresholds based on
    model size/capability. Smaller models get stricter filtering to avoid
    overwhelming them with tool options.

    **Design Notes:**
    - tiny (0.5B-3B): Strict filtering, minimal tools to avoid confusion
    - small (7B-8B): Moderate filtering, suitable for most tasks
    - medium (13B-15B): Lenient filtering, can handle more tools
    - large (30B+): Very lenient, can handle complex tool sets
    - cloud (Claude/GPT): Optimized for commercial models
    """

    tiny: Dict[str, float] = None  # Will be set in __post_init__
    small: Dict[str, float] = None
    medium: Dict[str, float] = None
    large: Dict[str, float] = None
    cloud: Dict[str, float] = None

    def __post_init__(self):
        # Use object.__setattr__ since dataclass is frozen
        object.__setattr__(self, 'tiny', {
            'base_threshold': 0.35,
            'base_max_tools': 5,
        })
        object.__setattr__(self, 'small', {
            'base_threshold': 0.25,
            'base_max_tools': 7,
        })
        object.__setattr__(self, 'medium', {
            'base_threshold': 0.20,
            'base_max_tools': 10,
        })
        object.__setattr__(self, 'large', {
            'base_threshold': 0.15,
            'base_max_tools': 12,
        })
        object.__setattr__(self, 'cloud', {
            'base_threshold': 0.18,
            'base_max_tools': 10,
        })


# ============================================================================
# Singleton instances (read-only)
# ============================================================================
# These are the default values used throughout Victor. Import these instances
# directly rather than creating new ones.

CONTEXT_LIMITS = ContextLimits()
SEMANTIC_THRESHOLDS = SemanticThresholds()
BUDGET_LIMITS = BudgetLimits()
COMPACTION_CONFIG = CompactionConfig()
TOOL_SELECTION_PRESETS = ToolSelectionPresets()


# ============================================================================
# Convenience functions
# ============================================================================

def get_budget_for_task(task_type: str) -> int:
    """Get recommended tool budget for a task type.

    Args:
        task_type: Task type (simple, medium, complex, action, analysis)

    Returns:
        Recommended tool budget

    Raises:
        ValueError: If task_type is not recognized
    """
    task_type_lower = task_type.lower()

    if task_type_lower in ('simple', 'quick', 'trivial'):
        return BUDGET_LIMITS.simple_task
    elif task_type_lower in ('medium', 'moderate', 'standard'):
        return BUDGET_LIMITS.medium_task
    elif task_type_lower in ('complex', 'difficult', 'advanced'):
        return BUDGET_LIMITS.complex_task
    elif task_type_lower in ('action', 'build', 'implement', 'feature'):
        return BUDGET_LIMITS.action_task
    elif task_type_lower in ('analysis', 'explore', 'research', 'audit'):
        return BUDGET_LIMITS.analysis_task
    else:
        raise ValueError(
            f"Unknown task type: {task_type}. "
            f"Valid types: simple, medium, complex, action, analysis"
        )


def is_budget_warning_threshold(used: int, total: int) -> bool:
    """Check if tool budget usage has reached warning threshold.

    Args:
        used: Number of tool calls used
        total: Total tool budget

    Returns:
        True if warning threshold reached
    """
    if total <= 0:
        return False
    utilization = used / total
    return utilization >= BUDGET_LIMITS.warning_threshold_pct


def should_compact_context(utilization: float, proactive: bool = True) -> bool:
    """Determine if context should be compacted based on utilization.

    Args:
        utilization: Current context utilization (0.0-1.0)
        proactive: If True, use proactive threshold; otherwise use overflow threshold

    Returns:
        True if compaction should be triggered
    """
    if proactive:
        return utilization >= CONTEXT_LIMITS.proactive_compaction_threshold
    else:
        return utilization >= CONTEXT_LIMITS.overflow_threshold
