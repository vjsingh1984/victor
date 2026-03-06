"""Base task hints shared across all verticals.

Provides common hints for exploration, debugging, refactoring,
documentation, and review. Verticals override get_vertical_hints().
"""

from __future__ import annotations

from typing import Dict, Optional

from victor.core.vertical_types import TaskTypeHint

COMMON_HINTS: dict[str, TaskTypeHint] = {
    "exploration": TaskTypeHint(
        task_type="exploration",
        hint=(
            "[EXPLORE] Start with high-level overview. Read directory structure, "
            "key configuration files, and entry points before diving into specifics."
        ),
    ),
    "debugging": TaskTypeHint(
        task_type="debugging",
        hint=(
            "[DEBUG] PHASE 1 - Understand the error. PHASE 2 - Identify root cause "
            "via stack trace and relevant code. PHASE 3 - Fix with minimal changes. "
            "PHASE 4 - Verify the fix."
        ),
    ),
    "refactoring": TaskTypeHint(
        task_type="refactoring",
        hint=(
            "[REFACTOR] 1. Understand current behavior. 2. Ensure tests exist. "
            "3. Make incremental changes. 4. Run tests after each change."
        ),
    ),
    "documentation": TaskTypeHint(
        task_type="documentation",
        hint=(
            "[DOCS] Focus on WHY, not WHAT. Document intent, trade-offs, and "
            "non-obvious behavior. Keep examples runnable."
        ),
    ),
    "review": TaskTypeHint(
        task_type="review",
        hint=(
            "[REVIEW] Check for: correctness, edge cases, error handling, "
            "security implications, performance, and maintainability."
        ),
    ),
}


class BaseTaskHints:
    """Base task hints for all verticals.

    Verticals should subclass and override get_vertical_hints()
    to add domain-specific hints.
    """

    @classmethod
    def get_hint(cls, task_type: str) -> Optional[str]:
        """Get hint string for a task type, checking vertical hints first."""
        vertical_hints = cls.get_vertical_hints()
        if task_type in vertical_hints:
            return vertical_hints[task_type]
        hint_obj = COMMON_HINTS.get(task_type)
        return hint_obj.hint if hint_obj else None

    @classmethod
    def get_vertical_hints(cls) -> dict[str, str]:
        """Override in vertical subclass to add domain-specific hints."""
        return {}

    @classmethod
    def get_all_hints(cls) -> dict[str, str]:
        """Get all hints as string dict: common + vertical-specific."""
        result = {k: v.hint for k, v in COMMON_HINTS.items()}
        result.update(cls.get_vertical_hints())
        return result

    @classmethod
    def get_task_type_hints(cls) -> Dict[str, TaskTypeHint]:
        """Get all hints as TaskTypeHint objects (protocol-compatible)."""
        return dict(COMMON_HINTS)
