# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Task-specific context reminder templates (P1 refinement).

Provides context-aware continuation prompts after compaction, improving
task continuity based on OpenDev research findings.

Key insights from OpenDev (arXiv:2603.05344):
- Short, targeted reminders at maximum recency are most effective
- Use user role for reminders (higher salience than system role)
- Reminder frequency must be capped to avoid becoming background noise
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TaskReminderTemplates:
    """Container for task-specific reminder templates.

    This class provides a centralized location for managing reminder
    templates used for post-compaction continuation.
    """

    def __init__(self, custom_templates: Optional[Dict[str, str]] = None):
        """Initialize with optional custom templates.

        Args:
            custom_templates: Optional dict of custom templates to add/override defaults
        """
        self._templates = TASK_TEMPLATES.copy()
        if custom_templates:
            self._templates.update(custom_templates)

    def get_template(self, task_type: str) -> str:
        """Get template for a task type.

        Args:
            task_type: Type of task (analysis, implementation, etc.)

        Returns:
            Template string, or default template if not found
        """
        return self._templates.get(task_type.lower(), TASK_TEMPLATES["default"])

    def add_template(self, task_type: str, template: str) -> None:
        """Add or update a template.

        Args:
            task_type: Task type identifier
            template: Template string
        """
        self._templates[task_type.lower()] = template

    def list_task_types(self) -> list[str]:
        """List available task types.

        Returns:
            List of task type identifiers
        """
        return list(self._templates.keys())


# Task-specific reminder templates
# Format: "You were {task_action}. Continue {guidance}"
TASK_TEMPLATES: Dict[str, str] = {
    "analysis": (
        "You were analyzing code/files. Continue your investigation — "
        "use tools to gather more information or provide findings."
    ),
    "implementation": (
        "You were implementing a feature. Continue coding — "
        "use tools to complete the implementation."
    ),
    "debugging": (
        "You were debugging an issue. Continue troubleshooting — "
        "use tools to investigate and fix the problem."
    ),
    "coding": (
        "You were working on a coding task. Continue — "
        "use tools to read, edit, or test code as needed."
    ),
    "search": (
        "You were searching for information. Continue — "
        "use tools to find what you need."
    ),
    "refactoring": (
        "You were refactoring code. Continue — "
        "use tools to modify and improve the code structure."
    ),
    "testing": (
        "You were running tests. Continue — "
        "use tools to execute tests and check results."
    ),
    # Default fallback
    "default": (
        "Continue with what you were doing — "
        "use tools to gather information or complete your task."
    ),
}


# Post-compaction reminder templates
# These are more detailed and include compaction context
POST_COMPACTION_TEMPLATES: Dict[str, str] = {
    "analysis": (
        "Context was compacted to continue the session. "
        "You were analyzing code/files. Continue your investigation — "
        "use tools to gather more information or provide findings."
    ),
    "implementation": (
        "Context was compacted. You were implementing a feature. "
        "Continue coding — use tools to complete the implementation."
    ),
    "debugging": (
        "Context was compacted. You were debugging an issue. "
        "Continue troubleshooting — use tools to investigate and fix."
    ),
    "coding": (
        "Context was compacted. You were working on code. "
        "Continue — use tools to read, edit, or test as needed."
    ),
    "search": (
        "Context was compacted. You were searching. "
        "Continue — use tools to find what you need."
    ),
    "default": (
        "Context was compacted. Continue your task — "
        "use tools to gather information or complete your work."
    ),
}


def get_reminder_for_task(
    task_type: str,
    context: Optional[Dict[str, str]] = None,
) -> str:
    """Get a task-specific reminder for continuation.

    Args:
        task_type: Type of task (analysis, implementation, etc.)
        context: Optional context variables to format into template

    Returns:
        Formatted reminder string
    """
    template = TASK_TEMPLATES.get(task_type.lower(), TASK_TEMPLATES["default"])

    # If context provided and template has placeholders, try to format
    if context and "{" in template:
        try:
            # Only format if we have values for the placeholders
            available_vars = {k: v for k, v in context.items() if f"{{{k}}}" in template}
            if available_vars:
                return template.format(**available_vars)
        except (KeyError, ValueError) as e:
            logger.debug(f"Could not format reminder template: {e}")

    return template


def get_post_compaction_reminder(
    task_type: str,
    compaction_summary: str = "",
    messages_removed: int = 0,
) -> str:
    """Get a post-compaction reminder with task-specific guidance.

    Based on OpenDev findings:
    - Reminders should be short and targeted
    - Use user role for higher salience
    - Include compaction context when relevant

    Args:
        task_type: Type of task being performed
        compaction_summary: Summary of what was compacted
        messages_removed: Number of messages removed (for severity)

    Returns:
        Formatted post-compaction reminder
    """
    base_template = POST_COMPACTION_TEMPLATES.get(
        task_type.lower(),
        POST_COMPACTION_TEMPLATES["default"]
    )

    # Add compaction context if significant
    if messages_removed > 20:
        # High severity - add more detail
        if compaction_summary:
            return (
                f"{base_template} "
                f"(Previous context: {compaction_summary[:100]}...)"
            )
        else:
            return (
                f"{base_template} "
                f"(Previous context was compacted: {messages_removed} messages removed)"
            )
    elif messages_removed > 5:
        # Medium severity - brief mention
        if compaction_summary:
            return (
                f"{base_template} "
                f"(Context: {compaction_summary[:50]}...)"
            )
        else:
            return f"{base_template} (Context was compacted)"

    return base_template


def get_compaction_summary_reminder(
    compaction_summary: str,
    messages_removed: int,
    current_task: str = "",
) -> str:
    """Get a focused reminder about what was compacted.

    This is used in the post-compaction continuation logic to remind
    the model of what was lost during compaction.

    Args:
        compaction_summary: Summary from compaction
        messages_removed: Number of messages removed
        current_task: Optional current task description

    Returns:
        Brief reminder string for user-role injection
    """
    if not compaction_summary and messages_removed == 0:
        return ""

    if messages_removed > 50:
        # Large compaction - include summary
        if current_task:
            return (
                f"[Context was compacted to continue. "
                f"You were working on: {current_task}. "
                f"Previous: {compaction_summary[:80]}...]"
            )
        return (
            f"[Context was compacted to continue. "
            f"Previous: {compaction_summary[:80]}...]"
        )
    elif messages_removed > 10:
        # Medium compaction
        if current_task:
            return (
                f"[Context compacted. You were: {current_task}. "
                f"Previous: {compaction_summary[:50]}...]"
            )
        return (
            f"[Context compacted. Previous: {compaction_summary[:50]}...]"
        )
    else:
        # Small compaction - minimal reminder
        if current_task:
            return (
                f"[Context compacted. You were: {current_task}. "
                f"Continue what you were doing.]"
            )
        return "[Context compacted. Continue what you were doing.]"


# Reminder frequency caps (per OpenDev: reminders must be capped)
# These define maximum frequency per reminder type to avoid noise
REMINDER_FREQUENCY_CAPS: Dict[str, int] = {
    "compaction": 1,  # Only remind once per compaction event
    "budget": 2,  # Remind about budget every 2 tool calls
    "progress": 3,  # Progress reminders every 3 tool calls
    "evidence": 5,  # Evidence reminders every 5 tool calls
    "task_hint": 1,  # Task hints only once per turn
    "grounding": 2,  # Grounding reminders every 2 tool calls
}


def get_max_frequency_for_reminder(reminder_type: str) -> int:
    """Get maximum frequency cap for a reminder type.

    Args:
        reminder_type: Type of reminder (compaction, budget, etc.)

    Returns:
        Maximum frequency (minimum calls between reminders)
    """
    return REMINDER_FREQUENCY_CAPS.get(reminder_type.lower(), 1)


def should_inject_reminder(
    reminder_type: str,
    calls_since_last: int,
    max_frequency: Optional[int] = None,
) -> bool:
    """Check if a reminder should be injected based on frequency capping.

    Args:
        reminder_type: Type of reminder
        calls_since_last: Tool calls since last injection
        max_frequency: Optional override frequency

    Returns:
        True if reminder should be injected
    """
    freq = max_frequency or get_max_frequency_for_reminder(reminder_type)
    return calls_since_last >= freq
