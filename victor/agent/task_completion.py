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

"""Task completion detection for agent workflow optimization.

This module provides mechanisms to detect when a task's objectives have been
achieved, preventing unnecessary continuation loops and improving efficiency.

Issue Reference: workflow-test-issues-v2.md Issue #1
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

logger = logging.getLogger(__name__)


class DeliverableType(Enum):
    """Types of deliverables an agent can produce."""

    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    PLAN_PROVIDED = "plan_provided"
    ANSWER_PROVIDED = "answer_provided"
    CODE_EXECUTED = "code_executed"
    ANALYSIS_PROVIDED = "analysis_provided"
    ERROR_REPORTED = "error_reported"


@dataclass
class TaskDeliverable:
    """Represents a completed deliverable."""

    type: DeliverableType
    description: str
    artifact_path: Optional[str] = None
    verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskCompletionState:
    """Tracks the completion state of a task."""

    expected_deliverables: List[DeliverableType] = field(default_factory=list)
    completed_deliverables: List[TaskDeliverable] = field(default_factory=list)
    completion_signals: Set[str] = field(default_factory=set)
    continuation_requests: int = 0
    max_continuation_requests: int = 2

    @property
    def is_complete(self) -> bool:
        """Check if all expected deliverables are met."""
        # If we've had too many continuation requests, force completion
        if self.continuation_requests >= self.max_continuation_requests:
            return True

        # If no expected deliverables, check for completion signals
        if not self.expected_deliverables:
            return bool(self.completion_signals) or bool(self.completed_deliverables)

        # Check if all expected types are in completed
        completed_types = {d.type for d in self.completed_deliverables}
        return all(dt in completed_types for dt in self.expected_deliverables)

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if not self.expected_deliverables:
            return 100.0 if self.is_complete else 0.0

        completed_types = {d.type for d in self.completed_deliverables}
        matched = sum(1 for dt in self.expected_deliverables if dt in completed_types)
        return (matched / len(self.expected_deliverables)) * 100


@runtime_checkable
class ITaskCompletionDetector(Protocol):
    """Protocol for task completion detection."""

    def analyze_intent(self, user_message: str) -> List[DeliverableType]:
        """Infer expected deliverables from user request."""
        ...

    def record_tool_result(self, tool_name: str, result: Dict[str, Any]) -> None:
        """Record deliverables from tool execution."""
        ...

    def analyze_response(self, response_text: str) -> None:
        """Detect completion signals in response text."""
        ...

    def should_stop(self) -> bool:
        """Check if task objectives are met and agent should stop."""
        ...

    def get_completion_summary(self) -> str:
        """Generate summary of completed deliverables."""
        ...

    def reset(self) -> None:
        """Reset state for new task."""
        ...


class TaskCompletionDetector:
    """Detects when a task's objectives have been achieved.

    This detector analyzes:
    1. User intent to determine expected deliverables
    2. Tool results to track completed deliverables
    3. Response text for completion signals
    4. Continuation request patterns to detect loops

    Usage:
        detector = TaskCompletionDetector()
        detector.analyze_intent("Create a cache_manager.py file")
        # ... agent executes tools ...
        detector.record_tool_result("write", {"success": True, "path": "cache.py"})
        if detector.should_stop():
            print(detector.get_completion_summary())
    """

    # Phrases indicating task completion
    COMPLETION_PHRASES: frozenset = frozenset({
        # File operations
        "successfully created",
        "has been created",
        "file created",
        "has been implemented",
        "implementation complete",
        "successfully written",
        "has been written",
        "file saved",
        # Task completion
        "task complete",
        "task completed",
        "done",
        "finished",
        "completed successfully",
        # Delivery phrases
        "here's the",
        "here is the",
        "i've created",
        "i have created",
        "the file is now",
        "the implementation is",
        # Summary phrases
        "in summary",
        "to summarize",
        "that covers",
        "this completes",
    })

    # Phrases indicating continuation loop (agent asking to continue)
    CONTINUATION_PHRASES: frozenset = frozenset({
        "continue with the implementation",
        "what would you like me to",
        "how would you like to proceed",
        "if you need further",
        "let me know if you",
        "would you like me to",
        "please specify",
        "please provide more details",
        "if there's anything else",
        "any other",
    })

    # Tools that produce file deliverables
    WRITE_TOOLS: frozenset = frozenset({
        "write",
        "edit",
        "create_file",
        "write_file",
        "save_file",
        "create",
    })

    # Tools that execute code
    EXECUTE_TOOLS: frozenset = frozenset({
        "execute_bash",
        "bash",
        "run_command",
        "execute_code",
        "run_python",
    })

    def __init__(self):
        """Initialize the task completion detector."""
        self._state = TaskCompletionState()
        self._intent_keywords: Dict[str, List[DeliverableType]] = {
            # File creation keywords
            "create": [DeliverableType.FILE_CREATED],
            "add": [DeliverableType.FILE_CREATED],
            "write": [DeliverableType.FILE_CREATED],
            "implement": [DeliverableType.FILE_CREATED, DeliverableType.CODE_EXECUTED],
            "build": [DeliverableType.FILE_CREATED],
            # File modification keywords
            "update": [DeliverableType.FILE_MODIFIED],
            "modify": [DeliverableType.FILE_MODIFIED],
            "change": [DeliverableType.FILE_MODIFIED],
            "fix": [DeliverableType.FILE_MODIFIED],
            "refactor": [DeliverableType.FILE_MODIFIED],
            # Planning keywords
            "plan": [DeliverableType.PLAN_PROVIDED],
            "design": [DeliverableType.PLAN_PROVIDED],
            "how to": [DeliverableType.PLAN_PROVIDED],
            "strategy": [DeliverableType.PLAN_PROVIDED],
            # Analysis keywords
            "what": [DeliverableType.ANSWER_PROVIDED],
            "explain": [DeliverableType.ANSWER_PROVIDED],
            "describe": [DeliverableType.ANSWER_PROVIDED],
            "analyze": [DeliverableType.ANALYSIS_PROVIDED],
            "review": [DeliverableType.ANALYSIS_PROVIDED],
            # Execution keywords
            "run": [DeliverableType.CODE_EXECUTED],
            "execute": [DeliverableType.CODE_EXECUTED],
            "test": [DeliverableType.CODE_EXECUTED],
        }

    def analyze_intent(self, user_message: str) -> List[DeliverableType]:
        """Infer expected deliverables from user request.

        Args:
            user_message: The user's request message

        Returns:
            List of expected deliverable types
        """
        message_lower = user_message.lower()
        deliverables: Set[DeliverableType] = set()

        # Check for file extension mentions (strong signal for file creation)
        if re.search(r'\.\w{2,4}\b', user_message):  # .py, .js, .yaml, etc.
            deliverables.add(DeliverableType.FILE_CREATED)

        # Check for intent keywords
        for keyword, types in self._intent_keywords.items():
            if keyword in message_lower:
                deliverables.update(types)

        # Store expected deliverables
        self._state.expected_deliverables = list(deliverables)

        logger.debug(f"Analyzed intent, expecting: {self._state.expected_deliverables}")
        return self._state.expected_deliverables

    def record_tool_result(self, tool_name: str, result: Dict[str, Any]) -> None:
        """Record deliverables from tool execution.

        Args:
            tool_name: Name of the tool that was executed
            result: Result dictionary from tool execution
        """
        tool_lower = tool_name.lower()
        success = result.get("success", True)  # Assume success if not specified

        if not success:
            return

        # Check for file write operations
        if tool_lower in self.WRITE_TOOLS:
            path = result.get("path", result.get("file_path", "unknown"))
            deliverable = TaskDeliverable(
                type=DeliverableType.FILE_CREATED,
                description=f"Created/modified {path}",
                artifact_path=path,
                verified=True,
                metadata={"tool": tool_name},
            )
            self._state.completed_deliverables.append(deliverable)
            logger.debug(f"Recorded file deliverable: {path}")

        # Check for code execution
        elif tool_lower in self.EXECUTE_TOOLS:
            deliverable = TaskDeliverable(
                type=DeliverableType.CODE_EXECUTED,
                description=f"Executed {result.get('command', 'code')[:50]}",
                verified=True,
                metadata={"tool": tool_name, "exit_code": result.get("exit_code", 0)},
            )
            self._state.completed_deliverables.append(deliverable)
            logger.debug("Recorded code execution deliverable")

    def analyze_response(self, response_text: str) -> None:
        """Detect completion signals in response text.

        Args:
            response_text: The agent's response text
        """
        response_lower = response_text.lower()

        # Check for completion phrases
        for phrase in self.COMPLETION_PHRASES:
            if phrase in response_lower:
                self._state.completion_signals.add(phrase)
                logger.debug(f"Detected completion signal: {phrase}")

        # Check for continuation loop patterns
        for phrase in self.CONTINUATION_PHRASES:
            if phrase in response_lower:
                self._state.continuation_requests += 1
                logger.debug(
                    f"Detected continuation request ({self._state.continuation_requests}): {phrase}"
                )
                break

        # Infer deliverables from response content
        if not self._state.expected_deliverables:
            # If we see plan-like content, record as plan provided
            if any(
                marker in response_lower
                for marker in ["steps:", "step 1", "## implementation", "here's how"]
            ):
                self._state.completed_deliverables.append(
                    TaskDeliverable(
                        type=DeliverableType.PLAN_PROVIDED,
                        description="Implementation plan provided",
                        verified=True,
                    )
                )

            # If we see explanation content
            if any(
                marker in response_lower
                for marker in ["this file", "the function", "it handles", "this class"]
            ):
                self._state.completed_deliverables.append(
                    TaskDeliverable(
                        type=DeliverableType.ANSWER_PROVIDED,
                        description="Explanation provided",
                        verified=True,
                    )
                )

    def should_stop(self) -> bool:
        """Check if task objectives are met and agent should stop.

        Returns:
            True if agent should stop, False otherwise
        """
        is_complete = self._state.is_complete

        if is_complete:
            logger.info(
                f"Task completion detected: {len(self._state.completed_deliverables)} deliverables, "
                f"{len(self._state.completion_signals)} signals, "
                f"{self._state.continuation_requests} continuation requests"
            )

        return is_complete

    def get_completion_summary(self) -> str:
        """Generate summary of completed deliverables.

        Returns:
            Markdown-formatted summary string
        """
        if not self._state.completed_deliverables and not self._state.completion_signals:
            return "No deliverables recorded."

        lines = ["## Task Completion Summary", ""]

        if self._state.completed_deliverables:
            lines.append("### Deliverables")
            for d in self._state.completed_deliverables:
                status = "✅" if d.verified else "⏳"
                path_info = f" ({d.artifact_path})" if d.artifact_path else ""
                lines.append(f"- {status} {d.type.value}: {d.description}{path_info}")
            lines.append("")

        if self._state.completion_signals:
            lines.append(f"### Completion Signals: {len(self._state.completion_signals)}")
            lines.append("")

        lines.append(f"**Completion: {self._state.completion_percentage:.0f}%**")

        return "\n".join(lines)

    def get_state(self) -> TaskCompletionState:
        """Get the current completion state.

        Returns:
            Current TaskCompletionState
        """
        return self._state

    def reset(self) -> None:
        """Reset state for new task."""
        self._state = TaskCompletionState()
        logger.debug("Task completion detector reset")

    def force_complete(self, reason: str) -> None:
        """Force task completion with a reason.

        Args:
            reason: Reason for forcing completion
        """
        self._state.completion_signals.add(f"forced:{reason}")
        logger.info(f"Forced task completion: {reason}")


def create_task_completion_detector() -> TaskCompletionDetector:
    """Factory function for creating TaskCompletionDetector.

    Returns:
        Configured TaskCompletionDetector instance
    """
    return TaskCompletionDetector()
