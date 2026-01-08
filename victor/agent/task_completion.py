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


class ResponsePhase(Enum):
    """Phases of agent response for completion detection."""

    EXPLORATION = "exploration"  # Reading files, searching codebase
    SYNTHESIS = "synthesis"  # Summarizing, planning, preparing output
    FINAL_OUTPUT = "final_output"  # Delivering answer, completed work
    BLOCKED = "blocked"  # Cannot complete, needs user input


class CompletionConfidence(Enum):
    """Confidence levels for task completion detection."""

    HIGH = "high"  # Active signal detected (_DONE_, _TASK_DONE_) - deterministic
    MEDIUM = "medium"  # File modifications + passive signal
    LOW = "low"  # Only passive phrase detected
    NONE = "none"  # No completion signal detected


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
    """Tracks the completion state of a task.

    Note: max_continuation_requests defaults to MEDIUM complexity budget.
    Use configure_for_complexity() to set based on task complexity.
    """

    expected_deliverables: List[DeliverableType] = field(default_factory=list)
    completed_deliverables: List[TaskDeliverable] = field(default_factory=list)
    completion_signals: Set[str] = field(default_factory=set)
    continuation_requests: int = 0
    # Default to MEDIUM complexity budget (5) - was 2 which caused premature termination
    max_continuation_requests: int = 5
    # Active signal detection flag - set when explicit completion signal detected
    active_signal_detected: bool = False

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

    This detector uses a priority-based approach:
    1. ACTIVE SIGNALS (Priority 1): Explicit completion phrases instructed in system prompt
    2. TOOL EVIDENCE (Priority 2): File modifications recorded from tool results
    3. PASSIVE PHRASES (Priority 3): Fallback phrase detection for models that ignore instructions

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

    # Priority 1: Active signals - deterministic, instructed in system prompt
    # These are checked first and if detected, immediately signal completion
    # Using underscore-prefixed signals to avoid confusion with natural language
    ACTIVE_SIGNALS: frozenset = frozenset(
        {
            "_task_done_",
            "_done_",
            "_summary_",
            "_blocked_",
            "_cannot_complete_",
            # Also accept natural language variants for models that don't follow exactly
            "task complete:",
            "done:",
            "summary:",
        }
    )

    # Priority 3: Passive phrases indicating task completion (fallback)
    COMPLETION_PHRASES: frozenset = frozenset(
        {
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
            # Bug fix / SWE-bench completion phrases
            "fix applied",
            "fix has been applied",
            "bug fixed",
            "bug has been fixed",
            "issue resolved",
            "issue has been resolved",
            "the fix is",
            "i've fixed",
            "i have fixed",
            "successfully fixed",
            "has been fixed",
            "patch applied",
            "change applied",
            "modification complete",
            "edit complete",
            # Active completion signaling (instructed in system prompt)
            "task complete:",
            "task complete.",
            # Additional patterns for broader LLM coverage
            "has been modified",
            "has been updated",
            "has been changed",
            "modification is complete",
            "update is complete",
            "changes have been made",
            "file updated",
            "code updated",
            "the changes are",
            "i've updated",
            "i have updated",
            "i've modified",
            "i have modified",
        }
    )

    # Phrases indicating continuation loop (agent asking to continue)
    CONTINUATION_PHRASES: frozenset = frozenset(
        {
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
            # Additional patterns for broader LLM coverage
            "shall i proceed",
            "do you want me to",
            "should i continue",
            "need me to",
            "want me to add",
            "want me to make",
            "anything else you'd like",
            "any changes you'd like",
            "any modifications",
        }
    )

    # Tools that produce file deliverables
    WRITE_TOOLS: frozenset = frozenset(
        {
            "write",
            "edit",
            "edit_file",
            "create_file",
            "write_file",
            "save_file",
            "create",
            "file_edit",
            "modify_file",
        }
    )

    # Tools that execute code
    EXECUTE_TOOLS: frozenset = frozenset(
        {
            "execute_bash",
            "bash",
            "run_command",
            "execute_code",
            "run_python",
        }
    )

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
        if re.search(r"\.\w{2,4}\b", user_message):  # .py, .js, .yaml, etc.
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
        """Detect completion signals in response text with priority ordering.

        Priority:
        1. ACTIVE SIGNALS: Explicit completion signals instructed in system prompt
        2. PASSIVE PHRASES: Fallback detection for models ignoring instructions

        Args:
            response_text: The agent's response text
        """
        response_lower = response_text.lower()

        # Priority 1: Check active signals first (deterministic, instructed)
        for signal in self.ACTIVE_SIGNALS:
            if signal in response_lower:
                self._state.completion_signals.add(f"active:{signal}")
                self._state.active_signal_detected = True
                logger.info(f"Active completion signal detected: {signal}")
                # Active signal is definitive - skip passive detection
                return

        # Priority 3: Passive phrase detection (fallback)
        for phrase in self.COMPLETION_PHRASES:
            if phrase in response_lower:
                self._state.completion_signals.add(f"passive:{phrase}")
                logger.debug(f"Passive completion signal: {phrase}")

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

        Uses priority-based completion detection:
        1. ACTIVE SIGNAL: Explicit completion signal detected (deterministic)
        2. FILE MODS + SIGNAL: File edits + any completion signal
        3. FILE MODS + CONTINUATION: File edits + continuation requests
        4. STANDARD: All expected deliverables met

        Returns:
            True if agent should stop, False otherwise
        """
        # Priority 1: Active signal detected (deterministic, highest confidence)
        if self._state.active_signal_detected:
            logger.info("Stopping: Active completion signal detected")
            return True

        is_complete = self._state.is_complete

        # Priority 2: File modifications + signal (existing SWE-bench logic)
        # For bug_fix tasks: if we've made file edits, that's a strong completion signal
        # This handles SWE-bench style tasks where the agent edits but doesn't say "done"
        if not is_complete and self._has_file_modifications():
            # If we have file edits + any completion signal, we're done
            if self._state.completion_signals:
                is_complete = True
                logger.info("Bug fix completion: file edits + completion signal detected")
            # If we have file edits + 2+ continuation requests, likely done
            elif self._state.continuation_requests >= 2:
                is_complete = True
                logger.info("Bug fix completion: file edits + continuation requests detected")

        if is_complete:
            logger.info(
                f"Task completion detected: {len(self._state.completed_deliverables)} deliverables, "
                f"{len(self._state.completion_signals)} signals, "
                f"{self._state.continuation_requests} continuation requests"
            )

        return is_complete

    def _has_file_modifications(self) -> bool:
        """Check if any file modification deliverables have been recorded.

        Returns:
            True if file edits/creations detected
        """
        file_types = {DeliverableType.FILE_CREATED, DeliverableType.FILE_MODIFIED}
        return any(d.type in file_types for d in self._state.completed_deliverables)

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

    def configure_for_complexity(self, complexity: str) -> None:
        """Configure completion limits based on task complexity.

        Args:
            complexity: Complexity level (simple, medium, complex, generation, action, analysis)

        Uses ComplexityBudget from victor.framework.task for consolidated limits.
        """
        try:
            from victor.framework.task.complexity import ComplexityBudget
            from victor.framework.task.protocols import TaskComplexity

            complexity_enum = TaskComplexity(complexity.lower())
            budget = ComplexityBudget.for_complexity(complexity_enum)
            self._state.max_continuation_requests = budget.max_continuation_requests
            logger.debug(
                f"Configured completion for {complexity}: "
                f"max_continuation_requests={budget.max_continuation_requests}"
            )
        except (ValueError, ImportError) as e:
            logger.warning(f"Could not configure for complexity '{complexity}': {e}")

    def force_complete(self, reason: str) -> None:
        """Force task completion with a reason.

        Args:
            reason: Reason for forcing completion
        """
        self._state.completion_signals.add(f"forced:{reason}")
        logger.info(f"Forced task completion: {reason}")

    def detect_response_phase(self, response_text: str) -> ResponsePhase:
        """Detect the current phase of the agent's response.

        This helps distinguish between "thinking" (exploration/synthesis) and "output" (final delivery).

        Args:
            response_text: The agent's response text

        Returns:
            Detected ResponsePhase
        """
        response_lower = response_text.lower()

        # Check for blocked signals first
        blocked_signals = [
            "_blocked_",
            "_cannot_complete_",
            "i cannot",
            "i'm unable to",
            "unable to complete",
        ]
        if any(signal in response_lower for signal in blocked_signals):
            return ResponsePhase.BLOCKED

        # Check for final output signals (active completion signals + delivery phrases)
        final_output_indicators = [
            "_done_",
            "_task_done_",
            "_summary_",
            "task complete:",
            "done:",
            "summary:",
            "here's the",
            "here is the",
            "i've created",
            "i have created",
            "the file is now",
            "the implementation is",
            "successfully created",
        ]
        if any(indicator in response_lower for indicator in final_output_indicators):
            return ResponsePhase.FINAL_OUTPUT

        # Check for synthesis phase (summarizing, preparing output)
        synthesis_indicators = [
            "in summary",
            "to summarize",
            "in conclusion",
            "let me summarize",
            "here's what i found",
            "here is what i found",
            "based on my analysis",
            "after reviewing",
            "having examined",
        ]
        if any(indicator in response_lower for indicator in synthesis_indicators):
            return ResponsePhase.SYNTHESIS

        # Check for exploration phase (reading, searching)
        exploration_indicators = [
            "let me read",
            "let me check",
            "let me search",
            "let me look",
            "i'll read",
            "i'll check",
            "i'll search",
            "i'll examine",
            "reading",
            "searching",
            "looking at",
            "examining",
            "first, i need to",
            "let me start by",
            "i'll start by",
        ]
        if any(indicator in response_lower for indicator in exploration_indicators):
            return ResponsePhase.EXPLORATION

        # Default: if no clear phase indicators, assume exploration
        # (conservative default - better to continue than stop prematurely)
        return ResponsePhase.EXPLORATION

    def get_completion_confidence(self) -> CompletionConfidence:
        """Get the current completion confidence level.

        This provides a graded assessment of completion likelihood:
        - HIGH: Active signal detected (deterministic)
        - MEDIUM: File modifications + passive completion signal
        - LOW: Only passive completion phrase detected
        - NONE: No completion signals detected

        Returns:
            Current CompletionConfidence level
        """
        # Priority 1: Active signal detected (HIGH confidence - deterministic)
        if self._state.active_signal_detected:
            return CompletionConfidence.HIGH

        # Priority 2: File modifications + any completion signal (MEDIUM confidence)
        if self._has_file_modifications() and self._state.completion_signals:
            return CompletionConfidence.MEDIUM

        # Priority 3: Only passive completion signals (LOW confidence)
        if self._state.completion_signals:
            # Check if signals are only passive phrases (not active)
            has_only_passive = all(
                signal.startswith("passive:") for signal in self._state.completion_signals
            )
            if has_only_passive:
                return CompletionConfidence.LOW

        # Priority 4: No completion signals (NONE confidence)
        return CompletionConfidence.NONE


def create_task_completion_detector() -> TaskCompletionDetector:
    """Factory function for creating TaskCompletionDetector.

    Returns:
        Configured TaskCompletionDetector instance
    """
    return TaskCompletionDetector()
