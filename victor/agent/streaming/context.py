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

"""Streaming chat context dataclass.

Contains all state needed for a streaming chat session, enabling
clean dependency injection and easier testing.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from victor.agent.stream_handler import StreamMetrics
from victor.agent.unified_classifier import TaskType


@dataclass
class StreamingChatContext:
    """Context holding all state for a streaming chat session.

    This dataclass encapsulates all the mutable state that was previously
    scattered across local variables in stream_chat. By centralizing state,
    we enable:
    - Easier testing with controlled initial states
    - Clear documentation of what state exists
    - Simpler parameter passing between methods
    """

    # User input
    user_message: str

    # Timing
    start_time: float = field(default_factory=time.time)

    # Metrics
    stream_metrics: StreamMetrics = field(default_factory=StreamMetrics)
    total_tokens: float = 0.0
    cumulative_usage: Dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
    )

    # Iteration control
    max_total_iterations: int = 30
    max_exploration_iterations: int = 10
    total_iterations: int = 0
    force_completion: bool = False

    # Task classification
    unified_task_type: TaskType = TaskType.DEFAULT
    task_classification: Optional[Dict[str, Any]] = None
    complexity_tool_budget: Optional[int] = None
    is_analysis_task: bool = False
    is_action_task: bool = False
    needs_execution: bool = False
    coarse_task_type: str = "default"

    # Content accumulation
    total_accumulated_chars: int = 0
    context_msg: str = ""

    # Goals for tool selection
    goals: List[str] = field(default_factory=list)

    # Quality tracking
    last_quality_score: float = 0.5

    # Blocked attempts tracking
    consecutive_blocked_attempts: int = 0
    max_blocked_before_force: int = 3

    # Garbage detection
    garbage_detected: bool = False

    # Continuation tracking
    continuation_prompts: int = 0
    asking_input_prompts: int = 0

    # Tool execution tracking
    tool_calls: Optional[List[Dict[str, Any]]] = None
    full_content: str = ""
    mentioned_tools_detected: List[str] = field(default_factory=list)

    # Recovery tracking
    consecutive_empty_responses: int = 0
    total_blocked_attempts: int = 0
    force_tool_execution_attempts: int = 0

    # Substantial content threshold for recovery decisions
    substantial_content_threshold: int = 500

    # Tool execution tracking (for budget and progress checks)
    tool_budget: int = 200  # Default tool budget
    tool_calls_used: int = 0
    unique_resources: Set[str] = field(default_factory=set)

    def elapsed_time(self) -> float:
        """Get elapsed time since session start."""
        return time.time() - self.start_time

    def increment_iteration(self) -> int:
        """Increment and return the iteration count."""
        self.total_iterations += 1
        return self.total_iterations

    def is_over_time_limit(self, limit_seconds: float) -> bool:
        """Check if session has exceeded time limit."""
        return self.elapsed_time() > limit_seconds

    def is_over_iteration_limit(self) -> bool:
        """Check if we've exceeded max iterations."""
        return self.total_iterations >= self.max_total_iterations

    def should_force_completion(self) -> bool:
        """Check if we should force completion based on various conditions."""
        return (
            self.force_completion
            or self.consecutive_blocked_attempts >= self.max_blocked_before_force
        )

    def record_blocked_attempt(self) -> bool:
        """Record a blocked tool attempt and return True if force threshold reached."""
        self.consecutive_blocked_attempts += 1
        return self.consecutive_blocked_attempts >= self.max_blocked_before_force

    def reset_blocked_attempts(self) -> None:
        """Reset the blocked attempts counter."""
        self.consecutive_blocked_attempts = 0

    def accumulate_content(self, content: str) -> None:
        """Add content to accumulated total."""
        self.total_accumulated_chars += len(content)

    def update_context_message(self, content: str) -> None:
        """Update the context message for next iteration."""
        self.context_msg = content or self.user_message

    def record_empty_response(self) -> bool:
        """Record an empty response and return True if threshold exceeded."""
        self.consecutive_empty_responses += 1
        return self.consecutive_empty_responses >= 3  # Default threshold

    def reset_empty_responses(self) -> None:
        """Reset the empty response counter."""
        self.consecutive_empty_responses = 0

    def has_substantial_content(self) -> bool:
        """Check if we've accumulated substantial content."""
        return self.total_accumulated_chars >= self.substantial_content_threshold

    def record_tool_blocked(self) -> None:
        """Record a blocked tool attempt."""
        self.total_blocked_attempts += 1

    def record_force_tool_attempt(self) -> int:
        """Record a forced tool execution attempt and return count."""
        self.force_tool_execution_attempts += 1
        return self.force_tool_execution_attempts

    def reset_force_tool_attempts(self) -> None:
        """Reset the forced tool execution counter."""
        self.force_tool_execution_attempts = 0

    def get_remaining_budget(self) -> int:
        """Get remaining tool budget."""
        return max(0, self.tool_budget - self.tool_calls_used)

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted."""
        return self.get_remaining_budget() <= 0

    def is_approaching_budget_limit(self, warning_threshold: int = 250) -> bool:
        """Check if approaching budget limit."""
        return self.tool_calls_used >= warning_threshold and self.get_remaining_budget() > 0

    def record_tool_execution(self, count: int = 1) -> None:
        """Record tool calls used."""
        self.tool_calls_used += count

    def add_unique_resource(self, resource: str) -> None:
        """Track a unique resource accessed by tools."""
        self.unique_resources.add(resource)

    def check_progress(self, base_max_consecutive: int = 8) -> bool:
        """Check if progress is being made relative to tool calls.

        Returns True if progress is adequate, False if stuck.
        """
        max_consecutive = base_max_consecutive
        if self.is_analysis_task:
            max_consecutive = 50
        elif self.is_action_task:
            max_consecutive = 30

        if self.tool_calls_used < max_consecutive:
            return True  # Haven't hit limit yet

        # Check unique resources accessed
        requires_lenient = self.is_analysis_task or self.is_action_task
        threshold = self.tool_calls_used // 4 if requires_lenient else self.tool_calls_used // 2
        return len(self.unique_resources) >= threshold

    def update_quality_score(self, score: float) -> None:
        """Update the last quality score."""
        self.last_quality_score = score


@dataclass
class PreparedStreamContext:
    """Result of preparing a stream context.

    Returned by the preparation phase, containing everything needed
    to start the streaming loop.
    """

    context: StreamingChatContext
    tools: List[Dict[str, Any]]
    provider_kwargs: Dict[str, Any]


def create_stream_context(
    user_message: str,
    max_iterations: int = 30,
    max_exploration: int = 10,
    tool_budget: Optional[int] = None,
) -> StreamingChatContext:
    """Factory function to create a StreamingChatContext.

    Args:
        user_message: The user's input message
        max_iterations: Maximum total iterations
        max_exploration: Maximum exploration iterations
        tool_budget: Optional tool budget override

    Returns:
        Initialized StreamingChatContext
    """
    ctx = StreamingChatContext(
        user_message=user_message,
        max_total_iterations=max_iterations,
        max_exploration_iterations=max_exploration,
        context_msg=user_message,
    )
    if tool_budget is not None:
        ctx.complexity_tool_budget = tool_budget
    return ctx
