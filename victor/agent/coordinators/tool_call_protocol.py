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

"""Protocol for tool call coordination.

This protocol defines the interface for coordinating tool call operations,
including validation, parsing, execution, and retry logic.
"""

from typing import Any, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.tool_calling.base import ToolCall


class ToolCallContext:
    """Context for tool call execution.

    Attributes:
        iteration: Current iteration number
        max_iterations: Maximum allowed iterations
        tool_budget: Remaining tool budget
        user_message: Original user message
        conversation_stage: Current conversation stage
    """

    def __init__(
        self,
        iteration: int,
        max_iterations: int,
        tool_budget: int,
        user_message: str,
        conversation_stage: str = "initial",
    ) -> None:
        self.iteration = iteration
        self.max_iterations = max_iterations
        self.tool_budget = tool_budget
        self.user_message = user_message
        self.conversation_stage = conversation_stage


class ToolCallResult:
    """Result of tool call execution.

    Attributes:
        tool_name: Name of the tool that was called
        arguments: Arguments passed to the tool
        output: Tool execution output
        error: Error if execution failed
        duration_ms: Execution duration in milliseconds
        success: Whether execution succeeded
    """

    def __init__(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        output: Any,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
        success: bool = True,
    ) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.output = output
        self.error = error
        self.duration_ms = duration_ms
        self.success = success

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "output": self.output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "success": self.success,
        }


class IToolCallCoordinator(Protocol):
    """Protocol for tool call coordination.

    This protocol defines the interface for coordinating tool call operations,
    including validation, parsing, execution, and retry logic.

    The coordinator is responsible for:
    - Parsing tool calls from LLM responses
    - Validating tool calls against budget and constraints
    - Executing tool calls with retry logic
    - Aggregating tool call results
    - Handling errors and edge cases

    Example:
        coordinator = container.get(IToolCallCoordinator)
        context = ToolCallContext(
            iteration=1,
            max_iterations=10,
            tool_budget=100,
            user_message="Read Python files"
        )
        results = await coordinator.handle_tool_calls(tool_calls, context)
    """

    async def handle_tool_calls(
        self,
        tool_calls: list["ToolCall"],
        context: ToolCallContext,
    ) -> list[ToolCallResult]:
        """Handle multiple tool calls with validation, execution, and retry.

        Args:
            tool_calls: List of tool calls to execute
            context: Execution context with budget and constraints

        Returns:
            List of tool call results

        Raises:
            ToolCallValidationError: If tool calls fail validation
            ToolExecutionError: If tool execution fails critically
        """
        ...

    async def execute_tool_with_retry(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolCallContext,
    ) -> ToolCallResult:
        """Execute a single tool with retry logic.

        Implements exponential backoff and intelligent retry strategies
        based on error type and tool semantics.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            context: Execution context

        Returns:
            Tool call result with output or error

        Raises:
            ToolExecutionError: If all retry attempts fail
        """
        ...

    def parse_tool_calls(
        self,
        raw_calls: list[dict[str, Any]],
    ) -> list["ToolCall"]:
        """Parse raw tool calls from LLM response.

        Normalizes tool call format across different providers and handles
        argument normalization and validation.

        Args:
            raw_calls: Raw tool call data from LLM

        Returns:
            List of parsed ToolCall objects

        Raises:
            ToolCallParseError: If parsing fails
        """
        ...

    def validate_tool_calls(
        self,
        tool_calls: list["ToolCall"],
        context: ToolCallContext,
    ) -> list[str]:
        """Validate tool calls against budget and constraints.

        Checks:
        - Tool availability and enablement
        - Budget constraints
        - Access permissions
        - Argument validity

        Args:
            tool_calls: Tool calls to validate
            context: Execution context

        Returns:
            List of validation error messages (empty if valid)

        Raises:
            ToolCallValidationError: If validation fails critically
        """
        ...

    def format_tool_output(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        output: Any,
    ) -> str:
        """Format tool output for inclusion in LLM context.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            output: Tool execution output

        Returns:
            Formatted output string
        """
        ...


class ToolCallCoordinatorConfig:
    """Configuration for ToolCallCoordinator.

    Attributes:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial retry delay in seconds
        retry_backoff_multiplier: Backoff multiplier for exponential backoff
        parallel_execution: Whether to execute tools in parallel
        timeout_seconds: Tool execution timeout
        strict_validation: Whether to enforce strict validation
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff_multiplier: float = 2.0,
        parallel_execution: bool = False,
        timeout_seconds: float = 30.0,
        strict_validation: bool = True,
    ) -> None:
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff_multiplier = retry_backoff_multiplier
        self.parallel_execution = parallel_execution
        self.timeout_seconds = timeout_seconds
        self.strict_validation = strict_validation


__all__ = [
    "IToolCallCoordinator",
    "ToolCallContext",
    "ToolCallResult",
    "ToolCallCoordinatorConfig",
]
