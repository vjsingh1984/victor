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

"""Budget and utility service protocols.

This module contains protocols related to budget management, utility services,
and helper functions. These protocols define contracts for:

- Budget tracking and multiplier calculation
- Mode completion detection
- Tool call classification
- Debug logging
- Task type hints
- Safety checking
- Auto-committing
- MCP bridging
- System prompt building
- Parallel execution
- Response completion
- Streaming handling

Usage:
    from victor.protocols.agent_budget import (
        IBudgetTracker,
        IMultiplierCalculator,
        SafetyCheckerProtocol,
    )
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from collections.abc import AsyncIterator


# =============================================================================
# Budget Management Refinements
# =============================================================================


@runtime_checkable
class IBudgetTracker(Protocol):
    """Protocol for budget tracking.

    Defines interface for tracking and consuming budget.
    Core budget functionality, separated from other concerns.
    """

    def consume(self, budget_type: Any, amount: int) -> bool:
        """Consume from budget.

        Args:
            budget_type: Type of budget to consume from
            amount: Amount to consume

        Returns:
            True if consumption succeeded, False if exhausted
        """
        ...

    def get_status(self, budget_type: Any) -> Any:
        """Get current budget status.

        Args:
            budget_type: Type of budget to query

        Returns:
            BudgetStatus instance
        """
        ...

    def reset(self) -> None:
        """Reset all budgets."""
        ...


@runtime_checkable
class IMultiplierCalculator(Protocol):
    """Protocol for budget multiplier calculation.

    Defines interface for calculating effective budget with multipliers.
    Separated from IBudgetTracker to follow ISP.
    """

    def calculate_effective_max(self, base_max: int) -> int:
        """Calculate effective maximum with multipliers.

        Args:
            base_max: Base maximum budget

        Returns:
            Effective maximum after applying multipliers
        """
        ...

    def set_model_multiplier(self, multiplier: float) -> None:
        """Set model-specific multiplier.

        Args:
            multiplier: Multiplier value (e.g., 1.0-1.5)
        """
        ...

    def set_mode_multiplier(self, multiplier: float) -> None:
        """Set mode-specific multiplier.

        Args:
            multiplier: Multiplier value (e.g., 1.0-3.0)
        """
        ...


@runtime_checkable
class IModeCompletionChecker(Protocol):
    """Protocol for mode completion detection.

    Defines interface for checking if mode should complete early.
    Separated from budget tracking to follow ISP.
    """

    def should_early_exit(self, mode: str, response: str) -> tuple[bool, str]:
        """Check if should exit mode early.

        Args:
            mode: Current mode
            response: Response to check

        Returns:
            Tuple of (should_exit, reason)
        """
        ...


@runtime_checkable
class IToolCallClassifier(Protocol):
    """Protocol for classifying tool calls.

    Defines interface for classifying tools by operation type.
    Supports Open/Closed Principle via strategy pattern.
    """

    def is_write_operation(self, tool_name: str) -> bool:
        """Check if tool is a write operation.

        Args:
            tool_name: Name of the tool

        Returns:
            True if write operation, False otherwise
        """
        ...

    def classify_operation(self, tool_name: str) -> str:
        """Classify tool operation type.

        Args:
            tool_name: Name of the tool

        Returns:
            Operation type category
        """
        ...

    def add_write_tool(self, tool_name: str) -> None:
        """Add a tool to the write operation classification.

        Args:
            tool_name: Name of the tool to add
        """
        ...


# =============================================================================
# Utility Service Protocols
# =============================================================================


@runtime_checkable
class DebugLoggerProtocol(Protocol):
    """Protocol for debug logging service.

    Provides clean, scannable debug output focused on meaningful events.
    """

    def reset(self) -> None:
        """Reset state for new conversation."""
        ...

    def log_iteration_start(self, iteration: int, **context: Any) -> None:
        """Log iteration start."""
        ...

    def log_iteration_end(
        self, iteration: int, has_tool_calls: bool = False, **context: Any
    ) -> None:
        """Log iteration end summary."""
        ...

    def log_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        iteration: int,
    ) -> None:
        """Log tool call."""
        ...

    def log_tool_result(
        self,
        tool_name: str,
        success: bool,
        output: Any,
        elapsed_ms: float,
    ) -> None:
        """Log tool result."""
        ...


@runtime_checkable
class TaskTypeHinterProtocol(Protocol):
    """Protocol for task type hint retrieval.

    Provides task-specific guidance for the LLM.
    """

    def get_hint(self, task_type: str) -> str:
        """Get prompt hint for a specific task type.

        Args:
            task_type: Type of task (edit, search, explain, etc.)

        Returns:
            Formatted hint string for system prompt
        """
        ...


@runtime_checkable
class SafetyCheckerProtocol(Protocol):
    """Protocol for safety checking service.

    Detects dangerous operations and requests confirmation.
    """

    def is_write_tool(self, tool_name: str) -> bool:
        """Check if a tool is a write/modify operation."""
        ...

    async def check_and_confirm(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """Check operation safety and request confirmation if needed.

        Returns:
            Tuple of (should_proceed, optional_rejection_reason)
        """
        ...

    def add_custom_pattern(
        self,
        pattern: str,
        description: str,
        risk_level: str = "HIGH",
        category: str = "custom",
    ) -> None:
        """Add a custom safety pattern from vertical extensions."""
        ...


@runtime_checkable
class AutoCommitterProtocol(Protocol):
    """Protocol for automatic git commits service.

    Handles automatic git commits for AI-assisted changes.
    """

    def is_git_repo(self) -> bool:
        """Check if workspace is a git repository."""
        ...

    def has_changes(self, files: Optional[list[str]] = None) -> bool:
        """Check if there are uncommitted changes."""
        ...

    def commit_changes(
        self,
        files: Optional[list[str]] = None,
        description: str = "AI-assisted changes",
        change_type: Optional[str] = None,
        scope: Optional[str] = None,
        auto_stage: bool = True,
    ) -> Any:
        """Commit changes to git."""
        ...


@runtime_checkable
class MCPBridgeProtocol(Protocol):
    """Protocol for Model Context Protocol bridge.

    Provides access to MCP tools as Victor tools.
    """

    def configure_client(self, client: Any, prefix: str = "mcp") -> None:
        """Configure the MCP client.

        Args:
            client: MCPClient instance
            prefix: Prefix for tool names
        """
        ...

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return MCP tools as Victor tool definitions with a name prefix."""
        ...


# =============================================================================
# Helper/Adapter Service Protocols
# =============================================================================


@runtime_checkable
class SystemPromptBuilderProtocol(Protocol):
    """Protocol for system prompt building service.

    Constructs system prompts from various components.
    """

    def build(
        self,
        base_prompt: str,
        tool_descriptions: Optional[str] = None,
        project_context: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Build system prompt from components.

        Args:
            base_prompt: Base system prompt
            tool_descriptions: Tool descriptions to include
            project_context: Project-specific context
            **kwargs: Additional prompt components

        Returns:
            Complete system prompt
        """
        ...


@runtime_checkable
class ParallelExecutorProtocol(Protocol):
    """Protocol for parallel tool execution service.

    Executes multiple tools in parallel.
    """

    async def execute_parallel(
        self,
        tool_calls: list[Any],
        **kwargs: Any,
    ) -> list[Any]:
        """Execute multiple tool calls in parallel.

        Args:
            tool_calls: List of tool calls to execute
            **kwargs: Additional execution parameters

        Returns:
            List of tool results
        """
        ...


@runtime_checkable
class ResponseCompleterProtocol(Protocol):
    """Protocol for response completion service.

    Completes partial responses and handles tool failures.
    """

    async def complete_response(
        self,
        partial_response: str,
        context: Any,
        **kwargs: Any,
    ) -> str:
        """Complete a partial response.

        Args:
            partial_response: Partial response text
            context: Completion context
            **kwargs: Additional completion parameters

        Returns:
            Completed response
        """
        ...


@runtime_checkable
class StreamingHandlerProtocol(Protocol):
    """Protocol for streaming chat handler service.

    Handles streaming chat responses.
    """

    async def handle_stream(
        self,
        stream: AsyncIterator[Any],
        context: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Handle streaming chat response.

        Args:
            stream: Input stream
            context: Streaming context
            **kwargs: Additional handling parameters

        Yields:
            Processed stream chunks
        """
        ...


@runtime_checkable
class UsageLoggerProtocol(Protocol):
    """Protocol for usage logging service.

    Logs tool and provider usage for analytics.
    """

    def log_tool_call(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        **metadata: Any,
    ) -> None:
        """Log a tool call.

        Args:
            tool_name: Name of the tool
            success: Whether the call succeeded
            duration_ms: Duration in milliseconds
            **metadata: Additional metadata
        """
        ...

    def log_provider_call(
        self,
        provider: str,
        model: str,
        tokens_used: int,
        duration_ms: float,
        **metadata: Any,
    ) -> None:
        """Log a provider API call.

        Args:
            provider: Provider name
            model: Model identifier
            tokens_used: Number of tokens consumed
            duration_ms: Duration in milliseconds
            **metadata: Additional metadata
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary of usage statistics
        """
        ...


@runtime_checkable
class StreamingMetricsCollectorProtocol(Protocol):
    """Protocol for streaming metrics collection.

    Collects real-time metrics during streaming responses.
    """

    def record_chunk(
        self,
        chunk_size: int,
        timestamp: float,
        **metadata: Any,
    ) -> None:
        """Record a streaming chunk.

        Args:
            chunk_size: Size of the chunk
            timestamp: Timestamp of the chunk
            **metadata: Additional metadata
        """
        ...

    def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics.

        Returns:
            Dictionary of streaming metrics
        """
        ...

    def reset(self) -> None:
        """Reset metrics for new session."""
        ...


@runtime_checkable
class IntentClassifierProtocol(Protocol):
    """Protocol for intent classification service.

    Classifies user intents using ML models.
    """

    def classify(self, text: str) -> Any:
        """Classify user intent.

        Args:
            text: User input text

        Returns:
            Classified intent (IntentType or similar)
        """
        ...

    def get_confidence(self, text: str, intent: Any) -> float:
        """Get confidence score for a specific intent.

        Args:
            text: User input text
            intent: Intent to check

        Returns:
            Confidence score (0-1)
        """
        ...


@runtime_checkable
class RLCoordinatorProtocol(Protocol):
    """Protocol for reinforcement learning coordinator.

    Manages all RL learners with unified SQLite storage.
    """

    def record_outcome(
        self,
        learner_name: str,
        outcome: Any,
        vertical: str = "coding",
    ) -> None:
        """Record an outcome for a specific learner."""
        ...

    def get_recommendation(
        self,
        learner_name: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> Optional[Any]:
        """Get recommendation from a learner."""
        ...

    def export_metrics(self) -> dict[str, Any]:
        """Export all learned values and metrics for monitoring."""
        ...

    def close(self) -> None:
        """Close database connection."""
        ...


__all__ = [
    # Budget management refinements
    "IBudgetTracker",
    "IMultiplierCalculator",
    "IModeCompletionChecker",
    "IToolCallClassifier",
    # Utility service protocols
    "DebugLoggerProtocol",
    "TaskTypeHinterProtocol",
    "SafetyCheckerProtocol",
    "AutoCommitterProtocol",
    "MCPBridgeProtocol",
    # Helper/adapter service protocols
    "SystemPromptBuilderProtocol",
    "ParallelExecutorProtocol",
    "ResponseCompleterProtocol",
    "StreamingHandlerProtocol",
    # Analytics protocols
    "UsageLoggerProtocol",
    "StreamingMetricsCollectorProtocol",
    "IntentClassifierProtocol",
    # RL coordinator
    "RLCoordinatorProtocol",
]
