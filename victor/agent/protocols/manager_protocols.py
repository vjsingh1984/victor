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

"""Protocols for manager/coordinator interfaces.

This module defines Protocol interfaces for manager and coordinator components,
following the Interface Segregation Principle (ISP). Each protocol is focused
on a single responsibility to prevent clients from depending on methods they
don't use.

Part of SOLID-based refactoring to eliminate god class anti-pattern.
"""

from typing import Protocol, Optional, List, Dict, Any, Tuple


# =============================================================================
# Provider Management Protocols
# =============================================================================


class IProviderHealthMonitor(Protocol):
    """Protocol for provider health monitoring.

    Defines interface for monitoring provider health and triggering fallbacks.
    Separated from IProviderSwitcher to follow ISP.
    """

    async def check_health(self, provider: Any) -> bool:
        """Check if provider is healthy.

        Args:
            provider: Provider instance to check

        Returns:
            True if provider is healthy, False otherwise
        """

    def start_health_checks(self, interval: float) -> None:
        """Start periodic health checks.

        Args:
            interval: Interval between health checks in seconds
        """

    def stop_health_checks(self) -> None:
        """Stop health checks."""


class IProviderSwitcher(Protocol):
    """Protocol for provider switching operations.

    Defines interface for switching between providers and models.
    Separated from IProviderHealthMonitor to follow ISP.
    """

    def switch_provider(
        self,
        provider_name: str,
        model: str,
        reason: str = "manual"
    ) -> bool:
        """Switch to a different provider/model.

        Args:
            provider_name: Name of provider to switch to
            model: Model identifier
            reason: Reason for switch (default "manual")

        Returns:
            True if switch succeeded, False otherwise
        """

    def switch_model(self, model: str) -> bool:
        """Switch to a different model on current provider.

        Args:
            model: Model identifier

        Returns:
            True if switch succeeded, False otherwise
        """

    def get_switch_history(self) -> List[Dict[str, Any]]:
        """Get history of provider switches.

        Returns:
            List of switch event dictionaries
        """


class IToolAdapterCoordinator(Protocol):
    """Protocol for tool adapter coordination.

    Defines interface for initializing and managing tool adapters.
    Separated to allow independent testing and mocking.
    """

    def initialize_adapter(self) -> Any:
        """Initialize tool adapter for current provider.

        Returns:
            ToolCallingCapabilities instance
        """

    def get_capabilities(self) -> Any:
        """Get tool calling capabilities.

        Returns:
            ToolCallingCapabilities instance
        """


class IProviderEventEmitter(Protocol):
    """Protocol for provider-related events.

    Defines interface for emitting and handling provider events.
    Separated to support different event implementations.
    """

    def emit_switch_event(self, event: Dict[str, Any]) -> None:
        """Emit provider switch event.

        Args:
            event: Event dictionary with switch details
        """

    def on_switch(self, callback: Any) -> None:
        """Register callback for provider switches.

        Args:
            callback: Callable to invoke on switch
        """


class IProviderClassificationStrategy(Protocol):
    """Protocol for provider classification.

    Defines interface for classifying providers by type.
    Supports Open/Closed Principle via strategy pattern.
    """

    def is_cloud_provider(self, provider_name: str) -> bool:
        """Check if provider is cloud-based.

        Args:
            provider_name: Name of the provider

        Returns:
            True if cloud provider, False otherwise
        """

    def is_local_provider(self, provider_name: str) -> bool:
        """Check if provider is local.

        Args:
            provider_name: Name of the provider

        Returns:
            True if local provider, False otherwise
        """

    def get_provider_type(self, provider_name: str) -> str:
        """Get provider type category.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider type ("cloud", "local", "unknown")
        """


# =============================================================================
# Conversation Management Protocols
# =============================================================================


class IMessageStore(Protocol):
    """Protocol for message storage and retrieval.

    Defines interface for persisting and retrieving messages.
    Separated from other conversation concerns to follow ISP.
    """

    def add_message(self, role: str, content: str, **metadata) -> None:
        """Add a message to storage.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            **metadata: Additional message metadata
        """

    def get_messages(self, limit: Optional[int] = None) -> List[Any]:
        """Retrieve messages.

        Args:
            limit: Optional limit on number of messages

        Returns:
            List of messages
        """

    def persist(self) -> bool:
        """Persist messages to storage.

        Returns:
            True if persistence succeeded, False otherwise
        """


class IContextOverflowHandler(Protocol):
    """Protocol for context overflow handling.

    Defines interface for detecting and handling context overflow.
    Separated from IMessageStore to follow ISP.
    """

    def check_overflow(self) -> bool:
        """Check if context has overflowed.

        Returns:
            True if overflow detected, False otherwise
        """

    def handle_compaction(self) -> Optional[Any]:
        """Handle context compaction.

        Returns:
            Compaction result or None
        """


class ISessionManager(Protocol):
    """Protocol for session lifecycle management.

    Defines interface for creating and managing sessions.
    Separated to support different session backends.
    """

    def create_session(self) -> str:
        """Create a new session.

        Returns:
            Session ID
        """

    def recover_session(self, session_id: str) -> bool:
        """Recover an existing session.

        Args:
            session_id: Session ID to recover

        Returns:
            True if recovery succeeded, False otherwise
        """

    def persist_session(self) -> bool:
        """Persist session state.

        Returns:
            True if persistence succeeded, False otherwise
        """


class IEmbeddingManager(Protocol):
    """Protocol for embedding and semantic search.

    Defines interface for semantic search over conversations.
    Separated because not all conversations need embeddings.
    """

    def initialize_embeddings(self) -> None:
        """Initialize embedding store."""

    def semantic_search(self, query: str, k: int = 5) -> List[Any]:
        """Perform semantic search.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of search results
        """


# =============================================================================
# Budget Management Protocols
# =============================================================================


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

    def get_status(self, budget_type: Any) -> Any:
        """Get current budget status.

        Args:
            budget_type: Type of budget to query

        Returns:
            BudgetStatus instance
        """

    def reset(self) -> None:
        """Reset all budgets."""


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

    def set_model_multiplier(self, multiplier: float) -> None:
        """Set model-specific multiplier.

        Args:
            multiplier: Multiplier value (e.g., 1.0-1.5)
        """

    def set_mode_multiplier(self, multiplier: float) -> None:
        """Set mode-specific multiplier.

        Args:
            multiplier: Multiplier value (e.g., 1.0-3.0)
        """


class IModeCompletionChecker(Protocol):
    """Protocol for mode completion detection.

    Defines interface for checking if mode should complete early.
    Separated from budget tracking to follow ISP.
    """

    def should_early_exit(self, mode: str, response: str) -> Tuple[bool, str]:
        """Check if should exit mode early.

        Args:
            mode: Current mode
            response: Response to check

        Returns:
            Tuple of (should_exit, reason)
        """


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

    def classify_operation(self, tool_name: str) -> str:
        """Classify tool operation type.

        Args:
            tool_name: Name of the tool

        Returns:
            Operation type category
        """

    def add_write_tool(self, tool_name: str) -> None:
        """Add a tool to the write operation classification.

        Args:
            tool_name: Name of the tool to add
        """
