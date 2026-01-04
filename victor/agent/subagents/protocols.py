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

"""Protocols for SubAgent dependencies - ISP compliant.

This module defines the minimal interface required by SubAgent instances,
following the Interface Segregation Principle (ISP). Instead of depending
on the full AgentOrchestrator interface, sub-agents depend only on the
specific capabilities they need.

Design Principles:
- ISP Compliance: SubAgentContext contains only methods needed by SubAgent
- DIP Compliance: SubAgent depends on abstractions, not concrete implementations
- Adapter Pattern: SubAgentContextAdapter bridges AgentOrchestrator to protocol

Usage:
    from victor.agent.subagents.protocols import (
        SubAgentContext,
        SubAgentContextAdapter,
    )

    # Type hint with protocol
    def create_subagent(context: SubAgentContext) -> SubAgent:
        return SubAgent(config, context)

    # Adapt from orchestrator
    context = SubAgentContextAdapter(orchestrator)
    subagent = create_subagent(context)

Example:
    # Direct usage with adapter
    adapter = SubAgentContextAdapter(parent_orchestrator)
    print(f"Using provider: {adapter.provider_name}")
    print(f"Model: {adapter.model}")

    # Protocol enables testing with mocks
    mock_context = MagicMock(spec=SubAgentContext)
    mock_context.provider_name = "test_provider"
    mock_context.model = "test_model"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator


@runtime_checkable
class SubAgentContext(Protocol):
    """Minimal context required by SubAgent.

    This protocol defines the subset of AgentOrchestrator's interface
    that SubAgent actually needs. By depending on this narrow interface
    instead of the full orchestrator, SubAgent follows the Interface
    Segregation Principle (ISP).

    Properties Required:
        settings: Configuration settings for the sub-agent
        provider_name: Name of the LLM provider to use
        model: Model identifier to use
        tool_registry: Registry containing available tools

    Benefits:
        - Easier testing: Mock only what's needed
        - Clearer contracts: Document actual dependencies
        - Reduced coupling: SubAgent doesn't depend on unrelated features
    """

    @property
    def settings(self) -> Any:
        """Get configuration settings.

        Returns:
            Settings object with configuration for the sub-agent.
            Typically contains tool_budget, max_context_chars, etc.
        """
        ...

    @property
    def provider_name(self) -> str:
        """Get the provider name.

        Returns:
            Name of the LLM provider (e.g., 'anthropic', 'openai', 'ollama')
        """
        ...

    @property
    def model(self) -> str:
        """Get the model identifier.

        Returns:
            Model identifier (e.g., 'claude-sonnet-4-20250514', 'gpt-4o')
        """
        ...

    @property
    def tool_registry(self) -> Any:
        """Get the tool registry.

        Returns:
            ToolRegistry instance containing available tools.
            Sub-agents filter this to only allowed tools.
        """
        ...

    @property
    def temperature(self) -> float:
        """Get the temperature setting.

        Returns:
            Temperature value for LLM sampling (e.g., 0.0 to 1.0)
        """
        ...


class SubAgentContextAdapter:
    """Adapter to create SubAgentContext from AgentOrchestrator.

    This adapter implements the Adapter pattern to bridge the gap between
    the full AgentOrchestrator interface and the minimal SubAgentContext
    protocol that SubAgent needs.

    The adapter:
    - Wraps an existing AgentOrchestrator instance
    - Exposes only the properties defined in SubAgentContext
    - Delegates to the wrapped orchestrator for implementation

    Usage:
        orchestrator = AgentOrchestrator(settings, provider="anthropic")
        context = SubAgentContextAdapter(orchestrator)

        # Now use context instead of orchestrator for SubAgent creation
        subagent = SubAgent(config, context)

    Attributes:
        _orchestrator: The wrapped AgentOrchestrator instance
    """

    def __init__(self, orchestrator: "AgentOrchestrator"):
        """Initialize adapter with orchestrator.

        Args:
            orchestrator: AgentOrchestrator instance to adapt
        """
        self._orchestrator = orchestrator

    @property
    def settings(self) -> Any:
        """Get configuration settings from orchestrator.

        Returns:
            Settings object from the wrapped orchestrator
        """
        return self._orchestrator.settings

    @property
    def provider_name(self) -> str:
        """Get provider name from orchestrator.

        Returns:
            Provider name from the wrapped orchestrator
        """
        return self._orchestrator.provider_name

    @property
    def model(self) -> str:
        """Get model identifier from orchestrator.

        Returns:
            Model identifier from the wrapped orchestrator
        """
        return self._orchestrator.model

    @property
    def tool_registry(self) -> Any:
        """Get tool registry from orchestrator.

        Returns:
            Tool registry from the wrapped orchestrator
        """
        return self._orchestrator.tool_registry

    @property
    def temperature(self) -> float:
        """Get temperature setting from orchestrator.

        Returns:
            Temperature value from the wrapped orchestrator
        """
        return self._orchestrator.temperature


__all__ = [
    "SubAgentContext",
    "SubAgentContextAdapter",
]
