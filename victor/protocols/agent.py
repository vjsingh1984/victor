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

"""Agent orchestrator protocol for breaking circular dependencies.

This module defines the IAgentOrchestrator protocol that captures the core
interface of AgentOrchestrator without requiring its import. This breaks
circular dependencies where modules need orchestrator functionality.

Problem:
    - victor/agent/orchestrator.py imports from providers, tools, workflows
    - Those modules often need orchestrator features for context
    - Direct imports create circular dependency chains

Solution:
    - Define IAgentOrchestrator protocol in neutral location
    - Modules depend on protocol, not concrete class
    - Orchestrator implements protocol implicitly (duck typing)
    - Protocol composed of focused sub-protocols (ISP compliance)

Usage:
    # Instead of:
    from victor.agent.orchestrator import AgentOrchestrator
    def process(orchestrator: AgentOrchestrator): ...

    # Use:
    from victor.protocols.agent import IAgentOrchestrator
    def process(orchestrator: IAgentOrchestrator): ...

    # Or use focused protocols (ISP):
    from victor.protocols.chat import ChatProtocol
    def chat(chat_interface: ChatProtocol): ...

Design Principles:
    - DIP: Depend on abstractions (protocol), not concretions (class)
    - ISP: Protocol composed of focused sub-protocols
    - OCP: New features added via protocol extension, not modification

Architecture:
    IAgentOrchestrator (composite)
    ├── ChatProtocol (chat methods)
    ├── ProviderProtocol (provider properties)
    ├── ToolProtocol (tool registry access)
    ├── StateProtocol (session state properties)
    └── ConfigProtocol (configuration properties)
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    pass

# Import focused protocols for composition
from victor.protocols.chat import ChatProtocol
from victor.protocols.config_agent import ConfigProtocol
from victor.protocols.provider import ProviderProtocol
from victor.protocols.state import StateProtocol
from victor.protocols.tools import ToolProtocol


class IAgentOrchestrator(
    ChatProtocol,
    ProviderProtocol,
    ToolProtocol,
    StateProtocol,
    ConfigProtocol,
    Protocol,
):
    """Composite protocol for agent orchestrator functionality.

    This protocol composes focused protocols (ChatProtocol, ProviderProtocol,
    ToolProtocol, StateProtocol, ConfigProtocol) to provide backward
    compatibility with existing code that depends on IAgentOrchestrator.

    This design follows the Interface Segregation Principle (ISP) by:
        - Providing focused protocols for specific use cases
        - Composing them into a composite for backward compatibility
        - Allowing code to depend on minimal interfaces

    Backward Compatibility:
        Existing code using IAgentOrchestrator continues to work without
        changes. New code can depend on focused protocols for better
        separation of concerns.

    Examples:
        # Old way (still works):
        from victor.protocols.agent import IAgentOrchestrator
        def process(orchestrator: IAgentOrchestrator): ...

        # New way (ISP-compliant):
        from victor.protocols.chat import ChatProtocol
        def chat_only(chat: ChatProtocol): ...

        # Composition:
        def multi_tool(chat: ChatProtocol, provider: ProviderProtocol): ...
    """

    pass


@runtime_checkable
class IAgentOrchestratorFactory(Protocol):
    """Protocol for creating agent orchestrators.

    This enables dependency injection of orchestrator creation
    without importing the concrete factory.
    """

    def create(
        self,
        provider_name: str = "anthropic",
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> IAgentOrchestrator:
        """Create a new agent orchestrator.

        Args:
            provider_name: Name of LLM provider to use
            model: Model identifier (provider-specific)
            **kwargs: Additional configuration

        Returns:
            Configured IAgentOrchestrator instance
        """
        ...


__all__ = [
    "IAgentOrchestrator",
    "IAgentOrchestratorFactory",
    # Re-export focused protocols for convenience
    "ChatProtocol",
    "ProviderProtocol",
    "ToolProtocol",
    "StateProtocol",
    "ConfigProtocol",
]
