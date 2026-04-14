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

"""Service bootstrap for SOLID-refactored architecture.

This module provides factory functions to bootstrap the new service-oriented
architecture when feature flags are enabled. It creates and wires up all the
new services (ChatService, ToolService, etc.) and registers them with the
ServiceContainer.

Usage:
    container = ServiceContainer()
    bootstrap_new_services(container)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.core.container import ServiceContainer, ServiceLifetime

if TYPE_CHECKING:
    from victor.agent.services.chat_service import ChatService
    from victor.agent.services.context_service import ContextService
    from victor.agent.services.protocol_registry import ServiceProtocolRegistry
    from victor.agent.services.provider_service import ProviderService
    from victor.agent.services.recovery_service import RecoveryService
    from victor.agent.services.session_service import SessionService
    from victor.agent.services.tool_service import ToolService
    from victor.agent.conversation_controller import ConversationController
    from victor.agent.streaming_coordinator import StreamingCoordinator

logger = logging.getLogger(__name__)


def bootstrap_new_services(
    container: ServiceContainer,
    conversation_controller: "ConversationController",
    streaming_coordinator: "StreamingCoordinator",
    tool_selector: Optional[Any] = None,
    tool_executor: Optional[Any] = None,
) -> None:
    """Bootstrap all new services with the container.

    This function creates and registers all the new service-oriented
    implementations (ChatService, ToolService, etc.). Services are
    always created for availability, but the orchestrator only uses
    them when USE_SERVICE_LAYER flag is enabled.

    Args:
        container: Service container to register services with
        conversation_controller: Conversation controller for chat operations
        streaming_coordinator: Streaming coordinator for chat operations
        tool_selector: Optional tool selector for ToolService
        tool_executor: Optional tool executor for ToolService

    Example:
        container = ServiceContainer()
        bootstrap_new_services(
            container,
            conversation_controller=my_conversation,
            streaming_coordinator=my_streaming,
        )
    """
    # Import feature flags first
    try:
        from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager
    except ImportError:
        logger.warning("Feature flags not available, skipping service bootstrap")
        return

    feature_flags = get_feature_flag_manager()

    # Import service protocols and implementations
    from victor.agent.services.protocols import (
        ChatServiceProtocol,
        ToolServiceProtocol,
        ContextServiceProtocol,
        ProviderServiceProtocol,
        RecoveryServiceProtocol,
        SessionServiceProtocol,
    )

    # Create service dependencies first (lower-level services)
    services_created: Dict[str, Any] = {}

    # Bootstrap ContextService (always create, used conditionally by orchestrator)
    context_service = _create_context_service(container)
    services_created["context"] = context_service
    container.register(ContextServiceProtocol, lambda c: context_service, ServiceLifetime.SINGLETON)
    logger.info("Bootstrapped ContextService")

    # Bootstrap ProviderService (always create)
    provider_service = _create_provider_service(container)
    services_created["provider"] = provider_service
    container.register(
        ProviderServiceProtocol,
        lambda c: provider_service,
        ServiceLifetime.SINGLETON,
    )
    logger.info("Bootstrapped ProviderService")

    # Bootstrap RecoveryService (always create)
    recovery_service = _create_recovery_service(container)
    services_created["recovery"] = recovery_service
    container.register(
        RecoveryServiceProtocol,
        lambda c: recovery_service,
        ServiceLifetime.SINGLETON,
    )
    logger.info("Bootstrapped RecoveryService")

    # Bootstrap ToolService (always create)
    tool_service = _create_tool_service(
        container,
        tool_selector=tool_selector,
        tool_executor=tool_executor,
    )
    services_created["tool"] = tool_service
    container.register(ToolServiceProtocol, lambda c: tool_service, ServiceLifetime.SINGLETON)
    logger.info("Bootstrapped ToolService")

    # Bootstrap SessionService (always create)
    session_service = _create_session_service(container)
    services_created["session"] = session_service
    container.register(SessionServiceProtocol, lambda c: session_service, ServiceLifetime.SINGLETON)
    logger.info("Bootstrapped SessionService")

    # Bootstrap decision service — prefer tiered (edge+balanced+performance),
    # fall back to single edge, then cloud
    decision_service = None

    if feature_flags.is_enabled(FeatureFlag.USE_EDGE_MODEL):
        # Try tiered service first (routes different DecisionTypes to different tiers)
        decision_service = _create_tiered_decision_service()
        if decision_service is not None:
            logger.info("Bootstrapped TieredDecisionService (edge/balanced/performance)")
        else:
            # Fall back to single edge model
            decision_service = _create_edge_decision_service()
            if decision_service is not None:
                logger.info("Bootstrapped LLMDecisionService with edge model")

    if decision_service is None and feature_flags.is_enabled(FeatureFlag.USE_LLM_DECISION_SERVICE):
        decision_service = _create_llm_decision_service(container)
        if decision_service is not None:
            logger.info("Bootstrapped LLMDecisionService with cloud provider")

    if decision_service is not None:
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )

        services_created["llm_decision"] = decision_service
        container.register(
            LLMDecisionServiceProtocol,
            lambda c: decision_service,
            ServiceLifetime.SINGLETON,
        )

    # Bootstrap ChatService (always create - depends on other services)
    chat_service = _create_chat_service(
        container,
        conversation_controller=conversation_controller,
        streaming_coordinator=streaming_coordinator,
        provider_service=services_created.get("provider"),
        tool_service=services_created.get("tool"),
        context_service=services_created.get("context"),
        recovery_service=services_created.get("recovery"),
    )
    container.register(ChatServiceProtocol, lambda c: chat_service, ServiceLifetime.SINGLETON)
    logger.info("Bootstrapped ChatService")


def _create_context_service(container: ServiceContainer) -> "ContextService":
    """Create ContextService instance.

    Args:
        container: Service container

    Returns:
        Configured ContextService instance
    """
    from victor.agent.services.context_service import (
        ContextService,
        ContextServiceConfig,
    )

    config = ContextServiceConfig(
        max_tokens=100000,  # Default, can be overridden
        overflow_threshold_percent=90.0,
    )
    return ContextService(config=config)


def _create_provider_service(container: ServiceContainer) -> "ProviderService":
    """Create ProviderService instance.

    Args:
        container: Service container

    Returns:
        Configured ProviderService instance
    """
    from victor.agent.services.provider_service import ProviderService

    # Try to get provider registry from container or create mock
    try:
        from victor.providers.registry import ProviderRegistry

        registry = container.get(ProviderRegistry)
    except Exception:
        registry = _MockProviderRegistry()

    return ProviderService(registry=registry)


def _create_recovery_service(container: ServiceContainer) -> "RecoveryService":
    """Create RecoveryService instance.

    Args:
        container: Service container

    Returns:
        Configured RecoveryService instance
    """
    from victor.agent.services.recovery_service import RecoveryService

    return RecoveryService()


def _create_tool_service(
    container: ServiceContainer,
    tool_selector: Optional[Any] = None,
    tool_executor: Optional[Any] = None,
) -> "ToolService":
    """Create ToolService instance.

    Args:
        container: Service container
        tool_selector: Optional tool selector
        tool_executor: Optional tool executor

    Returns:
        Configured ToolService instance
    """
    from victor.agent.services.tool_service import ToolService, ToolServiceConfig

    config = ToolServiceConfig(
        default_max_tools=10,
        default_tool_budget=100,
    )

    # Use provided or create defaults
    if tool_selector is None:
        tool_selector = _create_default_tool_selector(container)
    if tool_executor is None:
        tool_executor = _create_default_tool_executor(container)

    # Create default tool registrar
    tool_registrar = _create_default_tool_registrar(container)

    return ToolService(
        config=config,
        tool_selector=tool_selector,
        tool_executor=tool_executor,
        tool_registrar=tool_registrar,
    )


def _create_default_tool_registrar(container: ServiceContainer) -> Any:
    """Create default tool registrar.

    Args:
        container: Service container

    Returns:
        Default tool registrar instance
    """
    # Return a simple mock registrar
    return _MockToolRegistrar()


class _MockToolRegistrar:
    """Mock tool registrar for bootstrap when real registrar not available."""

    async def register(self, tool: Any, enabled: bool = True) -> None:
        """Register tool - mock implementation."""
        pass


def _create_session_service(container: ServiceContainer) -> "SessionService":
    """Create SessionService instance.

    Args:
        container: Service container

    Returns:
        Configured SessionService instance
    """
    from victor.agent.services.session_service import SessionService

    return SessionService()


def _create_chat_service(
    container: ServiceContainer,
    conversation_controller: "ConversationController",
    streaming_coordinator: "StreamingCoordinator",
    provider_service: Optional["ProviderService"] = None,
    tool_service: Optional["ToolService"] = None,
    context_service: Optional["ContextService"] = None,
    recovery_service: Optional["RecoveryService"] = None,
) -> "ChatService":
    """Create ChatService instance with dependencies.

    Args:
        container: Service container
        conversation_controller: Conversation controller
        streaming_coordinator: Streaming coordinator
        provider_service: Optional provider service
        tool_service: Optional tool service
        context_service: Optional context service
        recovery_service: Optional recovery service

    Returns:
        Configured ChatService instance
    """
    from victor.agent.services.chat_service import ChatService, ChatServiceConfig

    config = ChatServiceConfig(
        max_iterations=200,
        stream_chunk_size=100,
    )

    return ChatService(
        config=config,
        provider_service=provider_service,
        tool_service=tool_service,
        context_service=context_service,
        recovery_service=recovery_service,
        conversation_controller=conversation_controller,
        streaming_coordinator=streaming_coordinator,
    )


def _create_tiered_decision_service() -> Optional[Any]:
    """Create TieredDecisionService with per-DecisionType tier routing.

    Routes different decision types to different model tiers:
    - edge (local): tool_selection, skill_selection, intent, completion
    - balanced (cloud): task_type_classification, multi_skill_decomposition
    - performance (frontier): opt-in only via settings

    Returns None if creation fails (falls back to single edge service).
    """
    try:
        from victor.agent.services.tiered_decision_service import (
            create_tiered_decision_service,
        )
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        return create_tiered_decision_service(config)
    except Exception as e:
        logger.debug("Tiered decision service unavailable: %s", e)
        return None


def _create_edge_decision_service() -> Optional[Any]:
    """Create LLMDecisionService backed by a local edge model (Ollama).

    Uses a small model (qwen2.5-coder:1.5b) for zero-cost micro-decisions.
    Returns None if Ollama is unavailable.
    """
    try:
        from victor.agent.edge_model import (
            EdgeModelConfig,
            create_edge_decision_service,
        )

        config = EdgeModelConfig()
        return create_edge_decision_service(config)
    except Exception as e:
        logger.debug("Edge decision service unavailable: %s", e)
        return None


def _create_llm_decision_service(container: ServiceContainer) -> Optional[Any]:
    """Create LLMDecisionService instance.

    Requires a provider to be available in the container.

    Args:
        container: Service container

    Returns:
        Configured LLMDecisionService instance, or None if provider unavailable
    """
    try:
        from victor.agent.services.decision_service import (
            LLMDecisionService,
            LLMDecisionServiceConfig,
        )

        # Get provider from container or existing services
        provider = None
        model = None
        try:
            from victor.providers.base import BaseProvider

            provider = container.get(BaseProvider)
        except Exception as e:
            logger.debug("Failed to resolve provider from container: %s", e)

        if provider is None:
            logger.warning("No provider available for LLMDecisionService, skipping")
            return None

        # Get model name from provider if available
        model = getattr(provider, "model", None) or getattr(provider, "model_name", None) or ""

        config = LLMDecisionServiceConfig(
            confidence_threshold=0.7,
            micro_budget=10,
            timeout_ms=2000,
            cache_ttl=60,
        )
        return LLMDecisionService(provider=provider, model=model, config=config)
    except Exception as e:
        logger.warning("Failed to create LLMDecisionService: %s", e)
        return None


def _create_default_tool_selector(container: ServiceContainer) -> Any:
    """Create default tool selector.

    Args:
        container: Service container

    Returns:
        Default tool selector instance
    """
    # Try to get existing tool selector from container
    try:
        from victor.agent.tool_selection import ToolSelector

        return container.get(ToolSelector)
    except Exception:
        # Return a simple mock selector
        return _MockToolSelector()


def _create_default_tool_executor(container: ServiceContainer) -> Any:
    """Create default tool executor.

    Args:
        container: Service container

    Returns:
        Default tool executor instance
    """
    # Try to get existing tool executor from container
    try:
        from victor.agent.tool_execution import ToolExecutor

        return container.get(ToolExecutor)
    except Exception:
        # Return a simple mock executor
        return _MockToolExecutor()


class _MockToolSelector:
    """Mock tool selector for bootstrap when real selector not available."""

    async def select(self, context: Any, max_tools: int = 10) -> List[str]:
        """Select tools - returns empty list for mock."""
        return []


class _MockToolExecutor:
    """Mock tool executor for bootstrap when real executor not available."""

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute tool - returns mock result."""
        from victor.tools.base import ToolResult

        return ToolResult(
            success=False,
            output=None,
            error="Mock tool executor - tool not actually executed",
        )


class _MockProviderRegistry:
    """Mock provider registry for bootstrap when real registry not available."""

    def get_provider(self, provider_name: str) -> Optional[Any]:
        """Get provider by name - returns None for mock."""
        return None


__all__ = [
    "bootstrap_new_services",
]
