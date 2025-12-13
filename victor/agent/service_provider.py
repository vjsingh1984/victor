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

"""Service provider for orchestrator dependencies.

Registers all services required by AgentOrchestrator with the DI container.
This enables:
- Centralized service configuration
- Consistent lifecycle management
- Easy testing via override_services
- Type-safe service resolution

Design Pattern: Service Provider
- Groups related service registrations
- Separates singleton vs scoped lifetimes
- Provides factory functions for complex service creation

Usage:
    from victor.core.container import ServiceContainer
    from victor.agent.service_provider import (
        OrchestratorServiceProvider,
        configure_orchestrator_services,
    )

    # Option 1: Full registration
    container = ServiceContainer()
    configure_orchestrator_services(container, settings)

    # Option 2: Selective registration
    provider = OrchestratorServiceProvider(settings)
    provider.register_singleton_services(container)  # Only singletons
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from victor.core.container import ServiceContainer, ServiceLifetime

if TYPE_CHECKING:
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class OrchestratorServiceProvider:
    """Service provider for orchestrator dependencies.

    Manages registration of all services required by AgentOrchestrator.
    Services are categorized by lifetime:

    Singleton Services (application lifetime):
        - ToolRegistry: Shared tool definitions
        - ObservabilityIntegration: Event bus
        - TaskAnalyzer: Shared analysis
        - IntentClassifier: Semantic classification
        - ComplexityClassifier: Task complexity
        - ActionAuthorizer: Action authorization
        - SearchRouter: Search routing
        - ResponseSanitizer: Response cleanup
        - ArgumentNormalizer: Argument normalization
        - ProjectContext: Project-specific instructions

    Scoped Services (per-session):
        - ConversationStateMachine: Per-session state
        - UnifiedTaskTracker: Per-session tracking
        - MessageHistory: Per-session messages

    Attributes:
        _settings: Application settings for service configuration

    Example:
        container = ServiceContainer()
        provider = OrchestratorServiceProvider(settings)
        provider.register_services(container)

        # Resolve singletons directly
        sanitizer = container.get(ResponseSanitizerProtocol)

        # Resolve scoped services within a scope
        with container.create_scope() as scope:
            state_machine = scope.get(ConversationStateMachineProtocol)
    """

    def __init__(self, settings: "Settings"):
        """Initialize the service provider.

        Args:
            settings: Application settings for service configuration
        """
        self._settings = settings

    def register_services(self, container: ServiceContainer) -> None:
        """Register all orchestrator services.

        Registers both singleton and scoped services. Call this method
        during application bootstrap to set up all orchestrator dependencies.

        Args:
            container: DI container to register services in
        """
        self.register_singleton_services(container)
        self.register_scoped_services(container)
        logger.info("Registered all orchestrator services")

    def register_singleton_services(self, container: ServiceContainer) -> None:
        """Register singleton (application-lifetime) services.

        These services are created once and shared across all sessions.
        Use for stateless services or those with expensive initialization.

        Args:
            container: DI container to register services in
        """
        from victor.agent.protocols import (
            ComplexityClassifierProtocol,
            ActionAuthorizerProtocol,
            SearchRouterProtocol,
            ResponseSanitizerProtocol,
            ArgumentNormalizerProtocol,
            TaskAnalyzerProtocol,
            ObservabilityProtocol,
            ToolRegistryProtocol,
            ProjectContextProtocol,
        )

        # ToolRegistry - shared tool definitions
        self._register_tool_registry(container)

        # ObservabilityIntegration - event bus
        self._register_observability(container)

        # TaskAnalyzer - shared analysis facade
        self._register_task_analyzer(container)

        # IntentClassifier - singleton by design (ML model)
        self._register_intent_classifier(container)

        # ComplexityClassifier - stateless
        container.register(
            ComplexityClassifierProtocol,
            lambda c: self._create_complexity_classifier(),
            ServiceLifetime.SINGLETON,
        )

        # ActionAuthorizer - stateless
        container.register(
            ActionAuthorizerProtocol,
            lambda c: self._create_action_authorizer(),
            ServiceLifetime.SINGLETON,
        )

        # SearchRouter - stateless
        container.register(
            SearchRouterProtocol,
            lambda c: self._create_search_router(),
            ServiceLifetime.SINGLETON,
        )

        # ResponseSanitizer - stateless
        container.register(
            ResponseSanitizerProtocol,
            lambda c: self._create_response_sanitizer(),
            ServiceLifetime.SINGLETON,
        )

        # ArgumentNormalizer - stateless
        container.register(
            ArgumentNormalizerProtocol,
            lambda c: self._create_argument_normalizer(),
            ServiceLifetime.SINGLETON,
        )

        # ProjectContext - shared project instructions
        container.register(
            ProjectContextProtocol,
            lambda c: self._create_project_context(),
            ServiceLifetime.SINGLETON,
        )

        logger.debug("Registered singleton orchestrator services")

    def register_scoped_services(self, container: ServiceContainer) -> None:
        """Register scoped (per-session) services.

        These services are created fresh for each orchestrator session.
        Use for stateful services that need isolation between sessions.

        Args:
            container: DI container to register services in
        """
        from victor.agent.protocols import (
            ConversationStateMachineProtocol,
            TaskTrackerProtocol,
            MessageHistoryProtocol,
        )

        # ConversationStateMachine - per-session state
        container.register(
            ConversationStateMachineProtocol,
            lambda c: self._create_conversation_state_machine(),
            ServiceLifetime.SCOPED,
        )

        # UnifiedTaskTracker - per-session tracking
        container.register(
            TaskTrackerProtocol,
            lambda c: self._create_unified_task_tracker(),
            ServiceLifetime.SCOPED,
        )

        # MessageHistory - per-session messages
        container.register(
            MessageHistoryProtocol,
            lambda c: self._create_message_history(),
            ServiceLifetime.SCOPED,
        )

        logger.debug("Registered scoped orchestrator services")

    # =========================================================================
    # Factory methods for singleton services
    # =========================================================================

    def _register_tool_registry(self, container: ServiceContainer) -> None:
        """Register ToolRegistry as singleton."""
        from victor.agent.protocols import ToolRegistryProtocol
        from victor.tools.base import ToolRegistry

        container.register(
            ToolRegistryProtocol,
            lambda c: ToolRegistry(),
            ServiceLifetime.SINGLETON,
        )

    def _register_observability(self, container: ServiceContainer) -> None:
        """Register ObservabilityIntegration as singleton."""
        from victor.agent.protocols import ObservabilityProtocol

        def create_observability(_: ServiceContainer) -> Any:
            enable = getattr(self._settings, "enable_observability", True)
            if not enable:
                return _NullObservability()

            try:
                from victor.observability.integration import ObservabilityIntegration

                return ObservabilityIntegration()
            except ImportError:
                logger.warning("ObservabilityIntegration not available")
                return _NullObservability()

        container.register(
            ObservabilityProtocol,
            create_observability,
            ServiceLifetime.SINGLETON,
        )

    def _register_task_analyzer(self, container: ServiceContainer) -> None:
        """Register TaskAnalyzer as singleton."""
        from victor.agent.protocols import TaskAnalyzerProtocol

        def create_task_analyzer(_: ServiceContainer) -> Any:
            try:
                from victor.agent.task_analyzer import get_task_analyzer

                return get_task_analyzer()
            except ImportError:
                logger.warning("TaskAnalyzer not available")
                return _NullTaskAnalyzer()

        container.register(
            TaskAnalyzerProtocol,
            create_task_analyzer,
            ServiceLifetime.SINGLETON,
        )

    def _register_intent_classifier(self, container: ServiceContainer) -> None:
        """Register IntentClassifier as singleton."""
        from victor.embeddings.intent_classifier import IntentClassifier

        container.register(
            IntentClassifier,
            lambda c: IntentClassifier.get_instance(),
            ServiceLifetime.SINGLETON,
        )

    def _create_complexity_classifier(self) -> Any:
        """Create ComplexityClassifier instance."""
        from victor.agent.complexity_classifier import ComplexityClassifier

        return ComplexityClassifier()

    def _create_action_authorizer(self) -> Any:
        """Create ActionAuthorizer instance."""
        from victor.agent.action_authorizer import ActionAuthorizer

        return ActionAuthorizer()

    def _create_search_router(self) -> Any:
        """Create SearchRouter instance."""
        from victor.agent.search_router import SearchRouter

        return SearchRouter()

    def _create_response_sanitizer(self) -> Any:
        """Create ResponseSanitizer instance."""
        from victor.agent.response_sanitizer import ResponseSanitizer

        return ResponseSanitizer()

    def _create_argument_normalizer(self) -> Any:
        """Create ArgumentNormalizer instance."""
        from victor.agent.argument_normalizer import ArgumentNormalizer

        # Provider name will be updated when orchestrator is created
        return ArgumentNormalizer(provider_name="unknown")

    def _create_project_context(self) -> Any:
        """Create ProjectContext instance."""
        from victor.context.project_context import ProjectContext

        context = ProjectContext()
        context.load()
        return context

    # =========================================================================
    # Factory methods for scoped services
    # =========================================================================

    def _create_conversation_state_machine(self) -> Any:
        """Create ConversationStateMachine instance."""
        from victor.agent.conversation_state import ConversationStateMachine

        return ConversationStateMachine()

    def _create_unified_task_tracker(self) -> Any:
        """Create UnifiedTaskTracker instance."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        return UnifiedTaskTracker()

    def _create_message_history(self) -> Any:
        """Create MessageHistory instance."""
        from victor.agent.message_history import MessageHistory

        return MessageHistory(
            system_prompt="",  # Will be set by orchestrator
            max_history_messages=getattr(self._settings, "max_conversation_history", 100),
        )


# =============================================================================
# Null implementations for graceful degradation
# =============================================================================


class _NullObservability:
    """No-op observability implementation."""

    def on_tool_start(self, tool_name: str, arguments: dict, tool_id: str) -> None:
        pass

    def on_tool_end(
        self,
        tool_name: str,
        result: Any,
        success: bool,
        tool_id: str,
        error: Optional[str] = None,
    ) -> None:
        pass

    def wire_state_machine(self, state_machine: Any) -> None:
        pass

    def on_error(self, error: Exception, context: dict) -> None:
        pass


class _NullTaskAnalyzer:
    """No-op task analyzer implementation."""

    def analyze(self, prompt: str) -> dict:
        return {"complexity": "unknown", "intent": "unknown"}

    def classify_complexity(self, prompt: str) -> Any:
        return None

    def detect_intent(self, prompt: str) -> Any:
        return None


# =============================================================================
# Convenience function
# =============================================================================


def configure_orchestrator_services(
    container: ServiceContainer,
    settings: "Settings",
) -> None:
    """Configure all orchestrator services in one call.

    Convenience function for application bootstrap. Creates a service
    provider and registers all services.

    Args:
        container: DI container to register services in
        settings: Application settings

    Example:
        from victor.core.container import ServiceContainer
        from victor.agent.service_provider import configure_orchestrator_services

        container = ServiceContainer()
        configure_orchestrator_services(container, settings)
    """
    provider = OrchestratorServiceProvider(settings)
    provider.register_services(container)


__all__ = [
    "OrchestratorServiceProvider",
    "configure_orchestrator_services",
]
