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
            RecoveryHandlerProtocol,
            CodeExecutionManagerProtocol,
            WorkflowRegistryProtocol,
            UsageAnalyticsProtocol,
            ToolSequenceTrackerProtocol,
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

        # RecoveryHandler - model failure recovery with Q-learning
        self._register_recovery_handler(container)

        # CodeExecutionManager - manages code execution sandboxes
        container.register(
            CodeExecutionManagerProtocol,
            lambda c: self._create_code_execution_manager(),
            ServiceLifetime.SINGLETON,
        )

        # WorkflowRegistry - shared workflow definitions
        container.register(
            WorkflowRegistryProtocol,
            lambda c: self._create_workflow_registry(),
            ServiceLifetime.SINGLETON,
        )

        # UsageAnalytics - singleton for data-driven optimization
        container.register(
            UsageAnalyticsProtocol,
            lambda c: self._create_usage_analytics(),
            ServiceLifetime.SINGLETON,
        )

        # ToolSequenceTracker - singleton for pattern learning
        container.register(
            ToolSequenceTrackerProtocol,
            lambda c: self._create_tool_sequence_tracker(),
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

    def _register_recovery_handler(self, container: ServiceContainer) -> None:
        """Register RecoveryHandler as singleton.

        The RecoveryHandler integrates with:
        - Q-learning for adaptive recovery strategy selection
        - UsageAnalytics for telemetry
        - ContextCompactor for proactive context management

        Note: Session-specific state (recent_responses, consecutive_failures)
        is reset via set_session_id() when orchestrator creates a new session.
        """
        from victor.agent.protocols import RecoveryHandlerProtocol

        def create_recovery_handler(_: ServiceContainer) -> Any:
            enabled = getattr(self._settings, "enable_recovery_system", True)
            if not enabled:
                return _NullRecoveryHandler()

            try:
                from victor.agent.recovery import RecoveryHandler

                return RecoveryHandler.create(settings=self._settings)
            except ImportError as e:
                logger.warning(f"RecoveryHandler not available: {e}")
                return _NullRecoveryHandler()
            except Exception as e:
                logger.warning(f"RecoveryHandler creation failed: {e}")
                return _NullRecoveryHandler()

        container.register(
            RecoveryHandlerProtocol,
            create_recovery_handler,
            ServiceLifetime.SINGLETON,
        )

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

    def _create_code_execution_manager(self) -> Any:
        """Create CodeExecutionManager instance."""
        from victor.tools.code_executor_tool import CodeExecutionManager

        manager = CodeExecutionManager()
        manager.start()
        return manager

    def _create_workflow_registry(self) -> Any:
        """Create WorkflowRegistry instance."""
        from victor.workflows.base import WorkflowRegistry

        return WorkflowRegistry()

    def _create_usage_analytics(self) -> Any:
        """Create UsageAnalytics singleton instance."""
        from victor.agent.usage_analytics import UsageAnalytics, AnalyticsConfig
        from pathlib import Path

        # Use settings cache_dir if available
        analytics_cache_dir = (
            Path(self._settings.cache_dir)
            if hasattr(self._settings, "cache_dir") and self._settings.cache_dir
            else None
        )

        return UsageAnalytics.get_instance(
            AnalyticsConfig(
                cache_dir=analytics_cache_dir,
                enable_prometheus_export=getattr(self._settings, "enable_prometheus_export", True),
            )
        )

    def _create_tool_sequence_tracker(self) -> Any:
        """Create ToolSequenceTracker instance."""
        from victor.agent.tool_sequence_tracker import create_sequence_tracker

        return create_sequence_tracker(
            use_predefined=getattr(self._settings, "use_predefined_patterns", True),
            learning_rate=getattr(self._settings, "sequence_learning_rate", 0.3),
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


class _NullRecoveryHandler:
    """No-op recovery handler implementation for disabled mode."""

    @property
    def enabled(self) -> bool:
        return False

    @property
    def consecutive_failures(self) -> int:
        return 0

    def detect_failure(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def recover(self, *args: Any, **kwargs: Any) -> Any:
        # Return a no-op outcome
        from dataclasses import dataclass, field
        from enum import Enum, auto

        class _RecoveryAction(Enum):
            CONTINUE = auto()

        @dataclass
        class _RecoveryResult:
            action: _RecoveryAction = _RecoveryAction.CONTINUE
            success: bool = True
            strategy_name: str = "disabled"
            reason: str = "Recovery system disabled"

        @dataclass
        class _RecoveryOutcome:
            result: _RecoveryResult = field(default_factory=_RecoveryResult)

        return _RecoveryOutcome()

    def record_outcome(self, success: bool, quality_improvement: float = 0.0) -> None:
        pass

    def track_response(self, content: str) -> None:
        pass

    def reset_session(self, session_id: str) -> None:
        pass

    def set_context_compactor(self, compactor: Any) -> None:
        pass

    def set_session_id(self, session_id: str) -> None:
        pass

    def get_diagnostics(self) -> dict:
        return {"enabled": False}


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
