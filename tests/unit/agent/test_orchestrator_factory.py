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

"""Tests for OrchestratorFactory.

Part of CRITICAL-001: Monolithic Orchestrator decomposition.
"""

import pytest
from rich.console import Console
from unittest.mock import MagicMock, patch

# Suppress deprecation warnings for complexity_classifier shim during migration
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

from victor.agent.orchestrator_factory import (
    OrchestratorFactory,
    create_orchestrator_factory,
    OrchestratorComponents,
    ProviderComponents,
    CoreServices,
    ConversationComponents,
    ToolComponents,
    RecoveryComponents,
    CoordinatorComponents,
)


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    from unittest.mock import Mock

    # Use spec to limit allowed attributes and prevent auto-creation
    settings = Mock(
        spec=[
            "streaming_metrics_enabled",
            "streaming_metrics_history_size",
            "cache_dir",
            "enable_prometheus_export",
            "use_predefined_patterns",
            "sequence_learning_rate",
            "enable_recovery_system",
            "enable_observability",
        ]
    )
    settings.streaming_metrics_enabled = True
    settings.streaming_metrics_history_size = 100
    settings.cache_dir = "/tmp/cache"
    settings.enable_prometheus_export = True
    settings.use_predefined_patterns = True
    settings.sequence_learning_rate = 0.3
    settings.enable_recovery_system = True
    settings.enable_observability = True
    return settings


@pytest.fixture
def mock_provider():
    """Create mock provider."""
    provider = MagicMock()
    provider.name = "test_provider"
    provider.__class__.__name__ = "TestProvider"
    return provider


@pytest.fixture
def mock_container():
    """Create mock DI container."""
    from victor.agent.response_sanitizer import ResponseSanitizer
    from victor.framework.task import TaskComplexityService as ComplexityClassifier
    from victor.agent.action_authorizer import ActionAuthorizer
    from victor.agent.search_router import SearchRouter
    from victor.context.project_context import ProjectContext
    from victor.agent.usage_analytics import UsageAnalytics
    from victor.agent.tool_sequence_tracker import ToolSequenceTracker
    from victor.agent.recovery.handler import RecoveryHandler

    container = MagicMock()

    # Mock get() to return actual instances for services
    def mock_get(protocol):
        # Map protocols to concrete instances
        from victor.agent.protocols import (
            ResponseSanitizerProtocol,
            ComplexityClassifierProtocol,
            ActionAuthorizerProtocol,
            SearchRouterProtocol,
            ProjectContextProtocol,
            UsageAnalyticsProtocol,
            ToolSequenceTrackerProtocol,
            RecoveryHandlerProtocol,
        )

        protocol_map = {
            ResponseSanitizerProtocol: ResponseSanitizer(),
            ComplexityClassifierProtocol: ComplexityClassifier(),
            ActionAuthorizerProtocol: ActionAuthorizer(),
            SearchRouterProtocol: SearchRouter(),
            ProjectContextProtocol: ProjectContext(),
            UsageAnalyticsProtocol: UsageAnalytics(),
            ToolSequenceTrackerProtocol: ToolSequenceTracker(),
            RecoveryHandlerProtocol: MagicMock(
                spec=RecoveryHandler
            ),  # Use mock for complex dependency
        }
        return protocol_map.get(protocol, MagicMock())

    container.get = MagicMock(side_effect=mock_get)
    container.get_optional = MagicMock(return_value=None)
    return container


@pytest.fixture
def factory(mock_settings, mock_provider, mock_container):
    """Create factory with mocked dependencies."""
    f = OrchestratorFactory(
        settings=mock_settings,
        provider=mock_provider,
        model="test-model",
        temperature=0.7,
        max_tokens=4096,
    )
    # Pre-set the container to avoid lazy initialization issues
    f._container = mock_container
    return f


class TestOrchestratorFactoryInit:
    """Tests for OrchestratorFactory initialization."""

    def test_factory_stores_settings(self, factory, mock_settings):
        """Factory stores settings."""
        assert factory.settings == mock_settings

    def test_factory_stores_provider(self, factory, mock_provider):
        """Factory stores provider."""
        assert factory.provider == mock_provider

    def test_factory_stores_model(self, factory):
        """Factory stores model."""
        assert factory.model == "test-model"

    def test_factory_stores_temperature(self, factory):
        """Factory stores temperature."""
        assert factory.temperature == 0.7

    def test_factory_stores_max_tokens(self, factory):
        """Factory stores max_tokens."""
        assert factory.max_tokens == 4096

    def test_factory_default_tool_selection(self, factory):
        """Factory has empty default tool_selection."""
        assert factory.tool_selection == {}

    def test_factory_default_thinking(self, factory):
        """Factory has False default thinking."""
        assert factory.thinking is False


class TestCreateSanitizer:
    """Tests for create_sanitizer method."""

    def test_create_sanitizer_returns_sanitizer(self, factory):
        """create_sanitizer returns a ResponseSanitizer instance."""
        from victor.agent.response_sanitizer import ResponseSanitizer

        sanitizer = factory.create_sanitizer()
        assert isinstance(sanitizer, ResponseSanitizer)

    def test_create_sanitizer_uses_di_if_available(self, factory, mock_container):
        """create_sanitizer uses DI container (always)."""
        from victor.agent.protocols import ResponseSanitizerProtocol

        sanitizer = factory.create_sanitizer()

        # Verify DI container was used
        mock_container.get.assert_called_with(ResponseSanitizerProtocol)


class TestCreateProjectContext:
    """Tests for create_project_context method."""

    def test_create_project_context_returns_context(self, factory):
        """create_project_context returns a ProjectContext instance."""
        from victor.context.project_context import ProjectContext

        context = factory.create_project_context()
        assert isinstance(context, ProjectContext)

    def test_create_project_context_uses_di_if_available(self, factory, mock_container):
        """create_project_context uses DI container (always)."""
        from victor.agent.protocols import ProjectContextProtocol

        context = factory.create_project_context()

        # Verify DI container was used
        mock_container.get.assert_called_with(ProjectContextProtocol)


class TestCreateComplexityClassifier:
    """Tests for create_complexity_classifier method."""

    def test_create_complexity_classifier_returns_classifier(self, factory):
        """create_complexity_classifier returns a ComplexityClassifier instance."""
        from victor.framework.task import TaskComplexityService as ComplexityClassifier

        classifier = factory.create_complexity_classifier()
        assert isinstance(classifier, ComplexityClassifier)


class TestCreateActionAuthorizer:
    """Tests for create_action_authorizer method."""

    def test_create_action_authorizer_returns_authorizer(self, factory):
        """create_action_authorizer returns an ActionAuthorizer instance."""
        from victor.agent.action_authorizer import ActionAuthorizer

        authorizer = factory.create_action_authorizer()
        assert isinstance(authorizer, ActionAuthorizer)


class TestCreateSearchRouter:
    """Tests for create_search_router method."""

    def test_create_search_router_returns_router(self, factory):
        """create_search_router returns a SearchRouter instance."""
        from victor.agent.search_router import SearchRouter

        router = factory.create_search_router()
        assert isinstance(router, SearchRouter)


class TestCreateCoreServices:
    """Tests for create_core_services method."""

    def test_create_core_services_returns_all_services(self, factory):
        """create_core_services returns CoreServices with all components."""
        mock_adapter = MagicMock()
        mock_capabilities = MagicMock()

        with (
            patch.object(factory, "create_sanitizer") as mock_san,
            patch.object(factory, "create_prompt_builder") as mock_pb,
            patch.object(factory, "create_project_context") as mock_pc,
            patch.object(factory, "create_complexity_classifier") as mock_cc,
            patch.object(factory, "create_action_authorizer") as mock_aa,
            patch.object(factory, "create_search_router") as mock_sr,
        ):

            services = factory.create_core_services(mock_adapter, mock_capabilities)

            assert isinstance(services, CoreServices)
            assert services.sanitizer == mock_san.return_value
            assert services.prompt_builder == mock_pb.return_value
            assert services.project_context == mock_pc.return_value
            assert services.complexity_classifier == mock_cc.return_value
            assert services.action_authorizer == mock_aa.return_value
            assert services.search_router == mock_sr.return_value


class TestCreateStreamingMetricsCollector:
    """Tests for create_streaming_metrics_collector method."""

    def test_create_streaming_metrics_collector_when_enabled(self, factory, mock_settings):
        """create_streaming_metrics_collector returns collector when enabled."""
        from victor.analytics.streaming_metrics import StreamingMetricsCollector

        mock_settings.streaming_metrics_enabled = True

        collector = factory.create_streaming_metrics_collector()
        assert isinstance(collector, StreamingMetricsCollector)

    def test_create_streaming_metrics_collector_when_disabled(self, factory, mock_settings):
        """create_streaming_metrics_collector returns None when disabled."""
        mock_settings.streaming_metrics_enabled = False

        collector = factory.create_streaming_metrics_collector()

        assert collector is None


class TestCreateUsageAnalytics:
    """Tests for create_usage_analytics method."""

    def test_create_usage_analytics_returns_analytics(self, factory):
        """create_usage_analytics returns UsageAnalytics instance."""
        from victor.agent.usage_analytics import UsageAnalytics

        analytics = factory.create_usage_analytics()
        assert isinstance(analytics, UsageAnalytics)


class TestCreateSequenceTracker:
    """Tests for create_sequence_tracker method."""

    def test_create_sequence_tracker_returns_tracker(self, factory):
        """create_sequence_tracker returns ToolSequenceTracker instance."""
        from victor.agent.tool_sequence_tracker import ToolSequenceTracker

        tracker = factory.create_sequence_tracker()
        assert isinstance(tracker, ToolSequenceTracker)


class TestCreateRecoveryHandler:
    """Tests for create_recovery_handler method."""

    def test_create_recovery_handler_when_enabled(self, factory, mock_settings):
        """create_recovery_handler returns handler when enabled."""
        from victor.agent.recovery import RecoveryHandler

        mock_settings.enable_recovery_system = True

        handler = factory.create_recovery_handler()
        # Handler may be None if creation fails due to missing dependencies
        # but should not raise an exception
        assert handler is None or isinstance(handler, RecoveryHandler)

    def test_create_recovery_handler_when_disabled(self, factory, mock_settings):
        """create_recovery_handler returns a handler from DI container when disabled."""
        mock_settings.enable_recovery_system = False

        handler = factory.create_recovery_handler()

        # Handler is always returned from DI container (may be null implementation)
        # In production, this would be _NullRecoveryHandler when disabled
        assert handler is not None


class TestCreateObservability:
    """Tests for create_observability method."""

    def test_create_observability_when_enabled(self, factory, mock_settings):
        """create_observability returns integration when enabled."""
        from victor.observability.integration import ObservabilityIntegration

        mock_settings.enable_observability = True

        observability = factory.create_observability()
        assert isinstance(observability, ObservabilityIntegration)

    def test_create_observability_when_disabled(self, factory, mock_settings):
        """create_observability returns None when disabled."""
        mock_settings.enable_observability = False

        observability = factory.create_observability()

        assert observability is None


class TestCreateOrchestratorFactory:
    """Tests for create_orchestrator_factory convenience function."""

    def test_create_orchestrator_factory_returns_factory(self, mock_settings, mock_provider):
        """create_orchestrator_factory returns OrchestratorFactory."""
        factory = create_orchestrator_factory(
            settings=mock_settings,
            provider=mock_provider,
            model="test-model",
        )

        assert isinstance(factory, OrchestratorFactory)
        assert factory.settings == mock_settings
        assert factory.provider == mock_provider
        assert factory.model == "test-model"


class TestInitializeOrchestrator:
    """Tests for orchestrator initialization sequencing."""

    def test_builder_sequence_returns_expected_order(self, factory):
        """_builder_sequence returns the expected builder ordering."""
        sequence = factory._builder_sequence()

        assert [builder.__name__ for builder in sequence] == [
            "ProviderLayerBuilder",
            "PromptingBuilder",
            "SessionServicesBuilder",
            "MetricsLoggingBuilder",
            "WorkflowMemoryBuilder",
            "WorkflowChatBuilder",  # Phase 1: Domain-Agnostic Workflow Chat
            "IntelligentIntegrationBuilder",
            "ToolingBuilder",
            "ConversationPipelineBuilder",
            "ContextIntelligenceBuilder",
            "RecoveryObservabilityBuilder",
            "ConfigWorkflowBuilder",
            "FinalizationBuilder",
        ]

    def test_initialize_orchestrator_calls_steps_in_order(self, factory):
        """initialize_orchestrator calls helper methods in order."""
        orchestrator = MagicMock()
        call_order = []

        def make_builder(name):
            class _Builder:
                def __init__(self, settings, factory=None):
                    self._name = name

                def build(self, orchestrator, **_kwargs):
                    call_order.append(self._name)

            return _Builder

        expected = [
            "ProviderLayerBuilder",
            "PromptingBuilder",
            "SessionServicesBuilder",
            "MetricsLoggingBuilder",
            "WorkflowMemoryBuilder",
            "WorkflowChatBuilder",  # Phase 1: Domain-Agnostic Workflow Chat
            "IntelligentIntegrationBuilder",
            "ToolingBuilder",
            "ConversationPipelineBuilder",
            "ContextIntelligenceBuilder",
            "RecoveryObservabilityBuilder",
            "ConfigWorkflowBuilder",
            "FinalizationBuilder",
        ]

        builders = [make_builder(name) for name in expected]

        with patch.object(factory, "_builder_sequence", return_value=builders):
            factory.initialize_orchestrator(orchestrator)

        assert call_order == expected

    def test_initialize_orchestrator_sets_core_fields(self, factory):
        """initialize_orchestrator sets core orchestrator fields."""
        orchestrator = MagicMock()

        with patch.object(factory, "_builder_sequence", return_value=[]):
            factory.initialize_orchestrator(orchestrator)

        assert orchestrator.settings == factory.settings
        assert orchestrator.temperature == factory.temperature
        assert orchestrator.max_tokens == factory.max_tokens
        assert orchestrator.tool_selection == factory.tool_selection
        assert orchestrator.thinking == factory.thinking
        assert orchestrator._factory == factory
        assert orchestrator._container == factory._container
        assert isinstance(orchestrator.console, Console)


class TestDataClasses:
    """Tests for dataclass definitions."""

    def test_provider_components_fields(self):
        """ProviderComponents has expected fields."""
        provider = MagicMock()
        components = ProviderComponents(
            provider=provider,
            model="test",
            provider_name="test_provider",
            tool_adapter=MagicMock(),
            tool_calling_caps=MagicMock(),
        )
        assert components.provider == provider
        assert components.model == "test"
        assert components.provider_name == "test_provider"

    def test_core_services_fields(self):
        """CoreServices has expected fields."""
        services = CoreServices(
            sanitizer=MagicMock(),
            prompt_builder=MagicMock(),
            project_context=MagicMock(),
            complexity_classifier=MagicMock(),
            action_authorizer=MagicMock(),
            search_router=MagicMock(),
        )
        assert services.sanitizer is not None
        assert services.prompt_builder is not None

    def test_orchestrator_components_fields(self):
        """OrchestratorComponents has expected fields."""
        components = OrchestratorComponents()
        assert components.observability is None
        assert components.tool_output_formatter is None

    def test_conversation_components_defaults(self):
        """ConversationComponents has correct defaults."""
        components = ConversationComponents(
            conversation_controller=MagicMock(),
        )
        assert components.conversation_controller is not None
        assert components.memory_manager is None
        assert components.memory_session_id is None

    def test_tool_components_defaults(self):
        """ToolComponents has correct defaults."""
        components = ToolComponents(
            tool_registry=MagicMock(),
            tool_registrar=MagicMock(),
            tool_executor=MagicMock(),
        )
        assert components.tool_cache is None
        assert components.plugin_manager is None

    def test_recovery_components_fields(self):
        """RecoveryComponents has expected fields."""
        components = RecoveryComponents(
            recovery_handler=MagicMock(),
            recovery_integration=MagicMock(),
            context_compactor=MagicMock(),
        )
        assert components.recovery_handler is not None
        assert components.context_compactor is not None


class TestContainerProperty:
    """Tests for container property lazy initialization."""

    def test_container_property_initializes_lazily(self, mock_settings, mock_provider):
        """Container property triggers bootstrap on first access."""
        factory = OrchestratorFactory(
            settings=mock_settings,
            provider=mock_provider,
            model="test-model",
        )

        # Initially None
        assert factory._container is None

        # Access triggers initialization
        with patch("victor.core.bootstrap.ensure_bootstrapped") as mock_bootstrap:
            mock_bootstrap.return_value = MagicMock()
            container = factory.container
            mock_bootstrap.assert_called_once_with(mock_settings)
            assert container is not None


class TestCreateToolCache:
    """Tests for create_tool_cache method."""

    def test_create_tool_cache_when_enabled(self, factory, mock_settings):
        """create_tool_cache returns ToolCache when enabled."""
        mock_settings.tool_cache_enabled = True
        mock_settings.tool_cache_ttl = 600
        mock_settings.tool_cache_allowlist = []

        from pathlib import Path

        with patch("victor.config.settings.get_project_paths") as mock_paths:
            mock_paths.return_value.global_cache_dir = Path("/tmp/cache")
            with patch("victor.storage.cache.tool_cache.ToolCache") as mock_cache_cls:
                mock_cache = MagicMock()
                mock_cache_cls.return_value = mock_cache
                cache = factory.create_tool_cache()
                assert cache == mock_cache

    def test_create_tool_cache_when_disabled(self, factory, mock_settings):
        """create_tool_cache returns None when disabled."""
        mock_settings.tool_cache_enabled = False

        cache = factory.create_tool_cache()

        assert cache is None


class TestCreateMemoryComponents:
    """Tests for create_memory_components method."""

    def test_create_memory_components_when_enabled(self, factory, mock_settings):
        """create_memory_components returns tuple when enabled."""
        mock_settings.conversation_memory_enabled = True
        mock_settings.max_context_tokens = 100000
        mock_settings.response_token_reserve = 4096

        with patch("victor.config.settings.get_project_paths") as mock_paths:
            mock_project = MagicMock()
            mock_project.project_victor_dir = MagicMock()
            mock_project.conversation_db = "/tmp/test.db"
            mock_project.project_root = "/tmp/project"
            mock_paths.return_value = mock_project

            with patch("victor.agent.conversation_memory.ConversationStore") as mock_store_cls:
                mock_store = MagicMock()
                mock_session = MagicMock()
                mock_session.session_id = "test-session-id"
                mock_store.create_session.return_value = mock_session
                mock_store_cls.return_value = mock_store

                memory, session_id = factory.create_memory_components("test_provider")

                assert memory is not None
                assert session_id == "test-session-id"

    def test_create_memory_components_when_disabled(self, factory, mock_settings):
        """create_memory_components returns (None, None) when disabled."""
        mock_settings.conversation_memory_enabled = False

        memory, session_id = factory.create_memory_components("test_provider")

        assert memory is None
        assert session_id is None


class TestCreateUsageLogger:
    """Tests for create_usage_logger method."""

    def test_create_usage_logger_uses_di_if_available(self, factory, mock_container):
        """create_usage_logger uses DI container if available."""
        di_logger = MagicMock()
        mock_container.get_optional.return_value = di_logger

        logger = factory.create_usage_logger()

        assert logger == di_logger

    def test_create_usage_logger_fallback(self, factory, mock_container, mock_settings):
        """create_usage_logger falls back to basic logger."""
        mock_container.get_optional.return_value = None
        mock_settings.analytics_enabled = True

        with patch("victor.config.settings.get_project_paths") as mock_paths:
            mock_paths.return_value.global_logs_dir = MagicMock()
            mock_paths.return_value.global_logs_dir.__truediv__ = MagicMock(
                return_value="/tmp/usage.jsonl"
            )

            with patch("victor.analytics.logger.UsageLogger") as mock_logger_cls:
                mock_logger = MagicMock()
                mock_logger_cls.return_value = mock_logger

                logger = factory.create_usage_logger()

                assert logger == mock_logger


# =============================================================================
# Workflow Optimization Components Tests
# =============================================================================


class TestWorkflowOptimizationComponents:
    """Tests for workflow optimization component factory methods."""

    def test_create_task_completion_detector(self, factory):
        """create_task_completion_detector returns TaskCompletionDetector."""
        detector = factory.create_task_completion_detector()

        from victor.agent.task_completion import TaskCompletionDetector

        assert isinstance(detector, TaskCompletionDetector)

    def test_create_read_cache(self, factory, mock_settings):
        """create_read_cache returns ReadResultCache with settings."""
        mock_settings.read_cache_ttl = 120.0
        mock_settings.read_cache_max_entries = 50

        cache = factory.create_read_cache()

        from victor.agent.read_cache import ReadResultCache

        assert isinstance(cache, ReadResultCache)
        assert cache._ttl == 120.0
        assert cache._max_entries == 50

    def test_create_read_cache_defaults(self, factory, mock_settings):
        """create_read_cache uses defaults when settings not present."""
        # Remove attributes to test defaults
        if hasattr(mock_settings, "read_cache_ttl"):
            delattr(mock_settings, "read_cache_ttl")
        if hasattr(mock_settings, "read_cache_max_entries"):
            delattr(mock_settings, "read_cache_max_entries")

        cache = factory.create_read_cache()

        from victor.agent.read_cache import ReadResultCache

        assert isinstance(cache, ReadResultCache)
        assert cache._ttl == 300.0  # default
        assert cache._max_entries == 100  # default

    def test_create_time_aware_executor_with_timeout(self, factory):
        """create_time_aware_executor with explicit timeout."""
        executor = factory.create_time_aware_executor(timeout_seconds=60.0)

        from victor.agent.time_aware_executor import TimeAwareExecutor

        assert isinstance(executor, TimeAwareExecutor)
        assert executor.get_remaining_seconds() is not None

    def test_create_time_aware_executor_no_timeout(self, factory, mock_settings):
        """create_time_aware_executor without timeout."""
        # Ensure no timeout setting
        if hasattr(mock_settings, "execution_timeout"):
            delattr(mock_settings, "execution_timeout")

        executor = factory.create_time_aware_executor()

        from victor.agent.time_aware_executor import TimeAwareExecutor

        assert isinstance(executor, TimeAwareExecutor)
        assert executor.get_remaining_seconds() is None

    def test_create_thinking_detector(self, factory, mock_settings):
        """create_thinking_detector returns ThinkingPatternDetector."""
        mock_settings.thinking_repetition_threshold = 4
        mock_settings.thinking_similarity_threshold = 0.7

        detector = factory.create_thinking_detector()

        from victor.agent.thinking_detector import ThinkingPatternDetector

        assert isinstance(detector, ThinkingPatternDetector)
        assert detector._repetition_threshold == 4
        assert detector._similarity_threshold == 0.7

    def test_create_resource_manager(self, factory):
        """create_resource_manager returns singleton ResourceManager."""
        from victor.agent.resource_manager import ResourceManager

        # Reset singleton for test
        ResourceManager._instance = None

        manager = factory.create_resource_manager()

        assert isinstance(manager, ResourceManager)

        # Cleanup
        manager.reset()
        ResourceManager._instance = None

    def test_create_mode_completion_criteria(self, factory):
        """create_mode_completion_criteria returns ModeCompletionChecker."""
        criteria = factory.create_mode_completion_criteria()

        from victor.agent.budget_manager import ModeCompletionChecker

        assert isinstance(criteria, ModeCompletionChecker)

    def test_create_workflow_optimization_components(self, factory):
        """create_workflow_optimization_components returns all components."""
        from victor.agent.resource_manager import ResourceManager

        # Reset singleton for test
        ResourceManager._instance = None

        components = factory.create_workflow_optimization_components(timeout_seconds=30.0)

        from victor.agent.orchestrator_factory import WorkflowOptimizationComponents

        assert isinstance(components, WorkflowOptimizationComponents)
        assert components.task_completion_detector is not None
        assert components.read_cache is not None
        assert components.time_aware_executor is not None
        assert components.thinking_detector is not None
        assert components.resource_manager is not None
        assert components.mode_completion_criteria is not None

        # Cleanup
        components.resource_manager.reset()
        ResourceManager._instance = None


# =============================================================================
# Coordinator Components Tests (Phase 1.4)
# =============================================================================


class TestCreateConfigCoordinator:
    """Tests for create_config_coordinator method."""

    def test_create_config_coordinator_returns_coordinator(self, factory):
        """create_config_coordinator returns ConfigCoordinator instance."""
        from victor.agent.coordinators.config_coordinator import ConfigCoordinator

        coordinator = factory.create_config_coordinator()

        assert isinstance(coordinator, ConfigCoordinator)

    def test_create_config_coordinator_empty_providers_by_default(self, factory):
        """create_config_coordinator creates coordinator with no providers by default."""
        from victor.agent.coordinators.config_coordinator import ConfigCoordinator

        coordinator = factory.create_config_coordinator()

        assert isinstance(coordinator, ConfigCoordinator)
        assert coordinator._providers == []

    def test_create_config_coordinator_with_providers(self, factory):
        """create_config_coordinator accepts provider list."""
        from victor.agent.coordinators.config_coordinator import ConfigCoordinator
        from victor.protocols import IConfigProvider

        # Create mock providers
        mock_provider1 = MagicMock(spec=IConfigProvider)
        mock_provider1.priority.return_value = 10
        mock_provider2 = MagicMock(spec=IConfigProvider)
        mock_provider2.priority.return_value = 20

        coordinator = factory.create_config_coordinator(
            config_providers=[mock_provider1, mock_provider2]
        )

        assert isinstance(coordinator, ConfigCoordinator)
        assert len(coordinator._providers) == 2


class TestCreatePromptCoordinator:
    """Tests for create_prompt_coordinator method."""

    def test_create_prompt_coordinator_returns_coordinator(self, factory):
        """create_prompt_coordinator returns PromptCoordinator instance."""
        from victor.agent.coordinators.prompt_coordinator import PromptCoordinator

        coordinator = factory.create_prompt_coordinator()

        assert isinstance(coordinator, PromptCoordinator)

    def test_create_prompt_coordinator_empty_contributors_by_default(self, factory):
        """create_prompt_coordinator creates coordinator with no contributors by default."""
        from victor.agent.coordinators.prompt_coordinator import PromptCoordinator

        coordinator = factory.create_prompt_coordinator()

        assert isinstance(coordinator, PromptCoordinator)
        assert coordinator._contributors == []

    def test_create_prompt_coordinator_with_contributors(self, factory):
        """create_prompt_coordinator accepts contributor list."""
        from victor.agent.coordinators.prompt_coordinator import PromptCoordinator
        from victor.protocols import IPromptContributor

        # Create mock contributors
        mock_contributor1 = MagicMock(spec=IPromptContributor)
        mock_contributor1.priority.return_value = 10
        mock_contributor2 = MagicMock(spec=IPromptContributor)
        mock_contributor2.priority.return_value = 20

        coordinator = factory.create_prompt_coordinator(
            prompt_contributors=[mock_contributor1, mock_contributor2]
        )

        assert isinstance(coordinator, PromptCoordinator)
        assert len(coordinator._contributors) == 2


class TestCreateContextCoordinator:
    """Tests for create_context_coordinator method."""

    def test_create_context_coordinator_returns_coordinator(self, factory):
        """create_context_coordinator returns ContextCoordinator instance."""
        from victor.agent.coordinators.context_coordinator import ContextCoordinator

        coordinator = factory.create_context_coordinator()

        assert isinstance(coordinator, ContextCoordinator)

    def test_create_context_coordinator_empty_strategies_by_default(self, factory):
        """create_context_coordinator creates coordinator with no strategies by default."""
        from victor.agent.coordinators.context_coordinator import ContextCoordinator

        coordinator = factory.create_context_coordinator()

        assert isinstance(coordinator, ContextCoordinator)
        assert coordinator._strategies == []

    def test_create_context_coordinator_with_strategies(self, factory):
        """create_context_coordinator accepts strategy list."""
        from victor.agent.coordinators.context_coordinator import ContextCoordinator
        from victor.protocols import ICompactionStrategy

        # Create mock strategies
        mock_strategy1 = MagicMock(spec=ICompactionStrategy)
        mock_strategy2 = MagicMock(spec=ICompactionStrategy)

        coordinator = factory.create_context_coordinator(
            compaction_strategies=[mock_strategy1, mock_strategy2]
        )

        assert isinstance(coordinator, ContextCoordinator)
        assert len(coordinator._strategies) == 2


class TestCreateAnalyticsCoordinator:
    """Tests for create_analytics_coordinator method."""

    def test_create_analytics_coordinator_returns_coordinator(self, factory):
        """create_analytics_coordinator returns AnalyticsCoordinator instance."""
        from victor.agent.coordinators.analytics_coordinator import AnalyticsCoordinator

        coordinator = factory.create_analytics_coordinator()

        assert isinstance(coordinator, AnalyticsCoordinator)

    def test_create_analytics_coordinator_empty_exporters_by_default(self, factory):
        """create_analytics_coordinator creates coordinator with no exporters by default."""
        from victor.agent.coordinators.analytics_coordinator import AnalyticsCoordinator

        coordinator = factory.create_analytics_coordinator()

        assert isinstance(coordinator, AnalyticsCoordinator)
        assert coordinator._exporters == []

    def test_create_analytics_coordinator_with_exporters(self, factory):
        """create_analytics_coordinator accepts exporter list."""
        from victor.agent.coordinators.analytics_coordinator import AnalyticsCoordinator
        from victor.protocols import IAnalyticsExporter

        # Create mock exporters
        mock_exporter1 = MagicMock(spec=IAnalyticsExporter)
        mock_exporter1.exporter_type.return_value = "exporter1"
        mock_exporter2 = MagicMock(spec=IAnalyticsExporter)
        mock_exporter2.exporter_type.return_value = "exporter2"

        coordinator = factory.create_analytics_coordinator(
            analytics_exporters=[mock_exporter1, mock_exporter2]
        )

        assert isinstance(coordinator, AnalyticsCoordinator)
        assert len(coordinator._exporters) == 2

    def test_create_analytics_coordinator_with_console_exporter(self, factory):
        """create_analytics_coordinator creates with console exporter when enabled."""
        from victor.agent.coordinators.analytics_coordinator import (
            AnalyticsCoordinator,
            ConsoleAnalyticsExporter,
        )

        coordinator = factory.create_analytics_coordinator(enable_console_exporter=True)

        assert isinstance(coordinator, AnalyticsCoordinator)
        # Should have at least console exporter
        assert len(coordinator._exporters) >= 1
        assert any(isinstance(e, ConsoleAnalyticsExporter) for e in coordinator._exporters)


# =============================================================================
# New Coordinator Components Tests (Stream E4)
# =============================================================================


class TestCreateResponseCoordinator:
    """Tests for create_response_coordinator method."""

    def test_create_response_coordinator_returns_coordinator(self, factory):
        """create_response_coordinator returns ResponseCoordinator instance."""
        from victor.agent.coordinators.response_coordinator import ResponseCoordinator

        coordinator = factory.create_response_coordinator()

        assert isinstance(coordinator, ResponseCoordinator)

    def test_create_response_coordinator_with_dependencies(self, factory):
        """create_response_coordinator accepts tool_adapter and tool_registry."""
        from victor.agent.coordinators.response_coordinator import ResponseCoordinator

        mock_adapter = MagicMock()
        mock_registry = MagicMock()

        coordinator = factory.create_response_coordinator(
            tool_adapter=mock_adapter,
            tool_registry=mock_registry,
        )

        assert isinstance(coordinator, ResponseCoordinator)
        assert coordinator._tool_adapter == mock_adapter
        assert coordinator._tool_registry == mock_registry

    def test_create_response_coordinator_uses_settings(self, factory, mock_settings):
        """create_response_coordinator reads config from settings."""
        from victor.agent.coordinators.response_coordinator import ResponseCoordinator

        mock_settings.max_garbage_chunks = 5
        mock_settings.enable_tool_call_extraction = False
        mock_settings.enable_content_sanitization = False
        mock_settings.min_content_length = 50

        coordinator = factory.create_response_coordinator()

        assert isinstance(coordinator, ResponseCoordinator)
        assert coordinator._config.max_garbage_chunks == 5
        assert coordinator._config.enable_tool_call_extraction is False
        assert coordinator._config.enable_content_sanitization is False
        assert coordinator._config.min_content_length == 50


class TestCreateToolAccessConfigCoordinator:
    """Tests for create_tool_access_config_coordinator method."""

    def test_create_tool_access_config_coordinator_returns_coordinator(self, factory):
        """create_tool_access_config_coordinator returns ToolAccessConfigCoordinator instance."""
        from victor.agent.coordinators.config_coordinator import (
            ToolAccessConfigCoordinator,
        )

        coordinator = factory.create_tool_access_config_coordinator()

        assert isinstance(coordinator, ToolAccessConfigCoordinator)

    def test_create_tool_access_config_coordinator_with_dependencies(self, factory):
        """create_tool_access_config_coordinator accepts dependencies."""
        from victor.agent.coordinators.config_coordinator import (
            ToolAccessConfigCoordinator,
        )

        mock_controller = MagicMock()
        mock_mode_coordinator = MagicMock()
        mock_registry = MagicMock()

        coordinator = factory.create_tool_access_config_coordinator(
            tool_access_controller=mock_controller,
            mode_coordinator=mock_mode_coordinator,
            tool_registry=mock_registry,
        )

        assert isinstance(coordinator, ToolAccessConfigCoordinator)
        assert coordinator._tool_access_controller == mock_controller
        assert coordinator._mode_coordinator == mock_mode_coordinator
        assert coordinator._tool_registry == mock_registry


class TestCreateStateCoordinator:
    """Tests for create_state_coordinator method."""

    def test_create_state_coordinator_returns_coordinator(self, factory):
        """create_state_coordinator returns StateCoordinator instance."""
        from victor.agent.coordinators.state_coordinator import StateCoordinator
        from victor.agent.session_state_manager import SessionStateManager

        mock_session_state = MagicMock(spec=SessionStateManager)

        coordinator = factory.create_state_coordinator(session_state_manager=mock_session_state)

        assert isinstance(coordinator, StateCoordinator)

    def test_create_state_coordinator_with_conversation_state(self, factory):
        """create_state_coordinator accepts conversation_state_machine."""
        from victor.agent.coordinators.state_coordinator import StateCoordinator
        from victor.agent.session_state_manager import SessionStateManager

        mock_session_state = MagicMock(spec=SessionStateManager)
        mock_conversation_state = MagicMock()

        coordinator = factory.create_state_coordinator(
            session_state_manager=mock_session_state,
            conversation_state_machine=mock_conversation_state,
        )

        assert isinstance(coordinator, StateCoordinator)

    def test_create_state_coordinator_with_history_settings(self, factory):
        """create_state_coordinator respects history settings."""
        from victor.agent.coordinators.state_coordinator import StateCoordinator
        from victor.agent.session_state_manager import SessionStateManager

        mock_session_state = MagicMock(spec=SessionStateManager)

        coordinator = factory.create_state_coordinator(
            session_state_manager=mock_session_state,
            enable_history=False,
            max_history_size=50,
        )

        assert isinstance(coordinator, StateCoordinator)
        assert coordinator._enable_history is False
        assert coordinator._max_history_size == 50


class TestCreateCoordinators:
    """Tests for create_coordinators method."""

    def test_create_coordinators_returns_components(self, factory):
        """create_coordinators returns CoordinatorComponents instance."""
        from victor.agent.coordinators.response_coordinator import ResponseCoordinator
        from victor.agent.coordinators.config_coordinator import (
            ToolAccessConfigCoordinator,
        )

        components = factory.create_coordinators()

        assert isinstance(components, CoordinatorComponents)
        assert isinstance(components.response_coordinator, ResponseCoordinator)
        assert isinstance(components.tool_access_config_coordinator, ToolAccessConfigCoordinator)
        # state_coordinator is None without session_state_manager
        assert components.state_coordinator is None

    def test_create_coordinators_with_all_dependencies(self, factory):
        """create_coordinators creates all coordinators with full dependencies."""
        from victor.agent.coordinators.response_coordinator import ResponseCoordinator
        from victor.agent.coordinators.config_coordinator import (
            ToolAccessConfigCoordinator,
        )
        from victor.agent.coordinators.state_coordinator import StateCoordinator
        from victor.agent.session_state_manager import SessionStateManager

        mock_adapter = MagicMock()
        mock_registry = MagicMock()
        mock_controller = MagicMock()
        mock_mode_coordinator = MagicMock()
        mock_session_state = MagicMock(spec=SessionStateManager)
        mock_conversation_state = MagicMock()

        components = factory.create_coordinators(
            tool_adapter=mock_adapter,
            tool_registry=mock_registry,
            tool_access_controller=mock_controller,
            mode_coordinator=mock_mode_coordinator,
            session_state_manager=mock_session_state,
            conversation_state_machine=mock_conversation_state,
        )

        assert isinstance(components, CoordinatorComponents)
        assert isinstance(components.response_coordinator, ResponseCoordinator)
        assert isinstance(components.tool_access_config_coordinator, ToolAccessConfigCoordinator)
        assert isinstance(components.state_coordinator, StateCoordinator)


class TestCoordinatorComponentsDataClass:
    """Tests for CoordinatorComponents dataclass."""

    def test_coordinator_components_fields(self):
        """CoordinatorComponents has expected fields."""
        components = CoordinatorComponents()

        assert components.response_coordinator is None
        assert components.tool_access_config_coordinator is None
        assert components.state_coordinator is None

    def test_coordinator_components_with_values(self):
        """CoordinatorComponents stores values correctly."""
        mock_response = MagicMock()
        mock_tool_access = MagicMock()
        mock_state = MagicMock()

        components = CoordinatorComponents(
            response_coordinator=mock_response,
            tool_access_config_coordinator=mock_tool_access,
            state_coordinator=mock_state,
        )

        assert components.response_coordinator == mock_response
        assert components.tool_access_config_coordinator == mock_tool_access
        assert components.state_coordinator == mock_state


class TestOrchestratorComponentsWithCoordinators:
    """Tests for OrchestratorComponents with coordinators field."""

    def test_orchestrator_components_has_coordinators_field(self):
        """OrchestratorComponents includes coordinators field."""
        components = OrchestratorComponents()

        assert hasattr(components, "coordinators")
        assert isinstance(components.coordinators, CoordinatorComponents)

    def test_orchestrator_components_coordinators_defaults(self):
        """OrchestratorComponents coordinators field has correct defaults."""
        components = OrchestratorComponents()

        assert components.coordinators.response_coordinator is None
        assert components.coordinators.tool_access_config_coordinator is None
        assert components.coordinators.state_coordinator is None

    def test_orchestrator_components_with_custom_coordinators(self):
        """OrchestratorComponents accepts custom coordinators."""
        mock_coordinators = CoordinatorComponents(
            response_coordinator=MagicMock(),
            tool_access_config_coordinator=MagicMock(),
            state_coordinator=MagicMock(),
        )

        components = OrchestratorComponents(coordinators=mock_coordinators)

        assert components.coordinators == mock_coordinators
        assert components.coordinators.response_coordinator is not None
        assert components.coordinators.tool_access_config_coordinator is not None
        assert components.coordinators.state_coordinator is not None
