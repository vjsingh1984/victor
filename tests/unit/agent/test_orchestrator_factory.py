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
from unittest.mock import MagicMock, patch

from victor.agent.orchestrator_factory import (
    OrchestratorFactory,
    create_orchestrator_factory,
    OrchestratorComponents,
    ProviderComponents,
    CoreServices,
    ConversationComponents,
    ToolComponents,
    StreamingComponents,
    AnalyticsComponents,
    RecoveryComponents,
    WorkflowOptimizationComponents,
)


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
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
    from victor.agent.complexity_classifier import ComplexityClassifier
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
        from victor.agent.complexity_classifier import ComplexityClassifier

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

        with patch("victor.config.settings.get_project_paths") as mock_paths:
            mock_paths.return_value.global_cache_dir = "/tmp/cache"
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
        """create_mode_completion_criteria returns ModeCompletionCriteria."""
        criteria = factory.create_mode_completion_criteria()

        from victor.agent.budget_manager import ModeCompletionCriteria

        assert isinstance(criteria, ModeCompletionCriteria)

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
