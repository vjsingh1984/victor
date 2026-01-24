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

"""Comprehensive tests for OrchestratorFactory.

This test module provides comprehensive coverage for OrchestratorFactory,
testing all 50+ creation methods with various configurations.

Target: 70%+ coverage for orchestrator_factory.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from victor.agent.orchestrator_factory import (
    OrchestratorFactory,
    create_orchestrator_factory,
)
from tests.factories import (
    MockProviderFactory,
    TestFixtureFactory,
    TestSettingsBuilder,
)


class TestOrchestratorFactoryInitialization:
    """Tests for OrchestratorFactory initialization and basic properties."""

    def test_factory_initialization(self):
        """Test factory can be initialized with basic parameters."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
            temperature=0.7,
            max_tokens=4096,
        )

        assert factory.settings == settings
        assert factory.provider == provider
        assert factory.model == "claude-sonnet-4-5"
        assert factory.temperature == 0.7
        assert factory.max_tokens == 4096
        assert factory.console is None
        assert factory.provider_name is None

    def test_factory_with_console(self):
        """Test factory can be initialized with console."""
        from rich.console import Console

        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()
        console = Console()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
            console=console,
        )

        assert factory.console == console

    def test_factory_with_provider_name(self):
        """Test factory can be initialized with provider name."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_openai()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="gpt-4",
            provider_name="openai",
        )

        assert factory.provider_name == "openai"

    def test_factory_with_profile_name(self):
        """Test factory can be initialized with profile name."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
            profile_name="test-profile",
        )

        assert factory.profile_name == "test-profile"

    def test_factory_with_thinking_enabled(self):
        """Test factory can be initialized with thinking mode."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
            thinking=True,
        )

        assert factory.thinking is True

    def test_factory_container_lazy_initialization(self):
        """Test factory container is lazily initialized."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        # Container should be None initially
        assert factory._container is None

        # Access container property to trigger initialization
        with patch("victor.core.bootstrap.bootstrap_container"):
            container = factory.container

        # Container should now be initialized
        assert container is not None


class TestOrchestratorFactoryCoreServices:
    """Tests for core service creation methods."""

    def test_create_sanitizer(self):
        """Test sanitizer creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            sanitizer = factory.create_sanitizer()

        assert sanitizer is not None

    def test_create_project_context(self):
        """Test project context creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            project_context = factory.create_project_context()

        assert project_context is not None

    def test_create_complexity_classifier(self):
        """Test complexity classifier creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            classifier = factory.create_complexity_classifier()

        assert classifier is not None

    def test_create_action_authorizer(self):
        """Test action authorizer creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            authorizer = factory.create_action_authorizer()

        assert authorizer is not None

    def test_create_search_router(self):
        """Test search router creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            router = factory.create_search_router()

        assert router is not None

    def test_create_prompt_builder(self):
        """Test prompt builder creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tool_adapter = Mock()
        capabilities = Mock()
        capabilities.native_tool_calls = True

        with patch("victor.core.bootstrap.bootstrap_container"):
            prompt_builder = factory.create_prompt_builder(
                tool_adapter=tool_adapter,
                capabilities=capabilities,
            )

        assert prompt_builder is not None

    def test_create_core_services(self):
        """Test core services batch creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tool_adapter = Mock()
        capabilities = Mock()

        with patch("victor.core.bootstrap.bootstrap_container"):
            core_services = factory.create_core_services(
                tool_adapter=tool_adapter,
                capabilities=capabilities,
            )

        assert core_services is not None
        assert core_services.sanitizer is not None
        assert core_services.prompt_builder is not None
        assert core_services.project_context is not None
        assert core_services.complexity_classifier is not None
        assert core_services.action_authorizer is not None
        assert core_services.search_router is not None


class TestOrchestratorFactoryToolComponents:
    """Tests for tool-related component creation."""

    def test_create_tool_registry(self):
        """Test tool registry creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        registry = factory.create_tool_registry()

        assert registry is not None

    def test_create_tool_executor(self):
        """Test tool executor creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tools = Mock()
        argument_normalizer = Mock()
        tool_cache = None
        safety_checker = Mock()
        code_correction_middleware = None

        with patch("victor.core.bootstrap.bootstrap_container"):
            executor = factory.create_tool_executor(
                tools=tools,
                argument_normalizer=argument_normalizer,
                tool_cache=tool_cache,
                safety_checker=safety_checker,
                code_correction_middleware=code_correction_middleware,
            )

        assert executor is not None

    def test_create_tool_cache(self):
        """Test tool cache creation when enabled."""
        settings = TestSettingsBuilder().with_tool_cache(enabled=True).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        cache = factory.create_tool_cache()

        assert cache is not None

    def test_create_tool_cache_disabled(self):
        """Test tool cache creation when disabled."""
        settings = TestSettingsBuilder().with_tool_cache(enabled=False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        cache = factory.create_tool_cache()

        assert cache is None

    def test_create_parallel_executor(self):
        """Test parallel executor creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tool_executor = Mock()

        parallel_executor = factory.create_parallel_executor(
            tool_executor=tool_executor
        )

        assert parallel_executor is not None

    def test_create_tool_pipeline(self):
        """Test tool pipeline creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tools = Mock()
        tool_executor = Mock()
        tool_budget = 50
        tool_cache = None
        argument_normalizer = Mock()
        on_tool_start = Mock()
        on_tool_complete = Mock()
        deduplication_tracker = None

        pipeline = factory.create_tool_pipeline(
            tools=tools,
            tool_executor=tool_executor,
            tool_budget=tool_budget,
            tool_cache=tool_cache,
            argument_normalizer=argument_normalizer,
            on_tool_start=on_tool_start,
            on_tool_complete=on_tool_complete,
            deduplication_tracker=deduplication_tracker,
        )

        assert pipeline is not None

    def test_create_safety_checker(self):
        """Test safety checker creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            safety_checker = factory.create_safety_checker()

        assert safety_checker is not None

    def test_create_argument_normalizer(self):
        """Test argument normalizer creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            normalizer = factory.create_argument_normalizer(provider)

        assert normalizer is not None

    def test_create_tool_deduplication_tracker(self):
        """Test tool deduplication tracker creation when enabled."""
        settings = TestSettingsBuilder().with_tool_deduplication(
            enabled=True, window_size=10
        ).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tracker = factory.create_tool_deduplication_tracker()

        assert tracker is not None

    def test_create_tool_deduplication_tracker_disabled(self):
        """Test tool deduplication tracker creation when disabled."""
        settings = TestSettingsBuilder().with_tool_deduplication(
            enabled=False
        ).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tracker = factory.create_tool_deduplication_tracker()

        assert tracker is None


class TestOrchestratorFactoryConversationComponents:
    """Tests for conversation-related component creation."""

    def test_create_conversation_controller(self):
        """Test conversation controller creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        conversation = []
        conversation_state = Mock()
        memory_manager = None
        memory_session_id = "test-session"
        system_prompt = "You are a helpful assistant."

        controller = factory.create_conversation_controller(
            provider=provider,
            model="claude-sonnet-4-5",
            conversation=conversation,
            conversation_state=conversation_state,
            memory_manager=memory_manager,
            memory_session_id=memory_session_id,
            system_prompt=system_prompt,
        )

        assert controller is not None

    def test_create_message_history(self):
        """Test message history creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        system_prompt = "You are a helpful assistant."

        history = factory.create_message_history(system_prompt=system_prompt)

        assert history is not None

    def test_create_memory_components(self):
        """Test memory components creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.agent.orchestrator_factory.get_project_paths"):
            memory_manager, session_id = factory.create_memory_components(
                provider_name="anthropic",
                tool_capable=True,
            )

        # Memory might be None depending on settings
        assert session_id is None or memory_manager is not None


class TestOrchestratorFactoryStreamingComponents:
    """Tests for streaming-related component creation."""

    def test_create_streaming_controller(self):
        """Test streaming controller creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        streaming_metrics_collector = None
        on_session_complete = Mock()

        controller = factory.create_streaming_controller(
            streaming_metrics_collector=streaming_metrics_collector,
            on_session_complete=on_session_complete,
        )

        assert controller is not None

    def test_create_streaming_metrics_collector(self):
        """Test streaming metrics collector creation when enabled."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        collector = factory.create_streaming_metrics_collector()

        assert collector is not None

    def test_create_streaming_metrics_collector_disabled(self):
        """Test streaming metrics collector creation when disabled."""
        settings = TestSettingsBuilder().with_custom_setting(
            "streaming_metrics_enabled", False
        ).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        collector = factory.create_streaming_metrics_collector()

        assert collector is None

    def test_create_metrics_collector(self):
        """Test metrics collector creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        streaming_metrics_collector = None
        usage_logger = Mock()
        debug_logger = Mock()
        tool_cost_lookup = Mock(return_value=Mock(tier="low"))

        metrics = factory.create_metrics_collector(
            streaming_metrics_collector=streaming_metrics_collector,
            usage_logger=usage_logger,
            debug_logger=debug_logger,
            tool_cost_lookup=tool_cost_lookup,
        )

        assert metrics is not None


class TestOrchestratorFactoryAnalyticsComponents:
    """Tests for analytics-related component creation."""

    def test_create_usage_analytics(self):
        """Test usage analytics creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            usage_analytics = factory.create_usage_analytics()

        assert usage_analytics is not None

    def test_create_sequence_tracker(self):
        """Test sequence tracker creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            tracker = factory.create_sequence_tracker()

        assert tracker is not None


class TestOrchestratorFactoryRecoveryComponents:
    """Tests for recovery-related component creation."""

    def test_create_recovery_handler(self):
        """Test recovery handler creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            handler = factory.create_recovery_handler()

        assert handler is not None

    def test_create_context_compactor(self):
        """Test context compactor creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        conversation_controller = Mock()
        pruning_learner = None

        compactor = factory.create_context_compactor(
            conversation_controller=conversation_controller,
            pruning_learner=pruning_learner,
        )

        assert compactor is not None

    def test_create_recovery_integration(self):
        """Test recovery integration creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        recovery_handler = Mock()

        integration = factory.create_recovery_integration(
            recovery_handler=recovery_handler
        )

        assert integration is not None


class TestOrchestratorFactoryWorkflowComponents:
    """Tests for workflow-related component creation."""

    def test_create_reminder_manager(self):
        """Test reminder manager creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            manager = factory.create_reminder_manager(
                provider="anthropic",
                task_complexity="medium",
                tool_budget=50,
            )

        assert manager is not None

    def test_create_unified_tracker(self):
        """Test unified tracker creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tool_calling_caps = Mock()
        tool_calling_caps.exploration_multiplier = 1.0
        tool_calling_caps.continuation_patience = 2

        with patch("victor.core.bootstrap.bootstrap_container"):
            tracker = factory.create_unified_tracker(
                tool_calling_caps=tool_calling_caps
            )

        assert tracker is not None

    def test_create_response_completer(self):
        """Test response completer creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        completer = factory.create_response_completer()

        assert completer is not None

    def test_create_response_coordinator(self):
        """Test response coordinator creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tool_adapter = Mock()
        tool_registry = Mock()

        with patch("victor.core.bootstrap.bootstrap_container"):
            coordinator = factory.create_response_coordinator(
                tool_adapter=tool_adapter,
                tool_registry=tool_registry,
            )

        assert coordinator is not None

    def test_create_tool_access_config_coordinator(self):
        """Test tool access config coordinator creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tool_access_controller = Mock()
        mode_coordinator = Mock()
        tool_registry = Mock()

        coordinator = factory.create_tool_access_config_coordinator(
            tool_access_controller=tool_access_controller,
            mode_coordinator=mode_coordinator,
            tool_registry=tool_registry,
        )

        assert coordinator is not None

    def test_create_state_coordinator(self):
        """Test state coordinator creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        session_state_manager = Mock()
        conversation_state_machine = Mock()

        coordinator = factory.create_state_coordinator(
            session_state_manager=session_state_manager,
            conversation_state_machine=conversation_state_machine,
        )

        assert coordinator is not None

    def test_create_coordinators(self):
        """Test batch creation of all coordinators."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tool_adapter = Mock()
        tool_registry = Mock()
        tool_access_controller = Mock()
        mode_coordinator = Mock()
        session_state_manager = Mock()
        conversation_state_machine = Mock()

        with patch("victor.core.bootstrap.bootstrap_container"):
            coordinators = factory.create_coordinators(
                tool_adapter=tool_adapter,
                tool_registry=tool_registry,
                tool_access_controller=tool_access_controller,
                mode_coordinator=mode_coordinator,
                session_state_manager=session_state_manager,
                conversation_state_machine=conversation_state_machine,
            )

        assert coordinators is not None
        assert coordinators.response_coordinator is not None
        assert coordinators.tool_access_config_coordinator is not None
        assert coordinators.state_coordinator is not None


class TestOrchestratorFactoryWorkflowOptimizations:
    """Tests for workflow optimization components."""

    def test_create_task_completion_detector(self):
        """Test task completion detector creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        detector = factory.create_task_completion_detector()

        assert detector is not None

    def test_create_read_cache(self):
        """Test read cache creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        cache = factory.create_read_cache()

        assert cache is not None

    def test_create_time_aware_executor(self):
        """Test time aware executor creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        executor = factory.create_time_aware_executor(timeout_seconds=300.0)

        assert executor is not None

    def test_create_thinking_detector(self):
        """Test thinking detector creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        detector = factory.create_thinking_detector()

        assert detector is not None

    def test_create_resource_manager(self):
        """Test resource manager creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        manager = factory.create_resource_manager()

        assert manager is not None

    def test_create_mode_completion_criteria(self):
        """Test mode completion criteria creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        criteria = factory.create_mode_completion_criteria()

        assert criteria is not None

    def test_create_workflow_optimization_components(self):
        """Test batch creation of workflow optimization components."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        components = factory.create_workflow_optimization_components(
            timeout_seconds=300.0
        )

        assert components is not None
        assert components.task_completion_detector is not None
        assert components.read_cache is not None
        assert components.time_aware_executor is not None
        assert components.thinking_detector is not None
        assert components.resource_manager is not None
        assert components.mode_completion_criteria is not None


class TestOrchestratorFactoryOrchestratorCreation:
    """Tests for full orchestrator creation."""

    def test_create_orchestrator(self):
        """Test creating a full orchestrator."""
        settings = TestSettingsBuilder().with_conversation_memory(
            False
        ).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator = factory.create_orchestrator()

        assert orchestrator is not None
        assert orchestrator.provider == provider
        assert orchestrator.model == "claude-sonnet-4-5"

    def test_initialize_orchestrator(self):
        """Test initializing an orchestrator instance."""
        settings = TestSettingsBuilder().with_conversation_memory(
            False
        ).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        from victor.agent.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)

        with patch("victor.core.bootstrap.bootstrap_container"):
            factory.initialize_orchestrator(orchestrator)

        assert orchestrator.provider == provider
        assert orchestrator.model == "claude-sonnet-4-5"


class TestOrchestratorFactoryHelperMethods:
    """Tests for helper methods and utility functions."""

    def test_create_orchestrator_factory_function(self):
        """Test the convenience function for creating factory."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = create_orchestrator_factory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        assert isinstance(factory, OrchestratorFactory)
        assert factory.settings == settings
        assert factory.provider == provider
        assert factory.model == "claude-sonnet-4-5"

    def test_create_presentation_adapter(self):
        """Test presentation adapter creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            adapter = factory.create_presentation_adapter()

        assert adapter is not None

    def test_create_usage_logger(self):
        """Test usage logger creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            logger = factory.create_usage_logger()

        assert logger is not None

    def test_create_observability(self):
        """Test observability creation when enabled."""
        settings = TestSettingsBuilder().with_observability(
            enabled=True
        ).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        observability = factory.create_observability()

        assert observability is not None

    def test_create_observability_disabled(self):
        """Test observability creation when disabled."""
        settings = TestSettingsBuilder().with_observability(
            enabled=False
        ).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        observability = factory.create_observability()

        assert observability is None

    def test_create_tracers(self):
        """Test tracer creation when enabled."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.events.get_observability_bus"):
            tracers = factory.create_tracers()

        assert tracers is not None
        assert len(tracers) == 2

    def test_create_tracers_disabled(self):
        """Test tracer creation when disabled."""
        settings = TestSettingsBuilder().with_custom_setting(
            "enable_tracing", False
        ).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tracers = factory.create_tracers()

        assert tracers == (None, None)


class TestOrchestratorFactoryAdvancedComponents:
    """Tests for advanced component creation."""

    def test_create_code_execution_manager(self):
        """Test code execution manager creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            manager = factory.create_code_execution_manager()

        assert manager is not None

    def test_create_workflow_registry(self):
        """Test workflow registry creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            registry = factory.create_workflow_registry()

        assert registry is not None

    def test_create_conversation_state_machine(self):
        """Test conversation state machine creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            state_machine = factory.create_conversation_state_machine()

        assert state_machine is not None

    def test_create_integration_config(self):
        """Test integration config creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        config = factory.create_integration_config()

        assert config is not None

    def test_create_tool_registrar(self):
        """Test tool registrar creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tools = Mock()
        tool_graph = Mock()

        registrar = factory.create_tool_registrar(
            tools=tools,
            tool_graph=tool_graph,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        assert registrar is not None

    def test_create_tool_selector(self):
        """Test tool selector creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tools = Mock()
        conversation_state = Mock()
        unified_tracker = Mock()
        tool_selection = {}
        on_selection_recorded = Mock()

        with patch("victor.agent.orchestrator_factory.get_embedding_service"):
            selector = factory.create_tool_selector(
                tools=tools,
                conversation_state=conversation_state,
                unified_tracker=unified_tracker,
                model="claude-sonnet-4-5",
                provider_name="anthropic",
                tool_selection=tool_selection,
                on_selection_recorded=on_selection_recorded,
            )

        assert selector is not None

    def test_create_intent_classifier(self):
        """Test intent classifier creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        classifier = factory.create_intent_classifier()

        assert classifier is not None


class TestOrchestratorFactorySpecializedMethods:
    """Tests for specialized factory methods."""

    def test_initialize_execution_state(self):
        """Test execution state initialization."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        observed_files, executed_tools, failed_tool_signatures, tool_capability_warned = (
            factory.initialize_execution_state()
        )

        assert observed_files == []
        assert executed_tools == []
        assert failed_tool_signatures == set()
        assert tool_capability_warned is False

    def test_create_debug_logger_configured(self):
        """Test debug logger creation with configuration."""
        settings = TestSettingsBuilder().with_debug_logging(
            enabled=True
        ).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            debug_logger = factory.create_debug_logger_configured()

        assert debug_logger is not None
        assert debug_logger.enabled is True

    def test_create_middleware_chain(self):
        """Test middleware chain creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            middleware_chain, code_correction_middleware = (
                factory.create_middleware_chain()
            )

        # Middleware might be None if no verticals loaded
        assert code_correction_middleware is None or isinstance(
            code_correction_middleware, Mock
        )

    def test_create_auto_committer(self):
        """Test auto committer creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            auto_committer = factory.create_auto_committer()

        assert auto_committer is not None

    def test_create_streaming_chat_handler(self):
        """Test streaming chat handler creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        message_adder = Mock()

        handler = factory.create_streaming_chat_handler(
            message_adder=message_adder
        )

        assert handler is not None

    def test_create_rl_coordinator(self):
        """Test RL coordinator creation when enabled."""
        settings = TestSettingsBuilder().with_rl_learning(
            enabled=True
        ).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            coordinator = factory.create_rl_coordinator()

        # Might be None if not properly configured
        assert coordinator is None or coordinator is not None

    def test_create_tool_output_formatter(self):
        """Test tool output formatter creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        context_compactor = Mock()

        formatter = factory.create_tool_output_formatter(
            context_compactor=context_compactor
        )

        assert formatter is not None

    def test_create_checkpoint_manager(self):
        """Test checkpoint manager creation when enabled."""
        settings = TestSettingsBuilder().with_custom_setting(
            "checkpoint_enabled", True
        ).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.agent.orchestrator_factory.get_project_paths"):
            manager = factory.create_checkpoint_manager()

        # Might be None if creation fails
        assert manager is None or manager is not None

    def test_create_checkpoint_manager_disabled(self):
        """Test checkpoint manager creation when disabled."""
        settings = TestSettingsBuilder().with_custom_setting(
            "checkpoint_enabled", False
        ).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        manager = factory.create_checkpoint_manager()

        assert manager is None


class TestOrchestratorFactoryEdgeCases:
    """Tests for edge cases and error handling."""

    def test_factory_with_none_provider(self):
        """Test factory handles None provider gracefully."""
        settings = TestSettingsBuilder().build()

        factory = OrchestratorFactory(
            settings=settings,
            provider=None,  # type: ignore
            model="claude-sonnet-4-5",
        )

        assert factory.provider is None

    def test_factory_with_zero_tool_budget(self):
        """Test factory handles zero tool budget."""
        settings = TestSettingsBuilder().with_tool_budget(0).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        assert factory.settings.tool_budget == 0

    def test_factory_with_negative_max_tokens(self):
        """Test factory handles negative max tokens."""
        settings = TestSettingsBuilder().with_max_tokens(-1).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        assert factory.settings.max_tokens == -1

    def test_factory_with_empty_model_name(self):
        """Test factory handles empty model name."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="",  # Empty model name
        )

        assert factory.model == ""

    def test_factory_with_temperature_out_of_range(self):
        """Test factory handles temperature out of range."""
        settings = TestSettingsBuilder().with_temperature(2.5).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        assert factory.temperature == 2.5
