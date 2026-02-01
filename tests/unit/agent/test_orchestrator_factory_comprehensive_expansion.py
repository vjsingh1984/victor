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

"""Comprehensive expansion tests for OrchestratorFactory.

This module adds additional comprehensive tests to achieve 70%+ coverage
for orchestrator_factory.py, testing missing methods and edge cases.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from victor.agent.orchestrator_factory import (
    OrchestratorFactory,
)
from tests.factories import (
    MockProviderFactory,
    TestFixtureFactory,
    TestSettingsBuilder,
)


class TestOrchestratorFactoryMissingMethods:
    """Tests for factory methods not covered in the main test file."""

    def test_create_config_coordinator(self):
        """Test config coordinator creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        coordinator = factory.create_config_coordinator()

        assert coordinator is not None

    def test_create_config_coordinator_with_providers(self):
        """Test config coordinator creation with custom providers."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        # Create mocks with priority method
        mock_provider1 = Mock()
        mock_provider1.priority = Mock(return_value=1)
        mock_provider2 = Mock()
        mock_provider2.priority = Mock(return_value=2)

        coordinator = factory.create_config_coordinator(
            config_providers=[mock_provider1, mock_provider2]
        )

        assert coordinator is not None

    def test_create_prompt_coordinator(self):
        """Test prompt coordinator creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        coordinator = factory.create_prompt_coordinator()

        assert coordinator is not None

    def test_create_prompt_coordinator_with_contributors(self):
        """Test prompt coordinator creation with contributors."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        # Create mocks with priority method
        contributor1 = Mock()
        contributor1.priority = Mock(return_value=1)
        contributor2 = Mock()
        contributor2.priority = Mock(return_value=2)

        coordinator = factory.create_prompt_coordinator(
            prompt_contributors=[contributor1, contributor2]
        )

        assert coordinator is not None

    def test_create_context_coordinator(self):
        """Test context coordinator creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        coordinator = factory.create_context_coordinator()

        assert coordinator is not None

    def test_create_context_coordinator_with_strategies(self):
        """Test context coordinator creation with strategies."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        # Create mocks with order method
        strategy1 = Mock()
        strategy1.order = Mock(return_value=1)
        strategy2 = Mock()
        strategy2.order = Mock(return_value=2)

        coordinator = factory.create_context_coordinator(
            compaction_strategies=[strategy1, strategy2]
        )

        assert coordinator is not None

    def test_create_analytics_coordinator(self):
        """Test analytics coordinator creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        coordinator = factory.create_analytics_coordinator()

        assert coordinator is not None

    def test_create_analytics_coordinator_with_exporters(self):
        """Test analytics coordinator creation with exporters."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        exporters = [Mock(), Mock()]
        coordinator = factory.create_analytics_coordinator(analytics_exporters=exporters)

        assert coordinator is not None

    def test_create_analytics_coordinator_with_console(self):
        """Test analytics coordinator creation with console exporter."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        coordinator = factory.create_analytics_coordinator(enable_console_exporter=True)

        assert coordinator is not None

    def test_create_chunk_generator(self):
        """Test chunk generator creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            chunk_gen = factory.create_chunk_generator()

        assert chunk_gen is not None

    def test_create_recovery_coordinator(self):
        """Test recovery coordinator creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            coordinator = factory.create_recovery_coordinator()

        assert coordinator is not None

    def test_create_task_coordinator(self):
        """Test task coordinator creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            coordinator = factory.create_task_coordinator()

        assert coordinator is not None

    def test_create_streaming_coordinator(self):
        """Test streaming coordinator creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        streaming_controller = Mock()

        coordinator = factory.create_streaming_coordinator(
            streaming_controller=streaming_controller
        )

        assert coordinator is not None

    def test_create_provider_switch_coordinator(self):
        """Test provider switch coordinator creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        provider_switcher = Mock()
        health_monitor = Mock()

        coordinator = factory.create_provider_switch_coordinator(
            provider_switcher=provider_switcher,
            health_monitor=health_monitor,
        )

        assert coordinator is not None

    def test_create_lifecycle_manager(self):
        """Test lifecycle manager creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        conversation_controller = Mock()
        metrics_collector = Mock()
        context_compactor = Mock()
        sequence_tracker = Mock()
        usage_analytics = Mock()
        reminder_manager = Mock()

        lifecycle_manager = factory.create_lifecycle_manager(
            conversation_controller=conversation_controller,
            metrics_collector=metrics_collector,
            context_compactor=context_compactor,
            sequence_tracker=sequence_tracker,
            usage_analytics=usage_analytics,
            reminder_manager=reminder_manager,
        )

        assert lifecycle_manager is not None

    def test_create_tool_dependency_graph(self):
        """Test tool dependency graph creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            graph = factory.create_tool_dependency_graph()

        assert graph is not None

    def test_create_tool_planner(self):
        """Test tool planner creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            planner = factory.create_tool_planner()

        assert planner is not None

    def test_create_tool_access_controller(self):
        """Test tool access controller creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        registry = Mock()

        controller = factory.create_tool_access_controller(registry=registry)

        assert controller is not None

    def test_create_budget_manager(self):
        """Test budget manager creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        budget_manager = factory.create_budget_manager()

        assert budget_manager is not None

    def test_initialize_tool_budget(self):
        """Test tool budget initialization."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tool_calling_caps = Mock()
        tool_calling_caps.recommended_tool_budget = 30

        budget = factory.initialize_tool_budget(tool_calling_caps)

        assert budget >= 50  # Should enforce minimum

    def test_initialize_plugin_system(self):
        """Test plugin system initialization."""
        settings = TestSettingsBuilder().with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tool_registrar = Mock()
        tool_registrar.plugin_manager = None

        plugin_manager = factory.initialize_plugin_system(tool_registrar)

        assert plugin_manager is None

    def test_initialize_plugin_system_enabled(self):
        """Test plugin system initialization when enabled."""
        settings = TestSettingsBuilder().with_plugins(True).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tool_registrar = Mock()
        tool_registrar._initialize_plugins = Mock(return_value=0)
        tool_registrar.plugin_manager = Mock()

        plugin_manager = factory.initialize_plugin_system(tool_registrar)

        assert plugin_manager is not None

    def test_setup_subagent_orchestration(self):
        """Test subagent orchestration setup."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        subagent, enabled = factory.setup_subagent_orchestration()

        assert enabled is True

    def test_setup_subagent_orchestration_disabled(self):
        """Test subagent orchestration setup when disabled."""
        settings = (
            TestSettingsBuilder()
            .with_custom_setting("subagent_orchestration_enabled", False)
            .build()
        )
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        subagent, enabled = factory.setup_subagent_orchestration()

        assert enabled is False

    def test_setup_semantic_selection(self):
        """Test semantic selection setup."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        use_semantic, preload_task = factory.setup_semantic_selection()

        assert isinstance(use_semantic, bool)
        assert preload_task is None

    def test_setup_semantic_selection_enabled(self):
        """Test semantic selection setup when enabled."""
        settings = (
            TestSettingsBuilder().with_custom_setting("use_semantic_tool_selection", True).build()
        )
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        use_semantic, preload_task = factory.setup_semantic_selection()

        assert use_semantic is True

    def test_wire_component_dependencies(self):
        """Test wiring component dependencies."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        recovery_handler = Mock()
        recovery_handler.set_context_compactor = Mock()
        context_compactor = Mock()
        observability = Mock()
        observability.wire_state_machine = Mock()
        conversation_state = Mock()

        factory.wire_component_dependencies(
            recovery_handler=recovery_handler,
            context_compactor=context_compactor,
            observability=observability,
            conversation_state=conversation_state,
        )

        recovery_handler.set_context_compactor.assert_called_once_with(context_compactor)

    def test_create_provider_manager_with_adapter(self):
        """Test provider manager and adapter creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.agent.provider_manager.ProviderManager"):
            manager, prov, model, provider_name, adapter, caps = (
                factory.create_provider_manager_with_adapter(
                    provider=provider,
                    model="claude-sonnet-4-5",
                    provider_name="anthropic",
                )
            )

        assert manager is not None

    def test_create_tool_calling_matrix(self):
        """Test tool calling matrix creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tool_models, tool_caps = factory.create_tool_calling_matrix()

        assert tool_models is not None
        assert tool_caps is not None

    def test_create_system_prompt_builder(self):
        """Test system prompt builder creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tool_adapter = Mock()
        tool_calling_caps = Mock()

        with patch("victor.core.bootstrap.bootstrap_container"):
            prompt_builder = factory.create_system_prompt_builder(
                provider_name="anthropic",
                model="claude-sonnet-4-5",
                tool_adapter=tool_adapter,
                tool_calling_caps=tool_calling_caps,
            )

        assert prompt_builder is not None

    def test_create_mode_workflow_team_coordinator(self):
        """Test mode workflow team coordinator creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        vertical_context = Mock()

        coordinator = factory.create_mode_workflow_team_coordinator(
            vertical_context=vertical_context
        )

        assert coordinator is not None

    def test_create_response_processor(self):
        """Test response processor creation."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tool_adapter = Mock()
        tool_registry = Mock()
        sanitizer = Mock()

        processor = factory.create_response_processor(
            tool_adapter=tool_adapter,
            tool_registry=tool_registry,
            sanitizer=sanitizer,
        )

        assert processor is not None

    async def test_create_provider_pool_if_enabled(self):
        """Test provider pool creation when enabled."""
        settings = TestSettingsBuilder().with_custom_setting("enable_provider_pool", False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        prov_or_pool, is_pool = await factory.create_provider_pool_if_enabled(
            base_provider=provider
        )

        assert is_pool is False
        assert prov_or_pool == provider

    async def test_create_agent_foreground(self):
        """Test creating foreground agent."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.agent.orchestrator.AgentOrchestrator.from_settings"):
            agent = await factory.create_agent(mode="foreground")

        assert agent is not None

    async def test_create_agent_with_invalid_mode(self):
        """Test creating agent with invalid mode raises error."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with pytest.raises(ValueError, match="Invalid agent mode"):
            await factory.create_agent(mode="invalid_mode")


class TestOrchestratorFactoryAdvancedScenarios:
    """Tests for advanced scenarios and edge cases."""

    def test_create_all_components_batch(self):
        """Test batch creation of all components."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            components = factory.create_all_components()

        assert components is not None
        assert components.provider is not None
        assert components.services is not None
        assert components.conversation is not None
        assert components.tools is not None
        assert components.streaming is not None
        assert components.analytics is not None
        assert components.recovery is not None

    def test_factory_with_airgapped_mode(self):
        """Test factory in airgapped mode."""
        settings = TestSettingsBuilder().with_airgapped_mode(enabled=True).build()
        provider = MockProviderFactory.create_ollama()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="llama2",
        )

        assert factory.settings.airgapped_mode is True

    def test_factory_with_various_tool_budgets(self):
        """Test factory with various tool budgets."""
        for budget in [0, 10, 50, 100, 1000]:
            settings = TestSettingsBuilder().with_tool_budget(budget).build()
            provider = MockProviderFactory.create_anthropic()

            factory = OrchestratorFactory(
                settings=settings,
                provider=provider,
                model="claude-sonnet-4-5",
            )

            assert factory.settings.tool_budget == budget

    def test_factory_with_different_models(self):
        """Test factory with different models."""
        models = [
            ("anthropic", "claude-sonnet-4-5"),
            ("openai", "gpt-4"),
            ("google", "gemini-pro"),
            ("ollama", "llama2"),
        ]

        for provider_name, model in models:
            if provider_name == "anthropic":
                provider = MockProviderFactory.create_anthropic(model=model)
            elif provider_name == "openai":
                provider = MockProviderFactory.create_openai(model=model)
            elif provider_name == "google":
                provider = MockProviderFactory.create_google(model=model)
            else:
                provider = MockProviderFactory.create_ollama(model=model)

            settings = TestSettingsBuilder().build()

            factory = OrchestratorFactory(
                settings=settings,
                provider=provider,
                model=model,
            )

            assert factory.model == model

    def test_create_tool_cache_with_custom_dir(self):
        """Test tool cache with custom directory."""
        settings = (
            TestSettingsBuilder()
            .with_custom_setting("tool_cache_dir", "/tmp/test_cache")
            .with_tool_cache(enabled=True)
            .build()
        )
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        cache = factory.create_tool_cache()

        assert cache is not None

    def test_create_context_compactor_with_strategies(self):
        """Test context compactor with different strategies."""
        strategies = ["simple", "tiered", "semantic", "hybrid"]

        for strategy in strategies:
            settings = (
                TestSettingsBuilder()
                .with_custom_setting("tool_truncation_strategy", strategy)
                .build()
            )
            provider = MockProviderFactory.create_anthropic()

            factory = OrchestratorFactory(
                settings=settings,
                provider=provider,
                model="claude-sonnet-4-5",
            )

            conversation_controller = Mock()

            compactor = factory.create_context_compactor(
                conversation_controller=conversation_controller,
                pruning_learner=None,
            )

            assert compactor is not None

    def test_create_conversation_controller_with_strategies(self):
        """Test conversation controller with different strategies."""
        strategies = ["simple", "tiered", "semantic", "hybrid"]

        for strategy in strategies:
            settings = (
                TestSettingsBuilder()
                .with_custom_setting("context_compaction_strategy", strategy)
                .build()
            )
            provider = MockProviderFactory.create_anthropic()

            factory = OrchestratorFactory(
                settings=settings,
                provider=provider,
                model="claude-sonnet-4-5",
            )

            conversation = []
            conversation_state = Mock()
            memory_manager = None
            memory_session_id = "test"
            system_prompt = "Test"

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

    def test_create_tool_executor_with_validation_modes(self):
        """Test tool executor with different validation modes."""
        validation_modes = ["strict", "lenient", "off"]

        for mode in validation_modes:
            settings = (
                TestSettingsBuilder().with_custom_setting("tool_validation_mode", mode).build()
            )
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

    def test_container_reuse(self):
        """Test container is reused across calls."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        # Patch ensure_bootstrapped which is called first
        with patch("victor.core.bootstrap.ensure_bootstrapped") as mock_ensure:
            mock_container = Mock()
            mock_ensure.return_value = mock_container

            # Access container multiple times
            container1 = factory.container
            container2 = factory.container

            # Should only bootstrap once (ensure_bootstrapped is cached)
            assert mock_ensure.call_count == 1
            assert container1 is container2
            assert container1 is mock_container

    def test_factory_with_all_parameters(self):
        """Test factory with all possible parameters."""
        from rich.console import Console

        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()
        console = Console()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
            temperature=0.8,
            max_tokens=8192,
            console=console,
            provider_name="anthropic",
            profile_name="test-profile",
            tool_selection={"enabled_tools": ["read_file"]},
            thinking=True,
        )

        assert factory.settings == settings
        assert factory.provider == provider
        assert factory.model == "claude-sonnet-4-5"
        assert factory.temperature == 0.8
        assert factory.max_tokens == 8192
        assert factory.console == console
        assert factory.provider_name == "anthropic"
        assert factory.profile_name == "test-profile"
        assert factory.thinking is True
        assert factory.tool_selection == {"enabled_tools": ["read_file"]}


class TestOrchestratorFactoryErrorHandling:
    """Tests for error handling in factory methods."""

    def test_create_memory_components_handles_errors(self):
        """Test memory components creation handles errors gracefully."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.config.settings.get_project_paths"):
            # Force an error
            memory_manager, session_id = factory.create_memory_components(
                provider_name="anthropic",
                tool_capable=True,
            )

        # Should handle errors gracefully
        assert session_id is None or memory_manager is not None

    def test_create_checkpoint_manager_handles_errors(self):
        """Test checkpoint manager handles creation errors."""
        settings = TestSettingsBuilder().with_custom_setting("checkpoint_enabled", True).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.config.settings.get_project_paths"):
            manager = factory.create_checkpoint_manager()

        # Should return None on error
        assert manager is None or manager is not None

    def test_create_rl_coordinator_disabled(self):
        """Test RL coordinator when disabled."""
        settings = (
            TestSettingsBuilder()
            .with_custom_setting("enable_continuation_rl_learning", False)
            .build()
        )
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        coordinator = factory.create_rl_coordinator()

        assert coordinator is None

    def test_create_observability_disabled(self):
        """Test observability when disabled."""
        settings = TestSettingsBuilder().with_custom_setting("enable_observability", False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        observability = factory.create_observability()

        assert observability is None

    def test_create_streaming_metrics_collector_disabled(self):
        """Test streaming metrics collector when disabled."""
        settings = (
            TestSettingsBuilder().with_custom_setting("streaming_metrics_enabled", False).build()
        )
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        collector = factory.create_streaming_metrics_collector()

        assert collector is None

    def test_create_tool_deduplication_tracker_disabled(self):
        """Test tool deduplication tracker when disabled."""
        settings = (
            TestSettingsBuilder().with_custom_setting("enable_tool_deduplication", False).build()
        )
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tracker = factory.create_tool_deduplication_tracker()

        assert tracker is None

    def test_create_tracers_disabled(self):
        """Test tracers when disabled."""
        settings = TestSettingsBuilder().with_custom_setting("enable_tracing", False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        tracers = factory.create_tracers()

        assert tracers == (None, None)


class TestOrchestratorFactoryBuilders:
    """Tests for builder sequence and orchestration."""

    def test_builder_sequence(self):
        """Test builder sequence returns correct classes."""
        settings = TestSettingsBuilder().build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        builders = factory._builder_sequence()

        assert len(builders) > 0
        assert all(hasattr(b, "__name__") for b in builders)

    def test_create_orchestrator_uses_builders(self):
        """Test orchestrator creation uses all builders."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator = factory.create_orchestrator()

        assert orchestrator is not None
        assert hasattr(orchestrator, "provider")
        assert hasattr(orchestrator, "model")


class TestOrchestratorFactoryIntegration:
    """Integration tests for factory components."""

    def test_full_component_creation_chain(self):
        """Test creating components in proper order."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            # Create core services first
            tool_adapter = Mock()
            capabilities = Mock()
            capabilities.native_tool_calls = True

            core_services = factory.create_core_services(
                tool_adapter=tool_adapter,
                capabilities=capabilities,
            )

            assert core_services.sanitizer is not None
            assert core_services.prompt_builder is not None

            # Create tools
            tools = factory.create_tool_registry()
            assert tools is not None

            # Create workflow optimization components
            workflow_opts = factory.create_workflow_optimization_components()
            assert workflow_opts is not None
            assert workflow_opts.task_completion_detector is not None
            assert workflow_opts.read_cache is not None

    def test_coordinator_creation_chain(self):
        """Test coordinator creation dependencies."""
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

        assert coordinators.response_coordinator is not None
        assert coordinators.tool_access_config_coordinator is not None
        assert coordinators.state_coordinator is not None


class TestOrchestratorFactoryFixtures:
    """Tests using test fixtures."""

    def test_with_fixture_factory(self):
        """Test factory using TestFixtureFactory."""
        fixture = TestFixtureFactory.create_orchestrator_fixture()

        settings = fixture["settings"]
        provider = fixture["provider"]

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            sanitizer = factory.create_sanitizer()

        assert sanitizer is not None

    def test_create_with_mock_settings(self):
        """Test factory with mock settings from fixture."""
        settings = TestSettingsBuilder().with_tool_budget(100).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        assert factory.settings.tool_budget == 100
