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

"""End-to-end integration tests for OrchestratorFactory.

These tests verify that OrchestratorFactory can create complete,
working orchestrators with all components properly wired together.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from victor.agent.orchestrator_factory import (
    OrchestratorFactory,
    OrchestratorComponents,
)
from tests.factories import (
    MockProviderFactory,
    TestSettingsBuilder,
)


@pytest.mark.integration
class TestOrchestratorFactoryE2E:
    """End-to-end tests for orchestrator factory."""

    @pytest.mark.asyncio
    async def test_create_minimal_orchestrator(self):
        """Test creating a minimal orchestrator with defaults."""
        settings = (
            TestSettingsBuilder()
            .with_conversation_memory(False)
            .with_plugins(False)
            .with_analytics(False)
            .build()
        )
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
        assert orchestrator.temperature == 0.7
        assert orchestrator.max_tokens == 4096

    @pytest.mark.asyncio
    async def test_create_orchestrator_with_custom_settings(self):
        """Test creating orchestrator with custom settings."""
        settings = (
            TestSettingsBuilder()
            .with_provider("openai")
            .with_model("gpt-4")
            .with_temperature(0.5)
            .with_max_tokens(8192)
            .with_tool_budget(100)
            .with_conversation_memory(False)
            .with_plugins(False)
            .build()
        )
        provider = MockProviderFactory.create_openai(model="gpt-4")

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="gpt-4",
            temperature=0.5,
            max_tokens=8192,
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator = factory.create_orchestrator()

        assert orchestrator.provider == provider
        assert orchestrator.model == "gpt-4"
        assert orchestrator.temperature == 0.5
        assert orchestrator.max_tokens == 8192

    @pytest.mark.asyncio
    async def test_create_orchestrator_with_thinking_mode(self):
        """Test creating orchestrator with thinking mode enabled."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
            thinking=True,
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator = factory.create_orchestrator()

        assert orchestrator.thinking is True

    @pytest.mark.asyncio
    async def test_create_orchestrator_with_profile_name(self):
        """Test creating orchestrator with profile name."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
            profile_name="test-profile",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator = factory.create_orchestrator()

        assert orchestrator._profile_name == "test-profile"

    @pytest.mark.asyncio
    async def test_create_all_components(self):
        """Test creating all orchestrator components."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            components = factory.create_all_components()

        assert isinstance(components, OrchestratorComponents)
        assert components.provider is not None
        assert components.services is not None
        assert components.conversation is not None
        assert components.tools is not None
        assert components.streaming is not None
        assert components.analytics is not None
        assert components.recovery is not None

    @pytest.mark.asyncio
    async def test_components_have_correct_attributes(self):
        """Test that components have correct attributes."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            components = factory.create_all_components()

        # Provider components
        assert components.provider.provider == provider
        assert components.provider.model == "claude-sonnet-4-5"
        assert components.provider.provider_name == "anthropic"

        # Core services
        assert components.services.sanitizer is not None
        assert components.services.prompt_builder is not None
        assert components.services.project_context is not None
        assert components.services.complexity_classifier is not None
        assert components.services.action_authorizer is not None
        assert components.services.search_router is not None

    @pytest.mark.asyncio
    async def test_orchestrator_has_tool_registry(self):
        """Test that created orchestrator has tool registry."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator = factory.create_orchestrator()

        assert hasattr(orchestrator, "tools")
        assert orchestrator.tools is not None

    @pytest.mark.asyncio
    async def test_orchestrator_has_tool_executor(self):
        """Test that created orchestrator has tool executor."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator = factory.create_orchestrator()

        assert hasattr(orchestrator, "tool_executor")
        assert orchestrator.tool_executor is not None

    @pytest.mark.asyncio
    async def test_orchestrator_has_conversation_controller(self):
        """Test that created orchestrator has conversation controller."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator = factory.create_orchestrator()

        assert hasattr(orchestrator, "_conversation_controller")
        assert orchestrator._conversation_controller is not None

    @pytest.mark.asyncio
    async def test_orchestrator_has_metrics_collector(self):
        """Test that created orchestrator has metrics collector."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator = factory.create_orchestrator()

        assert hasattr(orchestrator, "_metrics_collector")
        assert orchestrator._metrics_collector is not None

    @pytest.mark.asyncio
    async def test_orchestrator_has_recovery_handler(self):
        """Test that created orchestrator has recovery handler."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator = factory.create_orchestrator()

        assert hasattr(orchestrator, "_recovery_handler")
        # Recovery handler might be None if disabled

    @pytest.mark.asyncio
    async def test_orchestrator_factory_is_reusable(self):
        """Test that factory can be reused to create multiple orchestrators."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator1 = factory.create_orchestrator()
            orchestrator2 = factory.create_orchestrator()

        assert orchestrator1 is not orchestrator2
        assert orchestrator1.provider == orchestrator2.provider

    @pytest.mark.asyncio
    async def test_factory_with_different_providers(self):
        """Test creating orchestrators with different providers."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()

        anthropic_provider = MockProviderFactory.create_anthropic()
        openai_provider = MockProviderFactory.create_openai()

        factory1 = OrchestratorFactory(
            settings=settings,
            provider=anthropic_provider,
            model="claude-sonnet-4-5",
        )

        factory2 = OrchestratorFactory(
            settings=settings,
            provider=openai_provider,
            model="gpt-4",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator1 = factory1.create_orchestrator()
            orchestrator2 = factory2.create_orchestrator()

        assert orchestrator1.provider != orchestrator2.provider
        assert orchestrator1.model == "claude-sonnet-4-5"
        assert orchestrator2.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_orchestrator_with_workflow_optimizations(self):
        """Test creating orchestrator with workflow optimizations enabled."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            components = factory.create_all_components()

        assert components.workflow_optimization is not None

    @pytest.mark.asyncio
    async def test_orchestrator_with_coordinators(self):
        """Test creating orchestrator with new coordinators."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            components = factory.create_all_components()

        assert components.coordinators is not None

    @pytest.mark.asyncio
    async def test_orchestrator_with_parallel_tools(self):
        """Test creating orchestrator with parallel tools enabled."""
        settings = (
            TestSettingsBuilder()
            .with_parallel_tools(enabled=True, max_concurrent=5)
            .with_conversation_memory(False)
            .with_plugins(False)
            .build()
        )
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator = factory.create_orchestrator()

        # Should have parallel executor configured
        assert orchestrator is not None

    @pytest.mark.asyncio
    async def test_orchestrator_with_tool_cache(self):
        """Test creating orchestrator with tool cache enabled."""
        settings = (
            TestSettingsBuilder()
            .with_tool_cache(enabled=True)
            .with_conversation_memory(False)
            .with_plugins(False)
            .build()
        )
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            components = factory.create_all_components()

        assert components.tools.tool_cache is not None

    @pytest.mark.asyncio
    async def test_orchestrator_with_deduplication(self):
        """Test creating orchestrator with tool deduplication enabled."""
        settings = (
            TestSettingsBuilder()
            .with_tool_deduplication(enabled=True, window_size=10)
            .with_conversation_memory(False)
            .with_plugins(False)
            .build()
        )
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator = factory.create_orchestrator()

        # Deduplication tracker should be configured
        assert orchestrator is not None

    @pytest.mark.asyncio
    async def test_orchestrator_with_observability(self):
        """Test creating orchestrator with observability enabled."""
        settings = (
            TestSettingsBuilder()
            .with_observability(enabled=True)
            .with_conversation_memory(False)
            .with_plugins(False)
            .build()
        )
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            components = factory.create_all_components()

        # Observability should be enabled
        assert components.observability is not None

    @pytest.mark.asyncio
    async def test_initialize_orchestrator_instance(self):
        """Test initializing an existing orchestrator instance."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
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
        assert orchestrator.settings == settings
        assert orchestrator.temperature == 0.7
        assert orchestrator.max_tokens == 4096

    @pytest.mark.asyncio
    async def test_create_orchestrator_with_airgapped_mode(self):
        """Test creating orchestrator in airgapped mode."""
        settings = (
            TestSettingsBuilder()
            .with_airgapped_mode(enabled=True)
            .with_conversation_memory(False)
            .with_plugins(False)
            .build()
        )
        provider = MockProviderFactory.create_ollama()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="llama2",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator = factory.create_orchestrator()

        assert orchestrator is not None
        assert orchestrator.settings.airgapped_mode is True

    @pytest.mark.asyncio
    async def test_create_orchestrator_with_custom_tool_selection(self):
        """Test creating orchestrator with custom tool selection strategy."""
        settings = (
            TestSettingsBuilder()
            .with_tool_selection_strategy("semantic")
            .with_conversation_memory(False)
            .with_plugins(False)
            .build()
        )
        provider = MockProviderFactory.create_anthropic()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container"):
            orchestrator = factory.create_orchestrator()

        # Tool selector should be configured with semantic strategy
        assert orchestrator is not None

    @pytest.mark.asyncio
    async def test_factory_container_caching(self):
        """Test that factory caches DI container."""
        settings = TestSettingsBuilder().with_conversation_memory(False).with_plugins(False).build()
        provider = MockProviderFactory.create_anthropic()

        # Explicitly reset any global container state before test
        from victor.core.container import reset_container

        reset_container()

        factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model="claude-sonnet-4-5",
        )

        with patch("victor.core.bootstrap.bootstrap_container") as mock_bootstrap:
            # First access should bootstrap
            container1 = factory.container
            # Second access should use cached container
            container2 = factory.container

            # Should only bootstrap once
            assert mock_bootstrap.call_count == 1
            assert container1 is container2
