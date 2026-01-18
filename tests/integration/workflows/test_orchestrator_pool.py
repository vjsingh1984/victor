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

"""Integration tests for OrchestratorPool.

Tests multi-provider workflow functionality including:
- Default orchestrator retrieval
- Profile-based orchestrator creation
- Orchestrator caching and reuse
- Multiple profiles management
- Error handling
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock

from victor.config.settings import Settings, ProfileConfig
from victor.workflows.orchestrator_pool import OrchestratorPool


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_settings():
    """Create mock settings with test profiles."""
    settings = Mock(spec=Settings)

    # Create test profiles
    profiles = {
        "default": ProfileConfig(
            provider="ollama",
            model_name="qwen3-coder:30b",
            temperature=0.7,
            max_tokens=4096,
            description="Default profile",
        ),
        "anthropic": ProfileConfig(
            provider="anthropic",
            model_name="claude-sonnet-4-5",
            temperature=0.7,
            max_tokens=8192,
            description="Anthropic profile",
        ),
        "openai": ProfileConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=4096,
            description="OpenAI profile",
        ),
    }

    settings.load_profiles.return_value = profiles
    settings.get_provider_settings.return_value = {}

    return settings


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator."""
    orchestrator = Mock()
    orchestrator.close = Mock()
    return orchestrator


@pytest.fixture
def mock_provider():
    """Create mock provider."""
    provider = Mock()
    provider.name = "test_provider"
    provider.shutdown = Mock()
    return provider


# =============================================================================
# Default Orchestrator Tests
# =============================================================================


class TestDefaultOrchestrator:
    """Test default orchestrator retrieval."""

    def test_get_default_orchestrator_success(
        self, mock_settings, mock_orchestrator, mock_provider
    ):
        """Test successful retrieval of default orchestrator."""
        with patch("victor.workflows.orchestrator_pool.ProviderRegistry") as mock_registry:
            with patch(
                "victor.workflows.orchestrator_pool.create_orchestrator_factory"
            ) as mock_factory_fn:
                # Setup mocks
                mock_registry.create.return_value = mock_provider
                mock_factory = Mock()
                mock_factory.create_orchestrator.return_value = mock_orchestrator
                mock_factory_fn.return_value = mock_factory

                # Create pool and get default orchestrator
                pool = OrchestratorPool(mock_settings)
                orchestrator = pool.get_default_orchestrator()

                # Verify
                assert orchestrator is not None
                assert orchestrator == mock_orchestrator
                mock_registry.create.assert_called_once_with("ollama")
                mock_factory.create_orchestrator.assert_called_once()

    def test_get_orchestrator_none_uses_default(
        self, mock_settings, mock_orchestrator, mock_provider
    ):
        """Test that passing None as profile uses default."""
        with patch("victor.workflows.orchestrator_pool.ProviderRegistry") as mock_registry:
            with patch(
                "victor.workflows.orchestrator_pool.create_orchestrator_factory"
            ) as mock_factory_fn:
                # Setup mocks
                mock_registry.create.return_value = mock_provider
                mock_factory = Mock()
                mock_factory.create_orchestrator.return_value = mock_orchestrator
                mock_factory_fn.return_value = mock_factory

                # Create pool and get orchestrator with None
                pool = OrchestratorPool(mock_settings)
                orchestrator = pool.get_orchestrator(None)

                # Verify
                assert orchestrator is not None
                assert orchestrator == mock_orchestrator
                mock_registry.create.assert_called_once_with("ollama")

    def test_get_default_orchestrator_profile_not_found(self, mock_settings):
        """Test error when default profile is not found."""
        # Mock empty profiles
        mock_settings.load_profiles.return_value = {}

        pool = OrchestratorPool(mock_settings)

        with pytest.raises(ValueError, match="Profile 'default' not found"):
            pool.get_default_orchestrator()


# =============================================================================
# Profile-Based Orchestrator Tests
# =============================================================================


class TestProfileBasedOrchestrator:
    """Test profile-based orchestrator creation."""

    def test_get_orchestrator_for_profile(
        self, mock_settings, mock_orchestrator, mock_provider
    ):
        """Test getting orchestrator for a specific profile."""
        with patch("victor.workflows.orchestrator_pool.ProviderRegistry") as mock_registry:
            with patch(
                "victor.workflows.orchestrator_pool.create_orchestrator_factory"
            ) as mock_factory_fn:
                # Setup mocks
                mock_registry.create.return_value = mock_provider
                mock_factory = Mock()
                mock_factory.create_orchestrator.return_value = mock_orchestrator
                mock_factory_fn.return_value = mock_factory

                # Create pool and get anthropic profile
                pool = OrchestratorPool(mock_settings)
                orchestrator = pool.get_orchestrator("anthropic")

                # Verify
                assert orchestrator is not None
                assert orchestrator == mock_orchestrator
                mock_registry.create.assert_called_once_with("anthropic")

    def test_get_orchestrator_profile_not_found(self, mock_settings):
        """Test error when requested profile is not found."""
        pool = OrchestratorPool(mock_settings)

        with pytest.raises(ValueError, match="Profile 'nonexistent' not found"):
            pool.get_orchestrator("nonexistent")


# =============================================================================
# Caching Tests
# =============================================================================


class TestOrchestratorCaching:
    """Test orchestrator caching and reuse."""

    def test_orchestrator_reuse_from_cache(
        self, mock_settings, mock_orchestrator, mock_provider
    ):
        """Test that cached orchestrator is reused on subsequent calls."""
        with patch("victor.workflows.orchestrator_pool.ProviderRegistry") as mock_registry:
            with patch(
                "victor.workflows.orchestrator_pool.create_orchestrator_factory"
            ) as mock_factory_fn:
                # Setup mocks
                mock_registry.create.return_value = mock_provider
                mock_factory = Mock()
                mock_factory.create_orchestrator.return_value = mock_orchestrator
                mock_factory_fn.return_value = mock_factory

                # Create pool
                pool = OrchestratorPool(mock_settings)

                # Get orchestrator twice
                orchestrator1 = pool.get_orchestrator("default")
                orchestrator2 = pool.get_orchestrator("default")

                # Verify same instance returned
                assert orchestrator1 is orchestrator2
                # Factory should only be called once (cached)
                assert mock_factory.create_orchestrator.call_count == 1

    def test_provider_reuse_from_cache(
        self, mock_settings, mock_orchestrator, mock_provider
    ):
        """Test that cached provider is reused."""
        with patch("victor.workflows.orchestrator_pool.ProviderRegistry") as mock_registry:
            with patch(
                "victor.workflows.orchestrator_pool.create_orchestrator_factory"
            ) as mock_factory_fn:
                # Setup mocks
                mock_registry.create.return_value = mock_provider
                mock_factory = Mock()
                mock_factory.create_orchestrator.return_value = mock_orchestrator
                mock_factory_fn.return_value = mock_factory

                # Create pool and get two different profiles with same provider
                pool = OrchestratorPool(mock_settings)
                pool.get_orchestrator("default")
                pool.get_orchestrator("default")  # Same profile

                # Provider should only be created once per profile
                assert mock_registry.create.call_count == 1

    def test_multiple_profiles_cached_separately(
        self, mock_settings, mock_orchestrator, mock_provider
    ):
        """Test that different profiles are cached separately."""
        with patch("victor.workflows.orchestrator_pool.ProviderRegistry") as mock_registry:
            with patch(
                "victor.workflows.orchestrator_pool.create_orchestrator_factory"
            ) as mock_factory_fn:
                # Setup mocks - create unique orchestrators for each profile
                mock_registry.create.return_value = mock_provider
                mock_factory = Mock()

                orchestrator1 = Mock()
                orchestrator2 = Mock()
                orchestrator3 = Mock()

                mock_factory.create_orchestrator.side_effect = [
                    orchestrator1,
                    orchestrator2,
                    orchestrator3,
                ]
                mock_factory_fn.return_value = mock_factory

                # Create pool and get multiple profiles
                pool = OrchestratorPool(mock_settings)
                default_orch = pool.get_orchestrator("default")
                anthropic_orch = pool.get_orchestrator("anthropic")
                openai_orch = pool.get_orchestrator("openai")

                # Verify each profile gets unique orchestrator
                assert default_orch is orchestrator1
                assert anthropic_orch is orchestrator2
                assert openai_orch is orchestrator3

                # Verify all are cached
                cached = pool.get_cached_profiles()
                assert set(cached) == {"default", "anthropic", "openai"}

    def test_clear_cache_single_profile(
        self, mock_settings, mock_orchestrator, mock_provider
    ):
        """Test clearing cache for a single profile."""
        with patch("victor.workflows.orchestrator_pool.ProviderRegistry") as mock_registry:
            with patch(
                "victor.workflows.orchestrator_pool.create_orchestrator_factory"
            ) as mock_factory_fn:
                # Setup mocks
                mock_registry.create.return_value = mock_provider
                mock_factory = Mock()
                mock_factory.create_orchestrator.return_value = mock_orchestrator
                mock_factory_fn.return_value = mock_factory

                # Create pool and cache orchestrator
                pool = OrchestratorPool(mock_settings)
                pool.get_orchestrator("default")

                # Clear cache
                pool.clear_cache("default")

                # Verify cache cleared
                assert pool.get_cached_profiles() == []

                # Next call should create new orchestrator
                pool.get_orchestrator("default")
                assert mock_factory.create_orchestrator.call_count == 2

    def test_clear_cache_all_profiles(
        self, mock_settings, mock_orchestrator, mock_provider
    ):
        """Test clearing cache for all profiles."""
        with patch("victor.workflows.orchestrator_pool.ProviderRegistry") as mock_registry:
            with patch(
                "victor.workflows.orchestrator_pool.create_orchestrator_factory"
            ) as mock_factory_fn:
                # Setup mocks
                mock_registry.create.return_value = mock_provider
                mock_factory = Mock()
                mock_factory.create_orchestrator.return_value = mock_orchestrator
                mock_factory_fn.return_value = mock_factory

                # Create pool and cache multiple orchestrators
                pool = OrchestratorPool(mock_settings)
                pool.get_orchestrator("default")
                pool.get_orchestrator("anthropic")
                pool.get_orchestrator("openai")

                # Verify all cached
                assert len(pool.get_cached_profiles()) == 3

                # Clear all caches
                pool.clear_cache()

                # Verify all cleared
                assert pool.get_cached_profiles() == []

    def test_get_cached_profiles(self, mock_settings, mock_orchestrator, mock_provider):
        """Test getting list of cached profiles."""
        with patch("victor.workflows.orchestrator_pool.ProviderRegistry") as mock_registry:
            with patch(
                "victor.workflows.orchestrator_pool.create_orchestrator_factory"
            ) as mock_factory_fn:
                # Setup mocks
                mock_registry.create.return_value = mock_provider
                mock_factory = Mock()
                mock_factory.create_orchestrator.return_value = mock_orchestrator
                mock_factory_fn.return_value = mock_factory

                # Create pool
                pool = OrchestratorPool(mock_settings)

                # Initially empty
                assert pool.get_cached_profiles() == []

                # Add profiles
                pool.get_orchestrator("default")
                pool.get_orchestrator("anthropic")

                # Verify cached profiles
                cached = pool.get_cached_profiles()
                assert set(cached) == {"default", "anthropic"}


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in OrchestratorPool."""

    def test_provider_creation_failure(self, mock_settings):
        """Test handling of provider creation failure."""
        with patch("victor.workflows.orchestrator_pool.ProviderRegistry") as mock_registry:
            # Setup mock to raise error
            mock_registry.create.side_effect = Exception("Provider creation failed")

            pool = OrchestratorPool(mock_settings)

            with pytest.raises(RuntimeError, match="Failed to create provider"):
                pool.get_orchestrator("default")

    def test_orchestrator_creation_failure(
        self, mock_settings, mock_provider
    ):
        """Test handling of orchestrator creation failure."""
        with patch("victor.workflows.orchestrator_pool.ProviderRegistry") as mock_registry:
            with patch(
                "victor.workflows.orchestrator_pool.create_orchestrator_factory"
            ) as mock_factory_fn:
                # Setup mocks
                mock_registry.create.return_value = mock_provider
                mock_factory = Mock()
                mock_factory.create_orchestrator.side_effect = Exception("Orchestrator creation failed")
                mock_factory_fn.return_value = mock_factory

                pool = OrchestratorPool(mock_settings)

                with pytest.raises(RuntimeError, match="Failed to create orchestrator"):
                    pool.get_orchestrator("default")


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Test orchestrator pool lifecycle management."""

    def test_shutdown_closes_orchestrators(
        self, mock_settings, mock_provider
    ):
        """Test that shutdown closes all orchestrators."""
        with patch("victor.workflows.orchestrator_pool.ProviderRegistry") as mock_registry:
            with patch(
                "victor.workflows.orchestrator_pool.create_orchestrator_factory"
            ) as mock_factory_fn:
                # Setup mocks
                mock_registry.create.return_value = mock_provider

                orchestrator1 = Mock()
                orchestrator2 = Mock()
                orchestrator3 = Mock()

                mock_factory = Mock()
                mock_factory.create_orchestrator.side_effect = [
                    orchestrator1,
                    orchestrator2,
                    orchestrator3,
                ]
                mock_factory_fn.return_value = mock_factory

                # Create pool and cache orchestrators
                pool = OrchestratorPool(mock_settings)
                pool.get_orchestrator("default")
                pool.get_orchestrator("anthropic")
                pool.get_orchestrator("openai")

                # Shutdown
                pool.shutdown()

                # Verify all orchestrators closed
                orchestrator1.close.assert_called_once()
                orchestrator2.close.assert_called_once()
                orchestrator3.close.assert_called_once()

                # Verify caches cleared
                assert pool.get_cached_profiles() == []

    def test_shutdown_handles_close_errors(
        self, mock_settings, mock_provider, caplog
    ):
        """Test that shutdown handles errors gracefully."""
        import logging

        with patch("victor.workflows.orchestrator_pool.ProviderRegistry") as mock_registry:
            with patch(
                "victor.workflows.orchestrator_pool.create_orchestrator_factory"
            ) as mock_factory_fn:
                # Setup mocks
                mock_registry.create.return_value = mock_provider

                orchestrator = Mock()
                orchestrator.close.side_effect = Exception("Close failed")

                mock_factory = Mock()
                mock_factory.create_orchestrator.return_value = orchestrator
                mock_factory_fn.return_value = mock_factory

                # Create pool and shutdown
                pool = OrchestratorPool(mock_settings)
                pool.get_orchestrator("default")

                # Should not raise, just log warning
                pool.shutdown()

                # Verify warning logged
                assert any("Error closing orchestrator" in message for message in caplog.messages)
