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

"""Tests for ConfigCoordinator.

Tests the configuration loading and validation coordination functionality.
"""

import pytest

from victor.agent.coordinators.config_coordinator import (
    ConfigCoordinator,
    ValidationResult,
    OrchestratorConfig,
    SettingsConfigProvider,
    EnvironmentConfigProvider,
)


class MockSettings:
    """Mock Settings object for testing."""

    def __init__(self):
        self.temperature = 0.5
        self.max_tokens = 2048
        self.thinking = False
        self.tool_selection = {"base_threshold": 0.5}


class MockConfigProvider:
    """Mock config provider for testing."""

    def __init__(self, config, priority=50):
        self._config = config
        self._priority = priority

    async def get_config(self, session_id):
        return self._config

    def priority(self):
        return self._priority


class TestSettingsConfigProvider:
    """Tests for SettingsConfigProvider."""

    @pytest.mark.asyncio
    async def test_get_config(self):
        """Test getting config from Settings."""
        settings = MockSettings()
        provider = SettingsConfigProvider(settings)

        config = await provider.get_config("session123")

        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 2048
        assert config["thinking"] is False

    def test_priority(self):
        """Test provider priority."""
        provider = SettingsConfigProvider(MockSettings(), priority=100)
        assert provider.priority() == 100


class TestEnvironmentConfigProvider:
    """Tests for EnvironmentConfigProvider."""

    @pytest.mark.asyncio
    async def test_get_config_from_env(self, monkeypatch):
        """Test getting config from environment variables."""
        monkeypatch.setenv("VICTOR_TEMPERATURE", "0.8")
        monkeypatch.setenv("VICTOR_MAX_TOKENS", "8192")
        monkeypatch.setenv("VICTOR_THINKING", "true")

        provider = EnvironmentConfigProvider()
        config = await provider.get_config("session123")

        assert config["temperature"] == 0.8
        assert config["max_tokens"] == 8192
        assert config["thinking"] is True

    @pytest.mark.asyncio
    async def test_get_config_empty_env(self, monkeypatch):
        """Test getting config when env vars not set."""
        # Clear any existing env vars
        monkeypatch.delenv("VICTOR_TEMPERATURE", raising=False)
        monkeypatch.delenv("VICTOR_MAX_TOKENS", raising=False)

        provider = EnvironmentConfigProvider()
        config = await provider.get_config("session123")

        assert config == {}


class TestConfigCoordinator:
    """Tests for ConfigCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator with no providers."""
        return ConfigCoordinator(providers=[])

    @pytest.fixture
    def coordinator_with_providers(self):
        """Create coordinator with mock providers."""
        provider1 = MockConfigProvider({"temperature": 0.3}, priority=10)
        provider2 = MockConfigProvider(
            {"max_tokens": 1024, "temperature": 0.7}, priority=100
        )
        return ConfigCoordinator(providers=[provider1, provider2])

    def test_init_empty(self):
        """Test initialization with no providers."""
        coordinator = ConfigCoordinator(providers=[])
        assert coordinator._providers == []
        assert coordinator._enable_cache is True

    def test_init_with_providers(self):
        """Test initialization with providers."""
        provider = MockConfigProvider({}, priority=50)
        coordinator = ConfigCoordinator(
            providers=[provider], enable_cache=False
        )

        assert len(coordinator._providers) == 1
        assert coordinator._enable_cache is False

    def test_init_sorts_providers_by_priority(self):
        """Test that providers are sorted by priority (lowest first)."""
        provider1 = MockConfigProvider({}, priority=10)
        provider2 = MockConfigProvider({}, priority=100)
        provider3 = MockConfigProvider({}, priority=50)

        coordinator = ConfigCoordinator(providers=[provider1, provider2, provider3])

        # Should be sorted: provider1 (10), provider3 (50), provider2 (100)
        # Lower priority first, so higher priority can override during merge
        assert coordinator._providers[0]._priority == 10
        assert coordinator._providers[1]._priority == 50
        assert coordinator._providers[2]._priority == 100

    @pytest.mark.asyncio
    async def test_load_config_no_providers(self, coordinator):
        """Test loading config with no providers."""
        config = await coordinator.load_config("session123")
        assert config == {}

    @pytest.mark.asyncio
    async def test_load_config_merges_providers(self, coordinator_with_providers):
        """Test that config is merged from multiple providers."""
        config = await coordinator_with_providers.load_config("session123")

        # Higher priority provider should override temperature
        assert config["temperature"] == 0.7
        # max_tokens from lower priority provider
        assert config["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_load_config_with_override(self, coordinator_with_providers):
        """Test loading config with override values."""
        config = await coordinator_with_providers.load_config(
            "session123", config_override={"temperature": 0.9}
        )

        # Override should take precedence
        assert config["temperature"] == 0.9
        assert config["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_load_config_caching(self, coordinator_with_providers):
        """Test that config is cached."""
        # First call
        config1 = await coordinator_with_providers.load_config("session123")
        # Second call should use cache
        config2 = await coordinator_with_providers.load_config("session123")

        assert config1 == config2

    @pytest.mark.asyncio
    async def test_invalidate_cache(self, coordinator_with_providers):
        """Test cache invalidation."""
        # Load config
        await coordinator_with_providers.load_config("session123")

        # Invalidate cache
        coordinator_with_providers.invalidate_cache("session123")

        # Cache should be cleared
        assert "session123" not in coordinator_with_providers._config_cache

    @pytest.mark.asyncio
    async def test_invalidate_all_cache(self, coordinator_with_providers):
        """Test invalidating all cache."""
        # Load multiple configs
        await coordinator_with_providers.load_config("session1")
        await coordinator_with_providers.load_config("session2")

        # Clear all cache
        coordinator_with_providers.invalidate_cache()

        assert coordinator_with_providers._config_cache == {}

    @pytest.mark.asyncio
    async def test_load_orchestrator_config(self, coordinator_with_providers):
        """Test loading OrchestratorConfig."""
        config = await coordinator_with_providers.load_orchestrator_config(
            session_id="session123",
            provider="anthropic",
            model="claude-sonnet-4-5"
        )

        assert isinstance(config, OrchestratorConfig)
        assert config.session_id == "session123"
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-5"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024

    @pytest.mark.asyncio
    async def test_validate_config_valid(self, coordinator):
        """Test validating a valid config."""
        config = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "temperature": 0.7,
            "max_tokens": 4096,
        }

        result = await coordinator.validate_config(config)

        assert result.valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_config_missing_required_fields(self, coordinator):
        """Test validation fails with missing required fields."""
        config = {"temperature": 0.7}  # Missing provider and model

        result = await coordinator.validate_config(config)

        assert result.valid is False
        assert "provider" in result.errors[0].lower()
        assert "model" in result.errors[1].lower()

    @pytest.mark.asyncio
    async def test_validate_config_invalid_temperature(self, coordinator):
        """Test validation fails with invalid temperature."""
        config = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "temperature": 3.0,  # Invalid: > 2
            "max_tokens": 4096,
        }

        result = await coordinator.validate_config(config)

        assert result.valid is False
        assert "temperature" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validate_config_invalid_max_tokens(self, coordinator):
        """Test validation fails with invalid max_tokens."""
        config = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "temperature": 0.7,
            "max_tokens": -100,  # Invalid: negative
        }

        result = await coordinator.validate_config(config)

        assert result.valid is False
        assert "max_tokens" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validate_config_warnings(self, coordinator):
        """Test validation generates warnings for non-critical issues."""
        config = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "temperature": 1.5,  # High but valid
            "max_tokens": 200000,  # Very large
        }

        result = await coordinator.validate_config(config)

        assert result.valid is True
        assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_validate_orchestrator_config(self, coordinator):
        """Test validating OrchestratorConfig dataclass."""
        config = OrchestratorConfig(
            session_id="session123",
            provider="anthropic",
            model="claude-sonnet-4-5",
            temperature=0.7,
            max_tokens=4096,
        )

        result = await coordinator.validate_config(config)

        assert result.valid is True

    def test_add_provider(self, coordinator):
        """Test adding a provider."""
        provider = MockConfigProvider({}, priority=75)
        coordinator.add_provider(provider)

        assert len(coordinator._providers) == 1
        assert coordinator._providers[0]._priority == 75

    def test_add_provider_resorts(self, coordinator):
        """Test that adding a provider triggers re-sorting."""
        provider1 = MockConfigProvider({}, priority=10)
        provider2 = MockConfigProvider({}, priority=100)

        coordinator.add_provider(provider1)
        coordinator.add_provider(provider2)

        # Should be sorted ascending by priority: provider1 (10), provider2 (100)
        assert coordinator._providers[0] == provider1
        assert coordinator._providers[1] == provider2

    def test_remove_provider(self, coordinator):
        """Test removing a provider."""
        provider = MockConfigProvider({}, priority=50)
        coordinator.add_provider(provider)

        coordinator.remove_provider(provider)

        assert len(coordinator._providers) == 0

    def test_remove_nonexistent_provider(self, coordinator):
        """Test removing a provider that doesn't exist."""
        provider = MockConfigProvider({}, priority=50)

        # Should not raise
        coordinator.remove_provider(provider)

        assert len(coordinator._providers) == 0

    def test_deep_merge(self):
        """Test deep merge functionality."""
        coordinator = ConfigCoordinator(providers=[])

        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 20}, "e": 5}

        result = coordinator._deep_merge(base, override)

        assert result["a"] == 1
        assert result["b"]["c"] == 20  # Overridden
        assert result["b"]["d"] == 3  # Preserved
        assert result["e"] == 5  # Added


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_validation_result_success(self):
        """Test successful validation result."""
        result = ValidationResult(valid=True, errors=[], warnings=[])

        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_validation_result_with_errors(self):
        """Test validation result with errors."""
        result = ValidationResult(
            valid=False, errors=["Missing field", "Invalid value"]
        )

        assert result.valid is False
        assert len(result.errors) == 2

    def test_validation_result_with_warnings(self):
        """Test validation result with warnings."""
        result = ValidationResult(
            valid=True, errors=[], warnings=["High temperature"]
        )

        assert result.valid is True
        assert len(result.warnings) == 1

    def test_validation_result_with_metadata(self):
        """Test validation result with metadata."""
        result = ValidationResult(
            valid=True, errors=[], warnings=[], metadata={"config_type": "test"}
        )

        assert result.metadata["config_type"] == "test"


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_orchestrator_config_defaults(self):
        """Test default values."""
        config = OrchestratorConfig(
            session_id="session123",
            provider="anthropic",
            model="claude-sonnet-4-5"
        )

        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.tool_selection is None
        assert config.thinking is False
        assert config.profile_name is None
        assert config.metadata is None

    def test_orchestrator_config_custom_values(self):
        """Test custom values."""
        config = OrchestratorConfig(
            session_id="session123",
            provider="anthropic",
            model="claude-sonnet-4-5",
            temperature=0.5,
            max_tokens=8192,
            thinking=True,
            profile_name="fast",
            tool_selection={"base_threshold": 0.3},
            metadata={"custom": "value"}
        )

        assert config.temperature == 0.5
        assert config.max_tokens == 8192
        assert config.thinking is True
        assert config.profile_name == "fast"
        assert config.tool_selection == {"base_threshold": 0.3}
        assert config.metadata == {"custom": "value"}


class TestProfileConfigProvider:
    """Tests for ProfileConfigProvider."""

    @pytest.fixture
    def mock_settings_with_profiles(self):
        """Create mock settings with profiles."""
        class MockProfile:
            def __init__(self, provider, model, temperature=0.7, max_tokens=4096, tool_selection=None):
                self.provider = provider
                self.model = model
                self.temperature = temperature
                self.max_tokens = max_tokens
                self.tool_selection = tool_selection
                self.__pydantic_extra__ = None

        class MockSettings:
            def __init__(self):
                self._profiles = {
                    "default": MockProfile("anthropic", "claude-sonnet-4-5"),
                    "fast": MockProfile("openai", "gpt-4", temperature=0.5, max_tokens=2048),
                    "local": MockProfile("ollama", "llama3", temperature=0.8),
                }

            def load_profiles(self):
                return self._profiles

        return MockSettings()

    @pytest.mark.asyncio
    async def test_get_config_from_profile(self, mock_settings_with_profiles):
        """Test getting config from profile."""
        from victor.agent.coordinators.config_coordinator import ProfileConfigProvider

        provider = ProfileConfigProvider(mock_settings_with_profiles, profile_name="fast")
        config = await provider.get_config("session123")

        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4"
        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 2048
        assert config["profile_name"] == "fast"

    @pytest.mark.asyncio
    async def test_get_config_profile_not_found(self, mock_settings_with_profiles):
        """Test error when profile not found."""
        from victor.agent.coordinators.config_coordinator import ProfileConfigProvider

        provider = ProfileConfigProvider(mock_settings_with_profiles, profile_name="nonexistent")

        with pytest.raises(ValueError, match="Profile not found: 'nonexistent'"):
            await provider.get_config("session123")

    @pytest.mark.asyncio
    async def test_get_config_profile_not_found_with_suggestions(
        self, mock_settings_with_profiles
    ):
        """Test error message includes profile suggestions."""
        from victor.agent.coordinators.config_coordinator import ProfileConfigProvider

        provider = ProfileConfigProvider(mock_settings_with_profiles, profile_name="fst")

        with pytest.raises(ValueError, match="Profile not found: 'fst'") as exc_info:
            await provider.get_config("session123")

        error_msg = str(exc_info.value)
        assert "Did you mean" in error_msg
        assert "Available profiles" in error_msg

    def test_priority(self):
        """Test provider priority."""
        from victor.agent.coordinators.config_coordinator import ProfileConfigProvider

        provider = ProfileConfigProvider(MockSettings(), priority=200)
        assert provider.priority() == 200


class TestConfigCoordinatorErrorHandling:
    """Tests for ConfigCoordinator error handling."""

    @pytest.mark.asyncio
    async def test_load_config_provider_error(self):
        """Test error handling when provider fails."""
        from victor.agent.coordinators.config_coordinator import ConfigCoordinator

        class FailingProvider:
            async def get_config(self, session_id):
                raise RuntimeError("Provider failure")

            def priority(self):
                return 50

        coordinator = ConfigCoordinator(providers=[FailingProvider()])

        with pytest.raises(ValueError, match="Configuration loading failed"):
            await coordinator.load_config("session123")

    @pytest.mark.asyncio
    async def test_load_config_no_providers_returns_empty(self):
        """Test that no providers returns empty config."""
        from victor.agent.coordinators.config_coordinator import ConfigCoordinator

        coordinator = ConfigCoordinator(providers=[], enable_cache=False)

        config = await coordinator.load_config("session123")
        assert config == {}

    @pytest.mark.asyncio
    async def test_load_config_with_failing_and_working_providers(self):
        """Test that working providers are used even when some fail."""
        from victor.agent.coordinators.config_coordinator import ConfigCoordinator

        class FailingProvider:
            async def get_config(self, session_id):
                raise RuntimeError("Provider failure")

            def priority(self):
                return 10

        class WorkingProvider:
            async def get_config(self, session_id):
                return {"temperature": 0.5, "max_tokens": 2048}

            def priority(self):
                return 50

        coordinator = ConfigCoordinator(
            providers=[FailingProvider(), WorkingProvider()]
        )

        config = await coordinator.load_config("session123")
        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 2048


class TestConfigCoordinatorCreateProvider:
    """Tests for ConfigCoordinator.create_provider_from_config."""

    @pytest.mark.asyncio
    async def test_create_provider_missing_provider_field(self):
        """Test error when provider field is missing."""
        from victor.agent.coordinators.config_coordinator import ConfigCoordinator

        coordinator = ConfigCoordinator(providers=[])

        with pytest.raises(ValueError, match="Configuration missing 'provider' field"):
            await coordinator.create_provider_from_config(
                config={}, settings=MockSettings()
            )

    @pytest.mark.asyncio
    async def test_create_provider_with_mock_registry(self):
        """Test provider creation with mock registry."""
        from victor.agent.coordinators.config_coordinator import ConfigCoordinator
        from unittest.mock import MagicMock, Mock

        coordinator = ConfigCoordinator(providers=[])

        # Create mock provider registry
        mock_registry = Mock()
        mock_provider = MagicMock()
        mock_registry.create.return_value = mock_provider

        # Create mock settings with provider settings
        mock_settings = Mock()
        mock_settings.get_provider_settings.return_value = {
            "api_key": "test-key",
            "timeout": 300,
        }

        config = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
        }

        provider = await coordinator.create_provider_from_config(
            config=config,
            settings=mock_settings,
            provider_registry=mock_registry,
        )

        assert provider == mock_provider
        mock_registry.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_provider_with_profile_overrides(self):
        """Test provider creation with profile overrides."""
        from victor.agent.coordinators.config_coordinator import ConfigCoordinator
        from unittest.mock import Mock

        coordinator = ConfigCoordinator(providers=[])

        # Create mock provider registry
        mock_registry = Mock()
        mock_provider = Mock()
        mock_registry.create.return_value = mock_provider

        # Create mock settings
        mock_settings = Mock()
        mock_settings.get_provider_settings.return_value = {"api_key": "test-key"}

        config = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "profile_overrides": {"timeout": 600, "custom_param": "value"},
        }

        provider = await coordinator.create_provider_from_config(
            config=config,
            settings=mock_settings,
            provider_registry=mock_registry,
        )

        # Verify that profile overrides were applied
        call_kwargs = mock_registry.create.call_args[1]
        assert "timeout" in call_kwargs
        assert "custom_param" in call_kwargs


class TestToolAccessConfigCoordinator:
    """Tests for ToolAccessConfigCoordinator."""

    @pytest.fixture
    def mock_tool_access_controller(self):
        """Create mock tool access controller."""
        from unittest.mock import Mock

        controller = Mock()
        controller.get_allowed_tools.return_value = {"bash", "read_file", "write_file"}
        controller.check_access.return_value = Mock(allowed=True)
        return controller

    @pytest.fixture
    def mock_mode_coordinator(self):
        """Create mock mode coordinator."""
        from unittest.mock import Mock

        coordinator = Mock()
        coordinator.current_mode_name = "build"
        coordinator.get_mode_config.return_value = Mock(
            allow_all_tools=False, disallowed_tools=set()
        )
        coordinator.is_tool_allowed.return_value = True
        return coordinator

    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry."""
        from unittest.mock import Mock

        registry = Mock()
        registry.list_tools.return_value = ["bash", "read_file", "write_file", "search"]
        return registry

    @pytest.fixture
    def tool_access_coordinator(
        self, mock_tool_access_controller, mock_mode_coordinator, mock_tool_registry
    ):
        """Create ToolAccessConfigCoordinator with mocks."""
        from victor.agent.coordinators.config_coordinator import ToolAccessConfigCoordinator

        return ToolAccessConfigCoordinator(
            tool_access_controller=mock_tool_access_controller,
            mode_coordinator=mock_mode_coordinator,
            tool_registry=mock_tool_registry,
        )

    def test_init(self, tool_access_coordinator):
        """Test initialization."""
        assert tool_access_coordinator._tool_access_controller is not None
        assert tool_access_coordinator._mode_coordinator is not None
        assert tool_access_coordinator._tool_registry is not None

    def test_init_with_none_components(self):
        """Test initialization with None components."""
        from victor.agent.coordinators.config_coordinator import ToolAccessConfigCoordinator

        coordinator = ToolAccessConfigCoordinator(
            tool_access_controller=None, mode_coordinator=None, tool_registry=None
        )

        assert coordinator._tool_access_controller is None
        assert coordinator._mode_coordinator is None
        assert coordinator._tool_registry is None

    def test_build_tool_access_context(self, tool_access_coordinator):
        """Test building tool access context."""
        context = tool_access_coordinator.build_tool_access_context(
            session_enabled_tools={"bash", "read_file"}, current_mode="plan"
        )

        assert context.session_enabled_tools == {"bash", "read_file"}
        assert context.current_mode == "plan"

    def test_build_tool_access_context_defaults(self, tool_access_coordinator):
        """Test building context with default values."""
        context = tool_access_coordinator.build_tool_access_context()

        assert context.session_enabled_tools is None
        assert context.current_mode == "build"  # From mock

    def test_get_enabled_tools_with_controller(self, tool_access_coordinator):
        """Test getting enabled tools via controller."""
        tools = tool_access_coordinator.get_enabled_tools()

        assert tools == {"bash", "read_file", "write_file"}
        tool_access_coordinator._tool_access_controller.get_allowed_tools.assert_called_once()

    def test_get_enabled_tools_without_controller(self, mock_tool_registry):
        """Test getting enabled tools without controller."""
        from victor.agent.coordinators.config_coordinator import ToolAccessConfigCoordinator

        coordinator = ToolAccessConfigCoordinator(
            tool_access_controller=None,
            mode_coordinator=None,
            tool_registry=mock_tool_registry,
        )

        tools = coordinator.get_enabled_tools()
        assert tools == {"bash", "read_file", "write_file", "search"}

    def test_is_tool_enabled_with_controller(self, tool_access_coordinator):
        """Test checking if tool is enabled via controller."""
        result = tool_access_coordinator.is_tool_enabled("bash")

        assert result is True
        tool_access_coordinator._tool_access_controller.check_access.assert_called_once()

    def test_is_tool_enabled_without_controller(self, mock_tool_registry):
        """Test checking if tool is enabled without controller."""
        from victor.agent.coordinators.config_coordinator import ToolAccessConfigCoordinator

        coordinator = ToolAccessConfigCoordinator(
            tool_access_controller=None,
            mode_coordinator=None,
            tool_registry=mock_tool_registry,
        )

        result = coordinator.is_tool_enabled("bash")
        assert result is True

    def test_set_enabled_tools(self, tool_access_coordinator):
        """Test setting enabled tools."""
        from unittest.mock import Mock

        mock_selector = Mock()
        session_tools = set()

        tool_access_coordinator.set_enabled_tools(
            tools={"bash", "read_file"},
            session_enabled_tools_attr=session_tools,
            tool_selector=mock_selector,
        )

        assert session_tools == {"bash", "read_file"}
        mock_selector.set_enabled_tools.assert_called_once_with({"bash", "read_file"})

    def test_set_enabled_tools_with_vertical_context(self, tool_access_coordinator):
        """Test setting enabled tools with vertical context."""
        from unittest.mock import Mock

        mock_selector = Mock()
        mock_vertical_context = Mock()
        session_tools = set()

        tool_access_coordinator.set_enabled_tools(
            tools={"bash", "read_file"},
            session_enabled_tools_attr=session_tools,
            tool_selector=mock_selector,
            vertical_context=mock_vertical_context,
        )

        assert mock_vertical_context.enabled_tools == {"bash", "read_file"}

    def test_get_available_tools(self, tool_access_coordinator):
        """Test getting available tools."""
        tools = tool_access_coordinator.get_available_tools()

        assert tools == {"bash", "read_file", "write_file", "search"}

    def test_get_available_tools_no_registry(self):
        """Test getting available tools without registry."""
        from victor.agent.coordinators.config_coordinator import ToolAccessConfigCoordinator

        coordinator = ToolAccessConfigCoordinator(tool_registry=None)
        tools = coordinator.get_available_tools()

        assert tools == set()

    def test_detect_config_changes(self, tool_access_coordinator):
        """Test detecting configuration changes."""
        old_config = {"temperature": 0.7, "max_tokens": 4096}
        new_config = {"temperature": 0.5, "max_tokens": 4096, "thinking": True}

        changes = tool_access_coordinator.detect_config_changes(old_config, new_config)

        assert "temperature" in changes
        assert changes["temperature"] == (0.7, 0.5)
        assert "max_tokens" not in changes
        assert "thinking" in changes

    def test_detect_config_changes_no_changes(self, tool_access_coordinator):
        """Test detecting changes when none exist."""
        config = {"temperature": 0.7, "max_tokens": 4096}

        changes = tool_access_coordinator.detect_config_changes(config, config)

        assert changes == {}

    def test_validate_mode_transition_safe(self, tool_access_coordinator):
        """Test validating safe mode transition."""
        result = tool_access_coordinator.validate_mode_transition(
            from_mode="plan", to_mode="build", current_tools={"bash", "read_file"}
        )

        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_mode_transition_restrictive_with_warnings(
        self, tool_access_coordinator
    ):
        """Test validating transition to restrictive mode with write tools."""
        result = tool_access_coordinator.validate_mode_transition(
            from_mode="build",
            to_mode="plan",
            current_tools={"bash", "write_file", "read_file"},
        )

        assert result.valid is True
        assert len(result.warnings) > 0
        assert "write" in str(result.warnings).lower()

    def test_validate_mode_transition_restrictive_no_write_tools(
        self, tool_access_coordinator
    ):
        """Test validating transition to restrictive mode without write tools."""
        result = tool_access_coordinator.validate_mode_transition(
            from_mode="build", to_mode="explore", current_tools={"read_file", "search"}
        )

        assert result.valid is True
        assert len(result.warnings) == 0
