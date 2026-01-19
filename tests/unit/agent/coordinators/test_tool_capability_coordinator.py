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

"""Unit tests for ToolCapabilityCoordinator.

Tests tool capability checks and model capability queries.
"""

from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from victor.agent.coordinators.tool_capability_coordinator import (
    ToolCapabilityCoordinator,
    CapabilityCheckResult,
    ModelCapabilityInfo,
    create_tool_capability_coordinator,
)


@pytest.fixture
def mock_tool_capabilities():
    """Create a mock tool capabilities checker."""
    capabilities = MagicMock()
    return capabilities


@pytest.fixture
def mock_console():
    """Create a mock console."""
    console = MagicMock()
    return console


class TestToolCapabilityCoordinator:
    """Test suite for ToolCapabilityCoordinator."""

    def test_initialization(self, mock_tool_capabilities):
        """Test coordinator initialization."""
        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        assert coordinator._tool_capabilities == mock_tool_capabilities
        assert coordinator._console is None
        assert coordinator._warn_once is True

    def test_initialization_with_console(self, mock_tool_capabilities, mock_console):
        """Test coordinator initialization with console."""
        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
            console=mock_console,
            warn_once=False,
        )

        assert coordinator._console == mock_console
        assert coordinator._warn_once is False

    def test_check_tool_calling_capability_supported(self, mock_tool_capabilities):
        """Test check_tool_calling_capability when supported."""
        # Setup
        mock_tool_capabilities.is_tool_call_supported.return_value = True

        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        result = coordinator.check_tool_calling_capability(
            provider_name="anthropic",
            model="claude-sonnet-4-5",
        )

        # Verify
        assert result.supported is True
        assert result.provider == "anthropic"
        assert result.model == "claude-sonnet-4-5"
        assert result.capability == "tool_calling"
        assert len(result.alternative_models) == 0

    def test_check_tool_calling_capability_not_supported(self, mock_tool_capabilities):
        """Test check_tool_calling_capability when not supported."""
        # Setup
        mock_tool_capabilities.is_tool_call_supported.return_value = False
        mock_tool_capabilities.get_supported_models.return_value = [
            "claude-3-opus",
            "claude-3-sonnet",
        ]

        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        result = coordinator.check_tool_calling_capability(
            provider_name="anthropic",
            model="claude-2",
        )

        # Verify
        assert result.supported is False
        assert result.provider == "anthropic"
        assert result.model == "claude-2"
        assert len(result.alternative_models) == 2
        assert "claude-3-opus" in result.alternative_models
        assert result.reason is not None

    def test_check_tool_calling_capability_no_provider(self, mock_tool_capabilities):
        """Test check_tool_calling_capability without provider."""
        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        result = coordinator.check_tool_calling_capability(
            provider_name=None,
            model="unknown-model",
        )

        # Verify - should default to supported
        assert result.supported is True
        assert result.provider == "unknown"

    def test_check_tool_calling_capability_empty_provider(self, mock_tool_capabilities):
        """Test check_tool_calling_capability with empty provider."""
        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        result = coordinator.check_tool_calling_capability(
            provider_name="",
            model="unknown-model",
        )

        # Verify - should default to supported
        assert result.supported is True

    def test_log_capability_warning_first_time(self, mock_tool_capabilities, mock_console):
        """Test logging capability warning for first time."""
        # Setup
        mock_tool_capabilities.is_tool_call_supported.return_value = False
        mock_tool_capabilities.get_supported_models.return_value = ["claude-3-opus"]

        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
            console=mock_console,
            warn_once=True,
        )

        result = coordinator.check_tool_calling_capability(
            provider_name="anthropic",
            model="claude-2",
        )

        # Execute - log warning
        with patch("victor.agent.coordinators.tool_capability_coordinator.logger"):
            coordinator.log_capability_warning("anthropic", "claude-2", result)

        # Verify - console print should be called
        mock_console.print.assert_called_once()

    def test_log_capability_warning_subsequent_times(self, mock_tool_capabilities, mock_console):
        """Test logging capability warning is suppressed with warn_once."""
        # Setup
        mock_tool_capabilities.is_tool_call_supported.return_value = False
        mock_tool_capabilities.get_supported_models.return_value = ["claude-3-opus"]

        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
            console=mock_console,
            warn_once=True,
        )

        result = coordinator.check_tool_calling_capability(
            provider_name="anthropic",
            model="claude-2",
        )

        # Execute - log twice
        coordinator.log_capability_warning("anthropic", "claude-2", result)
        mock_console.print.reset_mock()  # Reset after first call

        coordinator.log_capability_warning("anthropic", "claude-2", result)

        # Verify - should not be called second time
        mock_console.print.assert_not_called()

    def test_log_capability_warning_warn_once_disabled(self, mock_tool_capabilities, mock_console):
        """Test logging capability warning with warn_once=False."""
        # Setup
        mock_tool_capabilities.is_tool_call_supported.return_value = False
        mock_tool_capabilities.get_supported_models.return_value = ["claude-3-opus"]

        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
            console=mock_console,
            warn_once=False,
        )

        result = coordinator.check_tool_calling_capability(
            provider_name="anthropic",
            model="claude-2",
        )

        # Execute - log twice
        coordinator.log_capability_warning("anthropic", "claude-2", result)
        coordinator.log_capability_warning("anthropic", "claude-2", result)

        # Verify - both calls should print
        assert mock_console.print.call_count == 2

    def test_get_supported_models(self, mock_tool_capabilities):
        """Test getting supported models for provider."""
        # Setup
        mock_tool_capabilities.get_supported_models.return_value = [
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
        ]

        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        models = coordinator.get_supported_models("anthropic")

        # Verify
        assert len(models) == 3
        assert "claude-3-opus" in models
        mock_tool_capabilities.get_supported_models.assert_called_once_with("anthropic")

    def test_get_supported_models_with_exception(self, mock_tool_capabilities):
        """Test get_supported_models handles exceptions."""
        # Setup
        mock_tool_capabilities.get_supported_models.side_effect = Exception("Models error")

        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        models = coordinator.get_supported_models("anthropic")

        # Verify - should return empty list
        assert models == []

    def test_get_capability_info_full(self, mock_tool_capabilities):
        """Test getting full capability information."""
        # Setup
        mock_tool_capabilities.is_tool_call_supported.return_value = True
        mock_tool_capabilities.supports_parallel_calls.return_value = True
        mock_tool_capabilities.get_capabilities.return_value = {"vision", "json_mode"}

        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        info = coordinator.get_capability_info("anthropic", "claude-sonnet-4-5")

        # Verify
        assert info is not None
        assert info.provider == "anthropic"
        assert info.model == "claude-sonnet-4-5"
        assert info.tool_calling_supported is True
        assert info.parallel_calls_supported is True
        assert "vision" in info.other_capabilities
        assert "json_mode" in info.other_capabilities

    def test_get_capability_info_minimal(self, mock_tool_capabilities):
        """Test getting capability info without extra methods."""
        # Setup - only has is_tool_call_supported
        mock_tool_capabilities.is_tool_call_supported.return_value = True
        del mock_tool_capabilities.supports_parallel_calls
        del mock_tool_capabilities.get_capabilities

        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        info = coordinator.get_capability_info("anthropic", "claude-sonnet-4-5")

        # Verify
        assert info is not None
        assert info.tool_calling_supported is True
        assert info.parallel_calls_supported is False
        assert len(info.other_capabilities) == 0

    def test_get_capability_info_with_exception(self, mock_tool_capabilities):
        """Test get_capability_info handles exceptions."""
        # Setup
        mock_tool_capabilities.is_tool_call_supported.side_effect = Exception("Capability error")

        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        info = coordinator.get_capability_info("anthropic", "claude-sonnet-4-5")

        # Verify - should return None
        assert info is None

    def test_should_use_tools(self, mock_tool_capabilities):
        """Test should_use_tools always returns True."""
        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        result = coordinator.should_use_tools()

        # Verify
        assert result is True

    def test_validate_capability_tool_calling_supported(self, mock_tool_capabilities):
        """Test validate_capability for tool_calling when supported."""
        # Setup
        mock_tool_capabilities.is_tool_call_supported.return_value = True

        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        result = coordinator.validate_capability(
            provider_name="anthropic",
            model="claude-sonnet-4-5",
            capability="tool_calling",
        )

        # Verify
        assert result is True

    def test_validate_capability_tool_calling_not_supported(self, mock_tool_capabilities):
        """Test validate_capability for tool_calling when not supported."""
        # Setup
        mock_tool_capabilities.is_tool_call_supported.return_value = False

        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        result = coordinator.validate_capability(
            provider_name="anthropic",
            model="claude-2",
            capability="tool_calling",
        )

        # Verify
        assert result is False

    def test_validate_capability_unknown_capability(self, mock_tool_capabilities):
        """Test validate_capability for unknown capability."""
        # Setup
        mock_tool_capabilities.is_tool_call_supported.return_value = True

        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        result = coordinator.validate_capability(
            provider_name="anthropic",
            model="claude-sonnet-4-5",
            capability="unknown_capability",
        )

        # Verify - should default to False for unknown capabilities
        # (because get_capability_info returns None when exception occurs)
        assert result is False

    def test_validate_capability_no_provider(self, mock_tool_capabilities):
        """Test validate_capability without provider."""
        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        result = coordinator.validate_capability(
            provider_name=None,
            model="unknown-model",
            capability="tool_calling",
        )

        # Verify - should default to True
        assert result is True

    def test_get_capable_models_for_tool_calling(self, mock_tool_capabilities):
        """Test get_capable_models for tool_calling capability."""
        # Setup
        mock_tool_capabilities.get_supported_models.return_value = [
            "claude-3-opus",
            "claude-3-sonnet",
        ]

        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        models = coordinator.get_capable_models("anthropic", "tool_calling")

        # Verify
        assert len(models) == 2
        assert "claude-3-opus" in models

    def test_get_capable_models_for_other_capability(self, mock_tool_capabilities):
        """Test get_capable_models for other capabilities."""
        coordinator = ToolCapabilityCoordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        # Execute
        models = coordinator.get_capable_models("anthropic", "vision")

        # Verify - should return empty for now (placeholder)
        assert models == []


class TestCreateToolCapabilityCoordinator:
    """Test suite for factory function."""

    def test_factory_with_defaults(self, mock_tool_capabilities):
        """Test factory creates coordinator with defaults."""
        coordinator = create_tool_capability_coordinator(
            tool_capabilities=mock_tool_capabilities,
        )

        assert coordinator._tool_capabilities == mock_tool_capabilities
        assert coordinator._console is None
        assert coordinator._warn_once is True

    def test_factory_with_console(self, mock_tool_capabilities, mock_console):
        """Test factory with console."""
        coordinator = create_tool_capability_coordinator(
            tool_capabilities=mock_tool_capabilities,
            console=mock_console,
        )

        assert coordinator._console == mock_console

    def test_factory_with_warn_once_false(self, mock_tool_capabilities):
        """Test factory with warn_once=False."""
        coordinator = create_tool_capability_coordinator(
            tool_capabilities=mock_tool_capabilities,
            warn_once=False,
        )

        assert coordinator._warn_once is False
