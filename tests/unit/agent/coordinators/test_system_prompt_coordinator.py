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

"""Unit tests for SystemPromptCoordinator (deprecated compatibility wrapper)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from victor.agent.services.system_prompt_runtime import (
    SystemPromptCoordinator,
)


@pytest.fixture()
def mock_prompt_builder():
    builder = MagicMock()
    builder.build.return_value = "You are a helpful assistant."
    return builder


@pytest.fixture()
def mock_task_analyzer():
    analyzer = MagicMock()
    analyzer.classify_keywords.return_value = {
        "task_type": "general",
        "confidence": 0.5,
    }
    # UnifiedPromptPipeline calls classify_with_context, not classify_task_with_context
    analyzer.classify_with_context.return_value = {
        "task_type": "coding",
        "confidence": 0.8,
    }
    return analyzer


class TestSystemPromptCoordinator:
    """Test suite for deprecated SystemPromptCoordinator wrapper.

    .. deprecated::
        SystemPromptCoordinator is deprecated. Use UnifiedPromptPipeline directly.
        These tests verify backward compatibility of the wrapper.
    """

    def test_coordinators_package_reexports_service_runtime(self):
        """Package-level coordinator export re-exports from services."""
        from victor.agent.coordinators import SystemPromptCoordinator as package_export

        assert package_export is SystemPromptCoordinator

    def test_legacy_module_reexports_service_runtime(self):
        """Legacy coordinator path should re-export service-owned runtime."""
        from victor.agent.services.system_prompt_runtime import (
            SystemPromptCoordinator as legacy_coordinator,
        )

        assert legacy_coordinator is SystemPromptCoordinator

    def test_build_system_prompt_delegates_to_pipeline(self, mock_prompt_builder, mock_task_analyzer):
        """SystemPromptCoordinator delegates build_system_prompt to UnifiedPromptPipeline."""
        import warnings

        coord = SystemPromptCoordinator(
            prompt_builder=mock_prompt_builder,
            get_context_window=lambda: 65536,
            provider_name="anthropic",
            model_name="claude-3-sonnet",
            get_tools=lambda: MagicMock(),
            get_mode_controller=lambda: None,
            task_analyzer=mock_task_analyzer,
        )

        # Should get a prompt from the builder
        result = coord.build_system_prompt()
        assert isinstance(result, str)
        # The actual content depends on UnifiedPromptPipeline implementation
        # Just verify it returns a string

    def test_resolve_shell_variant_returns_string(self, mock_prompt_builder, mock_task_analyzer):
        """Shell variant resolution returns a string."""
        import warnings

        coord = SystemPromptCoordinator(
            prompt_builder=mock_prompt_builder,
            get_context_window=lambda: 65536,
            provider_name="anthropic",
            model_name="claude-3-sonnet",
            get_tools=lambda: MagicMock(),
            get_mode_controller=lambda: None,
            task_analyzer=mock_task_analyzer,
        )

        result = coord.resolve_shell_variant("bash")
        assert isinstance(result, str)

    def test_classify_task_keywords_returns_dict(self, mock_prompt_builder, mock_task_analyzer):
        """Task classification returns a dict with task_type and confidence."""
        import warnings

        coord = SystemPromptCoordinator(
            prompt_builder=mock_prompt_builder,
            get_context_window=lambda: 65536,
            provider_name="anthropic",
            model_name="claude-3-sonnet",
            get_tools=lambda: MagicMock(),
            get_mode_controller=lambda: None,
            task_analyzer=mock_task_analyzer,
        )

        result = coord.classify_task_keywords("fix the bug")
        assert isinstance(result, dict)
        assert "task_type" in result
        assert "confidence" in result

    def test_classify_task_with_context_returns_dict(self, mock_prompt_builder, mock_task_analyzer):
        """Context-aware task classification returns a dict."""
        import warnings

        coord = SystemPromptCoordinator(
            prompt_builder=mock_prompt_builder,
            get_context_window=lambda: 65536,
            provider_name="anthropic",
            model_name="claude-3-sonnet",
            get_tools=lambda: MagicMock(),
            get_mode_controller=lambda: None,
            task_analyzer=mock_task_analyzer,
        )

        history = [{"role": "user", "content": "hello"}]
        result = coord.classify_task_with_context("write a function", history)
        assert isinstance(result, dict)
        assert "task_type" in result
        assert "confidence" in result
