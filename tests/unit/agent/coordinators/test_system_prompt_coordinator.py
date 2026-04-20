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

"""Unit tests for SystemPromptCoordinator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from victor.agent.coordinators.system_prompt_coordinator import (
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
    analyzer.classify_task_keywords.return_value = {
        "task_type": "general",
        "confidence": 0.5,
    }
    analyzer.classify_task_with_context.return_value = {
        "task_type": "coding",
        "confidence": 0.8,
    }
    return analyzer


@pytest.fixture()
def mock_tools():
    return MagicMock()


@pytest.fixture()
def coordinator(mock_prompt_builder, mock_task_analyzer, mock_tools):
    return SystemPromptCoordinator(
        prompt_builder=mock_prompt_builder,
        get_context_window=lambda: 65536,
        provider_name="anthropic",
        model_name="claude-3-sonnet",
        get_tools=lambda: mock_tools,
        get_mode_controller=lambda: None,
        task_analyzer=mock_task_analyzer,
        session_id="test-session",
    )


class TestSystemPromptCoordinator:
    """Test suite for SystemPromptCoordinator."""

    def test_build_system_prompt_large_context(self, mock_prompt_builder, mock_task_analyzer):
        """Prompt includes parallel read budget for >= 32K context."""
        coord = SystemPromptCoordinator(
            prompt_builder=mock_prompt_builder,
            get_context_window=lambda: 65536,
            provider_name="anthropic",
            model_name="claude-3-sonnet",
            get_tools=lambda: MagicMock(),
            get_mode_controller=lambda: None,
            task_analyzer=mock_task_analyzer,
        )
        result = coord.build_system_prompt()
        assert "PARALLEL READ BUDGET" in result
        assert "You are a helpful assistant." in result

    def test_build_system_prompt_small_context(self, mock_prompt_builder, mock_task_analyzer):
        """Prompt omits parallel read budget for < 32K context."""
        coord = SystemPromptCoordinator(
            prompt_builder=mock_prompt_builder,
            get_context_window=lambda: 8192,
            provider_name="anthropic",
            model_name="claude-3-sonnet",
            get_tools=lambda: MagicMock(),
            get_mode_controller=lambda: None,
            task_analyzer=mock_task_analyzer,
        )
        result = coord.build_system_prompt()
        assert "PARALLEL READ BUDGET" not in result
        assert result == "You are a helpful assistant."

    def test_resolve_shell_variant_delegates(self, coordinator, mock_tools):
        """Shell variant resolution delegates to shell_resolver module."""
        with patch(
            "victor.agent.shell_resolver.resolve_shell_variant",
            return_value="shell",
        ) as mock_resolve:
            result = coordinator.resolve_shell_variant("bash")
            assert result == "shell"
            mock_resolve.assert_called_once_with("bash", mock_tools, None)

    def test_classify_task_keywords(self, coordinator, mock_task_analyzer):
        """Task keyword classification delegates to TaskAnalyzer."""
        result = coordinator.classify_task_keywords("fix the bug")
        assert result["task_type"] == "general"
        mock_task_analyzer.classify_task_keywords.assert_called_once_with("fix the bug")

    def test_classify_task_with_context(self, coordinator, mock_task_analyzer):
        """Context-aware task classification delegates to TaskAnalyzer."""
        history = [{"role": "user", "content": "hello"}]
        result = coordinator.classify_task_with_context("write a function", history)
        assert result["task_type"] == "coding"
        mock_task_analyzer.classify_task_with_context.assert_called_once_with(
            "write a function", history
        )

    def test_classify_task_with_context_no_history(self, coordinator, mock_task_analyzer):
        """Context classification works without history."""
        coordinator.classify_task_with_context("write a function")
        mock_task_analyzer.classify_task_with_context.assert_called_once_with(
            "write a function", None
        )

    def test_emit_prompt_used_event_no_hooks(self, coordinator):
        """RL event emission is a no-op when hooks are not available."""
        with patch(
            "victor.framework.rl.hooks.get_rl_hooks",
            return_value=None,
        ):
            # Should not raise
            coordinator._emit_prompt_used_event("test prompt")

    def test_emit_prompt_used_event_local_provider(self, mock_prompt_builder, mock_task_analyzer):
        """Local provider gets 'detailed' prompt style in RL event."""
        coord = SystemPromptCoordinator(
            prompt_builder=mock_prompt_builder,
            get_context_window=lambda: 65536,
            provider_name="ollama",
            model_name="llama3",
            get_tools=lambda: MagicMock(),
            get_mode_controller=lambda: None,
            task_analyzer=mock_task_analyzer,
        )
        mock_hooks = MagicMock()
        with patch(
            "victor.framework.rl.hooks.get_rl_hooks",
            return_value=mock_hooks,
        ):
            coord._emit_prompt_used_event("test prompt")
            mock_hooks.emit.assert_called_once()
            event = mock_hooks.emit.call_args[0][0]
            assert event.metadata["prompt_style"] == "detailed"

    def test_emit_prompt_used_event_cloud_provider(self, coordinator):
        """Cloud provider gets 'structured' prompt style in RL event."""
        mock_hooks = MagicMock()
        with patch(
            "victor.framework.rl.hooks.get_rl_hooks",
            return_value=mock_hooks,
        ):
            coordinator._emit_prompt_used_event("test prompt")
            mock_hooks.emit.assert_called_once()
            event = mock_hooks.emit.call_args[0][0]
            assert event.metadata["prompt_style"] == "structured"

    def test_emit_prompt_used_event_exception_suppressed(self, coordinator):
        """RL hook exceptions are suppressed, not propagated."""
        with patch(
            "victor.framework.rl.hooks.get_rl_hooks",
            side_effect=RuntimeError("hook failed"),
        ):
            # Should not raise
            coordinator._emit_prompt_used_event("test prompt")
