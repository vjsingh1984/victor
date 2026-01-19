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

"""Tests for LegacyAPIMixin backward compatibility.

These tests verify that all deprecated methods in LegacyAPIMixin:
1. Issue appropriate deprecation warnings
2. Maintain backward compatibility
3. Provide correct migration paths
"""

import warnings
from pathlib import Path
from typing import Dict, Set
from unittest.mock import MagicMock, patch

import pytest

from victor.agent.mixins.legacy_api import LegacyAPIMixin


class MockOrchestrator(LegacyAPIMixin):
    """Minimal mock orchestrator for testing LegacyAPIMixin."""

    def __init__(self):
        # Mock all required coordinators
        self._mode_workflow_team_coordinator = MagicMock()
        self._vertical_context = MagicMock()
        self._configuration_manager = MagicMock()
        self._tool_access_controller = MagicMock()
        self._project_context = MagicMock()
        self._vertical_integration_adapter = MagicMock()
        self._vertical_middleware = []
        self._vertical_safety_patterns = []
        self._team_coordinator = MagicMock()
        self._metrics_coordinator = MagicMock()
        self._cumulative_token_usage = {"input": 0, "output": 0, "total": 0}
        self._state_coordinator = MagicMock()
        self._context_compactor = MagicMock()
        self._usage_analytics = MagicMock()
        self._sequence_tracker = MagicMock()
        self._code_correction_middleware = MagicMock()
        self._safety_checker = MagicMock()
        self._auto_committer = MagicMock()
        self.search_router = MagicMock()
        self.conversation_state = MagicMock()
        self.conversation_state.state = MagicMock()
        self.conversation_state.state.modified_files = []
        self.unified_tracker = MagicMock()
        self.unified_tracker.tool_calls_used = 5
        self.unified_tracker.tool_budget = 50
        self.unified_tracker.iteration_count = 2
        self.unified_tracker.max_iterations = 25
        self.provider_name = "anthropic"
        self.model = "claude-sonnet-4-5"
        self._provider_manager = MagicMock()
        self._provider_manager.get_info.return_value = {"provider": "anthropic"}
        self._tool_access_coordinator = MagicMock()
        self._tool_access_coordinator.is_tool_enabled.return_value = True
        self.tools = MagicMock()
        self.tools.list_tools.return_value = ["read_file", "write_file"]
        self.prompt_builder = MagicMock()
        self.prompt_builder.build.return_value = "System prompt"
        self.conversation = MagicMock()
        self.conversation.messages = []
        self._search_coordinator = MagicMock()
        self.tool_selector = MagicMock()


class TestLegacyAPIMixinWarnings:
    """Test that all deprecated methods issue warnings."""

    def setup_method(self):
        """Create a fresh mock orchestrator for each test."""
        self.orchestrator = MockOrchestrator()

    def test_set_vertical_context_issues_warning(self):
        """Test set_vertical_context issues deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.orchestrator.set_vertical_context(MagicMock())

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "set_vertical_context is deprecated" in str(w[0].message)
            assert "VerticalContext.set_context()" in str(w[0].message)

    def test_set_tiered_tool_config_issues_warning(self):
        """Test set_tiered_tool_config issues deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.orchestrator.set_tiered_tool_config(MagicMock())

            assert len(w) == 1
            assert "set_tiered_tool_config is deprecated" in str(w[0].message)

    def test_set_workspace_issues_warning(self):
        """Test set_workspace issues deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("victor.config.settings.set_project_root"):
                self.orchestrator.set_workspace(Path("/tmp"))

            assert len(w) == 1
            assert "set_workspace is deprecated" in str(w[0].message)

    def test_get_tool_usage_stats_issues_warning(self):
        """Test get_tool_usage_stats issues deprecation warning."""
        self.orchestrator._metrics_coordinator.get_tool_usage_stats.return_value = {}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.orchestrator.get_tool_usage_stats()

            assert len(w) == 1
            assert "get_tool_usage_stats is deprecated" in str(w[0].message)

    def test_get_token_usage_issues_warning(self):
        """Test get_token_usage issues deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.orchestrator.get_token_usage()

            assert len(w) == 1
            assert "get_token_usage is deprecated" in str(w[0].message)

    def test_get_conversation_stage_issues_warning(self):
        """Test get_conversation_stage issues deprecation warning."""
        self.orchestrator._state_coordinator.get_stage.return_value = "EXECUTION"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Import inside test to avoid import errors in module scope
            from victor.agent.orchestrator_imports import ConversationStage

            result = self.orchestrator.get_conversation_stage()

            assert len(w) == 1
            assert "get_conversation_stage is deprecated" in str(w[0].message)
            assert result == ConversationStage.EXECUTION

    def test_get_observed_files_issues_warning(self):
        """Test get_observed_files issues deprecation warning."""
        self.orchestrator._state_coordinator.observed_files = {"/tmp/file1.py"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.orchestrator.get_observed_files()

            assert len(w) == 1
            assert "get_observed_files is deprecated" in str(w[0].message)
            assert result == {"/tmp/file1.py"}


class TestLegacyAPIMixinBackwardCompatibility:
    """Test that deprecated methods maintain backward compatibility."""

    def setup_method(self):
        """Create a fresh mock orchestrator for each test."""
        self.orchestrator = MockOrchestrator()

    def test_set_vertical_context_still_works(self):
        """Test set_vertical_context still works correctly."""
        mock_context = MagicMock()
        mock_context.vertical_name = "coding"

        self.orchestrator.set_vertical_context(mock_context)

        assert self.orchestrator._vertical_context == mock_context
        self.orchestrator._mode_workflow_team_coordinator.set_vertical_context.assert_called_once()

    def test_get_tool_usage_stats_returns_correct_data(self):
        """Test get_tool_usage_stats returns correct data."""
        expected_stats = {"tool_calls": 10}
        self.orchestrator._metrics_coordinator.get_tool_usage_stats.return_value = expected_stats
        self.orchestrator.conversation_state.get_state_summary.return_value = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.orchestrator.get_tool_usage_stats()

        assert result == expected_stats

    def test_get_token_usage_returns_correct_data(self):
        """Test get_token_usage returns correct data."""
        mock_usage = MagicMock()
        self.orchestrator._metrics_coordinator.get_token_usage.return_value = mock_usage

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.orchestrator.get_token_usage()

        assert result == mock_usage

    def test_reset_token_usage_works(self):
        """Test reset_token_usage resets counters."""
        self.orchestrator._cumulative_token_usage = {"input": 100, "output": 200, "total": 300}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.orchestrator.reset_token_usage()

        assert self.orchestrator._cumulative_token_usage["input"] == 0
        assert self.orchestrator._cumulative_token_usage["output"] == 0
        assert self.orchestrator._cumulative_token_usage["total"] == 0

    def test_get_tool_calls_count_returns_correct_value(self):
        """Test get_tool_calls_count returns correct value."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.orchestrator.get_tool_calls_count()

        assert result == 5

    def test_get_tool_budget_returns_correct_value(self):
        """Test get_tool_budget returns correct value."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.orchestrator.get_tool_budget()

        assert result == 50

    def test_get_iteration_count_returns_correct_value(self):
        """Test get_iteration_count returns correct value."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.orchestrator.get_iteration_count()

        assert result == 2

    def test_get_max_iterations_returns_correct_value(self):
        """Test get_max_iterations returns correct value."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.orchestrator.get_max_iterations()

        assert result == 25

    def test_current_provider_returns_correct_value(self):
        """Test current_provider returns correct value."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.orchestrator.current_provider()

        assert result == "anthropic"

    def test_current_model_returns_correct_value(self):
        """Test current_model returns correct value."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.orchestrator.current_model()

        assert result == "claude-sonnet-4-5"

    def test_get_available_tools_returns_correct_tools(self):
        """Test get_available_tools returns correct tools."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.orchestrator.get_available_tools()

        assert result == {"read_file", "write_file"}

    def test_is_tool_enabled_delegates_to_coordinator(self):
        """Test is_tool_enabled delegates to coordinator."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.orchestrator.is_tool_enabled("read_file")

        assert result is True
        self.orchestrator._tool_access_coordinator.is_tool_enabled.assert_called_once_with("read_file")

    def test_get_system_prompt_returns_prompt(self):
        """Test get_system_prompt returns prompt."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.orchestrator.get_system_prompt()

        assert result == "System prompt"

    def test_set_system_prompt_sets_prompt(self):
        """Test set_system_prompt sets prompt."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.orchestrator.set_system_prompt("New prompt")

        self.orchestrator.prompt_builder.set_custom_prompt.assert_called_once_with("New prompt")

    def test_append_to_system_prompt_works(self):
        """Test append_to_system_prompt works correctly."""
        self.orchestrator.prompt_builder.build.return_value = "Base prompt"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.orchestrator.append_to_system_prompt("Additional")

        # Should call set_custom_prompt with combined content
        call_args = self.orchestrator.prompt_builder.set_custom_prompt.call_args
        assert "Base prompt" in str(call_args)
        assert "Additional" in str(call_args)

    def test_get_messages_returns_correct_format(self):
        """Test get_messages returns correct format."""
        from victor.integrations.protocol.messages import MessageRole

        msg1 = MagicMock()
        msg1.role = MessageRole.USER
        msg1.content = "Hello"
        msg2 = MagicMock()
        msg2.role = MessageRole.ASSISTANT
        msg2.content = "Hi there"
        self.orchestrator.conversation.messages = [msg1, msg2]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.orchestrator.get_messages()

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there"}

    def test_get_message_count_returns_correct_count(self):
        """Test get_message_count returns correct count."""
        msg1 = MagicMock()
        msg2 = MagicMock()
        self.orchestrator.conversation.messages = [msg1, msg2]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.orchestrator.get_message_count()

        assert result == 2


class TestLegacyAPIMixinCategories:
    """Test each category of deprecated methods."""

    def setup_method(self):
        """Create a fresh mock orchestrator for each test."""
        self.orchestrator = MockOrchestrator()

    def test_vertical_configuration_methods_all_issue_warnings(self):
        """Test Category 1: Vertical configuration methods issue warnings."""
        methods = [
            ("set_vertical_context", [MagicMock()]),
            ("set_tiered_tool_config", [MagicMock()]),
            ("set_workspace", [Path("/tmp")]),
        ]

        for method_name, args in methods:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                method = getattr(self.orchestrator, method_name)
                if method_name == "set_workspace":
                    with patch("victor.config.settings.set_project_root"):
                        method(*args)
                else:
                    method(*args)

                assert len(w) == 1, f"{method_name} should issue warning"
                assert issubclass(w[0].category, DeprecationWarning)

    def test_metrics_methods_all_issue_warnings(self):
        """Test Category 3: Metrics methods issue warnings."""
        self.orchestrator._metrics_coordinator.get_tool_usage_stats.return_value = {}
        self.orchestrator._metrics_coordinator.get_token_usage.return_value = MagicMock()
        self.orchestrator._metrics_coordinator.get_last_stream_metrics.return_value = None
        self.orchestrator._metrics_coordinator.get_streaming_metrics_summary.return_value = {}
        self.orchestrator._metrics_coordinator.get_streaming_metrics_history.return_value = []
        self.orchestrator._metrics_coordinator.get_session_cost_summary.return_value = {}
        self.orchestrator._metrics_coordinator.get_session_cost_formatted.return_value = "$0.00"

        methods = [
            "get_tool_usage_stats",
            "get_token_usage",
            "reset_token_usage",
            "get_last_stream_metrics",
            "get_streaming_metrics_summary",
            "get_streaming_metrics_history",
            "get_session_cost_summary",
            "get_session_cost_formatted",
        ]

        for method_name in methods:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                method = getattr(self.orchestrator, method_name)
                if method_name == "get_streaming_metrics_history":
                    method(10)
                elif method_name == "export_session_costs":
                    method("/tmp/path.json")
                else:
                    method()

                assert len(w) == 1, f"{method_name} should issue warning"
                assert issubclass(w[0].category, DeprecationWarning)

    def test_removal_version_in_all_warnings(self):
        """Test that all warnings include v0.7.0 removal version."""
        # Sample a few key methods
        methods = [
            "set_vertical_context",
            "get_tool_usage_stats",
            "get_token_usage",
            "get_conversation_stage",
        ]

        for method_name in methods:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                method = getattr(self.orchestrator, method_name)

                # Call with appropriate args
                if method_name == "set_vertical_context":
                    method(MagicMock())
                elif method_name == "get_conversation_stage":
                    self.orchestrator._state_coordinator.get_stage.return_value = "EXECUTION"
                    method()
                else:
                    method()

                assert len(w) == 1
                assert "0.7.0" in str(w[0].message), f"{method_name} should mention v0.7.0"
