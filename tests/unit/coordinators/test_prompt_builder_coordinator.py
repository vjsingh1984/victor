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

"""Unit tests for PromptBuilderCoordinator."""

import pytest
from unittest.mock import Mock

from victor.agent.coordinators.prompt_builder_coordinator import (
    PromptBuilderCoordinator,
    create_prompt_builder_coordinator,
)
from victor.agent.coordinators.prompt_builder_protocol import (
    PromptContext,
    PromptBuilderCoordinatorConfig,
)


@pytest.fixture
def coordinator_config():
    """Create coordinator config for testing."""
    return PromptBuilderCoordinatorConfig(
        cache_enabled=True,
        include_tool_hints=True,
        include_thinking_instructions=True,
        max_prompt_length=50000,
    )


@pytest.fixture
def base_prompt():
    """Create base system prompt."""
    return "You are a helpful AI assistant."


@pytest.fixture
def prompt_builder_coordinator(coordinator_config, base_prompt):
    """Create PromptBuilderCoordinator for testing."""
    return PromptBuilderCoordinator(
        config=coordinator_config,
        base_prompt=base_prompt,
    )


@pytest.fixture
def mock_prompt_contributor():
    """Create mock prompt contributor."""
    contributor = Mock()
    contributor.get_contribution = Mock(return_value="Additional context from contributor.")
    return contributor


class TestPromptBuilderCoordinator:
    """Test PromptBuilderCoordinator functionality."""

    def test_initialization(self, prompt_builder_coordinator, coordinator_config, base_prompt):
        """Test coordinator initialization."""
        assert prompt_builder_coordinator._config == coordinator_config
        assert prompt_builder_coordinator._base_prompt == base_prompt
        assert prompt_builder_coordinator._prompt_contributors == []
        assert prompt_builder_coordinator._prompt_cache == {}

    def test_create_prompt_builder_coordinator_factory(self):
        """Test factory function."""
        config = PromptBuilderCoordinatorConfig()
        coordinator = create_prompt_builder_coordinator(
            config=config,
            base_prompt="Test prompt",
        )

        assert isinstance(coordinator, PromptBuilderCoordinator)
        assert coordinator._config == config
        assert coordinator._base_prompt == "Test prompt"

    def test_get_system_prompt(self, prompt_builder_coordinator, base_prompt):
        """Test getting system prompt."""
        assert prompt_builder_coordinator.get_system_prompt() == base_prompt

    def test_set_system_prompt(self, prompt_builder_coordinator):
        """Test setting system prompt."""
        new_prompt = "You are a coding assistant."
        prompt_builder_coordinator.set_system_prompt(new_prompt)

        assert prompt_builder_coordinator.get_system_prompt() == new_prompt
        # Cache should be invalidated
        assert prompt_builder_coordinator._prompt_cache == {}

    def test_append_to_system_prompt(self, prompt_builder_coordinator, base_prompt):
        """Test appending to system prompt."""
        additional = "Focus on Python code."
        prompt_builder_coordinator.append_to_system_prompt(additional)

        current_prompt = prompt_builder_coordinator.get_system_prompt()
        assert additional in current_prompt
        assert current_prompt.startswith(base_prompt)
        # Cache should be invalidated
        assert prompt_builder_coordinator._prompt_cache == {}

    def test_build_prompt_basic(self, prompt_builder_coordinator, base_prompt):
        """Test basic prompt building."""
        context = PromptContext()
        prompt = prompt_builder_coordinator.build_prompt(context)

        assert base_prompt in prompt
        assert prompt_builder_coordinator._config.cache_enabled
        # Check caching
        assert len(prompt_builder_coordinator._prompt_cache) == 1

    def test_build_prompt_with_mode(self, prompt_builder_coordinator):
        """Test prompt building with mode."""
        context = PromptContext(mode="build")
        prompt = prompt_builder_coordinator.build_prompt(context, include_hints=False)

        assert "BUILD mode" in prompt
        assert "full edits" in prompt

    def test_build_prompt_plan_mode(self, prompt_builder_coordinator):
        """Test prompt building with plan mode."""
        context = PromptContext(mode="plan")
        prompt = prompt_builder_coordinator.build_prompt(context, include_hints=False)

        assert "PLAN mode" in prompt
        assert "planning and analysis" in prompt

    def test_build_prompt_explore_mode(self, prompt_builder_coordinator):
        """Test prompt building with explore mode."""
        context = PromptContext(mode="explore")
        prompt = prompt_builder_coordinator.build_prompt(context, include_hints=False)

        assert "EXPLORE mode" in prompt
        assert "exploration and understanding" in prompt

    def test_build_prompt_thinking_disabled(self, prompt_builder_coordinator):
        """Test prompt building with thinking disabled."""
        base_with_thinking = (
            "You are helpful.\n\n"
            "Use your thinking process to solve problems step by step.\n"
            "Always show your reasoning."
        )
        prompt_builder_coordinator._base_prompt = base_with_thinking

        context = PromptContext(thinking_enabled=False)
        prompt = prompt_builder_coordinator.build_prompt(context, include_hints=False)

        # Should remove thinking-related instructions
        assert "thinking process" not in prompt.lower() or len(prompt) < len(base_with_thinking)

    def test_build_prompt_with_tool_hints(self, prompt_builder_coordinator):
        """Test prompt building with tool hints."""
        # Need to provide a tool_set context
        context = PromptContext(tool_set=Mock())
        prompt = prompt_builder_coordinator.build_prompt(context, include_hints=True)

        # Tool hints should be included when tool_set is provided
        assert "Available Tools:" in prompt or "BUILD mode" in prompt

    def test_build_prompt_without_tool_hints(self, prompt_builder_coordinator):
        """Test prompt building without tool hints."""
        context = PromptContext()
        prompt = prompt_builder_coordinator.build_prompt(context, include_hints=False)

        assert "Available Tools:" not in prompt

    def test_build_prompt_caching(self, prompt_builder_coordinator):
        """Test prompt caching."""
        context = PromptContext(mode="build", thinking_enabled=False)

        # First call - should build and cache
        prompt1 = prompt_builder_coordinator.build_prompt(context)
        cache_size_1 = len(prompt_builder_coordinator._prompt_cache)

        # Second call with same context - should use cache
        prompt2 = prompt_builder_coordinator.build_prompt(context)
        cache_size_2 = len(prompt_builder_coordinator._prompt_cache)

        assert prompt1 == prompt2
        assert cache_size_1 == cache_size_2

    def test_build_prompt_cache_invalidation(self, prompt_builder_coordinator):
        """Test cache invalidation."""
        context = PromptContext()

        # Build prompt to populate cache
        prompt_builder_coordinator.build_prompt(context)
        assert len(prompt_builder_coordinator._prompt_cache) > 0

        # Invalidate cache
        prompt_builder_coordinator.invalidate_prompt_cache()
        assert len(prompt_builder_coordinator._prompt_cache) == 0

    def test_build_prompt_different_contexts(self, prompt_builder_coordinator):
        """Test building prompts with different contexts."""
        context1 = PromptContext(mode="build")
        context2 = PromptContext(mode="plan")

        prompt1 = prompt_builder_coordinator.build_prompt(context1)
        prompt2 = prompt_builder_coordinator.build_prompt(context2)

        assert prompt1 != prompt2
        assert "BUILD mode" in prompt1
        assert "PLAN mode" in prompt2

    def test_apply_mode_prompt_build(self, prompt_builder_coordinator):
        """Test applying build mode prompt."""
        base = "You are helpful."
        modified = prompt_builder_coordinator.apply_mode_prompt(base, "build")

        assert "BUILD mode" in modified
        # Mode modifications are appended to base
        assert base in modified

    def test_apply_mode_prompt_plan(self, prompt_builder_coordinator):
        """Test applying plan mode prompt."""
        base = "You are helpful."
        modified = prompt_builder_coordinator.apply_mode_prompt(base, "plan")

        assert "PLAN mode" in modified
        assert "planning and analysis" in modified

    def test_apply_mode_prompt_explore(self, prompt_builder_coordinator):
        """Test applying explore mode prompt."""
        base = "You are helpful."
        modified = prompt_builder_coordinator.apply_mode_prompt(base, "explore")

        assert "EXPLORE mode" in modified
        assert "exploration and understanding" in modified

    def test_apply_mode_prompt_unknown(self, prompt_builder_coordinator):
        """Test applying unknown mode (no modification)."""
        base = "You are helpful."
        modified = prompt_builder_coordinator.apply_mode_prompt(base, "unknown")

        # Should just append base without mode-specific content
        assert modified == base

    def test_apply_tool_hints(self, prompt_builder_coordinator):
        """Test applying tool hints."""
        base = "You are helpful."
        modified = prompt_builder_coordinator.apply_tool_hints(base, tool_set=Mock())

        assert "Available Tools:" in modified
        assert modified.startswith(base)

    def test_apply_tool_hints_no_tool_set(self, prompt_builder_coordinator):
        """Test applying tool hints without tool set."""
        base = "You are helpful."
        modified = prompt_builder_coordinator.apply_tool_hints(base, tool_set=None)

        # Should return base unchanged
        assert modified == base

    def test_build_thinking_disabled_prompt(self, prompt_builder_coordinator):
        """Test building thinking disabled prompt."""
        base = (
            "You are helpful.\n\n"
            "Thinking process:\n"
            "Step 1: Analyze\n"
            "Step 2: Solve\n"
        )

        modified = prompt_builder_coordinator.build_thinking_disabled_prompt(base)

        # Should remove or reduce thinking-related content
        assert "Thinking process:" not in modified or len(modified) < len(base)

    def test_register_prompt_contributor(
        self,
        prompt_builder_coordinator,
        mock_prompt_contributor,
    ):
        """Test registering prompt contributor."""
        prompt_builder_coordinator.register_prompt_contributor(mock_prompt_contributor)

        assert mock_prompt_contributor in prompt_builder_coordinator._prompt_contributors

    def test_build_prompt_with_contributor(
        self,
        prompt_builder_coordinator,
        mock_prompt_contributor,
    ):
        """Test building prompt with contributor."""
        prompt_builder_coordinator.register_prompt_contributor(mock_prompt_contributor)

        context = PromptContext()
        prompt = prompt_builder_coordinator.build_prompt(context, include_hints=False)

        assert "Additional context from contributor." in prompt
        mock_prompt_contributor.get_contribution.assert_called_once()

    def test_build_prompt_multiple_contributors(
        self,
        prompt_builder_coordinator,
    ):
        """Test building prompt with multiple contributors."""
        contributor1 = Mock()
        contributor1.get_contribution = Mock(return_value="Context 1")

        contributor2 = Mock()
        contributor2.get_contribution = Mock(return_value="Context 2")

        prompt_builder_coordinator.register_prompt_contributor(contributor1)
        prompt_builder_coordinator.register_prompt_contributor(contributor2)

        context = PromptContext()
        prompt = prompt_builder_coordinator.build_prompt(context, include_hints=False)

        assert "Context 1" in prompt
        assert "Context 2" in prompt

    def test_build_prompt_contributor_error_handling(
        self,
        prompt_builder_coordinator,
    ):
        """Test that contributor errors are handled gracefully."""
        failing_contributor = Mock()
        failing_contributor.get_contribution = Mock(side_effect=Exception("Contributor failed"))

        working_contributor = Mock()
        working_contributor.get_contribution = Mock(return_value="Working context")

        prompt_builder_coordinator.register_prompt_contributor(failing_contributor)
        prompt_builder_coordinator.register_prompt_contributor(working_contributor)

        context = PromptContext()
        # Should not raise exception
        prompt = prompt_builder_coordinator.build_prompt(context, include_hints=False)

        assert "Working context" in prompt

    def test_build_prompt_max_length_truncation(self, prompt_builder_coordinator):
        """Test that prompts exceeding max length are truncated."""
        # Set a small max length
        prompt_builder_coordinator._config.max_prompt_length = 100

        # Create a long base prompt
        long_prompt = "x" * 200
        prompt_builder_coordinator._base_prompt = long_prompt

        context = PromptContext()
        prompt = prompt_builder_coordinator.build_prompt(context, include_hints=False)

        # Should be truncated
        assert len(prompt) <= 100

    def test_get_prompt_contributors(
        self,
        prompt_builder_coordinator,
        mock_prompt_contributor,
    ):
        """Test getting prompt contributors."""
        prompt_builder_coordinator.register_prompt_contributor(mock_prompt_contributor)

        contributors = prompt_builder_coordinator.get_prompt_contributors()

        assert mock_prompt_contributor in contributors
        # Should return a copy, not the original list
        assert contributors is not prompt_builder_coordinator._prompt_contributors

    def test_cache_key_generation(self, prompt_builder_coordinator):
        """Test that cache keys are generated correctly."""
        context1 = PromptContext(mode="build", thinking_enabled=False)
        context2 = PromptContext(mode="build", thinking_enabled=False)
        context3 = PromptContext(mode="plan", thinking_enabled=False)

        key1 = prompt_builder_coordinator._get_cache_key(context1, include_hints=True)
        key2 = prompt_builder_coordinator._get_cache_key(context2, include_hints=True)
        key3 = prompt_builder_coordinator._get_cache_key(context3, include_hints=True)

        # Same context should produce same key
        assert key1 == key2
        # Different context should produce different key
        assert key1 != key3
