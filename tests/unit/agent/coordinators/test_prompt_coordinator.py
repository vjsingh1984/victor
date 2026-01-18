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

"""Tests for PromptCoordinator.

Tests the prompt building coordination functionality.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from victor.agent.coordinators.prompt_coordinator import (
    PromptCoordinator,
    PromptBuildError,
    BasePromptContributor,
    SystemPromptContributor,
    TaskHintContributor,
    PromptBuilderCoordinator,
    IPromptBuilderCoordinator,
)
from victor.protocols import PromptContext


class MockPromptContributor(BasePromptContributor):
    """Mock prompt contributor for testing."""

    def __init__(self, contribution, priority=50):
        super().__init__(priority=priority)
        self._contribution = contribution

    async def contribute(self, context):
        # If contribution is callable, call it with context
        if callable(self._contribution):
            return self._contribution(context)
        return self._contribution


class TestBasePromptContributor:
    """Tests for BasePromptContributor."""

    @pytest.mark.asyncio
    async def test_contribute_returns_empty_string(self):
        """Test default contribute implementation."""
        contributor = BasePromptContributor(priority=50)
        context = {}

        result = await contributor.contribute(context)

        assert result == ""

    def test_priority(self):
        """Test priority getter."""
        contributor = BasePromptContributor(priority=75)

        assert contributor.priority() == 75


class TestSystemPromptContributor:
    """Tests for SystemPromptContributor."""

    @pytest.mark.asyncio
    async def test_contribute(self):
        """Test contributing system prompt."""
        prompt = "You are a helpful assistant."
        contributor = SystemPromptContributor(prompt, priority=100)

        result = await contributor.contribute({})

        assert result == prompt

    def test_priority(self):
        """Test priority getter."""
        contributor = SystemPromptContributor("test", priority=50)

        assert contributor.priority() == 50


class TestTaskHintContributor:
    """Tests for TaskHintContributor."""

    @pytest.mark.asyncio
    async def test_contribute(self):
        """Test contributing task hint."""
        hints = {"simple": "Keep it brief", "medium": "Be thorough"}
        contributor = TaskHintContributor(hints)

        result = await contributor.contribute({"task_type": "simple"})

        assert result == "Keep it brief"

    @pytest.mark.asyncio
    async def test_contribute_unknown_task(self):
        """Test contributing hint for unknown task."""
        hints = {"simple": "Keep it brief"}
        contributor = TaskHintContributor(hints)

        result = await contributor.contribute({"task_type": "unknown"})

        assert result == ""

    def test_set_hint(self):
        """Test setting hint for a task type."""
        contributor = TaskHintContributor({})

        contributor.set_hint("debug", "Check logs first")

        assert contributor.get_hints()["debug"] == "Check logs first"


class TestPromptCoordinator:
    """Tests for PromptCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create empty coordinator."""
        return PromptCoordinator(contributors=[])

    @pytest.fixture
    def coordinator_with_contributors(self):
        """Create coordinator with mock contributors."""
        contributor1 = MockPromptContributor("Section 1\n", priority=10)
        contributor2 = MockPromptContributor("Section 2\n", priority=50)
        contributor3 = MockPromptContributor("Section 3\n", priority=100)

        return PromptCoordinator(contributors=[contributor1, contributor2, contributor3])

    def test_init_empty(self):
        """Test initialization with no contributors."""
        coordinator = PromptCoordinator(contributors=[])

        assert coordinator._contributors == []
        assert coordinator._enable_cache is True

    def test_init_with_contributors(self):
        """Test initialization with contributors."""
        contributor = MockPromptContributor("test", priority=50)
        coordinator = PromptCoordinator(contributors=[contributor], enable_cache=False)

        assert len(coordinator._contributors) == 1
        assert coordinator._enable_cache is False

    def test_init_sorts_contributors_by_priority(self):
        """Test that contributors are sorted by priority."""
        contributor1 = MockPromptContributor("", priority=10)
        contributor2 = MockPromptContributor("", priority=100)
        contributor3 = MockPromptContributor("", priority=50)

        coordinator = PromptCoordinator(contributors=[contributor1, contributor2, contributor3])

        # Should be sorted descending: 100, 50, 10
        assert coordinator._contributors[0].priority() == 100
        assert coordinator._contributors[1].priority() == 50
        assert coordinator._contributors[2].priority() == 10

    @pytest.mark.asyncio
    async def test_build_system_prompt_no_contributors(self, coordinator):
        """Test building prompt with no contributors."""
        context = {}

        prompt = await coordinator.build_system_prompt(context)

        assert prompt == ""

    @pytest.mark.asyncio
    async def test_build_system_prompt_merges_contributions(
        self,
        coordinator_with_contributors,
    ):
        """Test that prompt is built from all contributors."""
        context = {}

        prompt = await coordinator_with_contributors.build_system_prompt(context)

        assert "Section 1" in prompt
        assert "Section 2" in prompt
        assert "Section 3" in prompt

    @pytest.mark.asyncio
    async def test_build_system_prompt_order_by_priority(
        self,
        coordinator_with_contributors,
    ):
        """Test that higher priority contributors appear first."""
        context = {}

        prompt = await coordinator_with_contributors.build_system_prompt(context)

        # Priority 100 should be first
        assert prompt.startswith("Section 3")
        assert "Section 2" in prompt
        assert "Section 1" in prompt

    @pytest.mark.asyncio
    async def test_build_system_prompt_caching(self, coordinator_with_contributors):
        """Test that built prompts are cached."""
        context = {}

        # First call
        prompt1 = await coordinator_with_contributors.build_system_prompt(context)
        # Second call should use cache
        prompt2 = await coordinator_with_contributors.build_system_prompt(context)

        assert prompt1 == prompt2

    @pytest.mark.asyncio
    async def test_invalidate_cache(self, coordinator_with_contributors):
        """Test cache invalidation."""
        context = {"task": "test"}

        # Build prompt
        await coordinator_with_contributors.build_system_prompt(context)

        # Invalidate cache
        coordinator_with_contributors.invalidate_cache(context)

        # Cache should be cleared
        assert len(coordinator_with_contributors._prompt_cache) == 0

    @pytest.mark.asyncio
    async def test_build_task_hint(self, coordinator_with_contributors):
        """Test building task-specific hint."""
        contributor = TaskHintContributor({"simple": "Be brief", "complex": "Be thorough"})
        coordinator = PromptCoordinator(contributors=[contributor], enable_cache=False)

        hint = await coordinator.build_task_hint("simple", {})

        assert hint == "Be brief"

    @pytest.mark.asyncio
    async def test_build_task_hint_empty_context(self, coordinator):
        """Test building task hint with empty context."""
        hint = await coordinator.build_task_hint("simple", {})

        assert hint == ""

    @pytest.mark.asyncio
    async def test_build_task_hint_caches_by_task_type(self, coordinator):
        """Test that hints are cached per task type."""
        # Build with contributors that return different values based on context
        contributor = MockPromptContributor(
            lambda ctx: f"Hint for {ctx.get('task_type', 'unknown')}", priority=50
        )
        coordinator = PromptCoordinator(contributors=[contributor])

        hint1 = await coordinator.build_task_hint("simple", {})
        hint2 = await coordinator.build_task_hint("complex", {})

        assert hint1 == "Hint for simple"
        assert hint2 == "Hint for complex"

    def test_invalidate_all_cache(self, coordinator_with_contributors):
        """Test invalidating all cache."""
        # Add some cache entries
        coordinator_with_contributors._prompt_cache["key1"] = "value1"
        coordinator_with_contributors._prompt_cache["key2"] = "value2"

        # Clear all
        coordinator_with_contributors.invalidate_cache()

        assert coordinator_with_contributors._prompt_cache == {}

    def test_add_contributor(self, coordinator):
        """Test adding a contributor."""
        contributor = MockPromptContributor("test", priority=75)
        coordinator.add_contributor(contributor)

        assert len(coordinator._contributors) == 1
        assert coordinator._contributors[0].priority() == 75

    def test_add_contributor_resorts(self, coordinator):
        """Test that adding a contributor triggers re-sorting."""
        contributor1 = MockPromptContributor("", priority=10)
        contributor2 = MockPromptContributor("", priority=100)

        coordinator.add_contributor(contributor1)
        coordinator.add_contributor(contributor2)

        # Should be sorted: 100, 10
        assert coordinator._contributors[0].priority() == 100
        assert coordinator._contributors[1].priority() == 10

        # Cache should be cleared when contributors change
        assert coordinator._prompt_cache == {}

    def test_remove_contributor(self, coordinator):
        """Test removing a contributor."""
        contributor = MockPromptContributor("test", priority=50)
        coordinator.add_contributor(contributor)

        coordinator.remove_contributor(contributor)

        assert len(coordinator._contributors) == 0

    def test_make_cache_key(self, coordinator):
        """Test cache key generation."""
        context1 = {"task": "test"}
        context2 = {"task": "test"}
        context3 = {"task": "different"}

        key1 = coordinator._make_cache_key(context1)
        key2 = coordinator._make_cache_key(context2)
        key3 = coordinator._make_cache_key(context3)

        # Same context should produce same key
        assert key1 == key2
        # Different context should produce different key
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_contributor_error_handling(self, coordinator):
        """Test that contributor errors are handled gracefully."""

        class FailingContributor(BasePromptContributor):
            async def contribute(self, context):
                raise ValueError("Intentional error")

        coordinator = PromptCoordinator(
            contributors=[FailingContributor(priority=50)], enable_cache=False
        )

        # Should not raise, just return empty prompt
        prompt = await coordinator.build_system_prompt({})

        assert prompt == ""

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        contributor = MockPromptContributor("Test prompt\n", priority=50)
        coordinator = PromptCoordinator(
            contributors=[contributor], enable_cache=True, cache_ttl=0.1  # 100ms TTL
        )

        context = {"task": "test"}

        # First call - cache miss
        prompt1 = await coordinator.build_system_prompt(context)
        assert prompt1 == "Test prompt\n"

        # Second call immediately - cache hit
        prompt2 = await coordinator.build_system_prompt(context)
        assert prompt2 == "Test prompt\n"
        assert coordinator._cache_hits == 1

        # Wait for TTL to expire
        time.sleep(0.15)

        # Third call after expiration - cache miss
        prompt3 = await coordinator.build_system_prompt(context)
        assert prompt3 == "Test prompt\n"
        assert coordinator._cache_hits == 1  # No additional hit
        assert coordinator._cache_misses >= 2  # At least 2 misses (1 initial + 1 after expiration)

    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        """Test that caching can be disabled."""
        contributor = MockPromptContributor("Test prompt\n", priority=50)
        coordinator = PromptCoordinator(contributors=[contributor], enable_cache=False)

        context = {"task": "test"}

        # Multiple calls with cache disabled
        await coordinator.build_system_prompt(context)
        await coordinator.build_system_prompt(context)
        await coordinator.build_system_prompt(context)

        # Should have no cache hits
        assert coordinator._cache_hits == 0
        assert coordinator._cache_misses == 3

    @pytest.mark.asyncio
    async def test_get_cache_stats(self):
        """Test getting cache statistics."""
        contributor = MockPromptContributor("Test\n", priority=50)
        coordinator = PromptCoordinator(contributors=[contributor])

        context = {"task": "test"}

        # Initial stats
        stats = coordinator.get_cache_stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["cache_invalidations"] == 0
        assert stats["cache_size"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["total_requests"] == 0

        # Build prompt (miss)
        await coordinator.build_system_prompt(context)
        stats = coordinator.get_cache_stats()
        assert stats["cache_misses"] == 1
        assert stats["total_requests"] == 1
        assert stats["hit_rate"] == 0.0

        # Build again (hit)
        await coordinator.build_system_prompt(context)
        stats = coordinator.get_cache_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 0.5
        assert stats["cache_size"] == 1

    @pytest.mark.asyncio
    async def test_reset_cache_stats(self):
        """Test resetting cache statistics."""
        contributor = MockPromptContributor("Test\n", priority=50)
        coordinator = PromptCoordinator(contributors=[contributor])

        context = {"task": "test"}

        # Build some prompts
        await coordinator.build_system_prompt(context)
        await coordinator.build_system_prompt(context)
        coordinator.invalidate_cache()

        # Verify stats are non-zero
        assert coordinator._cache_hits > 0 or coordinator._cache_misses > 0

        # Reset stats
        coordinator.reset_cache_stats()

        # Verify all stats are zero
        assert coordinator._cache_hits == 0
        assert coordinator._cache_misses == 0
        assert coordinator._cache_invalidations == 0

        stats = coordinator.get_cache_stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["cache_invalidations"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["total_requests"] == 0

    def test_make_cache_key_with_unjsonifiable_context(self):
        """Test cache key generation with unjsonifiable context."""
        coordinator = PromptCoordinator(contributors=[])

        # Create a context with an unjsonifiable object
        class UnjsonifiableObject:
            pass

        context1 = {"obj": UnjsonifiableObject()}
        context2 = {"obj": UnjsonifiableObject()}

        # Should not raise, should create keys from str(context)
        key1 = coordinator._make_cache_key(context1)
        key2 = coordinator._make_cache_key(context2)

        # Keys should be valid hex strings (SHA256 hash)
        assert len(key1) == 64  # SHA256 hex length
        assert len(key2) == 64
        assert all(c in "0123456789abcdef" for c in key1)

    @pytest.mark.asyncio
    async def test_build_task_hint_with_error(self):
        """Test build_task_hint handles contributor errors gracefully."""

        class FailingContributor(BasePromptContributor):
            async def contribute(self, context):
                raise RuntimeError("Contributor failed")

        class WorkingContributor(BasePromptContributor):
            async def contribute(self, context):
                return "Working hint"

        coordinator = PromptCoordinator(
            contributors=[FailingContributor(priority=50), WorkingContributor(priority=40)]
        )

        hint = await coordinator.build_task_hint("simple", {})

        # Should return hint from working contributor
        assert hint == "Working hint"

    @pytest.mark.asyncio
    async def test_build_task_hint_all_failing(self):
        """Test build_task_hint when all contributors fail."""

        class FailingContributor(BasePromptContributor):
            async def contribute(self, context):
                raise ValueError("All failing")

        coordinator = PromptCoordinator(contributors=[FailingContributor(priority=50)])

        hint = await coordinator.build_task_hint("simple", {})

        # Should return empty string
        assert hint == ""

    @pytest.mark.asyncio
    async def test_build_system_prompt_multiple_contributors_with_errors(self):
        """Test build_system_prompt with mixed success/failure contributors."""

        class FailingContributor(BasePromptContributor):
            async def contribute(self, context):
                raise ValueError("Failed")

        contributor1 = MockPromptContributor("Section 1\n", priority=100)
        failing = FailingContributor(priority=75)
        contributor2 = MockPromptContributor("Section 2\n", priority=50)

        coordinator = PromptCoordinator(
            contributors=[contributor1, failing, contributor2], enable_cache=False
        )

        prompt = await coordinator.build_system_prompt({})

        # Should include both working contributors
        assert "Section 1" in prompt
        assert "Section 2" in prompt

    @pytest.mark.asyncio
    async def test_invalidate_cache_specific_context(self):
        """Test invalidating cache for specific context."""
        contributor = MockPromptContributor("Test\n", priority=50)
        coordinator = PromptCoordinator(contributors=[contributor])

        context1 = {"task": "test1"}
        context2 = {"task": "test2"}

        # Build prompts for both contexts
        await coordinator.build_system_prompt(context1)
        await coordinator.build_system_prompt(context2)

        assert len(coordinator._prompt_cache) == 2

        # Invalidate only context1
        coordinator.invalidate_cache(context1)

        # Only context1 should be removed
        assert len(coordinator._prompt_cache) == 1
        key1 = coordinator._make_cache_key(context1)
        key2 = coordinator._make_cache_key(context2)
        assert key1 not in coordinator._prompt_cache
        assert key2 in coordinator._prompt_cache

    @pytest.mark.asyncio
    async def test_remove_nonexistent_contributor(self):
        """Test removing a contributor that doesn't exist."""
        contributor = MockPromptContributor("test", priority=50)
        coordinator = PromptCoordinator(contributors=[])

        # Should not raise
        coordinator.remove_contributor(contributor)

        assert len(coordinator._contributors) == 0

    @pytest.mark.asyncio
    async def test_empty_prompt_not_cached(self):
        """Test that empty prompts are not cached."""
        contributor = MockPromptContributor("", priority=50)
        coordinator = PromptCoordinator(contributors=[contributor])

        context = {"task": "test"}

        # Build empty prompt
        prompt = await coordinator.build_system_prompt(context)

        assert prompt == ""
        # Empty prompt should not be cached
        assert len(coordinator._prompt_cache) == 0


class TestPromptBuilderCoordinator:
    """Tests for PromptBuilderCoordinator."""

    @pytest.fixture
    def mock_tool_calling_caps(self):
        """Create mock tool calling capabilities."""
        caps = MagicMock()
        caps.thinking_disable_prefix = "/no_think"
        return caps

    @pytest.fixture
    def coordinator(self, mock_tool_calling_caps):
        """Create PromptBuilderCoordinator with mock caps."""
        return PromptBuilderCoordinator(
            tool_calling_caps=mock_tool_calling_caps, enable_rl_events=False
        )

    @pytest.fixture
    def mock_prompt_builder(self):
        """Create mock prompt builder."""
        builder = MagicMock()
        builder.build.return_value = "You are a helpful assistant."
        return builder

    def test_init(self):
        """Test initialization."""
        coordinator = PromptBuilderCoordinator()

        assert coordinator._tool_calling_caps is None
        assert coordinator._enable_rl_events is True

    def test_init_with_tool_calling_caps(self, mock_tool_calling_caps):
        """Test initialization with tool calling capabilities."""
        coordinator = PromptBuilderCoordinator(
            tool_calling_caps=mock_tool_calling_caps, enable_rl_events=False
        )

        assert coordinator._tool_calling_caps == mock_tool_calling_caps
        assert coordinator._enable_rl_events is False

    def test_build_system_prompt_with_adapter_small_context(self, coordinator, mock_prompt_builder):
        """Test building prompt for small context window (< 32K)."""
        context_window = 16000  # Small context

        prompt = coordinator.build_system_prompt_with_adapter(
            prompt_builder=mock_prompt_builder,
            get_model_context_window=lambda: context_window,
            model="small-model",
            session_id="test-session",
            provider_name="provider",
        )

        # Should not add budget hint for small models
        assert prompt == "You are a helpful assistant."
        mock_prompt_builder.build.assert_called_once()

    def test_build_system_prompt_with_adapter_large_context(self, coordinator, mock_prompt_builder):
        """Test building prompt for large context window (>= 32K)."""
        context_window = 128000  # Large context

        with patch(
            "victor.agent.context_compactor.calculate_parallel_read_budget"
        ) as mock_budget_calc:
            mock_budget = MagicMock()
            mock_budget.to_prompt_hint.return_value = "Max 20 parallel reads"
            mock_budget_calc.return_value = mock_budget

            prompt = coordinator.build_system_prompt_with_adapter(
                prompt_builder=mock_prompt_builder,
                get_model_context_window=lambda: context_window,
                model="claude-sonnet-4-5",
                session_id="test-session",
                provider_name="anthropic",
            )

            # Should add budget hint for large models
            assert "You are a helpful assistant." in prompt
            assert "Max 20 parallel reads" in prompt
            mock_budget_calc.assert_called_once_with(context_window)

    def test_build_system_prompt_with_adapter_emits_rl_event_when_enabled(
        self, mock_prompt_builder, mock_tool_calling_caps
    ):
        """Test that RL event is emitted when enabled."""
        coordinator = PromptBuilderCoordinator(
            tool_calling_caps=mock_tool_calling_caps, enable_rl_events=True
        )

        with patch("victor.framework.rl.hooks.get_rl_hooks") as mock_get_hooks:
            mock_hooks = MagicMock()
            mock_get_hooks.return_value = mock_hooks

            prompt = coordinator.build_system_prompt_with_adapter(
                prompt_builder=mock_prompt_builder,
                get_model_context_window=lambda: 128000,
                model="test-model",
                session_id="test-session",
                provider_name="test-provider",
            )

            # Verify event was emitted
            mock_hooks.emit.assert_called_once()
            call_args = mock_hooks.emit.call_args
            event = call_args[0][0]
            assert event.provider == "test-provider"
            assert event.model == "test-model"
            assert event.metadata["session_id"] == "test-session"

    def test_build_system_prompt_with_adapter_rl_hooks_unavailable(
        self, coordinator, mock_prompt_builder
    ):
        """Test handling when RL hooks are not available."""
        with patch("victor.framework.rl.hooks.get_rl_hooks") as mock_get_hooks:
            mock_get_hooks.return_value = None

            # Should not raise
            prompt = coordinator.build_system_prompt_with_adapter(
                prompt_builder=mock_prompt_builder,
                get_model_context_window=lambda: 128000,
                model="test-model",
                session_id="test-session",
                provider_name="test-provider",
            )

            # Should still return prompt
            assert prompt is not None

    def test_build_system_prompt_with_adapter_rl_event_failure(
        self, coordinator, mock_prompt_builder
    ):
        """Test handling when RL event emission fails."""
        with patch("victor.framework.rl.hooks.get_rl_hooks") as mock_get_hooks:
            mock_hooks = MagicMock()
            mock_hooks.emit.side_effect = RuntimeError("RL system error")
            mock_get_hooks.return_value = mock_hooks

            # Should not raise, should log and continue
            prompt = coordinator.build_system_prompt_with_adapter(
                prompt_builder=mock_prompt_builder,
                get_model_context_window=lambda: 128000,
                model="test-model",
                session_id="test-session",
                provider_name="test-provider",
            )

            # Should still return prompt
            assert prompt is not None

    def test_emit_prompt_used_event_detects_local_provider(self, mock_tool_calling_caps):
        """Test that local providers are detected correctly."""
        coordinator = PromptBuilderCoordinator(
            tool_calling_caps=mock_tool_calling_caps, enable_rl_events=True
        )

        with patch("victor.framework.rl.hooks.get_rl_hooks") as mock_get_hooks:
            mock_hooks = MagicMock()
            mock_get_hooks.return_value = mock_hooks

            prompt = "You are a helpful assistant."

            # Test local provider
            coordinator._emit_prompt_used_event(
                prompt=prompt,
                provider_name="ollama",
                model="llama2",
                session_id="test",
            )

            call_args = mock_hooks.emit.call_args
            event = call_args[0][0]
            assert event.metadata["prompt_style"] == "detailed"

    def test_emit_prompt_used_event_detects_cloud_provider(self, mock_tool_calling_caps):
        """Test that cloud providers are detected correctly."""
        coordinator = PromptBuilderCoordinator(
            tool_calling_caps=mock_tool_calling_caps, enable_rl_events=True
        )

        with patch("victor.framework.rl.hooks.get_rl_hooks") as mock_get_hooks:
            mock_hooks = MagicMock()
            mock_get_hooks.return_value = mock_hooks

            prompt = "You are a helpful assistant."

            # Test cloud provider
            coordinator._emit_prompt_used_event(
                prompt=prompt,
                provider_name="anthropic",
                model="claude-sonnet-4-5",
                session_id="test",
            )

            call_args = mock_hooks.emit.call_args
            event = call_args[0][0]
            assert event.metadata["prompt_style"] == "structured"

    def test_emit_prompt_used_event_analyzes_prompt_content(self, mock_tool_calling_caps):
        """Test that prompt content is analyzed correctly."""
        coordinator = PromptBuilderCoordinator(
            tool_calling_caps=mock_tool_calling_caps, enable_rl_events=True
        )

        with patch("victor.framework.rl.hooks.get_rl_hooks") as mock_get_hooks:
            mock_hooks = MagicMock()
            mock_get_hooks.return_value = mock_hooks

            # Prompt with examples, thinking, and constraints
            prompt = "You MUST follow these guidelines. For example, do X. Think step by step."

            coordinator._emit_prompt_used_event(
                prompt=prompt,
                provider_name="anthropic",
                model="claude-sonnet-4-5",
                session_id="test",
            )

            call_args = mock_hooks.emit.call_args
            event = call_args[0][0]
            metadata = event.metadata
            assert metadata["has_examples"] is True
            assert metadata["has_thinking_prompt"] is True
            assert metadata["has_constraints"] is True
            assert metadata["prompt_length"] == len(prompt)

    def test_emit_prompt_used_event_disabled(self, mock_tool_calling_caps):
        """Test that event is not emitted when RL events are disabled."""
        coordinator = PromptBuilderCoordinator(
            tool_calling_caps=mock_tool_calling_caps, enable_rl_events=False
        )

        with patch("victor.framework.rl.hooks.get_rl_hooks") as mock_get_hooks:
            prompt = "Test prompt"

            coordinator._emit_prompt_used_event(
                prompt=prompt,
                provider_name="anthropic",
                model="claude-sonnet-4-5",
                session_id="test",
            )

            # Should not call get_rl_hooks when disabled
            mock_get_hooks.assert_not_called()

    def test_get_thinking_disabled_prompt_with_prefix(self, coordinator):
        """Test getting thinking disabled prompt when prefix is configured."""
        base_prompt = "Summarize what you've found."

        prompt = coordinator.get_thinking_disabled_prompt(base_prompt)

        assert prompt == "/no_think\nSummarize what you've found."

    def test_get_thinking_disabled_prompt_without_prefix(self):
        """Test getting thinking disabled prompt when no prefix is configured."""
        coordinator = PromptBuilderCoordinator(tool_calling_caps=None, enable_rl_events=False)

        base_prompt = "Summarize what you've found."

        prompt = coordinator.get_thinking_disabled_prompt(base_prompt)

        # Should return base prompt unchanged
        assert prompt == base_prompt

    def test_get_thinking_disabled_prompt_no_attribute(self):
        """Test when tool_calling_caps doesn't have thinking_disable_prefix."""
        # Create a mock that explicitly doesn't have the attribute
        caps = MagicMock(spec=[])  # Empty spec, no attributes
        del caps.thinking_disable_prefix  # Ensure it doesn't exist
        coordinator = PromptBuilderCoordinator(tool_calling_caps=caps, enable_rl_events=False)

        base_prompt = "Summarize what you've found."

        prompt = coordinator.get_thinking_disabled_prompt(base_prompt)

        # Should return base prompt unchanged when no prefix
        assert prompt == base_prompt

    def test_set_tool_calling_caps(self, coordinator):
        """Test updating tool calling capabilities."""
        new_caps = MagicMock()
        new_caps.thinking_disable_prefix = "/new_prefix"

        coordinator.set_tool_calling_caps(new_caps)

        assert coordinator._tool_calling_caps == new_caps

        # Verify new caps are used
        base_prompt = "Test"
        prompt = coordinator.get_thinking_disabled_prompt(base_prompt)
        assert prompt == "/new_prefix\nTest"


class TestIPromptBuilderCoordinator:
    """Tests for IPromptBuilderCoordinator protocol."""

    def test_protocol_methods_exist(self):
        """Test that protocol defines required methods."""
        # The protocol should define these methods
        assert hasattr(IPromptBuilderCoordinator, "build_system_prompt_with_adapter")
        assert hasattr(IPromptBuilderCoordinator, "get_thinking_disabled_prompt")
