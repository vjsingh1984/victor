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

import pytest

from victor.agent.coordinators.prompt_coordinator import (
    PromptCoordinator,
    PromptBuildError,
    BasePromptContributor,
    SystemPromptContributor,
    TaskHintContributor,
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
        coordinator = PromptCoordinator(
            contributors=[contributor], enable_cache=False
        )

        assert len(coordinator._contributors) == 1
        assert coordinator._enable_cache is False

    def test_init_sorts_contributors_by_priority(self):
        """Test that contributors are sorted by priority."""
        contributor1 = MockPromptContributor("", priority=10)
        contributor2 = MockPromptContributor("", priority=100)
        contributor3 = MockPromptContributor("", priority=50)

        coordinator = PromptCoordinator(
            contributors=[contributor1, contributor2, contributor3]
        )

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
        contributor = TaskHintContributor(
            {"simple": "Be brief", "complex": "Be thorough"}
        )
        coordinator = PromptCoordinator(
            contributors=[contributor], enable_cache=False
        )

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
