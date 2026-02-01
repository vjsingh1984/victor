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

"""Tests for prompt contributors.

This test suite uses TDD to verify all prompt contributor implementations
work correctly with the PromptCoordinator.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from victor.protocols import PromptContext
from victor.agent.coordinators.prompt_contributors import (
    VerticalPromptContributor,
    ContextPromptContributor,
    ProjectInstructionsContributor,
    ModeAwareContributor,
    StageAwareContributor,
    DynamicPromptContributor,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_context() -> PromptContext:
    """Sample prompt context for testing."""
    return {
        "vertical_name": "coding",
        "mode": "build",
        "stage": "EXECUTION",
        "task_type": "implementation",
        "metadata": {"language": "python"},
    }


@pytest.fixture
def temp_project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory with .victor/init.md."""
    victor_dir = tmp_path / ".victor"
    victor_dir.mkdir()

    init_md = victor_dir / "init.md"
    init_md.write_text(
        "# Project Instructions\n\n"
        "This is a test project.\n\n"
        "## Important Notes\n\n"
        "- Follow PEP 8 style\n"
        "- Write tests for all functions\n"
    )

    return tmp_path


# =============================================================================
# VerticalPromptContributor Tests
# =============================================================================


class TestVerticalPromptContributor:
    """Tests for VerticalPromptContributor."""

    @pytest.mark.asyncio
    async def test_contribute_returns_prompt_for_matching_vertical(
        self, sample_context: PromptContext
    ):
        """Test that contributor returns prompt when vertical matches."""
        contributor = VerticalPromptContributor(
            vertical_name="coding",
            system_prompt="You are an expert software developer.",
        )

        result = await contributor.contribute(sample_context)

        assert result == "You are an expert software developer."

    @pytest.mark.asyncio
    async def test_contribute_returns_empty_for_non_matching_vertical(
        self, sample_context: PromptContext
    ):
        """Test that contributor returns empty string when vertical doesn't match."""
        contributor = VerticalPromptContributor(
            vertical_name="research",
            system_prompt="You are an expert researcher.",
        )

        result = await contributor.contribute(sample_context)

        assert result == ""

    @pytest.mark.asyncio
    async def test_contribute_handles_empty_context(self):
        """Test that contributor handles empty context gracefully."""
        contributor = VerticalPromptContributor(
            vertical_name="coding",
            system_prompt="Test prompt.",
        )

        result = await contributor.contribute({})

        assert result == ""

    @pytest.mark.asyncio
    async def test_contribute_handles_context_without_vertical(self):
        """Test that contributor handles context without vertical_name key."""
        contributor = VerticalPromptContributor(
            vertical_name="coding",
            system_prompt="Test prompt.",
        )

        result = await contributor.contribute({"mode": "build"})

        assert result == ""

    def test_priority_returns_correct_value(self):
        """Test that priority() returns the configured priority."""
        contributor = VerticalPromptContributor(
            vertical_name="coding",
            system_prompt="Test.",
            priority=150,
        )

        assert contributor.priority() == 150

    def test_default_priority_is_100(self):
        """Test that default priority is 100."""
        contributor = VerticalPromptContributor(
            vertical_name="coding",
            system_prompt="Test.",
        )

        assert contributor.priority() == 100


# =============================================================================
# ContextPromptContributor Tests
# =============================================================================


class TestContextPromptContributor:
    """Tests for ContextPromptContributor."""

    @pytest.mark.asyncio
    async def test_contribute_with_callable_handler(self, sample_context: PromptContext):
        """Test that callable handlers are invoked correctly."""

        def mode_handler(mode: str, context: PromptContext) -> str:
            return f"Current mode: {mode}"

        contributor = ContextPromptContributor(context_handlers={"mode": mode_handler})

        result = await contributor.contribute(sample_context)

        assert result == "Current mode: build"

    @pytest.mark.asyncio
    async def test_contribute_with_static_string_handler(self):
        """Test that static string handlers work correctly."""
        contributor = ContextPromptContributor(
            context_handlers={"task_type": "Focus on implementation."}
        )

        result = await contributor.contribute({"task_type": "implementation"})

        assert result == "Focus on implementation."

    @pytest.mark.asyncio
    async def test_contribute_aggregates_multiple_handlers(self):
        """Test that multiple handlers are aggregated correctly."""
        handlers = {
            "mode": lambda m, c: f"Mode: {m}",
            "stage": "Stage: EXECUTION",
        }
        contributor = ContextPromptContributor(context_handlers=handlers)

        result = await contributor.contribute({"mode": "build", "stage": "EXECUTION"})

        assert "Mode: build" in result
        assert "Stage: EXECUTION" in result

    @pytest.mark.asyncio
    async def test_contribute_skips_missing_context_keys(self):
        """Test that missing context keys are handled gracefully."""
        contributor = ContextPromptContributor(
            context_handlers={
                "mode": lambda m, c: f"Mode: {m}",
                "missing": lambda m, c: "Should not appear",
            }
        )

        result = await contributor.contribute({"mode": "build"})

        assert "Mode: build" in result
        assert "Should not appear" not in result

    @pytest.mark.asyncio
    async def test_contribute_handles_handler_exceptions(self):
        """Test that exceptions in handlers are caught and logged."""

        def failing_handler(value: str, context: PromptContext) -> str:
            raise ValueError("Intentional error")

        contributor = ContextPromptContributor(context_handlers={"mode": failing_handler})

        # Should not raise exception
        result = await contributor.contribute({"mode": "build"})

        assert result == ""

    @pytest.mark.asyncio
    async def test_contribute_returns_empty_for_empty_context(self):
        """Test that empty context returns empty string."""
        contributor = ContextPromptContributor(context_handlers={"mode": lambda m, c: f"Mode: {m}"})

        result = await contributor.contribute({})

        assert result == ""

    def test_add_handler_adds_new_handler(self):
        """Test that add_handler() adds a new handler."""
        contributor = ContextPromptContributor()

        def new_handler(value: str, context: PromptContext) -> str:
            return f"Value: {value}"

        contributor.add_handler("test_key", new_handler)

        # Handler should be in internal dict
        assert "test_key" in contributor._context_handlers

    def test_remove_handler_removes_existing_handler(self):
        """Test that remove_handler() removes a handler."""
        contributor = ContextPromptContributor(context_handlers={"test": lambda v, c: "test"})

        contributor.remove_handler("test")

        assert "test" not in contributor._context_handlers

    def test_remove_handler_handles_missing_key(self):
        """Test that remove_handler() handles missing keys gracefully."""
        contributor = ContextPromptContributor()

        # Should not raise exception
        contributor.remove_handler("nonexistent")

    def test_priority_returns_correct_value(self):
        """Test that priority() returns configured value."""
        contributor = ContextPromptContributor(priority=85)

        assert contributor.priority() == 85


# =============================================================================
# ProjectInstructionsContributor Tests
# =============================================================================


class TestProjectInstructionsContributor:
    """Tests for ProjectInstructionsContributor."""

    @pytest.mark.asyncio
    async def test_contribute_loads_project_context(self, temp_project_dir: Path):
        """Test that project context is loaded and returned."""
        contributor = ProjectInstructionsContributor(
            root_path=str(temp_project_dir),
            enabled=True,
        )

        result = await contributor.contribute({})

        assert "<project-context>" in result
        assert "Project Instructions" in result
        assert "Follow PEP 8 style" in result

    @pytest.mark.asyncio
    async def test_contribute_returns_empty_when_disabled(self, temp_project_dir: Path):
        """Test that disabled contributor returns empty string."""
        contributor = ProjectInstructionsContributor(
            root_path=str(temp_project_dir),
            enabled=False,
        )

        result = await contributor.contribute({})

        assert result == ""

    @pytest.mark.asyncio
    async def test_contribute_returns_empty_when_no_project_file(self, tmp_path: Path):
        """Test that empty string is returned when no project file exists."""
        contributor = ProjectInstructionsContributor(
            root_path=str(tmp_path),
            enabled=True,
        )

        result = await contributor.contribute({})

        assert result == ""

    @pytest.mark.asyncio
    async def test_contribute_caches_content(self, temp_project_dir: Path):
        """Test that project content is cached after first load."""
        contributor = ProjectInstructionsContributor(
            root_path=str(temp_project_dir),
            enabled=True,
        )

        # First call
        result1 = await contributor.contribute({})
        # Second call should use cache
        result2 = await contributor.contribute({})

        assert result1 == result2
        # Should only load once (cache is set)
        assert contributor._content_cache is not None

    @pytest.mark.asyncio
    async def test_invalidate_cache_clears_cached_content(self, temp_project_dir: Path):
        """Test that invalidate_cache() clears cached content."""
        contributor = ProjectInstructionsContributor(
            root_path=str(temp_project_dir),
            enabled=True,
        )

        # Load content
        await contributor.contribute({})
        assert contributor._content_cache is not None

        # Invalidate cache
        contributor.invalidate_cache()
        assert contributor._content_cache is None

    def test_priority_returns_correct_value(self):
        """Test that priority() returns configured value."""
        contributor = ProjectInstructionsContributor(priority=55)

        assert contributor.priority() == 55

    @pytest.mark.asyncio
    async def test_contribute_handles_loading_errors(self, tmp_path: Path):
        """Test that loading errors are handled gracefully."""
        # Create directory but make it inaccessible (mock the error)
        contributor = ProjectInstructionsContributor(
            root_path=str(tmp_path),
            enabled=True,
        )

        with patch("victor.context.project_context.ProjectContext") as mock_pc:
            mock_pc_instance = MagicMock()
            mock_pc_instance.load.side_effect = Exception("Load error")
            mock_pc.return_value = mock_pc_instance

            result = await contributor.contribute({})

            # Should return empty string on error
            assert result == ""


# =============================================================================
# ModeAwareContributor Tests
# =============================================================================


class TestModeAwareContributor:
    """Tests for ModeAwareContributor."""

    @pytest.mark.asyncio
    async def test_contribute_returns_prompt_for_matching_mode(self):
        """Test that contributor returns prompt for matching mode."""
        mode_prompts = {
            "build": "Focus on implementation.",
            "plan": "Focus on planning.",
        }
        contributor = ModeAwareContributor(mode_prompts=mode_prompts)

        result = await contributor.contribute({"mode": "build"})

        assert result == "Focus on implementation."

    @pytest.mark.asyncio
    async def test_contribute_returns_empty_for_non_matching_mode(self):
        """Test that contributor returns empty string for non-matching mode."""
        mode_prompts = {"build": "Focus on implementation."}
        contributor = ModeAwareContributor(mode_prompts=mode_prompts)

        result = await contributor.contribute({"mode": "explore"})

        assert result == ""

    @pytest.mark.asyncio
    async def test_contribute_handles_empty_context(self):
        """Test that empty context is handled gracefully."""
        contributor = ModeAwareContributor(mode_prompts={"build": "Test."})

        result = await contributor.contribute({})

        assert result == ""

    @pytest.mark.asyncio
    async def test_contribute_handles_context_without_mode(self):
        """Test that context without mode key is handled gracefully."""
        contributor = ModeAwareContributor(mode_prompts={"build": "Test."})

        result = await contributor.contribute({"stage": "EXECUTION"})

        assert result == ""

    def test_set_mode_prompt_adds_new_mode(self):
        """Test that set_mode_prompt() adds a new mode."""
        contributor = ModeAwareContributor(mode_prompts={})

        contributor.set_mode_prompt("new_mode", "New mode prompt.")

        assert "new_mode" in contributor._mode_prompts
        assert contributor._mode_prompts["new_mode"] == "New mode prompt."

    def test_set_mode_prompt_updates_existing_mode(self):
        """Test that set_mode_prompt() updates existing mode."""
        contributor = ModeAwareContributor(mode_prompts={"build": "Old prompt."})

        contributor.set_mode_prompt("build", "New prompt.")

        assert contributor._mode_prompts["build"] == "New prompt."

    def test_get_supported_modes_returns_all_modes(self):
        """Test that get_supported_modes() returns all configured modes."""
        mode_prompts = {"build": "Build.", "plan": "Plan.", "explore": "Explore."}
        contributor = ModeAwareContributor(mode_prompts=mode_prompts)

        modes = contributor.get_supported_modes()

        assert set(modes) == {"build", "plan", "explore"}

    def test_priority_returns_correct_value(self):
        """Test that priority() returns configured value."""
        contributor = ModeAwareContributor(mode_prompts={}, priority=90)

        assert contributor.priority() == 90


# =============================================================================
# StageAwareContributor Tests
# =============================================================================


class TestStageAwareContributor:
    """Tests for StageAwareContributor."""

    @pytest.mark.asyncio
    async def test_contribute_returns_prompt_for_matching_stage(self):
        """Test that contributor returns prompt for matching stage."""
        stage_prompts = {
            "EXECUTION": "Focus on implementing carefully.",
            "VERIFICATION": "Focus on testing.",
        }
        contributor = StageAwareContributor(stage_prompts=stage_prompts)

        result = await contributor.contribute({"stage": "EXECUTION"})

        assert result == "Focus on implementing carefully."

    @pytest.mark.asyncio
    async def test_contribute_returns_empty_for_non_matching_stage(self):
        """Test that contributor returns empty string for non-matching stage."""
        stage_prompts = {"EXECUTION": "Test."}
        contributor = StageAwareContributor(stage_prompts=stage_prompts)

        result = await contributor.contribute({"stage": "PLANNING"})

        assert result == ""

    @pytest.mark.asyncio
    async def test_contribute_handles_empty_context(self):
        """Test that empty context is handled gracefully."""
        contributor = StageAwareContributor(stage_prompts={"EXECUTION": "Test."})

        result = await contributor.contribute({})

        assert result == ""

    def test_set_stage_prompt_adds_new_stage(self):
        """Test that set_stage_prompt() adds a new stage."""
        contributor = StageAwareContributor(stage_prompts={})

        contributor.set_stage_prompt("NEW_STAGE", "New stage prompt.")

        assert "NEW_STAGE" in contributor._stage_prompts
        assert contributor._stage_prompts["NEW_STAGE"] == "New stage prompt."

    def test_get_supported_stages_returns_all_stages(self):
        """Test that get_supported_stages() returns all configured stages."""
        stage_prompts = {
            "INITIAL": "Init.",
            "EXECUTION": "Exec.",
            "COMPLETION": "Complete.",
        }
        contributor = StageAwareContributor(stage_prompts=stage_prompts)

        stages = contributor.get_supported_stages()

        assert set(stages) == {"INITIAL", "EXECUTION", "COMPLETION"}

    def test_priority_returns_correct_value(self):
        """Test that priority() returns configured value."""
        contributor = StageAwareContributor(stage_prompts={}, priority=70)

        assert contributor.priority() == 70


# =============================================================================
# DynamicPromptContributor Tests
# =============================================================================


class TestDynamicPromptContributor:
    """Tests for DynamicPromptContributor."""

    @pytest.mark.asyncio
    async def test_contribute_with_async_function(self):
        """Test that async contributor function is called correctly."""

        async def async_contributor(context: PromptContext) -> str:
            return f"Task: {context.get('task_type', 'unknown')}"

        contributor = DynamicPromptContributor(async_contributor)

        result = await contributor.contribute({"task_type": "debugging"})

        assert result == "Task: debugging"

    @pytest.mark.asyncio
    async def test_contribute_with_sync_function(self):
        """Test that sync contributor function is called correctly."""

        def sync_contributor(context: PromptContext) -> str:
            return f"Mode: {context.get('mode', 'unknown')}"

        contributor = DynamicPromptContributor(sync_contributor)

        result = await contributor.contribute({"mode": "plan"})

        assert result == "Mode: plan"

    @pytest.mark.asyncio
    async def test_contribute_returns_empty_for_non_callable(self):
        """Test that non-callable contributor returns empty string."""
        contributor = DynamicPromptContributor("not a function")

        result = await contributor.contribute({})

        assert result == ""

    @pytest.mark.asyncio
    async def test_contribute_handles_function_returning_none(self):
        """Test that function returning None is handled gracefully."""

        def none_contributor(context: PromptContext) -> None:
            return None

        contributor = DynamicPromptContributor(none_contributor)

        result = await contributor.contribute({})

        assert result == ""

    @pytest.mark.asyncio
    async def test_contribute_propagates_exceptions(self):
        """Test that exceptions in contributor function are propagated."""

        async def failing_contributor(context: PromptContext) -> str:
            raise ValueError("Intentional error")

        contributor = DynamicPromptContributor(failing_contributor)

        with pytest.raises(ValueError, match="Intentional error"):
            await contributor.contribute({})

    @pytest.mark.asyncio
    async def test_contribute_passes_context_to_function(self):
        """Test that context is passed to contributor function."""
        received_context = None

        async def context_checker(context: PromptContext) -> str:
            nonlocal received_context
            received_context = context
            return "checked"

        contributor = DynamicPromptContributor(context_checker)
        test_context = {"key": "value"}

        await contributor.contribute(test_context)

        assert received_context == test_context

    def test_priority_returns_correct_value(self):
        """Test that priority() returns configured value."""

        async def dummy_func(context: PromptContext) -> str:
            return "test"

        contributor = DynamicPromptContributor(dummy_func, priority=95)

        assert contributor.priority() == 95


# =============================================================================
# Integration Tests
# =============================================================================


class TestPromptContributorIntegration:
    """Integration tests for prompt contributors."""

    @pytest.mark.asyncio
    async def test_multiple_contributors_with_priorities(self):
        """Test that multiple contributors work together correctly."""
        contributors = [
            VerticalPromptContributor(
                vertical_name="coding",
                system_prompt="You are a developer.",
                priority=100,
            ),
            ModeAwareContributor(
                mode_prompts={"build": "Focus on building."},
                priority=80,
            ),
            StageAwareContributor(
                stage_prompts={"EXECUTION": "Execute carefully."},
                priority=70,
            ),
        ]

        # Sort by priority (highest first)
        sorted_contributors = sorted(contributors, key=lambda c: c.priority(), reverse=True)

        context = {
            "vertical_name": "coding",
            "mode": "build",
            "stage": "EXECUTION",
        }

        contributions = []
        for contributor in sorted_contributors:
            contribution = await contributor.contribute(context)
            if contribution:
                contributions.append(contribution)

        # All three should contribute
        assert len(contributions) == 3
        assert "You are a developer." in contributions
        assert "Focus on building." in contributions
        assert "Execute carefully." in contributions

    @pytest.mark.asyncio
    async def test_contributors_with_empty_contributions(self):
        """Test that contributors that return empty strings are handled."""
        contributors = [
            VerticalPromptContributor(
                vertical_name="research",  # Wrong vertical
                system_prompt="Research prompt.",
                priority=100,
            ),
            ModeAwareContributor(
                mode_prompts={"plan": "Plan prompt."},  # Wrong mode
                priority=80,
            ),
            ContextPromptContributor(
                context_handlers={},  # No handlers
                priority=60,
            ),
        ]

        context = {
            "vertical_name": "coding",
            "mode": "build",
        }

        contributions = []
        for contributor in contributors:
            contribution = await contributor.contribute(context)
            if contribution:
                contributions.append(contribution)

        # None should contribute
        assert len(contributions) == 0

    @pytest.mark.asyncio
    async def test_dynamic_and_static_contributors(self):
        """Test that dynamic and static contributors work together."""

        async def dynamic_func(context: PromptContext) -> str:
            return f"Dynamic: {context.get('value', '')}"

        contributors = [
            DynamicPromptContributor(dynamic_func, priority=90),
            VerticalPromptContributor(
                vertical_name="coding",
                system_prompt="Static prompt.",
                priority=80,
            ),
        ]

        context = {"vertical_name": "coding", "value": "test123"}

        contributions = []
        for contributor in sorted(contributors, key=lambda c: c.priority(), reverse=True):
            contribution = await contributor.contribute(context)
            if contribution:
                contributions.append(contribution)

        assert len(contributions) == 2
        assert "Dynamic: test123" in contributions
        assert "Static prompt." in contributions
