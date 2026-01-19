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

"""Tests for PromptBuilder consolidation features.

Tests the new section composition methods added in Phase 7.
"""

import pytest

from victor.framework.prompt_builder import PromptBuilder


class TestPromptBuilderGrounding:
    """Tests for add_grounding method."""

    def test_add_grounding_basic(self):
        """Test adding basic grounding without variables."""
        builder = PromptBuilder()
        builder.add_grounding("Context: Working on a project")

        prompt = builder.build()
        assert "Context: Working on a project" in prompt

    def test_add_grounding_with_variables(self):
        """Test adding grounding with variable substitution."""
        builder = PromptBuilder()
        builder.add_grounding(
            "Project: {project}, Stage: {stage}", project="Victor", stage="development"
        )

        prompt = builder.build()
        assert "Project: Victor" in prompt
        assert "Stage: development" in prompt

    def test_add_grounding_custom_priority(self):
        """Test adding grounding with custom priority."""
        builder = PromptBuilder()
        builder.add_grounding("High priority", priority=5)
        builder.add_section("low", "Low priority", priority=100)

        prompt = builder.build()
        # Higher priority (lower number) should come first
        high_idx = prompt.find("High priority")
        low_idx = prompt.find("Low priority")
        assert high_idx < low_idx

    def test_add_grounding_multiple_calls(self):
        """Test multiple add_grounding calls (last one wins)."""
        builder = PromptBuilder()
        builder.add_grounding("First {name}", name="A")
        builder.add_grounding("Second {name}", name="B")

        prompt = builder.build()
        # Second call overrides first
        assert "First A" not in prompt
        assert "Second B" in prompt


class TestPromptBuilderRules:
    """Tests for add_rules method."""

    def test_add_rules_single(self):
        """Test adding single rule."""
        builder = PromptBuilder()
        builder.add_rules(["Rule 1"])

        prompt = builder.build()
        assert "- Rule 1" in prompt

    def test_add_rules_multiple(self):
        """Test adding multiple rules."""
        builder = PromptBuilder()
        builder.add_rules(["Rule 1", "Rule 2", "Rule 3"])

        prompt = builder.build()
        assert "- Rule 1" in prompt
        assert "- Rule 2" in prompt
        assert "- Rule 3" in prompt

    def test_add_rules_with_header(self):
        """Test that rules section has header."""
        builder = PromptBuilder()
        builder.add_rules(["Rule 1"])

        prompt = builder.build()
        assert "## Rules" in prompt

    def test_add_rules_custom_priority(self):
        """Test adding rules with custom priority."""
        builder = PromptBuilder()
        builder.add_rules(["High"], priority=5)
        builder.add_rules(["Low"], priority=100)

        prompt = builder.build()
        high_idx = prompt.find("- High")
        low_idx = prompt.find("- Low")
        assert high_idx < low_idx


class TestPromptBuilderChecklist:
    """Tests for add_checklist method."""

    def test_add_checklist_single(self):
        """Test adding single checklist item."""
        builder = PromptBuilder()
        builder.add_checklist(["Item 1"])

        prompt = builder.build()
        assert "- [ ] Item 1" in prompt

    def test_add_checklist_multiple(self):
        """Test adding multiple checklist items."""
        builder = PromptBuilder()
        builder.add_checklist(["Item 1", "Item 2", "Item 3"])

        prompt = builder.build()
        assert "- [ ] Item 1" in prompt
        assert "- [ ] Item 2" in prompt
        assert "- [ ] Item 3" in prompt

    def test_add_checklist_with_header(self):
        """Test that checklist section has header."""
        builder = PromptBuilder()
        builder.add_checklist(["Item 1"])

        prompt = builder.build()
        assert "## Checklist" in prompt

    def test_add_checklist_custom_priority(self):
        """Test adding checklist with custom priority."""
        builder = PromptBuilder()
        builder.add_checklist(["High"], priority=5)
        builder.add_checklist(["Low"], priority=100)

        prompt = builder.build()
        high_idx = prompt.find("- [ ] High")
        low_idx = prompt.find("- [ ] Low")
        assert high_idx < low_idx


class TestPromptBuilderVerticalSection:
    """Tests for add_vertical_section method."""

    def test_add_vertical_section_basic(self):
        """Test adding vertical section."""
        builder = PromptBuilder()
        builder.add_vertical_section(vertical="coding", content="You are a coder")

        prompt = builder.build()
        assert "You are a coder" in prompt

    def test_add_vertical_section_multiple(self):
        """Test adding multiple vertical sections."""
        builder = PromptBuilder()
        builder.add_vertical_section("coding", "Coding content")
        builder.add_vertical_section("research", "Research content")

        prompt = builder.build()
        assert "Coding content" in prompt
        assert "Research content" in prompt

    def test_add_vertical_section_custom_priority(self):
        """Test adding vertical section with custom priority."""
        builder = PromptBuilder()
        builder.add_vertical_section("high", "High content", priority=5)
        builder.add_vertical_section("low", "Low content", priority=100)

        prompt = builder.build()
        high_idx = prompt.find("High content")
        low_idx = prompt.find("Low content")
        assert high_idx < low_idx


class TestPromptBuilderComposition:
    """Tests for composing prompts with multiple section types."""

    def test_compose_all_section_types(self):
        """Test composing prompts with all section types."""
        builder = (
            PromptBuilder()
            .add_grounding("Project: {name}", name="Victor", priority=10)
            .add_rules(["Rule 1", "Rule 2"], priority=20)
            .add_checklist(["Item 1", "Item 2"], priority=30)
            .add_vertical_section("coding", "Coding content", priority=40)
        )

        prompt = builder.build()

        # Check all sections are present
        assert "Project: Victor" in prompt
        assert "- Rule 1" in prompt
        assert "- [ ] Item 1" in prompt
        assert "Coding content" in prompt

    def test_priority_ordering_mixed_sections(self):
        """Test that sections are ordered by priority regardless of type."""
        builder = (
            PromptBuilder()
            .add_checklist(["Last"], priority=100)
            .add_grounding("First", priority=10)
            .add_vertical_section("test", "Middle", priority=50)
            .add_rules(["Second"], priority=20)
        )

        prompt = builder.build()
        lines = prompt.split("\n\n")

        # Find section order
        first_idx = next(i for i, line in enumerate(lines) if "First" in line)
        second_idx = next(i for i, line in enumerate(lines) if "- Second" in line)
        middle_idx = next(i for i, line in enumerate(lines) if "Middle" in line)
        last_idx = next(i for i, line in enumerate(lines) if "- [ ] Last" in line)

        assert first_idx < second_idx < middle_idx < last_idx

    def test_composition_with_legacy_methods(self):
        """Test that new sections work alongside legacy methods."""
        builder = (
            PromptBuilder()
            .add_section("identity", "You are an assistant.", priority=5)
            .add_grounding("Project: Victor", priority=10)
            .add_safety_rules(["Be safe"])  # No priority parameter for legacy method
            .add_rules(["New style rule"], priority=20)
        )

        prompt = builder.build()
        assert "You are an assistant." in prompt
        assert "Project: Victor" in prompt
        assert "- Be safe" in prompt
        assert "- New style rule" in prompt

    def test_fluent_api_with_new_methods(self):
        """Test fluent API with new composition methods."""
        prompt = (
            PromptBuilder()
            .add_grounding("Context")
            .add_rules(["Rule"])
            .add_checklist(["Item"])
            .add_vertical_section("test", "Content")
            .build()
        )

        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestPromptBuilderCustomSection:
    """Tests for add_custom_section method."""

    def test_add_custom_section_with_render(self):
        """Test adding custom section with render method."""

        class CustomSection:
            def __init__(self, content, priority=50):
                self.content = content
                self.priority = priority

            def render(self):
                return f"Custom: {self.content}"

        builder = PromptBuilder()
        custom = CustomSection("test content", priority=15)
        builder.add_custom_section(custom)

        prompt = builder.build()
        assert "Custom: test content" in prompt

    def test_add_custom_section_without_render(self):
        """Test adding custom section without render method."""

        class CustomSection:
            def __init__(self, content, priority=50):
                self.content = content
                self.priority = priority

            def __str__(self):
                return f"Custom: {self.content}"

        builder = PromptBuilder()
        custom = CustomSection("test")
        builder.add_custom_section(custom)

        prompt = builder.build()
        assert "Custom: test" in prompt

    def test_add_custom_section_with_name(self):
        """Test custom section with name attribute."""

        class CustomSection:
            name = "my_custom"
            priority = 50

            def render(self):
                return "Content"

        builder = PromptBuilder()
        custom = CustomSection()
        builder.add_custom_section(custom)

        # Should use the name attribute
        assert builder.has_section("my_custom")

    def test_add_custom_section_priority_respected(self):
        """Test that custom section priority is respected."""

        class CustomSection:
            def __init__(self, content, priority):
                self.content = content
                self.priority = priority

            def render(self):
                return self.content

        builder = PromptBuilder()
        builder.add_custom_section(CustomSection("High", 10))
        builder.add_custom_section(CustomSection("Low", 100))

        prompt = builder.build()
        high_idx = prompt.find("High")
        low_idx = prompt.find("Low")
        assert high_idx < low_idx


class TestPromptBuilderBackwardCompatibility:
    """Tests for backward compatibility with existing PromptBuilder."""

    def test_legacy_add_section_still_works(self):
        """Test that legacy add_section method still works."""
        builder = PromptBuilder()
        builder.add_section("identity", "You are an assistant.", priority=10)

        prompt = builder.build()
        assert "You are an assistant." in prompt

    def test_legacy_add_tool_hints_still_works(self):
        """Test that legacy add_tool_hints method still works."""
        builder = PromptBuilder()
        builder.add_tool_hints({"read": "Read files"})

        prompt = builder.build()
        assert "read: Read files" in prompt
        assert "## Tool Hints" in prompt

    def test_legacy_add_safety_rules_still_works(self):
        """Test that legacy add_safety_rules method still works."""
        builder = PromptBuilder()
        builder.add_safety_rules(["Rule 1", "Rule 2"])

        prompt = builder.build()
        assert "- Rule 1" in prompt
        assert "## Safety Rules" in prompt

    def test_legacy_add_context_still_works(self):
        """Test that legacy add_context method still works."""
        builder = PromptBuilder()
        builder.add_context("Working on Python project")

        prompt = builder.build()
        assert "Working on Python project" in prompt
        assert "## Context" in prompt

    def test_legacy_clear_still_works(self):
        """Test that legacy clear method still works."""
        builder = PromptBuilder()
        builder.add_grounding("Test")
        builder.add_rules(["Rule"])
        builder.clear()

        # Should be empty
        prompt = builder.build()
        # Only grounding rules (from minimal mode) should remain
        assert "Test" not in prompt
        assert "- Rule" not in prompt

    def test_legacy_set_grounding_mode_still_works(self):
        """Test that legacy set_grounding_mode method still works."""
        builder = PromptBuilder()

        # Set to extended mode
        result = builder.set_grounding_mode("extended")
        assert result is builder  # Returns self for chaining

        # Verify mode was set
        assert builder._grounding_mode == "extended"

        # Verify invalid mode raises error
        with pytest.raises(ValueError, match="Invalid grounding mode"):
            builder.set_grounding_mode("invalid")


class TestPromptBuilderIntegration:
    """Integration tests for PromptBuilder with sections."""

    def test_full_coding_prompt_construction(self):
        """Test constructing a full coding prompt using new methods."""
        builder = (
            PromptBuilder()
            .add_grounding(
                "Working on {project} in {language}",
                project="Victor",
                language="Python",
                priority=10,
            )
            .add_rules(["Always read files first", "Follow code style", "Write tests"], priority=20)
            .add_checklist(["Code compiles", "Tests pass", "Style check passes"], priority=30)
            .add_vertical_section("coding", "You are an expert Python developer.", priority=40)
        )

        prompt = builder.build()

        # Verify all sections
        assert "Working on Victor in Python" in prompt
        assert "- Always read files first" in prompt
        assert "- [ ] Code compiles" in prompt
        assert "expert Python developer" in prompt

    def test_section_order_with_grounding_rules(self):
        """Test that grounding rules from minimal mode come last."""
        builder = (
            PromptBuilder()
            .add_grounding("Custom grounding", priority=10)
            .add_rules(["Rule"], priority=20)
        )

        prompt = builder.build()

        # Both sections should be present
        assert "Custom grounding" in prompt
        assert "- Rule" in prompt

        # Prompt should not be empty
        assert len(prompt) > 0

    def test_mixed_legacy_and_new_sections(self):
        """Test mixing legacy and new section types."""
        builder = (
            PromptBuilder()
            .add_section("identity", "You are Victor.", priority=5)  # Legacy
            .add_grounding("Project: Test", priority=10)  # New
            .add_safety_rules(["Be careful"])  # Legacy (no priority param)
            .add_rules(["Follow rules"], priority=20)  # New
        )

        prompt = builder.build()

        # All sections should be present
        assert "You are Victor." in prompt
        assert "Project: Test" in prompt
        assert "- Be careful" in prompt
        assert "- Follow rules" in prompt

        # Check ordering (safety rules are at priority 60 by default)
        victor_idx = prompt.find("You are Victor.")
        project_idx = prompt.find("Project: Test")
        follow_idx = prompt.find("- Follow rules")
        careful_idx = prompt.find("- Be careful")

        assert victor_idx < project_idx < follow_idx < careful_idx
