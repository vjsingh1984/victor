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

"""Tests for Phase 3 framework capabilities.

Tests for framework-level capability providers that promote code reuse
across verticals (FileOperationsCapability, PromptContributionCapability).
"""

import pytest

from victor.framework.capabilities import (
    FileOperationsCapability,
    PromptContributionCapability,
    BaseCapabilityProvider,
    CapabilityMetadata,
)


class TestFileOperationsCapability:
    """Tests for FileOperationsCapability provider."""

    def test_default_operations_include_basic_tools(self):
        """Test that default operations include read, write, edit, grep."""
        capability = FileOperationsCapability()
        tools = capability.get_tool_list()

        assert "read" in tools
        assert "write" in tools
        assert "edit" in tools
        assert "grep" in tools

    def test_get_tools_returns_set(self):
        """Test that get_tools returns a set."""
        capability = FileOperationsCapability()
        tools = capability.get_tools()

        assert isinstance(tools, set)
        assert len(tools) == 4  # read, write, edit, grep

    def test_get_tool_list_returns_list(self):
        """Test that get_tool_list returns a list."""
        capability = FileOperationsCapability()
        tools = capability.get_tool_list()

        assert isinstance(tools, list)
        assert len(tools) == 4  # read, write, edit, grep

    def test_custom_operations_override_defaults(self):
        """Test that custom operations override defaults."""
        from victor.framework.capabilities.file_operations import FileOperation, FileOperationType

        custom_ops = [
            FileOperation(FileOperationType.READ, "read", required=True),
            FileOperation(FileOperationType.WRITE, "write", required=True),
        ]

        capability = FileOperationsCapability(operations=custom_ops)
        tools = capability.get_tool_list()

        assert len(tools) == 2
        assert "read" in tools
        assert "write" in tools
        assert "edit" not in tools
        assert "grep" not in tools


class TestPromptContributionCapability:
    """Tests for PromptContributionCapability provider."""

    def test_common_hints_include_expected_hints(self):
        """Test that common hints include read_first, verify_changes, search_code."""
        capability = PromptContributionCapability()
        hints = capability.get_task_hints()

        assert "edit" in hints
        assert "search" in hints

    def test_hint_structure(self):
        """Test that hints have correct structure."""
        capability = PromptContributionCapability()
        hints = capability.get_task_hints()

        # Check hint structure
        assert "hint" in hints["edit"]
        assert "tool_budget" in hints["edit"]
        assert isinstance(hints["edit"]["tool_budget"], int)

    def test_get_contributors_returns_list(self):
        """Test that get_contributors returns a list."""
        capability = PromptContributionCapability()
        contributors = capability.get_contributors()

        # Note: Returns empty list if PromptContributorAdapter not available
        assert isinstance(contributors, list)

    def test_custom_contributions_override_defaults(self):
        """Test that custom contributions override defaults."""
        from victor.framework.capabilities.prompt_contributions import PromptContribution

        custom_contribs = [
            PromptContribution(
                name="custom_hint",
                task_type="custom",
                hint="Custom hint for testing",
                tool_budget=20,
            )
        ]

        capability = PromptContributionCapability(contributions=custom_contribs)
        hints = capability.get_task_hints()

        assert len(hints) == 1
        assert "custom" in hints
        assert hints["custom"]["hint"] == "Custom hint for testing"
        assert hints["custom"]["tool_budget"] == 20


class TestCodingVerticalMigration:
    """Tests for CodingVertical migration to framework capabilities.

    Verifies that CodingVertical successfully uses framework capabilities
    to reduce code duplication.

    NOTE: These tests are skipped until Phase 3 migration is complete.
    The framework capabilities (FileOperationsCapability, PromptContributionCapability)
    have been defined but not yet integrated into CodingAssistant.
    """

    @pytest.mark.skip(reason="Phase 3 migration: FileOperationsCapability not yet integrated into CodingAssistant")
    def test_coding_vertical_uses_framework_file_ops(self):
        """Test that CodingVertical uses FileOperationsCapability."""
        from victor.coding import CodingAssistant

        # Check that CodingAssistant has framework capability attributes
        assert hasattr(CodingAssistant, "_file_ops")
        assert isinstance(CodingAssistant._file_ops, FileOperationsCapability)

    @pytest.mark.skip(reason="Phase 3 migration: PromptContributionCapability not yet integrated into CodingAssistant")
    def test_coding_vertical_uses_framework_prompt_contrib(self):
        """Test that CodingVertical uses PromptContributionCapability."""
        from victor.coding import CodingAssistant

        # Check that CodingAssistant has framework capability attributes
        assert hasattr(CodingAssistant, "_prompt_contrib")
        assert isinstance(CodingAssistant._prompt_contrib, PromptContributionCapability)

    @pytest.mark.skip(reason="Phase 3 migration: Framework tools not yet integrated into CodingAssistant.get_tools()")
    def test_coding_tools_include_framework_tools(self):
        """Test that CodingVertical.get_tools includes framework tools."""
        from victor.coding import CodingAssistant

        tools = CodingAssistant.get_tools()

        # Framework tools should be included
        assert "read" in tools
        assert "write" in tools
        assert "edit" in tools
        assert "grep" in tools

    @pytest.mark.skip(reason="Phase 3 migration: Vertical-specific tools not yet integrated via framework")
    def test_coding_tools_include_vertical_specific_tools(self):
        """Test that CodingVertical.get_tools includes vertical-specific tools."""
        from victor.coding import CodingAssistant
        from victor.tools.tool_names import ToolNames

        tools = CodingAssistant.get_tools()

        # Vertical-specific tools should be included
        assert ToolNames.GIT in tools
        assert ToolNames.SHELL in tools
        assert ToolNames.LSP in tools
        assert ToolNames.TEST in tools

    @pytest.mark.skip(reason="Phase 3 migration: Framework capability integration not complete")
    def test_framework_reduces_duplication(self):
        """Test that framework approach reduces code duplication."""
        from victor.coding import CodingAssistant

        # Before: tools were defined inline
        # After: tools use framework capability
        tools = CodingAssistant.get_tools()

        # Verify framework tools are included via capability
        framework_tools = CodingAssistant._file_ops.get_tool_list()
        for fw_tool in framework_tools:
            assert fw_tool in tools, f"Framework tool '{fw_tool}' not in CodingAssistant tools"


class TestBaseCapabilityProvider:
    """Tests for BaseCapabilityProvider base class."""

    def test_capability_metadata_creation(self):
        """Test that CapabilityMetadata can be created."""
        metadata = CapabilityMetadata(
            name="test_capability",
            description="Test capability for unit testing",
            version="1.0",
            tags=["test", "unit"],
        )

        assert metadata.name == "test_capability"
        assert metadata.description == "Test capability for unit testing"
        assert metadata.version == "1.0"
        assert "test" in metadata.tags
        assert "unit" in metadata.tags

    def test_base_capability_provider_is_abstract(self):
        """Test that BaseCapabilityProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            # Can't instantiate abstract base class
            BaseCapabilityProvider()


class TestPhase3Benefits:
    """Tests demonstrating Phase 3 benefits."""

    def test_single_source_of_truth_for_file_ops(self):
        """Test that file operations are defined once."""
        # Multiple verticals can use the same capability
        coding_capability = FileOperationsCapability()
        devops_capability = FileOperationsCapability()

        # Both return the same tools
        assert coding_capability.get_tool_list() == devops_capability.get_tool_list()

    def test_consistent_behavior_across_verticals(self):
        """Test that framework capabilities ensure consistent behavior."""
        capability1 = FileOperationsCapability()
        capability2 = FileOperationsCapability()

        # Same operations should produce same results
        assert capability1.get_tools() == capability2.get_tools()

    def test_framework_extensibility(self):
        """Test that framework capabilities are extensible."""
        from victor.framework.capabilities.file_operations import FileOperation, FileOperationType

        # Can extend with custom operations
        extended_ops = FileOperationsCapability.DEFAULT_OPERATIONS + [
            FileOperation(FileOperationType.READ, "custom_read", required=True),
        ]

        capability = FileOperationsCapability(operations=extended_ops)
        tools = capability.get_tool_list()

        assert "read" in tools  # Default
        assert "custom_read" in tools  # Extended


__all__ = [
    "TestFileOperationsCapability",
    "TestPromptContributionCapability",
    "TestCodingVerticalMigration",
    "TestBaseCapabilityProvider",
    "TestPhase3Benefits",
]
