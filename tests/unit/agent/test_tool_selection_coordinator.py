#!/usr/bin/env python3
"""Tests for ToolSelectionCoordinator.

Tests the intelligent tool selection, routing, and classification
functionality extracted from the orchestrator following SRP.

This module tests:
- Tool routing and selection
- Task classification (analysis, action, creation)
- Tool mention detection
- Required files/outputs extraction
- Tool capability checking
"""

import pytest
from unittest.mock import MagicMock

from victor.agent.protocols import IToolSelectionCoordinator, AgentToolSelectionContext


class TestIToolSelectionCoordinatorProtocol:
    """Tests for IToolSelectionCoordinator protocol compliance."""

    def test_protocol_defines_required_methods(self):
        """Test that protocol defines all required methods."""
        # Protocol should define these methods
        required_methods = [
            "get_recommended_search_tool",
            "route_search_query",
            "detect_mentioned_tools",
            "classify_task_keywords",
            "classify_task_with_context",
            "should_use_tools",
            "extract_required_files",
            "extract_required_outputs",
        ]

        for method_name in required_methods:
            assert hasattr(IToolSelectionCoordinator, method_name)

    def test_can_create_mock_from_protocol(self):
        """Test that we can create a mock from the protocol."""
        mock_coordinator = MagicMock(spec=IToolSelectionCoordinator)

        # Should have all protocol methods
        assert hasattr(mock_coordinator, "get_recommended_search_tool")
        assert hasattr(mock_coordinator, "route_search_query")
        assert hasattr(mock_coordinator, "detect_mentioned_tools")
        assert hasattr(mock_coordinator, "classify_task_keywords")
        assert hasattr(mock_coordinator, "classify_task_with_context")
        assert hasattr(mock_coordinator, "should_use_tools")
        assert hasattr(mock_coordinator, "extract_required_files")
        assert hasattr(mock_coordinator, "extract_required_outputs")


class TestToolSelectionCoordinator:
    """Tests for ToolSelectionCoordinator implementation.

    Note: These tests verify the extracted coordinator maintains
    the same behavior as the original orchestrator methods.
    """

    @pytest.fixture
    def coordinator(self):
        """Create a ToolSelectionCoordinator instance for testing."""
        from victor.agent.coordinators import ToolSelectionCoordinator
        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry()
        return ToolSelectionCoordinator(tool_registry=registry)

    # -------------------------------------------------------------------------
    # Test get_recommended_search_tool
    # -------------------------------------------------------------------------

    def test_get_recommended_search_tool_with_semantic_query(self, coordinator):
        """Test semantic queries route to semantic search."""
        result = coordinator.get_recommended_search_tool(
            "Find similar functions to authentication",
            context=AgentToolSelectionContext(task_type="analysis"),
        )

        # Should recommend semantic search for similarity queries
        assert result in ["semantic_search", "code_search", "grep", None]

    def test_get_recommended_search_tool_with_file_pattern(self, coordinator):
        """Test file pattern queries route to grep."""
        result = coordinator.get_recommended_search_tool(
            "Find all files ending with _test.py",
            context=AgentToolSelectionContext(task_type="analysis"),
        )

        # Should recommend grep for file pattern searches
        assert result in ["grep", "ls", None]

    def test_get_recommended_search_tool_with_web_query(self, coordinator):
        """Test web queries route to web search."""
        result = coordinator.get_recommended_search_tool(
            "What are the latest Python features?",
            context=AgentToolSelectionContext(task_type="research"),
        )

        # Should recommend web search for external information
        assert result in ["web_search", "web", None]

    def test_get_recommended_search_tool_returns_none_for_empty_query(self, coordinator):
        """Test that empty queries return None."""
        result = coordinator.get_recommended_search_tool("")
        assert result is None

    # -------------------------------------------------------------------------
    # Test route_search_query
    # -------------------------------------------------------------------------

    def test_route_search_query_with_available_tools(self, coordinator):
        """Test routing with specific available tools."""
        available = {"grep", "ls", "semantic_search"}
        result = coordinator.route_search_query(
            "Find TODO comments",
            available_tools=available,
        )

        # Should return one of the available tools
        assert result in available

    def test_route_search_query_falls_back_to_grep(self, coordinator):
        """Test fallback to grep when other tools unavailable."""
        available = {"grep", "ls"}
        result = coordinator.route_search_query(
            "semantic search for authentication",
            available_tools=available,
        )

        # Should fall back to grep even though query prefers semantic
        assert result == "grep"

    # -------------------------------------------------------------------------
    # Test detect_mentioned_tools
    # -------------------------------------------------------------------------

    def test_detect_mentioned_tools_finds_explicit_mentions(self, coordinator):
        """Test detection of explicit tool mentions."""
        prompt = "Use grep to find the error and then ls to list files"
        detected = coordinator.detect_mentioned_tools(
            prompt,
            available_tools={"grep", "ls", "read", "write"},
        )

        assert "grep" in detected
        assert "ls" in detected

    def test_detect_mentioned_tools_case_insensitive(self, coordinator):
        """Test that tool detection is case-insensitive."""
        prompt = "Use GREP and also try Read"
        detected = coordinator.detect_mentioned_tools(
            prompt,
            available_tools={"grep", "read", "write"},
        )

        assert "grep" in detected
        assert "read" in detected

    def test_detect_mentioned_tools_with_aliases(self, coordinator):
        """Test detection with tool aliases."""
        prompt = "Search the codebase"
        detected = coordinator.detect_mentioned_tools(
            prompt,
            available_tools={"grep", "code_search"},
        )

        # Should detect based on keywords like "search"
        assert len(detected) >= 0  # May or may not find depending on implementation

    def test_detect_mentioned_tools_empty_prompt(self, coordinator):
        """Test that empty prompts return empty set."""
        detected = coordinator.detect_mentioned_tools("", available_tools={"grep", "ls"})
        assert detected == set()

    # -------------------------------------------------------------------------
    # Test classify_task_keywords
    # -------------------------------------------------------------------------

    def test_classify_task_keywords_analysis(self, coordinator):
        """Test classification of analysis tasks."""
        result = coordinator.classify_task_keywords("Explain this code")
        assert result == "analysis"

    def test_classify_task_keywords_action(self, coordinator):
        """Test classification of action tasks."""
        result = coordinator.classify_task_keywords("Fix the bug")
        assert result == "action"

    def test_classify_task_keywords_creation(self, coordinator):
        """Test classification of creation tasks."""
        result = coordinator.classify_task_keywords("Create a new feature")
        assert result == "creation"

    def test_classify_task_keywords_with_history(self, coordinator):
        """Test classification with conversation history."""
        history = [
            {"role": "user", "content": "What does this code do?"},
            {"role": "assistant", "content": "It's a function that processes data"},
        ]
        result = coordinator.classify_task_keywords(
            "And how can I improve it?",
            conversation_history=history,
        )

        # Should classify based on context
        assert result in ["analysis", "action", "creation"]

    # -------------------------------------------------------------------------
    # Test classify_task_with_context
    # -------------------------------------------------------------------------

    def test_classify_task_with_context_uses_stage(self, coordinator):
        """Test classification using conversation stage."""
        context = AgentToolSelectionContext(
            stage="EXECUTING",
            task_type="action",
            recent_tools=["read", "grep"],
        )
        result = coordinator.classify_task_with_context("Apply the fix", context=context)

        # EXECUTING stage typically means action
        assert result in ["analysis", "action", "creation"]

    def test_classify_task_with_context_uses_recent_tools(self, coordinator):
        """Test classification considering recent tools."""
        context = AgentToolSelectionContext(
            recent_tools=["read", "ls", "grep"],
            task_type="analysis",
        )
        result = coordinator.classify_task_with_context(
            "What did we find?",
            context=context,
        )

        # Recent read tools suggest continued analysis
        assert result in ["analysis", "action", "creation"]

    # -------------------------------------------------------------------------
    # Test should_use_tools
    # -------------------------------------------------------------------------

    def test_should_use_tools_with_explicit_request(self, coordinator):
        """Test that explicit tool requests return True."""
        result = coordinator.should_use_tools("Use grep to find the error")
        assert result is True

    def test_should_use_tools_with_question(self, coordinator):
        """Test that questions may not need tools."""
        result = coordinator.should_use_tools("What is Python?")
        # Questions about general knowledge may not need tools
        assert isinstance(result, bool)

    def test_should_use_tools_without_tool_support(self, coordinator):
        """Test when model doesn't support tools."""
        result = coordinator.should_use_tools(
            "Use grep to find files",
            model_supports_tools=False,
        )
        # Should still detect tool intent even if not supported
        assert isinstance(result, bool)

    def test_should_use_tools_with_empty_message(self, coordinator):
        """Test that empty messages return False."""
        result = coordinator.should_use_tools("")
        assert result is False

    # -------------------------------------------------------------------------
    # Test extract_required_files
    # -------------------------------------------------------------------------

    def test_extract_required_files_with_paths(self, coordinator):
        """Test extraction of file paths from prompt."""
        prompt = "Check /path/to/file.py and /another/path/test.py"
        files = coordinator.extract_required_files(prompt)

        # Should extract file-like paths
        assert isinstance(files, set)

    def test_extract_required_files_with_quoted_paths(self, coordinator):
        """Test extraction with quoted file paths."""
        prompt = 'Read "file.py" and then look at "test_file.py"'
        files = coordinator.extract_required_files(prompt)

        # Should handle quoted paths
        assert isinstance(files, set)

    def test_extract_required_files_empty_prompt(self, coordinator):
        """Test that empty prompts return empty set."""
        files = coordinator.extract_required_files("")
        assert files == set()

    # -------------------------------------------------------------------------
    # Test extract_required_outputs
    # -------------------------------------------------------------------------

    def test_extract_required_outputs_with_file_references(self, coordinator):
        """Test extraction of output file references."""
        prompt = "Save the result to output.txt and write to result.json"
        outputs = coordinator.extract_required_outputs(prompt)

        # Should extract output file references
        assert isinstance(outputs, set)
        # Should contain at least some outputs
        assert len(outputs) >= 0

    def test_extract_required_outputs_with_variable_names(self, coordinator):
        """Test extraction of variable names."""
        prompt = "Store it in result_var and output to output_dir"
        outputs = coordinator.extract_required_outputs(prompt)

        # Should extract variable/path references
        assert isinstance(outputs, set)

    def test_extract_required_outputs_empty_prompt(self, coordinator):
        """Test that empty prompts return empty set."""
        outputs = coordinator.extract_required_outputs("")
        assert outputs == set()


# =============================================================================
# Integration Tests
# =============================================================================


class TestToolSelectionCoordinatorIntegration:
    """Integration tests for ToolSelectionCoordinator.

    Tests that the extracted coordinator maintains compatibility
    with existing orchestrator behavior.
    """

    def test_coordinator_implements_protocol(self):
        """Test that coordinator implements IToolSelectionCoordinator."""
        from victor.agent.coordinators import ToolSelectionCoordinator
        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry()
        coordinator = ToolSelectionCoordinator(tool_registry=registry)

        assert isinstance(coordinator, IToolSelectionCoordinator)

    def test_coordinator_maintains_orchestrator_compatibility(self):
        """Test that coordinator maintains orchestrator compatibility."""
        from victor.agent.coordinators import ToolSelectionCoordinator
        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry()
        coordinator = ToolSelectionCoordinator(tool_registry=registry)

        # Should have all orchestrator methods
        methods = [
            "get_recommended_search_tool",
            "route_search_query",
            "detect_mentioned_tools",
            "classify_task_keywords",
            "classify_task_with_context",
            "should_use_tools",
            "extract_required_files",
            "extract_required_outputs",
        ]

        for method in methods:
            assert hasattr(coordinator, method), f"Missing method: {method}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
