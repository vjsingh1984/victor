"""Unit tests for tool call deduplication tracker."""

import pytest
import time

from victor.agent.tool_deduplication import (
    ToolCall,
    ToolDeduplicationTracker,
    get_deduplication_tracker,
    is_redundant_call,
    track_call,
)


class TestToolCall:
    """Test ToolCall dataclass."""

    def test_initialization(self):
        """Test ToolCall initialization."""
        call = ToolCall(tool_name="read_file", args={"path": "foo.py"})

        assert call.tool_name == "read_file"
        assert call.args == {"path": "foo.py"}
        assert isinstance(call.timestamp, float)
        assert call.timestamp > 0

    def test_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated."""
        call1 = ToolCall(tool_name="test", args={})
        time.sleep(0.01)
        call2 = ToolCall(tool_name="test", args={})

        assert call2.timestamp > call1.timestamp


class TestToolDeduplicationTracker:
    """Test ToolDeduplicationTracker class."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        tracker = ToolDeduplicationTracker()

        assert tracker.window_size == 10
        assert tracker.similarity_threshold == 0.7
        assert len(tracker.recent_calls) == 0

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        tracker = ToolDeduplicationTracker(window_size=5, similarity_threshold=0.8)

        assert tracker.window_size == 5
        assert tracker.similarity_threshold == 0.8

    def test_add_call(self):
        """Test adding a tool call."""
        tracker = ToolDeduplicationTracker()
        tracker.add_call("read_file", {"path": "foo.py"})

        assert len(tracker.recent_calls) == 1
        assert tracker.recent_calls[0].tool_name == "read_file"
        assert tracker.recent_calls[0].args == {"path": "foo.py"}

    def test_window_size_enforcement(self):
        """Test that window size is enforced."""
        tracker = ToolDeduplicationTracker(window_size=3)

        # Add 5 calls (exceeds window size)
        for i in range(5):
            tracker.add_call("test", {"index": i})

        # Should only keep last 3
        assert len(tracker.recent_calls) == 3
        assert tracker.recent_calls[-1].args["index"] == 4

    def test_is_redundant_exact_duplicate(self):
        """Test detection of exact duplicate calls."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("read_file", {"path": "foo.py"})
        is_dup = tracker.is_redundant("read_file", {"path": "foo.py"})

        assert is_dup is True

    def test_is_redundant_different_call(self):
        """Test that different calls are not marked redundant."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("read_file", {"path": "foo.py"})
        is_dup = tracker.is_redundant("read_file", {"path": "bar.py"})

        assert is_dup is False

    def test_is_redundant_empty_history(self):
        """Test redundancy check with empty history."""
        tracker = ToolDeduplicationTracker()
        is_dup = tracker.is_redundant("read_file", {"path": "foo.py"})

        assert is_dup is False


class TestSearchRedundancy:
    """Test search/grep redundancy detection."""

    def test_exact_query_match(self):
        """Test exact query match detection."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("grep", {"query": "tool registration", "mode": "semantic"})
        is_dup = tracker.is_redundant("grep", {"query": "tool registration", "mode": "regex"})

        assert is_dup is True

    def test_synonym_query_match(self):
        """Test synonym query match detection."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("grep", {"query": "tool registration"})
        is_dup = tracker.is_redundant("grep", {"query": "register tool"})

        assert is_dup is True

    def test_substring_query_match(self):
        """Test substring query match detection."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("code_search", {"query": "error handling in providers"})
        is_dup = tracker.is_redundant("code_search", {"query": "error handling"})

        assert is_dup is True

    def test_different_queries_not_redundant(self):
        """Test that different queries are not redundant."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("grep", {"query": "tool registration"})
        is_dup = tracker.is_redundant("grep", {"query": "provider implementation"})

        assert is_dup is False

    def test_search_pattern_parameter(self):
        """Test search with 'pattern' instead of 'query' parameter."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("grep", {"pattern": "BaseProvider"})
        is_dup = tracker.is_redundant("grep", {"pattern": "BaseProvider"})

        assert is_dup is True


class TestFileRedundancy:
    """Test file operation redundancy detection."""

    def test_read_file_twice(self):
        """Test detection of reading same file twice."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("read_file", {"path": "foo.py"})
        is_dup = tracker.is_redundant("read_file", {"path": "foo.py"})

        assert is_dup is True

    def test_read_different_files(self):
        """Test that reading different files is not redundant."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("read_file", {"path": "foo.py"})
        is_dup = tracker.is_redundant("read_file", {"path": "bar.py"})

        assert is_dup is False

    def test_read_file_different_offsets(self):
        """Test that reading with different offsets is not redundant."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("read_file", {"path": "foo.py", "offset": 0})
        is_dup = tracker.is_redundant("read_file", {"path": "foo.py", "offset": 100})

        assert is_dup is False

    def test_file_path_parameter_variations(self):
        """Test different parameter names for file path."""
        tracker = ToolDeduplicationTracker()

        # Test "file_path" parameter
        tracker.add_call("edit_file", {"file_path": "foo.py"})
        is_dup = tracker.is_redundant("edit_file", {"file_path": "foo.py"})
        assert is_dup is True

        # Test "file" parameter
        tracker.clear()
        tracker.add_call("write_file", {"file": "bar.py"})
        is_dup = tracker.is_redundant("write_file", {"file": "bar.py"})
        assert is_dup is True


class TestListRedundancy:
    """Test list/directory operation redundancy detection."""

    def test_list_directory_twice(self):
        """Test detection of listing same directory twice."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("list_directory", {"path": "."})
        is_dup = tracker.is_redundant("list_directory", {"path": "."})

        assert is_dup is True

    def test_list_different_directories(self):
        """Test that listing different directories is not redundant."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("list_directory", {"path": "."})
        is_dup = tracker.is_redundant("list_directory", {"path": "src"})

        assert is_dup is False

    def test_list_default_path(self):
        """Test list with default path (no path parameter)."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("list_directory", {})
        is_dup = tracker.is_redundant("list_directory", {})

        assert is_dup is True


class TestUtilityMethods:
    """Test utility methods."""

    def test_get_recent_calls(self):
        """Test getting recent calls."""
        tracker = ToolDeduplicationTracker()

        for i in range(5):
            tracker.add_call("test", {"index": i})

        recent = tracker.get_recent_calls()
        assert len(recent) == 5
        # Should be reversed (most recent first)
        assert recent[0].args["index"] == 4
        assert recent[-1].args["index"] == 0

    def test_get_recent_calls_with_limit(self):
        """Test getting recent calls with limit."""
        tracker = ToolDeduplicationTracker()

        for i in range(10):
            tracker.add_call("test", {"index": i})

        recent = tracker.get_recent_calls(limit=3)
        assert len(recent) == 3
        assert recent[0].args["index"] == 9

    def test_clear(self):
        """Test clearing tracker."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("test", {})
        assert len(tracker.recent_calls) > 0

        tracker.clear()
        assert len(tracker.recent_calls) == 0

    def test_get_duplicate_count(self):
        """Test getting duplicate count."""
        tracker = ToolDeduplicationTracker()

        # Add some duplicates
        tracker.add_call("read_file", {"path": "foo.py"})
        tracker.add_call("read_file", {"path": "foo.py"})  # Duplicate
        tracker.add_call("read_file", {"path": "bar.py"})
        tracker.add_call("read_file", {"path": "bar.py"})  # Duplicate

        dup_count = tracker.get_duplicate_count()
        assert dup_count == 2


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_deduplication_tracker_singleton(self):
        """Test that get_deduplication_tracker returns singleton."""
        tracker1 = get_deduplication_tracker()
        tracker2 = get_deduplication_tracker()

        assert tracker1 is tracker2

    def test_is_redundant_call_convenience(self):
        """Test is_redundant_call convenience function."""
        tracker = get_deduplication_tracker()
        tracker.clear()  # Clear any previous state

        # First call should not be redundant
        result1 = is_redundant_call("test", {"arg": "value"})
        assert result1 is False

        # Add the call
        track_call("test", {"arg": "value"})

        # Same call should now be redundant
        result2 = is_redundant_call("test", {"arg": "value"})
        assert result2 is True

    def test_track_call_convenience(self):
        """Test track_call convenience function."""
        tracker = get_deduplication_tracker()
        tracker.clear()

        track_call("test_tool", {"test": "args"})

        recent = tracker.get_recent_calls()
        assert len(recent) == 1
        assert recent[0].tool_name == "test_tool"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_args(self):
        """Test with empty args dictionary."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("test", {})
        is_dup = tracker.is_redundant("test", {})

        assert is_dup is True

    def test_none_args_values(self):
        """Test with None values in args."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("test", {"path": None})
        is_dup = tracker.is_redundant("test", {"path": None})

        assert is_dup is True

    def test_complex_args(self):
        """Test with complex nested args."""
        tracker = ToolDeduplicationTracker()

        args = {"path": "foo.py", "options": {"recursive": True, "depth": 3}}
        tracker.add_call("test", args)
        is_dup = tracker.is_redundant("test", args)

        assert is_dup is True

    def test_case_sensitivity_in_queries(self):
        """Test case sensitivity in query matching."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("grep", {"query": "Tool Registration"})
        is_dup = tracker.is_redundant("grep", {"query": "tool registration"})

        # Should be case-insensitive
        assert is_dup is True

    def test_very_short_query(self):
        """Test with very short query (should not match substring)."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("grep", {"query": "tool registration"})
        is_dup = tracker.is_redundant("grep", {"query": "to"})  # < 3 chars

        # Should not match (query too short for substring matching)
        assert is_dup is False


class TestPerformance:
    """Test performance characteristics."""

    def test_redundancy_check_is_fast(self):
        """Test that redundancy check completes quickly."""
        tracker = ToolDeduplicationTracker(window_size=10)

        # Fill tracker with calls
        for i in range(10):
            tracker.add_call(f"tool{i}", {"arg": f"value{i}"})

        # Check redundancy 100 times
        start = time.time()
        for _ in range(100):
            tracker.is_redundant("grep", {"query": "test query"})

        elapsed = time.time() - start
        assert elapsed < 0.5  # Should complete 100 checks in under 0.5 seconds

    def test_large_window_size(self):
        """Test with large window size."""
        tracker = ToolDeduplicationTracker(window_size=1000)

        # Add many calls
        for i in range(500):
            tracker.add_call("test", {"index": i})

        # Should still work efficiently
        is_dup = tracker.is_redundant("test", {"index": 0})
        assert is_dup is True


class TestSemanticSimilarity:
    """Test semantic similarity detection."""

    def test_error_handling_synonyms(self):
        """Test error handling synonym detection."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("grep", {"query": "error handling"})
        is_dup = tracker.is_redundant("grep", {"query": "exception"})

        assert is_dup is True

    def test_provider_synonyms(self):
        """Test provider synonym detection."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("grep", {"query": "provider"})
        is_dup = tracker.is_redundant("grep", {"query": "llm provider"})

        assert is_dup is True

    def test_unrelated_queries(self):
        """Test that unrelated queries are not matched."""
        tracker = ToolDeduplicationTracker()

        tracker.add_call("grep", {"query": "error handling"})
        is_dup = tracker.is_redundant("grep", {"query": "database connection"})

        assert is_dup is False
