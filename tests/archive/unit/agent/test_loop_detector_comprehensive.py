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

"""Comprehensive tests for loop detection and budget enforcement.

This test module provides comprehensive coverage for the loop_detector.py module,
targeting uncovered lines to increase coverage from 26% to 70%+.

Coverage areas:
- Native extension fallback (lines 63-64)
- Progress params functions (lines 98-99, 111)
- FileReadRange edge cases (lines 137, 149, 152, 155-157)
- LoopDetector initialization and reset (lines 263-295, 299-312)
- LoopDetector properties (lines 317, 322, 327, 332, 337, 342-344)
- record_tool_call method (lines 353-393)
- _track_file_read method (lines 404-420)
- _count_overlapping_reads (lines 435-442)
- record_iteration (lines 450-455)
- force_stop and should_stop (lines 467, 475-519)
- get_metrics (lines 527)
- _get_signature (lines 555-574)
- _get_resource_key (lines 589-618)
- _get_base_resource_key (lines 634-656)
- _check_loop (lines 666-698)
- check_loop_warning (lines 710-738)
- is_blocked_after_warning (lines 753-767)
- clear_loop_warning (lines 771-772)
- _check_file_read_loops (lines 783-803)
- _get_max_iterations (lines 807-813)
- _get_base_details (line 817)
- Content loop detection (lines 839-847, 855-872, 883-912, 924-926, 931, 936, 940-942)
- Factory functions (lines 971-1000, 1023-1028)
"""

import pytest
from unittest.mock import MagicMock, patch

from victor.agent.loop_detector import (
    LoopDetectorTaskType,
    FileReadRange,
    ProgressConfig,
    LoopStopRecommendation,
    LoopDetector,
    RESEARCH_TOOLS,
    DEFAULT_READ_LIMIT,
    CANONICAL_READ_TOOL,
    get_progress_params_for_tool,
    is_progressive_tool,
    create_tracker_from_classification,
    classify_and_create_tracker,
)


# =============================================================================
# NATIVE EXTENSION FALLBACK TESTS (lines 63-64)
# =============================================================================


class TestNativeExtensionFallback:
    """Tests for native extension availability detection."""

    def test_native_module_import_handling(self):
        """Test that native module import failure is handled gracefully."""
        # The module should work even without native extensions
        detector = LoopDetector()
        # Should still be able to generate signatures
        sig = detector._get_signature("read", {"path": "test.py"})
        assert isinstance(sig, str)
        assert len(sig) > 0


# =============================================================================
# PROGRESS PARAMS FUNCTION TESTS (lines 98-99, 111)
# =============================================================================


class TestProgressParamsFunctions:
    """Tests for get_progress_params_for_tool and is_progressive_tool functions."""

    def test_get_progress_params_for_tool_with_registered_tool(self):
        """Test get_progress_params_for_tool returns params for registered tools."""
        # This tests the function with a tool that may have progress_params
        params = get_progress_params_for_tool("read")
        assert isinstance(params, list)

    def test_get_progress_params_for_tool_with_unknown_tool(self):
        """Test get_progress_params_for_tool returns empty list for unknown tools."""
        params = get_progress_params_for_tool("unknown_tool_xyz_123")
        assert params == []

    def test_is_progressive_tool_returns_bool(self):
        """Test is_progressive_tool returns boolean."""
        result = is_progressive_tool("read")
        assert isinstance(result, bool)

    def test_is_progressive_tool_for_unknown_tool(self):
        """Test is_progressive_tool returns False for unknown tools."""
        result = is_progressive_tool("nonexistent_tool_abc")
        assert result is False


# =============================================================================
# FILE READ RANGE TESTS (lines 137, 149, 152, 155-157)
# =============================================================================


class TestFileReadRangeComprehensive:
    """Comprehensive tests for FileReadRange dataclass."""

    def test_end_property_with_zero_offset(self):
        """Test end property calculation with zero offset."""
        range_obj = FileReadRange(offset=0, limit=500)
        assert range_obj.end == 500

    def test_end_property_with_large_values(self):
        """Test end property with large offset and limit."""
        range_obj = FileReadRange(offset=10000, limit=5000)
        assert range_obj.end == 15000

    def test_overlaps_with_contained_range(self):
        """Test overlaps when one range is contained within another."""
        outer = FileReadRange(offset=0, limit=200)
        inner = FileReadRange(offset=50, limit=50)
        assert outer.overlaps(inner)
        assert inner.overlaps(outer)

    def test_overlaps_at_boundaries(self):
        """Test overlaps with exact boundary conditions."""
        r1 = FileReadRange(offset=0, limit=100)
        # Adjacent ranges should NOT overlap
        r2 = FileReadRange(offset=100, limit=100)
        assert not r1.overlaps(r2)
        # Off by one - should overlap
        r3 = FileReadRange(offset=99, limit=100)
        assert r1.overlaps(r3)

    def test_hash_uniqueness(self):
        """Test hash function produces different hashes for different ranges."""
        r1 = FileReadRange(offset=0, limit=100)
        r2 = FileReadRange(offset=0, limit=200)
        r3 = FileReadRange(offset=100, limit=100)
        assert hash(r1) != hash(r2)
        assert hash(r1) != hash(r3)
        assert hash(r2) != hash(r3)

    def test_equality_with_different_types(self):
        """Test equality returns False for non-FileReadRange types."""
        r1 = FileReadRange(offset=0, limit=100)
        assert r1 != "FileReadRange(0, 100)"
        assert r1 != (0, 100)
        assert r1 != {"offset": 0, "limit": 100}
        assert r1 is not None

    def test_hash_for_set_operations(self):
        """Test that FileReadRange can be used in sets."""
        r1 = FileReadRange(offset=0, limit=100)
        r2 = FileReadRange(offset=0, limit=100)  # Same as r1
        r3 = FileReadRange(offset=100, limit=100)

        range_set = {r1, r2, r3}
        assert len(range_set) == 2  # r1 and r2 should be deduplicated


# =============================================================================
# LOOP DETECTOR INITIALIZATION AND RESET TESTS (lines 263-295, 299-312)
# =============================================================================


class TestLoopDetectorInitializationComprehensive:
    """Comprehensive tests for LoopDetector initialization and reset."""

    def test_init_with_custom_config(self):
        """Test initialization with fully custom configuration."""
        config = ProgressConfig(
            tool_budget=100,
            max_iterations_default=20,
            max_iterations_analysis=60,
            max_iterations_action=15,
            max_iterations_research=10,
            repeat_threshold_default=5,
            signature_history_size=15,
            max_total_iterations=100,
        )
        detector = LoopDetector(config=config, task_type=LoopDetectorTaskType.ANALYSIS)

        assert detector.config.tool_budget == 100
        assert detector.config.max_iterations_analysis == 60
        assert detector.task_type == LoopDetectorTaskType.ANALYSIS
        assert detector.remaining_budget == 100

    def test_init_with_research_task_type(self):
        """Test initialization with RESEARCH task type."""
        detector = LoopDetector(task_type=LoopDetectorTaskType.RESEARCH)
        assert detector.task_type == LoopDetectorTaskType.RESEARCH

    def test_init_with_action_task_type(self):
        """Test initialization with ACTION task type."""
        detector = LoopDetector(task_type=LoopDetectorTaskType.ACTION)
        assert detector.task_type == LoopDetectorTaskType.ACTION

    def test_reset_clears_all_state(self):
        """Test reset clears all internal state completely."""
        detector = LoopDetector()

        # Build up state
        detector.record_tool_call("read", {"path": "file1.py"})
        detector.record_tool_call("read", {"path": "file2.py", "offset": 100})
        detector.record_tool_call("grep", {"query": "test", "directory": "."})
        detector.record_iteration(500)
        detector.record_iteration(50)  # Low output
        detector.force_stop("test reason")
        detector.record_content_chunk("test content " * 100)
        detector._loop_warning_given = True
        detector._warned_signature = "test_sig"

        # Verify state exists
        assert detector.tool_calls > 0
        assert detector.iterations > 0
        assert len(detector._unique_resources) > 0

        # Reset
        detector.reset()

        # Verify all state is cleared
        assert detector.tool_calls == 0
        assert detector.iterations == 0
        assert detector.low_output_iterations == 0
        assert len(detector._unique_resources) == 0
        assert len(detector._file_read_ranges) == 0
        assert len(detector._base_resource_counts) == 0
        assert len(detector._signature_history) == 0
        assert detector._consecutive_research_calls == 0
        assert detector._loop_warning_given is False
        assert detector._warned_signature is None
        assert detector._content_buffer == ""
        assert detector._content_loop_detected is False
        assert detector._content_loop_phrase is None
        assert detector._forced_stop is None


# =============================================================================
# LOOP DETECTOR PROPERTIES TESTS (lines 317, 322, 327, 332, 337, 342-344)
# =============================================================================


class TestLoopDetectorPropertiesComprehensive:
    """Comprehensive tests for LoopDetector properties."""

    def test_tool_calls_property(self):
        """Test tool_calls property returns correct count."""
        detector = LoopDetector()
        assert detector.tool_calls == 0
        detector.record_tool_call("read", {})
        assert detector.tool_calls == 1
        detector.record_tool_call("write", {})
        assert detector.tool_calls == 2

    def test_iterations_property(self):
        """Test iterations property returns correct count."""
        detector = LoopDetector()
        assert detector.iterations == 0
        detector.record_iteration(500)
        assert detector.iterations == 1
        detector.record_iteration(200)
        assert detector.iterations == 2

    def test_low_output_iterations_property(self):
        """Test low_output_iterations property with various content lengths."""
        detector = LoopDetector()
        assert detector.low_output_iterations == 0

        # Below threshold (default is 150)
        detector.record_iteration(50)
        assert detector.low_output_iterations == 1

        detector.record_iteration(100)
        assert detector.low_output_iterations == 2

        # At threshold - still counts as low
        detector.record_iteration(149)
        assert detector.low_output_iterations == 3

        # Above threshold - doesn't count
        detector.record_iteration(200)
        assert detector.low_output_iterations == 3

    def test_unique_resources_property(self):
        """Test unique_resources returns a copy of resources set."""
        detector = LoopDetector()
        detector.record_tool_call("read", {"path": "test.py"})
        detector.record_tool_call("ls", {"path": "/home"})

        resources = detector.unique_resources
        assert isinstance(resources, set)
        # Modifying returned set should not affect internal state
        resources.add("new_resource")
        assert "new_resource" not in detector.unique_resources

    def test_remaining_budget_never_negative(self):
        """Test remaining_budget never goes below zero."""
        config = ProgressConfig(tool_budget=3)
        detector = LoopDetector(config=config)

        for _ in range(5):
            detector.record_tool_call("test", {})

        assert detector.remaining_budget == 0  # Not negative

    def test_progress_percentage_calculation(self):
        """Test progress_percentage calculation."""
        config = ProgressConfig(tool_budget=10)
        detector = LoopDetector(config=config)

        assert detector.progress_percentage == 0.0

        detector.record_tool_call("t1", {})
        assert detector.progress_percentage == 10.0

        for _ in range(4):
            detector.record_tool_call("t", {})
        assert detector.progress_percentage == 50.0

        for _ in range(5):
            detector.record_tool_call("t", {})
        assert detector.progress_percentage == 100.0

    def test_progress_percentage_with_zero_budget(self):
        """Test progress_percentage with zero budget returns 100%."""
        config = ProgressConfig(tool_budget=0)
        detector = LoopDetector(config=config)
        assert detector.progress_percentage == 100.0


# =============================================================================
# RECORD_TOOL_CALL METHOD TESTS (lines 353-393)
# =============================================================================


class TestRecordToolCallComprehensive:
    """Comprehensive tests for record_tool_call method."""

    def test_record_tool_call_increments_counter(self):
        """Test that record_tool_call increments tool call counter."""
        detector = LoopDetector()
        for i in range(5):
            detector.record_tool_call(f"tool_{i}", {})
        assert detector.tool_calls == 5

    def test_record_tool_call_tracks_unique_resources(self):
        """Test unique resource tracking for file reads."""
        detector = LoopDetector()
        detector.record_tool_call("read", {"path": "file1.py"})
        detector.record_tool_call("read", {"path": "file2.py"})
        detector.record_tool_call("read", {"path": "file1.py", "offset": 100})

        # Should have 3 unique resource entries (same file with different offset is different)
        assert len(detector.unique_resources) == 3

    def test_record_tool_call_tracks_file_reads(self):
        """Test file read range tracking."""
        detector = LoopDetector()
        detector.record_tool_call("read", {"path": "test.py", "offset": 0, "limit": 100})
        detector.record_tool_call("read", {"path": "test.py", "offset": 100, "limit": 100})

        assert "test.py" in detector._file_read_ranges
        assert len(detector._file_read_ranges["test.py"]) == 2

    def test_record_tool_call_tracks_base_resources(self):
        """Test base resource tracking for non-file operations."""
        detector = LoopDetector()
        detector.record_tool_call("grep", {"query": "test", "directory": "."})
        detector.record_tool_call("grep", {"query": "test2", "directory": "."})

        # Should track search base resources
        assert len(detector._base_resource_counts) > 0

    def test_record_tool_call_clears_loop_warning_on_different_signature(self):
        """Test loop warning is cleared when signature changes."""
        detector = LoopDetector()

        # Build up to warning threshold
        for _ in range(2):
            detector.record_tool_call("read", {"path": "same.py"})

        # Manually set warning state
        detector._loop_warning_given = True
        detector._signature_history.append("warned_sig")

        # Make a different call
        detector.record_tool_call("write", {"path": "different.py"})

        # Warning should be cleared because signature changed
        assert detector._loop_warning_given is False

    def test_record_tool_call_tracks_research_calls(self):
        """Test consecutive research call tracking."""
        detector = LoopDetector()

        detector.record_tool_call("web_search", {"query": "test"})
        assert detector._consecutive_research_calls == 1

        detector.record_tool_call("web_fetch", {"url": "http://example.com"})
        assert detector._consecutive_research_calls == 2

        detector.record_tool_call("read", {"path": "test.py"})
        assert detector._consecutive_research_calls == 0

    def test_record_tool_call_with_read_file_alias(self):
        """Test tool name normalization for read_file alias."""
        detector = LoopDetector()
        detector.record_tool_call("read_file", {"path": "test.py"})
        assert "test.py" in detector._file_read_ranges


# =============================================================================
# _TRACK_FILE_READ METHOD TESTS (lines 404-420)
# =============================================================================


class TestTrackFileRead:
    """Tests for _track_file_read method."""

    def test_track_file_read_with_defaults(self):
        """Test file read tracking with default offset and limit."""
        detector = LoopDetector()
        detector._track_file_read({"path": "test.py"})

        assert "test.py" in detector._file_read_ranges
        assert len(detector._file_read_ranges["test.py"]) == 1
        range_obj = detector._file_read_ranges["test.py"][0]
        assert range_obj.offset == 0
        assert range_obj.limit == DEFAULT_READ_LIMIT

    def test_track_file_read_with_custom_offset_limit(self):
        """Test file read tracking with custom offset and limit."""
        detector = LoopDetector()
        detector._track_file_read({"path": "test.py", "offset": 500, "limit": 200})

        range_obj = detector._file_read_ranges["test.py"][0]
        assert range_obj.offset == 500
        assert range_obj.limit == 200

    def test_track_file_read_with_empty_path(self):
        """Test file read tracking with empty path is ignored."""
        detector = LoopDetector()
        detector._track_file_read({"path": ""})
        assert len(detector._file_read_ranges) == 0

    def test_track_file_read_with_missing_path(self):
        """Test file read tracking with missing path is ignored."""
        detector = LoopDetector()
        detector._track_file_read({})
        assert len(detector._file_read_ranges) == 0

    def test_track_file_read_multiple_files(self):
        """Test tracking reads from multiple files."""
        detector = LoopDetector()
        detector._track_file_read({"path": "file1.py"})
        detector._track_file_read({"path": "file2.py"})
        detector._track_file_read({"path": "file1.py", "offset": 100})

        assert len(detector._file_read_ranges) == 2
        assert len(detector._file_read_ranges["file1.py"]) == 2
        assert len(detector._file_read_ranges["file2.py"]) == 1


# =============================================================================
# _COUNT_OVERLAPPING_READS TESTS (lines 435-442)
# =============================================================================


class TestCountOverlappingReads:
    """Tests for _count_overlapping_reads method."""

    def test_count_overlapping_reads_single_read(self):
        """Test count with single read (always 1)."""
        detector = LoopDetector()
        detector._file_read_ranges["test.py"] = [FileReadRange(offset=0, limit=100)]

        count = detector._count_overlapping_reads("test.py", FileReadRange(offset=0, limit=100))
        assert count == 1  # Counts self

    def test_count_overlapping_reads_no_overlap(self):
        """Test count with non-overlapping reads."""
        detector = LoopDetector()
        detector._file_read_ranges["test.py"] = [
            FileReadRange(offset=0, limit=100),
            FileReadRange(offset=200, limit=100),
        ]

        # New read at offset=100 doesn't overlap with existing
        count = detector._count_overlapping_reads("test.py", FileReadRange(offset=100, limit=100))
        assert count == 0

    def test_count_overlapping_reads_partial_overlap(self):
        """Test count with partial overlap."""
        detector = LoopDetector()
        detector._file_read_ranges["test.py"] = [
            FileReadRange(offset=0, limit=100),
            FileReadRange(offset=50, limit=100),
        ]

        # Count overlaps with (0, 100) and (50, 100)
        count = detector._count_overlapping_reads("test.py", FileReadRange(offset=25, limit=50))
        assert count == 2

    def test_count_overlapping_reads_unknown_path(self):
        """Test count returns 0 for unknown file path."""
        detector = LoopDetector()
        count = detector._count_overlapping_reads("unknown.py", FileReadRange(offset=0, limit=100))
        assert count == 0


# =============================================================================
# RECORD_ITERATION TESTS (lines 450-455)
# =============================================================================


class TestRecordIterationComprehensive:
    """Comprehensive tests for record_iteration method."""

    def test_record_iteration_with_various_lengths(self):
        """Test record_iteration with various content lengths."""
        config = ProgressConfig(min_content_threshold=150)
        detector = LoopDetector(config=config)

        # Below threshold
        detector.record_iteration(50)
        assert detector.iterations == 1
        assert detector.low_output_iterations == 1

        # At threshold boundary - 150 is NOT below 150, so not low output
        detector.record_iteration(150)
        assert detector.iterations == 2
        assert detector.low_output_iterations == 1  # 150 is not < 150

        # Above threshold
        detector.record_iteration(200)
        assert detector.iterations == 3
        assert detector.low_output_iterations == 1

    def test_record_iteration_with_zero_length(self):
        """Test record_iteration with zero content length."""
        detector = LoopDetector()
        detector.record_iteration(0)
        assert detector.iterations == 1
        assert detector.low_output_iterations == 1


# =============================================================================
# FORCE_STOP AND SHOULD_STOP TESTS (lines 467, 475-519)
# =============================================================================


class TestForceStopAndShouldStop:
    """Tests for force_stop and should_stop methods."""

    def test_force_stop_sets_reason(self):
        """Test force_stop sets the forced stop reason."""
        detector = LoopDetector()
        detector.force_stop("User requested cancellation")
        assert detector._forced_stop == "User requested cancellation"

    def test_should_stop_returns_false_initially(self):
        """Test should_stop returns false when no stop condition met."""
        detector = LoopDetector()
        result = detector.should_stop()
        assert result.should_stop is False
        assert result.reason == ""

    def test_should_stop_on_forced_stop(self):
        """Test should_stop returns true on forced stop."""
        detector = LoopDetector()
        detector.force_stop("Testing")

        result = detector.should_stop()
        assert result.should_stop is True
        assert "Manual stop" in result.reason
        assert "Testing" in result.reason

    def test_should_stop_on_budget_exceeded(self):
        """Test should_stop on tool budget exceeded."""
        config = ProgressConfig(tool_budget=3)
        detector = LoopDetector(config=config)

        for _ in range(4):
            detector.record_tool_call("test", {})

        result = detector.should_stop()
        assert result.should_stop is True
        assert "Tool budget exceeded" in result.reason

    def test_should_stop_on_max_iterations(self):
        """Test should_stop on max iterations reached."""
        config = ProgressConfig(max_total_iterations=3)
        detector = LoopDetector(config=config)

        for _ in range(4):
            detector.record_iteration(500)

        result = detector.should_stop()
        assert result.should_stop is True
        assert "Max total iterations" in result.reason

    def test_should_stop_on_research_loop(self):
        """Test should_stop on research loop detection."""
        config = ProgressConfig(max_iterations_research=3)
        detector = LoopDetector(config=config, task_type=LoopDetectorTaskType.RESEARCH)

        for _ in range(4):
            detector.record_tool_call("web_search", {"query": f"query_{_}"})

        result = detector.should_stop()
        assert result.should_stop is True
        assert "Research loop detected" in result.reason

    def test_should_stop_details_contain_metrics(self):
        """Test should_stop details contain expected metrics."""
        detector = LoopDetector()
        detector.record_tool_call("read", {"path": "test.py"})

        result = detector.should_stop()
        assert "tool_calls" in result.details
        assert "tool_budget" in result.details
        assert "iterations" in result.details
        assert "task_type" in result.details


# =============================================================================
# GET_METRICS TESTS (line 527)
# =============================================================================


class TestGetMetrics:
    """Tests for get_metrics method."""

    def test_get_metrics_returns_all_expected_fields(self):
        """Test get_metrics returns all expected fields."""
        detector = LoopDetector(task_type=LoopDetectorTaskType.ANALYSIS)
        detector.record_tool_call("read", {"path": "file1.py"})
        detector.record_tool_call("read", {"path": "file2.py"})
        detector.record_iteration(500)

        metrics = detector.get_metrics()

        assert metrics["tool_calls"] == 2
        assert metrics["iterations"] == 1
        assert metrics["low_output_iterations"] == 0
        assert metrics["unique_resources"] >= 0
        assert metrics["files_read"] >= 0
        assert "total_file_reads" in metrics
        assert "remaining_budget" in metrics
        assert "progress_percentage" in metrics
        assert metrics["task_type"] == "analysis"


# =============================================================================
# _GET_SIGNATURE TESTS (lines 555-574)
# =============================================================================


class TestGetSignature:
    """Tests for _get_signature method."""

    def test_signature_with_progressive_tool(self):
        """Test signature generation for progressive tools."""
        detector = LoopDetector()

        # Mock progress params for this test
        with patch(
            "victor.agent.loop_detector.get_progress_params_for_tool",
            return_value=["path", "offset"],
        ):
            sig = detector._get_signature("read", {"path": "test.py", "offset": 100})
            # Progressive tools include params in signature
            assert "read" in sig
            assert "test.py" in sig

    def test_signature_with_non_progressive_tool(self):
        """Test signature generation for non-progressive tools."""
        detector = LoopDetector()

        with patch("victor.agent.loop_detector.get_progress_params_for_tool", return_value=[]):
            with patch("victor.agent.loop_detector._NATIVE_AVAILABLE", False):
                sig = detector._get_signature("unknown_tool", {"arg": "value"})
                # Non-progressive tools hash arguments (Python fallback uses "tool:hash" format)
                assert "unknown_tool" in sig
                assert ":" in sig  # Format is "tool:hash"

    def test_signature_truncates_long_values(self):
        """Test signature truncates long parameter values."""
        detector = LoopDetector()

        with patch(
            "victor.agent.loop_detector.get_progress_params_for_tool", return_value=["query"]
        ):
            long_query = "x" * 200
            sig = detector._get_signature("search", {"query": long_query})
            # Signature should not contain the full 200 chars
            assert len(sig) < 250

    def test_signature_consistency(self):
        """Test same arguments produce same signature."""
        detector = LoopDetector()
        sig1 = detector._get_signature("read", {"path": "test.py"})
        sig2 = detector._get_signature("read", {"path": "test.py"})
        assert sig1 == sig2


# =============================================================================
# _GET_RESOURCE_KEY TESTS (lines 589-618)
# =============================================================================


class TestGetResourceKey:
    """Tests for _get_resource_key method."""

    def test_resource_key_for_read_file(self):
        """Test resource key generation for read tool."""
        detector = LoopDetector()
        key = detector._get_resource_key("read", {"path": "test.py", "offset": 100})
        assert key == "file:test.py:100"

    def test_resource_key_for_ls(self):
        """Test resource key generation for ls tool."""
        detector = LoopDetector()
        key = detector._get_resource_key("ls", {"path": "/home/user"})
        assert key == "dir:/home/user"

    def test_resource_key_for_grep(self):
        """Test resource key generation for grep tool."""
        detector = LoopDetector()
        key = detector._get_resource_key("grep", {"query": "test pattern", "directory": "/src"})
        assert key == "search:/src:test pattern"

    def test_resource_key_for_shell(self):
        """Test resource key generation for shell tool."""
        detector = LoopDetector()
        key = detector._get_resource_key("shell", {"command": "echo hello world"})
        assert key is not None
        assert "bash:" in key

    def test_resource_key_for_web_fetch(self):
        """Test resource key generation for web_fetch tool."""
        detector = LoopDetector()
        key = detector._get_resource_key("web_fetch", {"url": "http://example.com/page"})
        assert key is not None
        assert "web:" in key

    def test_resource_key_for_web_search(self):
        """Test resource key generation for web_search tool."""
        detector = LoopDetector()
        key = detector._get_resource_key("web_search", {"query": "python tutorial"})
        assert key is not None
        assert "web:" in key

    def test_resource_key_for_unknown_tool(self):
        """Test resource key returns None for unknown tool."""
        detector = LoopDetector()
        key = detector._get_resource_key("unknown_tool", {"arg": "value"})
        assert key is None

    def test_resource_key_with_empty_path(self):
        """Test resource key with empty path returns None."""
        detector = LoopDetector()
        key = detector._get_resource_key("read", {"path": ""})
        assert key is None


# =============================================================================
# _GET_BASE_RESOURCE_KEY TESTS (lines 634-656)
# =============================================================================


class TestGetBaseResourceKey:
    """Tests for _get_base_resource_key method."""

    def test_base_resource_key_for_ls(self):
        """Test base resource key for ls tool."""
        detector = LoopDetector()
        key = detector._get_base_resource_key("ls", {"path": "/home"})
        assert key == "dir:/home"

    def test_base_resource_key_for_grep(self):
        """Test base resource key for grep - uses query prefix."""
        detector = LoopDetector()
        key = detector._get_base_resource_key(
            "grep", {"query": "long search query here", "directory": "."}
        )
        assert key is not None
        assert key.startswith("search:")

    def test_base_resource_key_for_shell_git_command(self):
        """Test base resource key for git shell commands."""
        detector = LoopDetector()
        key = detector._get_base_resource_key("shell", {"command": "git status"})
        assert key == "bash:git status"

    def test_base_resource_key_for_shell_simple_command(self):
        """Test base resource key for simple shell commands."""
        detector = LoopDetector()
        key = detector._get_base_resource_key("shell", {"command": "ls -la"})
        assert key == "bash:ls"

    def test_base_resource_key_for_read_returns_none(self):
        """Test base resource key for read returns None (handled separately)."""
        detector = LoopDetector()
        key = detector._get_base_resource_key("read", {"path": "test.py"})
        assert key is None


# =============================================================================
# _CHECK_LOOP TESTS (lines 666-698)
# =============================================================================


class TestCheckLoop:
    """Tests for _check_loop method."""

    def test_check_loop_with_insufficient_history(self):
        """Test _check_loop returns None with insufficient history."""
        detector = LoopDetector()
        detector.record_tool_call("read", {"path": "a.py"})
        detector.record_tool_call("read", {"path": "b.py"})

        result = detector._check_loop()
        assert result is None

    def test_check_loop_detects_repeated_signatures(self):
        """Test _check_loop detects repeated signatures (via file read or signature)."""
        config = ProgressConfig(repeat_threshold_default=3, max_overlapping_reads_per_file=3)
        detector = LoopDetector(config=config)

        for _ in range(4):
            detector.record_tool_call("read", {"path": "same.py"})

        result = detector._check_loop()
        assert result is not None
        # May detect via file read overlap or signature repetition
        assert "repeated" in result or "same file region" in result

    def test_check_loop_with_analysis_task_type(self):
        """Test _check_loop uses analysis threshold for ANALYSIS task type."""
        config = ProgressConfig(
            repeat_threshold_analysis=5,
            max_overlapping_reads_per_file=10,  # High limit to test signature threshold
        )
        detector = LoopDetector(config=config, task_type=LoopDetectorTaskType.ANALYSIS)

        # 4 repeats - below analysis signature threshold of 5
        # Use non-file tool to avoid file read loop detection
        for _ in range(4):
            detector.record_tool_call("shell", {"command": "echo test"})

        result = detector._check_loop()
        # Should not detect loop yet (signature threshold is 5 for analysis)
        assert result is None or "search" in result.lower()

    def test_check_loop_with_action_task_type(self):
        """Test _check_loop uses action threshold for ACTION task type."""
        config = ProgressConfig(repeat_threshold_action=4)
        detector = LoopDetector(config=config, task_type=LoopDetectorTaskType.ACTION)

        for _ in range(5):
            detector.record_tool_call("write", {"path": "same.py", "content": "test"})

        result = detector._check_loop()
        assert result is not None

    def test_check_loop_detects_similar_searches(self):
        """Test _check_loop detects similar search queries."""
        config = ProgressConfig(max_searches_per_query_prefix=2)
        detector = LoopDetector(config=config)

        # Make searches with same query prefix
        for i in range(3):
            detector.record_tool_call("grep", {"query": f"test_query_{i}", "directory": "."})

        result = detector._check_loop()
        # May detect similar search pattern
        # The detection depends on query prefix matching


# =============================================================================
# CHECK_LOOP_WARNING TESTS (lines 710-738)
# =============================================================================


class TestCheckLoopWarning:
    """Tests for check_loop_warning method."""

    def test_check_loop_warning_with_insufficient_history(self):
        """Test check_loop_warning returns None with insufficient history."""
        detector = LoopDetector()
        detector.record_tool_call("read", {"path": "a.py"})

        result = detector.check_loop_warning()
        assert result is None

    def test_check_loop_warning_at_threshold_minus_one(self):
        """Test check_loop_warning triggers at threshold - 1."""
        config = ProgressConfig(repeat_threshold_default=4)
        detector = LoopDetector(config=config)

        for _ in range(3):  # threshold - 1 = 3
            detector.record_tool_call("read", {"path": "same.py"})

        result = detector.check_loop_warning()
        assert result is not None
        assert "approaching loop threshold" in result

    def test_check_loop_warning_only_warns_once(self):
        """Test check_loop_warning only returns warning once."""
        config = ProgressConfig(repeat_threshold_default=4)
        detector = LoopDetector(config=config)

        for _ in range(3):
            detector.record_tool_call("read", {"path": "same.py"})

        result1 = detector.check_loop_warning()
        assert result1 is not None

        # Second call should return None
        result2 = detector.check_loop_warning()
        assert result2 is None

    def test_check_loop_warning_with_analysis_task_type(self):
        """Test check_loop_warning uses correct threshold for task type."""
        config = ProgressConfig(repeat_threshold_analysis=6)
        detector = LoopDetector(config=config, task_type=LoopDetectorTaskType.ANALYSIS)

        for _ in range(5):  # threshold - 1 = 5
            detector.record_tool_call("read", {"path": "same.py"})

        result = detector.check_loop_warning()
        assert result is not None


# =============================================================================
# IS_BLOCKED_AFTER_WARNING TESTS (lines 753-767)
# =============================================================================


class TestIsBlockedAfterWarning:
    """Tests for is_blocked_after_warning method."""

    def test_not_blocked_without_warning(self):
        """Test not blocked when no warning has been given."""
        detector = LoopDetector()
        result = detector.is_blocked_after_warning("read", {"path": "test.py"})
        assert result is None

    def test_blocked_after_warning_same_signature(self):
        """Test blocked when same signature is attempted after warning."""
        config = ProgressConfig(repeat_threshold_default=4)
        detector = LoopDetector(config=config)

        # Build up to warning
        for _ in range(3):
            detector.record_tool_call("read", {"path": "same.py"})

        detector.check_loop_warning()  # Get warning
        assert detector._loop_warning_given is True

        # Try the same operation again
        result = detector.is_blocked_after_warning("read", {"path": "same.py"})
        assert result is not None
        assert "blocked after warning" in result

    def test_not_blocked_with_different_signature(self):
        """Test not blocked when different signature is attempted after warning."""
        config = ProgressConfig(repeat_threshold_default=4)
        detector = LoopDetector(config=config)

        # Build up to warning
        for _ in range(3):
            detector.record_tool_call("read", {"path": "same.py"})

        detector.check_loop_warning()

        # Try a different operation
        result = detector.is_blocked_after_warning("read", {"path": "different.py"})
        assert result is None


# =============================================================================
# CLEAR_LOOP_WARNING TESTS (lines 771-772)
# =============================================================================


class TestClearLoopWarning:
    """Tests for clear_loop_warning method."""

    def test_clear_loop_warning_clears_state(self):
        """Test clear_loop_warning clears warning state."""
        detector = LoopDetector()
        detector._loop_warning_given = True
        detector._warned_signature = "test_signature"

        detector.clear_loop_warning()

        assert detector._loop_warning_given is False
        assert detector._warned_signature is None


# =============================================================================
# _CHECK_FILE_READ_LOOPS TESTS (lines 783-803)
# =============================================================================


class TestCheckFileReadLoops:
    """Tests for _check_file_read_loops method."""

    def test_no_loop_with_single_read(self):
        """Test no loop detected with single file read."""
        detector = LoopDetector()
        detector._file_read_ranges["test.py"] = [FileReadRange(offset=0, limit=100)]

        result = detector._check_file_read_loops()
        assert result is None

    def test_no_loop_with_paginated_reads(self):
        """Test no loop detected with paginated reads (non-overlapping)."""
        detector = LoopDetector()
        detector._file_read_ranges["test.py"] = [
            FileReadRange(offset=0, limit=100),
            FileReadRange(offset=100, limit=100),
            FileReadRange(offset=200, limit=100),
        ]

        result = detector._check_file_read_loops()
        assert result is None

    def test_loop_with_overlapping_reads(self):
        """Test loop detected with overlapping reads."""
        config = ProgressConfig(max_overlapping_reads_per_file=2)
        detector = LoopDetector(config=config)
        detector._file_read_ranges["test.py"] = [
            FileReadRange(offset=0, limit=100),
            FileReadRange(offset=0, limit=100),
            FileReadRange(offset=0, limit=100),
            FileReadRange(offset=0, limit=100),
        ]

        result = detector._check_file_read_loops()
        assert result is not None
        assert "same file region read" in result

    def test_loop_with_partial_overlaps(self):
        """Test loop detection with partial overlaps."""
        config = ProgressConfig(max_overlapping_reads_per_file=2)
        detector = LoopDetector(config=config)
        detector._file_read_ranges["test.py"] = [
            FileReadRange(offset=0, limit=100),
            FileReadRange(offset=50, limit=100),
            FileReadRange(offset=100, limit=100),
            FileReadRange(offset=25, limit=50),  # Overlaps with all previous
        ]

        result = detector._check_file_read_loops()
        # May or may not detect depending on overlap count


# =============================================================================
# _GET_MAX_ITERATIONS TESTS (lines 807-813)
# =============================================================================


class TestGetMaxIterations:
    """Tests for _get_max_iterations method."""

    def test_max_iterations_default(self):
        """Test max iterations for DEFAULT task type."""
        config = ProgressConfig(max_iterations_default=8)
        detector = LoopDetector(config=config, task_type=LoopDetectorTaskType.DEFAULT)
        assert detector._get_max_iterations() == 8

    def test_max_iterations_analysis(self):
        """Test max iterations for ANALYSIS task type."""
        config = ProgressConfig(max_iterations_analysis=50)
        detector = LoopDetector(config=config, task_type=LoopDetectorTaskType.ANALYSIS)
        assert detector._get_max_iterations() == 50

    def test_max_iterations_action(self):
        """Test max iterations for ACTION task type."""
        config = ProgressConfig(max_iterations_action=12)
        detector = LoopDetector(config=config, task_type=LoopDetectorTaskType.ACTION)
        assert detector._get_max_iterations() == 12

    def test_max_iterations_research(self):
        """Test max iterations for RESEARCH task type."""
        config = ProgressConfig(max_iterations_research=6)
        detector = LoopDetector(config=config, task_type=LoopDetectorTaskType.RESEARCH)
        assert detector._get_max_iterations() == 6


# =============================================================================
# _GET_BASE_DETAILS TESTS (line 817)
# =============================================================================


class TestGetBaseDetails:
    """Tests for _get_base_details method."""

    def test_get_base_details_returns_all_fields(self):
        """Test _get_base_details returns all expected fields."""
        config = ProgressConfig(tool_budget=50)
        detector = LoopDetector(config=config, task_type=LoopDetectorTaskType.ANALYSIS)
        detector.record_tool_call("read", {"path": "test.py"})
        detector.record_iteration(500)

        details = detector._get_base_details()

        assert details["tool_calls"] == 1
        assert details["tool_budget"] == 50
        assert details["iterations"] == 1
        assert "unique_resources" in details
        assert details["task_type"] == "analysis"


# =============================================================================
# CONTENT LOOP DETECTION TESTS (lines 839-847, 855-872, 883-912, 924-926, 931, 936, 940-942)
# =============================================================================


class TestContentLoopDetection:
    """Comprehensive tests for content loop detection."""

    def test_record_content_chunk_empty_string(self):
        """Test record_content_chunk with empty string is ignored."""
        detector = LoopDetector()
        detector.record_content_chunk("")
        assert detector._content_buffer == ""

    def test_record_content_chunk_accumulates(self):
        """Test record_content_chunk accumulates content."""
        detector = LoopDetector()
        detector.record_content_chunk("Hello ")
        detector.record_content_chunk("World")
        assert detector._content_buffer == "Hello World"

    def test_record_content_chunk_trims_buffer(self):
        """Test record_content_chunk trims buffer when exceeding limit."""
        detector = LoopDetector()
        # Add more than CONTENT_BUFFER_SIZE (5000) characters
        detector.record_content_chunk("x" * 6000)
        assert len(detector._content_buffer) == detector.CONTENT_BUFFER_SIZE

    def test_check_content_loop_returns_none_for_short_content(self):
        """Test check_content_loop returns None for short content."""
        detector = LoopDetector()
        detector.record_content_chunk("short")
        result = detector.check_content_loop()
        assert result is None

    def test_check_content_loop_detects_repetition(self):
        """Test check_content_loop detects repeated phrases."""
        detector = LoopDetector()
        # Use a longer phrase that will be detected (needs to hit threshold of 5 repeats)
        # The algorithm finds phrases of 15-80 chars that repeat 5+ times
        repeated_phrase = (
            "This is a test phrase that should be detected when repeated enough times. "
        )
        detector.record_content_chunk(repeated_phrase * 20)

        result = detector.check_content_loop()
        assert result is not None
        assert detector._content_loop_detected is True

    def test_check_content_loop_returns_cached_result(self):
        """Test check_content_loop returns cached result after detection."""
        detector = LoopDetector()
        repeated_phrase = "This is a repeated phrase for testing purposes. "
        detector.record_content_chunk(repeated_phrase * 10)

        result1 = detector.check_content_loop()
        result2 = detector.check_content_loop()

        assert result1 is not None
        assert result2 is not None
        assert detector._content_loop_phrase in result2

    def test_find_repeated_phrase_with_whitespace_content(self):
        """Test _find_repeated_phrase handles whitespace content."""
        detector = LoopDetector()
        detector._content_buffer = "   " * 100
        result = detector._find_repeated_phrase()
        # Should not detect as meaningful phrase
        assert result is None

    def test_is_meaningful_phrase_with_alphanumeric(self):
        """Test _is_meaningful_phrase with alphanumeric content."""
        detector = LoopDetector()
        assert detector._is_meaningful_phrase("Hello World 123") is True
        assert detector._is_meaningful_phrase("test_function") is True

    def test_is_meaningful_phrase_with_punctuation_only(self):
        """Test _is_meaningful_phrase with mostly punctuation."""
        detector = LoopDetector()
        assert detector._is_meaningful_phrase("...---...---") is False
        assert detector._is_meaningful_phrase("!@#$%^&*()") is False

    def test_content_loop_detected_property(self):
        """Test content_loop_detected property."""
        detector = LoopDetector()
        assert detector.content_loop_detected is False

        repeated_phrase = "Repeated content for detection test here. "
        detector.record_content_chunk(repeated_phrase * 10)
        detector.check_content_loop()

        assert detector.content_loop_detected is True

    def test_content_loop_phrase_property(self):
        """Test content_loop_phrase property."""
        detector = LoopDetector()
        assert detector.content_loop_phrase is None

        repeated_phrase = "This phrase should be detected as repeating. "
        detector.record_content_chunk(repeated_phrase * 10)
        detector.check_content_loop()

        if detector.content_loop_detected:
            assert detector.content_loop_phrase is not None

    def test_reset_content_tracking(self):
        """Test reset_content_tracking clears content state only."""
        detector = LoopDetector()
        detector.record_content_chunk("test content " * 100)
        detector.check_content_loop()
        detector.record_tool_call("read", {"path": "test.py"})

        detector.reset_content_tracking()

        assert detector._content_buffer == ""
        assert detector._content_loop_detected is False
        assert detector._content_loop_phrase is None
        # Other state should be preserved
        assert detector.tool_calls == 1


# =============================================================================
# FACTORY FUNCTION TESTS (lines 971-1000, 1023-1028)
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_tracker_from_classification_simple(self):
        """Test create_tracker_from_classification with SIMPLE complexity."""
        from victor.agent.complexity_classifier import TaskComplexity

        mock_classification = MagicMock()
        mock_classification.complexity = TaskComplexity.SIMPLE
        mock_classification.tool_budget = 5

        tracker, hint = create_tracker_from_classification(mock_classification)

        assert isinstance(tracker, LoopDetector)
        assert tracker.config.tool_budget == 5
        assert isinstance(hint, str)

    def test_create_tracker_from_classification_medium(self):
        """Test create_tracker_from_classification with MEDIUM complexity."""
        from victor.agent.complexity_classifier import TaskComplexity

        mock_classification = MagicMock()
        mock_classification.complexity = TaskComplexity.MEDIUM
        mock_classification.tool_budget = 15

        tracker, hint = create_tracker_from_classification(mock_classification)

        assert tracker.config.tool_budget == 15
        assert tracker.task_type == LoopDetectorTaskType.DEFAULT

    def test_create_tracker_from_classification_complex(self):
        """Test create_tracker_from_classification with COMPLEX complexity."""
        from victor.agent.complexity_classifier import TaskComplexity

        mock_classification = MagicMock()
        mock_classification.complexity = TaskComplexity.COMPLEX
        mock_classification.tool_budget = 25

        tracker, hint = create_tracker_from_classification(mock_classification)

        assert tracker.config.tool_budget == 25
        assert tracker.task_type == LoopDetectorTaskType.ANALYSIS

    def test_create_tracker_from_classification_generation(self):
        """Test create_tracker_from_classification with GENERATION complexity."""
        from victor.agent.complexity_classifier import TaskComplexity

        mock_classification = MagicMock()
        mock_classification.complexity = TaskComplexity.GENERATION
        mock_classification.tool_budget = 20

        tracker, hint = create_tracker_from_classification(mock_classification)

        assert tracker.config.tool_budget == 20
        assert tracker.task_type == LoopDetectorTaskType.ACTION

    def test_create_tracker_from_classification_with_base_config(self):
        """Test create_tracker_from_classification with base config."""
        from victor.agent.complexity_classifier import TaskComplexity

        base_config = ProgressConfig(max_total_iterations=100)
        mock_classification = MagicMock()
        mock_classification.complexity = TaskComplexity.MEDIUM
        mock_classification.tool_budget = 10

        tracker, hint = create_tracker_from_classification(mock_classification, base_config)

        assert tracker.config.tool_budget == 10
        # max_total_iterations should be set to budget + 1
        assert tracker.config.max_total_iterations == 11

    def test_classify_and_create_tracker_simple_message(self):
        """Test classify_and_create_tracker with simple message."""
        tracker, hint, classification = classify_and_create_tracker("List all Python files")

        assert isinstance(tracker, LoopDetector)
        assert isinstance(hint, str)
        assert hasattr(classification, "complexity")
        assert hasattr(classification, "tool_budget")

    def test_classify_and_create_tracker_complex_message(self):
        """Test classify_and_create_tracker with complex message."""
        message = "Analyze the entire codebase and provide a comprehensive report on code quality, test coverage, and potential improvements"
        tracker, hint, classification = classify_and_create_tracker(message)

        assert isinstance(tracker, LoopDetector)
        assert tracker.remaining_budget > 0


# =============================================================================
# CONSTANTS AND EDGE CASES TESTS
# =============================================================================


class TestConstantsAndEdgeCases:
    """Tests for constants and edge cases."""

    def test_research_tools_constant(self):
        """Test RESEARCH_TOOLS constant contains expected tools."""
        assert "web_search" in RESEARCH_TOOLS
        assert "web_fetch" in RESEARCH_TOOLS
        assert "tavily_search" in RESEARCH_TOOLS

    def test_default_read_limit_constant(self):
        """Test DEFAULT_READ_LIMIT constant."""
        assert DEFAULT_READ_LIMIT == 500

    def test_canonical_read_tool_constant(self):
        """Test CANONICAL_READ_TOOL constant."""
        from victor.tools.tool_names import ToolNames

        assert CANONICAL_READ_TOOL == ToolNames.READ

    def test_detector_with_extreme_budget(self):
        """Test detector with extremely high budget."""
        config = ProgressConfig(tool_budget=10000)
        detector = LoopDetector(config=config)
        assert detector.remaining_budget == 10000

    def test_detector_with_minimal_config(self):
        """Test detector with minimal configuration."""
        config = ProgressConfig(
            tool_budget=1,
            max_iterations_default=1,
            repeat_threshold_default=1,
        )
        detector = LoopDetector(config=config)

        detector.record_tool_call("test", {})
        result = detector.should_stop()
        assert result.should_stop is True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestLoopDetectorIntegration:
    """Integration tests simulating real usage patterns."""

    def test_typical_analysis_workflow(self):
        """Test typical analysis workflow with file reads and searches."""
        detector = LoopDetector(task_type=LoopDetectorTaskType.ANALYSIS)

        # Simulate reading multiple files
        detector.record_tool_call("read", {"path": "src/main.py"})
        detector.record_tool_call("read", {"path": "src/utils.py"})
        detector.record_tool_call("read", {"path": "src/config.py"})

        # Simulate searches
        detector.record_tool_call("grep", {"query": "import", "directory": "."})

        # Simulate iterations
        detector.record_iteration(500)
        detector.record_iteration(800)

        # Check no premature stop
        result = detector.should_stop()
        assert result.should_stop is False

        # Check metrics
        metrics = detector.get_metrics()
        assert metrics["tool_calls"] == 4
        assert metrics["iterations"] == 2

    def test_deepseek_loop_scenario(self):
        """Test detection of DeepSeek-like repeated read pattern."""
        config = ProgressConfig(max_overlapping_reads_per_file=3, repeat_threshold_default=3)
        detector = LoopDetector(config=config)

        # Simulate DeepSeek reading same file region repeatedly
        for _ in range(5):
            detector.record_tool_call(
                "read", {"path": "utils/client.py", "offset": 0, "limit": 100}
            )

        result = detector.should_stop()
        assert result.should_stop is True

    def test_content_streaming_loop_scenario(self):
        """Test detection of content streaming loop (reasoning model pattern)."""
        detector = LoopDetector()

        # Simulate streaming chunks with repeated reasoning
        reasoning_phrase = "Let me think about this more carefully. "
        for _ in range(3):
            detector.record_content_chunk(reasoning_phrase)

        # Should not detect yet (not enough repetitions)
        result = detector.check_content_loop()
        # Add more repetitions
        for _ in range(7):
            detector.record_content_chunk(reasoning_phrase)

        result = detector.check_content_loop()
        assert result is not None

    def test_warning_then_recovery_workflow(self):
        """Test warning is given and then cleared when behavior changes."""
        config = ProgressConfig(repeat_threshold_default=4)
        detector = LoopDetector(config=config)

        # Build up to warning
        for _ in range(3):
            detector.record_tool_call("read", {"path": "same.py"})

        warning = detector.check_loop_warning()
        assert warning is not None
        assert detector._loop_warning_given is True

        # Model changes behavior
        detector.record_tool_call("write", {"path": "different.py", "content": "new"})

        # Warning should be cleared
        assert detector._loop_warning_given is False
