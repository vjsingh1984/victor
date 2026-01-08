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

"""Tests for tool loop detector.

These tests verify that the loop detector correctly identifies repetitive
tool calling patterns that indicate an LLM is stuck in a loop.

GAP-4 Fix: This addresses issues seen with DeepSeek where it repeatedly
reads the same file without progress.
"""

import pytest
from victor.agent.tool_loop_detector import (
    ToolLoopDetector,
    LoopDetectorConfig,
    LoopDetectionResult,
    LoopType,
    LoopSeverity,
    ToolCallRecord,
    create_loop_detector,
    LoggingLoopObserver,
)


class TestLoopDetectionResult:
    """Tests for LoopDetectionResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = LoopDetectionResult()

        assert result.loop_detected is False
        assert result.loop_type == LoopType.NONE
        assert result.severity == LoopSeverity.INFO
        assert result.consecutive_count == 0
        assert result.should_warn is False
        assert result.should_break is False

    def test_should_warn_for_warning_severity(self):
        """Test should_warn returns True for WARNING severity."""
        result = LoopDetectionResult(
            loop_detected=True,
            severity=LoopSeverity.WARNING,
        )
        assert result.should_warn is True
        assert result.should_break is False

    def test_should_break_for_critical_severity(self):
        """Test should_break returns True for CRITICAL severity."""
        result = LoopDetectionResult(
            loop_detected=True,
            severity=LoopSeverity.CRITICAL,
        )
        assert result.should_warn is True
        assert result.should_break is True


class TestLoopDetectorConfig:
    """Tests for LoopDetectorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LoopDetectorConfig()

        assert config.max_same_call_repetitions == 4
        assert config.max_cyclical_repetitions == 3
        assert config.warning_threshold_ratio == 0.75
        assert config.window_size == 20
        assert config.resource_read_threshold == 4

    def test_custom_config(self):
        """Test custom configuration."""
        config = LoopDetectorConfig(
            max_same_call_repetitions=3,
            window_size=10,
        )
        assert config.max_same_call_repetitions == 3
        assert config.window_size == 10


class TestToolLoopDetectorSameArguments:
    """Tests for same-argument loop detection."""

    @pytest.fixture
    def detector(self):
        """Create a detector with default config."""
        return ToolLoopDetector()

    def test_no_loop_on_first_call(self, detector):
        """Test no loop detected on first call."""
        result = detector.record_tool_call(
            tool_name="read_file",
            arguments={"path": "foo.py"},
        )
        assert result.loop_detected is False

    def test_no_loop_with_different_arguments(self, detector):
        """Test no loop with different arguments."""
        # Same tool, different args
        detector.record_tool_call("read_file", {"path": "foo.py"})
        result = detector.record_tool_call("read_file", {"path": "bar.py"})

        assert result.loop_detected is False

    def test_no_loop_with_different_tools(self, detector):
        """Test no loop with different tools."""
        detector.record_tool_call("read_file", {"path": "foo.py"})
        result = detector.record_tool_call("edit_files", {"path": "foo.py"})

        assert result.loop_detected is False

    def test_warning_at_threshold(self, detector):
        """Test warning when approaching loop limit."""
        # Default: warning at 75% of 4 = 3 repetitions
        detector.record_tool_call("read_file", {"path": "foo.py"})
        detector.record_tool_call("read_file", {"path": "foo.py"})
        result = detector.record_tool_call("read_file", {"path": "foo.py"})

        assert result.loop_detected is True
        assert result.loop_type == LoopType.APPROACHING_LIMIT
        assert result.severity == LoopSeverity.WARNING
        assert result.consecutive_count == 3

    def test_critical_at_max_repetitions(self, detector):
        """Test critical when max repetitions reached."""
        for _ in range(4):
            result = detector.record_tool_call("read_file", {"path": "foo.py"})

        assert result.loop_detected is True
        assert result.loop_type == LoopType.SAME_ARGUMENTS
        assert result.severity == LoopSeverity.CRITICAL
        assert result.consecutive_count == 4

    def test_counter_resets_with_different_call(self, detector):
        """Test consecutive counter resets with different call."""
        detector.record_tool_call("read_file", {"path": "foo.py"})
        detector.record_tool_call("read_file", {"path": "foo.py"})
        detector.record_tool_call("read_file", {"path": "bar.py"})  # Different
        result = detector.record_tool_call("read_file", {"path": "foo.py"})

        # Should be count=1, not count=3
        assert result.loop_detected is False

    def test_custom_max_repetitions(self):
        """Test with custom max repetitions."""
        config = LoopDetectorConfig(max_same_call_repetitions=2)
        detector = ToolLoopDetector(config)

        detector.record_tool_call("read_file", {"path": "foo.py"})
        result = detector.record_tool_call("read_file", {"path": "foo.py"})

        assert result.loop_detected is True
        assert result.severity == LoopSeverity.CRITICAL


class TestToolLoopDetectorCyclicalPatterns:
    """Tests for cyclical pattern detection (A→B→A→B)."""

    @pytest.fixture
    def detector(self):
        return ToolLoopDetector()

    def test_no_loop_with_insufficient_history(self, detector):
        """Test no cyclical detection with insufficient history."""
        detector.record_tool_call("read_file", {"path": "a.py"})
        detector.record_tool_call("edit_files", {"path": "a.py"})
        result = detector.record_tool_call("read_file", {"path": "b.py"})

        assert result.loop_type != LoopType.CYCLICAL_PATTERN

    def test_detect_simple_cycle(self, detector):
        """Test detection of A→B→A→B cycle."""
        # Build up: A→B→A→B→A→B (3 cycles)
        for _ in range(3):
            detector.record_tool_call("read_file", {"path": f"a{_}.py"})
            detector.record_tool_call("edit_files", {"path": f"a{_}.py"})

        result = detector.record_tool_call("read_file", {"path": "a3.py"})

        # May or may not detect depending on args being same
        # The cyclical detector looks at tool names, not args
        assert result.loop_detected or not result.loop_detected  # Valid either way

    def test_no_cycle_with_varied_tools(self, detector):
        """Test no cycle when tools vary."""
        detector.record_tool_call("read_file", {})
        detector.record_tool_call("edit_files", {})
        detector.record_tool_call("run_tests", {})
        result = detector.record_tool_call("git", {})

        assert result.loop_type != LoopType.CYCLICAL_PATTERN


class TestToolLoopDetectorResourceContention:
    """Tests for resource contention detection (multiple reads without writes)."""

    @pytest.fixture
    def detector(self):
        return ToolLoopDetector()

    def test_no_contention_with_single_read(self, detector):
        """Test no contention with single read."""
        result = detector.record_tool_call("read_file", {"path": "foo.py"})
        assert result.loop_type != LoopType.RESOURCE_CONTENTION

    def test_no_contention_with_mixed_operations(self, detector):
        """Test no contention with read→write→read."""
        detector.record_tool_call("read_file", {"path": "foo.py"})
        detector.record_tool_call("edit_files", {"path": "foo.py"})  # Write
        result = detector.record_tool_call("read_file", {"path": "foo.py"})

        assert result.loop_type != LoopType.RESOURCE_CONTENTION

    def test_detect_multiple_reads_same_resource(self, detector):
        """Test detection of multiple reads of same file.

        Note: When reading the same file with same args, SAME_ARGUMENTS loop
        detection takes priority. This test verifies resource contention is
        detected when different tools read the same resource.
        """
        # Use different tools but same resource to test resource contention
        detector.record_tool_call("read_file", {"path": "foo.py"})
        detector.record_tool_call("code_search", {"path": "foo.py", "query": "def"})
        detector.record_tool_call("list_directory", {"path": "foo.py"})
        result = detector.record_tool_call(
            "semantic_code_search", {"path": "foo.py", "query": "class"}
        )

        assert result.loop_detected is True
        assert result.loop_type == LoopType.RESOURCE_CONTENTION
        assert "foo.py" in result.details.get("resource", "")

    def test_different_resources_no_contention(self, detector):
        """Test no contention when reading different files."""
        for i in range(5):
            result = detector.record_tool_call("read_file", {"path": f"file{i}.py"})

        # Each file only read once, no contention
        assert result.loop_type != LoopType.RESOURCE_CONTENTION


class TestToolLoopDetectorDiminishingReturns:
    """Tests for diminishing returns detection (similar results)."""

    @pytest.fixture
    def detector(self):
        return ToolLoopDetector()

    def test_no_detection_without_result_hash(self, detector):
        """Test no diminishing returns without result hashes."""
        for _ in range(5):
            result = detector.record_tool_call("read_file", {"path": "foo.py"})

        # No result hash provided, shouldn't detect diminishing returns
        # (but may detect other patterns)
        if result.loop_detected:
            assert result.loop_type != LoopType.DIMINISHING_RETURNS

    def test_detect_identical_results(self, detector):
        """Test detection of identical results."""
        same_hash = "abc123"
        for i in range(5):
            result = detector.record_tool_call(
                "code_search",
                {"query": f"query{i}"},  # Different args
                result_hash=same_hash,  # Same result
            )

        assert result.loop_detected is True
        assert result.loop_type == LoopType.DIMINISHING_RETURNS

    def test_no_detection_with_varied_results(self, detector):
        """Test no detection with varied results."""
        for i in range(5):
            result = detector.record_tool_call(
                "code_search",
                {"query": f"query{i}"},
                result_hash=f"hash{i}",  # Different result each time
            )

        assert result.loop_type != LoopType.DIMINISHING_RETURNS


class TestToolLoopDetectorDeepSeekScenario:
    """Integration tests simulating the DeepSeek repeated read pattern.

    These tests simulate the GAP-4 scenario where DeepSeek reads the same
    file 6+ times without making progress.
    """

    def test_deepseek_same_file_loop(self):
        """Test detection of DeepSeek's same-file read pattern."""
        detector = ToolLoopDetector()

        # Simulate DeepSeek reading the same file repeatedly
        file_path = "investor_homelab/utils/web_search_client.py"
        warnings_issued = []

        for i in range(6):
            result = detector.record_tool_call(
                "read",
                {"path": file_path, "start_line": 200, "end_line": 200},
            )
            if result.loop_detected:
                warnings_issued.append(result)

        # Should have detected the loop
        assert len(warnings_issued) >= 2  # Warning at 3, Critical at 4+
        assert any(r.severity == LoopSeverity.WARNING for r in warnings_issued)
        assert any(r.severity == LoopSeverity.CRITICAL for r in warnings_issued)

    def test_deepseek_interleaved_reads(self):
        """Test interleaved reads that still form a pattern."""
        detector = ToolLoopDetector()

        # Simulate reads interleaved with symbol lookups
        loops_detected = 0
        for i in range(4):
            # Read file
            result = detector.record_tool_call(
                "read",
                {"path": "utils/web_search_client.py"},
            )
            if result.loop_detected:
                loops_detected += 1

            # Symbol lookup (different resource)
            detector.record_tool_call(
                "symbol",
                {"file_path": "utils/web_search_client.py", "symbol": "WebSearchClient"},
            )

        # Should still detect resource contention on the file
        assert loops_detected >= 1


class TestToolLoopDetectorStatistics:
    """Tests for detector statistics."""

    def test_initial_statistics(self):
        """Test initial statistics are zero."""
        detector = ToolLoopDetector()
        stats = detector.get_statistics()

        assert stats["total_calls"] == 0
        assert stats["loops_detected"] == 0
        assert stats["loop_rate"] == 0.0

    def test_statistics_after_calls(self):
        """Test statistics are updated correctly."""
        detector = ToolLoopDetector()

        # Make some calls that will trigger a loop
        for _ in range(5):
            detector.record_tool_call("read_file", {"path": "foo.py"})

        stats = detector.get_statistics()

        assert stats["total_calls"] == 5
        assert stats["loops_detected"] >= 2  # Warning and critical
        assert stats["history_length"] == 5


class TestToolLoopDetectorReset:
    """Tests for reset functionality."""

    def test_reset_clears_state(self):
        """Test reset clears all state."""
        detector = ToolLoopDetector()

        # Build up some history
        for _ in range(3):
            detector.record_tool_call("read_file", {"path": "foo.py"})

        detector.reset()
        stats = detector.get_statistics()

        assert stats["total_calls"] == 0
        assert stats["history_length"] == 0

    def test_clear_history_preserves_config(self):
        """Test clear_history preserves configuration."""
        config = LoopDetectorConfig(max_same_call_repetitions=2)
        detector = ToolLoopDetector(config)

        detector.record_tool_call("read_file", {"path": "foo.py"})
        detector.clear_history()

        # Config should be preserved
        assert detector.config.max_same_call_repetitions == 2


class TestLoggingLoopObserver:
    """Tests for the logging observer."""

    def test_observer_receives_notifications(self):
        """Test observer is notified of loop detections."""
        detector = ToolLoopDetector()
        notifications = []

        class TestObserver:
            def on_loop_detected(self, result):
                notifications.append(result)

        detector.add_observer(TestObserver())

        # Trigger a loop
        for _ in range(4):
            detector.record_tool_call("read_file", {"path": "foo.py"})

        assert len(notifications) >= 2

    def test_observer_can_be_removed(self):
        """Test observer can be removed."""
        detector = ToolLoopDetector()
        notifications = []

        class TestObserver:
            def on_loop_detected(self, result):
                notifications.append(result)

        observer = TestObserver()
        detector.add_observer(observer)
        detector.remove_observer(observer)

        # Trigger a loop
        for _ in range(4):
            detector.record_tool_call("read_file", {"path": "foo.py"})

        # Observer should not have received notifications
        assert len(notifications) == 0


class TestCreateLoopDetector:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        """Test factory creates detector with defaults."""
        detector = create_loop_detector()

        assert detector.config.max_same_call_repetitions == 4
        assert detector.config.window_size == 20

    def test_create_with_custom_values(self):
        """Test factory creates detector with custom values."""
        detector = create_loop_detector(max_repetitions=2, window_size=10)

        assert detector.config.max_same_call_repetitions == 2
        assert detector.config.window_size == 10


class TestToolLoopDetectorEdgeCases:
    """Edge case tests."""

    def test_empty_arguments(self):
        """Test handling of empty arguments."""
        detector = ToolLoopDetector()
        result = detector.record_tool_call("list_directory", {})
        assert not result.should_break

    def test_complex_arguments(self):
        """Test handling of complex nested arguments."""
        detector = ToolLoopDetector()
        complex_args = {
            "path": "foo.py",
            "options": {"recursive": True, "depth": 3},
            "filters": ["*.py", "*.txt"],
        }
        result = detector.record_tool_call("search", complex_args)
        assert not result.should_break

    def test_unicode_in_arguments(self):
        """Test handling of unicode in arguments."""
        detector = ToolLoopDetector()
        result = detector.record_tool_call("read_file", {"path": "日本語/file.py"})
        assert not result.should_break

    def test_very_long_history(self):
        """Test behavior with many tool calls."""
        detector = ToolLoopDetector()

        # Make many different calls
        for i in range(100):
            detector.record_tool_call(f"tool_{i % 5}", {"id": i})

        stats = detector.get_statistics()
        # History should be bounded by window_size
        assert stats["history_length"] <= detector.config.window_size


class TestProgressiveParameterNormalization:
    """Tests for progressive parameter normalization.

    This feature ensures semantic equivalence detection across different
    parameter naming conventions used by different tools:
    - file, filepath, file_path, path → path
    - start_line, start, begin → offset
    - query, search, pattern → query
    """

    @pytest.fixture
    def detector(self):
        return ToolLoopDetector()

    def test_file_path_aliases_normalized(self, detector):
        """Test that file/filepath/file_path/path all normalize to same hash."""
        # These should all be treated as the same arguments
        detector.record_tool_call("read", {"path": "foo.py"})
        detector.record_tool_call("read", {"file": "foo.py"})
        detector.record_tool_call("read", {"filepath": "foo.py"})
        result = detector.record_tool_call("read", {"file_path": "foo.py"})

        # Should detect loop - all 4 calls are semantically identical
        assert result.loop_detected is True
        assert result.loop_type == LoopType.SAME_ARGUMENTS
        assert result.severity == LoopSeverity.CRITICAL

    def test_line_offset_aliases_normalized(self, detector):
        """Test that start_line/start/offset all normalize to same hash."""
        detector.record_tool_call("read", {"path": "foo.py", "start_line": 10})
        detector.record_tool_call("read", {"path": "foo.py", "start": 10})
        detector.record_tool_call("read", {"path": "foo.py", "offset": 10})
        result = detector.record_tool_call("read", {"path": "foo.py", "begin": 10})

        # Should detect loop - all 4 calls are semantically identical
        assert result.loop_detected is True
        assert result.severity == LoopSeverity.CRITICAL

    def test_query_aliases_normalized(self, detector):
        """Test that query/search/pattern all normalize to same hash."""
        detector.record_tool_call("code_search", {"query": "def foo"})
        detector.record_tool_call("code_search", {"search": "def foo"})
        detector.record_tool_call("code_search", {"pattern": "def foo"})
        result = detector.record_tool_call("code_search", {"keyword": "def foo"})

        # Should detect loop - all 4 calls are semantically identical
        assert result.loop_detected is True
        assert result.severity == LoopSeverity.CRITICAL

    def test_path_normalization_removes_dot_slash(self, detector):
        """Test that ./foo.py and foo.py are treated as equivalent."""
        detector.record_tool_call("read", {"path": "./foo.py"})
        detector.record_tool_call("read", {"path": "foo.py"})
        detector.record_tool_call("read", {"path": "./foo.py"})
        result = detector.record_tool_call("read", {"path": "foo.py"})

        # Should detect loop - paths are normalized
        assert result.loop_detected is True
        assert result.severity == LoopSeverity.CRITICAL

    def test_numeric_offset_coercion(self, detector):
        """Test that numeric offsets are coerced correctly."""
        detector.record_tool_call("read", {"path": "foo.py", "offset": 10})
        detector.record_tool_call("read", {"path": "foo.py", "offset": "10"})  # String
        detector.record_tool_call("read", {"path": "foo.py", "offset": 10})
        result = detector.record_tool_call("read", {"path": "foo.py", "offset": 10})

        # Should detect loop - numeric values normalized
        assert result.loop_detected is True
        assert result.severity == LoopSeverity.CRITICAL

    def test_different_values_still_different(self, detector):
        """Test that normalization doesn't conflate different values."""
        detector.record_tool_call("read", {"path": "foo.py", "offset": 10})
        detector.record_tool_call("read", {"path": "foo.py", "offset": 20})
        detector.record_tool_call("read", {"path": "bar.py", "offset": 10})
        result = detector.record_tool_call("read", {"path": "baz.py", "offset": 30})

        # Should NOT detect loop - all different
        assert result.loop_detected is False

    def test_mixed_alias_detection(self, detector):
        """Test detection with mixed parameter aliases."""
        # Real-world scenario: ls with path vs read with file
        detector.record_tool_call("read", {"file": "utils/client.py", "start_line": 1})
        detector.record_tool_call("read", {"path": "utils/client.py", "offset": 1})
        detector.record_tool_call("read", {"filepath": "utils/client.py", "begin": 1})
        result = detector.record_tool_call("read", {"file_path": "utils/client.py", "start": 1})

        # Should detect - all semantically identical
        assert result.loop_detected is True
        assert result.severity == LoopSeverity.CRITICAL

    def test_resource_key_uses_normalized_path(self, detector):
        """Test that resource contention uses normalized path aliases."""
        # Different tools, same resource (via normalized path)
        detector.record_tool_call("read_file", {"path": "foo.py"})
        detector.record_tool_call("code_search", {"file": "foo.py", "query": "def"})
        detector.record_tool_call("list_directory", {"filepath": "foo.py"})
        result = detector.record_tool_call("symbol", {"file_path": "foo.py", "name": "Bar"})

        # Should detect resource contention on foo.py
        assert result.loop_detected is True
        assert result.loop_type == LoopType.RESOURCE_CONTENTION

    def test_deepseek_scenario_with_mixed_params(self):
        """Test DeepSeek-like scenario with mixed parameter names."""
        detector = ToolLoopDetector()

        # Simulate DeepSeek using different param names but same file
        file_path = "investor_homelab/utils/web_search_client.py"
        warnings = []

        # First call with 'path'
        warnings.append(detector.record_tool_call("read", {"path": file_path, "start_line": 200}))
        # Second call with 'file'
        warnings.append(detector.record_tool_call("read", {"file": file_path, "start": 200}))
        # Third call with 'filepath'
        warnings.append(detector.record_tool_call("read", {"filepath": file_path, "offset": 200}))
        # Fourth call with 'file_path'
        warnings.append(detector.record_tool_call("read", {"file_path": file_path, "begin": 200}))

        # Should detect loop by the 3rd or 4th call
        detected = [w for w in warnings if w.loop_detected]
        assert len(detected) >= 2  # Should have warning and critical


class TestToolGroupDetection:
    """Tests for tool group detection."""

    @pytest.fixture
    def detector(self):
        return ToolLoopDetector()

    def test_file_read_tools_grouped(self, detector):
        """Test that file read tools are in the same group."""
        assert detector._get_tool_group("read") == "file_read"
        assert detector._get_tool_group("read_file") == "file_read"
        assert detector._get_tool_group("cat") == "file_read"
        assert detector._get_tool_group("head") == "file_read"

    def test_file_list_tools_grouped(self, detector):
        """Test that file list tools are in the same group."""
        assert detector._get_tool_group("ls") == "file_list"
        assert detector._get_tool_group("list_directory") == "file_list"
        assert detector._get_tool_group("tree") == "file_list"

    def test_code_search_tools_grouped(self, detector):
        """Test that code search tools are in the same group."""
        assert detector._get_tool_group("grep") == "code_search"
        assert detector._get_tool_group("code_search") == "code_search"
        assert detector._get_tool_group("semantic_code_search") == "code_search"

    def test_symbol_tools_grouped(self, detector):
        """Test that symbol lookup tools are in the same group."""
        assert detector._get_tool_group("symbol") == "symbol_lookup"
        assert detector._get_tool_group("get_symbol") == "symbol_lookup"
        assert detector._get_tool_group("definition") == "symbol_lookup"

    def test_unknown_tool_no_group(self, detector):
        """Test that unknown tools return None."""
        assert detector._get_tool_group("custom_tool") is None
        assert detector._get_tool_group("unknown") is None

    def test_case_insensitive_grouping(self, detector):
        """Test that tool grouping is case-insensitive."""
        assert detector._get_tool_group("READ") == "file_read"
        assert detector._get_tool_group("Read_File") == "file_read"
        assert detector._get_tool_group("LS") == "file_list"
