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

"""Tests for the loop_detector module.

Tests loop detection and budget enforcement:
- Tool call counting and budget enforcement
- Loop detection (repeated signatures)
- Progress tracking (unique resources)
- Task-type aware thresholds
- Research loop detection
- Integration with complexity classifier
"""


from victor.agent.loop_detector import (
    TaskType,
    ProgressConfig,
    StopReason,
    LoopDetector,
    FileReadRange,
    create_tracker_from_classification,
    classify_and_create_tracker,
    RESEARCH_TOOLS,
    get_progress_params_for_tool,
)


class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_type_values(self):
        """Test all task type enum values exist."""
        assert TaskType.DEFAULT.value == "default"
        assert TaskType.ANALYSIS.value == "analysis"
        assert TaskType.ACTION.value == "action"
        assert TaskType.RESEARCH.value == "research"


class TestProgressConfig:
    """Tests for ProgressConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProgressConfig()

        assert config.tool_budget == 50
        assert config.max_iterations_default == 8
        assert config.max_iterations_analysis == 50
        assert config.max_iterations_action == 12
        assert config.max_iterations_research == 6
        assert config.repeat_threshold_default == 3
        assert config.max_overlapping_reads_per_file == 3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProgressConfig(
            tool_budget=100,
            max_iterations_default=20,
            max_overlapping_reads_per_file=5,
        )

        assert config.tool_budget == 100
        assert config.max_iterations_default == 20
        assert config.max_overlapping_reads_per_file == 5


class TestFileReadRange:
    """Tests for FileReadRange dataclass."""

    def test_file_read_range_creation(self):
        """Test creating a FileReadRange."""
        range_ = FileReadRange(offset=0, limit=500)

        assert range_.offset == 0
        assert range_.limit == 500
        assert range_.end == 500

    def test_file_read_range_end_property(self):
        """Test end property calculation."""
        range_ = FileReadRange(offset=100, limit=200)

        assert range_.end == 300

    def test_file_read_ranges_overlap_exact(self):
        """Test overlapping ranges with exact same range."""
        range1 = FileReadRange(offset=0, limit=500)
        range2 = FileReadRange(offset=0, limit=500)

        assert range1.overlaps(range2)
        assert range2.overlaps(range1)

    def test_file_read_ranges_overlap_partial(self):
        """Test partially overlapping ranges."""
        range1 = FileReadRange(offset=0, limit=500)  # 0-500
        range2 = FileReadRange(offset=400, limit=500)  # 400-900

        assert range1.overlaps(range2)
        assert range2.overlaps(range1)

    def test_file_read_ranges_no_overlap_adjacent(self):
        """Test adjacent ranges don't overlap."""
        range1 = FileReadRange(offset=0, limit=500)  # 0-500
        range2 = FileReadRange(offset=500, limit=500)  # 500-1000

        assert not range1.overlaps(range2)
        assert not range2.overlaps(range1)

    def test_file_read_ranges_no_overlap_gap(self):
        """Test ranges with gap don't overlap."""
        range1 = FileReadRange(offset=0, limit=100)  # 0-100
        range2 = FileReadRange(offset=500, limit=100)  # 500-600

        assert not range1.overlaps(range2)
        assert not range2.overlaps(range1)

    def test_file_read_range_equality(self):
        """Test FileReadRange equality."""
        range1 = FileReadRange(offset=0, limit=500)
        range2 = FileReadRange(offset=0, limit=500)
        range3 = FileReadRange(offset=0, limit=600)

        assert range1 == range2
        assert range1 != range3

    def test_file_read_range_hash(self):
        """Test FileReadRange can be hashed."""
        range1 = FileReadRange(offset=0, limit=500)
        range2 = FileReadRange(offset=0, limit=500)

        # Same values should have same hash
        assert hash(range1) == hash(range2)

        # Can be used in sets
        range_set = {range1, range2}
        assert len(range_set) == 1


class TestStopReason:
    """Tests for StopReason dataclass."""

    def test_stop_reason_creation(self):
        """Test creating a StopReason."""
        reason = StopReason(
            should_stop=True,
            reason="Budget exceeded",
            details={"tool_calls": 50, "budget": 50},
        )

        assert reason.should_stop is True
        assert reason.reason == "Budget exceeded"
        assert reason.details["tool_calls"] == 50

    def test_stop_reason_default_details(self):
        """Test StopReason with default empty details."""
        reason = StopReason(should_stop=False, reason="")

        assert reason.details == {}


class TestLoopDetector:
    """Tests for the LoopDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = LoopDetector()

        assert detector.tool_calls == 0
        assert detector.iterations == 0
        assert detector.remaining_budget == 50

    def test_custom_config(self):
        """Test detector with custom config."""
        config = ProgressConfig(tool_budget=10)
        detector = LoopDetector(config=config)

        assert detector.remaining_budget == 10

    def test_task_type_setting(self):
        """Test setting task type."""
        detector = LoopDetector(task_type=TaskType.ANALYSIS)

        metrics = detector.get_metrics()
        assert metrics["task_type"] == "analysis"

    def test_record_tool_call(self):
        """Test recording tool calls."""
        detector = LoopDetector()

        detector.record_tool_call("read_file", {"path": "test.py"})

        assert detector.tool_calls == 1
        assert "file:test.py:0" in detector.unique_resources

    def test_record_tool_call_with_offset(self):
        """Test recording read_file with offset."""
        detector = LoopDetector()

        detector.record_tool_call("read_file", {"path": "test.py", "offset": 100})

        assert "file:test.py:100" in detector.unique_resources

    def test_record_iteration(self):
        """Test recording iterations."""
        detector = LoopDetector()

        detector.record_iteration(content_length=500)

        assert detector.iterations == 1
        assert detector.low_output_iterations == 0

    def test_record_low_output_iteration(self):
        """Test recording low output iterations."""
        detector = LoopDetector()

        detector.record_iteration(content_length=50)  # Below threshold

        assert detector.low_output_iterations == 1

    def test_reset(self):
        """Test resetting detector state."""
        detector = LoopDetector()

        detector.record_tool_call("read_file", {"path": "test.py"})
        detector.record_iteration(content_length=500)

        detector.reset()

        assert detector.tool_calls == 0
        assert detector.iterations == 0
        assert len(detector.unique_resources) == 0

    def test_remaining_budget_calculation(self):
        """Test remaining budget calculation."""
        config = ProgressConfig(tool_budget=10)
        detector = LoopDetector(config=config)

        detector.record_tool_call("read_file", {"path": "test.py"})
        detector.record_tool_call("read_file", {"path": "test2.py"})

        assert detector.remaining_budget == 8

    def test_progress_percentage(self):
        """Test progress percentage calculation."""
        config = ProgressConfig(tool_budget=10)
        detector = LoopDetector(config=config)

        for i in range(5):
            detector.record_tool_call("read_file", {"path": f"test{i}.py"})

        assert detector.progress_percentage == 50.0

    def test_force_stop(self):
        """Test manual force stop."""
        detector = LoopDetector()

        detector.force_stop("User cancelled")

        result = detector.should_stop()
        assert result.should_stop is True
        assert "Manual stop" in result.reason

    def test_should_stop_budget_exceeded(self):
        """Test should_stop when budget exceeded."""
        config = ProgressConfig(tool_budget=3)
        detector = LoopDetector(config=config)

        for i in range(4):
            detector.record_tool_call("read_file", {"path": f"test{i}.py"})

        result = detector.should_stop()
        assert result.should_stop is True
        assert "budget" in result.reason.lower()

    def test_should_stop_max_iterations(self):
        """Test should_stop when max iterations reached."""
        config = ProgressConfig(max_total_iterations=3)
        detector = LoopDetector(config=config)

        for _i in range(4):
            detector.record_iteration(content_length=500)

        result = detector.should_stop()
        assert result.should_stop is True
        assert "iteration" in result.reason.lower()

    def test_should_stop_loop_detected(self):
        """Test should_stop when loop detected (overlapping reads to same region)."""
        config = ProgressConfig(repeat_threshold_default=2, max_overlapping_reads_per_file=2)
        detector = LoopDetector(config=config)

        # Read same file region multiple times (overlapping offsets trigger loop)
        # Using same offset=0 means all reads overlap
        for _i in range(3):
            detector.record_tool_call("read_file", {"path": "test.py", "offset": 0})

        result = detector.should_stop()
        assert result.should_stop is True
        assert "loop" in result.reason.lower() or "same file" in result.reason.lower()

    def test_should_stop_research_loop(self):
        """Test should_stop when research loop detected."""
        config = ProgressConfig(max_iterations_research=3)
        detector = LoopDetector(config=config, task_type=TaskType.RESEARCH)

        # Consecutive research calls
        for i in range(4):
            detector.record_tool_call("web_search", {"query": f"query{i}"})

        result = detector.should_stop()
        assert result.should_stop is True
        assert "research" in result.reason.lower()

    def test_no_stop_within_budget(self):
        """Test that detector doesn't stop within budget."""
        config = ProgressConfig(tool_budget=10)
        detector = LoopDetector(config=config)

        for i in range(5):
            detector.record_tool_call("read_file", {"path": f"test{i}.py"})

        result = detector.should_stop()
        assert result.should_stop is False

    def test_get_metrics(self):
        """Test get_metrics returns correct data."""
        detector = LoopDetector(task_type=TaskType.ANALYSIS)

        detector.record_tool_call("read_file", {"path": "test.py"})
        detector.record_iteration(content_length=500)

        metrics = detector.get_metrics()

        assert metrics["tool_calls"] == 1
        assert metrics["iterations"] == 1
        assert metrics["unique_resources"] == 1
        assert metrics["task_type"] == "analysis"


class TestResourceTracking:
    """Tests for resource tracking."""

    def test_read_file_resource_key(self):
        """Test read_file generates correct resource key."""
        detector = LoopDetector()

        detector.record_tool_call("read_file", {"path": "test.py", "offset": 50})

        assert "file:test.py:50" in detector.unique_resources

    def test_list_directory_resource_key(self):
        """Test list_directory generates correct resource key."""
        detector = LoopDetector()

        detector.record_tool_call("list_directory", {"path": "/src"})

        assert "dir:/src" in detector.unique_resources

    def test_code_search_resource_key(self):
        """Test code_search generates correct resource key."""
        detector = LoopDetector()

        detector.record_tool_call("code_search", {"query": "def main", "directory": "."})

        resources = detector.unique_resources
        assert any("search" in r for r in resources)

    def test_different_offsets_count_as_different_resources(self):
        """Test that different offsets count as different resources."""
        detector = LoopDetector()

        detector.record_tool_call("read_file", {"path": "test.py", "offset": 0})
        detector.record_tool_call("read_file", {"path": "test.py", "offset": 100})

        assert len(detector.unique_resources) == 2


class TestLoopDetection:
    """Tests for loop detection algorithms."""

    def test_same_file_loop_detection_with_overlapping_offsets(self):
        """Test detection of same file region being read too many times (overlapping ranges)."""
        config = ProgressConfig(max_overlapping_reads_per_file=2)
        detector = LoopDetector(config=config)

        # Read same file region 3 times with overlapping offsets (default limit=500)
        # All reads at offset=0 overlap with each other
        for _i in range(3):
            detector.record_tool_call("read_file", {"path": "test.py", "offset": 0})

        result = detector.should_stop()
        assert result.should_stop is True
        assert "same file region" in result.reason.lower()

    def test_paginated_file_reads_allowed(self):
        """Test that paginated reads to different file regions are allowed."""
        config = ProgressConfig(max_overlapping_reads_per_file=2)
        detector = LoopDetector(config=config)

        # Read different non-overlapping regions (offset=0, 500, 1000 with default limit=500)
        # These should NOT trigger a loop
        for i in range(5):
            detector.record_tool_call("read_file", {"path": "test.py", "offset": i * 500})

        result = detector.should_stop()
        # Should not stop - different regions being read
        assert result.should_stop is False or "same file region" not in result.reason.lower()

    def test_similar_search_loop_detection(self):
        """Test detection of similar searches being repeated."""
        config = ProgressConfig(max_searches_per_query_prefix=2)
        detector = LoopDetector(config=config)

        # Similar searches with same 20-char prefix: "def main_function_te" (exactly 20 chars)
        # The detector uses query[:20] for similarity detection
        detector.record_tool_call("code_search", {"query": "def main_function_testA"})
        detector.record_tool_call("code_search", {"query": "def main_function_testB"})
        detector.record_tool_call("code_search", {"query": "def main_function_testC"})

        result = detector.should_stop()
        assert result.should_stop is True

    def test_signature_based_loop_detection(self):
        """Test detection based on repeated signatures."""
        config = ProgressConfig(repeat_threshold_default=2)
        detector = LoopDetector(config=config)

        # Exact same call repeated
        for _i in range(4):
            detector.record_tool_call("execute_bash", {"command": "ls -la"})

        result = detector.should_stop()
        assert result.should_stop is True

    def test_progressive_params_prevent_false_loops(self):
        """Test that different params don't trigger false loops."""
        config = ProgressConfig(repeat_threshold_default=2, max_overlapping_reads_per_file=10)
        detector = LoopDetector(config=config)

        # Different files - should not be detected as loop
        for i in range(5):
            detector.record_tool_call("read_file", {"path": f"file{i}.py"})

        result = detector.should_stop()
        # Should not stop due to loop (different files)
        assert result.should_stop is False or "loop" not in result.reason.lower()


class TestResearchTracking:
    """Tests for research-specific tracking."""

    def test_research_tools_constant(self):
        """Test RESEARCH_TOOLS contains expected tools."""
        assert "web_search" in RESEARCH_TOOLS
        assert "web_fetch" in RESEARCH_TOOLS

    def test_consecutive_research_tracking(self):
        """Test tracking of consecutive research calls."""
        detector = LoopDetector(task_type=TaskType.RESEARCH)

        detector.record_tool_call("web_search", {"query": "test"})
        detector.record_tool_call("web_fetch", {"url": "http://example.com"})

        # Count should be 2
        assert detector._consecutive_research_calls == 2

    def test_non_research_resets_count(self):
        """Test that non-research tool resets consecutive count."""
        detector = LoopDetector(task_type=TaskType.RESEARCH)

        detector.record_tool_call("web_search", {"query": "test"})
        detector.record_tool_call("read_file", {"path": "test.py"})  # Non-research

        assert detector._consecutive_research_calls == 0


class TestProgressiveParams:
    """Tests for progressive params via registry."""

    @staticmethod
    def _ensure_tools_loaded():
        """Force import of tool modules to populate the registry."""
        # Import tools to trigger decorator registration
        import victor.tools.filesystem  # noqa: F401
        import victor.tools.code_search_tool  # noqa: F401

    def test_get_progress_params_for_tool(self):
        """Test get_progress_params_for_tool returns expected params from registry.

        Note: Tools are registered by their function names (read, ls, search),
        not the external-facing names (read_file, list_directory).
        """
        self._ensure_tools_loaded()
        # 'read' function in filesystem.py has path and offset as progress params
        read_params = get_progress_params_for_tool("read")
        assert "path" in read_params
        assert "offset" in read_params

    def test_progress_params_for_search_tools(self):
        """Test search tools have query-related progress params."""
        self._ensure_tools_loaded()
        # The code_search function in code_search_tool.py is registered as 'grep'
        # due to TOOL_ALIASES resolving code_search -> grep (keyword search)
        search_params = get_progress_params_for_tool("grep")
        assert len(search_params) > 0
        assert "query" in search_params

    def test_progress_params_for_unknown_tool(self):
        """Test unknown tool returns empty list."""
        params = get_progress_params_for_tool("nonexistent_tool")
        assert params == []


class TestIntegration:
    """Tests for integration with complexity classifier."""

    def test_create_tracker_from_classification(self):
        """Test creating tracker from TaskClassification."""
        from victor.agent.complexity_classifier import TaskClassification, TaskComplexity

        classification = TaskClassification(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=2,
            prompt_hint="Simple hint",
            confidence=0.9,
            matched_patterns=["list_files"],
        )

        tracker, hint = create_tracker_from_classification(classification)

        assert isinstance(tracker, LoopDetector)
        assert tracker.config.tool_budget == 2
        assert "Simple" in hint

    def test_classify_and_create_tracker(self):
        """Test convenience function for classifying and creating tracker."""
        tracker, hint, classification = classify_and_create_tracker("list all files")

        assert isinstance(tracker, LoopDetector)
        assert len(hint) > 0
        assert classification is not None

    def test_simple_task_has_low_budget(self):
        """Test that simple tasks get low tool budgets."""
        tracker, hint, classification = classify_and_create_tracker("list files")

        # Simple tasks get budgets in the 2-10 range (depends on classifier)
        assert tracker.config.tool_budget <= 10

    def test_complex_task_has_high_budget(self):
        """Test that complex tasks get high tool budgets."""
        tracker, hint, classification = classify_and_create_tracker("analyze the entire codebase")

        assert tracker.config.tool_budget >= 10


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_budget(self):
        """Test behavior with zero budget."""
        config = ProgressConfig(tool_budget=0)
        detector = LoopDetector(config=config)

        detector.record_tool_call("read_file", {"path": "test.py"})

        result = detector.should_stop()
        assert result.should_stop is True

    def test_empty_arguments(self):
        """Test recording tool call with empty arguments."""
        detector = LoopDetector()

        detector.record_tool_call("read_file", {})

        assert detector.tool_calls == 1

    def test_unknown_tool(self):
        """Test recording unknown tool calls."""
        detector = LoopDetector()

        detector.record_tool_call("unknown_tool", {"arg": "value"})

        assert detector.tool_calls == 1

    def test_negative_content_length(self):
        """Test recording iteration with negative content length."""
        detector = LoopDetector()

        detector.record_iteration(content_length=-1)

        assert detector.iterations == 1
        assert detector.low_output_iterations == 1

    def test_large_tool_budget(self):
        """Test with very large tool budget."""
        config = ProgressConfig(tool_budget=10000)
        detector = LoopDetector(config=config)

        for i in range(100):
            detector.record_tool_call("read_file", {"path": f"file{i}.py"})

        result = detector.should_stop()
        assert result.should_stop is False

    def test_unique_resources_is_copy(self):
        """Test that unique_resources property returns a copy."""
        detector = LoopDetector()

        detector.record_tool_call("read_file", {"path": "test.py"})

        resources1 = detector.unique_resources
        resources2 = detector.unique_resources

        # Should be equal but not the same object
        assert resources1 == resources2
        assert resources1 is not resources2
