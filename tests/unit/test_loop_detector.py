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

"""Tests for loop detection and budget enforcement."""

import pytest
from unittest.mock import MagicMock

from victor.agent.loop_detector import (
    TaskType,
    FileReadRange,
    ProgressConfig,
    StopReason,
    LoopDetector,
    RESEARCH_TOOLS,
    get_progress_params_for_tool,
    is_progressive_tool,
    create_tracker_from_classification,
    classify_and_create_tracker,
)


# =============================================================================
# TASK TYPE TESTS
# =============================================================================


class TestTaskType:
    """Tests for TaskType enum."""

    def test_enum_values(self):
        """Test all expected enum values exist."""
        assert TaskType.DEFAULT.value == "default"
        assert TaskType.ANALYSIS.value == "analysis"
        assert TaskType.ACTION.value == "action"
        assert TaskType.RESEARCH.value == "research"


# =============================================================================
# FILE READ RANGE TESTS
# =============================================================================


class TestFileReadRange:
    """Tests for FileReadRange dataclass."""

    def test_creation(self):
        """Test basic range creation."""
        range_obj = FileReadRange(offset=0, limit=100)
        assert range_obj.offset == 0
        assert range_obj.limit == 100

    def test_end_property(self):
        """Test end property calculation."""
        range_obj = FileReadRange(offset=50, limit=100)
        assert range_obj.end == 150

    def test_overlaps_true_same_range(self):
        """Test overlaps returns True for identical ranges."""
        r1 = FileReadRange(offset=0, limit=100)
        r2 = FileReadRange(offset=0, limit=100)
        assert r1.overlaps(r2)

    def test_overlaps_true_partial(self):
        """Test overlaps returns True for partial overlap."""
        r1 = FileReadRange(offset=0, limit=100)
        r2 = FileReadRange(offset=50, limit=100)
        assert r1.overlaps(r2)
        assert r2.overlaps(r1)

    def test_overlaps_false_adjacent(self):
        """Test overlaps returns False for adjacent ranges."""
        r1 = FileReadRange(offset=0, limit=100)
        r2 = FileReadRange(offset=100, limit=100)
        assert not r1.overlaps(r2)
        assert not r2.overlaps(r1)

    def test_overlaps_false_separate(self):
        """Test overlaps returns False for separate ranges."""
        r1 = FileReadRange(offset=0, limit=100)
        r2 = FileReadRange(offset=200, limit=100)
        assert not r1.overlaps(r2)
        assert not r2.overlaps(r1)

    def test_hash_and_equality(self):
        """Test hash and equality."""
        r1 = FileReadRange(offset=0, limit=100)
        r2 = FileReadRange(offset=0, limit=100)
        r3 = FileReadRange(offset=0, limit=200)

        assert r1 == r2
        assert hash(r1) == hash(r2)
        assert r1 != r3

    def test_equality_with_non_range(self):
        """Test equality with non-FileReadRange."""
        r1 = FileReadRange(offset=0, limit=100)
        assert r1 != "not a range"
        assert r1 != 100


# =============================================================================
# PROGRESS CONFIG TESTS
# =============================================================================


class TestProgressConfig:
    """Tests for ProgressConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProgressConfig()
        assert config.tool_budget == 50
        assert config.max_iterations_default == 8
        assert config.max_iterations_analysis == 50
        assert config.max_iterations_action == 12
        assert config.max_iterations_research == 6
        assert config.repeat_threshold_default == 3
        assert config.signature_history_size == 10
        assert config.max_total_iterations == 50

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ProgressConfig(
            tool_budget=100,
            max_iterations_default=20,
            repeat_threshold_default=5,
        )
        assert config.tool_budget == 100
        assert config.max_iterations_default == 20
        assert config.repeat_threshold_default == 5


# =============================================================================
# STOP REASON TESTS
# =============================================================================


class TestStopReason:
    """Tests for StopReason dataclass."""

    def test_creation(self):
        """Test basic stop reason creation."""
        reason = StopReason(should_stop=True, reason="Budget exceeded")
        assert reason.should_stop is True
        assert reason.reason == "Budget exceeded"
        assert reason.details == {}
        assert reason.is_warning is False

    def test_creation_with_details(self):
        """Test stop reason with details."""
        reason = StopReason(
            should_stop=False,
            reason="",
            details={"tool_calls": 5},
            is_warning=True,
        )
        assert reason.should_stop is False
        assert reason.details["tool_calls"] == 5
        assert reason.is_warning is True


# =============================================================================
# LOOP DETECTOR TESTS
# =============================================================================


class TestLoopDetector:
    """Tests for LoopDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a default detector."""
        return LoopDetector()

    def test_init_default(self, detector):
        """Test default initialization."""
        assert detector.task_type == TaskType.DEFAULT
        assert detector.tool_calls == 0
        assert detector.iterations == 0
        assert detector.remaining_budget == 50

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = ProgressConfig(tool_budget=100)
        detector = LoopDetector(config=config)
        assert detector.remaining_budget == 100

    def test_reset(self, detector):
        """Test state reset."""
        detector.record_tool_call("read", {"path": "test.py"})
        detector.record_iteration(100)
        detector._forced_stop = "test"

        detector.reset()

        assert detector.tool_calls == 0
        assert detector.iterations == 0
        assert detector._forced_stop is None


class TestRecordToolCall:
    """Tests for record_tool_call method."""

    @pytest.fixture
    def detector(self):
        return LoopDetector()

    def test_increments_tool_calls(self, detector):
        """Test tool calls are incremented."""
        detector.record_tool_call("read", {"path": "test.py"})
        assert detector.tool_calls == 1

    def test_tracks_research_calls(self, detector):
        """Test research calls are tracked."""
        detector.record_tool_call("web_search", {"query": "test"})
        assert detector._consecutive_research_calls == 1


class TestRecordIteration:
    """Tests for record_iteration method."""

    @pytest.fixture
    def detector(self):
        return LoopDetector()

    def test_increments_iterations(self, detector):
        """Test iterations are incremented."""
        detector.record_iteration(500)
        assert detector.iterations == 1

    def test_tracks_low_output(self, detector):
        """Test low output iterations are tracked."""
        detector.record_iteration(50)
        assert detector.low_output_iterations == 1


class TestShouldStop:
    """Tests for should_stop method."""

    def test_should_not_stop_initially(self):
        """Test should not stop initially."""
        detector = LoopDetector()
        result = detector.should_stop()
        assert result.should_stop is False

    def test_stops_on_budget_exceeded(self):
        """Test stops when budget exceeded."""
        config = ProgressConfig(tool_budget=3)
        detector = LoopDetector(config=config)

        for i in range(4):
            detector.record_tool_call(f"tool_{i}", {})

        result = detector.should_stop()
        assert result.should_stop is True
        assert "Tool budget exceeded" in result.reason

    def test_stops_on_forced_stop(self):
        """Test stops on forced stop."""
        detector = LoopDetector()
        detector.force_stop("User cancelled")

        result = detector.should_stop()
        assert result.should_stop is True
        assert "Manual stop" in result.reason


class TestLoopDetection:
    """Tests for loop detection."""

    def test_detects_repeated_signature(self):
        """Test detects repeated signatures."""
        config = ProgressConfig(repeat_threshold_default=3)
        detector = LoopDetector(config=config)

        for _ in range(4):
            detector.record_tool_call("read", {"path": "test.py"})

        result = detector.should_stop()
        assert result.should_stop is True


class TestFileReadLoopDetection:
    """Tests for file read loop detection."""

    def test_detects_overlapping_reads(self):
        """Test detects overlapping reads."""
        config = ProgressConfig(max_overlapping_reads_per_file=2)
        detector = LoopDetector(config=config)

        for _ in range(3):
            detector.record_tool_call("read", {"path": "test.py", "offset": 0, "limit": 100})

        result = detector.should_stop()
        assert result.should_stop is True

    def test_allows_paginated_reads(self):
        """Test allows paginated reads."""
        config = ProgressConfig(max_overlapping_reads_per_file=2)
        detector = LoopDetector(config=config)

        detector.record_tool_call("read", {"path": "test.py", "offset": 0, "limit": 100})
        detector.record_tool_call("read", {"path": "test.py", "offset": 100, "limit": 100})
        detector.record_tool_call("read", {"path": "test.py", "offset": 200, "limit": 100})

        result = detector.should_stop()
        assert result.should_stop is False


class TestContentLoopDetection:
    """Tests for content loop detection."""

    def test_record_content_chunk(self):
        """Test recording content chunks."""
        detector = LoopDetector()
        detector.record_content_chunk("Hello world")
        assert len(detector._content_buffer) == 11

    def test_detects_repeated_phrase(self):
        """Test detects repeated phrases."""
        detector = LoopDetector()
        phrase = "Let me think about this carefully. "
        content = phrase * 10

        detector.record_content_chunk(content)
        result = detector.check_content_loop()

        assert result is not None

    def test_reset_content_tracking(self):
        """Test reset_content_tracking method."""
        detector = LoopDetector()
        detector.record_content_chunk("test" * 1000)

        detector.reset_content_tracking()

        assert detector._content_buffer == ""


class TestGetMetrics:
    """Tests for get_metrics method."""

    def test_returns_all_metrics(self):
        """Test returns all expected metrics."""
        detector = LoopDetector(task_type=TaskType.ANALYSIS)
        detector.record_tool_call("read", {"path": "test.py"})

        metrics = detector.get_metrics()

        assert "tool_calls" in metrics
        assert "iterations" in metrics
        assert "task_type" in metrics


class TestProperties:
    """Tests for detector properties."""

    def test_remaining_budget(self):
        """Test remaining_budget property."""
        config = ProgressConfig(tool_budget=10)
        detector = LoopDetector(config=config)

        assert detector.remaining_budget == 10

        detector.record_tool_call("test", {})
        assert detector.remaining_budget == 9

    def test_progress_percentage(self):
        """Test progress_percentage property."""
        config = ProgressConfig(tool_budget=10)
        detector = LoopDetector(config=config)

        for _ in range(5):
            detector.record_tool_call("test", {})

        assert detector.progress_percentage == 50.0


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_progress_params_for_tool(self):
        """Test get_progress_params_for_tool function."""
        params = get_progress_params_for_tool("read_file")
        assert isinstance(params, list)

    def test_is_progressive_tool(self):
        """Test is_progressive_tool function."""
        result = is_progressive_tool("some_tool")
        assert isinstance(result, bool)


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_tracker_from_classification(self):
        """Test create_tracker_from_classification."""
        from victor.agent.complexity_classifier import TaskComplexity

        mock_classification = MagicMock()
        mock_classification.complexity = TaskComplexity.MEDIUM
        mock_classification.tool_budget = 15
        mock_classification.prompt_hint = "Medium task hint"

        tracker, hint = create_tracker_from_classification(mock_classification)

        assert isinstance(tracker, LoopDetector)
        assert tracker.config.tool_budget == 15

    def test_classify_and_create_tracker(self):
        """Test classify_and_create_tracker function."""
        tracker, hint, classification = classify_and_create_tracker("List all Python files")

        assert isinstance(tracker, LoopDetector)
        assert isinstance(hint, str)


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_arguments(self):
        """Test handling empty arguments."""
        detector = LoopDetector()
        detector.record_tool_call("test", {})
        assert detector.tool_calls == 1

    def test_research_tools_constant(self):
        """Test RESEARCH_TOOLS constant."""
        assert "web_search" in RESEARCH_TOOLS
        assert isinstance(RESEARCH_TOOLS, frozenset)
