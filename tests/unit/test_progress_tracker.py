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

"""TDD tests for unified LoopDetector class.

This class consolidates all loop detection and progress tracking mechanisms:
- Tool call counting and budget enforcement
- True loop detection (repeated signatures)
- Progress tracking (unique files read)
- Task-type aware thresholds (analysis vs action vs default)
- Research loop detection
"""

import pytest
from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum


class LoopDetectorTaskType(Enum):
    """Task type for configuring progress thresholds."""

    DEFAULT = "default"
    ANALYSIS = "analysis"
    ACTION = "action"
    RESEARCH = "research"


@dataclass
class ProgressConfig:
    """Configuration for progress tracking thresholds."""

    # Tool budget
    tool_budget: int = 50

    # Iteration limits by task type
    max_iterations_default: int = 8
    max_iterations_analysis: int = 50
    max_iterations_action: int = 12
    max_iterations_research: int = 6

    # Loop detection
    repeat_threshold_default: int = 3
    repeat_threshold_analysis: int = 5
    signature_history_size: int = 10

    # Progress detection
    min_content_threshold: int = 150

    # Hard limits
    max_total_iterations: int = 20


@dataclass
class LoopStopRecommendation:
    """Reason why progress tracker recommends stopping."""

    should_stop: bool
    reason: str
    details: Dict[str, Any]


# Import the class we're testing (will fail initially - TDD red phase)
# from victor.agent.loop_detector import LoopDetector, LoopDetectorTaskType, ProgressConfig, LoopStopRecommendation


class TestLoopDetectorInitialization:
    """Tests for LoopDetector initialization."""

    def test_default_initialization(self):
        """Test LoopDetector initializes with default config."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()
        assert tracker.tool_calls == 0
        assert tracker.iterations == 0
        assert len(tracker.unique_resources) == 0
        assert not tracker.should_stop().should_stop

    def test_custom_config(self):
        """Test LoopDetector accepts custom configuration."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig, LoopDetectorTaskType

        config = ProgressConfig(tool_budget=100, max_iterations_analysis=100)
        tracker = LoopDetector(config=config, task_type=LoopDetectorTaskType.ANALYSIS)

        assert tracker.config.tool_budget == 100
        assert tracker.task_type == LoopDetectorTaskType.ANALYSIS

    def test_reset(self):
        """Test tracker can be reset for new conversation turn."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()
        tracker.record_tool_call("read_file", {"path": "test.py"})
        tracker.record_iteration(content_length=50)

        assert tracker.tool_calls > 0

        tracker.reset()

        assert tracker.tool_calls == 0
        assert tracker.iterations == 0


class TestToolCallTracking:
    """Tests for tool call counting and budget."""

    def test_record_tool_call_increments_count(self):
        """Test that recording a tool call increments the count."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()
        tracker.record_tool_call("read_file", {"path": "test.py"})

        assert tracker.tool_calls == 1

    def test_tool_budget_exceeded(self):
        """Test that exceeding tool budget triggers stop."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig

        config = ProgressConfig(tool_budget=3)
        tracker = LoopDetector(config=config)

        tracker.record_tool_call("read_file", {"path": "a.py"})
        tracker.record_tool_call("read_file", {"path": "b.py"})
        assert not tracker.should_stop().should_stop

        tracker.record_tool_call("read_file", {"path": "c.py"})
        result = tracker.should_stop()
        assert result.should_stop
        assert "budget" in result.reason.lower()

    def test_remaining_budget(self):
        """Test remaining budget calculation."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig

        config = ProgressConfig(tool_budget=10)
        tracker = LoopDetector(config=config)

        assert tracker.remaining_budget == 10

        tracker.record_tool_call("test", {})
        tracker.record_tool_call("test", {})

        assert tracker.remaining_budget == 8


class TestLoopDetection:
    """Tests for true loop detection (repeated signatures)."""

    def test_no_loop_with_different_calls(self):
        """Test that different tool calls don't trigger loop detection."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()

        tracker.record_tool_call("read_file", {"path": "a.py"})
        tracker.record_tool_call("read_file", {"path": "b.py"})
        tracker.record_tool_call("list_directory", {"path": "src/"})

        result = tracker.should_stop()
        assert not result.should_stop

    def test_loop_detected_same_signature_repeated(self):
        """Test that repeating exact same call triggers loop detection."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig

        config = ProgressConfig(repeat_threshold_default=3)
        tracker = LoopDetector(config=config)

        # Same exact call 3 times should trigger loop
        for _ in range(3):
            tracker.record_tool_call("read_file", {"path": "same.py"})

        result = tracker.should_stop()
        assert result.should_stop
        assert "loop" in result.reason.lower()

    def test_analysis_task_more_lenient_loop_detection(self):
        """Test that analysis tasks allow more repeated calls before loop detection."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig, LoopDetectorTaskType

        config = ProgressConfig(repeat_threshold_default=3, repeat_threshold_analysis=5)
        tracker = LoopDetector(config=config, task_type=LoopDetectorTaskType.ANALYSIS)

        # 3 repeated calls should NOT trigger loop for analysis task
        for _ in range(3):
            tracker.record_tool_call("read_file", {"path": "same.py"})

        result = tracker.should_stop()
        assert not result.should_stop

        # But 5 should
        for _ in range(2):
            tracker.record_tool_call("read_file", {"path": "same.py"})

        result = tracker.should_stop()
        assert result.should_stop

    def test_progressive_tool_with_different_offsets_not_loop(self):
        """Test that reading same file with different offsets is NOT a loop."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()

        tracker.record_tool_call("read_file", {"path": "large.py", "offset": 0})
        tracker.record_tool_call("read_file", {"path": "large.py", "offset": 100})
        tracker.record_tool_call("read_file", {"path": "large.py", "offset": 200})

        result = tracker.should_stop()
        assert not result.should_stop


class TestProgressTracking:
    """Tests for unique resource tracking and progress detection."""

    def test_unique_files_tracked(self):
        """Test that unique files are tracked correctly."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()

        tracker.record_tool_call("read_file", {"path": "a.py"})
        tracker.record_tool_call("read_file", {"path": "b.py"})
        tracker.record_tool_call("read_file", {"path": "a.py"})  # duplicate

        assert len(tracker.unique_resources) == 2

    def test_file_with_offset_tracked_separately(self):
        """Test that same file with different offsets counts as different resources."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()

        tracker.record_tool_call("read_file", {"path": "big.py", "offset": 0})
        tracker.record_tool_call("read_file", {"path": "big.py", "offset": 100})

        assert len(tracker.unique_resources) == 2

    def test_directories_tracked(self):
        """Test that directory listings are tracked."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()

        tracker.record_tool_call("list_directory", {"path": "src/"})
        tracker.record_tool_call("list_directory", {"path": "tests/"})

        assert len(tracker.unique_resources) == 2


class TestIterationTracking:
    """Tests for iteration counting and low-output detection."""

    def test_iteration_increments(self):
        """Test that recording iteration increments count."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()
        tracker.record_iteration(content_length=100)

        assert tracker.iterations == 1

    def test_low_output_iterations_tracked(self):
        """Test that low-output iterations are tracked separately."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig

        config = ProgressConfig(min_content_threshold=150)
        tracker = LoopDetector(config=config)

        tracker.record_iteration(content_length=50)  # low output
        tracker.record_iteration(content_length=200)  # good output
        tracker.record_iteration(content_length=30)  # low output

        assert tracker.low_output_iterations == 2
        assert tracker.iterations == 3

    def test_max_iterations_exceeded(self):
        """Test that exceeding max iterations triggers stop."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig

        config = ProgressConfig(max_iterations_default=3, max_total_iterations=5)
        tracker = LoopDetector(config=config)

        for _ in range(5):
            tracker.record_iteration(content_length=50)

        result = tracker.should_stop()
        assert result.should_stop
        assert "iteration" in result.reason.lower()

    def test_analysis_task_higher_iteration_limit(self):
        """Test that analysis tasks have higher iteration limits."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig, LoopDetectorTaskType

        config = ProgressConfig(max_iterations_default=3, max_iterations_analysis=10)
        tracker = LoopDetector(config=config, task_type=LoopDetectorTaskType.ANALYSIS)

        for _ in range(5):
            tracker.record_iteration(content_length=50)

        # Should not stop yet for analysis task
        result = tracker.should_stop()
        assert not result.should_stop


class TestResearchLoopDetection:
    """Tests for research-specific loop detection."""

    def test_consecutive_research_calls_tracked(self):
        """Test that consecutive research calls are tracked."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig, LoopDetectorTaskType

        config = ProgressConfig(max_iterations_research=3)
        tracker = LoopDetector(config=config, task_type=LoopDetectorTaskType.RESEARCH)

        tracker.record_tool_call("web_search", {"query": "test"})
        tracker.record_tool_call("web_fetch", {"url": "http://test.com"})
        tracker.record_tool_call("web_search", {"query": "another"})

        result = tracker.should_stop()
        assert result.should_stop
        assert "research" in result.reason.lower()

    def test_research_counter_resets_on_non_research_call(self):
        """Test that research counter resets when non-research tool is called."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig, LoopDetectorTaskType

        config = ProgressConfig(max_iterations_research=3)
        tracker = LoopDetector(config=config, task_type=LoopDetectorTaskType.RESEARCH)

        tracker.record_tool_call("web_search", {"query": "test1"})
        tracker.record_tool_call("web_search", {"query": "test2"})
        tracker.record_tool_call("read_file", {"path": "test.py"})  # resets
        tracker.record_tool_call("web_search", {"query": "test3"})

        result = tracker.should_stop()
        assert not result.should_stop


class TestForcedCompletion:
    """Tests for forced completion scenarios."""

    def test_force_completion_manual(self):
        """Test that force_completion can be set manually."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()

        assert not tracker.should_stop().should_stop

        tracker.force_stop("Manual stop requested")

        result = tracker.should_stop()
        assert result.should_stop
        assert "manual" in result.reason.lower()

    def test_stop_reason_includes_details(self):
        """Test that stop reason includes useful details."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig

        config = ProgressConfig(tool_budget=2)
        tracker = LoopDetector(config=config)

        tracker.record_tool_call("test1", {})
        tracker.record_tool_call("test2", {})

        result = tracker.should_stop()

        assert "tool_calls" in result.details
        assert "tool_budget" in result.details


class TestSignatureGeneration:
    """Tests for tool call signature generation."""

    def test_same_args_same_signature(self):
        """Test that same tool+args produce same signature."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()

        sig1 = tracker._get_signature("read_file", {"path": "test.py"})
        sig2 = tracker._get_signature("read_file", {"path": "test.py"})

        assert sig1 == sig2

    def test_different_args_different_signature(self):
        """Test that different args produce different signatures."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()

        sig1 = tracker._get_signature("read_file", {"path": "a.py"})
        sig2 = tracker._get_signature("read_file", {"path": "b.py"})

        assert sig1 != sig2

    def test_progressive_params_included_in_signature(self):
        """Test that progressive params (offset, limit) are included in signature."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()

        sig1 = tracker._get_signature("read_file", {"path": "test.py", "offset": 0})
        sig2 = tracker._get_signature("read_file", {"path": "test.py", "offset": 100})

        assert sig1 != sig2


class TestProgressMetrics:
    """Tests for progress metrics and statistics."""

    def test_get_metrics(self):
        """Test that progress metrics can be retrieved."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()

        tracker.record_tool_call("read_file", {"path": "a.py"})
        tracker.record_tool_call("read_file", {"path": "b.py"})
        tracker.record_iteration(content_length=100)

        metrics = tracker.get_metrics()

        assert metrics["tool_calls"] == 2
        assert metrics["iterations"] == 1
        assert metrics["unique_resources"] == 2

    def test_progress_percentage(self):
        """Test progress percentage calculation."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig

        config = ProgressConfig(tool_budget=10)
        tracker = LoopDetector(config=config)

        tracker.record_tool_call("test", {})
        tracker.record_tool_call("test2", {})

        assert tracker.progress_percentage == 20.0  # 2/10 * 100


class TestSameFileLoopDetection:
    """Tests for same-file loop detection (Gap 2 fix)."""

    def test_same_file_different_offsets_triggers_loop(self):
        """Test that reading same file with different offsets triggers loop after threshold."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig

        config = ProgressConfig(max_overlapping_reads_per_file=3)
        tracker = LoopDetector(config=config)

        # Read same file with different offsets
        tracker.record_tool_call("read_file", {"path": "big.py", "offset": 0})
        tracker.record_tool_call("read_file", {"path": "big.py", "offset": 100})
        tracker.record_tool_call("read_file", {"path": "big.py", "offset": 200})

        # Should NOT trigger yet (count == 3, threshold is > 3)
        result = tracker.should_stop()
        assert not result.should_stop

        # 4th read of same file - should trigger
        tracker.record_tool_call("read_file", {"path": "big.py", "offset": 300})
        result = tracker.should_stop()
        assert result.should_stop
        assert "same file region read" in result.reason.lower()

    def test_different_files_do_not_trigger_same_file_loop(self):
        """Test that reading different files doesn't trigger same-file loop."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig

        config = ProgressConfig(max_overlapping_reads_per_file=2)
        tracker = LoopDetector(config=config)

        # Read different files
        tracker.record_tool_call("read_file", {"path": "a.py"})
        tracker.record_tool_call("read_file", {"path": "b.py"})
        tracker.record_tool_call("read_file", {"path": "c.py"})
        tracker.record_tool_call("read_file", {"path": "d.py"})

        result = tracker.should_stop()
        assert not result.should_stop

    def test_similar_search_queries_trigger_loop(self):
        """Test that similar search queries trigger loop after threshold."""
        from victor.agent.loop_detector import LoopDetector, ProgressConfig

        config = ProgressConfig(max_searches_per_query_prefix=2)
        tracker = LoopDetector(config=config)

        # Same query prefix (first 20 chars = "class BaseTool in th")
        tracker.record_tool_call(
            "code_search", {"query": "class BaseTool in the project", "directory": "."}
        )
        tracker.record_tool_call(
            "code_search", {"query": "class BaseTool in the codebase", "directory": "."}
        )

        # Should NOT trigger yet (count == 2, threshold is > 2)
        result = tracker.should_stop()
        assert not result.should_stop

        # 3rd similar search - should trigger
        tracker.record_tool_call(
            "code_search", {"query": "class BaseTool in the repo", "directory": "."}
        )
        result = tracker.should_stop()
        assert result.should_stop
        assert "similar search" in result.reason.lower()

    def test_unique_resources_still_tracked_with_offset(self):
        """Test that unique resources are still tracked with offset for coverage."""
        from victor.agent.loop_detector import LoopDetector

        tracker = LoopDetector()

        tracker.record_tool_call("read_file", {"path": "big.py", "offset": 0})
        tracker.record_tool_call("read_file", {"path": "big.py", "offset": 100})
        tracker.record_tool_call("read_file", {"path": "big.py", "offset": 200})

        # Should have 3 unique resources (different offsets)
        assert len(tracker.unique_resources) == 3


class TestTaskClassifierIntegration:
    """Tests for task classifier integration with progress tracker.

    These tests interact with ComplexityClassifier and LoopDetectorTaskTypeClassifier.
    We reset the singleton instances before each test to ensure isolation.
    """

    @pytest.fixture(autouse=True)
    def reset_classifier_singletons(self):
        """Reset singleton classifiers before each test for isolation."""
        # Reset LoopDetectorTaskTypeClassifier singleton (used by ComplexityClassifier)
        try:
            from victor.storage.embeddings.task_classifier import LoopDetectorTaskTypeClassifier

            LoopDetectorTaskTypeClassifier.reset_instance()
        except ImportError:
            pass  # LoopDetectorTaskTypeClassifier not available

        yield

        # Reset again after test
        try:
            from victor.storage.embeddings.task_classifier import LoopDetectorTaskTypeClassifier

            LoopDetectorTaskTypeClassifier.reset_instance()
        except ImportError:
            pass

    def test_create_tracker_from_simple_classification(self):
        """Test creating tracker from SIMPLE task classification."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity
        from victor.agent.loop_detector import create_tracker_from_classification

        classifier = ComplexityClassifier()
        classification = classifier.classify("git status")

        tracker, hint = create_tracker_from_classification(classification)

        assert classification.complexity == TaskComplexity.SIMPLE
        assert tracker.config.tool_budget == 10  # Updated minimum budget
        assert tracker.config.max_total_iterations == 11  # budget + 1
        assert "SIMPLE" in hint

    def test_create_tracker_from_complex_classification(self):
        """Test creating tracker from COMPLEX task classification."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity
        from victor.agent.loop_detector import (
            create_tracker_from_classification,
            LoopDetectorTaskType,
        )

        classifier = ComplexityClassifier()
        classification = classifier.classify("generate a new feature")

        tracker, hint = create_tracker_from_classification(classification)

        assert classification.complexity == TaskComplexity.COMPLEX
        assert tracker.config.tool_budget == 25  # Updated budget
        assert tracker.task_type == LoopDetectorTaskType.ANALYSIS
        assert "COMPLEX" in hint

    def test_create_tracker_from_generation_classification(self):
        """Test creating tracker from GENERATION task classification."""
        from victor.agent.complexity_classifier import ComplexityClassifier, TaskComplexity
        from victor.agent.loop_detector import (
            create_tracker_from_classification,
            LoopDetectorTaskType,
        )

        classifier = ComplexityClassifier()
        classification = classifier.classify("write a function to add numbers")

        tracker, hint = create_tracker_from_classification(classification)

        assert classification.complexity == TaskComplexity.GENERATION
        # GENERATION tasks have minimum budget of 10 (may need file reads)
        assert tracker.config.tool_budget == 10
        assert tracker.config.max_total_iterations == 11  # budget + 1
        assert tracker.task_type == LoopDetectorTaskType.ACTION
        assert "GENERATE" in hint

    def test_classify_and_create_tracker_convenience(self):
        """Test the convenience function classify_and_create_tracker."""
        from victor.agent.loop_detector import classify_and_create_tracker
        from victor.agent.complexity_classifier import TaskComplexity

        tracker, hint, classification = classify_and_create_tracker("git status")

        assert classification.complexity == TaskComplexity.SIMPLE
        assert tracker.remaining_budget == 10  # Updated minimum budget
        assert "SIMPLE" in hint

    def test_tracker_enforces_task_budget(self):
        """Test that tracker correctly enforces task-specific budgets."""
        from victor.agent.loop_detector import classify_and_create_tracker

        tracker, _, _ = classify_and_create_tracker("git log")

        # Should have budget of 10 for simple task (minimum)
        assert tracker.remaining_budget == 10

        # Use up the budget
        for i in range(10):
            tracker.record_tool_call("list_directory", {"path": f"path_{i}"})

        assert tracker.remaining_budget == 0

        # Should now trigger stop
        result = tracker.should_stop()
        assert result.should_stop
        assert "budget" in result.reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
