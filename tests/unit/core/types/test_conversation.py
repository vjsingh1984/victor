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

"""Tests for core conversation types.

This test suite validates the canonical conversation stage enum
that has been moved to victor.core.state to enforce layer boundaries.
"""

from __future__ import annotations

import pytest

from victor.core.state import ConversationStage


# =============================================================================
# ConversationStage Enum Tests
# =============================================================================


class TestConversationStage:
    """Test the ConversationStage enum."""

    def test_all_stages_defined(self):
        """All expected stages should be defined."""
        expected_stages = [
            "INITIAL",
            "PLANNING",
            "READING",
            "ANALYSIS",
            "EXECUTION",
            "VERIFICATION",
            "COMPLETION",
        ]

        for stage_name in expected_stages:
            assert hasattr(ConversationStage, stage_name), f"Missing stage: {stage_name}"

    def test_stage_values_are_strings(self):
        """Stage values should be strings for serialization compatibility."""
        assert issubclass(ConversationStage, str)

    def test_stage_values_lowercase(self):
        """Stage values should be lowercase strings."""
        assert ConversationStage.INITIAL.value == "initial"
        assert ConversationStage.PLANNING.value == "planning"
        assert ConversationStage.READING.value == "reading"
        assert ConversationStage.ANALYSIS.value == "analysis"
        assert ConversationStage.EXECUTION.value == "execution"
        assert ConversationStage.VERIFICATION.value == "verification"
        assert ConversationStage.COMPLETION.value == "completion"

    def test_stage_iteration(self):
        """Should be able to iterate over all stages."""
        stages = list(ConversationStage)
        assert len(stages) == 7
        assert ConversationStage.INITIAL in stages
        assert ConversationStage.COMPLETION in stages

    def test_stage_comparison(self):
        """Stages should be comparable."""
        assert ConversationStage.INITIAL == ConversationStage.INITIAL
        assert ConversationStage.INITIAL != ConversationStage.PLANNING

    def test_stage_string_representation(self):
        """Stages should have nice string representations."""
        assert str(ConversationStage.INITIAL) == "initial"
        assert repr(ConversationStage.INITIAL) == "<ConversationStage.initial: 'initial'>"

    def test_stage_from_string(self):
        """Should be able to get stage from string value."""
        stage = ConversationStage("initial")
        assert stage == ConversationStage.INITIAL

        stage = ConversationStage("execution")
        assert stage == ConversationStage.EXECUTION

    def test_stage_ordering(self):
        """Stages should have a natural order.

        This test verifies that stages follow the expected workflow:
        INITIAL -> PLANNING -> READING -> ANALYSIS -> EXECUTION -> VERIFICATION -> COMPLETION
        """
        stages = list(ConversationStage)
        expected_order = [
            ConversationStage.INITIAL,
            ConversationStage.PLANNING,
            ConversationStage.READING,
            ConversationStage.ANALYSIS,
            ConversationStage.EXECUTION,
            ConversationStage.VERIFICATION,
            ConversationStage.COMPLETION,
        ]

        # Note: Enum order is preserved in Python 3.7+
        # This test verifies the enum is defined in the right order
        assert stages == expected_order

    def test_stage_membership(self):
        """Should be able to check if a value is a valid stage."""
        assert "initial" in [s.value for s in ConversationStage]
        assert "invalid_stage" not in [s.value for s in ConversationStage]

    def test_stage_serialization(self):
        """Stages should be JSON-serializable via their string values."""
        import json

        stage = ConversationStage.EXECUTION

        # Should serialize to string
        serialized = json.dumps({"stage": stage.value})
        assert serialized == '{"stage": "execution"}'

        # Should deserialize back to stage
        data = json.loads(serialized)
        deserialized = ConversationStage(data["stage"])
        assert deserialized == ConversationStage.EXECUTION


# =============================================================================
# Stage Workflow Tests
# =============================================================================


class TestStageWorkflow:
    """Test typical conversation stage workflows."""

    def test_typical_coding_workflow(self):
        """Test typical workflow for a coding task."""
        workflow = [
            ConversationStage.INITIAL,
            ConversationStage.PLANNING,
            ConversationStage.READING,
            ConversationStage.ANALYSIS,
            ConversationStage.EXECUTION,
            ConversationStage.VERIFICATION,
            ConversationStage.COMPLETION,
        ]

        # All stages should be valid
        for stage in workflow:
            assert isinstance(stage, ConversationStage)

        # Workflow should progress forward (no backward transitions)
        for i in range(len(workflow) - 1):
            current = workflow[i]
            next_stage = workflow[i + 1]
            # Just verify they're different stages
            assert current != next_stage

    def test_exploratory_workflow(self):
        """Test workflow for exploratory tasks (lots of reading)."""
        workflow = [
            ConversationStage.INITIAL,
            ConversationStage.READING,
            ConversationStage.READING,  # More reading
            ConversationStage.ANALYSIS,
            ConversationStage.READING,  # Back to reading
            ConversationStage.ANALYSIS,
            ConversationStage.COMPLETION,  # May skip execution
        ]

        # All stages should be valid
        for stage in workflow:
            assert isinstance(stage, ConversationStage)

    def test_debugging_workflow(self):
        """Test workflow for debugging tasks."""
        workflow = [
            ConversationStage.INITIAL,
            ConversationStage.READING,
            ConversationStage.ANALYSIS,
            ConversationStage.EXECUTION,
            ConversationStage.VERIFICATION,
            ConversationStage.ANALYSIS,  # Back to analysis
            ConversationStage.EXECUTION,
            ConversationStage.VERIFICATION,
            ConversationStage.COMPLETION,
        ]

        # All stages should be valid
        for stage in workflow:
            assert isinstance(stage, ConversationStage)


# =============================================================================
# Stage Metadata Tests
# =============================================================================


class TestStageMetadata:
    """Test stage metadata and properties."""

    def test_stage_names_are_uppercase(self):
        """Stage names should be uppercase (Python enum convention)."""
        for stage in ConversationStage:
            name = stage.name
            assert name.isupper(), f"Stage {name} is not uppercase"
            assert name.isidentifier(), f"Stage {name} is not a valid identifier"

    def test_stage_values_are_lowercase(self):
        """Stage values should be lowercase for API consistency."""
        for stage in ConversationStage:
            value = stage.value
            assert value.islower(), f"Stage value {value} is not lowercase"
            assert value == value.lower(), f"Stage value {value} is not lowercase"

    def test_stage_value_matches_name_lowercased(self):
        """Stage value should be the lowercase version of the name."""
        for stage in ConversationStage:
            assert stage.value == stage.name.lower(), f"Stage value {stage.value} != {stage.name.lower()}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestConversationStageIntegration:
    """Test ConversationStage integration with other components."""

    def test_import_from_core(self):
        """Should be importable from victor.core.state."""
        from victor.core.state import ConversationStage as CoreStage

        assert CoreStage is ConversationStage

    def test_framework_stage_alias(self):
        """Framework's Stage should be an alias to ConversationStage."""
        from victor.core.state import ConversationStage
        from victor.framework.state import Stage

        assert Stage is ConversationStage

    def test_stage_in_annotations(self):
        """ConversationStage should work in type annotations."""
        from typing import get_type_hints

        def get_current_stage() -> ConversationStage:
            return ConversationStage.READING

        hints = get_type_hints(get_current_stage)
        assert "return" in hints
        assert hints["return"] == ConversationStage


# =============================================================================
# Edge Cases
# =============================================================================


class TestConversationStageEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_stage_value(self):
        """Creating stage with invalid value should raise ValueError."""
        with pytest.raises(ValueError):
            ConversationStage("invalid_stage")

    def test_stage_hashability(self):
        """Stages should be hashable for use in sets and dicts."""
        stage_set = {
            ConversationStage.INITIAL,
            ConversationStage.PLANNING,
            ConversationStage.EXECUTION,
        }

        assert ConversationStage.INITIAL in stage_set
        assert ConversationStage.READING not in stage_set

        stage_dict = {ConversationStage.EXECUTION: "Executing"}
        assert stage_dict[ConversationStage.EXECUTION] == "Executing"

    def test_stage_immutability(self):
        """Stage enum should be immutable."""
        with pytest.raises(AttributeError):
            ConversationStage.INITIAL = "some_value"


# =============================================================================
# Performance Tests
# =============================================================================


class TestConversationStagePerformance:
    """Test performance characteristics of ConversationStage."""

    def test_stage_lookup_speed(self):
        """Stage lookups should be fast."""
        import time

        iterations = 10000
        start = time.time()

        for _ in range(iterations):
            stage = ConversationStage.EXECUTION
            _ = stage.value
            _ = stage.name

        elapsed = time.time() - start
        avg_time = elapsed / iterations

        # Should be very fast (< 1ms per lookup)
        assert avg_time < 0.001, f"Stage lookup too slow: {avg_time:.6f}s per iteration"

    def test_stage_comparison_speed(self):
        """Stage comparisons should be fast."""
        import time

        iterations = 10000
        start = time.time()

        for _ in range(iterations):
            _ = ConversationStage.EXECUTION == ConversationStage.EXECUTION
            _ = ConversationStage.EXECUTION != ConversationStage.READING

        elapsed = time.time() - start
        avg_time = elapsed / iterations

        # Should be very fast (< 1ms per comparison)
        assert avg_time < 0.001, f"Stage comparison too slow: {avg_time:.6f}s per iteration"
