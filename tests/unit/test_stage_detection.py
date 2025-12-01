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

"""Tests for conversation stage detection."""

import pytest

from victor.agent.conversation_state import (
    ConversationStage,
    ConversationState,
    ConversationStateMachine,
    STAGE_TOOL_MAPPING,
    STAGE_KEYWORDS,
)


class TestConversationStage:
    """Tests for ConversationStage enum."""

    def test_all_stages_defined(self):
        """Test that all expected stages are defined."""
        stages = list(ConversationStage)
        assert ConversationStage.INITIAL in stages
        assert ConversationStage.PLANNING in stages
        assert ConversationStage.READING in stages
        assert ConversationStage.ANALYSIS in stages
        assert ConversationStage.EXECUTION in stages
        assert ConversationStage.VERIFICATION in stages
        assert ConversationStage.COMPLETION in stages

    def test_stages_have_tool_mappings(self):
        """Test that all stages have tool mappings."""
        for stage in ConversationStage:
            assert stage in STAGE_TOOL_MAPPING, f"Missing tool mapping for {stage.name}"
            assert isinstance(STAGE_TOOL_MAPPING[stage], set)

    def test_stages_have_keyword_mappings(self):
        """Test that all stages have keyword mappings."""
        for stage in ConversationStage:
            assert stage in STAGE_KEYWORDS, f"Missing keyword mapping for {stage.name}"
            assert isinstance(STAGE_KEYWORDS[stage], list)


class TestConversationState:
    """Tests for ConversationState dataclass."""

    def test_initial_state(self):
        """Test initial state values."""
        state = ConversationState()
        assert state.stage == ConversationStage.INITIAL
        assert state.tool_history == []
        assert state.observed_files == set()
        assert state.modified_files == set()
        assert state.message_count == 0
        assert state.last_tools == []

    def test_record_tool_execution(self):
        """Test recording tool execution."""
        state = ConversationState()
        state.record_tool_execution("read_file", {"file_path": "/test/file.py"})

        assert "read_file" in state.tool_history
        assert "read_file" in state.last_tools
        assert "/test/file.py" in state.observed_files

    def test_record_tool_execution_modified_file(self):
        """Test recording tool that modifies files."""
        state = ConversationState()
        state.record_tool_execution("write_file", {"file_path": "/test/output.py"})

        assert "write_file" in state.tool_history
        assert "/test/output.py" in state.modified_files

    def test_last_tools_limit(self):
        """Test that last_tools maintains limit of 5."""
        state = ConversationState()

        for i in range(10):
            state.record_tool_execution(f"tool_{i}", {})

        assert len(state.last_tools) == 5
        assert state.last_tools == ["tool_5", "tool_6", "tool_7", "tool_8", "tool_9"]

    def test_record_message(self):
        """Test recording messages."""
        state = ConversationState()
        assert state.message_count == 0

        state.record_message()
        assert state.message_count == 1

        state.record_message()
        assert state.message_count == 2


class TestConversationStateMachine:
    """Tests for ConversationStateMachine."""

    def test_initial_state(self):
        """Test machine starts in INITIAL state."""
        machine = ConversationStateMachine()
        assert machine.get_stage() == ConversationStage.INITIAL

    def test_reset(self):
        """Test reset clears all state."""
        machine = ConversationStateMachine()
        machine.record_tool_execution("read_file", {"file_path": "/test.py"})
        machine.record_message("test message")

        machine.reset()

        assert machine.get_stage() == ConversationStage.INITIAL
        assert machine.state.tool_history == []
        assert machine.state.message_count == 0

    def test_detect_stage_from_planning_keywords(self):
        """Test stage detection from planning keywords."""
        machine = ConversationStateMachine()
        machine.record_message("Can you plan the approach for this feature?")

        # Should detect PLANNING from keywords
        assert machine.get_stage() in [
            ConversationStage.PLANNING,
            ConversationStage.INITIAL,
        ]

    def test_detect_stage_from_execution_keywords(self):
        """Test stage detection from execution keywords."""
        machine = ConversationStateMachine()
        machine.record_message("Please implement the changes and fix the bug")

        # Should detect EXECUTION stage
        assert machine.get_stage() == ConversationStage.EXECUTION

    def test_detect_stage_from_tool_execution(self):
        """Test stage detection from tool patterns."""
        machine = ConversationStateMachine()

        # Execute reading tools
        machine.record_tool_execution("read_file", {"file_path": "/test.py"})
        machine.record_tool_execution("code_search", {"query": "test"})

        # Should be in READING stage
        assert machine.get_stage() in [
            ConversationStage.READING,
            ConversationStage.PLANNING,
            ConversationStage.INITIAL,
        ]

    def test_get_stage_tools(self):
        """Test getting tools for current stage."""
        machine = ConversationStateMachine()
        tools = machine.get_stage_tools()

        assert isinstance(tools, set)
        assert len(tools) > 0

    def test_get_state_summary(self):
        """Test state summary contains expected keys."""
        machine = ConversationStateMachine()
        summary = machine.get_state_summary()

        assert "stage" in summary
        assert "confidence" in summary
        assert "message_count" in summary
        assert "tools_executed" in summary
        assert "files_observed" in summary
        assert "files_modified" in summary
        assert "last_tools" in summary
        assert "recommended_tools" in summary

    def test_should_include_tool_current_stage(self):
        """Test tool inclusion for current stage."""
        machine = ConversationStateMachine()
        # INITIAL stage should include planning tools
        assert machine.should_include_tool("code_search")
        assert machine.should_include_tool("list_directory")

    def test_should_include_tool_adjacent_stage(self):
        """Test tool inclusion for adjacent stages."""
        machine = ConversationStateMachine()
        # Adjacent stage (PLANNING) tools should also be included
        assert machine.should_include_tool("plan_files")

    def test_get_tool_priority_boost_current_stage(self):
        """Test priority boost for current stage tools."""
        machine = ConversationStateMachine()
        # INITIAL stage tools get high boost
        boost = machine.get_tool_priority_boost("code_search")
        assert boost > 0

    def test_get_tool_priority_boost_irrelevant_tool(self):
        """Test no boost for irrelevant tools."""
        machine = ConversationStateMachine()
        # Random tool not in any stage
        boost = machine.get_tool_priority_boost("nonexistent_tool")
        assert boost == 0.0


class TestStageTransitions:
    """Tests for stage transition logic."""

    def test_transition_initial_to_planning(self):
        """Test transition from INITIAL to PLANNING."""
        machine = ConversationStateMachine()
        machine.record_message("What's your plan for implementing this?")

        # May transition to PLANNING
        assert machine.state._stage_confidence >= 0.5

    def test_transition_requires_confidence(self):
        """Test that transitions require sufficient confidence."""
        machine = ConversationStateMachine()

        # Manually transition to EXECUTION
        machine.state.stage = ConversationStage.EXECUTION
        machine.state._stage_confidence = 0.9

        # Try to transition backward with low confidence
        machine._transition_to(ConversationStage.READING, confidence=0.3)

        # Should not transition backward with low confidence
        assert machine.get_stage() == ConversationStage.EXECUTION

    def test_forward_transition_allowed(self):
        """Test forward transitions are allowed."""
        machine = ConversationStateMachine()

        # Should transition forward
        machine._transition_to(ConversationStage.PLANNING, confidence=0.6)
        assert machine.get_stage() == ConversationStage.PLANNING

        machine._transition_to(ConversationStage.READING, confidence=0.7)
        assert machine.get_stage() == ConversationStage.READING


class TestToolMappings:
    """Tests for tool-to-stage mappings."""

    def test_initial_stage_tools(self):
        """Test INITIAL stage has appropriate tools."""
        tools = STAGE_TOOL_MAPPING[ConversationStage.INITIAL]
        assert "code_search" in tools
        assert "list_directory" in tools

    def test_execution_stage_tools(self):
        """Test EXECUTION stage has write/edit tools."""
        tools = STAGE_TOOL_MAPPING[ConversationStage.EXECUTION]
        assert "write_file" in tools
        assert "edit_file" in tools
        assert "execute_bash" in tools

    def test_verification_stage_tools(self):
        """Test VERIFICATION stage has testing tools."""
        tools = STAGE_TOOL_MAPPING[ConversationStage.VERIFICATION]
        assert "testing" in tools
        assert "execute_bash" in tools


class TestKeywordMappings:
    """Tests for keyword-to-stage mappings."""

    def test_execution_keywords(self):
        """Test EXECUTION stage keywords."""
        keywords = STAGE_KEYWORDS[ConversationStage.EXECUTION]
        assert "implement" in keywords
        assert "fix" in keywords
        assert "create" in keywords

    def test_verification_keywords(self):
        """Test VERIFICATION stage keywords."""
        keywords = STAGE_KEYWORDS[ConversationStage.VERIFICATION]
        assert "test" in keywords
        assert "verify" in keywords


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
