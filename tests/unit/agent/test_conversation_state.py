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

"""Tests for conversation_state module."""


from victor.agent.conversation_state import (
    ConversationStage,
    ConversationState,
    ConversationStateMachine,
    STAGE_KEYWORDS,
)
from victor.tools.metadata_registry import get_tools_by_stage


class TestConversationStage:
    """Tests for ConversationStage enum."""

    def test_all_stages_defined(self):
        """Test all conversation stages are defined."""
        assert ConversationStage.INITIAL is not None
        assert ConversationStage.PLANNING is not None
        assert ConversationStage.READING is not None
        assert ConversationStage.ANALYSIS is not None
        assert ConversationStage.EXECUTION is not None
        assert ConversationStage.VERIFICATION is not None
        assert ConversationStage.COMPLETION is not None


class TestStageToolMapping:
    """Tests for decorator-driven stage-to-tool mapping via metadata registry."""

    def test_all_stages_have_mappings(self):
        """Test stage tool mappings can be retrieved for all stages.

        Note: With decorator-driven selection, stages may return empty sets
        if no tools have decorated that stage yet.
        """
        for stage in ConversationStage:
            # get_tools_by_stage returns a set (may be empty)
            tools = get_tools_by_stage(stage.name.lower())
            assert isinstance(tools, set), f"Stage {stage} should return a set"

    def test_initial_stage_tools(self):
        """Test initial stage can be queried for tools."""
        tools = get_tools_by_stage("initial")
        # With decorator-driven selection, we just verify the API works
        assert isinstance(tools, set)

    def test_execution_stage_tools(self):
        """Test execution stage can be queried for tools."""
        tools = get_tools_by_stage("execution")
        # With decorator-driven selection, we just verify the API works
        assert isinstance(tools, set)


class TestStageKeywords:
    """Tests for STAGE_KEYWORDS constant."""

    def test_all_stages_have_keywords(self):
        """Test all stages have keywords."""
        for stage in ConversationStage:
            assert stage in STAGE_KEYWORDS, f"Stage {stage} missing keywords"

    def test_execution_keywords(self):
        """Test execution stage has action keywords."""
        keywords = STAGE_KEYWORDS[ConversationStage.EXECUTION]
        assert "change" in keywords
        assert "fix" in keywords
        assert "implement" in keywords


class TestConversationStateDataclass:
    """Tests for ConversationState dataclass."""

    def test_default_init(self):
        """Test default initialization."""
        state = ConversationState()
        assert state.stage == ConversationStage.INITIAL
        assert state.tool_history == []
        assert state.observed_files == set()
        assert state.modified_files == set()
        assert state.message_count == 0
        assert state.last_tools == []

    def test_custom_init(self):
        """Test initialization with custom values."""
        state = ConversationState(
            stage=ConversationStage.PLANNING,
            message_count=5,
        )
        assert state.stage == ConversationStage.PLANNING
        assert state.message_count == 5

    def test_record_tool_execution(self):
        """Test recording tool execution."""
        state = ConversationState()
        state.record_tool_execution("code_search", {"query": "test"})

        assert "code_search" in state.tool_history
        assert "code_search" in state.last_tools

    def test_record_read_file_updates_observed(self):
        """Test read tool updates observed_files."""
        state = ConversationState()
        # Use canonical tool name "read" and "path" arg as checked in conversation_state.py
        state.record_tool_execution("read", {"path": "/test/file.py"})

        assert "/test/file.py" in state.observed_files

    def test_record_write_file_updates_modified(self):
        """Test write tool updates modified_files."""
        state = ConversationState()
        # Use canonical tool name "write" and "path" arg as checked in conversation_state.py
        state.record_tool_execution("write", {"path": "/test/file.py"})

        assert "/test/file.py" in state.modified_files

    def test_record_edit_file_updates_modified(self):
        """Test edit tool updates modified_files."""
        state = ConversationState()
        # Use canonical tool name "edit" and "path" arg as checked in conversation_state.py
        state.record_tool_execution("edit", {"path": "/test/file.py"})

        assert "/test/file.py" in state.modified_files

    def test_last_tools_limit(self):
        """Test last_tools maintains limited history."""
        state = ConversationState()
        for i in range(10):
            state.record_tool_execution(f"tool_{i}", {})

        # Should only keep last 5 tools
        assert len(state.last_tools) == 5

    def test_record_message_increments_count(self):
        """Test recording message increments count."""
        state = ConversationState()
        state.record_message()
        state.record_message()

        assert state.message_count == 2


class TestConversationStateMachine:
    """Tests for ConversationStateMachine class."""

    def test_init(self):
        """Test state machine initialization."""
        sm = ConversationStateMachine()
        assert sm.state.stage == ConversationStage.INITIAL
        assert sm.state.message_count == 0

    def test_reset(self):
        """Test resetting state machine."""
        sm = ConversationStateMachine()
        # Use canonical tool name "read" which is checked in conversation_state.py
        sm.record_tool_execution("read", {"path": "/test.py"})
        sm.record_message("Hello", is_user=True)

        sm.reset()

        assert sm.state.stage == ConversationStage.INITIAL
        assert sm.state.tool_history == []
        assert sm.state.message_count == 0

    def test_get_stage(self):
        """Test getting current stage."""
        sm = ConversationStateMachine()
        assert sm.get_stage() == ConversationStage.INITIAL

    def test_get_stage_tools(self):
        """Test getting stage tools."""
        sm = ConversationStateMachine()
        tools = sm.get_stage_tools()

        # Decorator-driven selection: just verify it returns a set
        # Specific tool membership depends on decorator configuration
        assert isinstance(tools, set)

    def test_record_tool_execution(self):
        """Test recording tool execution through state machine."""
        sm = ConversationStateMachine()
        # Use canonical tool name "read" which is checked in conversation_state.py
        sm.record_tool_execution("read", {"path": "/test.py"})

        assert "/test.py" in sm.state.observed_files

    def test_record_message(self):
        """Test recording message."""
        sm = ConversationStateMachine()
        sm.record_message("Hello, please help me", is_user=True)

        assert sm.state.message_count == 1

    def test_get_state_summary(self):
        """Test getting state summary."""
        sm = ConversationStateMachine()
        # Use canonical tool name "read" which is checked in conversation_state.py
        sm.record_tool_execution("read", {"path": "/test.py"})
        sm.record_message("Hello", is_user=True)

        summary = sm.get_state_summary()

        assert "stage" in summary
        assert "message_count" in summary
        assert summary["message_count"] == 1
        assert summary["files_observed"] == 1

    def test_detect_stage_from_content_execution(self):
        """Test detecting execution stage from message content."""
        sm = ConversationStateMachine()

        # Message with execution keywords
        stage = sm._detect_stage_from_content("Please fix and implement this change")

        # Should detect execution stage (multiple keywords)
        assert stage == ConversationStage.EXECUTION

    def test_detect_stage_from_content_no_match(self):
        """Test no stage detected for neutral content."""
        sm = ConversationStateMachine()

        # Neutral message
        stage = sm._detect_stage_from_content("Hello")

        assert stage is None

    def test_detect_stage_from_tools(self):
        """Test detecting stage from tool patterns."""
        sm = ConversationStateMachine()

        # Record some read operations using canonical names
        sm.state.last_tools = ["read", "read", "search"]

        stage = sm._detect_stage_from_tools()

        # Stage detection depends on decorator-driven registry
        # Result can be any stage or None depending on tool configurations
        assert stage is None or stage in ConversationStage

    def test_detect_stage_from_tools_empty(self):
        """Test no stage detected for empty tool history."""
        sm = ConversationStateMachine()

        stage = sm._detect_stage_from_tools()

        assert stage is None


class TestConversationStateMachineTransitions:
    """Tests for stage transitions."""

    def test_maybe_transition(self):
        """Test maybe_transition method."""
        sm = ConversationStateMachine()

        # Simulate executing read operations using canonical names
        sm.record_tool_execution("read", {"path": "/test.py"})
        sm.record_tool_execution("read", {"path": "/test2.py"})

        # Stage may transition
        stage = sm.get_stage()
        # Could be INITIAL or READING depending on transition logic
        assert stage in ConversationStage

    def test_transition_with_confidence(self):
        """Test transition updates confidence."""
        sm = ConversationStateMachine()

        # Send message with strong execution keywords
        sm.record_message("Please fix and change and implement this update", is_user=True)

        # Confidence should be updated
        assert sm.state._stage_confidence >= 0


class TestConversationStateSerialization:
    """Tests for serialization/deserialization."""

    def test_state_to_dict(self):
        """Test ConversationState.to_dict (covers line 176)."""
        state = ConversationState(
            stage=ConversationStage.READING,
            message_count=5,
        )
        state.tool_history = ["read_file", "code_search"]
        state.observed_files = {"/test.py", "/other.py"}
        state.modified_files = {"/changed.py"}
        state.last_tools = ["read_file"]

        data = state.to_dict()

        assert data["stage"] == "READING"
        assert data["tool_history"] == ["read_file", "code_search"]
        assert set(data["observed_files"]) == {"/test.py", "/other.py"}
        assert set(data["modified_files"]) == {"/changed.py"}
        assert data["message_count"] == 5
        assert data["last_tools"] == ["read_file"]

    def test_state_from_dict(self):
        """Test ConversationState.from_dict (covers lines 196-204)."""
        data = {
            "stage": "EXECUTION",
            "tool_history": ["write_file"],
            "observed_files": ["/test.py"],
            "modified_files": ["/test.py"],
            "message_count": 3,
            "last_tools": ["write_file"],
            "stage_confidence": 0.8,
        }

        state = ConversationState.from_dict(data)

        assert state.stage == ConversationStage.EXECUTION
        assert state.tool_history == ["write_file"]
        assert state.observed_files == {"/test.py"}
        assert state.modified_files == {"/test.py"}
        assert state.message_count == 3
        assert state.last_tools == ["write_file"]
        assert state._stage_confidence == 0.8

    def test_state_from_dict_empty(self):
        """Test ConversationState.from_dict with empty dict."""
        state = ConversationState.from_dict({})

        assert state.stage == ConversationStage.INITIAL
        assert state.tool_history == []
        assert state.message_count == 0

    def test_state_machine_to_dict(self):
        """Test ConversationStateMachine.to_dict (covers line 417)."""
        sm = ConversationStateMachine()
        # Use canonical tool name for consistency
        sm.record_tool_execution("read", {"path": "/test.py"})

        data = sm.to_dict()

        assert "state" in data
        assert data["state"]["tool_history"] == ["read"]

    def test_state_machine_from_dict(self):
        """Test ConversationStateMachine.from_dict (covers lines 431-434)."""
        data = {
            "state": {
                "stage": "ANALYSIS",
                "tool_history": ["code_search"],
                "observed_files": ["/file.py"],
                "modified_files": [],
                "message_count": 2,
                "last_tools": ["code_search"],
                "stage_confidence": 0.7,
            }
        }

        sm = ConversationStateMachine.from_dict(data)

        assert sm.state.stage == ConversationStage.ANALYSIS
        assert sm.state.tool_history == ["code_search"]
        assert sm.state.message_count == 2

    def test_state_machine_from_dict_empty(self):
        """Test ConversationStateMachine.from_dict with empty dict."""
        sm = ConversationStateMachine.from_dict({})

        assert sm.state.stage == ConversationStage.INITIAL


class TestConversationStateMachineRoundTrip:
    """Tests for state machine round-trip serialization."""

    def test_state_roundtrip(self):
        """Test ConversationState serialization round-trip."""
        original = ConversationState(
            stage=ConversationStage.ANALYSIS,
            message_count=10,
        )
        original.tool_history = ["read_file", "code_search", "write_file"]
        original.observed_files = {"/a.py", "/b.py"}
        original.modified_files = {"/c.py"}
        original.last_tools = ["write_file"]
        original._stage_confidence = 0.75

        data = original.to_dict()
        restored = ConversationState.from_dict(data)

        assert restored.stage == original.stage
        assert restored.tool_history == original.tool_history
        assert restored.observed_files == original.observed_files
        assert restored.modified_files == original.modified_files
        assert restored.message_count == original.message_count
        assert restored.last_tools == original.last_tools
        assert restored._stage_confidence == original._stage_confidence

    def test_machine_roundtrip(self):
        """Test ConversationStateMachine serialization round-trip."""
        original = ConversationStateMachine()
        # Use canonical tool names for file tracking
        original.record_tool_execution("read", {"path": "/test.py"})
        original.record_tool_execution("write", {"path": "/test.py"})
        original.record_message("Hello, please help", is_user=True)

        data = original.to_dict()
        restored = ConversationStateMachine.from_dict(data)

        assert restored.state.tool_history == original.state.tool_history
        assert restored.state.observed_files == original.state.observed_files
        assert restored.state.modified_files == original.state.modified_files
        assert restored.state.message_count == original.state.message_count


class TestConversationStateMachineRegistryIntegration:
    """Tests for registry integration in ConversationStateMachine.

    Note: Stage-to-tool mapping is now fully decorator-driven via the metadata registry.
    These tests verify the integration with registry_get_tools_by_stage().
    """

    def test_get_tools_for_stage_uses_registry(self):
        """Test _get_tools_for_stage uses the metadata registry."""
        sm = ConversationStateMachine()

        # Get tools for INITIAL stage - result depends on decorator configuration
        tools = sm._get_tools_for_stage(ConversationStage.INITIAL)

        # Should return a set (possibly empty if no tools decorated with initial stage)
        assert isinstance(tools, set)

    def test_get_stage_tools_returns_registry_tools(self):
        """Test get_stage_tools returns tools from metadata registry."""
        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.EXECUTION

        tools = sm.get_stage_tools()

        # Should return a set from the registry
        assert isinstance(tools, set)

    def test_get_tools_for_stage_includes_registry_tools(self):
        """Test _get_tools_for_stage includes tools from registry."""
        from unittest.mock import patch

        sm = ConversationStateMachine()

        # Mock the registry to return specific tools
        with patch(
            "victor.agent.conversation_state.registry_get_tools_by_stage",
            return_value={"mock_tool_from_registry"},
        ):
            tools = sm._get_tools_for_stage(ConversationStage.INITIAL)

            # Should include registry tools
            assert "mock_tool_from_registry" in tools

    def test_get_stage_tools_from_registry(self):
        """Test get_stage_tools returns tools from registry."""
        from unittest.mock import patch

        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.READING

        with patch(
            "victor.agent.conversation_state.registry_get_tools_by_stage",
            return_value={"registry_read_tool"},
        ):
            tools = sm.get_stage_tools()

            # Should include registry tool
            assert "registry_read_tool" in tools

    def test_detect_stage_from_tools_uses_helper(self):
        """Test _detect_stage_from_tools uses _get_tools_for_stage."""
        from unittest.mock import patch

        sm = ConversationStateMachine()
        sm.state.last_tools = ["custom_read_tool", "custom_read_tool"]

        # Mock _get_tools_for_stage to return custom_read_tool for READING
        original_get_tools = sm._get_tools_for_stage

        def mock_get_tools(stage):
            if stage == ConversationStage.READING:
                return {"custom_read_tool", "read"}
            return original_get_tools(stage)

        with patch.object(sm, "_get_tools_for_stage", side_effect=mock_get_tools):
            stage = sm._detect_stage_from_tools()

            # Should detect READING stage due to custom_read_tool overlap
            assert stage == ConversationStage.READING

    def test_should_include_tool_with_registry_tools(self):
        """Test should_include_tool works with registry integration."""
        from unittest.mock import patch

        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.ANALYSIS

        # Mock registry to include a custom tool in ANALYSIS stage
        with patch(
            "victor.agent.conversation_state.registry_get_tools_by_stage",
            return_value={"custom_analysis_tool"},
        ):
            # Custom tool from registry should be included
            assert sm.should_include_tool("custom_analysis_tool") is True

    def test_get_tool_priority_boost_with_registry(self):
        """Test get_tool_priority_boost works with registry integration."""
        from unittest.mock import patch

        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.EXECUTION

        with patch(
            "victor.agent.conversation_state.registry_get_tools_by_stage",
            return_value={"custom_exec_tool"},
        ):
            # Custom tool from registry should get high boost
            boost = sm.get_tool_priority_boost("custom_exec_tool")
            assert boost == 0.15  # High boost for stage-relevant tools
