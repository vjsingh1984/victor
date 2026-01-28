"""Unit tests for framework protocols package.

These tests verify the correctness of the protocols package structure
and all protocol implementations.

Phase 6: Migration & Testing - Unit Tests
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import Mock

import pytest

from victor.framework.protocols import (
    # Main protocols
    OrchestratorProtocol,
    ChatStateProtocol,
    ChatResultProtocol,
    WorkflowChatProtocol,
    ConversationStateProtocol,
    ProviderProtocol,
    ToolsProtocol,
    SystemPromptProtocol,
    MessagesProtocol,
    StreamingProtocol,
    CapabilityRegistryProtocol,
    # Implementations
    ChatResult,
    MutableChatState,
    # Types and enums
    ChunkType,
    OrchestratorStreamChunk,
    CapabilityType,
    OrchestratorCapability,
    # Utilities
    verify_protocol_conformance,
    # Exceptions
    IncompatibleVersionError,
)


@pytest.mark.unit
class TestProtocolsPackageStructure:
    """Test protocols package structure and exports."""

    def test_all_protocols_importable(self):
        """Verify all protocols can be imported from package."""
        from victor.framework.protocols import (
            OrchestratorProtocol,
            ChatStateProtocol,
            ChatResultProtocol,
            WorkflowChatProtocol,
            ConversationStateProtocol,
            ProviderProtocol,
            ToolsProtocol,
            SystemPromptProtocol,
            MessagesProtocol,
            StreamingProtocol,
            CapabilityRegistryProtocol,
        )

        # Verify all are protocols
        assert hasattr(OrchestratorProtocol, "__protocol_attrs__")
        assert hasattr(ChatStateProtocol, "__protocol_attrs__")
        assert hasattr(ChatResultProtocol, "__protocol_attrs__")
        assert hasattr(WorkflowChatProtocol, "__protocol_attrs__")

        print("\n✓ All protocols importable")

    def test_all_implementations_importable(self):
        """Verify all implementations can be imported."""
        from victor.framework.protocols import ChatResult, MutableChatState

        # Verify implementations
        assert isinstance(ChatResult("test", 1), ChatResult)
        assert isinstance(MutableChatState(), MutableChatState)

        print("\n✓ All implementations importable")

    def test_all_types_importable(self):
        """Verify all types and enums can be imported."""
        from victor.framework.protocols import (
            ChunkType,
            OrchestratorStreamChunk,
            CapabilityType,
            OrchestratorCapability,
            IncompatibleVersionError,
        )

        # Verify types
        assert ChunkType.CONTENT == "content"
        assert CapabilityType.TOOL == "tool"

        # Verify exception
        with pytest.raises(IncompatibleVersionError):
            raise IncompatibleVersionError("test", "1.0", "0.5")

        print("\n✓ All types importable")

    def test_direct_module_imports(self):
        """Verify direct imports from specific modules work."""
        from victor.framework.protocols.chat import ChatStateProtocol, MutableChatState
        from victor.framework.protocols.orchestrator import OrchestratorProtocol
        from victor.framework.protocols.streaming import ChunkType
        from victor.framework.protocols.capability import CapabilityType, OrchestratorCapability

        # Verify consistency
        from victor.framework.protocols import ChatStateProtocol as CSP
        from victor.framework.protocols import OrchestratorProtocol as OP

        assert ChatStateProtocol is CSP
        assert OrchestratorProtocol is OP

        print("\n✓ Direct module imports work")


@pytest.mark.unit
class TestMutableChatState:
    """Test MutableChatState implementation."""

    def test_initial_state(self):
        """Test initial state is empty."""
        state = MutableChatState()

        assert len(state.messages) == 0
        assert state.iteration_count == 0
        assert state.get_metadata("any_key") is None

        print("\n✓ Initial state is empty")

    def test_add_message(self):
        """Test adding messages to state."""
        state = MutableChatState()

        state.add_message("user", "Hello")
        state.add_message("assistant", "Hi there!")

        assert len(state.messages) == 2
        assert state.messages[0]["role"] == "user"
        assert state.messages[0]["content"] == "Hello"
        assert state.messages[1]["role"] == "assistant"
        assert state.messages[1]["content"] == "Hi there!"

        print("\n✓ Messages added correctly")

    def test_add_message_with_tool_calls(self):
        """Test adding messages with tool calls."""
        state = MutableChatState()

        tool_calls = [{"name": "read_file", "arguments": {"path": "main.py"}}]
        state.add_message("assistant", "Reading file", tool_calls=tool_calls)

        assert len(state.messages) == 1
        assert "tool_calls" in state.messages[0]
        assert state.messages[0]["tool_calls"] == tool_calls

        print("\n✓ Tool calls stored correctly")

    def test_increment_iteration(self):
        """Test incrementing iteration count."""
        state = MutableChatState()

        assert state.iteration_count == 0

        state.increment_iteration()
        assert state.iteration_count == 1

        state.increment_iteration()
        state.increment_iteration()
        assert state.iteration_count == 3

        print("\n✓ Iteration count increments correctly")

    def test_metadata_operations(self):
        """Test metadata get/set operations."""
        state = MutableChatState()

        # Set metadata
        state.set_metadata("key1", "value1")
        state.set_metadata("key2", 42)
        state.set_metadata("key3", {"nested": "value"})

        # Get metadata
        assert state.get_metadata("key1") == "value1"
        assert state.get_metadata("key2") == 42
        assert state.get_metadata("key3") == {"nested": "value"}

        # Get with default
        assert state.get_metadata("nonexistent", "default") == "default"
        assert state.get_metadata("nonexistent") is None

        print("\n✓ Metadata operations work correctly")

    def test_to_dict(self):
        """Test serializing state to dictionary."""
        state = MutableChatState()
        state.add_message("user", "Test")
        state.increment_iteration()
        state.set_metadata("key", "value")

        state_dict = state.to_dict()

        assert "messages" in state_dict
        assert "iteration_count" in state_dict
        assert "metadata" in state_dict
        assert len(state_dict["messages"]) == 1
        assert state_dict["iteration_count"] == 1
        assert state_dict["metadata"]["key"] == "value"

        print("\n✓ State serializes to dict correctly")

    def test_from_dict(self):
        """Test creating state from dictionary."""
        state_dict = {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ],
            "iteration_count": 2,
            "metadata": {"key": "value"},
        }

        state = MutableChatState.from_dict(state_dict)

        assert len(state.messages) == 2
        assert state.iteration_count == 2
        assert state.get_metadata("key") == "value"

        print("\n✓ State deserializes from dict correctly")

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        original = MutableChatState()
        original.add_message("user", "Test message")
        original.add_message("assistant", "Test response")
        original.increment_iteration()
        original.set_metadata("test", True)

        # Serialize
        state_dict = original.to_dict()

        # Deserialize
        restored = MutableChatState.from_dict(state_dict)

        # Verify equality
        assert restored.messages == original.messages
        assert restored.iteration_count == original.iteration_count
        assert restored.get_metadata("test") == original.get_metadata("test")

        print("\n✓ Serialization roundtrip preserves data")

    def test_clear_state(self):
        """Test clearing state."""
        state = MutableChatState()
        state.add_message("user", "Test")
        state.increment_iteration()
        state.set_metadata("key", "value")

        # Clear
        state.clear()

        # Verify cleared
        assert len(state.messages) == 0
        assert state.iteration_count == 0
        assert state.get_metadata("key") is None

        print("\n✓ State clears correctly")

    def test_repr(self):
        """Test string representation."""
        state = MutableChatState()
        state.add_message("user", "Test")
        state.set_metadata("key", "value")

        repr_str = repr(state)

        assert "MutableChatState" in repr_str
        assert "messages=1" in repr_str
        assert "iteration=0" in repr_str
        assert "metadata_keys" in repr_str

        print("\n✓ String representation works")


@pytest.mark.unit
class TestChatResult:
    """Test ChatResult implementation."""

    def test_create_result(self):
        """Test creating a chat result."""
        result = ChatResult(content="Success!", iteration_count=3, metadata={"files": ["main.py"]})

        assert result.content == "Success!"
        assert result.iteration_count == 3
        assert result.metadata["files"] == ["main.py"]

        print("\n✓ ChatResult created correctly")

    def test_to_dict(self):
        """Test serializing result to dictionary."""
        result = ChatResult(content="Test", iteration_count=1, metadata={"key": "value"})

        result_dict = result.to_dict()

        assert result_dict["content"] == "Test"
        assert result_dict["iteration_count"] == 1
        assert result_dict["metadata"]["key"] == "value"

        print("\n✓ ChatResult serializes correctly")

    def test_get_summary(self):
        """Test getting result summary."""
        result = ChatResult(content="x" * 100, iteration_count=5, metadata={"key": "value"})

        summary = result.get_summary()

        assert "iterations=5" in summary
        assert "content_length=100" in summary
        assert "metadata_keys" in summary

        print("\n✓ ChatResult summary works")

    def test_immutable(self):
        """Test ChatResult is immutable (frozen dataclass)."""
        result = ChatResult(content="Test", iteration_count=1)

        # Verify frozen
        with pytest.raises(Exception):  # FrozenInstanceError
            result.content = "Modified"

        print("\n✓ ChatResult is immutable")


@pytest.mark.unit
class TestOrchestratorStreamChunk:
    """Test OrchestratorStreamChunk implementation."""

    def test_create_content_chunk(self):
        """Test creating a content chunk."""
        chunk = OrchestratorStreamChunk(
            chunk_type=ChunkType.CONTENT, content="Hello", is_final=False
        )

        assert chunk.chunk_type == ChunkType.CONTENT
        assert chunk.content == "Hello"
        assert chunk.is_final is False

        print("\n✓ Content chunk created correctly")

    def test_create_tool_call_chunk(self):
        """Test creating a tool call chunk."""
        chunk = OrchestratorStreamChunk(
            chunk_type=ChunkType.TOOL_CALL,
            tool_name="read_file",
            tool_id="123",
            tool_arguments={"path": "main.py"},
        )

        assert chunk.chunk_type == ChunkType.TOOL_CALL
        assert chunk.tool_name == "read_file"
        assert chunk.tool_id == "123"
        assert chunk.tool_arguments["path"] == "main.py"

        print("\n✓ Tool call chunk created correctly")

    def test_create_stage_change_chunk(self):
        """Test creating a stage change chunk."""
        chunk = OrchestratorStreamChunk(
            chunk_type=ChunkType.STAGE_CHANGE, old_stage="READING", new_stage="EXECUTING"
        )

        assert chunk.chunk_type == ChunkType.STAGE_CHANGE
        assert chunk.old_stage == "READING"
        assert chunk.new_stage == "EXECUTING"

        print("\n✓ Stage change chunk created correctly")


@pytest.mark.unit
class TestOrchestratorCapability:
    """Test OrchestratorCapability implementation."""

    def test_create_capability(self):
        """Test creating a capability."""
        cap = OrchestratorCapability(
            name="test_capability",
            capability_type=CapabilityType.TOOL,
            version="1.0",
            setter="set_test",
            description="Test capability",
        )

        assert cap.name == "test_capability"
        assert cap.capability_type == CapabilityType.TOOL
        assert cap.version == "1.0"
        assert cap.setter == "set_test"

        print("\n✓ Capability created correctly")

    def test_capability_requires_accessor(self):
        """Test capability must have at least one accessor."""
        with pytest.raises(ValueError):
            OrchestratorCapability(name="invalid", capability_type=CapabilityType.TOOL)

        print("\n✓ Capability validation works")

    def test_version_compatibility(self):
        """Test version compatibility checking."""
        cap = OrchestratorCapability(
            name="test", capability_type=CapabilityType.TOOL, version="2.1", setter="set_test"
        )

        # Compatible versions
        assert cap.is_compatible_with("1.0") is True
        assert cap.is_compatible_with("2.0") is True
        assert cap.is_compatible_with("2.1") is True

        # Incompatible versions
        assert cap.is_compatible_with("2.2") is False
        assert cap.is_compatible_with("3.0") is False

        print("\n✓ Version compatibility checking works")

    def test_version_validation(self):
        """Test version format validation."""
        # Valid versions
        valid_versions = ["1.0", "2.1", "10.20", "1.0.0", "2.1.3"]
        for version in valid_versions:
            cap = OrchestratorCapability(
                name="test", capability_type=CapabilityType.TOOL, version=version, setter="set_test"
            )
            assert cap.version == version

        # Invalid versions
        invalid_versions = ["1", "1.x", "a.b", "1"]
        for version in invalid_versions:
            with pytest.raises(ValueError):
                OrchestratorCapability(
                    name="test",
                    capability_type=CapabilityType.TOOL,
                    version=version,
                    setter="set_test",
                )

        print("\n✓ Version validation works")


@pytest.mark.unit
class TestProtocolVerification:
    """Test protocol verification utilities."""

    def test_verify_protocol_conformance_success(self):
        """Test verifying conforming object."""
        from victor.framework.protocols import ChatStateProtocol, MutableChatState

        state = MutableChatState()
        conforms, missing = verify_protocol_conformance(state, ChatStateProtocol)

        assert conforms is True
        assert len(missing) == 0

        print("\n✓ Protocol verification passes for conforming object")

    def test_verify_protocol_conformance_failure(self):
        """Test verifying non-conforming object."""
        from victor.framework.protocols import ChatStateProtocol

        non_conforming = object()
        conforms, missing = verify_protocol_conformance(non_conforming, ChatStateProtocol)

        assert conforms is False
        assert len(missing) > 0

        print("\n✓ Protocol verification fails for non-conforming object")

    def test_verify_custom_protocol(self):
        """Test verifying custom protocol."""
        from typing import Protocol

        class CustomProtocol(Protocol):
            def custom_method(self) -> str: ...

        class ConformingImplementation:
            def custom_method(self) -> str:
                return "custom"

        class NonConformingImplementation:
            pass

        # Conforming
        conforms, missing = verify_protocol_conformance(ConformingImplementation(), CustomProtocol)
        assert conforms is True

        # Non-conforming
        conforms, missing = verify_protocol_conformance(
            NonConformingImplementation(), CustomProtocol
        )
        assert conforms is False
        assert "custom_method" in missing

        print("\n✓ Custom protocol verification works")
