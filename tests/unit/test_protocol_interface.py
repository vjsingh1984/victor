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

"""Tests for Victor Protocol Interface types and data structures."""

import pytest
from datetime import datetime

from victor.protocol.interface import (
    AgentMode,
    AgentStatus,
    ChatMessage,
    ChatResponse,
    SearchResult,
    StreamChunk,
    ToolCall,
    ToolResult,
    UndoRedoResult,
    VictorProtocol,
)


# =============================================================================
# ENUM TESTS
# =============================================================================


class TestAgentMode:
    """Tests for AgentMode enum."""

    def test_build_mode(self):
        """Test build mode."""
        assert AgentMode.BUILD.value == "build"
        assert AgentMode.BUILD == "build"  # str enum comparison

    def test_plan_mode(self):
        """Test plan mode."""
        assert AgentMode.PLAN.value == "plan"

    def test_explore_mode(self):
        """Test explore mode."""
        assert AgentMode.EXPLORE.value == "explore"


# =============================================================================
# TOOL RESULT TESTS
# =============================================================================


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_creation_success(self):
        """Test successful tool result creation."""
        result = ToolResult(
            success=True,
            output="File created successfully",
        )
        assert result.success is True
        assert result.output == "File created successfully"
        assert result.error is None
        assert result.metadata == {}

    def test_creation_failure(self):
        """Test failed tool result creation."""
        result = ToolResult(
            success=False,
            output="",
            error="File not found",
            metadata={"attempted_path": "/test.py"},
        )
        assert result.success is False
        assert result.error == "File not found"
        assert result.metadata["attempted_path"] == "/test.py"

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = ToolResult(
            success=True,
            output="Done",
            error=None,
            metadata={"lines": 42},
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == "Done"
        assert d["error"] is None
        assert d["metadata"]["lines"] == 42

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "success": True,
            "output": "Result output",
            "error": None,
            "metadata": {"key": "value"},
        }
        result = ToolResult.from_dict(data)
        assert result.success is True
        assert result.output == "Result output"
        assert result.metadata["key"] == "value"

    def test_from_dict_missing_optionals(self):
        """Test from_dict with missing optional fields."""
        data = {
            "success": True,
            "output": "Done",
        }
        result = ToolResult.from_dict(data)
        assert result.error is None
        assert result.metadata == {}


# =============================================================================
# TOOL CALL TESTS
# =============================================================================


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_creation_without_result(self):
        """Test tool call without result."""
        call = ToolCall(
            id="call_123",
            name="read_file",
            arguments={"path": "/test.py"},
        )
        assert call.id == "call_123"
        assert call.name == "read_file"
        assert call.arguments["path"] == "/test.py"
        assert call.result is None

    def test_creation_with_result(self):
        """Test tool call with result."""
        result = ToolResult(success=True, output="Content")
        call = ToolCall(
            id="call_123",
            name="read_file",
            arguments={"path": "/test.py"},
            result=result,
        )
        assert call.result is not None
        assert call.result.success is True

    def test_to_dict_without_result(self):
        """Test to_dict without result."""
        call = ToolCall(
            id="call_123",
            name="write_file",
            arguments={"path": "/test.py", "content": "hello"},
        )
        d = call.to_dict()
        assert d["id"] == "call_123"
        assert d["name"] == "write_file"
        assert d["arguments"]["path"] == "/test.py"
        assert d["result"] is None

    def test_to_dict_with_result(self):
        """Test to_dict with result."""
        result = ToolResult(success=True, output="Done")
        call = ToolCall(
            id="call_456",
            name="execute",
            arguments={"cmd": "ls"},
            result=result,
        )
        d = call.to_dict()
        assert d["result"] is not None
        assert d["result"]["success"] is True

    def test_from_dict_without_result(self):
        """Test from_dict without result."""
        data = {
            "id": "call_789",
            "name": "search",
            "arguments": {"query": "hello"},
        }
        call = ToolCall.from_dict(data)
        assert call.id == "call_789"
        assert call.name == "search"
        assert call.result is None

    def test_from_dict_with_result(self):
        """Test from_dict with result."""
        data = {
            "id": "call_789",
            "name": "search",
            "arguments": {"query": "hello"},
            "result": {
                "success": True,
                "output": "Found 3 matches",
                "error": None,
                "metadata": {},
            },
        }
        call = ToolCall.from_dict(data)
        assert call.result is not None
        assert call.result.output == "Found 3 matches"


# =============================================================================
# CHAT MESSAGE TESTS
# =============================================================================


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_creation_simple(self):
        """Test simple chat message creation."""
        msg = ChatMessage(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.tool_calls == []
        assert isinstance(msg.timestamp, datetime)

    def test_creation_with_tool_calls(self):
        """Test chat message with tool calls."""
        call = ToolCall("call_1", "read_file", {"path": "/test.py"})
        msg = ChatMessage(
            role="assistant",
            content="I'll read that file.",
            tool_calls=[call],
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "read_file"

    def test_to_dict(self):
        """Test to_dict serialization."""
        msg = ChatMessage(role="user", content="Test message")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Test message"
        assert d["tool_calls"] == []
        assert "timestamp" in d

    def test_to_dict_with_tool_calls(self):
        """Test to_dict with tool calls."""
        call = ToolCall("call_1", "search", {"query": "test"})
        msg = ChatMessage(
            role="assistant",
            content="Searching...",
            tool_calls=[call],
        )
        d = msg.to_dict()
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["name"] == "search"

    def test_from_dict(self):
        """Test from_dict deserialization."""
        now = datetime.now()
        data = {
            "role": "assistant",
            "content": "Hello!",
            "tool_calls": [],
            "timestamp": now.isoformat(),
        }
        msg = ChatMessage.from_dict(data)
        assert msg.role == "assistant"
        assert msg.content == "Hello!"
        assert msg.timestamp.isoformat() == now.isoformat()

    def test_from_dict_with_tool_calls(self):
        """Test from_dict with tool calls."""
        data = {
            "role": "assistant",
            "content": "Processing",
            "tool_calls": [
                {"id": "call_1", "name": "exec", "arguments": {"cmd": "ls"}},
            ],
        }
        msg = ChatMessage.from_dict(data)
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "exec"

    def test_from_dict_missing_timestamp(self):
        """Test from_dict without timestamp."""
        data = {
            "role": "user",
            "content": "Test",
        }
        msg = ChatMessage.from_dict(data)
        assert isinstance(msg.timestamp, datetime)


# =============================================================================
# CHAT RESPONSE TESTS
# =============================================================================


class TestChatResponse:
    """Tests for ChatResponse dataclass."""

    def test_creation_simple(self):
        """Test simple chat response creation."""
        resp = ChatResponse(content="Hello!")
        assert resp.content == "Hello!"
        assert resp.tool_calls == []
        assert resp.finish_reason == "stop"
        assert resp.usage == {}

    def test_creation_with_tool_calls(self):
        """Test chat response with tool calls."""
        call = ToolCall("call_1", "write", {"content": "test"})
        resp = ChatResponse(
            content="Writing file...",
            tool_calls=[call],
            finish_reason="tool_use",
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        assert len(resp.tool_calls) == 1
        assert resp.finish_reason == "tool_use"
        assert resp.usage["input_tokens"] == 100

    def test_to_dict(self):
        """Test to_dict serialization."""
        resp = ChatResponse(
            content="Done",
            finish_reason="stop",
            usage={"total_tokens": 150},
        )
        d = resp.to_dict()
        assert d["content"] == "Done"
        assert d["finish_reason"] == "stop"
        assert d["usage"]["total_tokens"] == 150

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "content": "Response content",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {"tokens": 100},
        }
        resp = ChatResponse.from_dict(data)
        assert resp.content == "Response content"
        assert resp.finish_reason == "stop"

    def test_from_dict_with_tool_calls(self):
        """Test from_dict with tool calls."""
        data = {
            "content": "Running tool",
            "tool_calls": [
                {"id": "c1", "name": "bash", "arguments": {"cmd": "ls"}},
            ],
            "finish_reason": "tool_use",
        }
        resp = ChatResponse.from_dict(data)
        assert len(resp.tool_calls) == 1

    def test_from_dict_missing_fields(self):
        """Test from_dict with minimal fields."""
        data = {"content": "Minimal"}
        resp = ChatResponse.from_dict(data)
        assert resp.finish_reason == "stop"
        assert resp.usage == {}


# =============================================================================
# STREAM CHUNK TESTS
# =============================================================================


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_creation_content_only(self):
        """Test stream chunk with content only."""
        chunk = StreamChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.tool_call is None
        assert chunk.finish_reason is None

    def test_creation_with_tool_call(self):
        """Test stream chunk with tool call."""
        call = ToolCall("call_1", "search", {"q": "test"})
        chunk = StreamChunk(content="", tool_call=call)
        assert chunk.tool_call is not None
        assert chunk.tool_call.name == "search"

    def test_creation_with_finish_reason(self):
        """Test stream chunk with finish reason."""
        chunk = StreamChunk(content="", finish_reason="stop")
        assert chunk.finish_reason == "stop"

    def test_to_dict_simple(self):
        """Test to_dict with simple content."""
        chunk = StreamChunk(content="Text")
        d = chunk.to_dict()
        assert d["content"] == "Text"
        assert d["tool_call"] is None
        assert d["finish_reason"] is None

    def test_to_dict_with_tool_call(self):
        """Test to_dict with tool call."""
        call = ToolCall("c1", "run", {"cmd": "ls"})
        chunk = StreamChunk(content="", tool_call=call)
        d = chunk.to_dict()
        assert d["tool_call"] is not None
        assert d["tool_call"]["name"] == "run"


# =============================================================================
# SEARCH RESULT TESTS
# =============================================================================


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_creation_minimal(self):
        """Test minimal search result creation."""
        result = SearchResult(
            file="test.py",
            line=42,
            content="def test_func():",
            score=0.95,
        )
        assert result.file == "test.py"
        assert result.line == 42
        assert result.score == 0.95
        assert result.context == ""

    def test_creation_with_context(self):
        """Test search result with context."""
        result = SearchResult(
            file="module.py",
            line=10,
            content="class MyClass:",
            score=0.85,
            context="# Comment above\nclass MyClass:\n    pass",
        )
        assert result.context != ""

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = SearchResult(
            file="file.py",
            line=5,
            content="x = 1",
            score=0.9,
            context="prev\nx = 1\nnext",
        )
        d = result.to_dict()
        assert d["file"] == "file.py"
        assert d["line"] == 5
        assert d["score"] == 0.9
        assert d["context"] == "prev\nx = 1\nnext"

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "file": "main.py",
            "line": 100,
            "content": "print('hello')",
            "score": 0.75,
            "context": "context lines",
        }
        result = SearchResult.from_dict(data)
        assert result.file == "main.py"
        assert result.line == 100
        assert result.context == "context lines"

    def test_from_dict_missing_context(self):
        """Test from_dict without context."""
        data = {
            "file": "test.py",
            "line": 1,
            "content": "import os",
            "score": 0.5,
        }
        result = SearchResult.from_dict(data)
        assert result.context == ""


# =============================================================================
# UNDO REDO RESULT TESTS
# =============================================================================


class TestUndoRedoResult:
    """Tests for UndoRedoResult dataclass."""

    def test_creation_success(self):
        """Test successful undo/redo result."""
        result = UndoRedoResult(
            success=True,
            message="Undid last change",
            files_modified=["test.py", "main.py"],
        )
        assert result.success is True
        assert result.message == "Undid last change"
        assert len(result.files_modified) == 2

    def test_creation_failure(self):
        """Test failed undo/redo result."""
        result = UndoRedoResult(
            success=False,
            message="Nothing to undo",
        )
        assert result.success is False
        assert result.files_modified == []

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = UndoRedoResult(
            success=True,
            message="Redone",
            files_modified=["a.py"],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["message"] == "Redone"
        assert "a.py" in d["files_modified"]


# =============================================================================
# AGENT STATUS TESTS
# =============================================================================


class TestAgentStatus:
    """Tests for AgentStatus dataclass."""

    def test_creation(self):
        """Test agent status creation."""
        status = AgentStatus(
            provider="anthropic",
            model="claude-3-opus",
            mode=AgentMode.BUILD,
            connected=True,
            tools_available=65,
            conversation_length=5,
        )
        assert status.provider == "anthropic"
        assert status.model == "claude-3-opus"
        assert status.mode == AgentMode.BUILD
        assert status.connected is True
        assert status.tools_available == 65

    def test_to_dict(self):
        """Test to_dict serialization."""
        status = AgentStatus(
            provider="ollama",
            model="llama3",
            mode=AgentMode.EXPLORE,
            connected=True,
            tools_available=30,
            conversation_length=10,
        )
        d = status.to_dict()
        assert d["provider"] == "ollama"
        assert d["model"] == "llama3"
        assert d["mode"] == "explore"
        assert d["connected"] is True


# =============================================================================
# VICTOR PROTOCOL TESTS
# =============================================================================


class ConcreteProtocol(VictorProtocol):
    """Concrete implementation for testing."""

    async def chat(self, messages):
        return ChatResponse(content="Test response")

    async def stream_chat(self, messages):
        yield StreamChunk(content="Test")

    async def reset_conversation(self):
        pass

    async def semantic_search(self, query, max_results=10):
        return []

    async def code_search(self, query, regex=False, case_sensitive=False, file_pattern=None):
        return []

    async def switch_model(self, provider, model):
        pass

    async def switch_mode(self, mode):
        pass

    async def get_status(self):
        return AgentStatus(
            provider="test",
            model="test-model",
            mode=AgentMode.BUILD,
            connected=True,
            tools_available=1,
            conversation_length=0,
        )

    async def undo(self):
        return UndoRedoResult(success=True, message="Undone")

    async def redo(self):
        return UndoRedoResult(success=True, message="Redone")

    async def get_history(self, limit=10):
        return []

    async def apply_patch(self, patch, dry_run=False):
        return {"success": True}

    async def close(self):
        pass


class FailingProtocol(ConcreteProtocol):
    """Protocol that fails health check."""

    async def get_status(self):
        raise ConnectionError("Not connected")


class TestVictorProtocol:
    """Tests for VictorProtocol abstract class."""

    @pytest.mark.asyncio
    async def test_concrete_implementation_works(self):
        """Test that concrete implementation works."""
        proto = ConcreteProtocol()
        resp = await proto.chat([ChatMessage("user", "Hi")])
        assert resp.content == "Test response"

    @pytest.mark.asyncio
    async def test_get_definition_default(self):
        """Test default get_definition returns empty list."""
        proto = ConcreteProtocol()
        result = await proto.get_definition("test.py", 0, 0)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_references_default(self):
        """Test default get_references returns empty list."""
        proto = ConcreteProtocol()
        result = await proto.get_references("test.py", 0, 0)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_hover_default(self):
        """Test default get_hover returns None."""
        proto = ConcreteProtocol()
        result = await proto.get_hover("test.py", 0, 0)
        assert result is None

    @pytest.mark.asyncio
    async def test_check_health_success(self):
        """Test check_health returns True when healthy."""
        proto = ConcreteProtocol()
        result = await proto.check_health()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_health_failure(self):
        """Test check_health returns False when unhealthy."""
        proto = FailingProtocol()
        result = await proto.check_health()
        assert result is False
