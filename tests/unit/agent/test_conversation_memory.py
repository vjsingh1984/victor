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

"""Unit tests for conversation_memory module.

Tests for model metadata parsing, known model lookups, and message serialization.
"""

import pytest
from datetime import datetime
from victor.agent.conversation_memory import (
    MessageRole,
    MessagePriority,
    ModelFamily,
    ModelSize,
    ContextSize,
    ModelMetadata,
    parse_model_metadata,
    get_known_model_context,
    get_known_model_params,
    ConversationMessage,
)

# =============================================================================
# MessageRole Tests
# =============================================================================


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_message_role_values(self):
        """MessageRole has expected values."""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL_CALL.value == "tool_call"
        assert MessageRole.TOOL_RESULT.value == "tool_result"


# =============================================================================
# MessagePriority Tests
# =============================================================================


class TestMessagePriority:
    """Tests for MessagePriority enum."""

    def test_priority_ordering(self):
        """Priority levels have correct numeric values."""
        assert MessagePriority.CRITICAL.value == 100
        assert MessagePriority.HIGH.value == 75
        assert MessagePriority.MEDIUM.value == 50
        assert MessagePriority.LOW.value == 25
        assert MessagePriority.EPHEMERAL.value == 0


# =============================================================================
# ModelFamily Tests
# =============================================================================


class TestModelFamily:
    """Tests for ModelFamily enum."""

    def test_model_families_exist(self):
        """All expected model families exist."""
        assert ModelFamily.LLAMA.value == "llama"
        assert ModelFamily.QWEN.value == "qwen"
        assert ModelFamily.MISTRAL.value == "mistral"
        assert ModelFamily.MIXTRAL.value == "mixtral"
        assert ModelFamily.CLAUDE.value == "claude"
        assert ModelFamily.GPT.value == "gpt"
        assert ModelFamily.GEMINI.value == "gemini"
        assert ModelFamily.DEEPSEEK.value == "deepseek"


# =============================================================================
# ModelSize Tests
# =============================================================================


class TestModelSize:
    """Tests for ModelSize enum."""

    def test_size_categories(self):
        """Model size categories have correct values."""
        assert ModelSize.TINY.value == "tiny"
        assert ModelSize.SMALL.value == "small"
        assert ModelSize.MEDIUM.value == "medium"
        assert ModelSize.LARGE.value == "large"
        assert ModelSize.XLARGE.value == "xlarge"


# =============================================================================
# ContextSize Tests
# =============================================================================


class TestContextSize:
    """Tests for ContextSize enum."""

    def test_context_size_values(self):
        """Context size categories have correct values."""
        assert ContextSize.SMALL.value == "small"
        assert ContextSize.MEDIUM.value == "medium"
        assert ContextSize.LARGE.value == "large"
        assert ContextSize.XLARGE.value == "xlarge"


# =============================================================================
# parse_model_metadata Tests
# =============================================================================


class TestParseModelMetadataLlama:
    """Tests for parsing Llama model names."""

    def test_parse_llama_3_3_70b(self):
        """Parse llama-3.3-70b model."""
        metadata = parse_model_metadata("llama-3.3-70b-versatile")

        assert metadata.model_family == ModelFamily.LLAMA
        assert metadata.model_size == ModelSize.XLARGE  # 70B is >= 70
        assert metadata.model_params_b == 70.0
        assert metadata.is_moe is False
        assert metadata.is_reasoning is False

    def test_parse_llama_3_1_8b(self):
        """Parse llama-3.1-8b model."""
        metadata = parse_model_metadata("llama-3.1-8b-instant")

        assert metadata.model_family == ModelFamily.LLAMA
        assert metadata.model_size == ModelSize.MEDIUM  # 8B is >= 8
        assert metadata.model_params_b == 8.0

    def test_parse_llama_generic(self):
        """Parse generic llama model."""
        metadata = parse_model_metadata("llama-2-13b")

        assert metadata.model_family == ModelFamily.LLAMA
        assert metadata.model_size == ModelSize.MEDIUM
        assert metadata.model_params_b == 13.0


class TestParseModelMetadataMixtral:
    """Tests for parsing Mixtral model names."""

    def test_parse_mixtral_8x7b(self):
        """Parse mixtral-8x7b model."""
        metadata = parse_model_metadata("mixtral-8x7b-32768")

        assert metadata.model_family == ModelFamily.MIXTRAL
        assert metadata.is_moe is True
        # MoE effective params
        assert metadata.model_params_b == 46.7
        assert metadata.model_size == ModelSize.LARGE  # 46.7B is >= 32

    def test_parse_mixtral_generic(self):
        """Parse generic mixtral model."""
        metadata = parse_model_metadata("mixtral-large")

        assert metadata.model_family == ModelFamily.MIXTRAL


class TestParseModelMetadataClaude:
    """Tests for parsing Claude model names."""

    def test_parse_claude_3_opus(self):
        """Parse claude-3-opus model."""
        metadata = parse_model_metadata("claude-3-opus")

        assert metadata.model_family == ModelFamily.CLAUDE
        assert metadata.model_params_b == 200.0
        assert metadata.model_size == ModelSize.XXLARGE  # 200B is >= 175

    def test_parse_claude_3_sonnet(self):
        """Parse claude-3-sonnet model."""
        metadata = parse_model_metadata("claude-3-sonnet")

        assert metadata.model_family == ModelFamily.CLAUDE
        assert metadata.model_params_b == 70.0
        assert metadata.model_size == ModelSize.XLARGE  # 70B is >= 70

    def test_parse_claude_3_5_sonnet(self):
        """Parse claude-3.5-sonnet model."""
        metadata = parse_model_metadata("claude-3.5-sonnet")

        assert metadata.model_family == ModelFamily.CLAUDE
        assert metadata.model_params_b == 70.0

    def test_parse_claude_3_haiku(self):
        """Parse claude-3-haiku model."""
        metadata = parse_model_metadata("claude-3-haiku")

        assert metadata.model_family == ModelFamily.CLAUDE
        assert metadata.model_params_b == 20.0
        assert metadata.model_size == ModelSize.MEDIUM


class TestParseModelMetadataGPT:
    """Tests for parsing GPT model names."""

    def test_parse_gpt_4(self):
        """Parse gpt-4 model."""
        metadata = parse_model_metadata("gpt-4-turbo")

        assert metadata.model_family == ModelFamily.GPT
        assert metadata.model_params_b == 175.0

    def test_parse_gpt_3_5(self):
        """Parse gpt-3.5-turbo model."""
        metadata = parse_model_metadata("gpt-3.5-turbo")

        assert metadata.model_family == ModelFamily.GPT
        assert metadata.model_params_b == 175.0


class TestParseModelMetadataQwen:
    """Tests for parsing Qwen model names."""

    def test_parse_qwen_2_5(self):
        """Parse qwen2.5 model."""
        metadata = parse_model_metadata("qwen2.5-coder:32b")

        assert metadata.model_family == ModelFamily.QWEN
        assert metadata.model_params_b == 32.0
        assert metadata.model_size == ModelSize.LARGE  # 32B is >= 32

    def test_parse_qwen_3(self):
        """Parse qwen3 model."""
        metadata = parse_model_metadata("qwen3:32b")

        assert metadata.model_family == ModelFamily.QWEN


class TestParseModelMetadataDeepSeek:
    """Tests for parsing DeepSeek model names."""

    def test_parse_deepseek_r1(self):
        """Parse deepseek-r1 reasoning model."""
        metadata = parse_model_metadata("deepseek-r1:32b")

        assert metadata.model_family == ModelFamily.DEEPSEEK
        assert metadata.is_reasoning is True
        assert metadata.model_params_b == 32.0

    def test_parse_deepseek_chat(self):
        """Parse deepseek-chat model."""
        metadata = parse_model_metadata("deepseek-chat")

        assert metadata.model_family == ModelFamily.DEEPSEEK
        assert metadata.is_reasoning is False


class TestParseModelMetadataContext:
    """Tests for context window parsing."""

    def test_parse_32k_context(self):
        """Parse 32k context window (with 'k' suffix)."""
        metadata = parse_model_metadata("mixtral-8x7b-32k")

        assert metadata.context_tokens == 32768  # 32 * 1024
        assert metadata.context_size == ContextSize.LARGE  # 32768 is >= 32000

    def test_parse_128k_context(self):
        """Parse 128k context window (with 'k' suffix)."""
        metadata = parse_model_metadata("llama-3.3-70b-128k")

        assert metadata.context_tokens == 131072  # 128 * 1024
        assert metadata.context_size == ContextSize.XLARGE

    def test_default_claude_context(self):
        """Default Claude context is 200k."""
        metadata = parse_model_metadata("claude-3-sonnet")

        assert metadata.context_tokens == 200000
        assert metadata.context_size == ContextSize.XLARGE

    def test_default_gpt_4_context(self):
        """Default GPT-4 context is 128k."""
        metadata = parse_model_metadata("gpt-4-turbo")

        assert metadata.context_tokens == 128000

    def test_default_gpt_3_5_context(self):
        """Default GPT-3.5 context is 16k."""
        metadata = parse_model_metadata("gpt-3.5-turbo")

        assert metadata.context_tokens == 16000
        assert metadata.context_size == ContextSize.MEDIUM


class TestParseModelMetadataProvider:
    """Tests for provider-based fallback."""

    def test_anthropic_provider_fallback(self):
        """Provider hint helps identify Claude models."""
        metadata = parse_model_metadata("unknown-model", provider="anthropic")

        assert metadata.model_family == ModelFamily.CLAUDE

    def test_openai_provider_fallback(self):
        """Provider hint helps identify GPT models."""
        metadata = parse_model_metadata("unknown-model", provider="openai")

        assert metadata.model_family == ModelFamily.GPT

    def test_google_provider_fallback(self):
        """Provider hint helps identify Gemini models."""
        metadata = parse_model_metadata("unknown-model", provider="google")

        assert metadata.model_family == ModelFamily.GEMINI


class TestParseModelMetadataKnownOverrides:
    """Tests for known model overrides."""

    def test_known_context_override(self):
        """Known context window overrides parsed value."""
        metadata = parse_model_metadata("llama-3.3-70b-versatile", known_context=999999)

        assert metadata.context_tokens == 999999

    def test_known_params_override(self):
        """Known parameter count overrides parsed value."""
        metadata = parse_model_metadata("llama-3.3-70b-versatile", known_params_b=99.9)

        assert metadata.model_params_b == 99.9


# =============================================================================
# get_known_model_context Tests
# =============================================================================


class TestGetKnownModelContext:
    """Tests for get_known_model_context function."""

    def test_known_llama_context(self):
        """Get context for known Llama model."""
        context = get_known_model_context("llama-3.3-70b-versatile")

        assert context == 128000

    def test_known_claude_context(self):
        """Get context for known Claude model."""
        context = get_known_model_context("claude-3-opus")

        assert context == 200000

    def test_known_gpt_context(self):
        """Get context for known GPT model."""
        context = get_known_model_context("gpt-4o")

        assert context == 128000

    def test_known_gemini_context(self):
        """Get context for known Gemini model."""
        context = get_known_model_context("gemini-1.5-pro")

        assert context == 1000000

    def test_unknown_model_returns_none(self):
        """Unknown model returns None."""
        context = get_known_model_context("unknown-model-xyz")

        assert context is None


# =============================================================================
# get_known_model_params Tests
# =============================================================================


class TestGetKnownModelParams:
    """Tests for get_known_model_params function."""

    def test_known_llama_params(self):
        """Get params for known Llama model."""
        params = get_known_model_params("llama-3.3-70b-versatile")

        assert params == 70.0

    def test_known_mixtral_params(self):
        """Get params for known Mixtral model (MoE effective)."""
        params = get_known_model_params("mixtral-8x7b-32768")

        assert params == 46.7

    def test_known_deepseek_params(self):
        """Get params for known DeepSeek model."""
        params = get_known_model_params("deepseek-r1:32b")

        assert params == 32.0

    def test_unknown_model_returns_none(self):
        """Unknown model returns None."""
        params = get_known_model_params("unknown-model-xyz")

        assert params is None


# =============================================================================
# ConversationMessage Tests
# =============================================================================


class TestConversationMessageInit:
    """Tests for ConversationMessage initialization."""

    def test_init_minimal(self):
        """Initialize with minimal required fields."""
        msg = ConversationMessage(
            id="msg-1",
            role=MessageRole.USER,
            content="Hello",
            timestamp=datetime.now(),
            token_count=5,
        )

        assert msg.id == "msg-1"
        assert msg.role == MessageRole.USER
        assert msg.priority == MessagePriority.MEDIUM  # Default
        assert msg.tool_name is None
        assert msg.tool_call_id is None

    def test_init_with_all_fields(self):
        """Initialize with all fields."""
        now = datetime.now()
        msg = ConversationMessage(
            id="msg-1",
            role=MessageRole.TOOL_CALL,
            content="call_tool",
            timestamp=now,
            token_count=10,
            priority=MessagePriority.HIGH,
            metadata={"key": "value"},
            tool_name="read_file",
            tool_call_id="call-123",
        )

        assert msg.tool_name == "read_file"
        assert msg.tool_call_id == "call-123"
        assert msg.metadata == {"key": "value"}


class TestConversationMessageSerialization:
    """Tests for message serialization."""

    def test_to_provider_format_user(self):
        """Convert user message to provider format."""
        msg = ConversationMessage(
            id="msg-1",
            role=MessageRole.USER,
            content="Hello",
            timestamp=datetime.now(),
            token_count=5,
        )

        provider_format = msg.to_provider_format()

        assert provider_format["role"] == "user"
        assert provider_format["content"] == "Hello"

    def test_to_provider_format_assistant(self):
        """Convert assistant message to provider format."""
        msg = ConversationMessage(
            id="msg-1",
            role=MessageRole.ASSISTANT,
            content="Hi there!",
            timestamp=datetime.now(),
            token_count=7,
        )

        provider_format = msg.to_provider_format()

        assert provider_format["role"] == "assistant"
        assert provider_format["content"] == "Hi there!"

    def test_to_provider_format_tool_result(self):
        """Convert tool result to provider format."""
        msg = ConversationMessage(
            id="msg-1",
            role=MessageRole.TOOL_RESULT,
            content="Result",
            timestamp=datetime.now(),
            token_count=6,
            tool_call_id="call-123",
        )

        provider_format = msg.to_provider_format()

        assert provider_format["role"] == "assistant"  # Tool results map to assistant
        assert provider_format["tool_call_id"] == "call-123"

    def test_to_dict_roundtrip(self):
        """Serialize to dict and back."""
        now = datetime.now()
        original = ConversationMessage(
            id="msg-1",
            role=MessageRole.USER,
            content="Test",
            timestamp=now,
            token_count=4,
            priority=MessagePriority.HIGH,
            tool_name="test_tool",
            metadata={"key": "value"},
        )

        # Convert to dict
        msg_dict = original.to_dict()

        # Convert back
        restored = ConversationMessage.from_dict(msg_dict)

        assert restored.id == original.id
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.priority == original.priority
        assert restored.tool_name == original.tool_name
        assert restored.metadata == original.metadata
