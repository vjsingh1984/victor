# Copyright 2025 Vijaykumar Singh <singhvijd@gmail.com>
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

"""Type stubs and common type definitions for Victor.

This module provides TypedDict and Protocol definitions for common types
used throughout the codebase. These improve type safety and IDE support.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, Protocol
from typing_extensions import Required, NotRequired


# =============================================================================
# Tool Calling Types
# =============================================================================


class ToolCall(TypedDict):
    """Representation of a tool/function call from an LLM.

    Attributes:
        name: The name of the tool/function to call
        arguments: Parameters to pass to the tool (JSON-serializable)
        id: Optional unique identifier for this call (for response tracking)
    """

    name: str
    arguments: Dict[str, Any]
    id: NotRequired[str]


class ToolCallResult(TypedDict):
    """Result of a tool execution.

    Attributes:
        tool_call: The original tool call that produced this result
        result: The return value from the tool execution
        success: Whether the tool executed successfully
        error: Optional error message if execution failed
        execution_time: Optional execution time in seconds
    """

    tool_call: ToolCall
    result: Any
    success: bool
    error: NotRequired[Optional[str]]
    execution_time: NotRequired[Optional[float]]


class ToolExecutionContext(TypedDict):
    """Context information for tool execution.

    Attributes:
        iteration: Current iteration number in the conversation
        budget_remaining: Remaining tool budget
        max_tools: Maximum tools allowed in this context
        categories: Optional list of tool categories to restrict to
        exclude_tools: Optional list of specific tools to exclude
    """

    iteration: int
    budget_remaining: int
    max_tools: int
    categories: NotRequired[Optional[List[str]]]
    exclude_tools: NotRequired[Optional[List[str]]]


# =============================================================================
# Agent Selection Context Types
# =============================================================================


class AgentToolSelectionContext(TypedDict):
    """Context for agent tool selection decisions.

    Attributes:
        max_tools: Maximum number of tools to select
        categories: Optional list of tool categories to consider
        exclude_tools: Optional list of tools to exclude from selection
        budget: Optional budget constraint for tool execution
        task_complexity: Optional complexity classification of the task
        intent: Optional classified intent of the user query
    """

    max_tools: int
    categories: NotRequired[Optional[List[str]]]
    exclude_tools: NotRequired[Optional[List[str]]]
    budget: NotRequired[Optional[int]]
    task_complexity: NotRequired[Optional[str]]
    intent: NotRequired[Optional[str]]


# =============================================================================
# Conversation Types
# =============================================================================


class ConversationStage:
    """Enumeration of conversation stages.

    The agent progresses through these stages during a conversation:
    - INITIAL: Starting state, no user input yet
    - PLANNING: Analyzing requirements and planning approach
    - READING: Reading files and gathering context
    - ANALYSIS: Analyzing codebase and understanding structure
    - EXECUTION: Executing tools and making changes
    - VERIFICATION: Verifying changes and testing
    - COMPLETION: Task complete, ready for next query
    """

    INITIAL = "initial"
    PLANNING = "planning"
    READING = "reading"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    COMPLETION = "completion"


ConversationStageType = Literal[
    "initial",
    "planning",
    "reading",
    "analysis",
    "execution",
    "verification",
    "completion",
]


class ConversationContext(TypedDict):
    """Context information about the current conversation.

    Attributes:
        stage: Current conversation stage
        messages: List of messages in the conversation
        tool_calls: List of tool calls made in this conversation
        iteration: Current iteration number
        tokens_used: Total tokens used so far
        budget_remaining: Remaining token budget
    """

    stage: str
    messages: List[Dict[str, Any]]
    tool_calls: List[ToolCall]
    iteration: int
    tokens_used: int
    budget_remaining: int


# =============================================================================
# Provider Types
# =============================================================================


class ProviderCapabilities(TypedDict):
    """Capabilities of an LLM provider.

    Attributes:
        supports_streaming: Whether provider supports streaming responses
        supports_tools: Whether provider supports tool/function calling
        supports_parallel_tools: Whether provider supports parallel tool calls
        supports_vision: Whether provider supports vision/multimodal inputs
        max_tokens: Maximum tokens supported by provider
        input_cost_per_1k: Cost per 1k input tokens
        output_cost_per_1k: Cost per 1k output tokens
    """

    supports_streaming: bool
    supports_tools: bool
    supports_parallel_tools: bool
    supports_vision: bool
    max_tokens: int
    input_cost_per_1k: NotRequired[Optional[float]]
    output_cost_per_1k: NotRequired[Optional[float]]


class ModelCapabilities(TypedDict):
    """Capabilities of a specific model.

    Attributes:
        model: Model identifier
        training_cutoff: Training data cutoff date
        tool_calling: Whether model supports tool calling
        native_tool_calls: Whether model has native tool call format
        parallel_tool_calls: Whether model supports parallel tool calls
        supports_vision: Whether model supports vision inputs
        max_tokens: Maximum context length
    """

    model: str
    training_cutoff: NotRequired[Optional[str]]
    tool_calling: bool
    native_tool_calls: NotRequired[bool]
    parallel_tool_calls: NotRequired[bool]
    supports_vision: NotRequired[bool]
    max_tokens: int


# =============================================================================
# Orchestrator Types
# =============================================================================


class OrchestratorContext(TypedDict):
    """Context for orchestrator operations.

    Attributes:
        settings: Application settings
        provider: Current LLM provider
        model: Current model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        console: Optional Rich console instance
        tool_selection: Optional tool selection configuration
        thinking: Whether extended thinking is enabled
        provider_name: Optional provider name for disambiguation
        profile_name: Optional profile name for session tracking
    """

    settings: Any  # Settings from victor.config.settings
    provider: Any  # BaseProvider
    model: str
    temperature: float
    max_tokens: int
    console: NotRequired[Optional[Any]]  # Optional[Console]
    tool_selection: NotRequired[Optional[Dict[str, Any]]]
    thinking: NotRequired[bool]
    provider_name: NotRequired[Optional[str]]
    profile_name: NotRequired[Optional[str]]


# =============================================================================
# Message Types
# =============================================================================


class Message(TypedDict):
    """A message in a conversation.

    Attributes:
        role: Message role (system, user, assistant, tool)
        content: Message content (string or multimodal content)
        tool_calls: Optional list of tool calls (for assistant messages)
        tool_call_id: Optional tool call ID (for tool messages)
        name: Optional name (for tool messages)
    """

    role: str
    content: str
    tool_calls: NotRequired[Optional[List[ToolCall]]]
    tool_call_id: NotRequired[Optional[str]]
    name: NotRequired[Optional[str]]


# =============================================================================
# Streaming Types
# =============================================================================


class StreamChunk(TypedDict):
    """A chunk of streamed response.

    Attributes:
        content: Partial content chunk
        delta: Delta since last chunk
        finish_reason: Optional reason for finishing (stop, length, error)
        usage: Optional token usage information
        is_complete: Whether this is the final chunk
    """

    content: str
    delta: NotRequired[str]
    finish_reason: NotRequired[Optional[str]]
    usage: NotRequired[Optional[Dict[str, int]]]
    is_complete: NotRequired[bool]


# =============================================================================
# Completion Types
# =============================================================================


class CompletionResponse(TypedDict):
    """Response from a non-streaming completion.

    Attributes:
        content: Generated content
        model: Model that generated the response
        usage: Token usage information
        finish_reason: Reason for finishing
        tool_calls: Optional list of tool calls made by the model
    """

    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: NotRequired[Optional[str]]
    tool_calls: NotRequired[Optional[List[ToolCall]]]
    # Additional fields for compatibility with different providers
    role: NotRequired[str]
    stop_reason: NotRequired[str]
    raw_response: NotRequired[Any]
    metadata: NotRequired[Any]


# =============================================================================
# Error Types
# =============================================================================


class ErrorContext(TypedDict):
    """Context information for error handling.

    Attributes:
        error_type: Type of error that occurred
        error_message: Error message
        retry_count: Number of retry attempts made
        last_successful_operation: Last operation that succeeded
        context: Additional context information
    """

    error_type: str
    error_message: str
    retry_count: int
    last_successful_operation: NotRequired[Optional[str]]
    context: NotRequired[Dict[str, Any]]


# =============================================================================
# Import Literal at runtime
# =============================================================================

from typing import Literal
