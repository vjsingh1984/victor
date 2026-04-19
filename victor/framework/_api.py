"""Core eager imports for the Victor framework public API.

This module contains all eagerly-imported symbols that form the core
public API surface of victor.framework. Extracted from __init__.py
for maintainability.
"""

from victor.framework.agent import Agent, ChatSession
from victor.framework.config import AgentConfig
from victor.framework.protocols import (
    ChunkType,
    ConversationStateProtocol,
    FrameworkOrchestratorProtocol,
    MessagesProtocol,
    OrchestratorStreamChunk,
    ProviderProtocol,
    StreamingProtocol,
    SystemPromptProtocol,
    ToolsProtocol,
    verify_protocol_conformance,
)

# Backward compatibility alias
OrchestratorProtocol = FrameworkOrchestratorProtocol
from victor.framework.errors import (
    AgentError,
    BudgetExhaustedError,
    CancellationError,
    ConfigurationError,
    ProviderError,
    StateTransitionError,
    ToolError,
)
from victor.framework.events import (
    AgentExecutionEvent,
    EventType,
    content_event,
    error_event,
    milestone_event,
    progress_event,
    stage_change_event,
    stream_end_event,
    stream_start_event,
    thinking_event,
    tool_call_event,
    tool_error_event,
    tool_result_event,
)
from victor.framework.state import Stage, State, StateHooks, StateObserver
from victor.framework.task import FrameworkTaskType, Task, TaskResult
from victor.framework.shim import FrameworkShim, get_vertical, list_verticals
from victor.framework.tools import ToolCategory, Tools, ToolSet, ToolsInput
from victor.framework.decorators import agent, task as task_decorator, AgentCallable, TaskDefinition

PUBLIC_API_NAMES = [
    # Decorator API
    "agent",
    "task_decorator",
    "AgentCallable",
    "TaskDefinition",
    # Core classes (the 5 concepts)
    "Agent",
    "Task",
    "Tools",
    "State",
    "AgentExecutionEvent",
    # Agent
    "ChatSession",
    # Task
    "TaskResult",
    "FrameworkTaskType",
    # Tools
    "ToolSet",
    "ToolCategory",
    "ToolsInput",
    # State
    "Stage",
    "StateHooks",
    "StateObserver",
    # Protocols
    "FrameworkOrchestratorProtocol",
    "OrchestratorProtocol",  # Backward compatibility alias
    "ConversationStateProtocol",
    "ProviderProtocol",
    "ToolsProtocol",
    "SystemPromptProtocol",
    "MessagesProtocol",
    "StreamingProtocol",
    "OrchestratorStreamChunk",
    "ChunkType",
    "verify_protocol_conformance",
    # Events
    "EventType",
    "content_event",
    "thinking_event",
    "tool_call_event",
    "tool_result_event",
    "tool_error_event",
    "stage_change_event",
    "stream_start_event",
    "stream_end_event",
    "error_event",
    "progress_event",
    "milestone_event",
    # Config
    "AgentConfig",
    # Shim
    "FrameworkShim",
    "get_vertical",
    "list_verticals",
    # Errors
    "AgentError",
    "ProviderError",
    "ToolError",
    "ConfigurationError",
    "BudgetExhaustedError",
    "CancellationError",
    "StateTransitionError",
]

__all__ = PUBLIC_API_NAMES
