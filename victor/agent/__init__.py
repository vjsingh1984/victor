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

"""Agent module - orchestrator and supporting components."""

from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
from victor.agent.config_loader import ConfigLoader
from victor.agent.tool_selection import get_critical_tools
from victor.agent.message_history import MessageHistory
from victor.agent.observability import (
    TracingProvider,
    Span,
    SpanKind,
    SpanStatus,
    get_observability,
    set_observability,
    traced,
)
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.stream_handler import StreamHandler, StreamResult, StreamMetrics, StreamBuffer
from victor.agent.tool_executor import ToolExecutor, ToolExecutionResult

# New decomposed components
from victor.agent.conversation_controller import (
    ConversationController,
    ConversationConfig,
    ContextMetrics,
)
from victor.agent.tool_pipeline import (
    ToolPipeline,
    ToolPipelineConfig,
    ToolCallResult,
    PipelineExecutionResult,
)
from victor.agent.streaming_controller import (
    StreamingController,
    StreamingControllerConfig,
    StreamingSession,
)
from victor.agent.task_analyzer import (
    TaskAnalyzer,
    TaskAnalysis,
    get_task_analyzer,
    reset_task_analyzer,
)

# Intelligent agent components (RL-based learning, quality scoring, prompt optimization)
from victor.agent.intelligent_pipeline import (
    IntelligentAgentPipeline,
    RequestContext,
    ResponseResult,
    PipelineStats,
    get_pipeline,
    clear_pipeline_cache,
)
from victor.agent.orchestrator_integration import (
    OrchestratorIntegration,
    IntegrationConfig,
    IntegrationMetrics,
    enhance_orchestrator,
)

__all__ = [
    "AgentOrchestrator",
    "ArgumentNormalizer",
    "NormalizationStrategy",
    "ConfigLoader",
    "get_critical_tools",
    # Conversation
    "MessageHistory",
    "ConversationController",
    "ConversationConfig",
    "ContextMetrics",
    # Observability
    "TracingProvider",
    "Span",
    "SpanKind",
    "SpanStatus",
    "get_observability",
    "set_observability",
    "traced",
    # Streaming
    "StreamHandler",
    "StreamResult",
    "StreamMetrics",
    "StreamBuffer",
    "StreamingController",
    "StreamingControllerConfig",
    "StreamingSession",
    # Tool execution
    "ToolExecutor",
    "ToolExecutionResult",
    "ToolPipeline",
    "ToolPipelineConfig",
    "ToolCallResult",
    "PipelineExecutionResult",
    # Task analysis
    "TaskAnalyzer",
    "TaskAnalysis",
    "get_task_analyzer",
    "reset_task_analyzer",
    # Intelligent Pipeline
    "IntelligentAgentPipeline",
    "RequestContext",
    "ResponseResult",
    "PipelineStats",
    "get_pipeline",
    "clear_pipeline_cache",
    # Orchestrator Integration
    "OrchestratorIntegration",
    "IntegrationConfig",
    "IntegrationMetrics",
    "enhance_orchestrator",
]
