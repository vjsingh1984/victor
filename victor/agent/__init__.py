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
from victor.agent.config_loader import ConfigLoader, CORE_TOOLS
from victor.agent.conversation import ConversationManager
from victor.agent.observability import (
    ObservabilityManager,
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

__all__ = [
    "AgentOrchestrator",
    "ArgumentNormalizer",
    "NormalizationStrategy",
    "ConfigLoader",
    "CORE_TOOLS",
    "ConversationManager",
    # Observability
    "ObservabilityManager",
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
    # Tool execution
    "ToolExecutor",
    "ToolExecutionResult",
]
