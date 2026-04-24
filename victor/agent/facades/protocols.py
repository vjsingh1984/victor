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

"""Protocol definitions for domain facades.

Each protocol specifies the public API surface that the orchestrator (and
other consumers) depend on. Concrete facade implementations satisfy these
protocols structurally via Python's Protocol system.

Design Principles:
    - ISP: Each protocol exposes only what its consumers need.
    - DIP: Consumers depend on protocols, not concrete classes.
    - Future-proof: New facades or facade methods require only a protocol
      update; no consumer changes unless they use the new surface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ChatFacadeProtocol(Protocol):
    """Protocol for the chat/conversation domain facade.

    Manages message history, conversation state, memory integration,
    semantic embedding, intent classification, and response completion.
    """

    @property
    def conversation(self) -> Any:
        """Message history (MessageHistory instance)."""
        ...

    @property
    def conversation_controller(self) -> Any:
        """ConversationController for dialogue state management."""
        ...

    @property
    def conversation_state(self) -> Any:
        """ConversationStateMachine for stage transitions."""
        ...

    @property
    def memory_manager(self) -> Optional[Any]:
        """Optional conversation memory store."""
        ...

    @property
    def embedding_store(self) -> Optional[Any]:
        """Optional conversation embedding store."""
        ...

    @property
    def intent_classifier(self) -> Optional[Any]:
        """Optional intent classifier."""
        ...


@runtime_checkable
class ToolFacadeProtocol(Protocol):
    """Protocol for the tool execution domain facade.

    Manages tool registry, execution pipeline, selection strategy,
    caching, deduplication, output formatting, and budgeting.
    """

    @property
    def tools(self) -> Any:
        """Tool registry (SharedToolRegistry instance)."""
        ...

    @property
    def tool_pipeline(self) -> Any:
        """ToolPipeline for execution orchestration."""
        ...

    @property
    def tool_executor(self) -> Any:
        """ToolExecutor for individual tool invocation."""
        ...

    @property
    def tool_selector(self) -> Any:
        """Tool selection strategy."""
        ...

    @property
    def tool_cache(self) -> Optional[Any]:
        """Optional tool result cache."""
        ...


@runtime_checkable
class ProviderFacadeProtocol(Protocol):
    """Protocol for the LLM provider domain facade.

    Manages provider instance, model selection, switching,
    health monitoring, and rate coordination.
    """

    @property
    def provider(self) -> Any:
        """Active LLM provider instance."""
        ...

    @property
    def model(self) -> str:
        """Active model identifier."""
        ...

    @property
    def provider_manager(self) -> Any:
        """ProviderManager for lifecycle management."""
        ...


@runtime_checkable
class SessionFacadeProtocol(Protocol):
    """Protocol for the session state domain facade.

    Manages session state, ledger, persistence, checkpoint,
    and lifecycle coordination.
    """

    @property
    def session_state(self) -> Any:
        """SessionState for tracking tool calls, budget, etc."""
        ...

    @property
    def session_ledger(self) -> Any:
        """SessionLedger for append-only event log."""
        ...

    @property
    def lifecycle_manager(self) -> Optional[Any]:
        """Optional lifecycle manager for session boundaries."""
        ...


@runtime_checkable
class MetricsFacadeProtocol(Protocol):
    """Protocol for the metrics/observability domain facade.

    Manages usage tracking, analytics, streaming metrics,
    cost accounting, and debug logging.
    """

    @property
    def usage_analytics(self) -> Optional[Any]:
        """Optional usage analytics collector."""
        ...

    @property
    def usage_logger(self) -> Optional[Any]:
        """Optional usage logger."""
        ...

    @property
    def metrics_collector(self) -> Optional[Any]:
        """Optional metrics collector."""
        ...


@runtime_checkable
class ResilienceFacadeProtocol(Protocol):
    """Protocol for the resilience domain facade.

    Manages error recovery, context management, RL coordination,
    code execution, and background task management.
    """

    @property
    def recovery_handler(self) -> Optional[Any]:
        """Optional recovery handler."""
        ...

    @property
    def context_manager(self) -> Optional[Any]:
        """Optional context window manager."""
        ...


@runtime_checkable
class WorkflowFacadeProtocol(Protocol):
    """Protocol for the workflow domain facade.

    Manages workflow registry, execution, optimization,
    and mode-workflow-team coordination.
    """

    @property
    def workflow_registry(self) -> Optional[Any]:
        """Optional workflow registry."""
        ...


@runtime_checkable
class OrchestrationFacadeProtocol(Protocol):
    """Protocol for the top-level orchestration domain facade.

    Manages coordinators, protocol adapters, streaming handlers,
    intelligent pipeline integration, and subagent orchestration.
    """

    @property
    def protocol_adapter(self) -> Any:
        """Protocol adapter for coordinator communication."""
        ...

    @property
    def chat_stream_runtime(self) -> Optional[Any]:
        """Canonical service-owned chat streaming runtime."""
        ...

    @property
    def task_analyzer(self) -> Optional[Any]:
        """Optional task analyzer."""
        ...
