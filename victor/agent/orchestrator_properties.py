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

"""Property facade for AgentOrchestrator.

Extracts ~25 property definitions from AgentOrchestrator to reduce its LOC
while preserving the same access patterns.  Properties are installed onto
the AgentOrchestrator class via ``install_properties()``, keeping full
compatibility with ``unittest.mock.patch.object`` and introspection.

Groups:
  1. Simple component accessors -- return self._xxx directly
  2. Lazy coordinator constructors -- check-and-create on first access
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.recovery_coordinator import StreamingRecoveryCoordinator
    from victor.agent.chunk_generator import ChunkGenerator
    from victor.agent.tool_planner import ToolPlanner
    from victor.agent.task_coordinator import TaskCoordinator
    from victor.agent.orchestrator_integration import OrchestratorIntegration
    from victor.agent.conversation_controller import ConversationController
    from victor.agent.tool_pipeline import ToolPipeline
    from victor.agent.streaming_controller import StreamingController
    from victor.agent.task_analyzer import TaskAnalyzer
    from victor.agent.provider_manager import ProviderManager
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.tool_output_formatter import ToolOutputFormatter
    from victor.agent.usage_analytics import UsageAnalytics
    from victor.agent.tool_sequence_tracker import ToolSequenceTracker
    from victor.agent.streaming.handler import StreamingChatHandler
    from victor.verticals.vertical_context import VerticalContext
    from victor.agent.recovery import RecoveryHandler
    from victor.agent.orchestrator_recovery import OrchestratorRecoveryIntegration
    from victor.agent.session_state_manager import SessionStateManager

logger = logging.getLogger(__name__)


# =====================================================================
# Group 1: Simple component accessor properties
# =====================================================================

def _conversation_controller(self: "AgentOrchestrator") -> "ConversationController":
    """Get the conversation controller component."""
    return self._conversation_controller


def _tool_pipeline(self: "AgentOrchestrator") -> "ToolPipeline":
    """Get the tool pipeline component."""
    return self._tool_pipeline


def _streaming_controller(self: "AgentOrchestrator") -> "StreamingController":
    """Get the streaming controller component."""
    return self._streaming_controller


def _streaming_handler(self: "AgentOrchestrator") -> "StreamingChatHandler":
    """Get the streaming chat handler component."""
    return self._streaming_handler


def _task_analyzer(self: "AgentOrchestrator") -> "TaskAnalyzer":
    """Get the task analyzer component."""
    return self._task_analyzer


def _provider_manager(self: "AgentOrchestrator") -> "ProviderManager":
    """Get the provider manager component."""
    return self._provider_manager


def _context_compactor(self: "AgentOrchestrator") -> "ContextCompactor":
    """Get the context compactor component."""
    return self._context_compactor


def _tool_output_formatter(self: "AgentOrchestrator") -> "ToolOutputFormatter":
    """Get the tool output formatter."""
    return self._tool_output_formatter


def _usage_analytics(self: "AgentOrchestrator") -> "UsageAnalytics":
    """Get the usage analytics singleton."""
    return self._usage_analytics


def _sequence_tracker(self: "AgentOrchestrator") -> "ToolSequenceTracker":
    """Get the tool sequence tracker."""
    return self._sequence_tracker


def _recovery_coordinator(self: "AgentOrchestrator") -> "StreamingRecoveryCoordinator":
    """Get the recovery coordinator for centralized recovery logic."""
    return self._recovery_coordinator


def _chunk_generator(self: "AgentOrchestrator") -> "ChunkGenerator":
    """Get the chunk generator for streaming output."""
    return self._chunk_generator


def _tool_planner(self: "AgentOrchestrator") -> "ToolPlanner":
    """Get the tool planner for tool planning operations."""
    return self._tool_planner


def _task_coordinator(self: "AgentOrchestrator") -> "TaskCoordinator":
    """Get the task coordinator for task coordination operations."""
    return self._task_coordinator


def _code_correction_middleware(self: "AgentOrchestrator") -> Optional[Any]:
    """Get the code correction middleware for automatic code validation/fixing."""
    return self._code_correction_middleware


def _checkpoint_manager(self: "AgentOrchestrator") -> Optional[Any]:
    """Get the checkpoint manager for time-travel debugging."""
    return self._checkpoint_manager


def _vertical_context(self: "AgentOrchestrator") -> "VerticalContext":
    """Get the vertical context for unified vertical state access."""
    return self._vertical_context


# =====================================================================
# Group 2: Lazy coordinator constructor properties
# =====================================================================

def _protocol_adapter(self: "AgentOrchestrator") -> Any:
    """Get the protocol adapter for DIP compliance (lazy init)."""
    if self._protocol_adapter is None:
        from victor.agent.coordinators.protocol_adapters import OrchestratorProtocolAdapter

        self._protocol_adapter = OrchestratorProtocolAdapter(self)
    return self._protocol_adapter


def _execution_coordinator(self: "AgentOrchestrator") -> Any:
    """Get the execution coordinator for agentic loop (lazy init)."""
    if self._execution_coordinator is None:
        from victor.agent.coordinators.execution_coordinator import ExecutionCoordinator

        self._execution_coordinator = ExecutionCoordinator(
            chat_context=self.protocol_adapter,
            tool_context=self.protocol_adapter,
            provider_context=self.protocol_adapter,
            execution_provider=self.protocol_adapter,
        )
    return self._execution_coordinator


def _sync_chat_coordinator(self: "AgentOrchestrator") -> Any:
    """Get the sync chat coordinator for non-streaming execution (lazy init)."""
    if self._sync_chat_coordinator is None:
        from victor.agent.coordinators.sync_chat_coordinator import SyncChatCoordinator

        self._sync_chat_coordinator = SyncChatCoordinator(
            chat_context=self.protocol_adapter,
            tool_context=self.protocol_adapter,
            provider_context=self.protocol_adapter,
            execution_coordinator=self.execution_coordinator,
            orchestrator=self,
        )
    return self._sync_chat_coordinator


def _streaming_chat_coordinator(self: "AgentOrchestrator") -> Any:
    """Get the streaming chat coordinator for streaming execution (lazy init)."""
    if self._streaming_chat_coordinator is None:
        from victor.agent.coordinators.streaming_chat_coordinator import (
            StreamingChatCoordinator,
        )

        self._streaming_chat_coordinator = StreamingChatCoordinator(
            chat_context=self.protocol_adapter,
            tool_context=self.protocol_adapter,
            provider_context=self.protocol_adapter,
            event_emitter=self.observability,
        )
    return self._streaming_chat_coordinator


def _unified_chat_coordinator(self: "AgentOrchestrator") -> Any:
    """Get the unified chat coordinator facade (lazy init)."""
    if self._unified_chat_coordinator is None:
        from victor.agent.coordinators.unified_chat_coordinator import (
            UnifiedChatCoordinator,
        )
        from victor.agent.coordinators.protocols import ExecutionMode

        self._unified_chat_coordinator = UnifiedChatCoordinator(
            sync_coordinator=self.sync_chat_coordinator,
            streaming_coordinator=self.streaming_chat_coordinator,
            default_mode=ExecutionMode.SYNC,
        )
    return self._unified_chat_coordinator


def _intelligent_integration(self: "AgentOrchestrator") -> Optional["OrchestratorIntegration"]:
    """Get the intelligent pipeline integration (lazy init)."""
    if not self._intelligent_pipeline_enabled:
        return None

    if self._intelligent_integration is None:
        try:
            from victor.agent.orchestrator_integration import OrchestratorIntegration
            from victor.agent.intelligent_pipeline import IntelligentAgentPipeline
            from victor.config.settings import get_project_paths
            from victor.context.project_context import VICTOR_DIR_NAME

            if self.project_context.context_file:
                parent_dir = self.project_context.context_file.parent
                if parent_dir.name == VICTOR_DIR_NAME:
                    intelligent_project_root = str(parent_dir.parent)
                else:
                    intelligent_project_root = str(parent_dir)
            else:
                intelligent_project_root = str(get_project_paths().project_root)

            pipeline = IntelligentAgentPipeline(
                provider_name=self.provider_name,
                model=self.model,
                profile_name=f"{self.provider_name}:{self.model}",
                project_root=intelligent_project_root,
            )
            self._intelligent_integration = OrchestratorIntegration(
                orchestrator=self,
                pipeline=pipeline,
                config=self._intelligent_integration_config,
            )
            logger.info(
                f"IntelligentPipeline initialized for "
                f"{self.provider_name}:{self.model}"
            )
        except ImportError as e:
            logger.debug(f"IntelligentPipeline dependencies not available: {e}")
            self._intelligent_pipeline_enabled = False
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                f"Failed to initialize IntelligentPipeline (config error): {e}"
            )
            self._intelligent_pipeline_enabled = False

    return self._intelligent_integration


def _subagent_orchestrator(self: "AgentOrchestrator") -> Optional[Any]:
    """Get the sub-agent orchestrator (lazy init)."""
    if not self._subagent_orchestration_enabled:
        return None

    if self._subagent_orchestrator is None:
        try:
            from victor.agent.subagents import SubAgentOrchestrator

            self._subagent_orchestrator = SubAgentOrchestrator(parent=self)
            logger.info("SubAgentOrchestrator initialized")
        except ImportError as e:
            logger.debug(f"SubAgentOrchestrator module not available: {e}")
            self._subagent_orchestration_enabled = False
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                f"Failed to initialize SubAgentOrchestrator (config error): {e}"
            )
            self._subagent_orchestration_enabled = False

    return self._subagent_orchestrator


def _coordination(self: "AgentOrchestrator") -> Any:
    """Get the mode-workflow-team coordinator (lazy init)."""
    if self._mode_workflow_team_coordinator is None:
        self._mode_workflow_team_coordinator = (
            self._factory.create_mode_workflow_team_coordinator(
                self._vertical_context
            )
        )
        logger.debug("ModeWorkflowTeamCoordinator initialized on first access")

    return self._mode_workflow_team_coordinator


# =====================================================================
# Group 3: Recovery properties (lazy resolution via get_instance)
# =====================================================================

def _recovery_handler(self: "AgentOrchestrator") -> Optional["RecoveryHandler"]:
    """Get the recovery handler for model failure recovery."""
    handler = getattr(self, "_recovery_handler", None)
    if hasattr(handler, "get_instance"):
        resolved = handler.get_instance()
        self._recovery_handler = resolved
        return resolved
    return handler


def _recovery_integration(self: "AgentOrchestrator") -> "OrchestratorRecoveryIntegration":
    """Get the recovery integration submodule."""
    integration = getattr(self, "_recovery_integration", None)
    if hasattr(integration, "get_instance"):
        resolved = integration.get_instance()
        self._recovery_integration = resolved
        return resolved
    return integration


# =====================================================================
# Group 4: Session state delegation properties (with setters)
# =====================================================================

def _session_state_get(self: "AgentOrchestrator") -> "SessionStateManager":
    """Get the session state manager."""
    return self._session_accessor.session_state


def _tool_calls_used_get(self: "AgentOrchestrator") -> int:
    """Get the number of tool calls used in this session."""
    return self._session_accessor.tool_calls_used


def _tool_calls_used_set(self: "AgentOrchestrator", value: int) -> None:
    self._session_accessor.tool_calls_used = value


def _observed_files_get(self: "AgentOrchestrator") -> set:
    """Get set of files observed/read during this session."""
    return self._session_accessor.observed_files


def _observed_files_set(self: "AgentOrchestrator", value: set) -> None:
    self._session_accessor.observed_files = value


def _executed_tools_get(self: "AgentOrchestrator") -> list:
    """Get list of executed tool names in order."""
    return self._session_accessor.executed_tools


def _executed_tools_set(self: "AgentOrchestrator", value: list) -> None:
    self._session_accessor.executed_tools = value


def _failed_tool_signatures_get(self: "AgentOrchestrator") -> set:
    """Get set of (tool_name, args_hash) tuples for failed calls."""
    return self._session_accessor.failed_tool_signatures


def _failed_tool_signatures_set(self: "AgentOrchestrator", value: set) -> None:
    self._session_accessor.failed_tool_signatures = value


def _tool_capability_warned_get(self: "AgentOrchestrator") -> bool:
    """Get whether we've warned about tool capability limitations."""
    return self._session_accessor.tool_capability_warned


def _tool_capability_warned_set(self: "AgentOrchestrator", value: bool) -> None:
    self._session_accessor.tool_capability_warned = value


def _read_files_session_get(self: "AgentOrchestrator") -> set:
    """Get files read during this session for task completion detection."""
    return self._session_accessor.read_files_session


def _required_files_get(self: "AgentOrchestrator") -> list:
    """Get required files extracted from user prompts."""
    return self._session_accessor.required_files


def _required_files_set(self: "AgentOrchestrator", value: list) -> None:
    self._session_accessor.required_files = value


def _required_outputs_get(self: "AgentOrchestrator") -> list:
    """Get required outputs extracted from user prompts."""
    return self._session_accessor.required_outputs


def _required_outputs_set(self: "AgentOrchestrator", value: list) -> None:
    self._session_accessor.required_outputs = value


def _all_files_read_nudge_sent_get(self: "AgentOrchestrator") -> bool:
    """Get whether we've sent a nudge that all required files are read."""
    return self._session_accessor.all_files_read_nudge_sent


def _all_files_read_nudge_sent_set(self: "AgentOrchestrator", value: bool) -> None:
    self._session_accessor.all_files_read_nudge_sent = value


def _cumulative_token_usage_get(self: "AgentOrchestrator") -> dict:
    """Get cumulative token usage for evaluation/benchmarking."""
    return self._session_accessor.cumulative_token_usage


def _cumulative_token_usage_set(self: "AgentOrchestrator", value: dict) -> None:
    self._session_accessor.cumulative_token_usage = value


# =====================================================================
# Property installation registry
# =====================================================================

# Map of public property name -> (getter, setter_or_None)
_PROPERTY_REGISTRY: dict[str, Any] = {
    # Group 1: Simple accessors (getter only)
    "conversation_controller": (_conversation_controller, None),
    "tool_pipeline": (_tool_pipeline, None),
    "streaming_controller": (_streaming_controller, None),
    "streaming_handler": (_streaming_handler, None),
    "task_analyzer": (_task_analyzer, None),
    "provider_manager": (_provider_manager, None),
    "context_compactor": (_context_compactor, None),
    "tool_output_formatter": (_tool_output_formatter, None),
    "usage_analytics": (_usage_analytics, None),
    "sequence_tracker": (_sequence_tracker, None),
    "recovery_coordinator": (_recovery_coordinator, None),
    "chunk_generator": (_chunk_generator, None),
    "tool_planner": (_tool_planner, None),
    "task_coordinator": (_task_coordinator, None),
    "code_correction_middleware": (_code_correction_middleware, None),
    "checkpoint_manager": (_checkpoint_manager, None),
    "vertical_context": (_vertical_context, None),
    # Group 2: Lazy coordinators (getter only)
    "protocol_adapter": (_protocol_adapter, None),
    "execution_coordinator": (_execution_coordinator, None),
    "sync_chat_coordinator": (_sync_chat_coordinator, None),
    "streaming_chat_coordinator": (_streaming_chat_coordinator, None),
    "unified_chat_coordinator": (_unified_chat_coordinator, None),
    "intelligent_integration": (_intelligent_integration, None),
    "subagent_orchestrator": (_subagent_orchestrator, None),
    "coordination": (_coordination, None),
    # Group 3: Recovery (getter only, with lazy resolution)
    "recovery_handler": (_recovery_handler, None),
    "recovery_integration": (_recovery_integration, None),
    # Group 4: Session state delegation (getter + setter)
    "session_state": (_session_state_get, None),
    "tool_calls_used": (_tool_calls_used_get, _tool_calls_used_set),
    "observed_files": (_observed_files_get, _observed_files_set),
    "executed_tools": (_executed_tools_get, _executed_tools_set),
    "failed_tool_signatures": (_failed_tool_signatures_get, _failed_tool_signatures_set),
    "_tool_capability_warned": (_tool_capability_warned_get, _tool_capability_warned_set),
    "_read_files_session": (_read_files_session_get, None),
    "_required_files": (_required_files_get, _required_files_set),
    "_required_outputs": (_required_outputs_get, _required_outputs_set),
    "_all_files_read_nudge_sent": (
        _all_files_read_nudge_sent_get,
        _all_files_read_nudge_sent_set,
    ),
    "_cumulative_token_usage": (_cumulative_token_usage_get, _cumulative_token_usage_set),
}


def install_properties(cls: type) -> type:
    """Install extracted property descriptors onto *cls*.

    Called once at module load time (after ``AgentOrchestrator`` class body
    is defined) to inject the property descriptors.  Because they live in
    the class ``__dict__``, ``unittest.mock.patch.object`` and normal
    attribute introspection continue to work.

    Supports both getter-only and getter+setter properties.

    Args:
        cls: The AgentOrchestrator class (or compatible subclass).

    Returns:
        The same class, with property descriptors installed.
    """
    for name, (getter, setter) in _PROPERTY_REGISTRY.items():
        prop = property(getter, setter, doc=getter.__doc__)
        setattr(cls, name, prop)
    return cls
