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

"""Coordinator package for Victor agentic AI framework.

This package retains coordinator exports for backward compatibility, but the
runtime has moved to service-first ownership for chat, tool, session, recovery,
and provider flows. Deprecated coordinator exports remain available here as
lazy compatibility shims so existing imports continue to work while making the
migration boundary explicit.

Preferred surfaces for new code:
- ``victor.agent.services`` for service-owned chat, tool, and session flows
- ``OrchestrationFacade.chat_stream_runtime`` for the canonical streaming runtime
- ``victor.agent.coordinators.ExplorationCoordinator`` for read-only exploration
- ``victor.agent.coordinators.ExplorationStatePassedCoordinator``
- ``victor.agent.coordinators.SystemPromptStatePassedCoordinator``
- ``victor.agent.coordinators.SafetyStatePassedCoordinator``

If you already depend on orchestration facades, prefer the matching
``OrchestrationFacade`` properties instead of reaching into deprecated
coordinator shims.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

_SUBMODULE_MAP: dict[str, str] = {}

_MODULE_MEMBERS = {
    "exploration_coordinator": [
        "ExplorationCoordinator",
        "ExplorationResult",
    ],
    "exploration_state_passed": [
        "ExplorationStatePassedCoordinator",
    ],
    "tool_coordinator": [
        "ToolCoordinator",
        "ToolCoordinatorConfig",
        "TaskContext",
        "IToolCoordinator",
        "create_tool_coordinator",
    ],
    "tool_observability": [
        "ToolObservabilityHandler",
    ],
    "tool_retry": [
        "ToolRetryExecutor",
    ],
    "chat_coordinator": [
        "ChatCoordinator",
    ],
    "sync_chat_coordinator": [
        "SyncChatCoordinator",
    ],
    "streaming_chat_coordinator": [
        "StreamingChatCoordinator",
    ],
    "unified_chat_coordinator": [
        "UnifiedChatCoordinator",
    ],
    "chat_protocols": [
        "ChatContextProtocol",
        "ChatOrchestratorProtocol",
        "ProviderContextProtocol",
        "ToolContextProtocol",
    ],
    "session_coordinator": [
        "SessionCoordinator",
        "SessionInfo",
        "create_session_coordinator",
    ],
    "planning_coordinator": [
        "PlanningCoordinator",
        "PlanningConfig",
        "PlanningMode",
        "PlanningResult",
    ],
    "protocols": [
        "CacheProvider",
        "CacheEventType",
        "EventEmitter",
        "ConfigProvider",
        "NoOpCacheProvider",
        "NoOpEventEmitter",
        "DictConfigProvider",
    ],
    "conversation_coordinator": [
        "ConversationCoordinator",
        "TurnType",
        "ConversationTurn",
        "ConversationStats",
        "ConversationContext",
    ],
    "safety_coordinator": [
        "SafetyCoordinator",
        "SafetyRule",
        "SafetyCheckResult",
        "SafetyStats",
        "SafetyAction",
        "SafetyCategory",
    ],
    "metrics_coordinator": [
        "MetricsCoordinator",
    ],
    "system_prompt_coordinator": [
        "SystemPromptCoordinator",
    ],
    "system_prompt_state_passed": [
        "SystemPromptStatePassedCoordinator",
    ],
    "safety_state_passed": [
        "SafetyStatePassedCoordinator",
    ],
}

for _module_name, _exported_names in _MODULE_MEMBERS.items():
    for _exported_name in _exported_names:
        _SUBMODULE_MAP[_exported_name] = _module_name

__all__ = list(_SUBMODULE_MAP.keys())

_DEPRECATED_EXPORTS = {
    "ChatContextProtocol": (
        "victor.agent.coordinators.ChatContextProtocol is deprecated compatibility "
        "surface. Prefer ChatServiceProtocol and service-owned runtime boundaries "
        "from victor.agent.services."
    ),
    "ChatOrchestratorProtocol": (
        "victor.agent.coordinators.ChatOrchestratorProtocol is deprecated "
        "compatibility surface. Prefer ChatServiceProtocol and service-owned "
        "runtime boundaries from victor.agent.services."
    ),
    "ProviderContextProtocol": (
        "victor.agent.coordinators.ProviderContextProtocol is deprecated "
        "compatibility surface. Prefer ProviderServiceProtocol and service-owned "
        "runtime boundaries from victor.agent.services."
    ),
    "ToolContextProtocol": (
        "victor.agent.coordinators.ToolContextProtocol is deprecated compatibility "
        "surface. Prefer ToolServiceProtocol and service-owned runtime boundaries "
        "from victor.agent.services."
    ),
    "ChatCoordinator": (
        "victor.agent.coordinators.ChatCoordinator is deprecated compatibility "
        "surface. Prefer ChatService from victor.agent.services."
    ),
    "SyncChatCoordinator": (
        "victor.agent.coordinators.SyncChatCoordinator is deprecated compatibility "
        "surface. Prefer ChatService from victor.agent.services."
    ),
    "StreamingChatCoordinator": (
        "victor.agent.coordinators.StreamingChatCoordinator is deprecated compatibility "
        "surface. Prefer ChatService from victor.agent.services."
    ),
    "UnifiedChatCoordinator": (
        "victor.agent.coordinators.UnifiedChatCoordinator is deprecated compatibility "
        "surface. Prefer ChatService from victor.agent.services."
    ),
    "ToolCoordinator": (
        "victor.agent.coordinators.ToolCoordinator is deprecated compatibility "
        "surface. Prefer ToolService from victor.agent.services."
    ),
    "create_tool_coordinator": (
        "victor.agent.coordinators.create_tool_coordinator is deprecated "
        "compatibility surface. Prefer ToolService from victor.agent.services."
    ),
    "SessionCoordinator": (
        "victor.agent.coordinators.SessionCoordinator is deprecated compatibility "
        "surface. Prefer SessionService from victor.agent.services."
    ),
    "create_session_coordinator": (
        "victor.agent.coordinators.create_session_coordinator is deprecated "
        "compatibility surface. Prefer SessionService from victor.agent.services."
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve coordinator exports lazily and warn on deprecated shims."""
    if name in _SUBMODULE_MAP:
        if _SUBMODULE_MAP[name] == "chat_protocols":
            module = importlib.import_module("victor.agent.services.protocols.chat_runtime")
        else:
            module = importlib.import_module(f"victor.agent.coordinators.{_SUBMODULE_MAP[name]}")
        value = getattr(module, name)
        if name in _DEPRECATED_EXPORTS:
            warnings.warn(
                _DEPRECATED_EXPORTS[name],
                DeprecationWarning,
                stacklevel=2,
            )
            return value

        globals()[name] = value
        return value
    raise AttributeError(f"module 'victor.agent.coordinators' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the lazily exported coordinator surface."""
    return sorted(list(globals().keys()) + __all__)
