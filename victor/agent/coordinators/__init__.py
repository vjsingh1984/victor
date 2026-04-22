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
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

_SUBMODULE_MAP: dict[str, str] = {}

_MODULE_MEMBERS = {
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
}

for _module_name, _exported_names in _MODULE_MEMBERS.items():
    for _exported_name in _exported_names:
        _SUBMODULE_MAP[_exported_name] = _module_name

__all__ = list(_SUBMODULE_MAP.keys())

_DEPRECATED_EXPORTS = {
    "ChatCoordinator": (
        "victor.agent.coordinators.ChatCoordinator is deprecated compatibility "
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
