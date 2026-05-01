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

This package provides state-passed coordinators for specialized functionality.
The deprecated service-first coordinators (chat, tool, session) have been removed.

Preferred surfaces:
- ``victor.agent.services`` for service-owned chat, tool, and session flows
- ``victor.agent.coordinators.ExplorationStatePassedCoordinator`` for exploration
- ``victor.agent.coordinators.CoordinationStatePassedCoordinator`` for
  coordination recommendations
- ``victor.agent.coordinators.SystemPromptStatePassedCoordinator`` for system prompts
- ``victor.agent.coordinators.SafetyStatePassedCoordinator`` for safety

Note: ToolCoordinator, ChatCoordinator, SessionCoordinator have been removed.
Use ToolService, ChatService, SessionService instead.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

_SUBMODULE_MAP: dict[str, str] = {
    # Runtime coordinators from services
    "ExplorationCoordinator": "services.exploration_runtime",
    "ExplorationResult": "services.exploration_runtime",
    "MetricsCoordinator": "services.metrics_service",
    "create_metrics_coordinator": "services.metrics_service",
    "SafetyCoordinator": "victor_sdk.safety",
    "SafetyRule": "victor_sdk.safety",
    "SafetyCheckResult": "victor_sdk.safety",
    "SafetyStats": "victor_sdk.safety",
    "SafetyAction": "victor_sdk.safety",
    "SafetyCategory": "victor_sdk.safety",
    "SystemPromptCoordinator": "services.system_prompt_runtime",
    "PlanningCoordinator": "services.planning_runtime",
    "PlanningConfig": "services.planning_runtime",
    "PlanningMode": "services.planning_runtime",
    "PlanningResult": "services.planning_runtime",
    # SDK-based coordinators
    "ConversationCoordinator": "victor_sdk.conversation",
    "TurnType": "victor_sdk.conversation",
    "ConversationTurn": "victor_sdk.conversation",
    "ConversationStats": "victor_sdk.conversation",
    "ConversationContext": "victor_sdk.conversation",
}

_MODULE_MEMBERS = {
    "exploration_state_passed": [
        "ExplorationStatePassedCoordinator",
    ],
    "coordination_state_passed": [
        "CoordinationStatePassedCoordinator",
    ],
    # NOTE: tool_coordinator, tool_observability, tool_retry removed
    # These now import directly from victor.agent.services for backward compatibility
    # See __getattr__ below for service-level re-exports
    # NOTE: chat_coordinator, sync_chat_coordinator, streaming_chat_coordinator,
    # unified_chat_coordinator removed as deprecated shims.
    # These now import directly from victor.agent.services.chat_compat for backward compatibility.
    # See __getattr__ below for service-level re-exports.
    "chat_protocols": [
        "ChatContextProtocol",
        "ChatOrchestratorProtocol",
        "ProviderContextProtocol",
        "ToolContextProtocol",
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
    "system_prompt_state_passed": [
        "SystemPromptStatePassedCoordinator",
    ],
    "safety_state_passed": [
        "SafetyStatePassedCoordinator",
    ],
    # Stage transition coordination (Phase 1 optimization)
    "stage_transition_coordinator": [
        "StageTransitionCoordinator",
        "TransitionDecision",
        "TransitionResult",
        "TurnContext",
    ],
    "transition_strategies": [
        "TransitionStrategyProtocol",
        "HeuristicOnlyTransitionStrategy",
        "EdgeModelTransitionStrategy",
        "HybridTransitionStrategy",
        "create_transition_strategy",
    ],
    # NOTE: exploration_coordinator, metrics_coordinator, safety_coordinator,
    # system_prompt_coordinator, planning_coordinator, session_coordinator,
    # turn_executor, conversation_coordinator removed as deprecated shims.
    # These now import directly from services for backward compatibility.
    # See __getattr__ below for service-level re-exports.
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
}


def __getattr__(name: str) -> Any:
    """Resolve coordinator exports lazily and warn on deprecated shims.

    For deleted coordinator shims: re-exports from services or SDK.
    For chat_protocols: imports from services.protocols.chat_runtime.
    For other coordinators: imports from coordinators.{module_name}.
    """
    # Service-level re-exports for coordinators
    if name in {
        # Runtime coordinators from services
        "ExplorationCoordinator",
        "ExplorationResult",
        "MetricsCoordinator",
        "create_metrics_coordinator",
        "SafetyCoordinator",
        "SafetyRule",
        "SafetyCheckResult",
        "SafetyStats",
        "SafetyAction",
        "SafetyCategory",
        "SystemPromptCoordinator",
        "PlanningCoordinator",
        "PlanningConfig",
        "PlanningMode",
        "PlanningResult",
        # SDK-based coordinators
        "ConversationCoordinator",
        "TurnType",
        "ConversationTurn",
        "ConversationStats",
        "ConversationContext",
    }:
        # Import from appropriate service module
        if name in {"ExplorationCoordinator", "ExplorationResult"}:
            module = importlib.import_module("victor.agent.services.exploration_runtime")
        elif name in {"MetricsCoordinator", "create_metrics_coordinator"}:
            module = importlib.import_module("victor.agent.services.metrics_service")
        elif name in {
            "SafetyCoordinator",
            "SafetyRule",
            "SafetyCheckResult",
            "SafetyStats",
            "SafetyAction",
            "SafetyCategory",
        }:
            module = importlib.import_module("victor_sdk.safety")
        elif name == "SystemPromptCoordinator":
            module = importlib.import_module("victor.agent.services.system_prompt_runtime")
        elif name in {"PlanningCoordinator", "PlanningConfig", "PlanningMode", "PlanningResult"}:
            module = importlib.import_module("victor.agent.services.planning_runtime")
        elif name in {
            "ConversationCoordinator",
            "TurnType",
            "ConversationTurn",
            "ConversationStats",
            "ConversationContext",
        }:
            module = importlib.import_module("victor_sdk.conversation")

        value = getattr(module, name)
        # Apply deprecation warning
        if name in _DEPRECATED_EXPORTS:
            warnings.warn(_DEPRECATED_EXPORTS[name], DeprecationWarning, stacklevel=2)
        globals()[name] = value
        return value

    # Original coordinator imports
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
        globals()[name] = value
        return value
    raise AttributeError(f"module 'victor.agent.coordinators' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the lazily exported coordinator surface."""
    return sorted(list(globals().keys()) + __all__)
