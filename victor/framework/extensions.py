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

"""Stable public API surface for external vertical extensions.

External verticals (packages like victor-coding, victor-research) should
import framework internals ONLY through this module or other
victor.framework.* modules. This provides a stable contract that
insulates verticals from internal refactoring.

Importing directly from victor.agent.*, victor.core.container.*,
victor.workflows.executor.*, or victor.evaluation.* is deprecated
and may break without notice.

Usage:
    from victor.framework.extensions import (
        SafetyCoordinator,
        ConversationCoordinator,
        WorkflowExecutor,
        WorkflowContext,
        ComputeNode,
        CodeCorrectionMiddleware,
        ModeConfigRegistry,
        AgentSpec,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    # Safety coordination
    "SafetyCoordinator",
    "SafetyAction",
    "SafetyCategory",
    "SafetyRule",
    # Conversation coordination
    "ConversationCoordinator",
    "ConversationStats",
    "ConversationTurn",
    "TurnType",
    # Workflow execution
    "WorkflowExecutor",
    "WorkflowContext",
    "ComputeNode",
    "NodeResult",
    "ExecutorNodeStatus",
    "register_compute_handler",
    "get_compute_handler",
    # Workflow definition
    "WorkflowBuilder",
    "WorkflowDefinition",
    "workflow",
    "AgentNode",
    "ConditionNode",
    "ParallelNode",
    # Code correction
    "CodeCorrectionMiddleware",
    "CodeCorrectionConfig",
    "CorrectionResult",
    # Code validation
    "CodeValidationResult",
    "Language",
    # Mode configuration
    "ModeConfigRegistry",
    "ModeDefinition",
    "ModeConfig",
    "RegistryBasedModeConfigProvider",
    # Service container
    "ServiceContainer",
    "ServiceLifetime",
    # Agent specs
    "AgentSpec",
    "AgentCapabilities",
    "AgentConstraints",
    "ModelPreference",
    "OutputFormat",
    "DelegationPolicy",
    # Sub-agents
    "SubAgent",
    "SubAgentConfig",
    "SubAgentResult",
    "SubAgentRole",
    "set_role_tool_provider",
    # Middleware
    "MiddlewareChain",
    "MiddlewareAbortError",
    "create_middleware_chain",
    # Vertical context
    "VerticalContext",
    "create_vertical_context",
    "VerticalContextProtocol",
    "MutableVerticalContextProtocol",
    # Handler registry
    "HandlerRegistry",
    "get_handler_registry",
    "register_global_handler",
    "register_vertical_handlers",
    # Provider access
    "ProviderRegistry",
]


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies and unnecessary loading."""
    _LAZY_IMPORTS = {
        # Safety
        "SafetyCoordinator": "victor.agent.coordinators.safety_coordinator",
        "SafetyAction": "victor.agent.coordinators.safety_coordinator",
        "SafetyCategory": "victor.agent.coordinators.safety_coordinator",
        "SafetyRule": "victor.agent.coordinators.safety_coordinator",
        # Conversation
        "ConversationCoordinator": "victor.agent.coordinators.conversation_coordinator",
        "ConversationStats": "victor.agent.coordinators.conversation_coordinator",
        "ConversationTurn": "victor.agent.coordinators.conversation_coordinator",
        "TurnType": "victor.agent.coordinators.conversation_coordinator",
        # Workflow execution
        "WorkflowExecutor": "victor.workflows.executor",
        "WorkflowContext": "victor.workflows.executor",
        "NodeResult": "victor.workflows.executor",
        "ExecutorNodeStatus": "victor.workflows.executor",
        "register_compute_handler": "victor.workflows.executor",
        "get_compute_handler": "victor.workflows.executor",
        # Workflow definition
        "ComputeNode": "victor.workflows.definition",
        "WorkflowBuilder": "victor.workflows.definition",
        "WorkflowDefinition": "victor.workflows.definition",
        "workflow": "victor.workflows.definition",
        "AgentNode": "victor.workflows.definition",
        "ConditionNode": "victor.workflows.definition",
        "ParallelNode": "victor.workflows.definition",
        # Code correction
        "CodeCorrectionMiddleware": "victor.agent.code_correction_middleware",
        "CodeCorrectionConfig": "victor.agent.code_correction_middleware",
        "CorrectionResult": "victor.agent.code_correction_middleware",
        # Code validation
        "CodeValidationResult": "victor.evaluation.correction.types",
        "Language": "victor.evaluation.correction.types",
        # Mode configuration
        "ModeConfigRegistry": "victor.core.mode_config",
        "ModeDefinition": "victor.core.mode_config",
        "ModeConfig": "victor.core.mode_config",
        "RegistryBasedModeConfigProvider": "victor.core.mode_config",
        # Service container
        "ServiceContainer": "victor.core.container",
        "ServiceLifetime": "victor.core.container",
        # Agent specs
        "AgentSpec": "victor.agent.specs.models",
        "AgentCapabilities": "victor.agent.specs.models",
        "AgentConstraints": "victor.agent.specs.models",
        "ModelPreference": "victor.agent.specs.models",
        "OutputFormat": "victor.agent.specs.models",
        "DelegationPolicy": "victor.agent.specs.models",
        # Sub-agents
        "SubAgent": "victor.agent.subagents.base",
        "SubAgentConfig": "victor.agent.subagents.base",
        "SubAgentResult": "victor.agent.subagents.base",
        "SubAgentRole": "victor.agent.subagents.base",
        "set_role_tool_provider": "victor.agent.subagents.protocols",
        # Middleware
        "MiddlewareChain": "victor.agent.middleware_chain",
        "MiddlewareAbortError": "victor.agent.middleware_chain",
        "create_middleware_chain": "victor.agent.middleware_chain",
        # Vertical context
        "VerticalContext": "victor.agent.vertical_context",
        "create_vertical_context": "victor.agent.vertical_context",
        "VerticalContextProtocol": "victor.agent.vertical_context",
        "MutableVerticalContextProtocol": "victor.agent.vertical_context",
        # Handler registry
        "HandlerRegistry": "victor.framework.handler_registry",
        "get_handler_registry": "victor.framework.handler_registry",
        "register_global_handler": "victor.framework.handler_registry",
        "register_vertical_handlers": "victor.framework.handler_registry",
        # Provider access
        "ProviderRegistry": "victor.providers.registry",
    }

    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)

    raise AttributeError(f"module 'victor.framework.extensions' has no attribute {name!r}")
