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
    # Workflow execution
    "WorkflowExecutor",
    "WorkflowContext",
    "ComputeNode",
    "NodeResult",
    # Code correction
    "CodeCorrectionMiddleware",
    # Mode configuration
    "ModeConfigRegistry",
    "ModeDefinition",
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
        # Workflow execution
        "WorkflowExecutor": "victor.workflows.executor",
        "WorkflowContext": "victor.workflows.executor",
        "ComputeNode": "victor.workflows.definition",
        "NodeResult": "victor.workflows.executor",
        # Code correction
        "CodeCorrectionMiddleware": "victor.agent.code_correction_middleware",
        # Mode configuration
        "ModeConfigRegistry": "victor.core.mode_config",
        "ModeDefinition": "victor.core.mode_config",
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
