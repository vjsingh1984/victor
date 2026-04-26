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

"""Service-owned compatibility runtime for deprecated prompt coordination.

This module hosts the deprecated PromptCoordinator compatibility shim while
service-first orchestration keeps prompt assembly outside the orchestrator
facade path. The canonical orchestrator-owned prompt assembly surface is
``victor.agent.prompt_pipeline.UnifiedPromptPipeline``. The canonical
``PromptRuntimeProtocol`` implementation is
``victor.agent.services.prompt_runtime.PromptRuntimeAdapter``. The narrower
``SystemPromptCoordinator`` wrapper remains compatibility-only and is not the
end-state replacement.

The legacy ``victor.agent.prompt_coordinator`` module now re-exports this
implementation for compatibility.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Optional, Protocol, runtime_checkable

from victor.agent.services.prompt_runtime import (
    PromptRuntimeAdapter,
    PromptRuntimeConfig as PromptCoordinatorConfig,
    PromptRuntimeContext as TaskContext,
)

if TYPE_CHECKING:
    from victor.agent.vertical_context import VerticalContext
    from victor.agent.system_prompt_policy import SystemPromptPolicy
    from victor.framework.prompt_builder import PromptBuilder

_PROMPT_COORDINATOR_DEPRECATION_MESSAGE = (
    "PromptCoordinator is deprecated. Use UnifiedPromptPipeline from "
    "victor.agent.prompt_pipeline instead. "
    "This adapter will be removed in a future release."
)


@runtime_checkable
class IPromptCoordinator(Protocol):
    """Protocol for prompt coordination operations."""

    def build_system_prompt(self, context: TaskContext, include_hints: bool = True) -> str: ...
    def add_task_hint(self, task_type: str, hint: str) -> None: ...
    def get_task_hint(self, task_type: str) -> Optional[str]: ...


class PromptCoordinator(PromptRuntimeAdapter):
    """Deprecated compatibility shim over PromptRuntimeAdapter."""

    def __init__(
        self,
        prompt_builder: Optional["PromptBuilder"] = None,
        vertical_context: Optional["VerticalContext"] = None,
        config: Optional[PromptCoordinatorConfig] = None,
        base_identity: Optional[str] = None,
        on_prompt_built: Optional[Callable[[str, TaskContext], None]] = None,
        policy: Optional["SystemPromptPolicy"] = None,
        warn_on_init: bool = True,
    ) -> None:
        """Initialize the deprecated compatibility shim."""
        if warn_on_init:
            warnings.warn(
                _PROMPT_COORDINATOR_DEPRECATION_MESSAGE,
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(
            prompt_builder=prompt_builder,
            vertical_context=vertical_context,
            config=config,
            base_identity=base_identity,
            on_prompt_built=on_prompt_built,
            policy=policy,
        )


def create_prompt_coordinator(
    prompt_builder: Optional["PromptBuilder"] = None,
    vertical_context: Optional["VerticalContext"] = None,
    config: Optional[PromptCoordinatorConfig] = None,
    base_identity: Optional[str] = None,
    policy: Optional["SystemPromptPolicy"] = None,
    warn_on_init: bool = True,
) -> PromptCoordinator:
    """Factory function to create a PromptCoordinator.

    Args:
        prompt_builder: Builder for prompt composition
        vertical_context: Optional vertical context for sections
        config: Configuration options
        base_identity: Base identity section for the prompt
        policy: Optional prompt enforcement policy

    Returns:
        Configured PromptCoordinator instance
    """
    if warn_on_init:
        warnings.warn(
            "create_prompt_coordinator() builds a deprecated PromptCoordinator shim. "
            "Prefer UnifiedPromptPipeline for active prompt assembly.",
            DeprecationWarning,
            stacklevel=2,
        )

    return PromptCoordinator(
        prompt_builder=prompt_builder,
        vertical_context=vertical_context,
        config=config,
        base_identity=base_identity,
        policy=policy,
        warn_on_init=False,
    )


__all__ = [
    "PromptCoordinator",
    "PromptCoordinatorConfig",
    "TaskContext",
    "IPromptCoordinator",
    "create_prompt_coordinator",
]
