# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deprecated shim for PromptCoordinator compatibility types."""

from victor.agent.services.prompt_compat import (
    IPromptCoordinator,
    PromptCoordinator,
    PromptCoordinatorConfig,
    TaskContext,
    create_prompt_coordinator,
)

__all__ = [
    "PromptCoordinator",
    "PromptCoordinatorConfig",
    "TaskContext",
    "IPromptCoordinator",
    "create_prompt_coordinator",
]
