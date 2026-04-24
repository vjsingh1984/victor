# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deprecated coordinator-path shim for ToolCoordinator."""

from victor.agent.services.tool_compat import (
    IToolCoordinator,
    NormalizedArgs,
    TaskContext,
    ToolCallValidation,
    ToolCoordinator,
    ToolCoordinatorConfig,
    ToolExecutionResult,
    ToolObservabilityHandler,
    ToolResultContext,
    ToolRetryExecutor,
    create_tool_coordinator,
)

__all__ = [
    "ToolCoordinator",
    "ToolCoordinatorConfig",
    "ToolResultContext",
    "TaskContext",
    "ToolCallValidation",
    "NormalizedArgs",
    "ToolExecutionResult",
    "IToolCoordinator",
    "create_tool_coordinator",
    "ToolObservabilityHandler",
    "ToolRetryExecutor",
]
