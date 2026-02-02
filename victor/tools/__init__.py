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

"""Tool framework - public API exports.

This module provides the canonical public API for the tools framework
by re-exporting commonly used classes from their respective modules.

Canonical import pattern:
    from victor.tools import BaseTool, ToolRegistry, ToolResult

Internal structure:
- victor/tools/base.py - BaseTool, ToolResult, and supporting classes
- victor/tools/enums.py - All enum classes
- victor/tools/metadata.py - ToolMetadata and ToolMetadataRegistry
- victor/tools/registry.py - ToolRegistry, Hook, HookError
- victor/tools/context.py - ToolExecutionContext and permissions
"""

# Re-export core tool classes for public API
from victor.tools.base import (
    BaseTool,
    ToolConfig,
    ToolMetadataProvider,
    ToolParameter,
    ToolResult,
    ToolValidationResult,
)

# Re-export enums from enums.py
from victor.tools.enums import (
    AccessMode,
    CostTier,
    DangerLevel,
    ExecutionCategory,
    Priority,
)

# Re-export metadata classes from metadata.py
from victor.tools.metadata import (
    ToolMetadata,
    ToolMetadataRegistry,
)

# Re-export authorization metadata classes (Phase 5)
from victor.tools.auth_metadata import (
    ToolAuthMetadata,
    ToolAuthMetadataRegistry,
    ToolSafety,
    get_tool_auth_metadata_registry,
    register_tool_auth_metadata,
)

# Re-export registry classes from registry.py
from victor.tools.registry import (
    Hook,
    HookError,
    ToolRegistry,
)

# Re-export tool graph classes
from victor.tools.tool_graph import (
    ToolNode,
    ToolTransition,
    ToolExecutionGraph,
    ToolGraphRegistry,
)

# Re-export typed context
from victor.tools.context import (
    DEFAULT_PERMISSIONS,
    FULL_PERMISSIONS,
    Permission,
    SAFE_PERMISSIONS,
    STANDARD_PERMISSIONS,
    ToolExecutionContext,
    create_context,
    create_full_access_context,
    create_readonly_context,
)

# Re-export selection protocol and registry
from victor.tools.selection import (
    BaseToolSelectionStrategy,
    PerformanceProfile,
    CrossVerticalToolSelectionContext,
    ToolSelectionStrategy,
    ToolSelectionStrategyRegistry,
    ToolSelectorFeatures,
    get_best_strategy,
    get_strategy,
    get_strategy_registry,
    list_strategies,
    register_strategy,
)

# Re-export LCEL-style composition (Promotion 4.3)
from victor.tools.composition import (
    Runnable,
    RunnableConfig,
    RunnableSequence,
    RunnableParallel,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableBinding,
    ToolRunnable,
    FunctionToolRunnable,
    as_runnable,
    chain,
    parallel,
    branch,
    extract_output,
    extract_if_success,
    map_keys,
    select_keys,
)

# Define __all__ for explicit exports
__all__ = [
    # Base classes
    "BaseTool",
    "ToolResult",
    "ToolParameter",
    "ToolConfig",
    "ToolValidationResult",
    "ToolMetadataProvider",
    # Enums
    "CostTier",
    "Priority",
    "AccessMode",
    "ExecutionCategory",
    "DangerLevel",
    # Metadata
    "ToolMetadata",
    "ToolMetadataRegistry",
    # Authorization Metadata (Phase 5)
    "ToolAuthMetadata",
    "ToolAuthMetadataRegistry",
    "ToolSafety",
    "get_tool_auth_metadata_registry",
    "register_tool_auth_metadata",
    # Registry
    "ToolRegistry",
    "Hook",
    "HookError",
    # Tool Graph
    "ToolNode",
    "ToolTransition",
    "ToolExecutionGraph",
    "ToolGraphRegistry",
    # Typed Context (Promotion 5)
    "ToolExecutionContext",
    "Permission",
    "DEFAULT_PERMISSIONS",
    "SAFE_PERMISSIONS",
    "STANDARD_PERMISSIONS",
    "FULL_PERMISSIONS",
    "create_context",
    "create_readonly_context",
    "create_full_access_context",
    # Selection Protocol (Promotion 4)
    "ToolSelectionStrategy",
    "BaseToolSelectionStrategy",
    "PerformanceProfile",
    "CrossVerticalToolSelectionContext",
    "ToolSelectorFeatures",
    "ToolSelectionStrategyRegistry",
    "get_strategy_registry",
    "register_strategy",
    "get_strategy",
    "get_best_strategy",
    "list_strategies",
    # LCEL-style Composition (Promotion 4.3)
    "Runnable",
    "RunnableConfig",
    "RunnableSequence",
    "RunnableParallel",
    "RunnableBranch",
    "RunnableLambda",
    "RunnablePassthrough",
    "RunnableBinding",
    "ToolRunnable",
    "FunctionToolRunnable",
    "as_runnable",
    "chain",
    "parallel",
    "branch",
    "extract_output",
    "extract_if_success",
    "map_keys",
    "select_keys",
]
