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

"""Tool framework - backward compatible re-exports.

This module provides backward compatibility by re-exporting all classes
from their new locations. Existing code can still import from victor.tools.base
or from victor.tools directly without any changes.

New structure:
- victor/tools/base.py - BaseTool, ToolResult, and supporting classes
- victor/tools/enums.py - All enum classes
- victor/tools/metadata.py - ToolMetadata and ToolMetadataRegistry
- victor/tools/registry.py - ToolRegistry, Hook, HookError
"""

# Re-export everything from base.py for backward compatibility
from victor.tools.base import (
    BaseTool,
    ToolConfig,
    ToolMetadataProvider,
    ToolParameter,
    ToolResult,
    ValidationResult,
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

# Re-export registry classes from registry.py
from victor.tools.registry import (
    Hook,
    HookError,
    ToolRegistry,
)

# Define __all__ for explicit exports
__all__ = [
    # Base classes
    "BaseTool",
    "ToolResult",
    "ToolParameter",
    "ToolConfig",
    "ValidationResult",
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
    # Registry
    "ToolRegistry",
    "Hook",
    "HookError",
]
