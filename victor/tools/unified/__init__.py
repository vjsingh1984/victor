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

"""Unified Tool Registry

Consolidates tool discovery, registration, selection, and lifecycle
management into a single, coherent interface.

Usage:
    from victor.tools.unified import UnifiedToolRegistry

    registry = UnifiedToolRegistry.get_instance()

    # Discover tools
    await registry.discover()

    # Select tools
    tools = await registry.select_tools("Read files")

    # Get tool
    tool = registry.get("read_file")

Migration:
    from victor.tools.unified.adapters import migrate_to_unified_registry

    migrate_to_unified_registry()
"""

from victor.tools.unified.registry import (
    HookPhase,
    SelectionStrategy,
    ToolMetadata,
    ToolMetrics,
    UnifiedToolRegistry,
)

from victor.tools.unified.adapters import (
    SharedToolRegistryAdapter,
    ToolRegistryAdapter,
    migrate_to_unified_registry,
)

# NOTE: The bash-style command tools (code_tool, fs_tool, git_tool, search_tool,
# shell_tool, web_tool) are intentionally NOT re-exported here. Importing them as
# ``from victor.tools.unified.<name>_tool import <name>_tool`` rebinds the
# submodule name to the function and shadows the module object. That breaks
# ``mock.patch("victor.tools.unified.<name>_tool.<func>")`` on Python 3.10,
# whose ``mock`` dotted resolver does not fall back to ``sys.modules``. The
# canonical import path
# ``from victor.tools.unified.<name>_tool import <name>_tool`` remains the
# supported way to access these tools.

__all__ = [
    # Core
    "UnifiedToolRegistry",
    "SelectionStrategy",
    "HookPhase",
    "ToolMetadata",
    "ToolMetrics",
    # Adapters
    "SharedToolRegistryAdapter",
    "ToolRegistryAdapter",
    "migrate_to_unified_registry",
]
