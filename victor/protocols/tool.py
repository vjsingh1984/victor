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

"""Tool protocols for dependency inversion.

This module defines protocol interfaces for tools, breaking circular
dependencies between tool capabilities and tool base classes.

Phase 3 Architectural Improvement: Breaking circular dependencies via protocols.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ITool(Protocol):
    """Protocol for tool instances.

    This protocol breaks the circular dependency between:
    - victor/tools/capabilities/system.py (needs BaseTool for type hints)
    - victor/tools/base.py (implements BaseTool)

    By using this protocol, capabilities system can reference the tool interface
    without importing the concrete BaseTool implementation.

    The protocol defines the minimal interface required for tool capabilities
    auto-discovery and metadata extraction.

    Example:
        from victor.protocols.tool import ITool

        def discover_capabilities(tools: List[ITool]) -> None:
            for tool in tools:
                metadata = tool.get_metadata()
                print(f"Tool: {tool.name}, Category: {metadata.category}")
    """

    name: str
    """Tool name/identifier."""

    def get_metadata(self) -> Any:
        """Get tool metadata for capability discovery.

        Returns:
            ToolMetadata object with category, keywords, use_cases, etc.
        """
        ...


__all__ = [
    "ITool",
]
