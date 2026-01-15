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

"""Victor Integrations Module.

This module consolidates all integration-related functionality:

- api: HTTP API servers for IDE integrations (VS Code, JetBrains)
- mcp: Model Context Protocol client/server implementations
- protocol: Victor Protocol interface for unified client communication
- protocols: Query enhancement protocols (core protocols live in victor.protocols)

Example usage:
    # HTTP API
    from victor.integrations.api import VictorFastAPIServer

    # MCP
    from victor.integrations.mcp import MCPClient, MCPServer, MCPRegistry

    # Protocol interface
    from victor.integrations.protocol import VictorProtocol, DirectProtocolAdapter

    # SOLID protocols
    from victor.protocols import IProviderAdapter, IGroundingStrategy

    # Query enhancement protocols
    from victor.integrations.protocols.query_enhancement import EnhancementContext

LAZY LOADING: Integration submodules are loaded on-demand via __getattr__ to reduce
startup time. This improves Victor's import performance by ~0.3s (19% reduction).
Submodules are only imported when actually accessed (e.g., when MCP features are used).
"""

from __future__ import annotations

__all__ = [
    "api",
    "mcp",
    "protocol",
    "protocols",
]


def __getattr__(name: str):
    """Lazy load integration submodules on first access.

    This function is called by Python when an attribute is not found in the module.
    It lazy-loads the integration submodule to improve startup performance.

    Args:
        name: Name of the submodule being accessed

    Returns:
        The requested submodule

    Raises:
        AttributeError: If the requested submodule doesn't exist
    """
    if name in __all__:
        # Lazy import the submodule
        import importlib

        module = importlib.import_module(f"victor.integrations.{name}")

        # Cache it in sys.modules so future accesses don't call __getattr__ again
        import sys

        sys.modules[f"victor.integrations.{name}"] = module

        return module

    raise AttributeError(f"module 'victor.integrations' has no attribute '{name}'")
